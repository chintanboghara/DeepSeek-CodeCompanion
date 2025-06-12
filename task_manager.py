import streamlit as st
import concurrent.futures
import uuid
import logging # Optional: for logging task statuses

@st.experimental_singleton # Ensure only one instance of the manager exists
class LLMTaskManager:
    def __init__(self):
        # Using ThreadPoolExecutor for I/O bound LLM calls
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=5) # max_workers can be tuned
        # Using st.session_state to store task futures and results, keyed by task_id
        # This makes them inherently session-specific.
        if '_llm_tasks_futures' not in st.session_state:
            st.session_state._llm_tasks_futures = {}
        if '_llm_tasks_results' not in st.session_state:
            st.session_state._llm_tasks_results = {}
        # For storing full streamed content if needed, though primary is via session_state["task_{task_id}_stream_content"]
        if '_llm_tasks_full_streamed_results' not in st.session_state:
            st.session_state._llm_tasks_full_streamed_results = {}


    def _process_streaming_task(self, task_id, target_func, args_tuple, kwargs_dict):
        """
        Internal method to execute the target function (which is a generator)
        and handle its streaming output or errors.
        Updates st.session_state for live streaming and stores final result/error.
        """
        st.session_state[f"task_{task_id}_stream_content"] = ""
        st.session_state[f"task_{task_id}_stream_status"] = "streaming"
        # Optional: st.session_state[f"task_{task_id}_stream_error_details"] = None

        accumulated_content_for_final_result = []
        target_func_name = target_func.__name__ # For logging

        try:
            generator = target_func(*args_tuple, **kwargs_dict)
            first_chunk = True

            for chunk in generator:
                if first_chunk:
                    # Check for prefixed error message from llm_logic.py
                    if isinstance(chunk, str) and (chunk.startswith("OLLAMA_CONNECTION_ERROR:") or \
                       chunk.startswith("LLM_RUNTIME_ERROR:") or \
                       chunk.startswith("LLM_UNEXPECTED_ERROR:")):
                        logging.error(f"Task {task_id} ({target_func_name}) yielded a prefixed error: {chunk}")
                        st.session_state[f"task_{task_id}_stream_status"] = "error"
                        # st.session_state[f"task_{task_id}_stream_error_details"] = chunk # Optional: store specific error string if needed separately
                        # Store the error string itself as an Exception. This allows get_task_result
                        # to raise it, maintaining compatibility with how errors were handled before streaming.
                        return Exception(chunk)
                    first_chunk = False

                # Append chunk to session state for app.py to display
                # Ensure chunk is a string before concatenation
                chunk_str = chunk if isinstance(chunk, str) else str(chunk)
                st.session_state[f"task_{task_id}_stream_content"] += chunk_str
                accumulated_content_for_final_result.append(chunk_str)

            # Stream completed successfully
            final_result_str = "".join(accumulated_content_for_final_result)
            st.session_state[f"task_{task_id}_stream_status"] = "completed"
            # Store the full result for get_task_result
            # st.session_state._llm_tasks_full_streamed_results[task_id] = final_result_str
            return final_result_str # This becomes the result of the future

        except Exception as e:
            # Handle exceptions that occur if target_func itself fails before/during iteration
            logging.exception(f"Exception during task {task_id} ({target_func_name}) execution: {e}")
            st.session_state[f"task_{task_id}_stream_status"] = "error"
            # st.session_state[f"task_{task_id}_stream_error_details"] = str(e)
            return e # This exception instance becomes the result of the future


    def submit_task(self, func, *args, **kwargs):
        """
        Submits a function to the executor and returns a unique task ID.
        If the function is a generator (for streaming), it's wrapped by _process_streaming_task.
        """
        task_id = str(uuid.uuid4())

        # Initialize stream-related session state immediately on submission
        st.session_state[f"task_{task_id}_stream_content"] = ""
        st.session_state[f"task_{task_id}_stream_status"] = "submitted" # Initial status

        # The actual function submitted to the executor is now _process_streaming_task,
        # which will call the original 'func' (generate_ai_response).
        future = self._executor.submit(self._process_streaming_task, task_id, func, args, kwargs)
        st.session_state._llm_tasks_futures[task_id] = future
        logging.info(f"Task {task_id} submitted for function {func.__name__} (wrapped by _process_streaming_task).")
        return task_id

    def get_task_status(self, task_id):
        """
        Returns the status of the task ("running", "completed", "error", or "not_found").
        If completed or error, stores result/exception in st.session_state._llm_tasks_results
        and removes future from st.session_state._llm_tasks_futures.
        """
        # Initial Checks (Task Not Found / Already Processed):
        if task_id not in st.session_state._llm_tasks_futures:
            # Task future not found. Check if it already completed and its result/error is stored.
            if task_id in st.session_state._llm_tasks_results:
                return "error" if isinstance(st.session_state._llm_tasks_results[task_id], Exception) else "completed"
            # Alternatively, check stream status if future was removed very quickly after completion/error.
            # This handles edge cases where the task finishes and cleans up its future before this status check.
            stream_status = st.session_state.get(f"task_{task_id}_stream_status")
            if stream_status == "completed": return "completed"
            if stream_status == "error": return "error"
            return "not_found" # Truly not found or cleaned up

        future = st.session_state._llm_tasks_futures[task_id]

        # Future Running:
        if future.running():
            # The ThreadPoolExecutor future is running, meaning _process_streaming_task is active.
            # app.py should primarily rely on st.session_state[f"task_{task_id}_stream_status"]
            # (e.g., "submitted", "streaming") for fine-grained status during the run.
            # This "running" state is more about the executor's perspective.
            return "running"

        # Future Done (Main Processing Block):
        elif future.done():
            # The _process_streaming_task (the wrapper handling the generator) has finished.
            # Its return value (final accumulated string or an Exception) needs to be processed and stored.
            try:
                result_or_exception = future.result() # This is what _process_streaming_task returned.
                st.session_state._llm_tasks_results[task_id] = result_or_exception # Store final outcome.

                # Ensure st.session_state stream_status is definitively set to 'completed' or 'error'.
                # This covers cases where _process_streaming_task might have failed before setting
                # its own stream_status, or if the stream was successful but we need to ensure
                # the final status reflects that for any subsequent checks.
                current_stream_status = st.session_state.get(f"task_{task_id}_stream_status", "unknown")

                if isinstance(result_or_exception, Exception):
                    # If the wrapper task itself returned an exception (e.g., prefixed error from stream, or other exception).
                    if current_stream_status != "error": # Avoid overwriting if _process_streaming_task already set it.
                        st.session_state[f"task_{task_id}_stream_status"] = "error"
                    logging.error(f"Task {task_id} (wrapper) resulted in error: {result_or_exception}")
                    # Future is done, remove it.
                    del st.session_state._llm_tasks_futures[task_id]
                    return "error"
                else:
                    # If the wrapper task returned a successful result (the fully accumulated string).
                    if current_stream_status not in ["completed", "error"]: # Avoid overwriting a specific error status set from within the stream.
                         st.session_state[f"task_{task_id}_stream_status"] = "completed"
                    logging.info(f"Task {task_id} (wrapper) completed. Final result stored.")
                    # Future is done, remove it.
                    del st.session_state._llm_tasks_futures[task_id]
                    return "completed"

            except Exception as e:
                # This catches exceptions if future.result() itself raises something unexpected
                # (e.g., an error in ThreadPoolExecutor, though _process_streaming_task is designed
                # to return exceptions rather than letting them be raised here from its own execution).
                st.session_state._llm_tasks_results[task_id] = e
                st.session_state[f"task_{task_id}_stream_status"] = "error" # Ensure status reflects this critical failure.
                logging.error(f"Task {task_id} (wrapper) future.result() raised an unexpected exception: {e}")
                if task_id in st.session_state._llm_tasks_futures: # Ensure cleanup if not already deleted.
                    del st.session_state._llm_tasks_futures[task_id]
                return "error"

        # Fallback/Final Status Check:
        # If future is not 'running' and not 'done' (an unlikely state for a future managed by this class),
        # or if this method is called at a point where _process_streaming_task has updated stream_status
        # but the future.done() state hasn't been processed in this same call yet.
        # This primarily relies on the stream_status set by _process_streaming_task as the source of truth
        # once the stream has actively started or finished.
        return st.session_state.get(f"task_{task_id}_stream_status", "unknown") # Default to 'unknown' if no status set yet.


    def get_task_result(self, task_id):
        """
        Returns the final result of a completed task (e.g., full accumulated string for streams).
        Raises the exception if the task resulted in an error.
        """
        # Prioritize result from _llm_tasks_results as it's set when future completes
        if task_id in st.session_state._llm_tasks_results:
            result = st.session_state._llm_tasks_results[task_id]
            if isinstance(result, Exception):
                raise result
            return result

        # Fallback for tasks that might have completed streaming but future not yet processed by get_task_status
        # This is less likely with the current structure but provides a fallback.
        # if st.session_state.get(f"task_{task_id}_stream_status") == "completed":
        #     return st.session_state.get(f"task_{task_id}_stream_content", "") # Or from _llm_tasks_full_streamed_results

        return None # Task not found or not yet completed

    def cleanup_task_result(self, task_id):
        """
        Removes a task's result from session state. Useful after result is processed.
        """
        if task_id in st.session_state._llm_tasks_results:
            del st.session_state._llm_tasks_results[task_id]
        # Clean up stream-specific states as well
        if f"task_{task_id}_stream_content" in st.session_state:
            del st.session_state[f"task_{task_id}_stream_content"]
        if f"task_{task_id}_stream_status" in st.session_state:
            del st.session_state[f"task_{task_id}_stream_status"]
        # if f"task_{task_id}_stream_error_details" in st.session_state:
        #     del st.session_state[f"task_{task_id}_stream_error_details"]
        # if task_id in st.session_state._llm_tasks_full_streamed_results: # If used
        #     del st.session_state._llm_tasks_full_streamed_results[task_id]

        logging.info(f"Cleaned up result and stream states for task {task_id}")
