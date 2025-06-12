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
                        # st.session_state[f"task_{task_id}_stream_error_details"] = chunk
                        # Store the error string itself as an Exception for get_task_result compatibility
                        return Exception(chunk) # This becomes the result of the future
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
        if task_id not in st.session_state._llm_tasks_futures:
            # Check if result is already stored (task finished and future removed)
            if task_id in st.session_state._llm_tasks_results:
                return "error" if isinstance(st.session_state._llm_tasks_results[task_id], Exception) else "completed"
            # Or if stream status indicates completion/error, but future might be gone due to quick completion
            stream_status = st.session_state.get(f"task_{task_id}_stream_status")
            if stream_status == "completed": return "completed"
            if stream_status == "error": return "error"
            return "not_found"

        future = st.session_state._llm_tasks_futures[task_id]

        if future.running():
            # For streaming tasks, "running" means the _process_streaming_task is running.
            # The actual stream progress is in st.session_state[f"task_{task_id}_stream_status"]
            # ("submitted", "streaming", "completed", "error")
            # This method can return "running" and app.py can check the stream_status for finer details.
            return "running"
        elif future.done():
            # _process_streaming_task has finished. Its return value (or exception) is the "result".
            try:
                result_or_exception = future.result() # This is what _process_streaming_task returned
                st.session_state._llm_tasks_results[task_id] = result_or_exception

                # Update stream_status one last time based on future's result if not already set by _process_streaming_task
                # This handles cases where _process_streaming_task itself might fail before setting stream_status.
                current_stream_status = st.session_state.get(f"task_{task_id}_stream_status", "unknown")
                if isinstance(result_or_exception, Exception):
                    if current_stream_status != "error": # Don't overwrite if already set by _process_streaming_task
                        st.session_state[f"task_{task_id}_stream_status"] = "error"
                    logging.error(f"Task {task_id} (wrapper) resulted in error: {result_or_exception}")
                    del st.session_state._llm_tasks_futures[task_id]
                    return "error"
                else: # Successfully completed
                    if current_stream_status not in ["completed", "error"]: # Avoid overwriting specific error from inside stream
                         st.session_state[f"task_{task_id}_stream_status"] = "completed"
                    logging.info(f"Task {task_id} (wrapper) completed. Final result stored.")
                    del st.session_state._llm_tasks_futures[task_id]
                    return "completed"

            except Exception as e: # Should ideally be caught by the future.result() line above
                st.session_state._llm_tasks_results[task_id] = e
                st.session_state[f"task_{task_id}_stream_status"] = "error" # Ensure status reflects this
                logging.error(f"Task {task_id} (wrapper) future.result() raised an unexpected exception: {e}")
                if task_id in st.session_state._llm_tasks_futures: # Ensure cleanup
                    del st.session_state._llm_tasks_futures[task_id]
                return "error"

        # Fallback based on stream status if future is somehow not running or done but still present
        # This also helps if called before future.done() is true but stream is already working
        return st.session_state.get(f"task_{task_id}_stream_status", "unknown")


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
