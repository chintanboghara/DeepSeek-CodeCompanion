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

    def submit_task(self, func, *args, **kwargs):
        """
        Submits a function to the executor and returns a unique task ID.
        Stores the future in st.session_state._llm_tasks_futures.
        """
        task_id = str(uuid.uuid4())
        future = self._executor.submit(func, *args, **kwargs)
        st.session_state._llm_tasks_futures[task_id] = future
        logging.info(f"Task {task_id} submitted for function {func.__name__}")
        return task_id

    def get_task_status(self, task_id):
        """
        Returns the status of the task ("running", "completed", "error", or "not_found").
        If completed or error, stores result/exception in st.session_state._llm_tasks_results
        and removes future from st.session_state._llm_tasks_futures.
        """
        if task_id not in st.session_state._llm_tasks_futures:
            # Check if it was completed and future removed, but result is available
            if task_id in st.session_state._llm_tasks_results:
                 # If result is an Exception, it's an error status
                return "error" if isinstance(st.session_state._llm_tasks_results[task_id], Exception) else "completed"
            return "not_found"

        future = st.session_state._llm_tasks_futures[task_id]

        if future.running():
            return "running"
        elif future.done():
            try:
                result = future.result()
                st.session_state._llm_tasks_results[task_id] = result
                logging.info(f"Task {task_id} completed. Result stored.")
                del st.session_state._llm_tasks_futures[task_id] # Clean up future
                return "completed"
            except Exception as e:
                st.session_state._llm_tasks_results[task_id] = e # Store exception as result
                logging.error(f"Task {task_id} resulted in error: {e}")
                del st.session_state._llm_tasks_futures[task_id] # Clean up future
                return "error"
        return "unknown" # Should ideally not happen

    def get_task_result(self, task_id):
        """
        Returns the result of a completed task from st.session_state._llm_tasks_results.
        Returns None if task is not found or result not yet available (should check status first).
        Raises the exception if the task resulted in an error.
        """
        result = st.session_state._llm_tasks_results.get(task_id)
        if isinstance(result, Exception):
            raise result # Re-raise the exception so caller can handle
        return result

    def cleanup_task_result(self, task_id):
        """
        Removes a task's result from session state. Useful after result is processed.
        """
        if task_id in st.session_state._llm_tasks_results:
            del st.session_state._llm_tasks_results[task_id]
            logging.info(f"Cleaned up result for task {task_id}")


# To ensure the singleton is initialized when the module is imported.
# This allows app.py to just import and use it.
# get_llm_task_manager = LLMTaskManager() # This is not how singletons are typically "pre-initialized" for use.
# Instead, the app will call LLMTaskManager() and st.experimental_singleton handles it.
