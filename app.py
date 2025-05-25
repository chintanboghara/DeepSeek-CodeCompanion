"""
Streamlit front-end for the DeepSeek Code Companion application.

This module handles the user interface, including displaying the chat,
configuring the LLM model, and managing user interactions.
It uses `llm_logic.py` for the core language model interactions.
"""
import streamlit as st
from streamlit_local_storage import LocalStorage # Import LocalStorage
from streamlit_autorefresh import st_autorefresh # Import autorefresh
from llm_logic import (
    init_llm_engine,
    build_prompt_chain,
    generate_ai_response
)
from task_manager import LLMTaskManager # Import TaskManager
import logging # Import logging

# Initialize Task Manager
task_manager = LLMTaskManager()

# Autorefresh every 2 seconds
st_autorefresh(interval=2000, limit=None, key="llm_refresh")

# Function to load CSS
def load_css(file_name):
    """Loads a CSS file into the Streamlit application.

    Args:
        file_name (str): The path to the CSS file.
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS from styles.css
load_css("styles.css")

# UI Components
def display_header():
    """Displays the main header and caption of the application."""
    st.title("🧠 DeepSeek Code Companion")
    st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

def display_sidebar():
    """
    Displays the sidebar with configuration options.

    Returns:
        tuple: A tuple containing selected_model (str), temperature (float),
               top_k (int), top_p (float), and clear_history_button (bool).
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Model selection dropdown
        selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
        # Temperature slider for controlling LLM randomness
        temperature = st.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        # Top K input
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=40, step=1,
                                help="Controls diversity. Limits the set of next tokens to the K most probable.")
        # Top P slider
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.01,
                          help="Controls diversity via nucleus sampling. Selects tokens with cumulative probability > P.")
        st.divider()
        # Clear chat history button
        clear_history_button = st.button("Clear Chat History")
        st.divider()
        st.markdown("### Model Capabilities")
        st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
        st.divider()
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    return selected_model, temperature, top_k, top_p, clear_history_button

def display_chat_interface(message_log):
    """
    Displays the chat interface with messages from the session state.

    Args:
        message_log (list): A list of dictionaries, where each dictionary
                              represents a chat message with "role" and "content".
    """
    chat_container = st.container()
    with chat_container:
        for message in message_log:
            with st.chat_message(message["role"]): # Display message with role (user/ai)
                st.markdown(message["content"])

# --- App Execution Flow ---

# Display the main header
display_header()

# Initialize LocalStorage
localS = LocalStorage(key="deepseek_code_companion_chat")

# Display the sidebar and get configuration settings
selected_model, temperature, top_k, top_p, clear_history_pressed = display_sidebar()

# Handle Clear Chat History button click
if clear_history_pressed:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
    localS.deleteItem('message_log')
    # Clear active task if any, as history is gone
    st.session_state.active_llm_task = None 
    st.rerun()

# Initialize the LLM engine with selected settings
llm_engine = init_llm_engine(selected_model, temperature, top_k, top_p)

# Session State Management for chat history and active task
if "message_log" not in st.session_state:
    # Try to load chat history from local storage
    stored_message_log = localS.getItem('message_log')
    if stored_message_log:
        st.session_state.message_log = stored_message_log
    else:
        # Default welcome message if no history is found
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

if "active_llm_task" not in st.session_state:
    st.session_state.active_llm_task = None

# Result Checking and UI Update (on each rerun, including autorefresh)
if st.session_state.active_llm_task:
    task_id = st.session_state.active_llm_task["id"]
    placeholder_idx = st.session_state.active_llm_task["placeholder_idx"]
    status = task_manager.get_task_status(task_id)

    if status == "completed":
        ai_response = task_manager.get_task_result(task_id)
        st.session_state.message_log[placeholder_idx]["content"] = ai_response
        task_manager.cleanup_task_result(task_id)
        st.session_state.active_llm_task = None
        localS.setItem('message_log', st.session_state.message_log) # Save final history
        st.rerun()
    elif status == "error":
        error_details_val = "Unknown error" # Default error message
        try:
            # This re-raises the exception, so we catch it to get details
            task_manager.get_task_result(task_id) 
        except Exception as e:
            error_details_val = str(e)
        st.session_state.message_log[placeholder_idx]["content"] = f"⚠️ Error: {error_details_val}"
        task_manager.cleanup_task_result(task_id)
        st.session_state.active_llm_task = None
        localS.setItem('message_log', st.session_state.message_log) # Save history with error
        st.rerun()
    elif status == "not_found" and task_id in st.session_state.get('_llm_tasks_results', {}):
        # This case implies status was checked after result was processed and future removed
        # but before active_llm_task was cleared. Or a general logic error.
        # For safety, clear active_llm_task if result is present
        logging.warning(f"Task {task_id} was not found in futures but result exists. Clearing active task.")
        # Attempt to finalize the message if possible, or mark as unknown error
        if task_id in st.session_state._llm_tasks_results:
            result = st.session_state._llm_tasks_results[task_id]
            if isinstance(result, Exception):
                 st.session_state.message_log[placeholder_idx]["content"] = f"⚠️ Error: {str(result)}"
            else:
                 st.session_state.message_log[placeholder_idx]["content"] = result
            localS.setItem('message_log', st.session_state.message_log) # Save history
        task_manager.cleanup_task_result(task_id) # ensure cleanup
        st.session_state.active_llm_task = None
        st.rerun()

# Display the existing chat messages
display_chat_interface(st.session_state.message_log)

# User Input Handling
chat_input_disabled = st.session_state.active_llm_task is not None
user_query = st.chat_input("Type your coding question here...", disabled=chat_input_disabled, key="user_query_input")

if user_query and user_query.strip() and not chat_input_disabled:
    # Add user's query to the message log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Add a placeholder AI message
    ai_placeholder = {"role": "ai", "content": "🧠 Thinking..."}
    st.session_state.message_log.append(ai_placeholder)
    placeholder_idx = len(st.session_state.message_log) - 1
    
    # Submit to Task Manager
    # Exclude the placeholder from the prompt chain
    prompt_chain = build_prompt_chain(st.session_state.message_log[:-1]) 
    task_id = task_manager.submit_task(generate_ai_response, llm_engine, prompt_chain)
    
    # Store task info
    st.session_state.active_llm_task = {"id": task_id, "placeholder_idx": placeholder_idx}
    
    # Clear the actual chat input field by resetting its key
    st.session_state.user_query_input = "" 
    
    # Rerun to immediately show the user message and placeholder, and disable input
    st.rerun()
elif user_query and not user_query.strip(): # If user_query exists but is only whitespace
    st.toast("⚠️ Input cannot be empty or whitespace.", icon="🚫")
