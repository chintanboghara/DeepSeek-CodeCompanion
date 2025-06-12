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
import os # Import os

# Initialize Task Manager
task_manager = LLMTaskManager()

# Get Ollama Base URL from environment variable or use default
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Autorefresh every 2 seconds
st_autorefresh(interval=2000, limit=None, key="llm_refresh")

# Predefined actions for the AI
PREDEFINED_ACTIONS = ["General Chat", "Explain Code", "Debug Code", "Write Documentation", "Optimize Code", "Write Unit Tests", "Translate Code"]

# Initialize session state keys for LLM engine caching and configuration
if "llm_engine_instance" not in st.session_state:
    st.session_state.llm_engine_instance = None
if "current_llm_model_name" not in st.session_state:
    st.session_state.current_llm_model_name = None
if "current_temperature" not in st.session_state:
    st.session_state.current_temperature = None
if "current_top_k" not in st.session_state:
    st.session_state.current_top_k = None
if "current_top_p" not in st.session_state:
    st.session_state.current_top_p = None
if "use_custom_dark_theme" not in st.session_state:
    st.session_state.use_custom_dark_theme = True # Default to True
if 'dedicated_code_input' not in st.session_state:
    st.session_state.dedicated_code_input = ""
if "uploaded_file_object" not in st.session_state:
    st.session_state.uploaded_file_object = None
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

# Function to load CSS
def load_css(file_name):
    """Loads a CSS file into the Streamlit application.

    Args:
        file_name (str): The path to the CSS file.
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Conditionally load custom CSS based on session state
if st.session_state.get("use_custom_dark_theme", True):
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
        tuple: A tuple containing selected_action (str), selected_model (str),
               temperature (float), top_k (int), top_p (float),
               and clear_history_button (bool).
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Action selection
        selected_action = st.selectbox(
            "Choose Action",
            PREDEFINED_ACTIONS,
            index=0, # Default to "General Chat"
            help="Select a specific task for the AI to focus on. This will tailor its responses and system prompt."
        )

        # Conditionally display dedicated code input area
        if selected_action in ["Explain Code", "Debug Code", "Optimize Code", "Write Unit Tests", "Translate Code"]:
            st.session_state.dedicated_code_input = st.text_area(
                "Paste your code here:",
                value=st.session_state.dedicated_code_input,
                height=200,
                key="dedicated_code_input_area"
            )

            uploaded_file = st.file_uploader(
                "Or upload a code file:",
                type=['py', 'js', 'java', 'c', 'cpp', 'txt', 'md', 'json', 'yaml', 'html', 'css'],
                key="code_file_uploader"
            )

            if uploaded_file is not None:
                if st.session_state.uploaded_file_name != uploaded_file.name or st.session_state.uploaded_file_object != uploaded_file:
                    try:
                        bytes_content = uploaded_file.getvalue()
                        st.session_state.uploaded_file_content = bytes_content.decode('utf-8')
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.uploaded_file_object = uploaded_file # Keep track of the object for comparison
                        st.toast(f"File '{uploaded_file.name}' uploaded and ready.", icon="📄")
                        # Optionally, clear the text area if a file is uploaded
                        # st.session_state.dedicated_code_input = ""
                    except Exception as e:
                        st.session_state.uploaded_file_content = f"Error reading file: {e}"
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.uploaded_file_object = uploaded_file
                        st.error(f"Error reading file {uploaded_file.name}: {e}")

            elif uploaded_file is None and st.session_state.uploaded_file_object is not None:
                # File was cleared from uploader
                st.session_state.uploaded_file_object = None
                st.session_state.uploaded_file_content = ""
                st.session_state.uploaded_file_name = ""
                st.toast("Uploaded file cleared.", icon="🗑️")

        st.divider() # Visual separation
        
        # Theme toggle
        st.toggle(
            "Enable Custom Dark Theme",
            value=st.session_state.get("use_custom_dark_theme", True),
            key="use_custom_dark_theme",
            help="Toggle the custom dark theme for the application. When off, Streamlit's default theme will be used."
        )
        st.divider()

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
    return selected_action, selected_model, temperature, top_k, top_p, clear_history_button

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
                if message["role"] == "ai":
                    content = message["content"]
                    parts = content.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # This part is inside ``` ... ```
                            lines = part.split('\n', 1)
                            language = None
                            code_content = part # Default to the whole part if no language line
                            if lines: # Check if there's at least one line
                                potential_lang = lines[0].strip().lower()
                                # List of common languages, extend as needed
                                known_languages = ["python", "javascript", "java", "c", "cpp", "sql", "html", "css", "shell", "bash", "json", "yaml", "markdown", ""] # Added "" for ``` block without lang
                                if potential_lang in known_languages:
                                    language = potential_lang if potential_lang else None # Use None if lang is ""
                                    if len(lines) > 1:
                                        code_content = lines[1]
                                    else:
                                        code_content = "" # No code content after language specifier
                                else:
                                    # No language specified or not recognized, treat whole block as code
                                    language = None
                                    code_content = part

                            if code_content.strip(): # Only display if there's actual code
                                st.code(code_content, language=language)
                        elif part.strip():  # This part is outside ``` ... ``` and not empty
                            st.markdown(part)
                else:  # For user messages or other roles
                    st.markdown(message["content"])

# --- App Execution Flow ---

# Display the main header
display_header()

# Initialize LocalStorage
localS = LocalStorage(key="deepseek_code_companion_chat")

# Display the sidebar and get configuration settings
selected_action, selected_model, temperature, top_k, top_p, clear_history_pressed = display_sidebar()

# Handle Clear Chat History button click
if clear_history_pressed:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
    localS.deleteItem('message_log')
    # Clear active task if any, as history is gone
    st.session_state.active_llm_task = None 
    st.rerun()

# Conditional LLM Engine Initialization & Caching
needs_reinitialization = (
    st.session_state.llm_engine_instance is None or
    selected_model != st.session_state.current_llm_model_name or
    temperature != st.session_state.current_temperature or
    top_k != st.session_state.current_top_k or
    top_p != st.session_state.current_top_p
)

if needs_reinitialization:
    with st.spinner(f"Initializing AI model: {selected_model}... Please wait."):
        llm_engine = init_llm_engine(selected_model, temperature, top_k, top_p, OLLAMA_BASE_URL)
        st.session_state.llm_engine_instance = llm_engine
        st.session_state.current_llm_model_name = selected_model
        st.session_state.current_temperature = temperature
        st.session_state.current_top_k = top_k
        st.session_state.current_top_p = top_p
else:
    llm_engine = st.session_state.llm_engine_instance

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
    final_query_content = user_query  # Start with the original query
    if selected_action in ["Explain Code", "Debug Code", "Optimize Code", "Write Unit Tests", "Translate Code"]:
        uploaded_content = st.session_state.get("uploaded_file_content", "").strip()
        dedicated_code = st.session_state.get("dedicated_code_input", "").strip()

        if uploaded_content:
            file_name = st.session_state.get("uploaded_file_name", "uploaded file")
            final_query_content = f"Code from uploaded file '{file_name}':\n```\n{uploaded_content}\n```\n\nUser question: {user_query}"
            st.session_state.dedicated_code_input = "" # Clear text area if file is used
        elif dedicated_code:
            final_query_content = f"Code to analyze:\n```\n{dedicated_code}\n```\n\nUser question: {user_query}"
        # If neither, final_query_content remains user_query

    # Add user's query to the message log
    st.session_state.message_log.append({"role": "user", "content": final_query_content})
    
    # Add a placeholder AI message
    ai_placeholder = {"role": "ai", "content": "🧠 Thinking..."}
    st.session_state.message_log.append(ai_placeholder)
    placeholder_idx = len(st.session_state.message_log) - 1
    
    # Submit to Task Manager
    # Exclude the placeholder from the prompt chain
    prompt_chain = build_prompt_chain(st.session_state.message_log[:-1], selected_action) 
    task_id = task_manager.submit_task(generate_ai_response, llm_engine, prompt_chain)
    
    # Store task info
    st.session_state.active_llm_task = {"id": task_id, "placeholder_idx": placeholder_idx}

    # Clear the dedicated code input area if the action was Explain Code or Debug Code.
    # This happens regardless of whether a file or the text area was used.
    # If a file was used, dedicated_code_input was already cleared before submission.
    # If the text area was used, this clears it.
    # If neither was used, this clears it.
    # uploaded_file_content is not cleared here, allowing follow-up on the same file.
    if selected_action in ["Explain Code", "Debug Code", "Optimize Code", "Write Unit Tests", "Translate Code"]:
        st.session_state.dedicated_code_input = ""
    
    # Clear the actual chat input field by resetting its key
    st.session_state.user_query_input = "" 
    
    # Rerun to immediately show the user message and placeholder, and disable input
    st.rerun()
elif user_query and not user_query.strip(): # If user_query exists but is only whitespace
    st.toast("⚠️ Input cannot be empty or whitespace.", icon="🚫")
