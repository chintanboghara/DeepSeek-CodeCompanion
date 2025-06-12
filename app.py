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
from datetime import datetime # Added for chat history formatting

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
if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = "New Session (Unsaved)"
if "session_name_input_value" not in st.session_state:
    st.session_state.session_name_input_value = ""
if "selected_session_to_load" not in st.session_state:
    st.session_state.selected_session_to_load = "" # Keep it simple, default to empty string
if "selected_session_to_delete" not in st.session_state:
    st.session_state.selected_session_to_delete = "" # Keep it simple
if 'last_loaded_session' not in st.session_state:
    st.session_state.last_loaded_session = ""

# --- Global Constants and Default Settings ---
DEFAULT_MODEL = "deepseek-r1:1.5b"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9
MODEL_OPTIONS = ["deepseek-r1:1.5b", "deepseek-r1:3b"]

# Initialize LocalStorage with the new key for session data
localS = LocalStorage(key="deepseek_code_companion_sessions_v2")

# --- Initialize Session State for Loaded Settings ---
# These will hold the values loaded from storage to initialize widgets.
# They are set once at the beginning of the session.
if 'loaded_selected_model' not in st.session_state:
    st.session_state.loaded_selected_model = DEFAULT_MODEL
if 'loaded_temperature' not in st.session_state:
    st.session_state.loaded_temperature = DEFAULT_TEMPERATURE
if 'loaded_top_k' not in st.session_state:
    st.session_state.loaded_top_k = DEFAULT_TOP_K
if 'loaded_top_p' not in st.session_state:
    st.session_state.loaded_top_p = DEFAULT_TOP_P
if 'settings_loaded' not in st.session_state:
    st.session_state.settings_loaded = False

# --- Helper Functions for Local Storage Interaction ---
def get_storage_data():
    """Retrieves the entire session data object from local storage."""
    storage_data = localS.getItem("chat_sessions_data")
    # Using global defaults defined at the top of the script
    global_defaults = {
        "selected_model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "top_k": DEFAULT_TOP_K,
        "top_p": DEFAULT_TOP_P
    }
    if storage_data is None:
        storage_data = {"sessions": {}, "last_active_session_name": None, "global_settings": global_defaults.copy()}

    if "sessions" not in storage_data:
        storage_data["sessions"] = {}

    if "global_settings" not in storage_data:
        storage_data["global_settings"] = global_defaults.copy()
    else: # Ensure all keys exist in global_settings, add if missing
        for key, value in global_defaults.items():
            if key not in storage_data["global_settings"]:
                storage_data["global_settings"][key] = value
    return storage_data

def save_storage_data(data):
    """Saves the entire session data object to local storage."""
    localS.setItem("chat_sessions_data", data)

def get_all_session_names():
    """Retrieves a sorted list of all saved session names."""
    data = get_storage_data()
    return sorted(list(data.get("sessions", {}).keys()))

def format_chat_history_for_markdown(message_log, session_name):
    """Formats the chat history for Markdown export."""
    formatted_lines = []

    # Determine title: Use session name or a generic title with timestamp
    if session_name and session_name != "New Session (Unsaved)":
        title = session_name
    else:
        title = f"Chat Conversation ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"

    formatted_lines.append(f"# Chat Session: {title}\n\n")

    for message in message_log:
        formatted_lines.append("---\n\n") # Separator

        role = message["role"].upper() # USER or AI
        content = message["content"]

        formatted_lines.append(f"**{role}:**\n\n{content}\n\n")

    return "".join(formatted_lines)
# --- End Helper Functions ---

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
        default_model_index = 0
        try:
            default_model_index = MODEL_OPTIONS.index(st.session_state.loaded_selected_model)
        except ValueError: # If loaded model isn't in options
            default_model_index = MODEL_OPTIONS.index(DEFAULT_MODEL) # Fallback
        selected_model = st.selectbox("Choose Model", MODEL_OPTIONS, index=default_model_index)

        # Temperature slider for controlling LLM randomness
        temperature = st.slider("Select Temperature", min_value=0.0, max_value=1.0, value=st.session_state.loaded_temperature, step=0.1)

        # Top K input
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=st.session_state.loaded_top_k, step=1,
                                help="Controls diversity. Limits the set of next tokens to the K most probable.")

        # Top P slider
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=st.session_state.loaded_top_p, step=0.01,
                          help="Controls diversity via nucleus sampling. Selects tokens with cumulative probability > P.")
        st.divider()
        # Clear chat history button
        clear_history_button = st.button("Clear Chat History")
        st.divider()
        st.markdown("### Model Capabilities")
        st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
        st.divider()
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
        st.divider() # Visual separation before session management

        # --- Session Management UI ---
        st.header("💾 Chat Sessions")
        st.caption(f"Current Session: {st.session_state.get('current_session_name', 'New Session (Unsaved)')}")

        # Save Session
        st.session_state.session_name_input_value = st.text_input(
            "Enter Session Name to Save/Overwrite:",
            value=st.session_state.get('session_name_input_value', st.session_state.get('current_session_name', 'New Session (Unsaved)') if st.session_state.get('current_session_name') != 'New Session (Unsaved)' else ''),
            key="session_name_text_input",
            placeholder="Enter a name for this session"
        )
        # Capture button clicks directly for logic in the main flow
        save_button_clicked = st.button("Save Current Session", key="save_session_button_logic")

        # Load Session
        saved_session_names = get_all_session_names()
        st.session_state.selected_session_to_load = st.selectbox(
            "Load Session:",
            options=[""] + saved_session_names, # Add empty option for placeholder
            key="load_session_selectbox",
            index=0, # Default to "Select a session"
            format_func=lambda x: "Select a session" if x == "" else x
        )
        # load_button_clicked will be implicitly handled by checking st.session_state.selected_session_to_load in main logic

        # Delete Session
        st.session_state.selected_session_to_delete = st.selectbox(
            "Delete Session:",
            options=[""] + saved_session_names, # Add empty option for placeholder
            key="delete_session_selectbox",
            index=0, # Default to "Select session to delete"
            format_func=lambda x: "Select session to delete" if x == "" else x
        )
        # Capture button clicks directly for logic in the main flow
        delete_button_clicked = st.button("Delete Selected Session", key="delete_session_button_logic")

        st.divider() # Divider before download button

        # Download Chat History Button
        if st.session_state.get("message_log"): # Only show if there's a history
            markdown_chat_history = format_chat_history_for_markdown(
                st.session_state.message_log,
                st.session_state.get("current_session_name", "Chat")
            )
            session_file_name_part = st.session_state.get("current_session_name", "Chat").replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
            download_file_name = f"{session_file_name_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            st.download_button(
                label="📥 Download Chat History",
                data=markdown_chat_history,
                file_name=download_file_name,
                mime="text/markdown",
                key="download_chat_button"
            )
        st.divider()


    # Return button states along with other config; selected values are in session_state
    return selected_action, selected_model, temperature, top_k, top_p, clear_history_button, save_button_clicked, delete_button_clicked

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

# Load settings from local storage ONCE upon app initialization/session start
if not st.session_state.settings_loaded:
    storage_data = get_storage_data()
    global_settings = storage_data.get("global_settings", {})

    st.session_state.loaded_selected_model = global_settings.get("selected_model", DEFAULT_MODEL)
    # Ensure loaded_selected_model is valid, otherwise default
    if st.session_state.loaded_selected_model not in MODEL_OPTIONS:
        st.session_state.loaded_selected_model = DEFAULT_MODEL

    st.session_state.loaded_temperature = float(global_settings.get("temperature", DEFAULT_TEMPERATURE))
    st.session_state.loaded_top_k = int(global_settings.get("top_k", DEFAULT_TOP_K))
    st.session_state.loaded_top_p = float(global_settings.get("top_p", DEFAULT_TOP_P))

    st.session_state.settings_loaded = True
    # Optional: Trigger a rerun if settings loaded modify session state that widgets depend on for their *initial* render.
    # st.rerun() # Usually not needed if session state is set before widget rendering.

# Display the main header
display_header()

# Display the sidebar and get configuration settings
# Sidebar widgets will now use the 'loaded_*' session state values as their defaults
selected_action, selected_model, temperature, top_k, top_p, clear_history_pressed, save_button_clicked, delete_button_clicked = display_sidebar()

# --- Save Configuration Settings on Change ---
storage_data_for_settings = get_storage_data()
current_global_settings = storage_data_for_settings.get("global_settings", {}).copy() # Use .copy()

settings_changed = False
if current_global_settings.get("selected_model") != selected_model:
    current_global_settings["selected_model"] = selected_model
    settings_changed = True
if current_global_settings.get("temperature") != temperature:
    current_global_settings["temperature"] = temperature
    settings_changed = True
if current_global_settings.get("top_k") != top_k:
    current_global_settings["top_k"] = top_k
    settings_changed = True
if current_global_settings.get("top_p") != top_p:
    current_global_settings["top_p"] = top_p
    settings_changed = True

if settings_changed:
    storage_data_for_settings["global_settings"] = current_global_settings
    save_storage_data(storage_data_for_settings)
    # st.toast("Settings saved to local storage!", icon="⚙️") # Optional for debugging

# --- Session Management Logic ---

# Save Session Logic
if save_button_clicked:
    session_name_to_save = st.session_state.session_name_input_value.strip()
    if not session_name_to_save:
        st.toast("⚠️ Session name cannot be empty.", icon="🚫")
    elif session_name_to_save in ["Select a session", "Select session to delete", "New Session (Unsaved)"]: # Prevent saving with placeholder names
        st.toast(f"⚠️ Invalid session name: '{session_name_to_save}'. Please choose a different name.", icon="🚫")
    else:
        data = get_storage_data()
        data["sessions"][session_name_to_save] = st.session_state.message_log
        # data["last_active_session_name"] = session_name_to_save # Optional
        save_storage_data(data)
        st.session_state.current_session_name = session_name_to_save
        st.toast(f"Session '{session_name_to_save}' saved successfully!", icon="✅")
        st.session_state.selected_session_to_load = ""
        st.session_state.selected_session_to_delete = ""
        st.rerun()

# Load Session Logic
selected_session_to_load_value = st.session_state.get("selected_session_to_load", "")
if selected_session_to_load_value and selected_session_to_load_value != st.session_state.get('last_loaded_session', ''):
    data = get_storage_data()
    if selected_session_to_load_value in data["sessions"]:
        st.session_state.message_log = data["sessions"][selected_session_to_load_value]
        st.session_state.current_session_name = selected_session_to_load_value
        st.session_state.session_name_input_value = selected_session_to_load_value
        st.session_state.active_llm_task = None
        st.session_state.last_loaded_session = selected_session_to_load_value
        st.toast(f"Session '{selected_session_to_load_value}' loaded.", icon="✅")
        st.session_state.selected_session_to_load = "" # Reset selectbox choice
        st.rerun()
    else:
        st.toast(f"Error: Could not load session '{selected_session_to_load_value}'.", icon="❌")
        st.session_state.selected_session_to_load = "" # Reset selectbox choice
elif not selected_session_to_load_value:
    st.session_state.last_loaded_session = ""

# Delete Session Logic
if delete_button_clicked:
    session_name_to_delete = st.session_state.selected_session_to_delete
    if not session_name_to_delete: # Check if it's an empty string (placeholder selected)
        st.toast("⚠️ No session selected for deletion.", icon="🚫")
    else:
        data = get_storage_data()
        if session_name_to_delete in data["sessions"]:
            del data["sessions"][session_name_to_delete]
            # if data.get("last_active_session_name") == session_name_to_delete:
            #     data["last_active_session_name"] = None
            save_storage_data(data)
            st.toast(f"Session '{session_name_to_delete}' deleted.", icon="🗑️")

            if st.session_state.current_session_name == session_name_to_delete:
                st.session_state.current_session_name = "New Session (Unsaved)"
                st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
                st.session_state.session_name_input_value = ""
                st.session_state.active_llm_task = None

            st.session_state.selected_session_to_load = ""
            st.session_state.selected_session_to_delete = "" # Reset this selectbox
            st.rerun()
        else:
            st.toast(f"Error: Could not find session '{session_name_to_delete}' to delete.", icon="❌")
            st.session_state.selected_session_to_delete = "" # Reset selectbox

# Handle Clear Chat History button click (now also resets current session name)
if clear_history_pressed:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
    st.session_state.current_session_name = "New Session (Unsaved)"
    st.session_state.session_name_input_value = ""
    # localS.deleteItem('message_log') # This was for the old single-session storage
    st.session_state.active_llm_task = None 
    st.toast("Current session cleared and reset.", icon="✨")
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

# Session State Management for chat history and active task (Initial Load)
if "message_log" not in st.session_state:
    # On first load, or if message_log is somehow cleared without session context,
    # initialize a fresh, unnamed session.
    # Specific session loading logic (including potentially last active) would go here if desired,
    # but for now, we start fresh or rely on explicit load.
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
    st.session_state.current_session_name = "New Session (Unsaved)"
    st.session_state.session_name_input_value = ""

if "active_llm_task" not in st.session_state:
    st.session_state.active_llm_task = None

# Result Checking and UI Update (on each rerun, including autorefresh)
if st.session_state.active_llm_task:
    task_id = st.session_state.active_llm_task["id"]
    placeholder_idx = st.session_state.active_llm_task["placeholder_idx"]

    # Check stream status from session state (populated by task_manager._process_streaming_task)
    stream_status = st.session_state.get(f"task_{task_id}_stream_status")
    current_stream_content = st.session_state.get(f"task_{task_id}_stream_content", "")

    # Update the placeholder content with the latest streamed content
    if st.session_state.message_log[placeholder_idx]["content"] != current_stream_content and current_stream_content:
        st.session_state.message_log[placeholder_idx]["content"] = current_stream_content
        # Autorefresh will handle the rerun to display the latest chunk.

    if stream_status == "streaming":
        # Content is being updated by the block above.
        # Keep input disabled, spinner active (implicitly via placeholder text or explicit spinner).
        pass # Wait for st_autorefresh to pick up next chunk

    elif stream_status == "completed":
        # Final content is already in current_stream_content and set in message_log.
        # Now, finalize the task with task_manager to get the official result (which should match)
        # and perform cleanup.
        try:
            final_content = task_manager.get_task_result(task_id) # This also raises if task had an internal error
            st.session_state.message_log[placeholder_idx]["content"] = final_content
        except Exception as e: # Should ideally be caught by stream_status == "error"
            logging.error(f"Error retrieving final result for completed stream task {task_id}: {e}")
            # This indicates a discrepancy, handle as an error
            st.session_state.critical_error_message = f"Error finalizing task: {str(e)}"
            st.session_state.message_log[placeholder_idx]["content"] = "🤖 Error finalizing response."

        task_manager.cleanup_task_result(task_id)
        st.session_state.active_llm_task = None
        st.rerun() # Rerun to enable chat input and reflect final state

    elif stream_status == "error":
        raw_error_details = "Unknown streaming error"
        try:
            # This will raise the exception stored by _process_streaming_task in self.results[task_id]
            task_manager.get_task_result(task_id) 
        except Exception as e:
            raw_error_details = str(e) # This should be the prefixed error string or exception message

        user_friendly_error = f"An unexpected error occurred: {raw_error_details}" # Default
        if raw_error_details.startswith("OLLAMA_CONNECTION_ERROR:"):
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            user_friendly_error = (
                "**Failed to Connect to AI Model**

"
                f"Could not connect to the Ollama server. Please ensure:
"
                f"1. Ollama is installed and running on your system.
"
                f"2. The Ollama server is accessible at the configured URL: **{ollama_url}**.
"
                f"3. The selected model (`{selected_model}`) is available in your Ollama instance."
            )
        elif raw_error_details.startswith("LLM_RUNTIME_ERROR:"):
            user_friendly_error = f"A runtime error occurred with the AI model: {raw_error_details.replace('LLM_RUNTIME_ERROR:', '').strip()}"
        elif raw_error_details.startswith("LLM_UNEXPECTED_ERROR:"):
            user_friendly_error = f"An unexpected error occurred with the AI model: {raw_error_details.replace('LLM_UNEXPECTED_ERROR:', '').strip()}"

        st.session_state.critical_error_message = user_friendly_error
        st.session_state.message_log[placeholder_idx]["content"] = "🤖 I encountered an issue. Please see the error message displayed."

        task_manager.cleanup_task_result(task_id)
        st.session_state.active_llm_task = None
        st.rerun() # Rerun to show error and enable input

    # Fallback for initial "submitted" status or if stream_status is somehow missed by autorefresh cycles
    # before task_manager sets it to "streaming", "completed", or "error".
    # This also covers the brief period after submission but before _process_streaming_task starts.
    elif stream_status == "submitted" or stream_status is None:
        # Check the underlying future status if stream status isn't definitive yet
        # This is more of a safeguard or for very fast non-streaming-like errors from the wrapper.
        underlying_status = task_manager.get_task_status(task_id) # Checks future.done() etc.
        if underlying_status == "completed": # Wrapper finished, means stream completed or errored out early
            try:
                final_content = task_manager.get_task_result(task_id)
                st.session_state.message_log[placeholder_idx]["content"] = final_content
                if st.session_state.get(f"task_{task_id}_stream_status") != "error": # Ensure we don't overwrite an error status
                    st.session_state[f"task_{task_id}_stream_status"] = "completed" # Mark explicitly
            except Exception as e:
                # Error handling similar to stream_status == "error"
                raw_error_details = str(e)
                user_friendly_error = f"An unexpected error occurred: {raw_error_details}"
                if raw_error_details.startswith("OLLAMA_CONNECTION_ERROR:"):
                    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                    user_friendly_error = (
                        "**Failed to Connect to AI Model**

"
                        f"Could not connect to the Ollama server. Please ensure:
"
                        f"1. Ollama is installed and running on your system.
"
                        f"2. The Ollama server is accessible at the configured URL: **{ollama_url}**.
"
                        f"3. The selected model (`{selected_model}`) is available in your Ollama instance."
                    )
                elif raw_error_details.startswith("LLM_RUNTIME_ERROR:"):
                    user_friendly_error = f"A runtime error occurred with the AI model: {raw_error_details.replace('LLM_RUNTIME_ERROR:', '').strip()}"
                elif raw_error_details.startswith("LLM_UNEXPECTED_ERROR:"):
                    user_friendly_error = f"An unexpected error occurred with the AI model: {raw_error_details.replace('LLM_UNEXPECTED_ERROR:', '').strip()}"
                st.session_state.critical_error_message = user_friendly_error
                st.session_state.message_log[placeholder_idx]["content"] = "🤖 I encountered an issue. Please see the error message displayed."
                st.session_state[f"task_{task_id}_stream_status"] = "error" # Mark explicitly

            task_manager.cleanup_task_result(task_id)
            st.session_state.active_llm_task = None
            st.rerun()
        elif underlying_status == "error": # Wrapper itself failed catastrophically or error caught by get_task_status
            raw_error_details = "Unknown error from task manager"
            try:
                task_manager.get_task_result(task_id)
            except Exception as e:
                raw_error_details = str(e)
            # (Error parsing and display logic as above)
            st.session_state.critical_error_message = f"A task execution error occurred: {raw_error_details}"
            st.session_state.message_log[placeholder_idx]["content"] = "🤖 I encountered an issue. Please see the error message displayed."
            st.session_state[f"task_{task_id}_stream_status"] = "error" # Mark explicitly
            task_manager.cleanup_task_result(task_id)
            st.session_state.active_llm_task = None
            st.rerun()
        # else, it's "running" or "submitted", so just wait for autorefresh.

# Display the existing chat messages
display_chat_interface(st.session_state.message_log)

# Display critical error message if it exists (below chat, above input)
if st.session_state.get("critical_error_message"):
    st.error(st.session_state.critical_error_message, icon="🚨")

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
