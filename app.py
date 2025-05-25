"""
Streamlit front-end for the DeepSeek Code Companion application.

This module handles the user interface, including displaying the chat,
configuring the LLM model, and managing user interactions.
It uses `llm_logic.py` for the core language model interactions.
"""
import streamlit as st
from llm_logic import (
    init_llm_engine,
    build_prompt_chain,
    generate_ai_response
)

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
        tuple: A tuple containing the selected model (str) and temperature (float).
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Model selection dropdown
        selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
        # Temperature slider for controlling LLM randomness
        temperature = st.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        st.divider()
        st.markdown("### Model Capabilities")
        st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
        st.divider()
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    return selected_model, temperature

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

# Display the sidebar and get model/temperature settings
selected_model, temperature = display_sidebar()

# Initialize the LLM engine with selected settings
llm_engine = init_llm_engine(selected_model, temperature)

# Session State Management for chat history
# Initialize message_log in session_state if it doesn't exist
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Display the existing chat messages
display_chat_interface(st.session_state.message_log)

# User Input Handling
# Get user input from the chat_input widget
user_query = st.chat_input("Type your coding question here...")

# Process user input if it's valid (not empty or just whitespace)
if user_query and user_query.strip():
    # Add user's query to the message log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Show a spinner while processing the query
    with st.spinner("🧠 Processing..."):
        # Build the prompt chain based on the current message log
        prompt_chain = build_prompt_chain(st.session_state.message_log)
        # Generate AI response using the LLM engine and prompt chain
        ai_response = generate_ai_response(llm_engine, prompt_chain)

    # Add AI's response to the message log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    # Rerun the Streamlit app to update the chat interface
    st.rerun()
elif user_query:  # If user_query exists but is only whitespace
    # Display a toast message for invalid input
    st.toast("⚠️ Input cannot be empty or whitespace.", icon="🚫")
