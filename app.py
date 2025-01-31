import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS Styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea, .stSelectbox div[data-baseweb="select"],
    .stSelectbox option, div[role="listbox"] div { 
        color: white !important; background-color: #3d3d3d !important;
    }
    .stSelectbox svg { fill: white !important; }
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize Chat Engine
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)

# System Prompt Configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session State Management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Function: Build Chat Prompt Chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Function: Generate AI Response
def generate_ai_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Chat Container (Displays Message Log)
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Input Handling
user_query = st.chat_input("Type your coding question here...")
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
