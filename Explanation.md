This Streamlit application is a **chat-based AI coding assistant** built using **LangChain, Ollama, and DeepSeek models**. It provides coding assistance, debugging support, and solution design capabilities. Below is a detailed breakdown of the components:

## **1️⃣ Importing Necessary Libraries**
```python
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
```
- **`streamlit`**: Used to create a web-based interface for the AI chatbot.
- **`langchain_ollama.ChatOllama`**: Allows interaction with the Ollama API, which runs the DeepSeek language model.
- **`langchain_core.output_parsers.StrOutputParser`**: Converts model outputs into plain text.
- **Prompt Templates (`SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, `AIMessagePromptTemplate`)**:
  - These help structure and format conversation prompts in a chat-like sequence.
  - `SystemMessagePromptTemplate` defines the AI's role.
  - `HumanMessagePromptTemplate` stores user messages.
  - `AIMessagePromptTemplate` stores AI responses.
- **`ChatPromptTemplate`**: Combines messages into a structured chat sequence.

---

## **2️⃣ Applying Custom CSS for Styling**
```python
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
```
- This **custom CSS** changes the color scheme:
  - **Dark mode** for the main background (`#1a1a1a`).
  - **Gray theme** for the sidebar (`#2d2d2d`).
  - **White text on dark backgrounds** for input fields and select boxes.

---

## **3️⃣ App Title and Description**
```python
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")
```
- Sets the **title** and **subtitle** of the app.

---

## **4️⃣ Sidebar Configuration**
```python
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
```
- **Dropdown (`st.selectbox`)**: Lets the user select between two DeepSeek AI models (`1.5B` and `3B` parameter versions).
- **Markdown (`st.markdown`)**:
  - Lists model capabilities (Python expertise, debugging, documentation, solution design).
  - Provides links to **Ollama** and **LangChain**.

---

## **5️⃣ Initializing the Chat Engine**
```python
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)
```
- **`ChatOllama`** initializes the AI model:
  - Uses the selected model (`deepseek-r1:1.5b` or `deepseek-r1:3b`).
  - **`base_url="http://localhost:11434"`** connects to a locally hosted Ollama API.
  - **`temperature=0.3`** makes responses deterministic and focused.

---

## **6️⃣ Configuring System Prompt (AI Personality)**
```python
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)
```
- This **defines the AI’s role**:
  - **Coding expert**.
  - **Provides debugging tips**.
  - **Responds only in English**.

---

## **7️⃣ Session State Management**
```python
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
```
- Uses **Streamlit's `session_state`** to store chat history across reruns.
- Initializes the conversation with an AI greeting message.

---

## **8️⃣ Chat Prompt Chain (Conversation Memory)**
```python
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)
```
- **Combines the conversation history** into a structured prompt sequence.
- Appends:
  - **System prompt**.
  - **User messages** (`HumanMessagePromptTemplate`).
  - **AI responses** (`AIMessagePromptTemplate`).
- Returns a **structured chat prompt** (`ChatPromptTemplate`) for AI processing.

---

## **9️⃣ AI Response Generation**
```python
def generate_ai_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"
```
- **Processes user queries**:
  - `prompt_chain | llm_engine | StrOutputParser()`:
    - **Feeds prompt sequence into the AI model**.
    - **Parses the AI response into plain text**.
- **Error handling**: Returns a warning message if the AI fails to respond.

---

## **🔟 Displaying the Chat History**
```python
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
```
- **Displays the chat history** using `st.chat_message()`.
- **Iterates over past messages**, displaying user and AI responses accordingly.

---

## **1️⃣1️⃣ Handling User Input**
```python
user_query = st.chat_input("Type your coding question here...")
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
```
- **Captures user input** via `st.chat_input()`.
- **Stores the user’s message** in session state.
- **Displays a loading spinner** (`st.spinner()`) while processing.
- **Calls `generate_ai_response()`** to get AI’s answer.
- **Appends AI’s response** to the chat log.
- **`st.rerun()` refreshes the UI** to display new messages.

---

## **💡 Summary**
### **What This Code Does**
- Creates a **chat-based coding assistant** with debugging support.
- Uses **LangChain + Ollama** to handle conversations.
- Implements a **memory system** to maintain chat history.
- Provides **custom theming** for dark mode styling.
- **Processes user queries** and generates AI responses dynamically.

### **How It Works**
1. **User asks a coding question**.
2. **Chat history + system instructions** are formatted into a structured prompt.
3. **DeepSeek AI model** generates a response.
4. **AI response is displayed** in a chat format.
5. **Session state maintains chat history** for continuity.

Would you like any improvements or feature additions? 🚀
