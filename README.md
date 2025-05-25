# DeepSeek Code Companion

**AI-powered pair programming assistant**

[![CodeQL Advanced](https://github.com/chintanboghara/DeepSeek-CodeCompanion/actions/workflows/codeql.yml/badge.svg)](https://github.com/chintanboghara/DeepSeek-CodeCompanion/actions/workflows/codeql.yml)
[![Dependency review](https://github.com/chintanboghara/DeepSeek-CodeCompanion/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/chintanboghara/DeepSeek-CodeCompanion/actions/workflows/dependency-review.yml)

**DeepSeek Code Companion** is an AI-powered pair programming assistant designed to assist with debugging, code documentation, and solution design. Built using **Streamlit**, **LangChain**, and **Ollama**, this tool integrates **DeepSeek models** to deliver intelligent, local coding assistance.

The application is structured for clarity, with the user interface managed by `app.py`, core LLM interactions handled by `llm_logic.py`, and styling defined in `styles.css`.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Ollama (Local LLM engine): [Download Ollama](https://ollama.ai/)

### Steps

1. **Clone the repository**

   ```sh
   git clone https://github.com/chintanboghara/DeepSeek-CodeCompanion.git
   cd DeepSeek-CodeCompanion
   ```

2. **Set up a virtual environment (Optional but recommended)**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**

   The `requirements.txt` file lists all necessary Python packages, including Streamlit, LangChain components, and Requests.
   ```sh
   pip install -r requirements.txt
   ```

4. **Run Ollama in the background**

   Ensure Ollama is running. If not, start it with:

   ```sh
   ollama run
   ```

   Pull the required DeepSeek models if you haven’t already:

   ```sh
   ollama pull deepseek-r1:1.5b
   ollama pull deepseek-r1:3b
   ```

5. **Launch the Streamlit app**

   ```sh
   streamlit run app.py
   ```

6. **Interact with the AI Assistant**

   Open the link provided in the terminal (usually `http://localhost:8501`) in your web browser. Type in your coding-related queries to receive AI-powered assistance.

## Features and Usage

Once the app is running, interact with the AI Assistant via the web interface:

- **Chat Interface**: Type your coding-related questions into the input box at the bottom of the screen.
- **Model Selection**: Choose between different DeepSeek models from the sidebar.
- **Temperature Control**: Adjust the "Select Temperature" slider in the sidebar (range 0.0 to 1.0, default 0.3).
    - Lower values (e.g., 0.1-0.3) produce more focused and deterministic responses.
    - Higher values (e.g., 0.7-0.9) lead to more creative and diverse, but potentially less accurate, responses.

### Example Queries:

- "How do I debug a Python script that crashes unexpectedly?"
- "Can you help me write documentation for this function?"
- "I need to design a solution for a caching mechanism. Any suggestions?"

The AI will provide insights, code snippets, or guidance tailored to your query and configuration.

## Technologies Used

- **[Streamlit](https://streamlit.io/)** – An open-source framework for building the web interface.
- **[LangChain](https://python.langchain.com/)** – A framework for managing AI-driven conversation flows.
- **[Ollama](https://ollama.ai/)** – A platform for running large language models locally.
- **DeepSeek Models** – Specifically, `deepseek-r1:1.5b` and `deepseek-r1:3b`, optimized for coding assistance.
