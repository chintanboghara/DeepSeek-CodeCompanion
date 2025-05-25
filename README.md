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

   The `requirements.txt` file lists all necessary Python packages, including Streamlit, LangChain components, Requests, `streamlit-autorefresh`, and other utilities.
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
- **Predefined Actions**: Select a specific task from the "Choose Action" dropdown in the sidebar to tailor the AI's focus and system prompt. Available actions include:
    - **General Chat**: For general coding questions, discussions, and assistance (default).
    - **Explain Code**: Provide a code snippet and ask the AI to explain its functionality, logic, and purpose.
    - **Debug Code**: Submit code along with error messages or a description of the issue to get help identifying bugs and potential fixes.
    - **Write Documentation**: Ask the AI to generate technical documentation (e.g., function summaries, parameter descriptions) for a given piece of code.
- **Model Selection**: Choose between different DeepSeek models from the sidebar.
- **Temperature Control**: Adjust the "Select Temperature" slider in the sidebar (range 0.0 to 1.0, default 0.3).
    - Lower values (e.g., 0.1-0.3) produce more focused and deterministic responses.
    - Higher values (e.g., 0.7-0.9) lead to more creative and diverse, but potentially less accurate, responses.
- **Top K Sampling**: Configure the "Top K" value using the number input in the sidebar (default 40, range 1-100).
    - This parameter limits the LLM's selection of the next token to the K most probable tokens, influencing response diversity.
- **Top P (Nucleus) Sampling**: Adjust the "Top P" slider in the sidebar (default 0.9, range 0.0-1.0).
    - This parameter selects tokens based on their cumulative probability, ensuring that only the most probable tokens whose sum exceeds P are considered. It provides another way to control response diversity.
- **Chat History Persistence**:
    - Your conversations are automatically saved to your browser's local storage.
    - This means you can close the browser or refresh the page, and your current chat history will be reloaded when you return.
- **Clear Chat History**:
    - A "Clear Chat History" button is available in the sidebar.
    - Clicking this button will remove the current conversation from both the application's session and your browser's local storage, allowing you to start a fresh chat.
- **Improved Code Formatting**:
    - The AI has been instructed to consistently use Markdown code blocks with language identifiers (e.g., ` ```python ... ``` `). This enhances the clarity and readability of code snippets in the chat.
- **Context Management**:
    - For very long conversations, the application manages the conversation context to stay within the LLM's processing limits.
    - It uses a token-based sliding window approach, prioritizing recent parts of the conversation if the total length becomes too extensive (currently around 3000 tokens using `cl100k_base` tokenizer).
    - This helps ensure stable performance and prevents errors during long interactions.
    - The AI's system prompt also includes a gentle reminder that it may not recall the earliest parts of a very long discussion.
- **Asynchronous LLM Calls (Non-Blocking UI)**:
    - The application now processes LLM requests in the background, ensuring the user interface remains responsive.
    - **User Experience**:
        - When you send a message, a "🧠 Thinking..." placeholder will appear immediately in the chat.
        - The chat input field will be temporarily disabled until the AI's response is received.
        - The application UI should not freeze during LLM processing.
        - The app uses `streamlit-autorefresh` to periodically check for results in the background, which may cause slight, brief screen refreshes while waiting for the AI's response.
    - This is achieved using a background task manager (`LLMTaskManager`) and the `streamlit-autorefresh` component for periodic updates.

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
