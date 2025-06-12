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

## Deployment with Docker

Deploying the DeepSeek Code Companion using Docker and Docker Compose offers a consistent and isolated environment, managing both the Streamlit application and the Ollama LLM service.

### Prerequisites

- **Docker**: Ensure Docker is installed on your system.
  - Installation guide: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
- **Docker Compose**: Ensure Docker Compose is installed.
  - Installation guide: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

### Running the Application

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/chintanboghara/DeepSeek-CodeCompanion.git
    cd DeepSeek-CodeCompanion
    ```

2.  **Start the application using Docker Compose**:
    ```bash
    docker-compose up --build
    ```
    - The `--build` flag is recommended for the first time you run the command or when there are changes to the `Dockerfile` or application code (e.g., `app.py`, `requirements.txt`). For subsequent runs, `docker-compose up` might be faster if no code changes occurred.
    - This command will build the Docker image for the Streamlit application and start both the `app` and `ollama` services.

3.  **Access the application**:
    - Once the services are running, the Streamlit application will be available at `http://localhost:8501` in your web browser.

### Ollama Model Management

Ollama models are persisted in a Docker volume named `ollama_data` (as defined in `docker-compose.yml`). This means models you pull will be available even if you stop and restart the containers.

To manage models within the Ollama service:

-   **List currently available models**:
    ```bash
    docker-compose exec ollama ollama list
    ```
-   **Pull a new model** (e.g., `deepseek-r1:1.5b`):
    ```bash
    docker-compose exec ollama ollama pull deepseek-r1:1.5b
    ```
-   **Pull another model** (e.g., `deepseek-r1:3b`):
    ```bash
    docker-compose exec ollama ollama pull deepseek-r1:3b
    ```
    You only need to pull each model once. The application's sidebar will allow you to select from the models available to the Ollama service.

### Ollama Base URL Configuration

- The `docker-compose.yml` file automatically configures the `OLLAMA_BASE_URL` environment variable for the Streamlit application service (`app`) to `http://ollama:11434`. This is the internal address of the `ollama` service within the Docker network.
- If you were running `app.py` locally for development (outside of this Docker Compose setup) and Ollama was also running locally, `app.py` would default to `http://localhost:11434` for the Ollama API, or you could set the `OLLAMA_BASE_URL` environment variable manually.

### Stopping the Application

To stop the application and remove the containers:
```bash
docker-compose down
```
- This command will stop and remove the containers defined in `docker-compose.yml`. The `ollama_data` volume will persist, preserving your downloaded Ollama models.

### GPU Support (Note)

The provided `docker-compose.yml` is configured for CPU usage by Ollama. For GPU support, you would need to uncomment and configure the `deploy` section within the `ollama` service definition in `docker-compose.yml`. This typically involves:
- Ensuring your host system has the necessary NVIDIA drivers and the NVIDIA Container Toolkit installed.
- Modifying the `deploy` section to specify GPU resources, for example:
  ```yaml
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: 1 # or 'all'
  #           capabilities: [gpu]
  ```
Refer to the official Docker and Ollama documentation for detailed instructions on GPU passthrough.

## Features and Usage

Once the app is running, interact with the AI Assistant via the web interface:

- **Chat Interface**: Type your coding-related questions into the input box at the bottom of the screen.
- **Predefined Actions**: Select a specific task from the "Choose Action" dropdown in the sidebar to tailor the AI's focus and system prompt. Available actions include:
    - **General Chat**: For general coding questions, discussions, and assistance (default).
    - **Explain Code**: Provide a code snippet and ask the AI to explain its functionality, logic, and purpose.
    - **Debug Code**: Submit code along with error messages or a description of the issue to get help identifying bugs and potential fixes.
    - **Write Documentation**: Ask the AI to generate technical documentation (e.g., function summaries, parameter descriptions) for a given piece of code.
    - **Optimize Code**: Provide a code snippet and ask the AI for optimization suggestions (e.g., for performance, readability, or efficiency).
    - **Write Unit Tests**: Ask the AI to generate unit test cases for a given function or class.
    - **Translate Code**: Ask the AI to translate a code snippet from a source language to a target language (you'll specify the languages in your query).
- **Flexible Code Input for Analysis**:
    - For actions like "Explain Code", "Debug Code", "Optimize Code", "Write Unit Tests", and "Translate Code", you have multiple ways to provide code:
        - **Dedicated Text Area**: A text area in the sidebar allows you to paste your code directly.
        - **File Upload**: You can upload code files (e.g., `.py`, `.js`, `.java`, `.c`, `.cpp`, `.txt`, `.md`, `.json`, `.yaml`, `.html`, `.css`) directly using the file uploader in the sidebar. If a file is uploaded, its content will be used for analysis, taking precedence over code in the text area. The assistant will be informed of the filename.
    - This allows for easy input of both small snippets and larger codebases.
- **Model Selection**: Choose between different DeepSeek models from the sidebar.
- **Temperature Control**: Adjust the "Select Temperature" slider in the sidebar (range 0.0 to 1.0, default 0.3).
    - Lower values (e.g., 0.1-0.3) produce more focused and deterministic responses.
    - Higher values (e.g., 0.7-0.9) lead to more creative and diverse, but potentially less accurate, responses.
- **Top K Sampling**: Configure the "Top K" value using the number input in the sidebar (default 40, range 1-100).
    - This parameter limits the LLM's selection of the next token to the K most probable tokens, influencing response diversity.
- **Top P (Nucleus) Sampling**: Adjust the "Top P" slider in the sidebar (default 0.9, range 0.0-1.0).
    - This parameter selects tokens based on their cumulative probability, ensuring that only the most probable tokens whose sum exceeds P are considered. It provides another way to control response diversity.
- **Configuration Persistence**: Your selections for the AI model, temperature, Top-K, and Top-P parameters are saved in your browser's local storage. These settings will be automatically reloaded the next time you open the application.
- **Session Management**:
    - **Saving Sessions**: You can save your current chat conversation for later use. In the sidebar under "💾 Chat Sessions", enter a unique name for your session in the text field and click "Save Current Session". This will store the current chat messages under that name in your browser's local storage. You can overwrite an existing session by saving with the same name.
    - **Loading Sessions**: To resume a previous conversation, select its name from the "Load Session:" dropdown in the sidebar. The chat history will be loaded, and the session name will be displayed.
    - **Deleting Sessions**: To remove a saved session, select its name from the "Delete Session:" dropdown and click "Delete Selected Session". This will permanently remove it from your browser's local storage.
    - **Starting a New Session**: Click the "Clear Chat History" button in the sidebar. This will clear the current chat display and start a fresh, unnamed session. It does not delete any of your saved sessions.
    - All session data is stored locally in your web browser.
- **Improved Error Notifications**: Critical errors, such as issues connecting to the AI model, are now displayed prominently at the top of the chat interface with troubleshooting tips. File processing errors will appear in the sidebar. This provides clearer feedback on any operational issues.
- **Improved Code Formatting**:
    - The AI has been instructed to consistently use Markdown code blocks with language identifiers (e.g., ` ```python ... ``` `). This enhances the clarity and readability of code snippets in the chat.
- **Context Management**:
    - For very long conversations, the application manages the conversation context to stay within the LLM's processing limits.
    - It uses a token-based sliding window approach, prioritizing recent parts of the conversation if the total length becomes too extensive (currently around 3000 tokens using `cl100k_base` tokenizer).
    - This helps ensure stable performance and prevents errors during long interactions.
    - The AI's system prompt also includes a gentle reminder that it may not recall the earliest parts of a very long discussion.
- **Model Loading Indicator & Caching**:
    - When you change the selected AI model or adjust its core parameters (like Temperature, Top K, Top P), a loading indicator (spinner) will appear while the new model configuration is being initialized. This provides clear feedback during model setup.
    - The AI model engine is cached within your current session. The loading indicator will only appear if the model or its key parameters need to be re-initialized, saving time on subsequent operations if the configuration hasn't changed.
- **Asynchronous LLM Calls (Non-Blocking UI)**:
    - The application now processes LLM requests in the background, ensuring the user interface remains responsive.
    - **User Experience**:
        - When you send a message, a "🧠 Thinking..." placeholder will appear immediately in the chat.
        - The chat input field will be temporarily disabled until the AI's response is received.
        - The application UI should not freeze during LLM processing.
        - The app uses `streamlit-autorefresh` to periodically check for results in the background, which may cause slight, brief screen refreshes while waiting for the AI's response.
    - This is achieved using a background task manager (`LLMTaskManager`) and the `streamlit-autorefresh` component for periodic updates.
- **Streaming Responses**: AI responses now stream in token by token. This means you'll start seeing the beginning of longer responses much faster, significantly improving the perceived responsiveness of the assistant.

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
