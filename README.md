# 🧠 DeepSeek Code Companion

**DeepSeek Code Companion** is an AI-powered pair programming assistant that helps with debugging, code documentation, and solution design. Built using **Streamlit**, **LangChain**, and **Ollama**, this tool integrates **DeepSeek models** to provide intelligent coding assistance.

## Installation

- **Python 3.8+**
- **pip** (Python package manager)
- **Ollama** (Local LLM engine): [Download Ollama](https://ollama.ai/)

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
   ```sh
   pip install -r requirements.txt
   ```

4. **Run Ollama in the background** (if not already running)
   ```sh
   ollama run
   ```

5. **Launch the Streamlit app**
   ```sh
   streamlit run app.py
   ```

6. **Interact with the AI Assistant**
   - Open the link provided in the terminal (usually `http://localhost:8501`).
   - Type in your coding-related queries and receive AI-powered assistance.

## Technologies Used
- **[Streamlit](https://streamlit.io/)** – UI for the web app.
- **[LangChain](https://python.langchain.com/)** – Framework for AI-driven conversations.
- **[Ollama](https://ollama.ai/)** – Local AI model execution.
