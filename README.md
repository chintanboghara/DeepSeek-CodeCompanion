# 🧠 DeepSeek Code Companion

AI-powered **Pair Programmer** with **Debugging Superpowers**. Built using **Streamlit**, **Ollama**, and **LangChain** for interactive coding assistance.  

## Features
- **Python Expert** – Get precise coding assistance  
- **Debugging Assistant** – Helps you debug with strategic print statements  
- **Code Documentation** – Generates explanations and documentation  
- **Solution Design** – Provides optimized coding solutions  

## Run with Docker
#### **Prerequisites**
- Install **Docker** (Ensure Docker is running)

#### **Steps**
1. **Build the Docker image**
   ```sh
   docker build -t deepseek-codecompanion .
   ```

2. **Run the container**
   ```sh
   docker run -p 8501:8501 deepseek-codecompanion
   ```

3. Open the browser and go to:  
   **[http://localhost:8501](http://localhost:8501)**


## Customization
### Modify **AI Model**
Change the model in `app.py`:
```python
selected_model = st.selectbox(
    "Choose Model",
    ["deepseek-r1:1.5b", "deepseek-r1:3b"],
    index=0
)
```

### Change **Temperature (Creativity Level)**
Modify this line in `app.py`:
```python
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)
```
- **Lower (`0.0 - 0.3`)** → More **deterministic**  
- **Higher (`0.7 - 1.0`)** → More **creative responses**  
