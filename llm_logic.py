"""
Handles the core logic for interacting with the Language Model (LLM).

This module includes functions for initializing the LLM engine,
constructing prompts based on chat history, and generating responses
from the LLM. It uses LangChain and Ollama for LLM interactions.
"""
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import requests
import logging
import tiktoken # Import tiktoken

# Define maximum context tokens
MAX_CONTEXT_TOKENS = 3000

# Predefined system prompts tailored for each specific action.
# Keys must match PREDEFINED_ACTIONS in app.py for correct mapping.
PREDEFINED_SYSTEM_PROMPTS = {
    "General Chat": "You are an expert AI coding assistant. Provide concise, correct solutions with strategic print statements for debugging. Always respond in English. When providing code, always use Markdown code blocks with the appropriate language identifier (e.g., ```python ... ``` or ```javascript ... ```). Please note that for very long conversations, I might not recall the earliest parts of our discussion to stay focused and efficient.",
    "Explain Code": "You are an expert AI coding assistant. Your current task is to explain the provided code snippet. Focus on its functionality, logic, key components, and how it works. Describe its purpose, inputs, and outputs. If relevant, suggest potential improvements or edge cases. Always respond in English. When providing code examples as part of your explanation, always use Markdown code blocks with the appropriate language identifier. Please note that for very long conversations, I might not recall the earliest parts of our discussion.",
    "Debug Code": "You are an expert AI coding assistant specializing in debugging code. Analyze the provided code snippet and any accompanying error messages or descriptions of issues. Identify potential bugs, logical errors, or areas that might not behave as expected. Suggest fixes or debugging strategies, including strategic print statements or checks. Explain your reasoning clearly. Always respond in English. When providing corrected code or examples, always use Markdown code blocks with the appropriate language identifier. Please note that for very long conversations, I might not recall the earliest parts of our discussion.",
    "Write Documentation": "You are an expert AI coding assistant with a talent for writing clear and concise technical documentation. For the provided code snippet, generate appropriate documentation. This may include:\n- A summary of the code's purpose and functionality.\n- Descriptions of functions/classes, including their parameters and return values.\n- Usage examples if applicable.\n- Notes on dependencies or important considerations.\nFormat the documentation clearly, using Markdown. For any code examples within the documentation, use Markdown code blocks with language identifiers. Always respond in English. Please note that for very long conversations, I might not recall the earliest parts of our discussion.",
    "Optimize Code": "You are an expert AI coding assistant focused on code optimization. Analyze the provided code snippet for potential improvements in areas such as performance, readability, efficiency, and adherence to best practices. Provide specific suggestions and explain the reasoning behind them. If you rewrite code, use Markdown code blocks with the appropriate language identifier. Always respond in English. Please note that for very long conversations, I might not recall the earliest parts of our discussion.",
    "Write Unit Tests": "You are an expert AI coding assistant specializing in software testing. Your task is to generate unit test cases for the provided code snippet (e.g., function or class). If possible, write tests suitable for common testing frameworks like Python's 'unittest' or 'pytest', or provide framework-agnostic test case descriptions. Aim to cover:\n    - Common use cases.\n    - Edge cases (e.g., empty inputs, boundary values).\n    - Error conditions or invalid inputs.\nExplain the purpose of each test case briefly. Present the test code using Markdown code blocks with the appropriate language identifier. Always respond in English. Please note that for very long conversations, I might not recall the earliest parts of our discussion.",
    "Translate Code": "You are an expert AI coding assistant with proficiency in multiple programming languages. Your task is to translate the provided code snippet from a source language to a target language, which the user will specify in their query (e.g., 'Translate this Python code to JavaScript'). Pay close attention to language-specific idioms, syntax, and common libraries or conventions. If the target language lacks a direct equivalent for a feature, suggest a suitable alternative. Provide the translated code using Markdown code blocks with the appropriate language identifier for the target language. Always respond in English. Please note that for very long conversations, I might not recall the earliest parts of our discussion."
}

def get_system_prompt(action: str) -> str:
    """
    Retrieves the system prompt text for a given action.

    Args:
        action (str): The selected action (e.g., "General Chat", "Explain Code").

    Returns:
        str: The system prompt text. Defaults to "General Chat" prompt if action is not found.
    """
    return PREDEFINED_SYSTEM_PROMPTS.get(action, PREDEFINED_SYSTEM_PROMPTS["General Chat"])

# Initialize Chat Engine
def init_llm_engine(selected_model, temperature, top_k, top_p, ollama_base_url: str):
    """
    Initializes and returns the ChatOllama LLM engine.

    Args:
        selected_model (str): The name of the Ollama model to use.
        temperature (float): The temperature setting for the LLM, controlling randomness.
        top_k (int): Limits the set of next tokens to the K most probable.
        top_p (float): Controls diversity via nucleus sampling. Selects tokens
                       with cumulative probability greater than P.
        ollama_base_url (str): The base URL for the Ollama API.

    Returns:
        ChatOllama: An instance of the ChatOllama engine.
    """
    model_kwargs = {'top_k': top_k, 'top_p': top_p}
    return ChatOllama(
        model=selected_model,
        base_url=ollama_base_url,
        temperature=temperature,
        model_kwargs=model_kwargs
    )

# System Prompt Configuration is now handled dynamically by get_system_prompt

# Function: Build Chat Prompt Chain
def build_prompt_chain(message_log, selected_action: str):
    """
    Constructs a LangChain ChatPromptTemplate from the message history,
    using an action-specific system prompt and managing the context window
    to stay within MAX_CONTEXT_TOKENS.

    The prompt includes the dynamically selected system prompt and the most
    recent messages that fit within the token limit. Messages are added in
    reverse chronological order (newest first) until the token limit is approached.

    Args:
        message_log (list): A list of dictionaries, where each dictionary
                              represents a chat message with "role" and "content".
        selected_action (str): The action selected by the user, used to determine
                               the system prompt.

    Returns:
        ChatPromptTemplate: A LangChain ChatPromptTemplate object.
    """
    # Initialize tokenizer (cl100k_base is a common encoder for OpenAI models)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        """Helper function to count tokens in a given text."""
        return len(tokenizer.encode(text))

    # Get action-specific system prompt text and create the prompt template
    action_specific_prompt_text = get_system_prompt(selected_action)
    current_system_prompt = SystemMessagePromptTemplate.from_template(action_specific_prompt_text)
    
    # Count tokens for the action-specific system prompt
    system_tokens = count_tokens(action_specific_prompt_text)
    
    current_tokens = system_tokens
    managed_message_log = []

    # Iterate through messages in reverse order (newest first)
    for msg in reversed(message_log):
        message_tokens = count_tokens(msg["content"])
        # Check if adding the current message would exceed the token limit
        if current_tokens + message_tokens < MAX_CONTEXT_TOKENS:
            managed_message_log.insert(0, msg) # Add to the beginning to maintain order
            current_tokens += message_tokens
        else:
            # If limit is reached, stop adding older messages
            logging.info(f"Context token limit reached. Truncating older messages. Current tokens: {current_tokens}")
            break
            
    # Construct the prompt sequence
    prompt_sequence = [current_system_prompt] # Start with the action-specific system prompt
    
    # Add managed messages to the prompt sequence
    for msg in managed_message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
            
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Function: Generate AI Response
def generate_ai_response(llm_engine, prompt_chain):
    """
    Generates a response from the LLM using the provided engine and prompt chain.

    It includes error handling for connection issues, runtime errors, and other
    unexpected exceptions during the LLM interaction.

    Args:
        llm_engine (ChatOllama): The initialized ChatOllama engine.
        prompt_chain (ChatPromptTemplate): The constructed prompt chain for the LLM.

    Returns:
        str: The AI's response as a string, or an error message string if an issue occurs.
    """
    try:
        # Define the processing pipeline: prompt -> LLM engine -> string output parser
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        # Invoke the pipeline to get the AI response
        return processing_pipeline.invoke({})
    except requests.exceptions.ConnectionError as e:
        # Handle errors where the Ollama server cannot be reached
        logging.exception("ConnectionError during LLM interaction:")
        return "OLLAMA_CONNECTION_ERROR: Could not connect to Ollama server. Please ensure Ollama is running."
    except RuntimeError as e:
        # Handle runtime errors that might occur within LangChain or Ollama
        logging.exception("RuntimeError during LLM interaction:")
        return f"LLM_RUNTIME_ERROR: A runtime error occurred: {str(e)}"
    except Exception as e:
        # Handle any other unexpected errors
        logging.exception("Unexpected error during LLM interaction:")
        return f"LLM_UNEXPECTED_ERROR: An unexpected error occurred: {str(e)}"
