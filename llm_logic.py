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

# Initialize Chat Engine
def init_llm_engine(selected_model, temperature, top_k, top_p):
    """
    Initializes and returns the ChatOllama LLM engine.

    Args:
        selected_model (str): The name of the Ollama model to use.
        temperature (float): The temperature setting for the LLM, controlling randomness.
        top_k (int): Limits the set of next tokens to the K most probable.
        top_p (float): Controls diversity via nucleus sampling. Selects tokens
                       with cumulative probability greater than P.

    Returns:
        ChatOllama: An instance of the ChatOllama engine.
    """
    model_kwargs = {'top_k': top_k, 'top_p': top_p}
    return ChatOllama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=temperature,
        model_kwargs=model_kwargs
    )

# System Prompt Configuration
# This system prompt defines the persona and behavior of the AI assistant.
# It instructs the AI to act as an expert coding assistant, provide concise
# and correct solutions, use print statements for debugging, always respond in English,
# specify code formatting, and manage expectations about context length.
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English. "
    "When providing code, always use Markdown code blocks with the appropriate "
    "language identifier (e.g., ```python ... ``` or ```javascript ... ```). "
    "Please note that for very long conversations, I might not recall the earliest "
    "parts of our discussion to stay focused and efficient."
)

# Function: Build Chat Prompt Chain
def build_prompt_chain(message_log):
    """
    Constructs a LangChain ChatPromptTemplate from the message history,
    managing the context window to stay within MAX_CONTEXT_TOKENS.

    The prompt includes the system prompt and the most recent messages that fit
    within the token limit. Messages are added in reverse chronological order
    (newest first) until the token limit is approached.

    Args:
        message_log (list): A list of dictionaries, where each dictionary
                              represents a chat message with "role" and "content".

    Returns:
        ChatPromptTemplate: A LangChain ChatPromptTemplate object.
    """
    # Initialize tokenizer (cl100k_base is a common encoder for OpenAI models)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        """Helper function to count tokens in a given text."""
        return len(tokenizer.encode(text))

    # Get system prompt text and count its tokens
    system_prompt_text = system_prompt.prompt.template
    system_tokens = count_tokens(system_prompt_text)
    
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
    prompt_sequence = [system_prompt] # Start with the system prompt
    
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
        return "⚠️ Error: Could not connect to Ollama server. Please ensure Ollama is running."
    except RuntimeError as e:
        # Handle runtime errors that might occur within LangChain or Ollama
        logging.exception("RuntimeError during LLM interaction:")
        return f"⚠️ Error: A runtime error occurred: {str(e)}"
    except Exception as e:
        # Handle any other unexpected errors
        logging.exception("Unexpected error during LLM interaction:")
        return f"⚠️ Error: An unexpected error occurred: {str(e)}"
