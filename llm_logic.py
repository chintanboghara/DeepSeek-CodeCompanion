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

# Initialize Chat Engine
def init_llm_engine(selected_model, temperature):
    """
    Initializes and returns the ChatOllama LLM engine.

    Args:
        selected_model (str): The name of the Ollama model to use.
        temperature (float): The temperature setting for the LLM, controlling randomness.

    Returns:
        ChatOllama: An instance of the ChatOllama engine.
    """
    return ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=temperature)

# System Prompt Configuration
# This system prompt defines the persona and behavior of the AI assistant.
# It instructs the AI to act as an expert coding assistant, provide concise
# and correct solutions, use print statements for debugging, and always respond in English.
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Function: Build Chat Prompt Chain
def build_prompt_chain(message_log):
    """
    Constructs a LangChain ChatPromptTemplate from the message history.

    The prompt includes the system prompt and alternates between human and AI messages
    based on the provided message log.

    Args:
        message_log (list): A list of dictionaries, where each dictionary
                              represents a chat message with "role" and "content".

    Returns:
        ChatPromptTemplate: A LangChain ChatPromptTemplate object.
    """
    prompt_sequence = [system_prompt] # Start with the system prompt
    # Iterate through the message log and append Human or AI message prompts
    for msg in message_log:
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
