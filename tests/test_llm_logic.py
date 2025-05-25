import unittest
from unittest.mock import patch, MagicMock
import requests
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

# Import functions and variables from llm_logic.py
from llm_logic import init_llm_engine, build_prompt_chain, generate_ai_response, system_prompt

class TestInitLlmEngine(unittest.TestCase):
    @patch('llm_logic.ChatOllama')
    def test_init_llm_engine_params(self, MockChatOllama):
        """Test that ChatOllama is initialized with correct model and temperature."""
        model_name = "test-model"
        temp = 0.5
        init_llm_engine(model_name, temp)
        MockChatOllama.assert_called_once_with(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=temp
        )

class TestBuildPromptChain(unittest.TestCase):
    def test_build_prompt_chain_empty_log(self):
        """Test prompt chain with an empty message log."""
        message_log = []
        prompt_chain = build_prompt_chain(message_log)
        self.assertIsInstance(prompt_chain, ChatPromptTemplate)
        self.assertEqual(len(prompt_chain.messages), 1)
        self.assertEqual(prompt_chain.messages[0], system_prompt)

    def test_build_prompt_chain_with_messages(self):
        """Test prompt chain with user and AI messages."""
        message_log = [
            {"role": "user", "content": "Hello AI"},
            {"role": "ai", "content": "Hello User"},
            {"role": "user", "content": "How are you?"}
        ]
        prompt_chain = build_prompt_chain(message_log)
        self.assertIsInstance(prompt_chain, ChatPromptTemplate)
        self.assertEqual(len(prompt_chain.messages), 4) # System + 3 messages
        self.assertEqual(prompt_chain.messages[0], system_prompt)
        self.assertIsInstance(prompt_chain.messages[1], HumanMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[1].prompt.template, "Hello AI")
        self.assertIsInstance(prompt_chain.messages[2], AIMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[2].prompt.template, "Hello User")
        self.assertIsInstance(prompt_chain.messages[3], HumanMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[3].prompt.template, "How are you?")

class TestGenerateAiResponse(unittest.TestCase):
    @patch('llm_logic.StrOutputParser')
    @patch('llm_logic.ChatOllama')
    def test_generate_response_success(self, MockChatOllama, MockStrOutputParser):
        """Test successful AI response generation."""
        mock_llm_engine = MockChatOllama.return_value
        mock_str_parser_instance = MockStrOutputParser.return_value

        # Setup mock for prompt_chain | llm_engine
        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        mock_first_pipe_result = MagicMock()
        mock_prompt_chain.__or__.return_value = mock_first_pipe_result

        # Setup mock for (prompt_chain | llm_engine) | StrOutputParser()
        mock_second_pipe_result = MagicMock()
        mock_first_pipe_result.__or__.return_value = mock_second_pipe_result
        
        # Setup mock for .invoke({})
        mock_second_pipe_result.invoke.return_value = "Successful response"
        
        response = generate_ai_response(mock_llm_engine, mock_prompt_chain)
        
        mock_prompt_chain.__or__.assert_called_once_with(mock_llm_engine)
        mock_first_pipe_result.__or__.assert_called_once_with(mock_str_parser_instance)
        mock_second_pipe_result.invoke.assert_called_once_with({})
        self.assertEqual(response, "Successful response")

    @patch('llm_logic.logging')
    @patch('llm_logic.StrOutputParser')
    @patch('llm_logic.ChatOllama')
    def test_generate_response_connection_error(self, MockChatOllama, MockStrOutputParser, mock_logging):
        """Test ConnectionError handling."""
        mock_llm_engine = MockChatOllama.return_value
        mock_str_parser_instance = MockStrOutputParser.return_value

        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        mock_first_pipe_result = MagicMock()
        mock_prompt_chain.__or__.return_value = mock_first_pipe_result
        
        mock_second_pipe_result = MagicMock()
        mock_first_pipe_result.__or__.return_value = mock_second_pipe_result
        
        mock_second_pipe_result.invoke.side_effect = requests.exceptions.ConnectionError("Test connection error")

        response = generate_ai_response(mock_llm_engine, mock_prompt_chain)
        
        self.assertEqual(response, "⚠️ Error: Could not connect to Ollama server. Please ensure Ollama is running.")
        mock_logging.exception.assert_called_once_with("ConnectionError during LLM interaction:")

    @patch('llm_logic.logging')
    @patch('llm_logic.StrOutputParser')
    @patch('llm_logic.ChatOllama')
    def test_generate_response_runtime_error(self, MockChatOllama, MockStrOutputParser, mock_logging):
        """Test RuntimeError handling."""
        mock_llm_engine = MockChatOllama.return_value
        mock_str_parser_instance = MockStrOutputParser.return_value

        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        mock_first_pipe_result = MagicMock()
        mock_prompt_chain.__or__.return_value = mock_first_pipe_result

        mock_second_pipe_result = MagicMock()
        mock_first_pipe_result.__or__.return_value = mock_second_pipe_result

        mock_second_pipe_result.invoke.side_effect = RuntimeError("Test runtime error")

        response = generate_ai_response(mock_llm_engine, mock_prompt_chain)

        self.assertEqual(response, "⚠️ Error: A runtime error occurred: Test runtime error")
        mock_logging.exception.assert_called_once_with("RuntimeError during LLM interaction:")

    @patch('llm_logic.logging')
    @patch('llm_logic.StrOutputParser')
    @patch('llm_logic.ChatOllama')
    def test_generate_response_generic_exception(self, MockChatOllama, MockStrOutputParser, mock_logging):
        """Test generic Exception handling."""
        mock_llm_engine = MockChatOllama.return_value
        mock_str_parser_instance = MockStrOutputParser.return_value

        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        mock_first_pipe_result = MagicMock()
        mock_prompt_chain.__or__.return_value = mock_first_pipe_result

        mock_second_pipe_result = MagicMock()
        mock_first_pipe_result.__or__.return_value = mock_second_pipe_result
        
        mock_second_pipe_result.invoke.side_effect = Exception("Test generic exception")
        
        response = generate_ai_response(mock_llm_engine, mock_prompt_chain)

        self.assertEqual(response, "⚠️ Error: An unexpected error occurred: Test generic exception")
        mock_logging.exception.assert_called_once_with("Unexpected error during LLM interaction:")

if __name__ == '__main__':
    unittest.main()
