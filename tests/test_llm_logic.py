import unittest
from unittest.mock import patch, MagicMock
import requests
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

# Import functions and variables from llm_logic.py
from llm_logic import init_llm_engine, build_prompt_chain, generate_ai_response, get_system_prompt, PREDEFINED_SYSTEM_PROMPTS

class TestInitLlmEngine(unittest.TestCase):
    @patch('llm_logic.ChatOllama')
    def test_init_llm_engine_params(self, MockChatOllama):
        model_name = "test-model"
        temp = 0.5
        top_k_val = 30
        top_p_val = 0.8
        ollama_url = "http://testhost:12345" # Test with a non-default URL

        init_llm_engine(model_name, temp, top_k_val, top_p_val, ollama_url)

        MockChatOllama.assert_called_once_with(
            model=model_name,
            base_url=ollama_url,
            temperature=temp,
            model_kwargs={'top_k': top_k_val, 'top_p': top_p_val}
        )

class TestBuildPromptChain(unittest.TestCase):
    # Test with "General Chat" action
    def test_build_prompt_chain_empty_log_general_chat(self):
        message_log = []
        action = "General Chat"
        # Ensure PREDEFINED_SYSTEM_PROMPTS is accessible or mock get_system_prompt
        expected_system_prompt_content = PREDEFINED_SYSTEM_PROMPTS[action]

        prompt_chain = build_prompt_chain(message_log, action)

        self.assertIsInstance(prompt_chain, ChatPromptTemplate)
        self.assertEqual(len(prompt_chain.messages), 1)
        self.assertIsInstance(prompt_chain.messages[0], SystemMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[0].prompt.template, expected_system_prompt_content)

    # Test with a different action, e.g., "Explain Code"
    def test_build_prompt_chain_with_messages_explain_code(self):
        message_log = [
            {"role": "user", "content": "Hello AI"},
            {"role": "ai", "content": "Hello User"}
        ]
        action = "Explain Code"
        expected_system_prompt_content = PREDEFINED_SYSTEM_PROMPTS[action]

        prompt_chain = build_prompt_chain(message_log, action)

        self.assertIsInstance(prompt_chain, ChatPromptTemplate)
        self.assertEqual(len(prompt_chain.messages), 3) # System + 2 messages
        self.assertIsInstance(prompt_chain.messages[0], SystemMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[0].prompt.template, expected_system_prompt_content)
        self.assertIsInstance(prompt_chain.messages[1], HumanMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[1].prompt.template, "Hello AI")
        self.assertIsInstance(prompt_chain.messages[2], AIMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[2].prompt.template, "Hello User")

    @patch('llm_logic.tiktoken.get_encoding')
    def test_build_prompt_chain_context_truncation(self, mock_get_encoding):
        # Mock tokenizer and its encode method to control token counts
        mock_tokenizer = MagicMock()
        # Simple mock: count each character as a token for predictability in test
        mock_tokenizer.encode.side_effect = lambda text: [0] * len(text)
        mock_get_encoding.return_value = mock_tokenizer

        action = "General Chat"
        # Actual system prompt content is fetched, its length will be added to token count
        system_prompt_content = PREDEFINED_SYSTEM_PROMPTS[action]
        system_prompt_tokens = len(system_prompt_content) # Based on our mock tokenizer

        # MAX_CONTEXT_TOKENS is 3000.
        # We want to test that truncation happens.
        # Let one message be very long, and another fit with the system prompt.

        # Message 1 (very long, should be truncated/excluded)
        content1 = "a" * (3000 - system_prompt_tokens + 500) # Definitely exceeds if taken with sys prompt
        # Message 2 (shorter, should be included)
        content2 = "b" * 100
        # Message 3 (most recent, should definitely be included)
        content3 = "c" * 50


        message_log = [
            {"role": "user", "content": content1},
            {"role": "ai", "content": content2},
            {"role": "user", "content": content3}
        ]

        prompt_chain = build_prompt_chain(message_log, action)

        # Assert that the system prompt is the first message
        self.assertIsInstance(prompt_chain.messages[0], SystemMessagePromptTemplate)
        self.assertEqual(prompt_chain.messages[0].prompt.template, system_prompt_content)

        # Calculate total tokens in the generated prompt
        total_tokens = system_prompt_tokens
        message_templates = []
        for msg_template in prompt_chain.messages[1:]: # Skip system prompt
            total_tokens += len(msg_template.prompt.template)
            message_templates.append(msg_template.prompt.template)

        self.assertTrue(total_tokens < 3000) # MAX_CONTEXT_TOKENS
        self.assertIn(content3, message_templates) # Most recent message should be there
        self.assertIn(content2, message_templates) # The one before should also fit
        self.assertNotIn(content1, message_templates) # The oldest, very long message should be excluded

class TestGenerateAiResponse(unittest.TestCase):
    @patch('llm_logic.StrOutputParser')
    @patch('llm_logic.ChatOllama') # Mock the ChatOllama class used in init_llm_engine
    def test_generate_response_success_streaming(self, MockChatOllama, MockStrOutputParser):
        mock_llm_engine_instance = MockChatOllama.return_value
        mock_str_parser_instance = MockStrOutputParser.return_value

        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        # Mock the behavior of the | operator for pipeline construction
        mock_pipeline_after_llm = MagicMock()
        mock_pipeline_final = MagicMock()

        mock_prompt_chain.__or__.return_value = mock_pipeline_after_llm
        mock_pipeline_after_llm.__or__.return_value = mock_pipeline_final
        
        mock_pipeline_final.stream.return_value = ["Successful ", "response ", "chunks"]
        
        response_generator = generate_ai_response(mock_llm_engine_instance, mock_prompt_chain)
        response_list = list(response_generator)
        
        mock_prompt_chain.__or__.assert_called_once_with(mock_llm_engine_instance)
        mock_pipeline_after_llm.__or__.assert_called_once_with(mock_str_parser_instance)
        mock_pipeline_final.stream.assert_called_once_with({})
        self.assertEqual(response_list, ["Successful ", "response ", "chunks"])

    @patch('llm_logic.logging')
    # Mock StrOutputParser and ChatOllama, though they might not be called if ConnectionError is early
    @patch('llm_logic.StrOutputParser', MagicMock())
    @patch('llm_logic.ChatOllama', MagicMock())
    def test_generate_response_connection_error_streaming(self, mock_logging):
        # This test assumes the error occurs when stream() is called on the pipeline.
        # To do this, we need to mock the pipeline construction and then the stream() call.
        mock_llm_engine_instance = MagicMock() # llm_logic.ChatOllama()
        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)
        
        # Mock the pipeline object that has the .stream() method
        mock_pipeline = MagicMock()
        mock_pipeline.stream.side_effect = requests.exceptions.ConnectionError("Test connection error")

        # Mock the pipeline construction (prompt | llm | parser) to return our mock_pipeline
        with patch.object(mock_prompt_chain, '__or__', return_value=mock_pipeline): # Patching the __or__ method
            # The StrOutputParser is part of the pipeline, so its direct mock might not be needed here
            # if the error occurs at the .stream() call of the final pipeline object.
            # For simplicity, we ensure the pipeline construction leads to an object whose .stream() fails.
            # This requires a bit more intricate mocking of the pipeline construction.

            # Let's simplify: assume the error is raised when processing_pipeline.stream() is called.
            # The pipeline construction itself is: prompt_chain | llm_engine | StrOutputParser()
            # We can mock the result of this entire chain's .stream() method.

            # A more direct way: patch the entire pipeline construction within generate_ai_response
            # For this test, we will assume the pipeline is constructed, and stream() fails.
            # The provided code for generate_ai_response constructs the pipeline internally.
            # So, we need to ensure that the constructed pipeline's stream() method throws the error.

            # To achieve this, we mock the __or__ method to return an object that has a faulty stream.
            # This is still tricky because of the chained __or__ calls.
            # The easiest way is to make the llm_engine.stream itself (or the final parser's stream) fail.
            # Let's assume the error is raised by `processing_pipeline.stream({})`

            # If ChatOllama or StrOutputParser is not instantiated due to an early error,
            # we might not need to mock them. But if stream() is called, they are.
            # The current structure of generate_ai_response creates the pipeline then calls stream.
            # So, we need to mock the pipeline's stream method.

            # Re-simplifying the mocking for this specific error case:
            # Patch the `stream` method of the object returned by StrOutputParser
            with patch('langchain_core.output_parsers.StrOutputParser.stream', side_effect=requests.exceptions.ConnectionError("Test connection error")):
                 # This approach won't work because StrOutputParser is an instance.
                 # We need to mock the stream method of the *instance* of StrOutputParser.
                 pass # Placeholder for a better mocking strategy if needed below.

            # The current `generate_ai_response` creates `processing_pipeline` then calls `stream`.
            # Let's mock the `stream` method of the object that `prompt_chain | llm_engine | StrOutputParser()` evaluates to.
            # This is the `mock_pipeline_final.stream.side_effect` approach from the prompt, which is good.
            # The issue is that the actual instances of ChatOllama and StrOutputParser are created inside `generate_ai_response`.
            # We need to make sure that the mocked instances are used.

            # The solution given in the prompt is the most viable:
            # Mock ChatOllama and StrOutputParser at the class level, then mock the methods on their return_value.
            # However, generate_ai_response takes llm_engine as an argument, so we can make *its* stream method fail if chained.
            # The pipeline is: prompt_chain | llm_engine | StrOutputParser()
            # The stream is called on the result of this.

            # Let's assume the StrOutputParser's stream method is what ultimately fails.
            # This requires StrOutputParser to be instantiated.

            # Simplified: The error should be raised by processing_pipeline.stream({})
            # We can achieve this by making the llm_engine.stream (which is called by the pipeline) raise the error.
            # Or, more directly, if the pipeline is (P | L | S), S.stream is called.

            # The prompt's approach for mocking pipeline_final.stream is good if we can inject it.
            # The issue is that processing_pipeline is created *inside* generate_ai_response.
            # So, we mock the components.

            # Let's assume StrOutputParser() is called, and its stream method fails.
            # Patching StrOutputParser to return a mock whose stream method fails.
            mock_parser_instance = MagicMock()
            mock_parser_instance.stream.side_effect = requests.exceptions.ConnectionError("Test connection error")
            MockStrOutputParser_class = MagicMock(return_value=mock_parser_instance) # Mock the class

            with patch('llm_logic.StrOutputParser', MockStrOutputParser_class):
                response_generator = generate_ai_response(mock_llm_engine_instance, mock_prompt_chain)
                response_list = list(response_generator)

            self.assertEqual(len(response_list), 1)
            self.assertEqual(response_list[0], "OLLAMA_CONNECTION_ERROR: Could not connect to Ollama server. Please ensure Ollama is running.")
            mock_logging.exception.assert_called_once_with("ConnectionError during LLM interaction setup:")


    @patch('llm_logic.logging')
    @patch('llm_logic.ChatOllama') # Mock ChatOllama
    def test_generate_response_runtime_error_streaming(self, MockChatOllama, mock_logging):
        mock_llm_engine_instance = MockChatOllama.return_value
        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)

        # Mock the StrOutputParser to return an object whose stream method fails
        mock_parser_instance = MagicMock()
        mock_parser_instance.stream.side_effect = RuntimeError("Test runtime error")
        MockStrOutputParser_class = MagicMock(return_value=mock_parser_instance)

        with patch('llm_logic.StrOutputParser', MockStrOutputParser_class):
            response_generator = generate_ai_response(mock_llm_engine_instance, mock_prompt_chain)
            response_list = list(response_generator)

        self.assertEqual(len(response_list), 1)
        self.assertEqual(response_list[0], "LLM_RUNTIME_ERROR: A runtime error occurred: Test runtime error")
        mock_logging.exception.assert_called_once_with("RuntimeError during LLM interaction setup/stream:")

    @patch('llm_logic.logging')
    @patch('llm_logic.ChatOllama') # Mock ChatOllama
    def test_generate_response_generic_exception_streaming(self, MockChatOllama, mock_logging):
        mock_llm_engine_instance = MockChatOllama.return_value
        mock_prompt_chain = MagicMock(spec=ChatPromptTemplate)

        # Mock the StrOutputParser to return an object whose stream method fails
        mock_parser_instance = MagicMock()
        mock_parser_instance.stream.side_effect = Exception("Test generic exception")
        MockStrOutputParser_class = MagicMock(return_value=mock_parser_instance)
        
        with patch('llm_logic.StrOutputParser', MockStrOutputParser_class):
            response_generator = generate_ai_response(mock_llm_engine_instance, mock_prompt_chain)
            response_list = list(response_generator)

        self.assertEqual(len(response_list), 1)
        self.assertEqual(response_list[0], "LLM_UNEXPECTED_ERROR: An unexpected error occurred: Test generic exception")
        mock_logging.exception.assert_called_once_with("Unexpected error during LLM interaction setup/stream:")

if __name__ == '__main__':
    unittest.main()
