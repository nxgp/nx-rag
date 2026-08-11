"""
Unit tests for Bedrock LLM Provider.
"""

from unittest.mock import MagicMock, patch

from mentera_rag.generation.bedrock import BedrockLLMProvider


class TestBedrockLLMProvider:
    """Tests for Bedrock LLM provider."""

    @patch("boto3.client")
    def test_bedrock_llm_provider_generate(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "Bedrock clinical answer"}]}}
        }
        mock_boto.return_value = mock_bedrock

        provider = BedrockLLMProvider(model_name="anthropic.claude-3-sonnet-20240229-v1:0")
        output = provider.generate("Medical question", system_prompt="System instructions")

        assert output == "Bedrock clinical answer"
        mock_bedrock.converse.assert_called_once()
