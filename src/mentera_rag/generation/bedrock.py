"""
AWS Bedrock LLM Provider Implementation.

Uses boto3 bedrock-runtime converse API for Claude 3.5 Sonnet / Llama 3 models.
"""

import boto3

from mentera_rag.config.settings import settings
from mentera_rag.generation.base import BaseLLMProvider


class BedrockLLMProvider(BaseLLMProvider):
    """
    AWS Bedrock LLM provider using direct boto3 bedrock-runtime converse API.
    """

    def __init__(
        self,
        model_name: str | None = None,
        region_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        resolved_model = (
            model_name
            or getattr(settings, "BEDROCK_LLM_MODEL_ID", None)
            or "us.anthropic.claude-3-haiku-20240307-v1:0"
        )
        super().__init__(model_name=resolved_model, max_tokens=max_tokens, temperature=temperature)
        self.region_name = region_name or settings.AWS_REGION
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Execute Converse API call on Bedrock runtime."""
        system_list = [{"text": system_prompt}] if system_prompt else []
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            system=system_list,
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        output_message = response.get("output", {}).get("message", {})
        contents = output_message.get("content", [])
        if contents and "text" in contents[0]:
            return str(contents[0]["text"]).strip()
        return ""
