"""
Abstract LLM Provider Interface.

Defines standard contract for LLM completion & chat providers.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM generation engines (Bedrock, Groq, Local).
    """

    def __init__(self, model_name: str, max_tokens: int = 1024, temperature: float = 0.0):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate text completion for a prompt string.

        Args:
            prompt: User prompt / context string.
            system_prompt: Optional system instruction prompt.

        Returns:
            Generated text string response from LLM.
        """
        pass
