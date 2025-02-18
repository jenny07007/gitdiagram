from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""

    @abstractmethod
    def call_api(
        self,
        system_prompt: str,
        data: dict,
        api_key: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
    ) -> str:
        """Makes a synchronous API call to the LLM service"""
        pass

    @abstractmethod
    async def call_api_stream(
        self,
        system_prompt: str,
        data: dict,
        api_key: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
    ) -> AsyncGenerator[str, None]:
        """Makes a streaming API call to the LLM service"""
        pass

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts tokens in the prompt"""
        pass
