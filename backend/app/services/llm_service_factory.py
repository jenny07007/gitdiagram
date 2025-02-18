from typing import Literal
from app.services.base_llm_service import BaseLLMService
from app.services.o3_mini_openai_service import OpenAIO3Service
from app.services.o3_mini_openrouter_service import OpenRouterO3Service
from app.services.claude_service import ClaudeService

LLMServiceType = Literal["openai-o3", "openrouter-o3", "claude"]


class LLMServiceFactory:
    @staticmethod
    def create_service(service_type: LLMServiceType) -> BaseLLMService:
        """Creates and returns the specified LLM service"""
        if service_type == "openai-o3":
            return OpenAIO3Service()
        elif service_type == "openrouter-o3":
            return OpenRouterO3Service()
        elif service_type == "claude":
            return ClaudeService()
        else:
            raise ValueError(f"Unknown service type: {service_type}")
