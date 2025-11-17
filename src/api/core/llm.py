from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.api.config import settings


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client wrapper"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using OpenAI"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS
        )

        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic (Claude) API client wrapper"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using Anthropic Claude"""

        # Convert messages format (OpenAI -> Anthropic)
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append(msg)

        response = await self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=conversation_messages,
            temperature=temperature or settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS
        )

        return response.content[0].text


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client"""

    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        return OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL
        )

    elif settings.LLM_PROVIDER == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        return AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.LLM_MODEL
        )

    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
