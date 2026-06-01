"""Provider abstraction layer for multi-API support.

Supports OpenAI, Anthropic, MiniMax, and Azure with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import os


@dataclass
class ProviderResponse:
    """Unified response from a provider."""

    content: str
    input_tokens: int
    output_tokens: int
    raw: Any = None
    tool_calls: list[dict[str, Any]] | None = None


class BaseProvider(ABC):
    """Abstract base class for API providers."""

    @abstractmethod
    def chat(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send a chat request and return unified response."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI API provider using the OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self._client = None
        self._init_client()

    def _init_client(self):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def chat(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send chat request via OpenAI SDK."""
        temperature = kwargs.get("temperature", self.temperature)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if kwargs.get("tools"):
            request["tools"] = kwargs["tools"]
            request["tool_choice"] = kwargs.get("tool_choice", "auto")
        response = self._client.chat.completions.create(**request)
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw=response,
            tool_calls=_openai_tool_calls(response),
        )


class AnthropicProvider(BaseProvider):
    """Anthropic API provider using the Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self._client = None
        self._init_client()

    def _init_client(self):
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def chat(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send chat request via Anthropic SDK."""
        temperature = kwargs.get("temperature", self.temperature)
        # Convert messages format for Anthropic
        # Anthropic uses roles: user, assistant (not system in messages)
        anthropic_messages = []
        system_content = None

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        response = self._client.messages.create(
            model=model,
            messages=anthropic_messages,
            system=system_content,
            temperature=temperature,
            max_tokens=4096,
        )
        return ProviderResponse(
            content=response.content[0].text if response.content else "",
            input_tokens=response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0,
            output_tokens=response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0,
            raw=response,
        )


class MiniMaxProvider(BaseProvider):
    """MiniMax API provider (compatible with OpenAI SDK)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self._client = None
        self._init_client()

    def _init_client(self):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "minimax"

    def chat(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send chat request via OpenAI SDK (MiniMax is OpenAI-compatible)."""
        temperature = kwargs.get("temperature", self.temperature)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if kwargs.get("tools"):
            request["tools"] = kwargs["tools"]
            request["tool_choice"] = kwargs.get("tool_choice", "auto")
        response = self._client.chat.completions.create(**request)
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw=response,
            tool_calls=_openai_tool_calls(response),
        )


class AzureProvider(BaseProvider):
    """Azure OpenAI API provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        api_version: str = "2024-02-01",
    ):
        if not base_url:
            raise ValueError("Azure requires a base_url endpoint")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.api_version = api_version
        self._client = None
        self._init_client()

    def _init_client(self):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version=self.api_version,
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "azure"

    def chat(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send chat request via Azure OpenAI SDK."""
        temperature = kwargs.get("temperature", self.temperature)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if kwargs.get("tools"):
            request["tools"] = kwargs["tools"]
            request["tool_choice"] = kwargs.get("tool_choice", "auto")
        response = self._client.chat.completions.create(**request)
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw=response,
            tool_calls=_openai_tool_calls(response),
        )


def _openai_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract OpenAI-compatible tool calls into provider-neutral dictionaries."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return []
    calls = getattr(message, "tool_calls", None) or []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        if not name:
            continue
        normalized.append(
            {
                "id": getattr(call, "id", None),
                "name": name,
                "arguments": getattr(function, "arguments", "") or "{}",
            }
        )
    return normalized


def create_provider(
    provider_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    temperature: float = 0.0,
) -> BaseProvider:
    """Factory function to create a provider instance.

    Args:
        provider_name: One of "openai", "anthropic", "minimax", "azure"
        api_key: API key for authentication
        base_url: API endpoint URL (required for Azure)
        timeout: Request timeout in seconds
        temperature: Default temperature for requests

    Returns:
        An instance of the appropriate provider class

    Raises:
        ValueError: If provider_name is unknown or base_url is missing for Azure
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "minimax": MiniMaxProvider,
        "azure": AzureProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")

    if provider_name == "azure":
        if not base_url:
            raise ValueError("Azure provider requires a base_url endpoint")
        return AzureProvider(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

    # For OpenAI-compatible providers
    preset_defaults = {
        "openai": {"base_url": "https://api.openai.com/v1"},
        "anthropic": {"base_url": "https://api.anthropic.com"},
        "minimax": {"base_url": "https://api.minimax.io/v1"},
    }

    defaults = preset_defaults.get(provider_name, {})
    final_base_url = base_url or defaults.get("base_url", "")

    return providers[provider_name](
        api_key=api_key,
        base_url=final_base_url,
        timeout=timeout,
        temperature=temperature,
    )
