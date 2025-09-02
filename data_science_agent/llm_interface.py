"""LLM interface and adapters.

This module defines a small abstraction layer over different language
model providers so that the rest of the agent does not need to
know which backend is being used. The base class :class:`BaseLLM`
encapsulates the simple callable interface used throughout the agent.

Two concrete adapters are provided for convenience:

- :class:`OpenRouterLLM` uses the `openrouter` client library to
  access any of the models hosted by OpenRouter. See
  https://openrouter.ai/docs for API details.
- :class:`LiteLLM` uses the `litellm` client library to access
  supported models from providers like OpenAI, Anthropic, Cohere, etc.

If neither library is installed the adapter will fall back to using
the Anthropic chat client from :mod:`langchain_anthropic` if available.

All adapters are initialised with an API key and a model name. The
agent will select the appropriate adapter based on the ``llm_provider``
argument passed to :class:`data_science_agent.agent.DataScienceAgent`.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

try:
    from openrouter import OpenRouter
except ImportError:  # pragma: no cover - optional dependency
    OpenRouter = None  # type: ignore

try:
    import litellm
except ImportError:  # pragma: no cover - optional dependency
    litellm = None  # type: ignore

try:
    # As a last resort we fall back to the Anthropic client provided via LangChain
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None  # type: ignore


class BaseLLM:
    """Abstract base class for language models used by the agent.

    Subclasses must implement the :meth:`__call__` method which
    accepts a list of messages and returns a response object with a
    ``content`` attribute containing the generated text.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    def __call__(self, messages: List[Dict[str, Any]]):  # pragma: no cover - interface
        raise NotImplementedError


class OpenRouterLLM(BaseLLM):
    """Adapter for the OpenRouter API.

    Requires the ``openrouter`` Python client to be installed. The
    client expects the API key to be set via the ``OPENROUTER_API_KEY``
    environment variable.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        if OpenRouter is None:
            raise RuntimeError(
                "openrouter library is not installed. Install via `pip install openrouter`"
            )
        super().__init__(model_name, api_key, **kwargs)
        # openrouter uses an environment variable for authentication
        os.environ["OPENROUTER_API_KEY"] = api_key
        self.client = OpenRouter()  # type: ignore[assignment]

    def __call__(self, messages: List[Dict[str, Any]]):
        # The OpenRouter client expects a list of dicts with role and content keys
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        # The shape of the returned object may vary; we normalise to expose
        # a content attribute for consistency with LangChain behaviour.
        return type("Response", (), {"content": response.choices[0].message.content})


class LiteLLM(BaseLLM):
    """Adapter for the litellm library.

    LiteLLM acts as a unified interface to multiple providers. See
    https://github.com/BerriAI/litellm for supported models. To use
    models requiring API keys (e.g. Anthropic or OpenAI), set the
    appropriate key in the environment (``ANTHROPIC_API_KEY`` or
    ``OPENAI_API_KEY``) or supply ``api_key`` directly when
    instantiating this class.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        if litellm is None:
            raise RuntimeError(
                "litellm library is not installed. Install via `pip install litellm`"
            )
        super().__init__(model_name, api_key, **kwargs)
        self.model_name = model_name
        # litellm uses global API keys per provider; we set environment variables
        # depending on the model prefix if not already set
        if model_name.startswith("claude"):
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif model_name.startswith("gpt"):
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        else:
            # generic fallback for other providers
            os.environ.setdefault("LITELLM_API_KEY", api_key)
        self.kwargs = kwargs

    def __call__(self, messages: List[Dict[str, Any]]):
        # litellm accepts a flat list of message dictionaries. It returns a dict
        # with the key "choices" which contains a message with a "content" field.
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        return type("Response", (), {"content": response["choices"][0]["message"]["content"]})


class AnthropicLLM(BaseLLM):
    """Fallback adapter using the Anthropic chat client from LangChain.

    This is used when neither OpenRouter nor litellm is installed. You
    can still pass Anthropic-specific models like ``claude-3-sonnet``.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        if ChatAnthropic is None:
            raise RuntimeError(
                "langchain_anthropic library is not installed. Install via `pip install langchain-anthropic`"
            )
        super().__init__(model_name, api_key, **kwargs)
        self.client = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model_name,
            **kwargs
        )

    def __call__(self, messages: List[Dict[str, Any]]):
        # The LangChain client expects a list of message objects but we
        # accept plain dicts for consistency. Convert them here.
        from langchain.schema import HumanMessage, SystemMessage, AIMessage

        def convert(msg: Dict[str, Any]):
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                return SystemMessage(content=content)
            elif role == "assistant":
                return AIMessage(content=content)
            else:
                return HumanMessage(content=content)

        lc_messages = [convert(m) for m in messages]
        response = self.client(lc_messages)
        return type("Response", (), {"content": response.content})


def get_llm(provider: str, model_name: str, api_key: str, **kwargs: Any) -> BaseLLM:
    """Factory function to return an appropriate LLM adapter.

    :param provider: one of ``"openrouter"``, ``"litellm"`` or ``"anthropic"``
    :param model_name: the name of the model to use
    :param api_key: API key for authentication
    :param kwargs: additional keyword arguments passed to the underlying client
    """
    provider = provider.lower()
    if provider == "openrouter":
        return OpenRouterLLM(model_name, api_key, **kwargs)
    elif provider == "litellm":
        return LiteLLM(model_name, api_key, **kwargs)
    elif provider == "anthropic":
        return AnthropicLLM(model_name, api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported llm_provider '{provider}'. Choose from 'openrouter', 'litellm', or 'anthropic'.")