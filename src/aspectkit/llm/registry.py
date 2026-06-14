"""Connector resolution: turn a model spec into a :class:`ChatLLM`.

The single entry point is :func:`resolve_llm`.  It accepts:

* a ready :class:`~aspectkit.llm.base.ChatLLM` — returned unchanged;
* a ``"provider:model"`` string, e.g. ``"openai:gpt-4o-mini"``,
  ``"anthropic:claude-opus-4-8"``, ``"deepseek:deepseek-chat"``,
  ``"gemini:gemini-2.0-flash"``, ``"vllm:meta-llama/Llama-3.1-8B-Instruct"``,
  ``"hf:Qwen/Qwen2.5-7B-Instruct"``;
* an in-memory ``transformers`` pipeline, or a model object plus a
  ``tokenizer=`` keyword;
* a ``(model, tokenizer)`` tuple;
* any ``messages -> str`` callable.

Provider presets are thin: OpenAI-compatible providers differ only in
``base_url`` and the API-key environment variable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aspectkit.llm.anthropic import AnthropicChat
from aspectkit.llm.base import CallableChat, ChatLLM
from aspectkit.llm.gemini import GeminiChat
from aspectkit.llm.local import TransformersChat, looks_like_hf_model, looks_like_hf_pipeline
from aspectkit.llm.openai import OpenAIChat

__all__ = ["PROVIDERS", "resolve_llm"]

#: Provider presets: connector class plus default constructor arguments.
#: OpenAI-compatible local servers get permissive default credentials —
#: the conventional placeholder keys their docs prescribe.
PROVIDERS: dict[str, tuple[Callable[..., ChatLLM], dict[str, Any]]] = {
    "openai": (OpenAIChat, {}),
    "deepseek": (
        OpenAIChat,
        {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
    ),
    "mistral": (
        OpenAIChat,
        {"base_url": "https://api.mistral.ai/v1", "api_key_env": "MISTRAL_API_KEY"},
    ),
    "together": (
        OpenAIChat,
        {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    ),
    "groq": (
        OpenAIChat,
        {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    ),
    "openrouter": (
        OpenAIChat,
        {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    ),
    "vllm": (
        OpenAIChat,
        {"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"},
    ),
    "ollama": (
        OpenAIChat,
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    ),
    "openai-compatible": (OpenAIChat, {}),
    "anthropic": (AnthropicChat, {}),
    "claude": (AnthropicChat, {}),
    "gemini": (GeminiChat, {}),
    "google": (GeminiChat, {}),
    "hf": (TransformersChat, {}),
    "transformers": (TransformersChat, {}),
    "local": (TransformersChat, {}),
}


def _from_string(spec: str, kwargs: dict[str, Any]) -> ChatLLM:
    if ":" not in spec:
        providers = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"model spec {spec!r} is ambiguous; use 'provider:model' "
            f"(e.g. 'openai:gpt-4o-mini'). Known providers: {providers}"
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        providers = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known providers: {providers}")
    cls, defaults = PROVIDERS[provider]
    if provider == "openai-compatible" and "base_url" not in kwargs:
        raise ValueError("provider 'openai-compatible' requires a base_url=... argument")
    return cls(model.strip(), **{**defaults, **kwargs})


def resolve_llm(spec: Any, **kwargs: Any) -> ChatLLM:
    """Resolve a model spec into a connector.

    Args:
        spec: A :class:`ChatLLM`, a ``"provider:model"`` string, a
            ``transformers`` pipeline or model object, a
            ``(model, tokenizer)`` tuple, or a ``messages -> str``
            callable.
        **kwargs: Forwarded to the connector constructor (e.g.
            ``api_key=...``, ``base_url=...``, ``temperature=...``,
            ``tokenizer=...`` for bare model objects).

    Returns:
        A ready-to-use :class:`ChatLLM`.

    Raises:
        ValueError: For ambiguous strings or unknown providers.
        TypeError: For specs of unsupported type, or extra kwargs with
            an already-constructed connector.
    """
    if isinstance(spec, ChatLLM):
        if kwargs:
            raise TypeError(
                f"{type(spec).__name__} is already constructed; "
                f"unexpected arguments: {sorted(kwargs)}"
            )
        return spec
    if isinstance(spec, str):
        return _from_string(spec, kwargs)
    if isinstance(spec, tuple) and len(spec) == 2:
        return TransformersChat(spec[0], spec[1], **kwargs)
    if looks_like_hf_pipeline(spec):
        return TransformersChat(spec, **kwargs)
    if looks_like_hf_model(spec):
        # Checked before the generic callable branch: model objects are
        # callable (nn.Module) but must not be wrapped as plain callables.
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            raise TypeError(
                "a transformers model object needs its tokenizer: "
                "resolve_llm(model, tokenizer=tokenizer)"
            )
        return TransformersChat(spec, tokenizer, **kwargs)
    if callable(spec):
        return CallableChat(spec, **kwargs)
    raise TypeError(
        f"cannot resolve {type(spec).__name__} into a chat model; expected a "
        "ChatLLM, 'provider:model' string, transformers pipeline/model, "
        "(model, tokenizer) tuple, or callable"
    )
