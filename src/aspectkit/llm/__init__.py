"""Chat-model connectors.

One small interface (:class:`ChatLLM`) spans the LLM landscape:

* :class:`OpenAIChat` — OpenAI and every OpenAI-compatible endpoint
  (DeepSeek, vLLM, Ollama, Together, Groq, OpenRouter, Mistral, ...);
* :class:`AnthropicChat` — Anthropic Messages API;
* :class:`GeminiChat` — Google Gemini;
* :class:`TransformersChat` — in-process Hugging Face models (hub id,
  pipeline object, or model/tokenizer pair);
* :class:`CallableChat` — any ``messages -> str`` callable.

Use :func:`resolve_llm` to construct a connector from a spec such as
``"openai:gpt-4o-mini"`` or a live ``transformers`` object.
"""

from __future__ import annotations

from aspectkit.llm.anthropic import AnthropicChat
from aspectkit.llm.base import CallableChat, ChatLLM, Message, split_system
from aspectkit.llm.gemini import GeminiChat
from aspectkit.llm.local import TransformersChat
from aspectkit.llm.openai import OpenAIChat
from aspectkit.llm.registry import PROVIDERS, resolve_llm

__all__ = [
    "PROVIDERS",
    "AnthropicChat",
    "CallableChat",
    "ChatLLM",
    "GeminiChat",
    "Message",
    "OpenAIChat",
    "TransformersChat",
    "resolve_llm",
    "split_system",
]
