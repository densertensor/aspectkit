"""Connector for the Google Gemini API (``google-genai`` SDK)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from aspectkit.exceptions import LLMError, MissingDependencyError
from aspectkit.llm.base import (
    ChatLLM,
    Message,
    UsageStats,
    backoff_sleep,
    is_transient_error,
    split_system,
)

__all__ = ["GeminiChat"]

#: Backoff retries for transient (429/503) errors; wrap in RetryingChat for more.
_RATE_LIMIT_RETRIES = 3


class GeminiChat(ChatLLM):
    """Chat connector for Gemini models.

    System messages map to ``system_instruction``; conversation roles
    map ``assistant`` → ``model``.  When a JSON schema is supplied the
    connector requests ``application/json`` output (the schema itself is
    carried in the prompt by aspectkit's backends, which keeps the
    connector robust across SDK schema dialects).

    Args:
        model: Model identifier, e.g. ``"gemini-2.0-flash"``.
        api_key: API key; defaults to ``GEMINI_API_KEY`` then
            ``GOOGLE_API_KEY``.
        temperature: Sampling temperature; ``None`` leaves the provider
            default.  Defaults to ``0.0`` for reproducible extraction.
        client: Pre-built ``google.genai.Client``-compatible client
            (dependency injection for tests).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._last_usage: UsageStats | None = None

        if client is not None:
            self._client = client
            return
        try:
            # Submodule form (not ``from google import genai``): ``google`` is a
            # namespace package, and the attribute form trips mypy's attr-defined
            # check under some google-genai versions.
            import google.genai as genai
        except ImportError as exc:
            raise MissingDependencyError("google-genai", "gemini", "GeminiChat") from exc
        key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if key is None:
            raise LLMError("no API key: pass api_key=... or set GEMINI_API_KEY / GOOGLE_API_KEY")
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        system, conversation = split_system(messages)
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in conversation
        ]
        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system is not None:
            config["system_instruction"] = system
        if self._temperature is not None:
            config["temperature"] = self._temperature
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
        rate_retries = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except Exception as exc:
                if is_transient_error(exc) and rate_retries < _RATE_LIMIT_RETRIES:
                    rate_retries += 1
                    backoff_sleep(rate_retries)
                    continue
                raise LLMError(f"generation failed for {self.name}: {exc}") from exc
        text = getattr(response, "text", None)
        if not text:
            raise LLMError(f"{self.name} returned an empty completion")
        usage = getattr(response, "usage_metadata", None)
        self._last_usage = (
            UsageStats(
                calls=1,
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )
            if usage is not None
            else None
        )
        return text
