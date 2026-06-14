"""Connector for OpenAI and OpenAI-compatible chat-completions endpoints.

One class covers the OpenAI API itself and every server speaking its
protocol — DeepSeek, vLLM, Ollama, llama.cpp server, Together, Groq,
OpenRouter, Mistral, LM Studio, ... — by pointing ``base_url`` at the
endpoint.  Named presets for common providers live in
:mod:`aspectkit.llm.registry`.

The connector degrades gracefully across protocol dialects: if the
server rejects ``json_schema`` response formats it falls back to
``json_object`` and then to plain prompting; if it requires
``max_completion_tokens`` instead of ``max_tokens`` (newer OpenAI
models) or rejects an explicit ``temperature``, the offending parameter
is adjusted once and remembered for the rest of the session.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

from aspectkit.exceptions import LLMError, MissingDependencyError
from aspectkit.llm.base import ChatLLM, Message, UsageStats, backoff_sleep, is_transient_error

__all__ = ["OpenAIChat"]

_SCHEMA_MODES = ("schema", "json_object", "none")
#: Backoff retries for transient (429/503) errors before giving up.  For finer
#: control, wrap the connector in :class:`~aspectkit.llm.wrappers.RetryingChat`.
_RATE_LIMIT_RETRIES = 3


def _is_bad_request(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 400:
        return True
    return type(exc).__name__ == "BadRequestError"


class OpenAIChat(ChatLLM):
    """Chat connector for OpenAI-protocol endpoints.

    Args:
        model: Model identifier as known by the endpoint
            (e.g. ``"gpt-4o-mini"``, ``"deepseek-chat"``, or a local
            model name served by vLLM).
        api_key: API key.  Defaults to the environment variable named by
            ``api_key_env``.  Local servers usually accept any value
            (the registry presets handle this).
        api_key_env: Environment variable consulted when ``api_key`` is
            not given.
        base_url: Endpoint base URL; ``None`` means the official OpenAI
            API.
        temperature: Sampling temperature; ``None`` leaves the provider
            default.  Defaults to ``0.0`` for reproducible extraction.
        timeout: Per-request timeout in seconds.
        max_retries: Transport-level retries delegated to the SDK.
        client: Pre-built ``openai.OpenAI``-compatible client (dependency
            injection for tests or custom transports); when given, the
            other connection arguments are ignored.
        **request_kwargs: Extra keyword arguments forwarded verbatim to
            every ``chat.completions.create`` call (e.g. ``seed=42``).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float | None = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
        **request_kwargs: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._temperature = temperature
        self._request_kwargs = request_kwargs
        self._token_param = "max_tokens"
        self._schema_mode = _SCHEMA_MODES[0]
        # Guards the dialect read-modify-write in _downgrade_after when a
        # single connector is shared across the LLMBackend thread pool.
        self._dialect_lock = threading.Lock()
        self._last_usage: UsageStats | None = None

        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai", "OpenAIChat") from exc
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        if key is None and base_url is None:
            raise LLMError(
                f"no API key: pass api_key=... or set the {api_key_env} environment variable"
            )
        self._client = OpenAI(
            api_key=key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _response_format(self, json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        if json_schema is None or self._schema_mode == "none":
            return None
        if self._schema_mode == "schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "aspectkit_output", "schema": json_schema, "strict": True},
            }
        return {"type": "json_object"}

    def _downgrade_after(self, exc: Exception, sent_schema: bool) -> bool:
        """Adjust dialect settings after a 400; return True to retry."""
        if not _is_bad_request(exc):
            return False
        message = str(exc)
        if self._token_param == "max_tokens" and "max_completion_tokens" in message:
            self._token_param = "max_completion_tokens"
            return True
        if self._temperature is not None and "temperature" in message:
            self._temperature = None
            return True
        if sent_schema and self._schema_mode != "none":
            idx = _SCHEMA_MODES.index(self._schema_mode)
            self._schema_mode = _SCHEMA_MODES[idx + 1]
            return True
        return False

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        downgrades = 0
        rate_retries = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                self._token_param: max_tokens,
                **self._request_kwargs,
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response_format = self._response_format(json_schema)
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                with self._dialect_lock:
                    downgrade = downgrades < 4 and self._downgrade_after(
                        exc, response_format is not None
                    )
                if downgrade:
                    downgrades += 1
                    continue
                if is_transient_error(exc) and rate_retries < _RATE_LIMIT_RETRIES:
                    rate_retries += 1
                    backoff_sleep(rate_retries)
                    continue
                raise LLMError(f"chat completion failed for {self.name}: {exc}") from exc
            content = response.choices[0].message.content
            if not content:
                raise LLMError(f"{self.name} returned an empty completion")
            usage = getattr(response, "usage", None)
            self._last_usage = (
                UsageStats(
                    calls=1,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
                if usage is not None
                else None
            )
            return content
