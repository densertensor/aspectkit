"""Connector for the Anthropic Messages API."""

from __future__ import annotations

import os
import threading
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

__all__ = ["AnthropicChat"]

#: Backoff retries for transient (429/503) errors; wrap in RetryingChat for more.
_RATE_LIMIT_RETRIES = 3


def _is_bad_request(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 400:
        return True
    return type(exc).__name__ == "BadRequestError"


class AnthropicChat(ChatLLM):
    """Chat connector for Anthropic models (Claude family).

    System messages are routed to the API's dedicated ``system``
    parameter.  When a JSON schema is supplied, the connector uses the
    native structured-output mechanism (``output_config.format``); if
    the target model rejects it, the connector falls back to plain
    prompting once and remembers the choice.

    Args:
        model: Model identifier, e.g. ``"claude-opus-4-8"``.
        api_key: API key; defaults to the environment variable named by
            ``api_key_env``.
        api_key_env: Environment variable consulted when ``api_key`` is
            not given.
        base_url: Optional API base URL override (proxies, gateways).
        temperature: Sampling temperature, or ``None`` to leave the
            provider default.  ``None`` is the default because recent
            Anthropic models reject explicit sampling parameters.
        timeout: Per-request timeout in seconds.
        max_retries: Transport-level retries delegated to the SDK.
        client: Pre-built ``anthropic.Anthropic``-compatible client
            (dependency injection for tests); when given, the other
            connection arguments are ignored.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._schema_supported = True
        # Guards the dialect read-modify-write below when a single
        # connector is shared across the LLMBackend thread pool.
        self._dialect_lock = threading.Lock()
        self._last_usage: UsageStats | None = None

        if client is not None:
            self._client = client
            return
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "anthropic", "AnthropicChat") from exc
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        if key is None:
            raise LLMError(
                f"no API key: pass api_key=... or set the {api_key_env} environment variable"
            )
        client_kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = Anthropic(**client_kwargs)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        system, conversation = split_system(messages)
        downgrades = 0
        rate_retries = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": conversation,
            }
            if system is not None:
                kwargs["system"] = system
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            use_schema = json_schema is not None and self._schema_supported
            if use_schema:
                kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
            try:
                response = self._client.messages.create(**kwargs)
            except Exception as exc:
                if downgrades < 3 and _is_bad_request(exc):
                    message = str(exc)
                    with self._dialect_lock:
                        if use_schema and ("output_config" in message or "format" in message):
                            self._schema_supported = False
                            retry = True
                        elif self._temperature is not None and "temperature" in message:
                            self._temperature = None
                            retry = True
                        else:
                            retry = False
                    if retry:
                        downgrades += 1
                        continue
                if is_transient_error(exc) and rate_retries < _RATE_LIMIT_RETRIES:
                    rate_retries += 1
                    backoff_sleep(rate_retries)
                    continue
                raise LLMError(f"message request failed for {self.name}: {exc}") from exc

            if getattr(response, "stop_reason", None) == "refusal":
                raise LLMError(f"{self.name} declined the request (stop_reason=refusal)")
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            if not text:
                raise LLMError(f"{self.name} returned an empty completion")
            usage = getattr(response, "usage", None)
            self._last_usage = (
                UsageStats(
                    calls=1,
                    prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                )
                if usage is not None
                else None
            )
            return text
