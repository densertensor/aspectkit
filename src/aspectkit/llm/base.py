"""Minimal chat-model abstraction shared by all LLM connectors.

Design notes
------------
The interface is deliberately small: a connector turns a list of
OpenAI-style messages into a single completion string.  Sampling
configuration (temperature, etc.) belongs to the connector instance —
providers disagree about which parameters exist, so per-call generation
arguments are limited to what every provider supports (``max_tokens``)
plus an optional JSON schema that connectors apply natively when the
provider supports structured output and ignore gracefully otherwise.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["CallableChat", "ChatLLM", "Message", "UsageStats", "split_system"]

#: An OpenAI-style chat message: ``{"role": "system"|"user"|"assistant", "content": str}``.
Message = dict[str, str]

#: Exception class names treated as transient (retried with backoff) when the
#: connector cannot read an HTTP status code.
_TRANSIENT_NAMES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
    }
)


@dataclass(frozen=True)
class UsageStats:
    """Token usage reported by a connector, accumulable across calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


def is_transient_error(exc: Exception, statuses: tuple[int, ...] = (429, 503)) -> bool:
    """Whether *exc* is a transient rate-limit/overload error worth retrying."""
    if getattr(exc, "status_code", None) in statuses:
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES


def backoff_sleep(attempt: int, *, base: float = 1.0, cap: float = 30.0) -> None:
    """Sleep with full-jitter exponential backoff (``attempt`` is 1-based)."""
    delay = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(random.uniform(0, delay))


class ChatLLM(ABC):
    """Abstract chat model.

    Implementations wrap one provider (or one in-memory model) and are
    cheap, stateless-per-call objects: construct once, reuse across
    thousands of completions.
    """

    #: Token usage of the most recent :meth:`complete` call, when the provider
    #: reports it; ``None`` otherwise.  Connectors that can read usage set it,
    #: and :class:`~aspectkit.llm.wrappers.CountingChat` reads it.
    _last_usage: UsageStats | None = None

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate one completion for a conversation.

        Args:
            messages: OpenAI-style messages.  At most one leading system
                message is expected; connectors for providers with a
                dedicated system channel split it out automatically.
            max_tokens: Upper bound on generated tokens.
            json_schema: Optional JSON schema for the reply.  Connectors
                use the provider's native structured-output mechanism
                when available; otherwise the schema is advisory and
                callers must parse defensively (aspectkit's backends
                always do).

        Returns:
            The completion text.

        Raises:
            aspectkit.exceptions.LLMError: On transport failures,
                refusals, or empty completions.
        """

    @property
    def name(self) -> str:
        """Human-readable identifier used in logs and reports."""
        model = getattr(self, "model", None)
        suffix = f"({model})" if isinstance(model, str) else ""
        return f"{type(self).__name__}{suffix}"

    def __repr__(self) -> str:
        return self.name


def split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Separate system messages from the conversation.

    Providers like Anthropic and Gemini take the system prompt as a
    dedicated argument.  Multiple system messages are joined with blank
    lines, preserving order.

    Returns:
        ``(system_text_or_None, non_system_messages)``.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return ("\n\n".join(system_parts) if system_parts else None), rest


class CallableChat(ChatLLM):
    """Adapter turning any ``messages -> str`` callable into a connector.

    The escape hatch for custom endpoints, request-level caching, or
    deterministic test doubles::

        llm = CallableChat(lambda messages: my_gateway.chat(messages))

    Extra keyword arguments (``max_tokens``, ``json_schema``) are passed
    through only if the callable accepts them.
    """

    def __init__(self, fn: Callable[..., str], *, name: str | None = None) -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "callable")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        import inspect

        try:
            params = dict(inspect.signature(self._fn).parameters)
        except (TypeError, ValueError):
            params = {}
        kwargs: dict[str, Any] = {}
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_kwargs or "max_tokens" in params:
            kwargs["max_tokens"] = max_tokens
        if accepts_kwargs or "json_schema" in params:
            kwargs["json_schema"] = json_schema
        return self._fn(list(messages), **kwargs)

    @property
    def name(self) -> str:
        return f"CallableChat({self._name})"
