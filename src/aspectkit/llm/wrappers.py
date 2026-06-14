"""Composable :class:`~aspectkit.llm.base.ChatLLM` wrappers.

Cross-cutting connector concerns are expressed as decorators over any
``ChatLLM`` rather than flags baked into a backend: a wrapper holds an
``inner`` connector and adds one behaviour, so users compose exactly the
capabilities they want, in the order they want::

    from aspectkit.llm import OpenAIChat, RetryingChat, CachingChat, CountingChat

    llm = CountingChat(RetryingChat(CachingChat(OpenAIChat("gpt-4o-mini"),
                                                cache_dir=".absa_cache")))

Each wrapper passes through :func:`~aspectkit.llm.registry.resolve_llm`'s
"a pre-built connector is used as-is" path, so wrapped connectors plug
into any backend unchanged.  Wrappers are independent of the
base-connector safety nets (e.g. the bounded 429 retry inside
``OpenAIChat``); :class:`RetryingChat` adds a configurable outer retry on
top.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aspectkit.llm.base import (
    ChatLLM,
    Message,
    UsageStats,
    backoff_sleep,
    is_transient_error,
)

__all__ = ["CachingChat", "CountingChat", "RetryingChat"]


class _Wrapper(ChatLLM):
    """Base for single-responsibility connector decorators."""

    def __init__(self, inner: ChatLLM) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.inner.name})"


class RetryingChat(_Wrapper):
    """Retry transient (rate-limit / overload) failures with backoff.

    Sits outside any connector and retries when the call raises a
    transient error — either directly, or wrapped (connectors raise
    :class:`~aspectkit.exceptions.LLMError` ``from`` the original, whose
    cause is inspected).  Non-transient errors propagate immediately.

    Args:
        inner: The connector to wrap.
        max_retries: Maximum retries after the first attempt.
        retry_on: HTTP status codes treated as transient.
        base_delay: Base seconds for full-jitter exponential backoff.
        max_delay: Backoff ceiling in seconds.
    """

    def __init__(
        self,
        inner: ChatLLM,
        *,
        max_retries: int = 6,
        retry_on: tuple[int, ...] = (429, 503),
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        super().__init__(inner)
        self.max_retries = max_retries
        self.retry_on = tuple(retry_on)
        self.base_delay = base_delay
        self.max_delay = max_delay

    def _retryable(self, exc: Exception) -> bool:
        if is_transient_error(exc, self.retry_on):
            return True
        cause = exc.__cause__
        return isinstance(cause, Exception) and is_transient_error(cause, self.retry_on)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        attempt = 0
        while True:
            try:
                result = self.inner.complete(
                    messages, max_tokens=max_tokens, json_schema=json_schema
                )
                self._last_usage = getattr(self.inner, "_last_usage", None)
                return result
            except Exception as exc:
                if attempt < self.max_retries and self._retryable(exc):
                    attempt += 1
                    backoff_sleep(attempt, base=self.base_delay, cap=self.max_delay)
                    continue
                raise


class CachingChat(_Wrapper):
    """Disk-backed exact-match cache over a connector.

    Keys completions by ``sha256`` of the inner connector's name, the
    messages, and (unless excluded) ``max_tokens`` and ``json_schema``.
    Re-running a corpus after a prompt iteration costs nothing for
    unchanged inputs.  Stdlib only; writes are atomic and safe under
    concurrency (a uniquely-named temp file per writer, then rename).

    Args:
        inner: The connector to wrap.
        cache_dir: Directory for cached responses (created if absent).
        ignore_keys: Names among ``("max_tokens", "json_schema")`` to
            exclude from the cache key.
    """

    def __init__(
        self,
        inner: ChatLLM,
        *,
        cache_dir: str | Path,
        ignore_keys: Sequence[str] = (),
    ) -> None:
        super().__init__(inner)
        unknown = set(ignore_keys) - {"max_tokens", "json_schema"}
        if unknown:
            raise ValueError(
                "ignore_keys must be a subset of {'max_tokens', 'json_schema'}; "
                f"got unknown {sorted(unknown)}"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ignore_keys = frozenset(ignore_keys)
        #: Cache hit / miss counters since construction.
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _key(
        self, messages: Sequence[Message], max_tokens: int, json_schema: dict[str, Any] | None
    ) -> str:
        payload: dict[str, Any] = {"model": self.inner.name, "messages": list(messages)}
        if "max_tokens" not in self.ignore_keys:
            payload["max_tokens"] = max_tokens
        if "json_schema" not in self.ignore_keys:
            payload["json_schema"] = json_schema
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        key = self._key(messages, max_tokens, json_schema)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            self._last_usage = UsageStats(calls=1)  # a hit costs no tokens
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        content = self.inner.complete(messages, max_tokens=max_tokens, json_schema=json_schema)
        with self._lock:
            self.misses += 1
        self._last_usage = getattr(self.inner, "_last_usage", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Per-writer temp name: concurrent misses on the same key (duplicate
        # texts under LLMBackend(concurrency>1)) must not share a temp file, or
        # the losing thread's tmp.replace() would raise FileNotFoundError.
        tmp = path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic publish
        return content


class CountingChat(_Wrapper):
    """Accumulate token usage across completions (thread-safe).

    Reads each call's usage from the inner connector's ``_last_usage``
    (the seam every bundled connector populates) and sums it into
    :attr:`total`; falls back to counting the call when the provider
    reports no usage.

    Args:
        inner: The connector to wrap.
        on_usage: Optional callback invoked with each call's
            :class:`~aspectkit.llm.base.UsageStats`.
    """

    def __init__(
        self,
        inner: ChatLLM,
        *,
        on_usage: Callable[[UsageStats], None] | None = None,
    ) -> None:
        super().__init__(inner)
        self.on_usage = on_usage
        #: Cumulative usage since construction (or the last :meth:`reset`).
        self.total = UsageStats()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Zero the cumulative :attr:`total`."""
        with self._lock:
            self.total = UsageStats()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        content = self.inner.complete(messages, max_tokens=max_tokens, json_schema=json_schema)
        reported = getattr(self.inner, "_last_usage", None) or UsageStats()
        per_call = UsageStats(
            calls=1,
            prompt_tokens=reported.prompt_tokens,
            completion_tokens=reported.completion_tokens,
        )
        with self._lock:
            self.total = self.total + per_call
        self._last_usage = per_call
        if self.on_usage is not None:
            self.on_usage(per_call)
        return content
