"""Tests for the composable ChatLLM wrappers (no network)."""

import threading

import pytest

from aspectkit.llm.base import ChatLLM, UsageStats
from aspectkit.llm.wrappers import CachingChat, CountingChat, RetryingChat

MSGS = [{"role": "user", "content": "q"}]


class RateLimitError(Exception):
    status_code = 429


class FakeChat(ChatLLM):
    """Scripted connector: each reply is a string or an exception (FIFO)."""

    def __init__(self, replies, *, usage=None, model="fake"):
        self.model = model
        self.replies = list(replies)
        self.calls = 0
        self._usage = usage

    def complete(self, messages, *, max_tokens=1024, json_schema=None):
        self.calls += 1
        action = self.replies.pop(0)
        if isinstance(action, Exception):
            raise action
        self._last_usage = self._usage
        return action


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("aspectkit.llm.wrappers.backoff_sleep", lambda *a, **k: None)


class TestRetryingChat:
    def test_retries_transient_then_succeeds(self):
        inner = FakeChat([RateLimitError(), RateLimitError(), "ok"])
        assert RetryingChat(inner).complete(MSGS) == "ok"
        assert inner.calls == 3

    def test_gives_up_after_max_retries(self):
        inner = FakeChat([RateLimitError()] * 10)
        with pytest.raises(RateLimitError):
            RetryingChat(inner, max_retries=2).complete(MSGS)
        assert inner.calls == 3  # initial + 2 retries

    def test_non_transient_propagates_immediately(self):
        inner = FakeChat([ValueError("boom")])
        with pytest.raises(ValueError):
            RetryingChat(inner).complete(MSGS)
        assert inner.calls == 1

    def test_retries_on_wrapped_transient_cause(self):
        wrapped = RuntimeError("chat completion failed")
        wrapped.__cause__ = RateLimitError()
        inner = FakeChat([wrapped, "ok"])
        assert RetryingChat(inner).complete(MSGS) == "ok"
        assert inner.calls == 2

    def test_name_chains(self):
        assert RetryingChat(FakeChat([])).name == "RetryingChat(FakeChat(fake))"


class TestCachingChat:
    def test_miss_then_hit(self, tmp_path):
        inner = FakeChat(["first"])
        cache = CachingChat(inner, cache_dir=tmp_path)
        assert cache.complete(MSGS) == "first"
        assert cache.complete(MSGS) == "first"
        assert inner.calls == 1
        assert (cache.hits, cache.misses) == (1, 1)

    def test_distinct_inputs_not_shared(self, tmp_path):
        inner = FakeChat(["a", "b"])
        cache = CachingChat(inner, cache_dir=tmp_path)
        assert cache.complete([{"role": "user", "content": "x"}]) == "a"
        assert cache.complete([{"role": "user", "content": "y"}]) == "b"
        assert inner.calls == 2

    def test_persists_across_instances(self, tmp_path):
        CachingChat(FakeChat(["cached"]), cache_dir=tmp_path).complete(MSGS)
        inner2 = FakeChat([])  # would IndexError if its complete() were reached
        assert CachingChat(inner2, cache_dir=tmp_path).complete(MSGS) == "cached"
        assert inner2.calls == 0

    def test_ignore_keys_shares_entry_across_max_tokens(self, tmp_path):
        inner = FakeChat(["once"])
        cache = CachingChat(inner, cache_dir=tmp_path, ignore_keys=("max_tokens",))
        assert cache.complete(MSGS, max_tokens=16) == "once"
        assert cache.complete(MSGS, max_tokens=999) == "once"  # different max_tokens, one entry
        assert inner.calls == 1

    def test_ignore_keys_rejects_unknown(self, tmp_path):
        with pytest.raises(ValueError, match="ignore_keys"):
            CachingChat(FakeChat([]), cache_dir=tmp_path, ignore_keys=("model",))

    def test_concurrent_same_key_writes_do_not_crash(self, tmp_path):
        # Every caller misses, then blocks until all have, so the writes race on
        # one key -- the deterministic temp path used to crash all but one writer.
        n = 8
        barrier = threading.Barrier(n)

        class BarrierChat(ChatLLM):
            model = "barrier"

            def complete(self, messages, *, max_tokens=1024, json_schema=None):
                barrier.wait()
                self._last_usage = None
                return "shared"

        cache = CachingChat(BarrierChat(), cache_dir=tmp_path)
        results: list[str] = []
        errors: list[Exception] = []

        def run() -> None:
            try:
                results.append(cache.complete(MSGS))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert results == ["shared"] * n
        assert cache.misses == n  # locked counter loses no concurrent increment


class TestCountingChat:
    def test_accumulates_usage(self):
        inner = FakeChat(
            ["a", "b"], usage=UsageStats(calls=1, prompt_tokens=10, completion_tokens=4)
        )
        counter = CountingChat(inner)
        counter.complete(MSGS)
        counter.complete(MSGS)
        assert (counter.total.calls, counter.total.prompt_tokens) == (2, 20)
        assert counter.total.completion_tokens == 8
        assert counter.total.total_tokens == 28

    def test_counts_calls_without_usage(self):
        counter = CountingChat(FakeChat(["a"], usage=None))
        counter.complete(MSGS)
        assert counter.total == UsageStats(calls=1)

    def test_on_usage_callback(self):
        seen = []
        inner = FakeChat(["a"], usage=UsageStats(calls=1, prompt_tokens=3, completion_tokens=1))
        CountingChat(inner, on_usage=seen.append).complete(MSGS)
        assert len(seen) == 1 and seen[0].total_tokens == 4

    def test_reset(self):
        counter = CountingChat(FakeChat(["a", "b"], usage=UsageStats(calls=1, prompt_tokens=5)))
        counter.complete(MSGS)
        counter.reset()
        counter.complete(MSGS)
        assert counter.total.prompt_tokens == 5


class TestComposition:
    def test_counting_over_retrying_over_caching(self, tmp_path):
        usage = UsageStats(calls=1, prompt_tokens=7, completion_tokens=2)
        inner = FakeChat([RateLimitError(), "ok"], usage=usage)
        stack = CountingChat(RetryingChat(CachingChat(inner, cache_dir=tmp_path)))
        assert stack.complete(MSGS) == "ok"  # retries past 429, then caches the result
        assert stack.complete(MSGS) == "ok"  # served from cache (no inner call)
        assert inner.calls == 2  # one failed + one successful real call
        assert stack.total.calls == 2  # two logical completions counted
        assert stack.total.prompt_tokens == 7  # only the real call billed; cache hit = 0
