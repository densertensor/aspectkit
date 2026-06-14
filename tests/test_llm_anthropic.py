"""Tests for the Anthropic connector, with a scripted fake client."""

from types import SimpleNamespace

import pytest

from aspectkit.exceptions import LLMError
from aspectkit.llm.anthropic import AnthropicChat

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class BadRequestError(Exception):
    status_code = 400


def reply(*texts, stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def make_client(script):
    calls = []
    queue = list(script)

    def create(**kwargs):
        calls.append(kwargs)
        action = queue.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


class TestRequestShape:
    def test_system_routed_to_parameter(self):
        client, calls = make_client([reply("ok")])
        llm = AnthropicChat("claude-opus-4-8", client=client)
        llm.complete(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=99,
        )
        (call,) = calls
        assert call["system"] == "be terse"
        assert call["messages"] == [{"role": "user", "content": "hi"}]
        assert call["max_tokens"] == 99
        assert call["model"] == "claude-opus-4-8"
        assert "temperature" not in call  # default None: modern models reject it

    def test_no_system_key_when_absent(self):
        client, calls = make_client([reply("ok")])
        AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])
        assert "system" not in calls[0]

    def test_explicit_temperature_sent(self):
        client, calls = make_client([reply("ok")])
        AnthropicChat("m", client=client, temperature=0.2).complete(
            [{"role": "user", "content": "hi"}]
        )
        assert calls[0]["temperature"] == 0.2

    def test_schema_sent_as_output_config(self):
        client, calls = make_client([reply("{}")])
        AnthropicChat("m", client=client).complete(
            [{"role": "user", "content": "hi"}], json_schema=SCHEMA
        )
        assert calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}

    def test_multiple_text_blocks_joined(self):
        client, _ = make_client([reply("part one ", "part two")])
        text = AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])
        assert text == "part one part two"


class TestDowngrades:
    def test_schema_rejected_falls_back_to_plain(self):
        client, calls = make_client(
            [BadRequestError("output_config: Extra inputs are not permitted"), reply("{}")]
        )
        llm = AnthropicChat("m", client=client)
        llm.complete([{"role": "user", "content": "hi"}], json_schema=SCHEMA)
        assert "output_config" not in calls[1]
        # remembered across calls
        client2, calls2 = make_client([reply("{}")])
        llm._client = client2
        llm.complete([{"role": "user", "content": "hi"}], json_schema=SCHEMA)
        assert "output_config" not in calls2[0]

    def test_rejected_temperature_dropped(self):
        client, calls = make_client(
            [BadRequestError("`temperature` is not supported"), reply("ok")]
        )
        AnthropicChat("m", client=client, temperature=0.5).complete(
            [{"role": "user", "content": "hi"}]
        )
        assert "temperature" not in calls[1]


class TestErrors:
    def test_refusal_raises(self):
        client, _ = make_client([reply("", stop_reason="refusal")])
        with pytest.raises(LLMError, match="refusal"):
            AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])

    def test_empty_completion(self):
        client, _ = make_client([SimpleNamespace(content=[], stop_reason="end_turn")])
        with pytest.raises(LLMError, match="empty"):
            AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])

    def test_transport_error_wrapped(self):
        client, _ = make_client([RuntimeError("connection reset")])
        with pytest.raises(LLMError, match="connection reset"):
            AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        anthropic_sdk = pytest.importorskip("anthropic")
        assert anthropic_sdk  # constructor path requires the SDK
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            AnthropicChat("claude-opus-4-8")


class RateLimitError(Exception):
    status_code = 429


class TestRetryAndUsage:
    def test_retries_transient_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("aspectkit.llm.anthropic.backoff_sleep", lambda *a, **k: None)
        client, calls = make_client([RateLimitError(), reply("ok")])
        out = AnthropicChat("m", client=client).complete([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert len(calls) == 2

    def test_usage_captured(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )
        client, _ = make_client([resp])
        llm = AnthropicChat("m", client=client)
        llm.complete([{"role": "user", "content": "hi"}])
        assert (llm._last_usage.prompt_tokens, llm._last_usage.completion_tokens) == (7, 3)
        assert llm._last_usage.total_tokens == 10
