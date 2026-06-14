"""Tests for the OpenAI-compatible connector, with a scripted fake client."""

from types import SimpleNamespace

import pytest

from aspectkit.exceptions import LLMError
from aspectkit.llm.openai import OpenAIChat

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "Return JSON."}]


class BadRequestError(Exception):
    status_code = 400


class ServerError(Exception):
    status_code = 500


def make_client(script):
    """Fake client: each script entry is an exception to raise or a reply string."""
    calls = []
    queue = list(script)

    def create(**kwargs):
        calls.append(kwargs)
        action = queue.pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=action))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


class TestRequestShape:
    def test_basic_call(self):
        client, calls = make_client(['{"ok": true}'])
        llm = OpenAIChat("gpt-4o-mini", client=client)
        reply = llm.complete(MESSAGES, max_tokens=64)
        assert reply == '{"ok": true}'
        (call,) = calls
        assert call["model"] == "gpt-4o-mini"
        assert call["messages"] == MESSAGES
        assert call["max_tokens"] == 64
        assert call["temperature"] == 0.0
        assert "response_format" not in call

    def test_temperature_none_omitted(self):
        client, calls = make_client(["ok"])
        OpenAIChat("m", client=client, temperature=None).complete(MESSAGES)
        assert "temperature" not in calls[0]

    def test_request_kwargs_forwarded(self):
        client, calls = make_client(["ok"])
        OpenAIChat("m", client=client, seed=7).complete(MESSAGES)
        assert calls[0]["seed"] == 7

    def test_schema_sent_as_strict_json_schema(self):
        client, calls = make_client(["{}"])
        OpenAIChat("m", client=client).complete(MESSAGES, json_schema=SCHEMA)
        fmt = calls[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["schema"] == SCHEMA
        assert fmt["json_schema"]["strict"] is True


class TestDialectDowngrades:
    def test_schema_falls_back_to_json_object(self):
        client, calls = make_client([BadRequestError("response_format not supported"), "{}"])
        llm = OpenAIChat("m", client=client)
        llm.complete(MESSAGES, json_schema=SCHEMA)
        assert calls[1]["response_format"] == {"type": "json_object"}
        # the downgrade is remembered for subsequent calls
        client2, calls2 = make_client(["{}"])
        llm._client = client2
        llm.complete(MESSAGES, json_schema=SCHEMA)
        assert calls2[0]["response_format"] == {"type": "json_object"}

    def test_json_object_falls_back_to_plain(self):
        client, calls = make_client([BadRequestError("nope"), BadRequestError("nope"), "{}"])
        OpenAIChat("m", client=client).complete(MESSAGES, json_schema=SCHEMA)
        assert "response_format" not in calls[2]

    def test_max_completion_tokens_switch(self):
        client, calls = make_client(
            [BadRequestError("Use 'max_completion_tokens' instead of 'max_tokens'"), "ok"]
        )
        OpenAIChat("m", client=client).complete(MESSAGES, max_tokens=33)
        assert "max_tokens" not in calls[1]
        assert calls[1]["max_completion_tokens"] == 33

    def test_unsupported_temperature_dropped(self):
        client, calls = make_client(
            [BadRequestError("'temperature' does not support 0.0 with this model"), "ok"]
        )
        OpenAIChat("m", client=client).complete(MESSAGES)
        assert "temperature" not in calls[1]


class TestErrors:
    def test_server_error_wrapped(self):
        client, _ = make_client([ServerError("boom")])
        with pytest.raises(LLMError, match="boom"):
            OpenAIChat("m", client=client).complete(MESSAGES)

    def test_bad_request_without_remedy_raises(self):
        client, _ = make_client([BadRequestError("model not found")])
        with pytest.raises(LLMError):
            OpenAIChat("m", client=client).complete(MESSAGES)

    def test_empty_completion(self):
        client, _ = make_client([""])
        with pytest.raises(LLMError, match="empty"):
            OpenAIChat("m", client=client).complete(MESSAGES)

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            OpenAIChat("gpt-4o-mini")

    def test_local_endpoint_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = OpenAIChat("served-model", base_url="http://localhost:8000/v1")
        assert llm.base_url == "http://localhost:8000/v1"

    def test_name(self):
        client, _ = make_client([])
        assert "gpt-4o-mini" in OpenAIChat("gpt-4o-mini", client=client).name


class RateLimitError(Exception):
    status_code = 429


class TestRetryAndUsage:
    def test_retries_transient_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("aspectkit.llm.openai.backoff_sleep", lambda *a, **k: None)
        client, calls = make_client([RateLimitError(), RateLimitError(), '{"ok": true}'])
        assert OpenAIChat("m", client=client).complete(MESSAGES) == '{"ok": true}'
        assert len(calls) == 3

    def test_transient_gives_up_after_bounded_retries(self, monkeypatch):
        monkeypatch.setattr("aspectkit.llm.openai.backoff_sleep", lambda *a, **k: None)
        client, calls = make_client([RateLimitError()] * 10)
        with pytest.raises(LLMError):
            OpenAIChat("m", client=client).complete(MESSAGES)
        assert len(calls) == 4  # 1 initial + 3 retries

    def test_usage_captured(self):
        def create(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = OpenAIChat("m", client=client)
        llm.complete(MESSAGES)
        assert (llm._last_usage.prompt_tokens, llm._last_usage.completion_tokens) == (11, 4)
        assert llm._last_usage.total_tokens == 15
        assert llm._last_usage.calls == 1

    def test_usage_absent_is_none(self):
        client, _ = make_client(["ok"])
        llm = OpenAIChat("m", client=client)
        llm.complete(MESSAGES)
        assert llm._last_usage is None
