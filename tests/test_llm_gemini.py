"""Tests for the Gemini connector, with a scripted fake client."""

from types import SimpleNamespace

import pytest

from aspectkit.exceptions import LLMError
from aspectkit.llm.gemini import GeminiChat


def make_client(script):
    calls = []
    queue = list(script)

    def generate_content(**kwargs):
        calls.append(kwargs)
        action = queue.pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(text=action)

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)), calls


class TestRequestShape:
    def test_roles_and_system_mapping(self):
        client, calls = make_client(["ok"])
        llm = GeminiChat("gemini-2.0-flash", client=client)
        llm.complete(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            max_tokens=55,
        )
        (call,) = calls
        assert call["model"] == "gemini-2.0-flash"
        assert [c["role"] for c in call["contents"]] == ["user", "model", "user"]
        assert call["contents"][0]["parts"] == [{"text": "one"}]
        config = call["config"]
        assert config["system_instruction"] == "be terse"
        assert config["max_output_tokens"] == 55
        assert config["temperature"] == 0.0

    def test_json_mode_when_schema_given(self):
        client, calls = make_client(["{}"])
        GeminiChat("m", client=client).complete(
            [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
        )
        assert calls[0]["config"]["response_mime_type"] == "application/json"

    def test_temperature_none_omitted(self):
        client, calls = make_client(["ok"])
        GeminiChat("m", client=client, temperature=None).complete(
            [{"role": "user", "content": "hi"}]
        )
        assert "temperature" not in calls[0]["config"]


class TestErrors:
    def test_empty_reply(self):
        client, _ = make_client([None])
        with pytest.raises(LLMError, match="empty"):
            GeminiChat("m", client=client).complete([{"role": "user", "content": "hi"}])

    def test_transport_error_wrapped(self):
        client, _ = make_client([RuntimeError("quota")])
        with pytest.raises(LLMError, match="quota"):
            GeminiChat("m", client=client).complete([{"role": "user", "content": "hi"}])


class RateLimitError(Exception):
    status_code = 429


class TestRetryAndUsage:
    def test_retries_transient_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("aspectkit.llm.gemini.backoff_sleep", lambda *a, **k: None)
        client, calls = make_client([RateLimitError(), "ok"])
        out = GeminiChat("m", client=client).complete([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert len(calls) == 2

    def test_usage_captured(self):
        def generate_content(**kwargs):
            return SimpleNamespace(
                text="ok",
                usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=2),
            )

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        llm = GeminiChat("m", client=client)
        llm.complete([{"role": "user", "content": "hi"}])
        assert (llm._last_usage.prompt_tokens, llm._last_usage.completion_tokens) == (9, 2)
        assert llm._last_usage.total_tokens == 11
