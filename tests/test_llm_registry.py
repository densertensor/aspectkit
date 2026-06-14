"""Tests for connector resolution."""

import pytest

from aspectkit.llm import (
    AnthropicChat,
    CallableChat,
    GeminiChat,
    OpenAIChat,
    TransformersChat,
    resolve_llm,
)
from tests.test_llm_local import FakeChatTokenizer, FakeModel, FakePipeline


class TestInstancePassthrough:
    def test_returned_unchanged(self):
        llm = CallableChat(lambda m: "ok")
        assert resolve_llm(llm) is llm

    def test_extra_kwargs_rejected(self):
        with pytest.raises(TypeError, match="already constructed"):
            resolve_llm(CallableChat(lambda m: "ok"), temperature=0.5)


class TestStringSpecs:
    def test_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        llm = resolve_llm("openai:gpt-4o-mini")
        assert isinstance(llm, OpenAIChat)
        assert llm.model == "gpt-4o-mini"
        assert llm.base_url is None

    def test_deepseek_preset(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        llm = resolve_llm("deepseek:deepseek-chat")
        assert isinstance(llm, OpenAIChat)
        assert llm.base_url == "https://api.deepseek.com"

    def test_vllm_preset_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = resolve_llm("vllm:meta-llama/Llama-3.1-8B-Instruct")
        assert llm.base_url == "http://localhost:8000/v1"
        assert llm.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_ollama_preset(self):
        llm = resolve_llm("ollama:llama3.1")
        assert llm.base_url == "http://localhost:11434/v1"

    def test_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        llm = resolve_llm("anthropic:claude-opus-4-8")
        assert isinstance(llm, AnthropicChat)
        assert llm.model == "claude-opus-4-8"

    def test_claude_alias(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert isinstance(resolve_llm("claude:claude-opus-4-8"), AnthropicChat)

    def test_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        llm = resolve_llm("gemini:gemini-2.0-flash")
        assert isinstance(llm, GeminiChat)

    def test_hf_lazy(self):
        llm = resolve_llm("hf:Qwen/Qwen2.5-7B-Instruct")
        assert isinstance(llm, TransformersChat)
        assert llm.model == "Qwen/Qwen2.5-7B-Instruct"

    def test_openai_compatible_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            resolve_llm("openai-compatible:my-model")

    def test_openai_compatible_with_base_url(self):
        llm = resolve_llm(
            "openai-compatible:my-model", base_url="http://gpu-box:8000/v1", api_key="x"
        )
        assert llm.base_url == "http://gpu-box:8000/v1"

    def test_kwargs_override_presets(self, monkeypatch):
        llm = resolve_llm("vllm:m", base_url="http://other:9000/v1")
        assert llm.base_url == "http://other:9000/v1"

    def test_bare_string_rejected_helpfully(self):
        with pytest.raises(ValueError, match="provider:model"):
            resolve_llm("gpt-4o-mini")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown provider"):
            resolve_llm("acme:model-x")


class TestObjectSpecs:
    def test_pipeline_object(self):
        llm = resolve_llm(FakePipeline())
        assert isinstance(llm, TransformersChat)

    def test_model_tokenizer_tuple(self):
        llm = resolve_llm((FakeModel(), FakeChatTokenizer()))
        assert isinstance(llm, TransformersChat)

    def test_model_with_tokenizer_kwarg(self):
        llm = resolve_llm(FakeModel(), tokenizer=FakeChatTokenizer())
        assert isinstance(llm, TransformersChat)

    def test_model_without_tokenizer_rejected(self):
        with pytest.raises(TypeError, match="tokenizer"):
            resolve_llm(FakeModel())

    def test_plain_callable(self):
        llm = resolve_llm(lambda messages: "ok")
        assert isinstance(llm, CallableChat)
        assert llm.complete([{"role": "user", "content": "x"}]) == "ok"

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="cannot resolve"):
            resolve_llm(42)
