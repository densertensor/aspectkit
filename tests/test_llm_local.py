"""Tests for the in-process transformers connector, with duck-typed fakes."""

import pytest

from aspectkit.exceptions import LLMError
from aspectkit.llm.local import TransformersChat, looks_like_hf_model, looks_like_hf_pipeline

MESSAGES = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "hi"},
]


class FakeChatTokenizer:
    """Tokenizer with a chat template (instruction-tuned model)."""

    chat_template = "{{ messages }}"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        return "<rendered>" + "|".join(m["content"] for m in messages)

    def __call__(self, prompt, return_tensors=None):
        assert return_tensors == "pt"
        self.last_prompt = prompt
        return {"input_ids": FakeTensor([[1, 2, 3]]), "attention_mask": FakeTensor([[1, 1, 1]])}

    def decode(self, token_ids, skip_special_tokens=False):
        ids = token_ids.data if isinstance(token_ids, FakeTensor) else list(token_ids)
        return " ".join(f"tok{i}" for i in ids)


class FakeBareTokenizer(FakeChatTokenizer):
    """Tokenizer without a chat template (base model)."""

    chat_template = None


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def __getitem__(self, key):
        value = self.data[key]
        return FakeTensor(value) if isinstance(value, list) else value

    def to(self, device):
        return self


class FakeModel:
    device = "cpu"
    config = object()
    name_or_path = "fake/model"

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(kwargs)
        # echo the prompt ids plus two generated tokens
        return FakeTensor([[*input_ids.data[0], 7, 8]])


class FakePipeline:
    """Duck-typed text-generation pipeline."""

    def __init__(self, reply=" generated"):
        self.tokenizer = FakeChatTokenizer()
        self.model = FakeModel()
        self.calls = []
        self._reply = reply

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return [{"generated_text": self._reply}]


class TestDuckTyping:
    def test_pipeline_detection(self):
        assert looks_like_hf_pipeline(FakePipeline())
        assert not looks_like_hf_pipeline(FakeModel())
        assert not looks_like_hf_pipeline("a string")

    def test_model_detection(self):
        assert looks_like_hf_model(FakeModel())
        assert not looks_like_hf_model(object())


class TestPipelinePath:
    def test_completion(self):
        pipe = FakePipeline()
        llm = TransformersChat(pipe)
        reply = llm.complete(MESSAGES, max_tokens=11)
        assert reply == " generated"
        prompt, kwargs = pipe.calls[0]
        assert prompt.startswith("<rendered>")
        assert kwargs["max_new_tokens"] == 11
        assert kwargs["do_sample"] is False
        assert kwargs["return_full_text"] is False

    def test_generate_kwargs_forwarded(self):
        pipe = FakePipeline()
        TransformersChat(pipe, repetition_penalty=1.1).complete(MESSAGES)
        assert pipe.calls[0][1]["repetition_penalty"] == 1.1

    def test_empty_reply_raises(self):
        pipe = FakePipeline(reply="   ")
        with pytest.raises(LLMError, match="empty"):
            TransformersChat(pipe).complete(MESSAGES)


class TestModelTokenizerPath:
    def test_completion_slices_new_tokens(self):
        model, tokenizer = FakeModel(), FakeChatTokenizer()
        llm = TransformersChat(model, tokenizer)
        reply = llm.complete(MESSAGES, max_tokens=5)
        # only the two generated tokens, not the 3 prompt tokens
        assert reply == "tok7 tok8"
        kwargs = model.calls[0]
        assert kwargs["max_new_tokens"] == 5
        assert kwargs["do_sample"] is False
        assert kwargs["pad_token_id"] == 0

    def test_chat_template_used_when_available(self):
        tokenizer = FakeChatTokenizer()
        TransformersChat(FakeModel(), tokenizer).complete(MESSAGES)
        assert tokenizer.last_prompt.startswith("<rendered>")

    def test_fallback_transcript_without_template(self):
        tokenizer = FakeBareTokenizer()
        TransformersChat(FakeModel(), tokenizer).complete(MESSAGES)
        prompt = tokenizer.last_prompt
        assert "System: be terse" in prompt
        assert "User: hi" in prompt
        assert prompt.endswith("Assistant:")


class TestConstruction:
    def test_model_without_tokenizer_rejected(self):
        with pytest.raises(TypeError, match="tokenizer"):
            TransformersChat(FakeModel())

    def test_hub_id_is_lazy(self):
        # no transformers import or download at construction time
        llm = TransformersChat("some-org/some-model")
        assert llm.model == "some-org/some-model"
