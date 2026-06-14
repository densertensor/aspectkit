"""Tests for the seq2seq backend, with scripted model/tokenizer fakes.

The generation seam (tokenize -> generate -> decode) is faked so the
predict contract — template decoding, span alignment, label
canonicalisation, batching, diagnostics — is tested deterministically
and without torch.  Real-model execution (including fit) is covered in
``test_integration_transformers.py``.
"""

import pytest

from aspectkit.backends.seq2seq import Seq2SeqBackend
from aspectkit.backends.templates import MarkersTemplate
from aspectkit.schema import ABSAExample, Span, is_implicit

TEXT = "the spicy tuna roll was unusually good !"


class ScriptedTokenizer:
    """Encodes texts as running indices; decodes indices to scripted targets."""

    def __init__(self, targets):
        self.targets = list(targets)
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append(list(texts))
        start = sum(len(batch) for batch in self.batches[:-1])
        return {"input_ids": list(range(start, start + len(texts)))}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.targets[i] for i in ids]


class ScriptedModel:
    """Echoes input ids; records every generate() call's kwargs."""

    def __init__(self):
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        return input_ids


def make_backend(targets, **kwargs):
    return Seq2SeqBackend(ScriptedModel(), tokenizer=ScriptedTokenizer(targets), **kwargs)


class TestPredict:
    def test_acos_end_to_end(self):
        backend = make_backend(
            [
                "[A] spicy tuna roll [C] food#quality [O] good [S] positive"
                " [SSEP] [A] null [C] RESTAURANT#GENERAL [O] null [S] negative"
            ],
            task="acos",
            categories=["FOOD#QUALITY", "RESTAURANT#GENERAL"],
        )
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert len(prediction) == 2
        explicit, implicit = prediction
        assert explicit.aspect == Span("spicy tuna roll", 4, 19)
        assert explicit.category == "FOOD#QUALITY"  # canonicalised casing
        assert explicit.opinion.text == "good"
        assert explicit.polarity == "positive"
        assert is_implicit(implicit.aspect) and is_implicit(implicit.opinion)
        assert implicit.polarity == "negative"
        assert backend.diagnostics == {"dropped_items": 0}

    def test_none_target_means_no_tuples(self):
        backend = make_backend(["none"], task="acos")
        assert backend.predict([ABSAExample(text=TEXT)]) == [[]]

    def test_unaligned_span_kept(self):
        backend = make_backend(["[A] tuna rolls"], task="ate")
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction[0].aspect == Span("tuna rolls")  # no offsets, still scored

    def test_malformed_fragment_dropped_with_warning(self):
        backend = make_backend(
            ["[A] roll [C] FOOD#QUALITY [S] positive [SSEP] complete garbage"],
            task="tasd",
        )
        with pytest.warns(UserWarning, match="malformed"):
            (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert len(prediction) == 1
        assert backend.diagnostics["dropped_items"] == 1

    def test_unknown_polarity_dropped(self):
        backend = make_backend(["[A] roll [O] good [S] amazing"], task="aste")
        with pytest.warns(UserWarning, match="amazing"):
            (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction == []
        assert backend.diagnostics["dropped_items"] == 1

    def test_batching_preserves_order(self):
        backend = make_backend(["[A] first", "[A] second", "[A] third"], task="ate", batch_size=2)
        predictions = backend.predict(
            [ABSAExample(text=f"the {word} item") for word in ("first", "second", "third")]
        )
        assert [p[0].aspect.text for p in predictions] == ["first", "second", "third"]
        tokenizer = backend._tokenizer
        assert [len(batch) for batch in tokenizer.batches] == [2, 1]

    def test_generation_arguments_forwarded(self):
        backend = make_backend(
            ["none"], task="acos", max_target_length=99, num_beams=4, do_sample=False
        )
        backend.predict([ABSAExample(text=TEXT)])
        (call,) = backend._model.calls
        assert call["max_new_tokens"] == 99
        assert call["num_beams"] == 4
        assert call["do_sample"] is False

    def test_paraphrase_style(self):
        backend = make_backend(
            ["FOOD#QUALITY is great because spicy tuna roll is good"],
            task="acos",
            style="paraphrase",
        )
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction[0].aspect.text == "spicy tuna roll"
        assert prediction[0].polarity == "positive"


class TestConstruction:
    def test_classification_task_rejected(self):
        with pytest.raises(ValueError, match="extraction views only"):
            Seq2SeqBackend(task="atsc")

    def test_unknown_style_rejected(self):
        with pytest.raises(ValueError, match="style"):
            Seq2SeqBackend(task="acos", style="json")

    def test_paraphrase_requires_quad_view(self):
        with pytest.raises(ValueError, match="quadruple"):
            Seq2SeqBackend(task="aste", style="paraphrase")

    def test_template_task_mismatch_rejected(self):
        with pytest.raises(ValueError, match="template"):
            Seq2SeqBackend(task="acos", template=MarkersTemplate("aste"))

    def test_custom_template_accepted(self):
        backend = Seq2SeqBackend(task="aste", template=MarkersTemplate("aste"))
        assert backend.template.task.name == "aste"

    def test_model_object_requires_tokenizer(self):
        with pytest.raises(TypeError, match="tokenizer"):
            Seq2SeqBackend(ScriptedModel())

    def test_fit_requires_examples(self):
        backend = make_backend([], task="acos")
        with pytest.raises(ValueError, match="at least one"):
            backend.fit([])

    def test_repr(self):
        backend = make_backend([], task="acos")
        text = repr(backend)
        assert "acos" in text and "MarkersTemplate" in text
