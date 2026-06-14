"""Tests for the pair-classifier backend, with duck-typed stubs."""

from types import SimpleNamespace

import pytest

from aspectkit.backends.pair import PairClassifierBackend
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span


class StubLogits:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class StubTokenizer:
    """Records batches; encodes nothing."""

    def __init__(self):
        self.batches = []

    def __call__(self, texts, aspects, **kwargs):
        self.batches.append(list(zip(texts, aspects, strict=True)))
        return {"batch": list(zip(texts, aspects, strict=True))}


class StubModel:
    """Maps aspect surface text to scripted logits."""

    name_or_path = "stub/absa"

    def __init__(self, logits_by_aspect):
        self.config = SimpleNamespace(id2label={0: "Negative", 1: "Neutral", 2: "Positive"})
        self._logits_by_aspect = logits_by_aspect

    def __call__(self, batch):
        rows = [self._logits_by_aspect[aspect] for _, aspect in batch]
        return SimpleNamespace(logits=StubLogits(rows))


def make_backend(logits_by_aspect, **kwargs):
    return PairClassifierBackend(StubModel(logits_by_aspect), tokenizer=StubTokenizer(), **kwargs)


class TestPredict:
    def test_polarity_assigned_per_target(self):
        backend = make_backend({"pasta": [0.1, 0.2, 0.9], "wine": [0.9, 0.1, 0.1]})
        example = ABSAExample(
            text="Good pasta, bad wine",
            tuples=[
                SentimentTuple(aspect=Span("pasta", 5, 10), category="FOOD#QUALITY"),
                SentimentTuple(aspect=Span("wine", 16, 20)),
            ],
        )
        (prediction,) = backend.predict([example])
        assert [t.polarity for t in prediction] == ["positive", "negative"]
        # given elements preserved
        assert prediction[0].aspect == Span("pasta", 5, 10)
        assert prediction[0].category == "FOOD#QUALITY"

    def test_batching(self):
        backend = make_backend({f"a{i}": [1.0, 0.0, 0.0] for i in range(5)}, batch_size=2)
        example = ABSAExample(
            text="x",
            tuples=[SentimentTuple(aspect=Span(f"a{i}")) for i in range(5)],
        )
        backend.predict([example])
        assert [len(b) for b in backend._tokenizer.batches] == [2, 2, 1]

    def test_implicit_aspect_warns_and_uses_empty_target(self):
        backend = make_backend({"": [0.0, 1.0, 0.0]})
        example = ABSAExample(text="Would not recommend", tuples=[SentimentTuple(aspect=IMPLICIT)])
        with pytest.warns(UserWarning, match="implicit"):
            (prediction,) = backend.predict([example])
        assert prediction[0].polarity == "neutral"

    def test_targets_required(self):
        backend = make_backend({})
        with pytest.raises(ValueError, match="given elements"):
            backend.predict([ABSAExample(text="no targets")])

    def test_multiple_examples_keep_alignment(self):
        backend = make_backend({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]})
        examples = [
            ABSAExample(text="x", tuples=[SentimentTuple(aspect=Span("a"))]),
            ABSAExample(text="y", tuples=[SentimentTuple(aspect=Span("b"))]),
        ]
        predictions = backend.predict(examples)
        assert predictions[0][0].polarity == "negative"
        assert predictions[1][0].polarity == "positive"

    def test_diagnostics_populated(self):
        backend = make_backend({"pasta": [0.1, 0.2, 0.9], "": [0.5, 0.3, 0.2]})
        example = ABSAExample(
            text="Good pasta",
            tuples=[SentimentTuple(aspect=Span("pasta", 5, 10)), SentimentTuple(aspect=IMPLICIT)],
        )
        with pytest.warns(UserWarning):
            backend.predict([example])
        assert backend.diagnostics == {"n_implicit": 1, "pairs": 2}


class TestLifecycle:
    def test_fit_rejects_unlabelled_examples(self):
        backend = make_backend({})
        with pytest.raises(ValueError, match="at least one"):
            backend.fit([ABSAExample(text="no labelled tuples here")])

    def test_model_object_without_tokenizer_rejected(self):
        with pytest.raises(TypeError, match="tokenizer"):
            PairClassifierBackend(StubModel({}))

    def test_hub_id_is_lazy(self):
        backend = PairClassifierBackend("some-org/some-checkpoint")
        assert backend.model_name == "some-org/some-checkpoint"

    def test_task_is_atsc(self):
        assert make_backend({}).task.name == "atsc"

    def test_repr(self):
        assert "stub/absa" in repr(make_backend({}))
