"""Tests for the ABSA facade."""

import json

import pytest

from aspectkit import ABSA, ABSAExample, SentimentTuple, Span
from aspectkit.backends.base import Backend
from aspectkit.backends.llm import LLMBackend
from aspectkit.backends.seq2seq import Seq2SeqBackend
from aspectkit.tasks import get_task
from tests.test_backend_llm import ACOS_REPLY, TEXT, FakeChat
from tests.test_backend_seq2seq import ScriptedModel, ScriptedTokenizer


class RecordingBackend(Backend):
    """Echoes scripted predictions and records what it was shown."""

    def __init__(self, task_name, predictions):
        self.task = get_task(task_name)
        self.predictions = predictions
        self.seen = None

    def predict(self, examples):
        self.seen = list(examples)
        return self.predictions[: len(examples)]


class TestConstruction:
    def test_llm_backend_requires_model(self):
        with pytest.raises(ValueError, match="needs a model"):
            ABSA(task="acos", backend="llm")

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown backend"):
            ABSA(task="acos", backend="crf", model="t5-base")

    def test_pair_backend_rejects_extraction_tasks(self):
        with pytest.raises(ValueError, match="ATSC only"):
            ABSA(task="aste", backend="pair")

    def test_seq2seq_backend_constructed(self):
        absa = ABSA(
            task="aste",
            backend="seq2seq",
            model=ScriptedModel(),
            tokenizer=ScriptedTokenizer(["[A] pasta [O] great [S] positive"]),
        )
        assert isinstance(absa.backend, Seq2SeqBackend)
        assert absa.task.name == "aste"
        prediction = absa.predict("the pasta was great")
        assert prediction[0].aspect.text == "pasta"

    def test_seq2seq_backend_defaults_to_acos(self):
        absa = ABSA(backend="seq2seq", model="t5-base")
        assert absa.task.name == "acos"

    def test_seq2seq_backend_rejects_classification_tasks(self):
        with pytest.raises(ValueError, match="extraction views only"):
            ABSA(task="atsc", backend="seq2seq", model="t5-base")

    def test_pair_backend_defaults_to_atsc(self):
        absa = ABSA(backend="pair", model="some-org/checkpoint")
        assert absa.task.name == "atsc"

    def test_backend_instance(self):
        backend = RecordingBackend("aste", [])
        absa = ABSA(backend=backend)
        assert absa.task.name == "aste"

    def test_backend_instance_with_conflicting_task(self):
        with pytest.raises(ValueError, match="contradicts"):
            ABSA(task="acos", backend=RecordingBackend("aste", []))

    def test_backend_instance_with_matching_task_ok(self):
        absa = ABSA(task="aste", backend=RecordingBackend("aste", []))
        assert absa.task.name == "aste"

    def test_backend_instance_rejects_model(self):
        with pytest.raises(ValueError, match="model"):
            ABSA(backend=RecordingBackend("aste", []), model="x")

    def test_default_task_is_acos(self):
        absa = ABSA(backend="llm", model=FakeChat([]))
        assert absa.task.name == "acos"

    def test_repr(self):
        absa = ABSA(backend="llm", model=FakeChat([]))
        assert "acos" in repr(absa)


class TestPredict:
    def test_single_text_returns_flat_list(self):
        absa = ABSA(task="acos", backend="llm", model=FakeChat([ACOS_REPLY]))
        prediction = absa.predict(TEXT)
        assert isinstance(prediction, list)
        assert all(isinstance(t, SentimentTuple) for t in prediction)
        assert len(prediction) == 2

    def test_list_of_texts_returns_nested_lists(self):
        absa = ABSA(
            task="acos",
            backend="llm",
            model=FakeChat([ACOS_REPLY, json.dumps({"tuples": []})]),
        )
        predictions = absa.predict([TEXT, "Nothing to see."])
        assert len(predictions) == 2
        assert predictions[1] == []

    def test_examples_accepted(self):
        absa = ABSA(task="acos", backend="llm", model=FakeChat([ACOS_REPLY]))
        prediction = absa.predict(ABSAExample(text=TEXT))
        assert len(prediction) == 2

    def test_fit_returns_self(self):
        absa = ABSA(task="acos", backend="llm", model=FakeChat([]))
        assert absa.fit([]) is absa


class TestEvaluate:
    def test_extraction_gold_is_hidden_from_backend(self):
        gold = [
            ABSAExample(
                text=TEXT,
                tuples=[SentimentTuple(aspect=Span("pasta"), polarity="positive")],
            )
        ]
        backend = RecordingBackend(
            "e2e", [[SentimentTuple(aspect=Span("pasta"), polarity="positive")]]
        )
        report = ABSA(backend=backend).evaluate(gold)
        assert backend.seen[0].tuples == []  # no gold leakage
        assert report.exact.f1 == 1.0

    def test_classification_labels_are_stripped(self):
        gold = [
            ABSAExample(
                text="Good pasta",
                tuples=[
                    SentimentTuple(aspect=Span("pasta"), polarity="positive", opinion=Span("Good"))
                ],
            )
        ]
        backend = RecordingBackend(
            "atsc", [[SentimentTuple(aspect=Span("pasta"), polarity="positive")]]
        )
        report = ABSA(backend=backend).evaluate(gold)
        shown = backend.seen[0].tuples[0]
        assert shown.aspect == Span("pasta")  # given element kept
        assert shown.polarity is None  # label stripped
        assert shown.opinion is None  # non-given elements stripped
        assert report.classification.accuracy == 1.0

    def test_precomputed_predictions(self):
        gold = [
            ABSAExample(
                text=TEXT,
                tuples=[SentimentTuple(aspect=Span("pasta"), polarity="positive")],
            )
        ]
        absa = ABSA(backend=RecordingBackend("e2e", [[]]))
        report = absa.evaluate(
            gold, predictions=[[SentimentTuple(aspect=Span("pasta"), polarity="positive")]]
        )
        assert report.exact.f1 == 1.0
        assert absa.backend.seen is None  # backend never invoked

    def test_lenient_flag_propagates(self):
        gold = [ABSAExample(text="x", tuples=[])]
        report = ABSA(backend=RecordingBackend("e2e", [[]])).evaluate(gold, lenient=True)
        assert report.lenient is not None


class TestSummarize:
    def test_end_to_end(self):
        absa = ABSA(task="acos", backend="llm", model=FakeChat([ACOS_REPLY]))
        summaries = absa.summarize([TEXT])
        keys = [s.key for s in summaries]
        assert "pasta" in keys and "(implicit)" in keys

    def test_precomputed_predictions(self):
        absa = ABSA(backend=RecordingBackend("e2e", [[]]))
        summaries = absa.summarize(
            ["Great wifi"],
            predictions=[[SentimentTuple(aspect=Span("wifi"), polarity="positive")]],
        )
        assert summaries[0].key == "wifi"
        assert absa.backend.seen is None

    def test_by_category(self):
        absa = ABSA(
            task="acos",
            backend="llm",
            model=FakeChat([ACOS_REPLY]),
            categories=["FOOD#QUALITY", "SERVICE#GENERAL"],
        )
        summaries = absa.summarize([TEXT], by="category")
        assert {s.key for s in summaries} == {"FOOD#QUALITY", "SERVICE#GENERAL"}


class TestLLMBackendIntegrationWithFacade:
    def test_transformers_style_object_accepted_as_model(self):
        from tests.test_llm_local import FakePipeline

        pipe = FakePipeline(reply='{"tuples": []}')
        absa = ABSA(task="acos", backend="llm", model=pipe)
        assert isinstance(absa.backend, LLMBackend)
        assert absa.predict(TEXT) == []
