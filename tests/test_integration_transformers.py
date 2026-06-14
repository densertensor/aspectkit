"""Integration tests against real torch/transformers execution paths.

No network access is assumed: instead of hub checkpoints, the tests
build tiny randomly-initialised models in-process.  Outputs are
meaningless, but every real seam is exercised — chat templating /
transcript rendering, tensor round-trips, ``generate`` slicing, pair
encoding, logits decoding — which is exactly what stubs cannot cover.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from aspectkit.backends.llm import LLMBackend  # noqa: E402
from aspectkit.backends.pair import PairClassifierBackend  # noqa: E402
from aspectkit.backends.seq2seq import Seq2SeqBackend  # noqa: E402
from aspectkit.llm.local import TransformersChat  # noqa: E402
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span  # noqa: E402

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "system",
    "user",
    "assistant",
    ":",
    ".",
    ",",
    "!",
    "?",
    "the",
    "pasta",
    "was",
    "great",
    "but",
    "we",
    "waited",
    "forever",
    "good",
    "bad",
    "wine",
    "service",
    "be",
    "terse",
    "hi",
    "text",
]


@pytest.fixture(scope="module")
def causal_lm():
    """A tiny GPT-2 with a programmatic word-level tokenizer (no downloads)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.normalizers import Lowercase
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")

    torch.manual_seed(7)
    # n_positions must accommodate the full ACOS system prompt;
    # bos/eos must lie inside the tiny vocabulary (defaults are GPT-2's 50256).
    config = GPT2Config(
        vocab_size=len(vocab),
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=1024,
        bos_token_id=9,
        eos_token_id=9,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tokenizer


def _build_pair_classifier(vocab_dir):
    """A tiny DistilBERT sequence-pair classifier (no downloads)."""
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )

    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    tokenizer = DistilBertTokenizer(str(vocab_file))

    torch.manual_seed(11)
    config = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=8,
        n_layers=1,
        n_heads=2,
        hidden_dim=16,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    model = DistilBertForSequenceClassification(config)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def pair_classifier(tmp_path_factory):
    """Shared read-only instance; tests that train build their own."""
    return _build_pair_classifier(tmp_path_factory.mktemp("vocab"))


T5_VOCAB = [
    "<pad>",
    "</s>",
    "<unk>",
    "[",
    "]",
    "a",
    "c",
    "o",
    "s",
    "ssep",
    "null",
    "none",
    "is",
    "because",
    "#",
    "food",
    "quality",
    "positive",
    "negative",
    "neutral",
    "great",
    "ok",
    "bad",
    "the",
    "pasta",
    "was",
    "but",
    "we",
    "waited",
    "forever",
    "wine",
    "service",
    "good",
    ",",
    ".",
    "!",
]


def _build_seq2seq():
    """A tiny T5 with a programmatic word-level tokenizer (no downloads)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.normalizers import Lowercase
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    vocab = {token: i for i, token in enumerate(T5_VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    # T5 convention: every sequence ends with </s>
    tok.post_processor = TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", vocab["</s>"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"
    )

    torch.manual_seed(13)
    config = T5Config(
        vocab_size=len(vocab),
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_heads=2,
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["</s>"],
        decoder_start_token_id=vocab["<pad>"],
    )
    model = T5ForConditionalGeneration(config)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def seq2seq_lm():
    """Shared read-only instance; tests that train build their own."""
    return _build_seq2seq()


class TestTransformersChatRealPath:
    def test_model_tokenizer_pair_generates_only_new_tokens(self, causal_lm):
        model, tokenizer = causal_lm
        llm = TransformersChat(model, tokenizer)
        reply = llm.complete(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "the pasta was great"},
            ],
            max_tokens=4,
        )
        assert isinstance(reply, str) and reply.strip()
        # the transcript fallback was used (tokenizer has no chat template)
        # and the prompt was sliced off: the reply is at most 4 tokens long
        assert len(reply.split()) <= 4

    def test_real_pipeline_object(self, causal_lm):
        from transformers import pipeline

        model, tokenizer = causal_lm
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        llm = TransformersChat(pipe)
        reply = llm.complete([{"role": "user", "content": "hi"}], max_tokens=3)
        assert isinstance(reply, str) and reply.strip()

    def test_llm_backend_survives_non_json_model(self, causal_lm):
        """A model that cannot emit JSON exercises repair + skip handling."""
        model, tokenizer = causal_lm
        backend = LLMBackend(
            TransformersChat(model, tokenizer),
            task="acos",
            max_repairs=1,
            on_error="skip",
            max_tokens=8,
        )
        with pytest.warns(UserWarning, match="prediction failed"):
            predictions = backend.predict([ABSAExample(text="the pasta was great")])
        assert predictions == [[]]
        assert backend.diagnostics["failed_examples"] == 1


class TestPairClassifierRealPath:
    def test_predicts_valid_polarities(self, pair_classifier):
        model, tokenizer = pair_classifier
        backend = PairClassifierBackend(model, tokenizer=tokenizer, batch_size=2)
        examples = [
            ABSAExample(
                text="good pasta , bad wine",
                tuples=[
                    SentimentTuple(aspect=Span("pasta"), category="FOOD#QUALITY"),
                    SentimentTuple(aspect=Span("wine")),
                ],
            ),
            ABSAExample(
                text="the service was great",
                tuples=[SentimentTuple(aspect=Span("service"))],
            ),
        ]
        predictions = backend.predict(examples)
        assert [len(p) for p in predictions] == [2, 1]
        for prediction in predictions:
            for t in prediction:
                assert t.polarity in ("positive", "negative", "neutral")
        # given elements survive the round-trip
        assert predictions[0][0].aspect == Span("pasta")
        assert predictions[0][0].category == "FOOD#QUALITY"

    def test_deterministic_across_calls(self, pair_classifier):
        model, tokenizer = pair_classifier
        backend = PairClassifierBackend(model, tokenizer=tokenizer)
        example = ABSAExample(text="good pasta", tuples=[SentimentTuple(aspect=Span("pasta"))])
        first = backend.predict([example])
        second = backend.predict([example])
        assert first == second


PAIR_TRAIN = [
    ABSAExample(
        text="good pasta , bad wine",
        tuples=[
            SentimentTuple(aspect=Span("pasta"), polarity="positive"),
            SentimentTuple(aspect=Span("wine"), polarity="negative"),
        ],
    ),
    ABSAExample(
        text="the service was great",
        tuples=[SentimentTuple(aspect=Span("service"), polarity="positive")],
    ),
]


class TestPairClassifierFit:
    def test_fit_reduces_loss_and_restores_eval_mode(self, tmp_path):
        model, tokenizer = _build_pair_classifier(tmp_path)
        backend = PairClassifierBackend(model, tokenizer=tokenizer, batch_size=4)
        backend.fit(PAIR_TRAIN, epochs=10, learning_rate=1e-3, seed=0)
        assert len(backend.history_) == 10
        assert backend.history_[-1] < backend.history_[0]
        assert model.training is False
        # the fine-tuned model still produces valid predictions
        (prediction,) = backend.predict(
            [ABSAExample(text="good wine", tuples=[SentimentTuple(aspect=Span("wine"))])]
        )
        assert prediction[0].polarity in ("positive", "negative", "neutral")

    def test_unmappable_label_rejected(self, tmp_path):
        model, tokenizer = _build_pair_classifier(tmp_path)
        backend = PairClassifierBackend(model, tokenizer=tokenizer)
        conflicted = [
            ABSAExample(
                text="good pasta",
                tuples=[SentimentTuple(aspect=Span("pasta"), polarity="conflict")],
            )
        ]
        with pytest.raises(ValueError, match="id2label"):
            backend.fit(conflicted)

    def test_no_labelled_pairs_rejected(self, pair_classifier):
        model, tokenizer = pair_classifier
        backend = PairClassifierBackend(model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="at least one"):
            backend.fit([ABSAExample(text="no labels here")])

    def test_implicit_aspects_warn(self, tmp_path):
        model, tokenizer = _build_pair_classifier(tmp_path)
        backend = PairClassifierBackend(model, tokenizer=tokenizer)
        train = [
            ABSAExample(
                text="bad service",
                tuples=[SentimentTuple(aspect=IMPLICIT, polarity="negative")],
            )
        ]
        with pytest.warns(UserWarning, match="implicit"):
            backend.fit(train, epochs=1)

    def test_save_and_reload(self, tmp_path):
        model, tokenizer = _build_pair_classifier(tmp_path / "build")
        backend = PairClassifierBackend(model, tokenizer=tokenizer)
        backend.save_pretrained(tmp_path / "saved")
        reloaded = PairClassifierBackend(str(tmp_path / "saved"))
        example = ABSAExample(text="good pasta", tuples=[SentimentTuple(aspect=Span("pasta"))])
        assert reloaded.predict([example]) == backend.predict([example])


SEQ2SEQ_TRAIN = [
    ABSAExample(
        text="the pasta was good",
        tuples=[
            SentimentTuple(
                aspect=Span("pasta", 4, 9),
                category="FOOD#QUALITY",
                opinion=Span("good", 14, 18),
                polarity="positive",
            )
        ],
    ),
    ABSAExample(
        text="bad wine !",
        tuples=[
            SentimentTuple(
                aspect=Span("wine", 4, 8),
                category="FOOD#QUALITY",
                opinion=Span("bad", 0, 3),
                polarity="negative",
            )
        ],
    ),
]


class TestSeq2SeqRealPath:
    def test_predict_runs_end_to_end(self, seq2seq_lm):
        model, tokenizer = seq2seq_lm
        backend = Seq2SeqBackend(model, tokenizer=tokenizer, task="acos", max_target_length=24)
        with warnings.catch_warnings():
            # a randomly-initialised model generates junk; drops are expected
            warnings.simplefilter("ignore")
            predictions = backend.predict(
                [ABSAExample(text="the pasta was good"), ABSAExample(text="bad wine !")]
            )
        assert len(predictions) == 2
        for prediction in predictions:
            assert all(isinstance(t, SentimentTuple) for t in prediction)
        assert "dropped_items" in backend.diagnostics

    def test_fit_reduces_loss_and_restores_eval_mode(self):
        model, tokenizer = _build_seq2seq()
        backend = Seq2SeqBackend(model, tokenizer=tokenizer, task="acos", batch_size=2)
        backend.fit(SEQ2SEQ_TRAIN, epochs=10, learning_rate=5e-3, seed=0)
        assert len(backend.history_) == 10
        assert backend.history_[-1] < backend.history_[0]
        assert model.training is False

    def test_fit_with_paraphrase_template(self):
        model, tokenizer = _build_seq2seq()
        backend = Seq2SeqBackend(
            model, tokenizer=tokenizer, task="acos", style="paraphrase", batch_size=2
        )
        backend.fit(SEQ2SEQ_TRAIN, epochs=2, learning_rate=5e-3, seed=0)
        assert len(backend.history_) == 2

    def test_save_and_reload(self, tmp_path):
        model, tokenizer = _build_seq2seq()
        backend = Seq2SeqBackend(model, tokenizer=tokenizer, task="acos")
        backend.save_pretrained(tmp_path / "saved")
        reloaded = Seq2SeqBackend(str(tmp_path / "saved"), task="acos", max_target_length=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = reloaded.predict([ABSAExample(text="the pasta was good")])
        assert len(predictions) == 1
