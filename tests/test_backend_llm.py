"""Tests for the prompted-LLM backend, with a scripted fake chat model."""

import json
import threading

import pytest

from aspectkit.backends.llm import LLMBackend
from aspectkit.backends.parsing import extract_json, payload_to_polarity, payload_to_tuples
from aspectkit.exceptions import LLMError, ParseError
from aspectkit.llm.base import ChatLLM
from aspectkit.schema import ABSAExample, SentimentTuple, Span, is_implicit
from aspectkit.tasks import get_task

TEXT = "The pasta was great but we waited forever."

ACOS_REPLY = json.dumps(
    {
        "tuples": [
            {
                "aspect": "pasta",
                "category": "FOOD#QUALITY",
                "opinion": "great",
                "polarity": "positive",
            },
            {
                "aspect": None,
                "category": "SERVICE#GENERAL",
                "opinion": "waited forever",
                "polarity": "negative",
            },
        ]
    }
)


class FakeChat(ChatLLM):
    """Scripted connector recording every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, *, max_tokens=1024, json_schema=None):
        self.calls.append(
            {"messages": list(messages), "max_tokens": max_tokens, "json_schema": json_schema}
        )
        action = self.replies.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fence_without_language(self):
        assert extract_json("```\n[1, 2]\n```") == [1, 2]

    def test_prose_around_json(self):
        text = 'Sure! Here is the result:\n{"tuples": []}\nLet me know if...'
        assert extract_json(text) == {"tuples": []}

    def test_no_json(self):
        with pytest.raises(ParseError, match="no JSON value"):
            extract_json("I cannot help with that.")


class TestPayloadToTuples:
    def test_bare_list_accepted(self):
        tuples, problems = payload_to_tuples(
            [{"aspect": "pasta", "polarity": "positive"}], get_task("e2e"), TEXT
        )
        assert problems == []
        assert tuples[0].aspect.text == "pasta"
        assert tuples[0].aspect.aligned

    def test_single_object_accepted(self):
        tuples, _ = payload_to_tuples(
            {"aspect": "pasta", "polarity": "positive"}, get_task("e2e"), TEXT
        )
        assert len(tuples) == 1

    def test_null_like_strings_become_implicit(self):
        for null in (None, "null", "NULL", "", "(implicit)"):
            tuples, _ = payload_to_tuples(
                [{"aspect": null, "polarity": "negative"}], get_task("e2e"), TEXT
            )
            assert is_implicit(tuples[0].aspect), repr(null)

    def test_bad_polarity_dropped_with_problem(self):
        tuples, problems = payload_to_tuples(
            [
                {"aspect": "pasta", "polarity": "amazing"},
                {"aspect": "pasta", "polarity": "positive"},
            ],
            get_task("e2e"),
            TEXT,
        )
        assert len(tuples) == 1
        assert "amazing" in problems[0]

    def test_category_canonicalised_against_inventory(self):
        tuples, _ = payload_to_tuples(
            [{"aspect": "pasta", "category": "food#quality", "polarity": "positive"}],
            get_task("tasd"),
            TEXT,
            categories=["FOOD#QUALITY"],
        )
        assert tuples[0].category == "FOOD#QUALITY"

    def test_non_dict_item_dropped(self):
        tuples, problems = payload_to_tuples(["oops"], get_task("e2e"), TEXT)
        assert tuples == [] and "not an object" in problems[0]

    def test_wrong_shape_raises(self):
        with pytest.raises(ParseError):
            payload_to_tuples("a string", get_task("e2e"), TEXT)


class TestPayloadToPolarity:
    def test_object(self):
        assert payload_to_polarity({"polarity": "Positive"}) == "positive"

    def test_bare_string(self):
        assert payload_to_polarity("negative") == "negative"

    def test_invalid(self):
        with pytest.raises(ParseError):
            payload_to_polarity({"polarity": 3.5})


class TestExtraction:
    def test_acos_end_to_end(self):
        llm = FakeChat([ACOS_REPLY])
        backend = LLMBackend(llm, task="acos", categories=["FOOD#QUALITY", "SERVICE#GENERAL"])
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert len(prediction) == 2
        explicit, implicit = prediction
        assert explicit.aspect == Span("pasta", 4, 9)
        assert explicit.category == "FOOD#QUALITY"
        assert explicit.opinion.text == "great"
        assert is_implicit(implicit.aspect)
        assert implicit.polarity == "negative"

    def test_schema_passed_to_connector(self):
        llm = FakeChat(['{"tuples": []}'])
        LLMBackend(llm, task="acos").predict([ABSAExample(text=TEXT)])
        schema = llm.calls[0]["json_schema"]
        assert schema["properties"]["tuples"]["items"]["required"] == [
            "aspect",
            "category",
            "opinion",
            "polarity",
        ]

    def test_schema_disabled(self):
        llm = FakeChat(['{"tuples": []}'])
        LLMBackend(llm, task="acos", use_schema=False).predict([ABSAExample(text=TEXT)])
        assert llm.calls[0]["json_schema"] is None

    def test_system_prompt_mentions_inventory(self):
        llm = FakeChat(['{"tuples": []}'])
        LLMBackend(llm, task="tasd", categories=["FOOD#QUALITY"]).predict([ABSAExample(text=TEXT)])
        system = llm.calls[0]["messages"][0]
        assert system["role"] == "system"
        assert "FOOD#QUALITY" in system["content"]

    def test_out_of_scheme_polarity_filtered(self):
        reply = json.dumps({"tuples": [{"aspect": "pasta", "polarity": "conflict"}]})
        llm = FakeChat([reply])
        backend = LLMBackend(llm, task="e2e")  # 3-way scheme by default
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction == []
        assert backend.diagnostics["dropped_items"] == 1

    def test_conflict_allowed_when_configured(self):
        reply = json.dumps({"tuples": [{"aspect": "pasta", "polarity": "conflict"}]})
        llm = FakeChat([reply])
        backend = LLMBackend(
            llm, task="e2e", polarities=("positive", "negative", "neutral", "conflict")
        )
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction[0].polarity == "conflict"


class TestFewShot:
    def make_train(self, n=10):
        return [
            ABSAExample(
                text=f"Example text {i}",
                tuples=[SentimentTuple(aspect=Span("thing"), polarity="positive")],
            )
            for i in range(n)
        ]

    def test_exemplars_rendered_as_turns(self):
        llm = FakeChat(['{"tuples": []}'])
        backend = LLMBackend(llm, task="e2e", n_exemplars=2)
        backend.fit(self.make_train(2))
        backend.predict([ABSAExample(text=TEXT)])
        messages = llm.calls[0]["messages"]
        # system + 2 * (user, assistant) + final user
        assert len(messages) == 6
        assert [m["role"] for m in messages[1:5]] == ["user", "assistant", "user", "assistant"]
        # assistant exemplars are valid JSON of the requested shape
        payload = json.loads(messages[2]["content"])
        assert payload["tuples"][0]["aspect"] == "thing"

    def test_exemplar_cap_and_seed_determinism(self):
        train = self.make_train(10)
        b1 = LLMBackend(FakeChat([]), task="e2e", n_exemplars=3, seed=1).fit(train)
        b2 = LLMBackend(FakeChat([]), task="e2e", n_exemplars=3, seed=1).fit(train)
        b3 = LLMBackend(FakeChat([]), task="e2e", n_exemplars=3, seed=2).fit(train)
        assert len(b1._exemplars) == 3
        assert [e.text for e in b1._exemplars] == [e.text for e in b2._exemplars]
        assert [e.text for e in b1._exemplars] != [e.text for e in b3._exemplars]

    def test_repr_reports_exemplar_count(self):
        fitted = LLMBackend(FakeChat([]), task="e2e", n_exemplars=3).fit(self.make_train(3))
        assert "exemplars=3" in repr(fitted)
        assert "exemplars=0" in repr(LLMBackend(FakeChat([]), task="e2e"))


class TestRepairAndErrors:
    def test_repair_after_garbage(self):
        llm = FakeChat(["I think the answer is...", '{"tuples": []}'])
        backend = LLMBackend(llm, task="acos")
        (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction == []
        assert backend.diagnostics["repairs"] == 1
        repair_conversation = llm.calls[1]["messages"]
        assert repair_conversation[-2]["role"] == "assistant"
        assert "ONLY" in repair_conversation[-1]["content"]

    def test_exhausted_repairs_raise(self):
        llm = FakeChat(["garbage", "more garbage"])
        backend = LLMBackend(llm, task="acos", max_repairs=1)
        with pytest.raises(ParseError):
            backend.predict([ABSAExample(text=TEXT)])

    def test_on_error_skip_records_empty(self):
        llm = FakeChat(["garbage", "more garbage"])
        backend = LLMBackend(llm, task="acos", max_repairs=1, on_error="skip")
        with pytest.warns(UserWarning, match="prediction failed"):
            (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction == []
        assert backend.diagnostics["failed_examples"] == 1

    def test_llm_error_skip(self):
        llm = FakeChat([LLMError("rate limited")])
        backend = LLMBackend(llm, task="acos", on_error="skip")
        with pytest.warns(UserWarning):
            (prediction,) = backend.predict([ABSAExample(text=TEXT)])
        assert prediction == []

    def test_llm_error_raise(self):
        llm = FakeChat([LLMError("rate limited")])
        with pytest.raises(LLMError):
            LLMBackend(llm, task="acos").predict([ABSAExample(text=TEXT)])

    def test_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            LLMBackend(FakeChat([]), on_error="ignore")

    def test_dropped_items_warn(self):
        reply = json.dumps({"tuples": [{"aspect": "pasta", "polarity": "amazing"}]})
        llm = FakeChat([reply])
        backend = LLMBackend(llm, task="e2e")
        with pytest.warns(UserWarning, match="malformed"):
            backend.predict([ABSAExample(text=TEXT)])


class TestClassification:
    def make_example(self):
        return ABSAExample(
            text="Good pasta, bad wine",
            tuples=[
                SentimentTuple(aspect=Span("pasta", 5, 10)),
                SentimentTuple(aspect=Span("wine", 16, 20)),
            ],
        )

    def test_one_call_per_target(self):
        llm = FakeChat(['{"polarity": "positive"}', '{"polarity": "negative"}'])
        backend = LLMBackend(llm, task="atsc")
        (prediction,) = backend.predict([self.make_example()])
        assert [t.polarity for t in prediction] == ["positive", "negative"]
        # given aspects preserved verbatim
        assert prediction[0].aspect == Span("pasta", 5, 10)
        assert len(llm.calls) == 2
        assert "Aspect: pasta" in llm.calls[0]["messages"][-1]["content"]

    def test_missing_targets_rejected(self):
        backend = LLMBackend(FakeChat([]), task="atsc")
        with pytest.raises(ValueError, match="given elements"):
            backend.predict([ABSAExample(text="no targets here")])

    def test_fit_uses_flat_exemplars(self):
        llm = FakeChat(['{"polarity": "positive"}'])
        backend = LLMBackend(llm, task="atsc", n_exemplars=4)
        train = [
            ABSAExample(
                text="Great screen",
                tuples=[SentimentTuple(aspect=Span("screen"), polarity="positive")],
            )
        ]
        backend.fit(train)
        example = ABSAExample(text="Bad screen", tuples=[SentimentTuple(aspect=Span("screen"))])
        backend.predict([example])
        messages = llm.calls[0]["messages"]
        assert "Great screen" in messages[1]["content"]
        assert json.loads(messages[2]["content"]) == {"polarity": "positive"}

    def test_out_of_scheme_polarity_skipped(self):
        llm = FakeChat(['{"polarity": "conflict"}'])
        backend = LLMBackend(llm, task="atsc", on_error="skip")
        with pytest.warns(UserWarning):
            (prediction,) = backend.predict(
                [ABSAExample(text="x", tuples=[SentimentTuple(aspect=Span("a"))])]
            )
        assert prediction == []


class KeyedFakeChat(ChatLLM):
    """Thread-safe fake that selects the reply by prompt content.

    Order-based scripting breaks under concurrency (call order is
    nondeterministic), so replies are keyed by a substring of the final
    user message instead.
    """

    def __init__(self, replies_by_needle):
        self.replies = dict(replies_by_needle)
        self.n_calls = 0
        self._lock = threading.Lock()

    def complete(self, messages, *, max_tokens=1024, json_schema=None):
        with self._lock:
            self.n_calls += 1
        content = messages[-1]["content"]
        for needle, reply in self.replies.items():
            if needle in content:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"no scripted reply matches {content!r}")


class TestConcurrency:
    def make_corpus(self, n=8):
        examples = [ABSAExample(text=f"we liked item{i} a lot") for i in range(n)]
        replies = {
            f"item{i}": json.dumps({"tuples": [{"aspect": f"item{i}", "polarity": "positive"}]})
            for i in range(n)
        }
        return examples, replies

    def test_matches_sequential_and_preserves_order(self):
        examples, replies = self.make_corpus()
        sequential = LLMBackend(KeyedFakeChat(replies), task="e2e").predict(examples)
        concurrent = LLMBackend(KeyedFakeChat(replies), task="e2e", concurrency=4).predict(examples)
        assert concurrent == sequential
        assert [p[0].aspect.text for p in concurrent] == [f"item{i}" for i in range(8)]

    def test_diagnostics_are_thread_safe(self):
        examples, _ = self.make_corpus()
        # every reply is garbage (the empty needle matches any prompt,
        # including repair turns): each example burns 2 calls (1 repair)
        # and then fails; the counters must not lose updates.
        llm = KeyedFakeChat({"": "not json"})
        backend = LLMBackend(llm, task="e2e", concurrency=8, max_repairs=1, on_error="skip")
        with pytest.warns(UserWarning, match="prediction failed"):
            predictions = backend.predict(examples)
        assert predictions == [[]] * 8
        assert backend.diagnostics["calls"] == 16
        assert backend.diagnostics["repairs"] == 8
        assert backend.diagnostics["failed_examples"] == 8
        assert llm.n_calls == 16

    def test_on_error_raise_propagates(self):
        examples, replies = self.make_corpus(4)
        replies["item2"] = LLMError("rate limited")
        backend = LLMBackend(KeyedFakeChat(replies), task="e2e", concurrency=4)
        with pytest.raises(LLMError, match="rate limited"):
            backend.predict(examples)

    def test_classification_concurrent(self):
        examples = [
            ABSAExample(text="good pasta", tuples=[SentimentTuple(aspect=Span("pasta"))]),
            ABSAExample(text="bad wine", tuples=[SentimentTuple(aspect=Span("wine"))]),
        ]
        llm = KeyedFakeChat(
            {
                "Aspect: pasta": '{"polarity": "positive"}',
                "Aspect: wine": '{"polarity": "negative"}',
            }
        )
        backend = LLMBackend(llm, task="atsc", concurrency=2)
        predictions = backend.predict(examples)
        assert [p[0].polarity for p in predictions] == ["positive", "negative"]

    def test_invalid_concurrency(self):
        with pytest.raises(ValueError, match="concurrency"):
            LLMBackend(FakeChat([]), concurrency=0)


class TestIntrospection:
    def test_describe_prompt_extraction(self):
        text = LLMBackend.describe_prompt("acos", categories=["FOOD#QUALITY"])
        assert "FOOD#QUALITY" in text
        assert "JSON schema" in text

    def test_describe_prompt_classification(self):
        text = LLMBackend.describe_prompt("atsc")
        assert "polarity" in text

    def test_repr(self):
        backend = LLMBackend(FakeChat([]), task="acos")
        assert "acos" in repr(backend)
