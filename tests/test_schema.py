"""Tests for the canonical data model."""

import copy
import pickle

import pytest

from aspectkit.schema import (
    IMPLICIT,
    ABSAExample,
    Implicit,
    SentimentTuple,
    Span,
    is_implicit,
)


class TestSpan:
    def test_aligned_span(self):
        span = Span(text="pasta", start=4, end=9)
        assert span.aligned
        assert span.text == "pasta"

    def test_unaligned_span(self):
        span = Span(text="pasta")
        assert not span.aligned

    def test_offsets_must_come_together(self):
        with pytest.raises(ValueError):
            Span(text="x", start=1)
        with pytest.raises(ValueError):
            Span(text="x", end=3)

    def test_invalid_offsets(self):
        with pytest.raises(ValueError):
            Span(text="x", start=5, end=5)
        with pytest.raises(ValueError):
            Span(text="x", start=-1, end=3)

    def test_dict_roundtrip(self):
        span = Span(text="pasta", start=4, end=9)
        assert Span.from_dict(span.to_dict()) == span
        unaligned = Span(text="pasta")
        assert Span.from_dict(unaligned.to_dict()) == unaligned

    def test_frozen(self):
        with pytest.raises(AttributeError):
            Span(text="x").text = "y"


class TestImplicit:
    def test_singleton(self):
        assert Implicit() is IMPLICIT
        assert copy.copy(IMPLICIT) is IMPLICIT
        assert copy.deepcopy(IMPLICIT) is IMPLICIT
        assert pickle.loads(pickle.dumps(IMPLICIT)) is IMPLICIT

    def test_repr(self):
        assert repr(IMPLICIT) == "IMPLICIT"

    def test_is_implicit(self):
        assert is_implicit(IMPLICIT)
        assert not is_implicit(None)
        assert not is_implicit(Span(text="x"))


class TestSentimentTuple:
    def test_basic(self):
        t = SentimentTuple(
            aspect=Span("pasta", 4, 9),
            category="FOOD#QUALITY",
            opinion=Span("great", 14, 19),
            polarity="positive",
        )
        assert t.aspect_text == "pasta"
        assert t.opinion_text == "great"

    def test_implicit_accessors(self):
        t = SentimentTuple(aspect=IMPLICIT, opinion=IMPLICIT, polarity="negative")
        assert t.aspect_text is None
        assert t.opinion_text is None

    def test_polarity_optional(self):
        t = SentimentTuple(aspect=Span("battery"))
        assert t.polarity is None

    def test_invalid_polarity(self):
        with pytest.raises(ValueError, match="polarity"):
            SentimentTuple(aspect=Span("x"), polarity="great")

    def test_aspect_none_rejected(self):
        with pytest.raises(ValueError, match="aspect"):
            SentimentTuple(aspect=None, polarity="positive")

    @pytest.mark.parametrize(
        "t",
        [
            SentimentTuple(aspect=Span("pasta", 4, 9), polarity="positive"),
            SentimentTuple(aspect=IMPLICIT, category="SERVICE#GENERAL", polarity="negative"),
            SentimentTuple(aspect=Span("staff"), opinion=IMPLICIT, polarity="neutral"),
            SentimentTuple(aspect=Span("ui")),
        ],
    )
    def test_dict_roundtrip(self, t):
        assert SentimentTuple.from_dict(t.to_dict()) == t

    def test_serialised_implicit_is_distinct_from_absent(self):
        implicit = SentimentTuple(aspect=IMPLICIT, opinion=IMPLICIT, polarity="positive")
        absent = SentimentTuple(aspect=IMPLICIT, opinion=None, polarity="positive")
        assert implicit.to_dict()["opinion"] == "IMPLICIT"
        assert absent.to_dict()["opinion"] is None
        assert SentimentTuple.from_dict(implicit.to_dict()) != SentimentTuple.from_dict(
            absent.to_dict()
        )


class TestABSAExample:
    def test_roundtrip(self):
        example = ABSAExample(
            text="The pasta was great",
            tuples=[SentimentTuple(aspect=Span("pasta", 4, 9), polarity="positive")],
            id="r1:s1",
            meta={"domain": "restaurants"},
        )
        assert ABSAExample.from_dict(example.to_dict()) == example

    def test_empty_tuples_roundtrip(self):
        example = ABSAExample(text="We arrived at nine.")
        restored = ABSAExample.from_dict(example.to_dict())
        assert restored.tuples == []
        assert restored.id is None


class TestFrozenExample:
    def test_attribute_reassignment_blocked(self):
        from dataclasses import FrozenInstanceError

        ex = ABSAExample(text="x")
        with pytest.raises(FrozenInstanceError):
            ex.text = "y"

    def test_in_place_tuple_mutation_still_allowed(self):
        # Freezing blocks attribute reassignment, not in-place list edits.
        ex = ABSAExample(text="x")
        ex.tuples.append(SentimentTuple(aspect=IMPLICIT, polarity="positive"))
        assert len(ex.tuples) == 1
