"""Tests for the seq2seq target-sequence templates."""

import pytest

from aspectkit.backends.templates import MarkersTemplate, ParaphraseTemplate
from aspectkit.schema import IMPLICIT, SentimentTuple, Span

QUAD_EXPLICIT = SentimentTuple(
    aspect=Span("spicy tuna roll", 4, 19),
    category="FOOD#QUALITY",
    opinion=Span("good", 24, 28),
    polarity="positive",
)
QUAD_IMPLICIT = SentimentTuple(
    aspect=IMPLICIT,
    category="RESTAURANT#GENERAL",
    opinion=IMPLICIT,
    polarity="negative",
)


class TestMarkersTemplate:
    def test_encode_quadruples(self):
        template = MarkersTemplate("acos")
        target = template.encode([QUAD_EXPLICIT, QUAD_IMPLICIT])
        assert target == (
            "[A] spicy tuna roll [C] FOOD#QUALITY [O] good [S] positive"
            " [SSEP] "
            "[A] null [C] RESTAURANT#GENERAL [O] null [S] negative"
        )

    def test_roundtrip(self):
        template = MarkersTemplate("acos")
        fragments, problems = template.decode(template.encode([QUAD_EXPLICIT, QUAD_IMPLICIT]))
        assert problems == []
        assert fragments == [
            {
                "aspect": "spicy tuna roll",
                "category": "FOOD#QUALITY",
                "opinion": "good",
                "polarity": "positive",
            },
            {
                "aspect": "null",
                "category": "RESTAURANT#GENERAL",
                "opinion": "null",
                "polarity": "negative",
            },
        ]

    def test_view_restricts_markers(self):
        template = MarkersTemplate("aste")
        target = template.encode([QUAD_EXPLICIT])
        assert target == "[A] spicy tuna roll [O] good [S] positive"
        assert "[C]" not in target

    def test_single_element_view(self):
        template = MarkersTemplate("ate")
        assert template.encode([QUAD_EXPLICIT]) == "[A] spicy tuna roll"
        fragments, problems = template.decode("[A] battery life")
        assert fragments == [{"aspect": "battery life"}] and problems == []

    def test_empty_tuples_roundtrip(self):
        template = MarkersTemplate("acos")
        assert template.encode([]) == "none"
        assert template.decode("none") == ([], [])
        assert template.decode("  NONE ") == ([], [])
        assert template.decode("") == ([], [])

    def test_missing_marker_dropped_with_problem(self):
        template = MarkersTemplate("acos")
        fragments, problems = template.decode(
            "[A] pasta [C] FOOD#QUALITY [S] positive"  # no [O]
            " [SSEP] [A] wine [C] DRINKS#QUALITY [O] flat [S] negative"
        )
        assert len(fragments) == 1
        assert fragments[0]["aspect"] == "wine"
        assert len(problems) == 1 and "missing opinion" in problems[0]

    def test_duplicate_marker_dropped(self):
        template = MarkersTemplate("ate")
        fragments, problems = template.decode("[A] pasta [A] wine")
        assert fragments == []
        assert "duplicate" in problems[0]

    def test_stray_prefix_tolerated(self):
        template = MarkersTemplate("ate")
        fragments, problems = template.decode("sure thing [A] pasta")
        assert fragments == [{"aspect": "pasta"}] and problems == []

    def test_marker_outside_view_ignored(self):
        template = MarkersTemplate("aste")
        fragments, problems = template.decode("[A] pasta [C] FOOD [O] great [S] positive")
        assert problems == []
        assert fragments == [{"aspect": "pasta", "opinion": "great", "polarity": "positive"}]

    def test_encode_rejects_missing_categorical(self):
        template = MarkersTemplate("acos")
        unlabelled = SentimentTuple(aspect=Span("pasta"), opinion=Span("great"))
        with pytest.raises(ValueError, match="polarity"):
            template.encode([unlabelled])

    def test_classification_view_rejected(self):
        with pytest.raises(ValueError, match="extraction"):
            MarkersTemplate("atsc")

    def test_repr(self):
        assert "acos" in repr(MarkersTemplate("acos"))


class TestParaphraseTemplate:
    def test_encode(self):
        template = ParaphraseTemplate()
        target = template.encode([QUAD_EXPLICIT, QUAD_IMPLICIT])
        assert target == (
            "FOOD#QUALITY is great because spicy tuna roll is good"
            " [SSEP] "
            "RESTAURANT#GENERAL is bad because it is it"
        )

    def test_roundtrip(self):
        template = ParaphraseTemplate()
        fragments, problems = template.decode(template.encode([QUAD_EXPLICIT, QUAD_IMPLICIT]))
        assert problems == []
        assert fragments[0] == {
            "aspect": "spicy tuna roll",
            "category": "FOOD#QUALITY",
            "opinion": "good",
            "polarity": "positive",
        }
        # "it" comes back as the shared null convention
        assert fragments[1]["aspect"] == "null"
        assert fragments[1]["opinion"] == "null"
        assert fragments[1]["polarity"] == "negative"

    def test_sentiment_words(self):
        template = ParaphraseTemplate()
        for polarity, word in (("positive", "great"), ("neutral", "ok"), ("negative", "bad")):
            t = SentimentTuple(
                aspect=Span("a"), category="C#G", opinion=Span("o"), polarity=polarity
            )
            assert f" is {word} because " in template.encode([t])

    def test_empty_tuples_roundtrip(self):
        template = ParaphraseTemplate()
        assert template.encode([]) == "none"
        assert template.decode("none") == ([], [])

    def test_malformed_fragments_dropped(self):
        template = ParaphraseTemplate()
        fragments, problems = template.decode(
            "no connective here"
            " [SSEP] FOOD#QUALITY is great because pasta is fresh"
            " [SSEP] FOOD#QUALITY is amazing because pasta is fresh"
        )
        assert len(fragments) == 1
        assert fragments[0]["opinion"] == "fresh"
        assert len(problems) == 2
        assert "because" in problems[0]
        assert "amazing" in problems[1]

    def test_non_quad_view_rejected(self):
        with pytest.raises(ValueError, match="quadruple"):
            ParaphraseTemplate("aste")

    def test_conflict_polarity_rejected_at_encode(self):
        template = ParaphraseTemplate()
        t = SentimentTuple(aspect=Span("a"), category="C#G", opinion=Span("o"), polarity="conflict")
        with pytest.raises(ValueError, match="MarkersTemplate"):
            template.encode([t])


class TestParaphraseLossyWarning:
    def test_warns_when_opinion_contains_is(self):
        t = SentimentTuple(
            aspect=Span("room"),
            category="HOTEL#GENERAL",
            opinion=Span("the view is great"),
            polarity="positive",
        )
        with pytest.warns(UserWarning, match="round-trip"):
            ParaphraseTemplate().encode([t])

    def test_no_warning_without_is(self):
        import warnings

        t = SentimentTuple(
            aspect=Span("room"),
            category="HOTEL#GENERAL",
            opinion=Span("great"),
            polarity="positive",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ParaphraseTemplate().encode([t])
