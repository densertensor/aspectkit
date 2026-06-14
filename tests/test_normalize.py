"""Tests for text/label normalisation and span alignment."""

import pytest

from aspectkit.normalize import align_span, canonical_polarity, normalize_text


class TestNormalizeText:
    def test_lowercase_and_collapse(self):
        assert normalize_text("  Spicy  Tuna\tRoll ") == "spicy tuna roll"

    def test_idempotent(self):
        assert normalize_text("spicy tuna roll") == "spicy tuna roll"


class TestCanonicalPolarity:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("positive", "positive"),
            ("POS", "positive"),
            ("2", "positive"),
            (2, "positive"),
            ("negative", "negative"),
            ("NEG", "negative"),
            (0, "negative"),
            ("neutral", "neutral"),
            ("NEU", "neutral"),
            (1, "neutral"),
            ("conflict", "conflict"),
            ("Mixed", "conflict"),
            ("  Positive ", "positive"),
        ],
    )
    def test_mappings(self, label, expected):
        assert canonical_polarity(label) == expected

    def test_unknown_label(self):
        with pytest.raises(ValueError, match="cannot map"):
            canonical_polarity("happy")


class TestAlignSpan:
    def test_exact_match(self):
        span = align_span("pasta", "The pasta was great")
        assert (span.start, span.end) == (4, 9)
        assert span.text == "pasta"

    def test_case_insensitive_keeps_sentence_casing(self):
        span = align_span("PASTA", "The Pasta was great")
        assert span.text == "Pasta"
        assert span.aligned

    def test_word_boundary_preferred(self):
        # "art" must align to the standalone word, not inside "started".
        span = align_span("art", "We started at the art gallery")
        assert (span.start, span.end) == (18, 21)

    def test_substring_fallback_when_no_word_boundary(self):
        span = align_span("rest", "The restaurant")
        assert (span.start, span.end) == (4, 8)

    def test_unalignable_text_kept(self):
        span = align_span("the meal", "Dinner was fine")
        assert not span.aligned
        assert span.text == "the meal"

    def test_empty_text(self):
        span = align_span("   ", "Dinner was fine")
        assert not span.aligned

    def test_regex_metacharacters_are_safe(self):
        span = align_span("price (high)", "I hated the price (high) here")
        assert span.aligned
        assert span.text == "price (high)"


class TestCasefoldOption:
    def test_casefold_folds_german_sharp_s(self):
        # casefold maps ß -> ss; the default .lower() leaves it unchanged.
        assert normalize_text("STRAßE", casefold=True) == "strasse"
        assert normalize_text("STRAßE") == "straße"

    def test_default_unchanged(self):
        assert normalize_text("  Spicy  Tuna ") == "spicy tuna"


class TestExtraPolarityMap:
    def test_extra_map_takes_precedence(self):
        assert canonical_polarity("0", extra_map={"0": "neutral"}) == "neutral"
        # the shared default map is not mutated by a one-off override
        assert canonical_polarity("0") == "negative"

    def test_extra_map_localized_label(self):
        assert canonical_polarity("正面", extra_map={"正面": "positive"}) == "positive"

    def test_extra_map_case_insensitive(self):
        assert canonical_polarity("Good", extra_map={"good": "positive"}) == "positive"


class TestAlignStartFrom:
    def test_start_from_skips_first_occurrence(self):
        sentence = "good food and good service"
        first = align_span("good", sentence)
        assert (first.start, first.end) == (0, 4)
        second = align_span("good", sentence, start_from=first.end)
        assert (second.start, second.end) == (14, 18)

    def test_start_from_zero_is_default(self):
        assert align_span("good", "good food and good service", start_from=0).start == 0
