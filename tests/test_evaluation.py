"""Tests for matching, metrics, and the evaluate() entry point.

All expected values are hand-computed.
"""

import pytest

from aspectkit.evaluation import evaluate
from aspectkit.evaluation.matching import (
    count_exact_matches,
    count_overlap_matches,
    element_key,
    tuple_key,
)
from aspectkit.evaluation.metrics import classification_scores, tuple_prf
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span


def t(aspect=None, category=None, opinion=None, polarity=None):
    """Terse tuple builder: strings become unaligned spans."""
    return SentimentTuple(
        aspect=IMPLICIT if aspect is None else Span(aspect),
        category=category,
        opinion=Span(opinion) if isinstance(opinion, str) else opinion,
        polarity=polarity,
    )


class TestElementKey:
    def test_aspect_normalised(self):
        assert element_key(t(aspect="  Spicy  Tuna "), "aspect") == "spicy tuna"

    def test_implicit_and_absent_are_distinct(self):
        implicit = SentimentTuple(aspect=IMPLICIT, opinion=IMPLICIT, polarity="positive")
        absent = SentimentTuple(aspect=IMPLICIT, opinion=None, polarity="positive")
        assert element_key(implicit, "opinion") != element_key(absent, "opinion")

    def test_category_case_insensitive(self):
        assert element_key(t(category="food#quality"), "category") == "FOOD#QUALITY"

    def test_unknown_element(self):
        with pytest.raises(ValueError):
            element_key(t(), "emotion")


class TestExactMatching:
    def test_full_quad_match(self):
        pred = [t("pasta", "FOOD#QUALITY", "great", "positive")]
        gold = [t("Pasta", "food#quality", "GREAT", "positive")]
        elements = ("aspect", "category", "opinion", "polarity")
        assert count_exact_matches(pred, gold, elements) == 1

    def test_one_wrong_element_fails(self):
        pred = [t("pasta", "FOOD#QUALITY", "great", "negative")]
        gold = [t("pasta", "FOOD#QUALITY", "great", "positive")]
        elements = ("aspect", "category", "opinion", "polarity")
        assert count_exact_matches(pred, gold, elements) == 0

    def test_duplicates_are_multiset_matched(self):
        pred = [t("wifi", polarity="negative"), t("wifi", polarity="negative")]
        gold = [t("wifi", polarity="negative")]
        assert count_exact_matches(pred, gold, ("aspect", "polarity")) == 1

    def test_restricted_view_ignores_other_elements(self):
        pred = [t("pasta", None, None, "positive")]
        gold = [t("pasta", "FOOD#QUALITY", "great", "positive")]
        assert count_exact_matches(pred, gold, ("aspect", "polarity")) == 1

    def test_tuple_key_order(self):
        key = tuple_key(
            t("pasta", "FOOD#QUALITY", "great", "positive"),
            ("aspect", "category", "opinion", "polarity"),
        )
        assert key == ("pasta", "FOOD#QUALITY", "great", "positive")


class TestOverlapMatching:
    def test_partial_span_matches(self):
        pred = [t("tuna roll", polarity="positive")]
        gold = [t("spicy tuna roll", polarity="positive")]
        # IoU = 2/3 >= 0.5
        assert count_overlap_matches(pred, gold, ("aspect", "polarity")) == 1
        assert count_exact_matches(pred, gold, ("aspect", "polarity")) == 0

    def test_below_threshold_fails(self):
        pred = [t("roll", polarity="positive")]
        gold = [t("spicy tuna roll", polarity="positive")]
        # IoU = 1/3 < 0.5
        assert count_overlap_matches(pred, gold, ("aspect", "polarity")) == 0
        assert count_overlap_matches(pred, gold, ("aspect", "polarity"), threshold=0.3) == 1

    def test_categorical_elements_still_exact(self):
        pred = [t("tuna roll", polarity="negative")]
        gold = [t("spicy tuna roll", polarity="positive")]
        assert count_overlap_matches(pred, gold, ("aspect", "polarity")) == 0

    def test_implicit_only_matches_implicit(self):
        pred = [t(None, polarity="positive")]
        gold = [t("pasta", polarity="positive")]
        assert count_overlap_matches(pred, gold, ("aspect", "polarity")) == 0
        assert (
            count_overlap_matches(pred, [t(None, polarity="positive")], ("aspect", "polarity")) == 1
        )

    def test_greedy_one_to_one(self):
        # one prediction cannot consume two golds
        pred = [t("tuna roll", polarity="positive")]
        gold = [t("tuna roll", polarity="positive"), t("tuna roll", polarity="positive")]
        assert count_overlap_matches(pred, gold, ("aspect", "polarity")) == 1


class TestTuplePRF:
    def test_hand_computed(self):
        # example 1: 2 pred, 1 correct; gold 2  | example 2: 1 pred, 1 correct; gold 2
        preds = [
            [t("pasta", polarity="positive"), t("wine", polarity="positive")],
            [t("service", polarity="negative")],
        ]
        golds = [
            [t("pasta", polarity="positive"), t("wine", polarity="negative")],
            [t("service", polarity="negative"), t("wait", polarity="negative")],
        ]
        prf = tuple_prf(preds, golds, ("aspect", "polarity"))
        assert prf.n_matched == 2 and prf.n_pred == 3 and prf.n_gold == 4
        assert prf.precision == pytest.approx(2 / 3)
        assert prf.recall == pytest.approx(0.5)
        assert prf.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))

    def test_empty_predictions(self):
        prf = tuple_prf([[]], [[t("pasta", polarity="positive")]], ("aspect", "polarity"))
        assert (prf.precision, prf.recall, prf.f1) == (0.0, 0.0, 0.0)

    def test_empty_everything(self):
        prf = tuple_prf([[]], [[]], ("aspect", "polarity"))
        assert (prf.precision, prf.recall, prf.f1) == (0.0, 0.0, 0.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="differ in length"):
            tuple_prf([[]], [[], []], ("aspect",))

    def test_unknown_matching_mode(self):
        with pytest.raises(ValueError, match="matching"):
            tuple_prf([[]], [[]], ("aspect",), matching="fuzzy")


class TestClassificationScores:
    def test_hand_computed(self):
        y_true = ["positive", "positive", "negative", "neutral"]
        y_pred = ["positive", "negative", "negative", None]
        scores = classification_scores(y_true, y_pred)
        assert scores.accuracy == pytest.approx(0.5)
        # positive: tp=1 fp=0 fn=1 -> P=1, R=.5, F1=2/3
        # negative: tp=1 fp=1 fn=0 -> P=.5, R=1, F1=2/3
        # neutral:  tp=0           -> F1=0
        assert scores.per_label["positive"].f1 == pytest.approx(2 / 3)
        assert scores.per_label["negative"].precision == pytest.approx(0.5)
        assert scores.per_label["neutral"].f1 == 0.0
        assert scores.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
        assert scores.per_label["positive"].support == 2

    def test_empty(self):
        scores = classification_scores([], [])
        assert scores.n == 0 and scores.accuracy == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            classification_scores(["positive"], [])


class TestEvaluate:
    def test_extraction_report(self):
        gold = [
            ABSAExample(
                text="The pasta was great",
                tuples=[t("pasta", "FOOD#QUALITY", "great", "positive")],
            )
        ]
        predictions = [[t("pasta", "FOOD#QUALITY", "great", "positive")]]
        report = evaluate(gold, predictions, "acos", lenient=True)
        assert report.kind == "extraction"
        assert report.exact.f1 == 1.0
        assert report.lenient.f1 == 1.0
        assert report.elements == ("aspect", "category", "opinion", "polarity")
        assert "exact match" in str(report)

    def test_extraction_without_lenient(self):
        gold = [ABSAExample(text="x", tuples=[t("a", polarity="positive")])]
        report = evaluate(gold, [[]], "e2e")
        assert report.lenient is None
        assert report.exact.recall == 0.0

    def test_classification_report_alignment(self):
        gold = [
            ABSAExample(
                text="Good pasta, bad wine",
                tuples=[t("pasta", polarity="positive"), t("wine", polarity="negative")],
            )
        ]
        # prediction order differs from gold order: alignment is by aspect
        predictions = [[t("wine", polarity="negative"), t("pasta", polarity="negative")]]
        report = evaluate(gold, predictions, "atsc")
        assert report.kind == "classification"
        assert report.classification.accuracy == pytest.approx(0.5)

    def test_classification_missing_prediction_counts_wrong(self):
        gold = [
            ABSAExample(
                text="Good pasta",
                tuples=[t("pasta", polarity="positive"), t("wine", polarity="negative")],
            )
        ]
        predictions = [[t("pasta", polarity="positive")]]
        report = evaluate(gold, predictions, "atsc")
        assert report.classification.accuracy == pytest.approx(0.5)

    def test_per_element_breakdown(self):
        gold = [
            ABSAExample(
                text="The pasta was great",
                tuples=[t("pasta", "FOOD#QUALITY", polarity="positive")],
            )
        ]
        # right aspect and polarity, wrong category: the tuple misses,
        # and the breakdown shows exactly which element is responsible
        predictions = [[t("pasta", "SERVICE#GENERAL", polarity="positive")]]
        report = evaluate(gold, predictions, "tasd")
        assert report.exact.f1 == 0.0
        assert report.per_element["aspect"].f1 == 1.0
        assert report.per_element["category"].f1 == 0.0
        assert report.per_element["polarity"].f1 == 1.0
        assert tuple(report.per_element) == ("aspect", "category", "polarity")
        assert "by element" in str(report)

    def test_per_element_absent_for_single_element_tasks(self):
        gold = [ABSAExample(text="x", tuples=[t("a")])]
        report = evaluate(gold, [[t("a")]], "ate")
        assert report.per_element is None
        assert "by element" not in str(report)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            evaluate([ABSAExample(text="x")], [], "acos")

    def test_report_to_dict_serialisable(self):
        import json

        gold = [ABSAExample(text="x", tuples=[t("a", polarity="positive")])]
        report = evaluate(gold, [[t("a", polarity="positive")]], "e2e")
        json.dumps(report.to_dict())
