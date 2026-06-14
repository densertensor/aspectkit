"""Tests for research.validate (Cohen's kappa), against hand-computed values."""

import pytest

from aspectkit.research import validate
from aspectkit.schema import ABSAExample, SentimentTuple, Span

# A 2x2 confusion designed for a known kappa:
#   both positive: 4, both negative: 3, model+/gold-: 2, model-/gold+: 1
#   po = 7/10 = 0.7 ; pe = (6/10)(5/10) + (4/10)(5/10) = 0.5 ; kappa = 0.2/0.5 = 0.4
PRED_POL = ["positive"] * 4 + ["negative"] * 3 + ["positive"] * 2 + ["negative"] * 1
GOLD_POL = ["positive"] * 4 + ["negative"] * 3 + ["negative"] * 2 + ["positive"] * 1


def _examples(polarities):
    return [
        ABSAExample(
            text=f"t{i}", id=f"id{i}", tuples=[SentimentTuple(aspect=Span("x"), polarity=p)]
        )
        for i, p in enumerate(polarities)
    ]


PREDICTIONS = _examples(PRED_POL)
GOLD = _examples(GOLD_POL)


class TestKappa:
    def test_gold_source_hand_computed(self):
        report = validate(PREDICTIONS, gold=GOLD, task="atsc")
        assert report.n == 10
        assert report.observed_agreement == pytest.approx(0.7)
        assert report.kappa == pytest.approx(0.4)
        # Fleiss-Cohen-Everitt (1969) large-sample SE for this 2x2
        assert report.kappa_se == pytest.approx(0.28397, abs=0.001)
        assert report.labels == ("negative", "positive")
        assert report.per_label["positive"] == pytest.approx(0.4)
        assert report.per_label["negative"] == pytest.approx(0.4)

    def test_judge_source_matches_gold(self):
        gold_by_id = {g.id: g.tuples for g in GOLD}
        report = validate(PREDICTIONS, lambda ex: gold_by_id[ex.id], "atsc")
        assert report.kappa == pytest.approx(0.4)

    def test_human_csv_source(self, tmp_path):
        path = tmp_path / "human.csv"
        rows = ["id,aspect,polarity"]
        rows += [f"id{i},x,{p}" for i, p in enumerate(GOLD_POL)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        report = validate(PREDICTIONS, human_csv=path, task="atsc")
        assert report.n == 10
        assert report.kappa == pytest.approx(0.4)

    def test_str_rendering(self):
        report = validate(PREDICTIONS, gold=GOLD, task="atsc")
        rendered = str(report)
        assert "kappa" in rendered and "polarity" in rendered


class TestAlignment:
    def test_unaligned_tuples_skipped(self):
        # predicted aspect "y" has no gold counterpart "x" -> no pairs
        preds = [
            ABSAExample(
                text="t", id="0", tuples=[SentimentTuple(aspect=Span("y"), polarity="positive")]
            )
        ]
        gold = [
            ABSAExample(
                text="t", id="0", tuples=[SentimentTuple(aspect=Span("x"), polarity="positive")]
            )
        ]
        report = validate(preds, gold=gold, task="atsc")
        assert report.n == 0

    def test_stratify_by_other_element(self):
        # agreement on aspect, aligning by polarity
        report = validate(PREDICTIONS, gold=GOLD, task="atsc", stratify_by="aspect")
        assert report.element == "aspect"

    def test_repeated_identity_alignment_order_independent(self):
        # one example, two "food" tuples; gold has the same labels reversed.
        # The multisets agree perfectly -> kappa must be +1, not order-dependent -1.
        pred = [
            ABSAExample(
                text="t",
                id="0",
                tuples=[
                    SentimentTuple(aspect=Span("food"), polarity="positive"),
                    SentimentTuple(aspect=Span("food"), polarity="negative"),
                ],
            )
        ]
        gold = [
            ABSAExample(
                text="t",
                id="0",
                tuples=[
                    SentimentTuple(aspect=Span("food"), polarity="negative"),
                    SentimentTuple(aspect=Span("food"), polarity="positive"),
                ],
            )
        ]
        report = validate(pred, gold=gold, task="atsc")
        assert report.n == 2
        assert report.observed_agreement == 1.0
        assert report.kappa == pytest.approx(1.0)


class TestValidation:
    def test_requires_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate(PREDICTIONS, task="atsc")
        with pytest.raises(ValueError, match="exactly one"):
            validate(PREDICTIONS, gold=GOLD, human_csv="x.csv", task="atsc")

    def test_stratify_not_in_task(self):
        with pytest.raises(ValueError, match="stratify_by"):
            validate(PREDICTIONS, gold=GOLD, task="atsc", stratify_by="opinion")

    def test_gold_length_mismatch(self):
        with pytest.raises(ValueError, match="differ in length"):
            validate(PREDICTIONS, gold=GOLD[:5], task="atsc")

    def test_human_csv_requires_id_column(self, tmp_path):
        path = tmp_path / "noid.csv"
        path.write_text("aspect,polarity\nx,positive\n", encoding="utf-8")
        with pytest.raises(ValueError, match="id"):
            validate(PREDICTIONS, human_csv=path, task="atsc")

    def test_reference_group_not_in_groups(self):
        from aspectkit.aggregate import summarize

        with pytest.raises(ValueError, match="reference_group"):
            summarize(["a"], [[]], group_by=["x"], reference_group="absent")


class TestStratifiedSample:
    def test_hits_n_exactly_with_many_small_strata(self):
        from aspectkit.research.validation import _stratified_sample

        # 100 strata of size 1: naive per-stratum round() would yield 0; Hamilton -> exactly 5
        pairs = [(f"label{i}", "x") for i in range(100)]
        out = _stratified_sample(pairs, 5, seed=0)
        assert len(out) == 5

    def test_returns_all_when_smaller_than_n(self):
        from aspectkit.research.validation import _stratified_sample

        pairs = [("a", "b")] * 3
        assert len(_stratified_sample(pairs, 10, seed=0)) == 3
