"""Tests for the corpus aggregator."""

from datetime import date

import pytest

from aspectkit.aggregate import summarize, summary_to_frame
from aspectkit.schema import IMPLICIT, SentimentTuple, Span


def t(aspect, polarity, category=None):
    return SentimentTuple(
        aspect=IMPLICIT if aspect is None else Span(aspect),
        polarity=polarity,
        category=category,
    )


@pytest.fixture()
def corpus():
    texts = [
        "The pasta was great",
        "Pasta was amazing, service was slow",
        "Awful pasta tonight",
        "Service was fine",
    ]
    predictions = [
        [t("pasta", "positive")],
        [t("Pasta", "positive"), t("service", "negative")],
        [t("pasta", "negative")],
        [t("service", "neutral")],
    ]
    return texts, predictions


class TestSummarize:
    def test_grouping_is_case_insensitive(self, corpus):
        texts, predictions = corpus
        summaries = summarize(texts, predictions)
        assert [s.key for s in summaries] == ["pasta", "service"]
        pasta = summaries[0]
        assert pasta.n_mentions == 3
        assert pasta.counts == {"positive": 2, "negative": 1}

    def test_score_and_share(self, corpus):
        texts, predictions = corpus
        pasta = summarize(texts, predictions)[0]
        assert pasta.score == pytest.approx((2 - 1) / 3)
        assert pasta.share["positive"] == pytest.approx(2 / 3)

    def test_quotes_capped_and_per_polarity(self, corpus):
        texts, predictions = corpus
        pasta = summarize(texts, predictions, max_quotes=1)[0]
        assert pasta.quotes["positive"] == ["The pasta was great"]
        assert pasta.quotes["negative"] == ["Awful pasta tonight"]

    def test_min_mentions_filter(self, corpus):
        texts, predictions = corpus
        summaries = summarize(texts, predictions, min_mentions=3)
        assert [s.key for s in summaries] == ["pasta"]

    def test_implicit_grouped_together(self):
        summaries = summarize(["a", "b"], [[t(None, "negative")], [t(None, "negative")]])
        assert summaries[0].key == "(implicit)"
        assert summaries[0].n_mentions == 2

    def test_by_category(self):
        predictions = [
            [t("pasta", "positive", category="FOOD#QUALITY")],
            [t(None, "negative", category="food#quality")],
            [t("decor", "positive")],  # no category: skipped in this view
        ]
        summaries = summarize(["a", "b", "c"], predictions, by="category")
        assert len(summaries) == 1
        assert summaries[0].key == "FOOD#QUALITY"
        assert summaries[0].n_mentions == 2

    def test_unlabelled_polarity_bucket(self):
        summaries = summarize(["a"], [[SentimentTuple(aspect=Span("ui"))]])
        assert summaries[0].counts == {"unlabelled": 1}
        assert summaries[0].score == 0.0

    def test_sorted_by_mentions_then_key(self):
        predictions = [[t("zzz", "positive"), t("aaa", "positive")]]
        summaries = summarize(["x"], predictions)
        assert [s.key for s in summaries] == ["aaa", "zzz"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            summarize(["a"], [])

    def test_invalid_by(self):
        with pytest.raises(ValueError, match="by"):
            summarize([], [], by="entity")

    def test_str_rendering(self, corpus):
        texts, predictions = corpus
        rendered = str(summarize(texts, predictions)[0])
        assert "pasta" in rendered and "n=3" in rendered


class TestSummaryToFrame:
    def test_frame_columns(self, corpus):
        pd = pytest.importorskip("pandas")
        texts, predictions = corpus
        frame = summary_to_frame(summarize(texts, predictions))
        assert isinstance(frame, pd.DataFrame)
        assert list(frame["aspect"]) == ["pasta", "service"]
        assert frame.loc[0, "n_positive"] == 2
        assert frame.loc[0, "share_negative"] == pytest.approx(1 / 3)


class TestSummarizeGrouping:
    def test_group_by_partitions(self):
        texts = ["a", "b", "c"]
        preds = [[t("wifi", "positive")], [t("wifi", "negative")], [t("wifi", "positive")]]
        summaries = summarize(texts, preds, group_by=["x", "x", "y"])
        by_group = {(s.group, s.key): s for s in summaries}
        assert by_group[("x", "wifi")].n_mentions == 2
        assert by_group[("x", "wifi")].counts == {"positive": 1, "negative": 1}
        assert by_group[("y", "wifi")].n_mentions == 1

    def test_timestamps_period_buckets(self):
        texts = ["a", "b", "c"]
        preds = [[t("x", "positive")], [t("x", "positive")], [t("x", "negative")]]
        ts = [date(2024, 1, 5), date(2024, 1, 20), date(2024, 2, 3)]
        summaries = summarize(texts, preds, timestamps=ts, window="month")
        periods = {s.period: s.n_mentions for s in summaries}
        assert periods == {"2024-01": 2, "2024-02": 1}

    def test_week_period_label(self):
        summaries = summarize(
            ["a"], [[t("x", "positive")]], timestamps=[date(2024, 1, 3)], window="week"
        )
        assert summaries[0].period == "2024-W01"

    def test_baseline_unchanged_when_no_extras(self):
        # the new fields stay None on the default path
        s = summarize(["a"], [[t("x", "positive")]])[0]
        assert (s.group, s.period, s.ci, s.test_p) == (None, None, None, None)


class TestSummarizeStats:
    def _polar(self, n_pos, n_neg, group=None):
        texts = [f"p{i}" for i in range(n_pos)] + [f"n{i}" for i in range(n_neg)]
        preds = [[t("x", "positive")]] * n_pos + [[t("x", "negative")]] * n_neg
        return texts, preds

    def test_wilson_ci_hand_computed(self):
        texts, preds = self._polar(8, 2)  # 8/10 positive
        s = summarize(texts, preds, ci=True)[0]
        assert s.ci is not None
        assert s.ci[0] == pytest.approx(0.4902, abs=0.002)
        assert s.ci[1] == pytest.approx(0.9433, abs=0.002)

    def test_bootstrap_ci_deterministic_and_brackets(self):
        texts, preds = self._polar(8, 2)
        a = summarize(texts, preds, ci=True, ci_method="bootstrap", seed=1)[0]
        b = summarize(texts, preds, ci=True, ci_method="bootstrap", seed=1)[0]
        assert a.ci == b.ci  # seeded -> reproducible
        assert 0.0 <= a.ci[0] <= 0.8 <= a.ci[1] <= 1.0

    def test_bootstrap_ci_independent_of_other_aspects(self):
        # per-bucket seeding: aspect "x"'s CI must not shift because of other aspects
        texts, preds = self._polar(8, 2)
        solo = summarize(texts, preds, ci=True, ci_method="bootstrap", seed=3)[0]
        texts2 = [*texts, "y1", "y2", "y3"]
        preds2 = [*preds, [t("y", "positive")], [t("y", "negative")], [t("y", "neutral")]]
        multi = next(
            s
            for s in summarize(texts2, preds2, ci=True, ci_method="bootstrap", seed=3)
            if s.key == "x"
        )
        assert solo.ci == multi.ci

    def test_two_proportion_ztest(self):
        # group A: 8/10 positive; reference: 2/10 positive -> z-test p ~ 0.0073
        texts = [f"a{i}" for i in range(10)] + [f"r{i}" for i in range(10)]
        preds = (
            [[t("x", "positive")]] * 8
            + [[t("x", "negative")]] * 2
            + [[t("x", "positive")]] * 2
            + [[t("x", "negative")]] * 8
        )
        groups = ["A"] * 10 + ["ref"] * 10
        summaries = summarize(texts, preds, group_by=groups, reference_group="ref")
        by_group = {s.group: s for s in summaries}
        assert by_group["A"].test_p == pytest.approx(0.0073, abs=0.0005)
        assert by_group["ref"].test_p is None  # the reference itself isn't compared


class TestSummarizeValidation:
    def test_group_by_length_mismatch(self):
        with pytest.raises(ValueError, match="group_by"):
            summarize(["a"], [[]], group_by=["x", "y"])

    def test_timestamps_require_window(self):
        with pytest.raises(ValueError, match="window"):
            summarize(["a"], [[]], timestamps=[date(2024, 1, 1)])

    def test_reference_group_requires_group_by(self):
        with pytest.raises(ValueError, match="reference_group"):
            summarize(["a"], [[]], reference_group="x")

    def test_bad_ci_method(self):
        with pytest.raises(ValueError, match="ci_method"):
            summarize(["a"], [[]], ci=True, ci_method="jackknife")

    def test_bad_window(self):
        with pytest.raises(ValueError, match="window"):
            summarize(
                ["a"], [[t("x", "positive")]], timestamps=[date(2024, 1, 1)], window="fortnight"
            )


class TestFrameExtensions:
    def test_extra_columns_present(self):
        pytest.importorskip("pandas")
        texts = [f"p{i}" for i in range(8)] + [f"n{i}" for i in range(2)]
        preds = [[t("x", "positive")]] * 8 + [[t("x", "negative")]] * 2
        frame = summary_to_frame(summarize(texts, preds, group_by=["g"] * 10, ci=True))
        assert "group" in frame.columns
        assert "ci_low" in frame.columns and "ci_high" in frame.columns
        assert frame.loc[0, "group"] == "g"

    def test_no_extra_columns_on_baseline(self):
        pytest.importorskip("pandas")
        frame = summary_to_frame(summarize(["a"], [[t("x", "positive")]]))
        assert "group" not in frame.columns and "ci_low" not in frame.columns
