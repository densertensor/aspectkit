"""Tests for the corpus aggregator."""

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
