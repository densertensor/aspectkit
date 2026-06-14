"""Corpus-level aggregation: the aspect-based opinion summary.

The original goal of ABSA (Hu & Liu 2004) was never per-sentence labels
but a per-aspect summary of a corpus — what applied researchers actually
want.  :func:`summarize` rolls predicted tuples up into per-aspect (or
per-category) sentiment distributions with representative quotes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from aspectkit.normalize import normalize_text
from aspectkit.schema import SentimentTuple, Span

__all__ = ["AspectSummary", "summarize", "summary_to_frame"]

#: Group label for opinions whose target is implicit.
IMPLICIT_GROUP = "(implicit)"
#: Polarity bucket for tuples without a polarity annotation.
UNLABELLED = "unlabelled"


@dataclass(frozen=True)
class AspectSummary:
    """Aggregated sentiment for one aspect (or category).

    Attributes:
        key: Display name of the group — the most frequent surface form
            of the aspect, or the category label.
        n_mentions: Total number of opinions about this group.
        counts: Opinion counts per polarity label.
        share: ``counts`` normalised to fractions of ``n_mentions``.
        score: Net sentiment in ``[-1, 1]``:
            ``(positive - negative) / n_mentions``.
        quotes: Up to ``max_quotes`` representative source texts per
            polarity.
    """

    key: str
    n_mentions: int
    counts: dict[str, int]
    share: dict[str, float]
    score: float
    quotes: dict[str, list[str]] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = ", ".join(f"{label}={count}" for label, count in sorted(self.counts.items()))
        return f"{self.key}: n={self.n_mentions}, score={self.score:+.2f} ({parts})"


def _group_key(t: SentimentTuple, by: str) -> str | None:
    if by == "aspect":
        return normalize_text(t.aspect.text) if isinstance(t.aspect, Span) else IMPLICIT_GROUP
    if t.category is None:
        return None
    return t.category.strip().upper()


def summarize(
    texts: Sequence[str],
    predictions: Sequence[Sequence[SentimentTuple]],
    *,
    by: Literal["aspect", "category"] = "aspect",
    min_mentions: int = 1,
    max_quotes: int = 3,
) -> list[AspectSummary]:
    """Aggregate per-text predictions into a per-aspect corpus summary.

    Args:
        texts: Source texts, aligned with *predictions* (used for
            representative quotes).
        predictions: Predicted tuples per text.
        by: Group by surface aspect (case-insensitive) or by category.
            Grouping by category requires a task view that predicts
            categories; aspect-less tuples are grouped under
            ``"(implicit)"`` when grouping by aspect, and tuples without
            a category are skipped when grouping by category.
        min_mentions: Drop groups mentioned fewer times than this.
        max_quotes: Representative source texts kept per polarity.

    Returns:
        Summaries sorted by mention count (descending), then key.
    """
    if len(texts) != len(predictions):
        raise ValueError(f"texts ({len(texts)}) and predictions ({len(predictions)}) differ")
    if by not in ("aspect", "category"):
        raise ValueError(f"by must be 'aspect' or 'category', got {by!r}")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    quotes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    surface_forms: dict[str, Counter[str]] = defaultdict(Counter)

    for text, tuples in zip(texts, predictions, strict=True):
        for t in tuples:
            key = _group_key(t, by)
            if key is None:
                continue
            polarity = t.polarity if t.polarity is not None else UNLABELLED
            counts[key][polarity] += 1
            bucket = quotes[key][polarity]
            if len(bucket) < max_quotes and text not in bucket:
                bucket.append(text)
            if by == "aspect" and isinstance(t.aspect, Span):
                surface_forms[key][t.aspect.text] += 1

    summaries: list[AspectSummary] = []
    for key, polarity_counts in counts.items():
        total = sum(polarity_counts.values())
        if total < min_mentions:
            continue
        display = key
        if surface_forms.get(key):
            display = surface_forms[key].most_common(1)[0][0]
        score = (polarity_counts["positive"] - polarity_counts["negative"]) / total
        summaries.append(
            AspectSummary(
                key=display,
                n_mentions=total,
                counts=dict(polarity_counts),
                share={label: count / total for label, count in polarity_counts.items()},
                score=score,
                quotes={label: list(items) for label, items in quotes[key].items()},
            )
        )
    summaries.sort(key=lambda s: (-s.n_mentions, s.key.lower()))
    return summaries


def summary_to_frame(summaries: Sequence[AspectSummary]) -> Any:
    """Convert summaries to a ``pandas.DataFrame`` (requires pandas).

    Columns: ``aspect``, ``n_mentions``, ``score``, one count column per
    polarity, one share column per polarity.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        from aspectkit.exceptions import MissingDependencyError

        raise MissingDependencyError("pandas", "pandas", "summary_to_frame") from exc

    labels = sorted({label for s in summaries for label in s.counts})
    rows = []
    for s in summaries:
        row: dict[str, Any] = {"aspect": s.key, "n_mentions": s.n_mentions, "score": s.score}
        for label in labels:
            row[f"n_{label}"] = s.counts.get(label, 0)
            row[f"share_{label}"] = s.share.get(label, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
