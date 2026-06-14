"""Corpus-level aggregation: the aspect-based opinion summary.

The original goal of ABSA (Hu & Liu 2004) was never per-sentence labels
but a per-aspect summary of a corpus — what applied researchers actually
want.  :func:`summarize` rolls predicted tuples up into per-aspect (or
per-category) sentiment distributions with representative quotes.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
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
        group: The ``group_by`` label this summary belongs to, if grouping
            was requested (else ``None``).
        period: The time-window label, if ``timestamps``/``window`` were
            given (else ``None``).
        ci: 95% confidence interval ``(low, high)`` of the positive-mention
            proportion, if ``ci=True`` was requested (else ``None``).
        test_p: Two-proportion z-test p-value of this group's positive rate
            vs the ``reference_group``'s, for the same aspect/period, if a
            reference was given (else ``None``).
    """

    key: str
    n_mentions: int
    counts: dict[str, int]
    share: dict[str, float]
    score: float
    quotes: dict[str, list[str]] = field(default_factory=dict)
    group: str | None = None
    period: str | None = None
    ci: tuple[float, float] | None = None
    test_p: float | None = None

    def __str__(self) -> str:
        parts = ", ".join(f"{label}={count}" for label, count in sorted(self.counts.items()))
        tags = []
        if self.group is not None:
            tags.append(f"group={self.group}")
        if self.period is not None:
            tags.append(f"period={self.period}")
        prefix = f"[{', '.join(tags)}] " if tags else ""
        ci = f" ci=[{self.ci[0]:.2f}, {self.ci[1]:.2f}]" if self.ci is not None else ""
        pval = f" p={self.test_p:.3g}" if self.test_p is not None else ""
        return (
            f"{prefix}{self.key}: n={self.n_mentions}, score={self.score:+.2f} ({parts}){ci}{pval}"
        )


def _group_key(t: SentimentTuple, by: str) -> str | None:
    if by == "aspect":
        return normalize_text(t.aspect.text) if isinstance(t.aspect, Span) else IMPLICIT_GROUP
    if t.category is None:
        return None
    return t.category.strip().upper()


def _period_label(ts: date | None, window: str) -> str | None:
    """Bucket a date/datetime into a period label at *window* granularity."""
    if ts is None:
        return None
    if window == "year":
        return f"{ts.year:04d}"
    if window == "month":
        return f"{ts.year:04d}-{ts.month:02d}"
    if window == "week":
        iso = ts.isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    if window == "day":
        return f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"
    raise ValueError(f"window must be 'day', 'week', 'month', or 'year', got {window!r}")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wilson_ci(pos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for the proportion ``pos / n``."""
    if n == 0:
        return (0.0, 0.0)
    p = pos / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _bootstrap_ci(pos: int, n: int, rng: random.Random, samples: int = 1000) -> tuple[float, float]:
    """Percentile bootstrap CI (95%) for the proportion ``pos / n``."""
    if n == 0:
        return (0.0, 0.0)
    p = pos / n
    props = sorted(sum(rng.random() < p for _ in range(n)) / n for _ in range(samples))
    return (props[int(0.025 * samples)], props[int(0.975 * samples)])


def _two_proportion_p(pos_a: int, n_a: int, pos_b: int, n_b: int) -> float | None:
    """Two-sided p-value of a two-proportion z-test on positive rates."""
    if n_a == 0 or n_b == 0:
        return None
    p_pool = (pos_a + pos_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0.0:
        return 1.0
    z = (pos_a / n_a - pos_b / n_b) / se
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def summarize(
    texts: Sequence[str],
    predictions: Sequence[Sequence[SentimentTuple]],
    *,
    by: Literal["aspect", "category"] = "aspect",
    min_mentions: int = 1,
    max_quotes: int = 3,
    group_by: Sequence[str | None] | None = None,
    timestamps: Sequence[date | None] | None = None,
    window: str | None = None,
    ci: bool = False,
    ci_method: Literal["wilson", "bootstrap"] = "wilson",
    reference_group: str | None = None,
    seed: int = 0,
) -> list[AspectSummary]:
    """Aggregate per-text predictions into a per-aspect corpus summary.

    With all the optional knobs at their defaults this is the plain
    per-aspect (or per-category) rollup; the extras add cross-section and
    statistical detail without changing that baseline output.

    Args:
        texts: Source texts, aligned with *predictions* (used for quotes).
        predictions: Predicted tuples per text.
        by: Group by surface aspect (case-insensitive) or by category.
            Aspect-less tuples group under ``"(implicit)"`` for ``"aspect"``;
            category-less tuples are skipped for ``"category"``.
        min_mentions: Drop groups mentioned fewer times than this.
        max_quotes: Representative source texts kept per polarity.
        group_by: Optional per-text label (aligned with *texts*); summaries
            are then produced per (group, aspect) and tagged with ``group``.
        timestamps: Optional per-text ``date``/``datetime`` (aligned);
            requires *window* and tags summaries with a ``period`` bucket.
        window: Period granularity for *timestamps*: ``"day"``, ``"week"``,
            ``"month"``, or ``"year"``.
        ci: Attach a 95% confidence interval of the positive-mention
            proportion to each summary (:attr:`AspectSummary.ci`).
        ci_method: ``"wilson"`` (default, closed-form) or ``"bootstrap"``.
        reference_group: A ``group_by`` value to compare every other group
            against (same aspect + period) via a two-proportion z-test on
            positive rates; the p-value lands in :attr:`AspectSummary.test_p`.
        seed: RNG seed for ``ci_method="bootstrap"`` (reproducibility).

    Returns:
        Summaries sorted by group, period, then mention count (desc), key.
    """
    if len(texts) != len(predictions):
        raise ValueError(f"texts ({len(texts)}) and predictions ({len(predictions)}) differ")
    if by not in ("aspect", "category"):
        raise ValueError(f"by must be 'aspect' or 'category', got {by!r}")
    if group_by is not None and len(group_by) != len(texts):
        raise ValueError(f"group_by ({len(group_by)}) and texts ({len(texts)}) differ")
    if timestamps is not None and len(timestamps) != len(texts):
        raise ValueError(f"timestamps ({len(timestamps)}) and texts ({len(texts)}) differ")
    if timestamps is not None and window is None:
        raise ValueError("window= is required when timestamps= is given")
    if ci_method not in ("wilson", "bootstrap"):
        raise ValueError(f"ci_method must be 'wilson' or 'bootstrap', got {ci_method!r}")
    if reference_group is not None and group_by is None:
        raise ValueError("reference_group= requires group_by=")
    if (
        reference_group is not None
        and group_by is not None
        and reference_group not in set(group_by)
    ):
        raise ValueError(f"reference_group={reference_group!r} not found among group_by values")

    Bucket = tuple[Any, Any, str]  # (group, period, normalized key)
    counts: dict[Bucket, Counter[str]] = defaultdict(Counter)
    quotes: dict[Bucket, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    surface_forms: dict[Bucket, Counter[str]] = defaultdict(Counter)

    for index, (text, tuples) in enumerate(zip(texts, predictions, strict=True)):
        group = group_by[index] if group_by is not None else None
        period = (
            _period_label(timestamps[index], window)
            if timestamps is not None and window is not None
            else None
        )
        for t in tuples:
            norm = _group_key(t, by)
            if norm is None:
                continue
            bucket: Bucket = (group, period, norm)
            polarity = t.polarity if t.polarity is not None else UNLABELLED
            counts[bucket][polarity] += 1
            quoted = quotes[bucket][polarity]
            if len(quoted) < max_quotes and text not in quoted:
                quoted.append(text)
            if by == "aspect" and isinstance(t.aspect, Span):
                surface_forms[bucket][t.aspect.text] += 1

    # positive-rate index for the two-proportion z-test, keyed by (period, norm)
    rates: dict[tuple[Any, str], dict[Any, tuple[int, int]]] = defaultdict(dict)
    if reference_group is not None:
        for (group, period, norm), polarity_counts in counts.items():
            total = sum(polarity_counts.values())
            rates[(period, norm)][group] = (polarity_counts.get("positive", 0), total)

    summaries: list[AspectSummary] = []
    for (group, period, norm), polarity_counts in counts.items():
        total = sum(polarity_counts.values())
        if total < min_mentions:
            continue
        display = norm
        if surface_forms.get((group, period, norm)):
            display = surface_forms[(group, period, norm)].most_common(1)[0][0]
        pos = polarity_counts.get("positive", 0)
        score = (pos - polarity_counts.get("negative", 0)) / total
        interval: tuple[float, float] | None = None
        if ci:
            if ci_method == "wilson":
                interval = _wilson_ci(pos, total)
            else:
                # Seed per bucket from its own identity so a bucket's CI does
                # not depend on how many draws earlier buckets consumed.
                interval = _bootstrap_ci(
                    pos, total, random.Random(repr((seed, group, period, norm)))
                )
        test_p: float | None = None
        if reference_group is not None and group != reference_group:
            reference = rates[(period, norm)].get(reference_group)
            if reference is not None:
                test_p = _two_proportion_p(pos, total, reference[0], reference[1])
        summaries.append(
            AspectSummary(
                key=display,
                n_mentions=total,
                counts=dict(polarity_counts),
                share={label: count / total for label, count in polarity_counts.items()},
                score=score,
                quotes={
                    label: list(items) for label, items in quotes[(group, period, norm)].items()
                },
                group=group,
                period=period,
                ci=interval,
                test_p=test_p,
            )
        )
    summaries.sort(key=lambda s: (s.group or "", s.period or "", -s.n_mentions, s.key.lower()))
    return summaries


def summary_to_frame(summaries: Sequence[AspectSummary]) -> Any:
    """Convert summaries to a ``pandas.DataFrame`` (requires pandas).

    Columns: ``aspect``, ``n_mentions``, ``score``, one count column per
    polarity, one share column per polarity.  ``group``, ``period``,
    ``ci_low``/``ci_high``, and ``test_p`` columns are added when the
    corresponding :func:`summarize` features were used.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        from aspectkit.exceptions import MissingDependencyError

        raise MissingDependencyError("pandas", "pandas", "summary_to_frame") from exc

    has_group = any(s.group is not None for s in summaries)
    has_period = any(s.period is not None for s in summaries)
    has_ci = any(s.ci is not None for s in summaries)
    has_p = any(s.test_p is not None for s in summaries)
    labels = sorted({label for s in summaries for label in s.counts})
    rows = []
    for s in summaries:
        row: dict[str, Any] = {}
        if has_group:
            row["group"] = s.group
        if has_period:
            row["period"] = s.period
        row.update(aspect=s.key, n_mentions=s.n_mentions, score=s.score)
        for label in labels:
            row[f"n_{label}"] = s.counts.get(label, 0)
            row[f"share_{label}"] = s.share.get(label, 0.0)
        if has_ci:
            row["ci_low"] = s.ci[0] if s.ci is not None else None
            row["ci_high"] = s.ci[1] if s.ci is not None else None
        if has_p:
            row["test_p"] = s.test_p
        rows.append(row)
    return pd.DataFrame(rows)
