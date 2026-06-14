"""Tuple matching strategies for evaluation.

The comparability standard in the ABSA literature is **exact match**: a
predicted tuple counts as correct only if every evaluated element equals
its gold counterpart (Pontiki et al. 2014-2016; Cai et al. 2021).  Span
elements are compared on normalised surface text, which keeps the
protocol well-defined for generative and LLM backends whose outputs
carry no offsets.

A **lenient** overlap mode (token-IoU on span elements) is provided for
error analysis.  Per the methodology literature it should be reported
*alongside* exact match, never instead of it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from aspectkit.normalize import normalize_text
from aspectkit.schema import SentimentTuple, Span

__all__ = ["count_exact_matches", "count_overlap_matches", "element_key", "tuple_key"]

#: Stand-in key for implicit span elements.
_IMPLICIT_KEY = "<implicit>"
#: Stand-in key for elements absent from a tuple.
_ABSENT_KEY = "<absent>"

_SPAN_ELEMENTS = frozenset({"aspect", "opinion"})


def element_key(t: SentimentTuple, element: str) -> str:
    """Normalised, comparable value of one element of a tuple."""
    if element == "aspect":
        return normalize_text(t.aspect.text) if isinstance(t.aspect, Span) else _IMPLICIT_KEY
    if element == "opinion":
        if t.opinion is None:
            return _ABSENT_KEY
        return normalize_text(t.opinion.text) if isinstance(t.opinion, Span) else _IMPLICIT_KEY
    if element == "category":
        return _ABSENT_KEY if t.category is None else t.category.strip().upper()
    if element == "polarity":
        return _ABSENT_KEY if t.polarity is None else t.polarity
    raise ValueError(f"unknown element {element!r}")


def tuple_key(t: SentimentTuple, elements: Sequence[str]) -> tuple[str, ...]:
    """Hashable key of a tuple restricted to *elements* (exact matching)."""
    return tuple(element_key(t, e) for e in elements)


def count_exact_matches(
    pred: Sequence[SentimentTuple],
    gold: Sequence[SentimentTuple],
    elements: Sequence[str],
) -> int:
    """Number of predicted tuples exactly matching gold tuples.

    Duplicates are handled as multisets: a gold tuple can satisfy at most
    one prediction and vice versa.
    """
    pred_counts = Counter(tuple_key(t, elements) for t in pred)
    gold_counts = Counter(tuple_key(t, elements) for t in gold)
    return sum((pred_counts & gold_counts).values())


def _token_iou(a: str, b: str) -> float:
    """Jaccard overlap of the normalised token sets of two span texts."""
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _span_pair_score(
    p: SentimentTuple, g: SentimentTuple, span_elements: Sequence[str], threshold: float
) -> float | None:
    """Mean token-IoU across span elements, or ``None`` if any element
    fails the threshold (implicit only matches implicit)."""
    scores: list[float] = []
    for element in span_elements:
        pk = element_key(p, element)
        gk = element_key(g, element)
        if pk in (_IMPLICIT_KEY, _ABSENT_KEY) or gk in (_IMPLICIT_KEY, _ABSENT_KEY):
            if pk != gk:
                return None
            scores.append(1.0)
            continue
        iou = _token_iou(pk, gk)
        if iou < threshold:
            return None
        scores.append(iou)
    return sum(scores) / len(scores) if scores else 1.0


def count_overlap_matches(
    pred: Sequence[SentimentTuple],
    gold: Sequence[SentimentTuple],
    elements: Sequence[str],
    *,
    threshold: float = 0.5,
) -> int:
    """Number of predictions matched to gold under lenient span overlap.

    Categorical elements (category, polarity) must still match exactly;
    span elements match when their token-IoU reaches *threshold*.  Pairs
    are resolved greedily by descending IoU so each tuple is used at most
    once (with exact spans this reduces to exact matching).
    """
    span_elements = [e for e in elements if e in _SPAN_ELEMENTS]
    cat_elements = [e for e in elements if e not in _SPAN_ELEMENTS]

    candidates: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if any(element_key(p, e) != element_key(g, e) for e in cat_elements):
                continue
            score = _span_pair_score(p, g, span_elements, threshold)
            if score is not None:
                candidates.append((score, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched = 0
    for _, i, j in candidates:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        matched += 1
    return matched
