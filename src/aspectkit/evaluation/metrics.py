"""Metric computation: micro tuple P/R/F1 and classification scores.

Implemented from first principles (no scikit-learn dependency) and unit
tested against hand-computed values.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from aspectkit.evaluation.matching import count_exact_matches, count_overlap_matches
from aspectkit.schema import SentimentTuple

__all__ = ["PRF", "LabelScores", "classification_scores", "tuple_prf"]


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class PRF:
    """Micro-averaged precision / recall / F1 over tuples."""

    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_matched: int

    @classmethod
    def from_counts(cls, n_matched: int, n_pred: int, n_gold: int) -> PRF:
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_gold if n_gold else 0.0
        return cls(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            n_pred=n_pred,
            n_gold=n_gold,
            n_matched=n_matched,
        )


def tuple_prf(
    predictions: Sequence[Sequence[SentimentTuple]],
    gold: Sequence[Sequence[SentimentTuple]],
    elements: Sequence[str],
    *,
    matching: str = "exact",
    overlap_threshold: float = 0.5,
) -> PRF:
    """Corpus-level micro P/R/F1 for tuple extraction.

    Args:
        predictions: Predicted tuples, one list per example.
        gold: Gold tuples, aligned with *predictions*.
        elements: The sentiment elements to compare (the task's view).
        matching: ``"exact"`` (SemEval protocol) or ``"overlap"`` (lenient
            span matching for error analysis).
        overlap_threshold: Minimum token-IoU for ``"overlap"`` matching.

    Raises:
        ValueError: If predictions and gold have different lengths or the
            matching mode is unknown.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"predictions ({len(predictions)}) and gold ({len(gold)}) differ in length"
        )
    if matching not in ("exact", "overlap"):
        raise ValueError(f"matching must be 'exact' or 'overlap', got {matching!r}")

    n_matched = n_pred = n_gold = 0
    for pred_tuples, gold_tuples in zip(predictions, gold, strict=True):
        n_pred += len(pred_tuples)
        n_gold += len(gold_tuples)
        if matching == "exact":
            n_matched += count_exact_matches(pred_tuples, gold_tuples, elements)
        else:
            n_matched += count_overlap_matches(
                pred_tuples, gold_tuples, elements, threshold=overlap_threshold
            )
    return PRF.from_counts(n_matched, n_pred, n_gold)


@dataclass(frozen=True)
class LabelScores:
    """Per-label precision / recall / F1 / support."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationScores:
    """Accuracy and macro-F1 for classification views (e.g. ATSC).

    Macro-F1 is reported because polarity classes are imbalanced and the
    neutral class is hard — accuracy alone overstates performance.
    """

    accuracy: float
    macro_f1: float
    per_label: dict[str, LabelScores]
    n: int


def classification_scores(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    labels: Sequence[str] | None = None,
) -> ClassificationScores:
    """Compute accuracy, macro-F1, and per-label P/R/F1.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels; ``None`` marks a missing prediction
            (counts as an error and as a false negative for the gold
            label).
        labels: Label set for macro averaging.  Defaults to the labels
            observed in ``y_true`` (predictions outside this set still
            count as errors/false positives where applicable).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) differ in length")
    if not y_true:
        return ClassificationScores(accuracy=0.0, macro_f1=0.0, per_label={}, n=0)

    label_set = list(labels) if labels is not None else sorted(set(y_true))
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    correct = 0
    for truth, pred in zip(y_true, y_pred, strict=True):
        if pred == truth:
            correct += 1
            tp[truth] += 1
        else:
            fn[truth] += 1
            if pred is not None:
                fp[pred] += 1

    support = Counter(y_true)
    per_label: dict[str, LabelScores] = {}
    for label in label_set:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        per_label[label] = LabelScores(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            support=support[label],
        )
    macro_f1 = sum(s.f1 for s in per_label.values()) / len(per_label) if per_label else 0.0
    return ClassificationScores(
        accuracy=correct / len(y_true),
        macro_f1=macro_f1,
        per_label=per_label,
        n=len(y_true),
    )
