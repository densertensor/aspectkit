"""Evaluation: exact-match tuple F1 (the SemEval protocol) first-class.

The headline function is :func:`evaluate`, which selects the correct
protocol from the task view: micro tuple P/R/F1 with exact matching for
extraction tasks, accuracy + macro-F1 for classification tasks.  A
lenient overlap mode can be reported *alongside* exact match for error
analysis.
"""

from __future__ import annotations

from collections.abc import Sequence

from aspectkit.evaluation.matching import (
    count_exact_matches,
    count_overlap_matches,
    element_key,
    tuple_key,
)
from aspectkit.evaluation.metrics import (
    PRF,
    ClassificationScores,
    LabelScores,
    classification_scores,
    tuple_prf,
)
from aspectkit.evaluation.report import EvaluationReport
from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = [
    "PRF",
    "ClassificationScores",
    "EvaluationReport",
    "LabelScores",
    "classification_scores",
    "count_exact_matches",
    "count_overlap_matches",
    "element_key",
    "evaluate",
    "tuple_key",
    "tuple_prf",
]


def _align_classification(
    gold: Sequence[ABSAExample],
    predictions: Sequence[Sequence[SentimentTuple]],
    task: Task,
) -> tuple[list[str], list[str | None]]:
    """Pair gold polarity labels with predicted ones via the given elements.

    For each gold tuple, the first unused prediction whose *given*
    elements (e.g. the aspect in ATSC) match is consumed; a gold tuple
    without a matching prediction contributes a ``None`` prediction,
    which scores as an error.
    """
    given = task.ordered_elements(task.given)
    y_true: list[str] = []
    y_pred: list[str | None] = []
    for example, pred_tuples in zip(gold, predictions, strict=True):
        used: set[int] = set()
        for gold_tuple in example.tuples:
            if gold_tuple.polarity is None:
                raise ValueError(
                    f"gold tuple without polarity in example {example.id or example.text[:40]!r}; "
                    "classification evaluation needs labelled targets"
                )
            gold_key = tuple_key(gold_tuple, given)
            chosen: str | None = None
            for i, pred_tuple in enumerate(pred_tuples):
                if i in used:
                    continue
                if tuple_key(pred_tuple, given) == gold_key:
                    used.add(i)
                    chosen = pred_tuple.polarity
                    break
            y_true.append(gold_tuple.polarity)
            y_pred.append(chosen)
    return y_true, y_pred


def evaluate(
    gold: Sequence[ABSAExample],
    predictions: Sequence[Sequence[SentimentTuple]],
    task: str | Task,
    *,
    lenient: bool = False,
    overlap_threshold: float = 0.5,
) -> EvaluationReport:
    """Score predictions against gold annotations under a task view.

    Args:
        gold: Gold examples carrying reference tuples.
        predictions: Predicted tuples, one list per gold example, in the
            same order.
        task: Task name or :class:`~aspectkit.tasks.Task`; determines the
            elements compared and the protocol (extraction vs
            classification).
        lenient: For extraction tasks, additionally report token-overlap
            matching alongside exact match.
        overlap_threshold: Minimum token-IoU for the lenient mode.

    Returns:
        An :class:`EvaluationReport`.

    Raises:
        ValueError: If lengths differ.
    """
    task = get_task(task)
    if len(gold) != len(predictions):
        raise ValueError(
            f"gold ({len(gold)}) and predictions ({len(predictions)}) differ in length"
        )

    if task.is_classification:
        y_true, y_pred = _align_classification(gold, predictions, task)
        scores = classification_scores(y_true, y_pred)
        return EvaluationReport(
            task=task.name,
            kind="classification",
            n_examples=len(gold),
            elements=task.ordered_elements(),
            classification=scores,
        )

    elements = task.ordered_elements(task.predicted)
    gold_tuples = [example.tuples for example in gold]
    exact = tuple_prf(predictions, gold_tuples, elements, matching="exact")
    lenient_prf = (
        tuple_prf(
            predictions,
            gold_tuples,
            elements,
            matching="overlap",
            overlap_threshold=overlap_threshold,
        )
        if lenient
        else None
    )
    # Per-element diagnostic: which element is responsible for tuple-level
    # misses (span boundaries vs categories vs polarity).  Exact matching,
    # each element scored in isolation — the breakdown reported by the
    # ACOS/ASQP literature.
    per_element = (
        {
            element: tuple_prf(predictions, gold_tuples, (element,), matching="exact")
            for element in elements
        }
        if len(elements) > 1
        else None
    )
    return EvaluationReport(
        task=task.name,
        kind="extraction",
        n_examples=len(gold),
        elements=elements,
        exact=exact,
        lenient=lenient_prf,
        per_element=per_element,
    )
