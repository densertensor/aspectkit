"""Validate predictions against a second opinion via Cohen's kappa.

Inter-rater agreement is the honest way to ask "should I trust these
predictions?": sample tuples, get a second label for each (an LLM judge, a
gold set, or a human CSV), align them on identity, and compute Cohen's
:math:`\\kappa` on the element of interest (polarity by default).  Tuples
are aligned by :func:`~aspectkit.evaluation.matching.tuple_key` over every
element *except* the one being judged, so agreement is measured on like for
like.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aspectkit.evaluation.matching import element_key, tuple_key
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span
from aspectkit.tasks import Task, get_task

__all__ = ["ValidationReport", "validate"]


@dataclass(frozen=True)
class ValidationReport:
    """Cohen's-kappa agreement between predictions and a reference rater.

    Attributes:
        element: The element agreement was measured on (e.g. ``"polarity"``).
        n: Number of aligned tuple pairs scored.
        kappa: Cohen's kappa over all classes.
        kappa_se: Large-sample standard error of *kappa*.
        observed_agreement: Raw proportion of agreeing pairs.
        per_label: One-vs-rest kappa for each class.
        labels: The class labels observed.
    """

    element: str
    n: int
    kappa: float
    kappa_se: float
    observed_agreement: float
    per_label: dict[str, float]
    labels: tuple[str, ...]

    def __str__(self) -> str:
        lines = [
            f"ValidationReport({self.element}, n={self.n}): kappa={self.kappa:.3f} "
            f"(SE={self.kappa_se:.3f}), agreement={self.observed_agreement:.3f}"
        ]
        for label, k in sorted(self.per_label.items()):
            lines.append(f"  {label:<10} kappa={k:.3f}")
        return "\n".join(lines)


def _cohens_kappa(
    rater_a: Sequence, rater_b: Sequence, labels: Sequence
) -> tuple[float, float, float]:
    """Return (kappa, asymptotic SE, observed agreement).

    The standard error is the Fleiss-Cohen-Everitt (1969) large-sample
    estimate computed from the full confusion matrix.
    """
    n = len(rater_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    classes = list(labels)
    joint = Counter(zip(rater_a, rater_b, strict=True))
    p = {(i, j): joint.get((i, j), 0) / n for i in classes for j in classes}
    row = {i: sum(p[(i, j)] for j in classes) for i in classes}  # rater_a marginals
    col = {j: sum(p[(i, j)] for i in classes) for j in classes}  # rater_b marginals
    po = sum(p[(i, i)] for i in classes)
    pe = sum(row[i] * col[i] for i in classes)
    if pe >= 1.0:  # only one class present: kappa undefined -> 0 (or 1 if perfect)
        return (1.0 if po >= 1.0 else 0.0), 0.0, po
    kappa = (po - pe) / (1.0 - pe)
    a = sum(p[(i, i)] * (1 - (row[i] + col[i]) * (1 - kappa)) ** 2 for i in classes)
    b = (1 - kappa) ** 2 * sum(
        p[(i, j)] * (col[i] + row[j]) ** 2 for i in classes for j in classes if i != j
    )
    c = (kappa - pe * (1 - kappa)) ** 2
    var = (a + b - c) / ((1.0 - pe) ** 2 * n)
    return kappa, (math.sqrt(var) if var > 0 else 0.0), po


def _load_human_csv(path: str | Path) -> dict[str, list[SentimentTuple]]:
    """Read reference tuples from a CSV keyed by example ``id``.

    Recognised columns: ``id``, ``aspect`` (blank -> implicit), ``polarity``,
    and optionally ``category`` and ``opinion``.
    """
    by_id: dict[str, list[SentimentTuple]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"human_csv {str(path)!r} must have an 'id' column to align by id")
        for row in reader:
            example_id = (row.get("id") or "").strip()
            aspect_text = (row.get("aspect") or "").strip()
            opinion_text = (row.get("opinion") or "").strip()
            by_id[example_id].append(
                SentimentTuple(
                    aspect=Span(aspect_text) if aspect_text else IMPLICIT,
                    polarity=(row.get("polarity") or "").strip() or None,
                    category=(row.get("category") or "").strip() or None,
                    opinion=Span(opinion_text) if opinion_text else None,
                )
            )
    return dict(by_id)


def _stratified_sample(
    pairs: list[tuple[str, str]], n_sample: int, seed: int
) -> list[tuple[str, str]]:
    """Sample exactly *n_sample* pairs, preserving the predicted-label mix.

    Quotas use largest-remainder (Hamilton) apportionment so the total is
    exactly *n_sample* and the sample is never emptied by many small strata
    each rounding to zero.
    """
    if len(pairs) <= n_sample:
        return list(pairs)
    rng = random.Random(seed)
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in pairs:
        strata[pair[0]].append(pair)
    total = len(pairs)
    quota: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for key, group in strata.items():
        exact = n_sample * len(group) / total
        quota[key] = int(exact)
        allocated += quota[key]
        remainders.append((exact - int(exact), key))
    remainders.sort(key=lambda item: (-item[0], item[1]))  # largest fractional remainder first
    for _, key in remainders[: n_sample - allocated]:
        quota[key] += 1
    sample: list[tuple[str, str]] = []
    for key, group in strata.items():
        rng.shuffle(group)
        sample.extend(group[: quota[key]])
    return sample


def _align_labels(pred_labels: list[str], ref_labels: list[str]) -> list[tuple[str, str]]:
    """Pair the stratify-labels of co-identified tuples, equal labels first.

    Several tuples can share one identity (e.g. a repeated aspect with
    differing polarity).  The marginals (hence pe) are fixed regardless of
    pairing, so matching equal labels first maximises the honest diagonal and
    avoids order-dependent mis-pairing; surplus on either side is unmatched.
    """
    available = Counter(ref_labels)
    leftover_pred: list[str] = []
    pairs: list[tuple[str, str]] = []
    for label in pred_labels:
        if available[label] > 0:
            available[label] -= 1
            pairs.append((label, label))
        else:
            leftover_pred.append(label)
    pairs.extend(zip(leftover_pred, available.elements(), strict=False))
    return pairs


def validate(
    predictions: Sequence[ABSAExample],
    judge: Callable[[ABSAExample], Sequence[SentimentTuple]] | None = None,
    task: str | Task = "acos",
    *,
    n_sample: int = 100,
    seed: int = 42,
    stratify_by: str = "polarity",
    gold: Sequence[ABSAExample] | None = None,
    human_csv: str | Path | None = None,
) -> ValidationReport:
    """Measure agreement between *predictions* and a second rater.

    Exactly one reference source must be supplied: *judge* (a callable that
    re-annotates each example), *gold* (parallel gold examples), or
    *human_csv* (a CSV of reference tuples keyed by example id).  Predicted
    and reference tuples are aligned by every element except *stratify_by*,
    and Cohen's kappa is computed on *stratify_by* over a sample of up to
    *n_sample* aligned pairs (stratified by the predicted label).

    Args:
        predictions: Model predictions as examples (text + predicted tuples).
        judge: ``example -> reference tuples`` for that example's text.
        task: Task view (determines which elements identify a tuple).
        n_sample: Maximum aligned pairs to score (stratified down to this).
        seed: RNG seed for sampling.
        stratify_by: The element agreement is measured on (default polarity).
        gold: Reference examples aligned by position with *predictions*.
        human_csv: CSV reference, aligned to *predictions* by example id.

    Raises:
        ValueError: If not exactly one reference source is given, lengths
            mismatch, or *stratify_by* is not an element of *task*.
    """
    resolved = get_task(task)
    if stratify_by not in resolved.elements:
        raise ValueError(
            f"stratify_by={stratify_by!r} is not an element of task {resolved.name!r} "
            f"({sorted(resolved.elements)})"
        )
    if sum(source is not None for source in (judge, gold, human_csv)) != 1:
        raise ValueError("exactly one of judge=, gold=, or human_csv= must be given")

    pred_list = list(predictions)
    if gold is not None:
        gold_list = list(gold)
        if len(gold_list) != len(pred_list):
            raise ValueError(
                f"gold ({len(gold_list)}) and predictions ({len(pred_list)}) differ in length"
            )
        ref_tuples = [list(g.tuples) for g in gold_list]
    elif human_csv is not None:
        by_id = _load_human_csv(human_csv)
        ref_tuples = [list(by_id.get(ex.id or "", [])) for ex in pred_list]
    else:
        assert judge is not None  # exactly-one check above guarantees this
        ref_tuples = [list(judge(ex)) for ex in pred_list]

    identity = tuple(e for e in resolved.ordered_elements() if e != stratify_by)
    pairs: list[tuple[str, str]] = []
    for pred_ex, refs in zip(pred_list, ref_tuples, strict=True):
        pred_by_id: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for pt in pred_ex.tuples:
            pred_by_id[tuple_key(pt, identity)].append(element_key(pt, stratify_by))
        ref_by_id: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for rt in refs:
            ref_by_id[tuple_key(rt, identity)].append(element_key(rt, stratify_by))
        for key, pred_labels in pred_by_id.items():
            if key in ref_by_id:
                pairs.extend(_align_labels(pred_labels, ref_by_id[key]))

    pairs = _stratified_sample(pairs, n_sample, seed)
    labels = tuple(sorted({label for pair in pairs for label in pair}))
    model_labels = [p[0] for p in pairs]
    ref_labels = [p[1] for p in pairs]
    kappa, se, po = _cohens_kappa(model_labels, ref_labels, labels)
    per_label = {}
    for label in labels:
        bin_a = [x == label for x in model_labels]
        bin_b = [y == label for y in ref_labels]
        per_label[label] = _cohens_kappa(bin_a, bin_b, [True, False])[0]
    return ValidationReport(
        element=stratify_by,
        n=len(pairs),
        kappa=kappa,
        kappa_se=se,
        observed_agreement=po,
        per_label=per_label,
        labels=labels,
    )
