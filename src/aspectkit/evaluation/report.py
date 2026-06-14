"""Evaluation report container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from aspectkit.evaluation.metrics import PRF, ClassificationScores

__all__ = ["EvaluationReport"]


@dataclass(frozen=True)
class EvaluationReport:
    """Result of evaluating predictions against gold annotations.

    Extraction tasks report micro tuple P/R/F1 under exact matching
    (``exact``), optionally accompanied by a lenient overlap score
    (``lenient``) — reported alongside, never instead of, exact match.
    Classification tasks (e.g. ATSC) report accuracy and macro-F1.

    Attributes:
        task: Canonical task name.
        kind: ``"extraction"`` or ``"classification"``.
        n_examples: Number of evaluated examples.
        exact: Exact-match P/R/F1 (extraction tasks).
        lenient: Overlap-match P/R/F1, if requested (extraction tasks).
        per_element: Exact-match P/R/F1 of each element scored in
            isolation (multi-element extraction tasks) — the diagnostic
            for *which* element drives tuple-level misses.
        classification: Accuracy / macro-F1 / per-label scores
            (classification tasks).
        elements: The sentiment elements compared.
    """

    task: str
    kind: Literal["extraction", "classification"]
    n_examples: int
    elements: tuple[str, ...]
    exact: PRF | None = None
    lenient: PRF | None = None
    per_element: dict[str, PRF] | None = None
    classification: ClassificationScores | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-dict form, suitable for JSON serialisation."""
        return asdict(self)

    def __str__(self) -> str:
        lines = [f"EvaluationReport(task={self.task}, n_examples={self.n_examples})"]
        if self.exact is not None:
            lines.append(
                f"  exact match   P={self.exact.precision:.4f}  "
                f"R={self.exact.recall:.4f}  F1={self.exact.f1:.4f}  "
                f"(pred={self.exact.n_pred}, gold={self.exact.n_gold})"
            )
        if self.lenient is not None:
            lines.append(
                f"  lenient       P={self.lenient.precision:.4f}  "
                f"R={self.lenient.recall:.4f}  F1={self.lenient.f1:.4f}"
            )
        if self.per_element:
            lines.append("  by element (exact):")
            for element, prf in self.per_element.items():
                lines.append(
                    f"    {element:<9} P={prf.precision:.4f}  R={prf.recall:.4f}  F1={prf.f1:.4f}"
                )
        if self.classification is not None:
            c = self.classification
            lines.append(f"  accuracy={c.accuracy:.4f}  macro_f1={c.macro_f1:.4f}  n={c.n}")
            for label, scores in sorted(c.per_label.items()):
                lines.append(
                    f"    {label:<10} P={scores.precision:.4f}  R={scores.recall:.4f}  "
                    f"F1={scores.f1:.4f}  support={scores.support}"
                )
        return "\n".join(lines)
