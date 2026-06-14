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

    def to_methods_text(
        self, *, model_name: str, dataset_name: str, template: str | None = None
    ) -> str:
        """Render a Methods-section paragraph from this report's own numbers.

        A copy-pasteable description for a paper's evaluation section — purely
        formatting, no new computation, and **no claim about what the model
        is**: the caller supplies *model_name* and *dataset_name* verbatim.

        Args:
            model_name: How to refer to the evaluated system (e.g.
                ``"GPT-4o-mini (zero-shot)"``).  Used as given.
            dataset_name: The evaluation set (e.g. ``"Rest16 (ACOS)"``).
            template: Optional ``str.format`` template that overrides the
                built-in wording; it receives the fields ``model_name``,
                ``dataset_name``, ``task``, ``kind``, ``n_examples``,
                ``elements``, ``precision``, ``recall``, ``f1``, ``n_pred``,
                ``n_gold``, ``lenient_f1``, ``accuracy``, ``macro_f1`` (numeric
                fields are pre-formatted strings; metrics absent for this
                report's kind are ``"n/a"``).
        """
        exact = self.exact
        cls = self.classification
        fields: dict[str, object] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "task": self.task,
            "kind": self.kind,
            "n_examples": self.n_examples,
            "elements": ", ".join(self.elements),
            "precision": f"{exact.precision:.3f}" if exact else "n/a",
            "recall": f"{exact.recall:.3f}" if exact else "n/a",
            "f1": f"{exact.f1:.3f}" if exact else "n/a",
            "n_pred": exact.n_pred if exact else 0,
            "n_gold": exact.n_gold if exact else 0,
            "lenient_f1": f"{self.lenient.f1:.3f}" if self.lenient else "n/a",
            "accuracy": f"{cls.accuracy:.3f}" if cls else "n/a",
            "macro_f1": f"{cls.macro_f1:.3f}" if cls else "n/a",
        }
        if template is not None:
            return template.format(**fields)

        if self.kind == "classification" and cls is not None:
            return (
                f"We evaluate {model_name} on {dataset_name} for the {self.task} task "
                f"(polarity classification, {self.n_examples} examples). It attains "
                f"accuracy {fields['accuracy']} and macro-F1 {fields['macro_f1']}."
            )

        parts = [
            f"We evaluate {model_name} on {dataset_name} for the {self.task} task "
            f"({self.n_examples} examples). Predicted tuples are scored by exact match "
            f"over ({fields['elements']}): a tuple is correct only when every element "
            f"matches the gold annotation."
        ]
        if exact is not None:
            parts.append(
                f" {model_name} attains precision {fields['precision']}, recall "
                f"{fields['recall']}, and micro-F1 {fields['f1']} (over {fields['n_pred']} "
                f"predicted and {fields['n_gold']} gold tuples)."
            )
        if self.lenient is not None:
            parts.append(
                f" A lenient token-overlap F1 of {fields['lenient_f1']} is reported "
                "alongside exact match."
            )
        if self.per_element:
            per = ", ".join(f"{el} {prf.f1:.3f}" for el, prf in self.per_element.items())
            parts.append(f" Per-element exact-match F1: {per}.")
        return "".join(parts)

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
