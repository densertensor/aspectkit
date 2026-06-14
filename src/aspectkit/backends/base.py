"""Backend interface.

A backend implements one prediction strategy (prompted LLM, fine-tuned
cross-encoder, ...) for one task view.  Backends consume and produce the
canonical schema only, so they are interchangeable behind the
:class:`~aspectkit.ABSA` facade and directly comparable under the same
evaluation protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task

__all__ = ["Backend"]


class Backend(ABC):
    """Abstract prediction backend.

    Attributes:
        task: The task view this backend instance is configured for.
    """

    task: Task

    def fit(self, examples: Sequence[ABSAExample]) -> Backend:
        """Adapt the backend to labelled examples.

        The default is a no-op for backends that need no adaptation.
        Backends that learn (or select few-shot exemplars) override
        this.  Returns ``self`` for chaining.
        """
        return self

    @abstractmethod
    def predict(self, examples: Sequence[ABSAExample]) -> list[list[SentimentTuple]]:
        """Predict sentiment tuples for each example.

        Args:
            examples: Inputs.  For tasks with *given* elements (e.g.
                ATSC), each example's ``tuples`` must carry those
                elements; for pure extraction tasks the texts suffice.

        Returns:
            One list of predicted tuples per input example, in order.
        """

    def _require_given_elements(self, examples: Sequence[ABSAExample]) -> None:
        """Validate that classification-style inputs carry their targets."""
        if not self.task.given:
            return
        for i, example in enumerate(examples):
            if not example.tuples:
                raise ValueError(
                    f"task '{self.task.name}' needs the given elements "
                    f"({', '.join(sorted(self.task.given))}) on each example, but "
                    f"examples[{i}] has no tuples. Construct examples such as "
                    "ABSAExample(text=..., tuples=[SentimentTuple(aspect=Span('...'))])."
                )
