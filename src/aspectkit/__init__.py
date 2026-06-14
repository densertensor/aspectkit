"""aspectkit: a configurable, evaluation-centric ABSA framework.

Quickstart::

    from aspectkit import ABSA

    absa = ABSA(task="acos", backend="llm", model="openai:gpt-4o-mini")
    absa.fit(train_examples)                       # optional few-shot
    preds = absa.predict(["The pasta was great but we waited forever."])
    report = absa.evaluate(test_examples, lenient=True)
    summary = absa.summarize(corpus, by="category")

The public surface is intentionally small:

* :class:`ABSA` — the pipeline facade;
* the canonical schema (:class:`ABSAExample`, :class:`SentimentTuple`,
  :class:`Span`, :data:`IMPLICIT`);
* task views (:func:`get_task`, :data:`TASKS`);
* loaders (:func:`load_examples` and the ``aspectkit.io`` readers);
* evaluation (:func:`evaluate`, :class:`EvaluationReport`);
* aggregation (:func:`summarize`, :class:`AspectSummary`);
* connectors (:mod:`aspectkit.llm`, :func:`resolve_llm`).
"""

from __future__ import annotations

from aspectkit.aggregate import AspectSummary, summarize, summary_to_frame
from aspectkit.backends import Backend, LLMBackend, PairClassifierBackend, Seq2SeqBackend
from aspectkit.evaluation import EvaluationReport, evaluate
from aspectkit.io import load_examples
from aspectkit.llm import resolve_llm
from aspectkit.pipeline import ABSA
from aspectkit.schema import (
    IMPLICIT,
    POLARITIES,
    ABSAExample,
    Implicit,
    SentimentTuple,
    Span,
    is_implicit,
)
from aspectkit.tasks import ELEMENTS, TASKS, Task, get_task

__version__ = "0.1.0"

__all__ = [
    "ABSA",
    "ELEMENTS",
    "IMPLICIT",
    "POLARITIES",
    "TASKS",
    "ABSAExample",
    "AspectSummary",
    "Backend",
    "EvaluationReport",
    "Implicit",
    "LLMBackend",
    "PairClassifierBackend",
    "SentimentTuple",
    "Seq2SeqBackend",
    "Span",
    "Task",
    "__version__",
    "evaluate",
    "get_task",
    "is_implicit",
    "load_examples",
    "resolve_llm",
    "summarize",
    "summary_to_frame",
]
