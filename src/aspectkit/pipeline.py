"""The :class:`ABSA` facade: one object from text to report.

Mirrors the ergonomics of modern text-mining frameworks: pick a task and
a backend, get ``fit`` / ``predict`` / ``evaluate`` / ``summarize`` with
sensible defaults, and swap components without touching the rest of the
pipeline::

    from aspectkit import ABSA

    absa = ABSA(task="acos", backend="llm", model="openai:gpt-4o-mini",
                categories=["FOOD#QUALITY", "SERVICE#GENERAL"])
    preds = absa.predict(["The pasta was great but we waited forever."])
    report = absa.evaluate(test_examples, lenient=True)
    summary = absa.summarize(corpus, by="category")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal, overload

from aspectkit.aggregate import AspectSummary
from aspectkit.aggregate import summarize as _summarize
from aspectkit.backends.base import Backend
from aspectkit.backends.llm import LLMBackend
from aspectkit.backends.pair import PairClassifierBackend
from aspectkit.backends.seq2seq import Seq2SeqBackend
from aspectkit.evaluation import EvaluationReport
from aspectkit.evaluation import evaluate as _evaluate
from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = ["ABSA"]

_BACKENDS: dict[str, str] = {
    "llm": "prompted chat model (LLMBackend)",
    "pair": "fine-tuned (text, aspect) cross-encoder (PairClassifierBackend)",
    "seq2seq": "fine-tunable generative extraction (Seq2SeqBackend)",
}


class ABSA:
    """Aspect-based sentiment analysis pipeline.

    Args:
        task: Task view name (``"acos"``, ``"aste"``, ``"tasd"``,
            ``"e2e"``, ``"atsc"``, ...).  Defaults to ``"acos"`` — the
            complete quadruple — when a new backend is constructed; when
            a ready :class:`~aspectkit.backends.Backend` is passed, the
            task comes from the backend.
        backend: ``"llm"``, ``"pair"``, ``"seq2seq"``, or a constructed
            backend instance (the extension point for custom
            strategies).
        model: Model spec for the chosen backend.  For ``"llm"``:
            anything :func:`~aspectkit.llm.resolve_llm` accepts —
            ``"openai:gpt-4o-mini"``, ``"anthropic:claude-opus-4-8"``,
            ``"deepseek:deepseek-chat"``, ``"vllm:<served-model>"``, a
            connector instance, a ``transformers`` pipeline, or a
            ``(model, tokenizer)`` pair.  For ``"pair"``: a Hugging Face
            checkpoint id (default: the DeBERTa-v3 ABSA checkpoint).
            For ``"seq2seq"``: a T5/BART-family checkpoint id or a saved
            fine-tuned directory (default ``"t5-base"`` — which must be
            fine-tuned with :meth:`fit` before predictions mean
            anything).
        categories: Category inventory for category-bearing tasks.
        polarities: Allowed polarity labels (3-way scheme by default).
        **backend_kwargs: Forwarded to the backend constructor (e.g.
            ``n_exemplars=4``, ``api_key=...``, ``device="cuda"``).

    Raises:
        ValueError: For unknown task/backend names, missing model specs,
            or a ``task`` that contradicts a passed backend instance.
    """

    def __init__(
        self,
        task: str | Task | None = None,
        backend: str | Backend = "llm",
        model: Any = None,
        *,
        categories: Sequence[str] | None = None,
        polarities: Sequence[str] = ("positive", "negative", "neutral"),
        **backend_kwargs: Any,
    ) -> None:
        if isinstance(backend, Backend):
            if task is not None and get_task(task).name != backend.task.name:
                raise ValueError(
                    f"task {get_task(task).name!r} contradicts the backend's task "
                    f"{backend.task.name!r}; omit task= when passing a backend instance"
                )
            if model is not None:
                raise ValueError("model= is ignored when passing a backend instance")
            self.backend = backend
            return

        if backend == "llm":
            if model is None:
                raise ValueError(
                    "the 'llm' backend needs a model: e.g. "
                    "ABSA(task='acos', backend='llm', model='openai:gpt-4o-mini'), "
                    "or pass a connector / transformers pipeline object"
                )
            self.backend = LLMBackend(
                model,
                task=task if task is not None else "acos",
                categories=categories,
                polarities=polarities,
                **backend_kwargs,
            )
        elif backend == "pair":
            resolved = get_task(task) if task is not None else get_task("atsc")
            if resolved.name != "atsc":
                raise ValueError(
                    f"the 'pair' backend implements ATSC only, not {resolved.name!r}; "
                    "use backend='llm' or backend='seq2seq' for extraction tasks"
                )
            if model is None:
                self.backend = PairClassifierBackend(**backend_kwargs)
            else:
                self.backend = PairClassifierBackend(model, **backend_kwargs)
        elif backend == "seq2seq":
            seq2seq_kwargs: dict[str, Any] = {
                "task": task if task is not None else "acos",
                "categories": categories,
                **backend_kwargs,
            }
            if model is None:
                self.backend = Seq2SeqBackend(**seq2seq_kwargs)
            else:
                self.backend = Seq2SeqBackend(model, **seq2seq_kwargs)
        else:
            options = "; ".join(f"'{name}' = {desc}" for name, desc in _BACKENDS.items())
            raise ValueError(f"unknown backend {backend!r}; available: {options}")

    # ------------------------------------------------------------ properties

    @property
    def task(self) -> Task:
        """The task view the pipeline is configured for."""
        return self.backend.task

    # ------------------------------------------------------------------- API

    def fit(self, examples: Sequence[ABSAExample]) -> ABSA:
        """Adapt the backend to labelled examples (chainable).

        For the LLM backend this selects few-shot exemplars (seeded,
        reproducible); other backends define their own behaviour.
        Optional: prompted and pretrained backends work without it.
        """
        self.backend.fit(examples)
        return self

    @overload
    def predict(
        self,
        inputs: str | ABSAExample | Sequence[str | ABSAExample],
        *,
        return_confidence: Literal[False] = False,
    ) -> list[SentimentTuple] | list[list[SentimentTuple]]: ...

    @overload
    def predict(
        self,
        inputs: str | ABSAExample | Sequence[str | ABSAExample],
        *,
        return_confidence: Literal[True],
    ) -> list[tuple[SentimentTuple, float]] | list[list[tuple[SentimentTuple, float]]]: ...

    def predict(
        self,
        inputs: str | ABSAExample | Sequence[str | ABSAExample],
        *,
        return_confidence: bool = False,
    ) -> (
        list[SentimentTuple]
        | list[list[SentimentTuple]]
        | list[tuple[SentimentTuple, float]]
        | list[list[tuple[SentimentTuple, float]]]
    ):
        """Predict sentiment tuples.

        Args:
            inputs: A single text/example, or a sequence of them.  Tasks
                with given elements (e.g. ATSC) need
                :class:`~aspectkit.schema.ABSAExample` inputs carrying
                the targets.
            return_confidence: When ``True``, pair each predicted tuple with a
                confidence in ``[0, 1]``.  Supported by the LLM backend (via
                ``n_samples`` self-consistency) and the pair backend (softmax
                probability); other backends raise :class:`NotImplementedError`.

        Returns:
            For a single input, one list of tuples; for a sequence, one
            list per input.  With ``return_confidence`` each tuple becomes a
            ``(tuple, confidence)`` pair.
        """
        single = isinstance(inputs, (str, ABSAExample))
        batch: Sequence[str | ABSAExample] = (
            [inputs] if isinstance(inputs, (str, ABSAExample)) else list(inputs)
        )
        examples = [
            item if isinstance(item, ABSAExample) else ABSAExample(text=item) for item in batch
        ]
        if return_confidence:
            if not isinstance(self.backend, (LLMBackend, PairClassifierBackend)):
                raise NotImplementedError(
                    f"return_confidence is not supported by {type(self.backend).__name__}; "
                    "use the LLM or pair backend."
                )
            scored = self.backend.predict(examples, return_confidence=True)
            return scored[0] if single else scored
        predictions = self.backend.predict(examples)
        return predictions[0] if single else predictions

    def evaluate(
        self,
        examples: Sequence[ABSAExample],
        *,
        lenient: bool = False,
        overlap_threshold: float = 0.5,
        predictions: Sequence[Sequence[SentimentTuple]] | None = None,
    ) -> EvaluationReport:
        """Predict on gold examples and score under the task's protocol.

        Gold annotations are never shown to the backend: extraction
        backends receive bare texts, classification backends receive
        only the given elements (targets) with labels stripped.

        Args:
            examples: Gold-annotated examples.
            lenient: Also report token-overlap matching alongside exact
                match (extraction tasks).
            overlap_threshold: Minimum token-IoU for lenient matching.
            predictions: Pre-computed predictions to score instead of
                running the backend (useful for cached or external runs).

        Returns:
            An :class:`~aspectkit.evaluation.EvaluationReport`.
        """
        if predictions is None:
            predictions = self.backend.predict(self._prediction_inputs(examples))
        return _evaluate(
            examples,
            predictions,
            self.task,
            lenient=lenient,
            overlap_threshold=overlap_threshold,
        )

    def summarize(
        self,
        corpus: Sequence[str | ABSAExample],
        *,
        by: Literal["aspect", "category"] = "aspect",
        min_mentions: int = 1,
        max_quotes: int = 3,
        predictions: Sequence[Sequence[SentimentTuple]] | None = None,
    ) -> list[AspectSummary]:
        """Predict over a corpus and aggregate per aspect or category.

        The corpus-level rollup — per-aspect sentiment distributions,
        net scores, and representative quotes — is the end product
        applied ABSA was invented for.

        Args:
            corpus: Texts (or examples) to analyse.
            by: Group by surface ``"aspect"`` or by ``"category"``.
            min_mentions: Drop groups with fewer mentions.
            max_quotes: Representative texts kept per polarity.
            predictions: Pre-computed predictions to aggregate instead
                of running the backend.

        Returns:
            :class:`~aspectkit.aggregate.AspectSummary` list, most
            mentioned first.
        """
        examples = [
            item if isinstance(item, ABSAExample) else ABSAExample(text=item) for item in corpus
        ]
        if predictions is None:
            predictions = self.backend.predict(examples)
        return _summarize(
            [example.text for example in examples],
            predictions,
            by=by,
            min_mentions=min_mentions,
            max_quotes=max_quotes,
        )

    # -------------------------------------------------------------- internals

    def _prediction_inputs(self, examples: Sequence[ABSAExample]) -> list[ABSAExample]:
        """Strip everything the backend must not see during evaluation."""
        if not self.task.given:
            return [ABSAExample(text=e.text, id=e.id, meta=e.meta) for e in examples]
        # Keep only the given elements on each tuple; blank the rest.
        cleared: dict[str, Any] = {}
        for element in ("category", "opinion", "polarity"):
            if element not in self.task.given:
                cleared[element] = None
        return [
            ABSAExample(
                text=e.text,
                tuples=[replace(t, **cleared) for t in e.tuples],
                id=e.id,
                meta=e.meta,
            )
            for e in examples
        ]

    def __repr__(self) -> str:
        return f"ABSA(task={self.task.name!r}, backend={self.backend!r})"
