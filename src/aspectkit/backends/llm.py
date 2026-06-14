"""Prompted-LLM backend: structured extraction with any chat model.

This is the cold-start path: full tuple views (including ACOS quads)
with zero training, via JSON-constrained prompting — the pattern used
in production review-mining pipelines.  The benchmark literature is
unambiguous that prompted LLMs trail fine-tuned models on exact-match
tuple extraction, so treat this backend as a strong baseline and
**validate it on a labelled sample with** :meth:`aspectkit.ABSA.evaluate`
**before trusting its output**.
"""

from __future__ import annotations

import json
import random
import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal

from aspectkit.backends.base import Backend
from aspectkit.backends.parsing import extract_json, payload_to_polarity, payload_to_tuples
from aspectkit.backends.prompts import (
    build_classification_schema,
    build_extraction_schema,
    classification_messages,
    extraction_messages,
)
from aspectkit.exceptions import LLMError, ParseError
from aspectkit.llm.base import ChatLLM, Message
from aspectkit.llm.registry import resolve_llm
from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = ["LLMBackend"]

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed. Respond again with ONLY the "
    "requested JSON object — no prose, no code fences, no explanations."
)


class LLMBackend(Backend):
    """Predict sentiment tuples by prompting a chat model.

    Works with any connector accepted by
    :func:`~aspectkit.llm.registry.resolve_llm`: hosted APIs, local
    OpenAI-compatible servers, or in-process ``transformers`` objects.

    Args:
        llm: Connector or model spec (e.g. ``"openai:gpt-4o-mini"``, an
            ``AnthropicChat`` instance, a ``transformers`` pipeline).
        task: Task view, e.g. ``"acos"`` (default), ``"aste"``, ``"atsc"``.
        categories: Category inventory for tasks that predict categories.
            Strongly recommended: an open category space is rarely what a
            study design intends.
        polarities: Allowed polarity labels.  The 3-way scheme is the
            default; add ``"conflict"`` for SemEval-2014-style data.
        n_exemplars: Maximum number of few-shot exemplars sampled by
            :meth:`fit`.
        seed: Random seed for exemplar sampling (reproducibility).
        max_tokens: Generation budget per call.
        use_schema: Pass a JSON schema to the connector so providers
            with native structured output enforce the format.
        max_repairs: How many times to re-prompt after an unparseable
            reply before giving up on the item.
        on_error: ``"raise"`` (default) propagates failures;
            ``"skip"`` records an empty prediction for the failing item
            and warns — useful for long corpus runs where one bad
            response must not kill the job.
        concurrency: Examples predicted in parallel (thread pool).  The
            default of 1 keeps calls sequential; raise it for
            corpus-scale runs against hosted APIs (mind your rate
            limits).  Prediction order is always preserved, and with
            ``on_error="raise"`` the first failure propagates.  The SDK
            clients behind the bundled connectors are thread-safe.
        **llm_kwargs: Forwarded to the connector constructor when *llm*
            is a spec rather than an instance.
    """

    def __init__(
        self,
        llm: Any,
        task: str | Task = "acos",
        *,
        categories: Sequence[str] | None = None,
        polarities: Sequence[str] = ("positive", "negative", "neutral"),
        n_exemplars: int = 8,
        seed: int = 42,
        max_tokens: int = 1024,
        use_schema: bool = True,
        max_repairs: int = 1,
        on_error: Literal["raise", "skip"] = "raise",
        concurrency: int = 1,
        **llm_kwargs: Any,
    ) -> None:
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self.llm: ChatLLM = resolve_llm(llm, **llm_kwargs)
        self.task = get_task(task)
        self.categories = list(categories) if categories else None
        self.polarities = tuple(polarities)
        self.n_exemplars = n_exemplars
        self.seed = seed
        self.max_tokens = max_tokens
        self.use_schema = use_schema
        self.max_repairs = max_repairs
        self.on_error = on_error
        self.concurrency = concurrency
        self._exemplars: list[ABSAExample] = []
        self._cls_exemplars: list[tuple[str, str | None, str]] = []
        #: Counters from the most recent :meth:`predict` call.
        self.diagnostics: dict[str, int] = {}
        self._diagnostics_lock = threading.Lock()

    # ------------------------------------------------------------------ fit

    def fit(self, examples: Sequence[ABSAExample]) -> LLMBackend:
        """Select few-shot exemplars from labelled examples.

        Sampling is seeded for reproducibility.  For classification
        views the exemplars are flattened ``(text, aspect, polarity)``
        triples; for extraction views whole examples are used (examples
        with no opinions included — they teach the empty-output case).
        """
        rng = random.Random(self.seed)
        if self.task.is_classification:
            triples = [
                (example.text, t.aspect_text, t.polarity)
                for example in examples
                for t in example.tuples
                if t.polarity is not None
            ]
            self._cls_exemplars = (
                triples
                if len(triples) <= self.n_exemplars
                else rng.sample(triples, self.n_exemplars)
            )
        else:
            pool = list(examples)
            self._exemplars = (
                pool if len(pool) <= self.n_exemplars else rng.sample(pool, self.n_exemplars)
            )
        return self

    # -------------------------------------------------------------- predict

    def predict(self, examples: Sequence[ABSAExample]) -> list[list[SentimentTuple]]:
        self._require_given_elements(examples)
        self.diagnostics = {"calls": 0, "repairs": 0, "dropped_items": 0, "failed_examples": 0}
        worker = self._classify_example if self.task.is_classification else self._extract_example
        if self.concurrency == 1 or len(examples) <= 1:
            return [worker(example) for example in examples]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(examples))) as pool:
            return list(pool.map(worker, examples))

    # ------------------------------------------------------------- internals

    def _bump(self, key: str, amount: int = 1) -> None:
        """Thread-safe diagnostics increment (predict may run concurrently)."""
        with self._diagnostics_lock:
            self.diagnostics[key] += amount

    def _complete_with_repairs(
        self, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> Any:
        """Call the model and decode JSON, re-prompting on parse failure."""
        conversation = list(messages)
        last_error: ParseError | None = None
        for attempt in range(self.max_repairs + 1):
            self._bump("calls")
            if attempt > 0:
                self._bump("repairs")
            reply = self.llm.complete(
                conversation, max_tokens=self.max_tokens, json_schema=json_schema
            )
            try:
                return extract_json(reply)
            except ParseError as exc:
                last_error = exc
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": _REPAIR_INSTRUCTION},
                ]
        assert last_error is not None
        raise last_error

    def _handle_failure(self, example: ABSAExample, exc: Exception) -> list[SentimentTuple]:
        if self.on_error == "raise":
            raise exc
        self._bump("failed_examples")
        warnings.warn(
            f"prediction failed for example {example.id or example.text[:60]!r}: {exc}",
            stacklevel=2,
        )
        return []

    def _extract_example(self, example: ABSAExample) -> list[SentimentTuple]:
        schema = (
            build_extraction_schema(self.task, self.categories, self.polarities)
            if self.use_schema
            else None
        )
        messages = extraction_messages(
            example.text, self.task, self.categories, self.polarities, self._exemplars
        )
        try:
            payload = self._complete_with_repairs(messages, schema)
            tuples, problems = payload_to_tuples(
                payload, self.task, example.text, categories=self.categories
            )
        except (LLMError, ParseError) as exc:
            return self._handle_failure(example, exc)
        if problems:
            self._bump("dropped_items", len(problems))
            warnings.warn(
                f"dropped {len(problems)} malformed item(s) for example "
                f"{example.id or example.text[:60]!r}: {'; '.join(problems)}",
                stacklevel=2,
            )
        return self._filter_polarities(tuples)

    def _filter_polarities(self, tuples: list[SentimentTuple]) -> list[SentimentTuple]:
        """Drop tuples whose polarity falls outside the configured scheme
        (a schema-less model may emit e.g. 'conflict' uninvited)."""
        if "polarity" not in self.task.predicted:
            return tuples
        kept = [t for t in tuples if t.polarity is None or t.polarity in self.polarities]
        if len(kept) < len(tuples):
            self._bump("dropped_items", len(tuples) - len(kept))
        return kept

    def _classify_example(self, example: ABSAExample) -> list[SentimentTuple]:
        predictions: list[SentimentTuple] = []
        schema = build_classification_schema(self.polarities) if self.use_schema else None
        for target in example.tuples:
            messages = classification_messages(
                example.text, target.aspect_text, self.polarities, self._cls_exemplars
            )
            try:
                payload = self._complete_with_repairs(messages, schema)
                polarity = payload_to_polarity(payload)
                if polarity not in self.polarities:
                    raise ParseError(
                        f"model predicted {polarity!r}, outside the configured "
                        f"polarity scheme {self.polarities}"
                    )
            except (LLMError, ParseError) as exc:
                self._handle_failure(example, exc)
                continue
            predictions.append(replace(target, polarity=polarity))
        return predictions

    def __repr__(self) -> str:
        return (
            f"LLMBackend(task={self.task.name!r}, llm={self.llm.name}, "
            f"exemplars={len(self._exemplars) + len(self._cls_exemplars)})"
        )

    @staticmethod
    def describe_prompt(
        task: str | Task = "acos",
        categories: Sequence[str] | None = None,
        polarities: Sequence[str] = ("positive", "negative", "neutral"),
    ) -> str:
        """Render the system prompt and schema for inspection.

        Researchers should be able to report exactly what a prompted
        baseline was asked to do; this returns the system prompt and the
        JSON schema for the given configuration.
        """
        resolved = get_task(task)
        if resolved.is_classification:
            messages = classification_messages("<text>", "<aspect>", polarities)
            schema = build_classification_schema(polarities)
        else:
            messages = extraction_messages("<text>", resolved, categories, polarities)
            schema = build_extraction_schema(resolved, categories, polarities)
        return messages[0]["content"] + "\n\n--- JSON schema ---\n" + json.dumps(schema, indent=2)
