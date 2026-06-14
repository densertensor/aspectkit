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
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal, overload

from aspectkit.backends.base import Backend
from aspectkit.backends.parsing import extract_json, payload_to_polarity, payload_to_tuples
from aspectkit.backends.prompts import (
    build_classification_schema,
    build_extraction_schema,
    classification_messages,
    extraction_messages,
)
from aspectkit.evaluation.matching import tuple_key
from aspectkit.exceptions import LLMError, ParseError
from aspectkit.llm.base import ChatLLM, Message
from aspectkit.llm.exemplars import ExemplarPool, ExemplarSelector, KNNSelector
from aspectkit.llm.registry import resolve_llm
from aspectkit.llm.wrappers import CachingChat, RetryingChat
from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = ["LLMBackend"]

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed. Respond again with ONLY the "
    "requested JSON object — no prose, no code fences, no explanations."
)


def _apparent_temperature(llm: ChatLLM) -> float | None:
    """Best-effort sampling temperature of a (possibly wrapped) connector.

    Unwraps composable wrappers to the base connector and reads its
    ``_temperature``; returns ``None`` when it cannot be determined.
    """
    while (inner := getattr(llm, "inner", None)) is not None:
        llm = inner
    temp = getattr(llm, "_temperature", None)
    return temp if isinstance(temp, (int, float)) and not isinstance(temp, bool) else None


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
        exemplar_selection: How :meth:`fit`-supplied exemplars are chosen
            per input.  ``"random"`` (default) reuses one seeded sample for
            every input (the historical behaviour); ``"none"`` is zero-shot;
            ``"knn"`` retrieves the most similar exemplars per input by a
            dependency-free TF-IDF cosine.  Pass an
            :class:`~aspectkit.llm.exemplars.ExemplarSelector` (e.g.
            ``KNNSelector(encode_fn=...)``) for full control.
        instructions: Extra guidance appended to the system prompt (the JSON
            contract is preserved).  Use for domain notes or label
            definitions without rewriting the prompt.
        system_prompt_fn: ``(task, categories, polarities) -> str`` to replace
            the built-in system prompt entirely; ``instructions`` is still
            appended.  For full control over the task framing.
        max_tokens: Generation budget per call.
        use_schema: Pass a JSON schema to the connector so providers
            with native structured output enforce the format.
        max_repairs: How many times to re-prompt after an unparseable
            reply before giving up on the item.
        n_samples: Self-consistency: sample the model this many times per
            input and aggregate.  ``1`` (default) calls once and assigns
            confidence ``1.0``.  Needs a sampling temperature > 0 to vary —
            a warning is emitted if the connector's temperature appears 0.
        vote: How the ``n_samples`` extractions are combined when
            ``n_samples > 1``: ``"majority"`` keeps tuples found in more than
            half the samples, ``"union"`` keeps any tuple found at least
            once.  Classification always takes the majority polarity per
            target.  Confidence is the fraction of samples a tuple appeared in.
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
        retry: Wrap the connector in
            :class:`~aspectkit.llm.wrappers.RetryingChat` for configurable
            rate-limit backoff.  ``True`` uses its defaults; a dict passes
            keyword arguments (e.g. ``{"max_retries": 10}``).  Compose
            wrappers manually for finer control.
        cache_dir: Wrap the connector in
            :class:`~aspectkit.llm.wrappers.CachingChat` with this
            directory, so repeated inputs are served from disk.
        on_progress: Optional ``(completed, total) -> None`` callback
            invoked as each example finishes (called from worker threads
            under concurrency, so it must be thread-safe).
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
        exemplar_selection: Literal["none", "random", "knn"] | ExemplarSelector = "random",
        instructions: str | None = None,
        system_prompt_fn: Callable[[Task, Sequence[str] | None, Sequence[str]], str] | None = None,
        max_tokens: int = 1024,
        use_schema: bool = True,
        max_repairs: int = 1,
        n_samples: int = 1,
        vote: Literal["majority", "union"] = "majority",
        on_error: Literal["raise", "skip"] = "raise",
        concurrency: int = 1,
        retry: bool | dict[str, Any] = False,
        cache_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if vote not in ("majority", "union"):
            raise ValueError(f"vote must be 'majority' or 'union', got {vote!r}")
        if isinstance(exemplar_selection, str):
            if exemplar_selection not in ("none", "random", "knn"):
                raise ValueError(
                    "exemplar_selection must be 'none', 'random', 'knn', or an "
                    f"ExemplarSelector instance, got {exemplar_selection!r}"
                )
        elif not isinstance(exemplar_selection, ExemplarSelector):
            raise TypeError(
                "exemplar_selection must be a str or ExemplarSelector, got "
                f"{type(exemplar_selection).__name__}"
            )
        self.llm: ChatLLM = resolve_llm(llm, **llm_kwargs)
        # Optional composable wrappers (innermost first): cache, then retry.
        if cache_dir is not None:
            self.llm = CachingChat(self.llm, cache_dir=cache_dir)
        if retry:
            self.llm = RetryingChat(self.llm, **(retry if isinstance(retry, dict) else {}))
        self.on_progress = on_progress
        self.task = get_task(task)
        self.categories = list(categories) if categories else None
        self.polarities = tuple(polarities)
        self.n_exemplars = n_exemplars
        self.seed = seed
        self.exemplar_selection = exemplar_selection
        # "none"/"random" are served from the static fit() sample below; "knn"
        # and custom selectors retrieve per-input from the fitted pool.
        self._selector: ExemplarSelector | None = (
            (KNNSelector() if exemplar_selection == "knn" else None)
            if isinstance(exemplar_selection, str)
            else exemplar_selection
        )
        self.instructions = instructions
        self.system_prompt_fn = system_prompt_fn
        # The override depends only on (task, categories, polarities), so build
        # it once rather than per example.
        self._system_override = (
            system_prompt_fn(self.task, self.categories, self.polarities)
            if system_prompt_fn is not None
            else None
        )
        self.max_tokens = max_tokens
        self.use_schema = use_schema
        self.max_repairs = max_repairs
        self.n_samples = n_samples
        self.vote = vote
        self.on_error = on_error
        self.concurrency = concurrency
        self._exemplars: list[ABSAExample] = []
        self._cls_exemplars: list[tuple[str, str | None, str]] = []
        self._extraction_pool = ExemplarPool([], [])
        self._cls_pool = ExemplarPool([], [])
        #: Counters from the most recent :meth:`predict` call.
        self.diagnostics: dict[str, int] = {}
        self._diagnostics_lock = threading.Lock()
        if n_samples > 1 and _apparent_temperature(self.llm) == 0:
            warnings.warn(
                "n_samples > 1 enables self-consistency, but the connector's sampling "
                "temperature appears to be 0 (deterministic), so samples will be "
                "identical; construct the connector with a temperature > 0.",
                stacklevel=2,
            )

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
            self._cls_pool = ExemplarPool(triples, [text for text, _, _ in triples])
        else:
            pool = list(examples)
            self._exemplars = (
                pool if len(pool) <= self.n_exemplars else rng.sample(pool, self.n_exemplars)
            )
            self._extraction_pool = ExemplarPool(pool, [example.text for example in pool])
        return self

    def _select_extraction(self, query: str) -> list[ABSAExample]:
        """Exemplars for one extraction input, per ``exemplar_selection``."""
        if self._selector is None:  # "none" or "random"
            return self._exemplars if self.exemplar_selection == "random" else []
        return self._selector.select(self._extraction_pool, query, self.n_exemplars)

    def _select_classification(self, query: str) -> list[tuple[str, str | None, str]]:
        """Classification exemplars for one input, per ``exemplar_selection``."""
        if self._selector is None:  # "none" or "random"
            return self._cls_exemplars if self.exemplar_selection == "random" else []
        return self._selector.select(self._cls_pool, query, self.n_exemplars)

    # -------------------------------------------------------------- predict

    @overload
    def predict(
        self, examples: Sequence[ABSAExample], *, return_confidence: Literal[False] = False
    ) -> list[list[SentimentTuple]]: ...

    @overload
    def predict(
        self, examples: Sequence[ABSAExample], *, return_confidence: Literal[True]
    ) -> list[list[tuple[SentimentTuple, float]]]: ...

    def predict(
        self, examples: Sequence[ABSAExample], *, return_confidence: bool = False
    ) -> list[list[SentimentTuple]] | list[list[tuple[SentimentTuple, float]]]:
        """Predict sentiment tuples for each example.

        With ``return_confidence=True`` each predicted tuple is paired with a
        confidence in ``[0, 1]`` — the fraction of the ``n_samples`` draws it
        appeared in (always ``1.0`` when ``n_samples == 1``).
        """
        self._require_given_elements(examples)
        self.diagnostics = {"calls": 0, "repairs": 0, "dropped_items": 0, "failed_examples": 0}
        base = self._classify_example if self.task.is_classification else self._extract_example
        worker = self._with_progress(base, len(examples)) if self.on_progress else base
        if self.concurrency == 1 or len(examples) <= 1:
            scored = [worker(example) for example in examples]
        else:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(examples))) as pool:
                scored = list(pool.map(worker, examples))
        if return_confidence:
            return scored
        return [[t for t, _conf in row] for row in scored]

    def _with_progress(
        self,
        worker: Callable[[ABSAExample], list[tuple[SentimentTuple, float]]],
        total: int,
    ) -> Callable[[ABSAExample], list[tuple[SentimentTuple, float]]]:
        """Wrap *worker* to report ``(completed, total)`` after each item."""
        callback = self.on_progress
        assert callback is not None  # only wrapped when on_progress is set
        done = [0]

        def tracked(example: ABSAExample) -> list[tuple[SentimentTuple, float]]:
            result = worker(example)
            with self._diagnostics_lock:
                done[0] += 1
                completed = done[0]
            callback(completed, total)
            return result

        return tracked

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

    def _extract_example(self, example: ABSAExample) -> list[tuple[SentimentTuple, float]]:
        """Extract for one input, aggregating ``n_samples`` draws by vote."""
        if self.n_samples == 1:
            return [(t, 1.0) for t in self._extract_once(example)]
        elements = self.task.ordered_elements(self.task.predicted)
        counts: Counter[tuple[str, ...]] = Counter()
        first: dict[tuple[str, ...], SentimentTuple] = {}
        for _ in range(self.n_samples):
            seen: set[tuple[str, ...]] = set()
            for t in self._extract_once(example):
                key = tuple_key(t, elements)
                if key not in seen:  # count presence per sample, not duplicates
                    seen.add(key)
                    counts[key] += 1
                    first.setdefault(key, t)
        return [
            (first[key], count / self.n_samples)
            for key, count in counts.items()
            if self._survives_vote(count)
        ]

    def _survives_vote(self, count: int) -> bool:
        """Whether a tuple seen in *count* of ``n_samples`` survives the vote."""
        if self.vote == "union":
            return count >= 1
        return count * 2 > self.n_samples  # strict majority

    def _extract_once(self, example: ABSAExample) -> list[SentimentTuple]:
        schema = (
            build_extraction_schema(self.task, self.categories, self.polarities)
            if self.use_schema
            else None
        )
        messages = extraction_messages(
            example.text,
            self.task,
            self.categories,
            self.polarities,
            self._select_extraction(example.text),
            extra_instructions=self.instructions,
            system_prompt_override=self._system_override,
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

    def _classify_example(self, example: ABSAExample) -> list[tuple[SentimentTuple, float]]:
        schema = build_classification_schema(self.polarities) if self.use_schema else None
        cls_exemplars = self._select_classification(example.text)
        predictions: list[tuple[SentimentTuple, float]] = []
        for target in example.tuples:
            votes: Counter[str] = Counter()
            for _ in range(self.n_samples):
                polarity = self._classify_target_once(example, target, cls_exemplars, schema)
                if polarity is not None:
                    votes[polarity] += 1
            if not votes:
                continue  # every sample failed (and on_error="skip" swallowed it)
            polarity, count = votes.most_common(1)[0]
            predictions.append((replace(target, polarity=polarity), count / self.n_samples))
        return predictions

    def _classify_target_once(
        self,
        example: ABSAExample,
        target: SentimentTuple,
        cls_exemplars: list[tuple[str, str | None, str]],
        schema: dict[str, Any] | None,
    ) -> str | None:
        """One polarity sample for *target*; ``None`` if it failed under skip."""
        messages = classification_messages(
            example.text,
            target.aspect_text,
            self.polarities,
            cls_exemplars,
            extra_instructions=self.instructions,
            system_prompt_override=self._system_override,
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
            return None
        return polarity

    def __repr__(self) -> str:
        return (
            f"LLMBackend(task={self.task.name!r}, llm={self.llm.name}, "
            f"exemplars={len(self._exemplars) + len(self._cls_exemplars)})"
        )

    def prompt_preview(self, text: str = "<text>", aspect: str = "<aspect>") -> str:
        """Render the prompt this backend would send for *text*.

        Unlike the static :meth:`describe_prompt`, this reflects the live
        configuration — selection mode and fitted exemplars, custom
        ``instructions``, and any ``system_prompt_fn`` override — as
        role-labelled message blocks.  ``aspect`` is used only for
        classification views.
        """
        if self.task.is_classification:
            messages = classification_messages(
                text,
                aspect,
                self.polarities,
                self._select_classification(text),
                extra_instructions=self.instructions,
                system_prompt_override=self._system_override,
            )
        else:
            messages = extraction_messages(
                text,
                self.task,
                self.categories,
                self.polarities,
                self._select_extraction(text),
                extra_instructions=self.instructions,
                system_prompt_override=self._system_override,
            )
        return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)

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
