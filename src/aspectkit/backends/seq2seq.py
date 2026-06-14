"""Generative seq2seq backend: the fine-tunable path to tuple extraction.

Linearises sentiment tuples into target strings (see
:mod:`~aspectkit.backends.templates`), fine-tunes an encoder-decoder
model (T5/BART family) to produce them, and parses generations back into
canonical tuples.  This is the backend family that holds the published
state of the art on exact-match ACOS/ASQP — the benchmark literature
consistently places fine-tuned seq2seq models ahead of prompted LLMs on
tuple extraction, which is why this backend exists alongside
:class:`~aspectkit.backends.llm.LLMBackend` rather than instead of it.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from aspectkit.backends.base import Backend
from aspectkit.backends.parsing import _NULL_STRINGS, _canonical_category
from aspectkit.backends.templates import (
    Fragment,
    MarkersTemplate,
    ParaphraseTemplate,
    TupleTemplate,
)
from aspectkit.exceptions import MissingDependencyError
from aspectkit.normalize import align_span, canonical_polarity
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = ["Seq2SeqBackend"]

DEFAULT_CHECKPOINT = "t5-base"

_STYLES = ("markers", "paraphrase")


def _snapshot(model: Any) -> bytes:
    """Serialise a model's parameters to an in-memory checkpoint."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _restore(model: Any, blob: bytes) -> None:
    """Load parameters saved by :func:`_snapshot` back into *model*."""
    import io

    import torch

    model.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))


class Seq2SeqBackend(Backend):
    """Tuple extraction with a fine-tuned encoder-decoder model.

    The model is trained (via :meth:`fit`) to generate a linearised form
    of the gold tuples; :meth:`predict` generates and parses it back.
    A fresh ``t5-base`` knows nothing about the templates — **this
    backend requires fine-tuning on labelled data before its
    predictions mean anything** (unlike the prompted LLM backend).

    Args:
        model: Hub id of an ``AutoModelForSeq2SeqLM`` checkpoint (or a
            directory saved by :meth:`save_pretrained`), or an
            already-loaded model object (pass ``tokenizer`` too).
        tokenizer: Tokenizer object, required when ``model`` is an
            object; loaded alongside the checkpoint otherwise.
        task: Extraction view, e.g. ``"acos"`` (default), ``"aste"``.
            Classification views (ATSC) are not supported — use the
            pair or LLM backend for those.
        style: Target linearisation: ``"markers"`` (default; MvP-style
            element markers, any view) or ``"paraphrase"`` (ASQP-style
            natural language, full quadruple view only).
        template: A custom :class:`~aspectkit.backends.templates.TupleTemplate`
            instance; overrides *style*.
        categories: Optional category inventory used to canonicalise the
            casing of generated category labels.
        device: Torch device string (e.g. ``"cuda"``); ``None`` keeps
            the model where it is and infers the device from its
            parameters when tensors must be moved.
        batch_size: Texts per generation/training batch.
        max_source_length: Tokenizer truncation length for input text.
        max_target_length: Generation budget / target truncation length.
        num_beams: Beam width for generation (1 = greedy).
        **generate_kwargs: Extra arguments forwarded to ``generate()``.
    """

    def __init__(
        self,
        model: Any = DEFAULT_CHECKPOINT,
        *,
        tokenizer: Any | None = None,
        task: str | Task = "acos",
        style: str = "markers",
        template: TupleTemplate | None = None,
        categories: Sequence[str] | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_source_length: int = 512,
        max_target_length: int = 192,
        num_beams: int = 1,
        **generate_kwargs: Any,
    ) -> None:
        resolved = get_task(task)
        if resolved.given:
            raise ValueError(
                f"Seq2SeqBackend covers extraction views only; for {resolved.name!r} "
                "use backend='pair' or backend='llm'"
            )
        self.task = resolved

        if template is not None:
            if template.task.name != resolved.name:
                raise ValueError(
                    f"template is for task {template.task.name!r}, "
                    f"backend is configured for {resolved.name!r}"
                )
            self.template = template
        elif style == "markers":
            self.template = MarkersTemplate(resolved)
        elif style == "paraphrase":
            self.template = ParaphraseTemplate(resolved)
        else:
            raise ValueError(f"unknown style {style!r}; expected one of {_STYLES} or template=")

        self.categories = list(categories) if categories else None
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self.generate_kwargs = generate_kwargs
        self._device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._hub_id: str | None = None
        #: Mean training loss per epoch from the most recent :meth:`fit`.
        self.history_: list[float] = []
        #: Mean validation loss per epoch when ``fit(val_examples=...)`` is used.
        self.val_history_: list[float] = []
        #: Counters from the most recent :meth:`predict` call.
        self.diagnostics: dict[str, int] = {}

        if isinstance(model, str):
            self._hub_id = model
            self.model_name = model
        else:
            if tokenizer is None:
                raise TypeError(
                    "a model object needs its tokenizer: Seq2SeqBackend(model, tokenizer=tokenizer)"
                )
            self._model = model
            self._tokenizer = tokenizer
            self.model_name = getattr(model, "name_or_path", "model")

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        examples: Sequence[ABSAExample],
        *,
        epochs: int = 20,
        learning_rate: float = 3e-4,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        shuffle: bool = True,
        seed: int = 42,
        val_examples: Sequence[ABSAExample] | None = None,
        patience: int | None = None,
        save_best: bool | None = None,
    ) -> Seq2SeqBackend:
        """Fine-tune the model to generate linearised gold tuples.

        A plain, transparent training loop: AdamW, seeded shuffling,
        gradient clipping — the setup of the ASQP/MvP papers (whose
        T5 defaults the hyperparameters follow; lower the learning rate
        for BART-family models).  Mean per-epoch losses are recorded in
        :attr:`history_`.

        Args:
            examples: Labelled training examples (examples with no
                tuples are included: they teach the empty output).
            epochs: Full passes over the data.
            learning_rate: AdamW learning rate (3e-4 suits T5).
            batch_size: Training batch size; defaults to the backend's.
            weight_decay: AdamW weight decay.
            max_grad_norm: Gradient clipping threshold (0 disables).
            shuffle: Reshuffle examples every epoch (seeded).
            seed: Seed for shuffling and dropout reproducibility.
            val_examples: Optional held-out examples; their mean per-epoch
                loss is recorded in :attr:`val_history_`.  Required for
                *patience* and *save_best*.
            patience: Early-stop after this many epochs without a new best
                validation loss (``None`` disables early stopping).
            save_best: Keep an in-memory checkpoint of the lowest-val-loss
                epoch and restore it at the end (``None``/``False`` keeps the
                final epoch's weights).

        Returns:
            ``self``, for chaining.
        """
        if not examples:
            raise ValueError("fit needs at least one labelled example")
        if (patience is not None or save_best) and not val_examples:
            raise ValueError("patience= and save_best= require val_examples=")
        pairs = [(e.text, self.template.encode(e.tuples)) for e in examples]
        val_pairs = (
            [(e.text, self.template.encode(e.tuples)) for e in val_examples] if val_examples else []
        )
        self._ensure_loaded()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - transformers implies torch
            raise MissingDependencyError("torch", "transformers", "Seq2SeqBackend.fit") from exc

        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded
        device = self._tensor_device()
        batch = batch_size or self.batch_size
        torch.manual_seed(seed)
        rng = random.Random(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        pad_id = tokenizer.pad_token_id

        self.history_ = []
        self.val_history_ = []
        best_val = float("inf")
        best_state: bytes | None = None
        stale_epochs = 0
        model.train()
        try:
            for _ in range(epochs):
                order = list(range(len(pairs)))
                if shuffle:
                    rng.shuffle(order)
                total, n_batches = 0.0, 0
                for offset in range(0, len(order), batch):
                    chunk = [pairs[i] for i in order[offset : offset + batch]]
                    loss = self._batch_loss(chunk, device, pad_id)
                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    total += float(loss.detach())
                    n_batches += 1
                self.history_.append(total / n_batches)

                if not val_pairs:
                    continue
                val_loss = self._validation_loss(val_pairs, batch, device, pad_id)
                self.val_history_.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    stale_epochs = 0
                    if save_best:
                        best_state = _snapshot(model)
                else:
                    stale_epochs += 1
                    if patience is not None and stale_epochs >= patience:
                        break
        finally:
            model.eval()
        if best_state is not None:
            _restore(model, best_state)
        return self

    def _batch_loss(
        self, batch_pairs: list[tuple[str, str]], device: Any, pad_id: int | None
    ) -> Any:
        """Encoder-decoder loss for one batch of (text, target) pairs."""
        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded
        encoded = tokenizer(
            [text for text, _ in batch_pairs],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=[target for _, target in batch_pairs],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids
        if pad_id is not None:
            labels[labels == pad_id] = -100  # ignored by the loss
        if device is not None:
            encoded = encoded.to(device)
            labels = labels.to(device)
        return model(**encoded, labels=labels).loss

    def _validation_loss(
        self, val_pairs: list[tuple[str, str]], batch: int, device: Any, pad_id: int | None
    ) -> float:
        """Mean per-batch loss over held-out pairs (eval mode, no grad)."""
        model = self._model
        assert model is not None
        model.eval()
        total, n_batches = 0.0, 0
        try:
            with self._no_grad():
                for offset in range(0, len(val_pairs), batch):
                    loss = self._batch_loss(val_pairs[offset : offset + batch], device, pad_id)
                    total += float(loss)
                    n_batches += 1
        finally:
            model.train()
        return total / n_batches if n_batches else 0.0

    # --------------------------------------------------------------- predict

    def predict(self, examples: Sequence[ABSAExample]) -> list[list[SentimentTuple]]:
        """Generate and parse tuples for each example.

        Malformed fragments in a generation are dropped (counted in
        :attr:`diagnostics` and warned about), never silently accepted.
        """
        self.diagnostics = {"dropped_items": 0}
        targets = self._generate([example.text for example in examples])
        predictions: list[list[SentimentTuple]] = []
        for example, target in zip(examples, targets, strict=True):
            fragments, problems = self.template.decode(target)
            tuples, conversion_problems = self._fragments_to_tuples(fragments, example.text)
            problems += conversion_problems
            if problems:
                self.diagnostics["dropped_items"] += len(problems)
                warnings.warn(
                    f"dropped {len(problems)} malformed fragment(s) for example "
                    f"{example.id or example.text[:60]!r}: {'; '.join(problems)}",
                    stacklevel=2,
                )
            predictions.append(tuples)
        return predictions

    # ------------------------------------------------------------ persistence

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the (fine-tuned) model and tokenizer to *directory*.

        The directory is reloadable via ``Seq2SeqBackend(directory)``.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        self._model.save_pretrained(str(directory))
        self._tokenizer.save_pretrained(str(directory))

    # -------------------------------------------------------------- internals

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError("transformers", "transformers", "Seq2SeqBackend") from exc
        assert self._hub_id is not None  # guaranteed by __init__
        self._tokenizer = AutoTokenizer.from_pretrained(self._hub_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._hub_id)
        self._model.eval()
        if self._device is not None:
            self._model.to(self._device)

    def _tensor_device(self) -> Any | None:
        """Where input tensors should live: the configured device, or the
        model's own (so models already on an accelerator just work)."""
        if self._device is not None:
            return self._device
        if self._model is None:
            return None
        try:
            return next(self._model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    def _no_grad(self) -> Any:
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.no_grad()

    def _generate(self, texts: list[str]) -> list[str]:
        """Generate one target string per text, in order."""
        self._ensure_loaded()
        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded
        device = self._tensor_device()
        outputs: list[str] = []
        for offset in range(0, len(texts), self.batch_size):
            encoded = tokenizer(
                texts[offset : offset + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
                return_tensors="pt",
            )
            if device is not None and hasattr(encoded, "to"):
                encoded = encoded.to(device)
            with self._no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.max_target_length,
                    num_beams=self.num_beams,
                    **self.generate_kwargs,
                )
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs

    def _fragments_to_tuples(
        self, fragments: list[Fragment], text: str
    ) -> tuple[list[SentimentTuple], list[str]]:
        """Convert decoded surface strings into canonical tuples."""
        tuples: list[SentimentTuple] = []
        problems: list[str] = []
        for i, fragment in enumerate(fragments):
            kwargs: dict[str, Any] = {"aspect": IMPLICIT}
            valid = True
            for element, value in fragment.items():
                if element in ("aspect", "opinion"):
                    if value.strip().lower() in _NULL_STRINGS:
                        kwargs[element] = IMPLICIT
                    else:
                        kwargs[element] = align_span(value, text)
                elif element == "category":
                    category = _canonical_category(value, self.categories)
                    if category is None:
                        problems.append(f"fragment {i}: empty category")
                        valid = False
                        break
                    kwargs["category"] = category
                else:  # polarity
                    try:
                        kwargs["polarity"] = canonical_polarity(value)
                    except ValueError:
                        problems.append(f"fragment {i}: unknown polarity {value!r}")
                        valid = False
                        break
            if valid:
                tuples.append(SentimentTuple(**kwargs))
        return tuples, problems

    def __repr__(self) -> str:
        return (
            f"Seq2SeqBackend(task={self.task.name!r}, model={self.model_name!r}, "
            f"template={type(self.template).__name__})"
        )
