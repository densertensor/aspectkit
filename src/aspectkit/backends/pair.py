"""Cross-encoder backend for aspect-term sentiment classification (ATSC).

Implements the auxiliary-sentence formulation of Sun et al. (2019): the
sentence and the aspect are fed to a sequence-pair classifier
(``[CLS] text [SEP] aspect``), which remains the practical workhorse for
polarity-given-aspect.  The default checkpoint,
``yangheng/deberta-v3-base-absa-v1.1``, is trained on merged
SemEval+MAMS data and gives a strong zero-config starting point.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, overload

from aspectkit.backends.base import Backend
from aspectkit.exceptions import MissingDependencyError
from aspectkit.normalize import canonical_polarity
from aspectkit.schema import ABSAExample, SentimentTuple, Span
from aspectkit.tasks import get_task

__all__ = ["PairClassifierBackend"]

DEFAULT_CHECKPOINT = "yangheng/deberta-v3-base-absa-v1.1"


def _softmax(row: list[float]) -> list[float]:
    """Numerically-stable softmax of a logit row (stdlib only)."""
    hi = max(row)
    exps = [math.exp(value - hi) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


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


class PairClassifierBackend(Backend):
    """ATSC via a Hugging Face sequence-pair classification model.

    Args:
        model: Hub id of a ``AutoModelForSequenceClassification``
            checkpoint whose labels map onto polarity classes, or an
            already-loaded model object (pass ``tokenizer`` too).
        tokenizer: Tokenizer object, required when ``model`` is an
            object; loaded from the hub otherwise.
        device: Torch device string (e.g. ``"cuda"``); ``None`` keeps
            the model where it is (hub loads default to CPU).
        batch_size: Pairs per forward pass.
        max_length: Tokenizer truncation length.
    """

    def __init__(
        self,
        model: Any = DEFAULT_CHECKPOINT,
        *,
        tokenizer: Any | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.task = get_task("atsc")
        self.batch_size = batch_size
        self.max_length = max_length
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
                    "a model object needs its tokenizer: "
                    "PairClassifierBackend(model, tokenizer=tokenizer)"
                )
            self._model = model
            self._tokenizer = tokenizer
            self.model_name = getattr(model, "name_or_path", "model")

    def fit(
        self,
        examples: Sequence[ABSAExample],
        *,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int | None = None,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        shuffle: bool = True,
        seed: int = 42,
        val_examples: Sequence[ABSAExample] | None = None,
        patience: int | None = None,
        save_best: bool | None = None,
    ) -> PairClassifierBackend:
        """Fine-tune the cross-encoder on labelled (text, aspect) pairs.

        A plain, transparent training loop: AdamW, seeded shuffling,
        gradient clipping — the standard BERT-classifier recipe.  Every
        tuple with a polarity contributes one training pair; implicit
        aspects pair with the empty string, matching :meth:`predict`.
        Mean per-epoch losses are recorded in :attr:`history_`.

        Args:
            examples: Labelled examples carrying aspects and polarities.
            epochs: Full passes over the data.
            learning_rate: AdamW learning rate (2e-5 suits BERT-family
                encoders).
            batch_size: Training batch size; defaults to the backend's.
            weight_decay: AdamW weight decay.
            max_grad_norm: Gradient clipping threshold (0 disables).
            shuffle: Reshuffle pairs every epoch (seeded).
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

        Raises:
            ValueError: If no labelled pairs are found, or a training
                polarity has no corresponding label in the model's
                ``id2label`` config (instantiate the model with matching
                ``num_labels``/``id2label`` in that case).
        """
        if (patience is not None or save_best) and not val_examples:
            raise ValueError("patience= and save_best= require val_examples=")
        pairs, n_implicit = self._build_pairs(examples)
        if not pairs:
            raise ValueError("fit needs at least one tuple with an aspect and a polarity")
        if n_implicit:
            warnings.warn(
                f"{n_implicit} implicit aspect(s) trained against an empty target, "
                "matching predict()'s convention",
                stacklevel=2,
            )
        val_pairs = self._build_pairs(val_examples)[0] if val_examples else []
        if (patience is not None or save_best) and not val_pairs:
            raise ValueError(
                "patience= and save_best= need labelled validation pairs, but val_examples "
                "produced none (no val tuple has both an aspect and a polarity)"
            )

        self._ensure_loaded()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - transformers implies torch
            raise MissingDependencyError(
                "torch", "transformers", "PairClassifierBackend.fit"
            ) from exc
        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded

        label_to_id: dict[str, int] = {}
        for index, label in dict(getattr(model.config, "id2label", {})).items():
            try:
                label_to_id[canonical_polarity(label)] = int(index)
            except ValueError:
                continue
        all_polarities = {p for _, _, p in pairs} | {p for _, _, p in val_pairs}
        unmapped = sorted(all_polarities - set(label_to_id))
        if unmapped:
            raise ValueError(
                f"training labels {unmapped} have no counterpart in the model's id2label "
                f"({dict(getattr(model.config, 'id2label', {}))}); instantiate the model "
                "with matching num_labels/id2label and pass it as the model object"
            )

        device = self._tensor_device()
        batch = batch_size or self.batch_size
        torch.manual_seed(seed)
        rng = random.Random(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

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
                    loss = self._batch_loss(chunk, device, label_to_id)
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
                val_loss = self._validation_loss(val_pairs, batch, device, label_to_id)
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

    def _build_pairs(
        self, examples: Sequence[ABSAExample]
    ) -> tuple[list[tuple[str, str, str]], int]:
        """Flatten labelled tuples into (text, aspect, polarity) pairs.

        Implicit aspects pair with the empty string (matching
        :meth:`predict`); the count of those is returned alongside.
        """
        pairs: list[tuple[str, str, str]] = []
        n_implicit = 0
        for example in examples:
            for t in example.tuples:
                if t.polarity is None:
                    continue
                if isinstance(t.aspect, Span):
                    pairs.append((example.text, t.aspect.text, t.polarity))
                else:
                    n_implicit += 1
                    pairs.append((example.text, "", t.polarity))
        return pairs, n_implicit

    def _batch_loss(
        self, batch_pairs: list[tuple[str, str, str]], device: Any, label_to_id: dict[str, int]
    ) -> Any:
        """Classification loss for one batch of (text, aspect, polarity) pairs."""
        import torch

        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded
        encoded = tokenizer(
            [text for text, _, _ in batch_pairs],
            [aspect for _, aspect, _ in batch_pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([label_to_id[polarity] for _, _, polarity in batch_pairs])
        if device is not None:
            encoded = encoded.to(device)
            labels = labels.to(device)
        return model(**encoded, labels=labels).loss

    def _validation_loss(
        self,
        val_pairs: list[tuple[str, str, str]],
        batch: int,
        device: Any,
        label_to_id: dict[str, int],
    ) -> float:
        """Mean per-batch loss over held-out pairs (eval mode, no grad)."""
        model = self._model
        assert model is not None
        model.eval()
        total, n_batches = 0.0, 0
        try:
            with self._no_grad():
                for offset in range(0, len(val_pairs), batch):
                    loss = self._batch_loss(val_pairs[offset : offset + batch], device, label_to_id)
                    total += float(loss)
                    n_batches += 1
        finally:
            model.train()
        return total / n_batches if n_batches else 0.0

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the (fine-tuned) model and tokenizer to *directory*.

        The directory is reloadable via ``PairClassifierBackend(directory)``.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        self._model.save_pretrained(str(directory))
        self._tokenizer.save_pretrained(str(directory))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError(
                "transformers", "transformers", "PairClassifierBackend"
            ) from exc
        assert self._hub_id is not None  # guaranteed by __init__
        self._tokenizer = AutoTokenizer.from_pretrained(self._hub_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._hub_id)
        self._model.eval()
        if self._device is not None:
            self._model.to(self._device)

    def _no_grad(self) -> Any:
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.no_grad()

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

    def _classify_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Classify (text, aspect) pairs into (polarity, softmax probability)."""
        self._ensure_loaded()
        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # set by _ensure_loaded
        id2label = dict(getattr(model.config, "id2label", {}))
        device = self._tensor_device()
        results: list[tuple[str, float]] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            encoded = tokenizer(
                [text for text, _ in batch],
                [aspect for _, aspect in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if device is not None and hasattr(encoded, "to"):
                encoded = encoded.to(device)
            with self._no_grad():
                logits = model(**encoded).logits
            for row in logits.tolist():
                index = max(range(len(row)), key=row.__getitem__)
                label = id2label.get(index, str(index))
                results.append((canonical_polarity(label), _softmax(row)[index]))
        return results

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
        """Classify the polarity of every given aspect in every example.

        Implicit aspects are classified against the empty string (the
        model judges the sentence as a whole), with a warning: pair
        classifiers are trained on explicit targets.  With
        ``return_confidence=True`` each tuple is paired with the model's
        softmax probability for the chosen polarity.
        """
        self._require_given_elements(examples)

        pairs: list[tuple[str, str]] = []
        layout: list[list[SentimentTuple]] = []
        n_implicit = 0
        for example in examples:
            layout.append(example.tuples)
            for target in example.tuples:
                if isinstance(target.aspect, Span):
                    pairs.append((example.text, target.aspect.text))
                else:
                    n_implicit += 1
                    pairs.append((example.text, ""))
        if n_implicit:
            warnings.warn(
                f"{n_implicit} implicit aspect(s) classified against an empty "
                "target; pair classifiers are trained on explicit aspects, "
                "treat these predictions with care",
                stacklevel=2,
            )

        self.diagnostics = {"n_implicit": n_implicit, "pairs": len(pairs)}
        scored = iter(self._classify_pairs(pairs))
        predictions: list[list[tuple[SentimentTuple, float]]] = []
        for targets in layout:
            row: list[tuple[SentimentTuple, float]] = []
            for target in targets:
                polarity, prob = next(scored)
                row.append((replace(target, polarity=polarity), prob))
            predictions.append(row)
        if return_confidence:
            return predictions
        return [[t for t, _prob in row] for row in predictions]

    def __repr__(self) -> str:
        return f"PairClassifierBackend(model={self.model_name!r})"
