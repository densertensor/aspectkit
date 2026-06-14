"""Resumable corpus prediction via an on-disk checkpoint.

:func:`predict_with_checkpoint` runs a backend over a corpus while
persisting completed predictions to a JSON-Lines checkpoint, so an
interrupted or rate-limited run can be resumed without re-predicting (and
re-paying for) the work already done.  Each input example is identified by
its stable ``id``; the checkpoint stores one example per line with
``tuples`` set to its prediction.  A later call with the same path skips
every example whose ``id`` is already recorded and predicts only the rest.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aspectkit.exceptions import DataFormatError
from aspectkit.io.jsonl import write_jsonl
from aspectkit.schema import ABSAExample, SentimentTuple

if TYPE_CHECKING:
    from aspectkit.backends.base import Backend

__all__ = ["predict_with_checkpoint"]


def _load_checkpoint(path: Path) -> tuple[list[ABSAExample], bool]:
    """Read checkpoint records, tolerating a single torn trailing line.

    Appends are sequential, so a hard kill (SIGKILL / OOM / power loss)
    mid-write can only truncate the *final* line.  Such a torn last line is
    reported (so the caller re-predicts that example, which is safe and
    idempotent); a malformed *earlier* line is genuine corruption and is
    raised via :class:`~aspectkit.exceptions.DataFormatError`.  ``read_jsonl``
    is deliberately left strict — this leniency is scoped to the checkpoint.

    Returns:
        The intact records and whether a torn trailing line was dropped.
    """
    records: list[ABSAExample] = []
    suspect: tuple[int, Exception] | None = None
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if suspect is not None:
                # A line failed to parse and was followed by more content, so
                # it is genuine mid-file corruption, not an interrupted append.
                bad_lineno, exc = suspect
                raise DataFormatError(f"{path}:{bad_lineno}: {exc}") from exc
            try:
                records.append(ABSAExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                suspect = (lineno, exc)
    return records, suspect is not None


def predict_with_checkpoint(
    backend: Backend,
    examples: Sequence[ABSAExample],
    checkpoint_path: str | Path,
    *,
    overwrite: bool = False,
    batch_size: int | None = None,
) -> list[list[SentimentTuple]]:
    """Predict over *examples*, checkpointing completed work to disk.

    Predictions are appended to *checkpoint_path* as aspectkit JSON-Lines
    (one example per line, ``tuples`` holding the prediction, keyed by
    ``id``).  On a later call with the same path, examples whose ``id`` is
    already recorded are served from the checkpoint instead of being
    re-predicted; only the remainder is passed to ``backend.predict``.  The
    merged result is always returned in the order of *examples*, however the
    run was split across calls.

    Every example must carry a unique, non-``None`` ``id`` — it is the
    checkpoint key.  If you change an example's text but keep its ``id`` the
    stale prediction is reused; pass ``overwrite=True`` or a fresh path to
    recompute from scratch.

    If a previous run was hard-killed mid-write, the checkpoint may end in a
    truncated line; it is detected on the next call, dropped (with a warning),
    and that one example is re-predicted — all other completed work is kept.

    Args:
        backend: The prediction backend (any
            :class:`~aspectkit.backends.base.Backend`).
        examples: Inputs to predict, each with a unique ``id``.
        checkpoint_path: JSON-Lines file accumulating completed predictions.
        overwrite: If ``True``, discard any existing checkpoint and start
            fresh; otherwise resume from it (the default).
        batch_size: If set, predict and persist the outstanding examples in
            chunks of this size, so a crash mid-run keeps the chunks already
            written.  ``None`` (the default) predicts all outstanding
            examples in a single ``backend.predict`` call.

    Returns:
        One list of predicted tuples per input example, in input order.

    Raises:
        ValueError: If an example lacks an ``id``, the ids are not unique,
            or ``batch_size`` is less than 1.
    """
    example_list = list(examples)
    ids: list[str] = []
    for index, example in enumerate(example_list):
        if example.id is None:
            raise ValueError(
                "predict_with_checkpoint requires a non-None id on every example "
                f"(the checkpoint key); examples[{index}] has none."
            )
        ids.append(example.id)
    seen: set[str] = set()
    for example_id in ids:
        if example_id in seen:
            raise ValueError(
                f"predict_with_checkpoint requires unique example ids; "
                f"{example_id!r} appears more than once."
            )
        seen.add(example_id)
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    checkpoint_path = Path(checkpoint_path)
    if overwrite:
        checkpoint_path.unlink(missing_ok=True)

    done: dict[str, list[SentimentTuple]] = {}
    if checkpoint_path.exists():
        records, torn = _load_checkpoint(checkpoint_path)
        for record in records:
            if record.id is not None:
                done[record.id] = record.tuples
        if torn:
            # Rewrite with only the intact records so the next append cannot
            # glue onto the partial line; the dropped example is re-predicted.
            write_jsonl(records, checkpoint_path)
            warnings.warn(
                f"checkpoint {checkpoint_path} had a truncated trailing line from an "
                "interrupted write; it was dropped and that example will be re-predicted.",
                stacklevel=2,
            )

    pending = [
        (example_id, example)
        for example_id, example in zip(ids, example_list, strict=True)
        if example_id not in done
    ]
    if pending:
        step = batch_size if batch_size is not None else len(pending)
        for start in range(0, len(pending), step):
            chunk = pending[start : start + step]
            predictions = backend.predict([example for _id, example in chunk])
            records = [
                ABSAExample(text=example.text, tuples=preds, id=example_id)
                for (example_id, example), preds in zip(chunk, predictions, strict=True)
            ]
            write_jsonl(records, checkpoint_path, append=True)
            for (example_id, _example), preds in zip(chunk, predictions, strict=True):
                done[example_id] = preds

    return [done[example_id] for example_id in ids]
