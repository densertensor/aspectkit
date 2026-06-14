"""Custom-data interop: records, CSV, JSON, pandas, Hugging Face datasets.

Most custom ABSA data lives in one of two shapes:

* **nested** — one record per text, with a list of opinion dicts::

      {"text": "...", "tuples": [{"aspect": "pasta", "polarity": "positive"}]}

* **flat** — one record per opinion (the typical CSV/DataFrame layout),
  grouped into examples by an id column or by the text itself::

      text,aspect,polarity
      "Good pasta, bad wine",pasta,positive
      "Good pasta, bad wine",wine,negative

:func:`from_records` converts either shape (auto-detected, every column
remappable) into the canonical schema; :func:`from_pandas`,
:func:`from_hf_dataset`, :func:`read_csv`, and :func:`read_json` are
thin entry points over it.  :func:`to_records` / :func:`to_pandas`
export examples (or predictions) back out for analysis.

Conventions applied during conversion:

* aspect values that are missing/empty/``NULL``/``IMPLICIT`` become
  :data:`~aspectkit.schema.IMPLICIT`;
* opinion values that are missing/empty become ``None`` (not annotated),
  while explicit ``NULL``/``IMPLICIT`` strings become implicit;
* string spans are offset-aligned against the text where possible, and
  ``{"text", "start", "end"}`` dicts are taken verbatim;
* polarity labels are normalised (``POS``, ``Positive``, ACOS integer
  codes, ...) via :func:`~aspectkit.normalize.canonical_polarity`.
"""

from __future__ import annotations

import csv as _csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aspectkit.exceptions import DataFormatError
from aspectkit.normalize import align_span, canonical_polarity
from aspectkit.schema import (
    IMPLICIT,
    ABSAExample,
    Implicit,
    SentimentTuple,
    Span,
)

__all__ = [
    "from_hf_dataset",
    "from_pandas",
    "from_records",
    "read_csv",
    "read_json",
    "to_pandas",
    "to_records",
]


class _Auto:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "AUTO"


#: Sentinel: detect nested vs flat layout from the data.
AUTO = _Auto()

_IMPLICIT_STRINGS = frozenset({"null", "implicit"})


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN from pandas
        return True
    return isinstance(value, str) and not value.strip()


def _span_value(value: Any, text: str, *, where: str) -> Span | Implicit:
    if isinstance(value, Mapping):
        try:
            return Span.from_dict(dict(value))
        except (KeyError, ValueError) as exc:
            raise DataFormatError(f"{where}: invalid span dict {value!r}: {exc}") from exc
    if isinstance(value, str):
        if value.strip().lower() in _IMPLICIT_STRINGS:
            return IMPLICIT
        return align_span(value, text)
    raise DataFormatError(f"{where}: expected a span string or dict, got {value!r}")


def _polarity_value(value: Any, *, where: str) -> str | None:
    if _is_missing(value):
        return None
    try:
        return canonical_polarity(value)
    except ValueError as exc:
        raise DataFormatError(f"{where}: {exc}") from exc


def _build_tuple(
    item: Mapping[str, Any],
    text: str,
    keys: dict[str, str],
    *,
    where: str,
) -> SentimentTuple | None:
    raw = {element: item.get(key) for element, key in keys.items()}
    if all(_is_missing(value) for value in raw.values()):
        return None

    aspect: Span | Implicit
    if _is_missing(raw["aspect"]):
        aspect = IMPLICIT
    else:
        aspect = _span_value(raw["aspect"], text, where=where)

    opinion: Span | Implicit | None
    if _is_missing(raw["opinion"]):
        opinion = None
    else:
        opinion = _span_value(raw["opinion"], text, where=where)

    category = None if _is_missing(raw["category"]) else str(raw["category"]).strip()
    return SentimentTuple(
        aspect=aspect,
        category=category,
        opinion=opinion,
        polarity=_polarity_value(raw["polarity"], where=where),
    )


def from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    text: str = "text",
    id: str | None = "id",
    tuples: str | None | _Auto = AUTO,
    aspect: str = "aspect",
    category: str = "category",
    opinion: str = "opinion",
    polarity: str = "polarity",
    group_by: str | None = None,
) -> list[ABSAExample]:
    """Convert an iterable of dict-like records into canonical examples.

    Args:
        records: Dict-like records (rows of a CSV, items of a Hugging
            Face dataset, ``DataFrame.to_dict("records")`` output, ...).
        text: Field holding the example text.
        id: Field holding a stable example id, used when present
            (records without it are unaffected).  Pass ``None`` to
            ignore ids entirely.
        tuples: Field holding a nested list of opinion dicts.  Leave at
            the default to auto-detect: nested when the first record has
            a ``"tuples"`` field, flat otherwise.  Pass ``None`` to force
            the flat (one record per opinion) layout.
        aspect: Field/key for the aspect term.
        category: Field/key for the aspect category.
        opinion: Field/key for the opinion term.
        polarity: Field/key for the polarity label.
        group_by: In the flat layout, field whose value groups records
            into one example.  Defaults to the id when a record has one,
            else the text itself.

    Returns:
        Examples in the canonical schema, in first-seen order.

    Raises:
        DataFormatError: On missing text fields, span/polarity values
            that cannot be interpreted, or inconsistent texts within a
            flat group.
    """
    materialised = [dict(record) for record in records]
    keys = {"aspect": aspect, "category": category, "opinion": opinion, "polarity": polarity}

    if isinstance(tuples, _Auto):
        nested_field = "tuples" if materialised and "tuples" in materialised[0] else None
    else:
        nested_field = tuples

    def record_id(record: dict[str, Any]) -> str | None:
        value = record.get(id) if id else None
        return None if _is_missing(value) else str(value)

    examples: list[ABSAExample] = []
    if nested_field is not None:
        for index, record in enumerate(materialised):
            where = f"records[{index}]"
            if _is_missing(record.get(text)):
                raise DataFormatError(f"{where}: missing text field {text!r}")
            example_text = str(record[text])
            items = record.get(nested_field) or []
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise DataFormatError(
                    f"{where}: field {nested_field!r} must be a list of opinion dicts"
                )
            example_tuples = []
            for item in items:
                if not isinstance(item, Mapping):
                    raise DataFormatError(f"{where}: opinion entries must be dicts, got {item!r}")
                built = _build_tuple(item, example_text, keys, where=where)
                if built is not None:
                    example_tuples.append(built)
            examples.append(
                ABSAExample(text=example_text, tuples=example_tuples, id=record_id(record))
            )
        return examples

    # Flat layout: group one-opinion-per-record rows into examples.  The
    # group key is tagged by its source so id values can never collide
    # with text values.  Tuples are accumulated and each example is
    # constructed once at the end (ABSAExample is immutable).
    grouped_tuples: dict[Any, list[SentimentTuple]] = {}
    grouped_info: dict[Any, tuple[str, str | None]] = {}
    order: list[Any] = []
    for index, record in enumerate(materialised):
        where = f"records[{index}]"
        if _is_missing(record.get(text)):
            raise DataFormatError(f"{where}: missing text field {text!r}")
        example_text = str(record[text])
        rid = record_id(record)
        if group_by is not None:
            key: Any = ("group", record.get(group_by))
        elif rid is not None:
            key = ("id", rid)
        else:
            key = ("text", example_text)

        if key not in grouped_info:
            grouped_info[key] = (example_text, rid)
            grouped_tuples[key] = []
            order.append(key)
        elif grouped_info[key][0] != example_text:
            raise DataFormatError(
                f"{where}: group {key[1]!r} mixes different texts "
                f"({grouped_info[key][0]!r} vs {example_text!r})"
            )

        built = _build_tuple(record, example_text, keys, where=where)
        if built is not None:
            grouped_tuples[key].append(built)
    return [
        ABSAExample(text=grouped_info[k][0], tuples=grouped_tuples[k], id=grouped_info[k][1])
        for k in order
    ]


def from_pandas(frame: Any, **mapping: Any) -> list[ABSAExample]:
    """Convert a ``pandas.DataFrame`` into canonical examples.

    A thin wrapper over :func:`from_records` (same keyword arguments);
    NaN cells are treated as missing values.
    """
    if not hasattr(frame, "to_dict"):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    return from_records(frame.to_dict(orient="records"), **mapping)


def from_hf_dataset(dataset: Iterable[Mapping[str, Any]], **mapping: Any) -> list[ABSAExample]:
    """Convert a Hugging Face ``datasets.Dataset`` into canonical examples.

    Works with anything that iterates as dict-like rows (a ``Dataset``,
    a dataset split, or a plain list of dicts); same keyword arguments
    as :func:`from_records`.
    """
    return from_records(dataset, **mapping)


def read_csv(path: str | Path, *, encoding: str = "utf-8", **mapping: Any) -> list[ABSAExample]:
    """Read a CSV file of flat (one opinion per row) records.

    Same keyword arguments as :func:`from_records`; the layout is forced
    to flat because CSV cells cannot nest.
    """
    mapping.setdefault("tuples", None)
    with Path(path).open(encoding=encoding, newline="") as handle:
        return from_records(_csv.DictReader(handle), **mapping)


def read_json(path: str | Path, **mapping: Any) -> list[ABSAExample]:
    """Read a JSON file containing an array of example records.

    Handles both the nested and flat layouts (auto-detected), including
    files produced by :func:`to_records`/native serialisation.  Same
    keyword arguments as :func:`from_records`.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise DataFormatError(f"{path}: expected a top-level JSON array of records")
    return from_records(payload, **mapping)


def to_records(examples: Sequence[ABSAExample], *, flat: bool = False) -> list[dict[str, Any]]:
    """Export examples to plain dict records.

    Args:
        examples: Canonical examples (gold or predictions attached to
            texts).
        flat: When ``False`` (default), one nested record per example in
            the native schema (round-trips through :func:`read_json`).
            When ``True``, one record per opinion with columns
            ``id, text, aspect, category, opinion, polarity`` — implicit
            spans export as ``None``; examples without opinions export a
            single row with all element columns ``None``.
    """
    if not flat:
        return [example.to_dict() for example in examples]

    rows: list[dict[str, Any]] = []
    for example in examples:
        base = {"id": example.id, "text": example.text}
        if not example.tuples:
            rows.append(
                {**base, "aspect": None, "category": None, "opinion": None, "polarity": None}
            )
            continue
        for t in example.tuples:
            rows.append(
                {
                    **base,
                    "aspect": t.aspect_text,
                    "category": t.category,
                    "opinion": t.opinion_text,
                    "polarity": t.polarity,
                }
            )
    return rows


def to_pandas(examples: Sequence[ABSAExample]) -> Any:
    """Export examples to a flat ``pandas.DataFrame`` (requires pandas).

    One row per opinion; see :func:`to_records` with ``flat=True``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        from aspectkit.exceptions import MissingDependencyError

        raise MissingDependencyError("pandas", "pandas", "to_pandas") from exc
    return pd.DataFrame(to_records(examples, flat=True))
