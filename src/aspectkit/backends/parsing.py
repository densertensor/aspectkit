"""Robust parsing of model output into the canonical schema.

LLM replies are noisy even under JSON instructions: code fences,
leading prose, trailing commentary.  :func:`extract_json` recovers the
first well-formed JSON value; the ``payload_to_*`` functions then map it
onto canonical tuples, normalising labels and aligning spans to the
source text.  Malformed *items* inside an otherwise valid reply are
dropped (and reported via the returned diagnostics) rather than failing
the whole prediction — per-item noise is expected from prompted models,
and silent acceptance of garbage is not an option either.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from aspectkit.exceptions import ParseError
from aspectkit.normalize import align_span, canonical_polarity
from aspectkit.schema import IMPLICIT, Implicit, SentimentTuple, Span
from aspectkit.tasks import Task

__all__ = ["extract_json", "payload_to_polarity", "payload_to_tuples"]

_NULL_STRINGS = frozenset({"", "null", "none", "nil", "n/a", "implicit", "(implicit)"})


def extract_json(text: str) -> Any:
    """Extract the first JSON value from a model reply.

    Tries, in order: the whole string; the contents of a fenced code
    block; the first decodable JSON object/array found by scanning.

    Raises:
        ParseError: If no JSON value can be decoded.
    """
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    if "```" in candidate:
        for block in candidate.split("```")[1::2]:  # odd segments are fenced
            inner = block.strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                continue

    decoder = json.JSONDecoder()
    for idx, char in enumerate(candidate):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            return value

    excerpt = candidate[:200] + ("..." if len(candidate) > 200 else "")
    raise ParseError(f"no JSON value found in model reply: {excerpt!r}")


def _span_or_implicit(value: Any, text: str) -> Span | Implicit | None:
    """Map a model-emitted aspect/opinion value onto the schema.

    ``None`` and null-like strings mean implicit; other strings are
    aligned to the source text.  Non-string values return ``None``
    (caller drops the item).
    """
    if value is None:
        return IMPLICIT
    if not isinstance(value, str):
        return None
    if value.strip().lower() in _NULL_STRINGS:
        return IMPLICIT
    return align_span(value, text)


def _canonical_category(value: Any, categories: Sequence[str] | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip()
    if categories:
        for known in categories:
            if known.lower() == cleaned.lower():
                return known
    return cleaned


def payload_to_tuples(
    payload: Any,
    task: Task,
    text: str,
    *,
    categories: Sequence[str] | None = None,
) -> tuple[list[SentimentTuple], list[str]]:
    """Convert a decoded extraction payload into canonical tuples.

    Accepts either ``{"tuples": [...]}`` (the requested shape) or a bare
    list (a common model deviation).

    Args:
        payload: Decoded JSON value.
        task: The task view (determines which keys are read).
        text: Source sentence, for span alignment.
        categories: Optional category inventory used to canonicalise
            label casing.

    Returns:
        ``(tuples, problems)`` where *problems* describes dropped items.

    Raises:
        ParseError: If the payload is not list-shaped at all.
    """
    if isinstance(payload, dict) and isinstance(payload.get("tuples"), list):
        items = payload["tuples"]
    elif isinstance(payload, list):
        items = payload
    elif (
        isinstance(payload, dict)
        and all(k in ("aspect", "category", "opinion", "polarity") for k in payload)
        and payload
    ):
        items = [payload]  # single-object reply for a single-opinion sentence
    else:
        raise ParseError(
            f"expected {{'tuples': [...]}} or a JSON array, got {type(payload).__name__}"
        )

    elements = task.ordered_elements(task.predicted)
    tuples: list[SentimentTuple] = []
    problems: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"item {i}: not an object")
            continue
        kwargs: dict[str, Any] = {"aspect": IMPLICIT}
        valid = True
        for element in elements:
            value = item.get(element)
            if element in ("aspect", "opinion"):
                span = _span_or_implicit(value, text)
                if span is None:
                    problems.append(f"item {i}: invalid {element} value {value!r}")
                    valid = False
                    break
                kwargs[element] = span
            elif element == "category":
                kwargs["category"] = _canonical_category(value, categories)
            elif element == "polarity":
                try:
                    kwargs["polarity"] = canonical_polarity(value) if value is not None else None
                except ValueError:
                    problems.append(f"item {i}: unknown polarity {value!r}")
                    valid = False
                    break
        if valid:
            tuples.append(SentimentTuple(**kwargs))
    return tuples, problems


def payload_to_polarity(payload: Any) -> str:
    """Convert a decoded classification payload into a polarity label.

    Accepts ``{"polarity": "..."}`` or a bare string.

    Raises:
        ParseError: If no usable polarity is present.
    """
    value = payload.get("polarity") if isinstance(payload, dict) else payload
    if not isinstance(value, str):
        raise ParseError(f"expected a polarity string, got {value!r}")
    try:
        return canonical_polarity(value)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
