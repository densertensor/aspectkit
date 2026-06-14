"""Text and label normalisation shared by loaders, backends, and evaluation.

Normalisation is deliberately centralised: exact-match evaluation is only
meaningful if every component canonicalises labels and span text the same
way.
"""

from __future__ import annotations

import re

from aspectkit.schema import Span

__all__ = ["align_span", "canonical_polarity", "normalize_text"]

_WHITESPACE = re.compile(r"\s+")

#: Dataset-specific polarity labels mapped onto the canonical scheme.
_POLARITY_MAP: dict[str, str] = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "2": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "conflict": "conflict",
    "mixed": "conflict",
}


def normalize_text(text: str, *, casefold: bool = False) -> str:
    """Lowercase and collapse whitespace, for matching purposes only.

    This is the canonical form used by exact-match evaluation and by the
    aggregator when grouping aspect mentions.  It never alters stored
    data.

    Args:
        text: The surface text to normalise.
        casefold: Use Unicode case folding (:meth:`str.casefold`) instead
            of :meth:`str.lower`.  Case folding is the correct
            equality-normalisation for non-ASCII scripts (e.g. German
            ``ß`` → ``ss``, Greek final sigma); ``False`` (the default)
            preserves the historical ``.lower()`` behaviour exactly.
    """
    lowered = text.strip().casefold() if casefold else text.strip().lower()
    return _WHITESPACE.sub(" ", lowered)


def canonical_polarity(label: str | int, *, extra_map: dict[str, str] | None = None) -> str:
    """Map a dataset- or model-specific polarity label to the canonical scheme.

    Handles SemEval labels (``positive``/``negative``/``neutral``/``conflict``),
    ASTE labels (``POS``/``NEG``/``NEU``), ACOS integer codes (``0``/``1``/``2``),
    and common variants, case-insensitively.

    Args:
        label: The dataset- or model-specific label.
        extra_map: Additional label → canonical-polarity entries, merged
            over (and taking precedence over) the built-in map.  Keys are
            matched case-insensitively.  Use this for datasets whose
            integer codes or localised labels differ from the ACOS/SemEval
            conventions (e.g. ``{"正面": "positive"}``) without
            patching the shared default.

    Raises:
        ValueError: If the label cannot be mapped.
    """
    key = str(label).strip().lower()
    if extra_map:
        extra = {str(k).strip().lower(): v for k, v in extra_map.items()}
        if key in extra:
            return extra[key]
    try:
        return _POLARITY_MAP[key]
    except KeyError:
        raise ValueError(f"cannot map polarity label {label!r} to canonical scheme") from None


def align_span(span_text: str, sentence: str, *, start_from: int = 0) -> Span:
    """Locate *span_text* in *sentence* and return an offset-aligned span.

    Generative and LLM backends produce surface text without offsets; this
    aligns the first case-insensitive occurrence at or after *start_from*,
    preferring matches on word boundaries.  If the text cannot be located
    (the model normalised or paraphrased it), the span is returned
    unaligned rather than discarded — string-level evaluation still works.

    Args:
        span_text: The surface form to locate.
        sentence: The sentence the span should occur in.
        start_from: Character offset to begin searching from.  Defaults to
            ``0`` (search the whole sentence).  Pass the end offset of a
            previously-aligned span to disambiguate repeated surface forms
            (sequential alignment of multiple spans in one sentence).

    Returns:
        A :class:`~aspectkit.schema.Span` with offsets when alignment
        succeeded, otherwise with ``start=end=None``.
    """
    needle = span_text.strip()
    if not needle:
        return Span(text=span_text)

    # Prefer a word-boundary match so "art" does not align inside "started".
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)
    match = pattern.search(sentence, start_from)
    if match is None:
        # Fall back to a plain substring search.
        lowered = sentence.lower()
        idx = lowered.find(needle.lower(), start_from)
        if idx == -1:
            return Span(text=needle)
        return Span(text=sentence[idx : idx + len(needle)], start=idx, end=idx + len(needle))
    return Span(text=match.group(0), start=match.start(), end=match.end())
