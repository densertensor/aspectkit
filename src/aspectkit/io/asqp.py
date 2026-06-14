"""Loader for ASQP-style quadruple text files (Zhang et al. 2021).

The format used by the generative-ABSA lineage (ASQP/Paraphrase, MvP and
descendants), one example per line::

    sentence####[['aspect', 'category', 'polarity', 'opinion'], ...]

Element order inside each label list is **[aspect term, aspect category,
sentiment polarity, opinion term]**; ``'NULL'`` marks implicit aspects or
opinions.  Categories in these files are typically lowercase with spaces
(``'service general'``) and are kept verbatim — evaluation matching is
case-insensitive.
"""

from __future__ import annotations

import ast
from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.normalize import align_span, canonical_polarity
from aspectkit.schema import IMPLICIT, ABSAExample, Implicit, SentimentTuple, Span

__all__ = ["read_asqp"]

_SEPARATOR = "####"


def _span_or_implicit(value: str, text: str) -> Span | Implicit:
    if value.strip().upper() == "NULL":
        return IMPLICIT
    return align_span(value, text)


def read_asqp(path: str | Path) -> list[ABSAExample]:
    """Read an ASQP quadruple file into the canonical schema.

    Each label becomes a full ``(aspect, category, opinion, polarity)``
    tuple; explicit spans are offset-aligned against the sentence where
    possible.

    Args:
        path: Path to e.g. ``rest15/train.txt`` from an ASQP-format
            dataset distribution.

    Raises:
        DataFormatError: On malformed lines or labels.
    """
    path = Path(path)
    examples: list[ABSAExample] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            if _SEPARATOR not in line:
                raise DataFormatError(f"{where}: missing '{_SEPARATOR}' separator")
            text, _, label_part = line.partition(_SEPARATOR)
            text = text.strip()
            try:
                labels = ast.literal_eval(label_part.strip())
            except (ValueError, SyntaxError) as exc:
                raise DataFormatError(f"{where}: invalid label literal: {exc}") from exc
            if not isinstance(labels, list):
                raise DataFormatError(f"{where}: label payload is not a list")

            tuples: list[SentimentTuple] = []
            for label in labels:
                if not isinstance(label, (list, tuple)) or len(label) != 4:
                    raise DataFormatError(
                        f"{where}: expected [aspect, category, polarity, opinion], got {label!r}"
                    )
                aspect_str, category, polarity_str, opinion_str = (str(v) for v in label)
                try:
                    polarity = canonical_polarity(polarity_str)
                except ValueError as exc:
                    raise DataFormatError(f"{where}: {exc}") from exc
                tuples.append(
                    SentimentTuple(
                        aspect=_span_or_implicit(aspect_str, text),
                        category=category,
                        opinion=_span_or_implicit(opinion_str, text),
                        polarity=polarity,
                    )
                )
            examples.append(ABSAExample(text=text, tuples=tuples))
    return examples
