"""Loader for ASTE-Data-V2 triplet files (Xu et al. 2020).

Format, one example per line::

    sentence####[([a_idx, ...], [o_idx, ...], 'POS'), ...]

where the index lists are word positions of the aspect and opinion spans
in the whitespace-tokenised sentence, and sentiment is ``POS``/``NEG``/``NEU``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.io._tokens import span_from_token_range, token_offsets
from aspectkit.normalize import canonical_polarity
from aspectkit.schema import ABSAExample, SentimentTuple

__all__ = ["read_aste"]

_SEPARATOR = "####"


def read_aste(path: str | Path) -> list[ABSAExample]:
    """Read an ASTE-Data-V2 file into the canonical schema.

    Each triplet becomes an ``(aspect, opinion, polarity)`` tuple with
    offset-aligned spans (``category`` is ``None``: the format does not
    annotate it).

    Args:
        path: Path to e.g. ``train_triplets.txt``.

    Raises:
        DataFormatError: On malformed lines, indices, or labels.
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
            text, _, triplet_part = line.partition(_SEPARATOR)
            try:
                triplets = ast.literal_eval(triplet_part.strip())
            except (ValueError, SyntaxError) as exc:
                raise DataFormatError(f"{where}: invalid triplet literal: {exc}") from exc
            if not isinstance(triplets, list):
                raise DataFormatError(f"{where}: triplet payload is not a list")

            offsets = token_offsets(text)
            tuples: list[SentimentTuple] = []
            for triplet in triplets:
                try:
                    aspect_idx, opinion_idx, label = triplet
                except (TypeError, ValueError) as exc:
                    raise DataFormatError(f"{where}: malformed triplet {triplet!r}") from exc
                if not aspect_idx or not opinion_idx:
                    raise DataFormatError(f"{where}: empty index list in {triplet!r}")
                try:
                    polarity = canonical_polarity(label)
                except ValueError as exc:
                    raise DataFormatError(f"{where}: {exc}") from exc
                tuples.append(
                    SentimentTuple(
                        aspect=span_from_token_range(
                            text, offsets, min(aspect_idx), max(aspect_idx), where=where
                        ),
                        opinion=span_from_token_range(
                            text, offsets, min(opinion_idx), max(opinion_idx), where=where
                        ),
                        polarity=polarity,
                    )
                )
            examples.append(ABSAExample(text=text, tuples=tuples))
    return examples
