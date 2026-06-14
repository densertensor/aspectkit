"""Loader for ACOS quadruple files (Cai et al. 2021).

Format, one example per line, fields separated by tabs::

    sentence<TAB>a_start,a_end CATEGORY#LABEL sentiment o_start,o_end<TAB>...

* Span fields are token ranges ``start,end`` (end-exclusive) into the
  whitespace-tokenised sentence; ``-1,-1`` marks an implicit element.
* Sentiment is an integer: ``0`` negative, ``1`` neutral, ``2`` positive.
* The same layout is used by the ASQP datasets distributed in this style.
"""

from __future__ import annotations

from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.io._tokens import span_from_token_range, token_offsets
from aspectkit.normalize import canonical_polarity
from aspectkit.schema import IMPLICIT, ABSAExample, Implicit, SentimentTuple, Span

__all__ = ["read_acos"]


def _parse_span(
    field: str, text: str, offsets: list[tuple[int, int]], where: str
) -> Span | Implicit:
    try:
        start_str, end_str = field.split(",")
        start, end = int(start_str), int(end_str)
    except ValueError as exc:
        raise DataFormatError(f"{where}: invalid span field {field!r}") from exc
    if start == -1 and end == -1:
        return IMPLICIT
    if end <= start:
        raise DataFormatError(f"{where}: empty token range {field!r}")
    # ACOS ranges are end-exclusive; span_from_token_range takes an
    # inclusive last index.
    return span_from_token_range(text, offsets, start, end - 1, where=where)


def read_acos(path: str | Path) -> list[ABSAExample]:
    """Read an ACOS/ASQP quadruple TSV file into the canonical schema.

    Each quadruple becomes a full ``(aspect, category, opinion, polarity)``
    tuple; implicit aspects and opinions map to
    :data:`~aspectkit.schema.IMPLICIT`.

    Args:
        path: Path to e.g. ``rest16_quad_train.tsv``.

    Raises:
        DataFormatError: On malformed lines, spans, or labels.
    """
    path = Path(path)
    examples: list[ABSAExample] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            text, *quads = line.split("\t")
            if not quads:
                raise DataFormatError(f"{where}: no quadruples on line")
            offsets = token_offsets(text)
            tuples: list[SentimentTuple] = []
            for quad in quads:
                parts = quad.split()
                if len(parts) != 4:
                    raise DataFormatError(
                        f"{where}: expected 4 space-separated fields per quad, "
                        f"got {len(parts)} in {quad!r}"
                    )
                aspect_field, category, sentiment, opinion_field = parts
                try:
                    polarity = canonical_polarity(sentiment)
                except ValueError as exc:
                    raise DataFormatError(f"{where}: {exc}") from exc
                tuples.append(
                    SentimentTuple(
                        aspect=_parse_span(aspect_field, text, offsets, where),
                        category=category,
                        opinion=_parse_span(opinion_field, text, offsets, where),
                        polarity=polarity,
                    )
                )
            examples.append(ABSAExample(text=text, tuples=tuples))
    return examples
