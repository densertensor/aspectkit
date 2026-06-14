"""Loader for CoNLL-style BIO token files (ATE / ATEPC / E2E tagging).

Covers the token-per-line family used by sequence-tagging ABSA:

* plain BIO for aspect term extraction::

      The     O
      battery B-ASP
      life    I-ASP

* PyABSA-style ATEPC with a polarity column (``-100``/``-999`` mark
  unlabelled positions; sentences with *k* aspects are conventionally
  repeated *k* times, each repeat labelling one aspect — consecutive
  repeats are merged back into one example by default)::

      cord    B-ASP  Neutral
      battery B-ASP  -100

* unified E2E tags that encode polarity in the tag suffix
  (Li et al. 2019)::

      battery B-POS
      life    I-POS

Sentences are separated by blank lines.  The example text is the tokens
joined by single spaces, and aspect spans carry offsets into that text.
"""

from __future__ import annotations

from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.io._tokens import span_from_token_range, token_offsets
from aspectkit.normalize import canonical_polarity
from aspectkit.schema import ABSAExample, SentimentTuple

__all__ = ["read_bio"]

#: Column-3 values meaning "no polarity annotated at this position".
_NULL_LABELS = frozenset({"-100", "-999", "O", ""})

#: Tag suffixes that encode polarity directly (unified E2E tagging).
_SUFFIX_POLARITY = {
    "POS": "positive",
    "NEG": "negative",
    "NEU": "neutral",
    "CON": "conflict",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "CONFLICT": "conflict",
}

_SpanInfo = tuple[int, int, str | None]  # first token, last token, polarity


def _resolve_polarity(suffix: str, labels: list[str | None], where: str) -> str | None:
    """Polarity of one span: tag suffix first, then the label column."""
    encoded = _SUFFIX_POLARITY.get(suffix.upper())
    if encoded is not None:
        return encoded
    for label in labels:
        if label is None or label.strip() in _NULL_LABELS:
            continue
        try:
            return canonical_polarity(label)
        except ValueError as exc:
            raise DataFormatError(f"{where}: {exc}") from exc
    return None


def _parse_block(
    rows: list[tuple[str, str, str | None]], where: str
) -> tuple[list[str], list[_SpanInfo]]:
    tokens = [token for token, _, _ in rows]
    spans: list[_SpanInfo] = []
    current: list[int] | None = None
    current_suffix = ""
    current_labels: list[str | None] = []

    def close() -> None:
        nonlocal current
        if current is not None:
            polarity = _resolve_polarity(current_suffix, current_labels, where)
            spans.append((current[0], current[-1], polarity))
            current = None

    for position, (_, tag, label) in enumerate(rows):
        upper = tag.upper()
        if upper == "O":
            close()
            continue
        if len(upper) < 3 or upper[1] != "-" or upper[0] not in "BI":
            raise DataFormatError(f"{where}: unrecognised tag {tag!r}")
        prefix, suffix = upper[0], tag[2:]
        starts_new = prefix == "B" or current is None or suffix.upper() != current_suffix.upper()
        if starts_new:
            close()
            current = [position]
            current_suffix = suffix
            current_labels = [label]
        else:
            assert current is not None
            current.append(position)
            current_labels.append(label)
    close()
    return tokens, spans


def _merge_span_lists(base: list[_SpanInfo], extra: list[_SpanInfo]) -> list[_SpanInfo]:
    """Union span lists from repeated copies of one sentence, preferring
    annotated polarity over the ``-100`` placeholder."""
    merged: dict[tuple[int, int], str | None] = {}
    order: list[tuple[int, int]] = []
    for first, last, polarity in [*base, *extra]:
        key = (first, last)
        if key not in merged:
            merged[key] = polarity
            order.append(key)
        elif merged[key] is None:
            merged[key] = polarity
    return [(first, last, merged[(first, last)]) for first, last in order]


def read_bio(path: str | Path, *, merge_repeats: bool = True) -> list[ABSAExample]:
    """Read a BIO/CoNLL token file into the canonical schema.

    Args:
        path: Path to the token file (e.g. ``*.atepc``, ``train.conll``).
        merge_repeats: Merge consecutive blocks with identical tokens
            into one example, combining their polarity annotations —
            undoing the one-repeat-per-aspect convention of ATEPC files.
            Set to ``False`` to keep blocks exactly as written.

    Returns:
        One example per sentence, with offset-aligned aspect spans and
        polarity where annotated (``None`` for plain ATE data).

    Raises:
        DataFormatError: On unrecognised tags, malformed lines, or
            unmappable polarity labels.
    """
    path = Path(path)
    blocks: list[tuple[list[str], list[_SpanInfo]]] = []
    rows: list[tuple[str, str, str | None]] = []
    block_line = 1

    def flush(where: str) -> None:
        nonlocal rows
        if not rows:
            return
        tokens, spans = _parse_block(rows, where)
        if merge_repeats and blocks and blocks[-1][0] == tokens:
            blocks[-1] = (tokens, _merge_span_lists(blocks[-1][1], spans))
        else:
            blocks.append((tokens, spans))
        rows = []

    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                flush(f"{path}:{block_line}")
                block_line = lineno + 1
                continue
            fields = line.split()
            if len(fields) == 2:
                rows.append((fields[0], fields[1], None))
            elif len(fields) == 3:
                rows.append((fields[0], fields[1], fields[2]))
            else:
                raise DataFormatError(
                    f"{path}:{lineno}: expected 'token TAG [polarity]', got {line!r}"
                )
    flush(f"{path}:{block_line}")

    examples: list[ABSAExample] = []
    for tokens, spans in blocks:
        text = " ".join(tokens)
        offsets = token_offsets(text)
        tuples = [
            SentimentTuple(
                aspect=span_from_token_range(text, offsets, first, last, where=str(path)),
                polarity=polarity,
            )
            for first, last, polarity in spans
        ]
        examples.append(ABSAExample(text=text, tuples=tuples))
    return examples
