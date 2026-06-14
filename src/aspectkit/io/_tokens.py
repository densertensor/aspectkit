"""Helpers for token-indexed annotation formats (ASTE, ACOS).

Both formats annotate spans as indices into the whitespace-tokenised
sentence.  These helpers map token ranges back to character offsets in
the original line so the canonical schema keeps exact alignment.
"""

from __future__ import annotations

from aspectkit.exceptions import DataFormatError
from aspectkit.schema import Span

__all__ = ["span_from_token_range", "token_offsets"]


def token_offsets(text: str) -> list[tuple[int, int]]:
    """Character offsets ``[start, end)`` of each whitespace token in *text*."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in text.split():
        start = text.index(token, cursor)
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def span_from_token_range(
    text: str,
    offsets: list[tuple[int, int]],
    first: int,
    last: int,
    *,
    where: str = "",
) -> Span:
    """Build an offset-aligned span covering tokens ``first..last`` (inclusive).

    Args:
        text: The original sentence.
        offsets: Output of :func:`token_offsets` for ``text``.
        first: Index of the first token in the span.
        last: Index of the last token in the span (inclusive).
        where: Location string used in error messages.

    Raises:
        DataFormatError: If the token indices fall outside the sentence.
    """
    if not (0 <= first <= last < len(offsets)):
        raise DataFormatError(
            f"{where}: token range [{first}, {last}] out of bounds "
            f"for sentence with {len(offsets)} tokens"
        )
    start = offsets[first][0]
    end = offsets[last][1]
    return Span(text=text[start:end], start=start, end=end)
