"""Loader for the Twitter ``$T$`` format (Dong et al. 2014).

The three-lines-per-example layout used by the ACL-14 Twitter dataset and
by many APC dataset distributions::

    sentence with $T$ placeholder
    aspect term
    polarity (-1 negative, 0 neutral, 1 positive)

Note the integer convention differs from ACOS files (where 0 is
negative): this loader applies the Twitter mapping explicitly.
"""

from __future__ import annotations

from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.schema import ABSAExample, SentimentTuple, Span

__all__ = ["read_twitter"]

_PLACEHOLDER = "$T$"
_POLARITY = {"-1": "negative", "0": "neutral", "1": "positive"}


def read_twitter(path: str | Path) -> list[ABSAExample]:
    """Read a ``$T$``-format file into the canonical schema.

    The placeholder is substituted with the aspect term (every occurrence)
    and the first occurrence becomes the offset-aligned aspect span.

    Args:
        path: Path to e.g. ``train.raw``.

    Raises:
        DataFormatError: If the file length is not a multiple of three,
            a template lacks the placeholder, or a polarity code is
            unknown.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 3 != 0:
        raise DataFormatError(
            f"{path}: expected groups of 3 lines (sentence, aspect, polarity); "
            f"got {len(lines)} lines"
        )

    examples: list[ABSAExample] = []
    for index in range(0, len(lines), 3):
        where = f"{path}:{index + 1}"
        template = lines[index].strip()
        aspect = lines[index + 1].strip()
        code = lines[index + 2].strip()
        if _PLACEHOLDER not in template:
            raise DataFormatError(f"{where}: sentence lacks the {_PLACEHOLDER} placeholder")
        try:
            polarity = _POLARITY[code]
        except KeyError:
            raise DataFormatError(
                f"{where}: unknown polarity code {code!r} (expected -1, 0, or 1)"
            ) from None

        start = template.index(_PLACEHOLDER)
        text = template.replace(_PLACEHOLDER, aspect)
        examples.append(
            ABSAExample(
                text=text,
                tuples=[
                    SentimentTuple(
                        aspect=Span(text=aspect, start=start, end=start + len(aspect)),
                        polarity=polarity,
                    )
                ],
            )
        )
    return examples
