"""Native JSON-Lines serialisation of the canonical schema.

One :class:`~aspectkit.schema.ABSAExample` per line.  Span elements are
objects ``{"text", "start", "end"}``, implicit elements are the string
``"IMPLICIT"``, and absent elements are ``null`` — three JSON types, no
ambiguity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from aspectkit.exceptions import DataFormatError
from aspectkit.schema import ABSAExample

__all__ = ["read_jsonl", "write_jsonl"]


def read_jsonl(path: str | Path) -> list[ABSAExample]:
    """Read examples from an aspectkit JSON-Lines file.

    Args:
        path: Path to a ``.jsonl`` file written by :func:`write_jsonl`
            (or conforming to the same schema).

    Raises:
        DataFormatError: If a line is not valid JSON or lacks required keys.
    """
    examples: list[ABSAExample] = []
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(ABSAExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise DataFormatError(f"{path}:{lineno}: {exc}") from exc
    return examples


def write_jsonl(examples: Iterable[ABSAExample], path: str | Path) -> None:
    """Write examples to a JSON-Lines file (UTF-8, one example per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False))
            handle.write("\n")
