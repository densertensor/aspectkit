"""Loaders for the SemEval ABSA benchmark XML formats.

Two distinct schemas exist in the SemEval lineage:

* **SemEval-2014 Task 4** (laptops, restaurants): sentence-level
  ``<aspectTerms>`` (term + polarity + offsets) and ``<aspectCategories>``
  (category + polarity).  Crucially, the two annotation layers are *not
  linked* to each other — a term carries no category and vice versa.
* **SemEval-2015 Task 12 / SemEval-2016 Task 5**: review-level documents
  whose sentences carry unified ``<Opinions>`` with
  ``target``/``category``/``polarity`` (+ offsets); ``target="NULL"``
  marks an implicit aspect.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from aspectkit.exceptions import DataFormatError
from aspectkit.normalize import canonical_polarity
from aspectkit.schema import IMPLICIT, ABSAExample, Implicit, SentimentTuple, Span

__all__ = ["read_semeval_2014", "read_semeval_2015"]


def _parse(path: str | Path) -> ET.Element:
    try:
        return ET.parse(Path(path)).getroot()
    except ET.ParseError as exc:
        raise DataFormatError(f"{path}: not well-formed XML: {exc}") from exc


def _offsets(element: ET.Element) -> tuple[int | None, int | None]:
    """Read SemEval ``from``/``to`` character offsets, if present and sane."""
    raw_from, raw_to = element.get("from"), element.get("to")
    if raw_from is None or raw_to is None:
        return None, None
    start, end = int(raw_from), int(raw_to)
    if end <= start:  # "0"/"0" is used for NULL targets in 2015/16 data
        return None, None
    return start, end


def read_semeval_2014(
    path: str | Path,
    annotations: Literal["terms", "categories", "both"] = "terms",
) -> list[ABSAExample]:
    """Read a SemEval-2014 Task 4 XML file.

    Because the 2014 term and category layers are unlinked, mixing them
    into one tuple would fabricate (aspect, category) pairs that were
    never annotated.  The loader therefore keeps them apart:

    * ``annotations="terms"`` (default) yields tuples with ``aspect`` and
      ``polarity`` (offset-aligned spans) — the ATE/ATSC/E2E view.
    * ``annotations="categories"`` yields tuples with ``category`` and
      ``polarity`` (aspect is :data:`~aspectkit.schema.IMPLICIT`) — the
      ACD/ACSA view.
    * ``annotations="both"`` yields both kinds of tuples side by side,
      still unlinked.

    Args:
        path: Path to e.g. ``Restaurants_Train.xml``.
        annotations: Which annotation layer(s) to convert.

    Returns:
        One :class:`~aspectkit.schema.ABSAExample` per ``<sentence>``.
    """
    if annotations not in ("terms", "categories", "both"):
        raise ValueError("annotations must be 'terms', 'categories', or 'both'")
    root = _parse(path)
    examples: list[ABSAExample] = []
    for sentence in root.iter("sentence"):
        text_el = sentence.find("text")
        if text_el is None or text_el.text is None:
            raise DataFormatError(f"{path}: <sentence> without <text>")
        text = text_el.text
        tuples: list[SentimentTuple] = []

        if annotations in ("terms", "both"):
            for term in sentence.iterfind("aspectTerms/aspectTerm"):
                start, end = _offsets(term)
                surface = term.get("term", "")
                # Trust offsets over the attribute text when both exist.
                if start is not None and end is not None:
                    surface = text[start:end]
                tuples.append(
                    SentimentTuple(
                        aspect=Span(text=surface, start=start, end=end),
                        polarity=canonical_polarity(term.get("polarity", "")),
                    )
                )
        if annotations in ("categories", "both"):
            for cat in sentence.iterfind("aspectCategories/aspectCategory"):
                tuples.append(
                    SentimentTuple(
                        aspect=IMPLICIT,
                        category=cat.get("category"),
                        polarity=canonical_polarity(cat.get("polarity", "")),
                    )
                )
        examples.append(ABSAExample(text=text, tuples=tuples, id=sentence.get("id")))
    return examples


def read_semeval_2015(path: str | Path) -> list[ABSAExample]:
    """Read a SemEval-2015 Task 12 / SemEval-2016 Task 5 XML file.

    Each sentence's ``<Opinion>`` elements become
    ``(aspect, category, polarity)`` tuples; ``target="NULL"`` maps to
    :data:`~aspectkit.schema.IMPLICIT`.  The review id is preserved in
    ``ABSAExample.meta["review_id"]``.

    Args:
        path: Path to e.g. ``ABSA16_Restaurants_Train_SB1.xml``.

    Returns:
        One :class:`~aspectkit.schema.ABSAExample` per ``<sentence>``.
        Sentences without ``<Opinions>`` are kept with empty tuples
        (they are genuine "no opinion" negatives, not noise).
    """
    root = _parse(path)
    examples: list[ABSAExample] = []
    for review in root.iter("Review"):
        review_id = review.get("rid")
        for sentence in review.iter("sentence"):
            text_el = sentence.find("text")
            if text_el is None or text_el.text is None:
                raise DataFormatError(f"{path}: <sentence> without <text>")
            text = text_el.text
            tuples: list[SentimentTuple] = []
            for opinion in sentence.iterfind("Opinions/Opinion"):
                target = opinion.get("target")
                aspect: Span | Implicit
                if target is None or target == "NULL":
                    aspect = IMPLICIT
                else:
                    start, end = _offsets(opinion)
                    surface = text[start:end] if start is not None else target
                    aspect = Span(text=surface, start=start, end=end)
                tuples.append(
                    SentimentTuple(
                        aspect=aspect,
                        category=opinion.get("category"),
                        polarity=canonical_polarity(opinion.get("polarity", "")),
                    )
                )
            meta = {"review_id": review_id} if review_id else {}
            examples.append(ABSAExample(text=text, tuples=tuples, id=sentence.get("id"), meta=meta))
    return examples
