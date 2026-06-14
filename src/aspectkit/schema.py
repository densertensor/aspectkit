"""Canonical data model for aspect-based sentiment analysis.

Every ABSA subtask (ATE, ATSC, E2E, ASTE, TASD, ACOS/ASQP, ...) is a
projection of one canonical record: a sentence paired with a set of
sentiment tuples ``(aspect, category, opinion, polarity)``.  This module
defines that record once, so loaders, backends, evaluation, and
aggregation all speak the same language.

Conventions
-----------
* Spans are character offsets ``[start, end)`` into ``ABSAExample.text``.
  Offsets are optional: tuples produced by generative models may carry
  surface text only.
* ``IMPLICIT`` marks an element that is expressed but has no surface span
  (e.g. *"Would not recommend."* has an implicit aspect).  It is distinct
  from ``None``, which means "not part of this annotation/task".
* An empty ``tuples`` list means the sentence expresses no opinions; it
  does **not** mean "neutral".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "IMPLICIT",
    "POLARITIES",
    "ABSAExample",
    "Implicit",
    "SentimentTuple",
    "Span",
    "is_implicit",
]

#: Canonical polarity labels.  Loaders map dataset-specific labels
#: (``POS``, ``2``, ...) onto these; see :func:`aspectkit.normalize.canonical_polarity`.
POLARITIES: tuple[str, ...] = ("positive", "negative", "neutral", "conflict")


class Implicit:
    """Singleton sentinel for implicit aspects/opinions.

    Use the module-level :data:`IMPLICIT` instance; do not instantiate.
    """

    _instance: Implicit | None = None

    def __new__(cls) -> Implicit:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "IMPLICIT"

    def __copy__(self) -> Implicit:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Implicit:
        return self

    def __reduce__(self) -> tuple[type[Implicit], tuple[()]]:
        return (Implicit, ())


#: The implicit-element sentinel.  ``aspect=IMPLICIT`` reads as "the
#: sentence targets an aspect that is not realised as a span of text".
IMPLICIT = Implicit()


def is_implicit(value: Any) -> bool:
    """Return ``True`` if *value* is the :data:`IMPLICIT` sentinel."""
    return isinstance(value, Implicit)


@dataclass(frozen=True)
class Span:
    """A contiguous span of surface text, optionally aligned to offsets.

    Attributes:
        text: The surface form of the span.
        start: Character offset of the first character in the parent
            sentence, or ``None`` if the span is unaligned.
        end: Character offset one past the last character, or ``None``.
    """

    text: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("Span offsets must be given together or not at all")
        if (
            self.start is not None
            and self.end is not None
            and (self.start < 0 or self.end <= self.start)
        ):
            raise ValueError(f"invalid span offsets [{self.start}, {self.end})")

    @property
    def aligned(self) -> bool:
        """Whether the span carries character offsets."""
        return self.start is not None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(text=data["text"], start=data.get("start"), end=data.get("end"))


def _element_to_json(value: Span | Implicit | None) -> Any:
    """Serialise a span-valued element: dict for a span, the string
    ``"IMPLICIT"`` for the sentinel, ``None`` for an absent element."""
    if value is None:
        return None
    if isinstance(value, Span):
        return value.to_dict()
    return "IMPLICIT"


def _element_from_json(value: Any) -> Span | Implicit | None:
    if value is None:
        return None
    if value == "IMPLICIT":
        return IMPLICIT
    return Span.from_dict(value)


@dataclass(frozen=True)
class SentimentTuple:
    """One opinion expressed in a sentence.

    The four sentiment elements of Zhang et al. (2022).  Elements that a
    task does not annotate are ``None``; elements that are expressed but
    have no surface form are :data:`IMPLICIT`.

    Attributes:
        aspect: The opinion target span, or :data:`IMPLICIT`.  Never
            ``None``: a tuple without an aspect annotation should still
            use ``IMPLICIT`` (e.g. category-only tasks), because every
            opinion has a target even when the dataset does not mark one.
        category: An ``ENTITY#ATTRIBUTE`` label from a domain taxonomy
            (e.g. ``FOOD#QUALITY``), or ``None`` if the task has no
            category inventory.
        opinion: The opinion-expression span, :data:`IMPLICIT`, or
            ``None`` if the task does not annotate opinion terms.
        polarity: One of :data:`POLARITIES`, or ``None`` for views that
            do not annotate polarity (e.g. plain aspect term extraction).
    """

    aspect: Span | Implicit
    polarity: str | None = None
    category: str | None = None
    opinion: Span | Implicit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aspect, (Span, Implicit)):
            raise ValueError("aspect must be a Span or IMPLICIT, not None")
        if self.polarity is not None and self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES} or None, got {self.polarity!r}")

    @property
    def aspect_text(self) -> str | None:
        """Surface text of the aspect, or ``None`` if implicit."""
        return self.aspect.text if isinstance(self.aspect, Span) else None

    @property
    def opinion_text(self) -> str | None:
        """Surface text of the opinion, or ``None`` if implicit/absent."""
        return self.opinion.text if isinstance(self.opinion, Span) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": _element_to_json(self.aspect),
            "category": self.category,
            "opinion": _element_to_json(self.opinion),
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentimentTuple:
        aspect = _element_from_json(data["aspect"])
        if aspect is None:
            aspect = IMPLICIT
        return cls(
            aspect=aspect,
            category=data.get("category"),
            opinion=_element_from_json(data.get("opinion")),
            polarity=data.get("polarity"),
        )


@dataclass(frozen=True)
class ABSAExample:
    """A sentence (or short text unit) with its sentiment tuples.

    Attributes:
        text: The raw text.  ABSA annotation is sentence-level; feeding
            whole documents to sentence-level models is a documented
            failure mode, so keep units short.
        tuples: The opinions expressed in ``text``.  An empty list means
            "no opinions", which is *not* the same as neutral.
        id: Optional stable identifier (e.g. the SemEval sentence id).
        meta: Free-form metadata (source file, review id, language, ...).
    """

    text: str
    tuples: list[SentimentTuple] = field(default_factory=list)
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "tuples": [t.to_dict() for t in self.tuples],
        }
        if self.id is not None:
            data["id"] = self.id
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ABSAExample:
        return cls(
            text=data["text"],
            tuples=[SentimentTuple.from_dict(t) for t in data.get("tuples", [])],
            id=data.get("id"),
            meta=data.get("meta", {}),
        )
