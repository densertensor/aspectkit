"""Target-sequence templates for generative (seq2seq) ABSA.

A template defines the mapping between a list of sentiment tuples and
the target string a sequence-to-sequence model is trained to produce —
and back.  Two schemes from the literature are provided:

* :class:`MarkersTemplate` — element markers ``[A] ... [C] ... [O] ...
  [S] ...`` per tuple, tuples joined by ``[SSEP]`` (the scheme
  popularised by MvP, Gou et al. 2023).  Adapts to any extraction view
  by dropping the markers of elements outside the view.
* :class:`ParaphraseTemplate` — the natural-language paraphrase of
  Zhang et al. (2021, ASQP): ``"<category> is <great|ok|bad> because
  <aspect> is <opinion>"``.  Defined for the full quadruple view only.

Both sides of the mapping use ``"null"`` for implicit span elements
(the ACOS file convention), so templates compose with the same
null-handling as the prompted backend.  Decoding is noise-tolerant:
malformed fragments are dropped and reported, never silently accepted.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from aspectkit.schema import SentimentTuple
from aspectkit.tasks import Task, get_task

__all__ = ["MarkersTemplate", "ParaphraseTemplate", "TupleTemplate"]

#: Separator between tuples in a target sequence (ASQP/MvP convention).
SSEP = "[SSEP]"

#: Target emitted/accepted for an example with no opinions.
NO_TUPLES = "none"

#: Stand-in for implicit span elements (matches the ACOS file convention).
NULL = "null"

# One decoded tuple: element name -> surface string ("null" for implicit).
Fragment = dict[str, str]


class TupleTemplate(ABC):
    """Mapping between sentiment tuples and a seq2seq target string.

    Subclasses implement one linearisation scheme.  Decoding returns
    surface strings only — converting them to canonical tuples (span
    alignment, label canonicalisation) is the backend's job, so all
    templates share one conversion path.

    Args:
        task: The extraction view whose elements the template covers.

    Raises:
        ValueError: If the task has given elements (classification-style
            views have no target sequence to generate).
    """

    def __init__(self, task: str | Task) -> None:
        resolved = get_task(task)
        if resolved.given:
            raise ValueError(
                f"templates linearise extraction views; task {resolved.name!r} has given "
                f"elements ({', '.join(sorted(resolved.given))}) and is not one"
            )
        self.task = resolved
        #: Elements rendered per tuple, in canonical order.
        self.elements: tuple[str, ...] = resolved.ordered_elements(resolved.predicted)

    @abstractmethod
    def encode(self, tuples: Sequence[SentimentTuple]) -> str:
        """Render gold tuples as the training target string.

        Raises:
            ValueError: If a tuple lacks a categorical element (category
                or polarity) that the view requires — silently writing a
                placeholder would teach the model to emit junk.
        """

    @abstractmethod
    def decode(self, target: str) -> tuple[list[Fragment], list[str]]:
        """Parse a generated target string back into element fragments.

        Returns:
            ``(fragments, problems)`` where each fragment maps every
            view element to its surface string (``"null"`` for implicit
            spans) and *problems* describes fragments that were dropped.
        """

    def _categorical_value(self, t: SentimentTuple, element: str) -> str:
        value = getattr(t, element)
        if value is None:
            raise ValueError(
                f"cannot encode a training tuple without a {element} under the "
                f"{self.task.name!r} view: {t!r}"
            )
        return str(value)

    def _span_value(self, t: SentimentTuple, element: str) -> str:
        text = t.aspect_text if element == "aspect" else t.opinion_text
        return text if text is not None else NULL

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task.name!r})"


class MarkersTemplate(TupleTemplate):
    """Element-marker linearisation (MvP-style).

    One tuple renders as ``[A] aspect [C] category [O] opinion [S]
    polarity`` (markers of elements outside the view are dropped);
    tuples are joined by ``[SSEP]``; an example with no opinions renders
    as ``none``.  Polarity is written as its canonical label.

    Example (ACOS view)::

        [A] spicy tuna roll [C] FOOD#QUALITY [O] good [S] positive
        [SSEP] [A] null [C] RESTAURANT#GENERAL [O] null [S] positive
    """

    _MARKERS: ClassVar[dict[str, str]] = {
        "aspect": "[A]",
        "category": "[C]",
        "opinion": "[O]",
        "polarity": "[S]",
    }
    _BY_LETTER: ClassVar[dict[str, str]] = {
        "A": "aspect",
        "C": "category",
        "O": "opinion",
        "S": "polarity",
    }
    _SPLIT = re.compile(r"\[([ACOS])\]")

    def encode(self, tuples: Sequence[SentimentTuple]) -> str:
        if not tuples:
            return NO_TUPLES
        parts = []
        for t in tuples:
            bits = []
            for element in self.elements:
                if element in ("aspect", "opinion"):
                    value = self._span_value(t, element)
                else:
                    value = self._categorical_value(t, element)
                bits.append(f"{self._MARKERS[element]} {value}")
            parts.append(" ".join(bits))
        return f" {SSEP} ".join(parts)

    def decode(self, target: str) -> tuple[list[Fragment], list[str]]:
        cleaned = target.strip()
        if not cleaned or cleaned.lower() == NO_TUPLES:
            return [], []
        fragments: list[Fragment] = []
        problems: list[str] = []
        for i, raw in enumerate(cleaned.split(SSEP)):
            pieces = self._SPLIT.split(raw)
            # pieces[0] is whatever precedes the first marker; model
            # warm-up text there carries no element value, so it is
            # tolerated.  pieces then alternate (letter, value).
            found: Fragment = {}
            duplicate = False
            for letter, value in zip(pieces[1::2], pieces[2::2], strict=True):
                element = self._BY_LETTER[letter]
                if element in found:
                    duplicate = True
                    break
                found[element] = value.strip()
            if duplicate:
                problems.append(f"fragment {i}: duplicate marker in {raw.strip()!r}")
                continue
            missing = [e for e in self.elements if e not in found]
            if missing:
                problems.append(f"fragment {i}: missing {', '.join(missing)} in {raw.strip()!r}")
                continue
            # Markers outside the view (a confused model) are ignored.
            fragments.append({e: found[e] for e in self.elements})
        return fragments, problems


class ParaphraseTemplate(TupleTemplate):
    """Natural-language paraphrase linearisation (ASQP-style).

    One quadruple renders as ``"<category> is <great|ok|bad> because
    <aspect> is <opinion>"`` with ``it`` standing in for implicit
    aspects/opinions, following Zhang et al. (2021).  Only defined for
    the full quadruple view (``acos``/``asqp``); recovery splits on
    ``" because "`` and ``" is "`` exactly as the original, so values
    containing ``" is "`` are a documented lossy edge case.

    Raises:
        ValueError: For views other than the full quadruple, or when
            encoding a polarity without a sentiment word (``conflict``
            — use :class:`MarkersTemplate` for 4-way schemes).
    """

    _WORD: ClassVar[dict[str, str]] = {"positive": "great", "neutral": "ok", "negative": "bad"}
    _POLARITY: ClassVar[dict[str, str]] = {word: polarity for polarity, word in _WORD.items()}
    _IMPLICIT = "it"

    def __init__(self, task: str | Task = "acos") -> None:
        super().__init__(task)
        if set(self.elements) != {"aspect", "category", "opinion", "polarity"}:
            raise ValueError(
                "the paraphrase template is defined for the full quadruple view "
                f"(acos/asqp), not {self.task.name!r}; use MarkersTemplate instead"
            )

    def encode(self, tuples: Sequence[SentimentTuple]) -> str:
        if not tuples:
            return NO_TUPLES
        parts = []
        for t in tuples:
            polarity = self._categorical_value(t, "polarity")
            if polarity not in self._WORD:
                raise ValueError(
                    f"the paraphrase template has no sentiment word for {polarity!r}; "
                    "use MarkersTemplate for schemes beyond positive/negative/neutral"
                )
            category = self._categorical_value(t, "category")
            aspect = t.aspect_text or self._IMPLICIT
            opinion = t.opinion_text or self._IMPLICIT
            for name, value in (("category", category), ("aspect", aspect), ("opinion", opinion)):
                if " is " in value:
                    warnings.warn(
                        f"ParaphraseTemplate: {name} {value!r} contains ' is ', which is also "
                        "the template delimiter — this tuple will not round-trip; use "
                        "MarkersTemplate for data containing ' is '.",
                        UserWarning,
                        stacklevel=2,
                    )
            parts.append(f"{category} is {self._WORD[polarity]} because {aspect} is {opinion}")
        return f" {SSEP} ".join(parts)

    def decode(self, target: str) -> tuple[list[Fragment], list[str]]:
        cleaned = target.strip()
        if not cleaned or cleaned.lower() == NO_TUPLES:
            return [], []
        fragments: list[Fragment] = []
        problems: list[str] = []
        for i, raw in enumerate(cleaned.split(SSEP)):
            fragment = raw.strip()
            if " because " not in fragment:
                problems.append(f"fragment {i}: no ' because ' in {fragment!r}")
                continue
            left, right = fragment.split(" because ", 1)
            if " is " not in left or " is " not in right:
                problems.append(f"fragment {i}: missing ' is ' in {fragment!r}")
                continue
            category, word = left.rsplit(" is ", 1)
            aspect, opinion = right.split(" is ", 1)
            polarity = self._POLARITY.get(word.strip().lower())
            if polarity is None:
                problems.append(f"fragment {i}: unknown sentiment word {word.strip()!r}")
                continue
            fragments.append(
                {
                    "aspect": self._null_if_implicit(aspect),
                    "category": category.strip(),
                    "opinion": self._null_if_implicit(opinion),
                    "polarity": polarity,
                }
            )
        return fragments, problems

    def _null_if_implicit(self, value: str) -> str:
        cleaned = value.strip()
        return NULL if cleaned.lower() == self._IMPLICIT else cleaned
