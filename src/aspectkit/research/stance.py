"""Entity-level stance analysis: aggregate ATSC predictions per entity.

A thin orchestrator over an ATSC backend and
:func:`~aspectkit.aggregate.summarize`: scan texts for entity aliases,
classify the stance toward each mentioned entity, and roll the results up
into one :class:`~aspectkit.aggregate.AspectSummary` per entity.  It is
*not* a new task — it composes the existing ATSC classification and
corpus-summary machinery.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from aspectkit.aggregate import AspectSummary, summarize
from aspectkit.normalize import normalize_text
from aspectkit.schema import ABSAExample, SentimentTuple, Span

if TYPE_CHECKING:
    from aspectkit.backends.base import Backend

__all__ = ["EntityStanceAnalyzer"]


class EntityStanceAnalyzer:
    """Classify and aggregate sentiment toward named entities.

    For each text, each entity whose alias appears (not flanked by word
    characters, optionally case-insensitive) becomes one ATSC ``(text,
    aspect)`` pair; the backend classifies its polarity, and the results are
    aggregated per entity.

    Args:
        backend: An ATSC backend (``task="atsc"``) classifying polarity
            given an aspect.
        entities: Mapping of entity name to the surface aliases to scan for,
            e.g. ``{"iPhone": ["iphone", "iphones"]}``.
        case_sensitive: Match aliases case-sensitively (default: no).
        min_mentions: Drop entities mentioned fewer times than this.

    Raises:
        ValueError: If *backend* is not an ATSC backend, or *entities* is empty.
    """

    def __init__(
        self,
        backend: Backend,
        entities: Mapping[str, Sequence[str]],
        *,
        case_sensitive: bool = False,
        min_mentions: int = 1,
    ) -> None:
        if backend.task.name != "atsc":
            raise ValueError(
                f"EntityStanceAnalyzer needs an ATSC backend, got task {backend.task.name!r}"
            )
        if not entities:
            raise ValueError("entities must be a non-empty {name: [aliases]} mapping")
        seen: dict[str, str] = {}
        for name in entities:
            key = normalize_text(name)
            if key in seen:
                raise ValueError(
                    "entity names must be distinct after normalisation; "
                    f"{seen[key]!r} and {name!r} both reduce to {key!r}"
                )
            seen[key] = name
        self.backend = backend
        self.case_sensitive = case_sensitive
        self.min_mentions = min_mentions
        flags = 0 if case_sensitive else re.IGNORECASE
        #: Per entity, the (alias, compiled boundary pattern) pairs to scan for.
        # ``(?<!\w)...(?!\w)`` matches whole tokens but, unlike ``\b``, also
        # matches aliases whose edges are non-word chars (e.g. "C++", ".NET").
        self._patterns: dict[str, list[tuple[str, re.Pattern[str]]]] = {
            name: [
                (alias, re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags)) for alias in aliases
            ]
            for name, aliases in entities.items()
        }

    def _first_alias(self, text: str, name: str) -> str | None:
        for alias, pattern in self._patterns[name]:
            if pattern.search(text):
                return alias
        return None

    def analyze(self, texts: Sequence[str]) -> dict[str, AspectSummary]:
        """Return one summary per mentioned entity, keyed by entity name.

        Entities not mentioned (or below ``min_mentions``) are absent.
        """
        mention_texts: list[str] = []
        examples: list[ABSAExample] = []
        entity_of: list[str] = []
        for text in texts:
            for name in self._patterns:
                alias = self._first_alias(text, name)
                if alias is not None:
                    mention_texts.append(text)
                    examples.append(
                        ABSAExample(text=text, tuples=[SentimentTuple(aspect=Span(alias))])
                    )
                    entity_of.append(name)
        if not examples:
            return {}
        predictions = self.backend.predict(examples)
        # Re-key each classified tuple under its entity name so the summary
        # groups all of an entity's aliases together.
        regrouped = [
            [replace(tpl, aspect=Span(name)) for tpl in preds]
            for name, preds in zip(entity_of, predictions, strict=True)
        ]
        summaries = summarize(mention_texts, regrouped, by="aspect", min_mentions=self.min_mentions)
        return {s.key: s for s in summaries}
