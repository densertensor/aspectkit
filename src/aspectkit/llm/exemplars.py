"""Pluggable few-shot exemplar selection for the prompted LLM backend.

By default the backend draws one seeded random set of exemplars at
:meth:`~aspectkit.backends.llm.LLMBackend.fit` time and reuses it for every
input.  This module exposes that choice as a strategy, so studies can compare
it against per-instance retrieval without monkey-patching private state: an
:class:`ExemplarSelector` maps a query plus a fitted :class:`ExemplarPool` to
the exemplars for that one input.

Selectors:

* :class:`NoneSelector` — zero-shot (no exemplars).
* :class:`RandomSelector` — a seeded random sample, identical for every query
  (reproduces the backend's historical behaviour).
* :class:`KNNSelector` — per-instance retrieval: the exemplars whose text is
  most similar to the query, scored by a dependency-free TF-IDF cosine
  (character 2-3-grams plus word tokens) or an optional embedding function.

Retrieval is offered *alongside* random and zero-shot, not as a replacement —
the maintainer's guiding principle is to expose the tools and let the study
design choose.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ExemplarPool",
    "ExemplarSelector",
    "KNNSelector",
    "NoneSelector",
    "RandomSelector",
]

_WORD = re.compile(r"\w+", re.UNICODE)


def _features(text: str) -> Counter[str]:
    """Bag of word tokens and character 2-/3-grams for TF-IDF retrieval."""
    tokens = _WORD.findall(text.lower())
    feats: Counter[str] = Counter("w:" + tok for tok in tokens)
    compact = " ".join(tokens)
    for n in (2, 3):
        for i in range(len(compact) - n + 1):
            feats["c:" + compact[i : i + n]] += 1
    return feats


class _TfidfIndex:
    """Minimal TF-IDF cosine index over a fixed document set (stdlib only)."""

    def __init__(self, documents: Sequence[str]) -> None:
        doc_features = [_features(doc) for doc in documents]
        df: Counter[str] = Counter()
        for feats in doc_features:
            df.update(feats.keys())
        n = len(doc_features) or 1
        self._idf = {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}
        self._vectors = [self._weight(feats) for feats in doc_features]
        self._norms = [math.sqrt(sum(w * w for w in vec.values())) for vec in self._vectors]

    def _weight(self, feats: Counter[str]) -> dict[str, float]:
        return {term: tf * self._idf.get(term, 0.0) for term, tf in feats.items()}

    def scores(self, query: str) -> list[float]:
        """Cosine similarity of *query* to every indexed document."""
        qv = self._weight(_features(query))
        qnorm = math.sqrt(sum(w * w for w in qv.values()))
        out: list[float] = []
        for vec, norm in zip(self._vectors, self._norms, strict=True):
            if qnorm == 0.0 or norm == 0.0:
                out.append(0.0)
                continue
            small, large = (qv, vec) if len(qv) <= len(vec) else (vec, qv)
            dot = sum(w * large.get(term, 0.0) for term, w in small.items())
            out.append(dot / (qnorm * norm))
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class ExemplarPool:
    """A fitted set of candidate exemplars and the text each is retrieved by.

    Args:
        items: The exemplar objects — whole examples for extraction,
            ``(text, aspect, polarity)`` triples for classification.
        texts: The parallel strings a retriever matches the query against
            (the sentence text of each item).

    The TF-IDF index (and any embedding vectors) are built once on first use
    and cached, so retrieving exemplars for a whole corpus indexes the pool
    a single time.
    """

    def __init__(self, items: Sequence[Any], texts: Sequence[str]) -> None:
        self.items = list(items)
        self.texts = list(texts)
        if len(self.items) != len(self.texts):
            raise ValueError(
                f"items and texts must be parallel; got {len(self.items)} and {len(self.texts)}"
            )
        self._tfidf: _TfidfIndex | None = None
        self._embeddings: list[Sequence[float]] | None = None
        self._embeddings_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def tfidf_scores(self, query: str) -> list[float]:
        """Cosine similarity of *query* to each item under the cached index."""
        if self._tfidf is None:
            self._tfidf = _TfidfIndex(self.texts)
        return self._tfidf.scores(query)

    def embedding_scores(
        self, query: str, encode_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]]
    ) -> list[float]:
        """Cosine similarity under a user embedding function (pool cached).

        The pool vectors are cached per ``encode_fn``: passing a different
        encoder recomputes them, so reusing one pool across encoders never
        mixes incompatible embedding spaces.
        """
        if self._embeddings is None or self._embeddings_fn is not encode_fn:
            self._embeddings = list(encode_fn(self.texts))
            self._embeddings_fn = encode_fn
        query_vec = next(iter(encode_fn([query])))
        return [_cosine(query_vec, vec) for vec in self._embeddings]


@runtime_checkable
class ExemplarSelector(Protocol):
    """Strategy mapping a query and a fitted pool to chosen exemplars."""

    def select(self, pool: ExemplarPool, query: str, n: int) -> list[Any]:
        """Return up to *n* exemplars from *pool* for *query*."""
        ...


class NoneSelector:
    """Zero-shot: never selects any exemplar."""

    def select(self, pool: ExemplarPool, query: str, n: int) -> list[Any]:
        return []


class RandomSelector:
    """A seeded random sample, identical for every query.

    Reproduces the backend's historical behaviour: one fixed sample of the
    pool, reused for all inputs.  The query is ignored.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def select(self, pool: ExemplarPool, query: str, n: int) -> list[Any]:
        if n <= 0 or len(pool) == 0:
            return []
        if len(pool) <= n:
            return list(pool.items)
        return random.Random(self.seed).sample(pool.items, n)


class KNNSelector:
    """Per-instance retrieval: the *n* exemplars most similar to the query.

    Similarity is a dependency-free TF-IDF cosine by default (character
    2-/3-grams plus word tokens).  Pass *encode_fn* — a function mapping a
    list of texts to a list of equal-length numeric vectors — to retrieve by
    embedding cosine instead; the pool's vectors are computed once and cached.

    Args:
        encode_fn: Optional batch embedding function for retrieval.
    """

    def __init__(
        self,
        *,
        encode_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self.encode_fn = encode_fn

    def select(self, pool: ExemplarPool, query: str, n: int) -> list[Any]:
        if n <= 0 or len(pool) == 0:
            return []
        if self.encode_fn is None:
            scores = pool.tfidf_scores(query)
        else:
            scores = pool.embedding_scores(query, self.encode_fn)
        # Highest score first; ties broken by original order for determinism.
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        return [pool.items[i] for i in order[:n]]
