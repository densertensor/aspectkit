"""Tests for pluggable exemplar selection (no network)."""

import random

import pytest

from aspectkit.llm.exemplars import (
    ExemplarPool,
    ExemplarSelector,
    KNNSelector,
    NoneSelector,
    RandomSelector,
)

# A small pool whose texts cluster into two clear topics.
TEXTS = [
    "the pasta and risotto were delicious",
    "great food, the pizza was tasty",
    "the waiter was rude and slow",
    "terrible service, we waited forever",
]
POOL = ExemplarPool(items=list(TEXTS), texts=list(TEXTS))


class TestProtocol:
    def test_builtins_satisfy_protocol(self):
        for selector in (NoneSelector(), RandomSelector(), KNNSelector()):
            assert isinstance(selector, ExemplarSelector)

    def test_pool_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="parallel"):
            ExemplarPool(items=[1, 2], texts=["only one"])


class TestNoneSelector:
    def test_returns_nothing(self):
        assert NoneSelector().select(POOL, "the pizza was great", 5) == []


class TestRandomSelector:
    def test_same_sample_for_every_query(self):
        sel = RandomSelector(seed=7)
        a = sel.select(POOL, "query one", 2)
        b = sel.select(POOL, "a totally different query", 2)
        assert a == b  # query is ignored — one fixed sample
        assert len(a) == 2

    def test_seed_determinism_and_variation(self):
        big = ExemplarPool(items=list(range(20)), texts=[str(i) for i in range(20)])
        assert RandomSelector(seed=1).select(big, "", 5) == RandomSelector(seed=1).select(
            big, "", 5
        )
        assert RandomSelector(seed=1).select(big, "", 5) != RandomSelector(seed=2).select(
            big, "", 5
        )

    def test_matches_random_sample_semantics(self):
        # Must reproduce the backend's historical fit() sampling exactly.
        items = list(range(20))
        pool = ExemplarPool(items=list(items), texts=[str(i) for i in items])
        expected = random.Random(42).sample(items, 8)
        assert RandomSelector(seed=42).select(pool, "ignored", 8) == expected

    def test_returns_all_when_pool_small(self):
        small = ExemplarPool(items=[1, 2], texts=["a", "b"])
        assert RandomSelector(seed=3).select(small, "q", 8) == [1, 2]  # original order, no sample


class TestKNNSelector:
    def test_retrieves_topically_similar(self):
        # A food query should retrieve the two food exemplars, service first dropped.
        picked = KNNSelector().select(POOL, "the pizza tasted amazing", 2)
        assert "great food, the pizza was tasty" in picked
        assert all("waiter" not in p and "service" not in p for p in picked)

    def test_respects_n_and_order(self):
        picked = KNNSelector().select(POOL, "the waiter was very slow", 1)
        assert picked == ["the waiter was rude and slow"]  # single best match

    def test_empty_pool_and_zero_n(self):
        empty = ExemplarPool(items=[], texts=[])
        assert KNNSelector().select(empty, "q", 3) == []
        assert KNNSelector().select(POOL, "q", 0) == []

    def test_deterministic_tiebreak_by_order(self):
        # Two identical texts tie on score; the earlier index wins deterministically.
        pool = ExemplarPool(items=["A", "B"], texts=["same words here", "same words here"])
        assert KNNSelector().select(pool, "same words here", 1) == ["A"]

    def test_embedding_encode_fn(self):
        # Embed each text by [length, 1.0]; cosine then ranks by closeness in
        # length, so the query retrieves the item of matching length.
        pool = ExemplarPool(items=["x", "yy", "zzzz"], texts=["x", "yy", "zzzz"])

        def encode(texts):
            return [[float(len(t)), 1.0] for t in texts]

        assert KNNSelector(encode_fn=encode).select(pool, "zzzz", 1) == ["zzzz"]

    def test_embedding_cache_not_mixed_across_encoders(self):
        # Reusing one pool with a different encoder must recompute the pool
        # vectors, not reuse the first encoder's (which would mix spaces).
        pool = ExemplarPool(items=["a", "b"], texts=["a", "b"])

        def enc1(texts):  # a -> [1, 0], b -> [0, 1]
            return [[1.0, 0.0] if t == "a" else [0.0, 1.0] for t in texts]

        def enc2(texts):  # axes swapped: a -> [0, 1], b -> [1, 0]
            return [[0.0, 1.0] if t == "a" else [1.0, 0.0] for t in texts]

        assert KNNSelector(encode_fn=enc1).select(pool, "a", 1) == ["a"]
        assert KNNSelector(encode_fn=enc2).select(pool, "a", 1) == ["a"]
