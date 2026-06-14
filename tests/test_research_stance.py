"""Tests for EntityStanceAnalyzer (no network, scripted ATSC backend)."""

from dataclasses import replace

import pytest

from aspectkit.backends.base import Backend
from aspectkit.research import EntityStanceAnalyzer
from aspectkit.schema import ABSAExample
from aspectkit.tasks import get_task


class FakeATSC(Backend):
    """ATSC backend that labels each (text, aspect) by the aspect surface text."""

    def __init__(self, polarity_by_aspect, task_name="atsc"):
        self.task = get_task(task_name)
        self.polarity_by_aspect = polarity_by_aspect
        self.seen: list[ABSAExample] = []

    def predict(self, examples):
        self.seen = list(examples)
        return [
            [
                replace(t, polarity=self.polarity_by_aspect.get(t.aspect.text, "neutral"))
                for t in ex.tuples
            ]
            for ex in examples
        ]


ENTITIES = {"Acme": ["acme", "acme corp"], "Globex": ["globex"]}


class TestAnalyze:
    def test_mention_detection_and_routing(self):
        backend = FakeATSC({"acme": "positive", "globex": "negative"})
        analyzer = EntityStanceAnalyzer(backend, ENTITIES)
        analyzer.analyze(
            ["Acme is great", "I hate globex", "acme corp and globex are ok", "nothing here"]
        )
        # one ATSC example per (text, mentioned entity); whole-word, case-insensitive
        assert [ex.tuples[0].aspect.text for ex in backend.seen] == [
            "acme",
            "globex",
            "acme",
            "globex",
        ]

    def test_per_entity_aggregation(self):
        backend = FakeATSC({"acme": "positive", "globex": "negative"})
        result = EntityStanceAnalyzer(backend, ENTITIES).analyze(
            ["Acme is great", "I hate globex", "acme corp and globex are ok"]
        )
        assert sorted(result) == ["Acme", "Globex"]
        assert result["Acme"].n_mentions == 2
        assert result["Acme"].counts == {"positive": 2}
        assert result["Acme"].score == 1.0
        assert result["Globex"].score == -1.0

    def test_whole_word_matching(self):
        backend = FakeATSC({"cat": "positive"})
        result = EntityStanceAnalyzer(backend, {"Cat": ["cat"]}).analyze(["the category page"])
        assert result == {}  # "cat" must not match inside "category"

    def test_case_sensitive(self):
        backend = FakeATSC({"acme": "positive"})
        result = EntityStanceAnalyzer(backend, {"Acme": ["acme"]}, case_sensitive=True).analyze(
            ["Acme rocks"]
        )
        assert result == {}  # "acme" != "Acme" when case-sensitive

    def test_min_mentions_filter(self):
        backend = FakeATSC({"acme": "positive", "globex": "negative"})
        result = EntityStanceAnalyzer(backend, ENTITIES, min_mentions=2).analyze(
            ["Acme is great", "Acme again", "globex once"]
        )
        assert sorted(result) == ["Acme"]  # Globex has only 1 mention

    def test_no_mentions_returns_empty(self):
        backend = FakeATSC({})
        assert EntityStanceAnalyzer(backend, ENTITIES).analyze(["nothing relevant"]) == {}

    def test_non_word_char_alias_matches(self):
        # "\b" would miss an alias whose edges are non-word chars; (?<!\w)..(?!\w) doesn't
        backend = FakeATSC({"C++": "positive"})
        result = EntityStanceAnalyzer(backend, {"Cpp": ["C++"]}).analyze(["I love C++ a lot"])
        assert result["Cpp"].score == 1.0


class TestConstruction:
    def test_non_atsc_backend_rejected(self):
        backend = FakeATSC({}, task_name="acos")
        with pytest.raises(ValueError, match="ATSC"):
            EntityStanceAnalyzer(backend, ENTITIES)

    def test_empty_entities_rejected(self):
        with pytest.raises(ValueError, match="entities"):
            EntityStanceAnalyzer(FakeATSC({}), {})

    def test_normalized_name_collision_rejected(self):
        with pytest.raises(ValueError, match="distinct"):
            EntityStanceAnalyzer(FakeATSC({}), {"Apple": ["apple"], "APPLE ": ["mac"]})
