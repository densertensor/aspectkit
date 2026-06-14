"""Tests for the task registry."""

import pytest

from aspectkit.tasks import ELEMENTS, TASKS, Task, get_task


class TestRegistry:
    def test_all_standard_tasks_present(self):
        assert set(TASKS) == {"ate", "atsc", "acd", "acsa", "e2e", "aste", "tasd", "acos"}

    def test_acos_predicts_all_elements(self):
        assert get_task("acos").predicted == frozenset(ELEMENTS)

    def test_atsc_view(self):
        task = get_task("atsc")
        assert task.given == frozenset({"aspect"})
        assert task.predicted == frozenset({"polarity"})
        assert task.is_classification

    def test_extraction_tasks_are_not_classification(self):
        for name in ("ate", "acd", "acsa", "e2e", "aste", "tasd", "acos"):
            assert not get_task(name).is_classification, name


class TestGetTask:
    @pytest.mark.parametrize(
        ("alias", "name"),
        [
            ("asqp", "acos"),
            ("quad", "acos"),
            ("asc", "atsc"),
            ("apc", "atsc"),
            ("atepc", "e2e"),
            ("e2e-absa", "e2e"),
            ("E2E_ABSA", "e2e"),
            ("ACOS", "acos"),
            ("  aste  ", "aste"),
        ],
    )
    def test_aliases(self, alias, name):
        assert get_task(alias).name == name

    def test_task_instance_passthrough(self):
        task = get_task("acos")
        assert get_task(task) is task

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="unknown task"):
            get_task("sentiment")

    def test_error_lists_options(self):
        with pytest.raises(ValueError, match="acos"):
            get_task("nope")


class TestTaskValidation:
    def test_unknown_element_rejected(self):
        with pytest.raises(ValueError, match="unknown elements"):
            Task("x", frozenset(), frozenset({"emotion"}), "bad")

    def test_overlapping_given_predicted_rejected(self):
        with pytest.raises(ValueError):
            Task("x", frozenset({"aspect"}), frozenset({"aspect"}), "bad")

    def test_must_predict_something(self):
        with pytest.raises(ValueError):
            Task("x", frozenset({"aspect"}), frozenset(), "bad")

    def test_ordered_elements_canonical_order(self):
        task = get_task("acos")
        assert task.ordered_elements() == ("aspect", "category", "opinion", "polarity")
        assert task.ordered_elements(frozenset({"polarity", "aspect"})) == ("aspect", "polarity")
