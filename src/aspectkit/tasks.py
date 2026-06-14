"""Task views over the canonical schema.

A *task* declares which sentiment elements are given as input and which
must be predicted.  Everything else in the library (prompt construction,
evaluation protocol, backend validation) is driven by this declaration,
so adding a new subtask is a one-line registry entry rather than a code
path.

The registry covers the standard subtask taxonomy:

========  ======================  ==========================================
name      output                  introduced by
========  ======================  ==========================================
ate       aspect                  SemEval-2014 Task 4 (SB1)
atsc      polarity (aspect given) SemEval-2014 Task 4 (SB2)
acd       category                SemEval-2014 Task 4 (SB3)
acsa      (category, polarity)    SemEval-2014 Task 4 (SB4)
e2e       (aspect, polarity)      Li et al. 2019 (E2E-ABSA)
aste      (aspect, opinion, pol.) Peng et al. 2020
tasd      (aspect, category, pol.) Wan et al. 2020
acos      full quadruple          Cai et al. 2021 / Zhang et al. 2021 (ASQP)
========  ======================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ELEMENTS", "TASKS", "Task", "get_task"]

#: The four sentiment elements, in canonical order.
ELEMENTS: tuple[str, ...] = ("aspect", "category", "opinion", "polarity")


@dataclass(frozen=True)
class Task:
    """A declarative view over :class:`~aspectkit.schema.SentimentTuple`.

    Attributes:
        name: Canonical task name (lowercase).
        given: Elements provided as input alongside the text.
        predicted: Elements the model must produce.
        description: One-line human-readable summary.
        aliases: Alternative names accepted by :func:`get_task`.
    """

    name: str
    given: frozenset[str]
    predicted: frozenset[str]
    description: str
    aliases: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        unknown = (self.given | self.predicted) - set(ELEMENTS)
        if unknown:
            raise ValueError(f"unknown elements {sorted(unknown)}; expected subset of {ELEMENTS}")
        if self.given & self.predicted:
            raise ValueError("an element cannot be both given and predicted")
        if not self.predicted:
            raise ValueError("a task must predict at least one element")

    @property
    def elements(self) -> frozenset[str]:
        """All elements involved in the task (given and predicted)."""
        return self.given | self.predicted

    @property
    def is_classification(self) -> bool:
        """True when the task classifies given targets rather than
        extracting new ones (drives the evaluation protocol)."""
        return bool(self.given) and self.predicted == frozenset({"polarity"})

    def ordered_elements(self, which: frozenset[str] | None = None) -> tuple[str, ...]:
        """Return elements in canonical order (defaults to all elements)."""
        pool = self.elements if which is None else which
        return tuple(e for e in ELEMENTS if e in pool)


def _task(
    name: str,
    given: tuple[str, ...],
    predicted: tuple[str, ...],
    description: str,
    aliases: tuple[str, ...] = (),
) -> Task:
    return Task(
        name=name,
        given=frozenset(given),
        predicted=frozenset(predicted),
        description=description,
        aliases=aliases,
    )


_REGISTRY: tuple[Task, ...] = (
    _task("ate", (), ("aspect",), "Aspect term extraction"),
    _task(
        "atsc",
        ("aspect",),
        ("polarity",),
        "Aspect-term sentiment classification (polarity given the aspect)",
        aliases=("asc", "apc", "alsc"),
    ),
    _task("acd", (), ("category",), "Aspect category detection"),
    _task(
        "acsa",
        (),
        ("category", "polarity"),
        "Aspect-category sentiment analysis",
        aliases=("acsc",),
    ),
    _task(
        "e2e",
        (),
        ("aspect", "polarity"),
        "End-to-end ABSA: joint aspect extraction and polarity",
        aliases=("e2e-absa", "atepc", "aesc", "uabsa"),
    ),
    _task(
        "aste",
        (),
        ("aspect", "opinion", "polarity"),
        "Aspect sentiment triplet extraction",
    ),
    _task(
        "tasd",
        (),
        ("aspect", "category", "polarity"),
        "Target-aspect-sentiment detection",
    ),
    _task(
        "acos",
        (),
        ("aspect", "category", "opinion", "polarity"),
        "Aspect-category-opinion-sentiment quadruple extraction "
        "(implicit aspects/opinions included)",
        aliases=("asqp", "quad", "quadruple"),
    ),
    _task(
        "document",
        (),
        ("polarity",),
        "Document-level sentiment: the overall polarity of the whole text "
        "(output tuple carries an implicit aspect)",
        aliases=("docsa", "documentsentiment"),
    ),
)

#: Registry of built-in tasks, keyed by canonical name.
TASKS: dict[str, Task] = {t.name: t for t in _REGISTRY}

_ALIASES: dict[str, str] = {alias: t.name for t in _REGISTRY for alias in t.aliases}


def get_task(task: str | Task) -> Task:
    """Resolve a task name (or alias, case-insensitive) to a :class:`Task`.

    Args:
        task: A task name such as ``"acos"``, an alias such as ``"asqp"``,
            or an already-resolved :class:`Task` (returned unchanged, which
            lets users define custom views).

    Raises:
        ValueError: If the name is not registered.
    """
    if isinstance(task, Task):
        return task
    key = task.strip().lower().replace("_", "-")
    key = {"e2e-absa": "e2e"}.get(key, key).replace("-", "")
    # canonical names/aliases contain no separators after this point
    if key in TASKS:
        return TASKS[key]
    if key in _ALIASES:
        return TASKS[_ALIASES[key]]
    available = sorted(set(TASKS) | set(_ALIASES))
    raise ValueError(f"unknown task {task!r}; available: {', '.join(available)}")
