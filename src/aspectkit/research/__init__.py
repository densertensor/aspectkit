"""Research-grade, opt-in extensions (not imported by the top-level package).

These compose the core primitives into higher-level study tools.  Import
them explicitly — keeping ``import aspectkit`` lean::

    from aspectkit.research import EntityStanceAnalyzer, validate, ValidationReport
"""

from __future__ import annotations

from aspectkit.research.stance import EntityStanceAnalyzer
from aspectkit.research.validation import ValidationReport, validate

__all__ = ["EntityStanceAnalyzer", "ValidationReport", "validate"]
