"""Exception hierarchy for aspectkit.

All library-specific errors derive from :class:`AspectKitError`, so user
code can catch one type at integration boundaries.
"""

from __future__ import annotations

__all__ = [
    "AspectKitError",
    "DataFormatError",
    "LLMError",
    "MissingDependencyError",
    "ParseError",
]


class AspectKitError(Exception):
    """Base class for all aspectkit errors."""


class DataFormatError(AspectKitError):
    """A dataset file does not conform to the declared format."""


class LLMError(AspectKitError):
    """A chat-model call failed (transport error, refusal, empty reply)."""


class ParseError(AspectKitError):
    """Model output could not be parsed into the canonical schema."""


class MissingDependencyError(AspectKitError, ImportError):
    """An optional dependency required by the selected component is absent."""

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} requires the '{package}' package. "
            f"Install it with: pip install 'aspectkit[{extra}]' (or pip install {package})"
        )
        self.package = package
        self.extra = extra
