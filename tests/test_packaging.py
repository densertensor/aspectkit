"""Packaging sanity checks (PEP 561 marker, single-sourced version)."""

import re
from pathlib import Path

import aspectkit


def test_import_and_version():
    assert isinstance(aspectkit.__version__, str)
    assert re.match(r"^\d+\.\d+", aspectkit.__version__)  # PEP 440-ish


def test_py_typed_marker_shipped():
    # PEP 561: the marker must live inside the installed package
    marker = Path(aspectkit.__file__).parent / "py.typed"
    assert marker.is_file()
