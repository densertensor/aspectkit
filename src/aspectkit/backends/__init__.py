"""Prediction backends.

* :class:`LLMBackend` — structured extraction with any chat model
  (hosted APIs, OpenAI-compatible local servers, in-process
  ``transformers`` objects).  Full tuple views with zero training.
* :class:`Seq2SeqBackend` — fine-tunable generative extraction
  (T5/BART family); the backend family holding the published state of
  the art on exact-match tuple extraction.  Target linearisations live
  in :mod:`~aspectkit.backends.templates`.
* :class:`PairClassifierBackend` — ATSC with a fine-tuned (and
  fine-tunable) cross-encoder checkpoint (the empirically stronger
  choice when aspects are given).

All backends implement the :class:`Backend` interface and exchange data
exclusively in the canonical schema.
"""

from __future__ import annotations

from aspectkit.backends.base import Backend
from aspectkit.backends.llm import LLMBackend
from aspectkit.backends.pair import PairClassifierBackend
from aspectkit.backends.seq2seq import Seq2SeqBackend
from aspectkit.backends.templates import MarkersTemplate, ParaphraseTemplate, TupleTemplate

__all__ = [
    "Backend",
    "LLMBackend",
    "MarkersTemplate",
    "PairClassifierBackend",
    "ParaphraseTemplate",
    "Seq2SeqBackend",
    "TupleTemplate",
]
