"""Dataset loaders and custom-data interop.

Each loader converts one published format into the canonical
:class:`~aspectkit.schema.ABSAExample` schema, eliminating the
dataset-version chaos documented across the ABSA literature: one
``to_canonical`` per format, used everywhere.

Benchmark file formats:

* :func:`read_semeval_2014` — SemEval-2014 Task 4 XML (``aspectTerms`` /
  ``aspectCategories``); also reads MAMS, which reuses the schema.
* :func:`read_semeval_2015` — SemEval-2015 Task 12 / SemEval-2016 Task 5
  XML (``Opinions`` with ``target``/``category``/``polarity``).
* :func:`read_aste` — ASTE-Data-V2 triplet files.
* :func:`read_acos` — ACOS quadruple TSV files.
* :func:`read_asqp` — ASQP/Paraphrase/MvP-style quadruple text files.
* :func:`read_twitter` — Dong et al. 2014 ``$T$`` three-line format.
* :func:`read_bio` — CoNLL-style BIO token files (ATE / PyABSA ATEPC /
  unified E2E tags).

Custom data and framework interop:

* :func:`read_jsonl` / :func:`write_jsonl` — aspectkit's native format.
* :func:`read_csv` / :func:`read_json` — flat or nested records with
  remappable columns.
* :func:`from_records` / :func:`from_pandas` / :func:`from_hf_dataset` —
  in-memory conversion from dicts, DataFrames, or Hugging Face datasets.
* :func:`to_records` / :func:`to_pandas` — export back out for analysis.
* :func:`predict_with_checkpoint` — run a backend over a corpus with an
  on-disk checkpoint so an interrupted run resumes without re-predicting.
"""

from __future__ import annotations

from pathlib import Path

from aspectkit.io.acos import read_acos
from aspectkit.io.asqp import read_asqp
from aspectkit.io.aste import read_aste
from aspectkit.io.bio import read_bio
from aspectkit.io.checkpoint import predict_with_checkpoint
from aspectkit.io.jsonl import read_jsonl, write_jsonl
from aspectkit.io.records import (
    from_hf_dataset,
    from_pandas,
    from_records,
    read_csv,
    read_json,
    to_pandas,
    to_records,
)
from aspectkit.io.semeval import read_semeval_2014, read_semeval_2015
from aspectkit.io.twitter import read_twitter
from aspectkit.schema import ABSAExample

__all__ = [
    "from_hf_dataset",
    "from_pandas",
    "from_records",
    "load_examples",
    "predict_with_checkpoint",
    "read_acos",
    "read_asqp",
    "read_aste",
    "read_bio",
    "read_csv",
    "read_json",
    "read_jsonl",
    "read_semeval_2014",
    "read_semeval_2015",
    "read_twitter",
    "to_pandas",
    "to_records",
    "write_jsonl",
]

_READERS = {
    "jsonl": read_jsonl,
    "json": read_json,
    "csv": read_csv,
    "semeval2014": read_semeval_2014,
    "mams": read_semeval_2014,  # MAMS reuses the 2014 XML schema
    "semeval2015": read_semeval_2015,
    "semeval2016": read_semeval_2015,  # 2016 reuses the 2015 schema
    "aste": read_aste,
    "acos": read_acos,
    "asqp": read_asqp,
    "twitter": read_twitter,
    "bio": read_bio,
    "conll": read_bio,
    "atepc": read_bio,
}


def load_examples(path: str | Path, format: str, **kwargs) -> list[ABSAExample]:
    """Load a dataset file by format name.

    Args:
        path: Path to the dataset file.
        format: One of ``jsonl``, ``json``, ``csv``, ``semeval2014``,
            ``mams``, ``semeval2015``, ``semeval2016``, ``aste``,
            ``acos``, ``asqp``, ``twitter``, ``bio`` (aliases ``conll``,
            ``atepc``).
        **kwargs: Passed through to the format-specific reader (e.g.
            ``annotations="categories"`` for SemEval-2014, column
            mappings for ``csv``/``json``).

    Returns:
        Examples in the canonical schema.
    """
    key = format.strip().lower().replace("-", "").replace("_", "")
    try:
        reader = _READERS[key]
    except KeyError:
        raise ValueError(
            f"unknown format {format!r}; available: {', '.join(sorted(_READERS))}"
        ) from None
    return reader(path, **kwargs)
