# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial release (0.1.0).

### Added

- Canonical schema (`Span`, `IMPLICIT`, `SentimentTuple`, `ABSAExample`) and a
  task registry covering `ate`, `atsc`, `acd`, `acsa`, `e2e`, `aste`, `tasd`,
  `acos`, and `document` (whole-text sentiment).
- Dataset loaders: native JSON-Lines, SemEval 2014/MAMS and 2015/2016 XML, ASTE,
  ACOS, ASQP, Twitter `$T$`, and CoNLL/ATEPC BIO; plus CSV/JSON/pandas/HF-dataset
  interop.
- Backends behind the `ABSA` facade:
  - `LLMBackend` — prompted extraction over any chat connector, with pluggable
    few-shot exemplar selection (`none`/`random`/`knn`, TF-IDF retrieval),
    prompt hooks (`instructions=`, `system_prompt_fn=`), `n_samples`
    self-consistency, parallel `predict(return_confidence=True)`, concurrency,
    and opt-in `retry=`/`cache_dir=`/`on_progress=`.
  - `Seq2SeqBackend` and `PairClassifierBackend` — fine-tunable extraction and
    ATSC classification, with `fit(val_examples=, patience=, save_best=)` and
    pair-classifier softmax confidence.
- Composable connector wrappers `RetryingChat`, `CachingChat`, `CountingChat`,
  and resumable corpus prediction via `predict_with_checkpoint` /
  `write_jsonl(append=)`.
- Evaluation: exact-match (primary) and lenient overlap tuple P/R/F1, per-element
  breakdown, classification accuracy/macro-F1, and
  `EvaluationReport.to_methods_text()`.
- Aggregation: `summarize()` corpus rollups with optional grouping, time windows,
  Wilson/bootstrap confidence intervals, and two-proportion z-tests.
- Opt-in `aspectkit.research`: `EntityStanceAnalyzer` and `validate()`
  (inter-rater Cohen's κ).
- PEP 561 typing marker (`py.typed`): the library ships its type information.

### Notes

- `ABSAExample` is a frozen dataclass: in-place `.tuples`/`.meta` mutation still
  works, but reassigning whole fields does not.
- `write_jsonl` gained a keyword-only `append=` parameter (default overwrites, as
  before).
