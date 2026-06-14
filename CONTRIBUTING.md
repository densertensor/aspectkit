# Contributing

Thanks for your interest in improving aspectkit. This guide covers the local
setup and the checks every change must pass.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.10+
pip install -e ".[dev]"                            # core + ruff, mypy, pytest
# optional extras for the model backends you touch:
pip install openai anthropic google-genai pandas
pip install torch --index-url https://download.pytorch.org/whl/cpu transformers
```

## Quality gate

Run all four before opening a pull request; CI runs the same on Python
3.10–3.13:

```bash
ruff format --check src tests
ruff check src tests
python -m mypy
python -m pytest
```

The suite is offline and fast: the integration tests build tiny,
randomly-initialised transformers in-process (guarded by
`pytest.importorskip("torch")`) rather than downloading models, and all provider
connectors are exercised through fakes.

## Conventions

- **The core stays dependency-free.** Model SDKs (`openai`, `anthropic`,
  `google-genai`, `transformers`/`torch`, `pandas`) are optional extras, imported
  lazily inside the components that need them — `import aspectkit` must never pull
  a heavy dependency.
- **New options are additive.** Prefer keyword-only parameters whose defaults
  reproduce existing behaviour; don't remove or reorder public signatures.
- **Everything is tested.** No stubs or dead code; unimplemented features raise
  `NotImplementedError` or do not exist. New behaviour ships with a test.
- **Exact match is the primary metric**, always reported; lenient overlap is
  opt-in and reported alongside, never instead.
- New subtasks are task-registry entries, not new code paths.

## Pull requests

Keep changes focused, update `CHANGELOG.md` under *Unreleased*, and make sure the
quality gate is green. Thank you!
