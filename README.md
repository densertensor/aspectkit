# aspectkit

**A configurable, evaluation-centric framework for aspect-based sentiment
analysis (ABSA).**

aspectkit gives researchers one coherent API over the whole ABSA task family —
from aspect term extraction to full **(aspect, category, opinion, polarity)**
quadruples with implicit elements — with swappable model backends, loaders for
the standard benchmarks, the exact-match evaluation protocol built in, and the
corpus-level opinion summary as a first-class output.

```python
from aspectkit import ABSA

absa = ABSA(task="acos", backend="llm", model="openai:gpt-4o-mini",
            categories=["FOOD#QUALITY", "SERVICE#GENERAL"])

absa.predict("The pasta was great but we waited forever.")
# [SentimentTuple(aspect=Span(text='pasta', start=4, end=9), polarity='positive',
#                 category='FOOD#QUALITY', opinion=Span(text='great', start=14, end=19)),
#  SentimentTuple(aspect=IMPLICIT, polarity='negative', category='SERVICE#GENERAL',
#                 opinion=Span(text='waited forever', start=27, end=41))]
```

## Why aspectkit

- **One schema, every subtask.** ATE, ATSC, ACD, ACSA, E2E, ASTE, TASD, and
  ACOS/ASQP are all declarative *views* (which elements are given, which are
  predicted) over one canonical tuple — so backends, datasets, and metrics
  compose instead of forking.
- **Implicit aspects and opinions are first-class.** Roughly a third of real
  review sentiment has no explicit target span; `IMPLICIT` is part of the data
  model, distinct from "not annotated".
- **Evaluation is the point, not an afterthought.** Exact-match tuple
  P/R/F1 (the SemEval comparability standard) is always computed; a lenient
  token-overlap mode can be reported *alongside* it; multi-element tasks get a
  per-element breakdown showing which element drives the misses; gold labels
  are stripped before the backend ever sees evaluation inputs.
- **LLM-era backends without lock-in.** One small connector interface spans
  hosted APIs, OpenAI-compatible local servers, and models already loaded in
  your notebook.
- **Fine-tuning when prompting is not enough.** A generative seq2seq backend
  (the model family holding the published state of the art on exact-match
  quad extraction) and a trainable ATSC cross-encoder, with plain,
  transparent training loops.
- **The output researchers actually want.** `summarize()` rolls predictions up
  into per-aspect sentiment distributions, net scores, and representative
  quotes — the aspect-based opinion summary ABSA was invented for.

## Installation

```bash
pip install aspectkit                  # core: zero dependencies
pip install "aspectkit[openai]"        # OpenAI + OpenAI-compatible endpoints
pip install "aspectkit[anthropic]"     # Anthropic models
pip install "aspectkit[gemini]"        # Google Gemini
pip install "aspectkit[transformers]"  # local Hugging Face models (+ torch)
pip install "aspectkit[all]"
```

Python 3.10+. The core library (schema, loaders, evaluation, aggregation) has
**no dependencies**; provider SDKs are optional extras imported lazily, so you
install only what your chosen backend needs.

## Connecting a model

Every chat model is addressed the same way — a `"provider:model"` string or a
live object:

| Spec | Connector |
|---|---|
| `"openai:gpt-4o-mini"` | OpenAI API (`OPENAI_API_KEY`) |
| `"anthropic:claude-opus-4-8"` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `"gemini:gemini-2.0-flash"` | Google Gemini (`GEMINI_API_KEY`) |
| `"deepseek:deepseek-chat"` | DeepSeek (`DEEPSEEK_API_KEY`) |
| `"vllm:<served-model>"` | local vLLM server (`http://localhost:8000/v1`) |
| `"ollama:llama3.1"` | local Ollama server |
| `"mistral:..."`, `"together:..."`, `"groq:..."`, `"openrouter:..."` | other OpenAI-compatible providers |
| `"openai-compatible:<model>"` + `base_url=...` | any OpenAI-protocol endpoint |
| `"hf:Qwen/Qwen2.5-7B-Instruct"` | Hugging Face hub id, loaded as a local pipeline |
| a `transformers` **pipeline object** | used in-process, as-is |
| a `(model, tokenizer)` **pair** | used in-process, as-is |
| any `messages -> str` **callable** | custom gateways, caching layers, test doubles |

The notebook workflow needs no ceremony — pass the object you already have:

```python
from transformers import pipeline
from aspectkit import ABSA

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct")
absa = ABSA(task="aste", backend="llm", model=pipe)
```

Connector behaviour worth knowing: generation defaults are deterministic
(temperature 0 where the provider allows it), JSON schemas are enforced
natively where the provider supports structured output, and protocol-dialect
quirks of OpenAI-compatible servers (unsupported `response_format`,
`max_completion_tokens`, rejected `temperature`) are detected, downgraded
once, and remembered.

For corpus-scale runs against hosted APIs, predict examples in parallel
(order is preserved, failures behave per `on_error`):

```python
absa = ABSA(task="acos", backend="llm", model="openai:gpt-4o-mini",
            concurrency=8, on_error="skip")
```

## Tasks

| name | given | predicted |
|---|---|---|
| `ate` | — | aspect |
| `atsc` (`asc`, `apc`) | aspect | polarity |
| `acd` | — | category |
| `acsa` | — | category, polarity |
| `e2e` (`atepc`) | — | aspect, polarity |
| `aste` | — | aspect, opinion, polarity |
| `tasd` | — | aspect, category, polarity |
| `acos` (`asqp`, `quad`) | — | aspect, category, opinion, polarity |

## Loading data: benchmarks, custom files, frameworks

Every published benchmark format has a loader, verified against the official
distributions:

```python
from aspectkit import load_examples

rest14 = load_examples("Restaurants_Train.xml", "semeval2014")   # terms/categories views
mams   = load_examples("train.xml", "mams")                      # same XML schema
rest16 = load_examples("ABSA16_Restaurants_Train_SB1.xml", "semeval2016")
quads  = load_examples("laptop_quad_train.tsv", "acos")          # token-span TSV
asqp   = load_examples("rest15/train.txt", "asqp")               # generative-ABSA txt
triples = load_examples("train_triplets.txt", "aste")            # ASTE-Data-V2
tweets = load_examples("train.raw", "twitter")                   # Dong et al. $T$ format
tagged = load_examples("Laptops_Train.atepc", "atepc")           # BIO/CoNLL token files
```

Each loader converts one published format into the canonical schema — spans
become character offsets, `NULL`/`-1,-1` targets become `IMPLICIT`, integer
polarity codes are mapped per each format's own convention (ACOS `0` is
negative, Twitter `0` is neutral — the loaders know) — so the dataset-version
chaos stays out of your experiment code.

**Custom datasets** come in through one remappable interface, whatever shape
they're in — nested records (one item per text) or flat rows (one opinion per
row, the usual CSV/DataFrame layout, grouped by id or text automatically):

```python
from aspectkit.io import from_records, from_pandas, from_hf_dataset, read_csv, read_json

examples = read_csv("reviews.csv", text="review", aspect="term", polarity="label")
examples = from_pandas(df)                       # pandas DataFrame
examples = from_hf_dataset(ds["train"])          # Hugging Face datasets
examples = from_records([{"text": "...", "tuples": [{"aspect": "...", "polarity": "POS"}]}])
```

And back out for analysis: `to_pandas(examples)` flattens gold data or
predictions into one row per opinion; `to_records(...)`/`read_json` round-trip
losslessly.

## Few-shot, evaluation, and the corpus summary

```python
absa = ABSA(task="acos", backend="llm", model="anthropic:claude-opus-4-8",
            categories=CATEGORIES)

absa.fit(train_examples)            # seeded few-shot exemplar selection (optional)

report = absa.evaluate(test_examples, lenient=True)
print(report)
# EvaluationReport(task=acos, n_examples=583)
#   exact match   P=0.61  R=0.57  F1=0.59  (pred=802, gold=843)
#   lenient       P=0.68  R=0.64  F1=0.66
#   by element (exact):
#     aspect      P=0.74  R=0.69  F1=0.71
#     category    P=0.81  R=0.76  F1=0.78
#     opinion     P=0.69  R=0.64  F1=0.66
#     polarity    P=0.88  R=0.82  F1=0.85

summary = absa.summarize(corpus, by="category", min_mentions=5)
for s in summary[:3]:
    print(s)
# FOOD#QUALITY: n=412, score=+0.55 (negative=71, neutral=44, positive=297)
# SERVICE#GENERAL: n=259, score=-0.18 (...)
```

A methodological note baked into the design: prompted LLMs are convenient and
strong at simple polarity, but the benchmark literature consistently shows
them trailing fine-tuned models on exact-match tuple extraction. Validate the
LLM backend on a labelled sample with `evaluate()` before trusting corpus-level
output — the API makes that the path of least resistance.

For polarity-given-aspect (ATSC), the empirically stronger default is the
fine-tuned cross-encoder backend:

```python
absa = ABSA(task="atsc", backend="pair")   # yangheng/deberta-v3-base-absa-v1.1
```

## Fine-tuning

When labelled data is available, the fine-tuned route is the strong one.  The
seq2seq backend trains a T5/BART-family model to *generate* linearised tuples
— either MvP-style element markers (`[A] pasta [C] FOOD#QUALITY [O] great
[S] positive`, any task view) or the ASQP paraphrase template (`"FOOD#QUALITY
is great because pasta is great"`, quad view):

```python
absa = ABSA(task="acos", backend="seq2seq", model="t5-base",
            categories=CATEGORIES)            # style="markers" by default

absa.fit(train_examples, epochs=20)           # plain AdamW loop, seeded
print(absa.backend.history_[-1])              # mean loss of the last epoch

report = absa.evaluate(test_examples)
absa.backend.save_pretrained("runs/acos-t5")  # reload via model="runs/acos-t5"
```

A fresh `t5-base` knows nothing about the templates: **fit before predict**.
The same `fit()` recipe (epochs, learning rate, seeded shuffling, loss
history) fine-tunes the ATSC cross-encoder:

```python
absa = ABSA(task="atsc", backend="pair")
absa.fit(train_examples, epochs=3)            # 2e-5 AdamW, the BERT recipe
```

## The data model

```python
from aspectkit import ABSAExample, SentimentTuple, Span, IMPLICIT

ABSAExample(
    text="Would not recommend.",
    tuples=[SentimentTuple(aspect=IMPLICIT,                 # no surface target
                           category="RESTAURANT#GENERAL",
                           opinion=Span("Would not recommend", 0, 19),
                           polarity="negative")],
)
```

- Spans carry character offsets `[start, end)`; offsets are optional, so
  generative outputs remain first-class citizens.
- `IMPLICIT` ≠ `None`: implicit means *expressed without a surface span*;
  `None` means *not part of this task's annotation*.
- An empty `tuples` list means "no opinions", which is not the same as neutral.
- Everything round-trips through JSONL (`aspectkit.io.write_jsonl` /
  `read_jsonl`).

## Extending

Custom strategies implement the two-method `Backend` interface (`fit`,
`predict`) over canonical examples and plug straight into the facade:

```python
from aspectkit import ABSA
from aspectkit.backends import Backend

class MyBackend(Backend):
    ...

absa = ABSA(backend=MyBackend(...))
```

Custom chat endpoints implement `ChatLLM.complete()` — or are just passed as a
callable.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
