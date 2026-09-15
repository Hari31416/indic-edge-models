# Indic browser embeddings

Build a compact embedding model for Indic text retrieval that runs entirely inside a browser tab.

The first release should demonstrate either better retrieval at a comparable download, memory, and latency budget, or comparable retrieval with a substantially smaller footprint. Specializing in Indic languages is a hypothesis to test, not evidence of an improvement.

## Status

E5-small is selected for the next bounded training pilot. The [Romanization v2 experiment](experiments/2026-09-15-romanization-v2/README.md) repaired script conversion and training/validation leakage, regenerated versioned data, and reran E5. Native nDCG@10 is 0.5513; corrected synthetic Romanized nDCG@10 is 0.0. Training preparation and browser validation remain pending.

Start with [STEPS.md](STEPS.md). It describes the initial sequence and the decisions that depend on results. Update it after each experiment rather than treating it as a fixed implementation plan.

Agent workflow is in [AGENTS.md](AGENTS.md). Run outputs and resulting decisions belong in [experiments/](experiments/README.md).

## Scope

The model maps queries and passages into vectors for local search. Evaluate these capabilities separately where suitable public data exists:

- Native-script queries retrieving passages in the same language.
- Romanized queries retrieving native-script passages.
- Hindi-English or other Indic-English code-mixed queries.
- Queries and passages in different languages, including English.

Two initial hypotheses are worth testing independently:

1. Targeted fine-tuning improves retrieval for Romanized or code-mixed queries without unacceptable native-script or English regressions.
2. Restricting language coverage and compressing the model reduces browser cost while preserving useful retrieval quality.

English plus Hindi is a provisional starting point. Final language selection should account for public evaluation data and the ability to inspect examples.

## Constraints

- One developer, using a Mac M4 Pro and short Kaggle sessions with two 16 GB T4 GPUs.
- Public datasets and existing checkpoints. No private corpus collection or large annotation effort.
- Start from an existing model rather than pretraining a foundation model.
- All inference in the released artifact must run locally in the browser, including any query preprocessing models.
- A useful model and reproducible evaluation come first. A small browser demonstration supports validation.

Reranking, translation, speech, and a chatbot are separate projects. A reranker may follow once retrieval measurements justify it.

## Starting candidates

The prior assessment identified these candidates for a bounded baseline comparison:

- `intfloat/multilingual-e5-small`
- `ibm-granite/granite-embedding-97m-multilingual-r2`
- `hotchpotch/bekko-embedding-v1-a8m`

These are candidates, not a commitment to a training base. Verify the exact revisions, terms, export behavior, and required preprocessing before use. Include lexical retrieval and a local transliteration-plus-embedding pipeline where practical.

## What a release should contain

- Model weights, tokenizer, and browser-compatible artifacts with clear usage instructions.
- Reproducible training and evaluation configurations, source revisions, and data-selection scripts.
- Quality results by language, query type, and task.
- Measured downloads, load times, inference latency, indexing throughput, and memory where measurable.
- A small browser demo and a model card describing supported use, limitations, and provenance.

Keep generated datasets, caches, and large checkpoints separate from tracked source files when the implementation is added. Record how to reproduce or obtain them.

## Background

- [Project context](../context.md)
- [Suggestions and options](../SUGGESTIONS_AND_OPTIONS.md)

The context defines the constraints. The consolidated assessment provides evidence and hypotheses; this folder records the embedding project's actual decisions and results.
