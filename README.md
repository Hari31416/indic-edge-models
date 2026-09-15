# Indic Edge Models

Fine-tune compact, efficient machine learning models for Indic languages designed to run entirely on the edge inside a browser tab.

## Overview

The primary objective is to build practical, edge-first models for Indian languages that operate completely client-side without relying on cloud APIs.

Development follows a staged priority order:

1. [Embeddings](embeddings/README.md): Compact Indic and multi-script embedding models for in-browser semantic search and retrieval.
2. [Reranking](rerank/README.md): Cross-encoder rerankers to improve candidate passage ranking within browser latency budgets.
3. [Translation](translation/README.md): Compact, pair-specific encoder-decoder translation models running locally in-browser.

Embeddings is the active first priority. Reranking and translation are deferred until the embedding stage is complete and evaluated.

## Core Constraints

- Runs entirely in the browser using client-side runtimes such as ONNX Runtime Web and Transformers.js.
- Built with public datasets and existing checkpoints without private corpus collection.
- Compute constrained to a single developer setup using a Mac M4 Pro and Kaggle 2x T4 GPU sessions.
- Evaluation focuses on measured in-browser performance: download size, memory footprint, initialization time, and inference latency alongside retrieval and generation quality.

## Project Structure

- [context.md](context.md): Developer goals, scope boundaries, hardware limits, and non-goals.
- [AGENTS.md](AGENTS.md): Guidelines for workflow, agent execution, and experiment tracking standards.
- [SUGGESTIONS_AND_OPTIONS.md](SUGGESTIONS_AND_OPTIONS.md): Consolidated technical feasibility assessment, baseline candidate comparisons, and architecture options.
- [embeddings/](embeddings/README.md): Embedding model subproject, roadmap, and experiment records.
- [rerank/](rerank/README.md): Cross-encoder reranking subproject and planning roadmap.
- [translation/](translation/README.md): Pair-specific translation subproject and planning roadmap.

## Experiment Tracking

All empirical evaluations, benchmarks, and model iterations are tracked under dedicated `experiments/` directories inside each subproject folder following the protocol defined in [AGENTS.md](AGENTS.md).
