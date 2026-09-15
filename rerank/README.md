# Indic browser reranking

Build a compact model that scores a query against retrieved passages and improves their order, entirely inside a browser tab.

## Status and priority

Deferred until the [embedding project](../embeddings/README.md) is complete. This folder currently contains planning documents only. No model, training recipe, language set, or runtime budget has been chosen.

Start with [STEPS.md](STEPS.md) when this project becomes active. Revise the roadmap using the embedding project's actual results.

Agent workflow is in [AGENTS.md](AGENTS.md). Run outputs and resulting decisions belong in [experiments/](experiments/README.md).

## Goal and scope

The first candidate is a cross-encoder: it reads the query and one passage together and produces a relevance score. It reranks a small candidate set, rather than searching the entire corpus.

A release should improve ranking enough to justify its additional download, memory, and latency. Compare both the reranker alone and the complete retrieval-plus-reranking pipeline.

Begin with the languages and query types validated by the embedding project. Native script, Romanized text, code-mixing, and cross-language retrieval require separate evaluation.

## Conditions for starting

- Relevant passages often enter the retriever's top candidates but appear too low in the results.
- Ordering those candidates by known relevance shows meaningful room for improvement.
- Scoring a small candidate set can fit the intended browser latency budget.

A reranker cannot recover passages absent from its input. Poor candidate recall is a reason to revisit retrieval before training this model.

## Constraints

- Solo developer with a Mac M4 Pro and short Kaggle sessions on two 16 GB T4s.
- Existing checkpoints and public relevance data; no private corpus collection.
- Fully local browser inference, including any query normalization.
- A model release with reproducible evaluation comes before a larger product.

## Model candidates

Hugging Face sources checked on 15 September 2026. Recheck revisions, terms, and exports when work begins. Listed roles are planning choices; Indic quality and browser performance have not been measured here.

| Candidate | Initial role | Qualification |
| --- | --- | --- |
| [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | First baseline and possible fine-tuning base | Retrieval-trained multilingual cross-encoder; card lists Apache 2.0 and ONNX support. Verify the browser export rather than assuming all ONNX variants work. |
| [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | Larger quality reference and possible teacher | Card lists Apache 2.0 and custom code. Teacher value depends on target-language results. |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Alternative multilingual reference and teacher | Profile bounded local scoring before committing to distillation. Not the default browser artifact. |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Optional additional reference | Uses a different scoring recipe; preserve its prescribed input formatting and score extraction. Only add if it answers an unresolved baseline question. |

Do not equate a low layer count with a small download. Count vocabulary weights, tokenizer files, and the full exported artifact. An embedding checkpoint also needs a trained relevance head/objective before it becomes a reranker.

## Public data candidates

| Source | Use | Qualification |
| --- | --- | --- |
| [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) | Indic relevance supervision | Translated queries and passages with selection labels; refers to original MS MARCO terms. |
| [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) | Existing multilingual training data | Shares original MS MARCO examples with other translations. Preserve original IDs to prevent overlap. |
| [Indic SWIM-IR](https://huggingface.co/datasets/nthakur/indic-swim-ir-cross-lingual) | Supplemental cross-language positives | Synthetic query/English-passage pairs; negatives must be obtained separately. |
| [MIRACL](https://huggingface.co/datasets/miracl/miracl) | Native retrieval evaluation and candidate relevance judgments | Reuse the embedding project's split policy and document model training exposure. |
| [Hindi-BEIR collection](https://huggingface.co/collections/ArkaAcharya/hindi-beir) | Broader Hindi evaluation | Select tasks rather than treating all sources as one independent benchmark. |

Mine training candidates with a pinned retriever over public training corpora. Unjudged passages are not necessarily irrelevant. Keep evaluation queries and their translations out of training and teacher-label generation.

## Expected release

- Checkpoint, tokenizer, browser artifacts, and a query/passage scoring example.
- Training configuration, candidate-generation recipe, and data manifests.
- Ranking gains, regressions, candidate recall, and complete pipeline costs.
- A browser demo and model card covering provenance, scope, and limitations.

Add code, configurations, and experiment records here when work starts. Keep large generated data and checkpoints outside tracked source files.

## Background

- [Project context](../context.md)
- [Validation and options](../VALIDATION_AND_OPTIONS.md)
- [Embedding roadmap](../embeddings/STEPS.md)
