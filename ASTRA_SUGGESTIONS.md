# Suggestions for Indic models on edge devices

Date: 15 September 2026

## Purpose and constraints

Explore useful, compact models for Indian languages that can run entirely on the user's device, ideally in a browser. Build and release the model first; a product can follow later.

Available resources:

- Solo developer, without access to proprietary datasets or a large annotation effort.
- Mac M4 Pro for development, preprocessing, evaluation, and browser testing. Available RAM remains unspecified.
- Kaggle access to two T4 GPUs, each with 16 GB VRAM. They do not automatically function as a single 32 GB GPU.
- Public datasets, existing checkpoints, and bounded distillation experiments are the practical starting points.

The shortlist is limited to an embedder, a reranker, and pair-specific translation. A chatbot is a distant possibility and outside the current scope.

## What success should mean

Aim for better quality at a comparable browser download, memory, and latency budget, or comparable quality with a substantially smaller footprint. Do not assume that specializing in Indic languages guarantees an improvement.

Select initial languages partly by the ability to inspect their outputs. Start with a small language set and expand after demonstrating repeatable gains.

Browser compatibility must be measured on the exported, quantized artifact. Report required download files, peak memory, cold loading, warm inference latency, and task quality. A successful run on an M4 Pro does not establish performance on an inexpensive phone.

The feasibility assessments below are planning judgments, not results from training or benchmarking these models.

## 1. Compact Indic embedder

### Basic idea

Map queries and passages into vectors so that relevant passages can be retrieved locally. The most promising specialization is short, informal queries in native scripts, Romanized text, and mixed English retrieving formal passages.

Distinguish these capabilities during evaluation:

- Native-script query to a passage in the same language.
- Indic query to an English passage, and the reverse where relevant.
- Romanized query to a native-script passage.
- Code-mixed query to a relevant passage.

### Suggested starting point

Fine-tune `intfloat/multilingual-e5-small`, about 118M parameters, rather than pretraining an encoder from scratch. It already has retrieval training, produces 384-dimensional vectors, and uses an MIT license.

Begin with two or three Indian languages plus English and a bounded pilot of roughly 100,000–300,000 selected training pairs. This is an experiment budget, not a guarantee of sufficient data. Start with short passages, around 256 tokens, and investigate size reduction after establishing quality.

Include relevant query–passage pairs, plausible but irrelevant passages as hard negatives, and carefully selected script variants. Ordinary gradient accumulation does not automatically supply the same competing examples as a large contrastive batch.

### Baselines

- [Multilingual E5 small](https://huggingface.co/intfloat/multilingual-e5-small): starting checkpoint and unchanged baseline.
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m): approximately 308M parameters, an on-device quality reference.
- [Jina embeddings v5 text nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano): 239M parameters; noncommercial weight license.
- [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): larger reference and possible teacher, subject to checking the intended training use.

Aggregate multilingual scores do not establish performance on a particular Indic language or Romanized queries.

### Data and evaluation

- [Indic SWIM-IR](https://huggingface.co/datasets/nthakur/indic-swim-ir-cross-lingual): synthetic Indic-query/English-passage training pairs.
- [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI): translated retrieval supervision; refers to original MS MARCO terms.
- [MIRACL](https://huggingface.co/datasets/miracl/miracl): independent retrieval evaluation for supported languages.
- [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA): supplementary evaluation after constructing a passage corpus; originally a QA dataset.

Translation pairs can help align languages but do not, by themselves, teach retrieval relevance. Avoid training/evaluation overlap, including translated or transliterated versions of the same underlying example.

[IndicXlit](https://github.com/AI4Bharat/IndicXlit) already provides compact word-level transliteration. Compare direct Romanized retrieval against IndicXlit followed by an existing embedder. Training a separate generic transliterator is not part of this proposal.

### Main risks

Public data still requires filtering, language balancing, and inspection. Synthetic transliteration may not represent natural spelling variation. Compression can damage retrieval quality. Vector dimensions affect index storage but do not necessarily shrink model weights.

## 2. Compact Indic reranker

### Basic idea

Take a query and a small set of retrieved passages, then score each passage more carefully. A reranker can improve the order of results from an embedder or keyword search.

### When to build it

Make this conditional on retrieval results:

- Relevant passages frequently appear in the top 20.
- They fail to reach the first few positions often enough to matter.
- Reranking a small candidate set fits the browser latency budget.

A reranker cannot recover a passage absent from its candidate set.

### Suggested approach

Fine-tune or distill a small cross-encoder, which reads the query and passage together. Reuse suitable retrieval pairs and mine difficult negatives. A rough initial deployment target is 30–100 MB, subject to actual architecture, export size, and quality.

Compare against the unchanged retriever and against keyword retrieval plus reranking. Larger quality references include [GTE multilingual reranker base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) and the [Jina reranker family](https://huggingface.co/jinaai/jina-reranker-v3). Check the exact checkpoint's license and deployment requirements.

### Main risks

Every search requires scoring multiple query–passage pairs. A small model can still add noticeable latency. Measure the combined pipeline's quality, download, and memory costs, not only the reranker's standalone accuracy.

## 3. Translation below 100 MB for one direction

### Basic idea

Build a downloadable translator for one direction, such as English → Hindi. English → Bengali uses `en-bn`; `be` means Belarusian.

Start with one direction and short passages. Supporting both directions within one 100 MB package is a separate, harder target.

### Suggested approach

Evaluate an existing compact translator before training. Then fine-tune it or distill a stronger teacher into a smaller encoder–decoder model. A roughly 40–70M-parameter student deployed in INT8 is a reasonable first design to investigate, not a promised size or quality result.

Fine-tuning alone does not shrink a model. Size reduction requires quantization, vocabulary reduction, architectural changes, or distillation.

### Baselines

- [OPUS-MT English → Hindi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi): compact Apache 2.0 baseline.
- [Its Transformers.js export](https://huggingface.co/Xenova/opus-mt-en-hi/tree/main/onnx): the listed INT8 encoder and merged decoder total approximately 110 MB before tokenizer files. Reaching 100 MB alone would therefore be a modest improvement.
- [OPUS-MT Bengali → English](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en) and [English → Indic](https://huggingface.co/Helsinki-NLP/opus-mt-en-inc): additional relevant baselines.
- [IndicTrans2 distilled English → Indic](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M): MIT-licensed quality reference and potential teacher; requires substantial compression to meet the budget.

[Chrome's on-device Translator API](https://developer.chrome.com/docs/ai/translator-api) already lists Hindi and Bengali. Its documentation currently describes desktop support, not mobile. Portability and measured mobile performance could provide value, but must be demonstrated.

### Data and evaluation

Investigate language-specific slices of [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC), particularly the human-translated BPCC-H-Wiki and BPCC-H-Daily subsets. Their listed license is CC BY 4.0. Check each component's provenance and terms: CC0 packaging does not imply unrestricted underlying text.

Use [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) and [IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) as held-out evaluation, not training material.

Measure translation quality alongside preservation of names, numbers, dates, negation, and completeness. Evaluate formal and conversational text separately. Human inspection of both languages remains necessary for a credible quality claim.

### Main risks

Fluent output can still change meaning. Teacher generation adds compute cost. Decoder inference is sequential, so sentence length and decoding strategy affect latency. Quantized downloads and runtime memory are different quantities; encoder/decoder exports can also duplicate weights.

## Recommended order

1. Define initial languages, target devices, and release/license requirements.
2. Benchmark existing models before choosing a training recipe.
3. Choose either the embedder or one-direction translation as the first substantial project.
4. Release a verified browser artifact with reproducible evaluation and explicit limitations.
5. Add the reranker when retrieval measurements justify it.
6. Expand language coverage only after the first model demonstrates useful gains.

The embedder offers cheaper iteration and simpler inference. Translation provides more immediately visible standalone utility. Both are feasible starting projects on the available hardware, but neither comes with a guarantee of outperforming existing models.
