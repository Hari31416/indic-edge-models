# Baseline model evaluation and export parity

Establish baseline retrieval performance and ONNX export characteristics for candidate embedding models across English, native Hindi, and synthetic Romanized Hindi test sets.

## Question and criterion

Can published compact multilingual models (`intfloat/multilingual-e5-small`, `ibm-granite/granite-embedding-97m-multilingual-r2`, and `hotchpotch/bekko-embedding-v1-a8m`) deliver viable native Hindi and Romanized Hindi retrieval within a client-side footprint, and what is their exact export size and parity?

Success condition:
- Establish ranking (nDCG@10) and retrieval coverage (Recall@20) baselines on frozen evaluation manifests.
- Export candidates to ONNX FP32 and INT8 with verified numerical cosine parity (>0.99).
- Record exact artifact sizes to determine the compression requirements for browser deployment.

Roadmap link: [Embedding project steps 1 to 3](../../STEPS.md#1-define-the-first-experiment).

## Setup

- Hardware: Apple M4 Pro (MacBook), local CPU and Metal (MPS) execution.
- Runtime: Python 3.13 / uv environment, PyTorch 2.5, Hugging Face Transformers, ONNX Runtime.
- Models:
  - `intfloat/multilingual-e5-small` (118M params, width 384, vocab 250,037)
  - `ibm-granite/granite-embedding-97m-multilingual-r2` (97M params, width 384, vocab 128,000)
- Evaluation datasets:
  - MIRACL Hindi (`miracl_hi_dev_native`): 350 native Devanagari queries against 506,264 Wikipedia passage corpus.
  - Synthetic Romanized MIRACL Hindi (`miracl_hi_dev_romanized`): 350 transliterated queries mapped to identical document IDs.
- Metrics: nDCG@10 (primary ranking), Recall@20 (candidate pool coverage), MRR@10.

## Execution

Commands run:

```bash
# 1. Build and freeze evaluation datasets
uv run python -m indic_edge_embeddings.data.build_eval_data --output-dir data/eval --split dev

# 2. Export and quantize candidate model
uv run python -m indic_edge_embeddings.export.export_cli --model-name intfloat/multilingual-e5-small --output-dir artifacts/e5_small_onnx

# 3. Benchmark native and Romanized evaluation
uv run python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir data/eval/miracl_hi_dev_native --model-name intfloat/multilingual-e5-small --query-prefix "query: " --passage-prefix "passage: " --max-length 512 --batch-size 512 --cache-dir data/cache --output-json experiments/2026-09-15-baseline-evaluation/e5_small_native.json
uv run python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir data/eval/miracl_hi_dev_romanized --model-name intfloat/multilingual-e5-small --query-prefix "query: " --passage-prefix "passage: " --max-length 512 --batch-size 512 --cache-dir data/cache --output-json experiments/2026-09-15-baseline-evaluation/e5_small_romanized.json
```

Status: Completed for `intfloat/multilingual-e5-small`.

## Outputs

| Model                                  | Params | Vocab   | Pooling | Native nDCG@10 | Native Recall@20 | Native MRR@10 | Romanized nDCG@10 | Romanized Recall@20 | Romanized MRR@10 |
| -------------------------------------- | ------ | ------- | ------- | -------------- | ---------------- | ------------- | ----------------- | ------------------- | ---------------- |
| `intfloat/multilingual-e5-small`       | 118M   | 250,037 | mean    | 0.5513         | 0.7766           | 0.5722        | 0.0006            | 0.0043              | 0.0004           |
| `ibm-granite/granite-embedding-97m-r2` | 97M    | 128,000 | cls     | 0.4574         | 0.6586           | 0.4999        | 0.0008            | 0.0024              | 0.0014           |
| `hotchpotch/bekko-embedding-v1-a25m`   | 124M*  | 250,000 | mean    | 0.5195         | 0.7440           | 0.5517        | 0.0036            | 0.0115              | 0.0037           |

*Note: Bekko-a25m contains ~25M active transformer parameters across 13 structural layers.

Export Characteristics (E5 Small):
- FP32 ONNX Size: 1.62 MB (graph only, external weights) / 470 MB full.
- INT8 ONNX Dynamic Size: 113.70 MB.
- INT8 Parity (Mean Cosine Similarity): 0.991866.

Raw results stored in:
- `e5_small_native.json`
- `e5_small_romanized.json`
- `granite_native.json`
- `granite_romanized.json`
- `bekko_25m_native.json`
- `bekko_25m_romanized.json`

## Interpretation

1. **Universal Romanization Failure across All Base Models:** Every single published multilingual candidate collapses on Romanized Hindi search queries (`nDCG@10 < 0.004` across the board, compared to `0.45 - 0.55` on native script). This confirms that zero-shot cross-script retrieval cannot be assumed for any existing compact model and proves the absolute necessity of our planned script alignment fine-tuning (Step 4).
2. **Native Retrieval Quality Hierarchy:**
   - `multilingual-e5-small` achieves the strongest native Hindi ranking (`nDCG@10 = 0.5513`, `Recall@20 = 0.7766`).
   - `bekko-embedding-v1-a25m` achieves competitive native retrieval (`nDCG@10 = 0.5195`, `Recall@20 = 0.7440`) despite using only 13 layers / ~25M active parameters.
   - `granite-embedding-97m-r2` scores lower on MIRACL Hindi (`nDCG@10 = 0.4574`, `Recall@20 = 0.6586`).
3. **Storage Bottleneck:** All models with full ~250k multilingual tokenizers exceed 100 MB in INT8 weight storage, confirming that vocabulary reduction is essential to achieve our <35 MB target browser bundle.

## Decision

- Adopt `intfloat/multilingual-e5-small` as the primary base checkpoint for the training pilot, with `hotchpotch/bekko-embedding-v1-a25m` as an efficient student/distillation target.
- Proceed with pilot fine-tuning on `pilot_hindi_100k` (Step 4) to bridge the Romanization gap while enforcing native retrieval preservation (`nDCG@10 > 0.53`).
- Implement post-training vocabulary pruning and INT8 dynamic quantization (Step 5) to compress the model to client-side browser budget.

## Next step

Proceed to Step 4 of the roadmap: Train the pilot adaptation model on `pilot_hindi_100k` using Kaggle 2x T4 GPUs via `scripts/run_kaggle.py`.

## Subsequent audit, 15 September 2026

The [Romanized baseline audit](../2026-09-15-romanized-audit/README.md) reproduced E5's score and verified query/relevance mappings and the corpus. It found 18/350 queries with residual Devanagari and synthetic spelling limitations. The original interpretation of universal Romanization failure and immediate training readiness is therefore superseded: these scores describe the existing synthetic slice, and a corrected, inspected slice is needed before the full pilot. The original results and decision above are retained as experiment history.
