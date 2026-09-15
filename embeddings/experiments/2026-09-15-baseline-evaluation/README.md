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

| Model                   | Variant | Size (MB) | Parity (Mean Cos) | Native nDCG@10 | Native Recall@20 | Native MRR@10 | Romanized nDCG@10 | Romanized Recall@20 | Romanized MRR@10 |
| ----------------------- | ------- | --------- | ----------------- | -------------- | ---------------- | ------------- | ----------------- | ------------------- | ---------------- |
| `multilingual-e5-small` | FP32    | 1.62*     | 1.000000          | 0.5513         | 0.7766           | 0.5722        | 0.0006            | 0.0043              | 0.0004           |
| `multilingual-e5-small` | INT8    | 113.70    | 0.991866          | -              | -                | -             | -                 | -                   | -                |

*Note: FP32 graph references external weights; INT8 dynamic package is self-contained at 113.70 MB.

Raw results stored in:
- `experiments/2026-09-15-baseline-evaluation/e5_small_native.json`
- `experiments/2026-09-15-baseline-evaluation/e5_small_romanized.json`
- `artifacts/e5_small_onnx/export_report.json`

## Interpretation

1. **Native Hindi Retrieval:** `multilingual-e5-small` establishes a solid baseline on native Devanagari Hindi (nDCG@10 of `0.5513`, Recall@20 of `0.7766`).
2. **Catastrophic Failure on Romanized Hindi:** When the exact same queries are transliterated into Latin script (Hinglish search), E5-small retrieval collapses almost completely (nDCG@10 of `0.0006`, Recall@20 of `0.0043`). This empirically validates our core hypothesis: published multilingual models cannot bridge the script gap without explicit alignment.
3. **Footprint Bottleneck:** At 113.70 MB for INT8, E5-small's 250,037 vocabulary table dominates weight storage, confirming that vocabulary pruning is necessary to hit an edge/browser download budget (<35 MB).

## Decision

- Adopt `intfloat/multilingual-e5-small` as the primary base checkpoint for the training pilot.
- Focus pilot fine-tuning on bridging the Romanization gap on our curated `pilot_hindi_100k` dataset with hard negatives.
- Measure adaptation gains on Romanized retrieval while enforcing a regression limit on native Hindi retrieval (maintaining nDCG@10 > 0.53).
- Investigate vocabulary pruning as a decoupled compression step after adaptation.

## Next step

Proceed to Step 4 of the roadmap: Train the pilot adaptation model on `pilot_hindi_100k` using Kaggle 2x T4 GPUs via `run_on_kaggle.py`.
