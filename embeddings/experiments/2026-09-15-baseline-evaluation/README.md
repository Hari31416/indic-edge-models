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
- Runtime: Python 3.14 / uv environment, PyTorch, Hugging Face Transformers, ONNX Runtime.
- Models:
  - `intfloat/multilingual-e5-small` (118M params, width 384, vocab 250,037)
  - `ibm-granite/granite-embedding-97m-multilingual-r2` (97M params, width 384, vocab 128,000)
  - `hotchpotch/bekko-embedding-v1-a8m` (106M params, width 512, vocab 102,400)
- Evaluation datasets:
  - MIRACL Hindi (`miracl/miracl` split `dev`): native Hindi queries against Wikipedia passage corpus.
  - Synthetic Romanized MIRACL Hindi: Transliterated Hindi queries mapped to the same document IDs.
- Metrics: nDCG@10 (primary ranking), Recall@20 (candidate pool coverage), MRR@10.

## Execution

Commands:

```bash
# Build evaluation manifest
python -m indic_edge_embeddings.eval.build_miracl_manifest --output-dir data/manifests --split dev

# Export and quantize candidate models
python -m indic_edge_embeddings.export.export_cli --model-name intfloat/multilingual-e5-small --output-dir artifacts/e5_small_onnx
python -m indic_edge_embeddings.export.export_cli --model-name ibm-granite/granite-embedding-97m-multilingual-r2 --output-dir artifacts/granite_97m_onnx

# Run evaluation benchmarks
python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir data/manifests/miracl_hi_dev --model-name intfloat/multilingual-e5-small --query-prefix "query: " --passage-prefix "passage: " --output-json experiments/2026-09-15-baseline-evaluation/e5_small_miracl_hi.json
```

Status: Setup and tooling initialized. Awaiting package installation and dataset generation run.

## Outputs

Metric logs, parity reports, and artifact size measurements will be recorded here once execution runs.

| Model                         | Variant | Size (MB) | Parity (Mean Cos) | MIRACL Native nDCG@10 | MIRACL Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 |
| ----------------------------- | ------- | --------- | ----------------- | --------------------- | ----------------------- | ----------------- | ------------------- |
| `multilingual-e5-small`       | FP32    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| `multilingual-e5-small`       | INT8    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| `granite-97m-multilingual-r2` | FP32    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| `granite-97m-multilingual-r2` | INT8    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| `bekko-embedding-v1-a8m`      | FP32    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| `bekko-embedding-v1-a8m`      | INT8    | TBD       | TBD               | TBD                   | TBD                     | TBD               | TBD                 |
| Lexical BM25                  | -       | -         | -                 | TBD                   | TBD                     | TBD               | TBD                 |

## Interpretation

To be populated upon baseline run completion.

## Decision

To be recorded based on baseline scores and INT8 model sizes.

## Next step

Run dependency sync and execute the baseline export and benchmark suites.
