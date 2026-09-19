# Full Benchmark Evaluation Across 4 Deployment Model Variants

## Question and criterion

Roadmap: [Step 6](../../STEPS.md).

Evaluate the retrieval performance of four deployment model variants across the full public benchmarks (MIRACL Hindi dev full 506,264-document corpus and SciFact English test set) on remote Kaggle GPU hardware:
1. `base_e5`: Base pretrained `multilingual-e5-small` in FP32 PyTorch.
2. `lora_fp32`: Merged 3-epoch symmetric LoRA model ($r=16, \alpha=32$) in FP32 PyTorch.
3. `lora_int8`: LoRA symmetric model exported to ONNX dynamic INT8 (full 250k vocabulary, ~114 MB).
4. `lora_int8_pruned`: LoRA symmetric model with vocabulary pruning (90.7k tokens) and ONNX dynamic INT8 (~55.4 MB).

Success criteria:
- **Quantization Parity:** INT8 quantized models retain $\ge 99.0\%$ of the FP32 LoRA ranking quality ($\Delta \text{nDCG@10} \ge -0.01$).
- **Pruning Parity:** Pruned INT8 model (55.4 MB) achieves identical or near-identical retrieval to the full-vocab INT8 model ($\Delta \text{nDCG@10} \ge -0.005$).
- **Romanized Retrieval:** Retain $\ge 20\times$ improvement over the unadapted base model on the full 506k passage corpus.
- **Native Devanagari Retention:** Retain $\ge 98\%$ candidate coverage (Recall@20) compared to the base model.

## Setup

- **Hardware:** Kaggle Nvidia Tesla T4-2x GPU
- **Evaluation Manifests:**
  - `miracl_hi_dev_native`: 350 native Devanagari queries against full 506,264 Wikipedia documents.
  - `miracl_hi_dev_romanized`: 350 synthetic Romanized Hindi queries against full 506,264 Wikipedia documents.
  - `scifact_en_test`: 300 English scientific queries against 5,183 scientific corpus documents.
- **Evaluation Settings:**
  - Query prefix: `"query: "`
  - Passage prefix: `"passage: "`
  - Max sequence length: 512 tokens
  - Batch size: 512
  - Corpus vector caching enabled in `/tmp/cache` (single corpus pass reused across native and Romanized queries)
- **Model Variants:**
  1. `base_e5`: `intfloat/multilingual-e5-small` (FP32 PyTorch)
  2. `lora_fp32`: Merged 3-Epoch Symmetric LoRA ($r=16, \alpha=32$, FP32 PyTorch)
  3. `lora_int8`: Full-vocab ONNX dynamic INT8 (~113.7 MB)
  4. `lora_int8_pruned`: Pruned-vocab (90.7k tokens) ONNX dynamic INT8 (~55.4 MB)

## Execution

Status: running.

Dispatched via `kagglex run`:
- Kernel slug: `e5-full-eval-campaign`
- Command:
```bash
python -m indic_edge_embeddings.eval.run_kaggle_full_eval
```

## Outputs

Pending completion of remote execution.

## Interpretation

Pending remote execution results.

## Decision

Pending remote execution results.

## Next step

1. Monitor execution logs and pull outputs via `kagglex pull`.
2. Update results table and verify export/pruning parity on the 506k corpus.
3. Benchmark the finalized pruned INT8 artifact in the browser test harness.
