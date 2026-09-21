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

Status: completed with failure (exit code 1, exit status 137).

Dispatched via `kagglex run`:
- Kernel: `hari31416/e5-full-4-model-evaluation`
- Dataset: `hari31416/indic-edge-e5-models`
- Command:
```bash
python -m indic_edge_embeddings.eval.run_kaggle_full_eval
```

Duration: 19,973 seconds (~5.5 hours). The run evaluated `base_e5` and `lora_fp32` across all splits, but was killed (OOM SIGKILL) at 79% through corpus encoding for the third model variant (`lora_int8`).

## Outputs

Output artifacts pulled from Kaggle:
- `base_e5_metrics.json`
- `lora_fp32_metrics.json`
- `run_summary.json`

Summary comparison on the full 506k MIRACL Hindi corpus and SciFact English benchmark:

| Metric | Base E5 (FP32) | LoRA Symmetric (FP32) | Delta |
| :--- | :--- | :--- | :--- |
| **Native Hindi nDCG@10** | 0.5513 | 0.4638 | -0.0875 (-15.9%) |
| **Native Hindi Recall@20** | 0.7766 | 0.6788 | -0.0978 (-12.6%) |
| **Romanized Hindi nDCG@10** | 0.0000 | 0.1103 | +0.1103 |
| **Romanized Hindi Recall@20** | 0.0000 | 0.1980 | +0.1980 |
| **English SciFact nDCG@10** | 0.6770 | 0.6489 | -0.0281 (-4.1%) |
| **English SciFact Recall@20** | 0.8469 | 0.8319 | -0.0150 (-1.8%) |

`lora_int8` and `lora_int8_pruned` variants did not finish evaluation due to memory exhaustion during corpus vector re-encoding.

## Interpretation

1. **Severe native regression under full corpus scale.** On the full 506,264 document corpus, adapting the model to Romanized Hindi caused a 0.0875 drop in native Devanagari nDCG@10 (0.5513 down to 0.4638) and dropped Recall@20 by nearly 10 percentage points.
2. **Underwhelming Romanized retrieval.** While Romanized retrieval showed high relative gains over zero baseline, absolute retrieval on the full corpus reached only 0.1103 nDCG@10 and 0.1980 Recall@20. Distractor density in the 506k corpus overwhelmed cross-script discriminability.
3. **Execution limits.** Evaluating full-corpus embeddings across multiple model variants exceeded Kaggle container RAM limits (OOM at 79% on batch 786/989).

## Decision

Stop the embedding adaptation project. The tradeoff between native representation degradation and modest Romanized retrieval gains does not produce a competitive edge retrieval model. Close out the embedding subproject and transition focus to pair-specific translation.

## Next step

Conclude embedding development. Shift active implementation to the translation subproject starting with baseline measurements on English to Hindi.

