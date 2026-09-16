# E5-Small Pilot Training on Hindi Retrieval

## Question and criterion

Can fine-tuning `intfloat/multilingual-e5-small` on `pilot_hindi_100k` (99,994 relevance pairs, 23% synthetic Romanized) bridge the Romanized retrieval gap (from 0.0006 baseline) while preserving native Devanagari retrieval quality without catastrophic forgetting? Roadmap step 4 bounded training pilot.

Acceptance criteria:
- Romanized Hindi nDCG@10 on held-out MIRACL dev >= 0.40 (baseline: 0.0006).
- Native Hindi nDCG@10 on held-out MIRACL dev >= 0.53 (baseline: 0.5513, max allowable regression <= 0.02).

## Setup

- Base model: `intfloat/multilingual-e5-small` (118M parameters, hidden size 384, vocabulary 250,037).
- Training dataset: `hari31416/indic-edge-hindi-pilot` (`pilot_hindi_100k/train.parquet`, 99,994 rows: 76,889 native Devanagari, 23,105 ASCII v2 Romanized).
- Validation dataset: `pilot_hindi_100k/validation.parquet` (5,000 rows, group-isolated from train).
- Evaluation datasets:
  - `miracl_hi_dev_native`: 350 native Devanagari queries over 506,264 Wikipedia passage corpus.
  - `miracl_hi_dev_romanized`: 350 transliterated queries over identical 506,264 passage corpus.
- Hardware: Kaggle 2x NVIDIA T4 GPUs (16 GB VRAM each).
- Hyperparameters:
  - Batch size per GPU: 32
  - Gradient accumulation steps: 4 (effective batch size: 128)
  - Optimizer: AdamW (weight decay 0.01)
  - Learning rate: 2e-5 with cosine decay and 10% warmup
  - Loss: InfoNCE / MultipleNegativesRankingLoss (temperature 0.05)
  - Precision: fp16 mixed precision
  - Sequence length: 512 tokens max
  - Epochs: 3 (2,343 optimization steps)
  - Seed: 42
- Training Kernel ID: `hari31416/e5-small-pilot-training`
- Evaluation Kernel ID: `hari31416/e5-pilot-evaluation`

## Execution

1. Remote training dispatched to Kaggle GPU:

```bash
PYTHONPATH=embeddings/src:. embeddings/.venv/bin/python scripts/run_kaggle.py push \
  --subproject embeddings \
  --title "e5-small-pilot-training" \
  --slug "e5-small-pilot-training" \
  --gpu t4-2x \
  --datasets hari31416/indic-edge-hindi-pilot \
  --command "python -m indic_edge_embeddings.train.train_cli --train-file train.parquet --val-file validation.parquet --model-name-or-path intfloat/multilingual-e5-small --output-dir /kaggle/working/outputs/pilot_e5_adapted --batch-size 32 --gradient-accumulation-steps 4 --learning-rate 2e-5 --num-epochs 3 --warmup-ratio 0.1 --max-length 512"
```

Training finished in 1,538 seconds (25.6 minutes, 195.0 examples/sec). Validation loss decreased monotonically from 0.8465 (step 250) to 0.6121 (step 2343).

2. Remote dense retrieval evaluation dispatched to Kaggle GPU:

```bash
PYTHONPATH=embeddings/src:. embeddings/.venv/bin/python scripts/run_kaggle.py push \
  --subproject embeddings \
  --title "e5-pilot-evaluation" \
  --slug "e5-pilot-evaluation" \
  --gpu t4-2x \
  --datasets hari31416/indic-edge-hindi-pilot hari31416/indic-edge-e5-pilot-adapted \
  --command "python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir eval/miracl_hi_dev_native --model-name indic-edge-e5-pilot-adapted --query-prefix 'query: ' --passage-prefix 'passage: ' --max-length 512 --batch-size 512 --cache-dir /tmp/cache --output-json /kaggle/working/outputs/pilot_e5_native.json && python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir eval/miracl_hi_dev_romanized --model-name indic-edge-e5-pilot-adapted --query-prefix 'query: ' --passage-prefix 'passage: ' --max-length 512 --batch-size 512 --cache-dir /tmp/cache --output-json /kaggle/working/outputs/pilot_e5_romanized.json"
```

Corpus encoding of 506,264 documents completed at 3.7 batches/sec on CUDA T4 FP16, and metrics were generated for both manifests.

## Outputs

### Retrieval Metrics Comparison

| Model Variant | Native nDCG@10 | Native Recall@20 | Native MRR@10 | Romanized nDCG@10 | Romanized Recall@20 | Romanized MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `intfloat/multilingual-e5-small` (Baseline) | 0.5513 | 0.7766 | 0.5722 | 0.0006 | 0.0043 | 0.0004 |
| `indic-edge-e5-pilot-adapted` (Step 4 Checkpoint) | 0.3921 | 0.5910 | 0.4392 | 0.1059 | 0.1825 | 0.1225 |
| **Delta** | **-0.1592** | **-0.1856** | **-0.1330** | **+0.1053** | **+0.1782** | **+0.1221** |

Raw metric files:
- `pilot_e5_native.json`
- `pilot_e5_romanized.json`

Visualizations:
- Loss progression curve: `loss_curve.png`
- Retrieval metric comparison: `retrieval_comparison.png`

## Interpretation

1. **Romanization Capability Bridge:** Synthetic Romanized fine-tuning produced a genuine capability jump. Romanized nDCG@10 moved from 0.0006 to 0.1059 (+176x), and Romanized Recall@20 jumped from 0.0043 to 0.1825 (+42x). The model learned to retrieve relevant documents for transliterated queries where the base model completely collapsed.
2. **Acceptance Criteria Failure:**
   - Romanized nDCG@10 (0.1059) remains well below the acceptance threshold (>= 0.40).
   - Native nDCG@10 degraded from 0.5513 to 0.3921 (-0.1592), failing the preservation criterion (>= 0.53).
3. **Root Cause Analysis:**
   - **Unregularized Full Fine-Tuning:** Standard InfoNCE loss across all 12 transformer layers at 2e-5 caused representation drift away from the base multilingual space.
   - **No Alignment Regularizer:** The model was not constrained to preserve its native representations (e.g. via distillation loss against the frozen base model or explicit cross-script query-query alignment loss: $L_{align} = \|E(q_{roman}) - E(q_{native})\|^2$).
   - **Data Volume and Diversity:** 100k pairs (with 23k Romanized queries) is insufficient for full representation realignment without distillation guidance.

## Decision

The pilot adaptation proves that cross-script retrieval can be learned from synthetic data, but simple end-to-end InfoNCE fine-tuning causes unacceptable native degradation and plateaued Romanized recall.

Decisions:
1. Do not proceed to INT8 export/quantization of this specific checkpoint as the primary production model.
2. Formulate the next iteration with two structural enhancements:
   - **Teacher-Student Distillation / Feature Preservation Loss:** Add MSE or KL divergence loss between candidate native representations and frozen baseline E5 representations to enforce $nDCG@10 \ge 0.53$.
   - **Cross-Script Embedding Alignment Loss:** Add explicit query-to-query alignment ($E(q_{romanized}) \leftrightarrow E(q_{native})$) so Romanized embeddings are pulled directly into the well-structured native manifold.

## Next step

Create experiment `2026-09-16-distillation-alignment-pilot` incorporating dual-objective training (InfoNCE + representation preservation distillation + cross-script alignment) to hit the >= 0.40 Romanized and >= 0.53 Native acceptance criteria.
