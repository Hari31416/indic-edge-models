# Low-Rank Adaptation (LoRA) for Parameter-Isolated Cross-Script Retrieval

## Question and criterion

Roadmap: [Step 4](../../STEPS.md).

Evaluate whether parameter-isolated Low-Rank Adaptation (LoRA) on `multilingual-e5-small` eliminates the catastrophic representation drift observed in full-weight fine-tuning (-0.0391 on native Devanagari) while enabling robust Romanized retrieval (target $\text{nDCG@10} \ge 0.4000$).

Success criteria:
- Native retention gate: $\Delta \text{nDCG@10} \ge -0.02$ from reference baseline ($\ge 0.8989$).
- English retention gate: $\Delta \text{nDCG@10} \ge -0.02$ from reference baseline ($\ge 0.6570$).
- Romanized retrieval target: $\text{nDCG@10} \ge 0.4000$.

## Setup

- **Base Model:** `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, frozen backbone)
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Dropout: 0.05
  - Target Modules: Attention projections (`query`, `key`, `value`, `dense`)
- **Training Data:** `quality_matched_v3/passages/train.parquet` (98,825 rows; 76,000 native Devanagari, 22,825 Romanized Hindi)
- **Validation Data:** `quality_matched_v3/passages/validation.parquet` (5,227 rows)
- **Evaluation Manifests:**
  - Native Devanagari: `eval/miracl_hi_dev_fast_native` (350 queries, 5,000 corpus docs)
  - Romanized Hindi: `eval/miracl_hi_dev_fast_romanized` (350 queries, 5,000 corpus docs)
  - English Reference: `eval/scifact_en_test` (300 queries, 5,183 corpus docs)
- **Loss Formulation:** InfoNCE with temperature $\tau = 0.05$ (in-batch contrastive negatives)
- **Candidate Architectures to Compare:**
  1. `lora_symmetric`: LoRA applied across both query and passage representations.
  2. `lora_query_asymmetric`: LoRA applied exclusively to query encodings while keeping passage embeddings 100% frozen in the base pretrained space.
- **Hardware:** Kaggle Nvidia T4-2x GPU

## Execution

Status: running.

Dispatched via `kagglex run`:
- Kernel slug: `e5-lora-adaptation`
- Command:
```bash
python -m indic_edge_embeddings.train.run_lora_campaign \
  --train-file quality_matched_v3/passages/train.parquet \
  --val-file quality_matched_v3/passages/validation.parquet \
  --native-eval-dir eval/miracl_hi_dev_fast_native \
  --romanized-eval-dir eval/miracl_hi_dev_fast_romanized \
  --english-eval-dir eval/scifact_en_test \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 1e-4 \
  --num-epochs 1 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --output-dir /kaggle/working/outputs/campaign_lora
```

## Outputs

Pending execution.

## Interpretation

Pending execution.

## Decision

Pending execution.

## Next step

Implement the LoRA training adapter module using `peft`, verify local parameter counting and unit tests, and dispatch the remote Kaggle run.
