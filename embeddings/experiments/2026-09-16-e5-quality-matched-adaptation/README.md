# Quality-matched passage supervision adaptation campaign

## Question and criterion
Roadmap: [Step 4](../../STEPS.md).
Evaluate whether training on genuine selected-passage supervision (`quality_matched_v3/passages`) improves Hindi retrieval and script adaptation across the 3 follow-up adaptation paradigms:
1. `conservative_control` (InfoNCE, LR 5e-6, 1 epoch)
2. `teacher_align` (Frozen teacher $L_{script} + L_{preserve}$, LR 5e-6, 1 epoch)
3. `dual_objective` ($L_{script} + L_{preserve} + L_{retrieval}$, LR 5e-6, 1 epoch)

Success criteria:
- Native retention gate: within 0.02 of reference baseline.
- English retention gate: within 0.02 of reference baseline.
- Romanized retrieval: evaluate progression and contrastive discriminability against distractor passages.

## Setup
- **Base Student Model:** `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`)
- **Frozen Teacher Model:** `intfloat/multilingual-e5-small`
- **Training Data:** `quality_matched_v3/passages/train.parquet` (98,825 rows; 76,000 native, 22,825 Romanized)
- **Validation Data:** `quality_matched_v3/passages/validation.parquet` (5,227 rows)
- **Evaluation Manifests:** `eval/miracl_hi_dev_fast_native`, `eval/miracl_hi_dev_fast_romanized`, `eval/scifact_en_test`
- **Attached Dataset:** `hari31416/indic-edge-hindi-pilot`
- **Hyperparameters:** Batch size 32, gradient accumulation 4 (effective batch size 128), learning rate 5e-6, 1 epoch, seed 42
- **Hardware:** Kaggle Nvidia T4-2x GPU

## Execution
Status: in_progress.

Dispatched via `kagglex run`:
- Kernel slug: `e5-quality-matched-adaptation`
- Command:
```bash
python -m indic_edge_embeddings.train.run_followup_campaign \
  --train-file quality_matched_v3/passages/train.parquet \
  --val-file quality_matched_v3/passages/validation.parquet \
  --native-eval-dir eval/miracl_hi_dev_fast_native \
  --romanized-eval-dir eval/miracl_hi_dev_fast_romanized \
  --english-eval-dir eval/scifact_en_test \
  --learning-rate 5e-6 \
  --num-epochs 1 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --checkpoints-per-epoch 1 \
  --output-dir /kaggle/working/outputs/campaign_quality_v3
```

## Outputs
Pending remote kernel completion.

## Interpretation
Pending results.

## Decision
Pending results.

## Next step
Pull outputs and checkpoints, record final metrics table, and analyze passage supervision vs answer supervision.
