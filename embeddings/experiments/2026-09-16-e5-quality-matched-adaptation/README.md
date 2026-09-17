# Quality-matched passage supervision adaptation campaign

## Question and criterion
Roadmap: [Step 4](../../STEPS.md).
Evaluate whether training on genuine selected-passage supervision (`quality_matched_v3/passages`) improves Hindi retrieval and script adaptation across the 3 follow-up adaptation paradigms:
1. `conservative_control` (InfoNCE, LR 5e-6, 1 epoch)
2. `teacher_align` (Frozen teacher $L_{script} + L_{preserve}$, LR 5e-6, 1 epoch)
3. `dual_objective` ($L_{script} + L_{preserve} + L_{retrieval}$, LR 5e-6, 1 epoch)

Success criteria:
- Native retention gate: within 0.02 of reference baseline (>= 0.8989).
- English retention gate: within 0.02 of reference baseline (>= 0.6570).
- Romanized retrieval: evaluate progression and contrastive discriminability against distractor passages (target >= 0.4000).

## Setup
- **Base Student Model:** `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`)
- **Frozen Teacher Model:** `intfloat/multilingual-e5-small`
- **Training Data:** `quality_matched_v3/passages/train.parquet` (98,825 rows; 76,000 native, 22,825 Romanized)
- **Validation Data:** `quality_matched_v3/passages/validation.parquet` (5,227 rows)
- **Evaluation Manifests:** `eval/miracl_hi_dev_fast_native`, `eval/miracl_hi_dev_fast_romanized`, `eval/scifact_en_test`
- **Attached Dataset:** `hari31416/indic-edge-hindi-pilot`
- **Hyperparameters:** Batch size 32, gradient accumulation 4 (effective batch size 128), learning rate 5e-6, 1 epoch, seed 42
- **Hardware:** Kaggle Nvidia T4-2x GPU (execution duration: 89.5 minutes)

## Execution
Status: complete.

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

| Experiment               | Checkpoint        | Native nDCG@10       | Native Recall@20 | Romanized nDCG@10    | Romanized Recall@20 | English nDCG@10      | English Recall@20 | Decision                      |
| :----------------------- | :---------------- | :------------------- | :--------------- | :------------------- | :------------------ | :------------------- | :---------------- | :---------------------------- |
| **Base Model Reference** | `pretrained_base` | **0.9189**           | 0.9856           | **0.0152**           | 0.0468              | **0.6770**           | 0.8469            | REFERENCE BASELINE            |
| **Conservative Control** | `final_model`     | **0.8798** (-0.0391) | 0.9720           | **0.3256** (+0.3104) | 0.4539              | **0.6569** (-0.0201) | 0.8419            | REJECT (Native drop: -0.0391) |
| **Teacher Alignment**    | `final_model`     | **0.9128** (-0.0061) | 0.9769           | **0.0156** (+0.0004) | 0.0356              | **0.6700** (-0.0070) | 0.8473            | REJECT (No Roman gain)        |
| **Dual-Objective**       | `final_model`     | **0.8633** (-0.0556) | 0.9706           | **0.0908** (+0.0756) | 0.1445              | **0.6777** (+0.0007) | 0.8562            | REJECT (Native drop: -0.0556) |

## Interpretation
1. **Teacher Alignment without contrastive negatives causes non-discriminative embedding collapse:**
   - Minimizing cosine distance between $S(q_{roman})$ and $T(q_{native})$ ($L_{script}$) pulls Romanized representations into the broad native embedding space ($\cos \approx 0.90$), but because there are no negative distractors or temperature-scaled cross-entropy losses, the student learns 0 ranking discriminability (nDCG@10 = 0.0156 vs 0.0152 baseline).
2. **Conservative Control (InfoNCE) learns strong discriminative mapping but suffers representation drift:**
   - InfoNCE with in-batch negatives successfully drives Romanized nDCG@10 from 0.0152 to 0.3256.
   - However, updating all 12 transformer encoder layers on the mixed dataset causes gradient interference that degrades native Devanagari embeddings (-0.0391 drop, violating the 0.02 preservation gate).
3. **Dual-Objective suffers from gradient conflict:**
   - $L_{preserve}$ (which penalizes moving away from pretrained teacher vectors) directly opposes $L_{retrieval}$ (which requires moving representations to separate hard negatives).
   - This conflict suppresses Romanized retrieval (only 0.0908 vs 0.3256 in InfoNCE) while still degrading native retrieval (-0.0556).

## Decision
Reject monolithic full-weight fine-tuning for cross-script adaptation on mE5-small. Full-encoder parameter updates create an inescapable tension between Devanagari stability and Romanized adaptation.

## Next step

Follow up with [2026-09-17 E5 LoRA adaptation](../2026-09-17-e5-lora-adaptation/README.md) to implement parameter-isolated Low-Rank Adaptation (LoRA) on `multilingual-e5-small`.
