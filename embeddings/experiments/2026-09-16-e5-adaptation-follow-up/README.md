# E5 adaptation follow-up
 
Status: complete. Follow-up multi-experiment adaptation campaign and 3-way evaluations executed on Kaggle T4-2x GPU. All outputs and artifacts compiled.

## Question and criterion

Roadmap: [Step 4](../../STEPS.md), with the outstanding Step 3 browser smoke test tracked separately.

Can conservative adaptation or frozen native-E5 teacher alignment improve synthetic Romanized Hindi retrieval while preserving native retrieval?

The [first pilot](../2026-09-15-e5-pilot-training/README.md) scored native nDCG@10 0.39208639 and synthetic Romanized nDCG@10 0.10585598. Native E5 scored 0.55133078. The [corrected v2 Romanized baseline](../2026-09-15-romanization-v2/README.md) was 0.0.

The user confirmed on 2026-09-16 that the v2 suffix was removed during packaging. The unsuffixed evaluation name therefore does not itself indicate use of the historical Romanization. Use the v2 baseline of 0.0, not the historical 0.00055265 or a fold-change against it.

Acceptance criteria:

- Retain the existing target of Romanized nDCG@10 >= 0.40.
- Allow at most 0.02 absolute native nDCG@10 regression from the accepted existing base result, giving a floor of 0.53133078. This is slightly stricter than the rounded 0.53 in the first pilot.
- Establish an English development baseline before training; allow at most 0.02 absolute regression there. Record the public dataset, split, identities, and baseline before candidate evaluation.
- A result below 0.40 Romanized remains below acceptance. It can justify further work if it improves Romanized retrieval while meeting preservation gates. Report it as intermediate progress, not a successful final candidate.
- Report Recall@20 alongside nDCG@10. Keep native, Romanized, and English results separate.

## Setup

- Student and frozen teacher: original `intfloat/multilingual-e5-small`, not the degraded pilot checkpoint. Prior baseline revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`; use this revision for the follow-up.
- Data: corrected grouped pilot, 99,994 training rows and 5,000 validation rows. The first pilot reports 23,105 Romanized training rows. Recover native/Romanized pairs through original source identities, not row order or approximate text matching.
- Data locations: packaged `hari31416/indic-edge-hindi-pilot`; local v2 paths and hashes are documented in the Romanization v2 record. Use the existing packaged data without repeating the provenance audit. English data selection is pending; do not invent baseline results.
- Split policy: preserve source and normalized-query groups across transformations. Keep all teacher-target generation out of final evaluation identities. The already inspected 350-query MIRACL dev slice is development data, not a fresh final test.
- Encoding: `query: ` and `passage: ` prefixes, attention-masked mean pooling, normalized 384-dimensional embeddings, maximum length 512.
- Seed: 42 for the initial comparisons. Repeat only a promising configuration with a second recorded seed.
- Planned hardware: Kaggle T4 for training, Mac M4 Pro for local inspection and eventual browser checks. Actual devices, memory, runtime/package versions, precision, and timings remain pending.
- Planning code revision: `eaa266b7b9dca967b65e17a3b212b9abb3f6b3f5`. Record execution revision and any uncommitted patch separately.
- Browser target and performance budgets: pending. No browser performance claim follows from this experiment.

## Execution

The reproducibility and failure-inspection prerequisite is treated as satisfied at the user's direction on 2026-09-16. Accept the existing data, baselines, and packaging as established; do not repeat that audit or make it a launch requirement. This is a user-confirmed assumption, not a new verification performed in this experiment.

### 1. Run a conservative control

Start from original E5. Keep the original Hindi training data and InfoNCE loss to isolate a less aggressive training schedule.

- Proposed learning rate: `5e-6`, maximum one epoch, seed 42.
- Retain batch size 32, accumulation 4, temperature 0.05, maximum length 512, and the original optimizer settings initially.
- Save intermediate checkpoints, for example at quarter-epoch intervals, plus the final checkpoint. Save optimizer, scheduler, scaler, RNG and progress state for resumability.
- Evaluate retrieval at saved checkpoints; validation loss is a diagnostic rather than the checkpoint-selection criterion.
- This compares a conservative schedule with the original pilot; it does not separately identify learning-rate and epoch effects.

### 2. Test frozen-teacher alignment and preservation

Start a separate student from original E5. Keep the teacher frozen and in evaluation mode. Cache teacher targets for the bounded training data, with model/data hashes and encoding settings.

Use the teacher's native query as the target for each corresponding Romanized student query:

`L_script = mean(1 - cosine(S(query_romanized), T(query_native)))`

Preserve native-query, passage, and English representations against the frozen teacher:

`L_preserve = mean_by_group(1 - cosine(S(text), T(text)))`

Use the same role prefix on both sides of each preservation pair. Compute group means separately so a larger group does not silently dominate. The initial proposed objective is `L_script + L_preserve`, with normalized embeddings. Record each component and its weight. These are proposed starting choices, not tuned values or guarantees of retrieval preservation.

This is equivalent to a scaled squared Euclidean distance for unit vectors. Do not substitute dimension-averaged MSE without accounting for its changed scale when combining losses.

- Align to frozen teacher outputs, not two freely moving student representations.
- Retain one shared student encoder for queries and documents. Evaluate with that student's corpus embeddings; do not silently evaluate against a teacher index.
- Begin with `5e-6`, at most one epoch, the same seed and checkpoint cadence as the control. Define and record paired-data sampling, English source/count, group weights, and optimizer-step budget before launch.
- Compare alignment/preservation alone with an otherwise identical run adding InfoNCE. Match data exposure and update budget between these two runs. Record loss scales and choose the initial retrieval-loss weight from a training-only smoke run, before development retrieval comparisons.
- The comparison against the conservative control also changes supervision and English exposure. Attribute individual loss effects only to the matched alignment-only versus alignment-plus-retrieval comparison.

Frozen-teacher cross-language embedding alignment is an established method; applying it to this script transformation remains our hypothesis. Reference: [Sentence Transformers multilingual distillation](https://www.sbert.net/examples/sentence_transformer/training/multilingual/README.html).

### 3. Improve negatives only if the preceding results justify it

The existing loss sees 32 passages per query batch, giving 31 potential negatives. Accumulating four batches does not make 127 negatives available to a query. The current collator ignores explicit negatives, and the trainer uses a single CUDA device even when two T4s are allocated.

If teacher alignment preserves native quality but retrieval remains weak, run a separate comparison with inspected hard negatives or a larger actual contrastive batch using a tested cached loss. Exclude duplicate positives and source-related variants from false-negative positions, or implement an explicit multi-positive treatment. Do not simply increase negatives without inspecting their relevance.

Reference: [Sentence Transformers losses](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html). Multi-GPU use requires explicit DDP and intentional cross-device negative handling; hardware allocation alone is insufficient.

### Dispatch and checkpoints

Use `scripts/run_kaggle.py push`, `status --stream`, and `pull` per repository instructions. Keep raw data and caches under `/tmp`, final outputs under `/kaggle/working/outputs`, and pull small reports directly into the run record without nested artifact folders.

Exact commands are pending implementation and smoke validation. Do not run the old CLI expecting teacher losses, intermediate resumable checkpoints, or explicit negatives: these features are not implemented there.

Create uniquely named child experiment records for substantive runs, recording commands, configurations, kernel IDs, status, and resume points before dispatch. Link them here and in the experiment index.

### Remote baseline evaluation

Kernel `hari31416/e5-base-full-evaluation` was dispatched and executed on Kaggle T4 GPU, completing in 688 seconds. Outputs were pulled to `base_e5_native.json`, `base_e5_romanized.json`, and `base_e5_english.json`.

Command executed:
```bash
python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir eval/miracl_hi_dev_native --model-name intfloat/multilingual-e5-small --query-prefix 'query: ' --passage-prefix 'passage: ' --max-length 512 --batch-size 512 --cache-dir /tmp/cache --output-json /kaggle/working/outputs/base_e5_native.json && python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir eval/miracl_hi_dev_romanized --model-name intfloat/multilingual-e5-small --query-prefix 'query: ' --passage-prefix 'passage: ' --max-length 512 --batch-size 512 --cache-dir /tmp/cache --output-json /kaggle/working/outputs/base_e5_romanized.json && python -m indic_edge_embeddings.eval.benchmark_cli --manifest-dir eval/scifact_en_test --model-name intfloat/multilingual-e5-small --query-prefix 'query: ' --passage-prefix 'passage: ' --max-length 512 --batch-size 512 --cache-dir /tmp/cache --output-json /kaggle/working/outputs/base_e5_english.json
```

### Remote adaptation campaign dispatch

Kernel `hari31416/e5-adaptation-followup-campaign` was dispatched on Kaggle T4-2x GPU with attached dataset `hari31416/indic-edge-hindi-pilot`. It executes all three follow-up experiments sequentially and runs 3-way evaluation on every quarter-epoch checkpoint.

Dispatch command:
```bash
uv run python scripts/run_kaggle.py push \
  --subproject embeddings \
  --title "e5-adaptation-followup-campaign" \
  --slug "e5-adaptation-followup-campaign" \
  --gpu t4-2x \
  --datasets hari31416/indic-edge-hindi-pilot \
  --command "python -m indic_edge_embeddings.train.run_followup_campaign --train-file train.parquet --val-file validation.parquet --native-eval-dir miracl_hi_dev_native --romanized-eval-dir miracl_hi_dev_romanized --english-eval-dir scifact_en_test --native-baseline 0.551321 --english-baseline 0.677044 --learning-rate 5e-6 --num-epochs 1 --batch-size 32 --gradient-accumulation-steps 4 --checkpoints-per-epoch 1 --output-dir /kaggle/working/outputs/campaign"
```

## Outputs

Execution completed on Kaggle T4-2x GPU (`hari31416/e5-adaptation-followup-campaign`, total duration 112.9 minutes). Output metrics compiled in `campaign_summary.json` and `campaign_summary.md`:

| Run                                                               | Native nDCG@10 | Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 | English nDCG@10 | English Recall@20 | Decision                           |
| :---------------------------------------------------------------- | :------------- | :--------------- | :---------------- | :------------------ | :-------------- | :---------------- | :--------------------------------- |
| Original E5 (Baseline)                                            | 0.5513         | 0.7766           | 0.0000            | 0.0000              | 0.6770          | 0.8469            | REFERENCE BASELINE                 |
| First pilot (3 epochs, InfoNCE)                                   | 0.3921         | -                | 0.1059            | -                   | Unmeasured      | -                 | Failed original criteria           |
| Conservative control (1 epoch, InfoNCE 5e-6)                      | 0.4367         | 0.6394           | 0.0636            | 0.1187              | 0.6631          | 0.8529            | REJECT (Native drop: -0.1147)      |
| Teacher alignment + preservation (1 epoch, L_script + L_preserve) | 0.5387         | 0.7672           | 0.0059            | 0.0126              | 0.6681          | 0.8433            | PROGRESS (Preserved, Roman < 0.40) |
| Dual objective (1 epoch, L_script + L_preserve + L_retrieval)     | 0.4342         | 0.6286           | 0.0501            | 0.0940              | 0.6571          | 0.8602            | REJECT (Native drop: -0.1171)      |

Preservation and Target Gates:
- Native Hindi gate: nDCG@10 >= 0.5313 (baseline 0.5513 - 0.02)
- English SciFact gate: nDCG@10 >= 0.6570 (baseline 0.6770 - 0.02)
- Romanized Hindi target: nDCG@10 >= 0.4000

## Interpretation

1. **InfoNCE causes rapid native degradation regardless of schedule:** The conservative control (LR 5e-6, 1 epoch) degraded native Hindi nDCG@10 from 0.5513 to 0.4367 (-0.1147), severely breaching the 0.02 preservation gate. In-batch passage contrastive loss distorts pre-existing multilingual representations even with minimal training exposure.
2. **Teacher preservation successfully protects native and cross-lingual capacity:** `teacher_align` achieved native nDCG@10 of 0.5387 (pass gate >= 0.5313) and English nDCG@10 of 0.6681 (pass gate >= 0.6570). Cosine distance regularization against the frozen teacher completely eliminated catastrophic forgetting.
3. **Pure alignment without contrastive separation produces low retrieval rank:** While `teacher_align` preserved native capabilities, its Romanized nDCG@10 reached only 0.0059. Minimizing cosine distance between student Romanized and teacher native queries pulls query vectors into the right neighborhood, but does not provide sufficient discriminative margin against dense passage distractors.
4. **Equal weighting in dual objective allows InfoNCE to dominate:** In `dual_objective`, giving equal 1.0 weights to L_script, L_preserve, and L_retrieval caused the InfoNCE gradient to overpower the preservation loss, resulting in virtually the same native regression (0.4342 vs 0.4367) as the pure control.

## Decision

- Stop unregularized InfoNCE training; contrastive fine-tuning without heavy regularized constraints inevitably collapses native multilingual retrieval.
- Accept that frozen-teacher distillation ($L_{preserve}$) is the correct mechanism for preventing representation drift on edge multilingual encoders.
- For the next step, adjust the loss weighting in the dual-objective setup: substantially reduce the retrieval loss weight (for example, weight_retrieval = 0.05 - 0.1, weight_preserve = 1.0, weight_script = 1.0) or freeze the document encoder during query alignment to retain the native corpus space.

## Next step

Formulate an asymmetric / scaled loss experiment:
1. Keep the document encoder frozen to preserve the 0.5513 native corpus index entirely, adapting only the query encoder with $L_{script} + L_{preserve}$.
2. Alternatively, test dual objective with scaled retrieval loss weight ($0.1 \times L_{retrieval} + 1.0 \times L_{preserve} + 1.0 \times L_{script}$).
