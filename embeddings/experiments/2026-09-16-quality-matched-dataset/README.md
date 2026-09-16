# Production quality-matched dataset generation

## Question and criterion
Generate the production quality-matched training dataset (`quality_matched_v3`) for [roadmap Step 4](../../STEPS.md) using the verified v3 builder and pinned MSMARCO-XI train data.
Success criterion:
- Complete generation of matched `passages/` and `answers/` arms with identical query IDs, row order, script variants, and split memberships.
- Clean isolation between train and validation splits via connected-component grouping.
- Comprehensive audit reports (`report.json`, `raw_samples.json`, `rejected_samples.json`, `romanization_failures.json`).
- Verified raw sample characteristics conforming to Data Creation Guidelines.

## Setup
- **Config:** `embeddings/configs/quality_matched.json`
- **Builder:** `indic_edge_embeddings.data.build_quality_data` (sha256: `7bb83ae3e9f2d528df4032dad72185af0441a0d8d2f08c398e81f2697fe78b9e`)
- **Source dataset:** `ai4bharat/MSMARCO-XI` (`train/hintrain.parquet`), pinned commit `bf5cdc1f26e581e519018e434db14edd1b77602b` (sha256: `dd88b273ad1ba381ae44bb0fc318e48bde20357b5fabce5211d62b343ec1e700`)
- **Evaluation exclusions:** `embeddings/data/eval/miracl_hi_dev_native/queries.jsonl` (sha256: `87dfdbd0296bf90e1ad414fa3e75362164bdc4b4f2ad4c8f63ce7c3b98adae2c`), `embeddings/data/eval/miracl_hi_dev_romanized/queries.jsonl` (sha256: `9e934565bf6a49d35c317719f75c09c408cd569a2d5c143ffec34b6f7d96dda9`)
- **Seed:** 42
- **Validation fraction:** 0.05
- **Max queries:** 80,000
- **Romanized probability:** 0.3
- **Hardware & Runtime:** macOS arm64, Python 3.13.5 (`embeddings/.venv`)

## Execution
Status: complete.

Command:
```bash
PYTHONPATH=embeddings/src embeddings/.venv/bin/python \
  -m indic_edge_embeddings.data.build_quality_data \
  --config embeddings/configs/quality_matched.json \
  --output-dir embeddings/data/train/quality_matched_v3
```

Execution scanned all 778,638 MSMARCO-XI rows, filtered out unselected/uninformative passage sets, reservoir sampled 80,000 queries, formed 71,544 connected split groups, and generated both arms.

## Outputs
- **Dataset location:** `embeddings/data/train/quality_matched_v3/`
- **Output metrics:**
  - `passages/train`: 98,825 rows (76,000 native Hindi `hi/native->hi`, 22,825 Romanized Hindi `hi/romanized->hi`; sha256: `efc84cac4e071e54db07c1fa0c8ce0b39785de709bb7a0d300df5f3ae51daf28`)
  - `answers/train`: 98,825 rows (76,000 native Hindi `hi/native->hi`, 22,825 Romanized Hindi `hi/romanized->hi`; sha256: `6a537ca3d860ac0d381189ca7b923e7d272637271cf2ae82917290d37e61f326`)
  - `passages/validation`: 5,227 rows (4,000 native Hindi `hi/native->hi`, 1,227 Romanized Hindi `hi/romanized->hi`; sha256: `99c7f9698b991a9825359a713fbe7fc1df80b5dafc71850af11b21cf09f71565`)
  - `answers/validation`: 5,227 rows (4,000 native Hindi `hi/native->hi`, 1,227 Romanized Hindi `hi/romanized->hi`; sha256: `25160ffb317b61d0b94f4a18d866ccd98519923b10e809ff8f5aaada40fd7fc3`)
- **Inspection audits:**
  - `report.json`: Full builder telemetry, input/output checksums, and heuristic review flags (8,059 short queries, 843 long positives, 603 Hindi queries without Devanagari).
  - `raw_samples.json`: 200 independent random samples.
  - `rejected_samples.json`: 200 sampled rejections (all verified as missing query or unselected positive arrays).
  - `romanization_failures.json`: 0 failures recorded.
- **Parity verification:**
  - Split group overlap: 0 between train and validation.
  - Source identity overlap: 0 between train and validation.
  - Query ID and row order alignment between `passages/` and `answers/` arms: exact 1:1 match.
  - Dataset loader verification: `RetrievalDataset` loads 98,825 rows with `split_group_id`, `positive_texts`, and script variants preserved.

## Interpretation
The dataset provides a clean, controlled benchmark arm pairing:
- `passages/` trains models directly on full selected passage context rather than truncated extracted answers.
- `answers/` acts as an identical matched control arm to isolate the empirical effect of supervision granularity under equal update budgets and identical query exposures.
- Connected component splitting prevents split leakage across related query transformations.

## Decision
Accept `quality_matched_v3` as the standard matched comparison dataset for roadmap Step 4. Keep both arms immutable for training benchmarks.

## Next step
Run the matched training comparison between `passages/train.parquet` and `answers/train.parquet` on Kaggle GPU under identical hyperparameters (learning rate 5e-6, 1 epoch, batch size 32) using dual-objective / teacher preservation, and evaluate retrieval on MIRACL Hindi dev (native & Romanized).
