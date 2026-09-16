# Training data quality inspection

## Question and criterion
Assess whether the current pilot provides suitable supervision for [roadmap Step 4](../../STEPS.md). Inspect composition, seeded raw examples, duplication, script coverage, and supervision format. Success means an evidence-based suitability decision, not a training improvement.

## Setup
- Local `embeddings/data/train/pilot_hindi_100k` JSONL and Parquet; v2 transformation metadata. Actual input hashes and runtime are in `stats.json`.
- Inspection seed 20260916. Randomly inspect 15 training rows in each of four source/script strata, 60 total. This deliberately oversamples SWIM and is not a population error-rate estimate.
- Mac local Python 3.14.5 for audit; project virtualenv for Parquet equivalence check. Model, optimizer and training hardware: inapplicable.
- Upstream revisions were not recorded by the builder. Working tree had pre-existing adaptation changes; these were not modified by this inspection.

## Execution
Complete. Ran `python3 embeddings/experiments/2026-09-16-training-data-quality/audit.py`. Separately loaded both Parquet files with project-venv PyArrow and compared their row dictionaries with JSONL: exact equality for training and validation.

Read preparation code, collator and prior experiment records. Verified upstream schema from [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) and inspected the [SWIM source card](https://huggingface.co/datasets/nthakur/indic-swim-ir-cross-lingual). No downloads, dataset changes, training, or remote-run changes.

## Outputs
- `stats.json`: full-dataset counts and local JSONL hashes.
- `samples.json`: 60 seeded raw examples, with original IDs.
- Training: 99,994 rows, 76,889 source identities. MSMARCO-XI 92,773 rows, SWIM 7,221. Native 76,889, Romanized 23,105.
- Validation: 5,000 rows. No source-ID, split-group-ID or exact query overlap with training. There are 47 shared positive texts; this is not query leakage, but validation is not document-disjoint. No semantic or external evaluation contamination audit was performed.
- Zero rows with explicit negatives. Native-only training rows contain 952 repeated-positive occurrences beyond the first occurrence. Script augmentation repeats positives by design in addition to these.
- Positive whitespace-word counts: median 19, p90 47, p99 87, maximum 2,338. These are not tokenizer lengths; truncation rates remain unmeasured.
- All rows carry language=hi, including SWIM's Hindi-query/English-passage supervision. No dedicated English-query retention slice in this pilot.

## Interpretation
Useful for a bounded synthetic script-adaptation experiment, weak as the sole foundation for a general Hindi passage retriever.

The highest-priority issue is supervision selection: `load_msmarco_xi_records` uses `Answer` as positive and does not load `passages`. The upstream schema supplies `Translated_passages` and `is_selected`. Answer retrieval can provide useful semantic supervision, but predominantly short answers differ from the Wikipedia passage retrieval evaluation. This mismatch is observed; its causal contribution to regression has not been isolated.

Sample defects include:
- Sample 35: "वे किस चीज़ का इस्तेमाल शव को अंकुरित करने के लिए करते हैं" with an embalming-related answer. "अंकुरित" means sprouting; the translation damages the intended query.
- Sample 57: "deeyaa kaa kyaa arth hai" paired with a self-reliance/DIY-like answer. The intended acronym meaning appears lost.
- Sample 12: "बॉटलेग सीरीज का अग", an incomplete SWIM question.
- Sample 14 asks which two Saint Lucia teams won a double, but the passage only says two teams did, without naming them. Topical relevance is present, answer support is incomplete.
- Other examples are usable, such as the advisor definition in sample 43 and cricket match counts in sample 8. This inspection does not establish an overall defect percentage.

Romanization uses a single synthetic spelling scheme. Samples such as `dakshin aphreekaa men ...` are readable but cannot establish robustness to naturally typed Hinglish spelling or code switching. Shared train/evaluation conversion narrows the claim further.

The builder takes the first available source rows before later shuffling; it does not randomly sample the complete upstream datasets. Its simple length and placeholder filters do not detect the defects above. Provenance lacks pinned upstream revisions.

Current collator uses query/positive pairs only. With batch size 32, each query has 31 other passage positions as potential negatives; gradient accumulation does not enlarge this pool. Duplicate or related positives can become false negatives, although collision frequency was not measured.

The previous pilot's synthetic Romanized nDCG@10 rose from corrected baseline 0.0 to 0.1059 while native fell from 0.5513 to 0.3921. This establishes some learning signal and failure of preservation, not a causal diagnosis of data or optimizer problems.

## Decision
Keep this dataset as the historical control and narrow script-alignment resource. Do not scale the unchanged recipe on the assumption that volume will fix quality. Prioritize a separate version using selected source passages for retrieval, with inspected translation quality, explicit supervision directions, English-query retention, and controlled negatives. Preserve complete source/variant groups and consider document grouping for a stricter internal validation design.

No changes to the running/planned adaptation campaign are authorized or made by this review.

## Next step
For a future dataset revision, inspect upstream selected passages against answers and English originals on a bounded random sample, then compare a corrected passage-supervision pilot with the existing control. Add spelling diversity and independently sourced evaluation before claiming natural Romanized/code-mixed quality. Keep current artifacts unchanged so the comparison remains reproducible.
