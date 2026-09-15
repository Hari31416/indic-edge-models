# Hindi Romanization v2

## Question and criterion

Fix conversion defects from the [Romanized audit](../2026-09-15-romanized-audit/README.md), preserve original data, and establish a corrected E5 baseline before the pilot. Roadmap steps 2–4. Criteria: no residual Devanagari, preserve English tokens, inspect seeded samples, preserve evaluation identities, use a versioned training transformation, and keep source variants out of opposite splits.

## Setup

- Existing local MIRACL Hindi dev: 350 queries and 506,264 passages. Existing pilot: 80,769 native source rows. Original upstream data revisions were not recorded; input/output SHA-256 hashes are in `data_report.json`.
- E5-small snapshot `614241f622f53c4eeff9890bdc4f31cfecc418b3`, unchanged model and audited corpus vectors. Hash validation precedes evaluation.
- Mac M4 Pro, CPU, Python 3.13, torch 2.14.0, transformers 5.17.0, numpy 2.5.3, indic-transliteration 2.3.82. Mean pooling, normalized embeddings, length 512, batch 32, `query: ` and `passage: ` prefixes.
- Source base `7fc85c7332bf63cf271b449ef14bdc85576c1db6` plus the changes accompanying this record. Split seed 42; inspection seed 20260915. No new downloads, training, browser measurements, or model export. Training hyperparameters inapplicable.

## Execution

Status: complete for conversion repair, regenerated data, and E5 comparison. Training not started.

Implemented `hindi_ascii_v2`: Unicode normalization; explicit nukta and borrowed-vowel handling; ASCII folding; conservative final/medial schwa removal; English-token preservation; strict errors for unsupported input. Removed the silent fallback. Explicit legacy modes remain available. This is approximate rule-based Romanization.

Evaluation generation writes a new `_romanized_v2` slice and preserves historical data. Training now groups source variants and identical normalized native query text in the same split. Existing nonempty output directories are rejected by the builders.

From repository root:

```bash
PYTHONPATH=embeddings/src embeddings/.venv/bin/python embeddings/experiments/2026-09-15-romanization-v2/prepare.py
PYTHONPATH=embeddings/src embeddings/.venv/bin/python embeddings/experiments/2026-09-15-romanization-v2/evaluate.py
PYTHONPATH=embeddings/src embeddings/.venv/bin/python embeddings/experiments/2026-09-15-romanization-v2/verify.py
PYTHONPATH=embeddings/src embeddings/.venv/bin/python -m pytest embeddings/tests -q
```

Preparation refuses to overwrite existing targets. Reproduction requires a fresh copy of the workspace data layout or explicit relocation of previously generated targets. `verify.py` checks the initial/final preparation comparison and uses the preserved initial report.

The first rebuild fixed source-ID overlap but left 54 equal native query strings across splits. It was superseded by normalized-query grouping. A compact initial report remains as `initial_data_report.json`; initial datasets remain in ignored directories ending `_v2_initial`. Final evaluation files are byte-identical to the scored initial v2 files, as verified by hashes, so the split-only change did not require another evaluation run.

## Outputs

| E5 development slice | nDCG@10 | Recall@20 |
| --- | ---: | ---: |
| Native Hindi | 0.55133078 | 0.77664626 |
| Historical ITRANS | 0.00055265 | 0.00428571 |
| Corrected Hindi ASCII v2 | 0.0 | 0.0 |

- Final evaluation: 350 queries, zero residual Devanagari; unchanged relevance judgments, corpus and query ancestry.
- Final pilot: 99,994 training rows and 5,000 validation rows, including 24,227 Romanized rows. The six-row shortfall preserves complete groups within size limits.
- Zero shared source IDs or normalized-query groups between splits. The original split had 2,161 shared source IDs.
- Two malformed source rows were quarantined in `rejected_training_rows.json`: an orphan nukta and an unsupported character. Original data remains intact.
- All 104,994 generated training rows and all 350 evaluation queries passed transformation, label preservation, English-token preservation, script and identity checks. Input/output hashes passed. See `verification.json`.
- 37 tests passed, including missing-dependency failure, nukta equivalence, vowel preservation, punctuation/digits, English preservation, and grouped splitting.
- `data_report.json` records 20 seeded evaluation examples and 20 final training samples. Inspected initial training samples too. Corrected examples include `कितने` → `kitne`, `वैज्ञानिक` → `vaigyaanik`, and `अलीगढ़` → `aleegarh`. Spellings such as `onalaain` and `raajadhaanee` remain approximate. Public source supervision is noisy: samples include awkward translations, medical queries, and a statement used as a query. Romanization does not repair those relevance/translation issues.
- Metrics are retained in `evaluation.json`. Verbose logs and top-20 ranking dumps were removed after inspection; scripts regenerate ranking dumps when rerun. These generated files are ignored by Git.
- Local generated datasets: `embeddings/data/eval/miracl_hi_dev_romanized_v2` and `embeddings/data/train/pilot_hindi_100k_romanized_v2`. Large files remain ignored, with hashes and preparation code tracked.

## Interpretation

The concrete script-conversion and split-leakage defects are repaired. E5 still fails on this corrected synthetic slice; conversion cleanup alone does not establish usable retrieval. The comparison does not establish that every plausible Romanization fails, nor that these rules model natural Hinglish spelling. Native retrieval remains unchanged to the reported precision.

Only E5 was rerun. Granite/Bekko, natural-query evaluation, transliteration-plus-embedding comparison, export ranking parity and browser performance remain unmeasured in this step. The general evaluator's incomplete cache key remains a separate risk; this experiment validates exact corpus/cache hashes and uses unchanged weights instead of relying on that key.

## Decision

Keep E5-small as the base for the next bounded adaptation pilot. Use v2 as the frozen synthetic development diagnostic; retain the old slice for history. Use the corrected, grouped pilot data instead of the original leaking split. Conversion repair is complete; do not interpret that as completion of training preparation or browser validation.

## Next step

Begin the pilot phase with a recorded training configuration and a small smoke run. Inspect/select supervision, add an English regression baseline and appropriate negatives, freeze acceptance criteria and final evaluation identities, and use a distinct validated corpus cache for changed checkpoints. Measure native and synthetic Romanized retrieval separately before compression. Natural Romanization and transliteration-plus-embedding comparisons remain follow-ups; no natural Hinglish superiority claim is supported.
