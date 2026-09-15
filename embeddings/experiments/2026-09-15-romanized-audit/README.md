# Romanized baseline audit

## Question and criterion

Do the near-zero Romanized baseline scores reflect model behavior on valid synthetic Hindi queries? [Roadmap steps 2 and 3](../../STEPS.md). Success requires preserved query/relevance/corpus identity, inspected transformations, correct encoding settings, and a reproducible score check before training.

## Setup

- Existing MIRACL Hindi dev native and Romanized manifests: 350 queries each, 506,264 passages. Exact file SHA-256 hashes in [audit.json](audit.json). Original dataset revision was not recorded by the baseline; hashes identify the audited local assets.
- E5-small cached snapshot: `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Source checkout: `7fc85c7332bf63cf271b449ef14bdc85576c1db6`, plus the audit script in this directory.
- Local Mac M4 Pro, CPU inference, batch size 32, maximum length 512, mean pooling, L2 normalization. Query prefix `query: ` and passage prefix `passage: `. A separate no-query-prefix diagnostic was also run.
- Python 3.13 environment; torch 2.14.0, transformers 5.17.0, numpy 2.5.3, indic-transliteration 2.3.82. These differ from the original README's stated torch version.
- Seed 20260915 for 20 query examples and 24 corpus-row checks. No training; training seed inapplicable.
- Reused `embeddings/data/cache/corpus_emb_8e0da0ebae.npy`. Shape and SHA-256 recorded. No network access or new model/data downloads.

## Execution

Status: audit complete. E5 evaluation reproduced; Granite and Bekko numerical reruns were not performed.

From repository root:

```bash
PYTHONPATH=embeddings/src embeddings/.venv/bin/python embeddings/experiments/2026-09-15-romanized-audit/audit.py > embeddings/experiments/2026-09-15-romanized-audit/run.log 2>&1
PYTHONPATH=embeddings/src embeddings/.venv/bin/python -m pytest embeddings/tests/test_metrics.py embeddings/tests/test_loader.py embeddings/tests/test_data.py -q
```

The subset and independent nDCG calculations were executed separately after the initial run, then incorporated into `audit.py` for reproduction. Inspected the staged Granite/Bekko bootstrap commands and packaged evaluator/transliteration code. Their commands specify CLS/mean pooling respectively; their historical model revisions and full execution environments remain unverified.

## Outputs

| E5 condition | Original nDCG@10 | Audit nDCG@10 | Audit Recall@20 |
| --- | ---: | ---: | ---: |
| Native Hindi | 0.55132143 | 0.55133078 | 0.77664626 |
| Existing synthetic Romanized Hindi | 0.00055265 | 0.00055265 | 0.00428571 |
| Romanized, missing query prefix diagnostic | unmeasured | 0 | 0 |

- All 350 source query mappings and relevance judgments match. No missing relevant documents, empty queries, duplicate Romanized query IDs, or qrels without queries.
- Native and Romanized corpus files have identical SHA-256 hashes, including row order.
- Fresh encodings of 24 sampled corpus rows match cached vectors: minimum cosine 0.99999994, mean 1.0. This is a sample check, not complete cache re-encoding.
- All 350 Romanized queries reproduce with the installed ITRANS implementation. No evidence that the character-replacement fallback generated these local queries.
- 18/350 queries retain Devanagari characters. The other 332 still score only 0.00058261 nDCG@10, so residual script alone does not explain the collapse.
- Independent NumPy nDCG calculation agrees with the project metric. Existing data/loader/metric checks: 10 passed.
- Twenty seeded raw query pairs are recorded in `audit.json`. Examples include `भारत का गणतंत्र दिवस किस तारीख पर आता है?` becoming `bharata ka ganatamtra divasa kisa tarikha para ata hai?`, `वैज्ञानिक` becoming `vaij~nanika`, and `अलीगढ़` becoming `aligaढ़`. These illustrate extra final vowels, transliteration notation, and incomplete conversion. This inspection is not a human-annotated natural-query benchmark.
- [audit.json](audit.json) contains metrics, sample checks and provenance. Verbose logs and ranking dumps were removed after inspection; `audit.py` regenerates top-20 rankings when rerun. Generated logs and rankings are ignored by Git.

## Interpretation

The E5 near-zero result is reproducible on the current synthetic evaluation. The inspected ID mapping, corpus ordering, sampled cache vectors, and metric implementation do not explain it as a scoring error. The tiny native discrepancy does not change the ranking or conclusion; its precise numerical cause was not isolated.

The transformation is not a validated representation of naturally typed Romanized Hindi or Hinglish. Lowercasing ITRANS loses case distinctions; the output retains extra vowels, notation, and sometimes Devanagari. This establishes failure on this specific synthetic slice. It does not prove all compact models universally fail or that fine-tuning is the only effective remedy.

The evaluator cache key also omits corpus content/order, model revision, and pooling. That is a reproducibility risk for later runs, although the sampled E5 cache check passed here. Historical Granite/Bekko scores remain recorded results, not independently reproduced results from this audit. No browser performance or export-quality evaluation was run.

## Decision

Keep E5-small as the provisional fine-tuning base on the strength of native retrieval. Preserve the baseline numbers as valid measurements of the existing synthetic slice, with narrower interpretation.

Do not treat this slice alone as sufficient validation for realistic Romanized retrieval training. Before launching the full pilot, create a separately versioned, inspected Romanization evaluation with explicit normalization and no unintended residual Devanagari. Check the training transformations too, since the training builder imports the same function. Preserve the existing slice as a diagnostic rather than overwriting it. A natural-query benchmark remains missing.

## Next step

Correct and version the transliteration data path, inspect raw examples, compare E5 on both old and corrected development slices, and include a local transliteration-plus-embedding comparison. Freeze evaluation identities before tuning. Harden corpus-cache identity before model/pooling/data changes. Record that work as a new experiment; this audit does not authorize or begin a training run.

## Follow-up

[Romanization v2](../2026-09-15-romanization-v2/README.md) completes conversion repair and a corrected E5 comparison. Its decision supersedes the conversion-related training hold above. The audit script now explicitly selects legacy ITRANS to keep this historical run reproducible after the default transformation changed.
