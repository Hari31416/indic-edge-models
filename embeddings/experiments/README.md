# Embedding experiment history

Baseline measurements, Romanization repairs, and the first training pilot are recorded below. Follow-up adaptation is planned; browser validation remains pending.

Before recording a run, read the [shared experiment-record rules](../../AGENTS.md#experiment-records). Keep a dated directory here for each substantive experiment or roadmap step, including failed and interrupted attempts.

## Records

- [2026-09-16 dataset builder v3](2026-09-16-dataset-builder-v3/README.md): Implementation complete. Matched answer/passage and mixed-source builders, Dakshina train spelling support, grouped exclusions, and explicit-negative/multi-positive training integration. 35 offline tests pass; production data build and retrieval comparison remain pending.

- [2026-09-16 training data quality](2026-09-16-training-data-quality/README.md): Complete. Inspected full pilot statistics and 60 seeded samples. Useful script-adaptation control, but 92.8% answer-based supervision, translation defects, one synthetic spelling scheme, and no explicit negatives limit passage-retrieval suitability. Recommend a separate corrected passage-supervision comparison.

- [2026-09-16 E5 adaptation follow-up](2026-09-16-e5-adaptation-follow-up/README.md): Complete. Evaluated conservative control (Native 0.4367, Romanized 0.0636), teacher alignment (Native 0.5387, Romanized 0.0059), and dual objective (Native 0.4342, Romanized 0.0501) on Kaggle GPU. Proved that teacher preservation protects native retrieval, while unregularized InfoNCE causes severe drift.

- [2026-09-15 baseline evaluation](2026-09-15-baseline-evaluation/README.md): Date 2026-09-15. Status: complete. Question: What are the baseline retrieval scores and INT8 export sizes for published compact multilingual models on native and Romanized Hindi? Decision: Confirmed E5-small as base model (Native nDCG@10: 0.5513; Romanized: 0.0006; INT8 size: 113.7 MB). Proceed to pilot fine-tuning on pilot_hindi_100k.

- [2026-09-15 Romanized baseline audit](2026-09-15-romanized-audit/README.md): Complete. Reproduced E5 Romanized nDCG@10 of 0.00055265; query/relevance mapping and corpus checks passed. Found residual Devanagari in 18/350 queries and unnatural synthetic spellings. E5 remains provisional; correct and inspect a separately versioned Romanization slice before the full training pilot. This qualifies the earlier training-readiness decision.

- [2026-09-15 Romanization v2](2026-09-15-romanization-v2/README.md): Complete. Conversion and split defects repaired; 99,994 training and 5,000 validation rows verified. E5 native nDCG@10 0.5513, corrected synthetic Romanized 0.0. Next: bounded E5 pilot preparation; browser validation remains pending.

- [2026-09-15 E5 pilot training](2026-09-15-e5-pilot-training/README.md): Complete. Fine-tuned E5-small on pilot_hindi_100k (3 epochs, Kaggle 2x T4). Romanized nDCG@10 improved from 0.0006 to 0.1059 (+176x), but native nDCG@10 regressed from 0.5513 to 0.3921 (-0.1592). Decision: Unregularized InfoNCE causes native representation drift. Formulate dual-objective training with teacher distillation and cross-script query alignment before INT8 export.
