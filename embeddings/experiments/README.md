# Embedding experiment history

Baseline measurements and a subsequent Romanization audit are recorded below. Browser validation and training are still pending.

Before recording a run, read the [shared experiment-record rules](../../AGENTS.md#experiment-records). Keep a dated directory here for each substantive experiment or roadmap step, including failed and interrupted attempts.

## Records

- [2026-09-15 baseline evaluation](2026-09-15-baseline-evaluation/README.md): Date 2026-09-15. Status: complete. Question: What are the baseline retrieval scores and INT8 export sizes for published compact multilingual models on native and Romanized Hindi? Decision: Confirmed E5-small as base model (Native nDCG@10: 0.5513; Romanized: 0.0006; INT8 size: 113.7 MB). Proceed to pilot fine-tuning on pilot_hindi_100k.

- [2026-09-15 Romanized baseline audit](2026-09-15-romanized-audit/README.md): Complete. Reproduced E5 Romanized nDCG@10 of 0.00055265; query/relevance mapping and corpus checks passed. Found residual Devanagari in 18/350 queries and unnatural synthetic spellings. E5 remains provisional; correct and inspect a separately versioned Romanization slice before the full training pilot. This qualifies the earlier training-readiness decision.

- [2026-09-15 Romanization v2](2026-09-15-romanization-v2/README.md): Complete. Conversion and split defects repaired; 99,994 training and 5,000 validation rows verified. E5 native nDCG@10 0.5513, corrected synthetic Romanized 0.0. Next: bounded E5 pilot preparation; browser validation remains pending.
