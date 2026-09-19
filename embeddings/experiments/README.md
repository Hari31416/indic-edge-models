# Embedding experiment history

Baseline measurements, Romanization repairs, and the first training pilot are recorded below. Follow-up adaptation is planned; browser validation remains pending.

Before recording a run, read the [shared experiment-record rules](../../AGENTS.md#experiment-records). Keep a dated directory here for each substantive experiment or roadmap step, including failed and interrupted attempts.

## Records

- [2026-09-19 E5 full evaluation campaign](2026-09-19-full-eval-campaign/README.md): In progress. Evaluating 4 deployment model variants (`base_e5`, `lora_fp32`, `lora_int8`, `lora_int8_pruned`) across full 506k-document MIRACL Hindi and SciFact English benchmarks on Kaggle GPU to establish full quantitative retrieval and quantization/pruning parity.

- [2026-09-18 E5 LoRA scaled](2026-09-18-e5-lora-scaled/README.md): Complete. Scaled parameter-isolated `LoRA Symmetric` adaptation ($r=16, \alpha=32$) to 3 epochs (2,316 steps) on Kaggle dual T4 GPUs. Romanized nDCG@10 reached 0.3937 (+26.0x over baseline) with 0.5403 Recall@20 (+11.5x gain) while retaining 0.8645 Native Devanagari nDCG@10 (0.9763 Recall@20) and 0.6489 English nDCG@10. Accepted `LoRA Symmetric` 3-epoch checkpoint for ONNX export and in-browser deployment.

- [2026-09-17 E5 LoRA adaptation](2026-09-17-e5-lora-adaptation/README.md): Complete. Evaluated parameter-isolated Low-Rank Adaptation (LoRA, $r=16, \alpha=32$, 1.13% trainable params) on frozen `multilingual-e5-small`. Accepted `LoRA Symmetric` achieving 0.3414 Romanized nDCG@10 (22.5x gain over baseline) and 0.6571 English nDCG@10 with a modest -0.0497 native Devanagari drop (0.8692 nDCG@10, 0.9728 Recall@20). Next: ONNX export and browser INT8 quantization.

- [2026-09-16 E5 quality-matched adaptation](2026-09-16-e5-quality-matched-adaptation/README.md): Complete. Evaluated 3 adaptation paradigms on 98.8k passage supervision rows (`quality_matched_v3/passages`). Found full-encoder InfoNCE achieves 0.3256 Romanized nDCG@10 but drops Native Devanagari by -0.0391; pairwise teacher regression aligns coordinates without learning discriminative ranking (+0.0004); dual objective suffers gradient conflict. Reject full-weight fine-tuning; proceed to parameter-isolated LoRA.

- [2026-09-16 quality-matched dataset](2026-09-16-quality-matched-dataset/README.md): Complete. Generated `quality_matched_v3` dataset (98,825 training and 5,227 validation rows per arm; 76k native, 22.8k Romanized train). Verified 1:1 order/query alignment between `passages/` and `answers/` arms, 0 split leakage across 71,544 connected groups, and clean raw sample quality.

- [2026-09-16 dataset builder v3](2026-09-16-dataset-builder-v3/README.md): Implementation complete. Matched answer/passage and mixed-source builders, Dakshina train spelling support, grouped exclusions, and explicit-negative/multi-positive training integration. 35 offline tests pass; production data build and retrieval comparison remain pending.

- [2026-09-16 training data quality](2026-09-16-training-data-quality/README.md): Complete. Inspected full pilot statistics and 60 seeded samples. Useful script-adaptation control, but 92.8% answer-based supervision, translation defects, one synthetic spelling scheme, and no explicit negatives limit passage-retrieval suitability. Recommend a separate corrected passage-supervision comparison.

- [2026-09-16 E5 adaptation follow-up](2026-09-16-e5-adaptation-follow-up/README.md): Complete. Evaluated conservative control (Native 0.4367, Romanized 0.0636), teacher alignment (Native 0.5387, Romanized 0.0059), and dual objective (Native 0.4342, Romanized 0.0501) on Kaggle GPU. Proved that teacher preservation protects native retrieval, while unregularized InfoNCE causes severe drift.

- [2026-09-15 baseline evaluation](2026-09-15-baseline-evaluation/README.md): Date 2026-09-15. Status: complete. Question: What are the baseline retrieval scores and INT8 export sizes for published compact multilingual models on native and Romanized Hindi? Decision: Confirmed E5-small as base model (Native nDCG@10: 0.5513; Romanized: 0.0006; INT8 size: 113.7 MB). Proceed to pilot fine-tuning on pilot_hindi_100k.

- [2026-09-15 Romanized baseline audit](2026-09-15-romanized-audit/README.md): Complete. Reproduced E5 Romanized nDCG@10 of 0.00055265; query/relevance mapping and corpus checks passed. Found residual Devanagari in 18/350 queries and unnatural synthetic spellings. E5 remains provisional; correct and inspect a separately versioned Romanization slice before the full training pilot. This qualifies the earlier training-readiness decision.

- [2026-09-15 Romanization v2](2026-09-15-romanization-v2/README.md): Complete. Conversion and split defects repaired; 99,994 training and 5,000 validation rows verified. E5 native nDCG@10 0.5513, corrected synthetic Romanized 0.0. Next: bounded E5 pilot preparation; browser validation remains pending.

- [2026-09-15 E5 pilot training](2026-09-15-e5-pilot-training/README.md): Complete. Fine-tuned E5-small on pilot_hindi_100k (3 epochs, Kaggle 2x T4). Romanized nDCG@10 improved from 0.0006 to 0.1059 (+176x), but native nDCG@10 regressed from 0.5513 to 0.3921 (-0.1592). Decision: Unregularized InfoNCE causes native representation drift. Formulate dual-objective training with teacher distillation and cross-script query alignment before INT8 export.
