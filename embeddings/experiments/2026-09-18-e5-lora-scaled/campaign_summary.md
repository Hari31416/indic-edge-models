# LoRA Adaptation Campaign Summary

Execution duration: 51.4 minutes

| Experiment | Checkpoint | Native nDCG@10 | Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 | English nDCG@10 | English Recall@20 | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| base_model | pretrained_base | 0.9189 | 0.9856 | 0.0152 | 0.0468 | 0.6770 | 0.8469 | REFERENCE BASELINE |
| lora_symmetric | best_model | 0.8646 | 0.9763 | 0.3936 | 0.5374 | 0.6488 | 0.8319 | REJECT (Native drop: -0.0543, English drop: -0.0283) |
| lora_symmetric | final_model | 0.8645 | 0.9763 | 0.3937 | 0.5403 | 0.6489 | 0.8319 | REJECT (Native drop: -0.0544, English drop: -0.0282) |

## Acceptance Criteria Reference
- Native Hindi baseline: 0.9189 (Gate: >= 0.8989)
- English SciFact baseline: 0.6770 (Gate: >= 0.6570)
- Romanized target: >= 0.4000
