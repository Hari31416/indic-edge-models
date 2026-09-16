# Follow-up Adaptation Campaign Summary

Execution duration: 112.9 minutes

| Experiment | Checkpoint | Native nDCG@10 | Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 | English nDCG@10 | English Recall@20 | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| conservative_control | final_model | 0.4367 | 0.6394 | 0.0636 | 0.1187 | 0.6631 | 0.8529 | REJECT (Native drop: -0.1147) |
| teacher_align | final_model | 0.5387 | 0.7672 | 0.0059 | 0.0126 | 0.6681 | 0.8433 | PROGRESS (Preserved, Roman < 0.40) |
| dual_objective | final_model | 0.4342 | 0.6286 | 0.0501 | 0.0940 | 0.6571 | 0.8602 | REJECT (Native drop: -0.1171) |

## Acceptance Criteria Reference
- Native Hindi baseline: 0.5513 (Gate: >= 0.5313)
- English SciFact baseline: 0.6770 (Gate: >= 0.6570)
- Romanized target: >= 0.4000
