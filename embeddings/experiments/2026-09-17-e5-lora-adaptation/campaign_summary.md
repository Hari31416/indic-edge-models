# LoRA Adaptation Campaign Summary

Execution duration: 26.3 minutes

| Experiment            | Checkpoint      | Native nDCG@10 | Native Recall@20 | Romanized nDCG@10 | Romanized Recall@20 | English nDCG@10 | English Recall@20 | Decision                                             |
| :-------------------- | :-------------- | :------------- | :--------------- | :---------------- | :------------------ | :-------------- | :---------------- | :--------------------------------------------------- |
| base_model            | pretrained_base | 0.9189         | 0.9856           | 0.0152            | 0.0468              | 0.6770          | 0.8469            | REFERENCE BASELINE                                   |
| lora_symmetric        | final_model     | 0.8692         | 0.9728           | 0.3414            | 0.4782              | 0.6571          | 0.8469            | REJECT (Native drop: -0.0497)                        |
| lora_query_asymmetric | final_model     | 0.8801         | 0.9778           | 0.2800            | 0.4316              | 0.6131          | 0.8029            | REJECT (Native drop: -0.0388, English drop: -0.0639) |

## Acceptance Criteria Reference
- Native Hindi baseline: 0.9189 (Gate: >= 0.8989)
- English SciFact baseline: 0.6770 (Gate: >= 0.6570)
- Romanized target: >= 0.4000
