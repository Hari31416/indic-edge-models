# LoRA Training Summary Report

| Parameter / Metric       | Value                          |
| ------------------------ | ------------------------------ |
| Base Model               | intfloat/multilingual-e5-small |
| Architecture Mode        | lora_symmetric                 |
| LoRA Rank (r)            | 16                             |
| LoRA Alpha (alpha)       | 32                             |
| LoRA Dropout             | 0.05                           |
| Target Modules           | query, key, value, dense       |
| Trainable Parameters     | 1,339,392 (1.13%)              |
| Total Parameters         | 118,993,152                    |
| Training Examples        | 98825                          |
| Validation Examples      | 5227                           |
| Epochs                   | 3                              |
| Per-Device Batch Size    | 32                             |
| Gradient Accumulation    | 4                              |
| Effective Batch Size     | 128                            |
| Learning Rate            | 1.00e-04                       |
| Total Optimization Steps | 2316                           |
| Training Duration (s)    | 3011.03                        |
| Throughput (examples/s)  | 98.5                           |
| Best Validation Loss     | 0.2796                         |
