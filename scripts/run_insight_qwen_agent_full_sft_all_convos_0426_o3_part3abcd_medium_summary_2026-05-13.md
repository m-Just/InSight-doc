# O3 Medium Part 3a/3b/3c/3d Summary

This note summarizes the newly converted O3 medium SFT parquets used by [run_insight_qwen_agent_full_sft_all_convos_0426.sh](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/run_insight_qwen_agent_full_sft_all_convos_0426.sh).

Parquets:
- [train_part3a](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3a/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet)
- [train_part3b](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3b/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet)
- [train_part3c](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3c/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet)
- [train_part3d](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3d/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet)

## Tool Calls

Tool calls were counted from `assistant.tool_calls` per row.

| split | rows | total tool calls | mean | p50 | p75 | p90 | max | zero-tool rows | 2+-tool rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `part3a` | 885 | 1387 | 1.5672 | 1 | 2 | 3 | 10 | 203 | 289 |
| `part3b` | 783 | 1250 | 1.5964 | 1 | 2 | 3 | 10 | 110 | 246 |
| `part3c` | 554 | 813 | 1.4675 | 1 | 2 | 4 | 10 | 148 | 163 |
| `part3d` | 1160 | 1901 | 1.6388 | 1 | 2 | 4 | 10 | 180 | 369 |
| `combined` | 3382 | 5351 | 1.5822 | 1 | 2 | 4 | 10 | 641 | 1067 |

Useful rates:
- zero-tool share: `part3a` 22.9%, `part3b` 14.0%, `part3c` 26.7%, `part3d` 15.5%, combined 18.9%
- `2+` tool-call share: `part3a` 32.7%, `part3b` 31.4%, `part3c` 29.4%, `part3d` 31.8%, combined 31.5%

## Estimated Tokens Per Row

Estimate definition matches the arxiv part3/4/5 note:
- text tokens ~= `word_count / 0.75`, counting message content only
- image tokens ~= `ceil(width / 32) * ceil(height / 32)` per image
- total tokens = text + image

| split | text mean | image mean | total mean | total p50 | total p75 | total p90 | total p95 | total max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `part3a` | 389.42 | 12011.40 | 12400.82 | 8546.00 | 19262.67 | 29710.13 | 36453.33 | 51761.67 |
| `part3b` | 380.48 | 6110.94 | 6491.42 | 4078.67 | 10294.00 | 16625.60 | 19686.80 | 24885.67 |
| `part3c` | 376.58 | 11475.11 | 11851.68 | 7839.50 | 18016.92 | 29804.27 | 37627.67 | 46308.67 |
| `part3d` | 381.41 | 6306.35 | 6687.76 | 4825.50 | 10315.92 | 16029.63 | 19470.48 | 27470.67 |
| `combined` | 382.50 | 8600.69 | 8983.19 | 5880.83 | 12950.92 | 21027.93 | 27948.25 | 51761.67 |

## Image Count

| split | image count mean | p50 | p75 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `part3a` | 13.03 | 9 | 20 | 33.0 | 40.0 | 46 |
| `part3b` | 13.33 | 8 | 21 | 36.8 | 41.0 | 45 |
| `part3c` | 12.88 | 8 | 20 | 34.0 | 40.0 | 45 |
| `part3d` | 13.28 | 8 | 20 | 34.0 | 41.0 | 46 |
| `combined` | 13.16 | 8 | 20 | 34.9 | 41.0 | 46 |

## Length Pressure

Rows above estimated length thresholds:

| split | rows | >32768 | >49152 | >65536 |
|---|---:|---:|---:|---:|
| `part3a` | 885 | 60 | 2 | 0 |
| `part3b` | 783 | 0 | 0 | 0 |
| `part3c` | 554 | 45 | 0 | 0 |
| `part3d` | 1160 | 0 | 0 | 0 |
| `combined` | 3382 | 105 | 2 | 0 |

Interpretation:
- `part3a` and `part3c` are the high-resolution-heavy O3 splits.
- `part3b` and `part3d` are much closer to the original arxiv `part3` length scale.
- For 48k training, only 2 rows exceed the estimate, both in `part3a`. For 32k training, the pressure is meaningful: 105 rows exceed the estimate across `part3a` and `part3c`.
