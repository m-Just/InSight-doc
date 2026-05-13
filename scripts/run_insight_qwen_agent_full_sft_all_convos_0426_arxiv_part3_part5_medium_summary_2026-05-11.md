# Arxiv Medium Part 3/4/5 Summary

This note summarizes the newly converted arxiv medium SFT parquets used by [run_insight_qwen_agent_full_sft_all_convos_0426.sh](/scratch/ywxzml3j/likaican/src/InSight-doc/verl/scripts/run_insight_qwen_agent_full_sft_all_convos_0426.sh).

Parquets:
- [train_part3](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part3/medium/processed_gpt5_nano_rewrite/sft_data.parquet)
- [train_part4](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part4/medium/processed_gpt5_nano_rewrite/sft_data.parquet)
- [train_part5](/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part5/medium/processed_gpt5_nano_rewrite/sft_data.parquet)

## Tool Calls

Tool calls were counted from `assistant.tool_calls` per row.

| split | rows | total tool calls | mean | p50 | p75 | p90 | max | zero-tool rows | 2+-tool rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `part3` | 1348 | 2445 | 1.8138 | 1 | 2 | 3 | 10 | 35 | 548 |
| `part4` | 1986 | 2646 | 1.3323 | 1 | 2 | 2 | 10 | 185 | 580 |
| `part5` | 881 | 996 | 1.1305 | 1 | 1 | 2 | 7 | 162 | 204 |

A few useful rates:
- zero-tool share:
  - `part3`: `2.6%`
  - `part4`: `9.3%`
  - `part5`: `18.4%`
- `2+` tool-call share:
  - `part3`: `40.7%`
  - `part4`: `29.2%`
  - `part5`: `23.2%`

So tool-use density decreases monotonically from `part3` to `part5`.

## Estimated Tokens Per Row

Estimate definition:
- text tokens ≈ `word_count / 0.75`
- image tokens ≈ `ceil(width / 32) * ceil(height / 32)` per image
- total tokens = text + image

| split | text mean | image mean | total mean | total p50 | total p75 | total p90 | total p95 | total max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `part3` | 492.11 | 6344.12 | 6836.23 | 7161.50 | 9606.92 | 10718.93 | 10929.67 | 12723.00 |
| `part4` | 471.61 | 12417.30 | 12888.91 | 13628.83 | 18597.33 | 20039.17 | 20461.67 | 23519.33 |
| `part5` | 465.21 | 25052.33 | 25517.54 | 27489.33 | 36304.67 | 38998.67 | 39434.33 | 44568.67 |

Interpretation:
- text-token load is nearly flat across the three parts
- the growth is almost entirely from image tokens
- image-token mean roughly doubles each step:
  - `part3 -> part4`: `6344 -> 12417`
  - `part4 -> part5`: `12417 -> 25052`
- total-token mean grows accordingly:
  - `part3`: `6.8k`
  - `part4`: `12.9k`
  - `part5`: `25.5k`

Practical implication:
- `part5` medium is substantially heavier than `part3`/`part4`
- the estimated max row in `part5` is `~44.6k` tokens, so row-level length pressure is real there
