# RL Parquet Prompt Token Estimate

Date: 2026-05-19

This note estimates how many rows exceed 11,000 input prompt tokens for two RL parquets under different initial image rescale settings.

## Assumptions

- Image token estimate: one token per `32 x 32` pixels after initial rescale and `gpt_image_max_area` capping.
- Text token estimate: one token per `0.75` words, including the system prompt already present in the parquet prompt.
- Resized image dimensions are rounded to integer pixels, then area-capped, then tokenized as `ceil(width / 32) * ceil(height / 32)`.
- For `per-sample rescale`, rows use `extra_info.initial_rescale` when present.
- For `insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet`, `extra_info.initial_rescale` is absent for all rows, so the per-sample case uses legacy fallback `0.25`.

## Parquets

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale-insight_qwen_agent.parquet`
- `/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet`

The new mixed answerable/unanswerable parquet has per-sample rescale distribution:

| initial_rescale | rows |
|---:|---:|
| 0.25 | 5,514 |
| 0.35 | 2,966 |
| 0.5 | 1,880 |

## `gpt_image_max_area = 1280 * 1280`

| Parquet | Rows | Rescale setting | Rows > 11k tokens | Percent |
|---|---:|---|---:|---:|
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | uniform `0.7` | 6,789 | 65.5% |
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | uniform `0.5` | 5,640 | 54.4% |
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | `extra_info.initial_rescale` | 2,286 | 22.1% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | uniform `0.7` | 2,286 | 55.7% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | uniform `0.5` | 1,898 | 46.3% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | fallback `0.25` | 0 | 0.0% |

## `gpt_image_max_area = 3500 * 3500`

| Parquet | Rows | Rescale setting | Rows > 11k tokens | Percent |
|---|---:|---|---:|---:|
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | uniform `0.7` | 7,408 | 71.5% |
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | uniform `0.5` | 5,833 | 56.3% |
| `answerable_plus_synthetic...per_sample_rescale` | 10,360 | `extra_info.initial_rescale` | 2,337 | 22.6% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | uniform `0.7` | 2,505 | 61.1% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | uniform `0.5` | 1,965 | 47.9% |
| `insight_doc_rl_balanced_dude_reduced_merged_u25` | 4,101 | fallback `0.25` | 0 | 0.0% |

## Takeaways

- The per-sample rescale metadata substantially reduces estimated over-11k rows in the new mixed parquet: about `22%` vs `54-72%` for uniform `0.5/0.7`.
- Increasing `gpt_image_max_area` from `1280 * 1280` to `3500 * 3500` mostly affects uniform `0.7`; it has limited effect on per-sample rescale because most rows are `0.25` or `0.35`.
- The legacy `merged_u25` parquet remains below the 11k threshold under fallback `0.25`; its maximum estimate was about `10,992` tokens in both area-cap settings.
