# Multiscale RL Parquet Construction

Date: 2026-05-20

Final parquet:

`/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.parquet`

Final rows: `3,983`

| category | rows |
|---|---:|
| answerable | 2,987 |
| synthetic unanswerable VR2-wrong | 996 |
| total | 3,983 |

Unanswerable fraction: `996 / 3983 = 25.01%`

Final rescale distribution:

| initial_rescale | rows |
|---:|---:|
| 0.25 | 2,182 |
| 0.35 | 1,080 |
| 0.5 | 721 |

The final parquet keeps the InSight tool-use system prompt on every row and keeps `extra_info.initial_rescale`, `extra_info.initial_rescale_source`, and `extra_info.initial_rescale_dpi`.

## Step 1: Map Answerable RL IDs

Inputs:

- `/scratch/ywxzml3j/likaican/src/InSight-doc/data/answerable_RL_afterfiltering0518.txt`
- `/scratch/ywxzml3j/likaican/src/InSight-doc/data/answerable_RL_afterfiltering0519.txt`

Script:

- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/map_answerable_rl_filtered_ids_to_source_manifests_20260519.py`

Outputs:

- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/answerable_rl_afterfiltering_manifest_mapping_20260519/answerable_RL_afterfiltering0518_mapped.jsonl`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/answerable_rl_afterfiltering_manifest_mapping_20260519/answerable_RL_afterfiltering0519_mapped.jsonl`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/answerable_rl_afterfiltering_manifest_mapping_20260519/summary.json`

Counts:

| split | mapped rows | unmapped rows |
|---|---:|---:|
| 0518 | 2,979 | 0 |
| 0519 | 3,033 | 0 |

Cross-batch qid overlap between 0518 and 0519: `1,123`.

These overlaps were kept in the later mixed parquet because different source/rescale variants can be distinct training samples.

## Step 2: Assign Answerable Rescales

Answerable per-sample `initial_rescale` assignment:

| source group | initial_rescale |
|---|---:|
| 0518 rows | 0.25 |
| 0519 `arxiv/train_part4` | 0.35 |
| 0519 `O3_data_0424/train_part3b` | 0.35 |
| 0519 `O3_data_0424/train_part3d` | 0.35 |
| 0519 `arxiv/train_part5` | 0.5 |
| 0519 `O3_data_0424/train_part3a` | 0.5 |
| 0519 `O3_data_0424/train_part3c` | 0.5 |

## Step 3: Gather Synthetic Unanswerable VR2-Wrong Rows

Script:

- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/gather_synthetic_unanswerable_vr2_wrong_for_rl_20260519.py`

Source criterion:

- Use final `wrong_question_ids.txt` from the synthetic unanswerable VR2 pipeline.

Outputs:

- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519/rescale025_50dpi/`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519/rescale035_70dpi/`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519/rescale05_100dpi/`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519/combined/mapped.jsonl`
- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519/summary.json`

Counts:

| split | dpi | rows |
|---|---:|---:|
| rescale025 | 50 | 2,535 |
| rescale035 | 70 | 1,087 |
| rescale05 | 100 | 726 |
| total | mixed | 4,348 |

## Step 4: Build Full Mixed Per-Sample-Rescale Parquet

Script:

- `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/build_answerable_plus_synthetic_unanswerable_rl_parquet_20260519.py`

Output:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale-insight_qwen_agent.parquet`

Sidecar:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale/summary.json`

Counts:

| category | rows |
|---|---:|
| answerable | 6,012 |
| synthetic unanswerable VR2-wrong | 4,348 |
| total | 10,360 |

Rescale counts:

| initial_rescale | rows |
|---:|---:|
| 0.25 | 5,514 |
| 0.35 | 2,966 |
| 0.5 | 1,880 |

## Step 5: Build Uniform-0.5 Length-Filtered Baseline Row Set

The full mixed parquet was filtered using an estimated prompt-token count under:

- uniform `initial_rescale=0.5`
- `gpt_image_max_area=3500*3500`
- max prompt tokens `<= 11000`

The intermediate no-override parquet removed `initial_rescale`, `initial_rescale_source`, and `initial_rescale_dpi` from `extra_info`.

Output:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale-insight_qwen_agent.parquet`

Sidecars:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale-insight_qwen_agent.summary.json`
- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale-insight_qwen_agent.dropped_question_ids.txt`

Counts:

| stage | rows |
|---|---:|
| source | 10,360 |
| kept | 4,527 |
| dropped | 5,833 |

## Step 6: Downsample Unanswerable Rows to U25

The u25 row set was built from the uniform-0.5 length-filtered row set.

Selection:

- Keep all answerable rows.
- Randomly sample unanswerable rows with seed `42`.
- Keep `996` unanswerable rows so that `996 / (2987 + 996) ~= 25%`.

Output:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale_u25-insight_qwen_agent.parquet`

Sidecars:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale_u25-insight_qwen_agent.summary.json`
- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_no_initial_rescale_u25-insight_qwen_agent.dropped_unanswerable_question_ids.txt`

Counts:

| category | rows |
|---|---:|
| answerable | 2,987 |
| synthetic unanswerable VR2-wrong | 996 |
| total | 3,983 |

## Step 7: Restore Per-Sample Rescale for Final Multiscale Parquet

The final multiscale parquet uses the exact same `3,983` rows as the u25 row set, but restores:

- `extra_info.initial_rescale`
- `extra_info.initial_rescale_source`
- `extra_info.initial_rescale_dpi`

Restoration key:

- `(extra_info.index, extra_info.question_id)`

Output:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.parquet`

Sidecar:

- `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.summary.json`

Final counts:

| category | rows |
|---|---:|
| answerable | 2,987 |
| synthetic unanswerable VR2-wrong | 996 |
| total | 3,983 |

Final rescale counts:

| initial_rescale | rows |
|---:|---:|
| 0.25 | 2,182 |
| 0.35 | 1,080 |
| 0.5 | 721 |

## Caveat

The final row set was selected using the conservative uniform `0.5` prompt-length filter, then per-sample rescale overrides were restored.

This means the final multiscale parquet is row-matched to the no-tool uniform-0.5 baseline row set, but it may exclude some rows that would have fit under true per-sample `0.25/0.35` rescale.
