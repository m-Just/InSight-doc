# Medium-Only SFT Data Comparison

Date: 2026-06-01

This note compares only the medium SFT data used by:

| run | checkpoint / experiment | train log |
|---|---|---|
| old | `freeze_vt_bs32_epoch2` | `/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt/train.log` |
| new | `full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519` | `/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch1_fp32_scratch/train.log` |

The old run contains both easy and medium data. This comparison intentionally ignores the easy rows and compares only the old medium data against the new medium-only data.

## Training Setup Context

| item | old | new |
|---|---:|---:|
| total train rows, all data | 16,855 | 17,913 |
| medium train rows considered here | 8,469 | 17,913 |
| easy train rows excluded here | 8,386 | 0 |
| train files, all data | 18 | 18 |
| max length | 32,768 | 65,536 |
| sequence parallel size | 1 | 4 |
| train batch size | 32 | 32 |
| epochs | 3 | 1 |
| freeze vision tower | true | true |
| engine dtype | `bfloat16` | `bfloat16` |
| model dtype | `fp32` | `fp32` |

## High-Level Medium Composition

| bucket | old rows | new rows | meaning |
|---|---:|---:|---|
| common old sources | 8,469 | 8,469 | Same medium source rows as old, but new uses `sft_data_base_model_tool_argument_order.parquet`. |
| O3 part3a-d additions | 0 | 3,381 | New O3 medium data with `processed_gpt5_nano_rewrite_aspect_drop`. |
| arxiv part4-5 additions | 0 | 2,867 | New arxiv medium data. |
| synthetic unanswerable additions | 0 | 3,196 | Synthetic unanswerable medium data at rescale 0.25, 0.35, and 0.5. |
| total medium | 8,469 | 17,913 | New medium data is old common medium plus 9,444 added medium rows. |

New medium data composition by fraction:

| bucket | rows | fraction of new medium |
|---|---:|---:|
| common old sources | 8,469 | 47.3% |
| O3 part3a-d additions | 3,381 | 18.9% |
| arxiv part4-5 additions | 2,867 | 16.0% |
| synthetic unanswerable additions | 3,196 | 17.8% |

## Medium-Only Aggregate Statistics

`text-token proxy` is a rough text-only estimate computed as word count divided by 0.75. It does not include image tokens.

| bucket | rows | rows with tool calls | avg tool calls | p50 tool calls | p90 tool calls | max tool calls | avg messages | avg images | p50 images | p90 images | max images | avg text-token proxy | avg assistant-token proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old medium all | 8,469 | 94.2% | 1.773 | 1 | 3 | 10 | 6.546 | 18.798 | 15 | 41 | 50 | 421.350 | 141.363 |
| new common medium | 8,469 | 94.2% | 1.773 | 1 | 3 | 10 | 6.546 | 18.798 | 15 | 41 | 50 | 421.350 | 141.363 |
| new O3 part3a-d | 3,381 | 81.0% | 1.581 | 1 | 4 | 10 | 6.162 | 13.156 | 8 | 35 | 46 | 375.540 | 118.332 |
| new arxiv part4-5 | 2,867 | 87.9% | 1.270 | 1 | 2 | 10 | 5.541 | 26.722 | 28 | 41 | 50 | 463.589 | 145.489 |
| new synthetic unanswerable | 3,196 | 85.4% | 1.513 | 1 | 3 | 10 | 6.025 | 20.814 | 19 | 41 | 48 | 468.957 | 178.380 |
| new medium all | 17,913 | 89.2% | 1.610 | 1 | 3 | 10 | 6.220 | 19.361 | 16 | 41 | 50 | 427.958 | 144.281 |

Interpretation:

The overlapping old/new medium subset is identical in these aggregate statistics. The new additions are slightly less tool-call dense than the old common medium subset, so the full new medium set has lower average tool calls per row than old medium, despite having much more total medium/tool-use supervision.

## Tool-Call Count Distribution

| bucket | 0 calls | 1 call | 2 calls | 3 calls | 4 calls | 5 calls | 6 calls | 7 calls | 8 calls | 9 calls | 10 calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old medium all | 487 | 4,883 | 1,697 | 571 | 286 | 172 | 92 | 70 | 49 | 32 | 130 |
| new common medium | 487 | 4,883 | 1,697 | 571 | 286 | 172 | 92 | 70 | 49 | 32 | 130 |
| new O3 part3a-d | 641 | 1,674 | 509 | 215 | 120 | 70 | 48 | 29 | 23 | 16 | 36 |
| new arxiv part4-5 | 347 | 1,736 | 590 | 116 | 48 | 14 | 6 | 5 | 2 | 1 | 2 |
| new synthetic unanswerable | 467 | 1,705 | 571 | 202 | 100 | 65 | 28 | 21 | 10 | 9 | 18 |
| new medium all | 1,942 | 9,998 | 3,367 | 1,104 | 554 | 321 | 174 | 125 | 84 | 58 | 186 |

All tool calls are `image_zoom_in_tool` in both old and new medium data.

| bucket | `image_zoom_in_tool` calls |
|---|---:|
| old medium all | 15,016 |
| new common medium | 15,016 |
| new O3 part3a-d | 5,346 |
| new arxiv part4-5 | 3,642 |
| new synthetic unanswerable | 4,834 |
| new medium all | 28,838 |

## Per-File Medium Statistics

| run | domain | source part | rows | tool-row frac | avg tool calls | avg images | avg text-token proxy |
|---|---|---|---:|---:|---:|---:|---:|
| old | O3 | `O3_data_0424/train_part1` | 1,553 | 93.1% | 1.739 | 12.897 | 374.298 |
| old | O3 | `O3_data_0424/train_part2a` | 1,339 | 90.6% | 1.435 | 12.280 | 357.280 |
| old | O3 | `O3_data_0424/train_part2b` | 1,102 | 91.9% | 1.725 | 13.857 | 377.443 |
| old | O3 | `O3_data_0424/train_part2c` | 857 | 92.2% | 1.797 | 13.466 | 375.984 |
| old | O3 | `O3_data_0424/dude_poster_unanswerable` | 135 | 94.1% | 6.163 | 14.370 | 582.854 |
| old | arxiv | `arxiv/train_part1` | 847 | 96.7% | 1.575 | 27.705 | 479.640 |
| old | arxiv | `arxiv/train_part2` | 1,217 | 97.9% | 1.706 | 27.073 | 482.220 |
| old | arxiv | `arxiv/train_part3` | 1,348 | 97.4% | 1.814 | 26.143 | 484.608 |
| old | arxiv | `arxiv/spanning_train_part1` | 71 | 98.6% | 3.732 | 32.704 | 641.108 |
| new | O3 | `O3_data_0424/train_part1` | 1,553 | 93.1% | 1.739 | 12.897 | 374.298 |
| new | O3 | `O3_data_0424/train_part2a` | 1,339 | 90.6% | 1.435 | 12.280 | 357.280 |
| new | O3 | `O3_data_0424/train_part2b` | 1,102 | 91.9% | 1.725 | 13.857 | 377.443 |
| new | O3 | `O3_data_0424/train_part2c` | 857 | 92.2% | 1.797 | 13.466 | 375.984 |
| new | O3 | `O3_data_0424/dude_poster_unanswerable` | 135 | 94.1% | 6.163 | 14.370 | 582.854 |
| new | arxiv | `arxiv/train_part1` | 847 | 96.7% | 1.575 | 27.705 | 479.640 |
| new | arxiv | `arxiv/train_part2` | 1,217 | 97.9% | 1.706 | 27.073 | 482.220 |
| new | arxiv | `arxiv/train_part3` | 1,348 | 97.4% | 1.814 | 26.143 | 484.608 |
| new | arxiv | `arxiv/spanning_train_part1` | 71 | 98.6% | 3.732 | 32.704 | 641.108 |
| new | arxiv | `arxiv/train_part4` | 1,986 | 90.7% | 1.332 | 26.585 | 465.390 |
| new | arxiv | `arxiv/train_part5` | 881 | 81.6% | 1.131 | 27.031 | 459.529 |
| new | O3 | `O3_data_0424/train_part3a` | 885 | 77.1% | 1.567 | 13.029 | 382.573 |
| new | O3 | `O3_data_0424/train_part3b` | 783 | 86.0% | 1.596 | 13.327 | 373.556 |
| new | O3 | `O3_data_0424/train_part3c` | 554 | 73.3% | 1.468 | 12.881 | 369.995 |
| new | O3 | `O3_data_0424/train_part3d` | 1,159 | 84.5% | 1.636 | 13.268 | 374.159 |
| new | synthetic unans | `first_batch_sft_parquets_20260517/rescale025` | 1,582 | 91.6% | 1.760 | 20.339 | 474.751 |
| new | synthetic unans | `first_batch_sft_parquets_20260517/rescale035` | 985 | 81.9% | 1.366 | 21.260 | 463.991 |
| new | synthetic unans | `first_batch_sft_parquets_20260517/rescale05` | 629 | 75.2% | 1.119 | 21.310 | 462.160 |

## Row-Level Check On Common Medium Data

For the 8,469 common medium rows, I compared old files against new `sft_data_base_model_tool_argument_order.parquet` files by matching source part and row index.

| check | matched rows |
|---|---:|
| row count equal | 8,469 / 8,469 |
| images equal | 8,469 / 8,469 |
| message role sequence equal | 8,469 / 8,469 |
| non-tool-call message content equal | 8,469 / 8,469 |
| tool-call semantics equal after canonical JSON parsing | 8,469 / 8,469 |
| raw tool-call argument strings exactly equal | 487 / 8,469 |
| raw tool-call argument strings changed but semantics equal | 7,982 / 8,469 |
| canonical row representation equal ignoring tool-argument key order | 8,469 / 8,469 |

The 487 rows with exact raw tool-call argument strings unchanged are the 487 no-tool-call medium rows. For rows with tool calls, the raw strings changed because the new files rewrite the argument order, but canonical JSON semantics are unchanged.

## Conclusion

The common old/new medium subset is semantically identical. The new common medium files differ from old medium only by the base-model tool-argument-order rewrite.

The substantive medium-only difference is the added 9,444 rows in the new run: 3,381 O3 part3a-d rows, 2,867 arxiv part4-5 rows, and 3,196 synthetic unanswerable rows.

The added medium data is slightly less tool-call dense than the old/common medium subset. Old/common medium has 94.2% rows with tool calls and 1.77 average tool calls per row, while the full new medium set has 89.2% rows with tool calls and 1.61 average tool calls per row. However, the new run has far more total medium/tool-use supervision: 17,913 medium rows and 28,838 total `image_zoom_in_tool` calls, compared with 8,469 medium rows and 15,016 tool calls in the old medium subset.
