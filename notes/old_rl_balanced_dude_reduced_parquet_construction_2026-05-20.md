# Old RL Parquet Construction: `insight_doc_rl_balanced_dude_reduced_merged_u25`

This note reconstructs the old RL parquet construction from the surviving scripts,
manifest metadata, and parquet summaries.

Final old parquet:

`/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet`

No-tool/no-system derivative:

`/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent_no_tool_no_system.parquet`

## Caveat

The later merge, u25 reduction, parquet conversion, and prompt-length filtering steps
are recoverable from scripts and metadata. The raw medium-only wrong-question
manifest extraction is documented by its `summary.json`, per-group `meta.json`, and
the generated manifests. The exact launcher for that first extraction step was not
located, so the step is still reconstructed from artifacts rather than a fully
captured end-to-end command.

## Construction Steps

1. Start from medium-only wrong-question manifests:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only`

   This directory contains eight grouped `manifest.jsonl` files, one `meta.json`
   per group, and a top-level `summary.json`.

   Raw extraction metadata:

   | item | value |
   |---|---:|
   | difficulty | `medium` |
   | processed wrong-ID files | 9 |
   | input wrong question IDs | 10,977 |
   | output manifest rows | 10,977 |
   | missing question IDs | 0 |
   | grouped manifests | 8 |
   | unique question IDs | 10,960 |
   | duplicate extra rows | 17 |
   | unique documents | 7,643 |
   | raw unanswerable rows | 4,446 |
   | raw unanswerable fraction | 40.5% |

   The source extraction excludes `val_sample_102` and is built from the medium
   `processed/wrong_question_ids.txt` files:

   | source processed path | contributes to group |
   |---|---|
   | `O3_data_0424/dude_poster_unanswerable/medium/processed/wrong_question_ids.txt` | `O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40` |
   | `O3_data_0424/train_part1/medium/processed/wrong_question_ids.txt` | `O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40` |
   | `O3_data_0424/train_part2a/medium/processed/wrong_question_ids.txt` | `O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40` |
   | `O3_data_0424/train_part2b/medium/processed/wrong_question_ids.txt` | `O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40` |
   | `O3_data_0424/train_part2c/medium/processed/wrong_question_ids.txt` | `O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40` |
   | `arxiv/spanning_train_part1/medium/processed/wrong_question_ids.txt` | `arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40` |
   | `arxiv/train_part1/medium/processed/wrong_question_ids.txt` | `arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40` |
   | `arxiv/train_part2/medium/processed/wrong_question_ids.txt` | `arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40` |
   | `arxiv/train_part3/medium/processed/wrong_question_ids.txt` | `arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0` |

   Raw grouped manifest counts:

   | group | rows | unanswerable rows | unique docs | mean pages | p90 pages | max pages |
   |---|---:|---:|---:|---:|---:|---:|
   | `O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40` | 1,578 | 499 | 1,311 | 12.54 | 33.0 | 40 |
   | `O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40` | 1,536 | 531 | 1,330 | 12.87 | 33.5 | 40 |
   | `O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40` | 1,171 | 400 | 1,001 | 13.65 | 34.0 | 40 |
   | `O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40` | 871 | 295 | 758 | 12.98 | 34.0 | 40 |
   | `O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40` | 1,214 | 1,214 | 811 | 9.08 | 24.0 | 39 |
   | `arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40` | 122 | 0 | 76 | 27.84 | 40.0 | 40 |
   | `arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0` | 2,728 | 1,507 | 2,173 | 26.02 | 39.0 | 40 |
   | `arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40` | 1,757 | 0 | 1,054 | 26.53 | 40.0 | 40 |

   Raw subset counts before balancing:

   | subset | rows | unanswerable rows | unique docs | mean pages | median pages | p90 pages | max pages |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `bigpage_info` | 147 | 0 | 144 | 3.78 | 1 | 9.4 | 20 |
   | `bigpage_map` | 1,183 | 0 | 488 | 3.28 | 1 | 8.0 | 20 |
   | `bigpage_poster` | 781 | 467 | 195 | 1.93 | 1 | 4.0 | 20 |
   | `docvqa` | 485 | 0 | 442 | 18.87 | 16 | 40.0 | 40 |
   | `dude` | 3,774 | 2,472 | 3,116 | 16.64 | 20 | 34.0 | 40 |
   | `mveqa` | 2,195 | 651 | 1,696 | 26.38 | 28 | 40.0 | 40 |
   | `veqa` | 2,412 | 856 | 2,062 | 26.15 | 28 | 40.0 | 40 |

   Raw page-count distribution:

   | subset | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-30 | 31-40 |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `bigpage_info` | 82 | 14 | 16 | 21 | 14 | 0 | 0 |
   | `bigpage_map` | 800 | 46 | 73 | 174 | 90 | 0 | 0 |
   | `bigpage_poster` | 685 | 15 | 18 | 34 | 29 | 0 | 0 |
   | `docvqa` | 149 | 8 | 13 | 31 | 71 | 47 | 166 |
   | `dude` | 312 | 520 | 249 | 366 | 900 | 791 | 636 |
   | `mveqa` | 0 | 0 | 19 | 285 | 385 | 607 | 899 |
   | `veqa` | 0 | 0 | 35 | 352 | 389 | 650 | 986 |

2. Balance/reduce the source manifests into:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced`

   The surviving summary records `5,446` selected rows.

   Subset counts:

   | subset | rows |
   |---|---:|
   | `bigpage_info` | 147 |
   | `bigpage_map` | 1,183 |
   | `bigpage_poster` | 781 |
   | `docvqa` | 485 |
   | `dude` | 486 |
   | `mveqa` | 1,182 |
   | `veqa` | 1,182 |

   Unanswerable rows after this balancing step:

   | subset | unanswerable rows |
   |---|---:|
   | `bigpage_poster` | 467 |
   | `dude` | 243 |
   | `mveqa` | 591 |
   | `veqa` | 591 |

   Retention from the raw medium-only manifests to this balanced stage:

   | subset | raw rows | selected rows | row retention | raw unanswerable | selected unanswerable | unanswerable retention |
   |---|---:|---:|---:|---:|---:|---:|
   | `bigpage_info` | 147 | 147 | 100.0% | 0 | 0 | n/a |
   | `bigpage_map` | 1,183 | 1,183 | 100.0% | 0 | 0 | n/a |
   | `bigpage_poster` | 781 | 781 | 100.0% | 467 | 467 | 100.0% |
   | `docvqa` | 485 | 485 | 100.0% | 0 | 0 | n/a |
   | `dude` | 3,774 | 486 | 12.9% | 2,472 | 243 | 9.8% |
   | `mveqa` | 2,195 | 1,182 | 53.8% | 651 | 591 | 90.8% |
   | `veqa` | 2,412 | 1,182 | 49.0% | 856 | 591 | 69.0% |

   The protected subsets were `bigpage_info`, `bigpage_map`, `bigpage_poster`,
   and `docvqa`, which were kept completely at this stage. The balanced-half
   subsets were `dude`, `mveqa`, and `veqa`. The non-protected cap was `1,182`
   rows, with a special `dude` override to `486` rows.

3. Merge grouped manifests for `create_parquet_dataset.py` using:

   `scripts/merge_rl_manifests_for_create_parquet.py`

   Output:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced_merged_for_parquet`

   This produced one `manifest.jsonl`, rewrote image paths relative to the common
   root, and created a `pdf_image` symlink to:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc`

   Group counts in the merged manifest:

   | group | rows |
   |---|---:|
   | `O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40` | 799 |
   | `O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40` | 771 |
   | `O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40` | 540 |
   | `O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40` | 430 |
   | `O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40` | 542 |
   | `arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40` | 46 |
   | `arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0` | 1,641 |
   | `arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40` | 677 |

4. Convert the merged manifest into the full merged parquet:

   `/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent.parquet`

   Rows: `5,446`.

5. Apply the old `u25` reduction with:

   `scripts/reduce_unanswerable_ratio_in_merged_rl_manifest.py`

   Input:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced_merged_for_parquet`

   Output:

   `/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced_merged_for_parquet_u25`

   Policy:

   - Keep all answerable rows.
   - For each subset, keep at most `floor(answerable_count / 3)` unanswerable rows.
   - Select unanswerable rows by deterministic hash tie-break with seed `0`.
   - Detect unanswerable rows by checking whether `question_type` contains `not-answerable`.

   Rows after u25 reduction: `4,133`.

   Selected subset counts:

   | subset | rows |
   |---|---:|
   | `bigpage_info` | 147 |
   | `bigpage_map` | 1,183 |
   | `bigpage_poster` | 418 |
   | `docvqa` | 485 |
   | `dude` | 324 |
   | `mveqa` | 788 |
   | `veqa` | 788 |

   Selected unanswerable counts:

   | subset | unanswerable rows |
   |---|---:|
   | `bigpage_poster` | 104 |
   | `dude` | 81 |
   | `mveqa` | 197 |
   | `veqa` | 197 |

   Total selected unanswerable rows before prompt-length filtering: `579`.

6. Convert the u25 manifest to the final train parquet and apply estimated
   prompt-length filtering.

   Final parquet:

   `/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet`

   Summary:

   `/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.estprompt_le11000.summary.json`

   Filtering settings:

   - Threshold: `11000` estimated prompt tokens.
   - Image-token estimate: approximately one token per `32x32` pixels after resize.
   - Historical context note: `notes/rl_balanced_dude_reduced_prompt_length_note_2026-05-04.md`.

   Prompt-length filtering result:

   | metric | value |
   |---|---:|
   | rows before filtering | 4,133 |
   | rows removed | 32 |
   | rows kept | 4,101 |

   Largest removed row:

   | question_id | document_id | images | estimated tokens |
   |---|---|---:|---:|
   | `docvqa_merged_twopass_38688_39907` | `docvqa/qsnc0227.pdf` | 40 | 13,482 |

## Final Old Parquet Properties

Final row count: `4,101`.

Final subset counts:

| subset | rows |
|---|---:|
| `bigpage_map` | 1,181 |
| `mveqa` | 787 |
| `veqa` | 787 |
| `docvqa` | 470 |
| `bigpage_poster` | 410 |
| `dude` | 324 |
| `bigpage_info` | 142 |

The final parquet keeps the system prompt and has no per-sample
`extra_info.initial_rescale` override. In later prompt-token estimates, rows without
`initial_rescale` were treated as legacy fallback `0.25`.

Although the name contains `u25`, the final parquet is not globally 25%
unanswerable. The u25 policy was applied per subset before the final prompt-length
filter and only to subsets that had unanswerable rows. The final row-level
unanswerable fraction is about `14%`.

## No-Tool/No-System Derivative

The no-tool/no-system derivative was created later from the final old u25 parquet:

`/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent_no_tool_no_system.parquet`

This keeps the same final row set but removes system messages for no-tool/no-system
training/evaluation.
