# RL Parquet: Per-Sample Rescale, Map-Reduced, Source-u20, Arxiv/Map 4of5

This note documents the final `4of5` RL parquet construction and its main statistics.

Final parquet:

`/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20_arxiv_map_4of5-insight_qwen_agent.parquet`

Summary JSON:

`/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20_arxiv_map_4of5-insight_qwen_agent.summary.json`

## Construction Lineage

1. Start from the full answerable plus synthetic-unanswerable parquet with per-sample rescale metadata:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale-insight_qwen_agent.parquet`

   This contains filtered answerable RL rows plus synthetic unanswerable VR2-wrong rows. Each row carries `extra_info.initial_rescale`, `initial_rescale_source`, and `initial_rescale_dpi`.

2. Apply prompt-length filtering using each row's own `extra_info.initial_rescale`, not uniform `0.5`.

   Intermediate:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000-insight_qwen_agent.parquet`

   Settings:

   | setting | value |
   |---|---:|
   | rescale mode | `extra_info.initial_rescale` |
   | image max area | `3500 * 3500` |
   | max estimated prompt tokens | `11000` |
   | source rows | 10,360 |
   | kept rows | 8,023 |
   | dropped rows | 2,337 |

3. Build a global u25 row set from the per-sample-rescale length-filtered rows.

   Script:

   `scripts/build_per_sample_rescale_u25_rl_parquet_20260520.py`

   Intermediate:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.parquet`

   Policy:

   - Keep all answerable rows.
   - Randomly sample synthetic unanswerable rows with seed `42`.
   - Target global unanswerable fraction: `25%`.

   Result:

   | metric | value |
   |---|---:|
   | source rows | 8,023 |
   | final rows | 6,149 |
   | answerable rows | 4,612 |
   | synthetic unanswerable rows | 1,537 |
   | unanswerable fraction | 24.996% |

4. Reduce map rows.

   Script:

   `scripts/build_per_sample_rescale_u25_map_reduced_rl_parquet_20260520.py`

   Intermediate:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced-insight_qwen_agent.parquet`

   Policy:

   - Keep `50%` of all travelmap rows.
   - Keep `75%` of all metromap rows.
   - Sample with stable seed `42`.
   - Sampling is across all map subtypes and rescale values.

   Result:

   | metric | before | after |
   |---|---:|---:|
   | total rows | 6,149 | 5,454 |
   | map rows | 1,923 | 1,228 |
   | travelmap rows | 859 | 430 |
   | metromap rows | 1,064 | 798 |

5. Cap unanswerable rows to about `20%` within each main source.

   Script:

   `scripts/build_map_reduced_source_u25_rl_parquet_20260520.py`

   Intermediate:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20-insight_qwen_agent.parquet`

   Policy:

   - Keep all answerable rows.
   - For each main source, keep at most `floor(answerable_count / 4)` unanswerable rows.
   - This gives roughly 20% unanswerable within sources that originally exceeded 20%.
   - Sources already below 20%, such as map and poster, are not upsampled.

   Result:

   | metric | value |
   |---|---:|
   | source rows | 5,454 |
   | kept rows | 4,767 |
   | dropped rows | 687 |
   | unanswerable rows | 850 |
   | overall unanswerable fraction | 17.83% |

6. Further reduce arxiv and map by `1/5`.

   Script:

   `scripts/build_source_u20_arxiv_map_reduced_rl_parquet_20260520.py`

   Final parquet:

   `/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20_arxiv_map_4of5-insight_qwen_agent.parquet`

   Policy:

   - Start from the `source_u20` parquet.
   - Keep `80%` of arxiv rows.
   - Keep `80%` of map rows.
   - Keep all other sources unchanged.
   - Sample with stable seed `42`.

## Final Dataset Summary

| metric | value |
|---|---:|
| total rows | 4,150 |
| answerable rows | 3,414 |
| synthetic unanswerable rows | 736 |
| overall unanswerable fraction | 17.73% |
| rows with system prompt | 4,150 |
| rows with `extra_info.initial_rescale` | 4,150 |

Initial rescale distribution:

| initial_rescale | rows |
|---|---:|
| 0.25 | 2,722 |
| 0.35 | 967 |
| 0.5 | 461 |

## Source Distribution

| source | rows | fraction |
|---|---:|---:|
| arxiv | 1,486 | 35.81% |
| map | 982 | 23.66% |
| poster | 568 | 13.69% |
| docvqa | 533 | 12.84% |
| dude | 318 | 7.66% |
| info | 263 | 6.34% |

Unanswerable rows by source:

| source | rows | unanswerable rows |
|---|---:|---:|
| arxiv | 1,486 | 293 |
| map | 982 | 133 |
| poster | 568 | 89 |
| docvqa | 533 | 106 |
| dude | 318 | 63 |
| info | 263 | 52 |

## Comparison With Old Parquet

Old parquet:

`/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet`

| source | old rows | old % | new rows | new % | delta |
|---|---:|---:|---:|---:|---:|
| arxiv | 1,574 | 38.38% | 1,486 | 35.81% | -2.57 pp |
| map | 1,181 | 28.80% | 982 | 23.66% | -5.14 pp |
| docvqa | 470 | 11.46% | 533 | 12.84% | +1.38 pp |
| poster | 410 | 10.00% | 568 | 13.69% | +3.69 pp |
| dude | 324 | 7.90% | 318 | 7.66% | -0.24 pp |
| info | 142 | 3.46% | 263 | 6.34% | +2.88 pp |
| total | 4,101 | 100.00% | 4,150 | 100.00% | +49 rows |

Unanswerable fraction:

| parquet | unanswerable fraction |
|---|---:|
| old, broad `question_type` contains `not-answerable` | 579 / 4,101 = 14.12% |
| old, strict `question_type == not-answerable` | 564 / 4,101 = 13.75% |
| new final `4of5` | 736 / 4,150 = 17.73% |

The final new dataset is close to the old parquet in total size and broad source mix, but it still has fewer maps and more poster/info rows. It also has a higher unanswerable rate than the old parquet, though lower than the earlier global u25 variants.

## Prompt-Length Verification

The final parquet was verified directly with the same rough estimator:

- Text tokens: approximate words / `0.75`.
- Image tokens: resized page dimensions under per-sample `extra_info.initial_rescale`, capped at `3500 * 3500`, then `ceil(width / 32) * ceil(height / 32)`.

Verification result:

| metric | value |
|---|---:|
| rows checked | 4,150 |
| missing `extra_info.initial_rescale` | 0 |
| system prompt rows | 4,150 |
| rows exceeding estimated 11,000 prompt tokens | 0 |
| max estimated prompt tokens | 10,981.33 |
| max row question id | `bigpage_combine_map_metromap_moscow005` |
| max row initial rescale | 0.5 |

## Page Count Distribution

Page count is `len(images)`.

Overall:

| metric | value |
|---|---:|
| rows | 4,150 |
| mean pages | 12.14 |
| median pages | 7 |
| p75 pages | 20 |
| p90 pages | 36 |
| p95 pages | 40 |
| max pages | 40 |

Overall buckets:

| pages | rows | fraction |
|---|---:|---:|
| 1 | 1,503 | 36.22% |
| 2-3 | 166 | 4.00% |
| 4-5 | 186 | 4.48% |
| 6-10 | 696 | 16.77% |
| 11-20 | 594 | 14.31% |
| 21-30 | 392 | 9.45% |
| 31-40 | 613 | 14.77% |

By source:

| source | rows | mean | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| arxiv | 1,486 | 22.57 | 22 | 40 | 40 |
| docvqa | 533 | 13.98 | 7 | 40 | 40 |
| dude | 318 | 13.83 | 13 | 31 | 39 |
| info | 263 | 2.79 | 1 | 7 | 16 |
| map | 982 | 2.92 | 1 | 8 | 20 |
| poster | 568 | 2.41 | 1 | 6 | 19 |

By source buckets:

| source | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-30 | 31-40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| arxiv | 0 | 0 | 1 | 384 | 323 | 323 | 455 |
| docvqa | 219 | 15 | 19 | 41 | 79 | 37 | 123 |
| dude | 37 | 45 | 21 | 30 | 118 | 32 | 35 |
| info | 161 | 34 | 24 | 33 | 11 | 0 | 0 |
| map | 668 | 36 | 75 | 164 | 39 | 0 | 0 |
| poster | 418 | 36 | 46 | 44 | 24 | 0 | 0 |

## Related Scripts

- `scripts/build_per_sample_rescale_u25_rl_parquet_20260520.py`
- `scripts/build_per_sample_rescale_u25_map_reduced_rl_parquet_20260520.py`
- `scripts/build_map_reduced_source_u25_rl_parquet_20260520.py`
- `scripts/build_source_u20_arxiv_map_reduced_rl_parquet_20260520.py`
