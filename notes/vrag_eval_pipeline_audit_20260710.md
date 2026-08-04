# VRAG Eval Pipeline Audit

Date: 2026-07-10

This audits the VRAG-RL evaluation process for `mmlongbench` and `longdocurl`: data/image preparation, index construction, scoped retrieval, model serving, multi-turn querying, answer extraction, scoring, and sampled failure modes.

## Bottom Line

I do not see evidence of a gross plumbing failure in the completed full run. The strongest checks passed:

- Source eval manifest: 3296 rows, 83837 unique page images, no missing paths, no duplicated row image references.
- Corpus manifest exactly equals the union of row image references.
- Index metadata: 83837 FAISS vectors and 83837 file metadata rows across 9 shards.
- Search API loaded all 83837 vectors.
- Full rollout output: 3296 unique `(dataset, question_id)` rows.
- Retrieval scoping: 14647 returned retrieval paths checked, 0 outside the current row's allowed image list.
- Sample scoring: 400/400 score records completed without compute-score API failures.

Main residual risks are quality/configuration risks, not data leakage or missing-output risks:

- Generation images are resized to about 0.4 MP, which can be too low for dense PDF tables.
- `search_top_k=3` but only one new retrieved image is fed per search turn, due `repeat_limit=1`.
- 565 search turns returned only already-seen candidates, so no new image was added.
- `max_steps=10` returns an empty prediction without a forced final-answer turn.
- The legacy scorer undercounts some complete final answers due answer extraction collapse.
- The preserved `shard00_existing` was built by the earlier single-GPU index run and later shards by the sharded run. Counts and model path match, but logs cannot prove there was no environment drift between those runs.

## Artifacts

- Full rollout: `external/VRAG/workspace/insight_doc_eval/runs/vrag_rl_full_index_allowed_eval_full_merged.jsonl`
- Legacy scored sample: `external/VRAG/workspace/insight_doc_eval/scores/legacy_sample200_per_benchmark_seed20260710_w32_b100/samples.jsonl`
- Diagnostic `single_call_v1` scored sample: `external/VRAG/workspace/insight_doc_eval/scores/single_call_v1_diagnostic_sample200_per_benchmark_seed20260710_w32_b100/samples.jsonl`
- Failure analysis: `external/VRAG/workspace/insight_doc_eval/scores/legacy_sample200_per_benchmark_seed20260710_w32_b100/failure_analysis.md`
- Existing progress memo: `notes/retrieval_pilot_memo_20260710.md`

## Data And Images

Source parquets:

- `notes/generated/testcase_0504_full_parquets/mmlongbench_full-insight_qwen_agent.parquet`
- `notes/generated/testcase_0504_full_parquets/longdocurl_full-insight_qwen_agent.parquet`

The prep script did not render PDFs itself. It read page image paths already present in the parquet `images` column and converted `file://` URIs to absolute paths.

Counts:

| Dataset | Rows | Page Refs | Mean Refs/Row | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mmlongbench` | 1089 | 32506 | 29.85 | 28 | 40 | 40 |
| `longdocurl` | 2207 | 51331 | 23.26 | 30 | 30 | 30 |
| total | 3296 | 83837 | 25.44 | - | - | - |

Checks:

- Missing image paths: 0.
- Rows with duplicate image paths: 0.
- Unique image refs: 83837.
- `corpus_full.jsonl` exactly matches the unique image refs.

Potential issue:

- The benchmark input is whatever is in those parquets. `mmlongbench` is capped at 40 page images per row and `longdocurl` at 30. If the intended benchmark needs pages beyond that source cap, the evaluator never had access to them.

## Retrieval Index

Retriever:

- Model path: `external/VRAG/hf_models/Qwen/Qwen3-VL-Embedding-2B`
- Model recorded in index metadata: `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/external/VRAG/hf_models/Qwen/Qwen3-VL-Embedding-2B`
- Embedding code: `external/VRAG/search_engine/models/Qwen3_VL_Embedding/qwen3_vl_embedding.py`
- Input instruction: default `"Represent the user's input."`
- Embedding normalization: enabled for both image and text embeddings.
- FAISS index: `IndexFlatIP`; with normalized embeddings this is cosine-like inner product search.

Retriever image preprocessing:

- Qwen3-VL embedder constants: `min_pixels=4 * 32 * 32 = 4096`, `max_pixels=1800 * 32 * 32 = 1843200`.
- The image content passed to `qwen_vl_utils.process_vision_info` includes those `min_pixels` / `max_pixels`.

Index shards:

| Shard | Vectors | Metadata Rows |
| --- | ---: | ---: |
| `index_full_qwen3vl2b_shard00_existing` | 17920 | 17920 |
| `index_full_qwen3vl2b_shard01` | 8240 | 8240 |
| `index_full_qwen3vl2b_shard02` | 8240 | 8240 |
| `index_full_qwen3vl2b_shard03` | 8239 | 8239 |
| `index_full_qwen3vl2b_shard04` | 8240 | 8240 |
| `index_full_qwen3vl2b_shard05` | 8240 | 8240 |
| `index_full_qwen3vl2b_shard06` | 8239 | 8239 |
| `index_full_qwen3vl2b_shard07` | 8240 | 8240 |
| `index_full_qwen3vl2b_shard08` | 8239 | 8239 |
| total | 83837 | 83837 |

Potential issue:

- `shard00_existing` came from the earlier single-GPU build, while shards 01-08 came from the later sharded build. It passes count/model-path validation and no embedding errors were found, but the log does not record exact package versions at build time.

## Scoped Retrieval

Implementation:

- Eval script passes each row's `record["images"]` as `allowed_image_paths`.
- Search API forwards `allowed_image_paths` to `SearchEngine.search`.
- `SearchEngine._resolve_allowed_indices` normalizes path keys and maps allowed paths to FAISS row indices.
- `SearchEngine._search_vectors` reconstructs only the allowed candidate vectors and ranks within that candidate set.

This is real candidate-pool restriction, not global retrieval followed by post-filtering.

Runtime retrieval config:

- `search_top_k=3`
- `repeat_limit=1`
- Every search call returned at least one candidate.
- Only the first returned path that had not already been used is added to the model context.

Full-run retrieval stats:

| Metric | Value |
| --- | ---: |
| Rows | 3296 |
| Search calls | 4960 |
| Returned retrieval paths | 14647 |
| Selected search images fed to generator | 4395 |
| Successful bbox crops fed to generator | 3723 |
| Search calls with 3 returned candidates | 4800 |
| Search calls with 2 returned candidates | 87 |
| Search calls with 1 returned candidate | 73 |
| Search calls where no new image was added | 565 |
| Retrieval-scope violations | 0 |

Potential issue:

- `top_k=3` does not mean three images are appended to the model per search. It means up to three candidates are returned, then the eval loop selects one unseen image. If the top candidates are repeats, the model gets no new page on that turn.

## Generation Model And Image Resolution

Generator:

- Weights: `external/VRAG/hf_models/Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`
- Served model name: `Qwen/Qwen2.5-VL-7B-Instruct`
- vLLM: `0.8.5.post1`
- vLLM settings: `dtype=bfloat16`, `max_model_len=32768`, `limit_mm_per_prompt={'image': 10}`, `gpu_memory_utilization=0.8`.
- Search API ran on GPU 0; generation replicas served ports 8002-8008 on GPUs 1-7.

Chat/completion settings:

- `max_tokens=2048`
- `temperature=0.0`
- `top_p=1.0`
- `stream=False`
- `max_steps=10`

Generation image preprocessing:

- Eval script uses `max_pixels=512 * 28 * 28 = 401408`.
- Eval script uses `min_pixels=256 * 28 * 28 = 200704`.
- Images are resized preserving aspect ratio, converted to RGB, JPEG-encoded as data URLs, and sent via OpenAI-compatible `image_url`.
- Selected page image processed dimensions over the full run are roughly:
  - Median: 556 x 720, area about 400588 pixels.
  - P90: 752 x 753, area about 400900 pixels.
  - Max processed area: 401349 pixels.

Potential issues:

- 0.4 MP is much smaller than many raw PDF pages. Raw selected page median was about 1700 x 2200, and max raw selected image was 7200 x 12800. Dense tables and small text may be unreadable after downscale.
- vLLM logs show the fast Qwen2-VL image processor default was used. That may differ slightly from an author's slow-processor baseline.
- `limit_mm_per_prompt image=10` matches `max_steps=10` in the worst case, but a row can use both search images and bbox crops. The eval loop stops at 10 assistant turns, so it should not exceed the limit in observed traces.

## Multi-Turn Retrieval Distribution

If "multi-turn retrieval" means more than one `search` action:

- Total rows with >=2 searches: 663/3296 = 20.1%.
- `mmlongbench`: 325/1089 = 29.8%.
- `longdocurl`: 338/2207 = 15.3%.

Search calls per row:

| Search Calls | Rows |
| ---: | ---: |
| 1 | 2633 |
| 2 | 297 |
| 3 | 68 |
| 4 | 179 |
| 5 | 63 |
| 6 | 6 |
| 7 | 9 |
| 8 | 5 |
| 9 | 1 |
| 10 | 35 |

If counting all image turns (`selected search image + successful bbox crop`):

| Image Turns | Rows |
| ---: | ---: |
| 1 | 112 |
| 2 | 2523 |
| 3 | 220 |
| 4 | 226 |
| 5 | 39 |
| 6 | 126 |
| 7 | 7 |
| 8 | 15 |
| 9 | 4 |
| 10 | 24 |

BBox calls per row:

| BBox Calls | Rows |
| ---: | ---: |
| 0 | 265 |
| 1 | 2651 |
| 2 | 204 |
| 3 | 125 |
| 4 | 14 |
| 5 | 13 |
| 6 | 2 |
| 7 | 2 |
| 8 | 8 |
| 9 | 12 |

Other loop stats:

- Invalid or missing action tags occurred in 2 rows.
- BBox errors occurred in 7 rows, with 30 bbox-error turns total.
- `max_steps` rows: 88 total, 45 `mmlongbench`, 43 `longdocurl`.

## Answer Extraction And Scoring

Legacy scored sample:

- 200 random `mmlongbench` + 200 random `longdocurl`, seed `20260710`.
- Judge model: `gpt-5-nano`
- Workers: 32
- Batch size: 100
- Task timeout: 180s
- Fallback judge: none
- Accuracy: 137/400 = 0.3425
- `mmlongbench`: 52/200 = 0.2600
- `longdocurl`: 85/200 = 0.4250

Diagnostic `single_call_v1` on the same sample:

- Judge model: `gpt-5-nano`
- Workers: 32
- Batch size: 100
- Task timeout: 900s
- Fallback judge: none
- Accuracy: 157/400 = 0.3925
- `mmlongbench`: 57/200 = 0.2850
- `longdocurl`: 100/200 = 0.5000

Scorer disagreement:

- Legacy wrong and diagnostic wrong: 233.
- Legacy wrong and diagnostic correct: 30.
- Legacy correct and diagnostic correct: 127.
- Legacy correct and diagnostic wrong: 10.

Interpretation:

- Legacy score is probably a lower bound for this sample because 30 legacy-wrong rows become correct when judging the raw final answer.
- The diagnostic scorer is not a perfect replacement, because 10 legacy-correct rows become wrong.
- The most reliable error set is the 233 rows wrong under both scorers.

Potential scoring issue:

- `wrong_cases_analysis.tsv` is comma-delimited despite the `.tsv` suffix.
- For non-answered rows, `score_vrag_results.py` attaches a zero score locally and sets `compute_score_success=True`; this means "no scorer API failure", not "model produced a valid answer".

## Failure Modes

Legacy wrong categories over the 400-row sample:

| Category | Count |
| --- | ---: |
| `numeric_wrong` | 97 |
| `list_or_multipart_wrong_or_incomplete` | 38 |
| `section_or_table_selection_wrong` | 35 |
| `unanswerable_false_positive` | 33 |
| `string_entity_wrong` | 30 |
| `no_final_answer_max_steps` | 14 |
| `figure_table_title_wrong` | 11 |
| `answerable_false_abstain` | 5 |

Stable wrong categories after diagnostic cross-check:

| Category | Count |
| --- | ---: |
| `numeric_wrong` | 93 |
| `unanswerable_false_positive` | 32 |
| `list_or_multipart_wrong_or_incomplete` | 31 |
| `section_or_table_selection_wrong` | 27 |
| `string_entity_wrong` | 24 |
| `no_final_answer_max_steps` | 14 |
| `figure_table_title_wrong` | 7 |
| `answerable_false_abstain` | 5 |

Failure-mode explanations:

- `numeric_wrong`: The model returned the wrong number, range, count, date, or computed quantity. Likely causes are unreadable downscaled tables, retrieval/crop landing on the right-looking but wrong row/page, or reasoning/calculation errors after seeing the right evidence.
- `list_or_multipart_wrong_or_incomplete`: The model returned only part of a required list or missed one component of a multipart answer. Likely causes are one-page-per-search context, insufficient aggregation across pages/crops, and final-answer truncation or summarization. Some legacy-only cases are scorer extraction artifacts.
- `section_or_table_selection_wrong`: The model selected the wrong table, page section, or nearby heading. This is most common in `longdocurl`, especially when many similar table/figure labels exist on adjacent pages.
- `unanswerable_false_positive`: The model answered even though the ground truth is not answerable. The prompt encourages search and answer generation but does not strongly calibrate abstention; scoped retrieval always returns some in-document page, which can create false evidence.
- `string_entity_wrong`: The model returned the wrong entity/title/phrase or an over-short string. Some are semantic misses; some legacy-only cases are extraction artifacts where the full final answer was more complete than the extracted answer.
- `figure_table_title_wrong`: The model confused figure/table titles, returned only the figure/table number, or selected the wrong title on a page with multiple objects. This overlaps with section selection but is specific to figure/table title tasks.
- `no_final_answer_max_steps`: The eval loop exhausted 10 assistant turns and wrote an empty prediction. The script does not force a final-answer prompt at the end, so search/bbox loops become zero-score rows.
- `answerable_false_abstain`: The model declined or said it could not determine despite the answer being present. Likely causes are missed retrieval/crop, poor visual readability, or over-conservative final reasoning.

## Recommended Follow-Ups

Highest signal checks/variants:

- Rebuild `shard00_existing` with the same sharded command/environment as the other shards if we want to eliminate the residual mixed-build risk.
- Run a matched-budget generation variant with higher `--max-pixels`, for example 1024 or 1536 units, because the current generator image area is only about 0.4 MP.
- Try feeding all top-3 retrieved pages per search, or increase `repeat_limit`/selection logic so repeated searches do not often produce no new evidence.
- Add a final forced answer turn before returning `max_steps`.
- Score the full run or a larger stratified sample with both legacy and `single_call_v1` to separate model failures from legacy extraction false negatives.
