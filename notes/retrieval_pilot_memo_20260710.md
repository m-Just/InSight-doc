# Retrieval Pilot Memo

Last updated: 2026-07-11T17:33:35Z

## Goal

Run a clean pilot of an external retrieval/evaluation stack on `mmlongbench` and `longdocurl` with the author's default settings first, then optionally add a matched-budget variant.

Before writing a batch evaluator, read and understand the target codebase, especially the retrieval API/driver path, and assess whether it can be patched to accept either:

- `allowed_ids`
- `allowed_image_paths`

## Current Status

- VRAG source is cloned and inspected at `external/VRAG`.
- Runtime env is prepared at `.conda_envs/vrag-pilot`.
- Generator weights are downloaded at `external/VRAG/hf_models/Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`.
- Retriever weights are downloaded at `external/VRAG/hf_models/Qwen/Qwen3-VL-Embedding-2B`.
- `mmlongbench` and `longdocurl` manifests are prepared under `external/VRAG/workspace/insight_doc_eval`.
- VRAG search API and search engine have been patched for lazy loading, GET compatibility, and exact `allowed_ids` / `allowed_image_paths` restricted search.
- A 10-row GPU-backed pilot evaluation completed on `mmlongbench` and `longdocurl`.
- The full 83837-image index build completed and passed validation.
- The full eval completed in restricted-retrieval mode with GPU 0 serving retrieval and GPUs 1-7 serving seven vLLM/eval shards.
- Pipeline audit and failure-mode analysis are summarized in `notes/vrag_eval_pipeline_audit_20260710.md`.
- Added optional trace artifact capture to `external/VRAG/scripts/eval_insight_vrag.py` via `--trace-artifact-dir`, plus an HTML renderer at `external/VRAG/scripts/render_vrag_trace_html.py`.

## Current Assumptions

- Target repository confirmed by user: `https://github.com/Alibaba-NLP/VRAG`.
- Target generator model confirmed by user: `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`.
- `Flyecnu/EviProp` was an inferred candidate from the task wording and public paper metadata, not a confirmed requirement, and is not useful for this pilot because the public repo is only a placeholder.
- The intended pilot datasets are `mmlongbench` and `longdocurl`.
- The first pilot should use author defaults before changing retrieval budget or filtering logic.
- Any filtering should constrain the retriever candidate pool, not post-hoc hide retrieved pages after scoring, unless the author stack makes true candidate restriction impractical.

## Environment

- Active VRAG env: `.conda_envs/vrag-pilot`.
- Python: `3.10.20`.
- Key installed packages:
  - `torch==2.6.0+cu124`
  - `transformers==4.57.6`
  - `vllm==0.8.5.post1`
  - `faiss-cpu==1.8.0`
  - `qwen-vl-utils==0.0.14`
  - `torchcodec==0.2.1`
- `pip check` passes.
- Import checks pass for `torch`, `transformers`, `vllm`, `faiss`, and VRAG's `Qwen3VLEmbedder`.
- Important compatibility finding: `transformers==4.51.3` was insufficient because it lacked `transformers.models.qwen3_vl`; upgrading to `4.57.6` fixed the import.

## Actions Taken

- Searched local source/notes for `allowed_ids`, `allowed_image_paths`, retrieval API/driver references, and pilot notes.
- No local note or source file clearly identified the exact external repository or model ID for this task.
- Tried cloning tentative repo `https://github.com/Flyecnu/EviProp.git` into `external/EviProp`.
- First clone failed with proxy/network error: `Received HTTP code 403 from proxy after CONNECT`.
- Retried with proxy environment variables unset for the `git clone` command.
- Clone succeeded: `external/EviProp`.
- Inspected `external/EviProp`; repository currently contains only `README.md` with `coming soon!`.
- Checked branches/tags/log:
  - Branches: `main`, `origin/main`
  - Tags: none
  - Latest commit: `2f51b28 Add 'coming soon!' message to README`
- Tracked files at `HEAD`: `README.md`
- Interpretation: this repo cannot currently be used to inspect or patch a retrieval API/driver because no implementation is present.
- User provided the likely intended source/model:
  - Repo: `https://github.com/Alibaba-NLP/VRAG`
  - Model: `https://huggingface.co/Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`
- Cloned VRAG with proxy environment variables unset: `external/VRAG`.
- VRAG clone commit: `3cc25fb43f4eaca99b7cf78ee05ca7ef54540776`.
- Inspected VRAG README, `requirements.txt`, `run_demo.sh`, demo agents, search API, search engine implementation, and retrieval call sites.
- Found that the README badge links `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`, while `run_demo.sh` and parts of the README manual launch use `autumncc/Qwen2.5-VL-7B-VRAG`.
- Found that the default search engine path expects an index at `search_engine/corpus/image_index`, but the clone does not include a prebuilt index.
- Found that the search API currently accepts JSON fields `queries`, `top_k`, and `vrag_ret`, then calls `SearchEngine.search(queries, top_k)`.
- Found that `SearchEngine` stores metadata in `file_data_list` and searches a `faiss.IndexFlatIP` index.
- Tried to download `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG` with `huggingface-cli` and proxy variables unset.
  - Escalated Hugging Face download approval timed out twice before running.
  - No model files were downloaded under `external/VRAG/hf_models`.
- Checked local caches/paths for exact model directories:
  - No exact `Qwen2.5-VL-7B-VRAG` directory found in the searched scratch paths.
  - No exact `Qwen3-VL-Embedding-2B` directory found in the searched scratch paths.
- Located likely full pilot parquet inputs:
  - `notes/generated/testcase_0504_full_parquets/mmlongbench_full-insight_qwen_agent.parquet`
  - `notes/generated/testcase_0504_full_parquets/longdocurl_full-insight_qwen_agent.parquet`
  - No-tool/no-system variants also exist under `notes/generated/testcase_0504_full_parquets_no_tool_no_system/`.
- Inspected full parquet schemas:
  - `mmlongbench_full`: 1089 rows, columns `images`, `data_source`, `prompt`, `reward_model`, `extra_info`, `agent_name`.
  - `longdocurl_full`: 2207 rows, same columns.
- Confirmed parquet rows contain exact page image paths in `images`, using `file:///scratch/.../pdf_image/.../*.jpg`.
- Confirmed corresponding image roots:
  - `/scratch/ywxzml3j/likaican/data/insight_doc/testcase_0504/testcase_0504/manifest_dpi200_max40_sourcemmlongbench_samplesall_qidsfromnone/pdf_image`
  - `/scratch/ywxzml3j/likaican/data/insight_doc/testcase_0504/testcase_0504/manifest_dpi200_max40_sourcelongdocurl_samplesall_qidsfromnone/pdf_image`

## Prepared Artifacts and Patches

- Downloaded generator model:
  - Path: `external/VRAG/hf_models/Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`
  - Size: about 16 GB
- Downloaded retriever model:
  - Path: `external/VRAG/hf_models/Qwen/Qwen3-VL-Embedding-2B`
  - Size: about 4 GB
- Prepared evaluation/corpus manifests:
  - `external/VRAG/workspace/insight_doc_eval/eval_full.jsonl`: 3296 rows
  - `external/VRAG/workspace/insight_doc_eval/corpus_full.jsonl`: 83837 unique page images
  - `external/VRAG/workspace/insight_doc_eval/eval_pilot.jsonl`: 10 rows
  - `external/VRAG/workspace/insight_doc_eval/corpus_pilot.jsonl`: 197 page images
- Added scripts:
  - `external/VRAG/scripts/prepare_insight_vrag_data.py`
  - `external/VRAG/scripts/build_insight_vrag_index.py`
  - `external/VRAG/scripts/eval_insight_vrag.py`
- Patched files:
  - `external/VRAG/.gitignore`
  - `external/VRAG/search_engine/__init__.py`
  - `external/VRAG/search_engine/models/Qwen3_VL_Embedding/qwen3_vl_embedding.py`
  - `external/VRAG/search_engine/search_engine.py`
  - `external/VRAG/search_engine/search_engine_api.py`
  - `external/VRAG/demo/vrag_agent.py`
- Fixed local issues found during preparation:
  - Package import failure from empty `search_engine/__init__.py`.
  - Top-level `models...` import that broke repo-root package imports.
  - Eager search engine/index construction at FastAPI import time.
  - Demo agent response parsing mismatch with the included search API.
  - Missing GET `/search` compatibility for `VRAG-RL/vrag_agent/generation.py`.
  - Dataset image array truth-value failure in the data-prep script.
  - Script-path import failure when running `scripts/build_insight_vrag_index.py` directly.
  - `qwen-vl-utils==0.0.10` incompatibility with Qwen3-VL `image_patch_size=16`; upgraded to `0.0.14`.
  - Silent bad-index success when every image embedding failed; index builder now verifies final vector count.
  - Excessive per-batch shape logging during index builds.
- Verified scripts and patched modules with `py_compile`.
- Verified restricted search semantics with a synthetic FAISS index.
- Verified evaluator dry-run loading for the 10-row pilot manifest.
- Added local ignore rules for downloaded weights and generated workspace artifacts.

## Pilot Execution Results

- Stopped placeholder `occupy_idle_gpus.sh` burners to free the GPUs.
- Built corrected pilot index:
  - Path: `external/VRAG/workspace/insight_doc_eval/index_pilot_qwen3vl2b`
  - Vectors: 197
  - Embedder: `Qwen/Qwen3-VL-Embedding-2B`
- Started pilot search API on GPU 0 and validated retrieval with a POST `/search` request.
- Started vLLM on GPU 1:
  - Model: `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`
  - Served name: `Qwen/Qwen2.5-VL-7B-Instruct`
  - Settings: `--dtype bfloat16 --max-model-len 32768 --limit-mm-per-prompt image=10`
- Ran one-row smoke eval successfully; it exercised search, image retrieval, bbox crop, and answer generation.
- Ran 10-row default pilot:
  - Output: `external/VRAG/workspace/insight_doc_eval/runs/vrag_rl_pilot_default.jsonl`
  - Rows: 10
  - Datasets: 5 `mmlongbench`, 5 `longdocurl`
  - Statuses: 9 `answered`, 1 `max_steps`
  - Qualitative correctness is mixed; several answered predictions do not match ground truth.

## Full Index Build

- Superseded single-GPU build:
  - Old PID: `3589125`
  - Old output: `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl2b`
  - Preserved as shard 00 after stopping the job.
- Preserved shard:
  - Path: `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl2b_shard00_existing`
  - Vectors: 17920
- Completed sharded build:
  - PID/log table: `external/VRAG/workspace/insight_doc_eval/logs/build_full_index_sharded_20260710T134518Z/pids.tsv`
  - Logs: `external/VRAG/workspace/insight_doc_eval/logs/build_full_index_sharded_20260710T134518Z`
  - Shard outputs: `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl2b_shard01` through `index_full_qwen3vl2b_shard08`
  - Index directory list for search API: `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl2b_dirs.txt`
- Corpus size: 83837 page images
- Remaining corpus after preserved shard: 65917 page images split across 8 shards.
- GPUs: 0-7, one builder per GPU.
- Batch size: 32.
- Observed sharded rate: roughly 16-19 seconds per 32-image batch per GPU.
- Final exact validation:
  - Shard corpus rows, FAISS vectors, and file metadata rows all match.
  - Total corpus rows: 83837
  - Total vectors: 83837
  - Total metadata rows: 83837
  - No shard logs contain tracebacks, OOMs, runtime errors, or vector-count mismatches.
- Combined functional validation:
  - `SearchEngine.load_multi_index_corpus_together()` loaded all 9 index dirs.
  - Combined `index.ntotal`: 83837
  - Combined `file_data_list`: 83837
  - Unrestricted search returned 3 results.
  - `allowed_image_paths` search returned only allowed paths.
## Full Evaluation Result

- Search API:
  - GPU: 0
  - Port: 8001
  - PID: 4012985
  - Index dirs: all directories listed in `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl2b_dirs.txt`
  - Log: `external/VRAG/workspace/insight_doc_eval/logs/full_eval_20260710T151009Z/search_api.log`
- Generation/eval workers:
  - GPUs: 1-7
  - Ports: 8002-8008
  - Eval shards: 7 deterministic shards over `eval_full.jsonl`
  - Shard outputs: `external/VRAG/workspace/insight_doc_eval/runs/vrag_rl_full_index_allowed_eval_full_shard00.jsonl` through `external/VRAG/workspace/insight_doc_eval/runs/vrag_rl_full_index_allowed_eval_full_shard06.jsonl`
  - Logs: `external/VRAG/workspace/insight_doc_eval/logs/full_eval_20260710T151009Z/eval_shards`
  - PID table: `external/VRAG/workspace/insight_doc_eval/logs/full_eval_20260710T151009Z/eval_shards/pids.tsv`
- Retrieval mode:
  - Full global index is loaded by the search API.
  - Each eval request passes `allowed_image_paths` from that row's dataset image list.
  - This keeps the author retrieval stack but restricts candidates to the current document/pages, avoiding the cross-document retrieval failures seen in the unrestricted smoke run.
- Smoke checks before launch:
  - Unrestricted full-index 10-row smoke completed but retrieved unrelated pages for some document-QA rows.
  - Restricted full-index 10-row smoke completed with all retrieved paths inside each sample's allowed page list.
- Current monitoring at launch:
  - All shard outputs are growing.
  - No tracebacks, OOMs, HTTP 5xx errors, or timeout errors observed in search, vLLM, or eval logs at the first post-launch check.
  - GPU 0 is intentionally reserved for retrieval; all 8 GPUs are active, with 7 GPUs doing generation.
- Completion summary:
  - Merged output: `external/VRAG/workspace/insight_doc_eval/runs/vrag_rl_full_index_allowed_eval_full_merged.jsonl`
  - Total rows: 3296
  - Unique `(dataset, question_id)` pairs: 3296
  - Dataset rows: 1089 `mmlongbench`, 2207 `longdocurl`
  - Statuses: 3208 `answered`, 88 `max_steps`
  - Per-dataset statuses: `mmlongbench` 1044 `answered` / 45 `max_steps`; `longdocurl` 2164 `answered` / 43 `max_steps`
  - JSON validation: no malformed rows
  - Retrieval-scope validation: 14647 retrieved paths checked, 0 paths outside each row's allowed image list
  - Runtime-log validation: no tracebacks, OOMs, HTTP 5xx errors, connection failures, or timeout errors found in the search, vLLM, or eval logs.

## VRAG Code Reading Findings

- Main demo search API: `external/VRAG/search_engine/search_engine_api.py`.
- Main retriever implementation: `external/VRAG/search_engine/search_engine.py`.
- Author default API fields: `queries`, `top_k`, `vrag_ret`.
- Author default top-k in demo agents is `3`.
- Supported embedding models listed by README:
  - `Alibaba-NLP/GVE-3B`
  - `Alibaba-NLP/GVE-7B`
  - `Qwen/Qwen3-VL-Embedding-2B`
  - `Qwen/Qwen3-VL-Embedding-8B`
- Code default embedding model in `search_engine_api.py` is `Qwen3-VL-Embedding-2B`; README examples use path-like `/path/to/Qwen3-VL-Embedding-2B`.
- `SearchEngine.build_index()` stores either `uid` records from JSONL input or `file_path` records from folder input.
- `SearchEngine.search()` encodes text queries, calls FAISS `search()`, and returns scores, row indices, and corresponding `file_data_list` records.
- Feasible patch path for `allowed_ids` / `allowed_image_paths`:
  - Extend request parsing in `search_engine_api.py`.
  - Extend `SearchEngine.search()` signature.
  - Build lookup maps from `file_data_list` after index load.
  - Restrict candidate row indices before scoring.
  - For `IndexFlatIP`, exact restricted search can be implemented by reconstructing candidate vectors and computing query-vector dot products over the allowed subset.
- The local parquets already expose per-sample allowed page images via `images`, so `allowed_image_paths` can be generated directly from each row without guessing document page IDs.
- Avoid post-hoc filtering of global top-k results because it is not equivalent to searching only within allowed candidates.
- Path normalization is required for `allowed_image_paths`, because stored paths may be relative to the VRAG repo while dataset paths may be absolute, symlinked, or `file://` style.
- `demo/vrag_agent.py` appears inconsistent with the API shape: it posts without `vrag_ret=True` but parses the response as if the API returned a raw list of image records.
- `demo/vimrag_agent.py` is consistent with the default API shape and reads `response.json()["results"]`.
- `VRAG-RL/vrag_agent/generation.py` uses `requests.get(..., params={"queries": batch_queries})`, while the demo FastAPI service exposes `POST /search`; this may refer to a different training-time retriever service.

## Current Blockers

- No current setup blocker.
- The full evaluation completed and the merged output has been validated.
- Trace patch testing completed:
  - Python syntax/import check passed for `external/VRAG/scripts/eval_insight_vrag.py` and `external/VRAG/scripts/render_vrag_trace_html.py`.
  - Direct `process_image()` sanity check returned resized image bytes, data URL, and expected processed dimensions.
  - Live GPU trace smoke run `trace_gpu_smoke_live_20260710T182808Z` produced 2 answered rows, 4 exact model-input image artifacts, and a rendered HTML viewer.
  - All model-input image artifacts from that live smoke existed and were non-empty.
  - End-to-end runner smoke `runner_trace_smoke_20260710T183344Z` used the sweep runner on GPUs 0-1 with `LIMIT=2`, produced 2 answered rows, 6 trace steps, 4 exact model-input image artifacts, and rendered HTML successfully.
  - Static runner validation passed with `bash -n external/VRAG/scripts/run_insight_vrag_eval_variant.sh`.
- Current GPU state after smoke cleanup: all 8 GPUs free according to `nvidia-smi`.

## Feasibility Answers After Reading Code

- Retrieval requests are built in the demo agent path and in `VRAG-RL/vrag_agent/generation.py`.
- The included FastAPI service accepts structured JSON on POST; it has been extended to also support GET query params for the VRAG-RL rollout path.
- Candidate metadata is represented by `file_data_list` records containing IDs and/or file paths, with row indices aligned to the FAISS index.
- Default retrieval is over a global page-image index.
- `allowed_ids` and `allowed_image_paths` can be passed through without rebuilding the index.
- Exact restricted search is feasible for `IndexFlatIP` by reconstructing allowed vectors and dot-scoring only that subset.
- For pilot reproducibility, pin top-k, max turns, generation temperature, model paths, and index manifest path in the launch commands.

## Potential Issues To Watch

- ID mismatch: dataset `question_id` / document page paths may not match retriever index IDs.
- Path mismatch: absolute image paths in our parquets may differ from paths stored in the retrieval index.
- Filtering semantics: filtering after top-k retrieval is not equivalent to searching only within allowed candidates.
- Performance: per-sample candidate masks can be expensive if implemented naively over FAISS/GPU indexes.
- Evaluation drift: author defaults may use a different judge, answer normalization, or prompt format than our sweep stack.
- Image preprocessing drift: retrieval may resize/crop/encode pages differently from our existing InSight/VERL inputs.
- Dependency conflicts: retrieval repos often pin old `torch`, `transformers`, `faiss`, `flash-attn`, or CUDA-specific wheels; keep isolated from existing conda envs.
- Demo/runtime mismatch: README/model badge points to `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`, but scripts hardcode `autumncc/Qwen2.5-VL-7B-VRAG`.
- API contract mismatch: `demo/vrag_agent.py` and `VRAG-RL/vrag_agent/generation.py` do not both match the included FastAPI search service.
- Missing index: `search_engine/corpus/image_index` must be built before the included search API can start.
- Dataset leakage: if allowed IDs are derived from answer-support pages, keep the pilot labels explicit and separate from normal retrieval evaluation.

## Next Steps

1. Run 2B retrieval sweep variants with the tested all-in-one runner: `search_top_k=5`, `search_top_k=8`, and `search_top_k=8` with `max_pixels=1536`, `min_pixels=768`.
2. Score completed rollout variants using the current single-call judge scorer settings: `gpt-5-nano`, 32 workers, batch size 100, 900s task timeout.
3. Download/build the 8B retrieval index, then test 8B variants conditionally based on the 8B default result.

## 2026-07-10 Sweep Progress Update

- Trace patch and runner validation completed with GPU smoke runs:
  - `trace_gpu_smoke_live_20260710T182808Z`: 2 answered rows, 4 exact model-input image artifacts, HTML rendered.
  - `runner_trace_smoke_20260710T183344Z`: all-in-one runner smoke, 2 answered rows, 6 trace steps, 4 exact model-input image artifacts, HTML rendered.
- Completed 2B rollout variants:
  - `vrag_2b_topk5_20260710T183701Z`: 3296 rows, 3135 answered, 161 max_steps, no duplicate keys.
  - `vrag_2b_topk8_20260710T190709Z`: 3296 rows, 3081 answered, 215 max_steps, no duplicate keys.
  - `vrag_2b_topk8_px1536_min768_20260710T193805Z`: 3296 rows, 3141 answered, 155 max_steps, no duplicate keys.
- Completed scorer results using current setting: `gpt-5-nano`, `single_call_v1`, 32 workers, batch size 100, 900s task timeout, 3 retries, no fallback judge, sample 200 per dataset seed 20260710:
  - 2B top_k=5: 156/400 = 0.3900 overall; mmlongbench 53/200 = 0.265; longdocurl 103/200 = 0.515.
  - 2B top_k=8: 153/400 = 0.3825 overall; mmlongbench 51/200 = 0.255; longdocurl 102/200 = 0.510.
  - 2B top_k=8 high-res: 161/400 = 0.4025 overall; mmlongbench 48/200 = 0.240; longdocurl 113/200 = 0.565.
- 8B retrieval model download completed under `external/VRAG/hf_models/Qwen/Qwen3-VL-Embedding-8B`.
- Prepared 8-way corpus shards under `external/VRAG/workspace/insight_doc_eval/corpus_full_shards_8way`: 83837 total rows.
- 8B index build:
  - Added `external/VRAG/scripts/build_insight_vrag_8b_index_8way.sh` to launch all 8 shards and validate vector counts before writing `workspace/insight_doc_eval/index_full_qwen3vl8b_dirs.txt`.
  - Batch size 8 was stable but projected around 3+ hours; stopped before checkpoint.
  - Batch size 16 is currently running on all 8 GPUs; GPU memory is about 57.7 GB per card.
  - Batch size 16 reached the first save checkpoint on all 8 shards at about 33/655 batches per shard, so the build can resume cleanly if interrupted.
  - Latest observed checkpoint: 2048 vectors saved per shard.
  - Latest observed checkpoint: 5632 vectors saved per shard.
  - Latest observed runtime progress: about 411-413/655 batches per shard, roughly 63% complete.
  - Current batch-16 projected remaining time is roughly 45-60 minutes for index completion.
  - Watcher session `76198` is waiting for `workspace/insight_doc_eval/index_full_qwen3vl8b_dirs.txt`; after the validated manifest appears it will launch `8B top_k=3` and score it with the current scorer settings.

## 2026-07-10T23:23Z Patch Test Verification

- Re-ran static checks after the trace patch:
  - `python -m py_compile external/VRAG/scripts/eval_insight_vrag.py external/VRAG/scripts/render_vrag_trace_html.py`
  - `bash -n external/VRAG/scripts/run_insight_vrag_eval_variant.sh`
  - `bash -n external/VRAG/scripts/build_insight_vrag_8b_index_8way.sh`
- Verified smoke trace outputs:
  - `trace_gpu_smoke_live_20260710T182808Z`: 2 rows, 6 trace steps, 4 `model_input_image_path` references.
  - `runner_trace_smoke_20260710T183344Z`: 2 rows, 6 trace steps, 4 `model_input_image_path` references.
  - Both smoke runs have 4 JPEG trace artifacts and 0 zero-byte artifacts.
  - Example model-input image sizes recorded from artifacts: `556x720`, `644x622`, `1276x314`.
  - Trace metadata contains raw/source sizes, model-input image sizes, selected image paths, bbox crop metadata, and artifact paths.
- Current 8B status:
  - 8B index build is still running, so the full 8B top-k rollout has not started yet.
  - Latest observed index progress is about 630-633/655 batches per shard.
  - Strict watcher session is waiting for build PID `419110`; after validated manifest creation it will launch `vrag_8b_topk3_*` and score it with the current scorer settings.

## 2026-07-11 Final 8B Sweep Results

- 8B index build completed and validated:
  - Manifest: `external/VRAG/workspace/insight_doc_eval/index_full_qwen3vl8b_dirs.txt`.
  - 8 shard indexes validated against corpus shards: 83,837 expected vectors, 83,837 actual vectors.
- Scorer settings for all listed sweep scores:
  - `gpt-5-nano`, `single_call_v1`, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, sample 200 per dataset, seed 20260710.
- Completed sweep results:
  - 2B top_k=5, run `vrag_2b_topk5_20260710T183701Z`: 3,296 rows, 3,135 answered, 161 max_steps, 0 duplicate keys, 156/400 = 0.3900 overall; mmlongbench 53/200 = 0.265, longdocurl 103/200 = 0.515.
  - 2B top_k=8, run `vrag_2b_topk8_20260710T190709Z`: 3,296 rows, 3,081 answered, 215 max_steps, 0 duplicate keys, 153/400 = 0.3825 overall; mmlongbench 51/200 = 0.255, longdocurl 102/200 = 0.510.
  - 2B top_k=8 high-res, run `vrag_2b_topk8_px1536_min768_20260710T193805Z`: 3,296 rows, 3,141 answered, 155 max_steps, 0 duplicate keys, 161/400 = 0.4025 overall; mmlongbench 48/200 = 0.240, longdocurl 113/200 = 0.565.
  - 8B top_k=3, run `vrag_8b_topk3_20260710T233006Z`: 3,296 rows, 3,195 answered, 101 max_steps, 0 duplicate keys, 169/400 = 0.4225 overall; mmlongbench 58/200 = 0.290, longdocurl 111/200 = 0.555.
  - 8B top_k=5, run `vrag_8b_topk5_20260711T000200Z`: 3,296 rows, 3,130 answered, 166 max_steps, 0 duplicate keys, 166/400 = 0.4150 overall; mmlongbench 57/200 = 0.285, longdocurl 109/200 = 0.545.
  - 8B top_k=8, run `vrag_8b_topk8_20260711T003242Z`: 3,296 rows, 3,090 answered, 206 max_steps, 0 duplicate keys, 166/400 = 0.4150 overall; mmlongbench 57/200 = 0.285, longdocurl 109/200 = 0.545.
- Conditional decisions:
  - 8B top_k=3 beat the 2B default sample baseline, 0.4225 vs 0.3925, so 8B top_k=5 and top_k=8 were launched.
  - 8B top_k=8 did not beat 8B top_k=3, 0.4150 vs 0.4225, so the conditional 8B top_k=8 high-res run was not launched.
- End state:
  - All requested non-conditional and satisfied conditional sweep cells are completed and scored.
  - No VRAG eval, vLLM, search API, or scorer processes remained after cleanup.
  - All 8 GPUs were free at the final check.

## 2026-07-11 `testcase_0504` Max40 Slice Sweep Scores

- Full scoring completed for the six sweep rollouts over all 3,296 rows in the local `testcase_0504` max40 rendered slice.
- Important caveat: these are not the canonical full benchmark cardinalities of 1,091 `mmlongbench` rows and 2,325 `longdocurl` rows. The VRAG prep used `notes/generated/testcase_0504_full_parquets/*_full-insight_qwen_agent.parquet`, which contain 1,089 `mmlongbench` rows and 2,207 `longdocurl` rows.
- Scorer settings: `gpt-5-nano`, `single_call_v1`, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, seed 20260710.
- Full-score output suffix: `_single_call_v1_full_w32_b100_t900`.

| Run | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| 2B top_k=5 | 1253/3296 = 38.02% | 265/1089 = 24.33% | 988/2207 = 44.77% | 73/88 |
| 2B top_k=8 | 1249/3296 = 37.89% | 262/1089 = 24.06% | 987/2207 = 44.72% | 103/112 |
| 2B top_k=8 high-res | 1347/3296 = 40.87% | 278/1089 = 25.53% | 1069/2207 = 48.44% | 84/71 |
| 8B top_k=3 | 1318/3296 = 39.99% | 276/1089 = 25.34% | 1042/2207 = 47.21% | 48/53 |
| 8B top_k=5 | 1307/3296 = 39.65% | 272/1089 = 24.98% | 1035/2207 = 46.90% | 66/100 |
| 8B top_k=8 | 1286/3296 = 39.02% | 264/1089 = 24.24% | 1022/2207 = 46.31% | 91/115 |

- Best slice overall score: 2B top_k=8 high-res at 40.87%.
- Best slice MMLongBench score: 2B top_k=8 high-res at 25.53%.
- Best slice LongDocURL score: 2B top_k=8 high-res at 48.44%.
- Score directories:
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk5_20260710T183701Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk8_20260710T190709Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk8_px1536_min768_20260710T193805Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk3_20260710T233006Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk5_20260711T000200Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk8_20260711T003242Z_single_call_v1_full_w32_b100_t900`

## 2026-07-11 2B top_k=3 Rollout and Three-Judge Scores

- Rollout run: `vrag_2b_topk3_20260711T184312Z`.
- Rollout settings: 2B Qwen3-VL retriever, `search_top_k=3`, `max_pixels=512`, `min_pixels=256`, scoped retrieval with `--use-allowed-image-paths`, generator `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`.
- Rollout outputs:
  - Merged JSONL: `external/VRAG/workspace/insight_doc_eval/runs/vrag_2b_topk3_20260711T184312Z_merged.jsonl`.
  - Summary JSON: `external/VRAG/workspace/insight_doc_eval/runs/vrag_2b_topk3_20260711T184312Z_summary.json`.
- Rollout status: 3,296 rows, 3,296 unique keys, 0 duplicate keys, 3,208 answered, 88 max_steps.
- Dataset split: 1,089 `mmlongbench`, 2,207 `longdocurl`.
- Judge settings for all three passes: `gpt-5-nano`, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, seed 20260710.
- Legacy judge had one `RewardWorker` 900s task-timeout warning, then continued and finished; `single_call_v1` and `single_call_v2` finished without observed timeout warnings.

| Judge | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| legacy | 1190/3296 = 36.10% | 262/1089 = 24.06% | 928/2207 = 42.05% | 45/43 |
| single_call_v1 | 1259/3296 = 38.20% | 268/1089 = 24.61% | 991/2207 = 44.90% | 45/43 |
| single_call_v2 | 1226/3296 = 37.20% | 267/1089 = 24.52% | 959/2207 = 43.45% | 45/43 |

- Score directories:
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk3_20260711T184312Z_legacy_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk3_20260711T184312Z_single_call_v1_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk3_20260711T184312Z_single_call_v2_full_w32_b100_t900`
- Cleanup: stopped the older resumable all-run legacy/v2 scorer before this scoring run to avoid judge API contention. After scoring, no VRAG eval, search API, vLLM, or scorer process remained. Unknown GPU worker processes from `/scratch/ywxzml3j/likaican/src/verl-merge` were killed, and all GPUs were free at final check.

## 2026-07-12 Legacy and single_call_v2 Full Sweep Completion

- Completed all remaining `legacy` and `single_call_v2` full-score cells over the local `testcase_0504` max40 slice.
- Scorer settings: `gpt-5-nano`, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, seed 20260710.
- Output suffixes:
  - Legacy: `_legacy_full_w32_b100_t900`.
  - single_call_v2: `_single_call_v2_full_w32_b100_t900`.
- All listed score directories have `done` markers and `num_scored_samples=3296`.
- The legacy judge had intermittent 900s task-timeout warnings but all cells completed. The observed `single_call_v2` cells completed without timeout warnings in the run log.

Legacy judge:

| Run | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| 2B top_k=3 | 1190/3296 = 36.10% | 262/1089 = 24.06% | 928/2207 = 42.05% | 45/43 |
| 2B top_k=5 | 1172/3296 = 35.56% | 258/1089 = 23.69% | 914/2207 = 41.41% | 73/88 |
| 2B top_k=8 | 1180/3296 = 35.80% | 259/1089 = 23.78% | 921/2207 = 41.73% | 103/112 |
| 2B top_k=8 high-res | 1262/3296 = 38.29% | 275/1089 = 25.25% | 987/2207 = 44.72% | 84/71 |
| 8B top_k=3 | 1231/3296 = 37.35% | 271/1089 = 24.89% | 960/2207 = 43.50% | 48/53 |
| 8B top_k=5 | 1224/3296 = 37.14% | 269/1089 = 24.70% | 955/2207 = 43.27% | 66/100 |
| 8B top_k=8 | 1200/3296 = 36.41% | 257/1089 = 23.60% | 943/2207 = 42.73% | 91/115 |

single_call_v2 judge:

| Run | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| 2B top_k=3 | 1226/3296 = 37.20% | 267/1089 = 24.52% | 959/2207 = 43.45% | 45/43 |
| 2B top_k=5 | 1201/3296 = 36.44% | 263/1089 = 24.15% | 938/2207 = 42.50% | 73/88 |
| 2B top_k=8 | 1198/3296 = 36.35% | 259/1089 = 23.78% | 939/2207 = 42.55% | 103/112 |
| 2B top_k=8 high-res | 1304/3296 = 39.56% | 276/1089 = 25.34% | 1028/2207 = 46.58% | 84/71 |
| 8B top_k=3 | 1266/3296 = 38.41% | 275/1089 = 25.25% | 991/2207 = 44.90% | 48/53 |
| 8B top_k=5 | 1247/3296 = 37.83% | 267/1089 = 24.52% | 980/2207 = 44.40% | 66/100 |
| 8B top_k=8 | 1245/3296 = 37.77% | 265/1089 = 24.33% | 980/2207 = 44.40% | 91/115 |

- Best legacy score: 2B top_k=8 high-res at 38.29% overall.
- Best single_call_v2 score: 2B top_k=8 high-res at 39.56% overall.
- End state: no scorer, VRAG eval, search API, or VRAG-owned vLLM process remained after the loop. An unrelated active Slurm/vLLM job was present on GPUs and was not killed.

## 2026-07-13 8B top_k=8 High-Res Completion

- Completed rollout run: `vrag_8b_topk8_px1536_min768_20260713T101237Z`.
- Rollout settings: 8B Qwen3-VL retriever, `search_top_k=8`, `max_pixels=1536`, `min_pixels=768`, scoped retrieval with the 8B full index file `workspace/insight_doc_eval/index_full_qwen3vl8b_dirs.txt`, generator `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`.
- Rollout outputs:
  - Merged JSONL: `external/VRAG/workspace/insight_doc_eval/runs/vrag_8b_topk8_px1536_min768_20260713T101237Z_merged.jsonl`.
  - Summary JSON: `external/VRAG/workspace/insight_doc_eval/runs/vrag_8b_topk8_px1536_min768_20260713T101237Z_summary.json`.
- Rollout status: 3,296 rows, 3,296 unique keys, 0 duplicate keys, 3,131 answered, 165 max_steps.
- Dataset split: 1,089 `mmlongbench`, 2,207 `longdocurl`.
- Judge settings for both passes: `gpt-5-nano`, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, seed 20260710.
- Both score directories have `done` markers and `num_scored_samples=3296`.
- Legacy judge had one observed 900s `RewardWorker` timeout warning, then continued and finished. `single_call_v2` finished without observed timeout warnings.
- No Slurm jobs or ambiguous GPU processes were killed for this completion run. Only the current Slurm allocation was used for rollout; post-run GPU processes that were not confirmed VRAG-owned were left untouched.

| Judge | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| legacy | 1302/3296 = 39.50% | 287/1089 = 26.35% | 1015/2207 = 45.99% | 84/81 |
| single_call_v2 | 1365/3296 = 41.41% | 291/1089 = 26.72% | 1074/2207 = 48.66% | 84/81 |

- Score directories:
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk8_px1536_min768_20260713T101237Z_legacy_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk8_px1536_min768_20260713T101237Z_single_call_v2_full_w32_b100_t900`
- This is now the best completed legacy score in the local full sweep table at 39.50% overall, and the best completed `single_call_v2` score at 41.41% overall.

## 2026-07-13 2B/8B top_k=3 High-Res Legacy Completion

- Completed rollout and legacy scoring for `search_top_k=3`, `max_pixels=1536`, `min_pixels=768`.
- Rollout settings: scoped retrieval with `--use-allowed-image-paths`, generator `Qiuchen-Wang/Qwen2.5-VL-7B-VRAG`, 3 generation shards per run, 1 search GPU per run.
- Judge settings: `gpt-5-nano`, legacy mode, 32 workers, batch size 100, 900s task timeout, 3 retries, 10s retry interval, no fallback judge, seed 20260710.
- 2B rollout run: `vrag_2b_topk3_px1536_min768_20260713T124945Z`.
  - Summary: 3,296 rows, 3,296 unique keys, 0 duplicate keys, 3,211 answered, 85 max_steps.
  - Index: `workspace/insight_doc_eval/index_full_qwen3vl2b_dirs.txt`.
- 8B rollout run: `vrag_8b_topk3_px1536_min768_20260713T134415Z`.
  - Summary: 3,296 rows, 3,296 unique keys, 0 duplicate keys, 3,216 answered, 80 max_steps.
  - Index: `workspace/insight_doc_eval/index_full_qwen3vl8b_dirs.txt`.
- Legacy judge had observed 900s `RewardWorker` timeout warnings in both runs, but both score directories completed with `done` markers and `num_scored_samples=3296`.
- The original sequential launcher had a queued duplicate 8B run on GPUs 4-7. It was interrupted after the completed 2B score and before any duplicate rollout outputs were written. The independent 8B run on GPUs 0-3 completed and was scored.
- Post-run GPU processes from `/scratch/ywxzml3j/likaican/src/verl-merge` were inspected and not killed. Recorded VRAG server PIDs had already exited; no Slurm job was touched.

| Run | Overall | MMLongBench | LongDocURL | Non-answered MMLB/LDURL |
|---|---:|---:|---:|---:|
| 2B top_k=3 high-res legacy | 1276/3296 = 38.71% | 282/1089 = 25.90% | 994/2207 = 45.04% | 49/36 |
| 8B top_k=3 high-res legacy | 1324/3296 = 40.17% | 293/1089 = 26.91% | 1031/2207 = 46.71% | 40/40 |

- Score directories:
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_2b_topk3_px1536_min768_20260713T124945Z_legacy_full_w32_b100_t900`
  - `external/VRAG/workspace/insight_doc_eval/scores/vrag_8b_topk3_px1536_min768_20260713T134415Z_legacy_full_w32_b100_t900`
- This makes 8B top_k=3 high-res the best completed legacy result in the local full sweep table at 40.17% overall.
