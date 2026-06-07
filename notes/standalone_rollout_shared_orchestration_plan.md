# Standalone Rollout Shared Orchestration Plan

## Goal

Split rollout into a shared orchestration layer plus backend-specific generation implementations. The shared layer should own run safety, resume, checkpointing, metrics, and final outputs. Backends should own only the pieces that genuinely differ: row materialization, model transport, prompt/message rendering, generation, and backend-specific export details.

## Target Files

- `standalone_eval/rollout.py`: thin CLI entrypoint and high-level `main`.
- `standalone_eval/judge.py`: separate reward/judging entrypoint.
- `standalone_eval/core/orchestrator.py`: shared rollout lifecycle.
- `standalone_eval/backends/base.py`: backend protocol/types.
- `standalone_eval/backends/ray_vllm.py`: Ray/vLLM backend.
- `standalone_eval/backends/https_openai_chat.py`: HTTPS backend.
- `standalone_eval/core/resume.py`: `basic_config.json`, stable job keys, checkpoint load/write.
- `standalone_eval/core/export.py`: shared export/sample helpers where possible.
- `standalone_eval/core/metrics.py`: shared rollout/judge summary metrics.
- `standalone_eval/core/utils.py`: small shared utility helpers.
- `standalone_eval/config/agent.py`: shared agent config, tool schema, dataset config, and sampling helpers used by top-level rollout and Ray worker subprocesses.
- `standalone_eval/config/model.py`: model config loading and backend-specific model defaults.

## Shared Data Types

`RolloutJob`

```python
@dataclass
class RolloutJob:
    output_index: int
    job_key: str
    sample_index: int
    trial_idx: int
    resume_val_file: str
    resume_file_row_idx: int
    row: dict[str, Any]
```

`RolloutRow`

Use the current row dict schema, but require source provenance:

- `resume_val_file`
- `resume_file_row_idx`

`RolloutSample`

Use the current sample dict schema, but require:

- `output_index`
- `sample_index`
- `trial_idx`
- `job_key`
- `resume_val_file`
- `resume_file_row_idx`
- `uid`
- `data_source`
- `ground_truth`
- `solution_str`
- `critical_failure`
- `failure_reasons`
- timing and length fields when available
- `conversation_export_json_path` when exported

`RolloutBackend`

```python
class RolloutBackend(Protocol):
    backend_name: str

    async def prepare(self) -> None: ...
    async def load_rows(self, val_files: list[str], max_samples: int) -> list[dict[str, Any]]: ...
    def basic_config_extra(self) -> dict[str, Any]: ...
    async def generate_many(
        self,
        jobs: list[RolloutJob],
        on_sample: Callable[[RolloutJob, dict[str, Any]], Awaitable[None]],
    ) -> None: ...
    async def close(self) -> None: ...
```

`generate_many` rather than `generate_one` is intentional: Ray/vLLM needs process-pool/global-queue implementation, while HTTPS may remain direct asyncio or later use a process pool. The orchestrator should not know those details.

`load_rows` receives canonicalized `val_files` from the orchestrator. This keeps file ordering and `max_samples` semantics shared rather than backend-defined.

## Shared Orchestration Flow

1. Parse CLI.
2. Load model config.
3. Instantiate backend from `backend` in model config.
4. `await backend.prepare()`.
5. Canonicalize `val_files` by resolved path, then load rows via `await backend.load_rows(...)`. Each row must include source provenance:
   - resolved source parquet path
   - file-local row index
6. Build `basic_config.json`:
   - backend name
   - model
   - model config sha
   - resolved `val_files`
   - `max_samples`
   - `num_trials`
   - agent settings
   - backend `basic_config_extra`
7. Validate or write `basic_config.json`; resume is automatic if compatible prior artifacts exist.
8. Build jobs using stable `job_key = sha256(trial_idx, resolved_val_file_path, file_local_row_idx)`.
9. Load existing samples by `job_key` from checkpoint and final samples.
10. Reuse matching successful samples.
11. Queue missing samples and failed samples.
12. Call `backend.generate_many(queued_jobs, on_sample=...)`.
13. `on_sample` writes every completed sample to the single checkpoint stream and updates in-memory results.
14. Write final `samples.jsonl` atomically.
15. Write rollout-only `metrics.json`.
16. Write final `manifest.json`.
17. Touch `done`.
18. `await backend.close()`.

## Resume Semantics

There should be no explicit `--resume` or `--resume-failed-only` modes. If the output directory contains compatible artifacts, rollout automatically resumes. If the output directory contains incompatible artifacts, rollout fails and the user should choose a new `--output-dir` or restore the original launch config.

Supported:

- Interrupted run resume.
- Resume after completed run with no queued jobs.
- Resume after completed run with failures by rerunning only failed/missing jobs.
- Increasing `num_trials`.
- Adding validation files to `val_files` when `max_samples < 0`, where negative means uncapped.
- Reordered `val_files`, because job identity uses resolved file path plus file-local row index.
- Changing throughput settings such as worker concurrency.

Compatibility rule:

- `basic_config.json` is the compatibility gate.
- `num_trials` may increase but not decrease.
- `val_files` should be canonicalized by resolved path before loading rows, so CLI ordering does not affect row selection or `max_samples`.
- If `max_samples < 0`, resolved existing `val_files` must be a subset of resolved current `val_files`.
- If `max_samples > 0`, resolved existing `val_files` must equal resolved current `val_files`; adding files is rejected to avoid ambiguous row selection.
- If `max_samples == 0`, no rows are selected; this mode is not useful for eval and should be rejected by the CLI.
- Throughput-only settings are not part of `basic_config.json`.
- Validation file contents are trusted not to mutate between runs.
- Rows are identified by `(resolved_val_file_path, file_local_row_idx)`, not by row content hash.

Rejected:

- Changed model config. This includes model path/name, backend-specific inference settings, and the model config file hash. Mixing outputs from two model configs in one output directory would make metrics uninterpretable.
- Changed backend. Ray/vLLM and HTTPS do not have identical transport, tokenization, timing, or image handling behavior, so their outputs should not share a resume directory.
- Changed agent settings. This includes tool availability, prompt/tool behavior, image rescale settings, crop policy, turn limits, and related agent config overrides. These directly affect prompt contents and trajectories.
- Removed or replaced validation files. Existing `val_files`, compared as resolved paths, must remain present. Reordering is allowed.
- Added validation files when `max_samples > 0`. With a capped sample count, adding files can change which rows are selected, so this is rejected.
- Changed `max_samples`. This changes the intended row selection contract. Use a new output directory for a different cap.
- Decreased `num_trials`. Dropping trials would leave old higher-trial outputs in checkpoints/final samples and make the intended result ambiguous.
- Existing rollout artifacts without `basic_config.json`. Without the compatibility gate, the runner cannot determine whether existing samples are safe to reuse.

Under this minimal implementation, validation file contents are trusted not to mutate between runs. If a parquet file is modified in place, resume behavior is undefined for that file; the correct action is to use a new output directory or a new validation file path.

## Job Identity

Use:

```text
job_key = sha256(trial_idx, resolved_val_file_path, file_local_row_idx)
```

This matches the intended resume contract:

- Reordering `val_files` is safe.
- Adding new `val_files` is safe only when `max_samples < 0`.
- Increasing `num_trials` is safe.
- Reordering rows inside an existing validation file is not supported.
- Mutating rows inside an existing validation file is not supported.

The loaded rows should carry resume metadata:

- `resume_val_file`: resolved source parquet path.
- `resume_file_row_idx`: row index within that parquet.

`sample_index` remains an output ordering index for the current run, not a resume identity.

Implementation requirements:

- Provenance must be attached before rows from multiple `val_files` are concatenated.
- `val_files` should be sorted by resolved path before loading.
- `max_samples` should truncate after provenance is attached.
- Ray/vLLM and HTTPS must use the same provenance convention.
- If a backend cannot recover file-local row indices, it should fail rather than fall back to global row position.

## Checkpoint And Final Output

Current target:

- `checkpoints/samples.jsonl`: incremental samples during rollout.
- `samples.jsonl`: final consolidated samples.

Append one envelope per completed sample to `checkpoints/samples.jsonl`:

```json
{
  "job_key": "...",
  "written_at": 123.4,
  "sample": {...}
}
```

Resume load order:

1. `checkpoints/samples.jsonl`.
2. `samples.jsonl`.

For duplicate `job_key` records, keep the newest record by `written_at`. If timestamps are tied or missing, prefer later source order, so `samples.jsonl` wins over `checkpoints/samples.jsonl`.

If a later resumed run writes new checkpoint records after an older final `samples.jsonl`, those newer checkpoint records should win by timestamp. At the end of each successful rollout, `samples.jsonl` is rewritten atomically from the latest per-`job_key` records.

Successful records are reused. Failed records are rerun.

`samples.jsonl` is written atomically at the end in canonical output order. It is intentionally separate from the checkpoint stream: checkpoints optimize recovery during active/interrupted rollout, while `samples.jsonl` is the clean downstream artifact for judging and analysis.

All checkpoint writes should go through the orchestrator or a single parent-process result queue. Backend workers should return samples; they should not own checkpoint file conventions.

## Backend Responsibilities

### Ray/vLLM Backend

Owns:

- tokenizer/processor loading
- `RLHFDataset`
- Ray server manifest validation
- Ray heartbeat
- `InSightQwenAgentRunner`
- process-pool/global-queue agent workers
- token-level Qwen/Hermes tool parsing
- local Qwen image processing path

Does not own:

- resume
- checkpoint writing
- final `samples.jsonl`
- final metrics/manifest
- compatibility checks

### HTTPS Backend

Owns:

- direct parquet row loading
- API client creation
- OpenAI-compatible message construction
- image URL/base64 serialization
- API retry/context-overflow retry behavior
- no-tool enforcement

Does not own:

- resume
- checkpoint writing
- final `samples.jsonl`
- final metrics/manifest
- compatibility checks

Future option:

- Add HTTPS process-pool generation behind `generate_many` without changing orchestrator.

## Reliability Benefits

- One resume implementation.
- One checkpoint implementation.
- One final-output implementation.
- Backend-specific code is isolated.
- HTTPS and Ray/vLLM produce the same sample envelope shape.
- Shared config compatibility and run-safety checks are enforced before backend generation.
- Adding future backends does not duplicate lifecycle code.

## Migration Steps

1. Move `basic_config` and stable job-key helpers to `standalone_eval/core/resume.py`.
2. Move checkpoint/final JSONL helpers to `standalone_eval/core/resume.py`.
3. Introduce `RolloutJob` and backend protocol.
4. Move HTTPS code into `HTTPSOpenAIChatBackend`.
5. Move Ray/vLLM code into `RayVLLMBackend`.
6. Implement `run_rollout(args, backend)` in `standalone_eval/core/orchestrator.py`.
7. Keep `standalone_eval/rollout.py` as parser plus backend factory.
8. Compile and smoke-test `--help`.
9. Run a small HTTPS no-tool sample.
10. Run a small Ray/vLLM sample with existing server manifest.
