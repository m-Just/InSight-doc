from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from standalone_eval.backends.base import RolloutBackend, RolloutJob
from standalone_eval.core.export import build_rollout_summary_metrics
from standalone_eval.core.resume import (
    CHECKPOINT_EVERY,
    EXPORT_GLOBAL_STEP,
    EXPORT_SPLIT,
    EXPORT_VALIDATE,
    append_checkpoint_records,
    attach_sample_metadata,
    build_basic_config,
    canonicalize_val_files,
    ensure_basic_config_compatible,
    load_resume_samples_by_key,
    should_rerun_existing_sample,
    stable_job_key,
    write_samples_jsonl_atomic,
)
from standalone_eval.core.utils import json_safe, parse_list_arg, progress_bar


def build_jobs(rows: list[dict[str, Any]], num_trials: int) -> list[RolloutJob]:
    jobs: list[RolloutJob] = []
    for trial_idx in range(num_trials):
        for sample_index, row in enumerate(rows):
            if "resume_val_file" not in row or "resume_file_row_idx" not in row:
                raise RuntimeError("backend row is missing required resume provenance")
            resume_val_file = str(row["resume_val_file"])
            resume_file_row_idx = int(row["resume_file_row_idx"])
            jobs.append(
                RolloutJob(
                    output_index=len(jobs),
                    job_key=stable_job_key(
                        trial_idx=trial_idx,
                        resume_val_file=resume_val_file,
                        resume_file_row_idx=resume_file_row_idx,
                    ),
                    sample_index=sample_index,
                    trial_idx=trial_idx,
                    resume_val_file=resume_val_file,
                    resume_file_row_idx=resume_file_row_idx,
                    row=row,
                )
            )
    return jobs


async def run_rollout(
    args: argparse.Namespace,
    *,
    backend: RolloutBackend,
    agent_settings: dict[str, Any],
    agent_name: str,
) -> None:
    eval_t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    export_dir = output_dir / "exported_conversations"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    canonical_val_files = canonicalize_val_files(parse_list_arg(args.val_files))
    args.val_files = canonical_val_files

    await backend.prepare()
    try:
        rows = await backend.load_rows(canonical_val_files, -1)
        if bool(getattr(args, "shuffle_rows", True)):
            random.Random(int(getattr(args, "shuffle_seed", 42))).shuffle(rows)
        max_samples = int(args.max_samples)
        if max_samples > 0:
            rows = rows[:max_samples]
        basic_config = build_basic_config(
            args,
            agent_settings=agent_settings,
            agent_name=agent_name,
            backend_name=backend.backend_name,
            canonical_val_files=canonical_val_files,
            backend_extra=backend.basic_config_extra(),
        )
        ensure_basic_config_compatible(output_dir, basic_config=basic_config)
    except Exception:
        await backend.close()
        raise

    jobs = build_jobs(rows, int(args.num_trials))
    samples: list[dict[str, Any] | None] = [None for _ in jobs]
    queued_jobs: list[RolloutJob] = []
    resume_existing_count = 0
    resume_reused_count = 0
    resume_rerun_failed_count = 0
    resume_missing_count = 0

    existing_samples = load_resume_samples_by_key(output_dir)
    resume_existing_count = len(existing_samples)
    for job in jobs:
        existing_sample = existing_samples.get(job.job_key)
        if existing_sample is None:
            resume_missing_count += 1
            queued_jobs.append(job)
            continue
        if should_rerun_existing_sample(existing_sample):
            resume_rerun_failed_count += 1
            queued_jobs.append(job)
            continue
        samples[job.output_index] = attach_sample_metadata(
            existing_sample,
            job=job,
            source="resume_reused",
        )
        resume_reused_count += 1
    print(
        "standalone resume: "
        f"loaded_existing={resume_existing_count} reused={resume_reused_count} "
        f"queued_missing={resume_missing_count} queued_failed={resume_rerun_failed_count} "
        f"queued_total={len(queued_jobs)}",
        flush=True,
    )

    generation_t0 = time.perf_counter()
    checkpoint_path = output_dir / "checkpoints" / "samples.jsonl"
    checkpoint_buffer: list[tuple[RolloutJob, dict[str, Any]]] = []

    def flush_parent_checkpoint(force: bool = False) -> None:
        if not checkpoint_buffer:
            return
        if not force and len(checkpoint_buffer) < CHECKPOINT_EVERY:
            return
        records = list(checkpoint_buffer)
        checkpoint_buffer.clear()
        append_checkpoint_records(checkpoint_path, records)

    try:
        if queued_jobs:
            print(f"standalone generation: queued {len(queued_jobs)} samples via {backend.backend_name}", flush=True)

            with progress_bar(total=len(queued_jobs), desc="Rollout generation") as pbar:

                async def on_sample(job: RolloutJob, sample: dict[str, Any]) -> None:
                    sample = attach_sample_metadata(sample, job=job, source="generated")
                    samples[job.output_index] = sample
                    checkpoint_buffer.append((job, sample))
                    flush_parent_checkpoint(force=False)
                    pbar.update(1)

                await backend.generate_many(queued_jobs, on_sample)
                flush_parent_checkpoint(force=True)
        else:
            print("standalone generation: no queued jobs; using existing resumed samples", flush=True)
    finally:
        await backend.close()
    generation_t1 = time.perf_counter()

    materialized_samples = [sample for sample in samples if sample is not None]

    scoring_t0 = time.perf_counter()
    for sample in materialized_samples:
        sample.pop("score", None)
    scoring_t1 = time.perf_counter()

    write_t0 = time.perf_counter()
    write_samples_jsonl_atomic(output_dir / "samples.jsonl", materialized_samples)
    summary = build_rollout_summary_metrics(materialized_samples) if materialized_samples else {}
    write_t1 = time.perf_counter()
    eval_t1 = write_t1
    wall_times = {
        "startup_wall_time_s": generation_t0 - eval_t0,
        "generation_wall_time_s": generation_t1 - generation_t0,
        "scoring_wall_time_s": scoring_t1 - scoring_t0,
        "write_wall_time_s": write_t1 - write_t0,
        "eval_wall_time_s": eval_t1 - eval_t0,
    }
    summary["wall_times"] = wall_times
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_extra = {}
    if hasattr(backend, "manifest_extra"):
        manifest_extra = backend.manifest_extra()  # type: ignore[attr-defined]
    server_metadata = getattr(backend, "server_metadata", [])
    standalone_parallelism = getattr(backend, "parallelism", {})
    manifest = {
        "run_id": uuid.uuid4().hex,
        "model": args.model,
        "val_files": canonical_val_files,
        "agent_config": args.agent_config,
        "agent_settings": agent_settings,
        "backend": backend.backend_name,
        "server_metadata": server_metadata,
        "model_config": {
            "path": str(Path(args.model_config).resolve()),
            "sha256": getattr(args, "_model_config_sha256", None),
            "data": getattr(args, "_model_config_data", None),
        },
        "basic_config": basic_config,
        "export_metadata": {
            "global_step": EXPORT_GLOBAL_STEP,
            "split": EXPORT_SPLIT,
            "validate": EXPORT_VALIDATE,
        },
        "standalone_parallelism": standalone_parallelism,
        "wall_times": wall_times,
        "num_samples": len(materialized_samples),
        "num_expected_samples": len(jobs),
        "num_queued_samples": len(queued_jobs),
        "num_trials": args.num_trials,
        "row_order": {
            "shuffle_rows": bool(getattr(args, "shuffle_rows", True)),
            "shuffle_seed": int(getattr(args, "shuffle_seed", 42)),
        },
        "resume": {
            "mode": "automatic",
            "loaded_existing": resume_existing_count,
            "reused_existing": resume_reused_count,
            "queued_missing": resume_missing_count,
            "queued_failed": resume_rerun_failed_count,
            "checkpoint_every": CHECKPOINT_EVERY,
        },
        **manifest_extra,
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "done").touch()
    print(f"standalone rollout complete: {len(materialized_samples)} samples -> {output_dir}")
