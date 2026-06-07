from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from standalone_eval.backends.base import RolloutJob
from standalone_eval.core.utils import json_safe


CHECKPOINT_EVERY = 1
EXPORT_GLOBAL_STEP = None
EXPORT_SPLIT = "val"
EXPORT_VALIDATE = True


def canonicalize_val_files(val_files: list[str]) -> list[str]:
    resolved = [str(Path(path).expanduser().resolve(strict=False)) for path in val_files]
    duplicates = sorted({path for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise ValueError(f"duplicate val_files after path resolution are not supported: {duplicates}")
    return sorted(resolved)


def parquet_num_rows(path: str) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import pandas as pd

        return int(pd.read_parquet(path).shape[0])


def build_row_provenance(val_files: list[str], max_samples: int) -> list[tuple[str, int]]:
    if max_samples == 0:
        raise ValueError("--max-samples=0 is not useful for eval and is rejected")
    provenance: list[tuple[str, int]] = []
    for val_file in val_files:
        for row_idx in range(parquet_num_rows(val_file)):
            provenance.append((val_file, row_idx))
            if max_samples > 0 and len(provenance) >= max_samples:
                return provenance
    return provenance


def stable_job_key(*, trial_idx: int, resume_val_file: str, resume_file_row_idx: int) -> str:
    payload = {
        "trial_idx": int(trial_idx),
        "resume_val_file": str(Path(resume_val_file).expanduser().resolve(strict=False)),
        "resume_file_row_idx": int(resume_file_row_idx),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_basic_config(
    args: Any,
    *,
    agent_settings: dict[str, Any],
    agent_name: str,
    backend_name: str,
    canonical_val_files: list[str],
    backend_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend_name,
        "model": args.model,
        "model_config_sha256": getattr(args, "_model_config_sha256", None),
        "val_files": list(canonical_val_files),
        "max_samples": int(args.max_samples),
        "num_trials": int(args.num_trials),
        "agent": {
            "name": agent_name,
            "settings": agent_settings,
        },
        "export_metadata": {
            "global_step": EXPORT_GLOBAL_STEP,
            "split": EXPORT_SPLIT,
            "validate": EXPORT_VALIDATE,
        },
    }
    if backend_extra:
        config["backend_extra"] = backend_extra
    return json_safe(config)


def _as_path_set(paths: list[Any]) -> set[str]:
    return {str(Path(str(path)).expanduser().resolve(strict=False)) for path in paths}


def compare_basic_config(existing: dict[str, Any], current: dict[str, Any]) -> tuple[bool, str | None]:
    existing_static = dict(existing)
    current_static = dict(current)
    existing_num_trials = int(existing_static.pop("num_trials", 1) or 1)
    current_num_trials = int(current_static.pop("num_trials", 1) or 1)
    existing_val_files = list(existing_static.pop("val_files", []) or [])
    current_val_files = list(current_static.pop("val_files", []) or [])
    existing_max_samples = int(existing_static.get("max_samples", -1))
    current_max_samples = int(current_static.get("max_samples", -1))

    if existing_static != current_static:
        return False, "non-trial config changed"
    if current_num_trials < existing_num_trials:
        return False, f"num_trials decreased from {existing_num_trials} to {current_num_trials}"
    if current_max_samples == 0:
        return False, "max_samples=0 is invalid"

    existing_set = _as_path_set(existing_val_files)
    current_set = _as_path_set(current_val_files)
    if current_max_samples > 0:
        if existing_set != current_set:
            return False, "val_files cannot change when max_samples > 0"
    elif not existing_set.issubset(current_set):
        missing = sorted(existing_set - current_set)
        return False, f"val_files cannot drop previously evaluated files: {missing}"
    return True, None


def ensure_basic_config_compatible(output_dir: Path, *, basic_config: dict[str, Any]) -> None:
    basic_config_path = output_dir / "basic_config.json"
    def has_content(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_dir():
            return any(path.iterdir())
        return path.stat().st_size > 0

    prior_outputs = [
        path.name
        for path in (
            output_dir / "samples.jsonl",
            output_dir / "manifest.json",
            output_dir / "done",
            output_dir / "checkpoints",
            output_dir / "exported_conversations",
        )
        if has_content(path)
    ]
    if basic_config_path.exists():
        existing = json.loads(basic_config_path.read_text(encoding="utf-8"))
        ok, reason = compare_basic_config(existing, basic_config)
        if not ok:
            raise RuntimeError(
                f"resume basic_config mismatch for {basic_config_path}: {reason}; "
                "choose a new --output-dir or restore the original launch config"
            )
    elif prior_outputs:
        raise RuntimeError(
            f"cannot safely resume {output_dir}: existing artifacts found but basic_config.json is missing"
        )
    basic_config_path.write_text(json.dumps(json_safe(basic_config), ensure_ascii=False, indent=2), encoding="utf-8")


def attach_sample_metadata(
    sample: dict[str, Any],
    *,
    job: RolloutJob,
    source: str,
    written_at: float | None = None,
) -> dict[str, Any]:
    timestamp = float(time.time() if written_at is None else written_at)
    sample = dict(sample)
    sample["output_index"] = int(job.output_index)
    sample["job_idx"] = int(job.output_index)
    sample["job_key"] = job.job_key
    sample["sample_index"] = int(job.sample_index)
    sample["trial_idx"] = int(job.trial_idx)
    sample["resume_val_file"] = job.resume_val_file
    sample["resume_file_row_idx"] = int(job.resume_file_row_idx)
    metadata = dict(sample.get("resume_metadata") or {})
    metadata.update(
        {
            "job_idx": int(job.output_index),
            "output_index": int(job.output_index),
            "job_key": job.job_key,
            "sample_index": int(job.sample_index),
            "trial_idx": int(job.trial_idx),
            "resume_val_file": job.resume_val_file,
            "resume_file_row_idx": int(job.resume_file_row_idx),
            "source": source,
            "written_at": timestamp,
        }
    )
    sample["resume_metadata"] = metadata
    return sample


def extract_sample_job_key(sample: dict[str, Any]) -> str | None:
    for value in (
        sample.get("job_key"),
        (sample.get("resume_metadata") or {}).get("job_key") if isinstance(sample.get("resume_metadata"), dict) else None,
    ):
        if value:
            return str(value)
    return None


def sample_resume_timestamp(sample: dict[str, Any], fallback: float) -> float:
    metadata = sample.get("resume_metadata")
    if isinstance(metadata, dict):
        for key in ("written_at", "finalized_at", "generated_at"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return float(fallback)


def iter_resume_record_samples(path: Path) -> list[tuple[int, dict[str, Any], float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    source_mtime = path.stat().st_mtime
    records: list[tuple[int, dict[str, Any], float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: failed to parse resume record {path}:{line_idx + 1}: {exc}", flush=True)
                continue
            if isinstance(record, dict) and "sample" in record:
                sample = record.get("sample")
                explicit_job_idx = record.get("job_idx")
                record_timestamp = record.get("written_at")
            else:
                sample = record
                explicit_job_idx = None
                record_timestamp = None
            if not isinstance(sample, dict):
                continue
            try:
                job_idx = int(explicit_job_idx if explicit_job_idx is not None else sample.get("job_idx"))
            except (TypeError, ValueError):
                continue
            timestamp = sample_resume_timestamp(sample, source_mtime)
            if record_timestamp is not None:
                try:
                    timestamp = float(record_timestamp)
                except (TypeError, ValueError):
                    pass
            records.append((job_idx, sample, timestamp))
    return records


def load_resume_samples_by_key(output_dir: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, tuple[tuple[float, int, int], dict[str, Any]]] = {}
    sequence = 0
    paths: list[tuple[int, Path]] = []
    checkpoint_path = output_dir / "checkpoints" / "samples.jsonl"
    if checkpoint_path.exists():
        paths.append((1, checkpoint_path))
    final_path = output_dir / "samples.jsonl"
    if final_path.exists():
        paths.append((2, final_path))
    for source_rank, path in paths:
        for _job_idx, sample, timestamp in iter_resume_record_samples(path):
            resume_key = extract_sample_job_key(sample)
            if resume_key is None:
                continue
            sequence += 1
            order_key = (float(timestamp), int(source_rank), sequence)
            previous = candidates.get(resume_key)
            if previous is None or order_key >= previous[0]:
                candidates[resume_key] = (order_key, sample)
    return {resume_key: sample for resume_key, (_, sample) in candidates.items()}


def sample_failure_reasons(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_reasons = sample.get("failure_reasons")
    if isinstance(raw_reasons, str):
        reasons.append(raw_reasons)
    elif isinstance(raw_reasons, (list, tuple)):
        reasons.extend(str(reason) for reason in raw_reasons if reason is not None)
    if sample.get("critical_failure") and not reasons:
        reasons.append("critical_failure")
    score = sample.get("score")
    if isinstance(score, dict):
        for key in ("fail_reason", "failure_reason", "error_type"):
            value = score.get(key)
            if value:
                reasons.append(str(value))
    return reasons


def should_rerun_existing_sample(sample: dict[str, Any]) -> bool:
    return bool(sample_failure_reasons(sample))


def sample_has_score(sample: dict[str, Any]) -> bool:
    score = sample.get("score")
    return isinstance(score, dict) and bool(score)


def append_checkpoint_records(checkpoint_path: Path, records: list[tuple[RolloutJob, dict[str, Any]]]) -> None:
    if not records:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for job, sample in records:
            written_at = time.time()
            sample = attach_sample_metadata(sample, job=job, source="worker_checkpoint", written_at=written_at)
            envelope = {
                "job_idx": int(job.output_index),
                "job_key": job.job_key,
                "written_at": written_at,
                "sample": sample,
            }
            f.write(json.dumps(json_safe(envelope), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_samples_jsonl_atomic(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    finalized_at = time.time()
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            sample = dict(sample)
            metadata = dict(sample.get("resume_metadata") or {})
            metadata["source"] = "final"
            metadata["written_at"] = finalized_at
            metadata["finalized_at"] = finalized_at
            sample["resume_metadata"] = metadata
            f.write(json.dumps(json_safe(sample), ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
