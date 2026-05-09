import dataclasses
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_ENV_VAR = "VSEARCH_PROFILE_DIR"


def get_profile_dir(config: Any = None) -> str | None:
    profile_dir = os.getenv(PROFILE_ENV_VAR)
    if profile_dir:
        return profile_dir

    if config is None:
        return None

    try:
        agent_config = config.actor_rollout_ref.rollout.agent
        profile_dir = agent_config.get("vreasoner_v2_profile_dir")
    except Exception:
        profile_dir = None
    return profile_dir or None


def profile_enabled(config: Any = None) -> bool:
    return bool(get_profile_dir(config))


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and value.__class__.__module__.startswith("torch"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "size") and value.__class__.__module__.startswith("PIL."):
        return {"pil_image_size": list(value.size), "mode": getattr(value, "mode", None)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_profile_event(stream: str, event: dict[str, Any], *, config: Any = None, profile_dir: str | None = None) -> None:
    profile_dir = profile_dir or get_profile_dir(config)
    if not profile_dir:
        return

    path = Path(profile_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event,
        }
        out_path = path / f"{stream}.{os.getpid()}.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True) + "\n")
    except Exception:
        # Profiling must never perturb validation.
        return


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }
