from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_MAX_TOKENS_AFTER_INITIAL_PROMPT = 16384

DEFAULT_RAY_VLLM_CONFIG = {
    "num_replicas": 4,
    "gpus_per_replica": 1,
    "max_model_len": 262144,
    "max_num_seqs": 1024,
    "max_num_batched_tokens": 32768,
    "gpu_memory_utilization": 0.9,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "enforce_eager": True,
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}

DEFAULT_HTTPS_OPENAI_CHAT_CONFIG = {
    "base_url": None,
    "api_key_env": "OPENAI_API_KEY",
    "timeout": 180,
    "max_retries": 1,
    "image_format": "png",
    "image_detail": "high",
    "reasoning_effort": "high",
}

# Internal adapter defaults needed by verl's Ray vLLM server wrapper. These are
# not part of the public model config schema.
RAY_VLLM_ADAPTER_DEFAULTS = {
    "dtype": "bfloat16",
    "load_format": "auto",
    "trust_remote_code": False,
    "enable_sleep_mode": True,
    "scheduling_policy": "fcfs",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_config_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def merge_dict(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_model_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    config = require_mapping(data, name=f"model config {path}")
    model = config.get("model")
    if not model:
        raise ValueError(f"model config {path} must define top-level 'model'")
    backend = str(config.get("backend") or "ray_vllm")
    if backend not in {"ray_vllm", "https_openai_chat"}:
        raise ValueError(f"model config {path} has unsupported backend={backend!r}")

    generation = merge_dict(
        {"max_tokens_after_initial_prompt": DEFAULT_MAX_TOKENS_AFTER_INITIAL_PROMPT},
        require_mapping(config.get("generation") or {}, name="generation"),
    )

    normalized = {
        "model": str(model),
        "backend": backend,
        "generation": generation,
    }
    if backend == "ray_vllm":
        normalized["ray_vllm"] = merge_dict(
            DEFAULT_RAY_VLLM_CONFIG,
            require_mapping(config.get("ray_vllm") or {}, name="ray_vllm"),
        )
    else:
        normalized["https_openai_chat"] = merge_dict(
            DEFAULT_HTTPS_OPENAI_CHAT_CONFIG,
            require_mapping(config.get("https_openai_chat") or {}, name="https_openai_chat"),
        )
    return normalized


def apply_model_config(args: Any, config: dict[str, Any]) -> None:
    if not config:
        return
    args.model_config_data = config
    args.model = config["model"]
    args.generation_backend = config["backend"]
    args.response_length = int(config["generation"]["max_tokens_after_initial_prompt"])

    if args.generation_backend == "ray_vllm":
        ray_cfg = config["ray_vllm"]
        sampling_cfg = ray_cfg["sampling"]
        args.model_path = args.model
        args.prompt_length = int(ray_cfg["max_model_len"])
        args.max_model_len = int(ray_cfg["max_model_len"])
        args.temperature = float(sampling_cfg["temperature"])
        args.top_p = float(sampling_cfg["top_p"])
        args.top_k = int(sampling_cfg["top_k"])
        args.presence_penalty = float(sampling_cfg["presence_penalty"])
        args.repetition_penalty = float(sampling_cfg["repetition_penalty"])
        args.ray_num_replicas = int(ray_cfg["num_replicas"])
        args.ray_gpus_per_replica = int(ray_cfg["gpus_per_replica"])
        args.ray_max_num_seqs = int(ray_cfg["max_num_seqs"])
        args.ray_max_num_batched_tokens = int(ray_cfg["max_num_batched_tokens"])
        args.ray_gpu_memory_utilization = float(ray_cfg["gpu_memory_utilization"])
        args.ray_enable_prefix_caching = bool(ray_cfg["enable_prefix_caching"])
        args.ray_enable_chunked_prefill = bool(ray_cfg["enable_chunked_prefill"])
        args.ray_enforce_eager = bool(ray_cfg["enforce_eager"])
        args.ray_dtype = RAY_VLLM_ADAPTER_DEFAULTS["dtype"]
        args.ray_load_format = RAY_VLLM_ADAPTER_DEFAULTS["load_format"]
        args.ray_trust_remote_code = RAY_VLLM_ADAPTER_DEFAULTS["trust_remote_code"]
        args.ray_enable_sleep_mode = RAY_VLLM_ADAPTER_DEFAULTS["enable_sleep_mode"]
        args.ray_scheduling_policy = RAY_VLLM_ADAPTER_DEFAULTS["scheduling_policy"]
    else:
        https_cfg = config["https_openai_chat"]
        args.model_path = None
        args.prompt_length = None
        args.max_model_len = None
        args.https_base_url = https_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")
        args.https_api_key_env = str(https_cfg.get("api_key_env") or "OPENAI_API_KEY")
        args.https_timeout = float(https_cfg.get("timeout", 180))
        args.https_max_retries = int(https_cfg.get("max_retries", 1))
        args.https_image_format = str(https_cfg.get("image_format") or "png").upper()
        args.https_image_detail = optional_config_string(https_cfg.get("image_detail", "high"))
        args.https_reasoning_effort = optional_config_string(https_cfg.get("reasoning_effort", "high"))
