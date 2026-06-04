#!/usr/bin/env python3
"""Project a resolved verl Hydra config into a standalone eval-defaults file.

Typical usage:

  python -m verl.trainer.main_ppo ... --cfg job --resolve > resolved_verl_config.yaml
  python scripts/recover_eval_defaults_from_verl_config.py \
    --resolved-config resolved_verl_config.yaml \
    --model-path /path/to/global_step_700/actor_merged_hf \
    --output /path/to/global_step_700/actor_merged_hf/eval_defaults.yaml

The output is intentionally smaller than the full Hydra config. It is meant to
be consumed by evaluate.py via --eval-config and to serve as reproducible
metadata colocated with a checkpoint or merged HF model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_nested(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first_value(mapping: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = get_nested(mapping, path)
        if value is not None:
            return value
    return default


def set_if_not_none(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def as_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(v) for v in value]
    return value


def parse_list_override(value: str | None) -> list[str] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def infer_global_step(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        match = re.search(r"global_step_(\d+)", str(value))
        if match:
            return match.group(1)
    return None


def build_eval_defaults(args: argparse.Namespace, config: dict[str, Any], resolved_config_sha256: str) -> dict[str, Any]:
    rollout = get_nested(config, "actor_rollout_ref.rollout", {}) or {}
    multi_turn = rollout.get("multi_turn") or {}
    rollout_agent = rollout.get("agent") or {}
    val_kwargs = rollout.get("val_kwargs") or {}
    engine_vllm = get_nested(config, "actor_rollout_ref.rollout.engine_kwargs.vllm", {}) or {}
    data_cfg = config.get("data") or {}
    trainer_cfg = config.get("trainer") or {}

    model_path = args.model_path or first_value(config, "actor_rollout_ref.model.path", "model.path")
    val_files = parse_list_override(args.val_files)
    if val_files is None:
        val_files = data_cfg.get("val_files")

    global_step = args.global_step or infer_global_step(args.model_path, args.output, trainer_cfg.get("resume_from_path"))
    run_name = args.run_name or trainer_cfg.get("experiment_name")

    vllm: dict[str, Any] = {}
    for key, value in {
        "dtype": first_value(config, "actor_rollout_ref.rollout.dtype", "actor_rollout_ref.model.torch_dtype"),
        "load_format": rollout.get("load_format"),
        "max_model_len": first_value(config, "actor_rollout_ref.rollout.max_model_len", default=engine_vllm.get("max_model_len")),
        "max_num_seqs": first_value(config, "actor_rollout_ref.rollout.max_num_seqs", default=engine_vllm.get("max_num_seqs")),
        "max_num_batched_tokens": first_value(
            config,
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            default=engine_vllm.get("max_num_batched_tokens"),
        ),
        "gpu_memory_utilization": first_value(
            config,
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            default=engine_vllm.get("gpu_memory_utilization"),
        ),
        "tensor_parallel_size": first_value(
            config,
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            default=engine_vllm.get("tensor_parallel_size"),
        ),
        "enable_prefix_caching": first_value(
            config,
            "actor_rollout_ref.rollout.enable_prefix_caching",
            default=engine_vllm.get("enable_prefix_caching"),
        ),
        "disable_custom_all_reduce": first_value(
            config,
            "actor_rollout_ref.rollout.disable_custom_all_reduce",
            default=engine_vllm.get("disable_custom_all_reduce"),
        ),
        "disable_log_stats": first_value(
            config,
            "actor_rollout_ref.rollout.disable_log_stats",
            default=engine_vllm.get("disable_log_stats"),
        ),
        "trust_remote_code": first_value(
            config,
            "actor_rollout_ref.model.trust_remote_code",
            "actor_rollout_ref.rollout.trust_remote_code",
            default=engine_vllm.get("trust_remote_code"),
        ),
        "generation_config": first_value(
            config,
            "actor_rollout_ref.rollout.generation_config",
            default=engine_vllm.get("generation_config"),
        ),
        "override_generation_config": first_value(
            config,
            "actor_rollout_ref.rollout.override_generation_config",
            default=engine_vllm.get("override_generation_config"),
        ),
        "scheduling_policy": first_value(
            config,
            "actor_rollout_ref.rollout.scheduling_policy",
            default=engine_vllm.get("scheduling_policy"),
        ),
        "enable_sleep_mode": first_value(
            config,
            "actor_rollout_ref.rollout.enable_sleep_mode",
            "actor_rollout_ref.rollout.free_cache_engine",
            default=engine_vllm.get("enable_sleep_mode"),
        ),
        "enable_chunked_prefill": first_value(
            config,
            "actor_rollout_ref.rollout.enable_chunked_prefill",
            default=engine_vllm.get("enable_chunked_prefill"),
        ),
        "enforce_eager": first_value(config, "actor_rollout_ref.rollout.enforce_eager", default=engine_vllm.get("enforce_eager")),
        "seed": first_value(config, "actor_rollout_ref.rollout.seed", default=engine_vllm.get("seed")),
    }.items():
        set_if_not_none(vllm, key, value)

    if "override_generation_config" not in vllm and rollout.get("response_length") is not None:
        vllm["override_generation_config"] = {
            "temperature": rollout.get("temperature", 1),
            "top_k": rollout.get("top_k", -1),
            "top_p": rollout.get("top_p", 1),
            "repetition_penalty": 1.0,
            "max_new_tokens": rollout.get("response_length"),
        }

    sampling: dict[str, Any] = {}
    for key, value in {
        "temperature": val_kwargs.get("temperature", rollout.get("temperature")),
        "top_p": val_kwargs.get("top_p", rollout.get("top_p")),
        "top_k": val_kwargs.get("top_k", rollout.get("top_k")),
        "presence_penalty": val_kwargs.get("presence_penalty", rollout.get("presence_penalty")),
        "repetition_penalty": val_kwargs.get("repetition_penalty", rollout.get("repetition_penalty")),
        "max_tokens": rollout.get("response_length"),
    }.items():
        set_if_not_none(sampling, key, value)

    agent_config_path = args.agent_config or rollout_agent.get("agent_loop_config_path")
    agent_config_name = args.agent_config_name
    qwen_tool_list = parse_list_override(args.qwen_tool_list) or multi_turn.get("qwen_tool_list")
    tool_parser = args.tool_parser or multi_turn.get("format")

    standalone_args: dict[str, Any] = {}
    for key, value in {
        "model_path": model_path,
        "val_files": val_files,
        "agent_config": agent_config_path,
        "agent_config_name": agent_config_name,
        "prompt_length": rollout.get("prompt_length"),
        "response_length": rollout.get("response_length"),
        "max_model_len": vllm.get("max_model_len"),
        "max_user_turns": multi_turn.get("max_user_turns"),
        "max_assistant_turns": multi_turn.get("max_assistant_turns"),
        "max_parallel_calls": multi_turn.get("max_parallel_calls"),
        "qwen_tool_list": qwen_tool_list,
        "tool_parser": tool_parser,
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "presence_penalty": sampling.get("presence_penalty"),
        "repetition_penalty": sampling.get("repetition_penalty"),
        "global_step": global_step,
        "split": args.split,
        "export_validate": args.validate,
        "run_name": run_name,
    }.items():
        set_if_not_none(standalone_args, key, value)

    return {
        "schema_version": "insight_eval_defaults_v1",
        "source": {
            "kind": "reconstructed_from_resolved_verl_config",
            "resolved_config": str(args.resolved_config.resolve()),
            "resolved_config_sha256": resolved_config_sha256,
            "launch_script": str(args.launch_script.resolve()) if args.launch_script else None,
            "launch_script_sha256": sha256_file(args.launch_script) if args.launch_script else None,
            "checkpoint": model_path,
            "global_step": global_step,
            "run_name": run_name,
            "notes": args.notes,
        },
        "model": {
            "path": model_path,
            "load_format": vllm.get("load_format"),
        },
        "vllm": vllm,
        "rollout": {
            "prompt_length": rollout.get("prompt_length"),
            "response_length": rollout.get("response_length"),
        },
        "sampling": sampling,
        "agent": {
            "name": rollout_agent.get("default_agent_loop"),
            "config_path": agent_config_path,
            "config_name": agent_config_name,
            "tool_parser": tool_parser,
            "qwen_tool_list": qwen_tool_list,
            "max_user_turns": multi_turn.get("max_user_turns"),
            "max_assistant_turns": multi_turn.get("max_assistant_turns"),
            "max_parallel_calls": multi_turn.get("max_parallel_calls"),
        },
        "dataset": {
            "train_files": data_cfg.get("train_files"),
            "val_files": val_files,
            "cache_dir": data_cfg.get("cache_dir"),
            "max_pixels": first_value(config, "data.max_pixels", "data.validation_max_pixels"),
            "image_patch_size": data_cfg.get("image_patch_size"),
        },
        "reward": {
            "reward_model": config.get("reward_model"),
            "custom_reward_function": config.get("custom_reward_function"),
        },
        "export": {
            "global_step": global_step,
            "split": args.split,
            "validate": args.validate,
            "run_name": run_name,
            "trial_name": args.trial_name,
        },
        "standalone_args": standalone_args,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--launch-script", type=Path)
    parser.add_argument("--val-files", help="Override val files as a comma-separated list or [a,b].")
    parser.add_argument("--agent-config")
    parser.add_argument("--agent-config-name", default="insight_qwen_agent_core")
    parser.add_argument("--qwen-tool-list")
    parser.add_argument("--tool-parser")
    parser.add_argument("--global-step")
    parser.add_argument("--run-name")
    parser.add_argument("--trial-name")
    parser.add_argument("--split", default="val")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_obj = OmegaConf.load(args.resolved_config)
    config = OmegaConf.to_container(config_obj, resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"resolved config must be a mapping: {args.resolved_config}")

    resolved_config_sha256 = sha256_file(args.resolved_config)
    eval_defaults = build_eval_defaults(args, config, resolved_config_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(as_plain(eval_defaults), sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
