#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
for extra_path in (
    REPO_ROOT,
    Path("/scratch/ywxzml3j/likaican/src/InSight-o3"),
    Path("/scratch/ywxzml3j/likaican/src/Qwen-Agent"),
):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from insight_agent_core import (
    CoreFunctionCall,
    InSightQwenAgentConfig,
    InSightQwenAgentRunner,
    StandaloneInSightRuntime,
)
from insight_agent_core.openai_https import OpenAIChatEndpointPool
from insight_agent_core.ray_vllm import RayVLLMEndpointPool
from verl.experimental.agent_loop.qwen_agent_loop import _build_insight_export_conversation
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.trainer.ppo.metric_utils import process_validation_metrics, process_vsearch_validation_metrics
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.reward_score.vsearch_batch import compute_score_batch
from verl.utils.vreasoner_v2_conversation_export import (
    build_export_record,
    build_repeated_conversation_export_id,
    build_root_conversation_export_id,
    export_conversation,
)


DEFAULT_REWARD_KWARGS = {
    "reward_type": "conditioned_on_tool_reward",
    "reward_weights": {
        "format": 0.2,
        "accuracy": 0.8,
        "iou": 0.8,
        "tool": 1.0,
    },
    "format_reward": {
        "must_have_answer": True,
        "simple": False,
    },
    "iou_reward": {
        "iou_low": 0.25,
        "iou_high": 1.0,
        "pseudo_iou_reward_type": "caller_feedback",
    },
    "tool_reward": {
        "max_consecutive_iou": 0.6,
    },
}


def parse_list_arg(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def job_key(sample_index: int, trial_idx: int) -> str:
    return f"{int(trial_idx)}:{int(sample_index)}"


def job_index(sample_index: int, trial_idx: int, rows_per_trial: int) -> int:
    return int(trial_idx) * int(rows_per_trial) + int(sample_index)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structured_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"eval config must be a mapping: {path}")
    return data


def _get_nested(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _first_config_value(mapping: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = _get_nested(mapping, path, None)
        if value is not None:
            return value
    return default


def build_arg_defaults_from_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map a recovered eval-defaults file onto standalone evaluate.py args."""
    defaults = dict(config.get("standalone_args") or {})

    mappings = {
        "generation_backend": ("generation.backend", "standalone.backend", "backend"),
        "https_base_url": ("generation.https_base_url", "generation.base_url", "standalone.https_base_url"),
        "https_model": ("generation.https_model", "generation.model", "standalone.https_model"),
        "https_api_key_env": ("generation.https_api_key_env", "standalone.https_api_key_env"),
        "https_timeout": ("generation.https_timeout", "standalone.https_timeout"),
        "https_max_retries": ("generation.https_max_retries", "standalone.https_max_retries"),
        "https_image_format": ("generation.https_image_format", "standalone.https_image_format"),
        "https_send_tool_schema": ("generation.https_send_tool_schema", "standalone.https_send_tool_schema"),
        "https_coerce_tool_role_to_user": (
            "generation.https_coerce_tool_role_to_user",
            "standalone.https_coerce_tool_role_to_user",
        ),
        "model_path": ("model.path", "model_path"),
        "val_files": ("dataset.val_files", "data.val_files"),
        "cache_dir": ("dataset.cache_dir", "data.cache_dir"),
        "max_pixels": ("dataset.max_pixels", "data.max_pixels"),
        "image_patch_size": ("dataset.image_patch_size", "data.image_patch_size"),
        "agent_config": ("agent.config_path", "agent.agent_config", "actor_rollout_ref.rollout.agent.agent_loop_config_path"),
        "agent_config_name": ("agent.config_name", "agent.agent_config_name"),
        "prompt_length": ("rollout.prompt_length", "actor_rollout_ref.rollout.prompt_length"),
        "response_length": ("rollout.response_length", "actor_rollout_ref.rollout.response_length"),
        "max_model_len": ("vllm.max_model_len", "actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len"),
        "max_user_turns": (
            "agent.max_user_turns",
            "multi_turn.max_user_turns",
            "actor_rollout_ref.rollout.multi_turn.max_user_turns",
        ),
        "max_assistant_turns": (
            "agent.max_assistant_turns",
            "multi_turn.max_assistant_turns",
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns",
        ),
        "max_parallel_calls": (
            "agent.max_parallel_calls",
            "multi_turn.max_parallel_calls",
            "actor_rollout_ref.rollout.multi_turn.max_parallel_calls",
        ),
        "qwen_tool_list": ("agent.qwen_tool_list", "multi_turn.qwen_tool_list", "actor_rollout_ref.rollout.multi_turn.qwen_tool_list"),
        "no_tool_schema": ("agent.no_tool_schema",),
        "tool_parser": ("agent.tool_parser", "multi_turn.format", "actor_rollout_ref.rollout.multi_turn.format"),
        "skip_reward": ("reward.skip_reward",),
        "temperature": ("sampling.temperature", "actor_rollout_ref.rollout.val_kwargs.temperature"),
        "top_p": ("sampling.top_p", "actor_rollout_ref.rollout.val_kwargs.top_p"),
        "top_k": ("sampling.top_k", "actor_rollout_ref.rollout.val_kwargs.top_k"),
        "presence_penalty": ("sampling.presence_penalty", "actor_rollout_ref.rollout.val_kwargs.presence_penalty"),
        "repetition_penalty": ("sampling.repetition_penalty", "actor_rollout_ref.rollout.val_kwargs.repetition_penalty"),
        "judge_model": ("reward.judge_model", "reward.model", "judge.model"),
        "fallback_judge_model": ("reward.fallback_judge_model", "judge.fallback_model"),
        "judge_workers": ("reward.judge_workers", "judge.workers"),
        "judge_task_timeout": ("reward.judge_task_timeout", "judge.task_timeout"),
        "judge_min_success_rate": ("reward.judge_min_success_rate", "judge.min_success_rate"),
        "judge_max_retries": ("reward.judge_max_retries", "judge.max_retries"),
        "judge_retry_interval": ("reward.judge_retry_interval", "judge.retry_interval"),
        "global_step": ("export.global_step", "source.global_step"),
        "split": ("export.split",),
        "export_validate": ("export.validate",),
        "run_name": ("export.run_name", "source.run_name"),
        "trial_name": ("export.trial_name",),
        "validation_image_token_reorder": (
            "dataset.validation_image_token_reorder.enabled",
            "data.validation_image_token_reorder.enabled",
            "standalone.validation_image_token_reorder",
        ),
        "validation_reorder_num_workers": (
            "dataset.validation_image_token_reorder.num_workers",
            "data.validation_image_token_reorder.num_workers",
        ),
        "validation_reorder_batch_size": (
            "dataset.validation_image_token_reorder.batch_size",
            "data.validation_image_token_reorder.batch_size",
        ),
        "validation_reorder_default_agent_loop": (
            "dataset.validation_image_token_reorder.default_agent_loop",
            "data.validation_image_token_reorder.default_agent_loop",
        ),
    }
    for arg_name, paths in mappings.items():
        if arg_name in defaults:
            continue
        value = _first_config_value(config, *paths)
        if value is not None:
            defaults[arg_name] = value

    if "max_tokens" in config.get("sampling", {}) and "response_length" not in defaults:
        defaults["response_length"] = config["sampling"]["max_tokens"]
    return defaults


def describe_processor(processor: Any) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    return {
        "processor_class": processor.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__ if image_processor is not None else None,
        "do_resize": getattr(image_processor, "do_resize", None),
        "size": getattr(image_processor, "size", None),
        "patch_size": getattr(image_processor, "patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
    }


def load_agent_settings(path: str, preferred_name: str) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    configs = list(loaded) if OmegaConf.is_list(loaded) else [loaded]
    fallback = None
    for config in configs:
        item = OmegaConf.to_container(config, resolve=True)
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name == preferred_name:
            item.pop("name", None)
            item.pop("_target_", None)
            return item
        if name in {"insight_qwen_agent_core", "insight_qwen_agent"}:
            fallback = item
    if fallback is None:
        raise ValueError(f"no InSight agent config found in {path}")
    fallback.pop("name", None)
    fallback.pop("_target_", None)
    return fallback


def build_tool_schemas(qwen_tool_list: list[str]) -> list[dict[str, Any]]:
    import importlib

    # Import for side-effect registration into qwen_agent's TOOL_REGISTRY.
    importlib.import_module("verl.experimental.agent_loop.qwen_agent_tools.image_zoom_in_qwen3vl")
    from qwen_agent.tools import TOOL_REGISTRY

    aliases = {"image_zoom_in_tool_qwen3vl": "image_zoom_in_tool"}
    schemas = []
    seen_names = set()
    for registry_tool_name in qwen_tool_list:
        if registry_tool_name not in TOOL_REGISTRY:
            raise ValueError(f"qwen tool {registry_tool_name!r} not found; available={list(TOOL_REGISTRY.keys())}")
        public_tool_name = aliases.get(registry_tool_name, registry_tool_name)
        if public_tool_name in seen_names:
            raise ValueError(f"duplicate public tool name: {public_tool_name}")
        seen_names.add(public_tool_name)
        tool_instance = TOOL_REGISTRY[registry_tool_name]()
        schema = dict(tool_instance.function)
        schema["name"] = public_tool_name
        schemas.append(schema)
    return schemas


def build_dataset_config(args: argparse.Namespace) -> Any:
    config = {
        "cache_dir": args.cache_dir,
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "image_patch_size": args.image_patch_size,
        "max_prompt_length": args.prompt_length,
        "return_raw_chat": True,
        "return_multi_modal_inputs": False,
        "truncation": "error",
        "filter_overlong_prompts": False,
        "apply_chat_template_kwargs": {"max_tool_calls": args.max_user_turns},
        "tool_config_path": None,
        "need_tools_kwargs": False,
        "filter_prompts": True,
        "shuffle": False,
        "use_shm": False,
        "force_dataset_concat": True,
        "use_vsearch": True,
        "max_pixels": args.max_pixels,
        "validation_max_pixels": args.max_pixels,
        "_is_train": False,
    }
    reorder_settings = getattr(args, "_validation_image_token_reorder_settings", None)
    if reorder_settings:
        config["_validation_image_token_reorder_settings"] = reorder_settings
    return OmegaConf.create(config)


def build_validation_image_token_reorder_settings(
    args: argparse.Namespace,
    core_config: InSightQwenAgentConfig,
) -> dict[str, Any] | None:
    if not args.validation_image_token_reorder:
        return None
    batch_size = int(args.validation_reorder_batch_size)
    num_workers = int(args.validation_reorder_num_workers)
    if batch_size <= 0:
        raise ValueError("--validation-reorder-batch-size must be positive")
    if num_workers <= 0:
        raise ValueError("--validation-reorder-num-workers must be positive")
    default_agent_loop = str(args.validation_reorder_default_agent_loop)
    return {
        "enabled": True,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "default_agent_loop": default_agent_loop,
        "agent_settings_by_name": {
            default_agent_loop: {
                "initial_rescale": core_config.initial_rescale,
                "gpt_image_max_area": core_config.gpt_image_max_area,
            }
        },
    }


def build_core_config(args: argparse.Namespace, agent_settings: dict[str, Any], tool_schemas: list[dict[str, Any]]):
    if "presented_initial_rescale" in agent_settings:
        agent_settings.setdefault("initial_rescale", agent_settings["presented_initial_rescale"])
    if "presented_max_area" in agent_settings:
        agent_settings.setdefault("gpt_image_max_area", agent_settings["presented_max_area"])
    if "crop_max_area" in agent_settings:
        agent_settings.setdefault("crop_image_max_area", agent_settings["crop_max_area"])
    if "presented_initial_pixels_lower_bound" in agent_settings:
        agent_settings.setdefault("initial_input_pixels_lower_bound", agent_settings["presented_initial_pixels_lower_bound"])

    return InSightQwenAgentConfig(
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        max_user_turns=args.max_user_turns,
        max_assistant_turns=args.max_assistant_turns,
        max_parallel_calls=args.max_parallel_calls,
        tool_schemas=None if args.no_tool_schema else tool_schemas,
        tool_parser_name=args.tool_parser,
        initial_rescale=float(agent_settings.get("initial_rescale", 0.25)),
        gpt_image_max_area=int(agent_settings.get("gpt_image_max_area", 1280 * 1280)),
        crop_image_max_area=int(agent_settings.get("crop_image_max_area", 1280 * 1280)),
        initial_input_pixels_lower_bound=int(agent_settings.get("initial_input_pixels_lower_bound", 0)),
        region_zoom_in_factor=float(agent_settings.get("region_zoom_in_factor", 4.0)),
        train_initial_rescale_randomization_prob=float(
            agent_settings.get("train_initial_rescale_randomization_prob", 0.0)
        ),
        train_initial_rescale_randomization_min=float(
            agent_settings.get("train_initial_rescale_randomization_min", 0.25)
        ),
        train_initial_rescale_randomization_max=float(
            agent_settings.get("train_initial_rescale_randomization_max", 0.25)
        ),
        train_initial_rescale_randomization_text_budget=int(
            agent_settings.get("train_initial_rescale_randomization_text_budget", 1024)
        ),
        agent_name=args.reward_agent_name,
    )


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "logprobs": args.logprobs,
    }


def build_reward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    reward_kwargs = json.loads(json.dumps(DEFAULT_REWARD_KWARGS))
    reward_kwargs.update(
        {
            "judge_model": args.judge_model,
            "fallback_judge_model": args.fallback_judge_model,
            "num_workers": args.judge_workers,
            "task_timeout": args.judge_task_timeout,
            "min_success_rate": args.judge_min_success_rate,
            "max_retries": args.judge_max_retries,
            "retry_interval": args.judge_retry_interval,
        }
    )
    return reward_kwargs


def _ray_namespace(args: argparse.Namespace) -> str:
    namespace = getattr(args, "_ray_namespace", None) or args.ray_namespace
    if namespace:
        return str(namespace)
    return f"standalone_eval_{uuid.uuid4().hex[:12]}"


def _ray_rollout_config(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "async",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "dtype": args.ray_dtype,
            "gpu_memory_utilization": args.ray_gpu_memory_utilization,
            "ignore_eos": False,
            "enforce_eager": args.ray_enforce_eager,
            "free_cache_engine": False,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "tensor_model_parallel_size": args.ray_gpus_per_replica,
            "pipeline_model_parallel_size": 1,
            "max_num_batched_tokens": args.ray_max_num_batched_tokens,
            "scheduling_policy": args.ray_scheduling_policy,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.ray_max_num_seqs,
            "disable_log_stats": True,
            "enable_chunked_prefill": args.ray_enable_chunked_prefill,
            "enable_prefix_caching": args.ray_enable_prefix_caching,
            "load_format": args.ray_load_format,
            "enable_sleep_mode": args.ray_enable_sleep_mode,
            "engine_kwargs": {"vllm": {"max_model_len": args.max_model_len}},
        }
    )


def _ray_model_config(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "path": args.model_path,
            "trust_remote_code": args.ray_trust_remote_code,
            "load_tokenizer": True,
            "use_shm": False,
            "lora_rank": 0,
            "override_config": {},
        }
    )


async def launch_ray_vllm_servers(args: argparse.Namespace) -> tuple[list[Any], list[str], list[dict[str, Any]], str]:
    import ray
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    namespace = _ray_namespace(args)
    ray_init_kwargs: dict[str, Any] = {
        "namespace": namespace,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": True,
    }
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    else:
        ray_init_kwargs["num_gpus"] = args.ray_num_replicas * args.ray_gpus_per_replica
        if args.ray_num_cpus:
            ray_init_kwargs["num_cpus"] = args.ray_num_cpus
        ray_temp_dir = args.ray_temp_dir or f"/tmp/vray_{uuid.uuid4().hex[:6]}"
        ray_init_kwargs["_temp_dir"] = ray_temp_dir
    ray_context = ray.init(**ray_init_kwargs)

    actor_cls = ray.remote(vLLMHttpServer)
    rollout_config = _ray_rollout_config(args)
    model_config = _ray_model_config(args)
    handles: list[Any] = []
    actor_names: list[str] = []
    metadata: list[dict[str, Any]] = []

    for replica_rank in range(args.ray_num_replicas):
        actor_name = f"standalone_vllm_{replica_rank}_{uuid.uuid4().hex[:8]}"
        server = actor_cls.options(
            num_gpus=args.ray_gpus_per_replica,
            num_cpus=args.ray_cpus_per_server,
            name=actor_name,
            namespace=namespace,
        ).remote(
            config=rollout_config,
            model_config=model_config,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=replica_rank,
            node_rank=0,
            gpus_per_node=args.ray_gpus_per_replica,
            nnodes=1,
        )
        handles.append(server)
        actor_names.append(actor_name)

    for replica_rank, server in enumerate(handles):
        master_address, master_port = await server.get_master_address.remote()
        print(
            f"launching Ray vLLM server replica={replica_rank} "
            f"actor={actor_names[replica_rank]} master={master_address}:{master_port}",
            flush=True,
        )
        await server.launch_server.remote(master_address=master_address, master_port=master_port)
        server_address, server_port = await server.get_server_address.remote()
        metadata.append(
            {
                "endpoint_type": "verl_ray_vllm",
                "actor_name": actor_names[replica_rank],
                "server_address": f"{server_address}:{server_port}",
                "replica_rank": replica_rank,
                "model": args.model_path,
                "max_model_len": args.max_model_len,
                "ray_namespace": namespace,
            }
        )
        print(
            f"ready Ray vLLM server replica={replica_rank} "
            f"actor={actor_names[replica_rank]} address={server_address}:{server_port}",
            flush=True,
        )

    args._ray_namespace = namespace
    args._ray_actor_names = actor_names
    address_info = getattr(ray_context, "address_info", {}) or {}
    args._ray_address = args.ray_address or address_info.get("address") or address_info.get("gcs_address") or "auto"
    return handles, actor_names, metadata, namespace


def connect_ray_vllm_servers(args: argparse.Namespace) -> list[Any]:
    import ray

    namespace = getattr(args, "_ray_namespace", None) or args.ray_namespace
    actor_names = list(getattr(args, "_ray_actor_names", None) or [])
    if not namespace or not actor_names:
        raise ValueError("Ray server actor names/namespace are missing; launch servers before workers")
    if not ray.is_initialized():
        ray.init(address=getattr(args, "_ray_address", None) or args.ray_address or "auto", namespace=namespace)
    return [ray.get_actor(name, namespace=namespace) for name in actor_names]


def cleanup_ray_vllm_servers(handles: list[Any]) -> None:
    if not handles:
        return
    try:
        import ray

        for handle in handles:
            ray.kill(handle, no_restart=True)
        ray.shutdown()
    except Exception as exc:
        print(f"warning: failed to clean up Ray vLLM servers: {exc}", flush=True)


def resolve_https_api_key(args: argparse.Namespace) -> str:
    if args.https_api_key:
        return str(args.https_api_key)
    env_name = args.https_api_key_env or "OPENAI_API_KEY"
    return os.getenv(env_name) or "EMPTY"


def optional_https_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def build_https_server_metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "endpoint_type": "openai_compatible_https_chat",
            "base_url": args.https_base_url,
            "model": args.https_model or args.model_path,
            "timeout": args.https_timeout,
            "max_retries": args.https_max_retries,
            "image_format": args.https_image_format,
            "image_detail": optional_https_string(args.https_image_detail),
            "reasoning_effort": optional_https_string(args.https_reasoning_effort),
            "send_tool_schema": args.https_send_tool_schema,
            "coerce_tool_role_to_user": args.https_coerce_tool_role_to_user,
            "api_stack": "insight_o3.utils.api",
            "api_logging": "enabled_required",
            "api_key_env": args.https_api_key_env,
            "api_key_provided": bool(args.https_api_key or os.getenv(args.https_api_key_env or "OPENAI_API_KEY")),
        }
    ]


def make_export_id(row: dict[str, Any], sample_index: int, trial_idx: int) -> str:
    extra_info = dict(row.get("extra_info") or {})
    if not extra_info.get("question_id"):
        extra_info["question_id"] = f"sample-{sample_index}"
    base_id = extra_info.get("conversation_export_base_id")
    if not base_id:
        base_id = build_root_conversation_export_id(
            extra_info=extra_info,
            data_source=row.get("data_source"),
            validate=True,
            val_trial_idx=trial_idx,
        )
    return build_repeated_conversation_export_id(str(base_id), 0)


def export_result(
    *,
    result,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: dict[str, Any],
    export_dir: Path,
    sample_index: int,
    trial_idx: int,
) -> str:
    payload = result.export_payload
    initial_question = payload.extra_info.get("question", "")
    core_export = getattr(args, "_core_config_for_export", {}) or {}
    eval_config_path = getattr(args, "eval_config", None)
    eval_config_sha256 = getattr(args, "_eval_config_sha256", None)
    record = build_export_record(
        job_id=payload.request_id,
        parent_job_id=None,
        root_job_id=payload.request_id,
        validate=bool(args.export_validate),
        initial_question=initial_question,
        messages_api=[],
        raw_prompt=payload.raw_prompt,
        original_images=payload.original_images,
        presented_image_refs=payload.presented_image_refs,
        request_params={
            "tool_parser": args.tool_parser,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "max_user_turns": args.max_user_turns,
            "max_assistant_turns": args.max_assistant_turns,
            "max_parallel_calls": args.max_parallel_calls,
        },
        loop_params={
            "implementation": getattr(args, "implementation_name", "insight_agent_core_standalone"),
            "initial_rescale": payload.actual_initial_rescale,
            "configured_initial_rescale": result.extra_fields.get("extra_info", {}).get(
                "initial_rescale",
                payload.actual_initial_rescale,
            ),
            "initial_rescale_randomization": payload.initial_rescale_metadata,
            "initial_prompt_tokens": result.extra_fields.get("initial_prompt_tokens"),
            "initial_prompt_tokens_before_shrink": result.extra_fields.get("initial_prompt_tokens_before_shrink"),
            "initial_prompt_tokens_after_shrink": result.extra_fields.get("initial_prompt_tokens_after_shrink"),
            "initial_prompt_shrink_count": result.extra_fields.get("initial_prompt_shrink_count", 0),
            "initial_prompt_shrink_applied": result.extra_fields.get("initial_prompt_shrink_applied", False),
            "initial_prompt_fit_succeeded": result.extra_fields.get("initial_prompt_fit_succeeded", True),
            "initial_prompt_shrink_warning": result.extra_fields.get("initial_prompt_shrink_warning"),
            "initial_input_pixels_lower_bound": core_export.get("initial_input_pixels_lower_bound"),
            "gpt_image_max_area": core_export.get("gpt_image_max_area"),
            "crop_image_max_area": core_export.get("crop_image_max_area"),
            "region_zoom_in_factor": core_export.get("region_zoom_in_factor"),
            "eval_config_path": str(Path(eval_config_path).resolve()) if eval_config_path else None,
            "eval_config_sha256": eval_config_sha256,
            "lengths": {
                "prompt_tokens": result.extra_fields.get("prompt_tokens"),
                "response_tokens_total": result.extra_fields.get("response_tokens_total"),
                "response_tokens_generated": result.extra_fields.get("response_tokens_generated"),
                "response_tokens_tool": result.extra_fields.get("response_tokens_tool"),
            },
            "timing": {
                "initial_prompt_fit_time": result.extra_fields.get("initial_prompt_fit_time", 0.0),
                "generate_sequences": result.extra_fields.get("generate_sequences", 0.0),
                "tool_parsing": result.extra_fields.get("tool_parsing", 0.0),
                "tool_calls": result.extra_fields.get("tool_calls", 0.0),
                "core_inference_time_raw": result.extra_fields.get("core_inference_time_raw"),
                "core_inference_time": result.extra_fields.get("core_inference_time", 0.0),
                "conversation_wall_time": result.extra_fields.get("conversation_wall_time", 0.0),
            },
            "agent_name": args.reward_agent_name,
        },
        sampling_params=dict(sampling_params),
        tools_kwargs=row.get("tools_kwargs", {}),
        extra_info=payload.extra_info,
        failure_events=payload.failure_events,
        critical_failure=payload.critical_failure,
        final_failure_reasons=payload.final_failure_reasons,
    )
    record["agent_name"] = args.reward_agent_name
    record["conversation"] = _build_insight_export_conversation(payload.messages, initial_question=initial_question)
    record["job"].update(
        {
            "global_step": args.global_step,
            "split": args.split,
            "validate": bool(args.export_validate),
            "trajectory_sample_index": sample_index,
            "rollout_n": trial_idx,
            "run_name": args.run_name,
            "trial_name": args.trial_name,
        }
    )
    index_metadata = {
        "global_step": args.global_step,
        "split": args.split,
        "validate": bool(args.export_validate),
        "trajectory_sample_index": sample_index,
        "rollout_n": trial_idx,
    }
    return export_conversation(
        str(export_dir),
        record,
        job_id=payload.request_id,
        export_id=payload.conversation_export_id,
        index_metadata=index_metadata,
    )


def build_generated_sample_record(
    *,
    result,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: dict[str, Any],
    export_dir: Path,
    sample_index: int,
    trial_idx: int,
    tokenizer,
    started: float,
) -> dict[str, Any]:
    export_path = export_result(
        result=result,
        row=row,
        args=args,
        sampling_params=sampling_params,
        export_dir=export_dir,
        sample_index=sample_index,
        trial_idx=trial_idx,
    )
    response_text = tokenizer.decode(result.response_ids, skip_special_tokens=True)
    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    sample_extra_info = dict(result.extra_fields.get("extra_info") or {})
    sample_extra_info["conversation_export_json_path"] = export_path
    return {
        "sample_index": sample_index,
        "trial_idx": trial_idx,
        "uid": str(row.get("uid")),
        "data_source": row.get("data_source"),
        "ground_truth": ground_truth,
        "solution_str": response_text,
        "extra_info": sample_extra_info,
        "conversation_export_json_path": export_path,
        "response_truncated": result.extra_fields.get("response_truncated"),
        "critical_failure": result.export_payload.critical_failure,
        "failure_reasons": result.export_payload.final_failure_reasons,
        "num_turns": result.num_turns,
        "wall_time_s": time.perf_counter() - started,
        "core_inference_time": result.extra_fields.get("core_inference_time"),
        "core_inference_time_raw": result.extra_fields.get("core_inference_time_raw"),
        "generate_sequences": result.extra_fields.get("generate_sequences"),
        "tool_parsing": result.extra_fields.get("tool_parsing"),
        "tool_calls": result.extra_fields.get("tool_calls"),
        "conversation_wall_time": result.extra_fields.get("conversation_wall_time"),
        "prompt_tokens": result.extra_fields.get("prompt_tokens"),
        "response_tokens_total": result.extra_fields.get("response_tokens_total"),
        "response_tokens_generated": result.extra_fields.get("response_tokens_generated"),
        "response_tokens_tool": result.extra_fields.get("response_tokens_tool"),
        "is_not_answerable": question_type_contains_not_answerable(sample_extra_info.get("question_type"))
        or ground_truth_is_not_answerable(ground_truth),
    }


def attach_resume_metadata(
    sample: dict[str, Any],
    *,
    job_idx: int,
    source: str,
    written_at: float | None = None,
) -> dict[str, Any]:
    """Attach stable resume identity without changing the metric payload."""
    sample = dict(sample)
    sample_idx = int(sample["sample_index"])
    trial_idx = int(sample.get("trial_idx", 0))
    timestamp = float(time.time() if written_at is None else written_at)
    metadata = dict(sample.get("resume_metadata") or {})
    metadata.update(
        {
            "job_idx": int(job_idx),
            "job_key": job_key(sample_idx, trial_idx),
            "sample_index": sample_idx,
            "trial_idx": trial_idx,
            "source": source,
            "written_at": timestamp,
        }
    )
    sample["job_idx"] = int(job_idx)
    sample["job_key"] = job_key(sample_idx, trial_idx)
    sample["resume_metadata"] = metadata
    return sample


def extract_sample_job_idx(sample: dict[str, Any], rows_per_trial: int) -> int | None:
    for value in (
        sample.get("job_idx"),
        (sample.get("resume_metadata") or {}).get("job_idx") if isinstance(sample.get("resume_metadata"), dict) else None,
    ):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    if "sample_index" not in sample:
        return None
    try:
        return job_index(int(sample["sample_index"]), int(sample.get("trial_idx", 0)), rows_per_trial)
    except (TypeError, ValueError):
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


def iter_resume_record_samples(path: Path, *, rows_per_trial: int) -> list[tuple[int, dict[str, Any], float]]:
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
            job_idx = None
            if explicit_job_idx is not None:
                try:
                    job_idx = int(explicit_job_idx)
                except (TypeError, ValueError):
                    job_idx = None
            if job_idx is None:
                job_idx = extract_sample_job_idx(sample, rows_per_trial)
            if job_idx is None:
                continue
            timestamp = sample_resume_timestamp(sample, source_mtime)
            if record_timestamp is not None:
                try:
                    timestamp = float(record_timestamp)
                except (TypeError, ValueError):
                    pass
            records.append((job_idx, sample, timestamp))
    return records


def assistant_solution_from_export_record(record: dict[str, Any]) -> str:
    reward = record.get("reward")
    if isinstance(reward, dict) and reward.get("extracted_answer") is not None:
        return str(reward.get("extracted_answer") or "")
    conversation = record.get("conversation") or []
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            if content.get("answer") is not None:
                return str(content.get("answer") or "")
            if content.get("text") is not None:
                return str(content.get("text") or "")
            return json.dumps(json_safe(content), ensure_ascii=False)
        if content is None:
            return ""
        return str(content)
    return ""


def sample_from_exported_conversation(
    path: Path,
    *,
    rows_by_index: dict[int, dict[str, Any]],
    rows_per_trial: int,
) -> tuple[int, dict[str, Any], float] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to parse exported conversation for resume {path}: {exc}", flush=True)
        return None
    if not isinstance(record, dict):
        return None
    job = record.get("job") or {}
    if not isinstance(job, dict):
        return None
    try:
        sample_index = int(job.get("trajectory_sample_index"))
        trial_idx = int(job.get("rollout_n", 0))
    except (TypeError, ValueError):
        return None
    row = rows_by_index.get(sample_index, {})
    reward = record.get("reward") if isinstance(record.get("reward"), dict) else {}
    score = reward.get("score") if isinstance(reward.get("score"), dict) else None
    status = record.get("status") if isinstance(record.get("status"), dict) else {}
    parameters = record.get("parameters") if isinstance(record.get("parameters"), dict) else {}
    loop_parameters = parameters.get("loop") if isinstance(parameters.get("loop"), dict) else {}
    lengths = loop_parameters.get("lengths") if isinstance(loop_parameters.get("lengths"), dict) else {}
    timing = loop_parameters.get("timing") if isinstance(loop_parameters.get("timing"), dict) else {}
    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    if ground_truth is None:
        ground_truth = reward.get("ground_truth")
    extra_info = dict(row.get("extra_info") or {})
    extra_info.update(record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {})
    extra_info["conversation_export_json_path"] = str(path)
    failure_reasons = status.get("final_failure_reasons") or reward.get("failure_reasons")
    if not failure_reasons and record.get("failures"):
        failure_reasons = [
            str((failure or {}).get("error_type") or (failure or {}).get("kind") or failure)
            for failure in record.get("failures") or []
        ]
    sample = {
        "sample_index": sample_index,
        "trial_idx": trial_idx,
        "uid": str(row.get("uid") or extra_info.get("question_id") or sample_index),
        "data_source": row.get("data_source") or reward.get("data_source"),
        "ground_truth": ground_truth,
        "solution_str": assistant_solution_from_export_record(record),
        "extra_info": extra_info,
        "conversation_export_json_path": str(path),
        "response_truncated": None,
        "critical_failure": bool(status.get("critical_failure")),
        "failure_reasons": failure_reasons,
        "num_turns": len(record.get("conversation") or []),
        "wall_time_s": timing.get("conversation_wall_time"),
        "core_inference_time": timing.get("core_inference_time"),
        "core_inference_time_raw": timing.get("core_inference_time_raw"),
        "generate_sequences": timing.get("generate_sequences"),
        "tool_parsing": timing.get("tool_parsing"),
        "tool_calls": timing.get("tool_calls"),
        "conversation_wall_time": timing.get("conversation_wall_time"),
        "prompt_tokens": lengths.get("prompt_tokens"),
        "response_tokens_total": lengths.get("response_tokens_total"),
        "response_tokens_generated": lengths.get("response_tokens_generated"),
        "response_tokens_tool": lengths.get("response_tokens_tool"),
        "is_not_answerable": question_type_contains_not_answerable(extra_info.get("question_type"))
        or ground_truth_is_not_answerable(ground_truth),
    }
    if score is not None:
        sample["score"] = score
    job_idx = job_index(sample_index, trial_idx, rows_per_trial)
    return job_idx, attach_resume_metadata(sample, job_idx=job_idx, source="exported_conversation", written_at=path.stat().st_mtime), path.stat().st_mtime


def load_resume_samples(
    output_dir: Path,
    *,
    rows_per_trial: int,
    rows_by_index: dict[int, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Load latest known sample per job from final samples and worker checkpoint shards."""
    candidates: dict[int, tuple[tuple[float, int, int], dict[str, Any]]] = {}
    sequence = 0
    paths: list[tuple[int, Path]] = []
    if rows_by_index is not None:
        export_dir = output_dir / "exported_conversations"
        if export_dir.exists():
            for path in sorted(export_dir.glob("*.json")):
                reconstructed = sample_from_exported_conversation(
                    path,
                    rows_by_index=rows_by_index,
                    rows_per_trial=rows_per_trial,
                )
                if reconstructed is None:
                    continue
                job_idx, sample, timestamp = reconstructed
                sequence += 1
                candidates[job_idx] = ((float(timestamp), 1, sequence), sample)
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        paths.extend((1, path) for path in sorted(checkpoint_dir.glob("worker_*.jsonl")))
    # Older/alternate checkpoint names are accepted to make resume forgiving.
    paths.extend((1, path) for path in (output_dir / "samples.checkpoint.jsonl",) if path.exists())
    paths.extend((2, path) for path in (output_dir / "samples.jsonl",) if path.exists())
    for source_rank, path in paths:
        for job_idx, sample, timestamp in iter_resume_record_samples(path, rows_per_trial=rows_per_trial):
            sequence += 1
            order_key = (float(timestamp), int(source_rank), sequence)
            previous = candidates.get(job_idx)
            if previous is None or order_key >= previous[0]:
                candidates[job_idx] = (order_key, attach_resume_metadata(sample, job_idx=job_idx, source=f"loaded:{path.name}"))
    return {job_idx: sample for job_idx, (_, sample) in candidates.items()}


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


def sample_is_failed(sample: dict[str, Any]) -> bool:
    return bool(sample_failure_reasons(sample))


def should_rerun_existing_sample(sample: dict[str, Any], filters: list[str] | None) -> bool:
    reasons = sample_failure_reasons(sample)
    if not reasons:
        return False
    if not filters:
        return True
    return any(reason.startswith(prefix) for reason in reasons for prefix in filters)


def sample_has_score(sample: dict[str, Any]) -> bool:
    score = sample.get("score")
    return isinstance(score, dict) and bool(score)


def parse_resume_fail_reason_filters(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"none", "null"}:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = parse_list_arg(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--resume-fail-reason-filters must be a JSON/list of strings")
    return parsed


def append_checkpoint_records(checkpoint_path: Path, records: list[tuple[int, dict[str, Any]]]) -> None:
    if not records:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for job_idx, sample in records:
            written_at = time.time()
            sample = attach_resume_metadata(sample, job_idx=job_idx, source="worker_checkpoint", written_at=written_at)
            envelope = {
                "job_idx": int(job_idx),
                "job_key": sample["job_key"],
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
            job_idx = int(sample.get("job_idx") if sample.get("job_idx") is not None else 0)
            sample = attach_resume_metadata(sample, job_idx=job_idx, source="final", written_at=finalized_at)
            sample["resume_metadata"]["finalized_at"] = finalized_at
            f.write(json.dumps(json_safe(sample), ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def question_type_contains_not_answerable(question_type: Any) -> bool:
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def ground_truth_is_not_answerable(ground_truth: Any) -> bool:
    return isinstance(ground_truth, str) and "not answerable" in ground_truth.lower()


def build_summary_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    data_sources = np.array([sample["data_source"] for sample in samples], dtype=object)
    uids = [sample["uid"] for sample in samples]
    reward_extra_infos: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        score = sample.get("score") or {}
        reward_extra_infos["reward"].append(score.get("score"))
        for key, value in score.items():
            reward_extra_infos[key].append(value)
        reward_extra_infos["response_truncated"].append(sample.get("response_truncated", False))
        reward_extra_infos["is_not_answerable"].append(sample.get("is_not_answerable", False))
        reward_extra_infos["critical_failure"].append(sample.get("critical_failure"))

    validation_metrics = process_validation_metrics(data_sources, uids, reward_extra_infos)
    if "accuracy_reward" in reward_extra_infos and (
        "format_reward" in reward_extra_infos or "has_answer" in reward_extra_infos
    ):
        vsearch_metrics = process_vsearch_validation_metrics(data_sources, reward_extra_infos)
    else:
        vsearch_metrics = {}

    summary: dict[str, Any] = {
        "validation_metrics": json_safe(validation_metrics),
        "vsearch_metrics": json_safe(vsearch_metrics),
        "by_data_source": {},
    }

    numeric_keys = [
        "core_inference_time",
        "core_inference_time_raw",
        "generate_sequences",
        "tool_parsing",
        "tool_calls",
        "conversation_wall_time",
        "prompt_tokens",
        "response_tokens_total",
        "response_tokens_generated",
        "response_tokens_tool",
        "n_valid_tool_calls",
        "accuracy_reward",
        "score",
    ]
    for data_source in sorted(set(data_sources.tolist())):
        ds_samples = [sample for sample in samples if sample["data_source"] == data_source]
        ds_summary: dict[str, Any] = {"n": len(ds_samples)}
        for key in numeric_keys:
            values = []
            for sample in ds_samples:
                if key in {"n_valid_tool_calls", "accuracy_reward", "score"}:
                    value = (sample.get("score") or {}).get(key)
                else:
                    value = sample.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    values.append(value)
            if values:
                ds_summary[f"{key}_mean"] = statistics.fmean(values)
        summary["by_data_source"][data_source] = ds_summary
    return summary


def compute_worker_concurrency(args: argparse.Namespace, n_workers: int | None = None) -> int:
    worker_concurrency = int(args.worker_concurrency)
    if worker_concurrency <= 0:
        divisor = n_workers if n_workers and n_workers > 0 else max(1, int(args.agent_worker_processes))
        worker_concurrency = max(1, math.ceil(int(args.concurrency) / divisor))
    return worker_concurrency


async def build_process_agent_runner_components(
    args_dict: dict[str, Any],
) -> tuple[argparse.Namespace, InSightQwenAgentRunner, Any, dict[str, Any]]:
    args = argparse.Namespace(**args_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    runtime_processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if args.custom_chat_template_file:
        template = Path(args.custom_chat_template_file).read_text(encoding="utf-8")
        tokenizer.chat_template = template
        runtime_processor.chat_template = template

    agent_settings = load_agent_settings(args.agent_config, args.agent_config_name)
    tool_schemas = build_tool_schemas(parse_list_arg(args.qwen_tool_list))
    core_config = build_core_config(args, agent_settings, tool_schemas)
    args._core_config_for_export = dict(core_config.__dict__)
    sampling_params = build_sampling_params(args)

    if args.generation_backend == "ray_vllm":
        endpoint_pool = RayVLLMEndpointPool(connect_ray_vllm_servers(args))
    elif args.generation_backend == "https_openai_chat":
        if not args.https_base_url:
            raise ValueError("--https-base-url or OPENAI_BASE_URL is required for --generation-backend=https_openai_chat")
        endpoint_pool = OpenAIChatEndpointPool(
            base_url=args.https_base_url,
            api_key=resolve_https_api_key(args),
            model=args.https_model or args.model_path,
            tokenizer=tokenizer,
            timeout=float(args.https_timeout),
            max_retries=int(args.https_max_retries),
            image_format=args.https_image_format,
            image_detail=optional_https_string(args.https_image_detail),
            reasoning_effort=optional_https_string(args.https_reasoning_effort),
            send_tool_schema=bool(args.https_send_tool_schema),
            coerce_tool_role_to_user=bool(args.https_coerce_tool_role_to_user),
        )
    else:
        raise ValueError(f"unsupported generation backend: {args.generation_backend}")

    tool_parser = ToolParser.get_tool_parser(args.tool_parser, tokenizer)

    async def extract_tool_calls(response_ids: list[int]) -> list[CoreFunctionCall]:
        _, calls = await tool_parser.extract_tool_calls(response_ids)
        return [CoreFunctionCall(name=call.name, arguments=call.arguments) for call in calls]

    runtime = StandaloneInSightRuntime(
        tokenizer=tokenizer,
        processor=runtime_processor,
        endpoint_pool=endpoint_pool,
        tool_call_extractor=extract_tool_calls,
        apply_chat_template_kwargs={"max_tool_calls": args.max_user_turns},
        processor_concurrency=args.processor_concurrency,
    )
    runner = InSightQwenAgentRunner(core_config, runtime)
    return args, runner, tokenizer, sampling_params


def run_agent_global_queue_worker_process(
    *,
    args_dict: dict[str, Any],
    job_queue: Any,
    export_dir: str,
    checkpoint_dir: str | None,
    worker_idx: int,
) -> list[tuple[int, dict[str, Any]]]:
    return asyncio.run(
        run_agent_global_queue_worker_process_async(
            args_dict=args_dict,
            job_queue=job_queue,
            export_dir=export_dir,
            checkpoint_dir=checkpoint_dir,
            worker_idx=worker_idx,
        )
    )


async def run_agent_global_queue_worker_process_async(
    *,
    args_dict: dict[str, Any],
    job_queue: Any,
    export_dir: str,
    checkpoint_dir: str | None,
    worker_idx: int,
) -> list[tuple[int, dict[str, Any]]]:
    args, runner, tokenizer, sampling_params = await build_process_agent_runner_components(args_dict)
    worker_concurrency = int(getattr(args, "_resolved_worker_concurrency", 0) or compute_worker_concurrency(args))
    print(
        f"agent worker {worker_idx}: global_queue local_concurrency={worker_concurrency} "
        f"processor_concurrency={args.processor_concurrency}",
        flush=True,
    )
    results: list[tuple[int, dict[str, Any]]] = []
    results_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0
    checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
    checkpoint_path = Path(checkpoint_dir) / f"worker_{worker_idx}.jsonl" if checkpoint_dir and checkpoint_every > 0 else None
    checkpoint_buffer: list[tuple[int, dict[str, Any]]] = []
    checkpoint_lock = asyncio.Lock()

    async def flush_checkpoint(force: bool = False) -> None:
        if checkpoint_path is None:
            return
        async with checkpoint_lock:
            if not checkpoint_buffer:
                return
            if not force and len(checkpoint_buffer) < checkpoint_every:
                return
            records = list(checkpoint_buffer)
            checkpoint_buffer.clear()
            await asyncio.to_thread(append_checkpoint_records, checkpoint_path, records)

    async def consume(slot_idx: int) -> None:
        nonlocal completed
        while True:
            item = await asyncio.to_thread(job_queue.get)
            if item is None:
                return
            job_idx, sample_idx, trial_idx, row = item
            extra_info = dict(row.get("extra_info") or {})
            extra_info.setdefault("question_id", f"sample-{sample_idx}")
            conversation_export_id = make_export_id({**row, "extra_info": extra_info}, sample_idx, trial_idx)
            started = time.perf_counter()
            result = await runner.run(
                dict(sampling_params),
                raw_prompt=row["raw_prompt"],
                extra_info=extra_info,
                tools_kwargs=row.get("tools_kwargs", {}),
                validate=True,
                conversation_export_id=conversation_export_id,
            )
            sample = build_generated_sample_record(
                result=result,
                row=row,
                args=args,
                sampling_params=sampling_params,
                export_dir=Path(export_dir),
                sample_index=sample_idx,
                trial_idx=trial_idx,
                tokenizer=tokenizer,
                started=started,
            )
            sample = attach_resume_metadata(sample, job_idx=job_idx, source="generated")
            async with results_lock:
                results.append((job_idx, sample))
            if checkpoint_path is not None:
                async with checkpoint_lock:
                    checkpoint_buffer.append((job_idx, sample))
                await flush_checkpoint(force=False)
            async with progress_lock:
                completed += 1
                if completed % max(1, args.progress_every) == 0:
                    print(f"worker {worker_idx} global_queue generated {completed} samples", flush=True)

    await asyncio.gather(*(consume(slot_idx) for slot_idx in range(worker_concurrency)))
    await flush_checkpoint(force=True)
    print(f"worker {worker_idx} global_queue complete: {len(results)} samples", flush=True)
    return results


async def run_eval(args: argparse.Namespace) -> None:
    eval_t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    export_dir = output_dir / "exported_conversations"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset_processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if args.custom_chat_template_file:
        template = Path(args.custom_chat_template_file).read_text(encoding="utf-8")
        tokenizer.chat_template = template
        dataset_processor.chat_template = template

    dataset_processor_metadata_before_dataset = describe_processor(dataset_processor)
    print(
        "standalone dataset processor before dataset: "
        f"{json.dumps(json_safe(dataset_processor_metadata_before_dataset), ensure_ascii=False)}"
    )

    agent_settings = load_agent_settings(args.agent_config, args.agent_config_name)
    tool_schemas = build_tool_schemas(parse_list_arg(args.qwen_tool_list))
    core_config = build_core_config(args, agent_settings, tool_schemas)
    args._core_config_for_export = dict(core_config.__dict__)
    validation_image_token_reorder_settings = build_validation_image_token_reorder_settings(args, core_config)
    args._validation_image_token_reorder_settings = validation_image_token_reorder_settings
    if validation_image_token_reorder_settings:
        print(
            "standalone validation image-token reordering enabled: "
            f"{json.dumps(json_safe(validation_image_token_reorder_settings), ensure_ascii=False)}",
            flush=True,
        )
    sampling_params = build_sampling_params(args)

    ray_server_handles: list[Any] = []
    if args.generation_backend == "ray_vllm":
        ray_server_handles, _, server_metadata, _ = await launch_ray_vllm_servers(args)
    elif args.generation_backend == "https_openai_chat":
        if not args.https_base_url:
            raise ValueError("--https-base-url or OPENAI_BASE_URL is required for --generation-backend=https_openai_chat")
        server_metadata = build_https_server_metadata(args)
    else:
        raise ValueError(f"unsupported generation backend: {args.generation_backend}")

    dataset = RLHFDataset(
        data_files=parse_list_arg(args.val_files),
        tokenizer=tokenizer,
        processor=dataset_processor,
        config=build_dataset_config(args),
        max_samples=args.max_samples,
    )
    dataset_processor_metadata_after_dataset = describe_processor(dataset_processor)
    print(
        "standalone dataset processor after dataset: "
        f"{json.dumps(json_safe(dataset_processor_metadata_after_dataset), ensure_ascii=False)}"
    )

    reward_kwargs = build_reward_kwargs(args)
    rows = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        if "uid" not in row:
            row["uid"] = str((row.get("extra_info") or {}).get("question_id") or idx)
        rows.append(row)

    jobs = [(sample_idx, trial_idx) for trial_idx in range(args.num_trials) for sample_idx in range(len(rows))]
    samples: list[dict[str, Any] | None] = [None for _ in jobs]
    indexed_jobs = [(job_idx, sample_idx, trial_idx) for job_idx, (sample_idx, trial_idx) in enumerate(jobs)]
    rows_by_index = {idx: row for idx, row in enumerate(rows)}
    queued_jobs = list(indexed_jobs)
    resume_enabled = bool(args.resume or args.resume_failed_only)
    resume_existing_count = 0
    resume_reused_count = 0
    resume_rerun_failed_count = 0
    resume_missing_count = 0
    resume_fail_reason_filters = parse_resume_fail_reason_filters(args.resume_fail_reason_filters)
    if resume_enabled:
        existing_samples = load_resume_samples(output_dir, rows_per_trial=len(rows), rows_by_index=rows_by_index)
        resume_existing_count = len(existing_samples)
        queued_jobs = []
        for job_idx, sample_idx, trial_idx in indexed_jobs:
            existing_sample = existing_samples.get(job_idx)
            if existing_sample is None:
                resume_missing_count += 1
                queued_jobs.append((job_idx, sample_idx, trial_idx))
                continue
            if args.resume_failed_only and should_rerun_existing_sample(existing_sample, resume_fail_reason_filters):
                resume_rerun_failed_count += 1
                queued_jobs.append((job_idx, sample_idx, trial_idx))
                continue
            samples[job_idx] = attach_resume_metadata(existing_sample, job_idx=job_idx, source="resume_reused")
            resume_reused_count += 1
        print(
            "standalone resume: "
            f"loaded_existing={resume_existing_count} reused={resume_reused_count} "
            f"queued_missing={resume_missing_count} queued_failed={resume_rerun_failed_count} "
            f"queued_total={len(queued_jobs)} filters={resume_fail_reason_filters or '<all failures>'}",
            flush=True,
        )
    resolved_agent_worker_processes = min(max(1, args.agent_worker_processes), len(queued_jobs)) if queued_jobs else 0
    resolved_worker_concurrency = (
        compute_worker_concurrency(args, resolved_agent_worker_processes) if resolved_agent_worker_processes else 0
    )

    generation_t0 = time.perf_counter()
    if queued_jobs:
        worker_args = vars(args).copy()
        worker_args["_resolved_worker_concurrency"] = resolved_worker_concurrency
        print(
            f"using {resolved_agent_worker_processes} global-queue agent worker processes for {len(queued_jobs)} jobs "
            f"(global_concurrency={args.concurrency}, worker_concurrency={args.worker_concurrency or 'auto'})",
            flush=True,
        )
        process_pool_kwargs: dict[str, Any] = {
            "max_workers": resolved_agent_worker_processes,
            "mp_context": mp.get_context("spawn"),
        }
        manager_context = mp.get_context("spawn")
        checkpoint_dir = str(output_dir / "checkpoints") if int(args.checkpoint_every or 0) > 0 else None
        with manager_context.Manager() as manager:
            job_queue = manager.Queue()
            for job_idx, sample_idx, trial_idx in queued_jobs:
                job_queue.put((job_idx, sample_idx, trial_idx, rows_by_index[sample_idx]))
            for _ in range(resolved_agent_worker_processes * resolved_worker_concurrency):
                job_queue.put(None)
            with ProcessPoolExecutor(**process_pool_kwargs) as executor:
                futures = [
                    executor.submit(
                        run_agent_global_queue_worker_process,
                        args_dict=worker_args,
                        job_queue=job_queue,
                        export_dir=str(export_dir),
                        checkpoint_dir=checkpoint_dir,
                        worker_idx=worker_idx,
                    )
                    for worker_idx in range(resolved_agent_worker_processes)
                ]
                completed = 0
                for future in as_completed(futures):
                    worker_results = future.result()
                    for job_idx, sample in worker_results:
                        samples[job_idx] = sample
                    completed += len(worker_results)
                    print(f"generated {completed}/{len(queued_jobs)} queued samples across worker processes", flush=True)
    else:
        print("standalone generation: no queued jobs; using existing resumed samples", flush=True)
    generation_t1 = time.perf_counter()

    materialized_samples = [sample for sample in samples if sample is not None]

    scoring_t0 = time.perf_counter()
    if not args.skip_reward:
        samples_to_score = [
            sample
            for sample in materialized_samples
            if args.rescore_existing or not sample_has_score(sample)
        ]
        if samples_to_score:
            score_dicts = await asyncio.to_thread(
                compute_score_batch,
                data_sources=[sample["data_source"] for sample in samples_to_score],
                solution_strs=[sample["solution_str"] for sample in samples_to_score],
                ground_truths=[sample["ground_truth"] for sample in samples_to_score],
                extra_infos=[sample["extra_info"] for sample in samples_to_score],
                **reward_kwargs,
            )
            for sample, score in zip(samples_to_score, score_dicts, strict=True):
                sample["score"] = score
    else:
        for sample in materialized_samples:
            sample["score"] = {}
    scoring_t1 = time.perf_counter()

    write_t0 = time.perf_counter()

    write_samples_jsonl_atomic(output_dir / "samples.jsonl", materialized_samples)

    summary = build_summary_metrics(materialized_samples) if materialized_samples else {}
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
    manifest = {
        "run_id": uuid.uuid4().hex,
        "model_path": args.model_path,
        "val_files": parse_list_arg(args.val_files),
        "agent_config": args.agent_config,
        "agent_settings": agent_settings,
        "core_config": core_config.__dict__,
        "sampling_params": sampling_params,
        "reward_kwargs": reward_kwargs,
        "backend": f"{args.generation_backend}_global_queue",
        "server_metadata": server_metadata,
        "eval_config": {
            "path": str(Path(args.eval_config).resolve()) if args.eval_config else None,
            "sha256": getattr(args, "_eval_config_sha256", None),
            "data": getattr(args, "_eval_config_data", None),
        },
        "export_metadata": {
            "global_step": args.global_step,
            "split": args.split,
            "validate": bool(args.export_validate),
            "run_name": args.run_name,
            "trial_name": args.trial_name,
        },
        "processor_metadata": {
            "dataset_before_dataset": dataset_processor_metadata_before_dataset,
            "dataset_after_dataset": dataset_processor_metadata_after_dataset,
            "runtime_processor_concurrency": args.processor_concurrency,
        },
        "validation_image_token_reorder_settings": validation_image_token_reorder_settings,
        "standalone_parallelism": {
            "concurrency": args.concurrency,
            "agent_worker_processes": args.agent_worker_processes,
            "resolved_agent_worker_processes": resolved_agent_worker_processes,
            "worker_concurrency": args.worker_concurrency,
            "resolved_worker_concurrency": resolved_worker_concurrency,
            "queue_policy": "global_queue",
            "processor_concurrency_per_process": args.processor_concurrency,
        },
        "wall_times": wall_times,
        "num_samples": len(materialized_samples),
        "num_expected_samples": len(indexed_jobs),
        "num_queued_samples": len(queued_jobs),
        "num_trials": args.num_trials,
        "skip_reward": args.skip_reward,
        "resume": {
            "enabled": resume_enabled,
            "resume_failed_only": bool(args.resume_failed_only),
            "loaded_existing": resume_existing_count,
            "reused_existing": resume_reused_count,
            "queued_missing": resume_missing_count,
            "queued_failed": resume_rerun_failed_count,
            "fail_reason_filters": resume_fail_reason_filters,
            "checkpoint_every": args.checkpoint_every,
            "rescore_existing": args.rescore_existing,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "done").touch()
    if ray_server_handles:
        cleanup_ray_vllm_servers(ray_server_handles)
    print(f"standalone eval complete: {len(materialized_samples)} samples -> {output_dir}")


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--eval-config")
    pre_args, _ = pre_parser.parse_known_args()
    eval_config_data = load_structured_config(pre_args.eval_config)
    eval_config_sha256 = sha256_file(pre_args.eval_config) if pre_args.eval_config else None
    defaults = build_arg_defaults_from_eval_config(eval_config_data)

    def default(name: str, fallback: Any = None) -> Any:
        return defaults.get(name, fallback)

    def default_bool(name: str, fallback: bool = False) -> bool:
        value = default(name, fallback)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    def required_without_default(name: str) -> bool:
        value = defaults.get(name)
        return value is None or value == ""

    parser = argparse.ArgumentParser(
        description="Standalone InSight Qwen agent evaluation with global-queue workers and Ray/HTTPS generation.",
        parents=[pre_parser],
    )
    parser.add_argument("--val-files", default=default("val_files"), required=required_without_default("val_files"))
    parser.add_argument("--model-path", default=default("model_path"), required=required_without_default("model_path"))
    parser.add_argument("--output-dir", default=default("output_dir"), required=required_without_default("output_dir"))
    parser.add_argument(
        "--agent-config",
        default=default(
            "agent_config",
            "recipe/vsearch/config/agent_insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml",
        ),
    )
    parser.add_argument("--agent-config-name", default=default("agent_config_name", "insight_qwen_agent_core"))
    parser.add_argument("--custom-chat-template-file", default=default("custom_chat_template_file"))
    parser.add_argument("--cache-dir", default=default("cache_dir", os.path.expanduser("~/.cache/verl/rlhf")))
    parser.add_argument("--max-samples", type=int, default=default("max_samples", -1))
    parser.add_argument("--num-trials", type=int, default=default("num_trials", 1))
    parser.add_argument(
        "--generation-backend",
        choices=["ray_vllm", "https_openai_chat"],
        default=default("generation_backend", "ray_vllm"),
        help="Generation transport. ray_vllm is the aligned local backend; https_openai_chat uses /v1/chat/completions.",
    )
    parser.add_argument("--https-base-url", default=default("https_base_url", os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--https-model", default=default("https_model"))
    parser.add_argument("--https-api-key", default=default("https_api_key"))
    parser.add_argument("--https-api-key-env", default=default("https_api_key_env", "OPENAI_API_KEY"))
    parser.add_argument("--https-timeout", type=float, default=default("https_timeout", 600.0))
    parser.add_argument("--https-max-retries", type=int, default=default("https_max_retries", 2))
    parser.add_argument("--https-image-format", default=default("https_image_format", "PNG"))
    parser.add_argument(
        "--https-image-detail",
        default=default("https_image_detail", "high"),
        help="OpenAI image_url detail setting for HTTPS generation images. Use none/null/empty to omit.",
    )
    parser.add_argument(
        "--https-reasoning-effort",
        default=default("https_reasoning_effort", "high"),
        help="reasoning_effort for API-model HTTPS generation. Use none/null/empty to omit.",
    )
    parser.add_argument(
        "--https-send-tool-schema",
        action=argparse.BooleanOptionalAction,
        default=default_bool("https_send_tool_schema", True),
        help="Send OpenAI tool schemas to the HTTPS chat endpoint when tool schemas are enabled.",
    )
    parser.add_argument(
        "--https-coerce-tool-role-to-user",
        action=argparse.BooleanOptionalAction,
        default=default_bool("https_coerce_tool_role_to_user", False),
        help="Convert role=tool messages to role=user for stricter OpenAI-compatible providers.",
    )
    parser.add_argument("--concurrency", type=int, default=default("concurrency", 32))
    parser.add_argument("--agent-worker-processes", type=int, default=default("agent_worker_processes", 1))
    parser.add_argument(
        "--worker-concurrency",
        type=int,
        default=default("worker_concurrency", 0),
        help="Per-process agent-loop concurrency. 0 means ceil(concurrency / agent_worker_processes).",
    )
    parser.add_argument("--processor-concurrency", type=int, default=default("processor_concurrency", 8))
    parser.add_argument(
        "--validation-image-token-reorder",
        action=argparse.BooleanOptionalAction,
        default=default_bool("validation_image_token_reorder", False),
        help="Apply verl validation image-token load balancing before standalone generation.",
    )
    parser.add_argument(
        "--validation-reorder-num-workers",
        type=int,
        default=default("validation_reorder_num_workers", 8),
    )
    parser.add_argument(
        "--validation-reorder-batch-size",
        type=int,
        default=default("validation_reorder_batch_size", 32),
    )
    parser.add_argument(
        "--validation-reorder-default-agent-loop",
        default=default("validation_reorder_default_agent_loop", "insight_qwen_agent"),
    )
    parser.add_argument("--ray-address", default=default("ray_address"))
    parser.add_argument("--ray-namespace", default=default("ray_namespace"))
    parser.add_argument("--ray-temp-dir", default=default("ray_temp_dir"))
    parser.add_argument("--ray-num-replicas", type=int, default=default("ray_num_replicas", 4))
    parser.add_argument("--ray-gpus-per-replica", type=int, default=default("ray_gpus_per_replica", 1))
    parser.add_argument("--ray-num-cpus", type=int, default=default("ray_num_cpus"))
    parser.add_argument("--ray-cpus-per-server", type=float, default=default("ray_cpus_per_server", 4.0))
    parser.add_argument("--ray-dtype", default=default("ray_dtype", "bfloat16"))
    parser.add_argument("--ray-load-format", default=default("ray_load_format", "auto"))
    parser.add_argument("--ray-max-num-seqs", type=int, default=default("ray_max_num_seqs", 1024))
    parser.add_argument("--ray-max-num-batched-tokens", type=int, default=default("ray_max_num_batched_tokens", 32768))
    parser.add_argument("--ray-gpu-memory-utilization", type=float, default=default("ray_gpu_memory_utilization", 0.9))
    parser.add_argument(
        "--ray-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=default_bool("ray_enable_prefix_caching", True),
    )
    parser.add_argument(
        "--ray-enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=default_bool("ray_enable_chunked_prefill", True),
    )
    parser.add_argument(
        "--ray-enable-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=default_bool("ray_enable_sleep_mode", True),
    )
    parser.add_argument(
        "--ray-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=default_bool("ray_enforce_eager", True),
    )
    parser.add_argument("--ray-scheduling-policy", default=default("ray_scheduling_policy", "fcfs"))
    parser.add_argument(
        "--ray-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=default_bool("ray_trust_remote_code", False),
    )
    parser.add_argument("--progress-every", type=int, default=default("progress_every", 25))
    parser.add_argument(
        "--resume",
        action="store_true",
        default=default_bool("resume", False),
        help="Resume from existing final/checkpoint samples and run only missing jobs.",
    )
    parser.add_argument(
        "--resume-failed-only",
        "--resume_failed_only",
        action="store_true",
        default=default_bool("resume_failed_only", False),
        help="Resume and rerun missing plus failed jobs, preserving successful existing jobs.",
    )
    parser.add_argument(
        "--resume-fail-reason-filters",
        "--fail_reason_filters",
        default=default("resume_fail_reason_filters"),
        help="Optional JSON/list of failure-reason prefixes to rerun with --resume-failed-only.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=default("checkpoint_every", 1),
        help="Append worker checkpoint records every N generated samples. 0 disables checkpoints.",
    )
    parser.add_argument(
        "--rescore-existing",
        action=argparse.BooleanOptionalAction,
        default=default_bool("rescore_existing", False),
        help="Recompute judge/reward scores for existing resumed samples instead of reusing stored scores.",
    )
    parser.add_argument("--prompt-length", type=int, default=default("prompt_length", 262144))
    parser.add_argument("--response-length", type=int, default=default("response_length", 15360))
    parser.add_argument("--max-model-len", type=int, default=default("max_model_len", 262144))
    parser.add_argument("--max-pixels", type=int, default=default("max_pixels", 1280 * 1280))
    parser.add_argument("--image-patch-size", type=int, default=default("image_patch_size", 14))
    parser.add_argument("--max-user-turns", type=int, default=default("max_user_turns", 6))
    parser.add_argument("--max-assistant-turns", type=int, default=default("max_assistant_turns", 7))
    parser.add_argument("--max-parallel-calls", type=int, default=default("max_parallel_calls", 1))
    parser.add_argument("--qwen-tool-list", default=default("qwen_tool_list", "[image_zoom_in_tool_qwen3vl]"))
    parser.add_argument("--no-tool-schema", action="store_true", default=bool(default("no_tool_schema", False)))
    parser.add_argument("--tool-parser", default=default("tool_parser", "hermes"))
    parser.add_argument("--reward-agent-name", default=default("reward_agent_name", "insight_qwen_agent"))
    parser.add_argument("--temperature", type=float, default=default("temperature", 0.7))
    parser.add_argument("--top-p", type=float, default=default("top_p", 0.8))
    parser.add_argument("--top-k", type=int, default=default("top_k", 20))
    parser.add_argument("--presence-penalty", type=float, default=default("presence_penalty", 1.5))
    parser.add_argument("--repetition-penalty", type=float, default=default("repetition_penalty", 1.0))
    parser.add_argument("--logprobs", action="store_true", default=bool(default("logprobs", False)))
    parser.add_argument("--skip-reward", action="store_true", default=bool(default("skip_reward", False)))
    parser.add_argument("--judge-model", default=default("judge_model", "gpt-5-nano"))
    parser.add_argument("--fallback-judge-model", default=default("fallback_judge_model"))
    parser.add_argument("--judge-workers", type=int, default=default("judge_workers", 32))
    parser.add_argument("--judge-task-timeout", type=int, default=default("judge_task_timeout", 60))
    parser.add_argument("--judge-min-success-rate", type=float, default=default("judge_min_success_rate", 0.99))
    parser.add_argument("--judge-max-retries", type=int, default=default("judge_max_retries", 10))
    parser.add_argument("--judge-retry-interval", type=int, default=default("judge_retry_interval", 30))
    parser.add_argument("--global-step", default=default("global_step"))
    parser.add_argument("--split", default=default("split", "val"))
    parser.add_argument(
        "--export-validate",
        action=argparse.BooleanOptionalAction,
        default=bool(default("export_validate", True)),
    )
    parser.add_argument("--run-name", default=default("run_name"))
    parser.add_argument("--trial-name", default=default("trial_name"))
    args = parser.parse_args()
    if args.generation_backend == "ray_vllm":
        if args.ray_num_replicas < 1:
            parser.error("--ray-num-replicas must be >= 1")
        if args.ray_gpus_per_replica < 1:
            parser.error("--ray-gpus-per-replica must be >= 1")
    if args.generation_backend == "https_openai_chat" and not args.https_base_url:
        parser.error("--https-base-url or OPENAI_BASE_URL is required for --generation-backend=https_openai_chat")
    if args.agent_worker_processes < 1:
        parser.error("--agent-worker-processes must be >= 1")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")
    try:
        args.resume_fail_reason_filters = parse_resume_fail_reason_filters(args.resume_fail_reason_filters)
    except ValueError as exc:
        parser.error(str(exc))
    if args.resume_failed_only:
        args.resume = True
    args.https_image_format = str(args.https_image_format).upper()
    args._eval_config_data = eval_config_data
    args._eval_config_sha256 = eval_config_sha256
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
