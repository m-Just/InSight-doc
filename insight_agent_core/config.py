from __future__ import annotations

import argparse
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .insight_qwen_agent import InSightQwenAgentConfig
from .utils import json_safe


DEFAULT_AGENT_CONFIG_NAME = "insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml"
DEFAULT_CACHE_DIR = "~/.cache/verl/rlhf"
DEFAULT_PROCESSOR_CONCURRENCY = 8
TOOL_NAME_ALIASES = {
    "image_zoom_in_tool_qwen3vl": "image_zoom_in_tool",
}
QWEN_IMAGE_ZOOM_TOOL_SCHEMA = {
    "name": "image_zoom_in_tool",
    "description": (
        "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) "
        "and an optional object label"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                    "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner"
                ),
            },
            "label": {
                "type": "string",
                "description": "The name or label of the object in the specified bounding box",
            },
            "img_idx": {
                "type": "number",
                "description": "The index of the zoomed-in image (starting from 0)",
            },
        },
        "required": ["bbox_2d", "label", "img_idx"],
    },
}


def default_agent_config_path() -> str:
    return str(resources.files("insight_agent_core.configs") / DEFAULT_AGENT_CONFIG_NAME)


def _apply_dotlist_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    base = OmegaConf.create(config)
    dot_conf = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(base, dot_conf)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def _flatten_agent_setting_overrides(overrides: list[str] | None) -> list[str]:
    """Map sectioned YAML override keys onto the flattened runtime settings."""

    if not overrides:
        return []
    section_names = {"tools", "limits", "images", "parser", "turns"}
    flattened: list[str] = []
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            flattened.append(override)
            continue
        parts = key.split(".")
        if len(parts) == 2 and parts[0] in section_names:
            key = parts[1]
        flattened.append(f"{key}={value}")
    return flattened


def _normalize_agent_settings(item: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "name": item.get("name", "insight_qwen_agent_core"),
        "qwen_tool_list": item.get("qwen_tool_list", []),
        "max_user_turns": item.get("max_user_turns", 10),
        "max_assistant_turns": item.get("max_assistant_turns", 11),
        "max_parallel_calls": item.get("max_parallel_calls", 1),
        "max_tool_response_length": item.get("max_tool_response_length", 256),
        "tool_response_truncate_side": item.get("tool_response_truncate_side", "middle"),
        "tool_parser": item.get("tool_parser", "qwen3_vl"),
        "initial_rescale": item.get("initial_rescale", 0.25),
        "gpt_image_max_area": item.get("gpt_image_max_area", 1280 * 1280),
        "crop_image_max_area": item.get("crop_image_max_area", 1280 * 1280),
        "initial_input_pixels_lower_bound": item.get("initial_input_pixels_lower_bound", 0),
        "region_zoom_in_factor": item.get("region_zoom_in_factor", 4.0),
    }
    for section in ("tools", "limits", "images", "parser"):
        value = item.get(section)
        if isinstance(value, dict):
            settings.update(value)
    return settings


def apply_agent_settings_to_args(args: argparse.Namespace, agent_settings: dict[str, Any]) -> None:
    args.qwen_tool_list = ",".join(str(item) for item in agent_settings.get("qwen_tool_list", []))
    args.max_user_turns = int(agent_settings.get("max_user_turns", 10))
    args.max_assistant_turns = int(agent_settings.get("max_assistant_turns", 11))
    args.max_parallel_calls = int(agent_settings.get("max_parallel_calls", 1))
    args.max_tool_response_length = int(agent_settings.get("max_tool_response_length", 256))
    args.tool_response_truncate_side = str(agent_settings.get("tool_response_truncate_side", "middle"))
    args.tool_parser = str(agent_settings.get("tool_parser", "qwen3_vl"))
    args.agent_name = str(agent_settings.get("name", "insight_qwen_agent_core"))


def load_agent_settings(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    if path is None:
        path = default_agent_config_path()
    config = OmegaConf.load(path)
    data = OmegaConf.to_container(config, resolve=True)
    if isinstance(data, dict) and "agent" in data:
        item = data["agent"]
    elif isinstance(data, dict) and "custom_agent" in data:
        custom_agents = data.get("custom_agent") or []
        if len(custom_agents) != 1:
            raise ValueError(f"{path} must define exactly one custom_agent entry for standalone rollout")
        item = custom_agents[0]
    elif isinstance(data, dict):
        item = data
    else:
        raise ValueError(f"unsupported agent config format: {path}")
    raw_item = _apply_dotlist_overrides(dict(item), overrides)
    settings = _normalize_agent_settings(raw_item)
    settings = _apply_dotlist_overrides(settings, _flatten_agent_setting_overrides(overrides))
    return settings, str(settings.get("name", "insight_qwen_agent_core"))


def build_tool_schemas(qwen_tool_list: list[str]) -> list[dict[str, Any]]:
    schemas = []
    for registry_tool_name in qwen_tool_list:
        public_tool_name = TOOL_NAME_ALIASES.get(registry_tool_name, registry_tool_name)
        if public_tool_name != "image_zoom_in_tool":
            raise ValueError(
                f"Tool {registry_tool_name!r} is not supported by standalone eval prompt-schema builder. "
                "Supported tools: image_zoom_in_tool_qwen3vl, image_zoom_in_tool"
            )
        if any(schema.get("name") == public_tool_name for schema in schemas):
            raise ValueError(f"Duplicate public tool name {public_tool_name!r} from qwen_tool_list={qwen_tool_list}")
        tool_schema = dict(QWEN_IMAGE_ZOOM_TOOL_SCHEMA)
        tool_schema["name"] = public_tool_name
        schemas.append(tool_schema)
    return schemas


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_prompt_signature(
    args: argparse.Namespace,
    agent_settings: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    *,
    processor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = {
        "schema_version": 1,
        "agent_name": getattr(args, "agent_name", agent_settings.get("name", "insight_qwen_agent")),
        "agent_settings": {
            key: agent_settings.get(key)
            for key in (
                "qwen_tool_list",
                "tool_parser",
                "max_user_turns",
                "max_assistant_turns",
                "max_tool_response_length",
                "tool_response_truncate_side",
                "initial_rescale",
                "gpt_image_max_area",
                "crop_image_max_area",
                "initial_input_pixels_lower_bound",
                "region_zoom_in_factor",
            )
        },
        "tool_schema_format": "qwen_agent_unwrapped_function",
        "tool_schema_sha256": _json_sha256(tool_schemas),
        "tool_schemas": tool_schemas,
        "apply_chat_template_kwargs": {"max_tool_calls": int(getattr(args, "max_user_turns", 10))},
        "prompt_length": getattr(args, "prompt_length", None),
        "custom_chat_template_file": getattr(args, "custom_chat_template_file", None),
        "processor_metadata": processor_metadata or {},
    }
    signature["sha256"] = _json_sha256({k: v for k, v in signature.items() if k != "sha256"})
    return signature


def build_dataset_config(
    args: argparse.Namespace,
    *,
    image_patch_size: int,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "tokenizer": None,
            "processor": None,
            "prompt_key": "prompt",
            "image_key": "images",
            "max_prompt_length": int(args.max_model_len),
            "truncation": "error",
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
            "return_full_prompt": False,
            "trust_remote_code": True,
            "custom_chat_template": None,
            "max_pixels": -1,
            "validation_max_pixels": -1,
            "image_patch_size": int(image_patch_size),
            "filter_overlong_prompts_workers": 1,
            "cache_dir": str(Path(DEFAULT_CACHE_DIR).expanduser()),
            "dataset_loading_config": None,
            "use_shm": False,
            "video_fps": 2.0,
        }
    )


def build_core_config(args: argparse.Namespace, agent_settings: dict[str, Any], tool_schemas: list[dict[str, Any]]):
    return InSightQwenAgentConfig(
        prompt_length=int(args.max_model_len),
        response_length=int(args.response_length),
        max_user_turns=int(args.max_user_turns),
        max_assistant_turns=int(args.max_assistant_turns),
        max_parallel_calls=int(args.max_parallel_calls),
        tool_schemas=tool_schemas or None,
        tool_parser_name=args.tool_parser,
        initial_rescale=float(agent_settings.get("initial_rescale", 0.25)),
        gpt_image_max_area=int(agent_settings.get("gpt_image_max_area", 1280 * 1280)),
        crop_image_max_area=int(agent_settings.get("crop_image_max_area", 1280 * 1280)),
        initial_input_pixels_lower_bound=int(agent_settings.get("initial_input_pixels_lower_bound", 0)),
        region_zoom_in_factor=float(agent_settings.get("region_zoom_in_factor", 4.0)),
        max_tool_response_length=int(agent_settings.get("max_tool_response_length", 256)),
        tool_response_truncate_side=str(agent_settings.get("tool_response_truncate_side", "middle")),
        train_initial_rescale_randomization_prob=0.0,
        train_initial_rescale_randomization_min=float(agent_settings.get("initial_rescale", 0.25)),
        train_initial_rescale_randomization_max=float(agent_settings.get("initial_rescale", 0.25)),
        train_initial_rescale_randomization_text_budget=1024,
        agent_name=args.agent_name,
        prompt_length_estimator=str(getattr(args, "ray_prompt_length_estimator", "tokenized")),
        prompt_length_safety_margin=int(getattr(args, "ray_prompt_length_safety_margin", 0) or 0),
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


def describe_processor(processor: Any) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    return {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__ if image_processor is not None else None,
        "do_resize": getattr(image_processor, "do_resize", None),
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "patch_size": getattr(image_processor, "patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
    }


def infer_processor_image_patch_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if patch_size is None:
        raise ValueError(
            "Cannot infer image_processor.patch_size from processor; standalone eval requires an explicit processor "
            "that exposes image patch size."
        )
    if merge_size is not None:
        try:
            return int(patch_size) * int(merge_size)
        except (TypeError, ValueError):
            pass
    return int(patch_size)


__all__ = [
    "DEFAULT_AGENT_CONFIG_NAME",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_PROCESSOR_CONCURRENCY",
    "QWEN_IMAGE_ZOOM_TOOL_SCHEMA",
    "TOOL_NAME_ALIASES",
    "apply_agent_settings_to_args",
    "build_core_config",
    "build_dataset_config",
    "build_prompt_signature",
    "build_sampling_params",
    "build_tool_schemas",
    "default_agent_config_path",
    "describe_processor",
    "infer_processor_image_patch_size",
    "load_agent_settings",
]
