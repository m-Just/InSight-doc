from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from standalone_eval.core.resume import EXPORT_GLOBAL_STEP, EXPORT_SPLIT, EXPORT_VALIDATE
from standalone_eval.core.utils import json_safe
from verl.experimental.agent_loop.qwen_agent_loop import _build_insight_export_conversation
from verl.utils.vreasoner_v2_conversation_export import (
    build_export_record,
    build_repeated_conversation_export_id,
    build_root_conversation_export_id,
    export_conversation,
)


def question_type_contains_not_answerable(question_type: Any) -> bool:
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def ground_truth_is_not_answerable(ground_truth: Any) -> bool:
    return isinstance(ground_truth, str) and "not answerable" in ground_truth.lower()


def make_export_id(row: dict[str, Any], sample_index: int, trial_idx: int) -> str:
    extra_info = dict(row.get("extra_info") or {})
    if not extra_info.get("question_id"):
        extra_info["question_id"] = f"sample-{sample_index}"
    base_id = extra_info.get("conversation_export_base_id")
    if not base_id:
        base_id = build_root_conversation_export_id(
            extra_info=extra_info,
            data_source=row.get("data_source"),
            validate=EXPORT_VALIDATE,
            val_trial_idx=trial_idx,
        )
    return build_repeated_conversation_export_id(str(base_id), 0)


def export_ray_result(
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
    model_config_path = getattr(args, "model_config", None)
    model_config_sha256 = getattr(args, "_model_config_sha256", None)
    record = build_export_record(
        job_id=payload.request_id,
        parent_job_id=None,
        root_job_id=payload.request_id,
        validate=EXPORT_VALIDATE,
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
            "model_config_path": str(Path(model_config_path).resolve()) if model_config_path else None,
            "model_config_sha256": model_config_sha256,
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
            "agent_name": args.agent_name,
        },
        sampling_params=dict(sampling_params),
        tools_kwargs=row.get("tools_kwargs", {}),
        extra_info=payload.extra_info,
        failure_events=payload.failure_events,
        critical_failure=payload.critical_failure,
        final_failure_reasons=payload.final_failure_reasons,
    )
    record["agent_name"] = args.agent_name
    record["conversation"] = _build_insight_export_conversation(payload.messages, initial_question=initial_question)
    record["job"].update(
        {
            "global_step": EXPORT_GLOBAL_STEP,
            "split": EXPORT_SPLIT,
            "validate": EXPORT_VALIDATE,
            "trajectory_sample_index": sample_index,
            "rollout_n": trial_idx,
        }
    )
    index_metadata = {
        "global_step": EXPORT_GLOBAL_STEP,
        "split": EXPORT_SPLIT,
        "validate": EXPORT_VALIDATE,
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


def build_ray_sample_record(
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
    export_path = export_ray_result(
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


def build_rollout_summary_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_data_source": {}}
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
    ]
    data_sources = sorted({sample["data_source"] for sample in samples})
    for data_source in data_sources:
        ds_samples = [sample for sample in samples if sample["data_source"] == data_source]
        ds_summary: dict[str, Any] = {"n": len(ds_samples)}
        for key in numeric_keys:
            values = []
            for sample in ds_samples:
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
