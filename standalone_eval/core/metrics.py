from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from standalone_eval.core.utils import json_safe
from verl.trainer.ppo.metric_utils import process_validation_metrics, process_vsearch_validation_metrics


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
        "n_tool_calls",
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
