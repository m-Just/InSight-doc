from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd
from openai._types import NotGiven
from PIL import Image

from insight_agent_core.images import cap_size_by_area, load_prompt_image, resize_dims_by_factor
from insight_agent_core.insight_qwen_agent import resolve_dynamic_initial_rescale
from evals.backends.base import RolloutJob
from evals.backends.openai_https import _image_to_data_url
from evals.core.export import (
    ground_truth_is_not_answerable,
    make_export_id,
    question_type_contains_not_answerable,
)
from evals.core.resume import EXPORT_GLOBAL_STEP, EXPORT_SPLIT, EXPORT_VALIDATE
from evals.core.utils import as_plain_list
from verl.utils.vreasoner_v2_conversation_export import build_export_record, export_conversation


def resolve_https_api_key(args: argparse.Namespace) -> str:
    env_name = args.https_api_key_env or "OPENAI_API_KEY"
    return os.getenv(env_name) or "EMPTY"


def optional_https_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def exportable_openai_option(value: Any) -> Any:
    if isinstance(value, NotGiven):
        return "default"
    return value


def image_ref_value(image_ref: Any) -> Any:
    if isinstance(image_ref, dict):
        return image_ref.get("image") or image_ref.get("image_url") or image_ref.get("url") or image_ref
    return image_ref


def load_https_presented_images(
    row: dict[str, Any],
    *,
    initial_rescale: float,
    max_area: int,
    initial_input_pixels_lower_bound: int,
) -> list[Any]:
    original_images = []
    for image_ref in as_plain_list(row.get("images")):
        image = load_prompt_image(image_ref_value(image_ref))
        if image is None:
            continue
        original_images.append(image)
    actual_rescale = resolve_dynamic_initial_rescale(
        image_sizes=[image.size for image in original_images],
        configured_initial_rescale=initial_rescale,
        total_pixels_lower_bound=initial_input_pixels_lower_bound,
        per_image_max_area=max_area,
    )
    images = []
    for image in original_images:
        if actual_rescale != 1.0:
            image = image.resize(resize_dims_by_factor(image.size, actual_rescale), Image.LANCZOS)
        if max_area:
            capped_size = cap_size_by_area(image.size, max_area)
            if capped_size != image.size:
                image = image.resize(capped_size, Image.LANCZOS)
        images.append(image)
    return images


def append_text_part(parts: list[dict[str, Any]], text: str) -> None:
    if text:
        parts.append({"type": "text", "text": text})


def content_with_image_urls(
    content: Any,
    *,
    image_urls_iter,
    image_detail: str | None,
) -> Any:
    if not isinstance(content, str):
        return content
    if "<image>" not in content:
        return content
    parts: list[dict[str, Any]] = []
    chunks = content.split("<image>")
    append_text_part(parts, chunks[0])
    for chunk in chunks[1:]:
        image_url = next(image_urls_iter, None)
        if image_url is not None:
            image_payload = {"url": image_url}
            if image_detail:
                image_payload["detail"] = image_detail
            parts.append({"type": "image_url", "image_url": image_payload})
        append_text_part(parts, chunk)
    return parts


def build_https_messages_from_parquet_row(
    row: dict[str, Any],
    *,
    images: list[Any],
    image_format: str,
    image_detail: str | None,
) -> list[dict[str, Any]]:
    image_urls = [_image_to_data_url(image, image_format=image_format) for image in images]
    image_urls_iter = iter(image_urls)
    messages = []
    for message in as_plain_list(row.get("prompt")):
        if not isinstance(message, dict):
            continue
        messages.append(
            {
                "role": str(message.get("role", "user")),
                "content": content_with_image_urls(
                    message.get("content", ""),
                    image_urls_iter=image_urls_iter,
                    image_detail=image_detail,
                ),
            }
        )
    remaining_urls = list(image_urls_iter)
    if remaining_urls:
        if not messages:
            messages.append({"role": "user", "content": []})
        last_message = messages[-1]
        content = last_message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]
        image_parts = []
        for image_url in remaining_urls:
            payload = {"url": image_url}
            if image_detail:
                payload["detail"] = image_detail
            image_parts.append({"type": "image_url", "image_url": payload})
        last_message["content"] = image_parts + content
    return messages


def is_context_overflow_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "context length",
            "maximum context",
            "max context",
            "input token",
            "too many tokens",
            "exceeds the model",
            "exceeded context",
        )
    )


def response_usage_value(response: Any, name: str) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def estimate_text_tokens(text: str) -> int:
    words = len(str(text or "").split())
    return int(math.ceil(words / 0.75)) if words else 0


def build_https_export_record(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    sample_index: int,
    trial_idx: int,
    request_id: str,
    messages: list[dict[str, Any]],
    response_text: str,
    images: list[Any],
    timing: dict[str, Any],
    lengths: dict[str, Any],
    failure_reasons: list[str],
    export_dir: Path,
) -> str:
    extra_info = dict(row.get("extra_info") or {})
    extra_info["agent_name"] = args.agent_name
    initial_question = str(extra_info.get("question") or "")
    export_id = make_export_id({**row, "extra_info": extra_info}, sample_index, trial_idx)
    messages_api = [*messages, {"role": "assistant", "content": response_text}]
    record = build_export_record(
        job_id=request_id,
        parent_job_id=None,
        root_job_id=request_id,
        validate=EXPORT_VALIDATE,
        initial_question=initial_question,
        messages_api=messages_api,
        raw_prompt=as_plain_list(row.get("prompt")),
        original_images=images,
        presented_image_refs=[],
        request_params={
            "backend": "https_openai_chat",
            "model": args.model,
            "max_tokens_after_initial_prompt": args.response_length,
        },
        loop_params={
            "implementation": "https_openai_chat_direct",
            "lengths": lengths,
            "timing": timing,
            "agent_name": args.agent_name,
        },
        sampling_params={
            "max_completion_tokens": args.response_length,
            "reasoning_effort": args.https_reasoning_effort,
        },
        tools_kwargs=row.get("tools_kwargs", {}) or {},
        extra_info=extra_info,
        failure_events=[{"kind": "https_generation_failure", "message": reason} for reason in failure_reasons],
        critical_failure=bool(failure_reasons),
        final_failure_reasons=failure_reasons,
    )
    record["agent_name"] = args.agent_name
    record["job"].update(
        {
            "global_step": EXPORT_GLOBAL_STEP,
            "split": EXPORT_SPLIT,
            "validate": EXPORT_VALIDATE,
            "trajectory_sample_index": sample_index,
            "rollout_n": trial_idx,
        }
    )
    return export_conversation(
        str(export_dir),
        record,
        job_id=request_id,
        export_id=export_id,
        index_metadata=record["job"],
    )


async def create_https_client(args: argparse.Namespace):
    from insight_o3.utils import api as insight_api

    if insight_api.log_chat_completion is None:
        try:
            from insight_o3.utils.api_logger import log_chat_completion
        except Exception as exc:
            raise RuntimeError("API logging is required for HTTPS generation but api_logger is unavailable") from exc
        insight_api.log_chat_completion = log_chat_completion
    client = insight_api.create_async_openai_client(
        api_key=resolve_https_api_key(args),
        base_url=args.https_base_url,
        timeout=args.https_timeout,
    )
    if not isinstance(args.https_max_retries, NotGiven):
        client = client.with_options(max_retries=args.https_max_retries)
    return client, insight_api.complete_chat_and_maybe_log


class HTTPSOpenAIChatBackend:
    backend_name = "https_openai_chat_direct"

    def __init__(self, args: argparse.Namespace, agent_settings: dict[str, Any], export_dir: Path):
        self.args = args
        self.agent_settings = agent_settings
        self.export_dir = export_dir
        self.client = None
        self.complete_chat_and_maybe_log = None
        self.server_metadata = self._build_server_metadata()
        self.parallelism: dict[str, Any] = {}

    def _build_server_metadata(self) -> list[dict[str, Any]]:
        args = self.args
        return [
            {
                "endpoint_type": "openai_compatible_https_chat",
                "base_url": args.https_base_url,
                "model": args.model,
                "timeout": exportable_openai_option(args.https_timeout),
                "max_retries": exportable_openai_option(args.https_max_retries),
                "image_format": args.https_image_format,
                "image_detail": optional_https_string(args.https_image_detail),
                "reasoning_effort": optional_https_string(args.https_reasoning_effort),
                "api_stack": "insight_o3.utils.api",
                "api_logging": "enabled_required",
                "api_key_env": args.https_api_key_env,
                "api_key_provided": bool(os.getenv(args.https_api_key_env or "OPENAI_API_KEY")),
            }
        ]

    async def prepare(self) -> None:
        if not self.args.https_base_url:
            raise ValueError("--model-config https_openai_chat.base_url or OPENAI_BASE_URL is required")
        if str(getattr(self.args, "qwen_tool_list", "") or "").strip():
            raise ValueError(
                "https_openai_chat standalone eval is direct no-tool API eval. "
                "Set tools.qwen_tool_list=[] in --agent-config or --agent-config-override for this backend."
            )
        self.client, self.complete_chat_and_maybe_log = await create_https_client(self.args)

    async def load_rows(self, val_files: list[str], max_samples: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for parquet_path in val_files:
            df = pd.read_parquet(parquet_path)
            for file_row_idx, (_, row) in enumerate(df.iterrows()):
                item = row.to_dict()
                item["prompt"] = as_plain_list(item.get("prompt"))
                item["images"] = as_plain_list(item.get("images"))
                item["resume_val_file"] = parquet_path
                item["resume_file_row_idx"] = int(file_row_idx)
                if "uid" not in item:
                    extra_info = item.get("extra_info") or {}
                    item["uid"] = str(extra_info.get("question_id") or len(rows))
                rows.append(item)
        return rows

    def basic_config_extra(self) -> dict[str, Any]:
        return {}

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()

    async def generate_many(
        self,
        jobs: list[RolloutJob],
        on_sample: Callable[[RolloutJob, dict[str, Any]], Awaitable[None]],
    ) -> None:
        semaphore = asyncio.Semaphore(max(1, int(self.args.worker_concurrency)))
        self.parallelism = {
            "worker_concurrency": self.args.worker_concurrency,
            "queue_policy": "https_direct_asyncio",
        }

        async def run_one(job: RolloutJob) -> None:
            async with semaphore:
                sample = await self._generate_sample(job)
            await on_sample(job, sample)

        await asyncio.gather(*(run_one(job) for job in jobs))

    async def _generate_sample(self, job: RolloutJob) -> dict[str, Any]:
        args = self.args
        row = job.row
        request_id = f"https-{uuid.uuid4().hex}"
        initial_rescale = float(
            (row.get("extra_info") or {}).get("initial_rescale")
            if (row.get("extra_info") or {}).get("initial_rescale") is not None
            else self.agent_settings.get("initial_rescale", 1.0)
        )
        max_area = int(self.agent_settings.get("gpt_image_max_area", 1280 * 1280) or 0)
        initial_input_pixels_lower_bound = int(self.agent_settings.get("initial_input_pixels_lower_bound", 0) or 0)
        current_rescale = initial_rescale
        context_retry_index = 0
        failure_reasons: list[str] = []
        response_text = ""
        finish_reason = None
        response = None
        messages: list[dict[str, Any]] = []
        images: list[Any] = []
        started = time.perf_counter()
        request_elapsed = None

        while True:
            images = load_https_presented_images(
                row,
                initial_rescale=current_rescale,
                max_area=max_area,
                initial_input_pixels_lower_bound=initial_input_pixels_lower_bound,
            )
            messages = build_https_messages_from_parquet_row(
                row,
                images=images,
                image_format=args.https_image_format,
                image_detail=args.https_image_detail,
            )
            kwargs = {
                "max_completion_tokens": args.response_length,
                "reasoning_effort": args.https_reasoning_effort,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                request_t0 = time.perf_counter()
                response = await self.complete_chat_and_maybe_log(
                    messages=messages,
                    model=args.model,
                    client=self.client,
                    show_detailed_error_message=True,
                    **kwargs,
                )
                request_elapsed = time.perf_counter() - request_t0
                if not response.choices:
                    failure_reasons = ["empty_response_choices"]
                    break
                choice = response.choices[0]
                message = choice.message
                response_text = str(message.content or "")
                finish_reason = getattr(choice, "finish_reason", None)
                failure_reasons = []
                break
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if is_context_overflow_error_text(error_text) and context_retry_index < args.context_overflow_max_halving_trials:
                    context_retry_index += 1
                    current_rescale /= 2.0
                    print(
                        f"WARNING: HTTPS sample {job.sample_index} exceeded context; retrying with "
                        f"image rescale={current_rescale} "
                        f"(retry {context_retry_index}/{args.context_overflow_max_halving_trials})",
                        flush=True,
                    )
                    continue
                failure_reasons = [error_text]
                break

        elapsed = time.perf_counter() - started
        prompt_tokens = response_usage_value(response, "prompt_tokens") if response is not None else None
        completion_tokens = response_usage_value(response, "completion_tokens") if response is not None else None
        total_tokens = response_usage_value(response, "total_tokens") if response is not None else None
        if completion_tokens is None and response_text:
            completion_tokens = estimate_text_tokens(response_text)
        lengths = {
            "prompt_tokens": prompt_tokens,
            "response_tokens_total": completion_tokens,
            "response_tokens_generated": completion_tokens,
            "response_tokens_tool": 0,
            "total_tokens": total_tokens,
        }
        timing = {
            "core_inference_time": elapsed,
            "core_inference_time_raw": elapsed,
            "conversation_wall_time": elapsed,
            "generate_sequences": elapsed,
            "tool_parsing": 0.0,
            "tool_calls": 0.0,
            "request_elapsed": request_elapsed,
        }
        export_path = build_https_export_record(
            row=row,
            args=args,
            sample_index=job.sample_index,
            trial_idx=job.trial_idx,
            request_id=request_id,
            messages=messages,
            response_text=response_text,
            images=images,
            timing=timing,
            lengths=lengths,
            failure_reasons=failure_reasons,
            export_dir=self.export_dir,
        )
        ground_truth = (row.get("reward_model") or {}).get("ground_truth")
        sample_extra_info = dict(row.get("extra_info") or {})
        sample_extra_info["agent_name"] = args.agent_name
        sample_extra_info["conversation_export_json_path"] = export_path
        return {
            "sample_index": job.sample_index,
            "trial_idx": job.trial_idx,
            "uid": str(row.get("uid") or sample_extra_info.get("question_id") or job.sample_index),
            "data_source": row.get("data_source"),
            "ground_truth": ground_truth,
            "solution_str": response_text,
            "extra_info": sample_extra_info,
            "conversation_export_json_path": export_path,
            "response_truncated": finish_reason == "length",
            "critical_failure": bool(failure_reasons),
            "failure_reasons": failure_reasons,
            "num_turns": len(messages) + (1 if response_text else 0),
            "n_tool_calls": 0,
            "wall_time_s": elapsed,
            "core_inference_time": timing["core_inference_time"],
            "core_inference_time_raw": timing["core_inference_time_raw"],
            "generate_sequences": timing["generate_sequences"],
            "tool_parsing": 0.0,
            "tool_calls": 0.0,
            "conversation_wall_time": elapsed,
            "prompt_tokens": prompt_tokens,
            "response_tokens_total": completion_tokens,
            "response_tokens_generated": completion_tokens,
            "response_tokens_tool": 0,
            "is_not_answerable": question_type_contains_not_answerable(sample_extra_info.get("question_type"))
            or ground_truth_is_not_answerable(ground_truth),
        }
