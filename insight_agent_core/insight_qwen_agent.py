from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from PIL import Image

from .images import (
    PresentedImageState,
    cap_size_by_area,
    clamp_bbox_to_image,
    load_prompt_image,
    presented_image_to_export_ref,
    resize_dims_by_factor,
    scale_bbox_from_qwen_range,
    translate_bbox_to_original,
    validate_qwen_image_aspect_ratio,
)
from .runtime import CoreFunctionCall, CoreRuntime


logger = logging.getLogger(__name__)

INITIAL_PROMPT_SHRINK_AREA_FACTOR = 0.5
INITIAL_PROMPT_SHRINK_DIM_FACTOR = math.sqrt(INITIAL_PROMPT_SHRINK_AREA_FACTOR)
INITIAL_PROMPT_MAX_SHRINK_STEPS = 4


class CoreAgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass
class InSightQwenAgentConfig:
    prompt_length: int | None
    response_length: int
    max_user_turns: int
    max_assistant_turns: int
    max_parallel_calls: int
    tool_schemas: list[dict[str, Any]] | None
    tool_parser_name: str
    initial_rescale: float = 0.25
    gpt_image_max_area: int = 1280 * 1280
    crop_image_max_area: int = 1280 * 1280
    initial_input_pixels_lower_bound: int = 0
    region_zoom_in_factor: float = 4.0
    train_initial_rescale_randomization_prob: float = 0.0
    train_initial_rescale_randomization_min: float = 0.25
    train_initial_rescale_randomization_max: float = 0.25
    train_initial_rescale_randomization_text_budget: int = 1024
    agent_name: str = "insight_qwen_agent"


@dataclass
class CoreToolExecutionResult:
    text_result: str
    images: list[Image.Image] = field(default_factory=list)
    presented_images: list[PresentedImageState] = field(default_factory=list)
    presented_image_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CoreAgentData:
    messages: list[dict[str, Any]]
    image_data: list[Any]
    video_data: list[Any]
    metrics: dict[str, Any]
    request_id: str
    tools_kwargs: dict[str, Any]
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    tool_calls: list[CoreFunctionCall] = field(default_factory=list)
    user_turns: int = 0
    assistant_turns: int = 0
    turn_scores: list[float] = field(default_factory=list)
    tool_rewards: list[float] = field(default_factory=list)
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class InSightQwenExportPayload:
    request_id: str
    conversation_export_id: str
    raw_prompt: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    original_images: list[Image.Image]
    presented_image_refs: list[dict[str, Any]]
    actual_initial_rescale: float
    initial_rescale_metadata: dict[str, Any]
    extra_info: dict[str, Any]
    failure_events: list[dict[str, Any]]
    critical_failure: bool
    final_failure_reasons: list[str] | None


@dataclass
class InSightQwenAgentResult:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    multi_modal_data: dict[str, Any]
    num_turns: int
    metrics: dict[str, Any]
    extra_fields: dict[str, Any]
    export_payload: InSightQwenExportPayload


FallbackToolExecutor = Callable[
    [CoreFunctionCall, CoreAgentData],
    Awaitable[CoreToolExecutionResult],
]


def _record_conversation_wall_time(agent_data: CoreAgentData, start_time: float) -> None:
    conversation_wall_time = time.perf_counter() - start_time
    agent_data.metrics["conversation_wall_time"] = conversation_wall_time
    agent_data.extra_fields["conversation_wall_time"] = conversation_wall_time


def _record_core_inference_metrics(agent_data: CoreAgentData) -> None:
    if "generate_sequences" not in agent_data.metrics:
        agent_data.extra_fields["generate_sequences"] = None
        agent_data.extra_fields["tool_parsing"] = None
        agent_data.extra_fields["tool_calls"] = None
        agent_data.extra_fields["core_inference_time"] = None
        return

    generate_sequences = float(agent_data.metrics.get("generate_sequences", 0.0) or 0.0)
    tool_parsing = float(agent_data.metrics.get("tool_parsing", 0.0) or 0.0)
    tool_calls = float(agent_data.metrics.get("tool_calls", 0.0) or 0.0)
    core_inference_time_raw = generate_sequences + tool_parsing + tool_calls
    agent_data.metrics["core_inference_time_raw"] = core_inference_time_raw
    agent_data.metrics["core_inference_time"] = core_inference_time_raw
    agent_data.extra_fields["generate_sequences"] = generate_sequences
    agent_data.extra_fields["tool_parsing"] = tool_parsing
    agent_data.extra_fields["tool_calls"] = tool_calls
    agent_data.extra_fields["core_inference_time_raw"] = core_inference_time_raw
    agent_data.extra_fields["core_inference_time"] = core_inference_time_raw


def _timer(metrics: dict[str, Any], key: str):
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            metrics[key] = metrics.get(key, 0.0) + time.perf_counter() - self.start

    return _Timer()


def resolve_dynamic_initial_rescale(
    image_sizes: list[tuple[int, int]],
    configured_initial_rescale: float,
    total_pixels_lower_bound: int,
    per_image_max_area: int,
) -> float:
    if configured_initial_rescale <= 0:
        raise ValueError(f"configured_initial_rescale must be positive, got {configured_initial_rescale}")
    if total_pixels_lower_bound <= 0 or not image_sizes:
        return min(configured_initial_rescale, 1.0)

    base_rescale = min(configured_initial_rescale, 1.0)

    def total_pixels_after_rescale(factor: float) -> int:
        total = 0
        for size in image_sizes:
            resized = resize_dims_by_factor(size, factor)
            capped = cap_size_by_area(resized, per_image_max_area)
            total += capped[0] * capped[1]
        return total

    if total_pixels_after_rescale(base_rescale) >= total_pixels_lower_bound:
        return base_rescale
    if total_pixels_after_rescale(1.0) <= total_pixels_lower_bound:
        return 1.0

    lo = base_rescale
    hi = 1.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if total_pixels_after_rescale(mid) >= total_pixels_lower_bound:
            hi = mid
        else:
            lo = mid
    return hi


class InSightQwenAgentRunner:
    """Token-level InSight Qwen agent runner with no verl dependency."""

    def __init__(
        self,
        config: InSightQwenAgentConfig,
        runtime: CoreRuntime,
        fallback_tool_executor: FallbackToolExecutor | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.fallback_tool_executor = fallback_tool_executor

    async def run(
        self,
        sampling_params: dict[str, Any],
        *,
        raw_prompt: list[dict[str, Any]],
        extra_info: dict[str, Any] | None = None,
        tools_kwargs: dict[str, Any] | None = None,
        validate: bool = False,
        conversation_export_id: str | None = None,
    ) -> InSightQwenAgentResult:
        conversation_wall_time_start = time.perf_counter()
        messages = copy.deepcopy(list(raw_prompt))
        extra_info = dict(extra_info or {})
        sample_initial_rescale = self.get_sample_initial_rescale(extra_info)
        aligned_prompt, original_images, presented_images, actual_initial_rescale, initial_rescale_metadata = (
            self.build_presented_prompt(
                messages,
                images_are_presented=False,
                training_mode=not validate,
                sample_initial_rescale=sample_initial_rescale,
            )
        )

        multi_modal_data = await self.runtime.process_vision_info(aligned_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        initial_prompt_fit_start = time.perf_counter()
        images, videos, initial_prompt_fit_metadata = await self.fit_initial_prompt_to_prompt_length(
            aligned_prompt,
            presented_images,
            images,
            videos,
        )
        initial_prompt_fit_time = time.perf_counter() - initial_prompt_fit_start
        initial_rescale_metadata.update(initial_prompt_fit_metadata)

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        export_id = conversation_export_id or extra_info.get("conversation_export_id") or request_id
        agent_data = CoreAgentData(
            messages=aligned_prompt,
            image_data=images if images else [],
            video_data=videos if videos else [],
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs or {},
        )
        agent_data.extra_fields["response_truncated"] = False
        agent_data.extra_fields["initial_prompt_tokens"] = initial_prompt_fit_metadata.get("prompt_tokens_after_shrink")
        agent_data.extra_fields["initial_prompt_tokens_before_shrink"] = initial_prompt_fit_metadata.get(
            "prompt_tokens_before_shrink"
        )
        agent_data.extra_fields["initial_prompt_tokens_after_shrink"] = initial_prompt_fit_metadata.get(
            "prompt_tokens_after_shrink"
        )
        agent_data.extra_fields["initial_prompt_shrink_count"] = initial_prompt_fit_metadata.get("prompt_shrink_count", 0)
        agent_data.extra_fields["initial_prompt_shrink_applied"] = bool(
            initial_prompt_fit_metadata.get("prompt_shrink_count", 0) > 0
        )
        agent_data.extra_fields["initial_prompt_fit_succeeded"] = bool(
            initial_prompt_fit_metadata.get("fits_prompt_length", True)
        )
        agent_data.extra_fields["initial_prompt_shrink_warning"] = initial_prompt_fit_metadata.get("prompt_shrink_warning")
        agent_data.extra_fields["initial_prompt_fit_time"] = initial_prompt_fit_time
        extra_info["agent_name"] = self.config.agent_name
        agent_data.extra_fields["agent_name"] = self.config.agent_name
        agent_data.extra_fields["extra_info"] = extra_info
        agent_data.extra_fields["export_failure_events"] = []
        agent_data.extra_fields["insight_original_images"] = original_images
        agent_data.extra_fields["insight_presented_images"] = presented_images
        agent_data.extra_fields["insight_presented_image_refs"] = [
            presented_image_to_export_ref(
                presented_img_idx,
                presented,
                kind="initial",
                original_images=original_images,
                initial_rescale=actual_initial_rescale,
            )
            for presented_img_idx, presented in enumerate(presented_images)
        ]

        if agent_data.extra_fields["initial_prompt_fit_succeeded"]:
            state = CoreAgentState.PENDING
            try:
                while state != CoreAgentState.TERMINATED:
                    if state == CoreAgentState.PENDING:
                        state = await self.handle_pending_state(agent_data)
                    elif state == CoreAgentState.GENERATING:
                        state = await self.handle_generating_state(agent_data, sampling_params)
                    elif state == CoreAgentState.PROCESSING_TOOLS:
                        state = await self.handle_processing_tools_state(agent_data)
                    else:
                        logger.error("Invalid state: %s", state)
                        state = CoreAgentState.TERMINATED
            except Exception as exc:
                failure_reason = f"agent_runtime_exception: {type(exc).__name__}: {exc}"
                logger.exception("Agent runtime failed for request_id=%s", request_id)
                agent_data.extra_fields["failure_reasons"] = [failure_reason]
                agent_data.extra_fields.setdefault("export_failure_events", []).append(
                    {
                        "kind": "agent_runtime",
                        "status": "exception",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "state": state.value,
                    }
                )
                if not agent_data.messages or agent_data.messages[-1].get("role") != "assistant":
                    agent_data.messages.append({"role": "assistant", "content": ""})
                    agent_data.assistant_turns += 1
        else:
            agent_data.prompt_ids = await self.runtime.apply_chat_template(
                aligned_prompt,
                tools=self.config.tool_schemas,
                images=images if images else None,
                videos=videos if videos else None,
            )
            agent_data.messages.append({"role": "assistant", "content": ""})
            agent_data.assistant_turns = 1
            failure_reason = (
                "initial_prompt_overflow_after_shrink: "
                f"{agent_data.extra_fields['initial_prompt_tokens']} > "
                f"{initial_prompt_fit_metadata.get('prompt_length_limit')}"
            )
            agent_data.extra_fields["failure_reasons"] = [failure_reason]
            agent_data.extra_fields.setdefault("export_failure_events", []).append(
                {
                    "kind": "initial_prompt_fit",
                    "status": "overflow_after_max_shrinks",
                    "error_message": failure_reason,
                    "shrink_count": agent_data.extra_fields.get("initial_prompt_shrink_count", 0),
                    "prompt_tokens": agent_data.extra_fields.get("initial_prompt_tokens"),
                    "prompt_length_limit": initial_prompt_fit_metadata.get("prompt_length_limit"),
                }
            )

        _record_core_inference_metrics(agent_data)
        _record_conversation_wall_time(agent_data, conversation_wall_time_start)

        if agent_data.response_mask:
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        else:
            response_ids = []
            prompt_ids = agent_data.prompt_ids
        response_tokens_total = len(response_ids)
        response_tokens_generated = int(sum(agent_data.response_mask))
        response_tokens_tool = response_tokens_total - response_tokens_generated
        agent_data.extra_fields["prompt_tokens"] = len(prompt_ids)
        agent_data.extra_fields["response_tokens_total"] = response_tokens_total
        agent_data.extra_fields["response_tokens_generated"] = response_tokens_generated
        agent_data.extra_fields["response_tokens_tool"] = response_tokens_tool
        if len(response_ids) > self.config.response_length or len(agent_data.response_mask) > self.config.response_length:
            agent_data.extra_fields["response_truncated"] = True

        multi_modal_output: dict[str, Any] = {}
        if agent_data.image_data:
            multi_modal_output["images"] = agent_data.image_data
        if agent_data.video_data:
            multi_modal_output["videos"] = agent_data.video_data

        export_payload = InSightQwenExportPayload(
            request_id=request_id,
            conversation_export_id=export_id,
            raw_prompt=raw_prompt,
            messages=agent_data.messages,
            original_images=original_images,
            presented_image_refs=agent_data.extra_fields.get("insight_presented_image_refs", []),
            actual_initial_rescale=actual_initial_rescale,
            initial_rescale_metadata=initial_rescale_metadata,
            extra_info=extra_info,
            failure_events=agent_data.extra_fields.get("export_failure_events", []),
            critical_failure=bool(agent_data.extra_fields.get("failure_reasons")),
            final_failure_reasons=agent_data.extra_fields.get("failure_reasons"),
        )

        agent_data.extra_fields.pop("insight_original_images", None)
        agent_data.extra_fields.pop("insight_presented_images", None)

        return InSightQwenAgentResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.config.response_length],
            response_mask=agent_data.response_mask[: self.config.response_length],
            response_logprobs=agent_data.response_logprobs[: self.config.response_length]
            if agent_data.response_logprobs
            else None,
            multi_modal_data=multi_modal_output,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
            export_payload=export_payload,
        )

    async def handle_pending_state(self, agent_data: CoreAgentData) -> CoreAgentState:
        agent_data.prompt_ids = await self.runtime.apply_chat_template(
            agent_data.messages,
            tools=self.config.tool_schemas,
            images=agent_data.image_data if agent_data.image_data else None,
            videos=agent_data.video_data if agent_data.video_data else None,
        )
        return CoreAgentState.GENERATING

    async def handle_generating_state(
        self,
        agent_data: CoreAgentData,
        sampling_params: dict[str, Any],
    ) -> CoreAgentState:
        remaining_response_length = self.config.response_length - len(agent_data.response_mask)
        if remaining_response_length <= 0:
            agent_data.extra_fields["response_truncated"] = True
            return CoreAgentState.TERMINATED

        generation_sampling_params = dict(sampling_params)
        requested_max_tokens = generation_sampling_params.pop("max_tokens", None)
        requested_max_new_tokens = generation_sampling_params.pop("max_new_tokens", None)
        if requested_max_tokens is None:
            requested_max_tokens = requested_max_new_tokens
        if requested_max_tokens is None:
            turn_max_tokens = remaining_response_length
        else:
            turn_max_tokens = min(int(requested_max_tokens), remaining_response_length)
        if turn_max_tokens <= 0:
            agent_data.extra_fields["response_truncated"] = True
            return CoreAgentState.TERMINATED
        generation_sampling_params["max_tokens"] = turn_max_tokens

        with _timer(agent_data.metrics, "generate_sequences"):
            output = await self.runtime.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=generation_sampling_params,
                image_data=agent_data.image_data if agent_data.image_data else None,
                video_data=agent_data.video_data if agent_data.video_data else None,
                messages=agent_data.messages,
                tools=self.config.tool_schemas,
            )
        if output.metrics:
            for key, value in output.metrics.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                agent_data.metrics[key] = float(agent_data.metrics.get(key, 0.0) or 0.0) + numeric_value
        if output.num_preempted is not None:
            agent_data.metrics["num_preempted"] = output.num_preempted

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        assistant_message = await self.runtime.decode(agent_data.response_ids, skip_special_tokens=True)
        agent_data.messages.append({"role": "assistant", "content": assistant_message})

        if len(agent_data.response_mask) >= self.config.response_length:
            agent_data.extra_fields["response_truncated"] = True
            return CoreAgentState.TERMINATED
        if self.config.max_assistant_turns and agent_data.assistant_turns >= self.config.max_assistant_turns:
            return CoreAgentState.TERMINATED
        if self.config.max_user_turns and agent_data.user_turns >= self.config.max_user_turns:
            return CoreAgentState.TERMINATED
        if self.config.tool_schemas is None:
            return CoreAgentState.TERMINATED

        with _timer(agent_data.metrics, "tool_parsing"):
            agent_data.tool_calls = await self.runtime.extract_tool_calls(agent_data.response_ids)

        if agent_data.tool_calls:
            return CoreAgentState.PROCESSING_TOOLS
        return CoreAgentState.TERMINATED

    async def handle_processing_tools_state(self, agent_data: CoreAgentData) -> CoreAgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Image.Image] = []
        base_image_count = len(agent_data.image_data) if isinstance(agent_data.image_data, list) else 0
        new_presented_this_turn: list[PresentedImageState] = []
        new_presented_refs_this_turn: list[dict[str, Any]] = []

        tasks = []
        for tool_call in agent_data.tool_calls[: self.config.max_parallel_calls]:
            tasks.append(self.call_tool(tool_call, agent_data))

        with _timer(agent_data.metrics, "tool_calls"):
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            tool_call = agent_data.tool_calls[i]
            if isinstance(response, Exception):
                logger.warning("Tool call %s failed: %s", tool_call.name, response)
                agent_data.extra_fields.setdefault("export_failure_events", []).append(
                    {
                        "kind": "tool_execution",
                        "status": "exception",
                        "tool_name": tool_call.name,
                        "error_message": str(response),
                    }
                )
                message = {"role": "tool", "content": f"Error executing tool: {response}"}
            else:
                if response.images:
                    content = []
                    for offset, _ in enumerate(response.images):
                        new_index = base_image_count + len(new_images_this_turn) + offset
                        content.append({"type": "text", "text": f"Image {new_index}:"})
                        content.append({"type": "image"})
                    if response.text_result:
                        content.append({"type": "text", "text": response.text_result})
                    new_images_this_turn.extend(response.images)
                    new_presented_this_turn.extend(response.presented_images)
                    new_presented_refs_this_turn.extend(response.presented_image_refs)
                    message = {"role": "tool", "content": content}
                else:
                    if response.text_result:
                        agent_data.extra_fields.setdefault("export_failure_events", []).append(
                            {
                                "kind": "tool_execution",
                                "status": "error",
                                "tool_name": tool_call.name,
                                "error_message": response.text_result,
                            }
                        )
                    message = {"role": "tool", "content": response.text_result or ""}
            add_messages.append(message)

        agent_data.messages.extend(add_messages)

        response_ids = await self.runtime.apply_chat_template(
            add_messages,
            images=new_images_this_turn if new_images_this_turn else None,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.config.response_length:
            agent_data.extra_fields["response_truncated"] = True
            return CoreAgentState.TERMINATED

        if new_images_this_turn:
            if not agent_data.image_data:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)
            agent_data.extra_fields.setdefault("insight_presented_images", []).extend(new_presented_this_turn)
            agent_data.extra_fields.setdefault("insight_presented_image_refs", []).extend(new_presented_refs_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        agent_data.user_turns += 1
        return CoreAgentState.GENERATING

    async def call_tool(self, tool_call: CoreFunctionCall, agent_data: CoreAgentData) -> CoreToolExecutionResult:
        if tool_call.name == "image_zoom_in_tool":
            return await asyncio.to_thread(self.call_insight_zoom_tool, tool_call, agent_data)
        if self.fallback_tool_executor is not None:
            return await self.fallback_tool_executor(tool_call, agent_data)
        return CoreToolExecutionResult(f"Tool '{tool_call.name}' not found.")

    def call_insight_zoom_tool(self, tool_call: CoreFunctionCall, agent_data: CoreAgentData) -> CoreToolExecutionResult:
        try:
            tool_args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            return CoreToolExecutionResult(f"Tool Execution Error Invalid JSON arguments: {exc}")

        img_idx = tool_args.get("img_idx")
        label = tool_args.get("label")
        bbox_2d = tool_args.get("bbox_2d")
        if not isinstance(img_idx, int):
            return CoreToolExecutionResult("Tool Execution Error img_idx must be an integer.")
        if not isinstance(label, str):
            return CoreToolExecutionResult("Tool Execution Error label must be a string.")
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
            return CoreToolExecutionResult("Tool Execution Error bbox_2d must be a list of four numbers.")

        presented_images: list[PresentedImageState] = agent_data.extra_fields.get("insight_presented_images", [])
        original_images: list[Image.Image] = agent_data.extra_fields.get("insight_original_images", [])
        if not (0 <= img_idx < len(presented_images)):
            return CoreToolExecutionResult(f"Error: Invalid input image index {img_idx}.")

        parent = presented_images[img_idx]
        bbox_on_presented = scale_bbox_from_qwen_range(bbox_2d, parent.display_size)
        if bbox_on_presented is None:
            return CoreToolExecutionResult("Tool Execution Error invalid bbox_2d.")

        bbox_on_original = translate_bbox_to_original(parent, bbox_on_presented)
        if bbox_on_original is None:
            return CoreToolExecutionResult("Tool Execution Error failed to translate bbox to original image.")

        source_original = original_images[parent.source_original_img_idx]
        bbox_on_original = clamp_bbox_to_image(bbox_on_original, source_original.size)
        if bbox_on_original is None:
            return CoreToolExecutionResult("Tool Execution Error translated bbox is invalid on original image.")

        x1, y1, x2, y2 = bbox_on_presented
        region_display_size = (max(1, x2 - x1), max(1, y2 - y1))
        target_display_size = cap_size_by_area(
            resize_dims_by_factor(region_display_size, self.config.region_zoom_in_factor),
            self.config.crop_image_max_area,
        )
        aspect_ratio_error = validate_qwen_image_aspect_ratio(target_display_size)
        if aspect_ratio_error is not None:
            return CoreToolExecutionResult(aspect_ratio_error)

        crop = source_original.crop(bbox_on_original)
        if crop.size != target_display_size:
            crop = crop.resize(target_display_size, Image.LANCZOS)
        aspect_ratio_error = validate_qwen_image_aspect_ratio(crop.size)
        if aspect_ratio_error is not None:
            return CoreToolExecutionResult(aspect_ratio_error)

        presented = PresentedImageState(
            image=crop,
            source_original_img_idx=parent.source_original_img_idx,
            bbox_on_original=bbox_on_original,
            display_size=crop.size,
        )
        export_ref = presented_image_to_export_ref(
            len(presented_images),
            presented,
            kind="region_crop",
            original_images=original_images,
            parent_presented_img_idx=img_idx,
            region_description=label,
            bbox_on_presented=bbox_on_presented,
            zoom_in_factor=self.config.region_zoom_in_factor,
            region_display_size_before_zoom=region_display_size,
        )
        return CoreToolExecutionResult("", [crop], [presented], [export_ref])

    def build_presented_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        images_are_presented: bool = False,
        training_mode: bool = False,
        sample_initial_rescale: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[Image.Image], list[PresentedImageState], float, dict[str, Any]]:
        original_images: list[Image.Image] = []
        staged_messages: list[tuple[dict[str, Any], list[Image.Image], list[str], bool]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            staged_items: list[Image.Image] = []
            saw_image = False
            trailing_text_parts: list[str] = []
            passthrough_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                image_value = None
                if item.get("type") == "image" and "image" in item:
                    image_value = item.get("image")
                elif item.get("type") == "image_url":
                    image_value = item.get("image_url", {})

                if image_value is None:
                    if "text" in item and isinstance(item["text"], str):
                        trailing_text_parts.append(item["text"])
                    else:
                        passthrough_content.append(copy.deepcopy(item))
                    continue

                image = load_prompt_image(image_value)
                if image is None:
                    continue
                original_images.append(image)
                staged_items.append(image)
                saw_image = True
            if not saw_image:
                message["content"] = passthrough_content or [
                    copy.deepcopy(item) for item in content if isinstance(item, dict)
                ]
            staged_messages.append((message, staged_items, trailing_text_parts, saw_image))

        actual_initial_rescale, initial_rescale_metadata = self.resolve_initial_rescale_for_prompt(
            image_sizes=[image.size for image in original_images],
            images_are_presented=images_are_presented,
            training_mode=training_mode,
            sample_initial_rescale=sample_initial_rescale,
        )

        presented_images: list[PresentedImageState] = []
        original_image_idx = 0
        for message, staged_items, trailing_text_parts, saw_image in staged_messages:
            if not saw_image:
                continue
            new_content: list[dict[str, Any]] = []
            for image in staged_items:
                presented_image = image.copy() if images_are_presented else self.build_presented_image(
                    image,
                    actual_initial_rescale,
                )
                presented_img_idx = len(presented_images)
                presented_images.append(
                    PresentedImageState(
                        image=presented_image,
                        source_original_img_idx=original_image_idx,
                        bbox_on_original=(0, 0, image.size[0], image.size[1]),
                        display_size=presented_image.size,
                    )
                )
                original_image_idx += 1
                if new_content:
                    new_content.append({"type": "text", "text": "\n---\n"})
                new_content.append({"type": "text", "text": f"Image {presented_img_idx}:"})
                new_content.append({"type": "image", "image": presented_image})
            if trailing_text_parts:
                new_content.append({"type": "text", "text": "".join(trailing_text_parts)})
            message["content"] = new_content

        return messages, original_images, presented_images, actual_initial_rescale, initial_rescale_metadata

    async def fit_initial_prompt_to_prompt_length(
        self,
        messages: list[dict[str, Any]],
        presented_images: list[PresentedImageState],
        images: list[Any] | None,
        videos: list[Any] | None,
    ) -> tuple[list[Any] | None, list[Any] | None, dict[str, Any]]:
        prompt_length_limit = self.config.prompt_length
        if prompt_length_limit is None:
            return images, videos, {
                "prompt_shrink_count": 0,
                "prompt_tokens_before_shrink": None,
                "prompt_tokens_after_shrink": None,
                "fits_prompt_length": True,
                "prompt_length_limit": None,
                "prompt_shrink_warning": None,
                "prompt_shrink_area_factor": INITIAL_PROMPT_SHRINK_AREA_FACTOR,
                "prompt_max_shrink_steps": INITIAL_PROMPT_MAX_SHRINK_STEPS,
            }

        prompt_ids = await self.runtime.apply_chat_template(
            messages,
            tools=self.config.tool_schemas,
            images=images if images else None,
            videos=videos if videos else None,
        )
        prompt_tokens_before = len(prompt_ids)
        prompt_tokens_after = prompt_tokens_before
        shrink_count = 0

        while (
            prompt_tokens_after > prompt_length_limit
            and shrink_count < INITIAL_PROMPT_MAX_SHRINK_STEPS
            and presented_images
        ):
            shrink_count += 1
            self.shrink_presented_prompt_images(messages, presented_images)
            multi_modal_data = await self.runtime.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self.runtime.apply_chat_template(
                messages,
                tools=self.config.tool_schemas,
                images=images if images else None,
                videos=videos if videos else None,
            )
            prompt_tokens_after = len(prompt_ids)

        fits_prompt_length = prompt_tokens_after <= prompt_length_limit
        warning = None
        if shrink_count > 0:
            warning = (
                "initial prompt exceeded prompt_length; shrank presented image area by 50% "
                f"{shrink_count} time(s) (max {INITIAL_PROMPT_MAX_SHRINK_STEPS}) "
                f"from {prompt_tokens_before} to {prompt_tokens_after} tokens with prompt_length={prompt_length_limit}"
            )
            logger.warning(warning)
            print(f"[InSightQwenAgentRunner] WARNING: {warning}")
        if not fits_prompt_length and prompt_tokens_after > prompt_length_limit:
            overflow_warning = (
                f"initial prompt still exceeds prompt_length after {shrink_count} shrink step(s): "
                f"{prompt_tokens_after} > {prompt_length_limit}"
            )
            logger.warning(overflow_warning)
            print(f"[InSightQwenAgentRunner] WARNING: {overflow_warning}")
            warning = overflow_warning if warning is None else f"{warning}; {overflow_warning}"

        return images, videos, {
            "prompt_shrink_count": shrink_count,
            "prompt_tokens_before_shrink": prompt_tokens_before,
            "prompt_tokens_after_shrink": prompt_tokens_after,
            "fits_prompt_length": fits_prompt_length,
            "prompt_length_limit": prompt_length_limit,
            "prompt_shrink_warning": warning,
            "prompt_shrink_area_factor": INITIAL_PROMPT_SHRINK_AREA_FACTOR,
            "prompt_max_shrink_steps": INITIAL_PROMPT_MAX_SHRINK_STEPS,
        }

    def shrink_presented_prompt_images(
        self,
        messages: list[dict[str, Any]],
        presented_images: list[PresentedImageState],
    ) -> None:
        presented_img_idx = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                if presented_img_idx >= len(presented_images):
                    return
                presented_state = presented_images[presented_img_idx]
                target_size = cap_size_by_area(
                    resize_dims_by_factor(presented_state.image.size, INITIAL_PROMPT_SHRINK_DIM_FACTOR),
                    self.config.gpt_image_max_area,
                )
                if target_size == presented_state.image.size:
                    shrunk_image = presented_state.image.copy()
                else:
                    shrunk_image = presented_state.image.resize(target_size, Image.LANCZOS)
                presented_state.image = shrunk_image
                presented_state.display_size = shrunk_image.size
                item["image"] = shrunk_image
                presented_img_idx += 1

    def resolve_initial_rescale_for_prompt(
        self,
        image_sizes: list[tuple[int, int]],
        *,
        images_are_presented: bool,
        training_mode: bool,
        sample_initial_rescale: float | None = None,
    ) -> tuple[float, dict[str, Any]]:
        sample_override_applied = sample_initial_rescale is not None
        configured_initial_rescale = self.config.initial_rescale
        base_initial_rescale = sample_initial_rescale if sample_override_applied else configured_initial_rescale
        if images_are_presented:
            return base_initial_rescale, {
                "images_are_presented": True,
                "randomized": False,
                "sampled_initial_rescale": base_initial_rescale,
                "actual_initial_rescale": base_initial_rescale,
                "configured_initial_rescale": configured_initial_rescale,
                "sample_initial_rescale_override": sample_initial_rescale,
                "sample_initial_rescale_override_applied": sample_override_applied,
            }

        sampled_initial_rescale = base_initial_rescale
        randomized = False
        max_rescale_under_budget = None

        if not sample_override_applied and training_mode and self.should_randomize_initial_rescale() and image_sizes:
            randomized_rescale, max_rescale_under_budget = self.sample_training_initial_rescale(image_sizes)
            if randomized_rescale is not None:
                sampled_initial_rescale = randomized_rescale
                randomized = True

        actual_initial_rescale = resolve_dynamic_initial_rescale(
            image_sizes=image_sizes,
            configured_initial_rescale=sampled_initial_rescale,
            total_pixels_lower_bound=self.config.initial_input_pixels_lower_bound,
            per_image_max_area=self.config.gpt_image_max_area,
        )
        if randomized and max_rescale_under_budget is not None:
            actual_initial_rescale = min(actual_initial_rescale, max_rescale_under_budget)
        return actual_initial_rescale, {
            "images_are_presented": False,
            "randomized": randomized,
            "sampled_initial_rescale": sampled_initial_rescale,
            "actual_initial_rescale": actual_initial_rescale,
            "configured_initial_rescale": configured_initial_rescale,
            "sample_initial_rescale_override": sample_initial_rescale,
            "sample_initial_rescale_override_applied": sample_override_applied,
            "train_randomization_prob": self.config.train_initial_rescale_randomization_prob,
            "train_randomization_min": self.config.train_initial_rescale_randomization_min,
            "train_randomization_max": self.config.train_initial_rescale_randomization_max,
            "train_randomization_text_budget": self.config.train_initial_rescale_randomization_text_budget,
            "image_token_budget_estimate": max(
                (self.config.prompt_length or 0) - self.config.train_initial_rescale_randomization_text_budget,
                0,
            ),
        }

    @staticmethod
    def get_sample_initial_rescale(extra_info: dict[str, Any]) -> float | None:
        raw_value = extra_info.get("initial_rescale")
        if raw_value is None or raw_value == "":
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"extra_info.initial_rescale must be a positive finite float, got {raw_value!r}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"extra_info.initial_rescale must be a positive finite float, got {raw_value!r}")
        return value

    def should_randomize_initial_rescale(self) -> bool:
        return (
            self.config.train_initial_rescale_randomization_prob > 0.0
            and self.config.train_initial_rescale_randomization_max > self.config.train_initial_rescale_randomization_min
            and random.random() < self.config.train_initial_rescale_randomization_prob
        )

    def sample_training_initial_rescale(self, image_sizes: list[tuple[int, int]]) -> tuple[float | None, float | None]:
        image_token_budget = max(
            (self.config.prompt_length or 0) - self.config.train_initial_rescale_randomization_text_budget,
            0,
        )
        if image_token_budget <= 0:
            return None, None

        max_rescale_under_budget = self.max_initial_rescale_under_image_token_budget(
            image_sizes=image_sizes,
            image_token_budget=image_token_budget,
        )
        if max_rescale_under_budget is None:
            return None, None

        lo = self.config.train_initial_rescale_randomization_min
        hi = min(self.config.train_initial_rescale_randomization_max, max_rescale_under_budget)
        if hi < lo:
            return None, None
        if abs(hi - lo) < 1e-8:
            return lo, max_rescale_under_budget
        return random.uniform(lo, hi), max_rescale_under_budget

    def estimate_image_tokens_after_rescale(
        self,
        image_sizes: list[tuple[int, int]],
        initial_rescale: float,
    ) -> int:
        total_tokens = 0
        for width, height in image_sizes:
            resized_w, resized_h = resize_dims_by_factor((width, height), initial_rescale)
            capped_w, capped_h = cap_size_by_area((resized_w, resized_h), self.config.gpt_image_max_area)
            total_tokens += math.ceil(capped_w / 32) * math.ceil(capped_h / 32)
        return total_tokens

    def max_initial_rescale_under_image_token_budget(
        self,
        *,
        image_sizes: list[tuple[int, int]],
        image_token_budget: int,
    ) -> float | None:
        if (
            self.estimate_image_tokens_after_rescale(
                image_sizes,
                self.config.train_initial_rescale_randomization_min,
            )
            > image_token_budget
        ):
            return None
        if (
            self.estimate_image_tokens_after_rescale(
                image_sizes,
                self.config.train_initial_rescale_randomization_max,
            )
            <= image_token_budget
        ):
            return self.config.train_initial_rescale_randomization_max

        lo = self.config.train_initial_rescale_randomization_min
        hi = self.config.train_initial_rescale_randomization_max
        best = lo
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if self.estimate_image_tokens_after_rescale(image_sizes, mid) <= image_token_budget:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    def build_presented_image(self, image: Image.Image, initial_rescale: float) -> Image.Image:
        target_size = cap_size_by_area(
            resize_dims_by_factor(image.size, initial_rescale),
            self.config.gpt_image_max_area,
        )
        if image.size == target_size:
            return image.copy()
        return image.resize(target_size, Image.LANCZOS)
