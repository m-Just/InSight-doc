from __future__ import annotations

import logging
from typing import Any

from transformers import AutoProcessor, AutoTokenizer

from insight_agent_core import (
    CoreFunctionCall,
    CoreGenerationOutput,
    InSightQwenAgentConfig,
    InSightQwenAgentRunner,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.insight_o3_agent_loop import VReasonerLoopV2
from verl.experimental.agent_loop.qwen_agent_loop import (
    QwenAgentLoop,
    _build_insight_export_conversation,
)
from verl.utils.vreasoner_v2_conversation_export import build_export_record, export_conversation


logger = logging.getLogger(__name__)


class VerlCoreRuntime:
    """Adapter from the core runner protocol to verl's token-level rollout stack."""

    def __init__(self, loop: QwenAgentLoop) -> None:
        self.loop = loop

    async def process_vision_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.loop.process_vision_info(messages)

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        return await self.loop.apply_chat_template(
            messages,
            tools=tools,
            images=images,
            videos=videos,
            remove_system_prompt=remove_system_prompt,
        )

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CoreGenerationOutput:
        output = await self.loop.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        return CoreGenerationOutput(
            token_ids=output.token_ids,
            log_probs=output.log_probs,
            num_preempted=output.num_preempted,
        )

    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return await self.loop.loop.run_in_executor(
            None,
            lambda: self.loop.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens),
        )

    async def extract_tool_calls(self, response_ids: list[int]) -> list[CoreFunctionCall]:
        _, tool_calls = await self.loop.tool_parser.extract_tool_calls(response_ids)
        return [CoreFunctionCall(name=call.name, arguments=call.arguments) for call in tool_calls]


@register("insight_qwen_agent_core")
class CoreInSightQwenAgentLoop(QwenAgentLoop):
    """verl wrapper for the extracted InSight Qwen core runner.

    The old ``insight_qwen_agent`` class remains unchanged. This class is a
    migration target that keeps verl-specific I/O, token tensors, and export
    side effects outside the core implementation.
    """

    DEFAULT_INITIAL_RESCALE = 0.25
    DEFAULT_GPT_IMAGE_MAX_AREA = 1280 * 1280
    DEFAULT_CROP_IMAGE_MAX_AREA = 1280 * 1280
    DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND = 0
    DEFAULT_REGION_ZOOM_IN_FACTOR = 4.0
    DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_PROB = 0.0
    DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_MIN = 0.25
    DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_MAX = 0.25
    DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_TEXT_BUDGET = 1024

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        initial_rescale: float = DEFAULT_INITIAL_RESCALE,
        gpt_image_max_area: int = DEFAULT_GPT_IMAGE_MAX_AREA,
        crop_image_max_area: int = DEFAULT_CROP_IMAGE_MAX_AREA,
        initial_input_pixels_lower_bound: int = DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND,
        region_zoom_in_factor: float = DEFAULT_REGION_ZOOM_IN_FACTOR,
        train_initial_rescale_randomization_prob: float = DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_PROB,
        train_initial_rescale_randomization_min: float = DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_MIN,
        train_initial_rescale_randomization_max: float = DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_MAX,
        train_initial_rescale_randomization_text_budget: int = DEFAULT_TRAIN_INITIAL_RESCALE_RANDOMIZATION_TEXT_BUDGET,
        **kwargs,
    ):
        if "presented_initial_rescale" in kwargs:
            initial_rescale = kwargs.pop("presented_initial_rescale")
        if "presented_max_area" in kwargs:
            gpt_image_max_area = kwargs.pop("presented_max_area")
        if "crop_max_area" in kwargs:
            crop_image_max_area = kwargs.pop("crop_max_area")
        if "presented_initial_pixels_lower_bound" in kwargs:
            initial_input_pixels_lower_bound = kwargs.pop("presented_initial_pixels_lower_bound")

        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.initial_rescale = initial_rescale
        self.gpt_image_max_area = gpt_image_max_area
        self.crop_image_max_area = crop_image_max_area
        self.initial_input_pixels_lower_bound = initial_input_pixels_lower_bound
        self.region_zoom_in_factor = region_zoom_in_factor
        self.train_initial_rescale_randomization_prob = train_initial_rescale_randomization_prob
        self.train_initial_rescale_randomization_min = train_initial_rescale_randomization_min
        self.train_initial_rescale_randomization_max = train_initial_rescale_randomization_max
        self.train_initial_rescale_randomization_text_budget = train_initial_rescale_randomization_text_budget

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        conversation_export_id = kwargs.get(
            "conversation_export_id",
            extra_info.get("conversation_export_id"),
        )
        core_config = InSightQwenAgentConfig(
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            max_user_turns=self.max_user_turns,
            max_assistant_turns=self.max_assistant_turns,
            max_parallel_calls=self.max_parallel_calls,
            tool_schemas=self.tool_schemas if self.tool_schemas else None,
            tool_parser_name=self.tool_parser_name,
            initial_rescale=self.initial_rescale,
            gpt_image_max_area=self.gpt_image_max_area,
            crop_image_max_area=self.crop_image_max_area,
            initial_input_pixels_lower_bound=self.initial_input_pixels_lower_bound,
            region_zoom_in_factor=self.region_zoom_in_factor,
            train_initial_rescale_randomization_prob=self.train_initial_rescale_randomization_prob,
            train_initial_rescale_randomization_min=self.train_initial_rescale_randomization_min,
            train_initial_rescale_randomization_max=self.train_initial_rescale_randomization_max,
            train_initial_rescale_randomization_text_budget=self.train_initial_rescale_randomization_text_budget,
            agent_name="insight_qwen_agent",
        )
        runner = InSightQwenAgentRunner(core_config, VerlCoreRuntime(self))
        result = await runner.run(
            dict(sampling_params),
            raw_prompt=kwargs["raw_prompt"],
            extra_info=extra_info,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
            validate=bool(kwargs.get("_validate", False)),
            conversation_export_id=conversation_export_id,
        )

        conversation_export_json_path = self._export_conversation_if_enabled(result, sampling_params, kwargs)
        if conversation_export_json_path:
            result.extra_fields["conversation_export_json_path"] = conversation_export_json_path
            result.extra_fields.setdefault("extra_info", {})["conversation_export_json_path"] = conversation_export_json_path
        if not self.conversation_export_dir:
            result.extra_fields.pop("insight_presented_image_refs", None)
            result.extra_fields.pop("export_failure_events", None)

        output = AgentLoopOutput(
            prompt_ids=result.prompt_ids,
            response_ids=result.response_ids,
            response_mask=result.response_mask,
            multi_modal_data=result.multi_modal_data,
            response_logprobs=result.response_logprobs,
            num_turns=result.num_turns,
            metrics=AgentLoopMetrics(**result.metrics),
            extra_fields=result.extra_fields,
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output

    def _export_conversation_if_enabled(
        self,
        result,
        sampling_params: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> str | None:
        if not self.conversation_export_dir:
            return None

        payload = result.export_payload
        validate = bool(kwargs.get("_validate", False))
        initial_question = payload.extra_info.get("question", "")
        try:
            record = build_export_record(
                job_id=payload.request_id,
                parent_job_id=kwargs.get("parent_job_id"),
                root_job_id=kwargs.get("root_job_id", payload.request_id),
                validate=validate,
                initial_question=initial_question,
                messages_api=[],
                raw_prompt=payload.raw_prompt,
                original_images=payload.original_images,
                presented_image_refs=payload.presented_image_refs,
                request_params={
                    "tool_parser": self.tool_parser_name,
                    "prompt_length": self.prompt_length,
                    "response_length": self.response_length,
                    "max_user_turns": self.max_user_turns,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_parallel_calls": self.max_parallel_calls,
                },
                loop_params={
                    "implementation": "insight_agent_core",
                    "initial_rescale": payload.actual_initial_rescale,
                    "configured_initial_rescale": self.initial_rescale,
                    "initial_rescale_randomization": payload.initial_rescale_metadata,
                    "initial_prompt_tokens": result.extra_fields.get("initial_prompt_tokens"),
                    "initial_prompt_tokens_before_shrink": result.extra_fields.get(
                        "initial_prompt_tokens_before_shrink"
                    ),
                    "initial_prompt_tokens_after_shrink": result.extra_fields.get(
                        "initial_prompt_tokens_after_shrink"
                    ),
                    "initial_prompt_shrink_count": result.extra_fields.get("initial_prompt_shrink_count", 0),
                    "initial_prompt_shrink_applied": result.extra_fields.get("initial_prompt_shrink_applied", False),
                    "initial_prompt_fit_succeeded": result.extra_fields.get("initial_prompt_fit_succeeded", True),
                    "initial_prompt_shrink_warning": result.extra_fields.get("initial_prompt_shrink_warning"),
                    "initial_input_pixels_lower_bound": self.initial_input_pixels_lower_bound,
                    "gpt_image_max_area": self.gpt_image_max_area,
                    "crop_image_max_area": self.crop_image_max_area,
                    "region_zoom_in_factor": self.region_zoom_in_factor,
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
                        "core_inference_time": result.extra_fields.get("core_inference_time", 0.0),
                        "conversation_wall_time": result.extra_fields.get("conversation_wall_time", 0.0),
                    },
                    "agent_name": "insight_qwen_agent",
                },
                sampling_params=dict(sampling_params),
                tools_kwargs=kwargs.get("tools_kwargs", {}),
                extra_info=payload.extra_info,
                failure_events=payload.failure_events,
                critical_failure=payload.critical_failure,
                final_failure_reasons=payload.final_failure_reasons,
            )
            record["agent_name"] = "insight_qwen_agent"
            record["conversation"] = _build_insight_export_conversation(
                payload.messages,
                initial_question=initial_question,
            )
            export_index_metadata = {
                "global_step": kwargs.get("_global_steps"),
                "split": "val" if validate else "train",
                "validate": validate,
                "trajectory_sample_index": kwargs.get("_trajectory_sample_index"),
                "rollout_n": kwargs.get("_rollout_n"),
            }
            record["job"].update(export_index_metadata)
            return export_conversation(
                self.conversation_export_dir,
                record,
                job_id=payload.request_id,
                export_id=payload.conversation_export_id,
                index_metadata=export_index_metadata,
            )
        except Exception as exc:
            logger.warning("failed to export core insight_qwen_agent conversation: %s", exc)
            result.extra_fields.setdefault("export_failure_events", []).append(
                {
                    "kind": "conversation_export",
                    "status": "error",
                    "error_message": str(exc),
                }
            )
            return None


@register("vreasoner_v2_core")
class CoreVReasonerLoopV2(VReasonerLoopV2):
    """Compatibility registration for the next vreasoner_v2 core migration step.

    This preserves the old implementation behind a new agent name while
    ``insight_qwen_agent_core`` is validated. The VReasoner orchestration can
    then be moved to the same core/runtime pattern without disrupting old runs.
    """

    pass
