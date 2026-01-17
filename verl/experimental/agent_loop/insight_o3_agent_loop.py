import asyncio
from dataclasses import dataclass, asdict
from functools import partial
import logging
import os
from typing import Any
from copy import deepcopy
from uuid import uuid4
import base64
import io
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    AgentLoopMetrics,
    _DummyConfig,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.profiler import simple_timer
from verl.utils.vsearch import BBox, parse_bbox, resize_bbox
from verl.utils.vsearch_gpt_async import get_gpt_visual_search_request

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ExtraFields:
    agent_name: str
    job_id: str
    parent_job_id: str | None
    root_job_id: str
    extra_info: dict[str, Any]
    n_tool_calls: int = 0
    caller_feedback: str | None = None
    final_bbox: BBox | None = None
    critical_failure: bool | None = None   # None if don't care
    messages: list[dict] | None = None
    multi_modal_data: dict[str, Any] | None = None


@register("vsearcher")
class VSearcherLoop(ToolAgentLoop):
    """Agent loop for the vsearcher agent, assuming the underlying model is Qwen2.5-VL."""

    MAX_ALLOWED_TOKEN_ID = 151664
    VALIDATION_CONTEXT_LENGTH = 32 * 1024
    MAX_TOKENS_PER_TURN = 1024

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Currently:
        # - we bake the system prompt into the dataset parquet files (or vreasoner calls), not using tool schemas
        # - we ignore tool messages returned by image_zoom_in_tool and manually add the messages to the chat template;
        #   however, errors returned by image_zoom_in_tool are still passed to the model, prompting its response
        self.tool_schemas = None

        validate = kwargs["_validate"]
        if validate:
            # Temporarily override prompt_length and response_length for validation
            # This allows for larger max_pixels and longer responses
            original_prompt_length = self.prompt_length
            original_response_length = self.response_length
            self.prompt_length = self.response_length = self.VALIDATION_CONTEXT_LENGTH

        # Validate sampling params
        assert "stop" not in sampling_params, f"{sampling_params['stop']=}"
        if validate:
            assert sampling_params["temperature"] == 0.0, f"{sampling_params['temperature']=}"

        sampling_params = deepcopy(sampling_params)
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = self.MAX_TOKENS_PER_TURN

        # Run the agent loop
        output = await super().run(sampling_params, **kwargs)

        # Extract final bbox from response text
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None, partial(self.tokenizer.decode, output.response_ids, skip_special_tokens=True)
        )
        last_response = response_text.split("user\n")[-1].split("assistant\n")[-1]
        answer_text = last_response.split("<answer>")[-1].split("</answer>")[0].strip()

        try:
            final_bbox = parse_bbox(answer_text)
        except Exception as e:
            logger.warning(f"no valid bbox found in vsearcher's answer: {e}")
            final_bbox = None

        if validate:
            # Drop all image tokens and truncate the rest (if necessary) to fit within the original prompt_length and response_length
            # Here we drop the image tokens first to avoid truncating the text, which will be used for validation
            # This should be fine as image tokens almost always take up most of the context length (at least for qwen2.5-vl and the data we have tested)
            output.prompt_ids = [i for i in output.prompt_ids if i != self.processor.image_token_id]
            output.response_ids, output.response_mask = map(
                list,
                zip(
                    *[
                        (i, j)
                        for i, j in zip(output.response_ids, output.response_mask, strict=False)
                        if i != self.processor.image_token_id
                    ],
                    strict=False,
                ),
            )
            output.prompt_ids = output.prompt_ids[-original_prompt_length:]
            output.response_ids = output.response_ids[-original_response_length:]
            output.response_mask = output.response_mask[-original_response_length:]
            self.prompt_length = original_prompt_length
            self.response_length = original_response_length

        # Prepare extra fields
        job_id = uuid4().hex

        extra_fields = ExtraFields(
            agent_name='vsearcher',
            job_id=job_id,
            parent_job_id=kwargs.get("parent_job_id", None),
            root_job_id=kwargs.get("root_job_id", job_id),
            extra_info=kwargs["extra_info"],
            n_tool_calls=(output.num_turns - 1) // 2,   # the number of successful tool calls = the number of user turns
            caller_feedback=None,
            final_bbox=final_bbox,
            messages=self.messages,
            multi_modal_data=output.multi_modal_data,
        )

        output.extra_fields.update(asdict(extra_fields))

        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        self.messages = agent_data.messages

        max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        if len(agent_data.prompt_ids) >= max_model_len:
            logger.warning(f"prompt len exceeded max model len: {len(agent_data.prompt_ids)=} >= {max_model_len=}")
            return AgentState.TERMINATED
    
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        # Check for invalid tokens
        if max(agent_data.response_ids) > self.MAX_ALLOWED_TOKEN_ID:
            logger.warning("generated out-of-vocabulary token")
            return AgentState.TERMINATED

        if self.processor.tokenizer.pad_token_id in agent_data.response_ids:
            logger.warning(f"generated pad token before eos")
            return AgentState.TERMINATED

        # vllm generation may stop at <|im_end|>, but for qwen2.5-vl, a complete message ends with <|im_end|>\n
        # so if the generation stopped at <|im_end|>, we add \n to the end of the response
        if agent_data.response_ids[-1] == self.processor.tokenizer.eos_token_id:
            agent_data.response_ids += [198]
            agent_data.prompt_ids += [198]
            agent_data.response_mask += [0]

        # Check for termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        return state


@register("vreasoner")
class VReasonerLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,                  # tokenizer for vSearcher
        processor: AutoProcessor,                  # processor for vSearcher
        model: str = "gpt-5-mini",                 # API model name for vReasoner
        max_tool_calls: int = 6,                   # max number of tool calls for vReasoner
        max_round_retries: int = 3,                # max number of retries for vReasoner
        max_round_retries_val: int = 5,            # max number of retries for vReasoner during validation
        gpt_image_max_area: int = 1280 * 1280,     # max area for GPT image in vReasoner
        max_completion_tokens: int | None = 2048,  # max completion tokens (per turn) for vReasoner
        reasoning_effort: str | None = None,       # reasoning effort for vReasoner
        enable_tool_feedback: bool = True,         # enable tool feedback for vReasoner
        **kwargs,                                  # extra kwargs for vReasoner
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_round_retries = max_round_retries
        self.max_round_retries_val = max_round_retries_val
        self.gpt_image_max_area = gpt_image_max_area
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.enable_tool_feedback = enable_tool_feedback
        self.vsearcher_loop = VSearcherLoop(trainer_config, server_manager, tokenizer, processor)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        job_id = uuid4().hex
        root_job_id = kwargs.get("root_job_id", job_id)

        validate = kwargs["_validate"]

        messages_api = []
        bbox = None
        n_tool_calls = 0
        vsearcher_outputs = []
        critical_failure = False
        multi_modal_data = {"image": []}

        metrics = AgentLoopMetrics()
        profile = {}

        # Start agent loop
        while True:
            with simple_timer("api_calls", profile):
                request = await get_gpt_visual_search_request(
                    initial_question=kwargs["extra_info"]["question"],
                    original_image=kwargs["extra_info"]["image_ori"][0],
                    messages=messages_api,
                    bbox=bbox,
                    model=self.model,
                    max_tool_calls=self.max_tool_calls,
                    max_round_retries=self.max_round_retries if not validate else self.max_round_retries_val,
                    gpt_image_max_area=self.gpt_image_max_area,
                    max_completion_tokens=self.max_completion_tokens,
                    reasoning_effort=self.reasoning_effort,
                    enable_tool_feedback=(False if validate else self.enable_tool_feedback),
                )

            if not request.success:
                logger.warning("API generation failed")
                critical_failure = True
                break

            messages_api = request.messages
            if vsearcher_outputs:
                vsearcher_outputs[-1].extra_fields["caller_feedback"] = request.tool_feedback

            if request.region_description is None:
                break

            n_tool_calls += 1

            if n_tool_calls > self.max_tool_calls:
                logger.warning(f"vreasoner: exceeded max vsearcher calls: {n_tool_calls} > {self.max_tool_calls}")
                break

            # Prepare vSearcherLoop.run() kwargs and sampling params
            raw_prompt = [
                {
                    "role": "system",
                    "content": 'You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n**Example**:  \n<tool_call>  \n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n</tool_call>',
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                f"\nLocate {request.region_description}."
                                "\nThink first, call **image_zoom_in_tool** if needed, then answer with the bbox coordinates in [x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can't locate it). "
                                "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  "
                                "<answer>[x1, y1, x2, y2]</answer> (otherwise)"
                            ),
                        },
                    ],
                },
            ]
            vsearcher_kwargs = {
                "raw_prompt": raw_prompt,
                "multi_modal_data": {"image": [kwargs["multi_modal_data"]["image"][0]]},
                "tools_kwargs": kwargs["tools_kwargs"],
                "extra_info": kwargs["extra_info"],
                "parent_job_id": job_id,
                "root_job_id": root_job_id,
                "_validate": validate,
            }

            vsearcher_sampling_params = {
                **sampling_params,
            }
            if validate:
                vsearcher_sampling_params["temperature"] = 0.0

            # Get the bbox from vsearcher output
            with simple_timer("vsearcher_loop.run", profile):
                vsearcher_output = await self.vsearcher_loop.run(vsearcher_sampling_params, **vsearcher_kwargs)
            vsearcher_outputs.append(vsearcher_output)

            bbox = vsearcher_output.extra_fields["final_bbox"]
            if bbox is None:
                logger.warning("vsearcher failed to return a valid bbox")
                break

            # Resize the target region bbox in scale with the original image resolution
            if bbox != (0, 0, 0, 0):
                source_wh = kwargs["extra_info"]["image_processed_wh"][0]
                target_wh = kwargs["extra_info"]["image_ori_wh"][0]
                bbox = resize_bbox(bbox, source_wh, target_wh)

        logger.info(f"vreasoner loop completed: {profile=}")

        # Construct messages for answer evaluation and visualization
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>" + kwargs["extra_info"]["question"]},
            ],
        }

        messages = [user_message]

        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        raw_prompt = self.processor.apply_chat_template(
            messages,
            tools=None,
            add_generation_prompt=False,
            tokenize=False,
            **apply_chat_template_kwargs,
        )

        loop = asyncio.get_event_loop()
        model_inputs = await loop.run_in_executor(
            None, partial(self.processor, text=[raw_prompt], return_tensors="pt")
        )
        prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()

        # Collect response tokens; we left-truncate over-length text so it can fit within response_length
        # This truncated text is only used for answer evaluation (note that max_length_char is in chars, not tokens)
        # We also construct a non-truncated version of the messages (with images) for later visualizing the response
        def maybe_truncate(text: str, max_length_char: int = 4096) -> str:
            """Left-truncate text if it is too long."""
            if len(text) > max_length_char:
                logger.info(f"vreasoner turn response is too long: {len(text)} chars; "
                            f"left-truncating to {max_length_char} chars")
                return "..." + text[-max_length_char:][len("..."):]
            return text

        messages_shortened = messages.copy()
        response_ids_shortened = []
        response_mask_shortened = []
        response_started = False

        for message in messages_api:
            if not isinstance(message, dict):
                message = message.to_dict()

            # Collect image data from the message for visualization
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        _, b64data = image_url.split(",", 1)
                        img = await loop.run_in_executor(
                            None, lambda b: Image.open(io.BytesIO(base64.b64decode(b))), b64data
                        )
                        multi_modal_data["image"].append(img)

            # Skip messages before the response starts
            if message["role"] in ("system", "user") and not response_started:
                continue

            if message["role"] == "assistant":
                response_started = True
                text = message["content"]
                if text is None:
                    logger.warning(f"vreasoner: assistant message content is None")
                    continue
                content = [{"type": "text", "text": text}]
                content_shortened = [{"type": "text", "text": maybe_truncate(text)}]
            elif message["role"] == "user":
                content = [
                    {"type": "text", "text": "<tool_response>"},
                    {"type": "image"},
                    {"type": "text", "text": "</tool_response>"},
                ]
                content_shortened = [
                    {"type": "text", "text": "<tool_response><|vision_start|><|vision_end|></tool_response>"},
                ]
            else:
                logger.warning(f"vreasoner: unexpected message role: {message['role']}")
                continue

            messages.append({"role": message["role"], "content": content})
            messages_shortened.append({"role": message["role"], "content": content_shortened})

            raw_prompt_shortened = self.processor.apply_chat_template(
                messages_shortened,
                tools=None,
                add_generation_prompt=False,
                tokenize=False,
                **apply_chat_template_kwargs,
            )

            model_inputs_shortened = await loop.run_in_executor(
                None, partial(self.processor, text=[raw_prompt_shortened], return_tensors="pt")
            )
            seq_ids_shortened = model_inputs_shortened.pop("input_ids").squeeze(0).tolist()
            new_response_ids = seq_ids_shortened[len(prompt_ids) + len(response_ids_shortened):]
            response_ids_shortened += new_response_ids
            response_mask_shortened += [int(message["role"] == "assistant")] * len(new_response_ids)

        # Left truncate the response if it is still too long
        response_length = self.config.actor_rollout_ref.rollout.response_length
        response_ids_shortened = response_ids_shortened[-response_length:]
        response_mask_shortened = response_mask_shortened[-response_length:]

        # Construct the extra fields for the final output
        extra_fields = ExtraFields(
            agent_name='vreasoner',
            job_id=job_id,
            parent_job_id=kwargs.get("parent_job_id", None),
            root_job_id=root_job_id,
            extra_info=kwargs["extra_info"],
            n_tool_calls=n_tool_calls,
            critical_failure=critical_failure,
            messages=messages,
            multi_modal_data=multi_modal_data,
        )

        extra_fields = asdict(extra_fields)

        # Add extra fields specific to ToolAgentLoop (which vSearcherLoop inherits from)
        # Normally, these fields would be automatically added in AgentLoopWorker._postprocess() when concatenating
        # the outputs of vreasoner and vsearcher. But in some rare cases where there is no vsearcher output in a worker 
        # batch, these fields would be missing and can trigger errors when AgentLoopManager tries to concatenate
        # multiple worker batches (some have vsearcher outputs, some don't)
        extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids_shortened,
            response_mask=response_mask_shortened,
            response_logprobs=None,
            multi_modal_data={},  # we pass multi_modal_data back through extra_fields instead
            reward_score=None,
            num_turns=len(messages_shortened),
            metrics=metrics,
            extra_fields=extra_fields,
            subagent_outputs=vsearcher_outputs,
        )