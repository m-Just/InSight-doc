import asyncio
import base64
import io
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Protocol, NewType, TypeAlias, runtime_checkable
from uuid import uuid4

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from omegaconf import DictConfig
from qwen_vl_utils import extract_vision_info

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    AgentLoopMetrics,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.qwen_agent_loop import QwenAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.profiler import simple_timer
from verl.utils.vsearch import BBox, parse_bbox, resize_bbox, extract_bbox_from_tool_call
from verl.utils.vsearch_gpt_async import get_gpt_visual_search_request

# Qwen3-VL uses relative coordinates in the range [0, 1000) for bounding boxes
# This is different from Qwen2.5-VL which uses absolute pixel coordinates
QWEN3_VL_COORD_RANGE = (1000, 1000)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ExtraFields:
    """Extra fields for VSearcher/VReasoner outputs."""
    agent_name: str
    job_id: str
    parent_job_id: str | None
    root_job_id: str
    extra_info: dict[str, Any]
    
    idx_as_child: int | None = None
    n_tool_calls: int = 0
    caller_feedback: str | None = None
    final_bbox: BBox | None = None
    tool_call_bboxes: list[BBox] | None = None  # Pre-converted bboxes from tool calls (absolute pixel coords)
    critical_failure: bool | None = None  # None if don't care
    messages: list[dict] | None = None
    multi_modal_data: dict[str, Any] | None = None
    failure_reasons: list[str] | None = None


@runtime_checkable
class AgentDataProtocol(Protocol):
    """Protocol for agent data classes that VSearcherMixin can work with."""
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]


ReconstructedMessageContent = NewType("ReconstructedMessageContent", list[dict[str, str]])
ReconstructedMessage: TypeAlias = dict[str, str | ReconstructedMessageContent]
ReconstructedMessages: TypeAlias = list[ReconstructedMessage]

def reconstruct_messages(text: str) -> ReconstructedMessages:
    """Reconstruct messages from text.
    Assuming Qwen2.5/3-VL chat template and multi-modal data only contains images.
    
    Args:
        text: Text to reconstruct messages from
        
    Returns:
        List of reconstructed messages
    """
    messages: ReconstructedMessages = []
    message_strs = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", text, re.DOTALL)
    for s in message_strs:
        content: ReconstructedMessageContent = []
        role, content_str = s.split("\n", 1)
        texts = re.split(r"<\|vision_start\|>.*?<\|vision_end\|>", content_str, re.DOTALL)
        for i, text in enumerate(texts):
            content.append({'type': 'text', 'text': text})
            if i < len(texts) - 1:
                content.append({'type': 'image'})
        messages.append({'role': role, 'content': content})
    return messages


class VSearcherMixin:
    """Mixin that adds VSearcher-specific behavior to agent loops.
    
    This mixin provides:
    - Class constants for token limits and context lengths
    - Pre-run setup (validation mode, sampling params)
    - Post-run processing (bbox extraction, validation cleanup, extra fields)
    - Generation state checks (invalid tokens, pad tokens, eos handling)
    - Complete run() and _handle_generating_state() implementations
    
    Usage:
        @register("vsearcher")
        class VSearcherLoop(VSearcherMixin, ToolAgentLoop):
            AGENT_NAME = "vsearcher"
            
        @register("vsearcher_qwen3vl")
        class VSearcherLoopQwen3VL(VSearcherMixin, QwenAgentLoop):
            AGENT_NAME = "vsearcher_qwen3vl"
    """
    
    AGENT_NAME: str
    DISABLE_TOOL_SCHEMAS: bool
    MAX_ALLOWED_TOKEN_ID: int  # for qwen-vl, this is max(tokenizer.get_vocab().values())
                               # see https://github.com/vllm-project/vllm/issues/13175 for more details
    EXPECTED_VALIDATION_SAMPLING_PARAMS: dict[str, Any]

    VALIDATION_CONTEXT_LENGTH = 32 * 1024
    MAX_TOKENS_PER_TURN = 1024

    def _pre_run(
        self, sampling_params: dict[str, Any], kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, int | None, int | None]:
        """Prepare for VSearcher run.
        
        Args:
            sampling_params: LLM sampling parameters
            kwargs: Run kwargs containing _validate flag
            
        Returns:
            Tuple of (modified_sampling_params, validate, original_prompt_length, original_response_length)
        """
        # Disable tool schemas if needed (when they have been baked into the system prompt)
        if self.DISABLE_TOOL_SCHEMAS:
            self.tool_schemas = None

        assert self.max_parallel_calls == 1, f"max_parallel_calls must be 1, got {self.max_parallel_calls}"

        validate = kwargs["_validate"]
        original_prompt_length = None
        original_response_length = None
        
        if validate:
            # Temporarily override prompt_length and response_length for validation
            original_prompt_length = self.prompt_length
            original_response_length = self.response_length
            self.prompt_length = self.response_length = self.VALIDATION_CONTEXT_LENGTH

        # Validate sampling params
        assert "stop" not in sampling_params, f"{sampling_params['stop']=}"
        if validate:
            assert sampling_params == self.EXPECTED_VALIDATION_SAMPLING_PARAMS, \
                f"expected {self.EXPECTED_VALIDATION_SAMPLING_PARAMS=}, got {sampling_params=}"

        sampling_params = deepcopy(sampling_params)
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = self.MAX_TOKENS_PER_TURN

        logger.info(f"vsearcher {sampling_params=}")

        return sampling_params, validate, original_prompt_length, original_response_length

    def _extract_bbox_within_last_answer_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> BBox | None:
        """Extract final bbox within <answer>...</answer> tags from the last assistant message.
        
        Looks for <answer>...</answer> tags in the last assistant message content.
        Subclasses can override this to perform coordinate conversion.
        
        Args:
            messages: Reconstructed messages from the conversation
            extra_info: Extra info dict containing image dimensions for coordinate conversion.
            
        Returns:
            Parsed bbox in absolute pixel coordinates, or None if not found
        """
        if not messages:
            logger.warning("no messages found, cannot extract bbox")
            return None

        if messages[-1]["role"] != "assistant":
            logger.warning("the last message is not an assistant message, cannot extract bbox")
            return None

        content: ReconstructedMessageContent = messages[-1]["content"]
        assert isinstance(content, list), f"expected list, got {type(content)}"
        assert len(content) == 1, f"expected 1 content item, got {len(content)}"
        assert content[0]["type"] == "text", f"expected text content, got {content[0]['type']}"
        answer_text = content[0]["text"].split("<answer>")[-1].split("</answer>")[0].strip()

        try:
            return parse_bbox(answer_text)
        except Exception as e:
            logger.warning(f"no valid bbox found in vsearcher's answer: {e}")
            return None

    def _extract_bboxes_within_tool_call_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> list[BBox]:
        """Extract bboxes from tool calls in assistant messages.
        
        Looks for <tool_call>...</tool_call> tags in all assistant messages.
        Subclasses can override this to perform coordinate conversion.
        
        Args:
            messages: Reconstructed messages from the conversation
            extra_info: Extra info dict containing image dimensions for coordinate conversion.
            
        Returns:
            List of bboxes from tool calls (in absolute pixel coordinates for base class)
        """
        bboxes = []
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content: ReconstructedMessageContent = msg["content"]
            assert isinstance(content, list), f"expected list, got {type(content)}"
            assert len(content) == 1, f"expected 1 content item, got {len(content)}"
            assert content[0]["type"] == "text", f"expected text content, got {content[0]['type']}"
            text = content[0]["text"]

            if "<tool_call>" not in text:
                continue

            tool_call_text = text.split("<tool_call>", 1)[-1].split("</tool_call>")[0].strip()
            try:
                bbox = extract_bbox_from_tool_call(tool_call_text)
            except Exception as e:
                logger.warning(f"failed to extract bbox from response ({e}): {text}")
                continue

            bboxes.append(bbox)
        
        return bboxes

    def _validation_cleanup(
        self,
        output: AgentLoopOutput,
        original_prompt_length: int,
        original_response_length: int,
    ) -> AgentLoopOutput:
        """Clean up output for validation mode.
        
        Drops image tokens and truncates to fit within original lengths.
        
        Args:
            output: The agent loop output
            original_prompt_length: Original prompt length before validation override
            original_response_length: Original response length before validation override
            
        Returns:
            Modified output
        """
        # Drop all image tokens and truncate
        image_token_id = self.processor.image_token_id
        
        output.prompt_ids = [i for i in output.prompt_ids if i != image_token_id]
        output.response_ids, output.response_mask = map(
            list,
            zip(
                *[
                    (i, j)
                    for i, j in zip(output.response_ids, output.response_mask, strict=False)
                    if i != image_token_id
                ],
                strict=False,
            ),
        )
        output.prompt_ids = output.prompt_ids[-original_prompt_length:]
        output.response_ids = output.response_ids[-original_response_length:]
        output.response_mask = output.response_mask[-original_response_length:]
        
        # Restore original lengths
        self.prompt_length = original_prompt_length
        self.response_length = original_response_length
        
        return output

    def _build_extra_fields(
        self,
        agent_name: str,
        output: AgentLoopOutput,
        kwargs: dict[str, Any],
        final_bbox: BBox | None,
        tool_call_bboxes: list[BBox],
        messages: ReconstructedMessages,
    ) -> dict[str, Any]:
        """Build extra fields for VSearcher output.
        
        Args:
            agent_name: Name of the agent (e.g., 'vsearcher', 'vsearcher_qwen3vl')
            output: The agent loop output
            kwargs: Run kwargs
            final_bbox: Extracted bbox (in absolute pixel coordinates)
            tool_call_bboxes: List of bboxes from tool calls (in absolute pixel coordinates)
            messages: Conversation messages
            
        Returns:
            Extra fields dict
        """
        job_id = uuid4().hex
        
        # Prune images from messages; images are passed back through multi_modal_data
        vision_infos = extract_vision_info(messages)
        for image in vision_infos:
            assert image["type"] in ("image", "image_url"), f"expected image or image_url, got {image['type']}"
            image.pop(image["type"], None)  # can't use `del` here because rollouts share the same input image dict

        extra_fields = ExtraFields(
            agent_name=agent_name,
            job_id=job_id,
            parent_job_id=kwargs.get("parent_job_id", None),
            root_job_id=kwargs.get("root_job_id", job_id),
            extra_info=kwargs["extra_info"],
            n_tool_calls=(output.num_turns - 1) // 2,  # number of successful tool calls = number of user turns
            caller_feedback=None,
            final_bbox=final_bbox,
            tool_call_bboxes=tool_call_bboxes,
            messages=messages,
            multi_modal_data=output.multi_modal_data,
        )

        return asdict(extra_fields)

    def _check_generation(
        self,
        agent_data: AgentDataProtocol,
        terminated_state: Any,
    ) -> Any | None:
        """Check generation state for VSearcher-specific termination conditions.
        
        Args:
            agent_data: Agent data with response_ids, prompt_ids, response_mask
            terminated_state: The terminated state value to return if checks fail
            
        Returns:
            terminated_state if should terminate, None otherwise
        """
        # Check max model length
        max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        if len(agent_data.prompt_ids) >= max_model_len:
            logger.warning(f"prompt len exceeded max model len: {len(agent_data.prompt_ids)=} >= {max_model_len=}")
            return terminated_state

        return None

    def _post_generation_checks(
        self,
        agent_data: AgentDataProtocol,
        terminated_state: Any,
    ) -> Any | None:
        """Post-generation checks for VSearcher.
        
        Args:
            agent_data: Agent data with response_ids
            terminated_state: The terminated state value to return if checks fail
            
        Returns:
            terminated_state if should terminate, None otherwise
        """
        if not agent_data.response_ids:
            return None
            
        # Check for invalid tokens
        if max(agent_data.response_ids) > self.MAX_ALLOWED_TOKEN_ID:
            logger.warning("generated out-of-vocabulary token")
            return terminated_state

        if self.processor.tokenizer.pad_token_id in agent_data.response_ids:
            logger.warning("generated pad token before eos")
            return terminated_state

        # vllm generation may stop at <|im_end|>, but for qwen-vl, a complete message ends with <|im_end|>\n
        # so if the generation stopped at <|im_end|>, we add \n to the end of the response
        if agent_data.response_ids[-1] == self.processor.tokenizer.eos_token_id:
            agent_data.response_ids += [198]
            agent_data.prompt_ids += [198]
            agent_data.response_mask += [0]

        # Check for response length termination
        if len(agent_data.response_mask) >= self.response_length:
            return terminated_state

        return None

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the VSearcher agent loop.
        
        This method handles:
        - Validation mode with extended context length
        - Bbox extraction from response text
        - Extra fields compatible with VSearcher format
        
        Args:
            sampling_params: LLM sampling parameters
            **kwargs: Dataset fields including raw_prompt, multi_modal_data, extra_info, etc.
            
        Returns:
            AgentLoopOutput with VSearcher extra fields
        """
        # Store extra_info for coordinate conversion (used by subclasses)
        self._extra_info = kwargs["extra_info"]

        # Pre-run setup using mixin
        sampling_params, validate, original_prompt_length, original_response_length = \
            self._pre_run(sampling_params, kwargs)

        # Run the parent agent loop
        try:
            output = await super().run(sampling_params, **kwargs)
        except ValueError as e:
            if "absolute aspect ratio must be smaller than 200" in str(e):
                logger.warning(f"invalid aspect ratio: {e}")
            else:
                raise e

        # Construct full messages from token IDs
        all_ids = output.prompt_ids + output.response_ids
        all_text = await asyncio.get_event_loop().run_in_executor(
            None, partial(self.tokenizer.decode, all_ids, skip_special_tokens=False)
        )
        try:
            messages = reconstruct_messages(all_text)
        except Exception as e:
            logger.warning(f"failed to reconstruct messages: {e}")
            messages = [{"role": "error", "content": str(e)}]

        # Extract final bbox from messages (subclasses may convert coordinates)
        final_bbox = self._extract_bbox_within_last_answer_tags(messages, self._extra_info)

        # Extract tool call bboxes from messages (subclasses may convert coordinates)
        tool_call_bboxes = self._extract_bboxes_within_tool_call_tags(messages, self._extra_info)

        # Validation cleanup
        if validate:
            output = self._validation_cleanup(
                output, original_prompt_length, original_response_length
            )

        # Build extra fields (including pre-converted tool_call_bboxes)
        extra_fields = self._build_extra_fields(
            agent_name=self.AGENT_NAME,
            output=output,
            kwargs=kwargs,
            final_bbox=final_bbox,
            tool_call_bboxes=tool_call_bboxes,
            messages=messages,
        )
        output.extra_fields.update(extra_fields)

        return output

    async def _handle_generating_state(
        self, agent_data: AgentDataProtocol, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> Any:
        """Handle the generating state with VSearcher-specific checks.
        
        Args:
            agent_data: Agent data with response_ids, prompt_ids, response_mask
            sampling_params: LLM sampling parameters
            ignore_termination: If True, skip turn-based termination checks
            
        Returns:
            Next agent state
        """
        # Pre-generation checks
        termination = self._check_generation(agent_data, AgentState.TERMINATED)
        if termination is not None:
            return termination

        # Call parent's generating state handler
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        # Post-generation checks
        termination = self._post_generation_checks(agent_data, AgentState.TERMINATED)
        if termination is not None:
            return termination

        return state


@register("vsearcher")
class VSearcherLoop(VSearcherMixin, ToolAgentLoop):
    """Agent loop for the vsearcher agent, assuming the underlying model is Qwen2.5-VL.
    
    Uses VSearcherMixin for run() and _handle_generating_state() implementations.
    """
    AGENT_NAME = "vsearcher"
    DISABLE_TOOL_SCHEMAS = True
    MAX_ALLOWED_TOKEN_ID = 151664
    EXPECTED_VALIDATION_SAMPLING_PARAMS = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": None,
    }


@register("vsearcher_qwen3_vl")
class VSearcherLoopQwen3VL(VSearcherMixin, QwenAgentLoop):
    """Agent loop for the vsearcher agent using qwen_agent's tool system.
    
    This class provides the same interface as VSearcherLoop but uses QwenAgentLoop
    as its base, leveraging qwen_agent's TOOL_REGISTRY for tool management.
    
    Uses VSearcherMixin for run() and _handle_generating_state() implementations.
    The underlying model is assumed to be Qwen3-VL or compatible.
    
    Qwen3-VL outputs bboxes in relative 0-1000 coordinates. This class overrides
    the bbox extraction methods to convert to absolute pixel coordinates so that
    the rest of verl always receives pixel coordinates.
    """
    AGENT_NAME = "vsearcher_qwen3_vl"
    DISABLE_TOOL_SCHEMAS = False
    MAX_ALLOWED_TOKEN_ID = 151668
    EXPECTED_VALIDATION_SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "logprobs": None,
    }  # https://hf.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/generation_config.json

    def _extract_bbox_within_last_answer_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> BBox | None:
        """Extract bbox from messages and convert from 0-1000 to pixel coordinates.
        
        Qwen3-VL outputs bboxes in relative 0-1000 coordinates. This method converts
        them to absolute pixel coordinates based on the processed image size.
        
        Args:
            messages: Reconstructed messages from the conversation
            extra_info: Extra info dict containing image_processed_wh for coordinate conversion.
            
        Returns:
            Parsed bbox in absolute pixel coordinates, or None if not found
        """
        bbox = super()._extract_bbox_within_last_answer_tags(messages, extra_info)
        if bbox is None or bbox == (0, 0, 0, 0):
            return bbox
        
        image_processed_wh: tuple[int, int] = extra_info["image_processed_wh"][0]
        return resize_bbox(bbox, QWEN3_VL_COORD_RANGE, image_processed_wh)

    def _extract_bboxes_within_tool_call_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> list[BBox]:
        """Extract tool call bboxes and convert from 0-1000 to pixel coordinates.
        
        Qwen3-VL outputs bboxes in relative 0-1000 coordinates. This method converts
        them to absolute pixel coordinates based on the processed image size.
        
        Args:
            messages: Reconstructed messages from the conversation
            extra_info: Extra info dict containing image_processed_wh for coordinate conversion.
            
        Returns:
            List of bboxes from tool calls in absolute pixel coordinates
        """
        # First extract the raw bboxes using parent's method
        bboxes = super()._extract_bboxes_within_tool_call_tags(messages, extra_info)
        if not bboxes:
            return bboxes
        
        # Convert all bboxes from 0-1000 relative coords to pixel coords
        converted_bboxes = []
        image_processed_wh: tuple[int, int] = extra_info["image_processed_wh"][0]
        for bbox in bboxes:
            if bbox == (0, 0, 0, 0):
                converted_bboxes.append(bbox)
            else:
                converted_bboxes.append(resize_bbox(bbox, QWEN3_VL_COORD_RANGE, image_processed_wh))
        return converted_bboxes


@register("vreasoner")
class VReasonerLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,                  # tokenizer for vSearcher
        processor: AutoProcessor,                  # processor for vSearcher
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
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
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs)
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_round_retries = max_round_retries
        self.max_round_retries_val = max_round_retries_val
        self.gpt_image_max_area = gpt_image_max_area
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.enable_tool_feedback = enable_tool_feedback
        vsearcher_loop_cls_name = self.config.actor_rollout_ref.rollout.agent.get("vsearcher_loop_cls", "VSearcherLoop")
        if vsearcher_loop_cls_name == "VSearcherLoop":
            vsearcher_loop_cls = VSearcherLoop
        elif vsearcher_loop_cls_name == "VSearcherLoopQwen3VL":
            vsearcher_loop_cls = VSearcherLoopQwen3VL
        else:
            raise ValueError(f"Invalid vsearcher loop class: {vsearcher_loop_cls_name}")
        self.vsearcher_loop = vsearcher_loop_cls(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
        )

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],  # sampling params for vSearcher
        **kwargs,
    ) -> AgentLoopOutput:
        job_id = uuid4().hex
        root_job_id = kwargs.get("root_job_id", job_id)

        validate = kwargs["_validate"]

        messages_api = []
        bbox = None
        n_tool_calls = 0
        vsearcher_outputs = []
        critical_failure = False
        multi_modal_data = {"images": []}

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
            # Extract vision info from raw_prompt messages using qwen_vl_utils
            vision_infos = extract_vision_info(kwargs["raw_prompt"])
            assert len(vision_infos) == 1, f"expected 1 vision info, got {len(vision_infos)}"
            image = vision_infos[0]
            assert image["type"] in ("image", "image_url"), f"expected image or image_url, got {image['type']}"


            if self.vsearcher_loop.AGENT_NAME == "vsearcher":
                system_prompt = 'You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n**Example**:  \n<tool_call>  \n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n</tool_call>',
            elif self.vsearcher_loop.AGENT_NAME == "vsearcher_qwen3_vl":
                system_prompt = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""
            else:
                raise ValueError(f"Invalid vsearcher loop class: {self.vsearcher_loop.AGENT_NAME}")

            raw_prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": image["type"], image["type"]: image[image["type"]]},
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
                # "multi_modal_data": {"image": [kwargs["multi_modal_data"]["image"][0]]},
                "tools_kwargs": kwargs["tools_kwargs"],
                "extra_info": kwargs["extra_info"],
                "parent_job_id": job_id,
                "root_job_id": root_job_id,
                "_validate": validate,
            }

            vsearcher_sampling_params = {
                **sampling_params,
            }

            # Get the bbox from vsearcher output
            with simple_timer("vsearcher_loop.run", profile):
                vsearcher_output = await self.vsearcher_loop.run(vsearcher_sampling_params, **vsearcher_kwargs)
                vsearcher_output.extra_fields["idx_as_child"] = len(vsearcher_outputs)
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
                        multi_modal_data["images"].append(img)

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
            failure_reasons=request.failure_reasons if critical_failure else None,
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


@register("vreasoner_qwen3_vl")
class VReasonerLoopQwen3VL(VSearcherLoopQwen3VL):
    AGENT_NAME = "vreasoner_qwen3_vl"
    MAX_TOKENS_PER_TURN = 4096

    def _extract_bbox_within_last_answer_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> BBox | None:
        return None