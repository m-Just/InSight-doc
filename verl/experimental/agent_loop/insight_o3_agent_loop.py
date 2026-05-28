import asyncio
import base64
import io
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Protocol, NewType, TypeAlias, runtime_checkable
from uuid import uuid4

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from omegaconf import DictConfig
from qwen_vl_utils import extract_vision_info, fetch_image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    AgentLoopMetrics,
    DictConfigWrap,
    register,
    resolve_dynamic_initial_rescale,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.qwen_agent_loop import QwenAgentLoop
from verl.experimental.agent_loop.presented_images import (
    PresentedImageState as PresentedImage,
    build_processed_child_image as _build_processed_child_image,
    cap_size_by_area as _cap_size_by_area,
    clamp_bbox_to_image as _clamp_bbox_to_image,
    resample_original_region as _resample_original_region,
    resize_bbox_by_rounding as _resize_bbox_by_rounding,
    resize_dims_by_factor as _resize_dims_by_factor,
    translate_bbox_to_original as _translate_bbox_to_original,
    translate_processed_bbox_to_original as _translate_processed_bbox_to_original,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.profiler import simple_timer
from verl.utils.vsearch import BBox, parse_bbox, resize_bbox, extract_bbox_from_tool_call
from verl.utils.vsearch_gpt_async import get_gpt_visual_search_request
from verl.utils.vsearch_gpt_async_v2 import ToolResult, get_gpt_visual_search_request_v2
from verl.utils.vreasoner_v2_conversation_export import (
    build_child_conversation_export_id,
    build_export_record,
    export_conversation,
)
from verl.utils.vsearch_profile import write_profile_event

# Qwen3-VL uses relative coordinates in the range [0, 1000) for bounding boxes
# This is different from Qwen2.5-VL which uses absolute pixel coordinates
QWEN3_VL_COORD_RANGE = (1000, 1000)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

UNANSWERABLE_ANSWER_VERIFICATION_HINT = (
    "Verification check: make sure your answer addresses the exact target in the question.\n"
    "If you answered a nearby related entity, field, unit, rank, year, or method instead of the exact target, "
    "correct yourself.\n"
    "If the document does not support the exact target as asked, say so directly in one sentence.\n"
    "Do not provide the answer for a similar substitute target.\n"
    "Return only a revised final answer in <answer>...</answer>."
)
UNANSWERABLE_EXPLICIT_REVISION_HINT = (
    "Verification check: this question is unanswerable as written from the provided document.\n"
    "Make sure your answer addresses the exact target in the question, and revise your final answer to reflect that, "
    "if not already so.\n"
    "If you answered a nearby related entity, field, unit, rank, year, or method instead of the exact target, "
    "correct yourself.\n"
    "If the document does not support the exact target as asked, say so directly in one sentence.\n"
    "Do not provide the answer for a similar substitute target.\n"
    "Return only a revised final answer in <answer>...</answer>."
)
UNANSWERABLE_EXPLICIT_REVISION_HINT_EVIDENCE_LINKED = (
    "Verification check: this question is unanswerable as written from the provided document.\n"
    "Revise your final answer to reflect that, if not already so.\n"
    "Base the revision on the relevant information you found in the document.\n"
    "If the document shows related information that does not answer the question as asked, briefly explain that "
    "mismatch in a natural way.\n"
    "Avoid generic 'insufficient information' answers when the document actually shows evidence that is related but "
    "does not answer the exact question.\n"
    "Do not answer a similar substitute target as if it were the requested one.\n"
    "Return only a revised final answer in <answer>...</answer>."
)
ANSWER_VERIFICATION_HINT_EXPORT_TYPE = "answer_verification_hint"
ANSWER_REVISION_EXPORT_TYPE = "answer_revision"


def _looks_like_multipart_question(question: str | None) -> bool:
    if not question:
        return False
    return len(re.findall(r"\([a-z]\)", question.lower())) >= 2


def get_unanswerable_answer_verification_hint(mode: str, extra_info: dict[str, Any] | None = None) -> str:
    extra_info = extra_info or {}
    if mode == "soft":
        return UNANSWERABLE_ANSWER_VERIFICATION_HINT
    if mode == "explicit_unanswerable_revision":
        return UNANSWERABLE_EXPLICIT_REVISION_HINT
    if mode == "explicit_unanswerable_revision_evidence_linked":
        multipart = _looks_like_multipart_question(str(extra_info.get("question") or ""))
        selected_part_label = extra_info.get("selected_part_label")
        if multipart:
            selected_part_msg = ""
            if isinstance(selected_part_label, str) and selected_part_label.strip():
                selected_part_msg = f" The unsupported sub-question is part ({selected_part_label.strip()})."
            return (
                "Verification check: this is a multipart question, and one sub-question is unanswerable as written "
                f"from the provided document.{selected_part_msg}\n"
                "Keep the answerable sub-questions unchanged, and revise only the unsupported sub-question, if not "
                "already so.\n"
                "Base the revision on the relevant information you found in the document.\n"
                "If the document shows related information that does not answer that sub-question as asked, briefly "
                "explain that mismatch in a natural way.\n"
                "Avoid generic 'insufficient information' answers when the document actually shows evidence that is "
                "related but does not answer the exact sub-question.\n"
                "Do not answer a similar substitute target as if it were the requested one.\n"
                "Return only a revised final answer in <answer>...</answer>."
            )
        return UNANSWERABLE_EXPLICIT_REVISION_HINT_EVIDENCE_LINKED
    raise ValueError(f"Unsupported unanswerable answer verification mode: {mode}")


@dataclass
class ExtraFields:
    """Extra fields for VSearcher/VReasoner outputs."""
    agent_name: str
    job_id: str
    parent_job_id: str | None
    root_job_id: str
    extra_info: dict[str, Any]
    
    idx_as_child: int | None = None
    img_idx: int | None = None  # For multi-image VReasoner: which input image this VSearcher operated on (0-indexed)
    n_tool_calls: int = 0
    caller_feedback: str | None = None
    final_bbox: BBox | None = None
    tool_call_bboxes: list[BBox] | None = None  # Pre-converted bboxes from tool calls (absolute pixel coords)
    critical_failure: bool | None = None  # None if don't care
    messages: list[dict] | None = None
    multi_modal_data: dict[str, Any] | None = None
    failure_reasons: list[str] | None = None
    conversation_export_json_path: str | None = None


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


def _process_presented_image(image: Image.Image, max_pixels: int | None) -> Image.Image:
    fetch_kwargs: dict[str, Any] = {"image": image}
    if max_pixels is not None:
        fetch_kwargs["max_pixels"] = max_pixels
    return fetch_image(fetch_kwargs)


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
        t_start = time.perf_counter()
        # Store extra_info for coordinate conversion (used by subclasses)
        self._extra_info = kwargs["extra_info"]

        # Pre-run setup using mixin
        sampling_params, validate, original_prompt_length, original_response_length = \
            self._pre_run(sampling_params, kwargs)

        # Run the parent agent loop
        t_before_parent = time.perf_counter()
        try:
            output = await super().run(sampling_params, **kwargs)
        except ValueError as e:
            if "absolute aspect ratio must be smaller than 200" in str(e):
                logger.warning(f"invalid aspect ratio: {e}")
                job_id = uuid4().hex
                return AgentLoopOutput(
                    prompt_ids=[],
                    response_ids=[],
                    response_mask=[],
                    response_logprobs=None,
                    multi_modal_data={},
                    reward_score=None,
                    num_turns=0,
                    metrics=AgentLoopMetrics(),
                    extra_fields={
                        "agent_name": self.AGENT_NAME,
                        "job_id": job_id,
                        "parent_job_id": kwargs.get("parent_job_id", None),
                        "root_job_id": kwargs.get("root_job_id", job_id),
                        "n_tool_calls": 0,
                        "caller_feedback": None,
                        "final_bbox": None,
                        "tool_call_bboxes": [],
                        "critical_failure": True,
                        "messages": [],
                        "multi_modal_data": {},
                        "failure_reasons": [f"invalid_aspect_ratio: {e}"],
                        "extra_info": kwargs.get("extra_info"),
                    },
                )
            else:
                raise e
        t_after_parent = time.perf_counter()

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
        t_end = time.perf_counter()
        write_profile_event(
            "vsearcher_sample",
            {
                "event": "vsearcher_sample",
                "agent_name": self.AGENT_NAME,
                "validate": validate,
                "job_id": output.extra_fields.get("job_id"),
                "parent_job_id": output.extra_fields.get("parent_job_id"),
                "root_job_id": output.extra_fields.get("root_job_id"),
                "idx_as_child": output.extra_fields.get("idx_as_child"),
                "img_idx": output.extra_fields.get("img_idx"),
                "final_bbox": final_bbox,
                "tool_call_count": len(tool_call_bboxes or []),
                "prompt_tokens": len(output.prompt_ids),
                "response_tokens": len(output.response_ids),
                "num_turns": output.num_turns,
                "timing_s": {
                    "pre_run": t_before_parent - t_start,
                    "parent_agent_run": t_after_parent - t_before_parent,
                    "decode_and_extract": t_end - t_after_parent,
                    "total": t_end - t_start,
                },
                "metrics": output.metrics.model_dump(),
            },
            config=self.config,
        )

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
    DEFAULT_QWEN_TOOL_LIST = ["image_zoom_in_tool"]
    DISABLE_TOOL_SCHEMAS = False
    MAX_ALLOWED_TOKEN_ID = 151668
    EXPECTED_VALIDATION_SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
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
        try:
            return resize_bbox(bbox, QWEN3_VL_COORD_RANGE, image_processed_wh)
        except ValueError as e:
            logger.warning(
                "failed to resize final bbox from Qwen3-VL coords to image pixels: "
                f"bbox={bbox}, image_processed_wh={image_processed_wh}, error={e}"
            )
            return None

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
                try:
                    converted_bboxes.append(resize_bbox(bbox, QWEN3_VL_COORD_RANGE, image_processed_wh))
                except ValueError as e:
                    logger.warning(
                        "failed to resize tool-call bbox from Qwen3-VL coords to image pixels: "
                        f"bbox={bbox}, image_processed_wh={image_processed_wh}, error={e}"
                    )
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
        prompt_variant: str = "default",           # system prompt variant for vReasoner
        enable_unanswerable_answer_verification: bool = False,
        unanswerable_answer_verification_mode: str = "soft",
        enable_tool_feedback: bool = True,         # enable tool feedback for vReasoner
        multi_image_input: bool = False,           # support multiple input images per query
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
        self.prompt_variant = prompt_variant
        self.enable_unanswerable_answer_verification = enable_unanswerable_answer_verification
        self.unanswerable_answer_verification_mode = unanswerable_answer_verification_mode
        self.enable_tool_feedback = enable_tool_feedback
        self.multi_image_input = multi_image_input
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
        conversation_export_id = kwargs.get(
            "conversation_export_id",
            kwargs.get("extra_info", {}).get("conversation_export_id", job_id),
        )

        validate = kwargs["_validate"]

        messages_api = []
        bbox = None
        n_tool_calls = 0
        last_img_idx = None
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
                    original_images=kwargs["extra_info"]["image_ori"],
                    messages=messages_api,
                    bbox=bbox,
                    model=self.model,
                    max_tool_calls=self.max_tool_calls,
                    max_round_retries=self.max_round_retries if not validate else self.max_round_retries_val,
                    gpt_image_max_area=self.gpt_image_max_area,
                    max_completion_tokens=self.max_completion_tokens,
                    reasoning_effort=self.reasoning_effort,
                    enable_tool_feedback=(False if validate else self.enable_tool_feedback),
                    multi_image_input=self.multi_image_input,
                    last_img_idx=last_img_idx,
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
            image_infos = [v for v in vision_infos if v["type"] in ("image", "image_url")]

            if self.multi_image_input:
                img_idx = request.img_idx if request.img_idx is not None else 0
                assert 0 <= img_idx < len(image_infos), (
                    f"img_idx {img_idx} out of range for {len(image_infos)} images"
                )
                image = image_infos[img_idx]
                last_img_idx = img_idx
            else:
                assert len(image_infos) == 1, f"expected 1 image, got {len(image_infos)}"
                image = image_infos[0]


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
            if self.multi_image_input:
                tools_kwargs_for_vsearcher = {
                    **kwargs["tools_kwargs"],
                    "image_zoom_in_tool": {
                        "create_kwargs": {
                            **kwargs["tools_kwargs"].get("image_zoom_in_tool", {}).get("create_kwargs", {}),
                            "image": kwargs["extra_info"]["image_ori"][img_idx],
                            "resized_image_size": kwargs["extra_info"]["image_processed_wh"][img_idx],
                        }
                    },
                }
                extra_info_for_vsearcher = {
                    **kwargs["extra_info"],
                    "image_ori": [kwargs["extra_info"]["image_ori"][img_idx]],
                    "image_ori_wh": [kwargs["extra_info"]["image_ori_wh"][img_idx]],
                    "image_processed_wh": [kwargs["extra_info"]["image_processed_wh"][img_idx]],
                }
            else:
                tools_kwargs_for_vsearcher = kwargs["tools_kwargs"]
                extra_info_for_vsearcher = kwargs["extra_info"]

            vsearcher_kwargs = {
                "raw_prompt": raw_prompt,
                "tools_kwargs": tools_kwargs_for_vsearcher,
                "extra_info": extra_info_for_vsearcher,
                "parent_job_id": job_id,
                "root_job_id": root_job_id,
                "conversation_export_id": build_child_conversation_export_id(
                    conversation_export_id,
                    len(vsearcher_outputs),
                ),
                "_validate": validate,
            }

            vsearcher_sampling_params = {
                **sampling_params,
            }

            # Get the bbox from vsearcher output
            with simple_timer("vsearcher_loop.run", profile):
                vsearcher_output = await self.vsearcher_loop.run(vsearcher_sampling_params, **vsearcher_kwargs)
                vsearcher_output.extra_fields["idx_as_child"] = len(vsearcher_outputs)
                if self.multi_image_input:
                    vsearcher_output.extra_fields["img_idx"] = img_idx
            vsearcher_outputs.append(vsearcher_output)

            bbox = vsearcher_output.extra_fields.get("final_bbox")
            if bbox is None:
                logger.warning("vsearcher failed to return a valid bbox")
                break

            # Resize the target region bbox in scale with the original image resolution
            if bbox != (0, 0, 0, 0):
                resize_idx = img_idx if self.multi_image_input else 0
                source_wh = kwargs["extra_info"]["image_processed_wh"][resize_idx]
                target_wh = kwargs["extra_info"]["image_ori_wh"][resize_idx]
                bbox = resize_bbox(bbox, source_wh, target_wh)

        logger.info(f"vreasoner loop completed: {profile=}")

        # Construct messages for answer evaluation and visualization
        if self.multi_image_input:
            n_images = len(kwargs["extra_info"]["image_ori"])
            vision_tokens = "".join("<|vision_start|><|vision_end|>" for _ in range(n_images))
        else:
            vision_tokens = "<|vision_start|><|vision_end|>"
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_tokens + kwargs["extra_info"]["question"]},
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
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
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
                has_image = (
                    isinstance(message["content"], list)
                    and any(c.get("type") == "image_url" for c in message["content"])
                )
                if has_image:
                    content = [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image"},
                        {"type": "text", "text": "</tool_response>"},
                    ]
                    content_shortened = [
                        {"type": "text", "text": "<tool_response><|vision_start|><|vision_end|></tool_response>"},
                    ]
                else:
                    text = message["content"] if isinstance(message["content"], str) else " ".join(
                        c.get("text", "") for c in message["content"] if c.get("type") == "text"
                    )
                    content = [{"type": "text", "text": text}]
                    content_shortened = [{"type": "text", "text": maybe_truncate(text)}]
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


@register("vreasoner_v2")
class VReasonerLoopV2(VReasonerLoop):
    DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND = 0

    @staticmethod
    def _is_not_answerable_question_type(question_type: Any) -> bool:
        if question_type == "not-answerable":
            return True
        if isinstance(question_type, (list, tuple, set)):
            return "not-answerable" in question_type
        return False

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        initial_rescale: float = 0.25,
        initial_input_pixels_lower_bound: int = DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND,
        region_zoom_in_factor: float = 4.0,
        png_max_area: int = 1280 * 1280,
        enable_stop: bool = False,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
            **kwargs,
        )
        if initial_rescale <= 0:
            raise ValueError(f"initial_rescale must be positive, got {initial_rescale}")
        if initial_input_pixels_lower_bound < 0:
            raise ValueError(
                f"initial_input_pixels_lower_bound must be non-negative, got {initial_input_pixels_lower_bound}"
            )
        if region_zoom_in_factor <= 0:
            raise ValueError(f"region_zoom_in_factor must be positive, got {region_zoom_in_factor}")
        if png_max_area <= 0:
            raise ValueError(f"png_max_area must be positive, got {png_max_area}")
        self.initial_rescale = initial_rescale
        self.initial_input_pixels_lower_bound = initial_input_pixels_lower_bound
        self.region_zoom_in_factor = region_zoom_in_factor
        self.png_max_area = png_max_area
        self.enable_stop = enable_stop
        self.conversation_export_dir = self.config.actor_rollout_ref.rollout.agent.get(
            "vreasoner_v2_conversation_export_dir",
            None,
        )

    def _build_initial_presented_images(
        self,
        original_images: list[Image.Image],
    ) -> tuple[list[PresentedImage], float]:
        actual_initial_rescale = resolve_dynamic_initial_rescale(
            image_sizes=[image.size for image in original_images],
            configured_initial_rescale=self.initial_rescale,
            total_pixels_lower_bound=self.initial_input_pixels_lower_bound,
            per_image_max_area=self.gpt_image_max_area,
        )
        presented_images: list[PresentedImage] = []
        for img_idx, image in enumerate(original_images):
            bbox_on_original = (0, 0, image.size[0], image.size[1])
            target_display_size = _cap_size_by_area(
                _resize_dims_by_factor(image.size, actual_initial_rescale),
                self.gpt_image_max_area,
            )
            displayed = image.resize(target_display_size, Image.LANCZOS) if image.size != target_display_size else image.copy()
            presented_images.append(
                PresentedImage(
                    image=displayed,
                    source_original_img_idx=img_idx,
                    bbox_on_original=bbox_on_original,
                    display_size=target_display_size,
                )
            )
        return presented_images, actual_initial_rescale

    def _append_presented_image(
        self,
        presented_images: list[PresentedImage],
        source_original: Image.Image,
        source_original_img_idx: int,
        bbox_on_original: BBox,
        target_display_size: tuple[int, int],
    ) -> int | None:
        resampled = _resample_original_region(
            source_original,
            bbox_on_original,
            target_display_size,
            self.gpt_image_max_area,
        )
        if resampled is None:
            return None
        display_image, display_size = resampled
        presented_images.append(
            PresentedImage(
                image=display_image,
                source_original_img_idx=source_original_img_idx,
                bbox_on_original=bbox_on_original,
                display_size=display_size,
            )
        )
        return len(presented_images) - 1

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs,
    ) -> AgentLoopOutput:
        t_sample_start = time.perf_counter()
        job_id = uuid4().hex
        root_job_id = kwargs.get("root_job_id", job_id)
        conversation_export_id = kwargs.get(
            "conversation_export_id",
            kwargs.get("extra_info", {}).get("conversation_export_id", job_id),
        )

        validate = kwargs["_validate"]

        messages_api = []
        n_tool_calls = 0
        vsearcher_outputs = []
        critical_failure = False
        multi_modal_data = {"images": []}

        metrics = AgentLoopMetrics()
        profile = {}

        original_images = kwargs["extra_info"]["image_ori"]
        presented_images, actual_initial_rescale = self._build_initial_presented_images(original_images)
        presented_image_refs = [
            {
                "presented_img_idx": img_idx,
                "kind": "initial",
                "source_original_img_idx": presented.source_original_img_idx,
                "bbox_on_original": list(presented.bbox_on_original),
                "display_size": list(presented.display_size),
                "original_size": list(original_images[presented.source_original_img_idx].size),
                "initial_rescale": actual_initial_rescale,
            }
            for img_idx, presented in enumerate(presented_images)
        ]
        tool_result = None
        request = None
        failure_events: list[dict[str, Any]] = []

        base_zoom_create_kwargs = kwargs["tools_kwargs"].get("image_zoom_in_tool", {}).get("create_kwargs", {})
        max_pixels = base_zoom_create_kwargs.get("max_pixels")
        api_rounds = []
        child_runs = []

        while True:
            api_round_idx = len(api_rounds)
            t_api_start = time.perf_counter()
            with simple_timer("api_calls", profile):
                request = await get_gpt_visual_search_request_v2(
                    initial_question=kwargs["extra_info"]["question"],
                    presented_images=[item.image for item in presented_images],
                    messages=messages_api,
                    model=self.model,
                    max_tool_calls=self.max_tool_calls,
                    max_round_retries=self.max_round_retries if not validate else self.max_round_retries_val,
                    gpt_image_max_area=self.gpt_image_max_area,
                    png_max_area=self.png_max_area,
                    max_completion_tokens=self.max_completion_tokens,
                    reasoning_effort=self.reasoning_effort,
                    tool_result=tool_result,
                    enable_stop=self.enable_stop,
                    prompt_variant=self.prompt_variant,
                )
            t_api_end = time.perf_counter()
            api_round_event = {
                "event": "vreasoner_v2_api_round",
                "job_id": job_id,
                "root_job_id": root_job_id,
                "conversation_export_id": conversation_export_id,
                "validate": validate,
                "round_idx": api_round_idx,
                "prior_tool_calls": n_tool_calls,
                "presented_image_count": len(presented_images),
                "success": request.success,
                "is_last_round": request.is_last_round,
                "requested_region_description": request.region_description,
                "requested_img_idx": request.img_idx,
                "has_answer": request.answer is not None,
                "failure_reasons": list(request.failure_reasons or []),
                "timing_s": {"api_round": t_api_end - t_api_start},
            }
            api_rounds.append(api_round_event)
            write_profile_event("vreasoner_v2_api", api_round_event, config=self.config)

            if request.failure_reasons:
                failure_events.append(
                    {
                        "kind": "assistant_generation_or_parsing",
                        "request_success": request.success,
                        "failure_reasons": list(request.failure_reasons),
                    }
                )

            if not request.success:
                logger.warning("API generation failed")
                critical_failure = True
                break

            messages_api = request.messages

            if request.region_description is None and request.img_idx is None:
                break

            n_tool_calls += 1
            assert n_tool_calls <= self.max_tool_calls, (
                f"vreasoner_v2: executed more tool calls than allowed: {n_tool_calls} > {self.max_tool_calls}"
            )

            img_idx = request.img_idx if request.img_idx is not None else 0
            if not (0 <= img_idx < len(presented_images)):
                logger.warning("received out-of-range img_idx %s for %s presented images", img_idx, len(presented_images))
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=f"ERROR: Image {img_idx} is not available. Choose an img_idx from the currently visible images.",
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                    }
                )
                continue
            presented = presented_images[img_idx]
            source_original = original_images[presented.source_original_img_idx]

            tool_result = None

            if request.region_description is None:
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=(
                        "ERROR: region_description is required. This zoom tool only supports region-based zoom requests."
                    ),
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                    }
                )
                continue

            if self.vsearcher_loop.AGENT_NAME == "vsearcher":
                system_prompt = 'You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n**Example**:  \n<tool_call>  \n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n</tool_call>'
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

            child_images = _build_processed_child_image(
                source_original,
                presented.bbox_on_original,
                max_pixels,
            )
            if child_images is None:
                raise RuntimeError(
                    "failed to build child vsearcher image from the original region represented by the presented image"
                )
            child_source_image, processed_presented = child_images
            processed_presented_wh = processed_presented.size
            raw_prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_presented},
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

            create_kwargs = {
                **base_zoom_create_kwargs,
                "image": child_source_image,
                "resized_image_size": processed_presented_wh,
            }
            tools_kwargs_for_vsearcher = {
                **kwargs["tools_kwargs"],
                "image_zoom_in_tool": {
                    **kwargs["tools_kwargs"].get("image_zoom_in_tool", {}),
                    "create_kwargs": create_kwargs,
                },
            }
            extra_info_for_vsearcher = {
                **kwargs["extra_info"],
                "image_ori": [child_source_image],
                "image_ori_wh": [child_source_image.size],
                "image_processed_wh": [processed_presented_wh],
            }
            vsearcher_kwargs = {
                "raw_prompt": raw_prompt,
                "tools_kwargs": tools_kwargs_for_vsearcher,
                "extra_info": extra_info_for_vsearcher,
                "parent_job_id": job_id,
                "root_job_id": root_job_id,
                "_validate": validate,
            }

            with simple_timer("vsearcher_loop.run", profile):
                t_child_start = time.perf_counter()
                vsearcher_output = await self.vsearcher_loop.run(dict(sampling_params), **vsearcher_kwargs)
                t_child_end = time.perf_counter()
                vsearcher_output.extra_fields["idx_as_child"] = len(vsearcher_outputs)
                vsearcher_output.extra_fields["img_idx"] = img_idx
            vsearcher_outputs.append(vsearcher_output)
            child_event = {
                "event": "vreasoner_v2_child_vsearcher",
                "job_id": job_id,
                "root_job_id": root_job_id,
                "child_idx": len(vsearcher_outputs) - 1,
                "child_job_id": vsearcher_output.extra_fields.get("job_id"),
                "requested_img_idx": img_idx,
                "requested_region_description": request.region_description,
                "final_bbox": vsearcher_output.extra_fields.get("final_bbox"),
                "tool_call_count": len(vsearcher_output.extra_fields.get("tool_call_bboxes") or []),
                "prompt_tokens": len(vsearcher_output.prompt_ids),
                "response_tokens": len(vsearcher_output.response_ids),
                "num_turns": vsearcher_output.num_turns,
                "timing_s": {"child_vsearcher": t_child_end - t_child_start},
                "metrics": vsearcher_output.metrics.model_dump(),
            }
            child_runs.append(child_event)
            write_profile_event("vreasoner_v2_child", child_event, config=self.config)

            bbox = vsearcher_output.extra_fields.get("final_bbox")
            error_message_due_to_vsearcher_failure = (
                "ERROR: The requested region could not be turned into a usable zoomed view. "
                "Try a more specific region description, or choose a different img_idx."
            )
            if bbox is None:
                logger.warning("vsearcher failed to return a valid bbox")
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=error_message_due_to_vsearcher_failure,
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                        "region_description": request.region_description,
                    }
                )
                continue

            if bbox == (0, 0, 0, 0):
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=(
                        "ERROR: The requested region could not be located in the selected image. The description may be "
                        "too vague, too broad, too specific, or mismatched to the visible content."
                    ),
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                        "region_description": request.region_description,
                    }
                )
                continue

            bbox_on_presented = _resize_bbox_by_rounding(bbox, processed_presented_wh, presented.display_size)
            if bbox_on_presented is None:
                logger.warning("failed to project bbox %s onto presented image %s", bbox, processed_presented_wh)
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=error_message_due_to_vsearcher_failure,
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                        "region_description": request.region_description,
                        "bbox_from_vsearcher": list(bbox),
                    }
                )
                continue

            bbox_on_original = _translate_processed_bbox_to_original(presented, bbox, processed_presented_wh)
            if bbox_on_original is None:
                logger.warning("failed to translate bbox %s from processed image to original coordinates", bbox)
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=error_message_due_to_vsearcher_failure,
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                        "region_description": request.region_description,
                        "bbox_from_vsearcher": list(bbox),
                    }
                )
                continue

            bbox_on_original = _clamp_bbox_to_image(bbox_on_original, source_original.size)
            if bbox_on_original is None:
                logger.warning("translated bbox is invalid on original image: %s", bbox_on_presented)
                tool_result = ToolResult(
                    status="error",
                    requested_img_idx=img_idx,
                    error_message=error_message_due_to_vsearcher_failure,
                )
                failure_events.append(
                    {
                        "kind": "tool_execution",
                        "status": "error",
                        "requested_img_idx": img_idx,
                        "error_message": tool_result.error_message,
                        "region_description": request.region_description,
                        "bbox_on_presented": list(bbox_on_presented),
                    }
                )
                continue

            region_display_size = (
                max(1, bbox_on_presented[2] - bbox_on_presented[0]),
                max(1, bbox_on_presented[3] - bbox_on_presented[1]),
            )

            converted_tool_call_bboxes = []
            for tool_bbox in vsearcher_output.extra_fields.get("tool_call_bboxes") or []:
                if tool_bbox == (0, 0, 0, 0):
                    converted_tool_call_bboxes.append(tool_bbox)
                    continue
                tool_bbox_on_original = _translate_processed_bbox_to_original(
                    presented,
                    tool_bbox,
                    processed_presented_wh,
                )
                if tool_bbox_on_original is None:
                    continue
                tool_bbox_on_original = _clamp_bbox_to_image(tool_bbox_on_original, source_original.size)
                if tool_bbox_on_original is None:
                    continue
                converted_tool_call_bboxes.append(tool_bbox_on_original)

            vsearcher_output.extra_fields["final_bbox"] = bbox_on_original
            vsearcher_output.extra_fields["tool_call_bboxes"] = converted_tool_call_bboxes

            new_img_idx = self._append_presented_image(
                presented_images,
                source_original,
                presented.source_original_img_idx,
                bbox_on_original,
                _resize_dims_by_factor(region_display_size, self.region_zoom_in_factor),
            )
            if new_img_idx is None:
                raise RuntimeError(
                    "failed to append region zoom result after successful localization and bbox conversion"
                )
            tool_result = ToolResult(
                status="success",
                requested_img_idx=img_idx,
                new_img_idx=new_img_idx,
            )
            new_presented = presented_images[new_img_idx]
            presented_image_refs.append(
                {
                    "presented_img_idx": new_img_idx,
                    "kind": "region_crop",
                    "parent_presented_img_idx": img_idx,
                    "source_original_img_idx": new_presented.source_original_img_idx,
                    "bbox_on_original": list(new_presented.bbox_on_original),
                    "display_size": list(new_presented.display_size),
                    "zoom_in_factor": self.region_zoom_in_factor,
                    "region_description": request.region_description,
                    "bbox_on_presented": list(bbox_on_presented),
                    "region_display_size_before_zoom": list(region_display_size),
                }
            )

        if (
            not critical_failure
            and request is not None
            and request.answer
            and self.enable_unanswerable_answer_verification
            and self._is_not_answerable_question_type(kwargs["extra_info"].get("question_type"))
        ):
            verification_round_idx = len(api_rounds)
            t_verify_start = time.perf_counter()
            verification_request = await get_gpt_visual_search_request_v2(
                initial_question=kwargs["extra_info"]["question"],
                presented_images=[item.image for item in presented_images],
                messages=messages_api,
                model=self.model,
                max_tool_calls=self.max_tool_calls,
                max_round_retries=self.max_round_retries if not validate else self.max_round_retries_val,
                gpt_image_max_area=self.gpt_image_max_area,
                png_max_area=self.png_max_area,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                enable_stop=self.enable_stop,
                prompt_variant=self.prompt_variant,
                followup_user_text=get_unanswerable_answer_verification_hint(
                    self.unanswerable_answer_verification_mode
                ),
                force_answer_only=True,
            )
            t_verify_end = time.perf_counter()
            verification_event = {
                "event": "vreasoner_v2_answer_verification",
                "job_id": job_id,
                "root_job_id": root_job_id,
                "conversation_export_id": conversation_export_id,
                "validate": validate,
                "round_idx": verification_round_idx,
                "prior_tool_calls": n_tool_calls,
                "presented_image_count": len(presented_images),
                "success": verification_request.success,
                "is_last_round": verification_request.is_last_round,
                "requested_region_description": verification_request.region_description,
                "requested_img_idx": verification_request.img_idx,
                "has_answer": verification_request.answer is not None,
                "failure_reasons": list(verification_request.failure_reasons or []),
                "timing_s": {"answer_verification": t_verify_end - t_verify_start},
            }
            api_rounds.append(verification_event)
            write_profile_event("vreasoner_v2_api", verification_event, config=self.config)
            if verification_request.failure_reasons:
                failure_events.append(
                    {
                        "kind": "answer_verification",
                        "request_success": verification_request.success,
                        "failure_reasons": list(verification_request.failure_reasons),
                    }
                )
            if verification_request.success and verification_request.answer:
                messages_api = verification_request.messages
                if len(messages_api) >= 2:
                    verification_user_message = messages_api[-2]
                    verification_assistant_message = messages_api[-1]
                    if (
                        isinstance(verification_user_message, dict)
                        and verification_user_message.get("role") == "user"
                    ):
                        verification_user_message["export_type"] = ANSWER_VERIFICATION_HINT_EXPORT_TYPE
                    if (
                        isinstance(verification_assistant_message, dict)
                        and verification_assistant_message.get("role") == "assistant"
                    ):
                        verification_assistant_message["export_type"] = ANSWER_REVISION_EXPORT_TYPE
                request = verification_request

        logger.info(f"vreasoner_v2 loop completed: {profile=}")

        t_post_start = time.perf_counter()
        if self.multi_image_input:
            n_images = len(kwargs["extra_info"]["image_ori"])
            vision_tokens = "".join("<|vision_start|><|vision_end|>" for _ in range(n_images))
        else:
            vision_tokens = "<|vision_start|><|vision_end|>"
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_tokens + kwargs["extra_info"]["question"]},
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
        t_prompt_ready = time.perf_counter()

        def maybe_truncate(text: str, max_length_char: int = 4096) -> str:
            if len(text) > max_length_char:
                logger.info(
                    f"vreasoner_v2 turn response is too long: {len(text)} chars; left-truncating to {max_length_char} chars"
                )
                return "..." + text[-max_length_char:][len("..."):]
            return text

        messages_shortened = messages.copy()
        response_ids_shortened = []
        response_mask_shortened = []
        response_started = False

        for message in messages_api:
            if not isinstance(message, dict):
                message = message.to_dict()

            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        _, b64data = image_url.split(",", 1)
                        img = await loop.run_in_executor(
                            None, lambda b: Image.open(io.BytesIO(base64.b64decode(b))), b64data
                        )
                        multi_modal_data["images"].append(img)

            if message["role"] in ("system", "user") and not response_started:
                continue

            if message["role"] == "assistant":
                response_started = True
                text = message["content"]
                if text is None:
                    logger.warning("vreasoner_v2: assistant message content is None")
                    continue
                content = [{"type": "text", "text": text}]
                content_shortened = [{"type": "text", "text": maybe_truncate(text)}]
            elif message["role"] == "user":
                has_image = (
                    isinstance(message["content"], list)
                    and any(c.get("type") == "image_url" for c in message["content"])
                )
                if has_image:
                    content = [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image"},
                        {"type": "text", "text": "</tool_response>"},
                    ]
                    content_shortened = [
                        {"type": "text", "text": "<tool_response><|vision_start|><|vision_end|></tool_response>"},
                    ]
                else:
                    text = message["content"] if isinstance(message["content"], str) else " ".join(
                        c.get("text", "") for c in message["content"] if c.get("type") == "text"
                    )
                    content = [{"type": "text", "text": text}]
                    content_shortened = [{"type": "text", "text": maybe_truncate(text)}]
            else:
                logger.warning(f"vreasoner_v2: unexpected message role: {message['role']}")
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

        response_length = self.config.actor_rollout_ref.rollout.response_length
        response_ids_shortened = response_ids_shortened[-response_length:]
        response_mask_shortened = response_mask_shortened[-response_length:]
        t_messages_replayed = time.perf_counter()

        conversation_export_json_path = None
        t_export_start = time.perf_counter()
        if self.conversation_export_dir:
            try:
                record = build_export_record(
                    job_id=job_id,
                    parent_job_id=kwargs.get("parent_job_id", None),
                    root_job_id=root_job_id,
                    validate=validate,
                    initial_question=kwargs["extra_info"]["question"],
                    messages_api=messages_api,
                    raw_prompt=kwargs["raw_prompt"],
                    original_images=original_images,
                    presented_image_refs=presented_image_refs,
                    request_params={
                        "model": self.model,
                        "temperature": 1.0,
                        "gpt_image_max_area": self.gpt_image_max_area,
                        "png_max_area": self.png_max_area,
                        "image_detail": "high",
                        "max_tool_calls": self.max_tool_calls,
                        "max_completion_tokens": self.max_completion_tokens,
                        "max_round_retries": self.max_round_retries if not validate else self.max_round_retries_val,
                        "reasoning_effort": self.reasoning_effort,
                        "enable_stop": self.enable_stop,
                    },
                    loop_params={
                        "initial_rescale": actual_initial_rescale,
                        "configured_initial_rescale": self.initial_rescale,
                        "initial_input_pixels_lower_bound": self.initial_input_pixels_lower_bound,
                        "region_zoom_in_factor": self.region_zoom_in_factor,
                        "png_max_area": self.png_max_area,
                        "model": self.model,
                        "max_tool_calls": self.max_tool_calls,
                        "max_round_retries": self.max_round_retries,
                        "max_round_retries_val": self.max_round_retries_val,
                        "gpt_image_max_area": self.gpt_image_max_area,
                        "max_completion_tokens": self.max_completion_tokens,
                        "reasoning_effort": self.reasoning_effort,
                        "multi_image_input": self.multi_image_input,
                    },
                    sampling_params=dict(sampling_params),
                    tools_kwargs=kwargs["tools_kwargs"],
                    extra_info=kwargs["extra_info"],
                    failure_events=failure_events,
                    critical_failure=critical_failure,
                    final_failure_reasons=(request.failure_reasons if request is not None and request.failure_reasons else None),
                )
                export_index_metadata = {
                    "global_step": kwargs.get("_global_steps"),
                    "split": "val" if validate else "train",
                    "validate": validate,
                    "trajectory_sample_index": kwargs.get("_trajectory_sample_index"),
                    "rollout_n": kwargs.get("_rollout_n"),
                }
                record["job"].update(export_index_metadata)
                conversation_export_json_path = export_conversation(
                    self.conversation_export_dir,
                    record,
                    job_id=job_id,
                    export_id=conversation_export_id,
                    index_metadata=export_index_metadata,
                )
            except Exception as exc:
                logger.warning("failed to export vreasoner_v2 conversation for %s: %s", job_id, exc)
                failure_events.append(
                    {
                        "kind": "conversation_export",
                        "status": "error",
                        "error_message": str(exc),
                    }
                )
        t_export_end = time.perf_counter()

        extra_fields = ExtraFields(
            agent_name='vreasoner_v2',
            job_id=job_id,
            parent_job_id=kwargs.get("parent_job_id", None),
            root_job_id=root_job_id,
            extra_info=kwargs["extra_info"],
            n_tool_calls=n_tool_calls,
            critical_failure=critical_failure,
            messages=messages,
            multi_modal_data=multi_modal_data,
            failure_reasons=(request.failure_reasons if request is not None and request.failure_reasons else None),
            conversation_export_json_path=conversation_export_json_path,
        )

        extra_fields = asdict(extra_fields)
        extra_fields.update({"turn_scores": [], "tool_rewards": []})

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids_shortened,
            response_mask=response_mask_shortened,
            response_logprobs=None,
            multi_modal_data={},
            reward_score=None,
            num_turns=len(messages_shortened),
            metrics=metrics,
            extra_fields=extra_fields,
            subagent_outputs=vsearcher_outputs,
        )
        t_sample_end = time.perf_counter()
        write_profile_event(
            "vreasoner_v2_sample",
            {
                "event": "vreasoner_v2_sample",
                "job_id": job_id,
                "root_job_id": root_job_id,
                "conversation_export_id": conversation_export_id,
                "conversation_export_json_path": conversation_export_json_path,
                "validate": validate,
                "critical_failure": critical_failure,
                "failure_reasons": (request.failure_reasons if request is not None and request.failure_reasons else None),
                "n_tool_calls": n_tool_calls,
                "api_round_count": len(api_rounds),
                "child_vsearcher_count": len(child_runs),
                "presented_image_count": len(presented_images),
                "messages_api_count": len(messages_api),
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids_shortened),
                "response_truncated": len(response_ids_shortened) >= response_length,
                "timing_s": {
                    "api_calls": profile.get("api_calls", 0.0),
                    "child_vsearcher": profile.get("vsearcher_loop.run", 0.0),
                    "build_prompt": t_prompt_ready - t_post_start,
                    "message_replay_tokenize": t_messages_replayed - t_prompt_ready,
                    "conversation_export": t_export_end - t_export_start,
                    "postprocess_total": t_sample_end - t_post_start,
                    "total": t_sample_end - t_sample_start,
                },
                "api_rounds": [
                    {
                        "round_idx": e.get("round_idx"),
                        "success": e.get("success"),
                        "is_last_round": e.get("is_last_round", False),
                        "has_answer": e.get("has_answer", False),
                        "requested_img_idx": e.get("requested_img_idx"),
                        "failure_reasons": e.get("failure_reasons"),
                        "timing_s": e.get("timing_s"),
                    }
                    for e in api_rounds
                ],
                "child_runs": [
                    {
                        "child_idx": e["child_idx"],
                        "child_job_id": e["child_job_id"],
                        "requested_img_idx": e["requested_img_idx"],
                        "timing_s": e["timing_s"],
                        "metrics": e["metrics"],
                    }
                    for e in child_runs
                ],
            },
            config=self.config,
        )
        return output


@register("vreasoner_qwen3_vl")
class VReasonerLoopQwen3VL(VSearcherLoopQwen3VL):
    AGENT_NAME = "vreasoner_qwen3_vl"
    MAX_TOKENS_PER_TURN = 4096

    def _extract_bbox_within_last_answer_tags(
        self, messages: ReconstructedMessages, extra_info: dict[str, Any] | None = None
    ) -> BBox | None:
        return None
