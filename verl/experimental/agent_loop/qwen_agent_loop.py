"""
QwenAgentLoop: A hybrid agent loop that uses qwen_agent's tool system with verl's rollout infrastructure.

This module bridges qwen_agent's tool ecosystem (TOOL_REGISTRY, BaseTool) with verl's training
infrastructure (AsyncLLMServerManager, AgentLoopOutput).
"""

import asyncio
import copy
import json
import logging
import os
import re
import tempfile
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from .qwen_agent_tools import image_zoom_in_qwen3vl as _image_zoom_in_qwen3vl

from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.llm.schema import ContentItem, Message

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
    resolve_dynamic_initial_rescale,
)
from verl.experimental.agent_loop.presented_images import (
    PresentedImageState,
    cap_size_by_area as _cap_size_by_area,
    clamp_bbox_to_image as _clamp_bbox_to_image,
    presented_image_to_export_ref as _presented_image_to_export_ref,
    resize_dims_by_factor as _resize_dims_by_factor,
    scale_bbox_from_qwen_range as _scale_bbox_from_qwen_range,
    translate_bbox_to_original as _translate_bbox_to_original,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, AgentData
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.vreasoner_v2_conversation_export import (
    build_export_record,
    export_conversation,
    parse_assistant_message,
    parse_user_message,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_NAME_ALIASES = {
    "image_zoom_in_tool_qwen3vl": "image_zoom_in_tool",
}

QWEN_IMAGE_MAX_ASPECT_RATIO = 200.0
IMAGE_LABEL_RE = re.compile(r"^Image (\d+):$")

def _image_aspect_ratio(size: tuple[int, int]) -> float:
    width, height = size
    if width <= 0 or height <= 0:
        return float("inf")
    return max(width, height) / min(width, height)


def _validate_qwen_image_aspect_ratio(size: tuple[int, int]) -> str | None:
    ratio = _image_aspect_ratio(size)
    if ratio > QWEN_IMAGE_MAX_ASPECT_RATIO:
        return (
            "Tool Execution Error "
            f"absolute aspect ratio must be smaller than {int(QWEN_IMAGE_MAX_ASPECT_RATIO)}, got {ratio}"
        )
    return None


def _build_visualization_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    ordered_images: list[Any] = []
    ordered_videos: list[Any] = []
    notebook_messages: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            notebook_messages.append(message)
            continue

        notebook_content: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                notebook_content.append(item)
                continue

            item_type = item.get("type")
            if item_type == "image" and "image" in item:
                ordered_images.append(item["image"])
                notebook_content.append({"type": "image"})
            elif item_type == "video" and "video" in item:
                ordered_videos.append(item["video"])
                notebook_content.append({"type": "video"})
            else:
                notebook_content.append(item)

        notebook_messages.append({"role": message.get("role"), "content": notebook_content})

    multi_modal_data: dict[str, list[Any]] = {}
    if ordered_images:
        multi_modal_data["image"] = ordered_images
    if ordered_videos:
        multi_modal_data["video"] = ordered_videos
    return notebook_messages, multi_modal_data


def _record_conversation_wall_time(agent_data: AgentData, start_time: float) -> None:
    conversation_wall_time = time.perf_counter() - start_time
    agent_data.metrics["conversation_wall_time"] = conversation_wall_time
    agent_data.extra_fields["conversation_wall_time"] = conversation_wall_time


def _message_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return "" if content is None else str(content)


def _strip_wrapped_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>.*?</{tag}>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def _text_before_tag(text: str, tag: str) -> str:
    marker = f"<{tag}>"
    if marker not in text:
        return text.strip()
    return text.split(marker, 1)[0].strip()


def _convert_export_multimodal_content(
    content: Any,
    *,
    next_presented_idx: int,
) -> tuple[Any, int]:
    if not isinstance(content, list):
        return _message_text_content(content), next_presented_idx

    converted_content: list[dict[str, Any]] = []
    pending_presented_idx: int | None = None
    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            converted_content.append({"type": "text", "text": text})
            match = IMAGE_LABEL_RE.match(text.strip())
            if match:
                pending_presented_idx = int(match.group(1))
            continue

        if item_type == "image":
            presented_idx = pending_presented_idx if pending_presented_idx is not None else next_presented_idx
            converted_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"presented://{presented_idx}",
                        "detail": "high",
                    },
                }
            )
            next_presented_idx = max(next_presented_idx, presented_idx + 1)
            pending_presented_idx = None
            continue

        if "text" in item and isinstance(item["text"], str):
            converted_content.append({"type": "text", "text": item["text"]})

    return converted_content, next_presented_idx


def _build_insight_export_conversation(
    messages: list[dict[str, Any]],
    *,
    initial_question: str,
) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    next_presented_idx = 0
    last_assistant_idx = max((idx for idx, message in enumerate(messages) if message.get("role") == "assistant"), default=-1)

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            conversation.append(
                {
                    "message_idx": idx,
                    "role": "system",
                    "type": "system_prompt",
                    "content": {"text": _message_text_content(content)},
                }
            )
            continue

        if role == "assistant":
            text = _message_text_content(content)
            parsed = parse_assistant_message(text)
            extracted_tags = parsed.get("content", {}).get("extracted_tags", {})
            parsed_tool_call = extracted_tags.get("tool_call")
            if parsed.get("type") == "others" and parsed_tool_call is not None:
                tool_call_parse_error = parsed.get("tool_call_parse_error")
                parsed = {
                    "type": "tool_call",
                    "content": {
                        "think": _text_before_tag(text, "tool_call"),
                        "tool_call": parsed_tool_call,
                    },
                    "tag_counts": parsed.get("tag_counts", {}),
                }
                if tool_call_parse_error is not None:
                    parsed["tool_call_parse_error"] = tool_call_parse_error
            elif parsed.get("type") == "others" and idx == last_assistant_idx and text.strip():
                parsed = {
                    "type": "answer",
                    "content": {
                        "think": "",
                        "answer": text.strip(),
                    },
                    "tag_counts": parsed.get("tag_counts", {}),
                }
            conversation.append({"message_idx": idx, "role": "assistant", **parsed})
            continue

        if role == "user":
            converted_content, next_presented_idx = _convert_export_multimodal_content(
                content,
                next_presented_idx=next_presented_idx,
            )
            parsed = parse_user_message(converted_content, initial_question=initial_question)
            conversation.append({"message_idx": idx, "role": "user", **parsed})
            continue

        if role == "tool":
            converted_content, next_presented_idx = _convert_export_multimodal_content(
                content,
                next_presented_idx=next_presented_idx,
            )
            parsed = parse_user_message(converted_content, initial_question=initial_question)
            if parsed.get("type") == "others":
                text = parsed.get("content", {}).get("text", "").strip()
                parsed = {
                    "type": "tool_result_fail_hint",
                    "content": {
                        "error_message": text,
                        "hint": "",
                    },
                }
            conversation.append({"message_idx": idx, "role": "user", **parsed})
            continue

        conversation.append(
            {
                "message_idx": idx,
                "role": role,
                "type": "others",
                "content": {"value": _message_text_content(content)},
            }
        )

    return conversation

@register("qwen_agent")
class QwenAgentLoop(AgentLoopBase):
    """Agent loop that integrates qwen_agent's tool system with verl's rollout infrastructure.
    
    This class provides:
    - Tool initialization from qwen_agent's TOOL_REGISTRY
    - State machine-based agentic loop (similar to ToolAgentLoop)
    - Tool call parsing using configurable parsers (hermes, gpt-oss, etc.)
    - Multimodal tool response handling (images via ContentItem)
    - Integration with verl's AsyncLLMServerManager for LLM inference
    
    Configuration:
        The agent loop is configured via the trainer config YAML:
        
        actor_rollout_ref:
          rollout:
            multi_turn:
              format: "hermes"  # Tool call format parser
              qwen_tool_list: ["image_zoom_in_tool_qwen3vl"]  # qwen_agent tool names
              max_user_turns: 6
              max_assistant_turns: 7
              max_parallel_calls: 1
              max_tool_response_length: 256
    """
    DEFAULT_QWEN_TOOL_LIST = ["image_zoom_in_tool_qwen3vl"]

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Multi-turn configuration
        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn
        self.max_user_turns = multi_turn_config.max_user_turns
        self.max_assistant_turns = multi_turn_config.max_assistant_turns
        self.max_parallel_calls = multi_turn_config.get("max_parallel_calls", 1)
        self.max_tool_response_length = multi_turn_config.get("max_tool_response_length", 256)
        self.tool_response_truncate_side = multi_turn_config.get("tool_response_truncate_side", "middle")

        # Initialize qwen_agent tools from TOOL_REGISTRY
        qwen_tool_list = list(multi_turn_config.get("qwen_tool_list", self.DEFAULT_QWEN_TOOL_LIST))
        self.tools: dict[str, BaseTool] = {}
        self.tool_schemas: list[dict] = []
        self.tool_name_map: dict[str, str] = {}
        for registry_tool_name in qwen_tool_list:
            if registry_tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Tool '{registry_tool_name}' not found in qwen_agent's TOOL_REGISTRY. "
                                f"Available tools: {list(TOOL_REGISTRY.keys())}")
            public_tool_name = TOOL_NAME_ALIASES.get(registry_tool_name, registry_tool_name)
            if public_tool_name in self.tools:
                raise ValueError(
                    f"Duplicate public tool name '{public_tool_name}' from qwen_tool_list={qwen_tool_list}"
                )

            tool_instance = TOOL_REGISTRY[registry_tool_name]()
            tool_schema = dict(tool_instance.function)
            tool_schema["name"] = public_tool_name

            self.tools[public_tool_name] = tool_instance
            self.tool_name_map[public_tool_name] = registry_tool_name
            self.tool_schemas.append(tool_schema)

        # Initialize tool parser for extracting tool calls from model output
        self.tool_parser = ToolParser.get_tool_parser(
            multi_turn_config.format, self.tokenizer
        )
        self.tool_parser_name = multi_turn_config.format

        # Sequence length configuration
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.conversation_export_dir = config.actor_rollout_ref.rollout.agent.get(
            "vreasoner_v2_conversation_export_dir",
            None,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the agent loop to interact with LLM and execute tools.
        
        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Dataset fields including raw_prompt, multi_modal_data, extra_info, etc.
            
        Returns:
            AgentLoopOutput with prompt_ids, response_ids, response_mask, etc.
        """
        conversation_wall_time_start = time.perf_counter()
        messages = list(kwargs["raw_prompt"])

        # Extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Create agent data to encapsulate state (reuses AgentData from ToolAgentLoop)
        agent_data = AgentData(
            messages=messages,
            image_data=images if images else [],
            video_data=videos if videos else [],
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,  # QwenAgentLoop doesn't support interaction yet
            interaction_kwargs={},
        )
        agent_data.extra_fields["response_truncated"] = False

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Includes prompt/image preparation, tool processing, and model generation.
        # Excludes export and later postprocessing.
        _record_conversation_wall_time(agent_data, conversation_wall_time_start)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        if len(response_ids) > self.response_length or len(agent_data.response_mask) > self.response_length:
            agent_data.extra_fields["response_truncated"] = True
        viz_messages, viz_multi_modal_data = _build_visualization_messages(agent_data.messages)
        agent_data.extra_fields["messages"] = viz_messages
        agent_data.extra_fields["multi_modal_data"] = viz_multi_modal_data

        multi_modal_output = {}
        if agent_data.image_data:
            multi_modal_output["images"] = agent_data.image_data
        if agent_data.video_data:
            multi_modal_output["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data=multi_modal_output,
            response_logprobs=agent_data.response_logprobs[:self.response_length] if agent_data.response_logprobs else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=AgentLoopMetrics(**agent_data.metrics),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Handle the pending state: prepare the prompt and transition to generating."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas if self.tool_schemas else None,
            images=agent_data.image_data if agent_data.image_data else None,
            videos=agent_data.video_data if agent_data.video_data else None,
        )
        agent_data.prompt_ids = prompt_ids

        logger.debug(f"[QwenAgentLoop._handle_pending_state] {self.tool_schemas=}")

        prompt = self.processor.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        logger.debug(f"[QwenAgentLoop._handle_pending_state] {prompt=}")

        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls.
        
        Args:
            agent_data: The agent data containing messages and state.
            sampling_params: LLM sampling parameters.
            ignore_termination: If True, skip turn-based termination checks (for subclasses).
        """
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data if agent_data.image_data else None,
                video_data=agent_data.video_data if agent_data.video_data else None,
            )

        if output.num_preempted is not None:
            agent_data.metrics["num_preempted"] = output.num_preempted

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        assistant_message = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        agent_data.messages.append({"role": "assistant", "content": assistant_message})

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data.extra_fields["response_truncated"] = True
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls from the response
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Image.Image] = []

        # Execute tool calls (up to max_parallel_calls)
        tasks = []
        for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
            tasks.append(self._call_qwen_tool(tool_call, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process tool responses
        for i, response in enumerate(responses):
            tool_call = agent_data.tool_calls[i]
            
            if isinstance(response, Exception):
                logger.warning(f"Tool call {tool_call.name} failed: {response}")
                message = {"role": "tool", "content": f"Error executing tool: {response}"}
            else:
                tool_result, new_images = response
                
                # Handle multimodal responses
                if new_images:
                    for img in new_images:
                        new_images_this_turn.append(img)
                    # Create multimodal tool response message
                    content = []
                    for _ in new_images:
                        content.append({"type": "image"})
                    if tool_result:
                        content.append({"type": "text", "text": tool_result})
                    message = {"role": "tool", "content": content}
                else:
                    # Text-only tool response
                    message = {"role": "tool", "content": tool_result or ""}

            add_messages.append(message)

        agent_data.messages.extend(add_messages)

        # Apply chat template to tool responses
        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn if new_images_this_turn else None,
            videos=None,
            remove_system_prompt=True,
        )

        # Check if adding tool response would exceed response length
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.extra_fields["response_truncated"] = True
            return AgentState.TERMINATED

        # Update image data with new images from tool calls
        if new_images_this_turn:
            if not agent_data.image_data:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        # Update token sequences
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)  # Tool responses are masked
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_qwen_tool(
        self, tool_call: FunctionCall, agent_data: AgentData
    ) -> tuple[str, list[Image.Image]]:
        """Call a qwen_agent tool and return the result.
        
        Args:
            tool_call: The parsed function call with name and arguments.
            agent_data: The agent data containing messages and state.
            
        Returns:
            Tuple of (text_result, list_of_images).
        """
        tool_name = tool_call.name
        new_images: list[Image.Image] = []

        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found.", new_images

        tool = self.tools[tool_name]

        try:
            # Parse tool arguments
            tool_args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            return f"Invalid JSON arguments: {e}", new_images

        try:
            # Prepare kwargs for tool call
            # qwen_agent tools expect messages in qwen_agent's Message format, not plain dicts
            qwen_messages = self._convert_to_qwen_messages(agent_data.messages)
            kwargs = {"messages": qwen_messages}
            
            # For tools with file_access, extract files/images from messages
            if tool.file_access:
                kwargs["files"] = self._extract_files_from_messages(agent_data.messages)

            # Call the qwen_agent tool (synchronous call wrapped in executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: tool.call(tool_args, **kwargs)
            )

            # Process the result
            text_result = ""
            if isinstance(result, str):
                text_result = result
            elif isinstance(result, list):
                # Handle ContentItem list (qwen_agent's multimodal response format)
                for item in result:
                    if isinstance(item, ContentItem):
                        item_type, item_value = item.get_type_and_value()
                        if item_type == "text":
                            text_result += item_value
                        elif item_type == "image":
                            # item_value is an image path
                            if isinstance(item_value, str):
                                img = Image.open(item_value)
                                new_images.append(img)
                            elif isinstance(item_value, Image.Image):
                                new_images.append(item_value)
                    elif isinstance(item, dict):
                        text_result += json.dumps(item, ensure_ascii=False)
                    else:
                        text_result += str(item)
            elif isinstance(result, dict):
                text_result = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                text_result = str(result)

            # Truncate long responses
            if text_result and len(text_result) > self.max_tool_response_length:
                text_result = self._truncate_text(text_result, self.max_tool_response_length)

            return text_result, new_images

        except Exception as e:
            logger.warning(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool: {e}", new_images

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length based on configured truncation side."""
        if self.tool_response_truncate_side == "left":
            return text[:max_length] + "...(truncated)"
        elif self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-max_length:]
        else:  # middle
            length = max_length // 2
            return text[:length] + "...(truncated)..." + text[-length:]

    def _convert_to_qwen_messages(self, messages: list[dict]) -> list[Message]:
        """Convert dict-formatted messages to qwen_agent's Message format.
        
        qwen_agent tools expect Message objects with ContentItem content, not plain dicts.
        This converts OpenAI-style message dicts to qwen_agent's schema format.
        
        Args:
            messages: List of message dicts in OpenAI format
            
        Returns:
            List of qwen_agent Message objects
        """
        qwen_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map 'tool' role to 'function' for qwen_agent compatibility
            if role == "tool":
                role = "function"
            
            if isinstance(content, str):
                # Simple text content
                qwen_messages.append(Message(role=role, content=content))
            elif isinstance(content, list):
                # Multimodal content - convert to ContentItem list
                content_items = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            content_items.append(ContentItem(text=item.get("text", "")))
                        elif item_type == "image":
                            # Handle image item with 'image' key
                            image_value = item.get("image", "")
                            if image_value:
                                image_path = self._ensure_image_path(image_value)
                                if image_path:
                                    content_items.append(ContentItem(image=image_path))
                        elif item_type == "image_url":
                            # Handle image_url format (OpenAI style)
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url
                            if url:
                                image_path = self._ensure_image_path(url)
                                if image_path:
                                    content_items.append(ContentItem(image=image_path))
                        elif item_type == "file":
                            content_items.append(ContentItem(file=item.get("file", "")))
                        elif item_type == "video":
                            content_items.append(ContentItem(video=item.get("video", "")))
                        elif item_type == "audio":
                            content_items.append(ContentItem(audio=item.get("audio", "")))
                    elif isinstance(item, ContentItem):
                        # Already a ContentItem
                        content_items.append(item)
                
                if content_items:
                    qwen_messages.append(Message(role=role, content=content_items))
                else:
                    # Empty content, use empty string
                    qwen_messages.append(Message(role=role, content=""))
            else:
                # Fallback - convert to string
                qwen_messages.append(Message(role=role, content=str(content) if content else ""))
        
        return qwen_messages

    def _ensure_image_path(self, image_value: Any) -> str | None:
        """Ensure image value is a string path that qwen_agent tools can use.
        
        If the image is a PIL Image, save it to a temp file and return the path.
        If it's already a string (path or URL), return it as-is.
        
        Args:
            image_value: Either a string path/URL or a PIL Image object
            
        Returns:
            String path to the image, or None if conversion failed
        """
        if isinstance(image_value, str):
            return image_value
        elif isinstance(image_value, Image.Image):
            # Save PIL Image to a temp file
            try:
                # Create temp file with .png extension
                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                image_value.save(temp_path)
                return temp_path
            except Exception as e:
                logger.warning(f"Failed to save PIL Image to temp file: {e}")
                return None
        else:
            logger.warning(f"Unsupported image type: {type(image_value)}")
            return None

    def _extract_files_from_messages(self, messages: list[dict]) -> list[str]:
        """Extract file URLs/paths from messages for tools that need file access.
        
        This mimics qwen_agent's extract_files_from_messages utility.
        """
        files = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Handle image items
                        if "image" in item:
                            files.append(item["image"])
                        elif "image_url" in item:
                            url = item["image_url"]
                            if isinstance(url, dict):
                                files.append(url.get("url", ""))
                            else:
                                files.append(url)
                        # Handle file items
                        if "file" in item:
                            files.append(item["file"])
        return files


@register("insight_qwen_agent")
class InSightQwenAgentLoop(QwenAgentLoop):
    """QwenAgentLoop variant aligned with InSight/VReasoner-style training data.

    Differences from QwenAgentLoop:
    - initial visible images are downscaled into presented-image views
    - successful tool results are emitted as indexed image observations like "Image N:"
    """

    DEFAULT_INITIAL_RESCALE = 0.25
    DEFAULT_GPT_IMAGE_MAX_AREA = 1280 * 1280
    DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND = 0
    DEFAULT_REGION_ZOOM_IN_FACTOR = 4.0

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        initial_rescale: float = DEFAULT_INITIAL_RESCALE,
        gpt_image_max_area: int = DEFAULT_GPT_IMAGE_MAX_AREA,
        initial_input_pixels_lower_bound: int = DEFAULT_INITIAL_INPUT_PIXELS_LOWER_BOUND,
        region_zoom_in_factor: float = DEFAULT_REGION_ZOOM_IN_FACTOR,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        # Backward-compatible aliases for older configs and scripts.
        if "presented_initial_rescale" in kwargs:
            initial_rescale = kwargs.pop("presented_initial_rescale")
        if "presented_max_area" in kwargs:
            gpt_image_max_area = kwargs.pop("presented_max_area")
        if "presented_initial_pixels_lower_bound" in kwargs:
            initial_input_pixels_lower_bound = kwargs.pop("presented_initial_pixels_lower_bound")

        if initial_input_pixels_lower_bound < 0:
            raise ValueError(
                f"initial_input_pixels_lower_bound must be non-negative, got {initial_input_pixels_lower_bound}"
            )
        self.initial_rescale = initial_rescale
        self.gpt_image_max_area = gpt_image_max_area
        self.initial_input_pixels_lower_bound = initial_input_pixels_lower_bound
        self.region_zoom_in_factor = region_zoom_in_factor
        # Keep alias attrs so older code paths continue to work.
        self.presented_initial_rescale = self.initial_rescale
        self.presented_max_area = self.gpt_image_max_area
        self.presented_initial_pixels_lower_bound = self.initial_input_pixels_lower_bound

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        conversation_wall_time_start = time.perf_counter()
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        # Smoke-test compatibility only:
        # some synthetic inference inputs are reconstructed from converted SFT rows whose images are already
        # in the presented form expected by InSightQwenAgentLoop. Normal eval/training flow should leave this
        # unset so the loop can build presented images from the raw prompt images itself.
        aligned_prompt, original_images, presented_images, actual_initial_rescale = self._build_presented_prompt(
            messages,
            images_are_presented=bool(kwargs.get("initial_images_already_presented", False)),
        )

        multi_modal_data = await self.process_vision_info(aligned_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=aligned_prompt,
            image_data=images if images else [],
            video_data=videos if videos else [],
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,
            interaction_kwargs={},
        )
        agent_data.extra_fields["response_truncated"] = False
        extra_info = dict(kwargs["extra_info"])
        extra_info["agent_name"] = "insight_qwen_agent"
        agent_data.extra_fields["agent_name"] = "insight_qwen_agent"
        agent_data.extra_fields["extra_info"] = extra_info
        agent_data.extra_fields["export_failure_events"] = []
        agent_data.extra_fields["insight_original_images"] = original_images
        agent_data.extra_fields["insight_presented_images"] = presented_images
        if self.conversation_export_dir:
            agent_data.extra_fields["insight_presented_image_refs"] = [
                _presented_image_to_export_ref(
                    presented_img_idx,
                    presented,
                    kind="initial",
                    original_images=original_images,
                    initial_rescale=actual_initial_rescale,
                )
                for presented_img_idx, presented in enumerate(presented_images)
            ]

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Includes prompt/image preparation, tool processing, and model generation.
        # Excludes export and later postprocessing.
        _record_conversation_wall_time(agent_data, conversation_wall_time_start)

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        if len(response_ids) > self.response_length or len(agent_data.response_mask) > self.response_length:
            agent_data.extra_fields["response_truncated"] = True

        conversation_export_json_path = None
        if self.conversation_export_dir:
            try:
                record = build_export_record(
                    job_id=request_id,
                    parent_job_id=kwargs.get("parent_job_id"),
                    root_job_id=kwargs.get("root_job_id", request_id),
                    validate=bool(kwargs.get("_validate", False)),
                    initial_question=kwargs["extra_info"]["question"],
                    messages_api=[],
                    raw_prompt=kwargs["raw_prompt"],
                    original_images=original_images,
                    presented_image_refs=agent_data.extra_fields.get("insight_presented_image_refs", []),
                    request_params={
                        "tool_parser": self.tool_parser_name,
                        "prompt_length": self.prompt_length,
                        "response_length": self.response_length,
                        "max_user_turns": self.max_user_turns,
                        "max_assistant_turns": self.max_assistant_turns,
                        "max_parallel_calls": self.max_parallel_calls,
                    },
                    loop_params={
                        "initial_rescale": actual_initial_rescale,
                        "configured_initial_rescale": self.initial_rescale,
                        "initial_input_pixels_lower_bound": self.initial_input_pixels_lower_bound,
                        "gpt_image_max_area": self.gpt_image_max_area,
                        "region_zoom_in_factor": self.region_zoom_in_factor,
                        "agent_name": "insight_qwen_agent",
                    },
                    sampling_params=dict(sampling_params),
                    tools_kwargs=kwargs["tools_kwargs"],
                    extra_info=extra_info,
                    failure_events=agent_data.extra_fields.get("export_failure_events", []),
                    critical_failure=bool(agent_data.extra_fields.get("failure_reasons")),
                    final_failure_reasons=agent_data.extra_fields.get("failure_reasons"),
                )
                record["agent_name"] = "insight_qwen_agent"
                record["conversation"] = _build_insight_export_conversation(
                    agent_data.messages,
                    initial_question=kwargs["extra_info"]["question"],
                )
                conversation_export_json_path = export_conversation(
                    self.conversation_export_dir,
                    record,
                    job_id=request_id,
                )
            except Exception as exc:
                logger.warning("failed to export insight_qwen_agent conversation for %s: %s", request_id, exc)

        if conversation_export_json_path:
            agent_data.extra_fields["conversation_export_json_path"] = conversation_export_json_path
            agent_data.extra_fields["extra_info"]["conversation_export_json_path"] = conversation_export_json_path

        # These fields are only needed while the loop is still running. Keeping them in the
        # returned DataProto multiplies CPU memory use across repeated samples and Ray workers.
        agent_data.extra_fields.pop("insight_original_images", None)
        agent_data.extra_fields.pop("insight_presented_images", None)
        if not self.conversation_export_dir:
            agent_data.extra_fields.pop("insight_presented_image_refs", None)
            agent_data.extra_fields.pop("export_failure_events", None)

        multi_modal_output = {}
        if agent_data.image_data:
            multi_modal_output["images"] = agent_data.image_data
        if agent_data.video_data:
            multi_modal_output["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data=multi_modal_output,
            response_logprobs=agent_data.response_logprobs[:self.response_length] if agent_data.response_logprobs else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=AgentLoopMetrics(**agent_data.metrics),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    def _build_presented_prompt(
        self,
        messages: list[dict[str, Any]],
        images_are_presented: bool = False,
    ) -> tuple[list[dict[str, Any]], list[Image.Image], list[PresentedImageState], float]:
        original_images: list[Image.Image] = []
        staged_messages: list[tuple[dict[str, Any], list[Any], list[str], bool]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            staged_items: list[Any] = []
            saw_image = False
            trailing_text_parts: list[str] = []
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
                        new_content.append(copy.deepcopy(item))
                    continue

                image = self._load_prompt_image(image_value)
                if image is None:
                    continue
                original_images.append(image)
                staged_items.append(image)
                saw_image = True
            if not saw_image:
                message["content"] = [copy.deepcopy(item) for item in content if isinstance(item, dict)]
            staged_messages.append((message, staged_items, trailing_text_parts, saw_image))

        actual_initial_rescale = self.initial_rescale if images_are_presented else resolve_dynamic_initial_rescale(
            image_sizes=[image.size for image in original_images],
            configured_initial_rescale=self.initial_rescale,
            total_pixels_lower_bound=self.initial_input_pixels_lower_bound,
            per_image_max_area=self.gpt_image_max_area,
        )

        presented_images: list[PresentedImageState] = []
        original_image_idx = 0
        for message, staged_items, trailing_text_parts, saw_image in staged_messages:
            if not saw_image:
                continue
            new_content: list[dict[str, Any]] = []
            for image in staged_items:
                presented_image = image.copy() if images_are_presented else self._build_presented_image(
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

        return messages, original_images, presented_images, actual_initial_rescale

    def _build_presented_image(self, image: Image.Image, initial_rescale: float) -> Image.Image:
        target_size = _cap_size_by_area(
            _resize_dims_by_factor(image.size, initial_rescale),
            self.gpt_image_max_area,
        )
        if image.size == target_size:
            return image.copy()
        return image.resize(target_size, Image.LANCZOS)

    def _load_prompt_image(self, image_value: Any) -> Image.Image | None:
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")

        if isinstance(image_value, dict):
            url = image_value.get("url", "")
            if url:
                image_value = url
            else:
                return None

        if not isinstance(image_value, str):
            return None

        path = image_value
        if image_value.startswith("file://"):
            parsed = urlparse(image_value)
            path = parsed.path
        try:
            image = Image.open(path)
            image.load()
            return image.convert("RGB")
        except Exception as exc:
            logger.warning(f"Failed to load prompt image {image_value}: {exc}")
            return None

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Image.Image] = []
        base_image_count = len(agent_data.image_data) if isinstance(agent_data.image_data, list) else 0
        new_presented_this_turn: list[PresentedImageState] = []
        new_presented_refs_this_turn: list[dict[str, Any]] = []

        tasks = []
        for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
            tasks.append(self._call_qwen_tool(tool_call, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            tool_call = agent_data.tool_calls[i]

            if isinstance(response, Exception):
                logger.warning(f"Tool call {tool_call.name} failed: {response}")
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
                tool_result, new_images, new_presented, new_presented_refs = response
                if new_images:
                    content = []
                    for offset, img in enumerate(new_images):
                        new_index = base_image_count + len(new_images_this_turn) + offset
                        content.append({"type": "text", "text": f"Image {new_index}:"})
                        content.append({"type": "image"})
                    if tool_result:
                        content.append({"type": "text", "text": tool_result})
                    for img in new_images:
                        new_images_this_turn.append(img)
                    new_presented_this_turn.extend(new_presented)
                    new_presented_refs_this_turn.extend(new_presented_refs)
                    message = {"role": "tool", "content": content}
                else:
                    if tool_result:
                        agent_data.extra_fields.setdefault("export_failure_events", []).append(
                            {
                                "kind": "tool_execution",
                                "status": "error",
                                "tool_name": tool_call.name,
                                "error_message": tool_result,
                            }
                        )
                    message = {"role": "tool", "content": tool_result or ""}

            add_messages.append(message)

        agent_data.messages.extend(add_messages)

        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn if new_images_this_turn else None,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.extra_fields["response_truncated"] = True
            return AgentState.TERMINATED

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
        return AgentState.GENERATING

    async def _call_qwen_tool(
        self, tool_call: FunctionCall, agent_data: AgentData
    ) -> tuple[str, list[Image.Image], list[PresentedImageState], list[dict[str, Any]]]:
        if tool_call.name == "image_zoom_in_tool":
            return self._call_insight_zoom_tool(tool_call, agent_data)
        text_result, new_images = await super()._call_qwen_tool(tool_call, agent_data)
        return text_result, new_images, [], []

    def _call_insight_zoom_tool(
        self, tool_call: FunctionCall, agent_data: AgentData
    ) -> tuple[str, list[Image.Image], list[PresentedImageState], list[dict[str, Any]]]:
        try:
            tool_args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            return f"Tool Execution Error Invalid JSON arguments: {exc}", [], [], []

        img_idx = tool_args.get("img_idx")
        label = tool_args.get("label")
        bbox_2d = tool_args.get("bbox_2d")
        if not isinstance(img_idx, int):
            return "Tool Execution Error img_idx must be an integer.", [], [], []
        if not isinstance(label, str):
            return "Tool Execution Error label must be a string.", [], [], []
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
            return "Tool Execution Error bbox_2d must be a list of four numbers.", [], [], []

        presented_images: list[PresentedImageState] = agent_data.extra_fields.get("insight_presented_images", [])
        original_images: list[Image.Image] = agent_data.extra_fields.get("insight_original_images", [])
        if not (0 <= img_idx < len(presented_images)):
            return f"Error: Invalid input image index {img_idx}.", [], [], []

        parent = presented_images[img_idx]
        bbox_on_presented = _scale_bbox_from_qwen_range(bbox_2d, parent.display_size)
        if bbox_on_presented is None:
            return "Tool Execution Error invalid bbox_2d.", [], [], []

        bbox_on_original = _translate_bbox_to_original(parent, bbox_on_presented)
        if bbox_on_original is None:
            return "Tool Execution Error failed to translate bbox to original image.", [], [], []

        source_original = original_images[parent.source_original_img_idx]
        bbox_on_original = _clamp_bbox_to_image(bbox_on_original, source_original.size)
        if bbox_on_original is None:
            return "Tool Execution Error translated bbox is invalid on original image.", [], [], []

        x1, y1, x2, y2 = bbox_on_presented
        region_display_size = (max(1, x2 - x1), max(1, y2 - y1))
        target_display_size = _cap_size_by_area(
            _resize_dims_by_factor(region_display_size, self.region_zoom_in_factor),
            self.gpt_image_max_area,
        )
        aspect_ratio_error = _validate_qwen_image_aspect_ratio(target_display_size)
        if aspect_ratio_error is not None:
            return aspect_ratio_error, [], [], []

        crop = source_original.crop(bbox_on_original)
        if crop.size != target_display_size:
            crop = crop.resize(target_display_size, Image.LANCZOS)
        aspect_ratio_error = _validate_qwen_image_aspect_ratio(crop.size)
        if aspect_ratio_error is not None:
            return aspect_ratio_error, [], [], []

        presented = PresentedImageState(
            image=crop,
            source_original_img_idx=parent.source_original_img_idx,
            bbox_on_original=bbox_on_original,
            display_size=crop.size,
        )
        export_ref = _presented_image_to_export_ref(
            len(presented_images),
            presented,
            kind="region_crop",
            original_images=original_images,
            parent_presented_img_idx=img_idx,
            region_description=label,
            bbox_on_presented=bbox_on_presented,
            zoom_in_factor=self.region_zoom_in_factor,
            region_display_size_before_zoom=region_display_size,
        )
        return "", [crop], [presented], [export_ref]
