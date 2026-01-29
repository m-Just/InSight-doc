"""
QwenAgentLoop: A hybrid agent loop that uses qwen_agent's tool system with verl's rollout infrastructure.

This module bridges qwen_agent's tool ecosystem (TOOL_REGISTRY, BaseTool) with verl's training
infrastructure (AsyncLLMServerManager, AgentLoopOutput).
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Any
from uuid import uuid4

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.llm.schema import ContentItem, Message

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, AgentData
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
              qwen_tool_list: ["image_zoom_in_tool"]  # qwen_agent tool names
              max_user_turns: 6
              max_assistant_turns: 7
              max_parallel_calls: 1
              max_tool_response_length: 256
    """

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
        # qwen_tool_list = multi_turn_config.get("qwen_tool_list", [])
        qwen_tool_list = ["image_zoom_in_tool"]  # hardcode for now
        self.tools: dict[str, BaseTool] = {}
        self.tool_schemas: list[dict] = []
        for tool_name in qwen_tool_list:
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Tool '{tool_name}' not found in qwen_agent's TOOL_REGISTRY. "
                                f"Available tools: {list(TOOL_REGISTRY.keys())}")
            tool_instance = TOOL_REGISTRY[tool_name]()
            self.tools[tool_name] = tool_instance
            self.tool_schemas.append(tool_instance.function)

        # Initialize tool parser for extracting tool calls from model output
        self.tool_parser = ToolParser.get_tool_parser(
            multi_turn_config.format, self.tokenizer
        )
        self.tool_parser_name = multi_turn_config.format

        # Sequence length configuration
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the agent loop to interact with LLM and execute tools.
        
        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Dataset fields including raw_prompt, multi_modal_data, extra_info, etc.
            
        Returns:
            AgentLoopOutput with prompt_ids, response_ids, response_mask, etc.
        """
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

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]

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

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
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
