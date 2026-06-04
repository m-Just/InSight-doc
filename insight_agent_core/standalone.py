from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Protocol

from .runtime import CoreFunctionCall, CoreGenerationOutput


ToolCallExtractor = Callable[[list[int]], Awaitable[list[CoreFunctionCall]]]


class GenerationEndpointPool(Protocol):
    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CoreGenerationOutput:
        ...


class StandaloneInSightRuntime:
    """CoreRuntime implementation backed by local HF processing and Ray vLLM generation."""

    def __init__(
        self,
        *,
        tokenizer,
        processor,
        endpoint_pool: GenerationEndpointPool,
        tool_call_extractor: ToolCallExtractor,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        processor_concurrency: int = 8,
    ) -> None:
        if processor_concurrency < 1:
            raise ValueError(f"processor_concurrency must be >= 1, got {processor_concurrency}")
        self.tokenizer = tokenizer
        self.processor = processor
        self.endpoint_pool = endpoint_pool
        self.tool_call_extractor = tool_call_extractor
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self.processor_concurrency = processor_concurrency
        self._processor_semaphore = asyncio.Semaphore(processor_concurrency)
        self._validate_processor_state()
        self.system_prompt = self._initialize_system_prompt()

    def _validate_processor_state(self) -> None:
        image_processor = getattr(self.processor, "image_processor", None)
        if getattr(image_processor, "do_resize", None) is False:
            raise ValueError(
                "StandaloneInSightRuntime received a processor with image_processor.do_resize=False. "
                "This runtime owns model-facing image processing; use a separate processor instance "
                "for RLHFDataset or other dataset-side image materialization."
            )

    def _initialize_system_prompt(self) -> list[int]:
        token1 = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
        )
        token2 = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
        )
        return token1[: -(len(token2) - len(token1))]

    async def process_vision_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        patch_size = self.processor.image_processor.patch_size
        images, videos = await asyncio.to_thread(
            process_vision_info,
            messages,
            image_patch_size=patch_size,
            return_video_metadata=True,
        )
        multi_modal_data: dict[str, Any] = {}
        if images is not None:
            multi_modal_data["images"] = images
        if videos is not None:
            multi_modal_data["videos"] = videos
        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        def apply() -> list[int]:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            video_metadatas = None
            video_values = videos
            if videos is not None:
                video_values, video_metadatas = zip(*videos, strict=False)
                video_values = list(video_values)
                video_metadatas = list(video_metadatas)

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=video_values,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return model_inputs.pop("input_ids").squeeze(0).tolist()

        async with self._processor_semaphore:
            prompt_ids = await asyncio.to_thread(apply)
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

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
        return await self.endpoint_pool.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            images=image_data,
            videos=video_data,
            messages=messages,
            tools=tools,
        )

    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return await asyncio.to_thread(
            self.tokenizer.decode,
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    async def extract_tool_calls(self, response_ids: list[int]) -> list[CoreFunctionCall]:
        start = time.perf_counter()
        calls = await self.tool_call_extractor(response_ids)
        # Keep the call async and explicit; timing is owned by the core runner.
        _ = time.perf_counter() - start
        return calls
