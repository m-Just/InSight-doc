from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class CoreFunctionCall:
    name: str
    arguments: str


@dataclass
class CoreGenerationOutput:
    token_ids: list[int]
    log_probs: list[float] | None = None
    num_preempted: int | None = None
    metrics: dict[str, float] | None = None


class CoreRuntime(Protocol):
    """Runtime services needed by token-level agent runners.

    Implementations can be backed by verl/Ray generation actors plus local
    HF tokenization and vision processing.
    """

    async def process_vision_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        ...

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        ...

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
        ...

    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ...

    async def extract_tool_calls(self, response_ids: list[int]) -> list[CoreFunctionCall]:
        ...
