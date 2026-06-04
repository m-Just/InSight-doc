from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from .runtime import CoreGenerationOutput


class RayVLLMEndpointPool:
    """Token-level generation backend using verl's Ray vLLM server actors.

    This mirrors the old verl agent-loop transport: agent workers call
    AsyncLLMServerManager.generate(), which forwards token ids and PIL images to
    vLLM server actors through Ray actor calls instead of a JSON/Base64 HTTP shim.
    """

    def __init__(self, server_handles: list[Any], *, manager_config: Any | None = None) -> None:
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

        if not server_handles:
            raise ValueError("at least one Ray vLLM server handle is required")
        self.server_handles = server_handles
        self.server_manager = AsyncLLMServerManager(manager_config or OmegaConf.create({}), server_handles)

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
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=dict(sampling_params),
            image_data=images,
            video_data=videos,
        )
        return CoreGenerationOutput(
            token_ids=list(output.token_ids),
            log_probs=output.log_probs,
            num_preempted=output.num_preempted,
        )
