from dataclasses import dataclass, field

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig, RolloutConfig

__all__ = [
    "VSearchFSDPActorConfig",
    "VSearchRolloutConfig",
]


@dataclass
class VSearchFSDPActorConfig(FSDPActorConfig):
    skip_old_log_prob_recompute: bool = True
    force_on_policy: bool = True


@dataclass
class VSearchRolloutConfig(RolloutConfig):
    # vLLM sampling params
    include_stop_str_in_output: bool = True
    stop: str | None = "</tool_call>"

    # vsearch-specific parameters
    use_vsearch: bool = True

    # behavior flags
    enable_tool_feedback: bool = False

    # safety/limits
    max_allowed_token_id: int | None = 151664
    max_num_batched_tokens: int = 8192
