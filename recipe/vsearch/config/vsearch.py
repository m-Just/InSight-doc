from dataclasses import dataclass

from omegaconf import MISSING

from verl.workers.config import AgentLoopConfig, FSDPActorConfig

__all__ = [
    "VSearchAgentLoopConfig",
    "VSearchFSDPActorConfig",
]


@dataclass
class VSearchAgentLoopConfig(AgentLoopConfig):
    vsearcher_loop_cls: str = MISSING


@dataclass
class VSearchFSDPActorConfig(FSDPActorConfig):
    skip_old_log_prob_recompute: bool = True
    force_on_policy: bool = True
