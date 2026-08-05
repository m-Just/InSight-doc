from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING

from verl.workers.config import AgentLoopConfig, FSDPActorConfig

__all__ = [
    "VSearchAgentLoopConfig",
    "VSearchFSDPActorConfig",
]


@dataclass
class VSearchAgentLoopConfig(AgentLoopConfig):
    vsearcher_loop_cls: str = MISSING
    vreasoner_v2_conversation_export_dir: Optional[str] = None
    vreasoner_v2_conversation_export_resume_mode: str = "off"
    vreasoner_v2_profile_dir: Optional[str] = None


@dataclass
class VSearchFSDPActorConfig(FSDPActorConfig):
    skip_old_log_prob_recompute: bool = True
    force_on_policy: bool = True
