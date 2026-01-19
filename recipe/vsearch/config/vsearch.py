from dataclasses import dataclass, field

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig, RolloutConfig

__all__ = [
    "VSearchFSDPActorConfig",
]


@dataclass
class VSearchFSDPActorConfig(FSDPActorConfig):
    skip_old_log_prob_recompute: bool = True
    force_on_policy: bool = True
