from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol


@dataclass(frozen=True)
class RolloutJob:
    output_index: int
    job_key: str
    sample_index: int
    trial_idx: int
    resume_val_file: str
    resume_file_row_idx: int
    row: dict[str, Any]


class RolloutBackend(Protocol):
    backend_name: str

    async def prepare(self) -> None: ...

    async def load_rows(self, val_files: list[str], max_samples: int) -> list[dict[str, Any]]: ...

    def basic_config_extra(self) -> dict[str, Any]: ...

    async def generate_many(
        self,
        jobs: list[RolloutJob],
        on_sample: Callable[[RolloutJob, dict[str, Any]], Awaitable[None]],
    ) -> None: ...

    async def close(self) -> None: ...
