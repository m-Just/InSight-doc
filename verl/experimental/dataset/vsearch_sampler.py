from collections.abc import Sized
from typing import Dict, Iterator, List, Optional
import logging
import os

import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractBatchSampler

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VSearchBatchSampler(AbstractBatchSampler):
    """Batch sampler that enforces fixed per-batch data-source ratios.

    Key features:
    - Deterministic per-epoch shuffling: Uses a seeded torch.Generator and `set_epoch(epoch)`
      for reproducible shuffling with `torchdata.stateful_dataloader.StatefulDataLoader`.
      If no seed is provided, a random seed is auto-generated (similar to TorchData's approach).
    - Fixed composition per batch: Computes per-source counts via the largest remainder method
      to match requested ratios within rounding error, and concatenates indices accordingly.
    - Exhaustion-based length: The number of batches equals the minimum over
      floor(len(source_indices) / per_batch_count[source]) across all participating sources.
    - Stateful resume: Supports `state_dict()` / `load_state_dict()` for seamless resumption
      mid-epoch with identical remaining order. The seed is saved in checkpoints, allowing
      shuffle regeneration on resume.

    Configuration:
    - Expects ratios under `data_config.batch_sampler.weights`, e.g.:
      {"batch_sampler": {"weights": {"datasetA": 0.2, "datasetB": 0.8}}}
      The sampler will also read `seed` from `data_config` if present. If not provided,
      a random seed is auto-generated.
    - The dataset must be an `RLHFDataset` exposing a HuggingFace `datasets.Dataset` as
      `dataframe` and include a `"data_source"` column.

    Caveats:
    - Designed to be used as a `batch_sampler` in `StatefulDataLoader`; the dataloader should
      not pass a separate `batch_size`/`shuffle`/`sampler`/`drop_last`. This sampler drops the
      last batch if it's incomplete (i.e. drop_last=True).
    - Sources with zero available samples are ignored. If rounding leads to a per-batch count
      of zero for a source, that source is dropped from the per-batch composition.
    - No oversampling is performed. Once a source is exhausted, iteration stops.
    - Raises `ValueError` if weights are missing or if none of the specified sources exist in
      the dataset.
    """
    def __init__(self, batch_size: int, data_source: Sized, data_config: DictConfig):
        # Basic args
        self.batch_size: int = int(batch_size)
        self.data_source: Sized = data_source
        self.data_config: DictConfig = data_config

        # Seed for reproducible shuffling across epochs.
        # If no seed is provided, auto-generate a random one (like TorchData's RandomSampler).
        config_seed = data_config.seed
        if config_seed is None:
            # Generate a random seed using torch's random number generator
            self.seed: int = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            self.seed: int = int(config_seed)

        # Extract ratios mapping from data_config. The user passes a DictConfig like
        # {"datasetA": 0.2, "datasetB": 0.8}. We treat string->float pairs as ratios.
        ratios: Dict[str, float] = {}
        for key, value in dict(data_config.batch_sampler.weights).items():
            if isinstance(key, str) and isinstance(value, int | float):
                ratios[key] = float(value)

        if not ratios:
            raise ValueError("VSearchBatchSampler requires data_config.batch_sampler.weights to specify source ratios")

        # Build mapping from source name -> list of indices
        try:
            hf_dataset = getattr(self.data_source, "dataframe")
        except AttributeError as exc:
            raise TypeError("data_source must be an RLHFDataset with a 'dataframe' attribute") from exc

        if "data_source" not in hf_dataset.column_names:
            raise ValueError("RLHFDataset.dataframe must contain a 'data_source' column")

        source_column: List[str] = hf_dataset["data_source"]

        # Keep only sources present in ratios and actually existing in the dataset
        desired_sources = set(ratios.keys())
        self.source_to_indices: Dict[str, List[int]] = {s: [] for s in desired_sources}
        for idx, src in enumerate(source_column):
            if src in self.source_to_indices:
                self.source_to_indices[src].append(idx)

        # Validate that all requested sources exist in the dataset
        missing_sources = {s for s in desired_sources if len(self.source_to_indices[s]) == 0}
        if missing_sources:
            available_sources = set(source_column)
            raise ValueError(
                f"Data sources specified in batch_sampler.weights not found in dataset: {missing_sources}. "
                f"Available data sources in dataset: {available_sources}"
            )

        # Renormalize ratios to present sources only
        present_sources = list(self.source_to_indices.keys())
        ratios_present: Dict[str, float] = {s: ratios.get(s, 0.0) for s in present_sources}
        ratio_sum = sum(ratios_present.values())
        if ratio_sum <= 0:
            raise ValueError("Sum of provided ratios over present sources must be > 0")
        for s in ratios_present:
            ratios_present[s] = ratios_present[s] / ratio_sum

        # Compute per-batch counts using largest remainder method for exact batch size
        base_counts: Dict[str, int] = {}
        remainders: List[tuple[str, float]] = []
        total_base = 0
        for s in present_sources:
            exact = ratios_present[s] * self.batch_size
            base = int(exact // 1)
            base_counts[s] = base
            total_base += base
            remainders.append((s, exact - base))

        remaining = self.batch_size - total_base
        # Deterministic tie-breaker by (-fractional, source_name)
        remainders.sort(key=lambda x: (-x[1], x[0]))
        for i in range(remaining):
            s = remainders[i % len(remainders)][0]
            base_counts[s] += 1

        # Drop any source that still has 0 count in a batch
        self.per_batch_counts: Dict[str, int] = {s: c for s, c in base_counts.items() if c > 0}
        if not self.per_batch_counts:
            raise ValueError("Per-batch counts are all zero; increase batch_size or adjust ratios")

        # Pre-compute number of batches before the first source exhausts
        def safe_div(a: int, b: int) -> int:
            return a // b if b > 0 else float("inf")

        batches_per_source: List[int] = [
            safe_div(len(self.source_to_indices[s]), self.per_batch_counts.get(s, 0))
            for s in self.per_batch_counts
        ]
        self._num_batches: int = min(batches_per_source) if batches_per_source else 0

        # Runtime iteration state (for StatefulDataLoader compatibility)
        self._epoch: int = 0
        self._prepared_epoch: Optional[int] = None
        self._shuffled_indices: Dict[str, List[int]] = {}
        self._positions: Dict[str, int] = {s: 0 for s in self.per_batch_counts.keys()}
        self._yielded_batches: int = 0

        logger.info(f'VSearchBatchSampler initialized: {self}')

    def __len__(self) -> int:
        return self._num_batches

    # StatefulDataLoader compatibility
    def set_epoch(self, epoch: int) -> None:
        new_epoch = int(epoch)
        prev_epoch = getattr(self, "_epoch", None)
        self._epoch = new_epoch
        # If advancing to a new epoch, reset progress so iteration restarts
        if prev_epoch is None or new_epoch != prev_epoch:
            self._yielded_batches = 0
        # Force reshuffle on next iteration
        self._prepared_epoch = None

    def state_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "prepared_epoch": self._prepared_epoch,
            "positions": dict(self._positions),
            "yielded_batches": self._yielded_batches,
        }

    def load_state_dict(self, state: Dict) -> None:
        # Restore the seed; default to 1 if not found in state_dict (e.g., legacy checkpoints)
        self.seed = int(state.get("seed", 1))
        self._epoch = int(state.get("epoch", 0))
        positions = state.get("positions", {})
        for s in self.per_batch_counts.keys():
            self._positions[s] = int(positions.get(s, 0))
        self._yielded_batches = int(state.get("yielded_batches", 0))
        # Force regeneration of deterministic shuffle for this epoch on next __iter__
        # so that resuming in a new process reproduces the same order.
        self._prepared_epoch = None

    def _prepare_epoch(self) -> None:
        # If the previous iteration exhausted all batches and a new iteration starts
        # without an explicit set_epoch(), advance to the next epoch and reset progress.
        if self._yielded_batches >= self._num_batches:
            self._epoch += 1
            self._yielded_batches = 0
            self._prepared_epoch = None

        if self._prepared_epoch == self._epoch:
            return

        # Deterministic per-epoch shuffling using torch.Generator
        # Use modulo 2**32 to ensure the seed fits in 32 bits (manual_seed requirement)
        g = torch.Generator()
        g.manual_seed((self.seed + self._epoch) % (2**32))

        self._shuffled_indices = {}
        for s, idxs in self.source_to_indices.items():
            if len(idxs) == 0:
                self._shuffled_indices[s] = []
                continue
            perm = torch.randperm(len(idxs), generator=g).tolist()
            self._shuffled_indices[s] = [idxs[i] for i in perm]
        # Only reset counters if starting a fresh epoch (not resuming mid-epoch)
        if self._yielded_batches == 0:
            self._positions = {s: 0 for s in self.per_batch_counts.keys()}
        self._prepared_epoch = self._epoch

    def __iter__(self) -> Iterator[List[int]]:
        self._prepare_epoch()

        for _ in range(self._yielded_batches, self._num_batches):
            batch: List[int] = []
            for s, count in self.per_batch_counts.items():
                pos = self._positions[s]
                next_pos = pos + count
                batch.extend(self._shuffled_indices[s][pos:next_pos])
                self._positions[s] = next_pos
            self._yielded_batches += 1
            yield batch

    def __repr__(self) -> str:
        return (
            f"VSearchBatchSampler(batch_size={self.batch_size}, num_batches={len(self)}, "
            f"batch_size_by_source={self.per_batch_counts}, seed={self.seed})"
        )