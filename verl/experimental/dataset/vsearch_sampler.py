from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import logging
import math
import os

import torch
from omegaconf import DictConfig, OmegaConf

from verl.experimental.dataset.sampler import AbstractBatchSampler

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _coerce_weight_mapping(raw_weights: Any, *, source: str) -> Dict[str, float]:
    raw_weights = _plain_container(raw_weights)
    if raw_weights is None:
        return {}
    if not isinstance(raw_weights, Mapping):
        raise TypeError(f"{source} must be a mapping of data_source -> numeric weight")

    weights: Dict[str, float] = {}
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, int | float):
            raise TypeError(f"{source}.{key} must be numeric, got {type(value).__name__}")
        if float(value) > 0:
            weights[key] = float(value)
    return weights


def _coerce_string_list(raw_values: Any, *, source: str) -> List[str]:
    raw_values = _plain_container(raw_values)
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        values = [part.strip() for part in raw_values.split(",")]
    elif isinstance(raw_values, list | tuple):
        values = list(raw_values)
    else:
        raise TypeError(f"{source} must be a list of strings or a comma-separated string")

    output: List[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{source} entries must be strings, got {type(value).__name__}")
        value = value.strip()
        if value:
            output.append(value)
    return output


def _coerce_skew_extra_groups(raw_groups: Any) -> Dict[str, List[str]]:
    raw_groups = _plain_container(raw_groups)
    if raw_groups is None:
        return {}
    if not isinstance(raw_groups, Mapping):
        raise TypeError("batch_sampler.skew_extra_groups must be a mapping of group_name -> members")

    groups: Dict[str, List[str]] = {}
    for group_name, members in raw_groups.items():
        if not isinstance(group_name, str):
            raise TypeError("batch_sampler.skew_extra_groups keys must be strings")
        group_name = group_name.strip()
        if not group_name:
            raise ValueError("batch_sampler.skew_extra_groups group names must be non-empty")
        groups[group_name] = _coerce_string_list(
            members, source=f"batch_sampler.skew_extra_groups.{group_name}"
        )
        if not groups[group_name]:
            raise ValueError(f"batch_sampler.skew_extra_groups.{group_name} must not be empty")
    return groups


def _extract_weights_from_yaml(raw_config: Any, path: Path) -> Dict[str, float]:
    raw_config = _plain_container(raw_config)
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"batch_sampler.weights_file must contain a YAML mapping: {path}")

    if "data" in raw_config:
        data_config = _plain_container(raw_config["data"])
        if not isinstance(data_config, Mapping):
            raise TypeError(f"data in weights_file must be a mapping: {path}")
        if "batch_sampler" not in data_config:
            raise ValueError(f"data.batch_sampler not found in weights_file: {path}")
        return _extract_weights_from_yaml({"batch_sampler": data_config["batch_sampler"]}, path)

    if "batch_sampler" in raw_config:
        batch_sampler = raw_config["batch_sampler"]
        batch_sampler = _plain_container(batch_sampler)
        if not isinstance(batch_sampler, Mapping):
            raise TypeError(f"batch_sampler in weights_file must be a mapping: {path}")
        if "weights" not in batch_sampler:
            raise ValueError(f"batch_sampler.weights not found in weights_file: {path}")
        return _coerce_weight_mapping(batch_sampler["weights"], source=f"{path}:batch_sampler.weights")

    if "weights" in raw_config:
        return _coerce_weight_mapping(raw_config["weights"], source=f"{path}:weights")

    return _coerce_weight_mapping(raw_config, source=str(path))


def _load_batch_sampler_weights(data_config: DictConfig) -> Dict[str, float]:
    batch_sampler_config = data_config.batch_sampler
    weights: Dict[str, float] = {}

    weights_file = batch_sampler_config.get("weights_file", None)
    if weights_file is None:
        weights_file = batch_sampler_config.get("weight_file", None)
    if weights_file:
        weights_path = Path(os.path.expandvars(str(weights_file))).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"batch_sampler.weights_file does not exist: {weights_path}")
        weights.update(_extract_weights_from_yaml(OmegaConf.load(weights_path), weights_path))

    inline_weights = batch_sampler_config.get("weights", None)
    weights.update(_coerce_weight_mapping(inline_weights, source="data_config.batch_sampler.weights"))
    return weights


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
      For long source lists, the same mapping can be placed in
      `data_config.batch_sampler.weights_file`. The YAML file may be a plain
      mapping, a `weights:` mapping, a `batch_sampler.weights:` mapping, or a
      `data.batch_sampler.weights:` mapping.
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

        # Extract ratios mapping from inline config and/or an optional YAML file.
        ratios = _load_batch_sampler_weights(data_config)

        if not ratios:
            raise ValueError(
                "VSearchBatchSampler requires data_config.batch_sampler.weights or "
                "data_config.batch_sampler.weights_file to specify source ratios"
            )

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


class VSearchExhaustiveBatchSampler(VSearchBatchSampler):
    """VSearch batch sampler that continues after individual sources exhaust.

    This sampler keeps the same configuration contract as ``VSearchBatchSampler``:
    source ratios are read from inline ``data_config.batch_sampler.weights`` and/or
    ``data_config.batch_sampler.weights_file``, and rows are grouped by the dataset
    ``data_source`` column. The difference is exhaustion
    behavior. ``VSearchBatchSampler`` stops when the first source cannot satisfy
    its fixed per-batch quota; this sampler drops exhausted sources from the
    active set, renormalizes the requested ratios over remaining sources, and
    keeps yielding full batches until fewer than ``batch_size`` rows remain.

    Rows are still sampled without replacement within each source for an epoch,
    and ``state_dict`` / ``load_state_dict`` are inherited for StatefulDataLoader
    checkpoint/resume support.
    """

    def __init__(self, batch_size: int, data_source: Sized, data_config: DictConfig):
        super().__init__(batch_size=batch_size, data_source=data_source, data_config=data_config)

        configured_weights = _load_batch_sampler_weights(data_config)
        self.source_weights: Dict[str, float] = {
            key: value
            for key, value in configured_weights.items()
            if key in self.source_to_indices and len(self.source_to_indices[key]) > 0 and value > 0
        }

        self.source_order: List[str] = sorted(self.source_weights.keys())
        if not self.source_order:
            raise ValueError("VSearchExhaustiveBatchSampler found no positive-weight sources")

        self._positions = {s: 0 for s in self.source_order}

        # Replace the parent length, which stops at the first exhausted source.
        self._num_batches = self._compute_num_batches_until_global_exhaustion()
        logger.info(f"VSearchExhaustiveBatchSampler initialized: {self}")

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["positions"] = {s: int(self._positions.get(s, 0)) for s in self.source_order}
        return state

    def load_state_dict(self, state: Dict) -> None:
        self.seed = int(state.get("seed", 1))
        self._epoch = int(state.get("epoch", 0))
        positions = state.get("positions", {})
        self._positions = {s: int(positions.get(s, 0)) for s in self.source_order}
        self._yielded_batches = int(state.get("yielded_batches", 0))
        self._prepared_epoch = None

    def _prepare_epoch(self) -> None:
        super()._prepare_epoch()
        for source in self.source_order:
            self._positions.setdefault(source, 0)

    def _remaining_by_source(self) -> Dict[str, int]:
        return {
            s: len(self.source_to_indices[s]) - int(self._positions.get(s, 0))
            for s in self.source_order
        }

    def _allocate_batch_counts(self, remaining_by_source: Dict[str, int]) -> Dict[str, int]:
        total_remaining = sum(max(0, remaining_by_source.get(s, 0)) for s in self.source_order)
        if total_remaining < self.batch_size:
            return {}

        active_sources = [s for s in self.source_order if remaining_by_source.get(s, 0) > 0]
        if not active_sources:
            return {}

        total_weight = sum(self.source_weights[s] for s in active_sources)
        if total_weight <= 0:
            return {}

        allocations: Dict[str, int] = {s: 0 for s in active_sources}
        remainders: List[tuple[str, float]] = []
        allocated = 0

        for s in active_sources:
            exact = self.batch_size * self.source_weights[s] / total_weight
            base = min(int(exact // 1), remaining_by_source[s])
            allocations[s] = base
            allocated += base
            remainders.append((s, exact - base))

        slots_left = self.batch_size - allocated
        # Deterministic tie-breaker by (-fractional, source_name), matching the parent style.
        remainders.sort(key=lambda x: (-x[1], x[0]))
        while slots_left > 0:
            made_progress = False
            for s, _ in remainders:
                if allocations[s] >= remaining_by_source[s]:
                    continue
                allocations[s] += 1
                slots_left -= 1
                made_progress = True
                if slots_left == 0:
                    break
            if not made_progress:
                break

        if sum(allocations.values()) != self.batch_size:
            return {}
        return {s: count for s, count in allocations.items() if count > 0}

    def _compute_num_batches_until_global_exhaustion(self) -> int:
        remaining = {s: len(self.source_to_indices[s]) for s in self.source_order}
        num_batches = 0
        while True:
            counts = self._allocate_batch_counts(remaining)
            if not counts:
                return num_batches
            for s, count in counts.items():
                remaining[s] -= count
            num_batches += 1

    def __iter__(self) -> Iterator[List[int]]:
        self._prepare_epoch()

        for _ in range(self._yielded_batches, self._num_batches):
            counts = self._allocate_batch_counts(self._remaining_by_source())
            if not counts:
                return

            batch: List[int] = []
            for s in self.source_order:
                count = counts.get(s, 0)
                if count <= 0:
                    continue
                pos = self._positions[s]
                next_pos = pos + count
                batch.extend(self._shuffled_indices[s][pos:next_pos])
                self._positions[s] = next_pos

            self._yielded_batches += 1
            yield batch

    def __repr__(self) -> str:
        return (
            f"VSearchExhaustiveBatchSampler(batch_size={self.batch_size}, num_batches={len(self)}, "
            f"initial_batch_size_by_source={self.per_batch_counts}, seed={self.seed})"
        )


class VSearchWeightedRandomRefillBatchSampler(AbstractBatchSampler):
    """Weighted random source sampler with per-source refill-on-exhaustion.

    This sampler is intended for many fine-grained ``data_source`` buckets where
    deterministic per-batch quotas would round small weights to zero. Each batch
    slot samples a source with probability proportional to configured source
    weights, then pops one row from that source's shuffled pool. When a source
    pool is empty, it is reshuffled and refilled, so low-cardinality sources are
    oversampled by cycling without replacement within each source cycle.

    The default and currently supported stopping rule is ``max_source_exhaustion``:
    generate full batches until every configured source has completed at least
    one pass through its original row pool.

    Optional skew rejection can redraw only extreme source-composition tails while
    preserving ordinary multinomial variation. It is disabled by default.
    """

    def __init__(self, batch_size: int, data_source: Sized, data_config: DictConfig):
        self.batch_size: int = int(batch_size)
        self.data_source: Sized = data_source
        self.data_config: DictConfig = data_config

        config_seed = data_config.seed
        if config_seed is None:
            self.seed: int = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            self.seed: int = int(config_seed)

        batch_sampler_config = data_config.batch_sampler
        self.stop_after = str(batch_sampler_config.get("stop_after", "max_source_exhaustion"))
        if self.stop_after != "max_source_exhaustion":
            raise ValueError(
                "VSearchWeightedRandomRefillBatchSampler currently only supports "
                "batch_sampler.stop_after=max_source_exhaustion"
            )
        self.skew_tail_p: float = float(batch_sampler_config.get("skew_tail_p", 0.0) or 0.0)
        self.skew_grouping: str = str(batch_sampler_config.get("skew_grouping", "family_prefix"))
        self.skew_extra_groups: Dict[str, List[str]] = _coerce_skew_extra_groups(
            batch_sampler_config.get("skew_extra_groups", None)
        )
        self.max_resample_attempts: int = int(batch_sampler_config.get("max_resample_attempts", 20) or 20)
        if self.skew_tail_p < 0:
            raise ValueError("batch_sampler.skew_tail_p must be >= 0")
        if self.max_resample_attempts <= 0:
            raise ValueError("batch_sampler.max_resample_attempts must be > 0")

        configured_weights = _load_batch_sampler_weights(data_config)
        if not configured_weights:
            raise ValueError(
                "VSearchWeightedRandomRefillBatchSampler requires data_config.batch_sampler.weights "
                "or data_config.batch_sampler.weights_file"
            )

        try:
            hf_dataset = getattr(self.data_source, "dataframe")
        except AttributeError as exc:
            raise TypeError("data_source must be an RLHFDataset with a 'dataframe' attribute") from exc

        if "data_source" not in hf_dataset.column_names:
            raise ValueError("RLHFDataset.dataframe must contain a 'data_source' column")

        source_column: List[str] = hf_dataset["data_source"]
        desired_sources = set(configured_weights.keys())
        self.source_to_indices: Dict[str, List[int]] = {s: [] for s in desired_sources}
        for idx, src in enumerate(source_column):
            if src in self.source_to_indices:
                self.source_to_indices[src].append(idx)

        missing_sources = {s for s in desired_sources if len(self.source_to_indices[s]) == 0}
        if missing_sources:
            available_sources = set(source_column)
            raise ValueError(
                f"Data sources specified in batch_sampler.weights not found in dataset: {missing_sources}. "
                f"Available data sources in dataset: {available_sources}"
            )

        self.source_weights: Dict[str, float] = {
            key: value
            for key, value in configured_weights.items()
            if key in self.source_to_indices and len(self.source_to_indices[key]) > 0 and value > 0
        }
        self.source_order: List[str] = sorted(self.source_weights.keys())
        if not self.source_order:
            raise ValueError("VSearchWeightedRandomRefillBatchSampler found no positive-weight sources")
        self._skew_groups = self._build_skew_groups()
        self._skew_group_probs = self._build_skew_group_probs()
        self._legacy_replay_until_yielded = False
        self._replay_skew_tail_p = self.skew_tail_p
        self._replay_skew_groups = self._skew_groups
        self._replay_skew_group_probs = self._skew_group_probs

        self._epoch: int = 0
        self._prepared_epoch: Optional[int] = None
        self._yielded_batches: int = 0
        self._epoch_batches: List[List[int]] = []
        self._num_batches: int = 0

        logger.info(f"VSearchWeightedRandomRefillBatchSampler initialized: {self}")

    def __len__(self) -> int:
        self._prepare_epoch()
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:
        new_epoch = int(epoch)
        prev_epoch = getattr(self, "_epoch", None)
        self._epoch = new_epoch
        if prev_epoch is None or new_epoch != prev_epoch:
            self._yielded_batches = 0
        self._prepared_epoch = None

    def state_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "prepared_epoch": self._prepared_epoch,
            "yielded_batches": self._yielded_batches,
            "skew_rejection_enabled": self.skew_tail_p > 0,
            "skew_tail_p": self.skew_tail_p,
            "skew_grouping": self.skew_grouping,
            "skew_extra_groups": self.skew_extra_groups,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.seed = int(state.get("seed", 1))
        self._epoch = int(state.get("epoch", 0))
        self._yielded_batches = int(state.get("yielded_batches", 0))
        state_skew_enabled = bool(state.get("skew_rejection_enabled", False))
        state_skew_tail_p = float(state.get("skew_tail_p", self.skew_tail_p) or 0.0) if state_skew_enabled else 0.0
        state_skew_grouping = str(state.get("skew_grouping", self.skew_grouping))
        state_skew_extra_groups = _coerce_skew_extra_groups(state.get("skew_extra_groups", None))
        state_skew_signature = (state_skew_tail_p, state_skew_grouping, state_skew_extra_groups)
        current_skew_signature = (self.skew_tail_p, self.skew_grouping, self.skew_extra_groups)
        self._legacy_replay_until_yielded = (
            self._yielded_batches > 0 and state_skew_signature != current_skew_signature
        )
        self._replay_skew_tail_p = state_skew_tail_p
        self._replay_skew_groups = self._build_skew_groups(
            skew_tail_p=state_skew_tail_p,
            skew_grouping=state_skew_grouping,
            skew_extra_groups=state_skew_extra_groups,
        )
        self._replay_skew_group_probs = self._build_skew_group_probs(self._replay_skew_groups)
        self._prepared_epoch = None

    def _reshuffle_source(self, source: str, generator: torch.Generator) -> List[int]:
        indices = self.source_to_indices[source]
        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in perm]

    def _skew_group_key(self, source: str, skew_grouping: Optional[str] = None) -> str:
        skew_grouping = self.skew_grouping if skew_grouping is None else skew_grouping
        if skew_grouping in ("none", ""):
            return source
        if skew_grouping == "source":
            return source
        if skew_grouping == "category":
            for suffix in ("_answerable", "_unanswerable"):
                if source.endswith(suffix):
                    return source[: -len(suffix)]
            return source
        if skew_grouping == "family_prefix":
            return source.split("_", 1)[0]
        raise ValueError(
            "batch_sampler.skew_grouping must be one of: "
            "family_prefix, category, source, none"
        )

    def _build_skew_groups(
        self,
        *,
        skew_tail_p: Optional[float] = None,
        skew_grouping: Optional[str] = None,
        skew_extra_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        skew_tail_p = self.skew_tail_p if skew_tail_p is None else skew_tail_p
        skew_grouping = self.skew_grouping if skew_grouping is None else skew_grouping
        skew_extra_groups = self.skew_extra_groups if skew_extra_groups is None else skew_extra_groups
        if skew_tail_p <= 0:
            return {}

        groups: Dict[str, List[str]] = {}
        base_group_by_source: Dict[str, str] = {
            source: self._skew_group_key(source, skew_grouping) for source in self.source_order
        }

        if skew_grouping not in ("none", ""):
            for source, group_key in base_group_by_source.items():
                groups.setdefault(group_key, []).append(source)

        for group_name, members in skew_extra_groups.items():
            member_set = set(members)
            extra_sources = [
                source
                for source in self.source_order
                if source in member_set or base_group_by_source[source] in member_set
            ]
            if not extra_sources:
                raise ValueError(
                    f"batch_sampler.skew_extra_groups.{group_name} did not match any "
                    "configured data_source or skew_grouping key"
                )
            merged_sources = set(groups.get(group_name, []))
            merged_sources.update(extra_sources)
            groups[group_name] = [source for source in self.source_order if source in merged_sources]
        return {group: sources for group, sources in groups.items() if len(sources) > 0}

    def _build_skew_group_probs(self, skew_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
        skew_groups = self._skew_groups if skew_groups is None else skew_groups
        if not skew_groups:
            return {}

        total_weight = sum(self.source_weights[source] for source in self.source_order)
        if total_weight <= 0:
            return {}
        return {
            group: sum(self.source_weights[source] for source in sources) / total_weight
            for group, sources in skew_groups.items()
        }

    @staticmethod
    def _binomial_upper_tail(n: int, p: float, k: int) -> float:
        if k <= 0:
            return 1.0
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0 if k <= n else 0.0
        return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))

    def _is_skewed_batch(
        self,
        source_counts: Dict[str, int],
        *,
        skew_tail_p: Optional[float] = None,
        skew_groups: Optional[Dict[str, List[str]]] = None,
        skew_group_probs: Optional[Dict[str, float]] = None,
    ) -> bool:
        skew_tail_p = self.skew_tail_p if skew_tail_p is None else skew_tail_p
        skew_groups = self._skew_groups if skew_groups is None else skew_groups
        skew_group_probs = self._skew_group_probs if skew_group_probs is None else skew_group_probs
        if skew_tail_p <= 0 or not skew_groups:
            return False

        for group, sources in skew_groups.items():
            group_count = sum(source_counts.get(source, 0) for source in sources)
            group_prob = skew_group_probs[group]
            if group_count <= self.batch_size * group_prob:
                continue
            tail_p = self._binomial_upper_tail(self.batch_size, group_prob, group_count)
            if tail_p <= skew_tail_p:
                return True
        return False

    def _draw_candidate_batch(
        self,
        *,
        pools: Dict[str, List[int]],
        positions: Dict[str, int],
        exhausted_once: Dict[str, bool],
        weights: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[List[int], Dict[str, List[int]], Dict[str, int], Dict[str, bool], Dict[str, int]]:
        next_pools = dict(pools)
        next_positions = dict(positions)
        next_exhausted_once = dict(exhausted_once)
        source_counts: Dict[str, int] = {}
        batch: List[int] = []

        for _ in range(self.batch_size):
            source_idx = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
            source = self.source_order[source_idx]
            batch.append(next_pools[source][next_positions[source]])
            source_counts[source] = source_counts.get(source, 0) + 1
            next_positions[source] += 1

            if next_positions[source] >= len(next_pools[source]):
                next_exhausted_once[source] = True
                next_pools[source] = self._reshuffle_source(source, generator)
                next_positions[source] = 0

        return batch, next_pools, next_positions, next_exhausted_once, source_counts

    def _should_apply_skew_rejection(self, batch_idx: int) -> bool:
        if self._legacy_replay_until_yielded and batch_idx < self._yielded_batches:
            return self._replay_skew_tail_p > 0 and bool(self._replay_skew_groups)
        return self.skew_tail_p > 0 and bool(self._skew_groups)

    def _is_skewed_batch_for_index(self, source_counts: Dict[str, int], batch_idx: int) -> bool:
        # When rejection config changes across resume, preserve the accepted
        # pre-checkpoint stream using the checkpoint's old skew config, then
        # switch to the current config for newly yielded batches.
        if self._legacy_replay_until_yielded and batch_idx < self._yielded_batches:
            return self._is_skewed_batch(
                source_counts,
                skew_tail_p=self._replay_skew_tail_p,
                skew_groups=self._replay_skew_groups,
                skew_group_probs=self._replay_skew_group_probs,
            )
        return self._is_skewed_batch(source_counts)

    def _prepare_epoch(self) -> None:
        if self._prepared_epoch == self._epoch:
            return

        generator = torch.Generator()
        generator.manual_seed((self.seed + self._epoch) % (2**32))

        pools = {source: self._reshuffle_source(source, generator) for source in self.source_order}
        positions = {source: 0 for source in self.source_order}
        exhausted_once = {source: False for source in self.source_order}
        weights = torch.tensor([self.source_weights[source] for source in self.source_order], dtype=torch.double)

        epoch_batches: List[List[int]] = []
        while True:
            batch_idx = len(epoch_batches)
            apply_rejection = self._should_apply_skew_rejection(batch_idx)
            for attempt_idx in range(self.max_resample_attempts):
                batch, next_pools, next_positions, next_exhausted_once, source_counts = self._draw_candidate_batch(
                    pools=pools,
                    positions=positions,
                    exhausted_once=exhausted_once,
                    weights=weights,
                    generator=generator,
                )
                if not apply_rejection or not self._is_skewed_batch_for_index(source_counts, batch_idx):
                    break
            else:
                logger.warning(
                    "Accepted skewed batch after %d rejected attempts "
                    "(skew_tail_p=%s, skew_grouping=%s)",
                    self.max_resample_attempts,
                    self.skew_tail_p,
                    self.skew_grouping,
                )

            pools = next_pools
            positions = next_positions
            exhausted_once = next_exhausted_once

            epoch_batches.append(batch)
            if all(exhausted_once.values()):
                break

        self._epoch_batches = epoch_batches
        self._num_batches = len(epoch_batches)
        self._prepared_epoch = self._epoch

    def __iter__(self) -> Iterator[List[int]]:
        if self._prepared_epoch == self._epoch and self._yielded_batches >= self._num_batches:
            self._epoch += 1
            self._yielded_batches = 0
            self._prepared_epoch = None

        self._prepare_epoch()

        for batch_idx in range(self._yielded_batches, self._num_batches):
            self._yielded_batches = batch_idx + 1
            yield list(self._epoch_batches[batch_idx])

    def __repr__(self) -> str:
        return (
            f"VSearchWeightedRandomRefillBatchSampler(batch_size={self.batch_size}, "
            f"num_batches={self._num_batches}, stop_after={self.stop_after}, "
            f"skew_tail_p={self.skew_tail_p}, skew_grouping={self.skew_grouping}, "
            f"skew_extra_groups={self.skew_extra_groups}, seed={self.seed})"
        )
