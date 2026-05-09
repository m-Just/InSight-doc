#!/usr/bin/env python3
"""Reproduce the 3D jagged position_ids chunking failure and validate a manual split workaround.

This script covers two Qwen-VL-style nested layouts:
1. equal sequence lengths -> nested shape like [batch, j, seq_len]
2. unequal sequence lengths -> nested shape like [batch, 4, j]

The real training crash came from case (1): two samples both shaped (4, 986).
For that case, forcing `_ragged_idx = 2` makes nested `unbind()` fail with the
same `split_with_sizes ... got [4, 4]` error seen in training.

The script shows:
1. the current nested `unbind()` path fails in the equal-length case
2. a manual split using `position_ids.values()` succeeds for both layouts
3. the reconstructed chunked tensors match the expected per-sample tensors exactly
"""

from __future__ import annotations

import sys

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu


def build_synthetic_batch(seq_lens: list[int]) -> tuple[TensorDict, list[torch.Tensor]]:
    input_ids_samples = []
    position_ids_samples = []
    for sample_idx, seq_len in enumerate(seq_lens):
        input_ids_samples.append(torch.arange(sample_idx * 1000, sample_idx * 1000 + seq_len, dtype=torch.long))
        position_ids_samples.append(
            torch.stack(
                [torch.arange(seq_len, dtype=torch.long) + (sample_idx + 1) * 1000 * (axis + 1) for axis in range(4)],
                dim=0,
            )
        )
    loss_mask_samples = [torch.ones_like(x) for x in input_ids_samples]

    td = tu.get_tensordict(
        tensor_dict={
            "input_ids": torch.nested.as_nested_tensor(input_ids_samples, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(position_ids_samples, layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(loss_mask_samples, layout=torch.jagged),
            "debug_sample_info": [
                {"sample": i, "length": int(input_ids_samples[i].numel())}
                for i in range(len(input_ids_samples))
            ],
        },
        non_tensor_dict={
            "micro_batch_size_per_gpu": 1,
            "use_dynamic_bsz": False,
        },
    )
    return td, position_ids_samples


def split_3d_nested_by_input_lengths(input_ids: torch.Tensor, position_ids: torch.Tensor) -> list[torch.Tensor]:
    """Split a 3D jagged nested tensor sample-by-sample without calling nested unbind().

    Handles both observed layouts for per-sample tensors shaped (4, seq_len):
    - equal seq_len across samples: position_ids.values() has shape (batch * 4, seq_len)
    - unequal seq_len across samples: position_ids.values() has shape (4, total_seq_len)
    """
    lengths = input_ids.offsets().diff().tolist()
    flat_values = position_ids.values()
    if flat_values.dim() != 2:
        raise RuntimeError(f"Expected position_ids.values() to be 2D, got {tuple(flat_values.shape)}")

    num_samples = len(lengths)
    if all(length == lengths[0] for length in lengths) and flat_values.shape[0] == num_samples * 4:
        return [flat_values[i * 4 : (i + 1) * 4, : lengths[i]].clone() for i in range(num_samples)]

    if flat_values.shape[0] == 4:
        pieces = []
        start = 0
        for length in lengths:
            end = start + int(length)
            pieces.append(flat_values[:, start:end].clone())
            start = end
        if start != flat_values.shape[1]:
            raise RuntimeError(
                f"Manual split consumed {start} tokens, but position_ids.values() has {flat_values.shape[1]} columns"
            )
        return pieces

    raise RuntimeError(
        "Unsupported 3D jagged position_ids values layout: "
        f"shape={tuple(flat_values.shape)} lengths={lengths}"
    )


def manual_chunk_tensordict(td: TensorDict, chunks: int) -> list[TensorDict]:
    """Chunk a TensorDict while manually splitting 3D jagged nested tensors."""
    assert len(td) % chunks == 0
    chunk_size = len(td) // chunks
    keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested and val.dim() >= 3}
    new_td = TensorDict({k: v for k, v in td.items() if k not in keys}, batch_size=td.batch_size, device=td.device)
    tds = new_td.chunk(chunks=chunks)

    for key in keys:
        if key != "position_ids":
            raise RuntimeError(f"Unexpected 3D nested key {key!r}; this test only handles position_ids")
        tensors = split_3d_nested_by_input_lengths(td["input_ids"], td[key])
        for i, chunk_td in enumerate(tds):
            chunk_td[key] = torch.nested.as_nested_tensor(
                tensors[i * chunk_size : (i + 1) * chunk_size], layout=torch.jagged
            )

    return tds


def main() -> int:
    cases = [
        ("equal_lengths_repro", [986, 986], True),
        ("unequal_lengths_sanity", [7, 5], False),
    ]

    for case_name, seq_lens, should_fail_unbind in cases:
        td, expected_position_ids = build_synthetic_batch(seq_lens)

        print(f"\n=== {case_name} ===")
        print("  input_ids.shape    =", td["input_ids"].shape)
        print("  position_ids.shape =", td["position_ids"].shape)
        print("  position_ids ragged_idx before =", getattr(td["position_ids"], "_ragged_idx", None))
        print("  position_ids.values().shape    =", tuple(td["position_ids"].values().shape))

        if should_fail_unbind:
            td["position_ids"]._ragged_idx = 2
            print("  position_ids ragged_idx after  =", getattr(td["position_ids"], "_ragged_idx", None))

            print("  repro current failure")
            try:
                _ = td["position_ids"].unbind(dim=0)
            except Exception as exc:
                print("    expected unbind failure:", type(exc).__name__, exc)
            else:
                print("    ERROR: unbind unexpectedly succeeded")
                return 1
        else:
            print("  direct unbind sanity")
            samples = td["position_ids"].unbind(dim=0)
            print("    sample_shapes =", [tuple(x.shape) for x in samples])

        print("  run manual workaround")
        chunks = manual_chunk_tensordict(td, chunks=2)
        for i, chunk in enumerate(chunks):
            sample = chunk["position_ids"].unbind(dim=0)[0]
            expected = expected_position_ids[i]
            ok = torch.equal(sample, expected)
            print(
                f"    chunk {i}: position_ids.shape={chunk['position_ids'].shape}, "
                f"sample_shape={tuple(sample.shape)}, matches_expected={ok}"
            )
            if not ok:
                print("    ERROR: reconstructed sample does not match expected tensor")
                return 1

        micro_batch_size = tu.get_non_tensor_data(chunks[0], "micro_batch_size_per_gpu", None)
        if micro_batch_size != 1:
            print(f"    ERROR: expected non-tensor metadata to survive chunking, got {micro_batch_size!r}")
            return 1
        print("    metadata preserved:", micro_batch_size)

    print("\nPASS: manual split bypasses the broken 3D nested unbind path and preserves outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
