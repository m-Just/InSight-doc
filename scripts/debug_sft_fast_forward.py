#!/usr/bin/env python
"""Dry-run the SFT data path for specific sampler steps without loading the training engine.

This reproduces:
1. MultiTurnSFTDataset row loading and multimodal preprocessing
2. DistributedSampler ordering
3. SFTTensorCollator batching
4. TensorDict construction
5. chunk_tensordict() micro-batch splitting

It is intended for fast-forwarding to a suspected failing training step.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler

from verl.trainer.sft_trainer import create_sft_dataset
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.tokenizer import hf_processor, hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", action="append", dest="train_files", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--step", type=int, default=None, help="1-based global training step within the epoch")
    parser.add_argument("--start-step", type=int, default=None, help="1-based inclusive")
    parser.add_argument("--end-step", type=int, default=None, help="1-based inclusive")
    parser.add_argument("--rank", type=int, action="append", dest="ranks", default=None)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--truncation", default="error")
    parser.add_argument("--ignore-input-ids-mismatch", action="store_true")
    return parser.parse_args()


def build_data_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "custom_cls": {},
            "pad_mode": "no_padding",
            "truncation": args.truncation,
            "max_length": args.max_length,
            "messages_key": "messages",
            "image_key": "images",
            "video_key": "videos",
            "tools_key": "tools",
            "message_loss_mask_key": "message_loss_mask",
            "enable_thinking_key": "enable_thinking",
            "apply_chat_template_kwargs": {},
            "shuffle": False,
            "seed": None,
            "ignore_input_ids_mismatch": args.ignore_input_ids_mismatch,
        }
    )


def resolve_step_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.step is not None:
        return args.step, args.step
    start_step = args.start_step if args.start_step is not None else 1
    end_step = args.end_step if args.end_step is not None else start_step
    if start_step < 1 or end_step < start_step:
        raise ValueError(f"invalid step range: start={start_step}, end={end_step}")
    return start_step, end_step


def summarize_batch(batch: dict) -> dict:
    summary = {}
    for key, value in batch.items():
        if hasattr(value, "is_nested") and value.is_nested:
            summary[key] = {"shape": str(value.shape), "dim": int(value.dim())}
        else:
            shape = getattr(value, "shape", None)
            summary[key] = {"type": type(value).__name__, "shape": str(shape)}
    return summary


def main() -> int:
    args = parse_args()
    if args.train_batch_size % args.nproc_per_node != 0:
        raise ValueError(
            f"train_batch_size={args.train_batch_size} must be divisible by nproc_per_node={args.nproc_per_node}"
        )

    local_batch_size = args.train_batch_size // args.nproc_per_node
    if local_batch_size < 1:
        raise ValueError("local batch size must be >= 1")

    start_step, end_step = resolve_step_range(args)
    ranks = args.ranks if args.ranks else list(range(args.nproc_per_node))
    model_path = Path(args.model_path)

    print(f"loading tokenizer from {model_path}", flush=True)
    tokenizer = hf_tokenizer(str(model_path), trust_remote_code=True, local_files_only=True)
    processor = hf_processor(str(model_path), trust_remote_code=True, local_files_only=True)
    if processor is None:
        raise RuntimeError(f"failed to load processor from {model_path}")

    data_config = build_data_config(args)
    print(f"building dataset from {len(args.train_files)} parquet files", flush=True)
    dataset = create_sft_dataset(args.train_files, data_config, tokenizer, processor, max_samples=-1)
    print(f"dataset len: {len(dataset)}", flush=True)

    collate_fn = SFTTensorCollator("no_padding")
    local_steps_per_epoch = len(dataset) // args.nproc_per_node // local_batch_size
    print(
        f"local_batch_size={local_batch_size} micro_batch_size_per_gpu={args.micro_batch_size_per_gpu} "
        f"epoch={args.epoch} step_range=[{start_step}, {end_step}] local_steps_per_epoch={local_steps_per_epoch}",
        flush=True,
    )

    if end_step > local_steps_per_epoch:
        raise ValueError(f"requested end_step={end_step} exceeds local_steps_per_epoch={local_steps_per_epoch}")

    for rank in ranks:
        sampler = DistributedSampler(
            list(range(len(dataset))),
            num_replicas=args.nproc_per_node,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        sampler.set_epoch(args.epoch)
        ordered_indices = list(iter(sampler))
        print(f"\nrank={rank} ordered_indices_len={len(ordered_indices)}", flush=True)

        for step in range(start_step, end_step + 1):
            start = (step - 1) * local_batch_size
            end = step * local_batch_size
            batch_indices = ordered_indices[start:end]
            samples = [dataset[idx] for idx in batch_indices]
            sample_info = [sample["debug_sample_info"] for sample in samples]
            print(f"rank={rank} step={step} batch_indices={batch_indices} sample_info={sample_info}", flush=True)

            batch = collate_fn(samples)
            print(f"rank={rank} step={step} batch_summary={summarize_batch(batch)}", flush=True)

            td = tu.get_tensordict(
                tensor_dict=batch,
                non_tensor_dict={
                    "use_dynamic_bsz": False,
                    "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                },
            )
            try:
                micro_batches = tu.chunk_tensordict(td, len(td) // args.micro_batch_size_per_gpu)
            except Exception as exc:
                print(
                    f"FAILED rank={rank} step={step} batch_indices={batch_indices} sample_info={sample_info} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                return 1

            print(
                f"rank={rank} step={step} chunk_ok num_micro_batches={len(micro_batches)}",
                flush=True,
            )

    print("completed without chunking failure", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
