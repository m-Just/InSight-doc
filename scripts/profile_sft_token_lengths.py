#!/usr/bin/env python
"""Profile post-tokenization SFT sequence lengths, including image tokens."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import create_sft_dataset
from verl.utils.tokenizer import hf_processor, hf_tokenizer


DEFAULT_TRAIN_FILES = [
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2b/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2b_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2c/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2c_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/dude_poster_unanswerable/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-dude_poster_unanswerable_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part2/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part2_resumable/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet",
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", action="append", dest="train_files", default=None)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true", help="Profile rows in seeded random order.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used with --shuffle.")
    parser.add_argument("--ignore-input-ids-mismatch", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def build_data_config(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "custom_cls": {},
            "pad_mode": "no_padding",
            "truncation": "right",
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


def percentile_summary(values: list[int]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.int64)
    percentiles = [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9, 100]
    summary: dict[str, int | float] = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": int(array.min()),
        "max": int(array.max()),
    }
    for pct in percentiles:
        key = f"p{str(pct).replace('.', '_')}"
        summary[key] = int(np.percentile(array, pct, method="nearest"))
    return summary


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def get_token_id(processor: Any, tokenizer: Any, name: str) -> int | None:
    value = getattr(processor, name, None)
    if isinstance(value, int):
        return value
    token_name = {
        "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>",
    }.get(name)
    if token_name is None:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
    except Exception:
        return None
    return token_id if isinstance(token_id, int) and token_id >= 0 else None


def main() -> int:
    args = parse_args()
    train_files = args.train_files or DEFAULT_TRAIN_FILES
    missing = [path for path in train_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing train parquet files:\n" + "\n".join(missing))

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    print(f"loading tokenizer/processor from {args.model_path}", flush=True)
    tokenizer = hf_tokenizer(args.model_path, **load_kwargs)
    processor = hf_processor(args.model_path, **load_kwargs)
    if processor is None:
        raise RuntimeError(f"Could not load multimodal processor from {args.model_path}")

    print(f"building dataset from {len(train_files)} parquet files", flush=True)
    dataset = create_sft_dataset(train_files, build_data_config(args), tokenizer, processor, max_samples=-1)
    total = len(dataset)
    limit = min(total, args.limit) if args.limit is not None else total
    indices = list(range(total))
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
    indices = indices[:limit]

    image_token_id = get_token_id(processor, tokenizer, "image_token_id")
    video_token_id = get_token_id(processor, tokenizer, "video_token_id")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    start_time = time.monotonic()
    for processed, idx in enumerate(indices, start=1):
        try:
            sample = dataset[idx]
            input_ids = sample["input_ids"]
            loss_mask = sample["loss_mask"]
            row = {
                "dataset_idx": idx,
                "sequence_length": int(input_ids.shape[-1]),
                "loss_tokens": int(loss_mask.sum().item()),
                "debug_sample_info": sample.get("debug_sample_info"),
            }
            if image_token_id is not None:
                row["image_tokens"] = int((input_ids == image_token_id).sum().item())
            if video_token_id is not None:
                row["video_tokens"] = int((input_ids == video_token_id).sum().item())
            multi_modal_inputs = sample.get("multi_modal_inputs", {})
            if "image_grid_thw" in multi_modal_inputs:
                row["num_images"] = int(multi_modal_inputs["image_grid_thw"].shape[0])
            if "video_grid_thw" in multi_modal_inputs:
                row["num_videos"] = int(multi_modal_inputs["video_grid_thw"].shape[0])
            rows.append(row)
        except Exception as exc:
            errors.append({"dataset_idx": idx, "error": f"{type(exc).__name__}: {exc}"})

        if args.report_every > 0 and processed % args.report_every == 0:
            elapsed = time.monotonic() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (limit - processed) / rate if rate > 0 else 0.0
            print(
                f"processed {processed}/{limit} "
                f"({processed / limit:.1%}) elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)} rate={rate:.2f} rows/s",
                flush=True,
            )

    if not rows:
        raise RuntimeError(f"No rows processed successfully; errors={errors[:5]}")

    lengths = [int(row["sequence_length"]) for row in rows]
    loss_tokens = [int(row["loss_tokens"]) for row in rows]
    longest = sorted(rows, key=lambda row: row["sequence_length"], reverse=True)[: args.top_k]

    report = {
        "model_path": args.model_path,
        "num_train_files": len(train_files),
        "dataset_len": total,
        "processed": len(rows),
        "num_errors": len(errors),
        "shuffle": args.shuffle,
        "seed": args.seed if args.shuffle else None,
        "max_length_used_for_profile": args.max_length,
        "sequence_length": percentile_summary(lengths),
        "loss_tokens": percentile_summary(loss_tokens),
        "longest": longest,
        "errors": errors[: args.top_k],
    }

    print(json.dumps(report, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
