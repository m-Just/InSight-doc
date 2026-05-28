#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/artifacts/"
    "synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517"
)


FIRST_BATCH_LAYOUT = {
    "rescale025": {
        "easy": {
            "parquet": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq/"
                "converted_sft/sft_data.parquet"
            ),
            "wrong_ids": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq/"
                "converted_sft/wrong_question_ids.txt"
            ),
        },
        "medium": {
            "parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq_resumable/"
                "sft_data.parquet"
            ),
            "base_model_tool_order_parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq_resumable/"
                "sft_data_base_model_tool_argument_order.parquet"
            ),
            "wrong_ids": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq_resumable/"
                "wrong_question_ids.txt"
            ),
        },
    },
    "rescale035": {
        "easy": {
            "parquet": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq/"
                "converted_sft/sft_data.parquet"
            ),
            "wrong_ids": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq/"
                "converted_sft/wrong_question_ids.txt"
            ),
        },
        "medium": {
            "parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq_resumable/"
                "sft_data.parquet"
            ),
            "base_model_tool_order_parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq_resumable/"
                "sft_data_base_model_tool_argument_order.parquet"
            ),
            "wrong_ids": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq_resumable/"
                "wrong_question_ids.txt"
            ),
        },
    },
    "rescale05": {
        "easy": {
            "parquet": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq/"
                "converted_sft/sft_data.parquet"
            ),
            "wrong_ids": Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
                "qwen3-vl-32b-instruct/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq/"
                "converted_sft/wrong_question_ids.txt"
            ),
        },
        "medium": {
            "parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq_resumable/"
                "sft_data.parquet"
            ),
            "base_model_tool_order_parquet": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq_resumable/"
                "sft_data_base_model_tool_argument_order.parquet"
            ),
            "wrong_ids": Path(
                "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/"
                "multi_agent_vsearch/synthetic_unanswerable_qwen3_region_loc/"
                "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq_resumable/"
                "wrong_question_ids.txt"
            ),
        },
    },
}


def ensure_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def parquet_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_root": str(output_root),
        "splits": {},
        "train_files": [],
    }

    for split_name, split_cfg in FIRST_BATCH_LAYOUT.items():
        split_summary = {}
        for difficulty, cfg in split_cfg.items():
            src_parquet = cfg["parquet"].resolve()
            src_wrong_ids = cfg["wrong_ids"].resolve()
            if not src_parquet.exists():
                raise FileNotFoundError(f"missing parquet: {src_parquet}")
            if not src_wrong_ids.exists():
                raise FileNotFoundError(f"missing wrong ids: {src_wrong_ids}")

            if difficulty == "easy":
                rel_dir = Path(split_name) / "easy" / "processed_drop_degenerate"
            else:
                rel_dir = Path(split_name) / "medium" / "processed_gpt5_nano_rewrite"

            out_dir = output_root / rel_dir
            out_parquet = out_dir / "sft_data.parquet"
            out_wrong_ids = out_dir / "wrong_question_ids.txt"

            ensure_symlink(out_parquet, src_parquet)
            ensure_symlink(out_wrong_ids, src_wrong_ids)

            base_model_tool_order_src = cfg.get("base_model_tool_order_parquet")
            base_model_tool_order_dst = out_dir / "sft_data_base_model_tool_argument_order.parquet"
            if base_model_tool_order_src is not None:
                base_model_tool_order_src = Path(base_model_tool_order_src).resolve()
                if base_model_tool_order_src.exists():
                    ensure_symlink(base_model_tool_order_dst, base_model_tool_order_src)

            rows = parquet_rows(src_parquet)
            wrong_count = count_lines(src_wrong_ids)
            split_summary[difficulty] = {
                "rows": rows,
                "wrong_question_ids": wrong_count,
                "parquet": str(out_parquet),
                "wrong_ids": str(out_wrong_ids),
                "source_parquet": str(src_parquet),
                "source_wrong_ids": str(src_wrong_ids),
            }
            if base_model_tool_order_src is not None and base_model_tool_order_src.exists():
                split_summary[difficulty]["base_model_tool_argument_order_parquet"] = str(
                    base_model_tool_order_dst
                )
                split_summary[difficulty]["source_base_model_tool_argument_order_parquet"] = str(
                    base_model_tool_order_src
                )
            summary["train_files"].append(str(out_parquet))

        summary["splits"][split_name] = split_summary

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "train_files.txt").write_text(
        "\n".join(summary["train_files"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
