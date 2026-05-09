#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_LABELS = {
    "base": "base",
    "base_no_tool_no_system": "base_no_tool_no_system",
    "freeze_vt_bs32_tool_arg_order_medium_only_epoch": "freeze_vt_bs32_tool_arg_order_medium_only",
    "rl_ckpt425_actor_merged_hf": "rl_ckpt425_actor_merged_hf",
}

MODEL_COLORS = {
    "base": "#d62728",
    "base_no_tool_no_system": "#9467bd",
    "freeze_vt_bs32_tool_arg_order_medium_only_epoch": "#2ca02c",
    "rl_ckpt425_actor_merged_hf": "#1f77b4",
}


def grab(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def load_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    status_path = output_root / "status.tsv"
    with status_path.open() as f:
        next(f)
        for line in f:
            lane, gpus, model_id, scale_id, initial_rescale, run_name, status, exit_code, work_dir = line.rstrip("\n").split("\t")
            if status != "success":
                continue
            log_path = Path(work_dir) / "ckpts" / "insight_doc" / run_name / "val_heldout.log"
            text = log_path.read_text(errors="ignore")
            longdocurl = grab(text, r"val-core/longdocurl0507_highpage/reward/mean@1:([0-9.]+)")
            mmlongbench = grab(text, r"val-core/mmlongbench0507_highpage/reward/mean@1:([0-9.]+)")
            longdocurl_core = grab(text, r"val-aux/longdocurl0507_highpage/core_inference_time/mean:([0-9.]+)")
            mmlongbench_core = grab(text, r"val-aux/mmlongbench0507_highpage/core_inference_time/mean:([0-9.]+)")
            if None in (longdocurl, mmlongbench, longdocurl_core, mmlongbench_core):
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "initial_rescale": initial_rescale,
                    "avg_accuracy": (longdocurl + mmlongbench) / 2,
                    "avg_core_time_s": (longdocurl_core + mmlongbench_core) / 2,
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "initial_rescale", "avg_core_time_s", "avg_accuracy"])
        for row in rows:
            writer.writerow([row["model_id"], row["initial_rescale"], row["avg_core_time_s"], row["avg_accuracy"]])


def plot(rows: list[dict], png_path: Path, svg_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=180)

    for model_id in MODEL_LABELS:
        pts = [row for row in rows if row["model_id"] == model_id]
        pts.sort(key=lambda row: float(row["initial_rescale"]))
        if not pts:
            continue
        xs = [row["avg_core_time_s"] for row in pts]
        ys = [row["avg_accuracy"] for row in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
        for row in pts:
            ax.annotate(
                row["initial_rescale"],
                (row["avg_core_time_s"], row["avg_accuracy"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=MODEL_COLORS[model_id],
            )

    ax.set_xlabel("Average core inference time on high-page QAs (s)")
    ax.set_ylabel("Average accuracy on high-page QAs")
    ax.set_title("High-page sweep: avg core time vs avg accuracy")
    ax.set_xlim(left=0)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507",
    )
    parser.add_argument(
        "--out-dir",
        default="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated",
    )
    parser.add_argument(
        "--stem",
        default="highpage_avg_core_time_vs_avg_accuracy_2026-05-07",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(output_root)
    csv_path = out_dir / f"{args.stem}.csv"
    png_path = out_dir / f"{args.stem}.png"
    svg_path = out_dir / f"{args.stem}.svg"
    write_csv(rows, csv_path)
    plot(rows, png_path, svg_path)
    print(csv_path)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
