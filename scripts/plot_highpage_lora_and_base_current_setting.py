#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROWS = [
    # model, initial_rescale, avg_core_time_s, avg_accuracy
    ("base_current_lora_setting", "0.25", 49.33, 0.2879),
    ("base_current_lora_setting", "0.35", 76.47, 0.3830),
    ("base_current_lora_setting", "0.5", 116.50, 0.4385),
    ("base_no_tool_no_system_current_lora_setting", "0.25", 16.06, 0.3166),
    ("base_no_tool_no_system_current_lora_setting", "0.35", 33.04, 0.4255),
    ("base_no_tool_no_system_current_lora_setting", "0.5", 71.83, 0.4804),
    ("base_no_tool_no_system", "0.25", 17.47, 0.3359),
    ("base_no_tool_no_system", "0.35", 32.57, 0.4531),
    ("base_no_tool_no_system", "0.5", 71.47, 0.4670),
    ("lora_basic_40k_freeze_vt_medium_only", "0.25", 14.18, 0.3868),
    ("lora_basic_40k_freeze_vt_medium_only", "0.35", 26.73, 0.4341),
    ("lora_basic_40k_freeze_vt_medium_only", "0.5", 77.28, 0.4255),
    ("lora_arxiv_w_higher_dpi", "0.25", 14.09, 0.3785),
    ("lora_arxiv_w_higher_dpi", "0.35", 26.17, 0.4357),
    ("lora_arxiv_w_higher_dpi", "0.5", 71.80, 0.4559),
    ("lora_O3_w_higher_dpi", "0.25", 13.82, 0.3837),
    ("lora_O3_w_higher_dpi", "0.35", 26.92, 0.4417),
    ("lora_O3_w_higher_dpi", "0.5", 81.33, 0.4661),
]

MODEL_LABELS = {
    "base_current_lora_setting": "base",
    "base_no_tool_no_system_current_lora_setting": "base_no_tool_no_system_current",
    "base_no_tool_no_system": "base_no_tool_no_system_ref",
    "lora_basic_40k_freeze_vt_medium_only": "lora_basic_40k",
    "lora_arxiv_w_higher_dpi": "lora_arxiv_higher_dpi",
    "lora_O3_w_higher_dpi": "lora_O3_higher_dpi",
}

MODEL_COLORS = {
    "base_current_lora_setting": "#d62728",
    "base_no_tool_no_system_current_lora_setting": "#9467bd",
    "base_no_tool_no_system": "#7f7f7f",
    "lora_basic_40k_freeze_vt_medium_only": "#2ca02c",
    "lora_arxiv_w_higher_dpi": "#1f77b4",
    "lora_O3_w_higher_dpi": "#ff7f0e",
}


def write_csv(rows: list[tuple[str, str, float, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "initial_rescale", "avg_core_time_s", "avg_accuracy"])
        writer.writerows(rows)


def plot(rows: list[tuple[str, str, float, float]], png_path: Path, svg_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=180)

    model_order = [
        "base_current_lora_setting",
        "base_no_tool_no_system_current_lora_setting",
        "base_no_tool_no_system",
        "lora_basic_40k_freeze_vt_medium_only",
        "lora_arxiv_w_higher_dpi",
        "lora_O3_w_higher_dpi",
    ]
    by_model: dict[str, list[tuple[str, float, float]]] = {model: [] for model in model_order}
    for model, rescale, core_time, accuracy in rows:
        by_model[model].append((rescale, core_time, accuracy))

    for model in model_order:
        points = sorted(by_model[model], key=lambda item: float(item[0]))
        xs = [core_time for _, core_time, _ in points]
        ys = [accuracy for _, _, accuracy in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.3,
            markersize=5.8,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        for rescale, core_time, accuracy in points:
            ax.annotate(
                rescale,
                (core_time, accuracy),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=MODEL_COLORS[model],
            )

    ax.set_xlabel("Average core inference time on high-page QAs (s)")
    ax.set_ylabel("Average accuracy on high-page QAs")
    ax.set_title("High-page eval: LoRA runs vs matched base reruns")
    ax.set_xlim(left=0)
    ax.set_ylim(0.26, 0.50)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated",
    )
    parser.add_argument(
        "--stem",
        default="highpage_lora_and_matched_base_current_setting_2026-05-15",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.stem}.csv"
    png_path = out_dir / f"{args.stem}.png"
    svg_path = out_dir / f"{args.stem}.svg"

    write_csv(ROWS, csv_path)
    plot(ROWS, png_path, svg_path)

    print(csv_path)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
