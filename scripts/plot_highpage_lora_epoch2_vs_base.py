#!/usr/bin/env python3
"""Plot highpage epoch-2 LoRA runs against matched base runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("notes/generated")


RUNS = [
    (
        "base_nonfast",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale025"),
    ),
    (
        "base_nonfast",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale035"),
    ),
    (
        "base_nonfast",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale05"),
    ),
    (
        "base_no_tool_no_system_fast",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_remaining/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale025"),
    ),
    (
        "base_no_tool_no_system_fast",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_remaining/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale035"),
    ),
    (
        "base_no_tool_no_system_fast",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale05"),
    ),
    (
        "lora_basic_40k_e2",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_basic_40k_epoch2_step528_highpage_0507_rescale025"),
    ),
    (
        "lora_basic_40k_e2",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_basic_40k_epoch2_retry_step528_highpage_0507_rescale035"),
    ),
    (
        "lora_basic_40k_e2",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_basic_40k_epoch2_retry_step528_highpage_0507_rescale05"),
    ),
    (
        "lora_arxiv_higher_dpi_e2",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_arxiv_higher_dpi_epoch2_retry_step708_highpage_0507_rescale025"),
    ),
    (
        "lora_arxiv_higher_dpi_e2",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_arxiv_higher_dpi_epoch2_retry_step708_highpage_0507_rescale035"),
    ),
    (
        "lora_arxiv_higher_dpi_e2",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_arxiv_higher_dpi_epoch2_retry_step708_highpage_0507_rescale05"),
    ),
    (
        "lora_O3_higher_dpi_e2",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_O3_higher_dpi_epoch2_retry_step740_highpage_0507_rescale025"),
    ),
    (
        "lora_O3_higher_dpi_e2",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_O3_higher_dpi_epoch2_retry_step740_highpage_0507_rescale035"),
    ),
    (
        "lora_O3_higher_dpi_e2",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_retry_noproxy_20260516/lora_O3_higher_dpi_epoch2_retry_step740_highpage_0507_rescale05"),
    ),
    (
        "lora_both_bs16_e2",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs16_epoch2_step1838_highpage_0507_rescale025"),
    ),
    (
        "lora_both_bs16_e2",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs16_epoch2_step1838_highpage_0507_rescale035"),
    ),
    (
        "lora_both_bs16_e2",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs16_epoch2_step1838_highpage_0507_rescale05"),
    ),
    (
        "lora_both_bs32_e2",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs32_epoch2_step918_highpage_0507_rescale025"),
    ),
    (
        "lora_both_bs32_e2",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs32_epoch2_step918_highpage_0507_rescale035"),
    ),
    (
        "lora_both_bs32_e2",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515/lora_both_bs32_epoch2_step918_highpage_0507_rescale05"),
    ),
]


METRICS = {
    "longdoc_acc": "val-core/longdocurl0507_highpage/reward/mean@1",
    "mmlong_acc": "val-core/mmlongbench0507_highpage/reward/mean@1",
    "core_time_s": "val-aux/core_inference_time/mean",
    "assistant_resp_tokens": "val-aux/response_tokens_generated/mean",
    "tool_resp_tokens": "val-aux/response_tokens_tool/mean",
    "turns": "val-aux/num_turns/mean",
}


DISPLAY = {
    "base_nonfast": "base tool-use non-fast",
    "base_no_tool_no_system_fast": "base no-tool/no-system fast",
    "lora_basic_40k_e2": "LoRA basic 40k e2",
    "lora_arxiv_higher_dpi_e2": "LoRA arxiv e2",
    "lora_O3_higher_dpi_e2": "LoRA O3 e2",
    "lora_both_bs16_e2": "LoRA both bs16 e2",
    "lora_both_bs32_e2": "LoRA both bs32 e2",
}


COLORS = {
    "base_nonfast": "#d62728",
    "base_no_tool_no_system_fast": "#9467bd",
    "lora_basic_40k_e2": "#2ca02c",
    "lora_arxiv_higher_dpi_e2": "#1f77b4",
    "lora_O3_higher_dpi_e2": "#ff7f0e",
    "lora_both_bs16_e2": "#111111",
    "lora_both_bs32_e2": "#8c564b",
}


ORDER = [
    "base_nonfast",
    "base_no_tool_no_system_fast",
    "lora_basic_40k_e2",
    "lora_arxiv_higher_dpi_e2",
    "lora_O3_higher_dpi_e2",
    "lora_both_bs16_e2",
    "lora_both_bs32_e2",
]


def parse_metric(text: str, key: str) -> float | None:
    pattern = re.escape(key) + r"['\"]?:\s*([+-]?(?:nan|\d+(?:\.\d+)?(?:e[+-]?\d+)?))"
    values = re.findall(pattern, text)
    if not values:
        pattern = re.escape("'" + key + "'") + r":\s*([+-]?(?:nan|\d+(?:\.\d+)?(?:e[+-]?\d+)?))"
        values = re.findall(pattern, text)
    if not values:
        return None
    value = values[-1]
    return float("nan") if value == "nan" else float(value)


def read_row(model: str, scale: str, run_dir: Path) -> dict[str, object] | None:
    logs = sorted(run_dir.glob("*.launch.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    text = logs[-1].read_text(errors="ignore")
    values = {name: parse_metric(text, key) for name, key in METRICS.items()}
    required = ("longdoc_acc", "mmlong_acc", "core_time_s")
    if any(values[name] is None for name in required):
        return None
    values["avg_acc"] = (float(values["longdoc_acc"]) + float(values["mmlong_acc"])) / 2.0
    return {
        "model": model,
        "scale": scale,
        "run_dir": str(run_dir),
        **values,
    }


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "model",
        "scale",
        "avg_acc",
        "longdoc_acc",
        "mmlong_acc",
        "core_time_s",
        "assistant_resp_tokens",
        "tool_resp_tokens",
        "turns",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot(rows: list[dict[str, object]], out_prefix: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=180)

    for model in ORDER:
        points = [row for row in rows if row["model"] == model]
        if not points:
            continue
        points.sort(key=lambda row: float(row["core_time_s"]))
        x = [float(row["core_time_s"]) for row in points]
        y = [float(row["avg_acc"]) for row in points]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6.5,
            color=COLORS[model],
            label=DISPLAY[model],
        )
        for row in points:
            ax.annotate(
                str(row["scale"]),
                (float(row["core_time_s"]), float(row["avg_acc"])),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                color=COLORS[model],
            )

    ax.set_title("Highpage eval: epoch-2 LoRA models vs base models", fontsize=16)
    ax.set_xlabel("Average core inference time (s)", fontsize=13)
    ax.set_ylabel("Average accuracy over longdocurl0507 + mmlongbench0507", fontsize=13)
    ax.set_xlim(left=0)
    max_acc = max(float(row["avg_acc"]) for row in rows)
    ax.set_ylim(0.26, max(0.50, max_acc + 0.025))
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="highpage_lora_epoch2_vs_base_2026-05-16")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for model, scale, run_dir in RUNS:
        row = read_row(model, scale, run_dir)
        if row is None:
            missing.append((model, scale, str(run_dir)))
        else:
            rows.append(row)

    out_prefix = OUT_DIR / args.output_prefix
    write_csv(rows, out_prefix.with_suffix(".csv"))
    plot(rows, out_prefix)

    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.png')}")
    print(f"wrote {out_prefix.with_suffix('.svg')}")
    if missing:
        print("missing/incomplete:")
        for item in missing:
            print("\t".join(item))


if __name__ == "__main__":
    main()
