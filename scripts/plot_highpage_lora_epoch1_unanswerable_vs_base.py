#!/usr/bin/env python3
"""Plot highpage epoch-1 LoRA/unanswerable runs against base references."""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("notes/generated")
OUT_STEM = "highpage_lora_epoch1_unanswerable_vs_base_2026-05-18"

METRICS = {
    "longdoc_acc": "val-core/longdocurl0507_highpage/reward/mean@1",
    "mmlong_acc": "val-core/mmlongbench0507_highpage/reward/mean@1",
    "core_time_s": "val-aux/core_inference_time/mean",
    "tool_calls_longdoc": "val-aux/longdocurl0507_highpage/n_valid_tool_calls/mean@1",
    "tool_calls_mmlong": "val-aux/mmlongbench0507_highpage/n_valid_tool_calls/mean@1",
    "turns": "val-aux/num_turns/mean",
}

RUNS: list[tuple[str, str, list[Path]]] = [
    (
        "base_nonfast",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_current_lora_setting_20260515/"
                "base_before_sft_current_lora_setting_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "base_nonfast",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_current_lora_setting_20260515/"
                "base_before_sft_current_lora_setting_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "base_nonfast",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_current_lora_setting_20260515/"
                "base_before_sft_current_lora_setting_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "base_no_tool_no_system_fast",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale025"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_20260515_remaining/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale025"
            ),
        ],
    ),
    (
        "base_no_tool_no_system_fast",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale035"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_20260515_remaining/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale035"
            ),
        ],
    ),
    (
        "base_no_tool_no_system_fast",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_20260515_remaining/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale05"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_current_fast_hf_variance_20260516/"
                "base_no_tool_no_system_current_lora_setting_fast_hf_val_only_repeat1_highpage_0507_rescale05"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_current_fast_hf_variance_20260516/"
                "base_no_tool_no_system_current_lora_setting_fast_hf_val_only_repeat2_highpage_0507_rescale05"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale05_07_repeats_20260519/"
                "base_no_tool_no_system_fast_hf_val_only_repeat5_highpage_0507_rescale05"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale05_07_repeats_20260519/"
                "base_no_tool_no_system_fast_hf_val_only_repeat6_highpage_0507_rescale05"
            ),
        ],
    ),
    (
        "base_no_tool_no_system_fast",
        "0.7",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale07_20260518/"
                "base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale07"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale07_repeats_20260518/"
                "base_no_tool_no_system_fast_hf_val_only_repeat2_highpage_0507_rescale07"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale07_repeats_20260518/"
                "base_no_tool_no_system_fast_hf_val_only_repeat3_highpage_0507_rescale07"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale05_07_repeats_20260519/"
                "base_no_tool_no_system_fast_hf_val_only_repeat4_highpage_0507_rescale07"
            ),
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_base_no_tool_fast_hf_val_only_rescale05_07_repeats_20260519/"
                "base_no_tool_no_system_fast_hf_val_only_repeat5_highpage_0507_rescale07"
            ),
        ],
    ),
    (
        "basic_medium_only_e1_32k_sp1",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_medium_only_epoch1_32k_sp1_20260517/"
                "lora_basic_medium_only_epoch1_32k_sp1_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "basic_medium_only_e1_32k_sp1",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_medium_only_epoch1_32k_sp1_20260517/"
                "lora_basic_medium_only_epoch1_32k_sp1_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "basic_medium_only_e1_32k_sp1",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_medium_only_epoch1_32k_sp1_20260517/"
                "lora_basic_medium_only_epoch1_32k_sp1_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "both_higher_dpi_medium_only_e1_64k_sp2",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_20260517/"
                "lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "both_higher_dpi_medium_only_e1_64k_sp2",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_20260517/"
                "lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "both_higher_dpi_medium_only_e1_64k_sp2",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_20260517/"
                "lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "basic_plus_unans025_e1_32k_sp1",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_unanswerable025_20260517/"
                "lora_basic_unanswerable025_len32768_sp1_epoch1_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "basic_plus_unans025_e1_32k_sp1",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_unanswerable025_20260517/"
                "lora_basic_unanswerable025_len32768_sp1_epoch1_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "basic_plus_unans025_e1_32k_sp1",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_basic_unanswerable025_20260517/"
                "lora_basic_unanswerable025_len32768_sp1_epoch1_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_20260517/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_20260517/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_20260517/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unans02503505_matched_basic264_20260518/"
                "lora_both_higher_dpi_unans02503505_len65536_sp2_steps264_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unans02503505_matched_basic264_20260518/"
                "lora_both_higher_dpi_unans02503505_len65536_sp2_steps264_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unans02503505_matched_basic264_20260518/"
                "lora_both_higher_dpi_unans02503505_len65536_sp2_steps264_highpage_0507_rescale05"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2",
        "0.25",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_epoch2_retry1_20260518/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch2_retry1_highpage_0507_rescale025"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2",
        "0.35",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_epoch2_retry1_20260518/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch2_retry1_highpage_0507_rescale035"
            )
        ],
    ),
    (
        "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2",
        "0.5",
        [
            Path(
                "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
                "highpage_lora_both_higher_dpi_unanswerable02503505_epoch2_retry1_20260518/"
                "lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch2_retry1_highpage_0507_rescale05"
            )
        ],
    ),
]

DISPLAY = {
    "base_nonfast": "base tool-use non-fast",
    "base_no_tool_no_system_fast": "base no-tool/no-system fast",
    "basic_medium_only_e1_32k_sp1": "LoRA basic medium-only e1 32k/sp1",
    "both_higher_dpi_medium_only_e1_64k_sp2": "LoRA both higher-DPI medium-only e1 64k/sp2",
    "basic_plus_unans025_e1_32k_sp1": "LoRA basic + unans025 e1",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2": "LoRA both higher-DPI + unans025/035/05 e1",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264": (
        "LoRA both higher-DPI + unans025/035/05 e1 steps264"
    ),
    "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2": (
        "LoRA both higher-DPI + unans025/035/05 e2 retry1"
    ),
}

COLORS = {
    "base_nonfast": "#d62728",
    "base_no_tool_no_system_fast": "#9467bd",
    "basic_medium_only_e1_32k_sp1": "#1f77b4",
    "both_higher_dpi_medium_only_e1_64k_sp2": "#ff7f0e",
    "basic_plus_unans025_e1_32k_sp1": "#2ca02c",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2": "#111111",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264": "#8c564b",
    "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2": "#e377c2",
}

ORDER = [
    "base_nonfast",
    "base_no_tool_no_system_fast",
    "basic_medium_only_e1_32k_sp1",
    "both_higher_dpi_medium_only_e1_64k_sp2",
    "basic_plus_unans025_e1_32k_sp1",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2",
    "both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2",
    "both_higher_dpi_plus_unans02503505_e1_64k_sp2_steps264",
]


def parse_metrics(run_dir: Path) -> dict[str, float] | None:
    logs = sorted(run_dir.glob("*.launch.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    text = logs[-1].read_text(errors="ignore")
    lines = [
        line
        for line in text.splitlines()
        if "step:0 - val-core/longdocurl0507_highpage/reward/mean@1" in line
    ]
    if not lines:
        return None
    line = lines[-1]
    parsed: dict[str, float] = {}
    for part in line.split(" - "):
        if ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        try:
            parsed[key.strip()] = float(value.strip().strip(","))
        except ValueError:
            parsed[key.strip()] = math.nan

    row = {}
    for short_key, metric_key in METRICS.items():
        if metric_key not in parsed:
            return None
        row[short_key] = parsed[metric_key]
    row["avg_acc"] = (row["longdoc_acc"] + row["mmlong_acc"]) / 2.0
    row["avg_tool_calls"] = (row["tool_calls_longdoc"] + row["tool_calls_mmlong"]) / 2.0
    return row


def mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for model, scale, run_dirs in RUNS:
        parsed_rows = []
        for run_dir in run_dirs:
            parsed = parse_metrics(run_dir)
            if parsed is None:
                missing.append(f"{model} {scale}: {run_dir}")
            else:
                parsed_rows.append((parsed, run_dir))
        if not parsed_rows:
            continue

        row: dict[str, object] = {
            "model": model,
            "scale": scale,
            "n_runs": len(parsed_rows),
            "run_dir": ";".join(str(run_dir) for _, run_dir in parsed_rows),
        }
        for key in [
            "avg_acc",
            "longdoc_acc",
            "mmlong_acc",
            "core_time_s",
            "avg_tool_calls",
            "turns",
        ]:
            mean, std = mean_std([parsed[key] for parsed, _ in parsed_rows])
            row[key] = mean
            row[f"{key}_std"] = std
        rows.append(row)

    if missing:
        print("Missing metrics:")
        for item in missing:
            print(f"  {item}")
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "model",
        "scale",
        "n_runs",
        "avg_acc",
        "avg_acc_std",
        "longdoc_acc",
        "mmlong_acc",
        "core_time_s",
        "core_time_s_std",
        "avg_tool_calls",
        "turns",
        "run_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot(rows: list[dict[str, object]], png_path: Path, svg_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8.5))

    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model in ORDER:
        model_rows = sorted(by_model.get(model, []), key=lambda row: float(row["scale"]))
        if not model_rows:
            continue
        xs = [float(row["core_time_s"]) for row in model_rows]
        ys = [float(row["avg_acc"]) for row in model_rows]
        yerr = [
            0.0 if row.get("avg_acc_std") in (None, "") else float(row["avg_acc_std"])
            for row in model_rows
        ]
        has_yerr = any(value > 0 for value in yerr)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr if has_yerr else None,
            marker="o",
            linewidth=2.7,
            markersize=7,
            capsize=4 if has_yerr else 0,
            label=DISPLAY[model],
            color=COLORS[model],
        )
        for row, x, y in zip(model_rows, xs, ys, strict=True):
            ax.annotate(
                str(row["scale"]),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color=COLORS[model],
            )

    ax.set_title("Highpage eval: LoRA medium/unanswerable models vs base models", fontsize=18)
    ax.set_xlabel("Average core inference time (s)", fontsize=13)
    ax.set_ylabel("Average accuracy over longdocurl0507 + mmlongbench0507", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.55)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = OUT_DIR / f"{OUT_STEM}.csv"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    svg_path = OUT_DIR / f"{OUT_STEM}.svg"
    write_csv(rows, csv_path)
    plot(rows, png_path, svg_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
