#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs")
EXPECTED_EXPORTS = 368

RUNS = [
    # model, initial_rescale, run_dir, avg_accuracy
    (
        "base_current_lora_setting",
        "0.25",
        OUTPUT_ROOT
        / "highpage_base_current_lora_setting_20260515"
        / "base_before_sft_current_lora_setting_highpage_0507_rescale025",
        0.2879,
    ),
    (
        "base_current_lora_setting",
        "0.35",
        OUTPUT_ROOT
        / "highpage_base_current_lora_setting_20260515"
        / "base_before_sft_current_lora_setting_highpage_0507_rescale035",
        0.3830,
    ),
    (
        "base_current_lora_setting",
        "0.5",
        OUTPUT_ROOT
        / "highpage_base_current_lora_setting_20260515"
        / "base_before_sft_current_lora_setting_highpage_0507_rescale05",
        0.4385,
    ),
    (
        "base_no_tool_no_system_current_lora_setting",
        "0.25",
        OUTPUT_ROOT
        / "highpage_base_no_tool_no_system_current_lora_setting_20260515"
        / "base_no_tool_no_system_current_lora_setting_highpage_0507_rescale025",
        0.3166,
    ),
    (
        "base_no_tool_no_system_current_lora_setting",
        "0.35",
        OUTPUT_ROOT
        / "highpage_base_no_tool_no_system_current_lora_setting_20260515"
        / "base_no_tool_no_system_current_lora_setting_highpage_0507_rescale035",
        0.4255,
    ),
    (
        "base_no_tool_no_system_current_lora_setting",
        "0.5",
        OUTPUT_ROOT
        / "highpage_base_no_tool_no_system_current_lora_setting_20260515"
        / "base_no_tool_no_system_current_lora_setting_highpage_0507_rescale05",
        0.4804,
    ),
    (
        "base_no_tool_no_system",
        "0.25",
        OUTPUT_ROOT / "highpage_initial_rescale_sweep_0507" / "base_no_tool_no_system_highpage_0507_rescale025",
        0.3359,
    ),
    (
        "base_no_tool_no_system",
        "0.35",
        OUTPUT_ROOT / "highpage_initial_rescale_sweep_0507" / "base_no_tool_no_system_highpage_0507_rescale035",
        0.4531,
    ),
    (
        "base_no_tool_no_system",
        "0.5",
        OUTPUT_ROOT / "highpage_initial_rescale_sweep_0507" / "base_no_tool_no_system_highpage_0507_rescale05",
        0.4670,
    ),
    (
        "lora_basic_40k_freeze_vt_medium_only",
        "0.25",
        OUTPUT_ROOT
        / "highpage_lora_basic_40k_freeze_vt_medium_only_20260514"
        / "lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale025",
        0.3868,
    ),
    (
        "lora_basic_40k_freeze_vt_medium_only",
        "0.35",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_basic_40k_freeze_vt_medium_only_retry_highpage_0507_rescale035",
        0.4341,
    ),
    (
        "lora_basic_40k_freeze_vt_medium_only",
        "0.5",
        OUTPUT_ROOT
        / "highpage_lora_basic_40k_freeze_vt_medium_only_20260514"
        / "lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale05",
        0.4255,
    ),
    (
        "lora_arxiv_w_higher_dpi",
        "0.25",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_arxiv_w_higher_dpi_highpage_0507_rescale025",
        0.3785,
    ),
    (
        "lora_arxiv_w_higher_dpi",
        "0.35",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_arxiv_w_higher_dpi_highpage_0507_rescale035",
        0.4357,
    ),
    (
        "lora_arxiv_w_higher_dpi",
        "0.5",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_arxiv_w_higher_dpi_highpage_0507_rescale05",
        0.4559,
    ),
    (
        "lora_O3_w_higher_dpi",
        "0.25",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_O3_w_higher_dpi_highpage_0507_rescale025",
        0.3837,
    ),
    (
        "lora_O3_w_higher_dpi",
        "0.35",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_O3_w_higher_dpi_highpage_0507_rescale035",
        0.4417,
    ),
    (
        "lora_O3_w_higher_dpi",
        "0.5",
        OUTPUT_ROOT
        / "highpage_lora_higher_dpi_chain_20260514"
        / "lora_O3_w_higher_dpi_highpage_0507_rescale05",
        0.4661,
    ),
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


def benchmark_name(data_source: str) -> str:
    if data_source.startswith("longdocurl"):
        return "longdocurl"
    if data_source.startswith("mmlongbench"):
        return "mmlongbench"
    return data_source


def export_files(export_dir: Path) -> list[Path]:
    files = sorted(export_dir.glob("*.json"))
    if len(files) > EXPECTED_EXPORTS:
        # Retry directories can retain stale exports from an earlier failed pass.
        files = sorted(files, key=lambda path: path.stat().st_mtime)[-EXPECTED_EXPORTS:]
    return files


def median_core_times(run_dir: Path) -> tuple[float, dict[str, float], int, int, int]:
    export_dir = run_dir / "exported_conversations"
    if not export_dir.exists():
        raise FileNotFoundError(export_dir)

    by_benchmark: dict[str, list[float]] = {}
    raw_total = len(list(export_dir.glob("*.json")))
    files = export_files(export_dir)
    total = len(files)
    skipped = 0
    for path in files:
        with path.open() as f:
            obj = json.load(f)
        timing = obj.get("parameters", {}).get("loop", {}).get("timing", {})
        core_time = timing.get("core_inference_time")
        if core_time is None:
            skipped += 1
            continue
        reward = obj.get("reward") or {}
        extra_info = obj.get("extra_info") or {}
        data_source = reward.get("data_source") or extra_info.get("data_source", "")
        bench = benchmark_name(str(data_source))
        by_benchmark.setdefault(bench, []).append(float(core_time))

    medians = {bench: statistics.median(values) for bench, values in by_benchmark.items() if values}
    if {"longdocurl", "mmlongbench"} <= set(medians):
        avg_median = (medians["longdocurl"] + medians["mmlongbench"]) / 2.0
    elif medians:
        avg_median = statistics.mean(medians.values())
    else:
        raise ValueError(f"No valid core_inference_time values in {export_dir}")

    return avg_median, medians, total, raw_total, skipped


def build_rows() -> list[dict[str, object]]:
    rows = []
    for model, rescale, run_dir, accuracy in RUNS:
        avg_median, medians, total, raw_total, skipped = median_core_times(run_dir)
        rows.append(
            {
                "model": model,
                "initial_rescale": rescale,
                "median_core_time_s": avg_median,
                "longdocurl_median_core_time_s": medians.get("longdocurl"),
                "mmlongbench_median_core_time_s": medians.get("mmlongbench"),
                "avg_accuracy": accuracy,
                "exported_conversations": total,
                "raw_exported_conversations": raw_total,
                "skipped_core_time_none": skipped,
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model",
        "initial_rescale",
        "median_core_time_s",
        "longdocurl_median_core_time_s",
        "mmlongbench_median_core_time_s",
        "avg_accuracy",
        "exported_conversations",
        "raw_exported_conversations",
        "skipped_core_time_none",
        "run_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], png_path: Path, svg_path: Path) -> None:
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
    by_model: dict[str, list[dict[str, object]]] = {model: [] for model in model_order}
    for row in rows:
        by_model[str(row["model"])].append(row)

    for model in model_order:
        points = sorted(by_model[model], key=lambda row: float(row["initial_rescale"]))
        xs = [float(row["median_core_time_s"]) for row in points]
        ys = [float(row["avg_accuracy"]) for row in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.3,
            markersize=5.8,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        for row in points:
            ax.annotate(
                str(row["initial_rescale"]),
                (float(row["median_core_time_s"]), float(row["avg_accuracy"])),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=MODEL_COLORS[model],
            )

    ax.set_xlabel("Average of benchmark median core inference times on high-page QAs (s)")
    ax.set_ylabel("Average accuracy on high-page QAs")
    ax.set_title("High-page eval: LoRA runs vs matched base reruns (median core time)")
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
        default="highpage_lora_and_matched_base_current_setting_median_core_2026-05-15",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
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
