#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt


SERIES = {
    "base_no_tool": {
        "avg": [
            ("0.175", 10.15, 0.2675),
            ("0.25", 13.15, 0.3125),
            ("0.35", 24.42, 0.4350),
            ("0.5", 53.03, 0.4850),
            ("0.7", 112.44, 0.5225),
        ],
        "longdocurl": [
            ("0.175", 10.99, 0.295),
            ("0.25", 14.16, 0.375),
            ("0.35", 26.57, 0.510),
            ("0.5", 63.38, 0.555),
            ("0.7", 121.68, 0.590),
        ],
    },
    "base_no_tool_no_system": {
        "avg": [
            ("0.175", 10.91, 0.2600),
            ("0.25", 15.33, 0.3550),
            ("0.35", 24.59, 0.4425),
            ("0.5", 53.04, 0.5075),
            ("0.7", 115.19, 0.5050),
        ],
        "longdocurl": [
            ("0.175", 13.60, 0.295),
            ("0.25", 15.08, 0.420),
            ("0.35", 27.85, 0.505),
            ("0.5", 59.20, 0.575),
            ("0.7", 123.60, 0.565),
        ],
    },
    "base_no_tool_no_system_answer_only": {
        "avg": [
            ("0.175", 8.25, 0.2225),
            ("0.25", 9.18, 0.3125),
            ("0.35", 19.43, 0.3850),
            ("0.5", 45.03, 0.4450),
            ("0.7", 102.35, 0.4850),
        ],
        "longdocurl": [
            ("0.175", 11.41, 0.260),
            ("0.25", 11.02, 0.380),
            ("0.35", 22.46, 0.475),
            ("0.5", 55.06, 0.500),
            ("0.7", 113.72, 0.580),
        ],
    },
    "base": {
        "avg": [
            ("0.175", 32.92, 0.2325),
            ("0.25", 39.90, 0.3100),
            ("0.35", 56.83, 0.3725),
            ("0.5", 92.96, 0.4725),
            ("0.7", 171.35, 0.5025),
        ],
        "longdocurl": [
            ("0.175", 34.91, 0.185),
            ("0.25", 44.02, 0.310),
            ("0.35", 60.19, 0.390),
            ("0.5", 103.15, 0.470),
            ("0.7", 178.93, 0.545),
        ],
    },
    "freeze_vt_bs32_tool_arg_order_medium_only_epoch": {
        "avg": [
            ("0.175", 18.00, 0.3275),
            ("0.25", 20.93, 0.4375),
            ("0.35", 37.10, 0.4750),
            ("0.5", 87.17, 0.5075),
            ("0.7", 211.99, 0.4625),
        ],
        "longdocurl": [
            ("0.175", 16.31, 0.400),
            ("0.25", 21.05, 0.500),
            ("0.35", 36.75, 0.550),
            ("0.5", 89.66, 0.560),
            ("0.7", 203.31, 0.510),
        ],
    },
}

COLORS = {
    "base_no_tool": "#1f77b4",
    "base_no_tool_no_system": "#9467bd",
    "base_no_tool_no_system_answer_only": "#ff7f0e",
    "base": "#d62728",
    "freeze_vt_bs32_tool_arg_order_medium_only_epoch": "#2ca02c",
}

LABELS = {
    "base_no_tool": "base_no_tool",
    "base_no_tool_no_system": "base_no_tool_no_system",
    "base_no_tool_no_system_answer_only": "base_no_tool_no_system_answer_only",
    "base": "base",
    "freeze_vt_bs32_tool_arg_order_medium_only_epoch": "freeze_vt_bs32_tool_arg_order_medium_only",
}


def build_outputs(out_dir: Path, view: str, stem_suffix: str):
    if view == "avg":
        stem = "uncapped_avg_core_time_vs_avg_accuracy_2026-05-07"
    else:
        stem = "uncapped_longdocurl_core_time_vs_accuracy_2026-05-07"
    if stem_suffix:
        stem = f"{stem}_{stem_suffix}"
    return (
        out_dir / f"{stem}.csv",
        out_dir / f"{stem}.png",
        out_dir / f"{stem}.svg",
    )


def filtered_series(selected_models: list[str] | None, excluded_rescales: set[str]):
    models = selected_models if selected_models else list(SERIES.keys())
    out = {}
    for model in models:
        by_view = SERIES[model]
        out[model] = {}
        for view_name, pts in by_view.items():
            out[model][view_name] = [pt for pt in pts if pt[0] not in excluded_rescales]
    return out


def write_csv(path: Path, view: str, data: dict):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        if view == "avg":
            writer.writerow(["model", "initial_rescale", "avg_core_time_s", "avg_accuracy"])
        else:
            writer.writerow(["model", "initial_rescale", "longdocurl_core_time_s", "longdocurl_accuracy"])
        for model, by_view in data.items():
            for rescale, x, y in by_view[view]:
                writer.writerow([model, rescale, x, y])


def plot(path_png: Path, path_svg: Path, view: str, data: dict):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=180)

    for model, by_view in data.items():
        pts = by_view[view]
        xs = [x for _, x, _ in pts]
        ys = [y for _, _, y in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=COLORS[model],
            label=LABELS[model],
        )
        for rescale, x, y in pts:
            ax.annotate(rescale, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=COLORS[model])

    if view == "avg":
        ax.set_xlabel("Average core inference time (s)")
        ax.set_ylabel("Average accuracy on longdocurl200 + mmlongbench200")
        ax.set_title("Uncapped sweep: avg core time vs avg accuracy")
        ax.set_ylim(0.2, 0.55)
    else:
        ax.set_xlabel("longdocurl200 core inference time (s)")
        ax.set_ylabel("longdocurl200 accuracy")
        ax.set_title("Uncapped sweep: longdocurl200 core time vs accuracy")
        ax.set_ylim(0.15, 0.62)

    ax.legend(frameon=True, fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["avg", "longdocurl"], default="avg")
    parser.add_argument("--models", nargs="+", choices=list(SERIES.keys()))
    parser.add_argument("--exclude-rescales", nargs="*", default=[])
    parser.add_argument("--stem-suffix", default="")
    parser.add_argument(
        "--out-dir",
        default="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = filtered_series(args.models, set(args.exclude_rescales))
    csv_path, png_path, svg_path = build_outputs(out_dir, args.view, args.stem_suffix)
    write_csv(csv_path, args.view, data)
    plot(png_path, svg_path, args.view, data)
    print(csv_path)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
