#!/usr/bin/env python3
"""Plot highpage HF-val-only LoRA/base speed-accuracy comparison."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("notes/generated")


RUNS = [
    # Recent fast-HF base no-tool/no-system runs. Prefer the later "remaining"
    # duplicate where available, otherwise use the completed fgtest row.
    (
        "base_no_tool_current_fast_hf",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/base_no_tool_no_system_current_lora_setting_fast_hf_val_only_highpage_0507_rescale025"),
    ),
    (
        "base_no_tool_current_fast_hf",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/base_no_tool_no_system_current_lora_setting_fast_hf_val_only_highpage_0507_rescale035"),
    ),
    (
        "base_no_tool_current_fast_hf",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_remaining/base_no_tool_no_system_current_lora_setting_fast_hf_val_only_highpage_0507_rescale05"),
    ),
    (
        "base_no_tool_fast_hf",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_remaining/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale025"),
    ),
    (
        "base_no_tool_fast_hf",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale035"),
    ),
    (
        "base_no_tool_fast_hf",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_20260515_fgtest/base_no_tool_no_system_fast_hf_val_only_highpage_0507_rescale05"),
    ),
    # LoRA highpage runs under the same HF-model rollout style.
    (
        "lora_basic_40k",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514/lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale025"),
    ),
    (
        "lora_basic_40k",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_basic_40k_freeze_vt_medium_only_retry_highpage_0507_rescale035"),
    ),
    (
        "lora_basic_40k",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514/lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale05"),
    ),
    (
        "lora_arxiv_higher_dpi",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale025"),
    ),
    (
        "lora_arxiv_higher_dpi",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale035"),
    ),
    (
        "lora_arxiv_higher_dpi",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale05"),
    ),
    (
        "lora_O3_higher_dpi",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale025"),
    ),
    (
        "lora_O3_higher_dpi",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale035"),
    ),
    (
        "lora_O3_higher_dpi",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale05"),
    ),
    (
        "lora_both_higher_dpi",
        "0.25",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_20260515_debug/lora_both_w_higher_dpi_highpage_0507_rescale025"),
    ),
    (
        "lora_both_higher_dpi",
        "0.35",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_20260515_debug/lora_both_w_higher_dpi_highpage_0507_rescale035"),
    ),
    (
        "lora_both_higher_dpi",
        "0.5",
        Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_20260515_debug/lora_both_w_higher_dpi_highpage_0507_rescale05"),
    ),
]


METRICS = {
    "longdoc_acc": "val-core/longdocurl0507_highpage/reward/mean@1",
    "mmlong_acc": "val-core/mmlongbench0507_highpage/reward/mean@1",
    "core_time_s": "val-aux/core_inference_time/mean",
    "longdoc_core_time_s": "val-aux/longdocurl0507_highpage/core_inference_time/mean",
    "mmlong_core_time_s": "val-aux/mmlongbench0507_highpage/core_inference_time/mean",
    "prompt_tokens": "val-aux/prompt_tokens/mean",
    "assistant_resp_tokens": "val-aux/response_tokens_generated/mean",
    "tool_resp_tokens": "val-aux/response_tokens_tool/mean",
    "tool_calls": "val-aux/n_valid_tool_calls/mean@1",
    "mmlong_unanswerable_acc": "val-aux/mmlongbench0507_highpage/acc/not_answerable/mean",
}


DISPLAY = {
    "base_no_tool_current_fast_hf": "base no-tool current HF",
    "base_no_tool_fast_hf": "base no-tool HF",
    "lora_basic_40k": "LoRA basic 40k",
    "lora_arxiv_higher_dpi": "LoRA arxiv higher dpi",
    "lora_O3_higher_dpi": "LoRA O3 higher dpi",
    "lora_both_higher_dpi": "LoRA both higher dpi",
}


COLORS = {
    "base_no_tool_current_fast_hf": "#7b3294",
    "base_no_tool_fast_hf": "#c2a5cf",
    "lora_basic_40k": "#1b9e77",
    "lora_arxiv_higher_dpi": "#377eb8",
    "lora_O3_higher_dpi": "#e6550d",
    "lora_both_higher_dpi": "#111111",
}


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
    if values["longdoc_acc"] is None or values["mmlong_acc"] is None or values["core_time_s"] is None:
        return None
    fast_path = "'val_only_hf_model_rollout': True" in text or "val_only_hf_model_rollout: True" in text
    values["avg_acc"] = (float(values["longdoc_acc"]) + float(values["mmlong_acc"])) / 2.0
    exports = len(list((run_dir / "exported_conversations").glob("*.json")))
    return {
        "model": model,
        "scale": scale,
        "run_dir": str(run_dir),
        "exports": exports,
        "fast_path": fast_path,
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
        "longdoc_core_time_s",
        "mmlong_core_time_s",
        "prompt_tokens",
        "assistant_resp_tokens",
        "tool_resp_tokens",
        "tool_calls",
        "mmlong_unanswerable_acc",
        "exports",
        "fast_path",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot(rows: list[dict[str, object]], out_prefix: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 8))

    order = [
        "base_no_tool_current_fast_hf",
        "base_no_tool_fast_hf",
        "lora_basic_40k",
        "lora_arxiv_higher_dpi",
        "lora_O3_higher_dpi",
        "lora_both_higher_dpi",
    ]
    for model in order:
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
            linewidth=2.6,
            markersize=7,
            color=COLORS.get(model),
            label=DISPLAY.get(model, model),
        )
        for row in points:
            ax.annotate(
                str(row["scale"]),
                (float(row["core_time_s"]), float(row["avg_acc"])),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                color=COLORS.get(model),
            )

    ax.set_title("Highpage HF-val-only comparison: LoRA vs base no-tool", fontsize=16)
    ax.set_xlabel("Average core inference time (s)", fontsize=13)
    ax.set_ylabel("Average accuracy over longdocurl + mmlongbench", fontsize=13)
    ax.legend(loc="best", frameon=True)
    ax.set_ylim(0.34, 0.49)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(out_prefix.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="highpage_hf_val_only_lora_comparison_2026-05-15")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for model, scale, run_dir in RUNS:
        row = read_row(model, scale, run_dir)
        if row is None:
            missing.append((model, scale, str(run_dir)))
            continue
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
