#!/usr/bin/env python3
"""Build a 250-row judge set with current-legacy-confirmed FP cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE_DIR = REPO / "notes/generated/judge_test_set_250_hard_evalmix_legacy_fn_20260717"
BASE_PARQUET = BASE_DIR / "judge_test_set_250.parquet"
MINING_DIR = REPO / "notes/generated/current_legacy_fp_mining_20260717"
OUT_DIR = REPO / "notes/generated/judge_test_set_250_current_legacy_fp_mix_20260717"

# Current-legacy-confirmed positives, manually retained as strict false
# positives. These are mostly incomplete list/table-title answers or clearly
# wrong MC answers. Rows are selected by `id` from the mining parquets.
SELECTED_CURRENT_LEGACY_FP_IDS = [
    "59abdd2700e9a0a5",  # Table 5 only, missing required full table name.
    "2dd21b6e3d2dbdaa",  # Table 10 only, missing required full table name.
    "1809ad992dbb2cf6",  # Contact answer gives only phone, misses address/email.
    "56041169ec0242e8",  # Dividend answer gives only 1-share value.
    "cda6d4be3ecf8d53",  # "All listed countries" instead of enumerating countries.
    "36057604fb003959",  # One Islamic title instead of four required titles.
    "115ea619b71d1d38",  # One Zen title instead of three required titles.
    "51c4ac68e2285a91",  # One 1936 name instead of three names.
    "2a51aee1f3df75ea",  # One chairperson instead of four listed names.
    "2d814e7cb5abbbf0",  # Address misses city/state detail.
    "9bd43b5b19e848a6",  # Logo answer misses visual elements and department text.
    "c162f2e49cddf0aa",  # Vague answer misses statutory wording.
    "578c981a640dbc69",  # Section-title answer truncates required hierarchy.
    "bf2ece8f81c81b96",  # Visual-field answer misses 0.75 degree / 4200 microns.
    "40d5367df40a7e3e",  # Court answer is too abbreviated.
    "537bda0eb6f56f25",  # Table numbers only, missing exact table names.
    "e04c2eb26a879015",  # Version numbers only, missing Spring Framework labels.
    "3ad4f2b7120e7777",  # Three atheist/secularist titles, missing ATHEISM.
    "74961da5fa9fcfd2",  # MC motion: GT stopped, answer parked.
    "e012209df8c4b1bc",  # MC motion: GT stopped, answer parked.
]

REMOVE_FN_BY_SOURCE = {
    "dude": 10,
    "mmlongbench": 7,
    "longdocurl": 3,
}


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item["id"])] = item
    return out


def load_mining_rows() -> pd.DataFrame:
    frames = []
    sources = [
        (
            MINING_DIR / "strict_negative_candidates.parquet",
            MINING_DIR / "legacy_eval_20260717_globalai/predictions_legacy.jsonl",
        ),
        (
            MINING_DIR / "incomplete_answer_candidates.parquet",
            MINING_DIR / "incomplete_answer_legacy_eval_20260717_globalai/predictions_legacy.jsonl",
        ),
    ]
    for parquet_path, pred_path in sources:
        df = pd.read_parquet(parquet_path)
        preds = load_predictions(pred_path)
        pred_df = pd.DataFrame(preds.values())
        merged = df.merge(
            pred_df[["id", "judge_score", "judge_pred", "score"]],
            on="id",
            how="left",
            suffixes=("", "_current_legacy"),
        )
        frames.append(merged)
    all_rows = pd.concat(frames, ignore_index=True)
    selected = all_rows[all_rows["id"].astype(str).isin(SELECTED_CURRENT_LEGACY_FP_IDS)].copy()
    missing = sorted(set(SELECTED_CURRENT_LEGACY_FP_IDS) - set(selected["id"].astype(str)))
    if missing:
        raise RuntimeError(f"missing selected current-legacy FP rows: {missing}")
    selected = selected.drop_duplicates(subset=["id"], keep="first")
    if len(selected) != len(SELECTED_CURRENT_LEGACY_FP_IDS):
        raise RuntimeError(f"expected {len(SELECTED_CURRENT_LEGACY_FP_IDS)} selected rows, got {len(selected)}")
    bad = selected[selected["judge_pred"] != 1]
    if len(bad):
        raise RuntimeError(f"selected rows not current-legacy positive: {bad['id'].tolist()}")
    return selected


def remove_fn_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    remove_indices: list[int] = []
    supplement = df[
        (df["row_origin"].astype(str) == "paired_score_file_legacy0_v2_1")
        & (df["answerability"].astype(str) == "unanswerable")
    ]
    for source, count in REMOVE_FN_BY_SOURCE.items():
        part = supplement[supplement["benchmark_or_data_source"].astype(str) == source]
        if len(part) < count:
            raise RuntimeError(f"not enough removable FN supplement rows for {source}: {len(part)} < {count}")
        remove_indices.extend(part.index[:count].tolist())
    removed = df.loc[remove_indices].copy()
    kept = df.drop(index=remove_indices).copy()
    return kept, removed


def selected_to_rows(selected: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in selected.iterrows():
        base = row.to_dict()
        flags = []
        raw_flags = base.get("heuristic_flags_json")
        if isinstance(raw_flags, str) and raw_flags.strip():
            try:
                flags = json.loads(raw_flags)
            except json.JSONDecodeError:
                flags = [raw_flags]
        flags = sorted(set(flags + ["current_legacy_false_positive", "manual_pass_accepted"]))
        base.update(
            {
                "old_judge_name": "current_legacy_judge",
                "old_judge_score": float(base.get("judge_score", 1.0)),
                "heuristic_flags_json": json.dumps(flags, ensure_ascii=False),
                "human_label": 0,
                "failure_mode_json": json.dumps(["current_legacy_false_positive"], ensure_ascii=False),
                "notes": "added_20260717; current legacy rerun scored this row correct; manually retained as strict false positive",
                "manual_pass_status": "accepted",
                "manual_pass_reviewer": "codex_20260717",
                "manual_pass_notes": "Question/GT/final answer manually inspected; retained as a current-legacy false-positive negative.",
                "row_origin": "current_legacy_fp_mining_confirmed",
                "single_call_v2_score": base.get("single_call_v2_score"),
                "current_legacy_score_json": json.dumps(base.get("score"), ensure_ascii=False),
            }
        )
        for key in ("score", "judge_score", "judge_pred"):
            base.pop(key, None)
        rows.append(base)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PARQUET)
    selected = load_mining_rows()
    kept, removed = remove_fn_rows(base)
    fp_rows = selected_to_rows(selected)
    out_rows = kept.to_dict(orient="records") + fp_rows
    out_df = pd.DataFrame(out_rows)
    if len(out_df) != 250:
        raise RuntimeError(f"expected 250 rows, got {len(out_df)}")

    out_df.to_parquet(OUT_DIR / "judge_test_set_250.parquet", index=False)
    write_jsonl(OUT_DIR / "judge_test_set_250.jsonl", out_rows)

    annotation_cols = [
        "id",
        "source_split",
        "model_name",
        "benchmark_or_data_source",
        "answerability",
        "question_type",
        "question",
        "ground_truth",
        "final_answer",
        "old_judge_name",
        "old_judge_score",
        "single_call_v2_score",
        "legacy_extracted_answer",
        "heuristic_flags_json",
        "human_label",
        "failure_mode_json",
        "notes",
        "manual_pass_status",
        "manual_pass_reviewer",
        "manual_pass_notes",
        "trajectory_path",
    ]
    out_df.reindex(columns=annotation_cols).to_csv(OUT_DIR / "judge_test_set_250_annotation.csv", index=False)
    selected.to_csv(OUT_DIR / "added_current_legacy_fp_candidates.csv", index=False)
    removed.to_csv(OUT_DIR / "removed_legacy_fn_rows.csv", index=False)

    fp_df = pd.DataFrame(fp_rows)
    summary = [
        "# Judge Test Set 250 Current-Legacy FP Mix",
        "",
        f"Base artifact: `{BASE_DIR.name}`.",
        "",
        f"Rows: {len(out_df)}",
        f"Removed legacy-FN supplement rows: {len(removed)}",
        f"Added current-legacy-confirmed FP rows: {len(fp_rows)}",
        "",
        "Important cleanup:",
        "",
        "- This artifact does not use the stale `paired_legacy1_v2_0` FP additions from `judge_test_set_250_legacy_fp_mix_20260717`.",
        "- Added FP rows were confirmed by a fresh current-legacy judge run in `current_legacy_fp_mining_20260717`.",
        "",
        "Label distribution:",
        "",
        out_df["human_label"].value_counts().sort_index().rename(index={0: "label_0", 1: "label_1"}).to_markdown(),
        "",
        "Added current-legacy FP rows by source:",
        "",
        fp_df["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "Removed legacy-FN rows by source:",
        "",
        removed["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "Outputs:",
        "",
        "- `judge_test_set_250.parquet`",
        "- `judge_test_set_250.jsonl`",
        "- `judge_test_set_250_annotation.csv`",
        "- `added_current_legacy_fp_candidates.csv`",
        "- `removed_legacy_fn_rows.csv`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {len(out_df)} rows to {OUT_DIR}")
    print("labels")
    print(out_df["human_label"].value_counts().sort_index().to_string())
    print("added FP by source")
    print(fp_df["benchmark_or_data_source"].value_counts().to_string())
    print("removed FN by source")
    print(removed["benchmark_or_data_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
