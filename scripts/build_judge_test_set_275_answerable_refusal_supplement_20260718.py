#!/usr/bin/env python3
"""Add focused answerable-refusal negatives to the corrected judge test set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
BASE_DIR = REPO / "notes/generated/judge_test_set_250_current_legacy_fp_mix_20260717"
BASE_PARQUET = BASE_DIR / "judge_test_set_250.parquet"
SOURCE_PARQUETS = [
    REPO / "notes/generated/judge_test_set_200_hard_20260717/judge_test_set_200.parquet",
    REPO / "notes/generated/judge_test_set_200_20260717/judge_test_set_200.parquet",
]
OUT_DIR = REPO / "notes/generated/judge_test_set_275_answerable_refusal_supplement_20260718"


# Manually selected from the broader mined pools. Each row is an answerable QA
# with a concrete GT and a refusal/insufficient-information-style answer that
# was previously scored correct by an existing judge/reward path.
SELECTED_IDS = [
    "787aac0af43f3f63",
    "661cbcbc79079f55",
    "5dcb7606d8bc198f",
    "98d2421edfd3b477",
    "223894154e25ef3a",
    "f1bb16af60496cbc",
    "0b813c09ceecd37d",
    "e706331a83c87500",
    "c125265e8cef1e3f",
    "4d30186fdb282693",
    "dede5df767b5fe6d",
    "ec67c8c7cda8ca3f",
    "2b9435917a8c8cb2",
    "2a3d5d5c60fe8387",
    "b301af0cae3e4a49",
    "7f7419ae9b48a73f",
    "4b27458bd1622629",
    "39313756b2026133",
    "ede03d044347bb45",
    "28c7c3af9808eb80",
    "2bd7e0acc4c707b8",
    "8ab93ca217bb81ee",
    "d1cbdaf8554e9048",
    "5a3b05159008cab3",
    "ffe4e84b4a78d51c",
]


def to_json_list(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps([value], ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[]"
    return json.dumps([str(value)], ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PARQUET)

    source_frames = []
    for path in SOURCE_PARQUETS:
        df = pd.read_parquet(path)
        df["source_artifact"] = str(path)
        source_frames.append(df)
    candidates = pd.concat(source_frames, ignore_index=True)
    selected = candidates[candidates["id"].isin(SELECTED_IDS)].copy()

    found_ids = set(selected["id"].astype(str))
    missing = [row_id for row_id in SELECTED_IDS if row_id not in found_ids]
    if missing:
        raise RuntimeError(f"missing selected ids: {missing}")
    if selected["id"].duplicated().any():
        selected = selected.drop_duplicates("id", keep="first")
    if len(selected) != len(SELECTED_IDS):
        raise RuntimeError(f"expected {len(SELECTED_IDS)} selected rows, got {len(selected)}")

    overlap = set(base["id"].astype(str)) & set(selected["id"].astype(str))
    if overlap:
        raise RuntimeError(f"selected rows already present in base set: {sorted(overlap)}")

    selected["human_label"] = 0
    selected["row_origin"] = "answerable_refusal_supplement_20260718"
    selected["manual_pass_status"] = "accepted"
    selected["manual_pass_reviewer"] = "codex"
    selected["manual_pass_notes"] = (
        "answerable QA with concrete GT; refusal/insufficient-information-style answer should be incorrect"
    )
    selected["failure_mode_json"] = selected.apply(
        lambda row: to_json_list(
            [
                "answerable_refusal_should_be_incorrect",
                "mc_answerable_refusal_should_be_incorrect",
            ]
            if row.get("question_type") == "mcq"
            else ["answerable_refusal_should_be_incorrect"]
        ),
        axis=1,
    )
    selected["notes"] = selected.get("notes", "").astype(str)
    selected.loc[:, "notes"] = selected["notes"].where(
        selected["notes"].str.len() > 0,
        "added_20260718; focused answerable-refusal negative supplement",
    )

    # Align source rows to the wider corrected artifact schema.
    for col in base.columns:
        if col not in selected.columns:
            selected[col] = None
    selected = selected[base.columns]

    out = pd.concat([base, selected], ignore_index=True)
    if out["id"].duplicated().any():
        dupes = out.loc[out["id"].duplicated(keep=False), "id"].tolist()
        raise RuntimeError(f"duplicate ids in output: {dupes}")

    out_parquet = OUT_DIR / "judge_test_set_275.parquet"
    out_jsonl = OUT_DIR / "judge_test_set_275.jsonl"
    out_csv = OUT_DIR / "judge_test_set_275_annotation.csv"
    added_csv = OUT_DIR / "added_answerable_refusal_rows.csv"

    out.to_parquet(out_parquet, index=False)
    write_jsonl(out_jsonl, out.to_dict(orient="records"))
    out.to_csv(out_csv, index=False)
    selected.to_csv(added_csv, index=False)

    def table(counter: Counter[tuple[Any, ...]], headers: list[str]) -> str:
        lines = [
            "| " + " | ".join(headers + ["count"]) + " |",
            "| " + " | ".join(["---"] * (len(headers) + 1)) + " |",
        ]
        for key, count in sorted(counter.items(), key=lambda item: tuple(str(x) for x in item[0])):
            vals = key if isinstance(key, tuple) else (key,)
            lines.append("| " + " | ".join(str(x) for x in (*vals, count)) + " |")
        return "\n".join(lines)

    label_counts = Counter((f"label_{x}",) for x in out["human_label"])
    added_qtype = Counter((x,) for x in selected["question_type"])
    added_source = Counter((x,) for x in selected["benchmark_or_data_source"])
    content = [
        "# Judge Test Set 275 Answerable-Refusal Supplement",
        "",
        f"Base artifact: `{BASE_DIR.name}`.",
        "",
        f"Rows: {len(out)}",
        f"Added answerable-refusal negative rows: {len(selected)}",
        "",
        "Purpose: strengthen regression coverage for the loophole where a judge gives credit to "
        "refusal/insufficient-information-style answers on answerable questions.",
        "",
        "The added rows are labeled `human_label=0` because the QA has a concrete answer in the GT, "
        "but the model answer refuses or claims the answer cannot be determined.",
        "",
        "## Label Distribution",
        "",
        table(label_counts, ["human_label"]),
        "",
        "## Added Rows By Question Type",
        "",
        table(added_qtype, ["question_type"]),
        "",
        "## Added Rows By Source",
        "",
        table(added_source, ["benchmark_or_data_source"]),
        "",
        "## Outputs",
        "",
        "- `judge_test_set_275.parquet`",
        "- `judge_test_set_275.jsonl`",
        "- `judge_test_set_275_annotation.csv`",
        "- `added_answerable_refusal_rows.csv`",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(content), encoding="utf-8")

    print(f"wrote {len(out)} rows to {OUT_DIR}")
    print(f"added question types: {dict(added_qtype)}")
    print(f"added sources: {dict(added_source)}")


if __name__ == "__main__":
    main()
