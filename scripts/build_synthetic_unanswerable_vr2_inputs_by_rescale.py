#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = REPO_ROOT / "artifacts" / "synthetic_unanswerable_pipeline"
OUTPUT_ROOT = OLD_ROOT / "by_rescale_20260518"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rescale_from_lane_name(name: str) -> str:
    if "rescale025" in name:
        return "025"
    if "rescale035" in name:
        return "035"
    if "rescale05" in name:
        return "05"
    raise ValueError(f"unrecognized rescale in lane name: {name}")


def write_question_ids(path: Path, question_ids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in question_ids), encoding="utf-8")


def build_group(kind: str, merged_root_name: str, parquet_field_name: str) -> dict[str, object]:
    source_root = OLD_ROOT / merged_root_name
    source_summary = load_summary(source_root / "summary.json")
    out_root = OUTPUT_ROOT / kind
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"output_root": str(out_root), "rescale_groups": {}}
    for rescale in ("025", "035", "05"):
        group_dir = out_root / f"rescale{rescale}"
        lane_dir = group_dir / "lanes"
        lane_dir.mkdir(parents=True, exist_ok=True)

        lanes = [lane for lane in source_summary["lanes"] if rescale_from_lane_name(lane["name"]) == rescale]
        if not lanes:
            raise ValueError(f"no lanes found for {kind} rescale{rescale}")

        group_frames: list[pd.DataFrame] = []
        group_qids: list[str] = []
        group_info: list[dict[str, object]] = []

        for lane in lanes:
            src_parquet = Path(lane["parquet"])
            src_qids_file = Path(lane["question_ids_file"])
            if not src_parquet.exists():
                raise FileNotFoundError(src_parquet)
            if not src_qids_file.exists():
                raise FileNotFoundError(src_qids_file)

            qids = [line.strip() for line in src_qids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not qids:
                raise ValueError(f"empty question ids for lane {lane['name']}")
            group_qids.extend(qids)

            df = pd.read_parquet(src_parquet)
            group_frames.append(df)

            dst_lane_dir = lane_dir / lane["name"]
            dst_lane_dir.mkdir(parents=True, exist_ok=True)
            dst_qids = dst_lane_dir / "question_ids.txt"
            write_question_ids(dst_qids, qids)
            dst_parquet = dst_lane_dir / src_parquet.name
            if dst_parquet.exists():
                dst_parquet.unlink()
            dst_parquet.symlink_to(src_parquet)

            group_info.append(
                {
                    "name": lane["name"],
                    "source_parquet": str(src_parquet),
                    "question_ids_file": str(dst_qids),
                    "rows": len(df),
                }
            )

        unique_qids = sorted(set(group_qids))
        if len(unique_qids) != len(group_qids):
            raise ValueError(f"duplicate question_ids within {kind} rescale{rescale}: {len(group_qids)} vs {len(unique_qids)}")

        merged = pd.concat(group_frames, ignore_index=True)
        merged_parquet = group_dir / parquet_field_name.format(rescale=rescale)
        merged.to_parquet(merged_parquet)
        write_question_ids(group_dir / "question_ids.txt", unique_qids)

        summary["rescale_groups"][f"rescale{rescale}"] = {
            "rescale": rescale,
            "parquet": str(merged_parquet),
            "question_ids_file": str(group_dir / "question_ids.txt"),
            "rows": len(merged),
            "question_ids": len(unique_qids),
            "lanes": group_info,
        }

    return summary


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    easy = build_group(
        kind="easy",
        merged_root_name="all_easy_vr2_input_20260518",
        parquet_field_name="sft_data.vreasoner_v2_easy_rescale{rescale}.parquet",
    )
    hard = build_group(
        kind="hard",
        merged_root_name="all_hard_retry_vr2_input_20260518",
        parquet_field_name="sft_data.vreasoner_v2_hard_retry_rescale{rescale}.parquet",
    )
    summary = {"output_root": str(OUTPUT_ROOT), "easy": easy, "hard": hard}
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
