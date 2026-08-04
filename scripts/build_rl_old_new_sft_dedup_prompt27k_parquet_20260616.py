#!/usr/bin/env python3
"""Build old+new RL parquet after SFT dedup and a 27k prompt cap.

This reconstructs the merged source pool discussed in the RL data notes:

1. Load the old medium-only wrong-question RL source manifests.
2. Load the new answerable + synthetic-unanswerable RL manifest.
3. Drop duplicated new 0.25-rescale rows first.
4. Drop old rows whose question_id appears in the remaining new rows.
5. Deduplicate against the SFT data used by
   full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519.
6. Drop only rows whose exact rough prompt estimate exceeds 27k tokens.

The output manifest uses absolute image paths because the selected rows span
both /home and /scratch source trees.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
INSIGHT_DOC_VERL_ROOT = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/verl")
CREATE_PARQUET = INSIGHT_DOC_VERL_ROOT / "recipe/vsearch/create_parquet_dataset.py"

OLD_RL_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only")
NEW_RL_MANIFEST = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale/manifest.jsonl"
)
PROMPT_11000_DROPPED_CSV = (
    REPO_ROOT / "notes/generated/prompt_cap_11000_dropped_rows_20260601/dropped_rows.csv"
)
GENERATED_BASE_DIR = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
UNANSWERABLE_SFT_ROOT = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517"

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notes/generated/rl_old_new_sft_dedup_prompt27k_20260616"
DEFAULT_OUTPUT_PARQUET = DEFAULT_OUTPUT_ROOT / (
    "insight_doc_rl_old_new_sft_dedup_prompt27k_granular_data_source-insight_qwen_agent.parquet"
)

OLD_INITIAL_RESCALE = 0.25
OLD_INITIAL_RESCALE_DPI = 50
PROMPT_CAP = 27000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--validate-images", action="store_true")
    parser.add_argument(
        "--data-source-mode",
        choices=["granular", "single"],
        default="granular",
        help="granular enables sampler weighting by origin/source/answerability/rescale.",
    )
    parser.add_argument(
        "--rescale-over-cap-to",
        type=float,
        default=None,
        help="If set, rows over the prompt cap are kept with this initial_rescale instead of dropped.",
    )
    parser.add_argument(
        "--rescale-over-cap-dpi",
        type=int,
        default=70,
        help="initial_rescale_dpi to use for rows kept by --rescale-over-cap-to.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def strip_image_preamble(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"(?m)^\s*Image\s+\d+\s*:\s*<image>\s*$", "", text)
    text = re.sub(r"(?m)^\s*---\s*$", "", text)
    text = text.replace("<image>", "")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def qid(row: dict[str, Any]) -> str:
    return str(row.get("question_id", ""))


def get_question(row: dict[str, Any]) -> str:
    return str(row.get("question", ""))


def pdf_image_root_from_manifest_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve().parent / "pdf_image"


def absolutize_images(row: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    out = dict(row)
    images = out.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"row missing images: {qid(row)}")
    abs_images = []
    for image in images:
        image_path = Path(str(image))
        if not image_path.is_absolute():
            if image_root is None:
                raise ValueError(f"relative image without image root for {qid(row)}: {image}")
            image_path = image_root / image_path
        abs_images.append(str(image_path.resolve()))
    out["images"] = abs_images
    return out


def source_bucket(row: dict[str, Any]) -> str:
    subset = str(row.get("subset") or "").lower()
    doc = str(row.get("document_id") or "").lower()
    row_text = " ".join([subset, doc, qid(row).lower()])
    if subset in {"veqa", "mveqa"} or "arxiv" in row_text:
        return "arxiv"
    if "bigpage_map" in row_text or "combine_map" in row_text or "metromap" in row_text:
        return "map"
    if "bigpage_poster" in row_text or "combine_p2p" in row_text or "poster" in row_text:
        return "poster"
    if "bigpage_info" in row_text or "combine_info" in row_text:
        return "info"
    if "dude" in row_text:
        return "dude"
    if "docvqa" in row_text:
        return "docvqa"
    if "mpdocvqa" in row_text:
        return "mpdocvqa"
    cleaned = re.sub(r"[^a-z0-9]+", "_", subset).strip("_")
    return cleaned or "unknown"


def is_unanswerable(row: dict[str, Any]) -> bool:
    data_source = str(row.get("data_source") or "").lower()
    if "unanswerable" in data_source:
        return True
    text = " ".join(
        [
            qid(row).lower(),
            json.dumps(row.get("question_type"), ensure_ascii=False).lower(),
            str(row.get("answer") or "").lower(),
        ]
    )
    return "unanswerable" in text or "not-answerable" in text or "not answerable" in text


def rescale_tag(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "runknown"
    if abs(numeric - 0.25) < 1e-6:
        return "r025"
    if abs(numeric - 0.35) < 1e-6:
        return "r035"
    if abs(numeric - 0.5) < 1e-6:
        return "r05"
    return "r" + re.sub(r"[^0-9]+", "", f"{numeric:.6g}")


def set_data_source(row: dict[str, Any], mode: str) -> None:
    original = row.get("data_source")
    origin = str(row.get("rl_merge_origin") or "unknown")
    bucket = str(row.get("source_bucket") or source_bucket(row))
    ans = "unanswerable" if bool(row.get("is_unanswerable")) else "answerable"
    tag = rescale_tag(row.get("initial_rescale"))
    row["original_data_source"] = original
    if mode == "single":
        row["data_source"] = "insight_doc_rl"
    else:
        row["data_source"] = f"insight_doc_rl_{origin}_{bucket}_{ans}_{tag}"


def load_old_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(OLD_RL_ROOT.glob("*/manifest.jsonl")):
        meta_path = manifest_path.parent / "meta.json"
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        image_root = pdf_image_root_from_manifest_path(meta["manifest_path"])
        for row in read_jsonl(manifest_path):
            out = absolutize_images(row, image_root)
            out["initial_rescale"] = OLD_INITIAL_RESCALE
            out["initial_rescale_source"] = "old_rl_u25"
            out["initial_rescale_dpi"] = OLD_INITIAL_RESCALE_DPI
            out["rl_merge_origin"] = "old"
            out["source_manifest_group"] = meta.get("group_name")
            out["source_bucket"] = source_bucket(out)
            out["is_unanswerable"] = is_unanswerable(out)
            rows.append(out)
    return rows


def load_new_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(NEW_RL_MANIFEST):
        out = absolutize_images(row, None)
        out["rl_merge_origin"] = "new"
        out["source_bucket"] = source_bucket(out)
        out["is_unanswerable"] = is_unanswerable(out)
        rows.append(out)
    return rows


def drop_new_rescale025_duplicates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = Counter(qid(row) for row in rows)
    kept = []
    dropped = []
    for row in rows:
        if counts[qid(row)] > 1 and abs(float(row.get("initial_rescale", -1)) - 0.25) < 1e-6:
            dropped.append(row)
        else:
            kept.append(row)
    return kept, dropped


def sft_train_files() -> list[Path]:
    parts = [
        "O3_data_0424/train_part1",
        "O3_data_0424/train_part2a",
        "O3_data_0424/train_part2b",
        "O3_data_0424/train_part2c",
        "O3_data_0424/dude_poster_unanswerable",
        "arxiv/train_part1",
        "arxiv/train_part2",
        "arxiv/train_part3",
        "arxiv/spanning_train_part1",
        "arxiv/train_part4",
        "arxiv/train_part5",
        "O3_data_0424/train_part3a",
        "O3_data_0424/train_part3b",
        "O3_data_0424/train_part3c",
        "O3_data_0424/train_part3d",
    ]
    files = []
    for part in parts:
        aspect_filtered = (
            GENERATED_BASE_DIR
            / part
            / "medium/processed_gpt5_nano_rewrite_aspect_drop/sft_data_base_model_tool_argument_order.parquet"
        )
        regular = (
            GENERATED_BASE_DIR
            / part
            / "medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet"
        )
        if part in {
            "O3_data_0424/train_part3a",
            "O3_data_0424/train_part3b",
            "O3_data_0424/train_part3c",
            "O3_data_0424/train_part3d",
        } and aspect_filtered.exists():
            files.append(aspect_filtered)
        else:
            files.append(regular)
    for scale in ["rescale025", "rescale035", "rescale05"]:
        files.append(
            UNANSWERABLE_SFT_ROOT
            / scale
            / "medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet"
        )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing SFT train parquets:\n" + "\n".join(missing))
    return files


def extract_raw_question(record: dict[str, Any]) -> str | None:
    extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    question = extra_info.get("question")
    if question:
        return str(question)
    for message in record.get("conversation") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, dict):
            role = content.get("role") or message.get("role")
            text = content.get("text")
        else:
            role = message.get("role") if isinstance(message, dict) else None
            text = content
        if role == "user" and text:
            return strip_image_preamble(text)
    return None


def collect_raw_question_to_qids(raw_files: set[Path]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for raw_path in sorted(raw_files):
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue
        extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
        raw_qid = extra_info.get("question_id") or record.get("question_id")
        question = extract_raw_question(record)
        if raw_qid and question:
            mapping[normalize_text(question)].add(str(raw_qid))
    return mapping


def collect_sft_raw_question_to_qids_by_dir(files: list[Path]) -> dict[Path, dict[str, set[str]]]:
    medium_dirs = {path.parents[1] for path in files}
    return {
        medium_dir: collect_raw_question_to_qids(set(medium_dir.glob("raw*/*.json")))
        for medium_dir in sorted(medium_dirs)
    }


def extract_sft_user_question(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return strip_image_preamble(message.get("content"))
    return ""


def collect_sft_dedup_keys(files: list[Path]) -> tuple[set[str], set[str], dict[str, Any]]:
    raw_question_to_qids_by_dir = collect_sft_raw_question_to_qids_by_dir(files)

    sft_qids: set[str] = set()
    fallback_questions: set[str] = set()
    total_rows = 0
    qid_mapped_rows = 0
    fallback_rows = 0
    per_file_rows: dict[str, int] = {}
    ambiguous_raw_questions = 0
    raw_question_keys = 0
    for path in files:
        raw_question_to_qids = raw_question_to_qids_by_dir.get(path.parents[1], {})
        unique_question_to_qid = {
            question: next(iter(qids)) for question, qids in raw_question_to_qids.items() if len(qids) == 1
        }
        ambiguous_raw_questions += sum(1 for qids in raw_question_to_qids.values() if len(qids) > 1)
        raw_question_keys += len(raw_question_to_qids)
        table = pq.read_table(path, columns=["messages"])
        rows = table.to_pylist()
        per_file_rows[str(path)] = len(rows)
        for row in rows:
            total_rows += 1
            question_norm = normalize_text(extract_sft_user_question(row["messages"]))
            recovered_qid = unique_question_to_qid.get(question_norm)
            if recovered_qid:
                sft_qids.add(recovered_qid)
                qid_mapped_rows += 1
            else:
                fallback_questions.add(question_norm)
                fallback_rows += 1
    summary = {
        "train_files": [str(path) for path in files],
        "per_file_rows": per_file_rows,
        "sft_total_rows": total_rows,
        "qid_mapped_rows": qid_mapped_rows,
        "fallback_rows": fallback_rows,
        "unique_sft_qids": len(sft_qids),
        "unique_fallback_questions": len(fallback_questions),
        "raw_question_keys": raw_question_keys,
        "ambiguous_raw_questions": ambiguous_raw_questions,
    }
    return sft_qids, fallback_questions, summary


def load_over_prompt_cap_keys(csv_path: Path, cap: float) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    keys = set()
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if float(record["estimated_prompt_tokens"]) <= cap:
                continue
            key = (str(record["question_id"]), rescale_tag(record["initial_rescale"]))
            keys.add(key)
            rows.append(record)
    return keys, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_create_parquet(data_root: Path, output_parquet: Path, validate_images: bool) -> None:
    cmd = [
        sys.executable,
        str(CREATE_PARQUET),
        "--dataset",
        "InSightDocRL",
        "--data_root",
        str(data_root),
        "--split",
        "all",
        "--prompt",
        "insight_qwen_agent",
        "--output_path",
        str(output_parquet),
        "--agent_name",
        "insight_qwen_agent",
        "--num_workers",
        "1",
        "--extra_options",
        json.dumps({"manifest_file": "manifest.jsonl"}),
    ]
    if validate_images:
        cmd.append("--validate_images")
    subprocess.run(cmd, cwd=INSIGHT_DOC_VERL_ROOT, check=True)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "origin_counts": dict(Counter(str(row.get("rl_merge_origin")) for row in rows)),
        "data_source_counts": dict(Counter(str(row.get("data_source")) for row in rows)),
        "source_bucket_counts": dict(Counter(str(row.get("source_bucket")) for row in rows)),
        "answerability_counts": dict(
            Counter("unanswerable" if row.get("is_unanswerable") else "answerable" for row in rows)
        ),
        "rescale_counts": dict(Counter(rescale_tag(row.get("initial_rescale")) for row in rows)),
        "source_answerability_counts": dict(
            Counter(
                f"{row.get('source_bucket')}/"
                f"{'unanswerable' if row.get('is_unanswerable') else 'answerable'}"
                for row in rows
            )
        ),
        "rescale_answerability_counts": dict(
            Counter(
                f"{rescale_tag(row.get('initial_rescale'))}/"
                f"{'unanswerable' if row.get('is_unanswerable') else 'answerable'}"
                for row in rows
            )
        ),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    old_rows = load_old_rows()
    new_rows_raw = load_new_rows()
    new_rows, dropped_new_025_dupes = drop_new_rescale025_duplicates(new_rows_raw)

    new_qids = {qid(row) for row in new_rows}
    old_overlap_dropped = [row for row in old_rows if qid(row) in new_qids]
    old_after_new_dedup = [row for row in old_rows if qid(row) not in new_qids]
    merged = old_after_new_dedup + new_rows

    sft_files = sft_train_files()
    sft_qids, sft_fallback_questions, sft_summary = collect_sft_dedup_keys(sft_files)
    sft_dropped = [
        row
        for row in merged
        if qid(row) in sft_qids or normalize_text(get_question(row)) in sft_fallback_questions
    ]
    after_sft = [
        row
        for row in merged
        if qid(row) not in sft_qids and normalize_text(get_question(row)) not in sft_fallback_questions
    ]

    over27_keys, over27_rows_from_csv = load_over_prompt_cap_keys(PROMPT_11000_DROPPED_CSV, PROMPT_CAP)
    over27_candidates = [
        row
        for row in after_sft
        if row.get("rl_merge_origin") == "new" and (qid(row), rescale_tag(row.get("initial_rescale"))) in over27_keys
    ]
    if args.rescale_over_cap_to is None:
        over27_dropped = over27_candidates
        over27_rescaled = []
        final_rows = [
            row
            for row in after_sft
            if not (
                row.get("rl_merge_origin") == "new"
                and (qid(row), rescale_tag(row.get("initial_rescale"))) in over27_keys
            )
        ]
    else:
        over27_dropped = []
        over27_rescaled = []
        final_rows = []
        for row in after_sft:
            if row.get("rl_merge_origin") == "new" and (qid(row), rescale_tag(row.get("initial_rescale"))) in over27_keys:
                row = dict(row)
                row["initial_rescale_before_prompt_cap_recovery"] = row.get("initial_rescale")
                row["initial_rescale_source_before_prompt_cap_recovery"] = row.get("initial_rescale_source")
                row["initial_rescale_dpi_before_prompt_cap_recovery"] = row.get("initial_rescale_dpi")
                row["initial_rescale"] = float(args.rescale_over_cap_to)
                row["initial_rescale_source"] = (
                    f"{row.get('initial_rescale_source')}_promptcap{int(PROMPT_CAP)}_"
                    f"rescaled_to_{rescale_tag(args.rescale_over_cap_to)}"
                )
                row["initial_rescale_dpi"] = int(args.rescale_over_cap_dpi)
                over27_rescaled.append(row)
            final_rows.append(row)

    for row in final_rows:
        set_data_source(row, args.data_source_mode)

    pdf_image = output_root / "pdf_image"
    if not pdf_image.exists():
        pdf_image.symlink_to(Path("/"))
    elif not pdf_image.is_symlink() or pdf_image.resolve() != Path("/"):
        raise FileExistsError(f"refusing to use existing non-root pdf_image path: {pdf_image}")

    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest_path, final_rows)
    write_csv(output_root / "over_27k_reference_rows.csv", over27_rows_from_csv)
    write_jsonl(output_root / "dropped_over_27k_manifest_rows.jsonl", over27_dropped)
    write_jsonl(output_root / "rescaled_over_27k_manifest_rows.jsonl", over27_rescaled)

    summary = {
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "output_parquet": str(output_parquet),
        "data_source_mode": args.data_source_mode,
        "prompt_cap": PROMPT_CAP,
        "stages": {
            "old_initial": len(old_rows),
            "new_initial": len(new_rows_raw),
            "new_rescale025_duplicate_dropped": len(dropped_new_025_dupes),
            "new_after_rescale025_duplicate_drop": len(new_rows),
            "old_dropped_because_qid_in_new": len(old_overlap_dropped),
            "old_after_new_qid_dedup": len(old_after_new_dedup),
            "merged_before_sft_dedup": len(merged),
            "sft_dedup_dropped": len(sft_dropped),
            "after_sft_dedup": len(after_sft),
            "over27_csv_keys": len(over27_keys),
            "over27_dropped_after_sft": len(over27_dropped),
            "over27_rescaled_after_sft": len(over27_rescaled),
            "final_rows": len(final_rows),
        },
        "rescale_over_cap_to": args.rescale_over_cap_to,
        "rescale_over_cap_dpi": args.rescale_over_cap_dpi,
        "sft_dedup": sft_summary,
        "final_distribution": summarize_rows(final_rows),
        "over27_distribution": summarize_rows(over27_dropped),
        "over27_rescaled_distribution": summarize_rows(over27_rescaled),
        "new_rescale025_duplicate_drop_distribution": summarize_rows(dropped_new_025_dupes),
        "old_overlap_drop_distribution": summarize_rows(old_overlap_dropped),
        "sft_drop_distribution": summarize_rows(sft_dropped),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    run_create_parquet(output_root, output_parquet, args.validate_images)

    table = pq.read_table(output_parquet, columns=["data_source"])
    parquet_rows = table.num_rows
    parquet_data_source_counts = Counter(table.column("data_source").to_pylist())
    verify = {
        "parquet_rows": parquet_rows,
        "parquet_data_source_counts": dict(parquet_data_source_counts),
    }
    with (output_root / "verify.json").open("w", encoding="utf-8") as handle:
        json.dump(verify, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary["stages"], "verify": verify}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
