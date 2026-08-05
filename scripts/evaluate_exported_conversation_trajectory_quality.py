#!/usr/bin/env python3
"""Evaluate trajectory quality for exported standalone/eval conversations."""

from __future__ import annotations

import argparse
import csv
import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_sft_trajectory_quality as sftq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--final-output-json",
        type=Path,
        default=None,
        help="Optional final_output JSON used to recover evidence pages/boxes by question id.",
    )
    parser.add_argument("--redundant-iou-threshold", type=float, default=0.8)
    parser.add_argument("--evidence-coverage-threshold", type=float, default=0.5)
    return parser.parse_args()


def bench_from_path(path: Path, record: dict[str, Any]) -> str:
    reward = record.get("reward") or {}
    data_source = reward.get("data_source")
    if data_source:
        return str(data_source)
    name = path.name
    match = re.match(r"(.+?)-val-", name)
    return match.group(1) if match else "<unknown>"


def rescale_from_path(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        if parent.name.startswith("rescale"):
            return parent.name
    return "<unknown>"


def nested_get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_maybe_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return re.split(r"[,\s]+", text)
    return [value]


def parse_int_pages(value: Any) -> list[int]:
    pages = []
    for item in parse_maybe_list(value):
        try:
            pages.append(int(item))
        except (TypeError, ValueError):
            continue
    return pages


def is_unanswerable_final_output(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key, "")).lower()
        for key in ("question_type", "answer", "subset")
    )
    return (
        "not-answerable" in text
        or "not answerable" in text
        or "unanswerable" in text
        or "cannot answer" in text
        or "cannot be answered" in text
    )


def input_image_stem_to_idx(record: dict[str, Any]) -> dict[int, int]:
    refs = record.get("image_references") or {}
    input_images = refs.get("input_images") if isinstance(refs, dict) else None
    if not isinstance(input_images, list):
        return {}
    out = {}
    for list_idx, image_ref in enumerate(input_images):
        if not isinstance(image_ref, dict):
            continue
        idx = as_int(image_ref.get("original_img_idx"))
        if idx is None:
            idx = list_idx
        value = image_ref.get("value") or image_ref.get("path") or image_ref.get("uri")
        if value is None:
            continue
        try:
            stem = int(Path(str(value).replace("file://", "")).stem)
        except (TypeError, ValueError):
            continue
        out[stem] = idx
    return out


def convert_final_output_pages_to_input_indices(
    record: dict[str, Any],
    final_record: dict[str, Any],
    raw_pages: list[int],
    source: str,
    field_name: str,
) -> list[int]:
    stem_to_idx = input_image_stem_to_idx(record)
    converted = []
    raw_pages_set = set(raw_pages)
    for page in raw_pages:
        stem = page
        if source == "longdocurl" and field_name == "evidence_pages":
            start_end = final_record.get("start_end_idx")
            if isinstance(start_end, list) and len(start_end) >= 1:
                start = as_int(start_end[0])
                if start is not None:
                    stem = page - start
        elif source == "mmlongbench":
            # MMLongBench evidence_pages/question_page_num are 1-based in the
            # final output, while rendered image stems are 0-based.
            stem = page - 1
        elif source == "dude":
            # DUDE is mixed in the source data: rows containing page 0 are
            # already 0-based; otherwise use the 1-based convention.
            stem = page if 0 in raw_pages_set else page - 1
        else:
            stem = page

        idx = stem_to_idx.get(stem)
        if idx is None:
            # If image filenames are unavailable, fall back to input index only
            # when it is directly in range.
            idx = stem
        if idx >= 0:
            converted.append(idx)
    return sorted(set(converted))


def parse_normalized_boxes_from_text(text: Any) -> list[list[float]]:
    if not isinstance(text, str):
        return []
    boxes = []
    for match in re.finditer(r"<box>\s*\(([^)]*)\)\s*</box>", text):
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) != 4:
            continue
        try:
            box = [float(part) for part in parts]
        except ValueError:
            continue
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def final_output_evidence(
    record: dict[str, Any],
    final_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not final_record or is_unanswerable_final_output(final_record):
        return []

    source = str(final_record.get("source") or "")
    raw_pages: list[int] = []
    field_name = "question_page_num"
    if source in {"longdocurl", "mmlongbench"} and final_record.get("evidence_pages") not in (None, "", [], {}):
        raw_pages = parse_int_pages(final_record.get("evidence_pages"))
        field_name = "evidence_pages"
    elif source == "mpdocvqa" and final_record.get("answer_page_idx") is not None:
        page = as_int(final_record.get("answer_page_idx"))
        raw_pages = [] if page is None else [page]
        field_name = "answer_page_idx"
    else:
        raw_pages = parse_int_pages(final_record.get("question_page_num"))

    page_ids = convert_final_output_pages_to_input_indices(record, final_record, raw_pages, source, field_name)
    evidence = [
        {
            "page_id": page_id,
            "bbox": None,
            "raw_bbox": None,
            "bbox_scaled_to_image": False,
            "kind": f"final_output_{field_name}",
        }
        for page_id in page_ids
    ]

    # LongDocURL sometimes encodes normalized evidence boxes in free-form text.
    # Assign boxes only when page assignment is unambiguous.
    if source == "longdocurl" and len(page_ids) == 1:
        boxes = parse_normalized_boxes_from_text(final_record.get("detailed_evidences"))
        if boxes:
            _, page_sizes = sftq.input_image_page_maps(record)
            width_height = page_sizes.get(page_ids[0])
            if width_height is not None:
                width, height = width_height
                for box in boxes:
                    scaled = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
                    evidence.append(
                        {
                            "page_id": page_ids[0],
                            "bbox": scaled,
                            "raw_bbox": box,
                            "bbox_scaled_to_image": True,
                            "kind": "final_output_longdocurl_normalized_box",
                        }
                    )
    return evidence


def tool_call_limit(record: dict[str, Any]) -> int | None:
    parameters = record.get("parameters") or {}
    for section_name, key in (
        ("loop", "max_tool_calls"),
        ("request", "max_tool_calls"),
        ("request", "max_user_turns"),
    ):
        section = parameters.get(section_name)
        if not isinstance(section, dict):
            continue
        value = as_int(section.get(key))
        if value and value > 0:
            return value
    return None


def stuck_stats(record: dict[str, Any], crops: list[dict[str, Any]], tool_calls: int) -> dict[str, Any]:
    limit = tool_call_limit(record)
    last_run_length, last_page_id = sftq.last_same_page_crop_run(crops)
    exhausted = tool_calls >= limit if limit is not None else None
    return {
        "tool_call_limit": limit,
        "exhausted_tool_call_limit": exhausted,
        "last_same_page_crop_run_length": last_run_length,
        "last_crop_page_id": last_page_id,
        "stuck_exhausted_same_page_tail": (exhausted and last_run_length >= 2) if exhausted is not None else None,
    }


def num_mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else None


def bool_mean(values: list[Any]) -> float | None:
    clean = [bool(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: list[Any], q: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not clean:
        return None
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def crop_count_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        n = int(row.get("num_crops") or 0)
        key = ">=5" if n >= 5 else str(n)
        counts[key] += 1
    return {key: counts[key] for key in ("0", "1", "2", "3", "4", ">=5")}


def evaluate_record(
    path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    final_output_by_qid: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    extra = record.get("extra_info") or {}
    msg_counts = sftq.count_message_types(record)
    crops = sftq.crop_entries(record)
    _, page_image_sizes = sftq.input_image_page_maps(record)
    qid = str(extra.get("question_id")) if extra.get("question_id") is not None else ""
    final_record = final_output_by_qid.get(qid) if final_output_by_qid else None
    evidence = final_output_evidence(record, final_record)

    row = {
        "json_path": str(path),
        "rescale": rescale_from_path(path),
        "benchmark": bench_from_path(path, record),
        "question_id": extra.get("question_id"),
        "document_id": extra.get("document_id"),
        "subset": extra.get("subset"),
        "category": extra.get("category"),
        "status": record.get("status"),
        "has_final_output_match": final_record is not None,
        "accuracy_reward": as_float(nested_get(record, ("reward", "score", "accuracy_reward"))),
        "score": as_float(nested_get(record, ("reward", "score", "score"))),
        "compute_score_success": nested_get(record, ("reward", "compute_score_success")),
        "num_crops": len(crops),
        "num_tool_call_messages": msg_counts.get("tool_call", 0),
        "num_tool_result_messages": msg_counts.get("tool_result", 0),
        "num_tool_result_fail_hints": msg_counts.get("tool_result_fail_hint", 0),
        "num_format_repair_hints": msg_counts.get("format_repair_hint", 0),
        "n_valid_tool_calls": as_int(nested_get(record, ("reward", "score", "n_valid_tool_calls"))),
        "prompt_tokens": as_int(nested_get(record, ("parameters", "loop", "lengths", "prompt_tokens"))),
        "response_tokens_generated": as_int(
            nested_get(record, ("parameters", "loop", "lengths", "response_tokens_generated"))
        ),
        "response_tokens_tool": as_int(nested_get(record, ("parameters", "loop", "lengths", "response_tokens_tool"))),
        "response_tokens_total": as_int(nested_get(record, ("parameters", "loop", "lengths", "response_tokens_total"))),
        "core_inference_time": as_float(
            nested_get(record, ("parameters", "loop", "timing", "core_inference_time"))
        ),
        "conversation_wall_time": as_float(
            nested_get(record, ("parameters", "loop", "timing", "conversation_wall_time"))
        ),
        "tool_calls_time": as_float(nested_get(record, ("parameters", "loop", "timing", "tool_calls"))),
    }
    row.update(sftq.evidence_match_stats(crops, evidence, args.evidence_coverage_threshold))
    row.update(sftq.zoom_area_stats(crops, page_image_sizes))
    row.update(sftq.evidence_step_stats(crops, evidence, args.evidence_coverage_threshold))
    row.update(stuck_stats(record, crops, msg_counts.get("tool_call", 0)))
    row.update(sftq.redundancy_stats(crops, args.redundant_iou_threshold))
    return row


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_crops = sum(int(row.get("num_crops") or 0) for row in rows)
    crop_dist = crop_count_distribution(rows)
    rows_with_tool_limit = [row for row in rows if row.get("exhausted_tool_call_limit") is not None]
    exhausted_limit_rows = [row for row in rows_with_tool_limit if row.get("exhausted_tool_call_limit")]
    stuck_exhausted_rows = [row for row in rows_with_tool_limit if row.get("stuck_exhausted_same_page_tail")]
    page_evidence_rows = [row for row in rows if row.get("has_evidence_pages")]
    box_evidence_rows = [row for row in rows if row.get("has_evidence_boxes")]
    page_evidence_nonzero_crop_rows = [row for row in page_evidence_rows if int(row.get("num_crops") or 0) > 0]
    box_evidence_nonzero_crop_rows = [row for row in box_evidence_rows if int(row.get("num_crops") or 0) > 0]
    evidence_box_pages_hit_by_crops = sum(int(row.get("num_cropped_evidence_box_pages") or 0) for row in rows)
    evidence_box_pages_hit_by_crops_with_coverage = sum(
        int(row.get("num_cropped_evidence_box_pages_covered") or 0) for row in rows
    )
    any_page_hit_rows = [row for row in page_evidence_rows if row.get("first_any_evidence_page_hit_step") is not None]
    all_page_hit_rows = [row for row in page_evidence_rows if row.get("first_all_evidence_pages_hit_step") is not None]
    region_hit_rows = [row for row in box_evidence_rows if row.get("first_evidence_region_coverage_hit_step") is not None]
    total_region_hit_crops = sum(int(row.get("num_crops_hit_evidence_region_at_threshold") or 0) for row in rows)
    box_evidence_total_crops = sum(int(row.get("num_crops") or 0) for row in box_evidence_rows)
    box_evidence_region_hit_crops = sum(
        int(row.get("num_crops_hit_evidence_region_at_threshold") or 0) for row in box_evidence_rows
    )
    return {
        "rows": len(rows),
        "rows_with_final_output_match": sum(1 for row in rows if row.get("has_final_output_match")),
        "accuracy_reward_mean": num_mean([row.get("accuracy_reward") for row in rows]),
        "compute_score_success_rate": bool_mean([row.get("compute_score_success") for row in rows]),
        "mean_crops": num_mean([row.get("num_crops") for row in rows]),
        "median_crops": percentile([row.get("num_crops") for row in rows], 0.5),
        "p90_crops": percentile([row.get("num_crops") for row in rows], 0.9),
        "total_crops": total_crops,
        "zero_crop_fraction": bool_mean([row.get("num_crops") == 0 for row in rows]),
        "crop_count_distribution": crop_dist,
        "mean_tool_call_messages": num_mean([row.get("num_tool_call_messages") for row in rows]),
        "mean_valid_tool_calls": num_mean([row.get("n_valid_tool_calls") for row in rows]),
        "mean_core_inference_time": num_mean([row.get("core_inference_time") for row in rows]),
        "median_core_inference_time": percentile([row.get("core_inference_time") for row in rows], 0.5),
        "p90_core_inference_time": percentile([row.get("core_inference_time") for row in rows], 0.9),
        "mean_conversation_wall_time": num_mean([row.get("conversation_wall_time") for row in rows]),
        "mean_prompt_tokens": num_mean([row.get("prompt_tokens") for row in rows]),
        "mean_response_tokens_generated": num_mean([row.get("response_tokens_generated") for row in rows]),
        "mean_response_tokens_tool": num_mean([row.get("response_tokens_tool") for row in rows]),
        "mean_response_tokens_total": num_mean([row.get("response_tokens_total") for row in rows]),
        "mean_total_tokens": num_mean(
            [
                (row.get("prompt_tokens") or 0) + (row.get("response_tokens_total") or 0)
                for row in rows
                if row.get("prompt_tokens") is not None and row.get("response_tokens_total") is not None
            ]
        ),
        "rows_with_tool_call_limit": len(rows_with_tool_limit),
        "rows_exhausting_tool_call_limit": len(exhausted_limit_rows),
        "tool_call_limit_exhaustion_rate": bool_mean([row["exhausted_tool_call_limit"] for row in rows_with_tool_limit]),
        "stuck_exhausted_same_page_tail_rows": len(stuck_exhausted_rows),
        "stuck_exhausted_same_page_tail_rate": bool_mean(
            [row["stuck_exhausted_same_page_tail"] for row in rows_with_tool_limit]
        ),
        "stuck_rate_among_tool_call_limit_exhausted": bool_mean(
            [row["stuck_exhausted_same_page_tail"] for row in exhausted_limit_rows]
        ),
        "mean_last_same_page_crop_run_length": num_mean([row.get("last_same_page_crop_run_length") for row in rows]),
        "mean_zoom_area_fraction_union": num_mean([row.get("zoom_area_fraction_union") for row in rows]),
        "median_zoom_area_fraction_union": percentile([row.get("zoom_area_fraction_union") for row in rows], 0.5),
        "p90_zoom_area_fraction_union": percentile([row.get("zoom_area_fraction_union") for row in rows], 0.9),
        "mean_zoom_area_fraction_sum": num_mean([row.get("zoom_area_fraction_sum") for row in rows]),
        "mean_crop_area_fraction_of_page": num_mean([row.get("mean_crop_area_fraction_of_page") for row in rows]),
        "mean_num_zoomed_source_pages": num_mean([row.get("num_zoomed_source_pages") for row in rows]),
        "rows_with_evidence": sum(1 for row in rows if row.get("has_evidence")),
        "rows_with_page_evidence": len(page_evidence_rows),
        "rows_with_box_evidence": len(box_evidence_rows),
        "rows_with_page_evidence_nonzero_crop": len(page_evidence_nonzero_crop_rows),
        "rows_with_box_evidence_nonzero_crop": len(box_evidence_nonzero_crop_rows),
        "evidence_page_hit_rate": bool_mean(
            [row.get("crop_hits_evidence_page") for row in page_evidence_nonzero_crop_rows]
        ),
        "evidence_coverage_hit_rate": bool_mean(
            [row.get("crop_covers_evidence_at_threshold") for row in box_evidence_nonzero_crop_rows]
        ),
        "mean_max_evidence_coverage": num_mean(
            [row.get("max_evidence_coverage") for row in box_evidence_nonzero_crop_rows]
        ),
        "mean_max_crop_evidence_iou": num_mean(
            [row.get("max_crop_evidence_iou") for row in box_evidence_nonzero_crop_rows]
        ),
        "evidence_box_pages_hit_by_crops": evidence_box_pages_hit_by_crops,
        "evidence_box_pages_hit_by_crops_with_coverage": evidence_box_pages_hit_by_crops_with_coverage,
        "evidence_box_page_coverage_hit_rate_on_cropped_pages": (
            evidence_box_pages_hit_by_crops_with_coverage / evidence_box_pages_hit_by_crops
            if evidence_box_pages_hit_by_crops
            else None
        ),
        "mean_fraction_cropped_evidence_box_pages_covered": num_mean(
            [row.get("fraction_cropped_evidence_box_pages_covered") for row in rows]
        ),
        "rows_with_any_evidence_page_hit": len(any_page_hit_rows),
        "rows_with_all_evidence_pages_hit": len(all_page_hit_rows),
        "rows_with_evidence_region_coverage_hit": len(region_hit_rows),
        "stopped_at_first_any_evidence_page_hit_fraction": bool_mean(
            [row.get("stopped_at_first_any_evidence_page_hit") for row in any_page_hit_rows]
        ),
        "stopped_at_first_all_evidence_pages_hit_fraction": bool_mean(
            [row.get("stopped_at_first_all_evidence_pages_hit") for row in all_page_hit_rows]
        ),
        "stopped_at_first_evidence_region_coverage_hit_fraction": bool_mean(
            [row.get("stopped_at_first_evidence_region_coverage_hit") for row in region_hit_rows]
        ),
        "mean_extra_crops_after_first_any_evidence_page_hit": num_mean(
            [row.get("extra_crops_after_first_any_evidence_page_hit") for row in any_page_hit_rows]
        ),
        "mean_extra_crops_after_first_all_evidence_pages_hit": num_mean(
            [row.get("extra_crops_after_first_all_evidence_pages_hit") for row in all_page_hit_rows]
        ),
        "mean_extra_crops_after_first_evidence_region_coverage_hit": num_mean(
            [row.get("extra_crops_after_first_evidence_region_coverage_hit") for row in region_hit_rows]
        ),
        "crops_hit_evidence_region_at_threshold": total_region_hit_crops,
        "box_evidence_total_crops": box_evidence_total_crops,
        "box_evidence_crops_hit_evidence_region_at_threshold": box_evidence_region_hit_crops,
        "box_evidence_crop_evidence_region_hit_rate": (
            box_evidence_region_hit_crops / box_evidence_total_crops if box_evidence_total_crops else None
        ),
        "box_evidence_total_crops_per_evidence_region_hit_crop": (
            box_evidence_total_crops / box_evidence_region_hit_crops if box_evidence_region_hit_crops else None
        ),
        "mean_max_crop_pair_iou": num_mean([row.get("max_crop_pair_iou") for row in rows]),
        "rows_with_redundant_pair_rate": bool_mean([row.get("redundant_crop_pairs", 0) > 0 for row in rows]),
        "mean_redundant_pairs": num_mean([row.get("redundant_crop_pairs") for row in rows]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rescale: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rescale_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_rescale[row["rescale"]].append(row)
        by_rescale_benchmark[row["rescale"]][row["benchmark"]].append(row)
    return {
        "total_rows": len(rows),
        "overall": group_summary(rows),
        "by_rescale": {key: group_summary(value) for key, value in sorted(by_rescale.items())},
        "by_rescale_benchmark": {
            rescale: {bench: group_summary(value) for bench, value in sorted(bench_rows.items())}
            for rescale, bench_rows in sorted(by_rescale_benchmark.items())
        },
    }


def fmt_pct(value: Any) -> str:
    value = as_float(value)
    return "" if value is None else f"{100 * value:.2f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    value = as_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def fmt_int(value: Any) -> str:
    value = as_int(value)
    return "" if value is None else f"{value:,}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# RL Ckpt700 Exported Conversation Trajectory Quality",
        "",
        f"Model: `{args.model_name or 'rl_ckpt700'}`.",
        "",
        f"Input root: `{args.export_root}`",
        "",
        f"Final-output evidence source: `{args.final_output_json}`" if args.final_output_json else "Final-output evidence source: not provided.",
        "",
        "Evidence-page metrics use recovered source evidence from the final-output JSON when available. Evidence-box metrics are only computed for rows with unambiguously recoverable boxes.",
        "",
        "Accuracy and aggregate rates are weighted by sample count, not macro-averaged over benchmarks.",
        "",
        "## Overall By Rescale",
        "",
        "| Rescale | Rows | Acc | Mean crops | Median crops | P90 crops | Zero crop | Evidence page hit | Box cov hit | Mean core time (s) | Mean prompt toks | Mean resp toks | Mean tool toks | Stuck rate | Redundant rows | Zoom union |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_int(data["rows"]),
                    fmt_pct(data["accuracy_reward_mean"]),
                    fmt_num(data["mean_crops"], 3),
                    fmt_num(data["median_crops"], 1),
                    fmt_num(data["p90_crops"], 1),
                    fmt_pct(data["zero_crop_fraction"]),
                    fmt_pct(data["evidence_page_hit_rate"]),
                    fmt_pct(data["evidence_coverage_hit_rate"]),
                    fmt_num(data["mean_core_inference_time"], 2),
                    fmt_num(data["mean_prompt_tokens"], 1),
                    fmt_num(data["mean_response_tokens_generated"], 1),
                    fmt_num(data["mean_response_tokens_tool"], 1),
                    fmt_pct(data["stuck_exhausted_same_page_tail_rate"]),
                    fmt_pct(data["rows_with_redundant_pair_rate"]),
                    fmt_pct(data["mean_zoom_area_fraction_union"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evidence Data Coverage",
            "",
            "| Rescale | Final-output matches | Rows with evidence | Page-evidence rows | Page-evidence rows with crops | Box-evidence rows | Box-evidence rows with crops |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_int(data["rows_with_final_output_match"]),
                    fmt_int(data["rows_with_evidence"]),
                    fmt_int(data["rows_with_page_evidence"]),
                    fmt_int(data["rows_with_page_evidence_nonzero_crop"]),
                    fmt_int(data["rows_with_box_evidence"]),
                    fmt_int(data["rows_with_box_evidence_nonzero_crop"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evidence Quality",
            "",
            "| Rescale | Page hit rate | Any page hits | All page hits | Box coverage hit >=0.5 | Mean max evidence coverage | Mean max crop/evidence IoU | Cropped evidence-box pages covered | Box-only crops / hit crop |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        covered = (
            f"{fmt_int(data['evidence_box_pages_hit_by_crops_with_coverage'])} / "
            f"{fmt_int(data['evidence_box_pages_hit_by_crops'])}"
            if data["evidence_box_pages_hit_by_crops"]
            else ""
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_pct(data["evidence_page_hit_rate"]),
                    fmt_int(data["rows_with_any_evidence_page_hit"]),
                    fmt_int(data["rows_with_all_evidence_pages_hit"]),
                    fmt_pct(data["evidence_coverage_hit_rate"]),
                    fmt_pct(data["mean_max_evidence_coverage"]),
                    fmt_pct(data["mean_max_crop_evidence_iou"]),
                    covered,
                    fmt_num(data["box_evidence_total_crops_per_evidence_region_hit_crop"], 4),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evidence-Based Stopping Efficiency",
            "",
            "| Rescale | Stop at first any-page hit | Extra crops after any-page hit | Stop at first all-page hit | Extra crops after all-page hit | Stop at first box-coverage hit | Extra crops after box-coverage hit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_pct(data["stopped_at_first_any_evidence_page_hit_fraction"]),
                    fmt_num(data["mean_extra_crops_after_first_any_evidence_page_hit"], 3),
                    fmt_pct(data["stopped_at_first_all_evidence_pages_hit_fraction"]),
                    fmt_num(data["mean_extra_crops_after_first_all_evidence_pages_hit"], 3),
                    fmt_pct(data["stopped_at_first_evidence_region_coverage_hit_fraction"]),
                    fmt_num(data["mean_extra_crops_after_first_evidence_region_coverage_hit"], 3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Crop Count Distribution",
            "",
            "| Rescale | 0 | 1 | 2 | 3 | 4 | >=5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        dist = data["crop_count_distribution"]
        rows = data["rows"] or 1
        cells = [f"{dist[key]:,} ({100 * dist[key] / rows:.1f}%)" for key in ("0", "1", "2", "3", "4", ">=5")]
        lines.append("| " + " | ".join([rescale, *cells]) + " |")

    lines.extend(
        [
            "",
            "## Timing And Token Lengths",
            "",
            "| Rescale | Mean core (s) | Median core (s) | P90 core (s) | Mean wall (s) | Mean prompt toks | Mean assistant toks | Mean tool toks | Mean response toks | Mean total toks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_num(data["mean_core_inference_time"], 2),
                    fmt_num(data["median_core_inference_time"], 2),
                    fmt_num(data["p90_core_inference_time"], 2),
                    fmt_num(data["mean_conversation_wall_time"], 2),
                    fmt_num(data["mean_prompt_tokens"], 1),
                    fmt_num(data["mean_response_tokens_generated"], 1),
                    fmt_num(data["mean_response_tokens_tool"], 1),
                    fmt_num(data["mean_response_tokens_total"], 1),
                    fmt_num(data["mean_total_tokens"], 1),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Stuck, Redundancy, And Zoom Area",
            "",
            "| Rescale | Tool-limit rows | Exhaust rows | Exhaust rate | Stuck rows | Stuck all-row rate | Stuck among exhausted | Redundant rows | Mean redundant pairs | Mean max crop-pair IoU | Zoom union mean | Zoom union median | Zoom union p90 | Zoom sum mean | Mean crop area/page |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_int(data["rows_with_tool_call_limit"]),
                    fmt_int(data["rows_exhausting_tool_call_limit"]),
                    fmt_pct(data["tool_call_limit_exhaustion_rate"]),
                    fmt_int(data["stuck_exhausted_same_page_tail_rows"]),
                    fmt_pct(data["stuck_exhausted_same_page_tail_rate"]),
                    fmt_pct(data["stuck_rate_among_tool_call_limit_exhausted"]),
                    fmt_pct(data["rows_with_redundant_pair_rate"]),
                    fmt_num(data["mean_redundant_pairs"], 4),
                    fmt_pct(data["mean_max_crop_pair_iou"]),
                    fmt_pct(data["mean_zoom_area_fraction_union"]),
                    fmt_pct(data["median_zoom_area_fraction_union"]),
                    fmt_pct(data["p90_zoom_area_fraction_union"]),
                    fmt_pct(data["mean_zoom_area_fraction_sum"]),
                    fmt_pct(data["mean_crop_area_fraction_of_page"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evidence Metadata Availability",
            "",
            "| Rescale | Rows with evidence | Rows with page evidence | Rows with box evidence |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for rescale, data in summary["by_rescale"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rescale,
                    fmt_int(data["rows_with_evidence"]),
                    fmt_int(data["rows_with_page_evidence"]),
                    fmt_int(data["rows_with_box_evidence"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## By Benchmark",
            "",
        ]
    )
    for rescale, bench_map in summary["by_rescale_benchmark"].items():
        lines.extend(
            [
                f"### {rescale}",
                "",
                "| Benchmark | Rows | Acc | Evidence rows | Page hit | Box cov hit | Mean crops | Mean valid tool calls | Mean core time (s) | Mean prompt toks | Mean resp toks | Mean tool toks | Stuck rate | Redundant rows | Zoom union |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for bench, data in bench_map.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        bench,
                        fmt_int(data["rows"]),
                        fmt_pct(data["accuracy_reward_mean"]),
                        fmt_int(data["rows_with_evidence"]),
                        fmt_pct(data["evidence_page_hit_rate"]),
                        fmt_pct(data["evidence_coverage_hit_rate"]),
                        fmt_num(data["mean_crops"], 3),
                        fmt_num(data["mean_valid_tool_calls"], 3),
                        fmt_num(data["mean_core_inference_time"], 2),
                        fmt_num(data["mean_prompt_tokens"], 1),
                        fmt_num(data["mean_response_tokens_generated"], 1),
                        fmt_num(data["mean_response_tokens_tool"], 1),
                        fmt_pct(data["stuck_exhausted_same_page_tail_rate"]),
                        fmt_pct(data["rows_with_redundant_pair_rate"]),
                        fmt_pct(data["mean_zoom_area_fraction_union"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    export_root = args.export_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(export_root.glob("rescale*/exported_conversations/*.json"))
    if not paths:
        raise FileNotFoundError(f"no exported conversations found under {export_root}")

    final_output_by_qid = None
    if args.final_output_json is not None:
        final_output_path = args.final_output_json.expanduser().resolve()
        final_output_by_qid = json.loads(final_output_path.read_text(encoding="utf-8"))
        if not isinstance(final_output_by_qid, dict):
            raise ValueError(f"final output JSON must be a dict keyed by question id: {final_output_path}")

    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.append(evaluate_record(path, json.load(handle), args, final_output_by_qid=final_output_by_qid))

    summary = summarize(rows)
    summary["settings"] = {
        "export_root": str(export_root),
        "model_name": args.model_name,
        "final_output_json": str(args.final_output_json.expanduser().resolve()) if args.final_output_json else None,
        "redundant_iou_threshold": args.redundant_iou_threshold,
        "evidence_coverage_threshold": args.evidence_coverage_threshold,
    }

    write_csv(output_root / "per_sample_metrics.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for rescale in sorted(summary["by_rescale"]):
        rescale_root = output_root / rescale
        rescale_root.mkdir(parents=True, exist_ok=True)
        rescale_rows = [row for row in rows if row["rescale"] == rescale]
        write_csv(rescale_root / "per_sample_metrics.csv", rescale_rows)
        rescale_summary = {
            "total_rows": len(rescale_rows),
            "overall": summary["by_rescale"][rescale],
            "by_benchmark": summary["by_rescale_benchmark"][rescale],
            "settings": summary["settings"],
        }
        (rescale_root / "summary.json").write_text(
            json.dumps(rescale_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    write_report(output_root / "full_quality_report.md", summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
