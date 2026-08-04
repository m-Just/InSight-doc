#!/usr/bin/env python3
"""Evaluate trajectory quality for InSight-doc SFT conversation data.

The SFT parquets are intentionally compact and do not retain question ids or
source evidence metadata. This script instead audits the raw exported
conversation JSONs linked under generated/*/{easy,medium}/raw*, then filters to
the rows that were retained by the SFT conversion using accuracy_reward == 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


GENERATED_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
OUTPUT_ROOT = Path("notes/generated/sft_trajectory_quality_20260713")
INSIGHT_DOC_DATA_ROOT = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/data")

O3_FINAL_OUTPUT_BY_PART = {
    "train_part1": INSIGHT_DOC_DATA_ROOT / "final_output_0426_selected_train_part1.json",
    "train_part2a": INSIGHT_DOC_DATA_ROOT / "final_output_0426_selected_train_part2a.json",
    "train_part2b": INSIGHT_DOC_DATA_ROOT / "final_output_0426_selected_train_part2b.json",
    "train_part2c": INSIGHT_DOC_DATA_ROOT / "final_output_0426_selected_train_part2c.json",
}
_O3_FINAL_OUTPUT_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class SourceSpec:
    group: str
    part: str
    difficulty: str

    @property
    def processed_dir(self) -> Path:
        name = "processed_drop_degenerate" if self.difficulty == "easy" else "processed_gpt5_nano_rewrite"
        return GENERATED_ROOT / self.group / self.part / self.difficulty / name

    @property
    def raw_dir(self) -> Path:
        name = "raw" if self.difficulty == "easy" else "raw_gpt5_nano_rewrite"
        return GENERATED_ROOT / self.group / self.part / self.difficulty / name

    @property
    def label(self) -> str:
        return f"{self.group}/{self.part}/{self.difficulty}"


DEFAULT_SOURCES = [
    SourceSpec(group, part, difficulty)
    for group, parts in (
        ("O3_data_0424", ["train_part1", "train_part2a", "train_part2b", "train_part2c", "dude_poster_unanswerable"]),
        ("arxiv", ["train_part1", "train_part2", "train_part3", "train_part4", "train_part5", "spanning_train_part1"]),
    )
    for part in parts
    for difficulty in ("easy", "medium")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--preset",
        choices=("current_22_files", "sft_ckpt1052_18_files"),
        default="current_22_files",
        help="Which TRAIN_FILES source set to audit.",
    )
    parser.add_argument(
        "--difficulty",
        choices=("all", "easy", "medium"),
        default="all",
        help="Optionally restrict the audited source set by difficulty.",
    )
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--redundant-iou-threshold", type=float, default=0.8)
    parser.add_argument("--evidence-coverage-threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-caption-evidence",
        action="store_true",
        help="Also treat caption boxes as evidence boxes. By default only visual boxes are used.",
    )
    parser.add_argument(
        "--evidence-coord-width",
        type=float,
        default=612.0,
        help="Reference page width for source evidence boxes before scaling to rendered image pixels.",
    )
    parser.add_argument(
        "--evidence-coord-height",
        type=float,
        default=792.0,
        help="Reference page height for source evidence boxes before scaling to rendered image pixels.",
    )
    parser.add_argument(
        "--no-scale-evidence-bboxes",
        action="store_true",
        help="Compare evidence boxes as-is instead of scaling them to rendered image pixel coordinates.",
    )
    return parser.parse_args()


def select_sources(preset: str) -> list[SourceSpec]:
    if preset == "current_22_files":
        return list(DEFAULT_SOURCES)
    if preset == "sft_ckpt1052_18_files":
        return [
            spec
            for spec in DEFAULT_SOURCES
            if not (spec.group == "arxiv" and spec.part in {"train_part4", "train_part5"})
        ]
    raise ValueError(f"unknown source preset: {preset}")


def accuracy_reward(record: dict[str, Any]) -> float | None:
    score = (record.get("reward") or {}).get("score") or {}
    value = score.get("accuracy_reward")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filtered_question_ids(spec: SourceSpec) -> set[str]:
    path = spec.processed_dir / "wrong_question_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def iter_retained_records(spec: SourceSpec, limit: int | None = None):
    count = 0
    filtered_ids = filtered_question_ids(spec)
    for path in sorted(spec.raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        reward = accuracy_reward(record)
        if reward != 1.0:
            continue
        extra_info = record.get("extra_info") or {}
        question_id = str(extra_info.get("question_id")) if extra_info.get("question_id") is not None else None
        if question_id is not None and question_id in filtered_ids:
            continue
        yield path, record
        count += 1
        if limit is not None and count >= limit:
            return


def parquet_rows(spec: SourceSpec) -> int | None:
    parquet = spec.processed_dir / "sft_data.parquet"
    if not parquet.exists():
        return None
    return pq.read_metadata(parquet).num_rows


def o3_final_output_record(spec: SourceSpec, question_id: Any) -> dict[str, Any] | None:
    if spec.group != "O3_data_0424" or question_id is None:
        return None
    source_path = O3_FINAL_OUTPUT_BY_PART.get(spec.part)
    if source_path is None:
        return None
    if spec.part not in _O3_FINAL_OUTPUT_CACHE:
        if not source_path.exists():
            _O3_FINAL_OUTPUT_CACHE[spec.part] = {}
        else:
            _O3_FINAL_OUTPUT_CACHE[spec.part] = json.loads(source_path.read_text(encoding="utf-8"))
    record = _O3_FINAL_OUTPUT_CACHE[spec.part].get(str(question_id))
    return record if isinstance(record, dict) else None


def is_unanswerable_record(record: dict[str, Any] | None, extra_info: dict[str, Any]) -> bool:
    values = []
    if record:
        values.extend(
            [
                record.get("question_type"),
                record.get("answer"),
                record.get("question_id"),
                record.get("qid"),
            ]
        )
    values.extend(
        [
            extra_info.get("question_type"),
            extra_info.get("answer"),
            extra_info.get("question_id"),
        ]
    )
    text = " ".join(str(value).lower() for value in values if value is not None)
    return (
        "not-answerable" in text
        or "unanswerable" in text
        or "cannot answer" in text
        or "cannot be answered" in text
    )


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_intersection(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    inter = bbox_intersection(a, b)
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def clamp_box_to_page(box: list[float], width: float, height: float) -> list[float] | None:
    clamped = [
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    ]
    return clamped if clamped[2] > clamped[0] and clamped[3] > clamped[1] else None


def rect_union_area(boxes: list[list[float]]) -> float:
    if not boxes:
        return 0.0
    xs = sorted({coord for box in boxes for coord in (box[0], box[2])})
    ys = sorted({coord for box in boxes for coord in (box[1], box[3])})
    area = 0.0
    for xi in range(len(xs) - 1):
        x1, x2 = xs[xi], xs[xi + 1]
        if x2 <= x1:
            continue
        for yi in range(len(ys) - 1):
            y1, y2 = ys[yi], ys[yi + 1]
            if y2 <= y1:
                continue
            if any(box[0] <= x1 and x2 <= box[2] and box[1] <= y1 and y2 <= box[3] for box in boxes):
                area += (x2 - x1) * (y2 - y1)
    return area


def as_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_page_numbers(value: Any) -> list[int]:
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    if isinstance(value, (list, tuple)):
        pages: list[int] = []
        for item in value:
            pages.extend(parse_page_numbers(item))
        return pages
    pages = []
    for part in re.split(r"[,\s]+", str(value).strip()):
        if not part:
            continue
        try:
            pages.append(int(part))
        except ValueError:
            continue
    return pages


def answer_bbox_entries(value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if all(key in item for key in ("left", "top", "width", "height", "page")):
                try:
                    left = float(item["left"])
                    top = float(item["top"])
                    width = float(item["width"])
                    height = float(item["height"])
                    page_id = int(item["page"])
                except (TypeError, ValueError):
                    pass
                else:
                    if width > 0 and height > 0:
                        entries.append(
                            {
                                "page_id": page_id,
                                "bbox": [left, top, left + width, top + height],
                            }
                        )
            for nested in item.values():
                walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)

    walk(value)
    return entries


def normalize_o3_question_pages(record: dict[str, Any], bbox_pages: set[int]) -> set[int]:
    try:
        total_pages = int(record.get("total_page_num"))
    except (TypeError, ValueError):
        return set()
    if total_pages <= 0:
        return set()

    raw_pages = parse_page_numbers(record.get("question_page_num"))
    if not raw_pages:
        return set()

    source = str(record.get("source") or "").lower()
    # DUDE source boxes are 0-based, and some DUDE question_page_num values are
    # also 0-based (including valid page 0). Other O3 construction sources use
    # the 1-based convention consumed by convert_final_output_to_manifest.py.
    if source == "dude" and (bbox_pages or any(page == 0 for page in raw_pages)):
        return {page for page in raw_pages if 0 <= page < total_pages}
    return {page - 1 for page in raw_pages if 1 <= page <= total_pages}


def image_ref_page_id(image_ref: dict[str, Any]) -> int | None:
    path = image_ref.get("path") or image_ref.get("uri")
    if path is None:
        return None
    try:
        return int(Path(str(path).replace("file://", "")).stem)
    except (TypeError, ValueError):
        return None


def input_image_page_maps(record: dict[str, Any]) -> tuple[dict[int, int], dict[int, tuple[float, float]]]:
    refs = record.get("image_references") or {}
    input_images = refs.get("input_images") if isinstance(refs, dict) else None
    if not isinstance(input_images, list):
        return {}, {}

    input_idx_to_page_id: dict[int, int] = {}
    page_id_to_size: dict[int, tuple[float, float]] = {}
    for list_idx, image_ref in enumerate(input_images):
        if not isinstance(image_ref, dict):
            continue
        try:
            input_idx = int(image_ref.get("original_img_idx", list_idx))
        except (TypeError, ValueError):
            input_idx = list_idx
        page_id = image_ref_page_id(image_ref)
        if page_id is None:
            page_id = input_idx
        input_idx_to_page_id[input_idx] = page_id

        size = image_ref.get("original_size")
        if isinstance(size, list) and len(size) == 2:
            try:
                width, height = float(size[0]), float(size[1])
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                page_id_to_size[page_id] = (width, height)

    return input_idx_to_page_id, page_id_to_size


def image_size_for_page(
    extra_info: dict[str, Any],
    page_id: int,
    page_image_sizes: dict[int, tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    if page_image_sizes is not None and page_id in page_image_sizes:
        return page_image_sizes[page_id]
    sizes = extra_info.get("image_ori_wh")
    if isinstance(sizes, str):
        try:
            sizes = json.loads(sizes)
        except json.JSONDecodeError:
            return None
    if not isinstance(sizes, list) or page_id < 0 or page_id >= len(sizes):
        return None
    size = sizes[page_id]
    if not isinstance(size, list) or len(size) != 2:
        return None
    try:
        width, height = float(size[0]), float(size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def scale_evidence_box(
    box: list[float],
    page_id: int,
    extra_info: dict[str, Any],
    args: argparse.Namespace,
    page_image_sizes: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[float], bool]:
    if args.no_scale_evidence_bboxes:
        return box, False
    image_size = image_size_for_page(extra_info, page_id, page_image_sizes=page_image_sizes)
    if image_size is None:
        return box, False

    coord_width = float(args.evidence_coord_width)
    coord_height = float(args.evidence_coord_height)
    if coord_width <= 0 or coord_height <= 0:
        return box, False

    image_width, image_height = image_size
    # Source arxiv annotations are in PDF/page coordinates (typically 612x792),
    # while generated crops are in rendered image pixels. Avoid double-scaling
    # boxes that are already close to pixel coordinates.
    if max(box[2], box[3]) > max(coord_width, coord_height) * 1.2:
        return box, False

    scaled = [
        box[0] * image_width / coord_width,
        box[1] * image_height / coord_height,
        box[2] * image_width / coord_width,
        box[3] * image_height / coord_height,
    ]
    return scaled, True


def evidence_boxes(
    extra_info: dict[str, Any],
    include_caption: bool,
    args: argparse.Namespace,
    page_image_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    details = extra_info.get("question_involved_visual_details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = None
    if not isinstance(details, list):
        return []

    out: list[dict[str, Any]] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        visual = item.get("visual")
        if isinstance(visual, dict):
            box = as_box(visual.get("bbox"))
            page_id = visual.get("page_id")
            if box is not None and page_id is not None:
                page_id = int(page_id)
                scaled_box, scaled = scale_evidence_box(
                    box,
                    page_id,
                    extra_info,
                    args,
                    page_image_sizes=page_image_sizes,
                )
                out.append(
                    {
                        "page_id": page_id,
                        "bbox": scaled_box,
                        "raw_bbox": box,
                        "bbox_scaled_to_image": scaled,
                        "kind": "visual",
                    }
                )
        if include_caption:
            captions = item.get("caption")
            if isinstance(captions, dict):
                captions = [captions]
            if isinstance(captions, list):
                for caption in captions:
                    if not isinstance(caption, dict):
                        continue
                    box = as_box(caption.get("bbox"))
                    page_id = caption.get("page_id")
                    if box is not None and page_id is not None:
                        page_id = int(page_id)
                        scaled_box, scaled = scale_evidence_box(
                            box,
                            page_id,
                            extra_info,
                            args,
                            page_image_sizes=page_image_sizes,
                        )
                        out.append(
                            {
                                "page_id": page_id,
                                "bbox": scaled_box,
                                "raw_bbox": box,
                                "bbox_scaled_to_image": scaled,
                                "kind": "caption",
                            }
                        )
    return out


def o3_evidence(
    source_record: dict[str, Any] | None,
    is_unanswerable: bool,
) -> list[dict[str, Any]]:
    if source_record is None or is_unanswerable:
        return []

    box_entries = answer_bbox_entries(source_record.get("answers_page_bounding_boxes"))
    bbox_pages = {entry["page_id"] for entry in box_entries}
    page_ids = normalize_o3_question_pages(source_record, bbox_pages) | bbox_pages
    evidence: list[dict[str, Any]] = [
        {
            "page_id": page_id,
            "bbox": None,
            "raw_bbox": None,
            "bbox_scaled_to_image": False,
            "kind": "o3_page",
        }
        for page_id in sorted(page_ids)
    ]
    evidence.extend(
        {
            "page_id": entry["page_id"],
            "bbox": entry["bbox"],
            "raw_bbox": entry["bbox"],
            "bbox_scaled_to_image": False,
            "kind": "o3_answer_bbox",
        }
        for entry in box_entries
    )
    return evidence


def crop_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    refs = record.get("image_references") or {}
    presented = refs.get("presented_images") if isinstance(refs, dict) else None
    if not isinstance(presented, list):
        return []
    input_idx_to_page_id, _ = input_image_page_maps(record)
    out: list[dict[str, Any]] = []
    for item in presented:
        if not isinstance(item, dict) or item.get("kind") != "region_crop":
            continue
        box = as_box(item.get("bbox_on_original"))
        page = item.get("source_original_img_idx")
        if box is None or page is None:
            continue
        page = int(page)
        out.append(
            {
                "presented_img_idx": item.get("presented_img_idx"),
                "source_original_img_idx": page,
                "source_original_page_id": input_idx_to_page_id.get(page, page),
                "bbox_on_original": box,
                "parent_presented_img_idx": item.get("parent_presented_img_idx"),
                "region_description": item.get("region_description"),
            }
        )
    return out


def count_message_types(record: dict[str, Any]) -> Counter:
    counter: Counter = Counter()
    for message in record.get("conversation") or []:
        if isinstance(message, dict):
            counter[str(message.get("type") or "<missing>")] += 1
    return counter


def tool_call_limit(record: dict[str, Any]) -> int | None:
    parameters = record.get("parameters") or {}
    for section_name in ("loop", "request"):
        section = parameters.get(section_name)
        if not isinstance(section, dict):
            continue
        value = section.get("max_tool_calls")
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def last_same_page_crop_run(crops: list[dict[str, Any]]) -> tuple[int, int | None]:
    if not crops:
        return 0, None
    last_page_id = crops[-1].get("source_original_page_id", crops[-1]["source_original_img_idx"])
    run_length = 0
    for crop in reversed(crops):
        page_id = crop.get("source_original_page_id", crop["source_original_img_idx"])
        if page_id != last_page_id:
            break
        run_length += 1
    return run_length, last_page_id


def stuck_stats(
    record: dict[str, Any],
    crops: list[dict[str, Any]],
    num_tool_call_messages: int,
) -> dict[str, Any]:
    limit = tool_call_limit(record)
    last_run_length, last_page_id = last_same_page_crop_run(crops)
    exhausted = None
    if limit is not None:
        exhausted = num_tool_call_messages >= limit
    return {
        "tool_call_limit": limit,
        "exhausted_tool_call_limit": exhausted,
        "last_same_page_crop_run_length": last_run_length,
        "last_crop_page_id": last_page_id,
        "stuck_exhausted_same_page_tail": (exhausted and last_run_length >= 2) if exhausted is not None else None,
    }


def redundancy_stats(crops: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ious: list[float] = []
    redundant_pairs = 0
    same_source_img_pairs = 0
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            source_i = crops[i].get("parent_presented_img_idx")
            source_j = crops[j].get("parent_presented_img_idx")
            if source_i is None:
                source_i = crops[i]["source_original_img_idx"]
            if source_j is None:
                source_j = crops[j]["source_original_img_idx"]
            if source_i != source_j:
                continue
            same_source_img_pairs += 1
            iou = bbox_iou(crops[i]["bbox_on_original"], crops[j]["bbox_on_original"])
            ious.append(iou)
            if iou >= threshold:
                redundant_pairs += 1
    return {
        "same_source_img_crop_pairs": same_source_img_pairs,
        "same_page_crop_pairs": same_source_img_pairs,
        "redundant_crop_pairs": redundant_pairs,
        "max_crop_pair_iou": max(ious) if ious else None,
        "mean_crop_pair_iou": sum(ious) / len(ious) if ious else None,
    }


def evidence_match_stats(crops: list[dict[str, Any]], evidence: list[dict[str, Any]], coverage_threshold: float) -> dict[str, Any]:
    evidence_pages = {item["page_id"] for item in evidence}
    evidence_with_boxes = [item for item in evidence if item.get("bbox") is not None]
    evidence_box_pages = {item["page_id"] for item in evidence_with_boxes}
    crop_pages = {
        crop.get("source_original_page_id", crop["source_original_img_idx"])
        for crop in crops
    }
    page_hit = bool(evidence_pages and crop_pages.intersection(evidence_pages))
    cropped_evidence_box_pages = evidence_box_pages.intersection(crop_pages)
    covered_cropped_evidence_box_pages: set[int] = set()
    crops_hit_evidence_region_at_threshold: set[int] = set()
    crops_on_evidence_box_pages = 0

    max_iou = 0.0
    max_evidence_coverage = 0.0
    max_crop_precision = 0.0
    for crop_idx, crop in enumerate(crops):
        crop_box = crop["bbox_on_original"]
        crop_area = bbox_area(crop_box)
        crop_page_id = crop.get("source_original_page_id", crop["source_original_img_idx"])
        if crop_page_id in evidence_box_pages:
            crops_on_evidence_box_pages += 1
        for ev in evidence_with_boxes:
            if crop_page_id != ev["page_id"]:
                continue
            ev_box = ev["bbox"]
            inter = bbox_intersection(crop_box, ev_box)
            ev_area = bbox_area(ev_box)
            max_iou = max(max_iou, bbox_iou(crop_box, ev_box))
            coverage = inter / ev_area if ev_area > 0 else 0.0
            max_evidence_coverage = max(max_evidence_coverage, coverage)
            max_crop_precision = max(max_crop_precision, inter / crop_area if crop_area > 0 else 0.0)
            if coverage >= coverage_threshold:
                covered_cropped_evidence_box_pages.add(crop_page_id)
                crops_hit_evidence_region_at_threshold.add(crop_idx)

    return {
        "has_evidence": bool(evidence),
        "has_evidence_pages": bool(evidence_pages),
        "has_evidence_boxes": bool(evidence_with_boxes),
        "num_evidence_pages": len(evidence_pages),
        "num_evidence_boxes": len(evidence_with_boxes),
        "num_evidence_box_pages": len(evidence_box_pages),
        "num_cropped_evidence_box_pages": len(cropped_evidence_box_pages),
        "num_cropped_evidence_box_pages_covered": len(covered_cropped_evidence_box_pages),
        "num_crops_on_evidence_box_pages": crops_on_evidence_box_pages,
        "num_crops_hit_evidence_region_at_threshold": len(crops_hit_evidence_region_at_threshold),
        "fraction_cropped_evidence_box_pages_covered": (
            len(covered_cropped_evidence_box_pages) / len(cropped_evidence_box_pages)
            if cropped_evidence_box_pages
            else None
        ),
        "num_scaled_evidence_boxes": sum(1 for item in evidence_with_boxes if item.get("bbox_scaled_to_image")),
        "crop_hits_evidence_page": page_hit if evidence else None,
        "max_crop_evidence_iou": max_iou if evidence_with_boxes else None,
        "max_evidence_coverage": max_evidence_coverage if evidence_with_boxes else None,
        "max_crop_precision_vs_evidence": max_crop_precision if evidence_with_boxes else None,
        "crop_covers_evidence_at_threshold": (max_evidence_coverage >= coverage_threshold) if evidence_with_boxes else None,
    }


def zoom_area_stats(
    crops: list[dict[str, Any]],
    page_image_sizes: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    if not crops:
        return {
            "zoom_area_fraction_union": 0.0,
            "zoom_area_fraction_sum": 0.0,
            "mean_crop_area_fraction_of_page": None,
            "max_crop_area_fraction_of_page": None,
            "num_zoomed_source_pages": 0,
            "num_crops_missing_page_size": 0,
        }

    boxes_by_page: dict[int, list[list[float]]] = defaultdict(list)
    page_areas: dict[int, float] = {}
    crop_area_fractions: list[float] = []
    missing_page_size = 0

    for crop in crops:
        page_id = crop.get("source_original_page_id", crop["source_original_img_idx"])
        size = page_image_sizes.get(page_id)
        if size is None:
            missing_page_size += 1
            continue
        width, height = size
        page_area = width * height
        if page_area <= 0:
            missing_page_size += 1
            continue
        box = clamp_box_to_page(crop["bbox_on_original"], width, height)
        if box is None:
            continue
        boxes_by_page[page_id].append(box)
        page_areas[page_id] = page_area
        crop_area_fractions.append(bbox_area(box) / page_area)

    if not boxes_by_page:
        return {
            "zoom_area_fraction_union": None,
            "zoom_area_fraction_sum": None,
            "mean_crop_area_fraction_of_page": None,
            "max_crop_area_fraction_of_page": None,
            "num_zoomed_source_pages": 0,
            "num_crops_missing_page_size": missing_page_size,
        }

    denominator = sum(page_areas[page_id] for page_id in boxes_by_page)
    union_area = sum(rect_union_area(page_boxes) for page_boxes in boxes_by_page.values())
    sum_area = sum(bbox_area(box) for page_boxes in boxes_by_page.values() for box in page_boxes)
    return {
        "zoom_area_fraction_union": union_area / denominator if denominator > 0 else None,
        "zoom_area_fraction_sum": sum_area / denominator if denominator > 0 else None,
        "mean_crop_area_fraction_of_page": sum(crop_area_fractions) / len(crop_area_fractions)
        if crop_area_fractions
        else None,
        "max_crop_area_fraction_of_page": max(crop_area_fractions) if crop_area_fractions else None,
        "num_zoomed_source_pages": len(boxes_by_page),
        "num_crops_missing_page_size": missing_page_size,
    }


def evidence_step_stats(
    crops: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    coverage_threshold: float,
) -> dict[str, Any]:
    evidence_pages = {item["page_id"] for item in evidence}
    evidence_with_boxes = [item for item in evidence if item.get("bbox") is not None]

    first_any_page_hit_step: int | None = None
    first_all_pages_hit_step: int | None = None
    first_region_coverage_hit_step: int | None = None
    seen_pages: set[int] = set()

    for step, crop in enumerate(crops, start=1):
        crop_page_id = crop.get("source_original_page_id", crop["source_original_img_idx"])
        seen_pages.add(crop_page_id)
        if evidence_pages and crop_page_id in evidence_pages and first_any_page_hit_step is None:
            first_any_page_hit_step = step
        if evidence_pages and evidence_pages.issubset(seen_pages) and first_all_pages_hit_step is None:
            first_all_pages_hit_step = step

        if evidence_with_boxes and first_region_coverage_hit_step is None:
            crop_box = crop["bbox_on_original"]
            for ev in evidence_with_boxes:
                if crop_page_id != ev["page_id"]:
                    continue
                ev_area = bbox_area(ev["bbox"])
                coverage = bbox_intersection(crop_box, ev["bbox"]) / ev_area if ev_area > 0 else 0.0
                if coverage >= coverage_threshold:
                    first_region_coverage_hit_step = step
                    break

    num_crops = len(crops)
    return {
        "first_any_evidence_page_hit_step": first_any_page_hit_step,
        "first_all_evidence_pages_hit_step": first_all_pages_hit_step,
        "first_evidence_region_coverage_hit_step": first_region_coverage_hit_step,
        "stopped_at_first_any_evidence_page_hit": (num_crops == first_any_page_hit_step)
        if first_any_page_hit_step is not None
        else None,
        "stopped_at_first_all_evidence_pages_hit": (num_crops == first_all_pages_hit_step)
        if first_all_pages_hit_step is not None
        else None,
        "stopped_at_first_evidence_region_coverage_hit": (num_crops == first_region_coverage_hit_step)
        if first_region_coverage_hit_step is not None
        else None,
        "extra_crops_after_first_any_evidence_page_hit": (num_crops - first_any_page_hit_step)
        if first_any_page_hit_step is not None
        else None,
        "extra_crops_after_first_all_evidence_pages_hit": (num_crops - first_all_pages_hit_step)
        if first_all_pages_hit_step is not None
        else None,
        "extra_crops_after_first_evidence_region_coverage_hit": (num_crops - first_region_coverage_hit_step)
        if first_region_coverage_hit_step is not None
        else None,
    }


def evaluate_record(spec: SourceSpec, path: Path, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    extra = record.get("extra_info") or {}
    crops = crop_entries(record)
    source_record = o3_final_output_record(spec, extra.get("question_id"))
    is_unanswerable = is_unanswerable_record(source_record, extra)
    _, page_image_sizes = input_image_page_maps(record)
    evidence: list[dict[str, Any]] = []
    if not is_unanswerable:
        evidence.extend(
            evidence_boxes(
                extra,
                include_caption=args.include_caption_evidence,
                args=args,
                page_image_sizes=page_image_sizes,
            )
        )
        evidence.extend(o3_evidence(source_record, is_unanswerable))
    msg_counts = count_message_types(record)
    row = {
        "source": spec.label,
        "group": spec.group,
        "part": spec.part,
        "difficulty": spec.difficulty,
        "json_path": str(path),
        "question_id": extra.get("question_id"),
        "document_id": extra.get("document_id"),
        "subset": extra.get("subset"),
        "is_unanswerable": is_unanswerable,
        "has_o3_final_output_match": source_record is not None,
        "accuracy_reward": accuracy_reward(record),
        "num_crops": len(crops),
        "num_tool_call_messages": msg_counts.get("tool_call", 0),
        "num_tool_result_messages": msg_counts.get("tool_result", 0),
        "num_tool_result_fail_hints": msg_counts.get("tool_result_fail_hint", 0),
        "num_format_repair_hints": msg_counts.get("format_repair_hint", 0),
    }
    row.update(evidence_match_stats(crops, evidence, args.evidence_coverage_threshold))
    row.update(zoom_area_stats(crops, page_image_sizes))
    row.update(evidence_step_stats(crops, evidence, args.evidence_coverage_threshold))
    row.update(stuck_stats(record, crops, msg_counts.get("tool_call", 0)))
    row.update(redundancy_stats(crops, args.redundant_iou_threshold))
    return row


def bool_mean(values: list[Any]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def num_mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    return sum(clean) / len(clean) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(rows: list[dict[str, Any]], specs: list[SourceSpec]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        by_difficulty[row["difficulty"]].append(row)
        by_subset[str(row.get("subset"))].append(row)

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        crop_counts = [float(row["num_crops"]) for row in group_rows]
        zoom_union_fractions = [
            float(row["zoom_area_fraction_union"])
            for row in group_rows
            if row.get("zoom_area_fraction_union") is not None
        ]
        zoom_sum_fractions = [
            float(row["zoom_area_fraction_sum"])
            for row in group_rows
            if row.get("zoom_area_fraction_sum") is not None
        ]
        has_evidence_rows = [row for row in group_rows if row["has_evidence"]]
        page_evidence_rows = [row for row in group_rows if row.get("has_evidence_pages")]
        box_evidence_rows = [row for row in group_rows if row.get("has_evidence_boxes")]
        page_evidence_nonzero_crop_rows = [row for row in page_evidence_rows if row.get("num_crops", 0) > 0]
        box_evidence_nonzero_crop_rows = [row for row in box_evidence_rows if row.get("num_crops", 0) > 0]
        evidence_box_pages_hit_by_crops = sum(int(row.get("num_cropped_evidence_box_pages") or 0) for row in group_rows)
        evidence_box_pages_hit_by_crops_with_coverage = sum(
            int(row.get("num_cropped_evidence_box_pages_covered") or 0) for row in group_rows
        )
        total_crops = sum(int(row.get("num_crops") or 0) for row in group_rows)
        crops_on_evidence_box_pages = sum(int(row.get("num_crops_on_evidence_box_pages") or 0) for row in group_rows)
        crops_hit_evidence_region_at_threshold = sum(
            int(row.get("num_crops_hit_evidence_region_at_threshold") or 0) for row in group_rows
        )
        box_evidence_total_crops = sum(int(row.get("num_crops") or 0) for row in box_evidence_rows)
        box_evidence_crops_hit_region_at_threshold = sum(
            int(row.get("num_crops_hit_evidence_region_at_threshold") or 0) for row in box_evidence_rows
        )
        any_page_hit_rows = [row for row in page_evidence_rows if row.get("first_any_evidence_page_hit_step") is not None]
        all_page_hit_rows = [row for row in page_evidence_rows if row.get("first_all_evidence_pages_hit_step") is not None]
        region_hit_rows = [
            row for row in box_evidence_rows if row.get("first_evidence_region_coverage_hit_step") is not None
        ]
        unanswerable_rows = [row for row in group_rows if row.get("is_unanswerable")]
        answerable_rows = [row for row in group_rows if not row.get("is_unanswerable")]
        answerable_with_evidence = [row for row in answerable_rows if row["has_evidence"]]
        answerable_without_evidence = [row for row in answerable_rows if not row["has_evidence"]]
        answerable_denominator = len(answerable_rows) or 1
        rows_with_tool_limit = [row for row in group_rows if row.get("exhausted_tool_call_limit") is not None]
        exhausted_limit_rows = [row for row in rows_with_tool_limit if row.get("exhausted_tool_call_limit")]
        stuck_exhausted_rows = [row for row in rows_with_tool_limit if row.get("stuck_exhausted_same_page_tail")]
        return {
            "rows": len(group_rows),
            "answerable_rows": len(answerable_rows),
            "unanswerable_rows_excluded_from_evidence_metrics": len(unanswerable_rows),
            "rows_with_evidence": len(has_evidence_rows),
            "rows_with_page_evidence": len(page_evidence_rows),
            "rows_with_box_evidence": len(box_evidence_rows),
            "rows_with_page_evidence_nonzero_crop": len(page_evidence_nonzero_crop_rows),
            "rows_with_box_evidence_nonzero_crop": len(box_evidence_nonzero_crop_rows),
            "answerable_rows_with_evidence": len(answerable_with_evidence),
            "answerable_rows_without_evidence": len(answerable_without_evidence),
            "answerable_rows_without_evidence_fraction": len(answerable_without_evidence) / answerable_denominator,
            "answerable_rows_with_evidence_fraction": len(answerable_with_evidence) / answerable_denominator,
            "mean_crops": num_mean([row["num_crops"] for row in group_rows]),
            "total_crops": total_crops,
            "crops_on_evidence_box_pages": crops_on_evidence_box_pages,
            "crops_hit_evidence_region_at_threshold": crops_hit_evidence_region_at_threshold,
            "crop_evidence_region_hit_rate": (
                crops_hit_evidence_region_at_threshold / total_crops if total_crops else None
            ),
            "total_crops_per_evidence_region_hit_crop": (
                total_crops / crops_hit_evidence_region_at_threshold
                if crops_hit_evidence_region_at_threshold
                else None
            ),
            "box_evidence_total_crops": box_evidence_total_crops,
            "box_evidence_crops_hit_evidence_region_at_threshold": box_evidence_crops_hit_region_at_threshold,
            "box_evidence_crop_evidence_region_hit_rate": (
                box_evidence_crops_hit_region_at_threshold / box_evidence_total_crops
                if box_evidence_total_crops
                else None
            ),
            "box_evidence_total_crops_per_evidence_region_hit_crop": (
                box_evidence_total_crops / box_evidence_crops_hit_region_at_threshold
                if box_evidence_crops_hit_region_at_threshold
                else None
            ),
            "median_crops": percentile(crop_counts, 0.5),
            "p90_crops": percentile(crop_counts, 0.9),
            "zero_crop_fraction": bool_mean([row["num_crops"] == 0 for row in group_rows]),
            "rows_with_tool_call_limit": len(rows_with_tool_limit),
            "rows_exhausting_tool_call_limit": len(exhausted_limit_rows),
            "tool_call_limit_exhaustion_rate": bool_mean(
                [row["exhausted_tool_call_limit"] for row in rows_with_tool_limit]
            ),
            "stuck_exhausted_same_page_tail_rows": len(stuck_exhausted_rows),
            "stuck_exhausted_same_page_tail_rate": bool_mean(
                [row["stuck_exhausted_same_page_tail"] for row in rows_with_tool_limit]
            ),
            "stuck_rate_among_tool_call_limit_exhausted": bool_mean(
                [row["stuck_exhausted_same_page_tail"] for row in exhausted_limit_rows]
            ),
            "mean_last_same_page_crop_run_length": num_mean(
                [row["last_same_page_crop_run_length"] for row in group_rows]
            ),
            "mean_zoom_area_fraction_union": num_mean([row["zoom_area_fraction_union"] for row in group_rows]),
            "median_zoom_area_fraction_union": percentile(zoom_union_fractions, 0.5),
            "p90_zoom_area_fraction_union": percentile(zoom_union_fractions, 0.9),
            "mean_zoom_area_fraction_sum": num_mean([row["zoom_area_fraction_sum"] for row in group_rows]),
            "median_zoom_area_fraction_sum": percentile(zoom_sum_fractions, 0.5),
            "p90_zoom_area_fraction_sum": percentile(zoom_sum_fractions, 0.9),
            "mean_crop_area_fraction_of_page": num_mean([row["mean_crop_area_fraction_of_page"] for row in group_rows]),
            "mean_num_zoomed_source_pages": num_mean([row["num_zoomed_source_pages"] for row in group_rows]),
            "evidence_page_hit_rate": bool_mean(
                [row["crop_hits_evidence_page"] for row in page_evidence_nonzero_crop_rows]
            ),
            "evidence_coverage_hit_rate": bool_mean(
                [row["crop_covers_evidence_at_threshold"] for row in box_evidence_nonzero_crop_rows]
            ),
            "mean_max_evidence_coverage": num_mean(
                [row["max_evidence_coverage"] for row in box_evidence_nonzero_crop_rows]
            ),
            "mean_max_crop_evidence_iou": num_mean(
                [row["max_crop_evidence_iou"] for row in box_evidence_nonzero_crop_rows]
            ),
            "evidence_box_pages_hit_by_crops": evidence_box_pages_hit_by_crops,
            "evidence_box_pages_hit_by_crops_with_coverage": evidence_box_pages_hit_by_crops_with_coverage,
            "evidence_box_page_coverage_hit_rate_on_cropped_pages": (
                evidence_box_pages_hit_by_crops_with_coverage / evidence_box_pages_hit_by_crops
                if evidence_box_pages_hit_by_crops
                else None
            ),
            "mean_fraction_cropped_evidence_box_pages_covered": num_mean(
                [row["fraction_cropped_evidence_box_pages_covered"] for row in group_rows]
            ),
            "rows_with_any_evidence_page_hit": len(any_page_hit_rows),
            "rows_with_all_evidence_pages_hit": len(all_page_hit_rows),
            "rows_with_evidence_region_coverage_hit": len(region_hit_rows),
            "stopped_at_first_any_evidence_page_hit_fraction": bool_mean(
                [row["stopped_at_first_any_evidence_page_hit"] for row in any_page_hit_rows]
            ),
            "stopped_at_first_all_evidence_pages_hit_fraction": bool_mean(
                [row["stopped_at_first_all_evidence_pages_hit"] for row in all_page_hit_rows]
            ),
            "stopped_at_first_evidence_region_coverage_hit_fraction": bool_mean(
                [row["stopped_at_first_evidence_region_coverage_hit"] for row in region_hit_rows]
            ),
            "mean_extra_crops_after_first_any_evidence_page_hit": num_mean(
                [row["extra_crops_after_first_any_evidence_page_hit"] for row in any_page_hit_rows]
            ),
            "mean_extra_crops_after_first_all_evidence_pages_hit": num_mean(
                [row["extra_crops_after_first_all_evidence_pages_hit"] for row in all_page_hit_rows]
            ),
            "mean_extra_crops_after_first_evidence_region_coverage_hit": num_mean(
                [row["extra_crops_after_first_evidence_region_coverage_hit"] for row in region_hit_rows]
            ),
            "mean_max_crop_pair_iou": num_mean([row["max_crop_pair_iou"] for row in group_rows]),
            "rows_with_redundant_pair": bool_mean([row["redundant_crop_pairs"] > 0 for row in group_rows]),
            "mean_redundant_pairs": num_mean([row["redundant_crop_pairs"] for row in group_rows]),
        }

    validation = {}
    for spec in specs:
        expected = parquet_rows(spec)
        actual = sum(1 for row in rows if row["source"] == spec.label)
        validation[spec.label] = {
            "processed_parquet_rows": expected,
            "retained_raw_accuracy1_rows": actual,
            "matches_parquet_rows": expected == actual if expected is not None else None,
        }

    return {
        "total_rows": len(rows),
        "validation": validation,
        "overall": group_summary(rows),
        "by_difficulty": {key: group_summary(value) for key, value in sorted(by_difficulty.items())},
        "by_subset": {key: group_summary(value) for key, value in sorted(by_subset.items())},
        "by_source": {key: group_summary(value) for key, value in sorted(by_source.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = select_sources(args.preset)
    if args.difficulty != "all":
        specs = [spec for spec in specs if spec.difficulty == args.difficulty]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.raw_dir.exists():
            raise FileNotFoundError(f"raw dir not found for {spec.label}: {spec.raw_dir}")
        for path, record in iter_retained_records(spec, limit=args.limit_per_source):
            rows.append(evaluate_record(spec, path, record, args))

    csv_path = output_root / "per_sample_metrics.csv"
    summary_path = output_root / "summary.json"
    write_csv(csv_path, rows)
    summary = summarize(rows, specs)
    summary["output_csv"] = str(csv_path)
    summary["settings"] = {
        "preset": args.preset,
        "difficulty": args.difficulty,
        "limit_per_source": args.limit_per_source,
        "redundant_iou_threshold": args.redundant_iou_threshold,
        "evidence_coverage_threshold": args.evidence_coverage_threshold,
        "include_caption_evidence": args.include_caption_evidence,
        "scale_evidence_bboxes": not args.no_scale_evidence_bboxes,
        "evidence_coord_width": args.evidence_coord_width,
        "evidence_coord_height": args.evidence_coord_height,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
