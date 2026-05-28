#!/usr/bin/env python3
"""Behavior analysis for exported InSight conversations at a train/val step."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.vreasoner_v2_conversation_export import restore_presented_images  # noqa: E402


PROMPT_VERSION = "conversation_behavior_judge_v1"
TOOL_PRIORITY = [
    "invalid_crop",
    "repetitive_crop",
    "deep_zoom_in",
    "bad_crop",
    "error_recovery",
    "expressing_uncertainty",
    "none",
]
ANSWER_PRIORITY = ["ungrounded_answer", "none"]
GOOD_LABELS = {"deep_zoom_in", "error_recovery", "expressing_uncertainty"}
BAD_LABELS = {"invalid_crop", "repetitive_crop", "bad_crop", "ungrounded_answer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze behavior labels in exported conversations.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--subset-key", default="subset")
    parser.add_argument("--repetitive-iou-threshold", type=float, default=0.8)
    parser.add_argument("--image-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-judge-images", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--messages-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
    )
    parser.add_argument("--ensure-api-logger", action="store_true", default=True)
    parser.add_argument("--no-ensure-api-logger", dest="ensure_api_logger", action="store_false")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            result = item.get("result")
            if isinstance(key, str) and isinstance(result, dict):
                cache[key] = result
    return cache


def collect_paths(export_dir: Path, global_step: int, split: str) -> tuple[list[Path], str]:
    index_dir = export_dir / "index" / f"global_step_{global_step}" / split
    paths: list[Path] = []
    seen: set[Path] = set()
    if index_dir.exists():
        for index_file in sorted(index_dir.glob("worker_*.jsonl")):
            with index_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw_path = item.get("path")
                    if not isinstance(raw_path, str):
                        continue
                    path = Path(raw_path)
                    if path not in seen:
                        paths.append(path)
                        seen.add(path)
        if paths:
            return paths, "index"

    for path in sorted(export_dir.glob("*.json")):
        record = load_json(path)
        if record is None:
            continue
        job = record.get("job")
        if not isinstance(job, dict):
            continue
        if job.get("global_step") == global_step and job.get("split") == split:
            paths.append(path)
    return paths, "flat_scan"


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        pieces = []
        for key in ("think", "answer", "text", "tool_call", "hint", "error_message"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
            elif isinstance(value, (dict, list)):
                pieces.append(json.dumps(value, ensure_ascii=False))
        return "\n\n".join(pieces).strip()
    return ""


def label_polarity(label: str) -> str:
    if label in GOOD_LABELS:
        return "good"
    if label in BAD_LABELS:
        return "bad"
    return "neutral"


def initial_question(record: dict[str, Any]) -> str:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"].strip()
    for message in record.get("conversation") or []:
        if isinstance(message, dict) and message.get("role") == "user" and message.get("type") == "query":
            text = content_text(message.get("content"))
            if text:
                return text
    return ""


def get_subset(record: dict[str, Any], subset_key: str) -> str:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        return "unknown"
    for key in (subset_key, "data_source", "source", "dataset"):
        value = extra_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def parse_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, dict):
        return None
    raw_tool_call = content.get("tool_call")
    if isinstance(raw_tool_call, dict):
        return raw_tool_call
    if not isinstance(raw_tool_call, str):
        return None
    try:
        return json.loads(raw_tool_call)
    except json.JSONDecodeError:
        return None


def tool_args(tool_call: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {}
    args = tool_call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def next_user_message(conversation: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for message in conversation[index + 1 :]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return message
        if message.get("role") == "assistant":
            return None
    return None


def tool_result_indices(message: dict[str, Any] | None) -> list[int]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, dict):
        indices = content.get("presented_img_indices")
        if isinstance(indices, list):
            return [int(v) for v in indices if isinstance(v, int)]
    return []


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def image_to_data_url(image: Image.Image, max_pixels: int) -> str:
    img = image.convert("RGB")
    if max_pixels > 0 and img.width * img.height > max_pixels:
        scale = (max_pixels / float(img.width * img.height)) ** 0.5
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def restored_image_map(record: dict[str, Any]) -> dict[int, Image.Image]:
    restored = restore_presented_images(record)
    out = {}
    for item in restored:
        idx = item.get("presented_img_idx")
        image = item.get("image")
        if isinstance(idx, int) and isinstance(image, Image.Image):
            out[idx] = image
    return out


def presented_refs(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    refs = record.get("image_references", {}).get("presented_images", [])
    out = {}
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and isinstance(ref.get("presented_img_idx"), int):
                out[ref["presented_img_idx"]] = ref
    return out


def cache_key(kind: str, payload: dict[str, Any], model: str) -> str:
    text = json.dumps({"kind": kind, "model": model, "prompt_version": PROMPT_VERSION, **payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("judge output is not a JSON object")
    return data


def load_api_helpers(insight_doc_root: Path, ensure_api_logger: bool):
    if ensure_api_logger:
        os.environ["ENSURE_API_LOGGER"] = "1"
    if str(insight_doc_root) not in sys.path:
        sys.path.insert(0, str(insight_doc_root))
    from insight_doc.utils.api import create_async_openai_client, query_model_with_retry

    return create_async_openai_client, query_model_with_retry


async def query_judge(
    *,
    query: str | list[dict[str, Any]],
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    insight_doc_root: Path,
    ensure_api_logger: bool,
) -> dict[str, Any]:
    create_async_openai_client, query_model_with_retry = load_api_helpers(insight_doc_root, ensure_api_logger)
    client = create_async_openai_client(timeout=timeout)
    try:
        call = await query_model_with_retry(
            query=query,
            model=model,
            client=client,
            context=[
                {
                    "role": "system",
                    "content": "You are a strict behavior-labeling judge for visual-document QA agent conversations. Output only JSON.",
                }
            ],
            max_attempts=max_retries + 1,
            retry_initial_delay_sec=1.0,
            max_completion_tokens=max_completion_tokens,
        )
    finally:
        await client.close()
    if not call.success or call.response is None:
        raise RuntimeError(call.error or "API call failed without an error message")
    content = call.response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        raise RuntimeError("API response did not contain string content")
    parsed = parse_json_object(content)
    parsed["raw_output"] = content
    return parsed


def tool_semantic_prompt(
    *,
    question: str,
    assistant_text: str,
    tool_args_dict: dict[str, Any],
    previous_tool_summaries: list[dict[str, Any]],
    crop_ref: dict[str, Any],
) -> str:
    return (
        "Classify this assistant tool-call message with exactly one label using this priority order:\n"
        "1. bad_crop: the requested/returned crop is clearly wrong, misses the target, or is much too broad/too narrow.\n"
        "2. error_recovery: the assistant realizes a previous tool call or crop was wrong/unhelpful and tries to correct it.\n"
        "3. expressing_uncertainty: the assistant explicitly says it is uncertain, cannot see clearly, or needs to zoom/check to be sure.\n"
        "4. none: none of the above.\n\n"
        "Return strict JSON: {\"label\": \"bad_crop|error_recovery|expressing_uncertainty|none\", "
        "\"confidence\": number, \"rationale\": string}\n\n"
        f"Question:\n{question}\n\n"
        f"Assistant message:\n{assistant_text}\n\n"
        f"Tool arguments:\n{json.dumps(tool_args_dict, ensure_ascii=False)}\n\n"
        f"Returned crop metadata:\n{json.dumps(crop_ref, ensure_ascii=False)}\n\n"
        f"Previous successful tool calls:\n{json.dumps(previous_tool_summaries[-5:], ensure_ascii=False)}\n"
    )


def answer_grounding_prompt(*, question: str, answer_text: str, crop_refs: list[dict[str, Any]]) -> str:
    return (
        "Decide whether this final answer is ungrounded with respect to the provided tool crop evidence.\n"
        "Mark ungrounded_answer=true only if the answer is clearly inconsistent with, or not supported by, "
        "the visible evidence in the tool crops. If there are no tool crops, judge from the answer text and "
        "do not mark insufficient evidence just because crops are absent.\n\n"
        "Return strict JSON: {\"ungrounded_answer\": boolean, \"confidence\": number, \"rationale\": string}\n\n"
        f"Question:\n{question}\n\n"
        f"Final answer:\n{answer_text}\n\n"
        f"Tool crop metadata:\n{json.dumps(crop_refs[-8:], ensure_ascii=False)}\n"
    )


def content_parts_with_images(prompt: str, images: list[Image.Image], max_pixels: int) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, image in enumerate(images):
        parts.append({"type": "text", "text": f"\nEvidence image {idx}:"})
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(image, max_pixels), "detail": "high"}})
    return parts


async def semantic_tool_label(
    *,
    base_payload: dict[str, Any],
    image: Image.Image | None,
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    key = cache_key("tool_semantic", {k: base_payload[k] for k in ("question", "assistant_text", "tool_args", "crop_ref", "previous_tool_summaries")}, args.model)
    if key in cache:
        return {**cache[key], "cache_key": key, "cache_hit": True}
    if args.dry_run:
        result = {"label": "none", "confidence": None, "rationale": "dry_run"}
    else:
        prompt = tool_semantic_prompt(
            question=base_payload["question"],
            assistant_text=base_payload["assistant_text"],
            tool_args_dict=base_payload["tool_args"],
            previous_tool_summaries=base_payload["previous_tool_summaries"],
            crop_ref=base_payload["crop_ref"],
        )
        query = content_parts_with_images(prompt, [image] if image is not None else [], args.image_max_pixels)
        async with semaphore:
            result = await query_judge(
                query=query,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                insight_doc_root=Path(args.insight_doc_root),
                ensure_api_logger=args.ensure_api_logger,
            )
        label = result.get("label")
        if label not in {"bad_crop", "error_recovery", "expressing_uncertainty", "none"}:
            raise ValueError(f"invalid tool semantic label: {label!r}")
    cache[key] = result
    append_jsonl(cache_path, {"key": key, "result": result})
    return {**result, "cache_key": key, "cache_hit": False}


async def semantic_answer_label(
    *,
    base_payload: dict[str, Any],
    images: list[Image.Image],
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    image_fingerprint = [ref.get("presented_img_idx") for ref in base_payload["crop_refs"][-args.max_judge_images :]]
    key = cache_key(
        "answer_grounding",
        {
            "question": base_payload["question"],
            "answer_text": base_payload["answer_text"],
            "crop_refs": base_payload["crop_refs"][-8:],
            "image_fingerprint": image_fingerprint,
        },
        args.model,
    )
    if key in cache:
        return {**cache[key], "cache_key": key, "cache_hit": True}
    if args.dry_run:
        result = {"ungrounded_answer": False, "confidence": None, "rationale": "dry_run"}
    else:
        prompt = answer_grounding_prompt(
            question=base_payload["question"],
            answer_text=base_payload["answer_text"],
            crop_refs=base_payload["crop_refs"],
        )
        query = content_parts_with_images(prompt, images[-args.max_judge_images :], args.image_max_pixels)
        async with semaphore:
            result = await query_judge(
                query=query,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                insight_doc_root=Path(args.insight_doc_root),
                ensure_api_logger=args.ensure_api_logger,
            )
        if not isinstance(result.get("ungrounded_answer"), bool):
            raise ValueError("answer judge output missing boolean ungrounded_answer")
    cache[key] = result
    append_jsonl(cache_path, {"key": key, "result": result})
    return {**result, "cache_key": key, "cache_hit": False}


async def analyze_record(
    *,
    path: Path,
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    record = load_json(path)
    if record is None:
        return [{"status": "failed", "conversation_path": str(path), "error": "invalid_json"}]
    conversation = [m for m in record.get("conversation") or [] if isinstance(m, dict)]
    refs = presented_refs(record)
    images = restored_image_map(record)
    question = initial_question(record)
    subset = get_subset(record, args.subset_key)
    job = record.get("job") if isinstance(record.get("job"), dict) else {}
    successful_tools: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for idx, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        message_type = message.get("type")
        assistant_text = content_text(message.get("content"))
        base = {
            "status": "ok",
            "conversation_path": str(path),
            "job": job,
            "subset": subset,
            "question": question,
            "message_idx": message.get("message_idx", idx),
            "conversation_index": idx,
            "assistant_type": message_type,
            "assistant_text": assistant_text,
            "primary_label": "none",
            "mechanistic_label": None,
            "judge_label": None,
            "judge": None,
        }

        if message_type == "tool_call":
            call = parse_tool_call(message)
            args_dict = tool_args(call)
            requested_img_idx = args_dict.get("img_idx")
            result_message = next_user_message(conversation, idx)
            result_indices = tool_result_indices(result_message)
            crop_idx = result_indices[-1] if result_indices else None
            crop_ref = refs.get(crop_idx) if isinstance(crop_idx, int) else None
            crop_bbox = bbox_tuple(crop_ref.get("bbox_on_original")) if isinstance(crop_ref, dict) else None
            source_original_img_idx = crop_ref.get("source_original_img_idx") if isinstance(crop_ref, dict) else None
            base.update(
                {
                    "tool_call": call,
                    "tool_args": args_dict,
                    "requested_img_idx": requested_img_idx,
                    "returned_presented_img_idx": crop_idx,
                    "returned_crop_ref": crop_ref,
                }
            )

            if not isinstance(crop_ref, dict) or crop_ref.get("kind") != "region_crop" or crop_bbox is None:
                base["primary_label"] = "invalid_crop"
                base["mechanistic_label"] = "invalid_crop"
                outputs.append(base)
                continue

            previous_success = successful_tools[-1] if successful_tools else None
            if previous_success and previous_success.get("source_original_img_idx") == source_original_img_idx:
                prev_bbox = bbox_tuple(previous_success.get("bbox_on_original"))
                if prev_bbox is not None:
                    iou = bbox_iou(prev_bbox, crop_bbox)
                    base["previous_successful_crop_iou"] = iou
                    if iou >= args.repetitive_iou_threshold:
                        base["primary_label"] = "repetitive_crop"
                        base["mechanistic_label"] = "repetitive_crop"
                        successful_tools.append(
                            {
                                "message_idx": base["message_idx"],
                                "presented_img_idx": crop_idx,
                                "source_original_img_idx": source_original_img_idx,
                                "bbox_on_original": list(crop_bbox),
                                "label": crop_ref.get("region_description"),
                                "primary_label": base["primary_label"],
                            }
                        )
                        outputs.append(base)
                        continue

            parent_ref = refs.get(requested_img_idx) if isinstance(requested_img_idx, int) else None
            if isinstance(parent_ref, dict) and parent_ref.get("kind") == "region_crop":
                base["primary_label"] = "deep_zoom_in"
                base["mechanistic_label"] = "deep_zoom_in"
                successful_tools.append(
                    {
                        "message_idx": base["message_idx"],
                        "presented_img_idx": crop_idx,
                        "source_original_img_idx": source_original_img_idx,
                        "bbox_on_original": list(crop_bbox),
                        "label": crop_ref.get("region_description"),
                        "primary_label": base["primary_label"],
                    }
                )
                outputs.append(base)
                continue

            judge = await semantic_tool_label(
                base_payload={
                    "question": question,
                    "assistant_text": assistant_text,
                    "tool_args": args_dict,
                    "crop_ref": crop_ref,
                    "previous_tool_summaries": successful_tools,
                },
                image=images.get(crop_idx) if isinstance(crop_idx, int) else None,
                args=args,
                cache=cache,
                cache_path=cache_path,
                semaphore=semaphore,
            )
            label = str(judge.get("label", "none"))
            base["judge_label"] = label
            base["judge"] = judge
            base["primary_label"] = label if label in TOOL_PRIORITY else "none"
            successful_tools.append(
                {
                    "message_idx": base["message_idx"],
                    "presented_img_idx": crop_idx,
                    "source_original_img_idx": source_original_img_idx,
                    "bbox_on_original": list(crop_bbox),
                    "label": crop_ref.get("region_description"),
                    "primary_label": base["primary_label"],
                }
            )
            outputs.append(base)
            continue

        if message_type in {"answer", "answer_revision"}:
            crop_refs = [item for item in refs.values() if item.get("kind") == "region_crop"]
            crop_refs_sorted = sorted(crop_refs, key=lambda item: item.get("presented_img_idx", -1))
            crop_images = [
                images[item["presented_img_idx"]]
                for item in crop_refs_sorted
                if isinstance(item.get("presented_img_idx"), int) and item["presented_img_idx"] in images
            ]
            judge = await semantic_answer_label(
                base_payload={
                    "question": question,
                    "answer_text": assistant_text,
                    "crop_refs": crop_refs_sorted,
                },
                images=crop_images,
                args=args,
                cache=cache,
                cache_path=cache_path,
                semaphore=semaphore,
            )
            base["judge"] = judge
            base["judge_label"] = "ungrounded_answer" if judge.get("ungrounded_answer") else "none"
            base["primary_label"] = base["judge_label"]
            outputs.append(base)
    for row in outputs:
        if row.get("status") == "ok":
            row["behavior_polarity"] = label_polarity(str(row.get("primary_label") or "none"))
    return outputs


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace, path_source: str, total_paths: int) -> dict[str, Any]:
    label_counts: dict[str, Counter] = defaultdict(Counter)
    assistant_type_counts: dict[str, Counter] = defaultdict(Counter)
    polarity_counts: dict[str, Counter] = defaultdict(Counter)
    failed = 0
    for row in rows:
        if row.get("status") != "ok":
            failed += 1
            continue
        subset = str(row.get("subset") or "unknown")
        label = str(row.get("primary_label") or "none")
        assistant_type = str(row.get("assistant_type") or "unknown")
        for key in ("overall", subset):
            label_counts[key][label] += 1
            label_counts[key]["total"] += 1
            assistant_type_counts[key][assistant_type] += 1
            if label in GOOD_LABELS:
                polarity_counts[key]["good"] += 1
            elif label in BAD_LABELS:
                polarity_counts[key]["bad"] += 1
            else:
                polarity_counts[key]["neutral"] += 1

    metrics = {}
    all_labels = ["invalid_crop", "repetitive_crop", "deep_zoom_in", "bad_crop", "error_recovery", "expressing_uncertainty", "ungrounded_answer", "none"]
    for subset in ["overall", *sorted(k for k in label_counts if k != "overall")]:
        total = int(label_counts[subset]["total"])
        metrics[subset] = {
            "total_assistant_messages": total,
            "assistant_type_counts": dict(assistant_type_counts[subset]),
            "polarity_counts": dict(polarity_counts[subset]),
            "label_counts": {label: int(label_counts[subset][label]) for label in all_labels},
            "label_rates": {
                label: (float(label_counts[subset][label]) / total if total else None)
                for label in all_labels
            },
            "polarity_rates": {
                polarity: (float(polarity_counts[subset][polarity]) / total if total else None)
                for polarity in ("good", "bad", "neutral")
            },
        }
    return {
        "schema_version": "conversation_behavior_analysis_v1",
        "prompt_version": PROMPT_VERSION,
        "export_dir": args.export_dir,
        "global_step": args.global_step,
        "split": args.split,
        "path_source": path_source,
        "total_conversation_paths": total_paths,
        "total_message_rows": len(rows),
        "failed_rows": failed,
        "judge_model": args.model,
        "repetitive_iou_threshold": args.repetitive_iou_threshold,
        "priority": {
            "tool_messages": TOOL_PRIORITY,
            "answer_messages": ANSWER_PRIORITY,
        },
        "metrics": metrics,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    labels = ["invalid_crop", "repetitive_crop", "deep_zoom_in", "bad_crop", "error_recovery", "expressing_uncertainty", "ungrounded_answer", "none"]
    lines = [
        "# Conversation Behavior Analysis",
        "",
        f"- Export dir: `{summary['export_dir']}`",
        f"- Global step: `{summary['global_step']}`",
        f"- Split: `{summary['split']}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Path source: `{summary['path_source']}`",
        "",
        "## Polarity",
        "",
        "| subset | total | good | bad | neutral |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, metrics in summary["metrics"].items():
        polarity = metrics["polarity_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    subset,
                    str(metrics["total_assistant_messages"]),
                    str(polarity.get("good", 0)),
                    str(polarity.get("bad", 0)),
                    str(polarity.get("neutral", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Labels",
            "",
        "| subset | total | " + " | ".join(labels) + " |",
        "|---|---:" + "|---:" * len(labels) + "|",
        ]
    )
    for subset, metrics in summary["metrics"].items():
        counts = metrics["label_counts"]
        lines.append("| " + " | ".join([subset, str(metrics["total_assistant_messages"]), *[str(counts[label]) for label in labels]]) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache_jsonl) if args.cache_jsonl else output_dir / "behavior_judge_cache.jsonl"
    status_path = Path(args.status_jsonl) if args.status_jsonl else output_dir / "behavior_status.jsonl"
    messages_path = Path(args.messages_jsonl) if args.messages_jsonl else output_dir / "behavior_messages.jsonl"
    summary_json_path = Path(args.summary_json) if args.summary_json else output_dir / "behavior_summary.json"
    summary_md_path = Path(args.summary_md) if args.summary_md else output_dir / "behavior_summary.md"

    paths, path_source = collect_paths(export_dir, args.global_step, args.split)
    if args.limit is not None:
        paths = paths[: args.limit]
    cache = load_cache(cache_path)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    all_rows: list[dict[str, Any]] = []

    async def process_path(path: Path) -> list[dict[str, Any]]:
        return await analyze_record(path=path, args=args, cache=cache, cache_path=cache_path, semaphore=semaphore)

    tasks = [asyncio.create_task(process_path(path)) for path in paths]
    for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
        try:
            rows = await task
        except Exception as exc:
            rows = [{"status": "failed", "error": str(exc)}]
        for row in rows:
            append_jsonl(status_path, row if row.get("status") != "ok" else {"status": "ok", "conversation_path": row.get("conversation_path")})
            append_jsonl(messages_path, row)
        all_rows.extend(rows)
        if args.progress_every > 0 and done_count % args.progress_every == 0:
            print(f"processed {done_count}/{len(paths)} conversations, message_rows={len(all_rows)}", flush=True)

    summary = build_summary(all_rows, args, path_source, len(paths))
    summary["outputs"] = {
        "messages_jsonl": str(messages_path),
        "status_jsonl": str(status_path),
        "cache_jsonl": str(cache_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
    }
    write_json(summary_json_path, summary)
    write_summary_md(summary_md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failed_rows"] == 0 else 1


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
