#!/usr/bin/env python3
"""Generate synthetic unanswerable question candidates from answerable manifest rows.

This script takes answerable seed questions from a manifest, shows only the
question-relevant pages to a multimodal API model, and asks it to produce
small, natural question mutations that should become unanswerable while still
looking document-related.

Outputs:
  - candidates.jsonl: accepted generation candidates, one per line
  - generation_status.jsonl: one status record per processed seed row
  - generation_cache.jsonl: cache keyed by prompt payload/model/version
  - summary.json: aggregate counters
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rewrite_exported_convos_final_answers_with_api import load_api_helpers  # noqa: E402


PROMPT_VERSION = "unanswerable_question_mutation_gen_v1"
MARKER_NAME = "scripts/generate_unanswerable_question_candidates_with_api.py"
DEFAULT_UNANSWERABLE_ANSWER = "the information provided in the document cannot answer this question"
PART_MARKER_RE = re.compile(r"(?:(?<=^)|(?<=[\s;]))\(([a-z])\)(?=\s)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input answerable manifest.jsonl.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for candidates.jsonl, cache/status JSONL, and summary.",
    )
    parser.add_argument("--model", default="gpt-5-nano", help="Generation model.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of one-candidate generation attempts per seed row.",
    )
    parser.add_argument("--target-dpi", type=int, default=140, help="Displayed DPI for generation images.")
    parser.add_argument("--source-dpi", type=int, default=200, help="DPI implied by the source manifest images.")
    parser.add_argument(
        "--max-relevant-pages",
        type=int,
        default=6,
        help="Cap relevant pages shown to the generator after page-anchor resolution.",
    )
    parser.add_argument(
        "--max-ungrounded-pages",
        type=int,
        default=40,
        help="If page grounding is missing or unusable, allow sending all pages only when the row has at most this many pages.",
    )
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--candidates-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-subset",
        action="append",
        default=[],
        help="Optional subset filter; may be repeated.",
    )
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
    )
    parser.add_argument(
        "--o3-final-output-json",
        default=None,
        help="Optional O3 backfill JSON keyed by question_id. Defaults to <insight-doc-root>/data/final_output_o3_data_mixed.json.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _stable_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % modulo


def _strip_outer_text_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in ("question:", "answer:"):
        if lowered.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text


def split_marked_parts(text: str) -> dict[str, Any] | None:
    raw = normalize_text(text)
    if not raw:
        return None
    matches = list(PART_MARKER_RE.finditer(raw))
    if len(matches) < 2:
        return None
    labels = [m.group(1).lower() for m in matches]
    expected = [chr(ord("a") + i) for i in range(len(labels))]
    if labels != expected:
        return None
    preamble = raw[: matches[0].start()]
    if preamble.strip() and _strip_outer_text_prefix(preamble).strip():
        return None
    parts: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        part = raw[start:end].strip()
        part = re.sub(r"^[;:\s]+", "", part)
        part = re.sub(r"\s*[;]+\s*$", "", part)
        if not part:
            return None
        parts.append(part)
    return {"preamble": preamble.strip(), "labels": labels, "parts": parts}


def reconstruct_marked_text(preamble: str, labels: list[str], parts: list[str]) -> str:
    prefix = f"{preamble.strip()} " if preamble.strip() else ""
    joined = " ".join(f"({label}) {normalize_text(part)}" for label, part in zip(labels, parts))
    return (prefix + joined).strip()


def _value_by_part(value: Any, part_count: int) -> list[Any] | None:
    if isinstance(value, list) and len(value) == part_count:
        return list(value)
    if isinstance(value, tuple) and len(value) == part_count:
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            parts = [item.strip() for item in text.split(",")]
            if len(parts) == part_count:
                return parts
    if part_count == 1 and value not in (None, ""):
        return [value]
    return None


def prepare_seed_row(row: dict[str, Any], *, sample_seed: int) -> dict[str, Any]:
    question_info = split_marked_parts(normalize_text(row.get("question")))
    answer_info = split_marked_parts(normalize_text(row.get("answer")))
    if question_info is None or answer_info is None:
        return row
    if len(question_info["parts"]) != len(answer_info["parts"]):
        return row
    part_count = len(question_info["parts"])
    qtype_parts = _value_by_part(row.get("question_type"), part_count)
    qpn_parts = _value_by_part(row.get("question_page_num"), part_count)
    original_qid_parts = _value_by_part(row.get("original_question_id"), part_count)
    selected_idx = _stable_index(f"{sample_seed}:{normalize_text(row.get('question_id'))}:multipart", part_count)
    selected_question = question_info["parts"][selected_idx]
    selected_answer = answer_info["parts"][selected_idx]
    selected_row = dict(row)
    selected_row["source_row_full"] = dict(row)
    selected_row["question"] = selected_question
    selected_row["answer"] = selected_answer
    if qtype_parts is not None:
        selected_row["question_type"] = qtype_parts[selected_idx]
    if qpn_parts is not None:
        selected_row["question_page_num"] = qpn_parts[selected_idx]
    selected_row["multipart_metadata"] = {
        "is_multipart": True,
        "preamble_question": question_info["preamble"],
        "preamble_answer": answer_info["preamble"],
        "part_labels": question_info["labels"],
        "question_parts_original": question_info["parts"],
        "answer_parts_original": answer_info["parts"],
        "selected_part_index": selected_idx,
        "selected_part_label": question_info["labels"][selected_idx],
        "selected_question_part": selected_question,
        "selected_answer_part": selected_answer,
        "question_type_parts": qtype_parts,
        "question_page_num_parts": qpn_parts,
        "original_question_id_parts": original_qid_parts,
    }
    return selected_row


def reconstruct_candidate_question(row: dict[str, Any], mutated_part: str) -> str:
    multipart = row.get("multipart_metadata")
    if not isinstance(multipart, dict) or not multipart.get("is_multipart"):
        return mutated_part
    parts = list(multipart.get("question_parts_original") or [])
    idx = int(multipart.get("selected_part_index", 0))
    if not (0 <= idx < len(parts)):
        return mutated_part
    parts[idx] = mutated_part
    return reconstruct_marked_text(
        normalize_text(multipart.get("preamble_question")),
        [str(x) for x in multipart.get("part_labels") or []],
        parts,
    )


def answer_type(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return "empty"
    if lowered in {"yes", "no"}:
        return "boolean"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", lowered):
        return "numeric"
    if "|" in lowered:
        return "multi"
    return "text"


def parse_question_pages(value: Any, total_pages: int) -> list[int]:
    pages, _ = parse_question_pages_with_mode(value, total_pages, zero_based=False)
    return pages


def _flatten_page_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_page_values(item))
        return out
    return [value]


def parse_question_pages_with_mode(value: Any, total_pages: int, *, zero_based: bool) -> tuple[list[int], bool]:
    """Parse question_page_num from the mixed real-world formats in the O3 manifests.

    Returns `(pages, had_explicit_value)`.
    """
    if value is None:
        return [], False

    raw_values: list[Any]
    if isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = _flatten_page_values(value)
    else:
        text = str(value).strip()
        if not text:
            return [], True
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if parsed is not None:
                raw_values = _flatten_page_values(parsed)
            else:
                raw_values = [part for part in re.split(r"[,\s]+", text) if part]
        else:
            raw_values = [part for part in re.split(r"[,\s]+", text) if part]

    nums: list[int] = []
    for item in raw_values:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            nums.append(int(text))
        except ValueError:
            continue

    if zero_based:
        nums = [n + 1 for n in nums if n >= 0]

    return sorted({page for page in nums if page >= 1}), True


def _image_stem_to_index(images: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for idx, rel in enumerate(images):
        stem = Path(str(rel)).stem
        try:
            page_num = int(stem)
        except ValueError:
            continue
        out.setdefault(page_num, idx)
    return out


def _resolve_page_numbers_to_indices(page_numbers: list[int], images: list[str]) -> list[int]:
    stem_to_index = _image_stem_to_index(images)
    resolved: list[int] = []
    seen: set[int] = set()
    for page_num in page_numbers:
        candidates: list[int] = []
        dense_idx = page_num - 1
        if 0 <= dense_idx < len(images):
            candidates.append(dense_idx)
        stem_idx = stem_to_index.get(page_num)
        if stem_idx is not None:
            candidates.append(stem_idx)
        zero_based_stem_idx = stem_to_index.get(page_num - 1)
        if zero_based_stem_idx is not None:
            candidates.append(zero_based_stem_idx)
        for idx in candidates:
            if idx not in seen:
                resolved.append(idx)
                seen.add(idx)
                break
    return resolved


def choose_pages_with_nearby(total_pages: int, max_pages: int, anchors: list[int]) -> list[int]:
    if total_pages <= 0:
        return []
    if max_pages >= total_pages:
        return list(range(1, total_pages + 1))
    anchors = sorted({page for page in anchors if 1 <= page <= total_pages})
    if not anchors:
        return []
    if len(anchors) >= max_pages:
        return anchors[:max_pages]
    selected: set[int] = set(anchors)
    distance = 1
    while len(selected) < max_pages and distance < total_pages:
        added_this_round = 0
        for anchor in anchors:
            for candidate in (anchor - distance, anchor + distance):
                if 1 <= candidate <= total_pages and candidate not in selected:
                    selected.add(candidate)
                    added_this_round += 1
                    if len(selected) >= max_pages:
                        break
            if len(selected) >= max_pages:
                break
        if added_this_round == 0:
            distance += 1
            continue
        distance += 1
    if len(selected) < max_pages:
        leftovers = [page for page in range(1, total_pages + 1) if page not in selected]
        leftovers.sort(key=lambda page: min(abs(page - anchor) for anchor in anchors))
        for page in leftovers:
            selected.add(page)
            if len(selected) >= max_pages:
                break
    return sorted(selected)


def extract_relevant_pages(row: dict[str, Any], max_pages: int, *, max_ungrounded_pages: int) -> list[int]:
    images = row.get("images")
    total_pages = len(images) if isinstance(images, list) else 0
    if total_pages <= 0:
        return []
    details = row.get("question_involved_visual_details")
    anchor_indices: list[int] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            visual = item.get("visual")
            if isinstance(visual, dict):
                page_id = visual.get("page_id")
                if isinstance(page_id, int) and 0 <= page_id < total_pages:
                    anchor_indices.append(page_id)
            caption = item.get("caption")
            if isinstance(caption, list):
                for cap in caption:
                    if isinstance(cap, dict):
                        page_id = cap.get("page_id")
                        if isinstance(page_id, int) and 0 <= page_id < total_pages:
                            anchor_indices.append(page_id)
    if anchor_indices:
        anchor_pages = sorted({idx + 1 for idx in anchor_indices})
        return choose_pages_with_nearby(total_pages, max_pages, anchor_pages)

    subset = normalize_text(row.get("subset")).lower()
    qpn_value = row.get("question_page_num")
    use_zero_based_qpn = subset == "dude" and (
        isinstance(qpn_value, list) or (isinstance(qpn_value, str) and qpn_value.strip().startswith("["))
    )
    anchor_page_numbers, had_qpn_value = parse_question_pages_with_mode(
        qpn_value,
        total_pages,
        zero_based=use_zero_based_qpn,
    )
    if not anchor_page_numbers and total_pages <= max_ungrounded_pages:
        explicit_missing_qpn = qpn_value is None or (isinstance(qpn_value, list) and len(qpn_value) == 0)
        if explicit_missing_qpn or had_qpn_value:
            return list(range(1, total_pages + 1))
    if not anchor_page_numbers and had_qpn_value:
        return []
    if not anchor_page_numbers:
        return []
    anchor_indices = _resolve_page_numbers_to_indices(anchor_page_numbers, images)
    if not anchor_indices:
        return []
    anchor_pages = sorted({idx + 1 for idx in anchor_indices})
    return choose_pages_with_nearby(total_pages, max_pages, anchor_pages)


def manifest_image_root(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent / "pdf_image"


def resize_image_bytes(image_path: Path, *, scale: float, quality: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if scale > 0 and not math.isclose(scale, 1.0):
            width = max(1, int(round(image.width * scale)))
            height = max(1, int(round(image.height * scale)))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_query_parts(
    *,
    text_prompt: str,
    image_paths: list[Path],
    image_labels: list[str],
    scale: float,
    detail: str,
    quality: int,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for label, path in zip(image_labels, image_paths):
        parts.append({"type": "text", "text": label})
        image_url = {"url": resize_image_bytes(path, scale=scale, quality=quality)}
        if detail != "auto":
            image_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": image_url})
    parts.append({"type": "text", "text": text_prompt})
    return parts


def response_text(call: Any) -> str:
    if not call.success or call.response is None:
        raise RuntimeError(call.error or "API call failed without response")
    content = call.response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise RuntimeError("API response did not contain string content")
    return content.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("response JSON is not an object")
    return payload


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def local_candidate_filter(candidate: dict[str, Any], *, seed_question: str, seed_answer: str) -> str | None:
    question = normalize_text(candidate.get("candidate_question"))
    mutation_type = normalize_text(candidate.get("mutation_type"))
    if not question:
        return "empty_candidate_question"
    if question.strip().lower() == seed_question.strip().lower():
        return "question_unchanged"
    if not mutation_type:
        return "missing_mutation_type"
    if not re.fullmatch(r"[a-z][a-z0-9_/-]*", mutation_type):
        return "invalid_mutation_type"
    if len(tokenize(question)) < 4:
        return "question_too_short"
    if normalize_text(seed_answer) and normalize_text(seed_answer).lower() in question.lower():
        return "question_contains_seed_answer"
    return None


def build_prompt(*, row: dict[str, Any], seed_answer_text: str) -> str:
    return (
        "You are generating synthetic unanswerable questions from answerable document QA seeds.\n"
        "\n"
        "You are shown only the pages directly relevant to the original question. Generate small, natural question "
        "mutations that remain readable and document-related, but should become unanswerable because the mutated "
        "target is not supported by the shown evidence.\n"
        "\n"
        "Rules:\n"
        "- Keep the wording close to the original question.\n"
        "- Change only one core factual target where possible.\n"
        "- Prefer mutations like entity swap, number/date/value swap, attribute swap, comparison flip, or antonym swap.\n"
        "- Do not make the question nonsensical, self-contradictory, or obviously fake.\n"
        "- Do not ask about missing pages, hidden labels, or dataset artifacts.\n"
        "- Do not output an answer; only output mutated questions and metadata.\n"
        "- Preserve the original answer style/type as much as possible.\n"
        "- The mutated question should look like a plausible question someone could ask about this document.\n"
        "\n"
        "Return strict JSON only with this schema:\n"
        "{"
        "\"candidate_question\": str, "
        "\"mutation_type\": str"
        "}\n"
        "\n"
        f"Original question:\n{normalize_text(row.get('question'))}\n\n"
        f"Original answer (for style/type only; do not repeat it verbatim in the question):\n{seed_answer_text}\n\n"
        f"Original question_type:\n{normalize_text(row.get('question_type'))}\n\n"
        f"Subset:\n{normalize_text(row.get('subset'))}\n"
    )


def cache_key(payload: dict[str, Any], *, model: str) -> str:
    encoded = json.dumps(
        {"model": model, "prompt_version": PROMPT_VERSION, **payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_generation_candidates(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        raise ValueError("response_not_object")
    if isinstance(parsed.get("candidate_question"), str):
        return [parsed]
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("response missing candidate_question or candidates list")
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            out.append(candidate)
    return out


async def query_generation_api(
    *,
    query_parts: list[dict[str, Any]],
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    insight_doc_root: Path,
) -> str:
    create_async_openai_client, query_model_with_retry = load_api_helpers(insight_doc_root, ensure_api_logger=True)
    client = create_async_openai_client(timeout=timeout)
    try:
        call = await query_model_with_retry(
            query=query_parts,
            model=model,
            client=client,
            context=[
                {
                    "role": "system",
                    "content": (
                        "You generate synthetic unanswerable questions for visual-document QA. "
                        "Return strict JSON only."
                    ),
                }
            ],
            max_attempts=max_retries + 1,
            retry_initial_delay_sec=1.0,
            max_completion_tokens=max_completion_tokens,
        )
    finally:
        await client.close()
    return response_text(call)


def iter_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_o3_backfill(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def _is_missing_page_grounding(row: dict[str, Any]) -> bool:
    details = row.get("question_involved_visual_details")
    if isinstance(details, list) and details:
        return False
    qpn = row.get("question_page_num")
    return qpn is None or qpn == [] or (isinstance(qpn, str) and not qpn.strip())


def apply_o3_backfill(row: dict[str, Any], backfill: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qid = normalize_text(row.get("question_id"))
    patch = backfill.get(qid)
    if patch is None:
        return row
    out = dict(row)
    if _is_missing_page_grounding(out):
        if "question_involved_visual_details" in patch and patch.get("question_involved_visual_details"):
            out["question_involved_visual_details"] = patch.get("question_involved_visual_details")
        if _is_missing_page_grounding(out) and "question_page_num" in patch:
            out["question_page_num"] = patch.get("question_page_num")
    if out.get("question_type") in (None, "", []):
        if "question_type" in patch:
            out["question_type"] = patch.get("question_type")
    return out


async def process_row(
    *,
    row: dict[str, Any],
    manifest_path: Path,
    image_root: Path,
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    num_candidates: int,
    target_dpi: int,
    source_dpi: int,
    max_relevant_pages: int,
    max_ungrounded_pages: int,
    image_detail: str,
    jpeg_quality: int,
    insight_doc_root: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    qid = normalize_text(row.get("question_id"))
    seed_question = normalize_text(row.get("question"))
    seed_answer_text = normalize_text(row.get("answer"))
    relevant_pages = extract_relevant_pages(
        row,
        max_relevant_pages,
        max_ungrounded_pages=max_ungrounded_pages,
    )
    if not relevant_pages:
        status = {"question_id": qid, "status": "filtered_no_relevant_pages"}
        return status, [], None
    images = row.get("images")
    if not isinstance(images, list) or not images:
        status = {"question_id": qid, "status": "filtered_no_images"}
        return status, [], None
    selected_rel_paths: list[str] = []
    selected_abs_paths: list[Path] = []
    for page_num in relevant_pages:
        idx = page_num - 1
        if 0 <= idx < len(images):
            rel = str(images[idx])
            abs_path = image_root / rel
            if abs_path.exists():
                selected_rel_paths.append(rel)
                selected_abs_paths.append(abs_path)
    if not selected_abs_paths:
        status = {"question_id": qid, "status": "filtered_missing_relevant_image_files"}
        return status, [], None
    payload = {
        "question_id": qid,
        "question": seed_question,
        "answer": seed_answer_text,
        "question_type": normalize_text(row.get("question_type")),
        "subset": normalize_text(row.get("subset")),
        "relevant_pages": relevant_pages,
        "relevant_images": selected_rel_paths,
        "num_candidates": num_candidates,
        "target_dpi": target_dpi,
        "source_dpi": source_dpi,
    }
    key = cache_key(payload, model=model)
    if dry_run:
        return {"question_id": qid, "status": "selected_dry_run", "cache_key": key}, [], None
    scale = max(0.01, target_dpi / max(1, source_dpi))
    labels = [f"Relevant page {page_num}" for page_num in relevant_pages[: len(selected_abs_paths)]]
    accepted: list[dict[str, Any]] = []
    local_filter_counts: Counter[str] = Counter()
    raw_responses: list[str] = []
    attempt_failures: list[dict[str, Any]] = []
    raw_candidates = 0
    for attempt_idx in range(max(1, num_candidates)):
        prompt = build_prompt(row=row, seed_answer_text=seed_answer_text)
        query_parts = build_query_parts(
            text_prompt=prompt,
            image_paths=selected_abs_paths,
            image_labels=labels,
            scale=scale,
            detail=image_detail,
            quality=jpeg_quality,
        )
        try:
            raw = await query_generation_api(
                query_parts=query_parts,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                max_completion_tokens=max_completion_tokens,
                insight_doc_root=insight_doc_root,
            )
            raw_responses.append(raw)
            candidates = normalize_generation_candidates(parse_json_response(raw))
        except Exception as exc:
            attempt_failures.append(
                {
                    "attempt_index": attempt_idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        raw_candidates += len(candidates)
        for idx, candidate in enumerate(candidates):
            reason = local_candidate_filter(candidate, seed_question=seed_question, seed_answer=seed_answer_text)
            if reason is not None:
                local_filter_counts[reason] += 1
                continue
            candidate_question_part = normalize_text(candidate.get("candidate_question"))
            candidate_question = reconstruct_candidate_question(row, candidate_question_part)
            mutation_type = normalize_text(candidate.get("mutation_type")) or "other"
            changed_span = normalize_text(candidate.get("changed_span"))
            rationale = normalize_text(candidate.get("why_it_should_be_unanswerable"))
            candidate_id = hashlib.sha256(f"{qid}:{candidate_question}:{attempt_idx}:{idx}".encode("utf-8")).hexdigest()[:16]
            accepted.append(
                {
                    "candidate_id": candidate_id,
                    "source_question_id": qid,
                    "source_manifest_path": str(manifest_path),
                    "source_row": dict(row),
                    "seed_question": seed_question,
                    "seed_answer": seed_answer_text,
                    "seed_question_type": row.get("question_type"),
                    "seed_question_full": normalize_text(
                        reconstruct_marked_text(
                            normalize_text((row.get("multipart_metadata") or {}).get("preamble_question")),
                            [str(x) for x in ((row.get("multipart_metadata") or {}).get("part_labels") or [])],
                            [str(x) for x in ((row.get("multipart_metadata") or {}).get("question_parts_original") or [])],
                        )
                    )
                    if isinstance(row.get("multipart_metadata"), dict)
                    else seed_question,
                    "seed_answer_full": normalize_text(
                        reconstruct_marked_text(
                            normalize_text((row.get("multipart_metadata") or {}).get("preamble_answer")),
                            [str(x) for x in ((row.get("multipart_metadata") or {}).get("part_labels") or [])],
                            [str(x) for x in ((row.get("multipart_metadata") or {}).get("answer_parts_original") or [])],
                        )
                    )
                    if isinstance(row.get("multipart_metadata"), dict)
                    else seed_answer_text,
                    "source_row_full": dict(row.get("source_row_full") or row),
                    "source_row_effective": dict(row),
                    "is_multipart": bool((row.get("multipart_metadata") or {}).get("is_multipart")),
                    "multipart_metadata": row.get("multipart_metadata"),
                    "candidate_question_part": candidate_question_part,
                    "candidate_question": candidate_question,
                    "mutation_type": mutation_type,
                    "changed_span": changed_span,
                    "why_it_should_be_unanswerable": rationale,
                    "preserved_answer_type": normalize_text(candidate.get("preserved_answer_type")) or answer_type(seed_answer_text),
                    "generation_model": model,
                    "generation_prompt_version": PROMPT_VERSION,
                    "generation_target_dpi": target_dpi,
                    "generation_source_dpi": source_dpi,
                    "generation_relevant_pages": relevant_pages,
                    "generation_relevant_images": selected_rel_paths,
                    "generation_attempt_index": attempt_idx,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    if accepted:
        final_status = "generated"
    elif attempt_failures and len(attempt_failures) == max(1, num_candidates):
        final_status = "api_failure"
    else:
        final_status = "generated_no_accepted_candidates"
    status = {
        "question_id": qid,
        "status": final_status,
        "cache_key": key,
        "requested_candidates": num_candidates,
        "generation_attempts": max(1, num_candidates),
        "failed_attempts": len(attempt_failures),
        "raw_candidates": raw_candidates,
        "accepted_candidates": len(accepted),
        "local_filter_counts": dict(local_filter_counts),
        "attempt_failures": attempt_failures,
    }
    cache_result = {
        "raw_responses": raw_responses,
        "status": status,
        "accepted_candidates": accepted,
    }
    return status, accepted, cache_result


async def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    insight_doc_root = Path(args.insight_doc_root).expanduser().resolve()
    o3_final_output_json = (
        Path(args.o3_final_output_json).expanduser().resolve()
        if args.o3_final_output_json
        else insight_doc_root / "data" / "final_output_o3_data_mixed.json"
    )
    cache_path = Path(args.cache_jsonl).expanduser().resolve() if args.cache_jsonl else output_dir / "generation_cache.jsonl"
    status_path = Path(args.status_jsonl).expanduser().resolve() if args.status_jsonl else output_dir / "generation_status.jsonl"
    candidates_path = (
        Path(args.candidates_jsonl).expanduser().resolve() if args.candidates_jsonl else output_dir / "candidates.jsonl"
    )
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / "summary.json"
    image_root = manifest_image_root(manifest_path)
    rows = iter_manifest(manifest_path)
    o3_backfill = load_o3_backfill(o3_final_output_json)
    if o3_backfill:
        rows = [apply_o3_backfill(row, o3_backfill) for row in rows]
    if args.only_subset:
        allowed = set(args.only_subset)
        rows = [row for row in rows if normalize_text(row.get("subset")) in allowed]
    rows = [row for row in rows if normalize_text(row.get("question_type")).lower() != "not-answerable"]
    rows = [prepare_seed_row(row, sample_seed=args.sample_seed) for row in rows]
    rows.sort(key=lambda row: stable_key(normalize_text(row.get("question_id")), args.sample_seed))
    if args.limit is not None:
        rows = rows[: args.limit]

    cache = load_jsonl_cache(cache_path)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    counters: Counter[str] = Counter()
    generated_candidates_total = 0

    async def worker(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, str]:
        qid = normalize_text(row.get("question_id"))
        relevant_pages = extract_relevant_pages(
            row,
            args.max_relevant_pages,
            max_ungrounded_pages=args.max_ungrounded_pages,
        )
        payload = {
            "question_id": qid,
            "question": normalize_text(row.get("question")),
            "answer": normalize_text(row.get("answer")),
            "question_type": normalize_text(row.get("question_type")),
            "subset": normalize_text(row.get("subset")),
            "relevant_pages": relevant_pages,
            "relevant_images": [
                str(row["images"][page - 1])
                for page in relevant_pages
                if isinstance(row.get("images"), list) and 0 <= page - 1 < len(row["images"])
            ],
            "num_candidates": args.num_candidates,
            "target_dpi": args.target_dpi,
            "source_dpi": args.source_dpi,
        }
        key = cache_key(payload, model=args.model)
        cached = cache.get(key)
        if cached is not None:
            status = dict(cached.get("status") or {})
            status["status"] = "cache_hit"
            accepted = list(cached.get("accepted_candidates") or [])
            return status, accepted, None, key
        async with sem:
            status, accepted, cache_result = await process_row(
                row=row,
                manifest_path=manifest_path,
                image_root=image_root,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                num_candidates=args.num_candidates,
                target_dpi=args.target_dpi,
                source_dpi=args.source_dpi,
                max_relevant_pages=args.max_relevant_pages,
                max_ungrounded_pages=args.max_ungrounded_pages,
                image_detail=args.image_detail,
                jpeg_quality=args.jpeg_quality,
                insight_doc_root=insight_doc_root,
                dry_run=args.dry_run,
            )
        return status, accepted, cache_result, key

    processed = 0
    for coro in asyncio.as_completed([worker(row) for row in rows]):
        try:
            status, accepted_candidates, cache_result, key = await coro
        except Exception as exc:
            status = {"status": "api_failure", "error": f"{type(exc).__name__}: {exc}"}
            accepted_candidates = []
            cache_result = None
            key = ""
        processed += 1
        counters[str(status.get("status"))] += 1
        generated_candidates_total += len(accepted_candidates)
        if key and cache_result is not None:
            append_jsonl(cache_path, {"key": key, "result": cache_result})
        append_jsonl(status_path, status)
        for candidate in accepted_candidates:
            append_jsonl(candidates_path, candidate)
        if processed % max(1, args.progress_every) == 0:
            print(
                f"[{processed}/{len(rows)}] generated_candidates={generated_candidates_total} "
                f"statuses={dict(counters)}",
                flush=True,
            )

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "rows_selected": len(rows),
        "generated_candidates_total": generated_candidates_total,
        "status_counts": dict(counters),
        "target_dpi": args.target_dpi,
        "source_dpi": args.source_dpi,
        "max_relevant_pages": args.max_relevant_pages,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
