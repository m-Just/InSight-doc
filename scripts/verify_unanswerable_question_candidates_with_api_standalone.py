#!/usr/bin/env python3
"""Verify synthetic unanswerable question candidates with broader document context.

This script reads candidate rows produced by
generate_unanswerable_question_candidates_with_api.py, rebuilds a broader page
window from the original source row, and asks a larger multimodal verifier
model whether each mutated question is still answerable.

Accepted candidates are materialized as new manifest rows whose images are the
verification page window, question_type is set to not-answerable, and answer is
set to the canonical missing-evidence string.
"""

from __future__ import annotations

import argparse
import asyncio
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

from scripts.generate_unanswerable_question_candidates_with_api import (  # noqa: E402
    DEFAULT_UNANSWERABLE_ANSWER,
    PROMPT_VERSION as GENERATION_PROMPT_VERSION,
    append_jsonl,
    build_query_parts,
    cache_key as generation_cache_key,
    choose_pages_with_nearby,
    load_jsonl_cache,
    load_api_helpers,
    manifest_image_root,
    normalize_text,
    parse_json_response,
    parse_question_pages,
    reconstruct_marked_text,
    resize_image_bytes,
    response_text,
    stable_key,
    write_json,
)


PROMPT_VERSION = "unanswerable_question_mutation_verify_v2"
MARKER_NAME = "scripts/verify_unanswerable_question_candidates_with_api.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Verification model.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--target-dpi", type=int, default=70, help="Displayed DPI for verification images.")
    parser.add_argument("--source-dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=40, help="Broader verification page window cap.")
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--verified-jsonl", default=None)
    parser.add_argument("--rejected-jsonl", default=None)
    parser.add_argument("--accepted-manifest", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_verification_pages(source_row: dict[str, Any], *, max_pages: int) -> list[int]:
    images = source_row.get("images")
    total_pages = len(images) if isinstance(images, list) else 0
    if total_pages <= 0:
        return []
    details = source_row.get("question_involved_visual_details")
    anchors: list[int] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            visual = item.get("visual")
            if isinstance(visual, dict):
                page_id = visual.get("page_id")
                if isinstance(page_id, int):
                    anchors.append(page_id + 1)
            captions = item.get("caption")
            if isinstance(captions, list):
                for cap in captions:
                    if isinstance(cap, dict):
                        page_id = cap.get("page_id")
                        if isinstance(page_id, int):
                            anchors.append(page_id + 1)
    if not anchors:
        anchors = parse_question_pages(source_row.get("question_page_num"), total_pages)
    if not anchors:
        return list(range(1, min(total_pages, max_pages) + 1))
    return choose_pages_with_nearby(total_pages, max_pages, anchors)


def build_prompt(candidate: dict[str, Any]) -> str:
    selected_seed_question = normalize_text(candidate.get("seed_question"))
    selected_seed_answer = normalize_text(candidate.get("seed_answer"))
    selected_candidate_question = normalize_text(candidate.get("candidate_question_part") or candidate.get("candidate_question"))
    return (
        "You are verifying whether a synthetic question mutation is genuinely unanswerable from the provided "
        "document pages.\n"
        "\n"
        "You are shown a broader page window from the same document than the generation stage used.\n"
        "Judge whether the mutated question can be answered from these pages.\n"
        "The question must be evaluated as a document-grounded QA item. If it can be answered from general world "
        "knowledge, commonsense, or external facts without needing the document, reject it.\n"
        "\n"
        "Be strict and separate these cases:\n"
        "- missing_evidence: the question is document-grounded, but the pages do not provide enough information to determine the answer.\n"
        "- document_mismatch: the question is still document-grounded, but the document does not support the premise as asked "
        "(for example, it discusses different entities, fields, units, or options than the mutated question asks about).\n"
        "- generic_answer_exists: the pages support a generic but still acceptable answer such as 'not stated', "
        "'none shown', 'blank', 'not provided', or equivalent.\n"
        "- answerable: the pages support a concrete answer.\n"
        "- externally_answerable: the mutated question does not need document evidence because a competent assistant could answer it from external knowledge or commonsense.\n"
        "- bad_mutation: the mutated question is malformed, unnatural, or semantically broken.\n"
        "- ambiguous: not clearly answerable, but the mutation is too vague to accept as a clean unanswerable.\n"
        "\n"
        "Return strict JSON only with this schema:\n"
        "{"
        "\"label\": str, "
        "\"confidence\": str, "
        "\"reason\": str, "
        "\"final_answer\": str, "
        "\"verifier_ref_pages\": [int]"
        "}\n"
        "The final_answer must be concise and clear, at most 1-2 sentences.\n"
        "For missing_evidence, final_answer should be a document-grounded insufficient-evidence answer.\n"
        "For document_mismatch, final_answer should concisely state that the document does not support the premise as asked.\n"
        "For generic_answer_exists, final_answer should be that concise generic answer.\n"
        "For answerable, final_answer should be the concise direct answer.\n"
        "For externally_answerable, bad_mutation, or ambiguous, final_answer may be empty.\n"
        "\n"
        f"Seed question:\n{selected_seed_question}\n\n"
        f"Seed answer:\n{selected_seed_answer}\n\n"
        f"Mutated candidate question:\n{selected_candidate_question}\n\n"
        f"Mutation type:\n{normalize_text(candidate.get('mutation_type'))}\n\n"
        f"Changed span:\n{normalize_text(candidate.get('changed_span'))}\n\n"
        f"Generator rationale:\n{normalize_text(candidate.get('why_it_should_be_unanswerable'))}\n"
    )


def validate_verification_payload(payload: dict[str, Any]) -> str | None:
    label = normalize_text(payload.get("label"))
    if label not in {
        "missing_evidence",
        "document_mismatch",
        "generic_answer_exists",
        "answerable",
        "externally_answerable",
        "bad_mutation",
        "ambiguous",
    }:
        return "invalid_label"
    if not isinstance(payload.get("reason"), str):
        return "reason_not_string"
    if not isinstance(payload.get("final_answer"), str):
        return "final_answer_not_string"
    verifier_ref_pages = payload.get("verifier_ref_pages")
    if verifier_ref_pages is not None and not isinstance(verifier_ref_pages, list):
        return "verifier_ref_pages_not_list"
    return None


def has_meaningful_answer_text(value: Any) -> bool:
    text = normalize_text(value).strip()
    if not text:
        return False
    return text.lower() not in {"none", "null", "n/a", "na"}


def materialize_pdf_image_tree(
    *,
    output_dir: Path,
    accepted_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    pdf_root = output_dir / "pdf_image"
    pdf_root.mkdir(parents=True, exist_ok=True)
    linked_dirs: set[str] = set()
    source_manifests: set[str] = set()
    missing_source_dirs: list[str] = []
    for accepted_row, verification_record in accepted_pairs:
        candidate = verification_record.get("candidate") or {}
        source_manifest_path = normalize_text(candidate.get("source_manifest_path"))
        if not source_manifest_path:
            continue
        source_manifests.add(source_manifest_path)
        source_pdf_root = manifest_image_root(Path(source_manifest_path).expanduser().resolve())
        for rel in accepted_row.get("images") or []:
            rel_dir = str(Path(str(rel)).parent)
            if not rel_dir or rel_dir == "." or rel_dir in linked_dirs:
                continue
            src_dir = source_pdf_root / rel_dir
            dst_dir = pdf_root / rel_dir
            if not src_dir.exists():
                missing_source_dirs.append(str(src_dir))
                continue
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            if not dst_dir.exists():
                os.symlink(src_dir, dst_dir, target_is_directory=True)
            linked_dirs.add(rel_dir)
    return {
        "pdf_image_root": str(pdf_root),
        "linked_document_dirs": len(linked_dirs),
        "source_manifest_paths": sorted(source_manifests),
        "missing_source_dirs": sorted(set(missing_source_dirs)),
    }


def build_accepted_manifest_row(
    candidate: dict[str, Any],
    *,
    selected_pages: list[int],
    selected_images: list[str],
    verifier_model: str,
    verification_payload: dict[str, Any],
) -> dict[str, Any]:
    source_row = candidate.get("source_row_full") or candidate["source_row"]
    seed_qid = normalize_text(candidate.get("source_question_id"))
    mutation_type = normalize_text(candidate.get("mutation_type")) or "other"
    candidate_id = normalize_text(candidate.get("candidate_id"))
    new_qid = f"{seed_qid}__mut_unanswerable_{mutation_type}_{candidate_id[:8]}"
    out = dict(source_row)
    final_answer = normalize_text(verification_payload.get("final_answer")) or DEFAULT_UNANSWERABLE_ANSWER
    multipart = candidate.get("multipart_metadata")
    question_text = normalize_text(candidate.get("candidate_question"))
    answer_text = final_answer
    if isinstance(multipart, dict) and multipart.get("is_multipart"):
        q_parts = list(multipart.get("question_parts_original") or [])
        a_parts = list(multipart.get("answer_parts_original") or [])
        idx = int(multipart.get("selected_part_index", 0))
        labels = [str(x) for x in multipart.get("part_labels") or []]
        if 0 <= idx < len(q_parts):
            q_parts[idx] = normalize_text(candidate.get("candidate_question_part") or question_text)
            question_text = reconstruct_marked_text(
                normalize_text(multipart.get("preamble_question")),
                labels,
                q_parts,
            )
        if 0 <= idx < len(a_parts):
            a_parts[idx] = final_answer
            answer_text = reconstruct_marked_text(
                normalize_text(multipart.get("preamble_answer")),
                labels,
                a_parts,
            )
    out["question_id"] = new_qid
    out["question"] = question_text
    out["answer"] = answer_text
    out["question_type"] = "not-answerable"
    out["images"] = selected_images
    question_page_num = out.get("question_page_num")
    if isinstance(question_page_num, (list, dict)):
        out["question_page_num"] = json.dumps(question_page_num, ensure_ascii=False)
    elif question_page_num is not None and not isinstance(question_page_num, str):
        out["question_page_num"] = str(question_page_num)
    out["synthetic_unanswerable_metadata"] = {
        "schema_version": "synthetic_unanswerable_v1",
        "marked_by": MARKER_NAME,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "verification_prompt_version": PROMPT_VERSION,
        "source_question_id": seed_qid,
        "seed_question": normalize_text(candidate.get("seed_question")),
        "seed_answer": normalize_text(candidate.get("seed_answer")),
        "seed_question_type": candidate.get("seed_question_type"),
        "candidate_id": candidate_id,
        "candidate_question": question_text,
        "candidate_question_part": normalize_text(candidate.get("candidate_question_part") or candidate.get("candidate_question")),
        "mutation_type": mutation_type,
        "changed_span": normalize_text(candidate.get("changed_span")),
        "generation_model": normalize_text(candidate.get("generation_model")),
        "verification_model": verifier_model,
        "verification_label": normalize_text(verification_payload.get("label")),
        "unanswerable_subtype": normalize_text(verification_payload.get("label")),
        "verification_reason": normalize_text(verification_payload.get("reason")),
        "verification_final_answer": final_answer,
        "verification_full_answer": answer_text,
        "verification_ref_pages": verification_payload.get("verifier_ref_pages") or [],
        "verification_selected_pages": selected_pages,
        "is_multipart": bool(isinstance(multipart, dict) and multipart.get("is_multipart")),
        "selected_part_index": multipart.get("selected_part_index") if isinstance(multipart, dict) else None,
        "selected_part_label": multipart.get("selected_part_label") if isinstance(multipart, dict) else None,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }
    return out


async def query_verification_api(
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
                        "You verify whether mutated document questions are genuinely unanswerable. "
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


async def process_candidate(
    *,
    candidate: dict[str, Any],
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    target_dpi: int,
    source_dpi: int,
    max_pages: int,
    image_detail: str,
    jpeg_quality: int,
    insight_doc_root: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    source_row = candidate.get("source_row_effective") or candidate.get("source_row")
    if not isinstance(source_row, dict):
        return {"candidate_id": normalize_text(candidate.get("candidate_id")), "status": "invalid_source_row"}, None, None, None
    source_manifest_path = Path(normalize_text(candidate.get("source_manifest_path"))).expanduser().resolve()
    image_root = manifest_image_root(source_manifest_path)
    selected_pages = select_verification_pages(source_row, max_pages=max_pages)
    if not selected_pages:
        return {"candidate_id": normalize_text(candidate.get("candidate_id")), "status": "filtered_no_verification_pages"}, None, None, None
    images = source_row.get("images")
    if not isinstance(images, list) or not images:
        return {"candidate_id": normalize_text(candidate.get("candidate_id")), "status": "filtered_no_images"}, None, None, None
    selected_rel_paths: list[str] = []
    selected_abs_paths: list[Path] = []
    for page_num in selected_pages:
        idx = page_num - 1
        if 0 <= idx < len(images):
            rel = str(images[idx])
            abs_path = image_root / rel
            if abs_path.exists():
                selected_rel_paths.append(rel)
                selected_abs_paths.append(abs_path)
    if not selected_abs_paths:
        return {"candidate_id": normalize_text(candidate.get("candidate_id")), "status": "filtered_missing_verification_image_files"}, None, None, None
    payload = {
        "candidate_id": normalize_text(candidate.get("candidate_id")),
        "source_question_id": normalize_text(candidate.get("source_question_id")),
        "candidate_question": normalize_text(candidate.get("candidate_question")),
        "mutation_type": normalize_text(candidate.get("mutation_type")),
        "selected_pages": selected_pages,
        "selected_images": selected_rel_paths,
        "target_dpi": target_dpi,
        "source_dpi": source_dpi,
        "max_pages": max_pages,
    }
    key = generation_cache_key(payload, model=model)
    if dry_run:
        return {"candidate_id": normalize_text(candidate.get("candidate_id")), "status": "selected_dry_run", "cache_key": key}, None, None, None
    prompt = build_prompt(candidate)
    scale = max(0.01, target_dpi / max(1, source_dpi))
    labels = [f"Verification page {page_num}" for page_num in selected_pages[: len(selected_abs_paths)]]
    query_parts = build_query_parts(
        text_prompt=prompt,
        image_paths=selected_abs_paths,
        image_labels=labels,
        scale=scale,
        detail=image_detail,
        quality=jpeg_quality,
    )
    raw = await query_verification_api(
        query_parts=query_parts,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        insight_doc_root=insight_doc_root,
    )
    parsed = parse_json_response(raw)
    validation_error = validate_verification_payload(parsed)
    if validation_error is not None:
        raise ValueError(validation_error)
    label = normalize_text(parsed.get("label"))
    final_answer = normalize_text(parsed.get("final_answer"))
    if label in {"missing_evidence", "document_mismatch", "generic_answer_exists", "answerable"}:
        if not final_answer:
            raise ValueError(f"missing_final_answer_for_label:{label}")
        if len(re.split(r"(?<=[.!?])\s+", final_answer.strip())) > 2:
            raise ValueError("final_answer_too_many_sentences")
    accepted_row = None
    if label in {"missing_evidence", "document_mismatch"}:
        accepted_row = build_accepted_manifest_row(
            candidate,
            selected_pages=selected_pages,
            selected_images=selected_rel_paths,
            verifier_model=model,
            verification_payload=parsed,
        )
    status = {
        "candidate_id": normalize_text(candidate.get("candidate_id")),
        "source_question_id": normalize_text(candidate.get("source_question_id")),
        "status": f"verified_{label}",
        "cache_key": key,
        "selected_pages": selected_pages,
    }
    verification_record = {
        "candidate": candidate,
        "verification": parsed,
        "verification_model": model,
        "verification_prompt_version": PROMPT_VERSION,
        "verification_selected_pages": selected_pages,
        "verification_selected_images": selected_rel_paths,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_result = {
        "status": status,
        "verification_record": verification_record,
        "accepted_manifest_row": accepted_row,
    }
    return status, verification_record, accepted_row, cache_result


async def run(args: argparse.Namespace) -> int:
    candidates_path = Path(args.candidates_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_path = Path(args.cache_jsonl).expanduser().resolve() if args.cache_jsonl else output_dir / "verification_cache.jsonl"
    status_path = Path(args.status_jsonl).expanduser().resolve() if args.status_jsonl else output_dir / "verification_status.jsonl"
    verified_path = Path(args.verified_jsonl).expanduser().resolve() if args.verified_jsonl else output_dir / "verified_candidates.jsonl"
    rejected_path = Path(args.rejected_jsonl).expanduser().resolve() if args.rejected_jsonl else output_dir / "rejected_candidates.jsonl"
    accepted_manifest_path = (
        Path(args.accepted_manifest).expanduser().resolve() if args.accepted_manifest else output_dir / "manifest.jsonl"
    )
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / "summary.json"

    candidates = load_candidates(candidates_path)
    candidates.sort(key=lambda row: stable_key(normalize_text(row.get("candidate_id")), args.sample_seed))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    cache = load_jsonl_cache(cache_path)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    counters: Counter[str] = Counter()
    accepted_rows: list[dict[str, Any]] = []
    accepted_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def worker(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str]:
        payload = {
            "candidate_id": normalize_text(candidate.get("candidate_id")),
            "source_question_id": normalize_text(candidate.get("source_question_id")),
            "candidate_question": normalize_text(candidate.get("candidate_question_part") or candidate.get("candidate_question")),
            "mutation_type": normalize_text(candidate.get("mutation_type")),
            "selected_pages": select_verification_pages(
                candidate.get("source_row_effective") or candidate.get("source_row") or {},
                max_pages=args.max_pages,
            ),
            "target_dpi": args.target_dpi,
            "source_dpi": args.source_dpi,
            "max_pages": args.max_pages,
        }
        key = generation_cache_key(payload, model=args.model)
        cached = cache.get(key)
        if cached is not None:
            status = dict(cached.get("status") or {})
            status["status"] = "cache_hit"
            return status, cached.get("verification_record"), cached.get("accepted_manifest_row"), None, key
        async with sem:
            status, verification_record, accepted_row, cache_result = await process_candidate(
                candidate=candidate,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                target_dpi=args.target_dpi,
                source_dpi=args.source_dpi,
                max_pages=args.max_pages,
                image_detail=args.image_detail,
                jpeg_quality=args.jpeg_quality,
                insight_doc_root=Path(args.insight_doc_root).expanduser().resolve(),
                dry_run=args.dry_run,
            )
        return status, verification_record, accepted_row, cache_result, key

    processed = 0
    for coro in asyncio.as_completed([worker(candidate) for candidate in candidates]):
        try:
            status, verification_record, accepted_row, cache_result, key = await coro
        except Exception as exc:
            status = {"status": "api_failure", "error": f"{type(exc).__name__}: {exc}"}
            verification_record = None
            accepted_row = None
            cache_result = None
            key = ""
        processed += 1
        counters[str(status.get("status"))] += 1
        if key and cache_result is not None:
            append_jsonl(cache_path, {"key": key, "result": cache_result})
        append_jsonl(status_path, status)
        if verification_record is not None:
            append_jsonl(verified_path, verification_record)
        if accepted_row is not None:
            accepted_rows.append(accepted_row)
            accepted_pairs.append((accepted_row, verification_record or {}))
        elif verification_record is not None:
            append_jsonl(rejected_path, verification_record)
        if processed % max(1, args.progress_every) == 0:
            print(f"[{processed}/{len(candidates)}] statuses={dict(counters)} accepted={len(accepted_rows)}", flush=True)

    accepted_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with accepted_manifest_path.open("w", encoding="utf-8") as handle:
        for row in accepted_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    materialization_summary = materialize_pdf_image_tree(
        output_dir=accepted_manifest_path.parent,
        accepted_pairs=accepted_pairs,
    )

    summary = {
        "candidates_jsonl": str(candidates_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "candidate_rows_selected": len(candidates),
        "accepted_manifest_rows": len(accepted_rows),
        "status_counts": dict(counters),
        "target_dpi": args.target_dpi,
        "source_dpi": args.source_dpi,
        "max_pages": args.max_pages,
        "pdf_image_materialization": materialization_summary,
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
