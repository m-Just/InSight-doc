#!/usr/bin/env python3
"""Audit overlap between the final RL parquet and the SFT training parquets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import build_rl_old_new_sft_dedup_prompt27k_parquet_20260616 as rl_build  # noqa: E402


DEFAULT_RL_PARQUET = (
    REPO_ROOT
    / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e05_plus_arxiv_struct1k_llm_20260722"
    / "insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e_arxiv_struct1k_llm-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notes/generated/rl_final_sft_overlap_audit_20260723"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-parquet", type=Path, default=DEFAULT_RL_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def strip_prompt_images(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("<image>", "")
    text = re.sub(r"(?m)^\s*Image\s+\d+\s*:\s*<image>\s*$", "", text)
    text = re.sub(r"(?m)^\s*---\s*$", "", text)
    return re.sub(r"\s+", " ", text.strip().lower())


def norm(text: Any) -> str:
    return rl_build.normalize_text(strip_prompt_images(text))


def prompt_user_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return ""
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "user":
            return strip_prompt_images(message.get("content"))
    return ""


def component(data_source: Any) -> str:
    data_source = str(data_source)
    if data_source.startswith("arxiv_struct_"):
        return "arxiv_struct_addon"
    if data_source.endswith("_mc_false_e"):
        return "mc_false_e_addon"
    return "base_16k_sft_deduped"


def answerability(data_source: Any) -> str:
    return "unanswerable" if "unanswerable" in str(data_source) else "answerable"


def category(data_source: Any) -> str:
    data_source = str(data_source)
    if data_source.startswith("arxiv_struct"):
        return "arxiv_struct"
    for candidate in [
        "arxiv_veqa",
        "arxiv_mveqa",
        "docvqa",
        "dude",
        "info",
        "map_metro",
        "map_travel",
        "poster",
    ]:
        if data_source.startswith(candidate):
            return candidate
    return data_source.split("_")[0] if data_source else "unknown"


def extract_raw_document_id(record: dict[str, Any]) -> str:
    extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    return str(extra_info.get("document_id") or record.get("document_id") or "")


def collect_raw_question_metadata(raw_files: set[Path]) -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for raw_path in sorted(raw_files):
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue
        extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
        raw_qid = str(extra_info.get("question_id") or record.get("question_id") or "")
        document_id = extract_raw_document_id(record)
        question = rl_build.extract_raw_question(record)
        if raw_qid and question:
            mapping[rl_build.normalize_text(question)].add((raw_qid, document_id))
    return mapping


def build_sft_keys() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], set[str]], dict[str, Any]]:
    """Return qid, question, doc+question keys, and a summary."""
    sft_files = rl_build.sft_train_files()
    medium_dirs = {path.parents[1] for path in sft_files}
    raw_meta_by_dir = {
        medium_dir: collect_raw_question_metadata(set(medium_dir.glob("raw*/*.json")))
        for medium_dir in sorted(medium_dirs)
    }

    sft_qid_to_files: dict[str, set[str]] = defaultdict(set)
    sft_question_to_files: dict[str, set[str]] = defaultdict(set)
    sft_doc_question_to_files: dict[tuple[str, str], set[str]] = defaultdict(set)
    total_rows = 0
    qid_mapped_rows = 0
    doc_question_mapped_rows = 0
    fallback_rows = 0

    for path in sft_files:
        raw_question_to_meta = raw_meta_by_dir.get(path.parents[1], {})
        unique_question_to_meta = {
            question: next(iter(items))
            for question, items in raw_question_to_meta.items()
            if len(items) == 1
        }
        table = pq.read_table(path, columns=["messages"])
        for row in table.to_pylist():
            total_rows += 1
            question_norm = rl_build.normalize_text(rl_build.extract_sft_user_question(row["messages"]))
            if not question_norm:
                continue
            sft_question_to_files[question_norm].add(str(path))
            recovered = unique_question_to_meta.get(question_norm)
            if recovered:
                recovered_qid, recovered_document_id = recovered
                sft_qid_to_files[str(recovered_qid)].add(str(path))
                if recovered_document_id:
                    sft_doc_question_to_files[(str(recovered_document_id), question_norm)].add(str(path))
                    doc_question_mapped_rows += 1
                qid_mapped_rows += 1
            else:
                fallback_rows += 1

    summary = {
        "sft_train_files": [str(path) for path in sft_files],
        "sft_total_rows": total_rows,
        "sft_qid_mapped_rows": qid_mapped_rows,
        "sft_doc_question_mapped_rows": doc_question_mapped_rows,
        "sft_fallback_rows": fallback_rows,
        "unique_sft_qids": len(sft_qid_to_files),
        "unique_sft_questions": len(sft_question_to_files),
        "unique_sft_doc_questions": len(sft_doc_question_to_files),
    }
    return sft_qid_to_files, sft_question_to_files, sft_doc_question_to_files, summary


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}%" if denominator else "n/a"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_qid_to_files, sft_question_to_files, sft_doc_question_to_files, sft_summary = build_sft_keys()
    sft_qids = set(sft_qid_to_files)
    sft_questions = set(sft_question_to_files)
    sft_doc_questions = set(sft_doc_question_to_files)

    rows = pq.read_table(args.rl_parquet, columns=["data_source", "prompt", "extra_info"]).to_pylist()

    overlaps: list[dict[str, Any]] = []
    match_type_counter: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    component_overlap: Counter[str] = Counter()
    data_source_counts: Counter[str] = Counter()
    data_source_overlap: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_overlap: Counter[str] = Counter()
    answerability_counts: Counter[str] = Counter()
    answerability_overlap: Counter[str] = Counter()
    strict_component_overlap: Counter[str] = Counter()
    strict_data_source_overlap: Counter[str] = Counter()
    strict_category_overlap: Counter[str] = Counter()
    strict_answerability_overlap: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        data_source = str(row.get("data_source"))
        row_component = component(data_source)
        row_category = category(data_source)
        row_answerability = answerability(data_source)

        component_counts[row_component] += 1
        data_source_counts[data_source] += 1
        category_counts[row_category] += 1
        answerability_counts[row_answerability] += 1

        extra_info = row.get("extra_info") or {}
        document_id = str(extra_info.get("document_id") or "")
        qid_candidates = []
        for key in ["question_id", "source_question_id"]:
            value = extra_info.get(key)
            if value:
                qid_candidates.append((key, str(value)))

        text_candidates = []
        for key in ["question", "original_question", "original_question_before_llm_rewrite"]:
            value = extra_info.get(key)
            if value:
                text_candidates.append((key, norm(value)))
        user_text = prompt_user_text(row.get("prompt"))
        if user_text:
            text_candidates.append(("prompt_user", user_text))

        matched = []
        for key, qid in qid_candidates:
            if qid in sft_qids:
                matched.append((f"qid:{key}", qid, sorted(sft_qid_to_files[qid])[:3]))
        for key, question in text_candidates:
            if question in sft_questions:
                matched.append((f"question:{key}", question, sorted(sft_question_to_files[question])[:3]))

        deduped_matches = []
        seen = set()
        for match in matched:
            match_key = (match[0], match[1])
            if match_key not in seen:
                seen.add(match_key)
                deduped_matches.append(match)

        if not deduped_matches:
            continue

        strict_matches = []
        weak_matches = []
        for match in deduped_matches:
            match_type, value, files = match
            is_strict = match_type.startswith("qid:")
            if not is_strict and document_id:
                is_strict = (document_id, value) in sft_doc_questions
                if is_strict:
                    files = sorted(sft_doc_question_to_files[(document_id, value)])[:3]
                    match = (match_type.replace("question:", "doc_question:"), value, files)
            if is_strict:
                strict_matches.append(match)
            else:
                weak_matches.append(match)

        component_overlap[row_component] += 1
        data_source_overlap[data_source] += 1
        category_overlap[row_category] += 1
        answerability_overlap[row_answerability] += 1
        if strict_matches:
            strict_component_overlap[row_component] += 1
            strict_data_source_overlap[data_source] += 1
            strict_category_overlap[row_category] += 1
            strict_answerability_overlap[row_answerability] += 1
        for match_type, _, _ in strict_matches + weak_matches:
            match_type_counter[match_type] += 1

        overlaps.append(
            {
                "row_index": idx,
                "component": row_component,
                "data_source": data_source,
                "answerability": row_answerability,
                "category": row_category,
                "overlap_class": "strict" if strict_matches else "question_only",
                "question_id": str(extra_info.get("question_id") or ""),
                "document_id": document_id,
                "question": str(extra_info.get("question") or "")[:500].replace("\n", " "),
                "original_question": str(
                    extra_info.get("original_question")
                    or extra_info.get("original_question_before_llm_rewrite")
                    or ""
                )[:500].replace("\n", " "),
                "match_reasons": "; ".join(match[0] for match in strict_matches + weak_matches),
                "matched_values": "; ".join(match[1][:220] for match in strict_matches + weak_matches),
                "sft_files": " | ".join({path for _, _, files in strict_matches + weak_matches for path in files}),
            }
        )

    detail_path = output_dir / "overlap_rows.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "row_index",
            "component",
            "data_source",
            "answerability",
            "category",
            "overlap_class",
            "question_id",
            "document_id",
            "question",
            "original_question",
            "match_reasons",
            "matched_values",
            "sft_files",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overlaps)

    summary = {
        "rl_parquet": str(args.rl_parquet.resolve()),
        "rl_rows": len(rows),
        **sft_summary,
        "overlap_rows": len(overlaps),
        "overlap_pct": len(overlaps) / len(rows) if rows else 0.0,
        "strict_overlap_rows": sum(strict_component_overlap.values()),
        "strict_overlap_pct": sum(strict_component_overlap.values()) / len(rows) if rows else 0.0,
        "question_only_overlap_rows": len(overlaps) - sum(strict_component_overlap.values()),
        "component_counts": dict(component_counts),
        "component_overlap": dict(component_overlap),
        "strict_component_overlap": dict(strict_component_overlap),
        "answerability_counts": dict(answerability_counts),
        "answerability_overlap": dict(answerability_overlap),
        "strict_answerability_overlap": dict(strict_answerability_overlap),
        "category_counts": dict(category_counts),
        "category_overlap": dict(category_overlap),
        "strict_category_overlap": dict(strict_category_overlap),
        "data_source_overlap_top": dict(data_source_overlap.most_common(50)),
        "strict_data_source_overlap_top": dict(strict_data_source_overlap.most_common(50)),
        "match_type_counts": dict(match_type_counter),
        "details_csv": str(detail_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Final RL vs SFT overlap audit",
        "",
        f"- RL parquet: `{args.rl_parquet.resolve()}`",
        f"- RL rows audited: `{len(rows)}`",
        f"- SFT rows used for keys: `{sft_summary['sft_total_rows']}`",
        f"- Unique recovered SFT qids: `{sft_summary['unique_sft_qids']}`",
        f"- Unique normalized SFT questions: `{sft_summary['unique_sft_questions']}`",
        f"- Unique SFT document-question keys: `{sft_summary['unique_sft_doc_questions']}`",
        f"- Strict overlapping RL rows: `{sum(strict_component_overlap.values())}` ({pct(sum(strict_component_overlap.values()), len(rows))})",
        f"- Question-only diagnostic matches: `{len(overlaps) - sum(strict_component_overlap.values())}`",
        f"- All diagnostic matches: `{len(overlaps)}` ({pct(len(overlaps), len(rows))})",
        "",
        "## Component breakdown",
        "",
        "| component | rows | strict overlaps | all diagnostic matches | strict pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, count in component_counts.items():
        strict_overlap = strict_component_overlap.get(key, 0)
        all_overlap = component_overlap.get(key, 0)
        lines.append(f"| `{key}` | {count} | {strict_overlap} | {all_overlap} | {pct(strict_overlap, count)} |")

    lines += [
        "",
        "## Answerability breakdown",
        "",
        "| answerability | rows | strict overlaps | all diagnostic matches | strict pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, count in answerability_counts.items():
        strict_overlap = strict_answerability_overlap.get(key, 0)
        all_overlap = answerability_overlap.get(key, 0)
        lines.append(f"| `{key}` | {count} | {strict_overlap} | {all_overlap} | {pct(strict_overlap, count)} |")

    lines += [
        "",
        "## Category breakdown",
        "",
        "| category | rows | strict overlaps | all diagnostic matches | strict pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, count in sorted(category_counts.items()):
        strict_overlap = strict_category_overlap.get(key, 0)
        all_overlap = category_overlap.get(key, 0)
        lines.append(f"| `{key}` | {count} | {strict_overlap} | {all_overlap} | {pct(strict_overlap, count)} |")

    lines += [
        "",
        "## Match-type counts",
        "",
        "| match type | count |",
        "|---|---:|",
    ]
    for key, count in match_type_counter.most_common():
        lines.append(f"| `{key}` | {count} |")

    lines += [
        "",
        "## Top strictly overlapping data sources",
        "",
        "| data_source | rows | strict overlaps | pct |",
        "|---|---:|---:|---:|",
    ]
    for data_source, overlap in strict_data_source_overlap.most_common(30):
        count = data_source_counts[data_source]
        lines.append(f"| `{data_source}` | {count} | {overlap} | {pct(overlap, count)} |")

    lines += [
        "",
        "## Files",
        "",
        f"- Details CSV: `{detail_path}`",
        f"- Summary JSON: `{summary_path}`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
