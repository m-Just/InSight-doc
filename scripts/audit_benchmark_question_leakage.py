#!/usr/bin/env python3
"""Audit benchmark question overlap against SFT and RL training parquets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


REPO_ROOT = Path("/scratch/ywxzml3j/likaican/src/verl-qwen3-vl")
GENERATED_BASE = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
SYN_UNANS_ROOT = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517"


BENCHMARK_FILES = [
    ("broad200_current", "mmlite200", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet"),
    ("broad200_current", "arxiv0307_test102", "/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet"),
    ("broad200_current", "dude200", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet"),
    ("broad200_current", "mmlongbench200", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet"),
    ("broad200_current", "longdocurl200", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet"),
    ("broad200_extra", "mpdocvqa200", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet"),
    ("broad200_extra", "o3bench0502", "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet"),
    ("highpage0507", "longdocurl_highpage_0507", str(REPO_ROOT / "notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet")),
    ("highpage0507", "mmlongbench_highpage_0507", str(REPO_ROOT / "notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet")),
    ("full0504", "longdocurl_full", str(REPO_ROOT / "notes/generated/testcase_0504_full_parquets/longdocurl_full-insight_qwen_agent.parquet")),
    ("full0504", "mpdocvqa_full", str(REPO_ROOT / "notes/generated/testcase_0504_full_parquets/mpdocvqa_full-insight_qwen_agent.parquet")),
    ("full0504", "mmlongbench_full", str(REPO_ROOT / "notes/generated/testcase_0504_full_parquets/mmlongbench_full-insight_qwen_agent.parquet")),
    ("full0504", "dude_full", str(REPO_ROOT / "notes/generated/testcase_0504_full_parquets/dude_full-insight_qwen_agent.parquet")),
    ("full0504", "mmlite_full", str(REPO_ROOT / "notes/generated/testcase_0504_full_parquets/mmlite_full-insight_qwen_agent.parquet")),
]


RL_FILE = (
    REPO_ROOT
    / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e05_plus_arxiv_struct1k_llm_20260722"
    / "insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e_arxiv_struct1k_llm-insight_qwen_agent.parquet"
)


def medium_file(part: str) -> str:
    if part in {
        "O3_data_0424/train_part3a",
        "O3_data_0424/train_part3b",
        "O3_data_0424/train_part3c",
        "O3_data_0424/train_part3d",
    }:
        aspect_filtered = (
            GENERATED_BASE
            / part
            / "medium/processed_gpt5_nano_rewrite_aspect_drop/sft_data_base_model_tool_argument_order.parquet"
        )
        if aspect_filtered.exists():
            return str(aspect_filtered)
    return str(GENERATED_BASE / part / "medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet")


def sft_files() -> list[tuple[str, str]]:
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
    files = [(part, medium_file(part)) for part in parts]
    for scale in ["rescale025", "rescale035", "rescale05"]:
        files.append(
            (
                f"synthetic_unanswerable/{scale}",
                str(SYN_UNANS_ROOT / scale / "medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet"),
            )
        )
    return files


@dataclass
class Record:
    split: str
    dataset: str
    parquet: str
    row_idx: int
    data_source: str
    question_id: str
    document_id: str
    question: str
    norm_question: str
    loose_question: str


def first_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def question_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return clean_question_prefix(first_text_content(msg.get("content")))
    return ""


def clean_question_prefix(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"Image\s+\d+\s*:\s*<image>\s*(?:-{3,}\s*)?", " ", text, flags=re.I)
    text = re.sub(r"(?:<image>\s*)+", " ", text, flags=re.I)
    text = re.sub(r"\n?Output answers only\. No thinking process or explanations?\.?\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*question\s*:\s*", "", text, flags=re.I)
    return text.strip()


def normalize_question(text: str) -> str:
    text = clean_question_prefix(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def loose_question(text: str) -> str:
    text = normalize_question(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_question(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    for key in ["question", "original_question", "original_question_before_llm_rewrite"]:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return clean_question_prefix(value)
    for key in ["messages", "prompt"]:
        value = row.get(key)
        question = question_from_messages(value)
        if question:
            return question
    return ""


def load_records(split: str, dataset: str, parquet: str) -> list[Record]:
    table = pq.read_table(parquet)
    records: list[Record] = []
    for row_idx, row in enumerate(table.to_pylist()):
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        question = row_question(row)
        norm = normalize_question(question)
        records.append(
            Record(
                split=split,
                dataset=dataset,
                parquet=parquet,
                row_idx=row_idx,
                data_source=str(row.get("data_source") or ""),
                question_id=str(extra.get("question_id") or ""),
                document_id=str(extra.get("document_id") or ""),
                question=question,
                norm_question=norm,
                loose_question=loose_question(question),
            )
        )
    return records


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def key_records(records: list[Record], field: str) -> dict[str, list[Record]]:
    keyed: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        value = getattr(rec, field)
        if value:
            keyed[value].append(rec)
    return keyed


def exact_matches(bench: list[Record], train: list[Record], key: str, train_split_name: str) -> list[dict[str, Any]]:
    train_by_key = key_records(train, key)
    matches: list[dict[str, Any]] = []
    for b in bench:
        value = getattr(b, key)
        if not value:
            continue
        for t in train_by_key.get(value, []):
            matches.append(
                {
                    "match_key": key,
                    "match_value": value,
                    "train_split": train_split_name,
                    "benchmark_group": b.split,
                    "benchmark": b.dataset,
                    "benchmark_row": b.row_idx,
                    "benchmark_qid": b.question_id,
                    "benchmark_doc": b.document_id,
                    "train_dataset": t.dataset,
                    "train_row": t.row_idx,
                    "train_qid": t.question_id,
                    "train_doc": t.document_id,
                    "benchmark_question": b.question,
                    "train_question": t.question,
                    "benchmark_parquet": b.parquet,
                    "train_parquet": t.parquet,
                }
            )
    return matches


STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "does",
    "with",
    "from",
    "that",
    "this",
    "into",
    "about",
    "answer",
    "question",
    "select",
    "table",
    "figure",
    "document",
}


def distinctive_tokens(text: str) -> set[str]:
    return {tok for tok in text.split() if len(tok) >= 5 and tok not in STOPWORDS}


def near_matches(
    bench: list[Record],
    train: list[Record],
    train_split_name: str,
    threshold: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    token_index: dict[str, set[int]] = defaultdict(set)
    for idx, rec in enumerate(train):
        for tok in distinctive_tokens(rec.loose_question):
            token_index[tok].add(idx)

    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for b_idx, b in enumerate(bench):
        if not b.loose_question:
            continue
        counter: Counter[int] = Counter()
        for tok in distinctive_tokens(b.loose_question):
            counter.update(token_index.get(tok, ()))
        for t_idx, _ in counter.most_common(max_candidates):
            if (b_idx, t_idx) in seen:
                continue
            seen.add((b_idx, t_idx))
            t = train[t_idx]
            if b.loose_question == t.loose_question:
                continue
            score = SequenceMatcher(None, b.loose_question, t.loose_question, autojunk=False).ratio()
            if score >= threshold:
                output.append(
                    {
                        "similarity": f"{score:.4f}",
                        "train_split": train_split_name,
                        "benchmark_group": b.split,
                        "benchmark": b.dataset,
                        "benchmark_row": b.row_idx,
                        "benchmark_qid": b.question_id,
                        "benchmark_doc": b.document_id,
                        "train_dataset": t.dataset,
                        "train_row": t.row_idx,
                        "train_qid": t.question_id,
                        "train_doc": t.document_id,
                        "benchmark_question": b.question,
                        "train_question": t.question,
                        "benchmark_parquet": b.parquet,
                        "train_parquet": t.parquet,
                    }
                )
    output.sort(key=lambda row: float(row["similarity"]), reverse=True)
    return output


def summarize_records(records: list[Record], label: str) -> list[dict[str, Any]]:
    by_group = Counter((rec.split, rec.dataset) for rec in records)
    return [
        {"collection": label, "group": group, "dataset": dataset, "rows": count}
        for (group, dataset), count in sorted(by_group.items())
    ]


def load_sft_raw_json_records(final_sft: list[Record]) -> list[Record]:
    """Recover SFT provenance from raw trajectory JSONs when present.

    The converted SFT parquet drops `question_id`/`document_id`. We use the
    raw JSONs only if their normalized question appears in the final parquet
    for the same train part, so provenance is restricted to trained rows.
    """
    final_questions_by_part: dict[str, set[str]] = defaultdict(set)
    for rec in final_sft:
        if rec.norm_question:
            final_questions_by_part[rec.dataset].add(rec.norm_question)

    records: list[Record] = []
    for part, _ in sft_files():
        if part.startswith("synthetic_unanswerable/"):
            scale = part.split("/", 1)[1]
            raw_dir = SYN_UNANS_ROOT / scale / "medium/raw_gpt5_nano_rewrite"
        else:
            raw_dir = GENERATED_BASE / part / "medium/raw_gpt5_nano_rewrite"
        if not raw_dir.exists():
            continue
        for json_path in glob.glob(str(raw_dir / "*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            extra = data.get("extra_info") if isinstance(data.get("extra_info"), dict) else {}
            question = clean_question_prefix(str(extra.get("question") or ""))
            norm = normalize_question(question)
            if norm not in final_questions_by_part.get(part, set()):
                continue
            reward = data.get("reward") if isinstance(data.get("reward"), dict) else {}
            records.append(
                Record(
                    split="sft_raw_json",
                    dataset=part,
                    parquet=json_path,
                    row_idx=-1,
                    data_source=str(reward.get("data_source") or ""),
                    question_id=str(extra.get("question_id") or ""),
                    document_id=str(extra.get("document_id") or ""),
                    question=question,
                    norm_question=norm,
                    loose_question=loose_question(question),
                )
            )
    return records


def document_question_matches(bench: list[Record], train: list[Record], train_split_name: str) -> list[dict[str, Any]]:
    train_by_key: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for rec in train:
        if rec.document_id and rec.norm_question:
            train_by_key[(rec.document_id, rec.norm_question)].append(rec)

    matches: list[dict[str, Any]] = []
    for b in bench:
        if not b.document_id or not b.norm_question:
            continue
        for t in train_by_key.get((b.document_id, b.norm_question), []):
            matches.append(
                {
                    "match_key": "document_id+norm_question",
                    "match_value": f"{b.document_id} | {b.norm_question}",
                    "train_split": train_split_name,
                    "benchmark_group": b.split,
                    "benchmark": b.dataset,
                    "benchmark_row": b.row_idx,
                    "benchmark_qid": b.question_id,
                    "benchmark_doc": b.document_id,
                    "train_dataset": t.dataset,
                    "train_row": t.row_idx,
                    "train_qid": t.question_id,
                    "train_doc": t.document_id,
                    "benchmark_question": b.question,
                    "train_question": t.question,
                    "benchmark_parquet": b.parquet,
                    "train_parquet": t.parquet,
                }
            )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "notes/generated/benchmark_question_leak_audit_20260724"))
    parser.add_argument("--near-threshold", type=float, default=0.92)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--skip-near", action="store_true", help="Skip slower near-duplicate review tables.")
    parser.add_argument("--skip-sft-raw", action="store_true", help="Skip raw SFT trajectory JSON provenance recovery.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs = [(group, dataset, path) for group, dataset, path in BENCHMARK_FILES if Path(path).exists()]
    sft_specs = [(name, path) for name, path in sft_files() if Path(path).exists()]

    benchmarks: list[Record] = []
    for group, dataset, path in benchmark_specs:
        benchmarks.extend(load_records(group, dataset, path))

    sft: list[Record] = []
    for dataset, path in sft_specs:
        sft.extend(load_records("sft_ckpt1118_train", dataset, path))

    rl = load_records("rl_train", "rl16k_prompt24k_plus_mc_false_e_arxiv_struct1k_llm", str(RL_FILE))
    sft_raw = [] if args.skip_sft_raw else load_sft_raw_json_records(sft)

    summary_rows = summarize_records(benchmarks, "benchmark")
    summary_rows.extend(summarize_records(sft, "sft"))
    summary_rows.extend(summarize_records(rl, "rl"))
    write_csv(out_dir / "record_counts.csv", summary_rows)

    exact_norm_sft = exact_matches(benchmarks, sft, "norm_question", "sft")
    exact_loose_sft = exact_matches(benchmarks, sft, "loose_question", "sft")
    exact_norm_rl = exact_matches(benchmarks, rl, "norm_question", "rl")
    exact_loose_rl = exact_matches(benchmarks, rl, "loose_question", "rl")
    qid_rl = exact_matches(benchmarks, rl, "question_id", "rl")
    qid_sft_raw = exact_matches(benchmarks, sft_raw, "question_id", "sft_raw_json")
    docq_sft = document_question_matches(benchmarks, sft, "sft")
    docq_sft_raw = document_question_matches(benchmarks, sft_raw, "sft_raw_json")
    docq_rl = document_question_matches(benchmarks, rl, "rl")

    write_csv(out_dir / "exact_norm_matches_sft.csv", exact_norm_sft)
    write_csv(out_dir / "exact_loose_matches_sft.csv", exact_loose_sft)
    write_csv(out_dir / "exact_norm_matches_rl.csv", exact_norm_rl)
    write_csv(out_dir / "exact_loose_matches_rl.csv", exact_loose_rl)
    write_csv(out_dir / "question_id_matches_rl.csv", qid_rl)
    write_csv(out_dir / "question_id_matches_sft_raw_json.csv", qid_sft_raw)
    write_csv(out_dir / "document_question_matches_sft.csv", docq_sft)
    write_csv(out_dir / "document_question_matches_sft_raw_json.csv", docq_sft_raw)
    write_csv(out_dir / "document_question_matches_rl.csv", docq_rl)

    if args.skip_near:
        near_sft: list[dict[str, Any]] = []
        near_rl: list[dict[str, Any]] = []
    else:
        near_sft = near_matches(benchmarks, sft, "sft", args.near_threshold, args.max_candidates)
        near_rl = near_matches(benchmarks, rl, "rl", args.near_threshold, args.max_candidates)
    write_csv(out_dir / "near_matches_sft.csv", near_sft)
    write_csv(out_dir / "near_matches_rl.csv", near_rl)

    def by_group(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(row["benchmark_group"] for row in rows).items()))

    report = {
        "benchmark_files": benchmark_specs,
        "sft_files": sft_specs,
        "rl_file": str(RL_FILE),
        "counts": {
            "benchmark_rows": len(benchmarks),
            "benchmark_unique_norm_questions": len({r.norm_question for r in benchmarks if r.norm_question}),
            "sft_rows": len(sft),
            "sft_unique_norm_questions": len({r.norm_question for r in sft if r.norm_question}),
            "sft_raw_json_rows_matched_to_final_sft_question": len(sft_raw),
            "sft_raw_json_unique_norm_questions": len({r.norm_question for r in sft_raw if r.norm_question}),
            "rl_rows": len(rl),
            "rl_unique_norm_questions": len({r.norm_question for r in rl if r.norm_question}),
        },
        "matches": {
            "near_match_status": "skipped" if args.skip_near else "computed",
            "sft_exact_norm": len(exact_norm_sft),
            "sft_exact_norm_by_benchmark_group": by_group(exact_norm_sft),
            "sft_exact_loose": len(exact_loose_sft),
            "sft_near": len(near_sft),
            "sft_raw_question_id": len(qid_sft_raw),
            "sft_raw_question_id_by_benchmark_group": by_group(qid_sft_raw),
            "sft_document_question": len(docq_sft),
            "sft_raw_document_question": len(docq_sft_raw),
            "rl_exact_norm": len(exact_norm_rl),
            "rl_exact_norm_by_benchmark_group": by_group(exact_norm_rl),
            "rl_exact_loose": len(exact_loose_rl),
            "rl_question_id": len(qid_rl),
            "rl_question_id_by_benchmark_group": by_group(qid_rl),
            "rl_document_question": len(docq_rl),
            "rl_near": len(near_rl),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Benchmark Question Leakage Audit",
        "",
        "This audit compares benchmark questions against the SFT training parquets for `sft_v2_ckpt1118` and the current 19,236-row RL training parquet.",
        "",
        "Matching levels:",
        "- `exact_norm`: case-folded, Unicode-normalized, whitespace-normalized question text after removing image placeholders and answer-only suffixes.",
        "- `exact_loose`: additionally strips punctuation; useful for punctuation-only variants.",
        "- `question_id`: only available for RL/benchmark rows because converted SFT parquets do not retain question IDs.",
        f"- `near`: SequenceMatcher similarity >= {args.near_threshold:.2f} after loose normalization; for manual review only.",
        "",
        "## Counts",
        "",
        f"- Benchmark rows audited: {len(benchmarks)} ({len({r.norm_question for r in benchmarks if r.norm_question})} unique normalized questions)",
        f"- SFT rows audited: {len(sft)} ({len({r.norm_question for r in sft if r.norm_question})} unique normalized questions)",
        f"- Raw SFT JSON provenance rows matched back to final SFT questions: {len(sft_raw)} ({len({r.norm_question for r in sft_raw if r.norm_question})} unique normalized questions)",
        f"- RL rows audited: {len(rl)} ({len({r.norm_question for r in rl if r.norm_question})} unique normalized questions)",
        "",
        "## Match Summary",
        "",
        f"Near-duplicate review status: {'skipped' if args.skip_near else 'computed'}.",
        "",
        "| Train set | exact_norm | exact_loose | question_id | near |",
        "|---|---:|---:|---:|---:|",
        f"| SFT | {len(exact_norm_sft)} | {len(exact_loose_sft)} | n/a | {len(near_sft)} |",
        f"| RL | {len(exact_norm_rl)} | {len(exact_loose_rl)} | {len(qid_rl)} | {len(near_rl)} |",
        "",
        "Additional provenance-backed checks:",
        "",
        f"- SFT raw JSON `question_id` matches: {len(qid_sft_raw)}.",
        f"- SFT final parquet document+question matches: {len(docq_sft)}.",
        f"- SFT raw JSON document+question matches: {len(docq_sft_raw)}.",
        f"- RL document+question matches: {len(docq_rl)}.",
        "",
        "## Output Files",
        "",
        "- `record_counts.csv`: audited row counts by collection and parquet.",
        "- `exact_norm_matches_sft.csv`, `exact_loose_matches_sft.csv`, `near_matches_sft.csv`.",
        "- `exact_norm_matches_rl.csv`, `exact_loose_matches_rl.csv`, `question_id_matches_rl.csv`, `near_matches_rl.csv`.",
        "- `question_id_matches_sft_raw_json.csv`, `document_question_matches_sft.csv`, `document_question_matches_sft_raw_json.csv`, `document_question_matches_rl.csv`.",
        "- `summary.json`: machine-readable summary.",
    ]
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["matches"], indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
