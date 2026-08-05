#!/usr/bin/env python3
"""Evaluate whether final exported answers predict questions as unanswerable.

The script reads exported validation conversations, classifies the final model
answer with an API judge, writes the predicted-unanswerable list, and reports
precision/recall/F1 against extra_info.question_type containing
"not-answerable".
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROMPT_VERSION = "unanswerable_prediction_judge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unanswerable prediction precision/recall from exported validation conversations."
    )
    parser.add_argument("--export-dir", required=True, help="Root directory containing exported conversation JSONs.")
    parser.add_argument(
        "--global-step",
        type=int,
        default=None,
        help="Validation global step to analyze. Uses index/global_step_<N>/val when present.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary and prediction outputs.")
    parser.add_argument("--model", default="gpt-5-nano", help="API judge model.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries after the first API attempt.")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--cache-jsonl",
        default=None,
        help="Successful judge cache. Defaults to OUTPUT_DIR/unanswerable_prediction_cache.jsonl.",
    )
    parser.add_argument(
        "--status-jsonl",
        default=None,
        help="Per-conversation status. Defaults to OUTPUT_DIR/unanswerable_prediction_status.jsonl.",
    )
    parser.add_argument(
        "--predicted-jsonl",
        default=None,
        help="Predicted-unanswerable rows. Defaults to OUTPUT_DIR/predicted_unanswerable.jsonl.",
    )
    parser.add_argument(
        "--all-results-jsonl",
        default=None,
        help="All classified rows. Defaults to OUTPUT_DIR/all_unanswerable_prediction_results.jsonl.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Metrics JSON. Defaults to OUTPUT_DIR/unanswerable_prediction_summary.json.",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Metrics markdown. Defaults to OUTPUT_DIR/unanswerable_prediction_summary.md.",
    )
    parser.add_argument(
        "--subset-key",
        default="subset",
        help=(
            "Primary extra_info key for grouping. Use 'benchmark' to derive dude/mmlongbench/o3bench/arxiv "
            "from extra_info/data_source/question_id prefixes. Otherwise falls back to data_source, source, dataset."
        ),
    )
    parser.add_argument(
        "--truth-key",
        default="question_type",
        help="Primary extra_info key used for true unanswerable labels.",
    )
    parser.add_argument(
        "--truth-fallback-keys",
        default="subset",
        help="Comma-separated extra_info fallback keys used when --truth-key is missing/None/empty.",
    )
    parser.add_argument(
        "--include-subsets",
        default=None,
        help="Comma-separated derived subset/group names to classify. Other rows are skipped before API calls.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N conversations for smoke tests.")
    parser.add_argument("--offset", type=int, default=0, help="Skip N selected conversations before applying --limit.")
    parser.add_argument("--dry-run", action="store_true", help="Load/filter records without calling the API.")
    parser.add_argument(
        "--strict-failures",
        action="store_true",
        help="Exit nonzero when any row cannot be classified. By default skipped rows are reported but tolerated.",
    )
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
        help="Path to InSight-doc repo containing insight_doc.utils.api.",
    )
    parser.add_argument(
        "--ensure-api-logger",
        action="store_true",
        default=True,
        help="Set ENSURE_API_LOGGER=1 before importing API helpers.",
    )
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


def json_contains_not_answerable(value: Any) -> bool:
    return "not-answerable" in json.dumps(value, ensure_ascii=False).lower()


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        pieces: list[str] = []
        for key in ("answer", "text", "value", "think"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
        return "\n\n".join(pieces).strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("value")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces).strip()
    return ""


def extract_initial_question(record: dict[str, Any]) -> str:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"].strip()
    for message in record.get("conversation") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user" and message.get("type") == "query":
            text = text_from_content(message.get("content"))
            if text:
                return text
    return ""


def extract_final_assistant_response(record: dict[str, Any]) -> str:
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if message.get("type") == "tool_call":
            continue
        text = text_from_content(message.get("content"))
        if text:
            return text
    return ""


def extra_value(extra_info: dict[str, Any], key: str) -> Any:
    return extra_info.get(key)


def derive_benchmark(extra_info: dict[str, Any]) -> str:
    for key in ("data_source", "source", "dataset", "benchmark"):
        value = extra_info.get(key)
        if value is not None and str(value).strip():
            text = str(value).strip().lower()
            if "dude" in text:
                return "dude"
            if "mmlong" in text:
                return "mmlongbench"
            if "o3bench" in text:
                return "o3bench"
            if "arxiv" in text:
                return "arxiv"
            return str(value).strip()
    qid = str(extra_info.get("question_id") or "").lower()
    if qid.startswith("dude_") or "dude" in qid:
        return "dude"
    if qid.startswith("mmlongbench_") or "mmlongbench" in qid:
        return "mmlongbench"
    if qid.startswith("o3bench_") or "o3bench" in qid:
        return "o3bench"
    if "arxiv" in qid:
        return "arxiv"
    return "unknown"


def get_subset(extra_info: dict[str, Any], subset_key: str) -> str:
    if subset_key == "benchmark":
        return derive_benchmark(extra_info)
    for key in (subset_key, "data_source", "source", "dataset"):
        value = extra_info.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def true_unanswerable_from_extra_info(extra_info: dict[str, Any], truth_key: str, fallback_keys: str) -> tuple[bool, str | None]:
    candidate_keys = [truth_key]
    candidate_keys.extend(key.strip() for key in fallback_keys.split(",") if key.strip())
    for key in candidate_keys:
        value = extra_value(extra_info, key)
        if value is None or value == "":
            continue
        return json_contains_not_answerable(value), key
    return False, None


def iter_paths_from_index(export_dir: Path, global_step: int | None) -> list[Path]:
    if global_step is None:
        index_root = export_dir / "index"
        if not index_root.exists():
            return []
        index_files = sorted(index_root.glob("global_step_*/val/worker_*.jsonl"))
    else:
        index_files = sorted((export_dir / "index" / f"global_step_{global_step}" / "val").glob("worker_*.jsonl"))
    paths: list[Path] = []
    seen: set[Path] = set()
    for index_path in index_files:
        with index_path.open("r", encoding="utf-8") as handle:
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
    return paths


def iter_paths_by_flat_scan(export_dir: Path, global_step: int | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(export_dir.glob("*.json")):
        record = load_json(path)
        if record is None:
            continue
        job = record.get("job")
        if isinstance(job, dict):
            if not bool(job.get("validate", False)):
                continue
            if global_step is not None and job.get("global_step") != global_step:
                continue
        elif global_step is not None:
            continue
        paths.append(path)
    return paths


def collect_conversation_paths(export_dir: Path, global_step: int | None) -> tuple[list[Path], str]:
    paths = iter_paths_from_index(export_dir, global_step)
    if paths:
        return paths, "index"
    return iter_paths_by_flat_scan(export_dir, global_step), "flat_scan"


def cache_key(*, question: str, answer: str, model: str) -> str:
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "question": question,
            "answer": answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_judge_prompt(question: str, answer: str) -> str:
    return (
        "Determine whether the assistant's final answer is predicting that the question is unanswerable from the "
        "provided document/image, rather than giving a concrete answer.\n\n"
        "Classify as predicted_unanswerable=true when the answer says or clearly implies that the requested "
        "information is not available, not shown, not provided, cannot be determined, insufficiently specified, "
        "or unanswerable as asked.\n"
        "Classify as false when the answer gives a substantive answer, even if it includes uncertainty or caveats.\n\n"
        "Return only strict JSON with keys:\n"
        '{"predicted_unanswerable": boolean, "confidence": number, "rationale": string}\n\n'
        f"Question:\n{question.strip()}\n\n"
        f"Assistant final answer:\n{answer.strip()}\n"
    )


def parse_judge_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        lowered = cleaned.lower()
        if "true" in lowered and "false" not in lowered:
            data = {"predicted_unanswerable": True, "confidence": None, "rationale": cleaned}
        elif "false" in lowered and "true" not in lowered:
            data = {"predicted_unanswerable": False, "confidence": None, "rationale": cleaned}
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("judge output is not a JSON object")
    if not isinstance(data.get("predicted_unanswerable"), bool):
        raise ValueError("judge output missing boolean predicted_unanswerable")
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    return {
        "predicted_unanswerable": bool(data["predicted_unanswerable"]),
        "confidence": confidence,
        "rationale": str(data.get("rationale", "")),
        "raw_output": text,
    }


def load_api_helpers(insight_doc_root: Path, ensure_api_logger: bool):
    if ensure_api_logger:
        os.environ["ENSURE_API_LOGGER"] = "1"
    if str(insight_doc_root) not in sys.path:
        sys.path.insert(0, str(insight_doc_root))
    from insight_doc.utils.api import create_async_openai_client, query_model_with_retry

    return create_async_openai_client, query_model_with_retry


async def judge_unanswerable(
    *,
    prompt: str,
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
            query=prompt,
            model=model,
            client=client,
            context=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict binary classifier for whether a visual-document QA answer claims the "
                        "question is unanswerable. Output only JSON."
                    ),
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
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise RuntimeError("API response did not contain string content")
    return parse_judge_output(content)


def build_result(path: Path, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}
    question = extract_initial_question(record)
    answer = extract_final_assistant_response(record)
    if not question or not answer:
        return None
    true_unanswerable, truth_source_key = true_unanswerable_from_extra_info(
        extra_info,
        args.truth_key,
        args.truth_fallback_keys,
    )
    return {
        "conversation_path": str(path),
        "job": record.get("job") if isinstance(record.get("job"), dict) else {},
        "subset": get_subset(extra_info, args.subset_key),
        "question_id": extra_info.get("question_id"),
        "document_id": extra_info.get("document_id"),
        "data_source": extra_info.get("data_source"),
        "question_type": extra_info.get("question_type"),
        "truth_source_key": truth_source_key,
        "true_unanswerable": true_unanswerable,
        "question": question,
        "final_answer": answer,
    }


def update_counts(counts: dict[str, Counter], subset: str, truth: bool, pred: bool) -> None:
    for key in ("overall", subset):
        if truth and pred:
            counts[key]["tp"] += 1
        elif not truth and pred:
            counts[key]["fp"] += 1
        elif truth and not pred:
            counts[key]["fn"] += 1
        else:
            counts[key]["tn"] += 1
        counts[key]["n"] += 1


def metrics_from_counts(counter: Counter) -> dict[str, Any]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    tn = int(counter["tn"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": int(counter["n"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_unanswerable": tp + fp,
        "true_unanswerable": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Unanswerable Prediction Evaluation",
        "",
        f"- Export dir: `{summary['export_dir']}`",
        f"- Global step: `{summary.get('global_step')}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Path source: `{summary['path_source']}`",
        f"- Classified rows: `{summary['classified_rows']}`",
        f"- Failed rows: `{summary['failed_rows']}`",
        f"- Skipped rows: `{summary.get('skipped_rows', 0)}`",
        "",
        "| subset | n | true_unans | pred_unans | TP | FP | FN | TN | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, metrics in summary["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(subset),
                    str(metrics["n"]),
                    str(metrics["true_unanswerable"]),
                    str(metrics["predicted_unanswerable"]),
                    str(metrics["tp"]),
                    str(metrics["fp"]),
                    str(metrics["fn"]),
                    str(metrics["tn"]),
                    format_float(metrics["precision"]),
                    format_float(metrics["recall"]),
                    format_float(metrics["f1"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache_jsonl) if args.cache_jsonl else output_dir / "unanswerable_prediction_cache.jsonl"
    status_path = Path(args.status_jsonl) if args.status_jsonl else output_dir / "unanswerable_prediction_status.jsonl"
    predicted_path = Path(args.predicted_jsonl) if args.predicted_jsonl else output_dir / "predicted_unanswerable.jsonl"
    all_results_path = Path(args.all_results_jsonl) if args.all_results_jsonl else output_dir / "all_unanswerable_prediction_results.jsonl"
    summary_json_path = Path(args.summary_json) if args.summary_json else output_dir / "unanswerable_prediction_summary.json"
    summary_md_path = Path(args.summary_md) if args.summary_md else output_dir / "unanswerable_prediction_summary.md"

    paths, path_source = collect_conversation_paths(export_dir, args.global_step)
    if args.offset:
        paths = paths[args.offset :]
    if args.limit is not None:
        paths = paths[: args.limit]
    cache = load_cache(cache_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter] = defaultdict(Counter)
    classified_rows = 0
    failed_rows = 0
    skipped_rows = 0
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    include_subsets = None
    if args.include_subsets:
        include_subsets = {item.strip() for item in args.include_subsets.split(",") if item.strip()}

    async def process_one(index: int, path: Path) -> dict[str, Any]:
        try:
            record = load_json(path)
            if record is None:
                return {"status": "failed", "path": str(path), "error": "invalid_json", "input_index": index}
            base = build_result(path, record, args)
            if base is None:
                return {
                    "status": "failed",
                    "path": str(path),
                    "error": "missing_question_or_final_answer",
                    "input_index": index,
                }
            if include_subsets is not None and str(base["subset"]) not in include_subsets:
                return {
                    "status": "skipped",
                    "path": str(path),
                    "reason": "subset_not_included",
                    "subset": base["subset"],
                    "input_index": index,
                }
            key = cache_key(question=base["question"], answer=base["final_answer"], model=args.model)
            if key in cache:
                judge = cache[key]
            elif args.dry_run:
                judge = {"predicted_unanswerable": False, "confidence": None, "rationale": "dry_run", "raw_output": ""}
            else:
                async with semaphore:
                    judge = await judge_unanswerable(
                        prompt=build_judge_prompt(base["question"], base["final_answer"]),
                        model=args.model,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        max_completion_tokens=args.max_completion_tokens,
                        insight_doc_root=Path(args.insight_doc_root),
                        ensure_api_logger=args.ensure_api_logger,
                    )
                append_jsonl(cache_path, {"key": key, "result": judge})
                cache[key] = judge
            return {"status": "ok", **base, "judge": judge, "cache_key": key, "input_index": index}
        except Exception as exc:
            return {"status": "failed", "path": str(path), "error": f"{type(exc).__name__}: {exc}", "input_index": index}

    tasks = [asyncio.create_task(process_one(index, path)) for index, path in enumerate(paths)]
    for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
        try:
            result = await task
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
        append_jsonl(status_path, result)
        if result.get("status") == "skipped":
            skipped_rows += 1
        elif result.get("status") != "ok":
            failed_rows += 1
        else:
            classified_rows += 1
            pred = bool(result["judge"]["predicted_unanswerable"])
            truth = bool(result["true_unanswerable"])
            update_counts(counts, str(result["subset"]), truth, pred)
            append_jsonl(all_results_path, result)
            if pred:
                append_jsonl(predicted_path, result)
        if args.progress_every > 0 and done_count % args.progress_every == 0:
            print(f"processed {done_count}/{len(paths)} classified={classified_rows} failed={failed_rows}", flush=True)

    ordered_metrics: dict[str, dict[str, Any]] = {}
    for subset in ["overall", *sorted(key for key in counts if key != "overall")]:
        if subset in counts:
            ordered_metrics[subset] = metrics_from_counts(counts[subset])
    summary = {
        "schema_version": "unanswerable_prediction_eval_v1",
        "prompt_version": PROMPT_VERSION,
        "export_dir": str(export_dir),
        "global_step": args.global_step,
        "path_source": path_source,
        "judge_model": args.model,
        "subset_key": args.subset_key,
        "truth_key": args.truth_key,
        "truth_fallback_keys": args.truth_fallback_keys,
        "include_subsets": sorted(include_subsets) if include_subsets is not None else None,
        "total_paths": len(paths),
        "offset": args.offset,
        "limit": args.limit,
        "classified_rows": classified_rows,
        "failed_rows": failed_rows,
        "skipped_rows": skipped_rows,
        "outputs": {
            "predicted_jsonl": str(predicted_path),
            "all_results_jsonl": str(all_results_path),
            "status_jsonl": str(status_path),
            "cache_jsonl": str(cache_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
        "metrics": ordered_metrics,
    }
    write_json(summary_json_path, summary)
    write_summary_md(summary_md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict_failures and failed_rows else 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
