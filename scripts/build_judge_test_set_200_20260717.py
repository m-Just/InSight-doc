#!/usr/bin/env python3
"""Build a small judge regression set from train/eval model outputs.

The output is intentionally annotation-oriented: current judge scores are kept
as metadata only, and human labels are left blank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(
    os.environ.get(
        "JUDGE_TEST_OUT_DIR",
        str(REPO / "notes/generated/judge_test_set_200_hard_20260717"),
    )
)
RNG = random.Random(20260717)
RESERVOIR_LIMIT = 500


EVAL_SCORE_FILES = {
    "rl_v4_ckpt700": [
        "workspace/standalone_full_eval_full_rl_v4_ckpt700_rescale025_035_05_1trial_gpu0_3_single_call_v2_20260716_rl_v4_ckpt700_single_call_v2_retry2/rl_v4_ckpt700/full5_tool/rescale025/scores_single_call_v2/samples.jsonl",
        "workspace/standalone_full_eval_full_rl_v4_ckpt700_rescale025_035_05_1trial_gpu0_3_single_call_v2_20260716_rl_v4_ckpt700_single_call_v2_retry2/rl_v4_ckpt700/full5_tool/rescale035/scores_single_call_v2/samples.jsonl",
        "workspace/standalone_full_eval_full_rl_v4_ckpt700_rescale025_035_05_1trial_gpu0_3_single_call_v2_20260716_rl_v4_ckpt700_single_call_v2_retry2/rl_v4_ckpt700/full5_tool/rescale05/scores_single_call_v2/samples.jsonl",
    ],
    "base_no_tool_no_system_clean_20260616": [
        "workspace/standalone_full_eval_num_trials1_base_no_tool_no_system_rescale025_035_05_localmodel_20260616/base_no_tool_no_system_local/full5_no_tool_no_system/rescale025/scores_single_call_v2/samples.jsonl",
        "workspace/standalone_full_eval_num_trials1_base_no_tool_no_system_rescale035_05_localmodel_20260616/base_no_tool_no_system_local/full5_no_tool_no_system/rescale035/scores_single_call_v2/samples.jsonl",
        "workspace/standalone_full_eval_num_trials1_base_no_tool_no_system_rescale035_05_localmodel_20260616/base_no_tool_no_system_local/full5_no_tool_no_system/rescale05/scores_single_call_v2/samples.jsonl",
    ],
    "gpt_5_4_mini": [
        "workspace/standalone_full_eval_full_gpt_https_rescale025_05_conc128_20260615/gpt_5_4_mini/full5_no_tool_no_system/rescale025/scores/samples.jsonl",
        "workspace/standalone_full_eval_full_gpt_https_rescale025_05_conc128_20260615/gpt_5_4_mini/full5_no_tool_no_system/rescale05/scores/samples.jsonl",
    ],
    "gemini_3_flash_clean_20260616": [
        "workspace/standalone_full_eval_num_trials1_gemini_3_flash_rescale05_025_20260616/gemini_3_flash/full5_no_tool_no_system/rescale025/scores/samples.jsonl",
        "workspace/standalone_full_eval_num_trials1_gemini_3_flash_rescale05_025_20260616/gemini_3_flash/full5_no_tool_no_system/rescale05/scores/samples.jsonl",
    ],
}

TRAIN_EXPORT_ROOTS = {
    "rl_training_unans014_e05_from_sft": Path(
        "/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/"
        "insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_"
        "new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_"
        "mc_false_e05_judge_single_call_v1_from_sft"
    ),
    "base_no_tool_no_system_training_run": Path(
        "/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/"
        "insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_"
        "new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_judge_single_call_v1_"
        "base_no_tool_no_system"
    ),
}

MODEL_TARGETS = {
    ("eval", "rl_v4_ckpt700"): 25,
    ("eval", "base_no_tool_no_system_clean_20260616"): 25,
    ("eval", "gpt_5_4_mini"): 25,
    ("eval", "gemini_3_flash_clean_20260616"): 25,
    ("train", "rl_training_unans014_e05_from_sft"): 50,
    ("train", "base_no_tool_no_system_training_run"): 50,
}

GT_UNANSWERABLE_RE = re.compile(
    r"\b(unanswerable|cannot answer|can't answer|can not answer|not answerable|"
    r"not enough information|insufficient information|not provided|not available|"
    r"not specified|not mentioned|unknown|n/?a|image unclear|not visible|"
    r"cannot be determined|can't be determined|there is no|there isn't|no such|"
    r"no answer|none)\b",
    re.I,
)
REFUSAL_RE = re.compile(
    r"^\s*(the|this)?\s*question\s+(as written\s+)?(is\s+)?unanswerable"
    r"|^\s*unanswerable\s+from\s+the\s+provided"
    r"|^\s*(based on [^,.]{0,120}[,.]\s*)?(i\s+)?(cannot|can't|could not|unable to|not able to)\s+"
    r"(determine|answer|identify|read|confirm|provide)"
    r"|^\s*(it is|it's)\s+(not possible|impossible)\s+to\s+(determine|answer|identify|read|confirm)"
    r"|\b(question|requested information|requested value|requested number|answer)\s+"
    r"(cannot|can't)\s+be\s+(determined|answered)\b"
    r"|\b(cannot|can't)\s+determine\b"
    r"|\b(cannot|can't)\s+answer\b"
    r"|\bnot\s+(enough|sufficient)\s+information\s+(to|for)"
    r"|\binsufficient information\s+(to|for)"
    r"|\bneed\s+(a\s+)?(clearer|higher[- ]resolution).{0,80}\b(to|before)\s+"
    r"(determine|answer|identify|read|confirm)",
    re.I,
)
MCQ_RE = re.compile(r"\([A-E]\)")
LETTER_RE = re.compile(r"(?:^|\b|\()([A-E])(?:\)|\b)", re.I)


def loads_line(line: bytes | str) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return json.dumps(value, ensure_ascii=False)


def norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", to_text(value).lower()).strip()


def score_value(score_obj: Any) -> float | None:
    if isinstance(score_obj, dict):
        value = score_obj.get("accuracy_reward", score_obj.get("score"))
    else:
        value = score_obj
    try:
        return float(value)
    except Exception:
        return None


def infer_answerability(data_source: str, ground_truth: Any, is_not_answerable: Any = None) -> str:
    if is_not_answerable is True:
        return "unanswerable"
    if is_not_answerable is False:
        explicit = None
    else:
        explicit = None
    ds = data_source.lower()
    if "unanswerable" in ds or "_unans" in ds:
        explicit = "unanswerable"
    elif "answerable" in ds:
        explicit = "answerable"
    if explicit is not None:
        return explicit
    gt = to_text(ground_truth).strip().lower()
    if gt in {"", "none", "null", "[]"} or GT_UNANSWERABLE_RE.search(gt):
        return "unanswerable"
    return "answerable"


def infer_question_type(question: str, extra_info: dict[str, Any], data_source: str, ground_truth: Any) -> str:
    if extra_info.get("mc_options") or extra_info.get("mc_correct_letter"):
        return "mcq"
    if "_mc_" in data_source.lower() or data_source.lower().endswith("_mc"):
        return "mcq"
    if len(MCQ_RE.findall(question)) >= 2:
        return "mcq"
    gt = to_text(ground_truth).strip()
    if len(gt) == 1 and gt.upper() in set("ABCDE") and len(MCQ_RE.findall(question)) >= 1:
        return "mcq"
    return "non_mcq"


def correct_option(extra_info: dict[str, Any], ground_truth: Any) -> str | None:
    for key in ("mc_correct_letter", "correct_option", "answer"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip().upper() in set("ABCDE"):
            return value.strip().upper()
    gt = to_text(ground_truth).strip()
    if len(gt) == 1 and gt.upper() in set("ABCDE"):
        return gt.upper()
    return None


def final_option(answer: str) -> str | None:
    letters = [m.group(1).upper() for m in LETTER_RE.finditer(answer[:500])]
    return letters[-1] if letters else None


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(" ".join(answer.split())[:320]))


def flags_for(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    answer = row["final_answer"]
    answerability = row["answerability"]
    old_score = row["old_judge_score"]
    question_type = row["question_type"]
    refusal = is_refusal(answer)
    if refusal:
        flags.append("final_refusal")
    if answerability == "answerable" and refusal:
        flags.append("answerable_refusal")
        if old_score == 1.0:
            flags.append("answerable_refusal_scored_correct")
    if answerability == "unanswerable" and not refusal:
        flags.append("unanswerable_non_refusal_answer")
        if old_score == 1.0:
            flags.append("unanswerable_non_refusal_scored_correct")
    opt = row.get("correct_option")
    if question_type == "mcq" and opt:
        ans_upper = answer.upper()
        if f"({opt})" in ans_upper or re.search(rf"\b{re.escape(opt)}\b", ans_upper):
            flags.append("mc_mentions_correct_option")
        fopt = final_option(answer)
        if fopt:
            row["final_option"] = fopt
            if fopt == opt:
                flags.append("mc_final_option_matches_correct")
            else:
                flags.append("mc_final_option_differs_from_correct")
                if old_score == 1.0:
                    flags.append("mc_wrong_final_option_scored_correct")
    gt_norm = norm_text(row["ground_truth"])
    ans_norm = norm_text(answer)
    if gt_norm and len(gt_norm) >= 3 and gt_norm[:80] in ans_norm:
        flags.append("mentions_ground_truth_text")
    if row.get("response_truncated"):
        flags.append("response_truncated")
    if row.get("critical_failure"):
        flags.append("critical_failure")
    return sorted(set(flags))


class Reservoirs:
    def __init__(self) -> None:
        self.pools: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        self.seen: Counter[tuple[str, ...]] = Counter()

    def add(self, row: dict[str, Any]) -> None:
        hard = "hard" if any(
            flag in row["heuristic_flags"]
            for flag in (
                "answerable_refusal_scored_correct",
                "answerable_refusal",
                "unanswerable_non_refusal_answer",
                "mc_wrong_final_option_scored_correct",
                "mc_final_option_differs_from_correct",
            )
        ) else "normal"
        keys = [
            (
                row["source_split"],
                row["model_name"],
                row["answerability"],
                row["question_type"],
                hard,
            ),
            (row["source_split"], row["model_name"], "any", "any", hard),
            (row["source_split"], row["model_name"], row["answerability"], row["question_type"], "any"),
            (row["source_split"], row["model_name"], "any", "any", "any"),
        ]
        for flag in row["heuristic_flags"]:
            keys.append((row["source_split"], row["model_name"], "flag", flag))
            keys.append(("any", "any", "flag", flag))
        for key in keys:
            self._reservoir_add(key, row)

    def _reservoir_add(self, key: tuple[str, ...], row: dict[str, Any]) -> None:
        self.seen[key] += 1
        pool = self.pools[key]
        if len(pool) < RESERVOIR_LIMIT:
            pool.append(row)
            return
        j = RNG.randrange(self.seen[key])
        if j < RESERVOIR_LIMIT:
            pool[j] = row


def make_id(parts: list[Any]) -> str:
    raw = "||".join(to_text(x) for x in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def old_judge_name_from_path(path: str) -> str:
    if "scores_single_call_v2" in path:
        return "single_call_v2"
    if "scores_single_call_v1" in path:
        return "single_call_v1"
    if "/scores/" in path:
        return "existing_scores"
    return "unknown"


def rescale_from_path(path: str) -> str | None:
    if "rescale025" in path:
        return "0.25"
    if "rescale035" in path:
        return "0.35"
    if "rescale05" in path:
        return "0.5"
    return None


def normalize_eval_sample(sample: dict[str, Any], model_name: str, source_file: Path) -> dict[str, Any] | None:
    score_obj = sample.get("score") or {}
    final_answer = score_obj.get("extracted_answer") if isinstance(score_obj, dict) else None
    final_answer = to_text(final_answer or sample.get("solution_str"))
    if not final_answer.strip():
        return None
    extra = sample.get("extra_info") or {}
    question = to_text(extra.get("question"))
    ground_truth = sample.get("ground_truth")
    data_source = to_text(sample.get("data_source"))
    answerability = infer_answerability(data_source, ground_truth, sample.get("is_not_answerable"))
    qtype = infer_question_type(question, extra, data_source, ground_truth)
    copt = correct_option(extra, ground_truth)
    conv_path = to_text(sample.get("conversation_export_json_path"))
    if conv_path and not os.path.isabs(conv_path):
        conv_path = str((REPO / conv_path).resolve())
    row = {
        "id": make_id(["eval", model_name, sample.get("uid"), source_file, sample.get("output_index")]),
        "qa_id": to_text(sample.get("uid") or extra.get("question_id")),
        "source_split": "eval",
        "model_name": model_name,
        "benchmark_or_data_source": data_source,
        "answerability": answerability,
        "question_type": qtype,
        "question": question,
        "ground_truth": to_text(ground_truth),
        "mc_options_json": json.dumps(extra.get("mc_options"), ensure_ascii=False),
        "correct_option": copt,
        "final_option": None,
        "final_answer": final_answer,
        "trajectory_path": conv_path,
        "trajectory_text": "",
        "old_judge_name": old_judge_name_from_path(str(source_file)),
        "old_judge_score": score_value(score_obj),
        "heuristic_flags": [],
        "heuristic_flags_json": "[]",
        "human_label": "",
        "failure_mode_json": "[]",
        "notes": "",
        "rescale": rescale_from_path(str(source_file)),
        "source_path": str(source_file.resolve()),
        "row_origin": "eval_score_file",
        "response_truncated": bool(sample.get("response_truncated")),
        "critical_failure": bool(sample.get("critical_failure")),
        "n_tool_calls": sample.get("n_tool_calls"),
    }
    row["heuristic_flags"] = flags_for(row)
    row["heuristic_flags_json"] = json.dumps(row["heuristic_flags"], ensure_ascii=False)
    return row


def compact_conversation_text(conversation: list[dict[str, Any]], max_chars: int = 6000) -> str:
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content")
        if isinstance(content, dict):
            if content.get("answer"):
                text = f"answer: {content.get('answer')}"
            elif content.get("tool_call"):
                text = f"tool_call: {json.dumps(content.get('tool_call'), ensure_ascii=False)}"
            elif content.get("hint") is not None:
                text = "tool_response"
            else:
                text = json.dumps(content, ensure_ascii=False)
        else:
            text = to_text(content)
        parts.append(f"{role}: {text}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)[:max_chars]


def normalize_train_conversation(conv: dict[str, Any], model_name: str, conv_path: Path) -> dict[str, Any] | None:
    reward = conv.get("reward") or {}
    extra = conv.get("extra_info") or {}
    final_answer = to_text(reward.get("extracted_answer"))
    if not final_answer.strip():
        return None
    question = to_text(extra.get("question"))
    ground_truth = reward.get("ground_truth")
    data_source = to_text(reward.get("data_source"))
    answerability = infer_answerability(data_source, ground_truth)
    qtype = infer_question_type(question, extra, data_source, ground_truth)
    copt = correct_option(extra, ground_truth)
    score_obj = reward.get("score") or reward.get("reward")
    job = conv.get("job") or {}
    row = {
        "id": make_id(["train", model_name, conv_path]),
        "qa_id": to_text(extra.get("question_id") or job.get("root_job_id") or conv_path.stem),
        "source_split": "train",
        "model_name": model_name,
        "benchmark_or_data_source": data_source,
        "answerability": answerability,
        "question_type": qtype,
        "question": question,
        "ground_truth": to_text(ground_truth),
        "mc_options_json": json.dumps(extra.get("mc_options"), ensure_ascii=False),
        "correct_option": copt,
        "final_option": None,
        "final_answer": final_answer,
        "trajectory_path": str(conv_path),
        "trajectory_text": "",
        "old_judge_name": "training_reward_judge",
        "old_judge_score": score_value(score_obj),
        "heuristic_flags": [],
        "heuristic_flags_json": "[]",
        "human_label": "",
        "failure_mode_json": "[]",
        "notes": "",
        "rescale": to_text(extra.get("initial_rescale")),
        "source_path": str(conv_path),
        "row_origin": "train_exported_conversation",
        "response_truncated": False,
        "critical_failure": bool((conv.get("status") or {}).get("critical_failure")),
        "n_tool_calls": (reward.get("score") or {}).get("n_valid_tool_calls") if isinstance(reward.get("score"), dict) else None,
        "global_step": job.get("global_step"),
    }
    row["heuristic_flags"] = flags_for(row)
    row["heuristic_flags_json"] = json.dumps(row["heuristic_flags"], ensure_ascii=False)
    return row


def collect_eval(reservoirs: Reservoirs) -> Counter[str]:
    counts: Counter[str] = Counter()
    for model_name, rel_paths in EVAL_SCORE_FILES.items():
        for rel_path in rel_paths:
            path = REPO / rel_path
            if not path.exists():
                print(f"missing eval score file: {path}")
                continue
            with path.open("rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = normalize_eval_sample(loads_line(line), model_name, path)
                    if row is None:
                        continue
                    reservoirs.add(row)
                    counts[f"eval:{model_name}"] += 1
    return counts


def collect_train(reservoirs: Reservoirs) -> Counter[str]:
    counts: Counter[str] = Counter()
    for model_name, root in TRAIN_EXPORT_ROOTS.items():
        if not root.exists():
            print(f"missing train export root: {root}")
            continue
        index_files = sorted(root.glob("**/index/global_step_*/train/*.jsonl"))
        for i, index_file in enumerate(index_files, 1):
            if i % 1000 == 0:
                print(f"train scan {model_name}: {i}/{len(index_files)} index shards")
            with index_file.open("rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    idx = loads_line(line)
                    conv_path = Path(to_text(idx.get("path")))
                    if not conv_path.exists():
                        continue
                    try:
                        conv = loads_line(conv_path.read_bytes())
                    except Exception:
                        continue
                    row = normalize_train_conversation(conv, model_name, conv_path)
                    if row is None:
                        continue
                    reservoirs.add(row)
                    counts[f"train:{model_name}"] += 1
    return counts


def pick_from_pool(
    reservoirs: Reservoirs,
    key: tuple[str, ...],
    n: int,
    selected_ids: set[str],
    selected_model_qa: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    pool = list(reservoirs.pools.get(key, []))
    RNG.shuffle(pool)
    out: list[dict[str, Any]] = []
    for row in pool:
        if len(out) >= n:
            break
        if row["id"] in selected_ids:
            continue
        model_qa = (row["model_name"], row["qa_id"])
        if model_qa in selected_model_qa:
            continue
        out.append(row)
        selected_ids.add(row["id"])
        selected_model_qa.add(model_qa)
    return out


def sample_rows(reservoirs: Reservoirs) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_model_qa: set[tuple[str, str]] = set()
    strata = [
        ("answerable", "mcq"),
        ("answerable", "non_mcq"),
        ("unanswerable", "mcq"),
        ("unanswerable", "non_mcq"),
    ]
    for (split, model_name), target in MODEL_TARGETS.items():
        if split == "eval":
            # Match the real broad-eval QA mix reasonably closely. In the
            # scanned eval pools, answerable MCQs are ~12.2% and unanswerable
            # MCQs are effectively absent. Hard mining otherwise over-selects
            # MCQ option failures.
            model_rows: list[dict[str, Any]] = []
            eval_quotas = [
                ("answerable", "non_mcq", 20),
                ("answerable", "mcq", 3),
                ("unanswerable", "non_mcq", 2),
            ]
            for answerability, qtype, quota in eval_quotas:
                hard_key = (split, model_name, answerability, qtype, "hard")
                normal_key = (split, model_name, answerability, qtype, "normal")
                got = pick_from_pool(
                    reservoirs,
                    hard_key,
                    max(1, quota // 2),
                    selected_ids,
                    selected_model_qa,
                )
                model_rows.extend(got)
                got2 = pick_from_pool(
                    reservoirs,
                    normal_key,
                    quota - len(got),
                    selected_ids,
                    selected_model_qa,
                )
                model_rows.extend(got2)
                if len(got) + len(got2) < quota:
                    model_rows.extend(
                        pick_from_pool(
                            reservoirs,
                            (split, model_name, answerability, qtype, "any"),
                            quota - len(got) - len(got2),
                            selected_ids,
                            selected_model_qa,
                        )
                    )
            while len(model_rows) < target:
                before = len(model_rows)
                for key in [
                    (split, model_name, "any", "any", "hard"),
                    (split, model_name, "any", "any", "normal"),
                    (split, model_name, "any", "any", "any"),
                ]:
                    need = target - len(model_rows)
                    if need <= 0:
                        break
                    model_rows.extend(pick_from_pool(reservoirs, key, need, selected_ids, selected_model_qa))
                if len(model_rows) == before:
                    break
            selected.extend(model_rows[:target])
            continue

        # Reserve a substantial fraction for known judge failure modes. These
        # rows are rare but much more informative than random examples.
        flag_priorities = [
            ("answerable_refusal_scored_correct", max(4, target // 3)),
            ("unanswerable_non_refusal_scored_correct", max(3, target // 5)),
            ("mc_wrong_final_option_scored_correct", max(2, target // 10)),
            ("mc_final_option_differs_from_correct", max(3, target // 6)),
            ("answerable_refusal", max(3, target // 6)),
        ]
        model_rows: list[dict[str, Any]] = []
        for flag, quota in flag_priorities:
            if len(model_rows) >= target:
                break
            need = min(quota, target - len(model_rows))
            model_rows.extend(
                pick_from_pool(
                    reservoirs,
                    (split, model_name, "flag", flag),
                    need,
                    selected_ids,
                    selected_model_qa,
                )
            )
        per = max(1, target // len(strata))
        for answerability, qtype in strata:
            if len(model_rows) >= target:
                break
            need = per
            hard_key = (split, model_name, answerability, qtype, "hard")
            normal_key = (split, model_name, answerability, qtype, "normal")
            got = pick_from_pool(reservoirs, hard_key, max(1, need // 2), selected_ids, selected_model_qa)
            model_rows.extend(got)
            got2 = pick_from_pool(reservoirs, normal_key, need - len(got), selected_ids, selected_model_qa)
            model_rows.extend(got2)
        while len(model_rows) < target:
            for key in [
                (split, model_name, "any", "any", "hard"),
                (split, model_name, "any", "any", "normal"),
                (split, model_name, "any", "any", "any"),
            ]:
                need = target - len(model_rows)
                if need <= 0:
                    break
                got = pick_from_pool(reservoirs, key, need, selected_ids, selected_model_qa)
                model_rows.extend(got)
            if len(model_rows) >= target:
                break
            break
        selected.extend(model_rows[:target])
    RNG.shuffle(selected)
    return selected[: sum(MODEL_TARGETS.values())]


def hydrate_trajectory_text(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = Path(row["trajectory_path"]) if row.get("trajectory_path") else None
        if not path or not path.exists():
            row["trajectory_text"] = row["final_answer"][:6000]
            continue
        try:
            obj = loads_line(path.read_bytes())
        except Exception:
            row["trajectory_text"] = row["final_answer"][:6000]
            continue
        if isinstance(obj, dict) and "conversation" in obj:
            row["trajectory_text"] = compact_conversation_text(obj.get("conversation") or [])
        else:
            row["trajectory_text"] = row["final_answer"][:6000]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "heuristic_flags"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], collect_counts: Counter[str]) -> None:
    def table(counter: Counter[tuple[Any, ...]], headers: list[str]) -> str:
        lines = ["| " + " | ".join(headers + ["count"]) + " |", "| " + " | ".join(["---"] * (len(headers) + 1)) + " |"]
        for key, count in sorted(counter.items(), key=lambda x: tuple(to_text(v) for v in x[0])):
            vals = key if isinstance(key, tuple) else (key,)
            lines.append("| " + " | ".join(to_text(v) for v in vals + (count,)) + " |")
        return "\n".join(lines)

    by_model = Counter((r["source_split"], r["model_name"]) for r in rows)
    by_slice = Counter((r["source_split"], r["answerability"], r["question_type"]) for r in rows)
    by_flags = Counter(flag for r in rows for flag in json.loads(r["heuristic_flags_json"]))
    by_old_score = Counter((r["old_judge_name"], r["old_judge_score"]) for r in rows)
    content = [
        "# Judge Test Set 200",
        "",
        "Purpose: small human-annotation set for judge regression testing. Current judge scores are metadata, not labels.",
        "",
        f"Rows: {len(rows)}",
        "",
        "## Candidate Inputs Scanned",
        "",
    ]
    for key, count in sorted(collect_counts.items()):
        content.append(f"- `{key}`: {count}")
    content.extend(
        [
            "",
            "## Selected Rows By Source",
            "",
            table(by_model, ["source_split", "model_name"]),
            "",
            "## Selected Rows By QA Slice",
            "",
            table(by_slice, ["source_split", "answerability", "question_type"]),
            "",
            "## Heuristic Flags",
            "",
            table(Counter((k,) for k in by_flags.elements()), ["flag"]),
            "",
            "## Existing Judge Score Metadata",
            "",
            table(by_old_score, ["old_judge_name", "old_judge_score"]),
            "",
            "## Outputs",
            "",
            "- `judge_test_set_200.jsonl`: primary annotation dataset.",
            "- `judge_test_set_200.parquet`: same rows in parquet.",
            "- `judge_test_set_200_annotation.csv`: compact annotation sheet.",
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reservoirs = Reservoirs()
    counts = Counter()
    counts.update(collect_eval(reservoirs))
    counts.update(collect_train(reservoirs))
    rows = sample_rows(reservoirs)
    hydrate_trajectory_text(rows)
    for row in rows:
        row.pop("heuristic_flags", None)
    jsonl_path = OUT_DIR / "judge_test_set_200.jsonl"
    parquet_path = OUT_DIR / "judge_test_set_200.parquet"
    csv_path = OUT_DIR / "judge_test_set_200_annotation.csv"
    write_jsonl(jsonl_path, rows)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    annotation_cols = [
        "id",
        "source_split",
        "model_name",
        "benchmark_or_data_source",
        "answerability",
        "question_type",
        "question",
        "ground_truth",
        "final_answer",
        "old_judge_name",
        "old_judge_score",
        "heuristic_flags_json",
        "human_label",
        "failure_mode_json",
        "notes",
        "trajectory_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=annotation_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_summary(OUT_DIR / "README.md", rows, counts)
    print(f"wrote {len(rows)} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
