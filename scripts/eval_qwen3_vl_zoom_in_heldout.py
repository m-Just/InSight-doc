#!/usr/bin/env python3
"""Evaluate base vs fine-tuned QwenAgentLoop on held-out VEQA rows.

This script:
1. Identifies the question_ids used by the small SFT smoke run from the
   converted train parquet plus the source export JSONs.
2. Excludes those question_ids from the raw VEQA parquet.
3. Runs QwenAgentLoop inference on the held-out rows for two models:
   - a base model
   - a fine-tuned HF checkpoint
4. Scores the final answers with:
   - a fixed local judge model using the same binary correctness prompt as
     `verl.utils.reward_score.vsearch_batch.compute_accuracy_reward`
   - normalized exact match diagnostics

The intended setup matches the currently best smoke configuration:
`system_prompt_mode=vsearcher_qwen3_vl`, plain assistant targets, and
QwenAgentLoop with the qwen3-vl zoom-in tool.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import ray
from hydra import compose, initialize_config_dir
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = REPO_ROOT.parent
EXTRA_REPOS = (WORKSPACE_SRC / "InSight-o3", WORKSPACE_SRC / "Qwen-Agent")
for extra_repo in EXTRA_REPOS:
    extra_repo_str = str(extra_repo)
    if extra_repo.exists() and extra_repo_str not in sys.path:
        sys.path.insert(0, extra_repo_str)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{extra_repo_str}:{current_pythonpath}" if current_pythonpath else extra_repo_str
os.environ.setdefault("VERL_PROJ_DIR", str(REPO_ROOT))

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer
from verl.utils.vsearch_role_play_prompt import qa_verify


VSEARCHER_QWEN3_VL_SYSTEM_PROMPT = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question.""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-parquet", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--train-row-limit", type=int, default=8)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--system-prompt-mode", choices=["vsearcher_qwen3_vl", "exported"], default="vsearcher_qwen3_vl")
    parser.add_argument("--rollout-backend", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_model_ref(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_ref


def parse_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got {uri}")
    return Path(parsed.path)


def load_rl_image(item: Any) -> Image.Image:
    if isinstance(item, dict):
        if "image" in item:
            path = parse_file_uri(item["image"])
            image = Image.open(path)
            image.load()
            return image.convert("RGB")
        if "bytes" in item:
            image = Image.open(io.BytesIO(item["bytes"]))
            image.load()
            return image.convert("RGB")
    if isinstance(item, (bytes, bytearray)):
        image = Image.open(io.BytesIO(item))
        image.load()
        return image.convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(item)}")


def split_text_with_images(text: str, images: list[Image.Image], image_cursor: int) -> tuple[str | list[dict[str, Any]], int]:
    if "<image>" not in text:
        return text, image_cursor

    parts: list[dict[str, Any]] = []
    chunks = text.split("<image>")
    for idx, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if idx < len(chunks) - 1:
            if image_cursor >= len(images):
                raise ValueError("Not enough images to satisfy <image> placeholders")
            parts.append({"type": "image", "image": images[image_cursor]})
            image_cursor += 1
    return parts, image_cursor


def extract_question_from_converted_user_content(content: str) -> str:
    marker = "\n\n"
    if marker in content:
        return content.split(marker)[-1].strip()
    return re.sub(r"(Image \d+:<image>\n---\n?)+", "", content).strip()


def exported_prompt_text_from_json(export_json: dict[str, Any]) -> str:
    for msg in export_json["conversation"]:
        if msg.get("type") == "system_prompt":
            return msg["content"]["text"]
    raise ValueError("No system_prompt found in export JSON")


def resolve_system_prompt(mode: str, export_json: dict[str, Any] | None = None) -> str:
    if mode == "vsearcher_qwen3_vl":
        return VSEARCHER_QWEN3_VL_SYSTEM_PROMPT
    if mode == "exported":
        if export_json is None:
            raise ValueError("export_json is required for exported system prompt mode")
        return exported_prompt_text_from_json(export_json)
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def build_training_question_ids(train_parquet: Path, export_dir: Path, train_row_limit: int) -> list[str]:
    train_df = pd.read_parquet(train_parquet)
    export_question_to_qid = {}
    for path in sorted(export_dir.glob("*.json")):
        data = json.loads(path.read_text())
        export_question_to_qid[data["extra_info"]["question"]] = data["extra_info"]["question_id"]

    question_ids: list[str] = []
    seen = set()
    for idx in range(min(train_row_limit, len(train_df))):
        row = train_df.iloc[idx]
        user_message = next(message for message in row["messages"] if message["role"] == "user")
        question = extract_question_from_converted_user_content(user_message["content"])
        question_id = export_question_to_qid.get(question)
        if question_id is None:
            raise KeyError(f"Could not map training row {idx} back to a question_id")
        if question_id not in seen:
            seen.add(question_id)
            question_ids.append(question_id)
    return question_ids


def build_raw_prompt_from_eval_row(row: pd.Series, system_prompt: str) -> list[dict[str, Any]]:
    prompt_messages = row["prompt"].tolist()
    if len(prompt_messages) < 2:
        raise ValueError("Expected at least system and user messages in raw prompt")

    user_message = prompt_messages[1]
    user_content = user_message["content"]
    flat_images = [load_rl_image(item) for item in row["images"]]
    rebuilt_content, image_cursor = split_text_with_images(user_content, flat_images, 0)
    if image_cursor != len(flat_images):
        raise ValueError(f"Unused images after rebuilding prompt: used={image_cursor}, total={len(flat_images)}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rebuilt_content},
    ]


def extract_final_answer(decoded_assistant_only: str) -> tuple[str | None, bool]:
    segments = [segment.strip() for segment in decoded_assistant_only.split("<|im_end|>") if segment.strip()]
    if not segments:
        return None, False
    last = segments[-1].strip()
    if "<tool_call>" in last:
        return None, False
    return last, True


def exact_match_score(answer: str | None, ground_truth: str) -> int:
    if not answer:
        return 0
    return int(normalize_answer(answer) == normalize_answer(ground_truth))


def substring_match_score(answer: str | None, ground_truth: str) -> int:
    if not answer:
        return 0
    normalized_answer = normalize_answer(answer)
    normalized_gt = normalize_answer(ground_truth)
    return int(normalized_gt in normalized_answer or normalized_answer in normalized_gt)


def make_eval_config(model_path: Path | str, rollout_backend: str):
    with initialize_config_dir(config_dir=os.path.abspath(REPO_ROOT / "verl" / "trainer" / "config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={model_path}",
                f"actor_rollout_ref.rollout.name={rollout_backend}",
                "actor_rollout_ref.rollout.mode=async",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.data_parallel_size=1",
                "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
                "actor_rollout_ref.rollout.enforce_eager=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=32768",
                "actor_rollout_ref.rollout.prompt_length=28672",
                "actor_rollout_ref.rollout.response_length=2048",
                "actor_rollout_ref.rollout.n=1",
                "actor_rollout_ref.rollout.agent.num_workers=1",
                "actor_rollout_ref.rollout.skip_tokenizer_init=True",
                "trainer.nnodes=1",
                "trainer.n_gpus_per_node=1",
                "reward_model.enable=False",
                "reward_model.use_reward_loop=False",
            ],
        )
    return config


def run_model_eval(model_name: str, model_path: Path | str, eval_df: pd.DataFrame, system_prompt_mode: str, export_lookup: dict[str, Any], rollout_backend: str) -> list[dict[str, Any]]:
    print(f"\n[eval] model={model_name} path={model_path}")
    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
    }
    if rollout_backend == "vllm":
        env_vars["VLLM_LOGGING_LEVEL"] = "INFO"
        env_vars["VLLM_USE_V1"] = "1"

    ray.shutdown()
    ray.init(runtime_env={"env_vars": env_vars}, ignore_reinit_error=True)

    config = make_eval_config(model_path=model_path, rollout_backend=rollout_backend)
    agent_loop_manager = AgentLoopManager(config)
    tokenizer = hf_tokenizer(str(model_path))

    results: list[dict[str, Any]] = []
    for eval_idx, row in eval_df.iterrows():
        question_id = row["extra_info"]["question_id"]
        export_json = export_lookup.get(question_id)
        system_prompt = resolve_system_prompt(system_prompt_mode, export_json)
        raw_prompt = build_raw_prompt_from_eval_row(row, system_prompt)

        batch = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array([raw_prompt], dtype=object),
                "agent_name": np.array(["qwen_agent"], dtype=object),
                "data_source": np.array([row["data_source"]], dtype=object),
                "reward_model": np.array([row["reward_model"]], dtype=object),
            }
        )
        result = agent_loop_manager.generate_sequences(prompts=batch)
        num_turns = int(result.non_tensor_batch["__num_turns__"][0])
        responses = result.batch["responses"]
        response_mask = result.batch["response_mask"]
        attention_mask = result.batch["attention_mask"]
        response_length = response_mask.size(1)

        valid_tokens_with_obs = responses[0][attention_mask[0][-response_length:].bool()]
        decoded_with_obs = tokenizer.decode(valid_tokens_with_obs)
        valid_tokens = responses[0][response_mask[0].bool()]
        decoded_assistant_only = tokenizer.decode(valid_tokens)
        final_answer, has_answer = extract_final_answer(decoded_assistant_only)
        tool_calls = decoded_assistant_only.count("<tool_call>")

        results.append(
            {
                "eval_index": int(eval_idx),
                "question_id": question_id,
                "question": row["extra_info"]["question"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "num_turns": num_turns,
                "tool_calls": tool_calls,
                "has_answer": has_answer,
                "final_answer": final_answer,
                "decoded_assistant_only": decoded_assistant_only,
                "decoded_with_obs": decoded_with_obs,
            }
        )

        print(
            f"[eval] {model_name} qid={question_id} turns={num_turns} tool_calls={tool_calls} has_answer={has_answer}"
        )

    ray.shutdown()
    return results


def judge_answers(outputs: list[dict[str, Any]], judge_model: str) -> list[dict[str, Any]]:
    print(f"\n[judge] model={judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=False)
    llm = LLM(model=judge_model, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    prompts = []
    for item in outputs:
        answer = item["final_answer"] if item["final_answer"] else "[NO_ANSWER]"
        user_prompt = qa_verify.format(
            question=item["question"],
            gt_answer=item["ground_truth"],
            model_answer=answer,
        )
        messages = [{"role": "user", "content": user_prompt}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    judge_outputs = llm.generate(prompts, sampling_params)
    judged = []
    for item, judge_output in zip(outputs, judge_outputs, strict=True):
        text = judge_output.outputs[0].text.strip()
        judge_correct = "<correct>" in text.lower()
        judged.append(
            {
                **item,
                "judge_raw": text,
                "judge_correct": judge_correct,
                "exact_match": bool(exact_match_score(item["final_answer"], item["ground_truth"])),
                "substring_match": bool(substring_match_score(item["final_answer"], item["ground_truth"])),
            }
        )
    del llm
    return judged


def summarize_outputs(label: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(outputs)
    answered = sum(int(item["has_answer"]) for item in outputs)
    judge_correct = sum(int(item["judge_correct"]) for item in outputs)
    exact_match = sum(int(item["exact_match"]) for item in outputs)
    substring_match = sum(int(item["substring_match"]) for item in outputs)
    avg_turns = sum(item["num_turns"] for item in outputs) / total if total else 0.0
    avg_tool_calls = sum(item["tool_calls"] for item in outputs) / total if total else 0.0
    return {
        "label": label,
        "n": total,
        "answer_rate": answered / total if total else 0.0,
        "judge_accuracy": judge_correct / total if total else 0.0,
        "exact_match_rate": exact_match / total if total else 0.0,
        "substring_match_rate": substring_match / total if total else 0.0,
        "avg_turns": avg_turns,
        "avg_tool_calls": avg_tool_calls,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_parquet = Path(args.eval_parquet).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    train_parquet = Path(args.train_parquet).expanduser().resolve()
    base_model = resolve_model_ref(args.base_model)
    finetuned_model = resolve_model_ref(args.finetuned_model)
    judge_model = resolve_model_ref(args.judge_model)

    train_question_ids = build_training_question_ids(train_parquet, export_dir, args.train_row_limit)
    print("[train_question_ids]")
    for qid in train_question_ids:
        print(qid)

    eval_df = pd.read_parquet(eval_parquet)
    heldout_rows = [
        row for _, row in eval_df.iterrows() if row["extra_info"]["question_id"] not in set(train_question_ids)
    ]
    heldout_df = pd.DataFrame(heldout_rows)
    if args.max_eval_samples is not None:
        heldout_df = heldout_df.iloc[: args.max_eval_samples].reset_index(drop=True)
    else:
        heldout_df = heldout_df.reset_index(drop=True)
    print(f"\n[heldout] total={len(heldout_df)} excluded_train_questions={len(train_question_ids)}")

    export_lookup = {}
    for path in sorted(export_dir.glob("*.json")):
        data = json.loads(path.read_text())
        export_lookup[data["extra_info"]["question_id"]] = data

    base_outputs = run_model_eval(
        model_name="base",
        model_path=base_model,
        eval_df=heldout_df,
        system_prompt_mode=args.system_prompt_mode,
        export_lookup=export_lookup,
        rollout_backend=args.rollout_backend,
    )
    finetuned_outputs = run_model_eval(
        model_name="finetuned",
        model_path=finetuned_model,
        eval_df=heldout_df,
        system_prompt_mode=args.system_prompt_mode,
        export_lookup=export_lookup,
        rollout_backend=args.rollout_backend,
    )

    base_judged = judge_answers(base_outputs, judge_model)
    finetuned_judged = judge_answers(finetuned_outputs, judge_model)

    base_summary = summarize_outputs("base", base_judged)
    finetuned_summary = summarize_outputs("finetuned", finetuned_judged)

    (work_dir / "base_outputs.json").write_text(json.dumps(base_judged, ensure_ascii=False, indent=2))
    (work_dir / "finetuned_outputs.json").write_text(json.dumps(finetuned_judged, ensure_ascii=False, indent=2))
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_question_ids": train_question_ids,
                "base": base_summary,
                "finetuned": finetuned_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n[summary]")
    print(json.dumps({"base": base_summary, "finetuned": finetuned_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
