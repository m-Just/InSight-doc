import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable
import logging
import re

from openai import AsyncOpenAI

from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer
from verl.utils.vsearch import (
    BBox,
    iou as compute_iou,
    compute_overall_iou_with_gt,
    intersection_area,
    extract_bbox_from_tool_call,
    parse_bbox,
)
from verl.utils.vsearch_role_play_prompt import qa_verify as verify_prompt
from verl.utils.vreasoner_v2_conversation_export import append_reward_info
from verl.utils.vsearch_profile import summarize_numbers, write_profile_event

from insight_o3.utils.api import create_async_openai_client, query_api  # pyright: ignore[reportMissingImports]
from insight_o3 import prompts as insight_o3_prompts  # pyright: ignore[reportMissingImports]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _judge_completion_token_budget(default: int = 2048) -> int:
    raw_value = os.getenv("JUDGE_MAX_COMPLETION_TOKENS")
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid JUDGE_MAX_COMPLETION_TOKENS=%r; using default %s", raw_value, default)
        return default


def _judge_completion_token_kwargs(judge_model: str, default: int = 2048) -> dict[str, int]:
    budget = _judge_completion_token_budget(default)
    if judge_model.lower().startswith("gpt-5"):
        return {"max_completion_tokens": budget}
    return {"max_tokens": budget}


VERIFY_WITH_RAW_FINAL_ANSWER_PROMPT = """
You are given an image/document question, the ground truth (GT) answer, an extracted model answer, and the model's full final answer text.

Decide whether the model answered the question correctly.

Rules:
- Mark <correct> when the final answer gives the same answer as GT. Semantically equivalent wording is okay for descriptive answers.
- Ignore harmless formatting differences: punctuation, capitalization, whitespace, equivalent units, and clearly approximate numeric rounding.
- For names, labels, organizations, IDs, option letters, dates, numbers, and extracted text, require the exact essential identifier. Do not accept spelling/OCR variants, missing characters, or extra characters inside the identifier unless GT explicitly lists aliases.
- A full-name or expanded-name answer is correct for a shorter GT name only when it contains the GT name as an exact token/string and does not identify a different entity.
- For titles, sections, headings, or paragraph-selection questions, the core GT text must be identified or quoted; ignore only location/source/caption context, not broader parent items, narrower child items, neighboring items, or extra distinct selections.
- For list, multi-part, or selection questions, require the exact requested item set: all essential GT items must be present, and broader parent items, merely related items, or extra distinct items are wrong.
- If GT is unanswerable, mark <correct> only when the final answer clearly says the answer cannot be determined, the requested page/table/field/item is absent, no matching item exists, or none satisfy the question.
- If the final answer's conclusion is wrong, unavailable, or contradicted by its own text, mark <wrong> even if it mentions the GT in passing.
- If uncertain, reply with <wrong>.

Only reply with <correct> or <wrong>, no explanations.

Question: {question}
GT Answer: {gt_answer}
Extracted Model Answer: {extracted_answer}
Full Final Answer: {full_final_answer}
""".strip()


VERIFY_FINAL_ANSWER_ONLY_PROMPT = """
You are given an image/document question, the ground truth (GT) answer, and the model's final answer text.

Decide whether the model answered the question correctly.

Rules:
- Mark <correct> when the final answer gives the same answer as GT. Semantically equivalent wording is okay for descriptive answers.
- Ignore harmless formatting differences: punctuation, capitalization, whitespace, equivalent units, and clearly approximate numeric rounding.
- For names, labels, organizations, IDs, option letters, dates, numbers, and extracted text, require the exact essential identifier. Do not accept spelling/OCR variants, missing characters, or extra characters inside the identifier unless GT explicitly lists aliases.
- A full-name or expanded-name answer is correct for a shorter GT name only when it contains the GT name as an exact token/string and does not identify a different entity.
- For titles, sections, headings, or paragraph-selection questions, the core GT text must be identified or quoted; ignore only location/source/caption context, not broader parent items, narrower child items, neighboring items, or extra distinct selections.
- For list, multi-part, or selection questions, require the exact requested item set: all essential GT items must be present, and broader parent items, merely related items, or extra distinct items are wrong.
- If GT is unanswerable, mark <correct> only when the final answer clearly says the answer cannot be determined, the requested page/table/field/item is absent, no matching item exists, or none satisfy the question.
- If the final answer's conclusion is wrong, unavailable, or contradicted by its own text, mark <wrong> even if it mentions the GT in passing.
- If uncertain, reply with <wrong>.

Only reply with <correct> or <wrong>, no explanations.

Question: {question}
GT Answer: {gt_answer}
Final Answer: {full_final_answer}
""".strip()


LEGACY_PROMPT_V2_OPEN_QA_EXTRACTION_PROMPT = """
You are given a question and a model-generated final response.
Your task is to extract the model's final answer for later grading.

Instructions:
- Prefer content inside <answer>...</answer> or \\boxed{{...}}; otherwise take the last decisive span.
- Keep the answer concise, but do not drop requested answer content or a stated no-answer/refusal.
- Do not invent, infer, or repair missing answer content.
- Output only the extracted answer text.

Question: {question}
Model Response: {model_response}

Extracted Answer:
""".strip()


LEGACY_PROMPT_V2_MCQA_EXTRACTION_PROMPT = """
You are given a multiple-choice question with options and a model-generated final answer.
Your task is to extract the option letter selected by the model.

Instructions:
- Output only one option letter, such as A, B, C, D, or E.
- Use the final selected option, not options merely mentioned or rejected in reasoning.
- If no option is selected or the selection is ambiguous, output UNKNOWN.

Question: {question}
Options: {options}
Model Answer: {model_answer}

Selected Option:
""".strip()


LEGACY_PROMPT_V2_QA_VERIFY_PROMPT = """
You are given an image/document question, the ground truth (GT) answer, and a model's extracted answer.

Compare the model's answer with the GT answer.

Rules:
- Mark <correct> only when the model answer gives the same answer as GT. Semantically equivalent wording is okay.
- If GT is answerable and the model answer says it cannot answer or cannot determine the answer, mark <wrong>.
- If GT says the question is not answerable, mark <correct> only when the model answer also says the answer cannot be determined or is absent.
- If uncertain, reply with <wrong>.

Only reply with <correct> or <wrong>, no explanations.

Question: {question}
GT Answer: {gt_answer}
Model Answer: {model_answer}
""".strip()


class ParseError(Exception): pass
class JudgeError(Exception): pass


class RewardComputer:
    """Manages reward computation with asyncio concurrency and timeout handling."""

    def __init__(
        self,
        num_workers: int = 32,
        task_timeout: int = 60,
        min_success_rate: float = 0.85,
        max_retries: int = 10,
        retry_interval: int = 30,
        **kwargs,
    ):
        self.num_workers = num_workers
        self.task_timeout = task_timeout
        self.min_success_rate = min_success_rate
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self._semaphore: asyncio.Semaphore | None = None

    def __enter__(self):
        raise RuntimeError("RewardComputer must be used as an async context manager")

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        self._semaphore = asyncio.Semaphore(self.num_workers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore = None

    async def _run_single(
        self,
        idx: int,
        trial_count: int,
        compute_single_fn: Callable,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        **reward_kwargs: dict,
    ) -> tuple[int, Any, Exception | None, float, str]:
        assert self._semaphore is not None
        t0 = time.perf_counter()
        fn_name = getattr(compute_single_fn, "__name__", str(compute_single_fn))
        judge_model = reward_kwargs.get("judge_model")
        fallback_judge_model = reward_kwargs.get("fallback_judge_model")
        use_fallback_judge = bool(fallback_judge_model) and trial_count == self.max_retries
        active_reward_kwargs = {
            **reward_kwargs,
            "primary_judge_model": judge_model,
            "judge_fallback_used": use_fallback_judge,
        }
        if use_fallback_judge:
            active_reward_kwargs["judge_model"] = fallback_judge_model
        active_judge_model = active_reward_kwargs.get("judge_model")
        try:
            async with self._semaphore:
                result = await asyncio.wait_for(
                    compute_single_fn(
                        data_source,
                        solution_str,
                        ground_truth,
                        extra_info,
                        **active_reward_kwargs,
                    ),
                    timeout=self.task_timeout,
                )
            duration = time.perf_counter() - t0
            write_profile_event(
                "reward_task",
                {
                    "event": "reward_task",
                    "idx": idx,
                    "trial": trial_count,
                    "data_source": data_source,
                    "agent_name": extra_info.get("agent_name"),
                    "job_id": extra_info.get("job_id"),
                    "root_job_id": extra_info.get("root_job_id"),
                    "parent_job_id": extra_info.get("parent_job_id"),
                    "compute_fn": fn_name,
                    "success": True,
                    "judge_model": active_judge_model,
                    "judge_fallback_used": use_fallback_judge,
                    "timing_s": {"task": duration},
                },
            )
            return idx, result, None, duration, fn_name
        except Exception as exc:  # bubble idx back to the collector
            duration = time.perf_counter() - t0
            write_profile_event(
                "reward_task",
                {
                    "event": "reward_task",
                    "idx": idx,
                    "trial": trial_count,
                    "data_source": data_source,
                    "agent_name": extra_info.get("agent_name"),
                    "job_id": extra_info.get("job_id"),
                    "root_job_id": extra_info.get("root_job_id"),
                    "parent_job_id": extra_info.get("parent_job_id"),
                    "compute_fn": fn_name,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "judge_model": active_judge_model,
                    "judge_fallback_used": use_fallback_judge,
                    "timing_s": {"task": duration},
                },
            )
            return idx, None, exc, duration, fn_name

    async def compute_batch(
        self,
        compute_single_fns: list[Callable],
        data_sources: list[str],
        solution_strs: list[str],
        ground_truths: list[str],
        extra_infos: list[dict],
        **reward_kwargs: dict,
    ) -> tuple[list[bool], list[dict[str, Any] | None]]:
        """
        Compute rewards for a batch with retry mechanism to handle network instability.
        Ensures at least `min_success_rate` success rate for verification to maintain training stability.
        """
        if not self._semaphore:
            raise RuntimeError("RewardComputer must be used as async context manager")

        total_samples = len(solution_strs)
        trial_count = 1
        results = [None] * total_samples
        success = [False] * total_samples

        while trial_count <= self.max_retries:
            trial_start_time = time.perf_counter()
            tasks = []
            for i, (data_source, solution_str, ground_truth, extra_info) in enumerate(
                zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
            ):
                if success[i]:
                    continue
                compute_single_fn = compute_single_fns[i]
                task = asyncio.create_task(
                    self._run_single(
                        i,
                        trial_count,
                        compute_single_fn,
                        data_source,
                        solution_str,
                        ground_truth,
                        extra_info,
                        **reward_kwargs,
                    )
                )
                tasks.append(task)

            trial_task_durations = []
            trial_error_types = defaultdict(int)
            trial_compute_fns = defaultdict(int)
            for task in asyncio.as_completed(tasks):
                idx, task_result, error, task_duration, fn_name = await task
                trial_task_durations.append(task_duration)
                trial_compute_fns[fn_name] += 1
                if error is None:
                    results[idx] = task_result
                    success[idx] = True
                    continue

                trial_error_types[type(error).__name__] += 1
                if isinstance(error, JudgeError):
                    active_judge_model = (
                        reward_kwargs.get("fallback_judge_model")
                        if trial_count == self.max_retries and reward_kwargs.get("fallback_judge_model")
                        else reward_kwargs.get("judge_model")
                    )
                    logger.warning(
                        f"[RewardWorker] Task {idx} failed: {error} "
                        f"(judge_model={active_judge_model}, "
                        f"fallback={trial_count == self.max_retries and bool(reward_kwargs.get('fallback_judge_model'))})"
                    )
                elif isinstance(error, asyncio.TimeoutError):
                    logger.warning(f"[RewardWorker] Task {idx} timed out after {self.task_timeout}s")
                else:
                    logger.error(f"[RewardWorker] Task {idx} encountered an unexpected error")
                    raise error

            task_success_rate = sum(success) / total_samples
            num_failed = total_samples - sum(success)
            trial_duration = time.perf_counter() - trial_start_time
            write_profile_event(
                "reward_trial",
                {
                    "event": "reward_trial",
                    "trial": trial_count,
                    "total_samples": total_samples,
                    "attempted_samples": len(tasks),
                    "cumulative_successful": sum(success),
                    "cumulative_failed": num_failed,
                    "success_rate": task_success_rate,
                    "error_types": dict(trial_error_types),
                    "compute_fns": dict(trial_compute_fns),
                    "task_duration_summary_s": summarize_numbers(trial_task_durations),
                    "timing_s": {"trial": trial_duration},
                },
            )
            if task_success_rate >= self.min_success_rate:
                logger.info(
                    f"[RewardWorker] Work finished with "
                    f"{sum(success)} successful, {num_failed} failed, "
                    f"success rate: {task_success_rate:.3f}."
                )
                return success, results

            if trial_count == self.max_retries:
                error_msg = (
                    f"[RewardWorker] ERROR: Failed to achieve required success rates after {self.max_retries} trials. "
                    f"Final task success rate: {task_success_rate:.3f}, "
                    f"Required minimum: {self.min_success_rate:.3f}. "
                    f"Please check your network and API availability."
                )
                raise RuntimeError(error_msg)

            logger.warning(
                f"[RewardWorker] Trial [{trial_count}/{self.max_retries}] has {num_failed} tasks failed, "
                f"success rate {task_success_rate:.3f} below threshold {self.min_success_rate:.3f}. "
                f"Retrying in {self.retry_interval} seconds."
            )
            trial_count += 1

            try:
                await self._interruptible_sleep(self.retry_interval)
            except KeyboardInterrupt:
                logger.info("[RewardWorker] Interrupted during retry wait, shutting down...")
                raise

        return success, results

    async def _interruptible_sleep(self, duration: int):
        """Sleep in small chunks to allow interruption by signals."""
        chunk_size = 1  # Sleep 1 second at a time
        elapsed = 0
        while elapsed < duration:
            await asyncio.sleep(min(chunk_size, duration - elapsed))
            elapsed += chunk_size


@dataclass
class Score:
    score: float
    format_reward: float = 0.0
    accuracy_reward: float = 0.0
    tool_reward: float = 0.0
    iou_reward: float = 0.0

    n_valid_tool_calls: int = 0
    extracted_answer: Any = None
    judge_model_used: str | None = None
    judge_fallback_used: bool = False
    primary_judge_model: str | None = None

    tool_iou: float = 0.0
    final_iou: float = 0.0

    def __init__(self, reward_weights: dict):
        self.reward_weights = reward_weights

    @property
    def score(self) -> float:
        return (
            self.format_reward * self.reward_weights["format"]
            + self.accuracy_reward * self.reward_weights["accuracy"]
            + self.tool_reward * self.reward_weights["tool"]
            + self.iou_reward * self.reward_weights["iou"]
        )

    @property
    def is_correct(self) -> bool:
        return self.accuracy_reward > 0.0


@dataclass
class ScoreConditionedOnToolReward(Score):
    def __init__(self, reward_weights: dict):
        self.reward_weights = reward_weights

    @property
    def score(self) -> float:
        assert self.accuracy_reward * self.iou_reward == 0.0
        return self.tool_reward * (
            self.format_reward * self.reward_weights["format"]
            + self.accuracy_reward * self.reward_weights["accuracy"]
            + self.iou_reward * self.reward_weights["iou"]
        )


@dataclass
class ScoreOnlyAccuracy(Score):
    def __init__(self):
        pass

    @property
    def score(self) -> float:
        return self.accuracy_reward


SCORE_CLASS_MAP = {
    "basic_weighted_addition": Score,
    "conditioned_on_tool_reward": ScoreConditionedOnToolReward,
}


def _record_judge_metadata(score: Score, reward_kwargs: dict) -> None:
    score.judge_model_used = reward_kwargs.get("judge_model")
    score.judge_fallback_used = bool(reward_kwargs.get("judge_fallback_used"))
    score.primary_judge_model = reward_kwargs.get("primary_judge_model") or reward_kwargs.get("judge_model")


def parse_response(
    response: str,
    *,
    required_tags: list[str],
    excluded_tags: list[str] | None = None,
) -> dict[str, str]:
    """Parse a response delimited by the given XML tags.

    Expect the response to be in the following format (with the order following `required_tags`):
        <required_tag_1>content_1</required_tag_1>
        <required_tag_2>content_2</required_tag_2>
        ...
        <required_tag_n>content_n</required_tag_n>
    Only whitespace characters are allowed before and after each tag.
    Tags in `excluded_tags` must not appear in the response.

    Args:
        - response: the response to parse
        - required_tags: the tags that must appear exactly once, in the exact order
        - excluded_tags (optional): the tags that must not appear in the response

    Returns:
        - A dictionary of the parsed content, with the keys being the tag names and the values being the contents.

    Raises:
        - ParseError: if the response is not in the expected format.
    """

    def contain_tag(response: str, tag: str) -> bool:
        return f"<{tag}>" in response or f"</{tag}>" in response

    # Check if the excluded tags appear in the response
    if excluded_tags is None:
        excluded_tags = []
    for tag in excluded_tags:
        if contain_tag(response, tag):
            raise ParseError(f"parse_response: excluded tag {tag} appears in the response")

    parsed_content = {}

    for tag in required_tags:
        # Check if the opening tag is present
        response = response.lstrip()
        if not response.startswith(f"<{tag}>"):
            raise ParseError(f"parse_response: missing opening tag for {tag}")

        # Remove the opening tag
        response = response[len(f"<{tag}>") :]

        # Ensure that the closing tag is present
        response_chunks = response.split(f"</{tag}>", 1)
        if len(response_chunks) == 1:
            raise ParseError(f"parse_response: missing closing tag for {tag}")

        # Separate the content from the remaining response
        content, response = response_chunks

        # Ensure that the content does not contain any other required tags
        for tag_other in required_tags:
            if contain_tag(content, tag_other):
                raise ParseError(f"parse_response: nested tags detected in {tag}")

        parsed_content[tag] = content

    # Ensure there is no unexpected content after the last pair of tags
    if response.strip():
        raise ParseError("parse_response: unexpected content after the last pair of tags")

    return parsed_content


def _extract_last_tagged_content(text: str, tag: str) -> str | None:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
        return None
    return text.rsplit(start_tag, 1)[1].split(end_tag, 1)[0].strip()


def _extract_last_boxed_answer(text: str) -> str | None:
    if "\\boxed{" not in text:
        return None
    return text.rsplit("\\boxed{", 1)[1].split("}", 1)[0].strip()


def _extract_responses(solution_str: str) -> list[str]:
    return [round.split("assistant\n")[-1] for round in solution_str.split("user\n")]


def _extract_last_plain_answer(text: str) -> str | None:
    responses = _extract_responses(text)
    if not responses:
        return None
    candidate = responses[-1].strip()
    candidate = candidate.removesuffix("<|im_end|>").strip()
    if not candidate or "<tool_call>" in candidate or "</tool_call>" in candidate:
        return None
    return candidate


def _extract_last_assistant_message(solution_str: str) -> str | None:
    responses = _extract_responses(solution_str)
    if not responses:
        return None
    candidate = responses[-1].strip()
    candidate = candidate.removesuffix("<|im_end|>").strip()
    return candidate or None


def _crop_judge_answer_context(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    crop_marker = " ... "
    kept_chars = max(1, max_len - len(crop_marker))
    prefix_chars = kept_chars // 2
    suffix_chars = kept_chars - prefix_chars
    return f"{text[:prefix_chars]}{crop_marker}{text[-suffix_chars:]}"


def _normalize_judge_extracted_answer(text: str | None) -> str | None:
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None

    tagged = re.findall(r"<answer>(.*?)</answer>", candidate, re.DOTALL)
    if tagged:
        candidate = tagged[-1].strip()
    else:
        candidate = candidate.removeprefix("<answer>").strip()

    boxed = re.findall(r"\\boxed\{(.*?)\}", candidate, re.DOTALL)
    if boxed:
        candidate = boxed[-1].strip()

    candidate = candidate.strip().strip("`").strip()
    return candidate or None


def _extract_raw_final_answer_for_judge(final_message: str) -> str:
    raw_answer = _extract_last_tagged_content(final_message, "answer")
    if raw_answer is None:
        raw_answer = _extract_last_boxed_answer(final_message)
    if raw_answer is None:
        raw_answer = final_message.strip()
    return _crop_judge_answer_context(raw_answer)


def _is_not_answerable_text(text: str) -> bool:
    normalized = normalize_answer(text)
    return any(
        phrase in normalized
        for phrase in (
            "not answerable",
            "cannot answer",
            "can not answer",
            "cannot be determined",
            "can not be determined",
            "cannot determine",
            "can not determine",
            "not provided",
            "not available",
            "not explicitly derived",
            "cannot be explicitly derived",
            "can not be explicitly derived",
        )
    )


def _raw_final_answer_matches_not_answerable(full_final_answer: str) -> bool:
    normalized = normalize_answer(full_final_answer)
    if _is_not_answerable_text(full_final_answer):
        return True
    return bool(
        re.search(r"\bnone\b", normalized)
        or re.search(r"\bno (?:listed )?(?:candidate|item|person|president|option|answer|matching item)\b", normalized)
        or re.search(r"\bno [a-z0-9 ]{0,80}\b(?:satisf(?:y|ies|ied)|match(?:es|ed)?|had|has)\b", normalized)
    )


def _raw_final_answer_is_hierarchy_leaf_only(ground_truth: str, full_final_answer: str) -> bool:
    """Detect path-like GTs where the answer only gives the leaf title."""

    if ">" not in str(ground_truth):
        return False
    parts = [normalize_answer(part) for part in str(ground_truth).split(">")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return False
    normalized_answer = normalize_answer(full_final_answer)
    if not normalized_answer:
        return False
    leaf = parts[-1]
    parents = parts[:-1]
    return leaf in normalized_answer and any(parent not in normalized_answer for parent in parents)


def _parse_mc_options_from_question(question: str) -> dict[str, str]:
    option_matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9])(?:\(([A-E])\)(?:[.:])?\s*|([A-E])\.\s+)",
            question,
        )
    )
    if len(option_matches) < 2:
        return {}

    labels = [(match.group(1) or match.group(2)) for match in option_matches]
    unique_labels = list(dict.fromkeys(labels))
    expected_labels = [chr(ord("A") + idx) for idx in range(len(unique_labels))]
    if unique_labels != expected_labels:
        return {}

    options: dict[str, str] = {}
    for i, match in enumerate(option_matches):
        label = labels[i]
        start = match.end()
        end = option_matches[i + 1].start() if i + 1 < len(option_matches) else len(question)
        option_text = question[start:end].strip()
        if option_text:
            options[label] = option_text
    return options


def _parse_mc_options_from_question_v1(question: str) -> dict[str, str]:
    """Historical single_call_v1 MC parser kept for provenance.

    It only recognizes parenthesized options and is intentionally less robust
    than the v2 parser.
    """

    option_matches = list(re.finditer(r"\(([A-E])\)\s*", question))
    if len(option_matches) < 2:
        return {}

    options: dict[str, str] = {}
    for i, match in enumerate(option_matches):
        label = match.group(1)
        start = match.end()
        end = option_matches[i + 1].start() if i + 1 < len(option_matches) else len(question)
        option_text = question[start:end].strip()
        if option_text:
            options[label] = option_text
    return options


def _raw_final_answer_matches_mc_ground_truth_v1(
    *,
    question: str,
    ground_truth: str,
    full_final_answer: str,
) -> bool | None:
    """Historical single_call_v1 MC rule kept for reproducibility.

    This preserves the known bug where mentioning the GT option anywhere can
    mark the row correct, even if the answer later rejects it.
    """

    gt = ground_truth.strip().upper()
    if not re.fullmatch(r"[A-E]", gt):
        return None

    options = _parse_mc_options_from_question_v1(question)
    if gt not in options:
        return None

    answer_text = full_final_answer.strip()
    upper_answer = answer_text.upper()
    explicit_labels = re.findall(r"\(([A-E])\)", upper_answer)
    explicit_labels.extend(
        re.findall(
            r"\b(?:ANSWER|OPTION|CHOICE|SELECTED|SELECT|CORRECT ANSWER|BEST ANSWER)\b\s*(?:IS|:)?\s*([A-E])\b",
            upper_answer,
        )
    )
    if explicit_labels:
        return gt in {label.upper() for label in explicit_labels}

    if re.fullmatch(rf"(?:\(?{gt}\)?|(?:answer|option|choice)\s*\(?{gt}\)?)", answer_text, flags=re.IGNORECASE):
        return True

    normalized_answer = normalize_answer(full_final_answer)
    normalized_option = normalize_answer(options[gt])
    if normalized_option and normalized_option in normalized_answer:
        return True

    return False


def _extract_explicit_mc_selected_label(answer_text: str) -> str | None:
    """Extract a final selected MC label from high-precision answer cues.

    This intentionally does not treat every ``(A)`` mention as a selection.
    Model explanations often compare or reject options before giving the final
    answer; accepting any GT-label mention creates false positives.
    """

    candidates: list[tuple[int, str]] = []
    patterns = (
        r"\b(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER|BEST\s+ANSWER|CORRECT\s+OPTION|BEST\s+OPTION|SELECTED\s+OPTION|CORRECT\s+CHOICE|BEST\s+CHOICE|SELECTED\s+CHOICE|MY\s+ANSWER)\b\s*(?:IS|:|=)?\s*\(?([A-E])\)?\b",
        r"\b(?:CORRECT|FINAL)\b[^.\n]{0,120}?\b(?:IS|:|=)\s*\(?([A-E])\)?\b",
        r"\b(?:CHOOSE|SELECT|PICK|GO\s+WITH)\b\s*(?:OPTION|CHOICE)?\s*\(?([A-E])\)?\b",
        r"\b(?:MATCHES|CORRESPONDS\s+TO|MAPS\s+TO)\b\s*(?:OPTION|CHOICE)\s*\(?([A-E])\)?\b",
        r"\b(?:THEREFORE|THUS|SO|HENCE|OVERALL)\b[^.\n]{0,120}?\b(?:ANSWER|OPTION|CHOICE)\b\s*(?:IS|:|=)?\s*\(?([A-E])\)?\b",
    )
    upper_answer = answer_text.upper().replace("*", "").replace("`", "")
    for pattern in patterns:
        for match in re.finditer(pattern, upper_answer):
            if _mc_selection_context_is_negative(upper_answer, match.start(), match.end()):
                continue
            candidates.append((match.start(), match.group(1).upper()))

    tail_match = re.search(r"\(([A-E])\)[).:]?(?:\s+[^()]{0,120})?\s*$", upper_answer)
    if tail_match:
        before = upper_answer[max(0, tail_match.start() - 40) : tail_match.start()]
        after = upper_answer[tail_match.end() :]
        earlier_parenthesized_labels = re.findall(r"\(([A-E])\)", upper_answer[: tail_match.start()])
        if not earlier_parenthesized_labels and not re.search(
            r"\b(?:NOT|INCORRECT|WRONG|EXCEPT|RATHER\s+THAN)\b",
            before + " " + after,
        ):
            candidates.append((tail_match.start(), tail_match.group(1).upper()))

    # A final line like "(E) All the above answers are wrong" is also an
    # explicit selection, but a mid-answer option mention/list is not.
    nonempty_lines = [(start, line.strip()) for start, line in _iter_answer_lines_with_offsets(upper_answer) if line.strip()]
    if nonempty_lines:
        line_start, stripped = nonempty_lines[-1]
        match = re.fullmatch(
            r"(?:\(([A-E])\)|([A-E])(?:[).:]|$)|(?:OPTION|CHOICE)\s*\(?([A-E])\)?)(?:\s+.*)?",
            stripped,
        )
        mentioned_line_labels = re.findall(r"\(([A-E])\)|\b(?:OPTION|CHOICE)\s*\(?([A-E])\)?\b", stripped)
        mentioned_line_labels = [match[0] or match[1] for match in mentioned_line_labels]
        if (
            match
            and len(mentioned_line_labels) <= 1
            and not _mc_selection_context_is_negative(stripped, match.start(), match.end())
        ):
            candidates.append((line_start, (match.group(1) or match.group(2) or match.group(3)).upper()))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _iter_answer_lines_with_offsets(text: str):
    offset = 0
    for line in text.splitlines():
        yield offset, line
        offset += len(line) + 1


def _mc_selection_context_is_negative(upper_text: str, start: int, end: int) -> bool:
    """Return true when an option mention is clearly being rejected, not selected."""

    before = upper_text[max(0, start - 40) : start]
    after = upper_text[end : min(len(upper_text), end + 80)]
    context = before + " " + after
    return bool(
        re.search(r"\b(?:NOT|NO|NEITHER|INCORRECT|WRONG|FALSE|REJECT(?:ED)?|EXCLUDE(?:D)?)\b", before)
        or re.search(
            r"^\s*\)?[).:]?\s*(?:IS|ARE|WAS|WERE|SEEMS|APPEARS|WOULD\s+BE)?\s*"
            r"(?:NOT\s+(?:CORRECT|ACCURATE|APPLICABLE|RIGHT|THE\s+ANSWER|A\s+GOOD\s+ANSWER|A\s+MATCH)|"
            r"INCORRECT|WRONG|FALSE|LESS\s+PRECISE|PARTIALLY\s+CORRECT|NOT\s+PERFECT)",
            after,
        )
    )


def _strip_direct_answer_prefix(answer_text: str) -> str:
    stripped = answer_text.strip().strip("`").strip()
    return re.sub(
        r"^(?:final\s+answer|answer|the\s+answer|model\s+answer)\s*(?:is|:|=)?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()


def _is_concise_mc_answer(answer_text: str) -> bool:
    words = re.findall(r"\w+", answer_text)
    return len(words) <= 32 and len(answer_text.strip()) <= 240


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", text))


def _mc_option_is_rejection_option(option_text: str) -> bool:
    normalized = normalize_answer(option_text)
    return (
        _is_not_answerable_text(option_text)
        or "all above answers are wrong" in normalized
        or "all the above answers are wrong" in normalized
        or "none of the above" in normalized
        or "none of the provided options" in normalized
        or "none of the listed options" in normalized
        or "none of the options" in normalized
        or "does not provide sufficient information" in normalized
        or "does not feature" in normalized
        or "image does not" in normalized
    )


def _match_concise_answer_to_mc_options(answer_text: str, options: dict[str, str]) -> tuple[str, bool] | None:
    direct_answer = _strip_direct_answer_prefix(answer_text)
    normalized_answer = normalize_answer(direct_answer)
    if not normalized_answer:
        return None

    matches: list[tuple[str, bool]] = []
    for label, option_text in options.items():
        normalized_option = normalize_answer(option_text)
        if not normalized_option:
            continue
        exact_match = normalized_answer == normalized_option
        loose_match = _contains_normalized_phrase(normalized_option, normalized_answer) or _contains_normalized_phrase(
            normalized_answer,
            normalized_option,
        )
        if exact_match or loose_match:
            matches.append((label, exact_match))

    matched_labels = {label for label, _ in matches}
    if len(matched_labels) == 1:
        return matches[0]
    return None


def _raw_final_answer_matches_mc_ground_truth(
    *,
    question: str,
    ground_truth: str,
    full_final_answer: str,
) -> bool | None:
    gt = ground_truth.strip().upper()
    if not re.fullmatch(r"[A-E]", gt):
        return None

    options = _parse_mc_options_from_question(question)
    if gt not in options:
        return None

    answer_text = full_final_answer.strip()
    selected_label = _extract_explicit_mc_selected_label(answer_text)
    if selected_label is not None:
        return selected_label == gt

    if _mc_option_is_rejection_option(answer_text) and _mc_option_is_rejection_option(options.get("E", "")):
        return gt == "E"

    explicit_label_mentions = re.findall(r"\(([A-E])\)|\b(?:OPTION|CHOICE)\s*([A-E])\b", answer_text.upper())
    mentioned_labels = {(match[0] or match[1]).upper() for match in explicit_label_mentions if match[0] or match[1]}
    if _is_concise_mc_answer(answer_text) and len(mentioned_labels) <= 1:
        matched_label = _match_concise_answer_to_mc_options(answer_text, options)
        if matched_label is not None:
            label, exact_match = matched_label
            if label == gt:
                return True
            if exact_match:
                return None

    return None


def _looks_like_list_ground_truth(ground_truth: str) -> bool:
    stripped = ground_truth.strip()
    if stripped.startswith("[") or stripped.startswith("(") or stripped.startswith("{"):
        return True
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if isinstance(parsed, str):
        parsed_stripped = parsed.strip()
        return parsed_stripped.startswith("[") or parsed_stripped.startswith("(") or parsed_stripped.startswith("{")
    return isinstance(parsed, (list, tuple, dict))


def _parse_list_ground_truth(ground_truth: str) -> list[str] | None:
    stripped = ground_truth.strip()
    for candidate in (stripped,):
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                return None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None


def _section_like_ids(text: str) -> set[str]:
    ids = set()
    for match in re.finditer(r"\b\d+(?:\.\d+)+\b", text):
        ids.add(match.group(0).replace(".", ""))
    return ids


def _is_title_or_section_selection_question(question: str) -> bool:
    normalized = normalize_answer(question)
    return bool(
        ("title" in normalized or "section" in normalized or "heading" in normalized)
        and ("select" in normalized or "which" in normalized or "best match" in normalized)
    )


def _raw_final_answer_matches_scalar_title_ground_truth(
    *,
    question: str,
    ground_truth: str,
    full_final_answer: str,
) -> bool:
    if _looks_like_list_ground_truth(ground_truth) or not _is_title_or_section_selection_question(question):
        return False

    normalized_gt = normalize_answer(ground_truth)
    normalized_answer = normalize_answer(full_final_answer)
    if not normalized_gt or not normalized_answer:
        return False
    if re.search(
        rf"\b(?:containing|contains|page containing|specifically|section containing|chapter containing)\b.{{0,80}}\b{re.escape(normalized_gt)}\b",
        normalized_answer,
    ):
        return False
    if normalized_gt in normalized_answer:
        return True

    # Accept scalar title/header answers that preserve a long core suffix while
    # dropping source/owner prefixes from the formal GT title.
    gt_tokens = normalized_gt.split()
    for start in range(1, len(gt_tokens) - 1):
        suffix = " ".join(gt_tokens[start:])
        if len(suffix) >= 18 and suffix in normalized_answer:
            return True
    return False


def _raw_final_answer_matches_list_ground_truth(
    *,
    question: str,
    ground_truth: str,
    full_final_answer: str,
) -> bool:
    items = _parse_list_ground_truth(ground_truth)
    if not items or not _is_title_or_section_selection_question(question):
        return False

    normalized_answer = normalize_answer(full_final_answer)
    normalized_gt = normalize_answer(ground_truth)
    if re.search(r"(?m)^\s*[-*]\s+", full_final_answer):
        return False
    if "section" not in normalized_gt and re.search(r"\bsection\s+\d+\b", normalized_answer):
        return False

    answer_tokens = set(normalized_answer.split())
    for item in items:
        item_tokens = [
            token
            for token in normalize_answer(item).split()
            if len(token) > 2 and not any(ch.isdigit() for ch in token) and token not in {"and", "the"}
        ]
        if not item_tokens:
            return False
        covered = sum(token in answer_tokens for token in item_tokens)
        if covered / len(item_tokens) < 0.70:
            return False

    gt_section_ids = set().union(*(_section_like_ids(item) for item in items))
    answer_section_ids = _section_like_ids(full_final_answer)
    if gt_section_ids and not answer_section_ids.issubset(gt_section_ids):
        return False

    return True


def _is_entity_identity_question(question: str) -> bool:
    normalized = normalize_answer(question)
    return bool(
        re.search(r"\bwho\b", normalized)
        or any(
            phrase in normalized
            for phrase in (
                "signator",
                "signatory",
                "signatories",
                "author",
                "auditor",
                "representative",
                "signed by",
            )
        )
    )


def _raw_final_answer_matches_entity_ground_truth(
    *,
    question: str,
    ground_truth: str,
    full_final_answer: str,
) -> bool:
    if _looks_like_list_ground_truth(ground_truth) or not _is_entity_identity_question(question):
        return False

    ignored_tokens = {
        "and",
        "for",
        "on",
        "behalf",
        "of",
        "the",
        "senior",
        "statutory",
        "auditor",
        "chartered",
        "accountant",
        "accountants",
        "representative",
        "author",
        "signed",
        "signatory",
        "signatories",
    }
    gt_tokens = [
        token
        for token in normalize_answer(ground_truth).split()
        if len(token) > 1 and token not in ignored_tokens
    ]
    if len(gt_tokens) < 2:
        return False

    answer_tokens = set(normalize_answer(full_final_answer).split())
    covered = sum(token in answer_tokens for token in gt_tokens)
    return covered / len(gt_tokens) >= 0.8


async def _extract_answer_for_insight_qwen_agent(
    *,
    question: str,
    final_message: str,
    extra_info: dict,
    judge_client: AsyncOpenAI,
    judge_model: str,
    prompt_version: str = "legacy",
) -> str | None:
    raw_answer = _extract_raw_final_answer_for_judge(final_message)
    if not raw_answer:
        return None

    options = extra_info.get("options")
    if options:
        prompt_template = (
            LEGACY_PROMPT_V2_MCQA_EXTRACTION_PROMPT
            if prompt_version == "legacy_prompt_v2"
            else insight_o3_prompts.GPT_EVAL_MCQA_PROMPT
        )
        prompt = prompt_template.format(
            question=question,
            options=options,
            model_answer=raw_answer,
        )
    else:
        prompt_template = (
            LEGACY_PROMPT_V2_OPEN_QA_EXTRACTION_PROMPT
            if prompt_version == "legacy_prompt_v2"
            else insight_o3_prompts.GPT_EVAL_OPEN_QA_PROMPT
        )
        prompt = prompt_template.format(
            question=question,
            model_response=raw_answer,
        )

    try:
        _, response = await query_api(
            query=prompt,
            model=judge_model,
            client=judge_client,
            **_judge_completion_token_kwargs(judge_model),
        )
    except Exception as e:
        raise JudgeError(f"answer extraction judge failed: {e}") from e

    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    elif content is None:
        raise JudgeError("answer extraction judge returned empty content")

    extracted_answer = _normalize_judge_extracted_answer(str(content))
    return extracted_answer or raw_answer


def compute_format_reward(solution_str: str, must_have_answer: bool = True) -> dict:
    """Compute the format reward of the model's conversation with the user.
       We assume that the user messages are all tool responses.

    The conversation may contain multiple rounds of exchanges between the model and the user.
    At each round, the model can choose to either
        a. call a tool; or
        b. answer the query.

    We reward the model if its response follows this format in every round:
        <think>...</think>
        <tool_call>...</tool_call> (if a tool is needed)
        <answer>...</answer>       (otherwise)

    In addition:
        - Every response except the last one must end with <tool_call>...</tool_call>.
        - If must_have_answer is True, the last response must end with <answer>...</answer>.

    Args:
        - solution_str: a conversation between the model and the user (excluding the system prompt
            and the initial user prompt), formatted as follows:
                [... initial_model_response ...]\n
                user\n
                [... tool_response ...]\n
                assistant\n
                [... model_response ...]\n
                ...
        - must_have_answer: whether the model must answer with <answer>...</answer> in the last response.

    Returns:
        - format_reward: a float that is either 0.0 or 1.0
    """

    # Extract the responses from the conversation
    responses = _extract_responses(solution_str)

    # Check if every response strictly follows the format
    for i, response in enumerate(responses):
        # Check if this is a tool-call response
        try:
            parse_response(response, required_tags=["think", "tool_call"], excluded_tags=["answer"])
        except ParseError:
            is_tool_call = False
        else:
            is_tool_call = True

        # If this is not a tool-call response, see if it is the last response
        # If it is the last response, it should have an answer; otherwise, the format is wrong
        if not is_tool_call:
            if i + 1 == len(responses):
                must_have_answer = True
            else:
                return 0.0

    # If must_have_answer is True, the last response must end with <answer>...</answer>
    if must_have_answer:
        try:
            parse_response(responses[-1], required_tags=["think", "answer"], excluded_tags=["tool_call"])
        except ParseError:
            return 0.0

    return 1.0


def compute_format_reward_simple(solution_str: str, must_have_answer: bool = True) -> float:
    """Simpler version of compute_format_reward that does not require <think>...</think>.

    We reward the model if its response follows this format in every round:
        ...
        <tool_call>...</tool_call> (if a tool is needed)
        <answer>...</answer>       (otherwise)

    Everything else follows that of compute_format_reward.
    """
    # Extract the responses from the conversation
    responses = _extract_responses(solution_str)

    # Check if every response strictly follows the format
    for i, response in enumerate(responses):
        # Check if this is a tool-call response
        if "<tool_call>" in response:
            # Ensure there is exactly one <tool_call> and one </tool_call>
            if response.count("<tool_call>") != 1 or response.count("</tool_call>") != 1:
                return 0.0
            # Ensure that </tool_call> is after <tool_call>
            if "</tool_call>" not in response.split("<tool_call>", 1)[1]:
                return 0.0
            # Ensure there is nothing else after </tool_call>
            if response.split("</tool_call>", 1)[1].strip():
                return 0.0
            # Ensure that there is no <answer> or </answer> in the response
            if "<answer>" in response or "</answer>" in response:
                return 0.0
        else:
            # If this is not a tool-call response, it should have an answer
            if i + 1 == len(responses):
                must_have_answer = True
            else:
                return 0.0

    # If must_have_answer is True, the last response must end with <answer>...</answer>
    if must_have_answer:
        # Ensure there is exactly one <answer> and one </answer>
        if responses[-1].count("<answer>") != 1 or responses[-1].count("</answer>") != 1:
            return 0.0
        # Ensure that </answer> is after <answer>
        if "</answer>" not in responses[-1].split("<answer>", 1)[1]:
            return 0.0
        # Ensure there is nothing else after </answer>
        if responses[-1].split("</answer>", 1)[1].strip():
            return 0.0
        # Ensure there is no tool call in the last response
        if "<tool_call>" in responses[-1] or "</tool_call>" in responses[-1]:
            return 0.0

    return 1.0


async def compute_accuracy_reward(
    data_source: str,
    question: str,
    extracted_answer: str,
    ground_truth: str,
    judge_client: AsyncOpenAI,
    judge_model: str = "gpt-5-nano",
    verify_prompt_template: str | None = None,
) -> float:
    """Compute the accuracy reward of the extracted answer."""

    if extracted_answer == ground_truth:
        return 1.0

    normalized_extracted = normalize_answer(extracted_answer)
    normalized_ground_truth = normalize_answer(ground_truth)
    fallback_accuracy = float(
        bool(normalized_extracted)
        and bool(normalized_ground_truth)
        and (
            normalized_extracted == normalized_ground_truth
            or normalized_extracted in normalized_ground_truth
            or normalized_ground_truth in normalized_extracted
        )
    )

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        logger.warning("No judge endpoint configured; falling back to normalized exact/substring accuracy.")
        return fallback_accuracy

    # Use judge model to verify the answer
    query = (verify_prompt_template or verify_prompt).format(
        question=question,
        gt_answer=ground_truth,
        model_answer=extracted_answer,
    )

    try:
        _, judge_response = await query_api(
            query,
            model=judge_model,
            client=judge_client,
            **_judge_completion_token_kwargs(judge_model),
        )
        response_text = judge_response.choices[0].message.content
    except Exception as e:
        raise JudgeError(
            f"judge query failed ({type(e).__name__}: {e!r}); "
            "normalized exact/substring fallback suppressed so batch retry can handle it"
        ) from e

    return float("<correct>" in response_text)


def _compute_rule_based_raw_final_answer_accuracy(
    question: str,
    full_final_answer: str,
    ground_truth: str,
) -> float | None:
    mc_match = _raw_final_answer_matches_mc_ground_truth_v1(
        question=question,
        ground_truth=ground_truth,
        full_final_answer=full_final_answer,
    )
    if mc_match is not None:
        return float(mc_match)
    normalized_ground_truth = normalize_answer(ground_truth)
    normalized_full_answer = normalize_answer(full_final_answer)
    if normalized_ground_truth and normalized_ground_truth == normalized_full_answer:
        return 1.0
    if _raw_final_answer_matches_scalar_title_ground_truth(
        question=question,
        ground_truth=ground_truth,
        full_final_answer=full_final_answer,
    ):
        return 1.0
    if _raw_final_answer_matches_list_ground_truth(
        question=question,
        ground_truth=ground_truth,
        full_final_answer=full_final_answer,
    ):
        return 1.0
    if _raw_final_answer_matches_entity_ground_truth(
        question=question,
        ground_truth=ground_truth,
        full_final_answer=full_final_answer,
    ):
        return 1.0
    if _is_not_answerable_text(ground_truth) and _raw_final_answer_matches_not_answerable(full_final_answer):
        return 1.0
    return None


def _compute_rule_based_raw_final_answer_accuracy_v2(
    question: str,
    full_final_answer: str,
    ground_truth: str,
) -> float | None:
    mc_match = _raw_final_answer_matches_mc_ground_truth(
        question=question,
        ground_truth=ground_truth,
        full_final_answer=full_final_answer,
    )
    if mc_match is not None:
        return float(mc_match)
    normalized_ground_truth = normalize_answer(ground_truth)
    normalized_full_answer = normalize_answer(full_final_answer)
    if normalized_ground_truth and normalized_ground_truth == normalized_full_answer:
        return 1.0
    return None


async def compute_accuracy_reward_with_raw_final_answer(
    data_source: str,
    question: str,
    extracted_answer: str,
    full_final_answer: str,
    ground_truth: str,
    judge_client: AsyncOpenAI,
    judge_model: str = "gpt-5-nano",
    rule_version: str = "v1",
) -> float:
    """Verify fallback answer correctness using extracted and raw final answer text."""

    rule_fn = (
        _compute_rule_based_raw_final_answer_accuracy_v2
        if rule_version == "v2"
        else _compute_rule_based_raw_final_answer_accuracy
    )
    rule_based_accuracy = rule_fn(
        question=question,
        full_final_answer=full_final_answer,
        ground_truth=ground_truth,
    )
    if rule_based_accuracy is not None:
        return rule_based_accuracy

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        logger.warning("No judge endpoint configured; raw-final-answer fallback unavailable.")
        return 0.0

    query = VERIFY_WITH_RAW_FINAL_ANSWER_PROMPT.format(
        question=question,
        gt_answer=ground_truth,
        extracted_answer=extracted_answer,
        full_final_answer=full_final_answer,
    )

    try:
        _, judge_response = await query_api(
            query,
            model=judge_model,
            client=judge_client,
            **_judge_completion_token_kwargs(judge_model),
        )
        response_text = judge_response.choices[0].message.content
    except Exception as e:
        raise JudgeError(
            f"raw-final-answer judge query failed ({type(e).__name__}: {e!r}); "
            "batch retry can handle this transient failure"
        ) from e

    return float("<correct>" in str(response_text))


async def compute_accuracy_reward_with_final_answer_only(
    data_source: str,
    question: str,
    full_final_answer: str,
    ground_truth: str,
    judge_client: AsyncOpenAI,
    judge_model: str = "gpt-5-nano",
    rule_version: str = "v1",
) -> float:
    """Verify single-call answer correctness using only the raw final answer text."""

    rule_fn = (
        _compute_rule_based_raw_final_answer_accuracy_v2
        if rule_version == "v2"
        else _compute_rule_based_raw_final_answer_accuracy
    )
    rule_based_accuracy = rule_fn(
        question=question,
        full_final_answer=full_final_answer,
        ground_truth=ground_truth,
    )
    if rule_based_accuracy is not None:
        return rule_based_accuracy

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        logger.warning("No judge endpoint configured; final-answer-only judge unavailable.")
        return 0.0

    query = VERIFY_FINAL_ANSWER_ONLY_PROMPT.format(
        question=question,
        gt_answer=ground_truth,
        full_final_answer=full_final_answer,
    )

    try:
        _, judge_response = await query_api(
            query,
            model=judge_model,
            client=judge_client,
            **_judge_completion_token_kwargs(judge_model),
        )
        response_text = judge_response.choices[0].message.content
    except Exception as e:
        raise JudgeError(
            f"final-answer-only judge query failed ({type(e).__name__}: {e!r}); "
            "batch retry can handle this transient failure"
        ) from e

    return float("<correct>" in str(response_text))


async def compute_score_single_vsearch_base(
    solution_str: str,
    extra_info: dict,
    **reward_kwargs: dict,
) -> Score:
    assert len(extra_info["image_processed_wh"]) >= 1, f"expected at least 1 image, got {len(extra_info['image_processed_wh'])}"
    image_processed_wh = extra_info["image_processed_wh"][0]

    score_cls = SCORE_CLASS_MAP[reward_kwargs["reward_type"]]
    score = score_cls(reward_kwargs["reward_weights"])

    # Compute the format reward
    fn = compute_format_reward_simple if reward_kwargs["format_reward"]["simple"] else compute_format_reward
    score.format_reward = fn(
        solution_str,
        must_have_answer=reward_kwargs["format_reward"]["must_have_answer"],
    )

    bboxes_crop = extra_info["tool_call_bboxes"]
    score.n_valid_tool_calls = min(len(bboxes_crop), solution_str.count("user\n<tool_response>"))

    # Compute the iou if the ground truth bboxes are available
    if extra_info.get("bboxes"):
        tool_iou = compute_overall_iou_with_gt(
            bboxes_crop, extra_info["bboxes"], image_processed_wh, extra_info["image_ori_wh"][0]
        )
        score.tool_iou = tool_iou

    # If the format is wrong, return 0.0 for all rewards
    if not score.format_reward:
        _record_judge_metadata(score, reward_kwargs)
        return score

    # Compute the tool reward
    score.tool_reward = float(score.n_valid_tool_calls > 0)

    # Penalize bad tool usage: cropping similar regions in two consecutive tool calls
    seen_bboxes = [(0, 0, *image_processed_wh), *bboxes_crop]
    for i in range(1, len(seen_bboxes)):
        iou = compute_iou(seen_bboxes[i-1], seen_bboxes[i])
        max_consecutive_iou = reward_kwargs["tool_reward"]["max_consecutive_iou"]
        if iou > max_consecutive_iou:
            logger.info(f"[RewardWorker] vsearcher consecutive iou {iou} > {max_consecutive_iou}")
            score.tool_reward = 0.0
            break

    _record_judge_metadata(score, reward_kwargs)
    return score


async def compute_score_single_vsearcher(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    score = await compute_score_single_vsearch_base(solution_str, extra_info, **reward_kwargs)

    # Compute the iou reward
    # Use pre-converted final_bbox if available, otherwise use extracted_answer
    bbox_2d = extra_info["final_bbox"]
    if bbox_2d is not None:
        bbox_2d = tuple(bbox_2d)  # Ensure it's a tuple
    if bbox_2d is None or bbox_2d == (0, 0, 0, 0):
        _record_judge_metadata(score, reward_kwargs)
        return score

    final_iou = compute_overall_iou_with_gt(
        [bbox_2d], extra_info["bboxes"], extra_info["image_processed_wh"][0], extra_info["image_ori_wh"][0]
    )
    score.final_iou = final_iou

    iou_low, iou_high = reward_kwargs["iou_reward"]["iou_low"], reward_kwargs["iou_reward"]["iou_high"]
    final_iou_clipped = max(iou_low, min(iou_high, score.final_iou))
    score.iou_reward = (final_iou_clipped - iou_low) / (iou_high - iou_low)

    _record_judge_metadata(score, reward_kwargs)
    return score


async def compute_score_single_vsearcher_as_subagent(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    score = await compute_score_single_vsearch_base(solution_str, extra_info, **reward_kwargs)
    _record_judge_metadata(score, reward_kwargs)
    return score


async def compute_score_single_vreasoner(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    """ For vReasoner, we only care about whether the answer is correct.  """
    score = ScoreOnlyAccuracy()

    score.extracted_answer = _extract_last_tagged_content(solution_str, "answer")
    if score.extracted_answer is None:
        score.extracted_answer = _extract_last_boxed_answer(solution_str)
    if score.extracted_answer is None and extra_info.get("agent_name") == "insight_qwen_agent":
        score.extracted_answer = _extract_last_plain_answer(solution_str)

    if score.extracted_answer:
        score.format_reward = 1.0
    else:
        score.format_reward = 0.0

    if score.extracted_answer:
        score.accuracy_reward = await compute_accuracy_reward(
            data_source,
            extra_info["question"],
            score.extracted_answer,
            ground_truth,
            reward_kwargs["judge_client"],
            reward_kwargs["judge_model"],
        )
    else:
        score.accuracy_reward = 0.0

    score.n_valid_tool_calls = solution_str.count("<tool_response>")
    _record_judge_metadata(score, reward_kwargs)
    return score


async def compute_score_single_insight_qwen_agent(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    score = ScoreOnlyAccuracy()

    final_message = _extract_last_assistant_message(solution_str)
    if final_message is None:
        score.format_reward = 0.0
        score.accuracy_reward = 0.0
        score.n_valid_tool_calls = solution_str.count("<tool_response>")
        _record_judge_metadata(score, reward_kwargs)
        return score

    insight_qwen_judge_mode = reward_kwargs.get("insight_qwen_judge_mode")
    if insight_qwen_judge_mode in {"single_call_v1", "single_call_v2"}:
        raw_final_answer = _extract_raw_final_answer_for_judge(final_message)
        score.extracted_answer = raw_final_answer
        score.format_reward = float(bool(raw_final_answer))
        if raw_final_answer:
            score.accuracy_reward = await compute_accuracy_reward_with_final_answer_only(
                data_source,
                extra_info["question"],
                raw_final_answer,
                ground_truth,
                reward_kwargs["judge_client"],
                reward_kwargs["judge_model"],
                rule_version="v2" if insight_qwen_judge_mode == "single_call_v2" else "v1",
            )
        else:
            score.accuracy_reward = 0.0
        score.n_valid_tool_calls = solution_str.count("<tool_response>")
        _record_judge_metadata(score, reward_kwargs)
        return score

    legacy_prompt_version = "legacy_prompt_v2" if insight_qwen_judge_mode == "legacy_prompt_v2" else "legacy"
    legacy_verify_prompt_template = (
        LEGACY_PROMPT_V2_QA_VERIFY_PROMPT
        if insight_qwen_judge_mode == "legacy_prompt_v2"
        else None
    )

    rule_based_answer = _extract_last_tagged_content(final_message, "answer")
    if rule_based_answer is None:
        rule_based_answer = _extract_last_boxed_answer(final_message)

    if rule_based_answer == ground_truth:
        score.extracted_answer = rule_based_answer
    else:
        score.extracted_answer = await _extract_answer_for_insight_qwen_agent(
            question=extra_info["question"],
            final_message=final_message,
            extra_info=extra_info,
            judge_client=reward_kwargs["judge_client"],
            judge_model=reward_kwargs["judge_model"],
            prompt_version=legacy_prompt_version,
        )

    score.format_reward = float(bool(score.extracted_answer))
    if score.extracted_answer:
        score.accuracy_reward = await compute_accuracy_reward(
            data_source,
            extra_info["question"],
            score.extracted_answer,
            ground_truth,
            reward_kwargs["judge_client"],
            reward_kwargs["judge_model"],
            verify_prompt_template=legacy_verify_prompt_template,
        )
    else:
        score.accuracy_reward = 0.0

    score.n_valid_tool_calls = solution_str.count("<tool_response>")

    if (
        score.accuracy_reward == 0.0
        and score.extracted_answer
        and reward_kwargs.get("insight_qwen_judge_mode") == "raw_final_answer_fallback_v2"
    ):
        raw_final_answer = _extract_raw_final_answer_for_judge(final_message)
        score.accuracy_reward = await compute_accuracy_reward_with_raw_final_answer(
            data_source,
            extra_info["question"],
            str(score.extracted_answer),
            raw_final_answer,
            ground_truth,
            reward_kwargs["judge_client"],
            reward_kwargs["judge_model"],
        )

    if score.accuracy_reward == 0.0 and insight_qwen_judge_mode == "legacy_raw_final_answer_fallback_v1":
        raw_final_answer = _extract_raw_final_answer_for_judge(final_message)
        if raw_final_answer:
            if not _raw_final_answer_is_hierarchy_leaf_only(ground_truth, raw_final_answer):
                score.accuracy_reward = await compute_accuracy_reward_with_raw_final_answer(
                    data_source,
                    extra_info["question"],
                    str(score.extracted_answer or ""),
                    raw_final_answer,
                    ground_truth,
                    reward_kwargs["judge_client"],
                    reward_kwargs["judge_model"],
                    rule_version="v2",
                )
                if not score.extracted_answer:
                    score.extracted_answer = raw_final_answer

    _record_judge_metadata(score, reward_kwargs)
    return score


def _update_subagent_iou_rewards(
    scores: list[Score | None],
    extra_infos: list[dict],
    pseudo_iou_reward_type: str,
) -> None:
    """Update subagent iou_reward with pseudo_iou_reward based on caller_feedback and root's accuracy_reward.

    This separates the reward computation from the advantage estimation in compute_advantage.
    For subagents (where parent_job_id is not None), we compute a pseudo_iou_reward based on:
    - The subagent's caller_feedback
    - The root job's accuracy_reward

    The pseudo_iou_reward is then set as the subagent's iou_reward.
    """

    # Build mapping from job_id to index
    job_id_to_idx = {}
    for i, extra_info in enumerate(extra_infos):
        job_id = extra_info.get("job_id")
        if job_id is not None:
            job_id_to_idx[job_id] = i

    # Update subagent iou_rewards
    count = 0
    for i, extra_info in enumerate(extra_infos):
        # Only process subagents (where parent_job_id is not None)
        if extra_info.get("parent_job_id") is None:
            continue

        # Get root job's accuracy_reward
        root_idx = job_id_to_idx[extra_info["root_job_id"]]
        root_score = scores[root_idx]
        if root_score is None:
            # Root job's score computation failed, skip this subagent
            continue
        root_accuracy_reward = root_score.accuracy_reward

        # Compute pseudo_iou_reward based on pseudo_iou_reward_type
        caller_feedback = extra_info["caller_feedback"]
        if "caller_feedback_no_outcome" in pseudo_iou_reward_type:
            pseudo_iou_reward = float(caller_feedback == "helpful")
        elif "outcome_no_caller_feedback" in pseudo_iou_reward_type:
            pseudo_iou_reward = float(root_accuracy_reward == 1.0)
        else:
            pseudo_iou_reward = float(caller_feedback == "helpful" and root_accuracy_reward == 1.0)

        # Update score
        scores[i].iou_reward = pseudo_iou_reward
        count += 1

    logger.info(f"[RewardWorker] Updated {count} subagent iou_rewards")


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs):
    """
    Optimized batch reward computation using asyncio-based concurrency and timeout handling.
    """

    async def _compute_score_batch():
        logger.info(
            f"[RewardWorker] Computing rewards for {len(solution_strs)} samples "
            f"with {reward_kwargs['num_workers']} workers "
            f"using {reward_kwargs['judge_model']} verification "
            f"(fallback_judge_model={reward_kwargs.get('fallback_judge_model') or None})."
        )
        start_time = time.time()

        compute_single_fns = []
        for i, extra_info in enumerate(extra_infos):
            if "agent_name" not in extra_info:
                raise KeyError(f"agent_name not found in extra_info: {extra_info}")

            if extra_info["agent_name"].startswith("vsearcher"):
                if extra_info.get("parent_job_id") is None:
                    compute_single_fns.append(compute_score_single_vsearcher)
                else:
                    compute_single_fns.append(compute_score_single_vsearcher_as_subagent)
            elif extra_info["agent_name"] == "insight_qwen_agent":
                compute_single_fns.append(compute_score_single_insight_qwen_agent)
            elif (
                extra_info["agent_name"].startswith("vreasoner")
            ):
                compute_single_fns.append(compute_score_single_vreasoner)
            else:
                raise ValueError(f"Unknown agent name: {extra_info['agent_name']}")

        judge_client = create_async_openai_client()

        reward_kwargs_with_client = {**reward_kwargs, "judge_client": judge_client}
        try:
            async with RewardComputer(**reward_kwargs_with_client) as reward_computer:
                success, scores = await reward_computer.compute_batch(
                    compute_single_fns=compute_single_fns,
                    data_sources=data_sources,
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    **reward_kwargs_with_client,
                )
        finally:
            await judge_client.close()

        # Post-process: update subagent iou_reward with pseudo_iou_reward
        pseudo_iou_reward_type = reward_kwargs["iou_reward"].get("pseudo_iou_reward_type", "")
        if "caller_feedback" in pseudo_iou_reward_type and any(ei.get("caller_feedback") for ei in extra_infos):
            _update_subagent_iou_rewards(scores, extra_infos, pseudo_iou_reward_type)

        n_skipped = sum(s is None for s in success)
        n_failed = sum(s is False for s in success)
        n_correct = sum(score.is_correct for score in scores if score is not None)
        n_wrong = sum(not score.is_correct for score in scores if score is not None)
        elapsed = time.time() - start_time
        logger.info(
            f"[RewardWorker] Accuracy ({reward_kwargs['judge_model']}): "
            f"{n_correct} correct, {n_wrong} wrong, {n_failed} failed, {n_skipped} skipped out of {len(scores)} samples"
        )
        agent_names = defaultdict(int)
        for extra_info in extra_infos:
            agent_names[extra_info.get("agent_name")] += 1
        write_profile_event(
            "reward_batch",
            {
                "event": "reward_batch",
                "total_samples": len(scores),
                "agent_names": dict(agent_names),
                "judge_model": reward_kwargs["judge_model"],
                "fallback_judge_model": reward_kwargs.get("fallback_judge_model") or None,
                "num_workers": reward_kwargs["num_workers"],
                "task_timeout": reward_kwargs["task_timeout"],
                "min_success_rate": reward_kwargs["min_success_rate"],
                "max_retries": reward_kwargs["max_retries"],
                "retry_interval": reward_kwargs["retry_interval"],
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "n_failed": n_failed,
                "n_skipped": n_skipped,
                "timing_s": {"total": elapsed},
            },
        )

        score_dicts = []
        for compute_score_success, score in zip(success, scores, strict=True):
            if score is None:
                score_cls = SCORE_CLASS_MAP[reward_kwargs["reward_type"]]
                score = score_cls(reward_kwargs["reward_weights"])  # dummy score
            score_dict = asdict(score)
            score_dict["compute_score_success"] = compute_score_success
            score_dicts.append(score_dict)

        for data_source, ground_truth, extra_info, score_dict in zip(
            data_sources,
            ground_truths,
            extra_infos,
            score_dicts,
            strict=True,
        ):
            export_path = extra_info.get("conversation_export_json_path")
            if not export_path:
                continue

            reward_payload = {
                "reward": score_dict.get("score"),
                "score": score_dict,
                "data_source": data_source,
                "ground_truth": ground_truth,
                "agent_name": extra_info.get("agent_name"),
                "failure_reasons": extra_info.get("failure_reasons"),
                "compute_score_success": score_dict.get("compute_score_success"),
            }
            if score_dict.get("extracted_answer") is not None:
                reward_payload["extracted_answer"] = score_dict["extracted_answer"]

            try:
                append_reward_info(str(export_path), reward_payload)
            except Exception as exc:
                logger.warning("[conversation_export_update_failed] path=%s error=%s", export_path, exc)

        logger.info(f"[RewardWorker] Reward computation completed, time: {time.time() - start_time:.2f}s")
        return score_dicts

    try:
        return asyncio.run(_compute_score_batch())
    except RuntimeError as e:
        # asyncio.run cannot be called from a running event loop; surface a clear error.
        if "asyncio.run()" in str(e):
            raise RuntimeError("compute_score_batch must be called from a non-async context") from e
        raise
