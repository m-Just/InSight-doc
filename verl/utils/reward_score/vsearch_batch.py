import asyncio
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


async def _extract_answer_for_insight_qwen_agent(
    *,
    question: str,
    final_message: str,
    extra_info: dict,
    judge_client: AsyncOpenAI,
    judge_model: str,
) -> str | None:
    raw_answer = _extract_last_tagged_content(final_message, "answer")
    if raw_answer is None:
        raw_answer = _extract_last_boxed_answer(final_message)
    if raw_answer is None:
        raw_answer = final_message.strip()
    if not raw_answer:
        return None

    raw_answer = _crop_judge_answer_context(raw_answer)
    options = extra_info.get("options")
    if options:
        prompt = insight_o3_prompts.GPT_EVAL_MCQA_PROMPT.format(
            question=question,
            options=options,
            model_answer=raw_answer,
        )
    else:
        prompt = insight_o3_prompts.GPT_EVAL_OPEN_QA_PROMPT.format(
            question=question,
            model_response=raw_answer,
        )

    try:
        _, response = await query_api(
            query=prompt,
            model=judge_model,
            client=judge_client,
            max_completion_tokens=2048,
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
    query = verify_prompt.format(
        question=question,
        gt_answer=ground_truth,
        model_answer=extracted_answer,
    )

    try:
        _, judge_response = await query_api(
            query,
            model=judge_model,
            client=judge_client,
            max_tokens=2048,
        )
        response_text = judge_response.choices[0].message.content
    except Exception as e:
        raise JudgeError(
            f"judge query failed ({type(e).__name__}: {e!r}); "
            "normalized exact/substring fallback suppressed so batch retry can handle it"
        ) from e

    return float("<correct>" in response_text)


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
        )
    else:
        score.accuracy_reward = 0.0

    score.n_valid_tool_calls = solution_str.count("<tool_response>")
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
