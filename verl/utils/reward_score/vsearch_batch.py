import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
import logging

from openai import AsyncOpenAI

from verl.utils.vsearch import (
    BBox,
    iou as compute_iou,
    compute_overall_iou_with_gt,
    intersection_area,
    extract_bbox_from_tool_call,
    parse_bbox,
)
from verl.utils.vsearch_role_play_prompt import qa_verify as verify_prompt

from insight_o3.utils.api import create_async_openai_client, query_api  # pyright: ignore[reportMissingImports]


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
        compute_single_fn: Callable,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        **reward_kwargs: dict,
    ) -> tuple[int, Any, Exception | None]:
        assert self._semaphore is not None
        try:
            async with self._semaphore:
                result = await asyncio.wait_for(
                    compute_single_fn(
                        data_source,
                        solution_str,
                        ground_truth,
                        extra_info,
                        **reward_kwargs,
                    ),
                    timeout=self.task_timeout,
                )
            return idx, result, None
        except Exception as exc:  # bubble idx back to the collector
            return idx, None, exc

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
                        compute_single_fn,
                        data_source,
                        solution_str,
                        ground_truth,
                        extra_info,
                        **reward_kwargs,
                    )
                )
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                idx, task_result, error = await task
                if error is None:
                    results[idx] = task_result
                    success[idx] = True
                    continue

                if isinstance(error, JudgeError):
                    logger.warning(f"[RewardWorker] Task {idx} failed: {error}")
                elif isinstance(error, asyncio.TimeoutError):
                    logger.warning(f"[RewardWorker] Task {idx} timed out after {self.task_timeout}s")
                else:
                    logger.error(f"[RewardWorker] Task {idx} encountered an unexpected error")
                    raise error

            task_success_rate = sum(success) / total_samples
            num_failed = total_samples - sum(success)
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


def _extract_last_assistant_response(solution_str: str) -> str:
    if "assistant\n" not in solution_str:
        return solution_str.strip()
    return solution_str.rsplit("assistant\n", 1)[-1].strip()


def _extract_vreasoner_v2_answer(response: str) -> str | None:
    try:
        parsed = parse_response(
            response,
            required_tags=["observation", "state", "plan", "response"],
            excluded_tags=["action"],
        )
    except ParseError:
        parsed = None

    if parsed is not None:
        answer = parsed["response"].strip()
        return answer or None

    try:
        parsed = parse_response(
            response,
            required_tags=["observation", "state", "plan", "action"],
            excluded_tags=["response"],
        )
    except ParseError:
        return None

    action = parsed["action"].strip()
    if not action:
        return None

    try:
        payload = json.loads(action)
    except json.JSONDecodeError:
        return action

    if (
        isinstance(payload, dict)
        and payload.get("name") == "image_zoom_in_tool"
        and isinstance(payload.get("arguments"), dict)
    ):
        return None

    return action


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
    responses = [round.split("assistant\n")[-1] for round in solution_str.split("user\n")]

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
    responses = [round.split("assistant\n")[-1] for round in solution_str.split("user\n")]

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
        raise JudgeError(f"failed to verify answer correctness: {e}")

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
        return score

    final_iou = compute_overall_iou_with_gt(
        [bbox_2d], extra_info["bboxes"], extra_info["image_processed_wh"][0], extra_info["image_ori_wh"][0]
    )
    score.final_iou = final_iou

    iou_low, iou_high = reward_kwargs["iou_reward"]["iou_low"], reward_kwargs["iou_reward"]["iou_high"]
    final_iou_clipped = max(iou_low, min(iou_high, score.final_iou))
    score.iou_reward = (final_iou_clipped - iou_low) / (iou_high - iou_low)

    return score


async def compute_score_single_vsearcher_as_subagent(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    score = await compute_score_single_vsearch_base(solution_str, extra_info, **reward_kwargs)
    return score


async def compute_score_single_vreasoner(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **reward_kwargs: dict
) -> Score:
    """ For vReasonser, we only care about whether the answer is correct.  """
    score = ScoreOnlyAccuracy()
    last_response = _extract_last_assistant_response(solution_str)

    if "\\boxed{" in solution_str:
        score.extracted_answer = solution_str.split("\\boxed{", 1)[1].split("}", 1)[0].strip()
    elif "<answer>" in solution_str and "</answer>" in solution_str:
        score.extracted_answer = solution_str.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    else:
        score.extracted_answer = _extract_vreasoner_v2_answer(last_response)

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
            f"using {reward_kwargs['judge_model']} verification."
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
            elif extra_info["agent_name"].startswith("vreasoner"):
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
        logger.info(
            f"[RewardWorker] Accuracy ({reward_kwargs['judge_model']}): "
            f"{n_correct} correct, {n_wrong} wrong, {n_failed} failed, {n_skipped} skipped out of {len(scores)} samples"
        )

        score_dicts = []
        for compute_score_success, score in zip(success, scores, strict=True):
            if score is None:
                score_cls = SCORE_CLASS_MAP[reward_kwargs["reward_type"]]
                score = score_cls(reward_kwargs["reward_weights"])  # dummy score
            score_dict = asdict(score)
            score_dict["compute_score_success"] = compute_score_success
            score_dicts.append(score_dict)

        logger.info(f"[RewardWorker] Reward computation completed, time: {time.time() - start_time:.2f}s")
        return score_dicts

    try:
        return asyncio.run(_compute_score_batch())
    except RuntimeError as e:
        # asyncio.run cannot be called from a running event loop; surface a clear error.
        if "asyncio.run()" in str(e):
            raise RuntimeError("compute_score_batch must be called from a non-async context") from e
        raise
