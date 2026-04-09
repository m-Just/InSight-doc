# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.vreasoner_v2_conversation_export import append_reward_info
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        # Inject additional information to the extras
        # First, we deepcopy the extras, so we can safely modify its content
        # NOTE: we can't use `extras = deepcopy(extras)` because it does *not* deepcopy the objects in the array!
        extras = np.array([deepcopy(e) for e in extras])
        for info in ("agent_name", "job_id", "parent_job_id", "root_job_id", "caller_feedback", "final_bbox", "tool_call_bboxes", "img_idx"):
            if info in data.non_tensor_batch:
                for i in range(len(data)):
                    extras[i] = extras[i] or {}
                    extras[i][info] = data.non_tensor_batch[info][i]

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]
            extracted_answer = None
            score_for_export = score.copy() if isinstance(score, dict) else {"score": score}

            if isinstance(score, dict):
                if "extracted_answer" in score:
                    extracted_answer = str(score.pop("extracted_answer"))
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            export_path = None
            if "conversation_export_json_path" in data.non_tensor_batch:
                export_path = data.non_tensor_batch["conversation_export_json_path"][i]
            if export_path:
                reward_payload = {
                    "reward": reward,
                    "score": score_for_export,
                    "data_source": data_source,
                    "ground_truth": data[i].non_tensor_batch["reward_model"].get("ground_truth", None),
                    "agent_name": data.non_tensor_batch["agent_name"][i] if "agent_name" in data.non_tensor_batch else None,
                    "failure_reasons": data.non_tensor_batch["failure_reasons"][i] if "failure_reasons" in data.non_tensor_batch else None,
                }
                if extracted_answer is not None:
                    reward_payload["extracted_answer"] = extracted_answer
                try:
                    append_reward_info(str(export_path), reward_payload)
                except Exception as exc:
                    print(f"[conversation_export_update_failed] path={export_path} error={exc}")

            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if extracted_answer is not None:
                    print("[extracted_answer]", extracted_answer)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print("[data_source]", data_source)
                if "agent_name" in data.non_tensor_batch:
                    print("[agent_name]", data.non_tensor_batch["agent_name"][i])
                if "failure_reasons" in data.non_tensor_batch:
                    print("[failure_reasons]", data.non_tensor_batch["failure_reasons"][i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            critical_failure = data.non_tensor_batch.pop("critical_failure", None)
            if critical_failure is not None:
                for i in range(len(data)):
                    reward_extra_info["critical_failure"].append(critical_failure[i])
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
