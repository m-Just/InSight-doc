# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import pickle
import shutil
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractBatchSampler, AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_agent_metrics,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
    process_vsearch_validation_metrics,
)
from verl.trainer.ppo.reward import (
    compute_reward,
    compute_reward_async,
    compute_reward_async_thread,
    get_async_reward_thread,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.vreasoner_v2_conversation_export import build_repeated_conversation_export_id
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _resolve_agent_loop_config_path(config_path: str) -> str:
    if os.path.isabs(config_path):
        return config_path

    cwd = os.path.abspath(os.getcwd())
    cwd_path = os.path.abspath(os.path.join(cwd, config_path))
    if (cwd_path == cwd or cwd_path.startswith(cwd + os.sep)) and os.path.exists(cwd_path):
        return cwd_path

    try:
        import verl

        verl_package_dir = os.path.abspath(os.path.dirname(verl.__file__))
        project_root = os.path.dirname(verl_package_dir)
        dev_path = os.path.abspath(os.path.join(project_root, config_path))
        if (dev_path == project_root or dev_path.startswith(project_root + os.sep)) and os.path.exists(dev_path):
            return dev_path

        install_path = os.path.abspath(os.path.join(verl_package_dir, config_path))
        if (install_path == verl_package_dir or install_path.startswith(verl_package_dir + os.sep)) and os.path.exists(
            install_path
        ):
            return install_path
    except (ImportError, AttributeError):
        pass

    raise FileNotFoundError(
        f"Agent loop configuration file not found: {config_path}. Tried current directory and verl project root."
    )


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        subagent_advantage_estimator = config.get("subagent_advantage_estimator")
        if not subagent_advantage_estimator:
            return data

        adv_new = data.batch["advantages"].clone()
        job_id_arr = data.non_tensor_batch["job_id"]
        root_job_id_arr = data.non_tensor_batch["root_job_id"]
        jobid_to_idx = {jid: idx for idx, jid in enumerate(job_id_arr)}

        if "broadcast" in subagent_advantage_estimator:
            # Replace the advantages and returns of subagent jobs by those of their root jobs
            # For each sample, if job_id != root_job_id, replace its adv/ret with those of its root
            for idx, (jid, rjid) in enumerate(zip(job_id_arr, root_job_id_arr, strict=False)):
                if jid != rjid:
                    root_idx = jobid_to_idx[rjid]
                    adv_new[idx] = advantages[root_idx]
                    if subagent_advantage_estimator == "broadcast_tool_only":
                        adv_new[idx] *= data.non_tensor_batch["tool_reward"][idx]
                        
        elif "global_norm" in subagent_advantage_estimator:
            # For subagents, the iou_reward has been updated with pseudo_iou_reward in compute_score_batch
            # based on caller_feedback and root's accuracy_reward. Here we just use it for advantage estimation.
            rewards = []
            for idx, (jid, rjid) in enumerate(zip(job_id_arr, root_job_id_arr, strict=False)):
                # Only process subagents (where job_id != root_job_id)
                if jid != rjid:
                    reward = data.batch["token_level_scores"][idx].sum().item()
                    rewards.append((idx, reward))
            if "no_global_norm" in config.subagent_advantage_estimator:
                for idx, reward in rewards:
                    adv_new[idx] = reward
            else:
                reward_mean = np.mean([reward for _, reward in rewards])
                reward_std = np.std([reward for _, reward in rewards])
                for idx, reward in rewards:
                    adv_new[idx] = (reward - reward_mean) / (reward_std + 1e-6)
        else:
            raise ValueError(f"Invalid subagent advantage estimator type: {config.subagent_advantage_estimator}")

        data.batch["advantages"] = adv_new
        data.batch["returns"] = adv_new

    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self._validation_image_token_reorder_settings = self._get_validation_image_token_reorder_settings()

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        val_only = self.config.trainer.get("val_only", False)

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self._collate_fn = collate_fn
        num_workers = self.config.data["dataloader_num_workers"]
        self._dataloader_num_workers = num_workers
        self._val_shuffle_enabled = self.config.data.get("validation_shuffle", True)
        if self._validation_image_token_reorder_settings and self._val_shuffle_enabled:
            print("validation image-token reordering enabled: overriding validation_shuffle=True to False")
            self._val_shuffle_enabled = False

        # Skip training dataset/dataloader creation if val_only is enabled
        if val_only:
            self.train_dataset = None
            self.train_dataloader = None
        else:
            if train_dataset is None:
                train_dataset = create_rl_dataset(
                    self.config.data.train_files,
                    self.config.data,
                    self.tokenizer,
                    self.processor,
                    is_train=True,
                    max_samples=self.config.data.get("train_max_samples", -1),
                )
            self.train_dataset = train_dataset

            if train_sampler is None:
                train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

            if isinstance(train_sampler, AbstractBatchSampler):
                self.train_dataloader = StatefulDataLoader(
                    dataset=self.train_dataset,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    batch_sampler=train_sampler,
                )
            else:
                self.train_dataloader = StatefulDataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                    num_workers=num_workers,
                    drop_last=True,
                    collate_fn=collate_fn,
                    sampler=train_sampler,
                )
            assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if val_dataset is None or self._validation_image_token_reorder_settings:
            val_dataset = self._create_val_dataset_for_trial(0)
        self.val_dataset = val_dataset

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self._val_shuffle_enabled,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self._current_val_trial_idx = 0

        if val_only:
            self.total_training_steps = 0
            print(f"val_only mode: skipping training dataset/dataloader creation")
            print(f"Size of val dataloader: {len(self.val_dataloader)}")
            return

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _create_val_dataset_for_trial(self, val_trial_idx: int):
        from verl.trainer.main_ppo import create_rl_dataset

        return create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
            conversation_export_dir=self.config.actor_rollout_ref.rollout.agent.get(
                "vreasoner_v2_conversation_export_dir",
                None,
            ),
            conversation_export_resume_mode=self.config.actor_rollout_ref.rollout.agent.get(
                "vreasoner_v2_conversation_export_resume_mode",
                "off",
            ),
            conversation_export_val_trial_idx=val_trial_idx,
            conversation_export_repeat_count=self.config.actor_rollout_ref.rollout.val_kwargs.n,
            validation_image_token_reorder_settings=self._validation_image_token_reorder_settings,
        )

    def _rebuild_val_dataloader_for_trial(self, val_trial_idx: int) -> None:
        self.val_dataset = self._create_val_dataset_for_trial(val_trial_idx)
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self._dataloader_num_workers,
            shuffle=self._val_shuffle_enabled,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
        self._current_val_trial_idx = val_trial_idx

    def _get_validation_image_token_reorder_settings(self) -> dict[str, Any] | None:
        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.get("agent_loop_config_path")
        if not agent_loop_config_path:
            return None

        resolved_path = _resolve_agent_loop_config_path(agent_loop_config_path)
        loaded_configs = OmegaConf.load(resolved_path)
        agent_loop_configs = list(loaded_configs) if OmegaConf.is_list(loaded_configs) else [loaded_configs]
        agent_settings_by_name: dict[str, dict[str, Any]] = {}
        for agent_loop_config in agent_loop_configs:
            agent_name = OmegaConf.select(agent_loop_config, "name")
            if agent_name not in {"insight_qwen_agent", "vreasoner_v2"}:
                continue
            agent_settings_by_name[agent_name] = {
                "initial_rescale": float(OmegaConf.select(agent_loop_config, "initial_rescale", default=0.25)),
                "gpt_image_max_area": int(
                    OmegaConf.select(agent_loop_config, "gpt_image_max_area", default=1280 * 1280)
                ),
            }

        if not agent_settings_by_name:
            return None

        default_agent_loop = self.config.actor_rollout_ref.rollout.agent.get("default_agent_loop")
        settings = {
            "enabled": True,
            "agent_settings_by_name": agent_settings_by_name,
            "num_workers": int(self.config.actor_rollout_ref.rollout.agent.num_workers),
            "batch_size": self.config.data.val_batch_size,
            "default_agent_loop": default_agent_loop,
        }
        print(
            "validation image-token reordering configured: "
            f"default_agent_loop={default_agent_loop} "
            f"agent_names={sorted(agent_settings_by_name.keys())} "
            f"agent_settings={agent_settings_by_name} "
            f"num_workers={settings['num_workers']} "
            f"batch_size={settings['batch_size']}"
        )
        return settings

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_batch_sample(self, batch, dump_dir, num_groups=5, packet_id=None):
        if num_groups is None:
            selected_batch = batch
        else:
            # Get the root job indexes
            if "parent_job_id" in batch.non_tensor_batch:
                root_job_idxs = (batch.non_tensor_batch["parent_job_id"] == None).nonzero()[0]  # noqa: E711
            else:
                root_job_idxs = np.arange(len(batch))

            # Group the root jobs by uid
            root_job_idxs_by_uid = defaultdict(list)
            for idx in root_job_idxs:
                root_job_idxs_by_uid[batch.non_tensor_batch["uid"][idx]].append(idx)

            # Randomly select at most num_groups
            rng = np.random.default_rng(42)
            uids = list(root_job_idxs_by_uid.keys())
            selected_uids = rng.choice(uids, min(num_groups, len(uids)), replace=False)

            # For each selected group, select the top, middle, and bottom jobs sorted by reward scores (if given)
            if "token_level_scores" in batch.batch:
                scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                selected_idxs = []
                for uid in selected_uids:
                    idxs = root_job_idxs_by_uid[uid]
                    idxs_argsort = np.argsort(scores[idxs])
                    selected_idxs.extend([idxs[idxs_argsort[i]] for i in [-1, len(idxs) // 2, 0]])
            else:
                selected_idxs = []
                for uid in selected_uids:
                    idxs = root_job_idxs_by_uid[uid]
                    selected_idxs.extend([idxs[i] for i in [-1, len(idxs) // 2, 0]])

            # Get the extra jobs derived from the selected jobs
            if "parent_job_id" in batch.non_tensor_batch:
                selected_root_job_ids = batch.non_tensor_batch["job_id"][selected_idxs]
                selected_extra_job_idxs = []
                for idx in range(len(batch)):
                    if batch.non_tensor_batch["parent_job_id"][idx] is None:
                        continue
                    if batch.non_tensor_batch["root_job_id"][idx] in selected_root_job_ids:
                        selected_extra_job_idxs.append(idx)
                selected_idxs.extend(selected_extra_job_idxs)

            selected_batch = batch.select_idxs(selected_idxs)

        # pop pixel_values (if any) to save disk space
        if "multi_modal_inputs" in selected_batch.non_tensor_batch:
            for mm_inputs in selected_batch.non_tensor_batch["multi_modal_inputs"]:
                if isinstance(mm_inputs, dict) and "pixel_values" in mm_inputs:
                    mm_inputs.pop("pixel_values", None)

        if packet_id is not None:
            dump_dir = Path(dump_dir, f"global_step_{self.global_steps}")
            dump_path = Path(dump_dir, f"packet_{packet_id}.pickle")
        else:
            dump_dir = Path(dump_dir)
            dump_path = Path(dump_dir, f"global_step_{self.global_steps}.pickle")

        dump_dir.mkdir(parents=True, exist_ok=True)
        with dump_path.open("wb") as f:
            pickle.dump(selected_batch, f)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _assign_conversation_export_ids(self, batch: DataProto) -> None:
        extra_infos = batch.non_tensor_batch.get("extra_info")
        if extra_infos is None:
            return

        occurrence_by_base_id: dict[str, int] = defaultdict(int)
        updated_extra_infos = []
        changed = False
        for extra_info in extra_infos:
            if not isinstance(extra_info, dict):
                updated_extra_infos.append(extra_info)
                continue
            base_export_id = extra_info.get("conversation_export_base_id")
            if not base_export_id:
                updated_extra_infos.append(extra_info)
                continue
            repeat_idx = occurrence_by_base_id[base_export_id]
            occurrence_by_base_id[base_export_id] += 1
            export_id = build_repeated_conversation_export_id(base_export_id, repeat_idx)
            new_extra_info = dict(extra_info)
            new_extra_info["conversation_export_id"] = export_id
            new_extra_info["conversation_export_repeat_idx"] = repeat_idx
            updated_extra_infos.append(new_extra_info)
            changed = True

        if changed:
            batch.non_tensor_batch["extra_info"] = np.array(updated_extra_infos, dtype=object)

    def _validate(self, merged: bool = False, val_trial_idx: int = 0):
        resume_mode = self.config.actor_rollout_ref.rollout.agent.get("vreasoner_v2_conversation_export_resume_mode", "off")
        if resume_mode != "off" and getattr(self, "_current_val_trial_idx", None) != val_trial_idx:
            self._rebuild_val_dataloader_for_trial(val_trial_idx)
        if len(self.val_dataset) == 0:
            return {
                "resume/remaining_validation_samples": 0.0,
                "resume/validation_trial_idx": float(val_trial_idx),
            }

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_conversation_wall_times = []
        sample_uids = []

        packets_count_by_data_source = defaultdict(int)

        for test_data in tqdm(self.val_dataloader, desc="Validation Progress"):
            t_batch_start = time.perf_counter()
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            self._assign_conversation_export_ids(test_batch)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            if "extra_info" in test_batch.non_tensor_batch:
                test_gen_batch.non_tensor_batch["extra_info"] = test_batch.non_tensor_batch.pop("extra_info")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            t_before_generate = time.perf_counter()
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            t_after_generate = time.perf_counter()

            # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch, test_output_gen_batch, new_pad_size = self._reorganize_batch(
                test_batch, test_output_gen_batch_padded, pad_size
            )
            t_after_reorganize = time.perf_counter()

            print("validation generation end")
            print(
                "Validation phase timing: "
                f"batch_prep={t_before_generate - t_batch_start:.2f}s "
                f"generate={t_after_generate - t_before_generate:.2f}s "
                f"reorganize={t_after_reorganize - t_after_generate:.2f}s"
            )

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store input prompts
            input_ids = test_output_gen_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # Store ground truths
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            if "response_truncated" in test_batch.non_tensor_batch and "response_truncated" not in reward_extra_info:
                reward_extra_info["response_truncated"] = test_batch.non_tensor_batch["response_truncated"]
            t_after_reward = time.perf_counter()
            print(
                "Validation reward timing: "
                f"reward_compute={t_after_reward - t_after_reorganize:.2f}s "
                f"total_batch={t_after_reward - t_batch_start:.2f}s"
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            overlapping_keys = {
                key for key in reward_extra_infos_dict if key in test_batch.non_tensor_batch
            }
            overlapping_keys.discard("response_truncated")
            assert not overlapping_keys, (
                f"{test_batch.non_tensor_batch.keys()=}, {reward_extra_infos_dict.keys()=}, "
                f"{overlapping_keys=}"
            )
            test_batch.non_tensor_batch.update(
                {
                    k: np.array(v, dtype=object)
                    for k, v in reward_extra_infos_dict.items()
                    if k not in test_batch.non_tensor_batch
                }
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
            if "conversation_wall_time" in test_batch.non_tensor_batch:
                sample_conversation_wall_times.append(
                    np.asarray(test_batch.non_tensor_batch["conversation_wall_time"], dtype=np.float64)
                )

            sample_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(sample_data_sources)

            # for each data source, dump a small sample to disk
            if dump_dir := self.config.trainer.get("val_dump_dir", None):
                max_val_sample_dump_per_data_source = self.config.trainer.get("max_val_sample_dump_per_data_source", 1)
                for idx, data_source in enumerate(sample_data_sources):
                    # skip if already dumped enough samples for this data source
                    if packets_count_by_data_source[data_source] >= max_val_sample_dump_per_data_source:
                        continue
                    self._dump_batch_sample(
                        test_batch[idx : idx + 1],
                        str(Path(dump_dir, data_source, f"trial_{val_trial_idx}")),
                        num_groups=None,
                        packet_id=packets_count_by_data_source[data_source],
                    )
                    packets_count_by_data_source[data_source] += 1

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "sample_conversation_wall_times": sample_conversation_wall_times,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_conversation_wall_times,
        )

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
        sample_conversation_wall_times,
    ):
        metric_dict = {}

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)

        if self.config.get("use_vsearch", False):
            vsearch_metrics = process_vsearch_validation_metrics(data_sources, reward_extra_infos_dict)
            for data_source, var2metric2val in vsearch_metrics.items():
                if data_source not in data_src2var2metric2val:
                    data_src2var2metric2val[data_source] = {}
                data_src2var2metric2val[data_source].update(var2metric2val)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_list = [int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys() if "@" in name]
                n_max = max(n_list) if n_list else 0
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if "response_truncated" in reward_extra_infos_dict:
            response_truncated = np.asarray(reward_extra_infos_dict["response_truncated"], dtype=bool)
            for data_source in np.unique(data_sources):
                ds_mask = data_sources == data_source
                metric_dict[f"val-aux/{data_source}/truncation_rate/mean"] = float(response_truncated[ds_mask].mean())

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if len(sample_conversation_wall_times) > 0:
            sample_conversation_wall_times = np.concatenate(sample_conversation_wall_times)
            metric_dict["val-aux/conversation_wall_time/min"] = sample_conversation_wall_times.min()
            metric_dict["val-aux/conversation_wall_time/max"] = sample_conversation_wall_times.max()
            metric_dict["val-aux/conversation_wall_time/mean"] = sample_conversation_wall_times.mean()
            for data_source in np.unique(data_sources):
                ds_mask = data_sources == data_source
                metric_dict[f"val-aux/{data_source}/conversation_wall_time/mean"] = float(
                    sample_conversation_wall_times[ds_mask].mean()
                )

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "sample_conversation_wall_times": [],
                "reward_extra_infos_dict": {},
            }
        if result_b is None:
            result_b = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "sample_conversation_wall_times": [],
                "reward_extra_infos_dict": {},
            }

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]
        sample_conversation_wall_times = (
            result_a["sample_conversation_wall_times"] + result_b["sample_conversation_wall_times"]
        )

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_conversation_wall_times,
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            if resume_from_step := self.config.trainer.get("resume_from_step", None):
                # TODO: merge this code with find_latest_ckpt_path
                global_step_folder = os.path.join(checkpoint_folder, "global_step_{}".format(resume_from_step))
                if not os.path.exists(global_step_folder):
                    raise ValueError("Checkpoint folder does not exist: {}".format(global_step_folder))
            else:
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        if self.train_dataloader is None:
            return
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def _update_multi_modal_inputs(self, batch: DataProto, gen_batch_output: DataProto):
        if "multi_modal_inputs" in batch.non_tensor_batch and "multi_modal_inputs" in gen_batch_output.non_tensor_batch:
            from verl.utils.dataset.rl_dataset import concat_multi_modal_inputs

            multi_modal_inputs = concat_multi_modal_inputs(
                batch.non_tensor_batch.pop("multi_modal_inputs"),
                gen_batch_output.non_tensor_batch.pop("multi_modal_inputs"),
            )
            gen_batch_output.non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

    def _reorganize_batch(
        self, batch: DataProto, gen_batch_output: DataProto, pad_size: int = 0, size_divisor: int = 1
    ) -> tuple[DataProto, DataProto, int]:
        """Reorganize batch and gen_batch_output for multi-agent joint training.

        In multi-agent joint training, gen_batch_output may contain more data than the (input) batch because agents may
        create new jobs for subagents. These new jobs then generate extra data that are not in the input batch.
        For example, given data-parallel (dp) size 2, the data structure is as follows:
            batch:            |<- dp0 ->|<- dp1 ->|
            gen_batch_output: |<- dp0 ->|<- dp0_extra ->|<- dp1 ->|<- dp1_extra ->|

        In cases where gen_batch is padded, we have:
            gen_batch:        |<- dp0 ->|<- dp1 ->|<- pad ->|
            gen_batch_output: |<- dp0 ->|<- dp0_extra ->|<- dp1 ->|<- pad ->|<- dp1_extra ->|
        where dp1_extra may contain new data derived from the pad data.

        This function removes the pad data and their derived data (if any) from gen_batch_output, creates an extra batch
        for the new jobs by duplicating their root jobs in the batch, and then reorganizes the batch and the unpadded
        gen_batch_output to align them.

        The returned batch and gen_batch_output are of the same length and have the same structure as follows:
            batch:            |<- dp0 ->|<- dp1 ->|<- dp0_extra ->|<- dp1_extra ->|<- new_pad ->|
            gen_batch_output: |<- dp0 ->|<- dp1 ->|<- dp0_extra ->|<- dp1_extra ->|<- new_pad ->|
        where new_pad is added so the batch can be evenly divided by size_divisor.

        NOTE: This function uses shallow copies whenever possible to avoid unnecessary memory usage.
              However, this may cause unexpected behavior if the data inside the batch is modified in place.
              So, it is recommended to avoid directly modifying the data inside the batch.
              Even with deepcopy, one must be careful because e.g., deepcopying a numpy array of dtype=object does not
              deepcopy the elements inside the array and the elements may still reference to the same original objects.
        """
        assert len(gen_batch_output) >= len(batch) + pad_size

        # There is no need to reorganize the batch if there is no new data
        if len(gen_batch_output) == len(batch) + pad_size:
            # Remove the pad
            gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size)

            # Store the root job ids
            if "job_id" in gen_batch_output.non_tensor_batch:
                root_job_ids = [job_id for job_id in gen_batch_output.non_tensor_batch["job_id"]]
                gen_batch_output.non_tensor_batch["root_job_id"] = np.array(root_job_ids, dtype=object)

            # Update multi_modal_inputs
            self._update_multi_modal_inputs(batch, gen_batch_output)

            return batch, gen_batch_output, 0

        print(f"_reorganize_batch input: {len(batch)=}, {len(gen_batch_output)=}, {pad_size=}, {size_divisor=}")

        # Construct the mapping from job_id to output index in gen_batch_output
        job_id_to_idx = {}
        for i in range(len(gen_batch_output)):
            job_id = gen_batch_output.non_tensor_batch["job_id"][i]
            assert job_id not in job_id_to_idx, f"Duplicate job_id: {job_id}"
            job_id_to_idx[job_id] = i

        # Construct the mapping from output index to input index
        # The construction is based on the following assumptions:
        # 1. non-pad jobs always precede pad jobs
        # 2. non-root jobs are always preceded by their parent jobs
        # 3. the order of root jobs is the same for batch and gen_batch_output
        output_to_input_idx = {}  # output idx of root jobs  -> input idx of the root jobs
        extra_output_to_input_idx = {}  # output idx of extra jobs -> input idx of the root jobs of the extra jobs
        pad_idxs = set()
        for i in range(len(gen_batch_output)):
            if gen_batch_output.non_tensor_batch["parent_job_id"][i] is None:
                if len(output_to_input_idx) < len(batch):
                    output_to_input_idx[i] = len(output_to_input_idx)
                else:
                    pad_idxs.add(i)
            else:
                root_job_id = gen_batch_output.non_tensor_batch["root_job_id"][i]
                root_idx = job_id_to_idx[root_job_id]
                if root_idx in pad_idxs:
                    pad_idxs.add(i)
                else:
                    extra_output_to_input_idx[i] = output_to_input_idx[root_idx]

        assert len(output_to_input_idx) == len(batch), f"{len(output_to_input_idx)=}, {len(batch)=}"
        assert len(output_to_input_idx) + len(extra_output_to_input_idx) + len(pad_idxs) == len(gen_batch_output), (
            f"{len(output_to_input_idx)=}, {len(extra_output_to_input_idx)=}, {len(pad_idxs)=}, "
            f"{len(gen_batch_output)=}"
        )

        # Construct the extra batch
        from tensordict import TensorDict

        if "dummy_tensor" in batch.batch:
            _ = batch.batch.pop("dummy_tensor")
        assert len(batch.batch.keys()) == 0, f"{batch.batch.keys()=}"
        extra_tensor_batch = TensorDict({}, batch_size=len(extra_output_to_input_idx))
        extra_non_tensor_batch = {}

        # Add suffixes to data_source of the extra batch
        extra_data_source_list = []
        for output_idx, input_idx in extra_output_to_input_idx.items():
            data_source = batch.non_tensor_batch["data_source"][input_idx]
            agent_name = gen_batch_output.non_tensor_batch["agent_name"][output_idx]
            extra_data_source_list.append(f"{data_source}_derived/{agent_name}")
        extra_non_tensor_batch["data_source"] = np.array(extra_data_source_list, dtype=object)

        reward_model_list = [{} for _ in range(len(extra_output_to_input_idx))]
        extra_non_tensor_batch["reward_model"] = np.array(reward_model_list, dtype=object)

        if "multi_modal_inputs" in batch.non_tensor_batch:
            # Initialize multi_modal_inputs as empty dicts for the extra batch
            multi_modal_inputs_list = [{} for _ in range(len(extra_output_to_input_idx))]
            extra_non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        if "uid" in batch.non_tensor_batch:
            # Generate random UIDs for the extra batch
            uid_list = [str(uuid.uuid4()) for _ in range(len(extra_output_to_input_idx))]
            extra_non_tensor_batch["uid"] = np.array(uid_list, dtype=object)

        for key in batch.non_tensor_batch.keys():
            if key in extra_non_tensor_batch:
                continue
            print(f"DEBUG: filling batch.non_tensor_batch.{key} with None for the extra batch of subagents")
            extra_non_tensor_batch[key] = np.array([None] * len(extra_output_to_input_idx), dtype=object)
        for key in batch.meta_info:
            print(f"DEBUG: copying batch.meta_info.{key} for the extra batch of subagents")

        extra_batch = DataProto(
            batch=extra_tensor_batch,
            non_tensor_batch=extra_non_tensor_batch,
            meta_info=batch.meta_info,
        )

        # Construct the fully aligned batch and gen_batch_output
        batch = DataProto.concat([batch, extra_batch])
        gen_batch_output = gen_batch_output.select_idxs(list(output_to_input_idx) + list(extra_output_to_input_idx))
        assert len(batch) == len(gen_batch_output), f"{len(batch)=}, {len(gen_batch_output)=}"

        # Update multi_modal_inputs
        self._update_multi_modal_inputs(batch, gen_batch_output)

        # Pad the batch to make it divisible by size_divisor
        batch, new_pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        gen_batch_output, _ = pad_dataproto_to_divisor(gen_batch_output, size_divisor)

        # [lkc] TODO: DataProto.select_idxs only creates shallow copy; should we enforce deepcopy for all data?

        # Print a summary
        print("_reorganize_batch completed.")
        print(f"Original batch size : {len(output_to_input_idx)}")
        print(f"Extra batch size    : {len(extra_output_to_input_idx)}")
        print(f"Old pad size        : {len(pad_idxs)}")
        print(f"New pad size        : {new_pad_size}")
        print(f"Final batch size    : {len(batch)} (original + extra + new_pad)")

        return batch, gen_batch_output, new_pad_size

    def _force_concat(self, batch_train: DataProto, batch_other: DataProto, train_first: bool = True) -> DataProto:
        print(f"DEBUG: _force_concat: {len(batch_train)=}, {len(batch_other)=}, {train_first=}")
        for key, val in batch_train.batch.items():
            if key not in batch_other.batch.keys():
                tensor_size = (len(batch_other), *val.shape[1:])
                print(f'DEBUG: _force_concat: adding dummy placeholders of size {tensor_size} for "{key}"')
                batch_other.batch[key] = torch.zeros(*tensor_size, dtype=val.dtype, device=val.device)
        for key in batch_train.non_tensor_batch.keys():
            if key not in batch_other.non_tensor_batch.keys():
                print(f'DEBUG: _force_concat: adding dummy placeholders of size {len(batch_other)} for "{key}"')
                batch_other.non_tensor_batch[key] = np.array([None] * len(batch_other), dtype=object)
        return DataProto.concat([batch_train, batch_other] if train_first else [batch_other, batch_train])

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader) if self.train_dataloader else 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            for val_trial_idx in range(self.config.trainer.get("val_before_train_n", 1)):
                val_metrics = self._validate(val_trial_idx=val_trial_idx)
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.finish()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                if "extra_info" in batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["extra_info"] = batch.non_tensor_batch.pop("extra_info")

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if "critical_failure" in gen_batch_output.non_tensor_batch:
                        critical_failure = gen_batch_output.non_tensor_batch["critical_failure"]
                        failure_ratio = (critical_failure == True).sum() / max((critical_failure != None).sum(), 1)
                        max_ratio = self.config.actor_rollout_ref.rollout.agent.get("max_critical_failure_ratio", 0.2)
                        if failure_ratio > max_ratio:
                            raise RuntimeError(
                                f"Critical failure ratio {failure_ratio:.2f} exceeds tolerance threshold {max_ratio}"
                            )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    # re-organize the batch and gen_batch_output for multi-agent joint training
                    # for padding, we assume ppo_mini_batch_size == train_batch_size for simplicity
                    assert self.config.actor_rollout_ref.actor.ppo_mini_batch_size == self.config.data.train_batch_size
                    assert (
                        self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                        % self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
                        == 0
                    )
                    size_divisor = (
                        self.actor_rollout_wg.world_size
                        * self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                    )
                    original_batch_size = len(batch)
                    with marked_timer("reorganize_batch", timing_raw, color="gray"):
                        batch, gen_batch_output, new_pad_size = self._reorganize_batch(
                            batch, gen_batch_output, size_divisor=size_divisor
                        )
                    subagent_batch_size = len(batch) - original_batch_size - new_pad_size

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if not self.config.trainer.get("train_on_vreasoner_response", True):
                        vreasoner_idxs = np.where(batch.non_tensor_batch["agent_name"] == "vreasoner")[0]
                        other_idxs = np.where(batch.non_tensor_batch["agent_name"] != "vreasoner")[0]
                        other_idxs = other_idxs[
                            : len(other_idxs) // size_divisor * size_divisor
                        ]  # align with size_divisor
                        vreasoner_batch = batch.select_idxs(vreasoner_idxs)
                        batch = batch.select_idxs(other_idxs)

                    # Separate the main-agent/subagent batch if we only train on one of them
                    subagent_keep_size = subagent_batch_size // size_divisor * size_divisor
                    if not self.config.trainer.get("train_on_main_agent_response", True):
                        main_agent_batch = batch[:original_batch_size]
                        batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                    elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                        subagent_batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                        batch = batch[:original_batch_size]
                    effective_batch_size = len(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # append the main-agent/subagent batch back to the original batch for reward computation
                        if not self.config.trainer.get("train_on_main_agent_response", True):
                            batch = self._force_concat(batch, main_agent_batch, train_first=False)
                        elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                            batch = self._force_concat(batch, subagent_batch)

                        if not self.config.trainer.get("train_on_vreasoner_response", True):
                            batch = self._force_concat(batch, vreasoner_batch)

                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            if self.config.reward_model.get("reward_fn_async_backend", "ray") == "ray":
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                future_reward, reward_fn_thread = compute_reward_async_thread(self.reward_fn, batch)
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                        if not self.config.trainer.get("train_on_vreasoner_response", True):
                            vreasoner_idxs = np.where(batch.non_tensor_batch["agent_name"] == "vreasoner")[0]
                            other_idxs = np.where(batch.non_tensor_batch["agent_name"] != "vreasoner")[0]
                            vreasoner_batch = batch.select_idxs(vreasoner_idxs)
                            batch = batch.select_idxs(other_idxs)

                        # remove the main-agent/subagent batch from the original batch
                        if not self.config.trainer.get("train_on_main_agent_response", True):
                            main_agent_batch = batch[:original_batch_size]
                            batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                        elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                            subagent_batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                            batch = batch[:original_batch_size]

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    skip_old_log_prob_recompute = self.config.actor_rollout_ref.actor.get("skip_old_log_prob_recompute", False)

                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    elif skip_old_log_prob_recompute:
                        # Skip old_log_prob computation to save time
                        print(
                            "WARNING: `skip_old_log_prob_recompute` is set. Make sure that .train() and .eval() "
                            "don't affect the log_prob produced by the model"
                        )
                        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                        if self.config.data.train_batch_size > ppo_mini_batch_size:
                            raise RuntimeError(
                                "`skip_old_log_prob_recompute` is set, but the train_batch_size is larger than "
                                "the ppo_mini_batch_size"
                            )
                        if self.config.algorithm.use_kl_in_reward:
                            raise RuntimeError(
                                "`skip_old_log_prob_recompute` is set, but `use_kl_in_reward` is enabled"
                            )
                        # old_log_prob is simply the current log_prob which will be computed in update_actor
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    else:  # Recompute old_log_probs (decoupled mode)
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    if not bypass_recomputing_logprobs and not skip_old_log_prob_recompute:
                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # append the main-agent/subagent batch back to the original batch for reward computation
                        if not self.config.trainer.get("train_on_main_agent_response", True):
                            batch = self._force_concat(batch, main_agent_batch, train_first=False)
                        elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                            batch = self._force_concat(batch, subagent_batch)

                        if not self.config.trainer.get("train_on_vreasoner_response", True):
                            batch = self._force_concat(batch, vreasoner_batch)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            if self.config.reward_model.get("reward_fn_async_backend", "ray") == "ray":
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            else:
                                reward_tensor, reward_extra_infos_dict = get_async_reward_thread(
                                    future_reward, reward_fn_thread
                                )
                                if reward_tensor is None:  # fallback to synchronous computation
                                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = reward_tensor

                        if self.config.get("use_vsearch", False):
                            batch.non_tensor_batch.pop("critical_failure", None)

                        if reward_extra_infos_dict:
                            assert all(key not in batch.non_tensor_batch for key in reward_extra_infos_dict), (
                                f"{batch.non_tensor_batch.keys()=}, {reward_extra_infos_dict.keys()=}"
                            )
                            batch.non_tensor_batch.update(
                                {k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()}
                            )

                        # score computation may fail occasionally (e.g., due to GPT API error)
                        # we tolerate a small number of failed samples by replacing them with successful ones
                        if "compute_score_success" in batch.non_tensor_batch:
                            # NOTE: elements in compute_score_success can take three values:
                            # - True: the score computation is successful
                            # - False: the score computation is failed
                            # - None: the score computation is not performed (e.g., for subagent results)
                            # We only replace the failed ones (and their descendants) here.
                            idxs_to_keep = np.where(batch.non_tensor_batch["compute_score_success"] != False)[0]  # noqa: E712
                            if "job_id" in batch.non_tensor_batch and "root_job_id" in batch.non_tensor_batch:
                                job_ids_to_keep = set(batch.non_tensor_batch["job_id"][idxs_to_keep])
                                idxs_to_keep = np.array(
                                    [
                                        idx
                                        for idx in idxs_to_keep
                                        if batch.non_tensor_batch["root_job_id"][idx] in job_ids_to_keep
                                    ]
                                )
                            assert len(idxs_to_keep), f"{len(idxs_to_keep)=}, {len(batch.batch)=}"

                            if len(idxs_to_keep) < len(batch.batch):
                                num_to_pad = len(batch.batch) - len(idxs_to_keep)
                                print(
                                    f"WARNING: compute_score failed for {num_to_pad} out of {len(batch.batch)} samples;"
                                    " replacing the failed ones"
                                )
                                # randomly sample indices from select_idxs to pad
                                rng = np.random.RandomState(42)
                                pad_idxs = rng.choice(idxs_to_keep, size=num_to_pad, replace=True)
                                # concatenate the selected samples and the padded samples
                                new_idxs = np.concatenate([idxs_to_keep, pad_idxs])
                                batch = batch.select_idxs(new_idxs)
                                # update reward info
                                reward_tensor = batch.batch["token_level_scores"]
                                reward_extra_infos_dict = {
                                    k: list(batch.non_tensor_batch[k]) for k in reward_extra_infos_dict
                                }

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if not self.config.trainer.get("train_on_vreasoner_response", True):
                            vreasoner_idxs = np.where(batch.non_tensor_batch["agent_name"] == "vreasoner")[0]
                            other_idxs = np.where(batch.non_tensor_batch["agent_name"] != "vreasoner")[0]
                            other_idxs = other_idxs[
                                : len(other_idxs) // size_divisor * size_divisor
                            ]  # align with size_divisor
                            vreasoner_batch = batch.select_idxs(vreasoner_idxs)
                            batch = batch.select_idxs(other_idxs)

                        # remove the main-agent/subagent batch from the original batch
                        if not self.config.trainer.get("train_on_main_agent_response", True):
                            main_agent_batch = batch[:original_batch_size]
                            batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                        elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                            subagent_batch = batch[original_batch_size : original_batch_size + subagent_keep_size]
                            batch = batch[:original_batch_size]

                        # rebalance the batch
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # append the main-agent/subagent batch back to the original batch for dumping results and metrics
                    if not self.config.trainer.get("train_on_main_agent_response", True):
                        batch = self._force_concat(batch, main_agent_batch, train_first=False)
                    elif subagent_batch_size and not self.config.trainer.get("train_on_subagent_response", False):
                        batch = self._force_concat(batch, subagent_batch)

                    if not self.config.trainer.get("train_on_vreasoner_response", True):
                        batch = self._force_concat(batch, vreasoner_batch)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # add refresh_freq to save checkpoint
                refresh_freq = self.config.trainer.get("refresh_freq", -1)
                if refresh_freq > 0 and (self.global_steps % refresh_freq == 0):
                    checkpoint_folder = self.config.trainer.default_local_dir
                    if not os.path.isabs(checkpoint_folder):
                        working_dir = os.getcwd()
                        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                    refresh_file = Path(checkpoint_folder, "last_refresh_iteration.txt")
                    if refresh_file.exists():
                        with refresh_file.open("r") as f:
                            last_refresh_step = int(f.read())
                        shutil.rmtree(Path(checkpoint_folder, f"global_step_{last_refresh_step}"), ignore_errors=True)
                        refresh_file.unlink()
                    if not Path(checkpoint_folder, f"global_step_{self.global_steps}").exists():
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        with refresh_file.open("w") as f:
                            f.write(str(self.global_steps))

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                if self.config.get("use_vsearch", False):
                    metrics.update(compute_agent_metrics(batch=batch))
                    metrics.update({"training/batch_size": effective_batch_size})
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                if dump_dir := self.config.trainer.get("train_dump_dir", None):
                    self._dump_batch_sample(batch, dump_dir)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    logger.finish()
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
