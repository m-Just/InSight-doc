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
Single Process Actor
"""

import logging
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_REF_MICROBATCH_DEBUG_ENABLED = os.environ.get(
    "VERL_DEBUG_REF_MICROBATCH", os.environ.get("VERL_DEBUG_MEM", "")
).lower() in {"1", "true", "yes", "on"}
_ACTOR_UPDATE_DEBUG_ENABLED = os.environ.get(
    "VERL_DEBUG_ACTOR_UPDATE", os.environ.get("VERL_DEBUG_MEM", "")
).lower() in {"1", "true", "yes", "on"}


def _debug_tensor_size_gb(tensor) -> float:
    if not hasattr(tensor, "element_size") or not hasattr(tensor, "numel"):
        return 0.0
    return tensor.element_size() * tensor.numel() / (1024**3)


def _debug_tensor_sum(tensor) -> int:
    if tensor is None or not hasattr(tensor, "detach"):
        return 0
    return int(tensor.detach().sum().item())


def _debug_response_token_sum(responses, pad_token_id: int) -> int:
    if responses is None or not hasattr(responses, "detach"):
        return 0
    return int(responses.detach().ne(pad_token_id).sum().item())


def _debug_distributed_rank() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return -1
    return int(torch.distributed.get_rank())


def _debug_cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    try:
        device = torch.cuda.current_device()
        return (
            f"cuda_device={device} "
            f"alloc_gb={torch.cuda.memory_allocated(device) / (1024**3):.3f} "
            f"reserved_gb={torch.cuda.memory_reserved(device) / (1024**3):.3f} "
            f"max_alloc_gb={torch.cuda.max_memory_allocated(device) / (1024**3):.3f} "
            f"max_reserved_gb={torch.cuda.max_memory_reserved(device) / (1024**3):.3f}"
        )
    except Exception as exc:
        return f"cuda_mem_error={type(exc).__name__}:{exc}"


def _debug_micro_batch_summary(micro_batch: DataProto, pad_token_id: int) -> str:
    batch = micro_batch.batch
    non_tensor_batch = micro_batch.non_tensor_batch

    attention_mask = batch.get("attention_mask") if batch is not None else None
    responses = batch.get("responses") if batch is not None else None
    valid_tokens = _debug_tensor_sum(attention_mask)
    response_tokens = _debug_response_token_sum(responses, pad_token_id)
    prompt_tokens = max(0, valid_tokens - response_tokens)

    input_shape = tuple(batch["input_ids"].shape) if batch is not None and "input_ids" in batch else None
    response_shape = tuple(responses.shape) if responses is not None and hasattr(responses, "shape") else None

    image_items = 0
    image_count = 0
    image_tokens_sum = 0
    image_tokens_max = 0
    pixel_values_gb = 0.0
    pixel_shapes = []

    if non_tensor_batch is not None and "multi_modal_inputs" in non_tensor_batch:
        for item in non_tensor_batch["multi_modal_inputs"]:
            if not isinstance(item, dict) or not item:
                continue
            image_items += 1
            grid = item.get("image_grid_thw")
            if grid is not None:
                try:
                    if hasattr(grid, "detach"):
                        grid_tensor = grid.detach().reshape(-1, 3)
                    else:
                        grid_tensor = torch.as_tensor(grid).reshape(-1, 3)
                    per_image_tokens = grid_tensor.prod(dim=1)
                    image_count += int(per_image_tokens.numel())
                    if per_image_tokens.numel() > 0:
                        image_tokens_sum += int(per_image_tokens.sum().item())
                        image_tokens_max = max(image_tokens_max, int(per_image_tokens.max().item()))
                except Exception:
                    pass
            pixel_values = item.get("pixel_values")
            if pixel_values is not None:
                pixel_values_gb += _debug_tensor_size_gb(pixel_values)
                if len(pixel_shapes) < 5 and hasattr(pixel_values, "shape"):
                    pixel_shapes.append(tuple(pixel_values.shape))

    return (
        f"mb_len={len(micro_batch)} input_shape={input_shape} response_shape={response_shape} "
        f"valid_tokens={valid_tokens} prompt_tokens={prompt_tokens} response_tokens={response_tokens} "
        f"mm_items={image_items} images={image_count} image_tokens_sum={image_tokens_sum} "
        f"image_tokens_max={image_tokens_max} pixel_values_gb={pixel_values_gb:.3f} "
        f"pixel_shapes={pixel_shapes}"
    )


def _debug_ref_micro_batch(
    *,
    label: str,
    micro_batch: DataProto,
    pad_token_id: int,
    micro_batch_idx: int,
    num_micro_batches: int,
    outputs: dict[str, torch.Tensor] | None = None,
) -> None:
    parts = [
        f"DEBUG_REF_MICROBATCH {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"pid={os.getpid()} rank={_debug_distributed_rank()} local_rank={os.environ.get('LOCAL_RANK', '?')}",
        f"{label} micro_batch={micro_batch_idx + 1}/{num_micro_batches}",
        _debug_cuda_memory_summary(),
        _debug_micro_batch_summary(micro_batch, pad_token_id),
    ]
    if outputs is not None:
        output_parts = []
        for key, value in outputs.items():
            if hasattr(value, "shape"):
                output_parts.append(f"{key}_shape={tuple(value.shape)}")
                output_parts.append(f"{key}_gb={_debug_tensor_size_gb(value):.6f}")
        if output_parts:
            parts.append(" ".join(output_parts))
    print(" | ".join(parts), flush=True)


def _debug_actor_update_event(
    *,
    label: str,
    pad_token_id: int,
    micro_batch: DataProto | None = None,
    epoch_idx: int | None = None,
    batch_idx: int | None = None,
    micro_batch_idx: int | None = None,
    num_micro_batches: int | None = None,
    outputs: dict[str, torch.Tensor] | None = None,
    extra: str | None = None,
) -> None:
    index_parts = []
    if epoch_idx is not None:
        index_parts.append(f"epoch={epoch_idx + 1}")
    if batch_idx is not None:
        index_parts.append(f"mini_batch={batch_idx + 1}")
    if micro_batch_idx is not None and num_micro_batches is not None:
        index_parts.append(f"micro_batch={micro_batch_idx + 1}/{num_micro_batches}")

    parts = [
        f"DEBUG_ACTOR_UPDATE {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"pid={os.getpid()} rank={_debug_distributed_rank()} local_rank={os.environ.get('LOCAL_RANK', '?')}",
        " ".join([label, *index_parts]).strip(),
        _debug_cuda_memory_summary(),
    ]
    if micro_batch is not None:
        parts.append(_debug_micro_batch_summary(micro_batch, pad_token_id))
    if outputs is not None:
        output_parts = []
        for key, value in outputs.items():
            if hasattr(value, "shape"):
                output_parts.append(f"{key}_shape={tuple(value.shape)}")
                output_parts.append(f"{key}_gb={_debug_tensor_size_gb(value):.6f}")
        if output_parts:
            parts.append(" ".join(output_parts))
    if extra:
        parts.append(extra)
    print(" | ".join(parts), flush=True)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                    if self.use_ulysses_sp and is_vlm_model:
                        model_type = getattr(
                            getattr(self.actor_module, "module", self.actor_module).config,
                            "model_type",
                            None,
                        )
                        if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
                            extra_args["shifted_labels"] = input_ids_rmpad_rolled

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        debug_ref_microbatch = _REF_MICROBATCH_DEBUG_ENABLED and bool(
            data.meta_info.get("debug_ref_log_prob_microbatch", False)
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        if debug_ref_microbatch:
            _debug_ref_micro_batch(
                label="before_select",
                micro_batch=data,
                pad_token_id=pad_token_id,
                micro_batch_idx=0,
                num_micro_batches=1,
            )
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if debug_ref_microbatch:
            _debug_ref_micro_batch(
                label="after_select",
                micro_batch=data,
                pad_token_id=pad_token_id,
                micro_batch_idx=0,
                num_micro_batches=1,
            )

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="before_prepare_dynamic_batch",
                    micro_batch=data,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=0,
                    num_micro_batches=1,
                )
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="after_prepare_dynamic_batch",
                    micro_batch=data,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=0,
                    num_micro_batches=len(micro_batches),
                )
        else:
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="before_fixed_split",
                    micro_batch=data,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=0,
                    num_micro_batches=1,
                )
            micro_batches = data.split(micro_batch_size)
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="after_fixed_split",
                    micro_batch=data,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=0,
                    num_micro_batches=len(micro_batches),
                )

        num_micro_batches = len(micro_batches)
        if debug_ref_microbatch:
            print(
                " | ".join(
                    [
                        f"DEBUG_REF_MICROBATCH {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"pid={os.getpid()} rank={_debug_distributed_rank()} "
                        f"local_rank={os.environ.get('LOCAL_RANK', '?')}",
                        "start",
                        f"data_len={len(data)} micro_batch_size={micro_batch_size} "
                        f"use_dynamic_bsz={use_dynamic_bsz} num_micro_batches={num_micro_batches}",
                        _debug_cuda_memory_summary(),
                        _debug_micro_batch_summary(data, pad_token_id),
                    ]
                ),
                flush=True,
            )

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="before_to_device",
                    micro_batch=micro_batch,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=micro_batch_idx,
                    num_micro_batches=num_micro_batches,
                )
            micro_batch = micro_batch.to(get_device_id())
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="after_to_device_before_forward",
                    micro_batch=micro_batch,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=micro_batch_idx,
                    num_micro_batches=num_micro_batches,
                )
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            if debug_ref_microbatch:
                _debug_ref_micro_batch(
                    label="after_forward",
                    micro_batch=micro_batch,
                    pad_token_id=pad_token_id,
                    micro_batch_idx=micro_batch_idx,
                    num_micro_batches=num_micro_batches,
                    outputs=outputs,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        if debug_ref_microbatch:
            print(
                " | ".join(
                    [
                        f"DEBUG_REF_MICROBATCH {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"pid={os.getpid()} rank={_debug_distributed_rank()} "
                        f"local_rank={os.environ.get('LOCAL_RANK', '?')}",
                        "before_concat",
                        f"log_probs_parts={len(log_probs_lst)}",
                        _debug_cuda_memory_summary(),
                    ]
                ),
                flush=True,
            )
        log_probs = torch.concat(log_probs_lst, dim=0)
        if debug_ref_microbatch:
            print(
                " | ".join(
                    [
                        f"DEBUG_REF_MICROBATCH {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"pid={os.getpid()} rank={_debug_distributed_rank()} "
                        f"local_rank={os.environ.get('LOCAL_RANK', '?')}",
                        "after_concat",
                        f"log_probs_shape={tuple(log_probs.shape)} log_probs_gb={_debug_tensor_size_gb(log_probs):.6f}",
                        _debug_cuda_memory_summary(),
                    ]
                ),
                flush=True,
            )
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if "old_log_probs" not in data.batch:
            # when skip old_log_prob recompute, we need to skip old_log_probs to the select_keys
            select_keys.remove("old_log_probs")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if self.config.force_on_policy:
            if self.config.ppo_mini_batch_size != data.batch.batch_size[0]:
                print(
                    f"WARNING: force on-policy (original ppo_mini_batch_size: {self.config.ppo_mini_batch_size}, "
                    f"new: {data.batch.batch_size[0]})"
                )
                self.config.ppo_mini_batch_size = data.batch.batch_size[0]
            if self.config.ppo_epochs != 1:
                print(f"WARNING: force on-policy (original ppo_epochs: {self.config.ppo_epochs}, new: 1)")
                self.config.ppo_epochs = 1
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        debug_actor_update = _ACTOR_UPDATE_DEBUG_ENABLED
        if debug_actor_update:
            _debug_actor_update_event(
                label="start",
                pad_token_id=pad_token_id,
                micro_batch=data,
                extra=(
                    f"ppo_epochs={self.config.ppo_epochs} mini_batches={len(mini_batches)} "
                    f"ppo_mini_batch_size={self.config.ppo_mini_batch_size} "
                    f"use_dynamic_bsz={self.config.use_dynamic_bsz} "
                    f"ppo_max_token_len_per_gpu={self.config.ppo_max_token_len_per_gpu}"
                ),
            )

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                num_micro_batches = len(micro_batches)
                if debug_actor_update:
                    _debug_actor_update_event(
                        label="mini_batch_start",
                        pad_token_id=pad_token_id,
                        micro_batch=mini_batch,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        extra=f"num_micro_batches={num_micro_batches}",
                    )

                self.actor_optimizer.zero_grad()
                if debug_actor_update:
                    _debug_actor_update_event(
                        label="after_zero_grad",
                        pad_token_id=pad_token_id,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                    )

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    if debug_actor_update:
                        _debug_actor_update_event(
                            label="before_to_device",
                            pad_token_id=pad_token_id,
                            micro_batch=micro_batch,
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            micro_batch_idx=micro_batch_idx,
                            num_micro_batches=num_micro_batches,
                        )
                    micro_batch = micro_batch.to(get_device_id())
                    if debug_actor_update:
                        _debug_actor_update_event(
                            label="after_to_device_before_forward",
                            pad_token_id=pad_token_id,
                            micro_batch=micro_batch,
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            micro_batch_idx=micro_batch_idx,
                            num_micro_batches=num_micro_batches,
                        )
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    if self.config.skip_old_log_prob_recompute:
                        calculate_entropy = True
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    if debug_actor_update:
                        _debug_actor_update_event(
                            label="after_forward",
                            pad_token_id=pad_token_id,
                            micro_batch=micro_batch,
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            micro_batch_idx=micro_batch_idx,
                            num_micro_batches=num_micro_batches,
                            outputs=outputs,
                        )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if debug_actor_update:
                        try:
                            loss_value = float(loss.detach().item())
                        except Exception:
                            loss_value = float("nan")
                        _debug_actor_update_event(
                            label="before_backward",
                            pad_token_id=pad_token_id,
                            micro_batch=micro_batch,
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            micro_batch_idx=micro_batch_idx,
                            num_micro_batches=num_micro_batches,
                            extra=f"loss={loss_value:.8g}",
                        )
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if debug_actor_update:
                        _debug_actor_update_event(
                            label="after_backward",
                            pad_token_id=pad_token_id,
                            micro_batch=micro_batch,
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            micro_batch_idx=micro_batch_idx,
                            num_micro_batches=num_micro_batches,
                        )

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                if debug_actor_update:
                    _debug_actor_update_event(
                        label="before_optimizer_step",
                        pad_token_id=pad_token_id,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                    )
                grad_norm = self._optimizer_step()
                if debug_actor_update:
                    try:
                        grad_norm_value = float(grad_norm.detach().item())
                    except Exception:
                        grad_norm_value = float("nan")
                    _debug_actor_update_event(
                        label="after_optimizer_step",
                        pad_token_id=pad_token_id,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        extra=f"grad_norm={grad_norm_value:.8g}",
                    )
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if debug_actor_update:
            _debug_actor_update_event(label="end", pad_token_id=pad_token_id)
        return metrics
