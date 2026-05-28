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

import copy
import logging
import math
import os
import re
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Sequence

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_vl_utils import extract_vision_info, fetch_image

from verl.utils.import_utils import load_extern_object
from verl.utils.vsearch import fetch_image_wo_resize
from verl.utils.vreasoner_v2_conversation_export import (
    build_conversation_export_path,
    build_repeated_conversation_export_id,
    build_root_conversation_export_id,
    is_conversation_export_complete,
)

logger = logging.getLogger(__name__)


def _resize_dims_by_factor(size: tuple[int, int], factor: float) -> tuple[int, int]:
    width, height = size
    return (max(1, round(width * factor)), max(1, round(height * factor)))


def _cap_size_by_area(size: tuple[int, int], max_area: int) -> tuple[int, int]:
    width, height = size
    area = width * height
    if area <= max_area or max_area <= 0:
        return size
    ratio = (max_area / float(area)) ** 0.5
    return (max(1, int(width * ratio)), max(1, int(height * ratio)))


def collate_fn(data_list: list[dict]) -> dict:
    r"""
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


def concat_multi_modal_inputs(mm_inputs: Sequence[dict], new_mm_inputs: Sequence[dict]) -> list[dict]:
    """Concatenate mm_inputs with new_mm_inputs."""
    assert len(mm_inputs) == len(new_mm_inputs), f"length mismatch: {len(mm_inputs)=}, {len(new_mm_inputs)=}"
    mm_inputs = [mm_input.copy() for mm_input in mm_inputs]
    for i in range(len(mm_inputs)):
        if not mm_inputs[i]:
            mm_inputs[i] = new_mm_inputs[i]
            continue
        if not new_mm_inputs[i]:
            continue
        assert set(mm_inputs[i]) == set(new_mm_inputs[i]), (
            f"mm_inputs[i] and new_mm_inputs[i] have different keys: {mm_inputs[i].keys()=}, {new_mm_inputs[i].keys()=}"
        )
        for key in mm_inputs[i]:
            # print(f"DEBUG: concat multi_modal_inputs.{key}", mm_inputs[i][key].shape, new_mm_inputs[i][key].shape)
            mm_inputs[i][key] = torch.cat([mm_inputs[i][key], new_mm_inputs[i][key]], dim=0)
    return mm_inputs


def _normalize_image_source_ref(image_value: Any) -> dict[str, Any] | None:
    if isinstance(image_value, str):
        if image_value.startswith("file://"):
            return {
                "source_type": "path",
                "path": image_value[7:],
                "uri": image_value,
            }
        if image_value.startswith(("http://", "https://")):
            return {
                "source_type": "url",
                "url": image_value,
            }
        if image_value.startswith("data:image"):
            return {
                "source_type": "data_url",
                "url": image_value,
            }
        return {
            "source_type": "path",
            "path": image_value,
        }
    if isinstance(image_value, Image.Image):
        return None
    return None


def _setup_vsearch_fields(row_dict: dict[str, Any], patch_size: int, config: DictConfig, is_train: bool) -> None:
    """Create vsearch fields (image_ori, image_ori_wh, image_processed_wh) in extra_info.

    Also sets up the image_zoom_in_tool create_kwargs in tools_kwargs, and replaces
    the original images in raw_prompt with processed (resized) images.

    Args:
        row_dict: The row dict returned by RLHFDataset.__getitem__.
        config: The dataset config.
        is_train: Whether this is training mode (affects max_pixels selection).
    """
    extra_info = row_dict["extra_info"]
    raw_prompt = row_dict["raw_prompt"]

    # Determine max_pixels from config and extra_info
    max_pixels_global = config.get("max_pixels")
    if not is_train and config.get("validation_max_pixels"):
        max_pixels_global = config.get("validation_max_pixels")
    max_pixels_sample = extra_info.get("max_pixels")
    if max_pixels_global and max_pixels_sample:
        max_pixels = min(max_pixels_global, max_pixels_sample)
    else:
        max_pixels = max_pixels_global or max_pixels_sample

    # Extract vision info from raw_prompt messages using qwen_vl_utils
    vision_infos = extract_vision_info(raw_prompt)

    # Filter for images only (exclude videos)
    image_elements = [info for info in vision_infos if "image" in info or "image_url" in info]
    assert len(image_elements) >= 1, f"expected at least 1 image element, got {len(image_elements)}"

    if "original_image_refs" not in extra_info or extra_info["original_image_refs"] is None:
        original_image_refs = []
        for img_elem in image_elements:
            if "image" in img_elem:
                original_image_refs.append(_normalize_image_source_ref(img_elem["image"]))
            elif "image_url" in img_elem:
                image_url = img_elem["image_url"]
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                original_image_refs.append(_normalize_image_source_ref(url))
        extra_info["original_image_refs"] = original_image_refs

    # Create image_ori by loading images without resize
    image_ori = [fetch_image_wo_resize(img) for img in image_elements]
    extra_info["image_ori"] = image_ori
    extra_info["image_ori_wh"] = [img.size for img in image_ori]

    # Set max_pixels on image elements so fetch_image uses it for resizing
    if max_pixels is not None:
        for img_elem in image_elements:
            img_elem["max_pixels"] = max_pixels

    # Create processed images using fetch_image (which applies resizing based on max_pixels)
    image_processed = [fetch_image(img) for img in image_elements]
    extra_info["image_processed_wh"] = [img.size for img in image_processed]

    # Replace original images in raw_prompt with processed images
    # image_elements are references to the dicts in raw_prompt, so we can modify them directly
    for img_elem, processed_img in zip(image_elements, image_processed, strict=True):
        img_elem["image"] = processed_img

    # Set up image_zoom_in_tool create_kwargs
    tools_kwargs = row_dict.get("tools_kwargs", {})
    if tools_kwargs is None:
        tools_kwargs = {}
        row_dict["tools_kwargs"] = tools_kwargs

    assert "image_zoom_in_tool" not in tools_kwargs, f"image_zoom_in_tool already in tools_kwargs: {tools_kwargs}"
    create_kwargs = tools_kwargs.get("image_zoom_in_tool", {}).get("create_kwargs", {})
    create_kwargs.update({
        "image": extra_info["image_ori"][0],
        "resized_image_size": extra_info["image_processed_wh"][0],
        "patch_size": patch_size,
    })
    if max_pixels is not None:
        create_kwargs["max_pixels"] = max_pixels
    tools_kwargs.setdefault("image_zoom_in_tool", {})["create_kwargs"] = create_kwargs


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.is_train = bool(config.get("_is_train", True))
        self.conversation_export_dir = config.get("_conversation_export_dir")
        self.conversation_export_resume_mode = config.get("_conversation_export_resume_mode", "off")
        self.conversation_export_validate = bool(config.get("_conversation_export_validate", not self.is_train))
        self.conversation_export_val_trial_idx = config.get("_conversation_export_val_trial_idx")
        self.conversation_export_repeat_count = int(max(1, config.get("_conversation_export_repeat_count", 1)))
        self.validation_image_token_reorder_settings = config.get("_validation_image_token_reorder_settings")

        self._download()
        self._read_files_and_tokenize()

        if self.config.get("use_vsearch", False):
            # TODO: only for Qwen2VLImageProcessor or for all other processors?
            self.processor.image_processor.do_resize = False

    def _build_conversation_export_base_id(self, row_dict: dict[str, Any], sample_index: int) -> str:
        return build_root_conversation_export_id(
            extra_info=row_dict.get("extra_info"),
            data_source=row_dict.get("data_source"),
            validate=self.conversation_export_validate,
            val_trial_idx=self.conversation_export_val_trial_idx,
        )

    def _is_conversation_export_finished(self, base_export_id: str) -> bool:
        if not self.conversation_export_dir:
            return False
        for repeat_idx in range(self.conversation_export_repeat_count):
            export_id = build_repeated_conversation_export_id(base_export_id, repeat_idx)
            export_path = build_conversation_export_path(self.conversation_export_dir, export_id)
            if not is_conversation_export_complete(export_path):
                return False
        return True

    def _prepare_conversation_export_resume(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.conversation_export_dir:
            return dataframe

        keep_indices: list[int] = []
        export_ids: list[str] = []
        skipped_completed = 0
        for sample_index in range(len(dataframe)):
            row_dict = dataframe[sample_index]
            base_export_id = self._build_conversation_export_base_id(row_dict, sample_index)
            if self.conversation_export_resume_mode == "skip_completed" and self._is_conversation_export_finished(base_export_id):
                skipped_completed += 1
                continue
            keep_indices.append(sample_index)
            export_ids.append(base_export_id)

        if len(keep_indices) != len(dataframe):
            dataframe = dataframe.select(keep_indices)

        def attach_export_id(example, idx):
            extra_info = dict(example.get("extra_info") or {})
            base_export_id = str(export_ids[idx])
            extra_info["conversation_export_base_id"] = base_export_id
            existing_export_id = extra_info.get("conversation_export_id")
            if not isinstance(existing_export_id, str) or not existing_export_id:
                extra_info["conversation_export_id"] = base_export_id
            example["extra_info"] = extra_info
            return example

        map_features = copy.deepcopy(dataframe.features)
        extra_info_features = map_features.get("extra_info")
        if isinstance(extra_info_features, dict):
            extra_info_features["conversation_export_base_id"] = datasets.Value("string")
            extra_info_features["conversation_export_id"] = datasets.Value("string")

        dataframe = dataframe.map(
            attach_export_id,
            with_indices=True,
            desc="Attaching deterministic conversation export ids",
            features=map_features,
        )
        if self.conversation_export_resume_mode == "skip_completed":
            print(f"conversation export resume: skipped {skipped_completed} completed samples")
        return dataframe

    def _get_image_size_for_token_estimate(self, image_value: Any) -> tuple[int, int] | None:
        if isinstance(image_value, Image.Image):
            return image_value.size

        if isinstance(image_value, dict):
            if "bytes" in image_value:
                try:
                    with Image.open(BytesIO(image_value["bytes"])) as image:
                        return image.size
                except Exception:
                    return None
            if "image" in image_value:
                return self._get_image_size_for_token_estimate(image_value["image"])
            if "image_url" in image_value:
                image_url = image_value["image_url"]
                if isinstance(image_url, dict):
                    return self._get_image_size_for_token_estimate(image_url.get("url"))
                return self._get_image_size_for_token_estimate(image_url)
            return None

        if isinstance(image_value, str):
            if image_value.startswith("file://"):
                image_value = image_value[7:]
            if image_value.startswith(("http://", "https://", "data:image")):
                return None
            try:
                with Image.open(image_value) as image:
                    return image.size
            except Exception:
                return None

        return None

    def _get_validation_image_reorder_agent_settings(self, row_dict: dict[str, Any]) -> dict[str, Any] | None:
        settings = self.validation_image_token_reorder_settings or {}
        agent_settings_by_name = settings.get("agent_settings_by_name") or {}
        if not agent_settings_by_name:
            return None

        agent_name = row_dict.get("agent_name")
        if agent_name in agent_settings_by_name:
            return agent_settings_by_name[agent_name]

        default_agent_loop = settings.get("default_agent_loop")
        if default_agent_loop in agent_settings_by_name:
            return agent_settings_by_name[default_agent_loop]

        if len(agent_settings_by_name) == 1:
            return next(iter(agent_settings_by_name.values()))

        return None

    def _estimate_image_token_cost(self, row_dict: dict[str, Any]) -> int:
        agent_settings = self._get_validation_image_reorder_agent_settings(row_dict)
        if agent_settings is None:
            return 0

        extra_info = row_dict.get("extra_info") or {}
        initial_rescale = float(extra_info.get("initial_rescale", agent_settings.get("initial_rescale", 1.0)))
        gpt_image_max_area = int(agent_settings.get("gpt_image_max_area", 0))

        total_tokens = 0
        for image_value in row_dict.get(self.image_key, None) or []:
            original_size = self._get_image_size_for_token_estimate(image_value)
            if original_size is None:
                continue
            presented_size = _cap_size_by_area(
                _resize_dims_by_factor(original_size, initial_rescale),
                gpt_image_max_area,
            )
            total_tokens += max(1, math.ceil(presented_size[0] / 32.0) * math.ceil(presented_size[1] / 32.0))
        return total_tokens

    @staticmethod
    def _split_group_sizes(total_items: int, num_groups: int) -> list[int]:
        if total_items <= 0 or num_groups <= 0:
            return []
        base = total_items // num_groups
        remainder = total_items % num_groups
        return [base + (1 if i < remainder else 0) for i in range(num_groups)]

    def _build_balanced_batch_order(
        self,
        batch_indices: list[int],
        batch_costs: list[int],
        num_groups: int,
    ) -> tuple[list[int], list[list[int]], list[int]]:
        if not batch_indices:
            return [], [], []

        num_groups = max(1, min(num_groups, len(batch_indices)))
        target_sizes = self._split_group_sizes(len(batch_indices), num_groups)
        groups: list[list[int]] = [[] for _ in range(num_groups)]
        group_costs = [0 for _ in range(num_groups)]

        for local_idx in sorted(range(len(batch_indices)), key=lambda idx: batch_costs[idx], reverse=True):
            candidate_group_idxs = [
                group_idx for group_idx, target_size in enumerate(target_sizes) if len(groups[group_idx]) < target_size
            ]
            chosen_group_idx = min(candidate_group_idxs, key=lambda group_idx: (group_costs[group_idx], len(groups[group_idx])))
            groups[chosen_group_idx].append(batch_indices[local_idx])
            group_costs[chosen_group_idx] += batch_costs[local_idx]

        ordered_indices = [idx for group in groups for idx in group]
        return ordered_indices, groups, group_costs

    def _reorder_validation_by_image_tokens(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        settings = self.validation_image_token_reorder_settings or {}
        if self.is_train or not settings or not settings.get("enabled", False):
            return dataframe
        if len(dataframe) == 0:
            return dataframe

        batch_size = settings.get("batch_size")
        if batch_size is None or int(batch_size) <= 0:
            batch_size = len(dataframe)
        batch_size = min(int(batch_size), len(dataframe))
        num_groups = max(1, int(settings.get("num_workers", 1)))
        num_batches = math.ceil(len(dataframe) / batch_size)

        costs = [self._estimate_image_token_cost(dataframe[idx]) for idx in range(len(dataframe))]
        batch_target_sizes = self._split_group_sizes(len(dataframe), num_batches)
        balanced_batches: list[list[int]] = [[] for _ in range(num_batches)]
        balanced_batch_costs = [0 for _ in range(num_batches)]

        for sample_idx in sorted(range(len(dataframe)), key=lambda idx: costs[idx], reverse=True):
            candidate_batch_idxs = [
                batch_idx
                for batch_idx, target_size in enumerate(batch_target_sizes)
                if len(balanced_batches[batch_idx]) < target_size
            ]
            chosen_batch_idx = min(
                candidate_batch_idxs,
                key=lambda batch_idx: (balanced_batch_costs[batch_idx], len(balanced_batches[batch_idx]), batch_idx),
            )
            balanced_batches[chosen_batch_idx].append(sample_idx)
            balanced_batch_costs[chosen_batch_idx] += costs[sample_idx]

        naive_batch_costs = []
        for batch_start in range(0, len(dataframe), batch_size):
            batch_indices = list(range(batch_start, min(batch_start + batch_size, len(dataframe))))
            naive_batch_costs.append(sum(costs[idx] for idx in batch_indices))

        reordered_indices: list[int] = []
        naive_group_gap_values: list[int] = []
        balanced_group_gap_values: list[int] = []
        naive_group_max_values: list[int] = []
        balanced_group_max_values: list[int] = []
        example_batch_summaries: list[str] = []

        for batch_idx, batch_indices in enumerate(balanced_batches):
            batch_costs = [costs[idx] for idx in batch_indices]
            effective_groups = max(1, min(num_groups, len(batch_indices)))

            naive_target_sizes = self._split_group_sizes(len(batch_indices), effective_groups)
            naive_group_costs = []
            cursor = 0
            for group_size in naive_target_sizes:
                naive_group_costs.append(sum(batch_costs[cursor : cursor + group_size]))
                cursor += group_size

            batch_reordered_indices, groups, balanced_group_costs = self._build_balanced_batch_order(
                batch_indices=batch_indices,
                batch_costs=batch_costs,
                num_groups=effective_groups,
            )
            reordered_indices.extend(batch_reordered_indices)

            naive_group_gap_values.append(max(naive_group_costs) - min(naive_group_costs))
            balanced_group_gap_values.append(max(balanced_group_costs) - min(balanced_group_costs))
            naive_group_max_values.append(max(naive_group_costs))
            balanced_group_max_values.append(max(balanced_group_costs))

            if len(example_batch_summaries) < 5:
                group_sizes = [len(group) for group in groups]
                example_batch_summaries.append(
                    f"batch[{batch_idx}] "
                    f"batch_total_cost={sum(batch_costs)} "
                    f"group_sizes={group_sizes} "
                    f"group_costs={balanced_group_costs}"
                )

        dataframe = dataframe.select(reordered_indices)
        print(
            "validation image-token reordering: "
            f"samples={len(dataframe)} "
            f"num_batches={num_batches} "
            f"batch_size={batch_size} "
            f"num_groups={num_groups} "
            f"estimated_tokens_total={sum(costs)} "
            f"estimated_tokens_avg={sum(costs) / len(costs):.1f} "
            f"naive_batch_cost_mean={np.mean(naive_batch_costs):.1f} "
            f"balanced_batch_cost_mean={np.mean(balanced_batch_costs):.1f} "
            f"naive_batch_gap_mean={np.mean([abs(cost - np.mean(naive_batch_costs)) for cost in naive_batch_costs]):.1f} "
            f"balanced_batch_gap_mean={np.mean([abs(cost - np.mean(balanced_batch_costs)) for cost in balanced_batch_costs]):.1f} "
            f"naive_batch_min={min(naive_batch_costs)} "
            f"naive_batch_max={max(naive_batch_costs)} "
            f"balanced_batch_min={min(balanced_batch_costs)} "
            f"balanced_batch_max={max(balanced_batch_costs)} "
            f"naive_group_max_cost_mean={np.mean(naive_group_max_values):.1f} "
            f"balanced_group_max_cost_mean={np.mean(balanced_group_max_values):.1f} "
            f"naive_group_gap_mean={np.mean(naive_group_gap_values):.1f} "
            f"balanced_group_gap_mean={np.mean(balanced_group_gap_values):.1f} "
            f"naive_group_gap_max={max(naive_group_gap_values)} "
            f"balanced_group_gap_max={max(balanced_group_gap_values)}"
        )
        for summary in example_batch_summaries:
            print(f"validation image-token reordering example: {summary}")
        return dataframe

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        if self.config.get("force_dataset_concat", False) and len(dataframes) > 1:
            print("WARNING: Force dataset concatenation")
            extra_infos = [set(df["extra_info"][0].keys()) for df in dataframes]
            extra_infos_union = set.union(*extra_infos)
            print(f"Extra infos union: {extra_infos_union}")

            def map_extra_info(example):
                extra_info = example.pop("extra_info")
                for key in list(extra_infos_union):
                    if key not in extra_info:
                        extra_info[key] = None
                assert "extra_info_union" not in example, "extra_info_union already exists"
                example["extra_info_union"] = extra_info
                return example

            def map_extra_info_back(example):
                example["extra_info"] = example.pop("extra_info_union")
                return example

            dataframes = [df.map(map_extra_info).map(map_extra_info_back) for df in dataframes]
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.conversation_export_resume_mode not in (None, "off"):
            self.dataframe = self._prepare_conversation_export_resume(self.dataframe)
            total = len(self.dataframe)
            print(f"dataset len after conversation export resume: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        self.dataframe = self._reorder_validation_by_image_tokens(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", **image}
        - <video>: {"type": "video", **video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[self.prompt_key]
        # When concatenating image and video datasets, pop will return None for image or video sample
        images = example.pop(self.image_key, None) or []
        videos = example.pop(self.video_key, None) or []

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        content_list.append({"type": "image", "image": image})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        content_list.append({"type": "image", **image})
                    else:
                        raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image)}")
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])

        # print(f"DEBUG: {tools_kwargs=}")
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        # InSightQwenAgentLoop loads prompt images on the rollout worker from refs in
        # raw_prompt, so eagerly materializing vsearch image blobs here only inflates
        # host RAM before the batch is repeated for rollout.
        if self.config.get("use_vsearch", False) and row_dict.get("agent_name") != "insight_qwen_agent":
            _setup_vsearch_fields(row_dict, self.processor.image_processor.patch_size, self.config, self._is_train)

        return row_dict

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[RLHFDataset]: list of RLHFDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in RLHFDataset\n"
                "reason: _read_files_and_tokenize() not called or Parquet file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("RLHFDataset dataframe 为 None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")
        if total_samples % num_splits != 0:
            raise ValueError(f"Cannot split dataset size {total_samples} into {num_splits} splits")
        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = self.dataframe.select(range(start_idx, end_idx))

            split_dataset = RLHFDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                processor=self.processor,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits


def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls
