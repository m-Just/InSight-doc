from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"
IMAGE_PLACEHOLDER = "<image>"
GLM4V_IMAGE_TOKEN = "<|image|>"


@dataclass
class PromptLengthEstimate:
    token_count: int
    estimator_name: str
    supported: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptLengthEstimator(Protocol):
    name: str

    def estimate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: list[Any] | None,
        videos: list[Any] | None,
        prompt_ids: list[int],
        processor: Any,
        tokenizer: Any,
        apply_chat_template_kwargs: dict[str, Any],
    ) -> PromptLengthEstimate:
        ...


class TokenizedPromptLengthEstimator:
    name = "tokenized"

    def estimate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: list[Any] | None,
        videos: list[Any] | None,
        prompt_ids: list[int],
        processor: Any,
        tokenizer: Any,
        apply_chat_template_kwargs: dict[str, Any],
    ) -> PromptLengthEstimate:
        del messages, tools, images, videos, processor, tokenizer, apply_chat_template_kwargs
        return PromptLengthEstimate(
            token_count=len(prompt_ids),
            estimator_name=self.name,
            supported=True,
            metadata={"prompt_ids_tokens": len(prompt_ids)},
        )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1
    if use_thumbnail and max_num != 1:
        max_num += 1
    return min_num, max_num


def _get_internvl_target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]:
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def _calculate_internvl_num_patches(
    *,
    width: int,
    height: int,
    image_size: int,
    target_ratios: list[tuple[int, int]],
    use_thumbnail: bool,
) -> int:
    aspect_ratio = width / height
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=width,
        height=height,
        image_size=image_size,
    )
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    if isinstance(image, dict):
        width = image.get("width")
        height = image.get("height")
        if width is not None and height is not None:
            return int(width), int(height)
    raise ValueError(f"cannot infer image size for prompt length estimation from {type(image).__name__}")


class InternVLPromptLengthEstimator:
    name = "internvl"

    def __init__(self, hf_config: Any) -> None:
        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("InternVL prompt length estimator requires hf_config.vision_config")
        self.image_size = int(getattr(vision_config, "image_size"))
        patch_size = int(getattr(vision_config, "patch_size"))
        downsample_ratio = float(getattr(hf_config, "downsample_ratio"))
        self.num_image_token = int((self.image_size // patch_size) ** 2 * (downsample_ratio**2))
        self.min_dynamic_patch = int(getattr(hf_config, "min_dynamic_patch", 1))
        self.max_dynamic_patch = int(getattr(hf_config, "max_dynamic_patch", 1))
        self.dynamic_image_size = bool(getattr(hf_config, "dynamic_image_size", False))
        self.use_thumbnail = bool(getattr(hf_config, "use_thumbnail", False))

    def _target_ratios(self) -> list[tuple[int, int]]:
        min_num, max_num = _resolve_internvl_min_max_num(
            min_dynamic_patch=self.min_dynamic_patch,
            max_dynamic_patch=self.max_dynamic_patch,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.use_thumbnail,
        )
        return _get_internvl_target_ratios(min_num, max_num)

    def _image_feature_size(self, image: Any, target_ratios: list[tuple[int, int]]) -> tuple[int, int]:
        width, height = _image_size(image)
        num_patches = _calculate_internvl_num_patches(
            width=width,
            height=height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )
        return num_patches * self.num_image_token, num_patches

    def estimate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: list[Any] | None,
        videos: list[Any] | None,
        prompt_ids: list[int],
        processor: Any,
        tokenizer: Any,
        apply_chat_template_kwargs: dict[str, Any],
    ) -> PromptLengthEstimate:
        if videos:
            raise ValueError("InternVL prompt length estimator does not support video prompts")

        raw_prompt = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        expanded_prompt = str(raw_prompt)
        image_items = list(images or [])
        placeholder_count = expanded_prompt.count(IMAGE_PLACEHOLDER)
        if placeholder_count != len(image_items):
            raise ValueError(
                "InternVL prompt/image mismatch during length estimation: "
                f"placeholders={placeholder_count} images={len(image_items)}"
            )

        target_ratios = self._target_ratios()
        image_feature_sizes: list[int] = []
        image_num_patches: list[int] = []
        for image in image_items:
            feature_size, num_patches = self._image_feature_size(image, target_ratios)
            image_feature_sizes.append(feature_size)
            image_num_patches.append(num_patches)
            replacement = f"{IMG_START}{IMG_CONTEXT * feature_size}{IMG_END}"
            expanded_prompt = expanded_prompt.replace(IMAGE_PLACEHOLDER, replacement, 1)

        tokenized = tokenizer(text=[expanded_prompt], return_tensors=None)
        input_ids = tokenized["input_ids"][0]
        return PromptLengthEstimate(
            token_count=len(input_ids),
            estimator_name=self.name,
            supported=True,
            metadata={
                "prompt_ids_tokens": len(prompt_ids),
                "image_count": len(image_items),
                "image_feature_tokens": int(sum(image_feature_sizes)),
                "image_feature_sizes": image_feature_sizes,
                "image_num_patches": image_num_patches,
                "placeholder_count": placeholder_count,
                "num_image_token": self.num_image_token,
                "image_size": self.image_size,
                "min_dynamic_patch": self.min_dynamic_patch,
                "max_dynamic_patch": self.max_dynamic_patch,
                "dynamic_image_size": self.dynamic_image_size,
                "use_thumbnail": self.use_thumbnail,
            },
        )


def _smart_resize_glm4v(
    *,
    height: int,
    width: int,
    temporal_patch_size: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    num_frames = temporal_patch_size
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_patch_size) * temporal_patch_size
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Glm4vPromptLengthEstimator:
    name = "glm4v"

    def __init__(self, hf_config: Any) -> None:
        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("GLM4V prompt length estimator requires hf_config.vision_config")
        self.patch_size = int(getattr(vision_config, "patch_size", 14))
        self.merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        self.temporal_patch_size = int(getattr(vision_config, "temporal_patch_size", 2))
        self.min_pixels = 112 * 112
        self.max_pixels = 28 * 28 * 2 * 6144

    def _image_feature_size(self, image: Any) -> tuple[int, tuple[int, int]]:
        width, height = _image_size(image)
        resized_height, resized_width = _smart_resize_glm4v(
            height=height,
            width=width,
            temporal_patch_size=self.temporal_patch_size,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        num_patches = grid_h * grid_w
        return num_patches // (self.merge_size**2), (resized_width, resized_height)

    def estimate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: list[Any] | None,
        videos: list[Any] | None,
        prompt_ids: list[int],
        processor: Any,
        tokenizer: Any,
        apply_chat_template_kwargs: dict[str, Any],
    ) -> PromptLengthEstimate:
        if videos:
            raise ValueError("GLM4V prompt length estimator does not support video prompts")

        raw_prompt = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        expanded_prompt = str(raw_prompt)
        image_items = list(images or [])
        placeholder_count = expanded_prompt.count(GLM4V_IMAGE_TOKEN)
        if placeholder_count != len(image_items):
            raise ValueError(
                "GLM4V prompt/image mismatch during length estimation: "
                f"placeholders={placeholder_count} images={len(image_items)}"
            )

        image_feature_sizes: list[int] = []
        resized_sizes: list[tuple[int, int]] = []
        for image in image_items:
            feature_size, resized_size = self._image_feature_size(image)
            image_feature_sizes.append(feature_size)
            resized_sizes.append(resized_size)
            expanded_prompt = expanded_prompt.replace(GLM4V_IMAGE_TOKEN, GLM4V_IMAGE_TOKEN * feature_size, 1)

        tokenized = tokenizer(text=[expanded_prompt], return_tensors=None)
        input_ids = tokenized["input_ids"][0]
        return PromptLengthEstimate(
            token_count=len(input_ids),
            estimator_name=self.name,
            supported=True,
            metadata={
                "prompt_ids_tokens": len(prompt_ids),
                "image_count": len(image_items),
                "image_feature_tokens": int(sum(image_feature_sizes)),
                "image_feature_sizes": image_feature_sizes,
                "resized_image_sizes": resized_sizes,
                "placeholder_count": placeholder_count,
                "patch_size": self.patch_size,
                "merge_size": self.merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
        )


def _looks_like_internvl(model_path: str | None, hf_config: Any | None) -> bool:
    values: list[str] = []
    if model_path:
        values.append(str(model_path))
    if hf_config is not None:
        values.append(type(hf_config).__name__)
        model_type = getattr(hf_config, "model_type", None)
        if model_type:
            values.append(str(model_type))
        architectures = getattr(hf_config, "architectures", None) or []
        values.extend(str(item) for item in architectures)
    return "internvl" in " ".join(values).lower()


def _looks_like_glm4v(model_path: str | None, hf_config: Any | None) -> bool:
    values: list[str] = []
    if model_path:
        values.append(str(model_path))
    if hf_config is not None:
        values.append(type(hf_config).__name__)
        model_type = getattr(hf_config, "model_type", None)
        if model_type:
            values.append(str(model_type))
        text_config = getattr(hf_config, "text_config", None)
        text_model_type = getattr(text_config, "model_type", None)
        if text_model_type:
            values.append(str(text_model_type))
        architectures = getattr(hf_config, "architectures", None) or []
        values.extend(str(item) for item in architectures)
    return "glm4v" in " ".join(values).lower() or "glm-4" in " ".join(values).lower()


def create_prompt_length_estimator(
    *,
    estimator_name: str | None,
    model_path: str | None,
    require_supported: bool = False,
) -> PromptLengthEstimator:
    normalized = str(estimator_name or "tokenized").strip().lower()
    if normalized in {"", "none", "tokenized", "default"}:
        if require_supported and normalized in {"", "none"}:
            raise ValueError("prompt length estimator is required but disabled")
        return TokenizedPromptLengthEstimator()

    hf_config = None
    if normalized in {"auto", "internvl", "glm4v"}:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:
            if normalized == "internvl" or require_supported:
                raise ValueError(f"failed to load model config for prompt length estimator: {exc}") from exc

    if normalized == "auto":
        if _looks_like_internvl(model_path, hf_config):
            return InternVLPromptLengthEstimator(hf_config)
        if _looks_like_glm4v(model_path, hf_config):
            return Glm4vPromptLengthEstimator(hf_config)
        if require_supported:
            raise ValueError(f"no exact prompt length estimator is available for model={model_path!r}")
        return TokenizedPromptLengthEstimator()

    if normalized == "internvl":
        if not _looks_like_internvl(model_path, hf_config):
            raise ValueError(f"prompt_length_estimator=internvl does not match model={model_path!r}")
        return InternVLPromptLengthEstimator(hf_config)

    if normalized == "glm4v":
        if not _looks_like_glm4v(model_path, hf_config):
            raise ValueError(f"prompt_length_estimator=glm4v does not match model={model_path!r}")
        return Glm4vPromptLengthEstimator(hf_config)

    raise ValueError(f"unsupported prompt_length_estimator={estimator_name!r}")


def prompt_fits_context(
    *,
    prompt_tokens: int,
    max_model_len: int | None,
    max_new_tokens: int = 0,
    safety_margin: int = 0,
) -> bool:
    if max_model_len is None:
        return True
    return prompt_tokens + max(0, max_new_tokens) + max(0, safety_margin) <= max_model_len


def available_new_tokens(
    *,
    prompt_tokens: int,
    max_model_len: int | None,
    safety_margin: int = 0,
) -> int | None:
    if max_model_len is None:
        return None
    return max_model_len - prompt_tokens - max(0, safety_margin)


__all__ = [
    "Glm4vPromptLengthEstimator",
    "InternVLPromptLengthEstimator",
    "PromptLengthEstimate",
    "PromptLengthEstimator",
    "TokenizedPromptLengthEstimator",
    "available_new_tokens",
    "create_prompt_length_estimator",
    "prompt_fits_context",
]
