from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any

from PIL import Image


BBox = tuple[int, int, int, int]
QWEN3_VL_COORD_RANGE = (1000, 1000)
QWEN_IMAGE_MAX_ASPECT_RATIO = 200.0


@dataclass
class PresentedImageState:
    image: Image.Image
    source_original_img_idx: int
    bbox_on_original: BBox
    display_size: tuple[int, int]


def resize_dims_by_factor(size: tuple[int, int], factor: float) -> tuple[int, int]:
    width, height = size
    return (max(1, round(width * factor)), max(1, round(height * factor)))


def cap_size_by_area(size: tuple[int, int], max_area: int) -> tuple[int, int]:
    width, height = size
    area = width * height
    if area <= max_area or max_area <= 0:
        return size
    ratio = (max_area / float(area)) ** 0.5
    return (max(1, int(width * ratio)), max(1, int(height * ratio)))


def clamp_bbox_to_image(bbox: BBox, image_size: tuple[int, int]) -> BBox | None:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    clamped = (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        return None
    return clamped


def translate_bbox_to_original(parent: PresentedImageState, bbox_on_presented: BBox) -> BBox | None:
    presented_w, presented_h = parent.display_size
    if presented_w <= 0 or presented_h <= 0:
        return None
    original_x1, original_y1, original_x2, original_y2 = parent.bbox_on_original
    original_w = original_x2 - original_x1
    original_h = original_y2 - original_y1
    if original_w <= 0 or original_h <= 0:
        return None
    x1, y1, x2, y2 = bbox_on_presented
    return (
        original_x1 + round(x1 * original_w / presented_w),
        original_y1 + round(y1 * original_h / presented_h),
        original_x1 + round(x2 * original_w / presented_w),
        original_y1 + round(y2 * original_h / presented_h),
    )


def scale_bbox_from_qwen_range(
    bbox_2d: list[float] | tuple[float, float, float, float],
    display_size: tuple[int, int],
) -> BBox | None:
    width, height = display_size
    if width <= 0 or height <= 0:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    except Exception:
        return None
    bbox = (
        round(x1 * width / QWEN3_VL_COORD_RANGE[0]),
        round(y1 * height / QWEN3_VL_COORD_RANGE[1]),
        round(x2 * width / QWEN3_VL_COORD_RANGE[0]),
        round(y2 * height / QWEN3_VL_COORD_RANGE[1]),
    )
    return clamp_bbox_to_image(bbox, display_size)


def image_aspect_ratio(size: tuple[int, int]) -> float:
    width, height = size
    if width <= 0 or height <= 0:
        return float("inf")
    return max(width, height) / min(width, height)


def validate_qwen_image_aspect_ratio(size: tuple[int, int]) -> str | None:
    ratio = image_aspect_ratio(size)
    if ratio > QWEN_IMAGE_MAX_ASPECT_RATIO:
        return (
            "Tool Execution Error "
            f"absolute aspect ratio must be smaller than {int(QWEN_IMAGE_MAX_ASPECT_RATIO)}, got {ratio}"
        )
    return None


def load_prompt_image(image_value: Any) -> Image.Image | None:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        url = image_value.get("url", "")
        if not url:
            return None
        image_value = url

    if not isinstance(image_value, str):
        return None

    path = image_value
    if image_value.startswith("file://"):
        path = urlparse(image_value).path
    try:
        image = Image.open(path)
        image.load()
        return image.convert("RGB")
    except Exception:
        return None


def presented_image_to_export_ref(
    presented_img_idx: int,
    presented: PresentedImageState,
    *,
    kind: str,
    original_images: list[Image.Image],
    parent_presented_img_idx: int | None = None,
    region_description: str | None = None,
    bbox_on_presented: BBox | None = None,
    initial_rescale: float | None = None,
    zoom_in_factor: float | None = None,
    region_display_size_before_zoom: tuple[int, int] | None = None,
) -> dict[str, object]:
    ref: dict[str, object] = {
        "presented_img_idx": presented_img_idx,
        "kind": kind,
        "source_original_img_idx": presented.source_original_img_idx,
        "bbox_on_original": list(presented.bbox_on_original),
        "display_size": list(presented.display_size),
        "original_size": list(original_images[presented.source_original_img_idx].size),
    }
    if parent_presented_img_idx is not None:
        ref["parent_presented_img_idx"] = parent_presented_img_idx
    if region_description is not None:
        ref["region_description"] = region_description
    if bbox_on_presented is not None:
        ref["bbox_on_presented"] = list(bbox_on_presented)
    if initial_rescale is not None:
        ref["initial_rescale"] = initial_rescale
    if zoom_in_factor is not None:
        ref["zoom_in_factor"] = zoom_in_factor
    if region_display_size_before_zoom is not None:
        ref["region_display_size_before_zoom"] = list(region_display_size_before_zoom)
    return ref
