from dataclasses import dataclass

from PIL import Image
from qwen_vl_utils import fetch_image


BBox = tuple[int, int, int, int]


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


def crop_original_image(image: Image.Image, bbox: BBox) -> Image.Image | None:
    clamped = clamp_bbox_to_image(bbox, image.size)
    if clamped is None:
        return None
    return image.crop(clamped)


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


def translate_processed_bbox_to_original(
    parent: PresentedImageState,
    bbox_on_processed: BBox,
    processed_size: tuple[int, int],
) -> BBox | None:
    processed_w, processed_h = processed_size
    if processed_w <= 0 or processed_h <= 0:
        return None
    original_x1, original_y1, original_x2, original_y2 = parent.bbox_on_original
    original_w = original_x2 - original_x1
    original_h = original_y2 - original_y1
    if original_w <= 0 or original_h <= 0:
        return None
    return (
        original_x1 + round(bbox_on_processed[0] * original_w / processed_w),
        original_y1 + round(bbox_on_processed[1] * original_h / processed_h),
        original_x1 + round(bbox_on_processed[2] * original_w / processed_w),
        original_y1 + round(bbox_on_processed[3] * original_h / processed_h),
    )


def resize_bbox_by_rounding(
    bbox: BBox,
    source_wh: tuple[int, int],
    target_wh: tuple[int, int],
) -> BBox | None:
    source_w, source_h = source_wh
    target_w, target_h = target_wh
    if source_w <= 0 or source_h <= 0:
        return None
    return (
        max(0, min(target_w, round(bbox[0] * target_w / source_w))),
        max(0, min(target_h, round(bbox[1] * target_h / source_h))),
        max(0, min(target_w, round(bbox[2] * target_w / source_w))),
        max(0, min(target_h, round(bbox[3] * target_h / source_h))),
    )


def process_presented_image(image: Image.Image, max_pixels: int | None) -> Image.Image:
    fetch_kwargs: dict[str, object] = {"image": image}
    if max_pixels is not None:
        fetch_kwargs["max_pixels"] = max_pixels
    return fetch_image(fetch_kwargs)


def build_processed_child_image(
    source_original: Image.Image,
    bbox_on_original: BBox,
    max_pixels: int | None,
) -> tuple[Image.Image, Image.Image] | None:
    child_image = crop_original_image(source_original, bbox_on_original)
    if child_image is None:
        return None
    return child_image, process_presented_image(child_image, max_pixels)


def resample_original_region(
    original: Image.Image,
    bbox_on_original: BBox,
    target_display_size: tuple[int, int],
    max_area: int,
) -> tuple[Image.Image, tuple[int, int]] | None:
    crop = crop_original_image(original, bbox_on_original)
    if crop is None:
        return None
    capped_size = cap_size_by_area(target_display_size, max_area)
    if crop.size != capped_size:
        crop = crop.resize(capped_size, Image.LANCZOS)
    return crop, capped_size


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
        round(x1 * width / 1000.0),
        round(y1 * height / 1000.0),
        round(x2 * width / 1000.0),
        round(y2 * height / 1000.0),
    )
    return clamp_bbox_to_image(bbox, display_size)


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
