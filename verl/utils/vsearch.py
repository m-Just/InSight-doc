import base64
import json
import re
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from qwen_vl_utils.vision_process import to_rgb

BBox = tuple[int, int, int, int]

SPECIAL_BBOXES = set([(0, 0, 0, 0)])


def validate_bbox(bbox: Any) -> BBox:
    """ Validate the bbox type and format [x1, y1, x2, y2]. """
    if not(
        isinstance(bbox, tuple) and
        len(bbox) == 4 and
        all(isinstance(n, int) for n in bbox) and
        0 <= bbox[0] < bbox[2] and
        0 <= bbox[1] < bbox[3]
    ):
        raise ValueError(f'invalid bbox: {bbox}')
    return bbox


def extract_bbox_from_tool_call(tool_call_json: str) -> BBox:
    """Extract the bbox from the tool call JSON.

    Expect the tool call JSON to be in the following format:
        {
            "name": "image_zoom_in_tool",
            "arguments": {
                "bbox_2d": [x1, y1, x2, y2]
            }
        }
    where x1, y1, x2, y2 are expected to be integers such that x1 < x2 and y1 < y2.

    Args:
        tool_call_json: the tool call JSON

    Returns:
        the bbox, a tuple of 4 integers
    """
    try:
        tool_call = json.loads(tool_call_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"failed to parse tool call JSON: {e}") from e
    bbox = tuple(tool_call["arguments"]["bbox_2d"])
    return validate_bbox(bbox)


def extract_final_bbox_from_response(response: str) -> BBox:
    def maybe_truncate(text: str, max_length: int = 500) -> str:
        """Left-truncate text if it is too long."""
        return ("..." + text[-max_length:]) if len(text) > max_length else text

    # Search for the *last* bbox match in the response
    pattern = r"\]" + ",".join([r"\s*\d+\s*"] * 4) + r"\["
    match = re.search(pattern, response[::-1])
    if not match:
        raise ValueError(f"no bbox found in response: {maybe_truncate(response)}")
    bbox_str = match.group(0)[::-1]
    try:
        bbox = tuple(eval(bbox_str))
        if bbox not in SPECIAL_BBOXES:
            validate_bbox(bbox)
    except Exception as e:
        raise ValueError(f"invalid bbox: {bbox_str}") from e

    return bbox


def resize_bbox(bbox: BBox, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> BBox:
    bbox_resized = [-1] * 4
    for i in range(4):
        bbox_resized[i] = round(bbox[i] * target_wh[i % 2] / source_wh[i % 2])
    return validate_bbox(tuple(bbox_resized))  # type: ignore


def bbox_area(box: BBox) -> float:
    """Return area of a 2D bbox in the format (x1, y1, x2, y2)."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_area(box1: BBox, box2: BBox) -> float:
    """Calculate intersection area between two bounding boxes in the format (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(box1: BBox, box2: BBox) -> float:
    """Calculate IoU between two bounding boxes in the format (x1, y1, x2, y2)."""
    intersection = intersection_area(box1, box2)
    union = bbox_area(box1) + bbox_area(box2) - intersection
    return intersection / union if union > 0 else 0.0


def compute_overall_iou_with_gt(
    bboxes_crop: list[BBox], bboxes_gt: list[BBox], image_processed_wh: tuple[int, int], image_ori_wh: tuple[int, int]
) -> float:
    # Resize the predicted bboxes to the original image size
    bboxes = [(0, 0, *image_ori_wh)]
    for bbox_crop in bboxes_crop:
        bbox = resize_bbox(bbox_crop, image_processed_wh, image_ori_wh)
        bbox = (max(0, bbox[0]), max(0, bbox[1]), min(image_ori_wh[0], bbox[2]), min(image_ori_wh[1], bbox[3]))
        bboxes.append(bbox)

    # Compute the iou between the predicted bboxes and the ground truth bboxes
    ious = []
    for gt_bbox in bboxes_gt:
        cur_ious = []
        gt_bbox_area = bbox_area(gt_bbox)
        assert gt_bbox_area > 0, f"gt_bbox_area is 0 for {gt_bbox}"
        for bbox in bboxes:
            # Count the iou only when the gt_bbox is mostly covered by the bbox
            if intersection_area(bbox, gt_bbox) / gt_bbox_area > 0.95:
                cur_ious.append(iou(bbox, gt_bbox))
            else:
                cur_ious.append(0.0)
        ious.append(cur_ious)
    overall_iou = min(max(cur_ious) for cur_ious in ious)
    return overall_iou


def avg_max_iou(bboxes: list[BBox]) -> float:
    """
    Calculate the average of the maximum IoU for each bounding box with all previous bounding boxes.
    bboxes: list of bounding boxes, each in format [x1, y1, x2, y2]
    """
    if len(bboxes) < 2:
        return 0.0

    ious = []
    for i in range(1, len(bboxes)):
        max_iou = 0.0
        for j in range(i):
            iou_value = iou(bboxes[i], bboxes[j])
            if iou_value > max_iou:
                max_iou = iou_value
        ious.append(max_iou)
    return sum(ious) / len(ious)


def get_bbox(
    text: str, image_wh: tuple[int, int], processed_image_wh: tuple[int, int], pattern: str
) -> tuple[BBox, str]:
    bbox = re.search(pattern, text, re.DOTALL)
    if not bbox:
        return None, "no regex match"

    bbox = bbox.group(1)
    try:
        bbox = json.loads(bbox)
    except json.JSONDecodeError:
        return None, "json.loads failed"
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, int) for n in bbox):
        for i in range(4):
            bbox[i] *= image_wh[i % 2] / processed_image_wh[i % 2]
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(image_wh[0], bbox[2])
        bbox[3] = min(image_wh[1], bbox[3])
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            return tuple(bbox), "success"
        else:
            return None, f"invalid bbox coordinates: {bbox}"
    else:
        return None, "fetched bbox is not a list of 4 integers"


# this function is adapted from qwen_vl_utils/vision_process.py:fetch_image
def fetch_image_wo_resize(ele: dict[str, str | Image.Image]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    return image


def crop_image(im: Image.Image, bbox: BBox) -> tuple[Image.Image, BBox]:
    # assuming bbox format is [x1, y1, x2, y2]
    w, h = im.size
    bbox_crop = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(w, bbox[2]),
        min(h, bbox[3])
    )
    if bbox_crop[0] < bbox_crop[2] and bbox_crop[1] < bbox_crop[3]:
        img_cropped = im.crop(bbox_crop)
        return img_cropped, bbox_crop
    else:
        raise ValueError(f'bbox_crop is not valid: {bbox_crop}')


def expanded_crop_resize(
    image: Image.Image,
    bboxes: list[BBox],
    frame: tuple[int, int] = (256, 768),
    expansion_factor: float = 0.2,
    bbox_format: str = "xywh",
) -> Image.Image:
    assert bbox_format in ["xywh", "xyxy"], "bbox_format must be 'xywh' or 'xyxy'"

    # Convert bboxes to xyxy format if they are in xywh format
    if bbox_format == "xywh":
        bboxes = [(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]) for bbox in bboxes]

    # Find the minimum covering box for all bboxes
    min_x = min([bbox[0] for bbox in bboxes])
    min_y = min([bbox[1] for bbox in bboxes])
    max_x = max([bbox[2] for bbox in bboxes])
    max_y = max([bbox[3] for bbox in bboxes])

    # Expand the region by expansion_factor
    expansion_x = (max_x - min_x) * expansion_factor
    expansion_y = (max_y - min_y) * expansion_factor

    min_x = max(0, int(min_x - expansion_x))
    min_y = max(0, int(min_y - expansion_y))
    max_x = min(image.size[0], int(max_x + expansion_x))
    max_y = min(image.size[1], int(max_y + expansion_y))

    # Crop the image to the expanded region
    cropped_img = image.crop((min_x, min_y, max_x, max_y))

    if frame[0] is None or frame[1] is None:
        return cropped_img
    else:
        # Resize the image to fit within the give frame while maintaining aspect ratio
        region_width, region_height = cropped_img.size

        max_size = max(region_width, region_height)
        min_size = min(region_width, region_height)
        scale_factor = min(max(frame) / max_size, min(frame) / min_size)

        new_width = int(region_width * scale_factor)
        new_height = int(region_height * scale_factor)

        resized_cropped_img = cropped_img.resize((new_width, new_height), Image.LANCZOS)
        return resized_cropped_img
