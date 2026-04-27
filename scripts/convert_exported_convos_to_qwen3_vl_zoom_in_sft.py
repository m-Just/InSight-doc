#!/usr/bin/env python3
"""
Convert exported vReasoner/VSearcher conversation JSON files into a parquet dataset
for the engine SFT trainer, preserving Qwen-style tool turns as closely as possible.

Output rows follow the stock MultiTurnSFTDataset schema:
  - messages: list[dict]
  - message_loss_mask: list[bool]
  - images: list[{"bytes": ...}] or list[{"image": "file:///..."}]
  - tools: list[tool_schema]

Key conventions:
  - assistant tool requests become structured assistant.tool_calls
  - tool outputs become role="tool"
  - image references become "<image>" placeholders plus entries in the flat images list

Conversion summary:
  - system_prompt -> role="system"
  - query -> role="user"
  - assistant answer -> assistant content formatted as
    "<think>...</think>\\n<answer>...</answer>"
  - assistant tool_call -> converted to qwen-agent-style image_zoom_in_tool calls only
    when it can be matched exactly to a successful exported region_crop
  - successful tool_result -> role="tool" with minimal indexed image content
    like "Image N:<image>"
  - assistant malformed "others" turns are only tolerated when they are immediately
    followed by format_repair_hint and --stitch-runtime-hints is enabled; in that case
    both the malformed assistant turn and the hint are removed
  - any remaining assistant "others" turn causes the whole conversation to be dropped
  - with --stitch-runtime-hints, embedded LAST_ROUND_HINT text is also stripped from
    multimodal tool_result messages that carry secondary_types=["last_round_hint"]
  - tool_result_fail_hint and other non-exactly-convertible tool failures cause the whole
    conversation to be dropped
  - with --stitch-runtime-hints:
    - assistant others + following format_repair_hint are removed together
    - last_round_hint is removed

Decision table:
  - system_prompt: keep, mask
  - query: keep, mask
  - assistant answer: keep, train
  - assistant tool_call + exact successful region_crop alignment: keep, train
  - successful tool_result: keep, mask
  - assistant others with following format_repair_hint and --stitch-runtime-hints:
    stitch out malformed assistant turn and hint
  - remaining assistant others: drop conversation
  - embedded last_round_hint inside a multimodal tool_result with --stitch-runtime-hints:
    strip the hint text from that kept tool_result
  - format_repair_hint with --stitch-runtime-hints: already handled by stitching out the
    preceding malformed assistant turn and this hint together
  - last_round_hint with --stitch-runtime-hints: stitch out hint
  - format_repair_hint without --stitch-runtime-hints: keep, mask
  - last_round_hint without --stitch-runtime-hints: keep, mask
  - tool_result_fail_hint: drop conversation
  - malformed or non-exact tool_call episode: drop conversation
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

import verl.utils.vreasoner_v2_prompt as prompts
from verl.utils.vreasoner_v2_conversation_export import load_exported_conversation, restore_presented_images


IMAGE_ZOOM_IN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) "
            "and an object label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where "
                        "(x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box.",
                },
                "img_idx": {
                    "type": "number",
                    "description": "The index of the zoomed-in image (starting from 0).",
                },
            },
            "required": ["bbox_2d", "label", "img_idx"],
        },
    },
}


VSEARCHER_QWEN3_VL_SYSTEM_PROMPT = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""


class DropConversationError(RuntimeError):
    """Raised when a conversation cannot be converted conservatively."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing exported conversation JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train.parquet and val.parquet.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Fraction of rows to put into validation.")
    parser.add_argument(
        "--output-parquet-name",
        type=str,
        default=None,
        help=(
            "Override the default output parquet filename. "
            "When specified, --val-ratio must be 0 and only a single parquet is written."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for train/val split.")
    parser.add_argument(
        "--system-prompt-mode",
        choices=["exported", "vsearcher_qwen3_vl"],
        default="vsearcher_qwen3_vl",
        help="Which system prompt to place into converted rows.",
    )
    parser.add_argument(
        "--assistant-format-mode",
        choices=["tagged", "plain"],
        default="plain",
        help="Whether assistant reasoning/answers keep <think>/<answer> tags or are written as plain text.",
    )
    parser.add_argument(
        "--stitch-runtime-hints",
        action="store_true",
        help=(
            "Drop last-round hints and stitch away format-repair hint episodes by removing the hint and the "
            "preceding malformed assistant turn when applicable."
        ),
    )
    parser.add_argument(
        "--image-storage-mode",
        choices=["bytes", "path"],
        default="path",
        help="Store images inline as bytes or on disk as file URIs in an output-dir image cache.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for per-conversation conversion.",
    )
    parser.add_argument(
        "--only-correct-answers",
        action="store_true",
        help="Keep only exported conversations whose accuracy_reward is exactly 1.0.",
    )
    parser.add_argument(
        "--wrong-question-ids-only",
        action="store_true",
        help=(
            "Only write wrong_question_ids.txt and skip parquet generation. "
            "Requires --only-correct-answers."
        ),
    )
    return parser.parse_args()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def materialize_image(image: Image.Image, image_storage_mode: str, image_cache_dir: Path | None) -> dict[str, Any]:
    png_bytes = image_to_png_bytes(image)
    if image_storage_mode == "bytes":
        return {"bytes": png_bytes}
    if image_storage_mode == "path":
        if image_cache_dir is None:
            raise ValueError("image_cache_dir is required when image_storage_mode='path'")
        image_cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(png_bytes).hexdigest()
        image_path = image_cache_dir / f"{digest}.png"
        if not image_path.exists():
            tmp_path = image_cache_dir / f".{digest}.{os.getpid()}.tmp"
            tmp_path.write_bytes(png_bytes)
            os.replace(tmp_path, image_path)
        return {"image": image_path.resolve().as_uri()}
    raise ValueError(f"Unsupported image_storage_mode: {image_storage_mode}")


def normalize_text(text: str) -> str:
    return text if text is not None else ""


def resolve_system_prompt(original_text: str, mode: str) -> str:
    if mode == "exported":
        return normalize_text(original_text)
    if mode == "vsearcher_qwen3_vl":
        return VSEARCHER_QWEN3_VL_SYSTEM_PROMPT
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def build_assistant_content(content: dict[str, Any], message_type: str, assistant_format_mode: str) -> str:
    think = normalize_text(content.get("think", ""))
    if message_type == "answer":
        answer = normalize_text(content.get("answer", ""))
        if assistant_format_mode == "plain":
            if think and answer:
                return f"{think}\n\n{answer}"
            return think or answer
        return f"<think>{think}</think>\n<answer>{answer}</answer>"
    if think:
        if assistant_format_mode == "plain":
            return think
        return f"<think>{think}</think>"
    return ""


def build_tool_calls(content: dict[str, Any]) -> list[dict[str, Any]]:
    payload = content.get("tool_call")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected parsed tool_call dict, got {type(payload)}")
    name = payload.get("name", "image_zoom_in_tool")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments_json = arguments
    else:
        arguments_json = json.dumps(arguments, ensure_ascii=False)
    return [{"type": "function", "function": {"name": name, "arguments": arguments_json}}]


def scale_bbox_to_qwen_range(bbox: list[int] | tuple[int, int, int, int], size: list[int] | tuple[int, int]) -> list[int]:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size for bbox scaling: {size}")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    scaled = [
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
        round(x2 * 1000 / width),
        round(y2 * 1000 / height),
    ]
    scaled = [max(0, min(1000, value)) for value in scaled]
    return scaled


def bbox_within_tolerance(
    actual_bbox: list[int] | tuple[int, int, int, int],
    expected_bbox: list[int] | tuple[int, int, int, int],
    *,
    tolerance: int = 1,
) -> bool:
    if len(actual_bbox) != 4 or len(expected_bbox) != 4:
        return False
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(actual_bbox, expected_bbox, strict=True))


def append_text(text_parts: list[str], new_text: str) -> None:
    if new_text:
        text_parts.append(new_text)


def convert_parts_to_content(
    parts: list[dict[str, Any]],
    image_map: dict[int, Image.Image],
    *,
    strip_last_round_hint: bool,
    qwen_style_tool_success: bool,
    image_storage_mode: str,
    image_cache_dir: Path | None,
) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    images: list[dict[str, bytes]] = []
    for part in parts:
        kind = part.get("kind")
        if kind in ("text", "separator"):
            if qwen_style_tool_success:
                continue
            if strip_last_round_hint and kind == "text" and part.get("text", "").strip() == prompts.LAST_ROUND_HINT.strip():
                continue
            append_text(text_parts, part.get("text", ""))
            continue
        if kind == "image_ref":
            label = part.get("label")
            if label:
                append_text(text_parts, f"{label}:")
            append_text(text_parts, "<image>")
            presented_img_idx = part.get("presented_img_idx")
            image = image_map.get(presented_img_idx)
            if image is None:
                raise ValueError(f"Missing presented image for presented_img_idx={presented_img_idx}")
            images.append(materialize_image(image, image_storage_mode, image_cache_dir))
            continue
        raise ValueError(f"Unsupported export part kind: {kind}")
    return "".join(text_parts), images


def convert_user_like_message(
    message: dict[str, Any],
    image_map: dict[int, Image.Image],
    stitch_runtime_hints: bool,
    image_storage_mode: str,
    image_cache_dir: Path | None,
) -> dict[str, Any] | None:
    message_type = message.get("type")
    content = message.get("content", {})
    parts = message.get("parts")
    secondary_types = message.get("secondary_types", [])

    if message_type in ("format_repair_hint", "last_round_hint") and stitch_runtime_hints:
        return None

    role = "user"
    if message_type in ("tool_result", "tool_result_fail_hint"):
        role = "tool"

    if isinstance(parts, list):
        qwen_style_tool_success = message_type == "tool_result"
        text, images = convert_parts_to_content(
            parts,
            image_map,
            strip_last_round_hint=bool(
                stitch_runtime_hints
                and isinstance(secondary_types, list)
                and "last_round_hint" in secondary_types
            ),
            qwen_style_tool_success=qwen_style_tool_success,
            image_storage_mode=image_storage_mode,
            image_cache_dir=image_cache_dir,
        )
        out: dict[str, Any] = {"role": role, "content": text}
        return out, images

    if message_type == "query":
        return {"role": "user", "content": normalize_text(content.get("question", ""))}, []
    if message_type == "tool_result":
        presented_img_indices = content.get("presented_img_indices")
        if isinstance(presented_img_indices, list) and presented_img_indices:
            images: list[dict[str, bytes]] = []
            text = ""
            for presented_img_idx in presented_img_indices:
                if not isinstance(presented_img_idx, int):
                    raise ValueError(f"Invalid presented_img_idx in tool_result: {presented_img_indices}")
                image = image_map.get(presented_img_idx)
                if image is None:
                    raise ValueError(f"Missing presented image for presented_img_idx={presented_img_idx}")
                text += f"Image {presented_img_idx}:<image>"
                images.append(materialize_image(image, image_storage_mode, image_cache_dir))
            return {"role": "tool", "content": text}, images
        return {"role": "tool", "content": ""}, []
    if message_type == "tool_result_fail_hint":
        error_message = normalize_text(content.get("error_message", ""))
        hint = normalize_text(content.get("hint", ""))
        text = f"{error_message}\n\n{hint}".strip()
        return {"role": "tool", "content": text}, []
    if message_type in ("format_repair_hint", "last_round_hint"):
        return {"role": "user", "content": normalize_text(content.get("hint", ""))}, []
    if message_type == "others":
        return {"role": "user", "content": normalize_text(content.get("text", ""))}, []

    raise ValueError(f"Unsupported user-side export type: {message_type}")


def find_presented_image(record: dict[str, Any], presented_img_idx: int) -> dict[str, Any] | None:
    for presented in record.get("image_references", {}).get("presented_images", []):
        if presented.get("presented_img_idx") == presented_img_idx:
            return presented
    return None


def convert_exact_tool_call(
    record: dict[str, Any],
    assistant_message: dict[str, Any],
    tool_result_message: dict[str, Any],
    assistant_format_mode: str,
) -> dict[str, Any]:
    content = assistant_message.get("content", {})
    payload = content.get("tool_call")
    if not isinstance(payload, dict):
        raise DropConversationError("assistant tool_call payload is not parsed JSON")

    qwen_arguments_from_export: dict[str, Any] | None = None
    region_description = payload.get("region_description")
    img_idx = payload.get("img_idx")
    if isinstance(region_description, str) and isinstance(img_idx, int):
        pass
    else:
        tool_name = payload.get("name")
        arguments = payload.get("arguments")
        if tool_name != "image_zoom_in_tool":
            raise DropConversationError("assistant tool_call is not image_zoom_in_tool")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise DropConversationError(f"assistant tool_call arguments are not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise DropConversationError("assistant tool_call arguments are not a dict")

        region_description = arguments.get("label")
        img_idx = arguments.get("img_idx")
        bbox_2d = arguments.get("bbox_2d")
        if (
            not isinstance(region_description, str)
            or not isinstance(img_idx, int)
            or not isinstance(bbox_2d, list)
            or len(bbox_2d) != 4
            or not all(isinstance(v, (int, float)) for v in bbox_2d)
        ):
            raise DropConversationError("assistant tool_call does not have exact Qwen-style arguments")
        qwen_arguments_from_export = {
            "img_idx": img_idx,
            "label": region_description,
            "bbox_2d": [int(round(v)) for v in bbox_2d],
        }

    result_content = tool_result_message.get("content", {})
    presented_img_indices = result_content.get("presented_img_indices")
    if not isinstance(presented_img_indices, list) or len(presented_img_indices) != 1:
        raise DropConversationError("tool_result does not point to exactly one presented image")
    new_presented_img_idx = presented_img_indices[0]
    if not isinstance(new_presented_img_idx, int):
        raise DropConversationError("tool_result presented image index is invalid")

    presented_ref = find_presented_image(record, new_presented_img_idx)
    if presented_ref is None:
        raise DropConversationError(f"missing presented image metadata for idx={new_presented_img_idx}")
    if presented_ref.get("kind") != "region_crop":
        raise DropConversationError(
            f"presented image idx={new_presented_img_idx} is not a region_crop ({presented_ref.get('kind')})"
        )
    if presented_ref.get("parent_presented_img_idx") != img_idx:
        raise DropConversationError(
            f"tool_result parent idx mismatch: expected {img_idx}, got {presented_ref.get('parent_presented_img_idx')}"
        )

    bbox_on_presented = presented_ref.get("bbox_on_presented")
    if not isinstance(bbox_on_presented, list) or len(bbox_on_presented) != 4:
        raise DropConversationError(f"missing bbox_on_presented for presented image idx={new_presented_img_idx}")

    parent_ref = find_presented_image(record, img_idx)
    if parent_ref is None:
        raise DropConversationError(f"missing parent presented image metadata for idx={img_idx}")
    parent_display_size = parent_ref.get("display_size")
    if not isinstance(parent_display_size, list) or len(parent_display_size) != 2:
        raise DropConversationError(f"missing parent display_size for presented image idx={img_idx}")

    qwen_arguments = {
        "img_idx": img_idx,
        "label": region_description,
        "bbox_2d": scale_bbox_to_qwen_range(bbox_on_presented, parent_display_size),
    }
    if (
        qwen_arguments_from_export is not None
        and (
            qwen_arguments_from_export.get("img_idx") != qwen_arguments["img_idx"]
            or qwen_arguments_from_export.get("label") != qwen_arguments["label"]
            or not bbox_within_tolerance(
                qwen_arguments_from_export.get("bbox_2d", []),
                qwen_arguments["bbox_2d"],
                tolerance=1,
            )
        )
    ):
        raise DropConversationError(
            "assistant tool_call Qwen-style arguments do not exactly match exported region_crop metadata"
        )
    return {
        "role": "assistant",
        "content": build_assistant_content(content, "tool_call", assistant_format_mode),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "arguments": json.dumps(qwen_arguments, ensure_ascii=False),
                },
            }
        ],
    }


def convert_record(
    record: dict[str, Any],
    stitch_runtime_hints: bool,
    system_prompt_mode: str,
    assistant_format_mode: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
) -> dict[str, Any]:
    presented_images = restore_presented_images(record)
    image_map = {item.get("presented_img_idx"): item.get("image") for item in presented_images}

    messages: list[dict[str, Any]] = []
    images: list[dict[str, bytes]] = []
    message_loss_mask: list[bool] = []

    conversation = record.get("conversation", [])
    index = 0
    while index < len(conversation):
        message = conversation[index]
        role = message.get("role")
        message_type = message.get("type")
        content = message.get("content", {})

        if role == "system":
            messages.append(
                {
                    "role": "system",
                    "content": resolve_system_prompt(content.get("text", ""), system_prompt_mode),
                }
            )
            message_loss_mask.append(False)
            index += 1
            continue

        if role == "assistant":
            if message_type == "tool_call":
                if index + 1 >= len(conversation):
                    raise DropConversationError("assistant tool_call is missing a following tool result message")
                next_message = conversation[index + 1]
                next_type = next_message.get("type")
                if next_message.get("role") != "user":
                    raise DropConversationError("assistant tool_call is not followed by a user/tool message")
                if next_type == "tool_result_fail_hint":
                    raise DropConversationError("conversation contains a non-exactly-convertible tool failure")
                if next_type != "tool_result":
                    raise DropConversationError(f"assistant tool_call is followed by unexpected user message type={next_type}")
                messages.append(convert_exact_tool_call(record, message, next_message, assistant_format_mode))
                message_loss_mask.append(True)
                index += 1
                continue
            if message_type == "answer":
                messages.append(
                    {
                        "role": "assistant",
                        "content": build_assistant_content(content, message_type, assistant_format_mode),
                    }
                )
                message_loss_mask.append(True)
                index += 1
                continue
            if (
                message_type == "others"
                and stitch_runtime_hints
                and index + 1 < len(conversation)
                and conversation[index + 1].get("role") == "user"
                and conversation[index + 1].get("type") == "format_repair_hint"
            ):
                index += 2
                continue
            else:
                raise DropConversationError("conversation contains an assistant 'others' turn outside a stitched repair episode")

        if role == "user":
            if message_type == "format_repair_hint":
                if stitch_runtime_hints:
                    index += 1
                    continue
            if message_type == "last_round_hint":
                if stitch_runtime_hints:
                    index += 1
                    continue
            if message_type == "tool_result_fail_hint":
                raise DropConversationError("conversation contains a non-exactly-convertible tool failure")
            converted = convert_user_like_message(
                message,
                image_map,
                stitch_runtime_hints,
                image_storage_mode=image_storage_mode,
                image_cache_dir=image_cache_dir,
            )
            index += 1
            if converted is None:
                continue
            converted_message, new_images = converted
            messages.append(converted_message)
            message_loss_mask.append(False)
            images.extend(new_images)
            continue

        messages.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
        message_loss_mask.append(False)
        index += 1

    return {
        "messages": messages,
        "message_loss_mask": message_loss_mask,
        "images": images,
        "tools": [IMAGE_ZOOM_IN_TOOL_SCHEMA],
    }


def convert_one_path(
    path: Path,
    *,
    stitch_runtime_hints: bool,
    system_prompt_mode: str,
    assistant_format_mode: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    only_correct_answers: bool,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    record = load_exported_conversation(str(path))
    extra_info = record.get("extra_info")
    question_id = None
    if isinstance(extra_info, dict) and extra_info.get("question_id") is not None:
        question_id = str(extra_info["question_id"])
    if only_correct_answers:
        reward = record.get("reward")
        accuracy_reward = None
        if isinstance(reward, dict):
            score = reward.get("score")
            if isinstance(score, dict) and score.get("accuracy_reward") is not None:
                accuracy_reward = float(score["accuracy_reward"])
            elif reward.get("accuracy_reward") is not None:
                accuracy_reward = float(reward["accuracy_reward"])
        if accuracy_reward != 1.0:
            return path.name, None, f"filtered out by only-correct-answers (accuracy_reward={accuracy_reward})", question_id
    try:
        converted = convert_record(
            record,
            stitch_runtime_hints=stitch_runtime_hints,
            system_prompt_mode=system_prompt_mode,
            assistant_format_mode=assistant_format_mode,
            image_storage_mode=image_storage_mode,
            image_cache_dir=image_cache_dir,
        )
    except DropConversationError as exc:
        return path.name, None, str(exc), question_id
    except Exception as exc:
        raise RuntimeError(f"Failed to convert {path}") from exc
    return path.name, converted, None, question_id


def _convert_one_path_star(args: tuple[Path, bool, str, str, str, Path | None, bool]) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    path, stitch_runtime_hints, system_prompt_mode, assistant_format_mode, image_storage_mode, image_cache_dir, only_correct_answers = args
    return convert_one_path(
        path,
        stitch_runtime_hints=stitch_runtime_hints,
        system_prompt_mode=system_prompt_mode,
        assistant_format_mode=assistant_format_mode,
        image_storage_mode=image_storage_mode,
        image_cache_dir=image_cache_dir,
        only_correct_answers=only_correct_answers,
    )


def load_records(
    input_dir: Path,
    stitch_runtime_hints: bool,
    system_prompt_mode: str,
    assistant_format_mode: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    num_workers: int,
    only_correct_answers: bool,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], int, Counter[str], list[str]]:
    records: list[dict[str, Any]] = []
    paths = sorted(input_dir.glob("*.json"))
    warning_counts: Counter[str] = Counter()
    wrong_question_ids: list[str] = []
    if num_workers <= 1:
        for path in paths:
            _, converted, warning, question_id = convert_one_path(
                path,
                stitch_runtime_hints=stitch_runtime_hints,
                system_prompt_mode=system_prompt_mode,
                assistant_format_mode=assistant_format_mode,
                image_storage_mode=image_storage_mode,
                image_cache_dir=image_cache_dir,
                only_correct_answers=only_correct_answers,
            )
            if warning is not None:
                warning_counts[warning] += 1
                if only_correct_answers and question_id is not None and warning.startswith("filtered out by only-correct-answers"):
                    wrong_question_ids.append(question_id)
                continue
            assert converted is not None
            records.append(converted)
    else:
        tasks = [
            (
                path,
                stitch_runtime_hints,
                system_prompt_mode,
                assistant_format_mode,
                image_storage_mode,
                image_cache_dir,
                only_correct_answers,
            )
            for path in paths
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for _, converted, warning, question_id in executor.map(_convert_one_path_star, tasks, chunksize=8):
                if warning is not None:
                    warning_counts[warning] += 1
                    if only_correct_answers and question_id is not None and warning.startswith("filtered out by only-correct-answers"):
                        wrong_question_ids.append(question_id)
                    continue
                assert converted is not None
                records.append(converted)
    if not records and not allow_empty:
        raise ValueError(f"No convertible records produced from JSON files in {input_dir}")
    return records, len(paths), warning_counts, wrong_question_ids


def split_dataframe(df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= 1 or val_ratio <= 0:
        return df, df.iloc[0:0]
    val_size = max(1, int(len(df) * val_ratio))
    if val_size >= len(df):
        val_size = len(df) - 1
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = output_dir / "images" if args.image_storage_mode == "path" else None

    if args.wrong_question_ids_only and not args.only_correct_answers:
        raise ValueError("--wrong-question-ids-only requires --only-correct-answers")
    if args.output_parquet_name is not None and args.val_ratio != 0:
        raise ValueError("--val-ratio must be 0 when --output-parquet-name is specified")
    if args.wrong_question_ids_only and args.output_parquet_name is not None:
        raise ValueError("--output-parquet-name cannot be used with --wrong-question-ids-only")

    rows, total_jsons, warning_counts, wrong_question_ids = load_records(
        input_dir,
        stitch_runtime_hints=args.stitch_runtime_hints,
        system_prompt_mode=args.system_prompt_mode,
        assistant_format_mode=args.assistant_format_mode,
        image_storage_mode=args.image_storage_mode,
        image_cache_dir=image_cache_dir,
        num_workers=args.num_workers,
        only_correct_answers=args.only_correct_answers,
        allow_empty=args.wrong_question_ids_only,
    )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    dropped = total_jsons - len(rows)
    print(f"Scanned {total_jsons} JSON files")
    print(f"Kept {len(rows)} convertible conversations")
    print(f"Dropped or filtered {dropped} conversations")
    if warning_counts:
        print("Skip reason summary:")
        for reason, count in warning_counts.most_common():
            print(f"  {count}\t{reason}")
    if args.only_correct_answers:
        wrong_question_ids_path = output_dir / "wrong_question_ids.txt"
        unique_wrong_question_ids = sorted(set(wrong_question_ids))
        wrong_question_ids_path.write_text(
            "".join(f"{question_id}\n" for question_id in unique_wrong_question_ids),
            encoding="utf-8",
        )
        print(f"Wrote {len(unique_wrong_question_ids)} wrong-answer question_ids to {wrong_question_ids_path}")
    if args.wrong_question_ids_only:
        return

    if args.output_parquet_name is not None:
        output_path = output_dir / args.output_parquet_name
        df.to_parquet(output_path)
        print(f"Wrote {len(df)} rows to {output_path}")
        return

    train_df, val_df = split_dataframe(df, args.val_ratio)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train_df.to_parquet(train_path)
    if len(val_df) > 0:
        val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} rows to {train_path}")
    if len(val_df) > 0:
        print(f"Wrote {len(val_df)} rows to {val_path}")
    else:
        print("Validation split is empty; set trainer.test_freq=-1 when training.")


if __name__ == "__main__":
    main()
