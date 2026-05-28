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
import asyncio
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import verl.utils.vreasoner_v2_prompt as prompts
from verl.utils.vreasoner_v2_conversation_export import load_exported_conversation, restore_presented_images
from scripts.mark_bad_exported_conversations import (
    QUALITY_FIELD,
    Thresholds as DegenerateThresholds,
    scan_record as scan_degenerate_record,
)


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


ANSWER_VERIFICATION_HINT_TYPE = "answer_verification_hint"
ANSWER_REVISION_TYPE = "answer_revision"


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
        "--system-prompt-insertion-mode",
        choices=["no_prepend_if_missing", "prepend_if_missing"],
        default="no_prepend_if_missing",
        help=(
            "Whether to leave system-message presence unchanged, or prepend the selected system prompt "
            "when an exported conversation has no system message."
        ),
    )
    parser.add_argument(
        "--assistant-format-mode",
        choices=["tagged", "plain"],
        default="plain",
        help="Whether assistant reasoning/answers keep <think>/<answer> tags or are written as plain text.",
    )
    parser.add_argument(
        "--tool-argument-order",
        choices=["legacy", "base_model"],
        default="legacy",
        help=(
            "Order for serialized image_zoom_in_tool arguments in SFT tool calls. "
            "legacy keeps the historical img_idx,label,bbox_2d order; base_model uses "
            "label,bbox_2d,img_idx to match observed Qwen3-VL base-model generations."
        ),
    )
    parser.add_argument(
        "--final-answer-rewrite-mode",
        choices=["none", "api"],
        default="none",
        help=(
            "Optionally rewrite exported final assistant answers into a sibling raw rewrite directory before "
            "SFT conversion. Rewritten rows are materialized as exported conversation JSONs and conversion reads "
            "from that directory. API failures never fall back into the parquet."
        ),
    )
    parser.add_argument(
        "--final-answer-rewrite-output-dir",
        default=None,
        help=(
            "Directory for rewritten exported conversation JSONs. Defaults to INPUT_DIR's sibling "
            "raw_gpt5_nano_rewrite when INPUT_DIR is named raw, otherwise OUTPUT_DIR/rewritten_exported_conversations."
        ),
    )
    parser.add_argument("--final-answer-rewrite-model", default="gpt-5-nano")
    parser.add_argument("--final-answer-rewrite-concurrency", type=int, default=8)
    parser.add_argument("--final-answer-rewrite-timeout", type=float, default=120.0)
    parser.add_argument("--final-answer-rewrite-max-retries", type=int, default=4)
    parser.add_argument("--final-answer-rewrite-max-completion-tokens", type=int, default=4096)
    parser.add_argument("--final-answer-rewrite-max-failure-ratio", type=float, default=0.005)
    parser.add_argument("--final-answer-rewrite-max-failures", type=int, default=None)
    parser.add_argument(
        "--final-answer-rewrite-retry-rounds",
        type=int,
        default=2,
        help=(
            "Extra full rewrite-stage retry rounds when API/validation failures exceed the configured threshold. "
            "Completed rewritten JSONs are skipped on each retry."
        ),
    )
    parser.add_argument(
        "--final-answer-rewrite-retry-sleep",
        type=float,
        default=30.0,
        help="Seconds to wait between final-answer rewrite retry rounds.",
    )
    parser.add_argument("--final-answer-rewrite-progress-every", type=int, default=50)
    parser.add_argument(
        "--final-answer-rewrite-openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://az.gptplus5.com/v1"),
    )
    parser.add_argument(
        "--api-logger-save-dir",
        default=os.environ.get("API_LOGGER_SAVE_DIR", str(Path.home() / ".dumps/api_requests")),
    )
    parser.add_argument(
        "--api-logger-project-name",
        default=os.environ.get("API_LOGGER_PROJECT_NAME", "final_answer_rewrite_gpt5_nano"),
    )
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
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
        "--invalid-image-aspect-policy",
        choices=["error", "pad", "drop"],
        default="drop",
        help=(
            "How to handle images whose aspect ratio exceeds Qwen-VL preprocessing limits. "
            "error keeps fail-fast behavior, pad expands the short side with white padding, "
            "and drop filters the whole conversation."
        ),
    )
    parser.add_argument(
        "--max-image-aspect-ratio",
        type=float,
        default=200.0,
        help="Maximum allowed max(width,height)/min(width,height) before applying invalid-image-aspect-policy.",
    )
    parser.add_argument(
        "--image-aspect-pad-target-ratio",
        type=float,
        default=198.0,
        help="Target aspect ratio used when padding invalid images; keep this below Qwen-VL's hard limit.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for per-conversation conversion.",
    )
    parser.add_argument(
        "--conversion-progress-every",
        type=int,
        default=100,
        help="Fallback conversion progress print interval when tqdm is unavailable or disabled.",
    )
    parser.add_argument(
        "--disable-conversion-progress-bar",
        action="store_true",
        help="Disable tqdm progress bar and use periodic text progress instead.",
    )
    parser.add_argument(
        "--only-correct-answers",
        action="store_true",
        help="Keep only exported conversations whose accuracy_reward is exactly 1.0.",
    )
    parser.add_argument(
        "--drop-degenerate-conversations",
        action="store_true",
        help=(
            "Drop exported conversations with degenerate assistant text. "
            "This also honors existing top-level quality_flags.bad_example=true markers."
        ),
    )
    parser.add_argument("--degenerate-max-assistant-chars", type=int, default=50_000)
    parser.add_argument("--degenerate-max-assistant-words", type=int, default=8_000)
    parser.add_argument("--degenerate-min-unique-word-ratio", type=float, default=0.20)
    parser.add_argument("--degenerate-min-words-for-unique-ratio", type=int, default=1_000)
    parser.add_argument("--degenerate-max-same-word-run", type=int, default=10)
    parser.add_argument("--degenerate-ngram-size", type=int, default=8)
    parser.add_argument("--degenerate-max-ngram-repeats", type=int, default=50)
    parser.add_argument("--degenerate-min-words-for-ngram", type=int, default=1_000)
    parser.add_argument("--degenerate-preview-chars", type=int, default=240)
    parser.add_argument(
        "--wrong-question-ids-only",
        action="store_true",
        help=(
            "Only write wrong_question_ids.txt and skip parquet generation. "
            "Requires --only-correct-answers."
        ),
    )
    parser.add_argument(
        "--rewrite-file-uri-prefix",
        action="append",
        default=[],
        metavar="OLD_ROOT=NEW_ROOT",
        help=(
            "Rewrite file:// input-image references inside exported conversations before reconstruction. "
            "Repeatable. OLD_ROOT and NEW_ROOT should be filesystem path prefixes."
        ),
    )
    return parser.parse_args()


def parse_file_uri_prefix_mappings(raw_mappings: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for raw in raw_mappings:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --rewrite-file-uri-prefix value {raw!r}; expected OLD_ROOT=NEW_ROOT"
            )
        old_root, new_root = raw.split("=", 1)
        old_root = old_root.strip()
        new_root = new_root.strip()
        if not old_root or not new_root:
            raise ValueError(
                f"Invalid --rewrite-file-uri-prefix value {raw!r}; both OLD_ROOT and NEW_ROOT are required"
            )
        if old_root.startswith("file://") or new_root.startswith("file://"):
            raise ValueError(
                "--rewrite-file-uri-prefix expects filesystem paths, not file:// URIs"
            )
        mappings.append((str(Path(old_root).expanduser()), str(Path(new_root).expanduser())))
    mappings.sort(key=lambda item: len(item[0]), reverse=True)
    return mappings


def rewrite_file_uri_value(value: str, prefix_mappings: list[tuple[str, str]]) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return value
    source_path = parsed.path
    for old_root, new_root in prefix_mappings:
        old_prefix = old_root.rstrip("/")
        if source_path == old_prefix or source_path.startswith(old_prefix + "/"):
            suffix = source_path[len(old_prefix) :].lstrip("/")
            rewritten_path = (Path(new_root) / suffix if suffix else Path(new_root)).resolve()
            if not rewritten_path.exists():
                raise FileNotFoundError(
                    "Rewritten file:// target does not exist: "
                    f"{rewritten_path} (from {value!r} via {old_root!r} -> {new_root!r})"
                )
            return rewritten_path.as_uri()
    return value


def rewrite_filesystem_path_value(value: str, prefix_mappings: list[tuple[str, str]]) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    source_path = str(Path(value).expanduser())
    for old_root, new_root in prefix_mappings:
        old_prefix = old_root.rstrip("/")
        if source_path == old_prefix or source_path.startswith(old_prefix + "/"):
            suffix = source_path[len(old_prefix) :].lstrip("/")
            rewritten_path = (Path(new_root) / suffix if suffix else Path(new_root)).resolve()
            if not rewritten_path.exists():
                raise FileNotFoundError(
                    "Rewritten filesystem target does not exist: "
                    f"{rewritten_path} (from {value!r} via {old_root!r} -> {new_root!r})"
                )
            return str(rewritten_path)
    return value


def rewrite_path_or_file_uri_value(
    value: str,
    prefix_mappings: list[tuple[str, str]],
) -> str:
    if value.startswith("file://"):
        return rewrite_file_uri_value(value, prefix_mappings)
    return rewrite_filesystem_path_value(value, prefix_mappings)


def rewrite_input_image_ref(
    ref: dict[str, Any],
    prefix_mappings: list[tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    rewritten_ref = dict(ref)
    changed = False
    for key in ("value", "uri", "path"):
        value = rewritten_ref.get(key)
        if not isinstance(value, str):
            continue
        new_value = rewrite_path_or_file_uri_value(value, prefix_mappings)
        if new_value != value:
            rewritten_ref[key] = new_value
            changed = True
    return rewritten_ref, changed


def rewrite_input_image_ref_list(
    refs: Any,
    prefix_mappings: list[tuple[str, str]],
) -> tuple[Any, bool]:
    if not isinstance(refs, list):
        return refs, False
    rewritten_refs: list[Any] = []
    changed = False
    for ref in refs:
        if not isinstance(ref, dict):
            rewritten_refs.append(ref)
            continue
        rewritten_ref, ref_changed = rewrite_input_image_ref(ref, prefix_mappings)
        rewritten_refs.append(rewritten_ref)
        changed = changed or ref_changed
    return rewritten_refs, changed


def rewrite_record_file_uri_refs(
    record: dict[str, Any],
    prefix_mappings: list[tuple[str, str]],
) -> dict[str, Any]:
    if not prefix_mappings:
        return record

    rewritten_record = dict(record)
    changed = False

    image_references = record.get("image_references")
    if isinstance(image_references, dict):
        rewritten_refs, refs_changed = rewrite_input_image_ref_list(
            image_references.get("input_images"),
            prefix_mappings,
        )
        if refs_changed:
            rewritten_record["image_references"] = {
                **image_references,
                "input_images": rewritten_refs,
            }
            changed = True

    extra_info = rewritten_record.get("extra_info")
    if isinstance(extra_info, dict):
        rewritten_original_refs, original_refs_changed = rewrite_input_image_ref_list(
            extra_info.get("original_image_refs"),
            prefix_mappings,
        )
        if original_refs_changed:
            rewritten_record["extra_info"] = {
                **extra_info,
                "original_image_refs": rewritten_original_refs,
            }
            changed = True

    return rewritten_record if changed else record


def build_degenerate_thresholds(args: argparse.Namespace) -> DegenerateThresholds:
    return DegenerateThresholds(
        max_assistant_chars=args.degenerate_max_assistant_chars,
        max_assistant_words=args.degenerate_max_assistant_words,
        min_unique_word_ratio=args.degenerate_min_unique_word_ratio,
        min_words_for_unique_ratio=args.degenerate_min_words_for_unique_ratio,
        max_same_word_run=args.degenerate_max_same_word_run,
        ngram_size=args.degenerate_ngram_size,
        max_ngram_repeats=args.degenerate_max_ngram_repeats,
        min_words_for_ngram=args.degenerate_min_words_for_ngram,
    )


def record_has_bad_quality_marker(record: dict[str, Any]) -> tuple[bool, str | None]:
    marker = record.get(QUALITY_FIELD)
    if isinstance(marker, dict) and marker.get("bad_example") is True:
        reasons = marker.get("reasons")
        if isinstance(reasons, list) and reasons:
            return True, "; ".join(str(reason) for reason in reasons[:3])
        return True, "quality_flags.bad_example=true"
    return False, None


def degenerate_drop_reason(
    record: dict[str, Any],
    thresholds: DegenerateThresholds,
    preview_chars: int,
) -> str | None:
    marked_bad, marker_reason = record_has_bad_quality_marker(record)
    if marked_bad:
        return f"marked bad by quality_flags ({marker_reason})"

    bad_example, top_level_marker, _ = scan_degenerate_record(record, thresholds, preview_chars)
    if not bad_example:
        return None
    reasons = top_level_marker.get("reasons", [])
    if isinstance(reasons, list) and reasons:
        reason_text = "; ".join(str(reason) for reason in reasons[:3])
        if len(reasons) > 3:
            reason_text += f"; ... ({len(reasons)} reasons total)"
        return reason_text
    return "degenerate assistant text"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def image_aspect_ratio(image: Image.Image) -> float:
    width, height = image.size
    short_side = min(width, height)
    if short_side <= 0:
        return math.inf
    return max(width, height) / short_side


def pad_image_to_aspect_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    if target_ratio <= 1:
        raise ValueError(f"image_aspect_pad_target_ratio must be > 1, got {target_ratio}")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot pad image with invalid size {image.size}")
    if image_aspect_ratio(image) <= target_ratio:
        return image

    if width > height:
        target_width = width
        target_height = max(height, math.ceil(width / target_ratio))
    else:
        target_width = max(width, math.ceil(height / target_ratio))
        target_height = height

    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    canvas.paste(image.convert("RGB"), ((target_width - width) // 2, (target_height - height) // 2))
    return canvas


def repair_or_validate_image_aspect(
    image: Image.Image,
    *,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
) -> Image.Image:
    ratio = image_aspect_ratio(image)
    if ratio <= max_image_aspect_ratio:
        return image

    message = (
        "image aspect ratio exceeds limit: "
        f"size={image.size}, ratio={ratio:.3f}, limit={max_image_aspect_ratio:g}"
    )
    if invalid_image_aspect_policy == "drop":
        raise DropConversationError(f"filtered out by invalid image aspect ratio ({message})")
    if invalid_image_aspect_policy == "pad":
        return pad_image_to_aspect_ratio(image, image_aspect_pad_target_ratio)
    if invalid_image_aspect_policy == "error":
        raise ValueError(message)
    raise ValueError(f"Unsupported invalid_image_aspect_policy={invalid_image_aspect_policy!r}")


def materialize_image(
    image: Image.Image,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    *,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
) -> dict[str, Any]:
    image = repair_or_validate_image_aspect(
        image,
        invalid_image_aspect_policy=invalid_image_aspect_policy,
        max_image_aspect_ratio=max_image_aspect_ratio,
        image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
    )
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


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (list, dict)):
        return json.dumps(text, ensure_ascii=False)
    return str(text)


def _text_after_last_label_parts(parts: list[dict[str, Any]]) -> str:
    text_fragments: list[str] = []
    for part in parts:
        if part.get("kind") == "text":
            text_fragments.append(str(part.get("text", "")))
    return "".join(text_fragments).strip()


def recover_legacy_user_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "user" or message.get("type") != "others":
        return message
    content = message.get("content", {})
    if not isinstance(content, dict):
        return message
    raw_parts = content.get("text")
    if not isinstance(raw_parts, list):
        return message

    parts: list[dict[str, Any]] = []
    pending_label: tuple[int, str] | None = None
    for item in raw_parts:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text", ""))
            stripped = text.strip()
            if stripped == prompts.IMAGE_SEPARATOR.strip():
                parts.append({"kind": "separator", "text": prompts.IMAGE_SEPARATOR})
                continue
            if stripped.startswith("Image ") and stripped.endswith(":"):
                idx_str = stripped[len("Image ") : -1]
                if idx_str.isdigit():
                    pending_label = (int(idx_str), stripped[:-1])
                    continue
            parts.append({"kind": "text", "text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url", {})
            detail = image_url.get("detail") if isinstance(image_url, dict) else None
            if pending_label is None:
                parts.append({"kind": "image_ref", "presented_img_idx": None, "label": None, "detail": detail})
            else:
                img_idx, label = pending_label
                parts.append(
                    {
                        "kind": "image_ref",
                        "presented_img_idx": img_idx,
                        "label": label,
                        "detail": detail,
                    }
                )
                pending_label = None

    image_indices = [part.get("presented_img_idx") for part in parts if part.get("kind") == "image_ref"]
    image_indices = [idx for idx in image_indices if isinstance(idx, int)]
    if not image_indices:
        return message

    main_text_parts = [
        str(part.get("text", ""))
        for part in parts
        if part.get("kind") == "text" and str(part.get("text", "")).strip() != prompts.LAST_ROUND_HINT.strip()
    ]
    main_text = "".join(main_text_parts).strip()
    secondary_types: list[str] = []
    if any(
        part.get("kind") == "text" and str(part.get("text", "")).strip() == prompts.LAST_ROUND_HINT.strip()
        for part in parts
    ):
        secondary_types.append("last_round_hint")

    recovered = {
        "message_idx": message.get("message_idx"),
        "role": "user",
        "type": "tool_result",
        "content": {
            "hint": main_text,
            "presented_img_indices": image_indices,
        },
        "parts": parts,
    }
    if secondary_types:
        recovered["secondary_types"] = secondary_types
    return recovered


def is_answer_verification_hint(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and message.get("type") == ANSWER_VERIFICATION_HINT_TYPE


def get_revision_answer_content(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get("role") != "assistant":
        return None
    message_type = message.get("type")
    if message_type not in ("answer", ANSWER_REVISION_TYPE):
        return None
    content = message.get("content", {})
    answer = normalize_text(content.get("answer", ""))
    if not answer.strip():
        return None
    return {
        "think": normalize_text(content.get("think", "")),
        "answer": answer,
    }


def resolve_system_prompt(original_text: str, mode: str) -> str:
    if mode == "exported":
        return normalize_text(original_text)
    if mode == "vsearcher_qwen3_vl":
        return VSEARCHER_QWEN3_VL_SYSTEM_PROMPT
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def build_assistant_content(content: dict[str, Any], message_type: str, assistant_format_mode: str) -> str:
    think = normalize_text(content.get("think", ""))
    if message_type in ("answer", ANSWER_REVISION_TYPE):
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
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
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
            images.append(
                materialize_image(
                    image,
                    image_storage_mode,
                    image_cache_dir,
                    invalid_image_aspect_policy=invalid_image_aspect_policy,
                    max_image_aspect_ratio=max_image_aspect_ratio,
                    image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
                )
            )
            continue
        raise ValueError(f"Unsupported export part kind: {kind}")
    return "".join(text_parts), images


def convert_user_like_message(
    message: dict[str, Any],
    image_map: dict[int, Image.Image],
    stitch_runtime_hints: bool,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
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
            invalid_image_aspect_policy=invalid_image_aspect_policy,
            max_image_aspect_ratio=max_image_aspect_ratio,
            image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
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
                images.append(
                    materialize_image(
                        image,
                        image_storage_mode,
                        image_cache_dir,
                        invalid_image_aspect_policy=invalid_image_aspect_policy,
                        max_image_aspect_ratio=max_image_aspect_ratio,
                        image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
                    )
                )
            return {"role": "tool", "content": text}, images
        return {"role": "tool", "content": ""}, []
    if message_type == "tool_result_fail_hint":
        error_message = normalize_text(content.get("error_message", ""))
        hint = normalize_text(content.get("hint", ""))
        text = f"{error_message}\n\n{hint}".strip()
        return {"role": "tool", "content": text}, []
    if message_type in ("format_repair_hint", "last_round_hint", ANSWER_VERIFICATION_HINT_TYPE):
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
    tool_argument_order: str,
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

    qwen_bbox_2d = scale_bbox_to_qwen_range(bbox_on_presented, parent_display_size)
    if tool_argument_order == "legacy":
        qwen_arguments = {
            "img_idx": img_idx,
            "label": region_description,
            "bbox_2d": qwen_bbox_2d,
        }
    elif tool_argument_order == "base_model":
        qwen_arguments = {
            "label": region_description,
            "bbox_2d": qwen_bbox_2d,
            "img_idx": img_idx,
        }
    else:
        raise ValueError(f"Unsupported tool_argument_order={tool_argument_order!r}")
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
    system_prompt_insertion_mode: str,
    assistant_format_mode: str,
    tool_argument_order: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
) -> dict[str, Any]:
    presented_images = restore_presented_images(record)
    image_map = {item.get("presented_img_idx"): item.get("image") for item in presented_images}

    messages: list[dict[str, Any]] = []
    images: list[dict[str, bytes]] = []
    message_loss_mask: list[bool] = []

    conversation = record.get("conversation", [])
    if (
        system_prompt_insertion_mode == "prepend_if_missing"
        and not any(message.get("role") == "system" for message in conversation)
    ):
        messages.append(
            {
                "role": "system",
                "content": resolve_system_prompt("", system_prompt_mode),
            }
        )
        message_loss_mask.append(False)

    index = 0
    while index < len(conversation):
        message = recover_legacy_user_message(conversation[index])
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
            if (
                message_type == "answer"
                and index + 2 < len(conversation)
                and is_answer_verification_hint(conversation[index + 1])
            ):
                revised_content = get_revision_answer_content(conversation[index + 2])
                if revised_content is not None:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": build_assistant_content(revised_content, "answer", assistant_format_mode),
                        }
                    )
                    message_loss_mask.append(True)
                    index += 3
                    continue
            if message_type == "tool_call":
                if index + 1 >= len(conversation):
                    raise DropConversationError("assistant tool_call is missing a following tool result message")
                next_message = recover_legacy_user_message(conversation[index + 1])
                next_type = next_message.get("type")
                if next_message.get("role") != "user":
                    raise DropConversationError("assistant tool_call is not followed by a user/tool message")
                if next_type == "tool_result_fail_hint":
                    raise DropConversationError("conversation contains a non-exactly-convertible tool failure")
                if next_type != "tool_result":
                    raise DropConversationError(f"assistant tool_call is followed by unexpected user message type={next_type}")
                messages.append(
                    convert_exact_tool_call(
                        record,
                        message,
                        next_message,
                        assistant_format_mode,
                        tool_argument_order,
                    )
                )
                message_loss_mask.append(True)
                index += 1
                continue
            if message_type == "answer":
                answer_text = normalize_text(content.get("answer", ""))
                if not answer_text.strip():
                    raise DropConversationError("assistant answer is empty")
                messages.append(
                    {
                        "role": "assistant",
                        "content": build_assistant_content(content, message_type, assistant_format_mode),
                    }
                )
                message_loss_mask.append(True)
                index += 1
                continue
            if message_type == ANSWER_REVISION_TYPE:
                answer_text = normalize_text(content.get("answer", ""))
                if not answer_text.strip():
                    raise DropConversationError("assistant answer_revision is empty")
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
                and recover_legacy_user_message(conversation[index + 1]).get("role") == "user"
                and recover_legacy_user_message(conversation[index + 1]).get("type") == "format_repair_hint"
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
                invalid_image_aspect_policy=invalid_image_aspect_policy,
                max_image_aspect_ratio=max_image_aspect_ratio,
                image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
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
    system_prompt_insertion_mode: str,
    assistant_format_mode: str,
    tool_argument_order: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
    only_correct_answers: bool,
    rewrite_file_uri_prefixes: list[tuple[str, str]],
    drop_degenerate_conversations: bool,
    degenerate_thresholds: DegenerateThresholds,
    degenerate_preview_chars: int,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    record = load_exported_conversation(str(path))
    extra_info = record.get("extra_info")
    question_id = None
    if isinstance(extra_info, dict) and extra_info.get("question_id") is not None:
        question_id = str(extra_info["question_id"])
    try:
        record = rewrite_record_file_uri_refs(record, rewrite_file_uri_prefixes)
    except FileNotFoundError as exc:
        return path.name, None, f"filtered out by rewrite-file-uri-prefix ({exc})", question_id
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
    if drop_degenerate_conversations:
        reason = degenerate_drop_reason(record, degenerate_thresholds, degenerate_preview_chars)
        if reason is not None:
            return path.name, None, f"filtered out by drop-degenerate-conversations ({reason})", question_id
    try:
        converted = convert_record(
            record,
            stitch_runtime_hints=stitch_runtime_hints,
            system_prompt_mode=system_prompt_mode,
            system_prompt_insertion_mode=system_prompt_insertion_mode,
            assistant_format_mode=assistant_format_mode,
            tool_argument_order=tool_argument_order,
            image_storage_mode=image_storage_mode,
            image_cache_dir=image_cache_dir,
            invalid_image_aspect_policy=invalid_image_aspect_policy,
            max_image_aspect_ratio=max_image_aspect_ratio,
            image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
        )
    except DropConversationError as exc:
        return path.name, None, str(exc), question_id
    except Exception as exc:
        raise RuntimeError(f"Failed to convert {path}") from exc
    return path.name, converted, None, question_id


def _convert_one_path_star(
    args: tuple[
        Path,
        bool,
        str,
        str,
        str,
        str,
        str,
        Path | None,
        str,
        float,
        float,
        bool,
        list[tuple[str, str]],
        bool,
        DegenerateThresholds,
        int,
    ]
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    (
        path,
        stitch_runtime_hints,
        system_prompt_mode,
        system_prompt_insertion_mode,
        assistant_format_mode,
        tool_argument_order,
        image_storage_mode,
        image_cache_dir,
        invalid_image_aspect_policy,
        max_image_aspect_ratio,
        image_aspect_pad_target_ratio,
        only_correct_answers,
        rewrite_file_uri_prefixes,
        drop_degenerate_conversations,
        degenerate_thresholds,
        degenerate_preview_chars,
    ) = args
    return convert_one_path(
        path,
        stitch_runtime_hints=stitch_runtime_hints,
        system_prompt_mode=system_prompt_mode,
        system_prompt_insertion_mode=system_prompt_insertion_mode,
        assistant_format_mode=assistant_format_mode,
        tool_argument_order=tool_argument_order,
        image_storage_mode=image_storage_mode,
        image_cache_dir=image_cache_dir,
        invalid_image_aspect_policy=invalid_image_aspect_policy,
        max_image_aspect_ratio=max_image_aspect_ratio,
        image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
        only_correct_answers=only_correct_answers,
        rewrite_file_uri_prefixes=rewrite_file_uri_prefixes,
        drop_degenerate_conversations=drop_degenerate_conversations,
        degenerate_thresholds=degenerate_thresholds,
        degenerate_preview_chars=degenerate_preview_chars,
    )


def load_records(
    input_dir: Path,
    stitch_runtime_hints: bool,
    system_prompt_mode: str,
    system_prompt_insertion_mode: str,
    assistant_format_mode: str,
    tool_argument_order: str,
    image_storage_mode: str,
    image_cache_dir: Path | None,
    invalid_image_aspect_policy: str,
    max_image_aspect_ratio: float,
    image_aspect_pad_target_ratio: float,
    num_workers: int,
    only_correct_answers: bool,
    rewrite_file_uri_prefixes: list[tuple[str, str]],
    drop_degenerate_conversations: bool,
    degenerate_thresholds: DegenerateThresholds,
    degenerate_preview_chars: int,
    conversion_progress_every: int,
    disable_conversion_progress_bar: bool,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], int, Counter[str], list[str]]:
    records: list[dict[str, Any]] = []
    paths = sorted(input_dir.glob("*.json"))
    warning_counts: Counter[str] = Counter()
    wrong_question_ids: list[str] = []

    def iter_with_progress(iterator: Any, total: int) -> Any:
        if tqdm is not None and not disable_conversion_progress_bar:
            return tqdm(iterator, total=total, desc=f"Converting {input_dir.name}", unit="conv")

        def generator() -> Any:
            started_at = time.time()
            for idx, item in enumerate(iterator, start=1):
                if conversion_progress_every > 0 and (idx == 1 or idx % conversion_progress_every == 0 or idx == total):
                    elapsed = time.time() - started_at
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    print(f"Converted {idx}/{total} conversations ({rate:.2f}/s)", flush=True)
                yield item

        return generator()

    if num_workers <= 1:
        for path in iter_with_progress(paths, len(paths)):
            _, converted, warning, question_id = convert_one_path(
                path,
                stitch_runtime_hints=stitch_runtime_hints,
                system_prompt_mode=system_prompt_mode,
                system_prompt_insertion_mode=system_prompt_insertion_mode,
                assistant_format_mode=assistant_format_mode,
                tool_argument_order=tool_argument_order,
                image_storage_mode=image_storage_mode,
                image_cache_dir=image_cache_dir,
                invalid_image_aspect_policy=invalid_image_aspect_policy,
                max_image_aspect_ratio=max_image_aspect_ratio,
                image_aspect_pad_target_ratio=image_aspect_pad_target_ratio,
                only_correct_answers=only_correct_answers,
                rewrite_file_uri_prefixes=rewrite_file_uri_prefixes,
                drop_degenerate_conversations=drop_degenerate_conversations,
                degenerate_thresholds=degenerate_thresholds,
                degenerate_preview_chars=degenerate_preview_chars,
            )
            if warning is not None:
                warning_counts[warning] += 1
                if only_correct_answers and question_id is not None:
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
                system_prompt_insertion_mode,
                assistant_format_mode,
                tool_argument_order,
                image_storage_mode,
                image_cache_dir,
                invalid_image_aspect_policy,
                max_image_aspect_ratio,
                image_aspect_pad_target_ratio,
                only_correct_answers,
                rewrite_file_uri_prefixes,
                drop_degenerate_conversations,
                degenerate_thresholds,
                degenerate_preview_chars,
            )
            for path in paths
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            mapped = executor.map(_convert_one_path_star, tasks, chunksize=8)
            for _, converted, warning, question_id in iter_with_progress(mapped, len(tasks)):
                if warning is not None:
                    warning_counts[warning] += 1
                    if only_correct_answers and question_id is not None:
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


def default_final_answer_rewrite_output_dir(input_dir: Path, output_dir: Path) -> Path:
    if input_dir.name == "raw":
        return input_dir.parent / "raw_gpt5_nano_rewrite"
    return output_dir / "rewritten_exported_conversations"


def run_final_answer_rewrite_stage(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    degenerate_thresholds: DegenerateThresholds,
) -> Path:
    if args.final_answer_rewrite_mode == "none":
        return input_dir

    if args.final_answer_rewrite_mode != "api":
        raise ValueError(f"Unsupported final-answer rewrite mode: {args.final_answer_rewrite_mode}")
    if args.assistant_format_mode != "plain":
        raise ValueError("--final-answer-rewrite-mode=api requires --assistant-format-mode=plain")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set for --final-answer-rewrite-mode=api")

    from scripts.rewrite_exported_convos_final_answers_with_api import (
        load_rewrite_cache,
        process_paths,
    )

    os.environ.setdefault("OPENAI_BASE_URL", args.final_answer_rewrite_openai_base_url)
    os.environ.setdefault("OPENAI_CLIENT_TIMEOUT", str(args.final_answer_rewrite_timeout))
    os.environ.setdefault("ENSURE_API_LOGGER", "1")
    os.environ.setdefault("API_LOGGER_SAVE_DIR", str(Path(args.api_logger_save_dir).expanduser()))
    os.environ.setdefault("API_LOGGER_PROJECT_NAME", args.api_logger_project_name)
    os.environ.setdefault("INSIGHT_DOC_ROOT", args.insight_doc_root)

    rewrite_output_dir = (
        Path(args.final_answer_rewrite_output_dir).expanduser().resolve()
        if args.final_answer_rewrite_output_dir
        else default_final_answer_rewrite_output_dir(input_dir, output_dir)
    )
    rewrite_output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = rewrite_output_dir / "rewrite_cache.jsonl"
    status_path = rewrite_output_dir / "rewrite_status.jsonl"
    rewrite_cache = load_rewrite_cache(cache_path)
    paths = sorted(input_dir.glob("*.json"))

    rewrite_args = argparse.Namespace(
        model=args.final_answer_rewrite_model,
        timeout=args.final_answer_rewrite_timeout,
        max_retries=args.final_answer_rewrite_max_retries,
        max_completion_tokens=args.final_answer_rewrite_max_completion_tokens,
        concurrency=args.final_answer_rewrite_concurrency,
        max_api_failure_ratio=args.final_answer_rewrite_max_failure_ratio,
        max_api_failures=args.final_answer_rewrite_max_failures,
        progress_every=args.final_answer_rewrite_progress_every,
        only_correct_answers=args.only_correct_answers,
        drop_degenerate_conversations=args.drop_degenerate_conversations,
        degenerate_preview_chars=args.degenerate_preview_chars,
        dry_run=False,
        ensure_api_logger=True,
        insight_doc_root=args.insight_doc_root,
    )

    max_rounds = max(0, args.final_answer_rewrite_retry_rounds) + 1
    last_status_counts = None
    last_eligible = 0
    last_failures = 0
    last_failure_ratio = 0.0
    for round_index in range(max_rounds):
        if round_index > 0:
            print(
                "Retrying final-answer rewrite stage "
                f"round {round_index + 1}/{max_rounds} after {args.final_answer_rewrite_retry_sleep:g}s; "
                "completed rewritten JSONs will be skipped."
            )
            time.sleep(max(0.0, args.final_answer_rewrite_retry_sleep))
            rewrite_cache = load_rewrite_cache(cache_path)

        print("Final-answer rewrite stage")
        print(f"  round={round_index + 1}/{max_rounds}")
        print(f"  input_dir={input_dir}")
        print(f"  output_dir={rewrite_output_dir}")
        print(f"  model={rewrite_args.model}")
        print(f"  concurrency={rewrite_args.concurrency}")
        print(f"  files={len(paths)} existing_cache_entries={len(rewrite_cache)}")
        status_counts, eligible, failures = asyncio.run(
            process_paths(
                paths=paths,
                input_dir=input_dir,
                output_dir=rewrite_output_dir,
                args=rewrite_args,
                rewrite_cache=rewrite_cache,
                cache_path=cache_path,
                status_path=status_path,
                degenerate_thresholds=degenerate_thresholds,
            )
        )

        failure_ratio = failures / eligible if eligible else 0.0
        print("Final-answer rewrite summary:")
        for reason, count in status_counts.most_common():
            print(f"  {count}\t{reason}")
        print(f"  eligible={eligible}")
        print(f"  api_or_validation_failures={failures}")
        print(f"  failure_ratio={failure_ratio:.6f}")

        last_status_counts = status_counts
        last_eligible = eligible
        last_failures = failures
        last_failure_ratio = failure_ratio
        too_many_failures = failure_ratio > args.final_answer_rewrite_max_failure_ratio
        if args.final_answer_rewrite_max_failures is not None and failures > args.final_answer_rewrite_max_failures:
            too_many_failures = True
        if not too_many_failures:
            break
    else:
        raise RuntimeError(
            "Too many final-answer API/validation failures after retry rounds: "
            f"{last_failures}/{last_eligible} ({last_failure_ratio:.6f}); "
            f"threshold ratio={args.final_answer_rewrite_max_failure_ratio}, "
            f"absolute={args.final_answer_rewrite_max_failures}; "
            f"last_status_counts={dict(last_status_counts or {})}"
        )

    return rewrite_output_dir


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
    rewrite_file_uri_prefixes = parse_file_uri_prefix_mappings(args.rewrite_file_uri_prefix)
    degenerate_thresholds = build_degenerate_thresholds(args)
    input_dir = run_final_answer_rewrite_stage(
        args=args,
        input_dir=input_dir,
        output_dir=output_dir,
        degenerate_thresholds=degenerate_thresholds,
    )

    rows, total_jsons, warning_counts, wrong_question_ids = load_records(
        input_dir,
        stitch_runtime_hints=args.stitch_runtime_hints,
        system_prompt_mode=args.system_prompt_mode,
        system_prompt_insertion_mode=args.system_prompt_insertion_mode,
        assistant_format_mode=args.assistant_format_mode,
        tool_argument_order=args.tool_argument_order,
        image_storage_mode=args.image_storage_mode,
        image_cache_dir=image_cache_dir,
        invalid_image_aspect_policy=args.invalid_image_aspect_policy,
        max_image_aspect_ratio=args.max_image_aspect_ratio,
        image_aspect_pad_target_ratio=args.image_aspect_pad_target_ratio,
        num_workers=args.num_workers,
        only_correct_answers=args.only_correct_answers,
        rewrite_file_uri_prefixes=rewrite_file_uri_prefixes,
        drop_degenerate_conversations=args.drop_degenerate_conversations,
        degenerate_thresholds=degenerate_thresholds,
        degenerate_preview_chars=args.degenerate_preview_chars,
        conversion_progress_every=args.conversion_progress_every,
        disable_conversion_progress_bar=args.disable_conversion_progress_bar,
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
