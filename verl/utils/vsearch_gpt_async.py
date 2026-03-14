import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import logging
import asyncio
from functools import partial

from openai.types.chat import ChatCompletionMessage
from PIL import Image

import verl.utils.vsearch_role_play_prompt as prompts_no_tool_feedback
import verl.utils.vsearch_role_play_prompt_with_tool_feedback as prompts_with_tool_feedback
from verl.utils.vsearch import BBox

from insight_o3.utils.api import create_async_openai_client, query_api  # pyright: ignore[reportMissingImports]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

client = create_async_openai_client()


@dataclass
class GPTVisualSearchRequest:
    success: bool
    messages: list[ChatCompletionMessage]
    region_description: Optional[str] = None  # None if GPT makes no (further) request
    img_idx: Optional[int] = None  # 0-indexed image to search (multi-image mode only)
    answer: Optional[str] = None  # None if GPT returns no answer
    is_last_round: bool = False  # Whether last-round prompt was used this call
    tool_feedback: Optional[str] = None  # Parsed <tool_feedback> content if present
    display_text: Optional[str] = None  # Text to display to the user
    failure_reasons: list[str] = field(default_factory=list)  # Reasons for failure


# --- Minimal image helpers --------------------------------------------------
def _scale_image_to_area(image: Image.Image, max_area: int) -> Image.Image:
    w, h = image.size
    area = w * h
    if area <= max_area or max_area <= 0:
        return image
    ratio = (max_area / float(area)) ** 0.5
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return image.resize((new_w, new_h), Image.LANCZOS)


def _pil_to_data_url_jpeg(image: Image.Image) -> str:
    if image.mode in ["RGBA", "P"]:
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _crop_prepare_data_url(
    original: Image.Image,
    bbox: tuple[int, int, int, int],
    expand_ratio: float,
    min_side: int,
    min_area: int,
    max_area: int,
) -> Optional[str]:
    # Validate and expand bbox
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    if x1 == x2 or y1 == y2:
        return None
    w, h = original.size
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw == 0 or bh == 0:
        return None
    ex = int(bw * float(expand_ratio))
    ey = int(bh * float(expand_ratio))
    x1e = max(0, x1 - ex)
    y1e = max(0, y1 - ey)
    x2e = min(w, x2 + ex)
    y2e = min(h, y2 + ey)
    if x2e <= x1e or y2e <= y1e:
        return None
    crop = original.crop((x1e, y1e, x2e, y2e))
    # Enforce minimum short side
    cw, ch = crop.size
    short_side = min(cw, ch)
    if short_side <= 0:
        return None
    if short_side < int(min_side):
        s = int(min_side) / float(short_side)
        cw, ch = max(1, int(cw * s)), max(1, int(ch * s))
        crop = crop.resize((cw, ch), Image.LANCZOS)
    # Enforce minimum area
    area = cw * ch
    if area < int(min_area):
        s = (int(min_area) / float(area)) ** 0.5
        cw2, ch2 = max(1, int(cw * s)), max(1, int(ch * s))
        crop = crop.resize((cw2, ch2), Image.LANCZOS)
        cw, ch = cw2, ch2
    # Cap by max area for GPT
    crop = _scale_image_to_area(crop, int(max_area))
    return _pil_to_data_url_jpeg(crop)


# --- Minimal parsing (no regex) ---------------------------------------------
def _parse_region_description(text: str) -> Optional[str]:
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return None
    segment = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    if "region_description=" not in segment:
        return None
    part = segment.split("region_description=", 1)[1]
    if "{" in part and "}" in part:
        return part.split("{", 1)[1].rsplit("}", 1)[0].strip()
    return None


def _parse_region_description_multi_image(text: str) -> Optional[tuple[str, int]]:
    """Parse JSON tool call format: {"region_description": "...", "img_idx": N}"""
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return None
    segment = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    try:
        parsed = json.loads(segment)
    except json.JSONDecodeError:
        return None
    region_desc = parsed.get("region_description")
    img_idx = parsed.get("img_idx")
    if not isinstance(region_desc, str) or not isinstance(img_idx, int):
        return None
    return (region_desc, img_idx)


def _parse_boxed_answer(text: str) -> Optional[str]:
    if "\\boxed{" not in text:
        return None
    return text.split("\\boxed{", 1)[1].split("}", 1)[0].strip()


def _parse_tool_feedback(text: str) -> Optional[str]:
    """Extract content between <tool_feedback>...</tool_feedback> if present."""
    start_tag = "<tool_feedback>"
    end_tag = "</tool_feedback>"
    if start_tag not in text or end_tag not in text:
        return None
    segment = text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return segment


async def get_gpt_visual_search_request(
    initial_question: str,  # the initial question from the user, e.g., "What is the color of the car?"
    original_images: list[Image.Image],  # the original image(s) (w/o any resize) from the user
    messages: list[dict],  # history messages, initially empty
    bbox: BBox,  # the bounding box returned by the visual search tool
    model: str = "gpt-5-nano",  # which gpt model to use
    temperature: float = 1.0,  # temperature for the gpt model (NOTE: some model don't support custom temperature)
    gpt_image_max_area: int = 1280 * 1280,
    image_detail: str = "high",
    crop_expand_ratio: float = 0.0,
    crop_min_side: int = 56,
    crop_min_area: int = 112 * 112,
    max_tool_calls: int = 6,
    max_completion_tokens: int | None = None,
    max_round_retries: int = 3,
    reasoning_effort: str = None,
    enable_tool_feedback: bool = False,
    multi_image_input: bool = False,
    last_img_idx: int | None = None,  # img_idx from the previous tool call (for cropping the right image)
) -> GPTVisualSearchRequest:
    # Start with history and compute prior tool-call count
    out_messages: list = [] if messages is None else list(messages)
    prior_tool_calls = 0
    for m in out_messages:
        try:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role == "assistant" and isinstance(content, str) and "<tool_call>" in content:
                prior_tool_calls += 1
        except Exception:
            continue

    is_first_round = prior_tool_calls == 0
    is_last_round = False

    # Select prompt set based on flag
    if enable_tool_feedback:
        prompts = prompts_with_tool_feedback
    else:
        prompts = prompts_no_tool_feedback
    
    # Prepare first prompt/image once
    def _prepare_first_image(img: Image.Image, gpt_image_max_area: int) -> str:
        img = img.convert("RGB") if img.mode != "RGB" else img
        img = _scale_image_to_area(img, gpt_image_max_area)
        return _pil_to_data_url_jpeg(img)

    loop = asyncio.get_event_loop()

    if not out_messages:
        if multi_image_input:
            # Multi-image first turn: prepare all images and build query content
            image_urls = []
            for img in original_images:
                url = await loop.run_in_executor(None, _prepare_first_image, img, gpt_image_max_area)
                image_urls.append(url)
            sys_prompt = prompts.vsearch_sys_prompt_multi_image
            out_messages = [{"role": "system", "content": sys_prompt}]
            # Build query content with labeled images followed by the question
            query_content: list[dict] = []
            for i, url in enumerate(image_urls):
                query_content.append({"type": "text", "text": f"Image {i}:"})
                query_content.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail},
                })
            query_content.append({"type": "text", "text": initial_question})
            # Embed images directly in the query; no separate image_url for query_api
            pending_image_url = None
            pending_question = query_content
        else:
            pending_image_url = await loop.run_in_executor(None, _prepare_first_image, original_images[0], gpt_image_max_area)
            out_messages = [{"role": "system", "content": prompts.vsearch_sys_prompt}]
            pending_question = initial_question
    else:
        # Check bbox first; treat [0,0,0,0] as invalid
        bbox_valid = bool(bbox) and any(int(v) != 0 for v in bbox)
        crop_source = original_images[last_img_idx] if (multi_image_input and last_img_idx is not None) else original_images[0]
        if bbox_valid:
            pending_image_url = await loop.run_in_executor(
                None,
                partial(
                    _crop_prepare_data_url,
                    original=crop_source,
                    bbox=bbox,
                    expand_ratio=crop_expand_ratio,
                    min_side=crop_min_side,
                    min_area=crop_min_area,
                    max_area=gpt_image_max_area,
                )
            )
            pending_question = prompts.vsearch_user_hint if pending_image_url else prompts.vsearch_user_hint_fail
        else:
            pending_image_url = None
            pending_question = prompts.vsearch_user_hint_fail

    # If exceeded tool-call limit already, force last-round prompt for the first attempt
    if prior_tool_calls >= int(max_tool_calls):
        pending_question = prompts.vsearch_user_hint_last_round
        is_last_round = True

    current_messages: list = out_messages
    updated_messages = None
    failure_reasons = []

    attempt = 0
    while attempt < int(max_round_retries):
        try:
            messages, response = await query_api(
                query=pending_question,
                model=model,
                client=client,
                image_url=pending_image_url,
                image_url_extra_settings={"detail": image_detail},
                context=current_messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            updated_messages = messages + [response.choices[0].message]
        except Exception as e:
            logger.warning(f"query_api failed on {model} (attempt {attempt + 1} of {max_round_retries}): {e}")
            updated_messages = None

        # If the underlying call failed, keep prior inputs/messages and try again
        if not updated_messages or not isinstance(updated_messages[-1], ChatCompletionMessage):
            failure_reasons.append("query_api_failed")
            attempt += 1
            continue

        # Successful call: update messages and inspect format
        current_messages = updated_messages
        assistant_msg: ChatCompletionMessage = current_messages[-1]
        content: str = assistant_msg.content or ""
        finish_reason = response.choices[0].finish_reason

        img_idx = None
        if multi_image_input:
            parsed = _parse_region_description_multi_image(content)
            if parsed is not None:
                region_desc, img_idx = parsed
            else:
                region_desc = None
        else:
            region_desc = _parse_region_description(content)
        answer = _parse_boxed_answer(content)
        tool_feedback = None
        if enable_tool_feedback:
            feedback_str = _parse_tool_feedback(content)
            if is_first_round and feedback_str == "NA":
                tool_feedback = feedback_str
            if not is_first_round and feedback_str in ("helpful", "unhelpful"):
                tool_feedback = feedback_str

        # Check if the response follows the expected format
        if is_last_round:
            if not answer:
                failure_reasons.append(f"no_answer_in_last_round({finish_reason})")
                success = False
                logger.warning(
                    f"no answer from {model} in last round (attempt {attempt + 1} of {max_round_retries}; {finish_reason=})"
                )
            else:
                success = True
        else:
            if not (region_desc or answer):
                failure_reasons.append(f"no_tool_call_nor_answer({finish_reason})")
                success = False
                logger.warning(
                    f"no tool call nor answer from {model} (attempt {attempt + 1} of {max_round_retries}; {finish_reason=})"
                )
            else:
                success = True

        if enable_tool_feedback and not tool_feedback:
            if success:
                failure_reasons.append(f"no_feedback({finish_reason})")
            else:
                failure_reasons[-1] += f"+no_feedback({finish_reason})"
            success = False
            logger.warning(
                f"no feedback from {model} (attempt {attempt + 1} of {max_round_retries}; {finish_reason=})"
            )

        if success:
            return GPTVisualSearchRequest(
                success=True,
                messages=current_messages,
                region_description=region_desc,
                img_idx=img_idx,
                answer=answer,
                is_last_round=is_last_round,
                tool_feedback=tool_feedback,
                display_text=content,
                failure_reasons=failure_reasons,
            )
        else:
            attempt += 1

        if is_last_round:
            # Retry with the same inputs/messages
            current_messages = out_messages
        else:
            # Neither tool call nor answer (or no feedback) -> follow-up with a format hint or last-round
            pending_question = prompts.format_user_hint_multi_image if multi_image_input else prompts.format_user_hint
            pending_image_url = None

    # Exhausted retries
    return GPTVisualSearchRequest(
        success=False,
        messages=updated_messages or current_messages,  # type: ignore[arg-type]
        region_description=None,
        answer=None,
        is_last_round=is_last_round,
        tool_feedback=None,
        display_text=None,
        failure_reasons=failure_reasons,
    )
