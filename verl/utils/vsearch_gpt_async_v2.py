import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from openai.types.chat import ChatCompletionMessage
from PIL import Image

from insight_o3.utils.api import create_async_openai_client, query_api  # pyright: ignore[reportMissingImports]

import verl.utils.vreasoner_v2_prompt as prompts

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

client = create_async_openai_client()
TERMINAL_STOP_SEQUENCES = ["</tool_call>"]


@dataclass
class GPTVisualSearchRequest:
    success: bool
    messages: list[ChatCompletionMessage]
    region_description: Optional[str] = None
    img_idx: Optional[int] = None
    answer: Optional[str] = None
    is_last_round: bool = False
    tool_feedback: Optional[str] = None
    display_text: Optional[str] = None
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    status: Literal["success", "error"]
    requested_img_idx: int | None = None
    new_img_idx: int | None = None
    error_message: str | None = None


def _scale_image_to_area(image: Image.Image, max_area: int) -> Image.Image:
    w, h = image.size
    area = w * h
    if area <= max_area or max_area <= 0:
        return image
    ratio = (max_area / float(area)) ** 0.5
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return image.resize((new_w, new_h), Image.LANCZOS)


def _pil_to_data_url(image: Image.Image, image_format: str) -> str:
    if image_format == "JPEG" and image.mode in ["RGBA", "P"]:
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=image_format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime_subtype = image_format.lower()
    return f"data:image/{mime_subtype};base64,{b64}"


def _prepare_image_data_url(image: Image.Image, max_area: int, png_max_area: int) -> str:
    scaled = _scale_image_to_area(image, max_area)
    image_format = "PNG" if png_max_area > 0 and scaled.size[0] * scaled.size[1] <= png_max_area else "JPEG"
    if image_format == "JPEG" and scaled.mode != "RGB":
        scaled = scaled.convert("RGB")
    return _pil_to_data_url(scaled, image_format)


def _normalize_tool_call_content(text: str) -> str:
    """Append </tool_call> when generation was cut off by a stop sequence."""
    stripped = text.rstrip()
    if "<tool_call>" in stripped and "</tool_call>" not in stripped:
        return stripped + "</tool_call>"
    return text


def _normalize_assistant_message(message: ChatCompletionMessage) -> dict:
    normalized = message.to_dict()
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = _normalize_tool_call_content(content)
    return normalized


def _parse_tool_call(text: str) -> Optional[tuple[str, int]]:
    """Parse <tool_call>{"region_description": "...", "img_idx": N}</tool_call>."""
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


def _build_multimodal_query(
    labeled_images: list[tuple[int, str]],
    trailing_text: str,
    image_detail: str,
) -> list[dict]:
    content: list[dict] = []
    for i, (img_idx, url) in enumerate(labeled_images):
        if i > 0:
            content.append({"type": "text", "text": prompts.IMAGE_SEPARATOR})
        content.append({"type": "text", "text": f"Image {img_idx}:"})
        content.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    content.append({"type": "text", "text": trailing_text})
    return content


async def get_gpt_visual_search_request_v2(
    initial_question: str,
    presented_images: list[Image.Image],
    messages: list[dict],
    model: str = "gpt-5-nano",
    temperature: float = 1.0,
    gpt_image_max_area: int = 1280 * 1280,
    png_max_area: int = 1280 * 1280,
    image_detail: str = "high",
    max_tool_calls: int = 6,
    max_completion_tokens: int | None = None,
    max_round_retries: int = 3,
    reasoning_effort: str = None,
    tool_result: ToolResult | None = None,
    enable_stop: bool = False,
) -> GPTVisualSearchRequest:
    out_messages: list = [] if messages is None else list(messages)
    prior_tool_calls = 0
    for m in out_messages:
        try:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role == "assistant" and isinstance(content, str) and _parse_tool_call(content) is not None:
                prior_tool_calls += 1
        except Exception:
            continue

    is_last_round = prior_tool_calls >= int(max_tool_calls)
    failure_reasons = []
    updated_messages = None

    if not out_messages:
        labeled_images = []
        for img_idx, image in enumerate(presented_images):
            labeled_images.append((img_idx, _prepare_image_data_url(image, gpt_image_max_area, png_max_area)))
        pending_question = _build_multimodal_query(
            labeled_images,
            f"\n\n{initial_question}",
            image_detail,
        )
        current_messages = [{"role": "system", "content": prompts.VSEARCH_SYS_PROMPT}]
    else:
        current_messages = out_messages
        if tool_result is None:
            raise RuntimeError("tool_result is required after the initial round")
        if tool_result.status == "error":
            hint = prompts.build_tool_result_fail_hint(tool_result.requested_img_idx)
            tool_error = tool_result.error_message or "The previous zoom request did not produce a usable result."
            pending_question = [{"type": "text", "text": f"{tool_error}\n\n{hint}"}]
        else:
            if tool_result.status != "success":
                raise RuntimeError(f"invalid tool_result status: {tool_result.status}")
            if tool_result.new_img_idx is None:
                raise RuntimeError(
                    "tool_result.new_img_idx is required after a successful tool result; "
                    "got None while tool_result.status='success'"
                )
            image = presented_images[tool_result.new_img_idx]
            hint = prompts.build_tool_result_hint(tool_result.new_img_idx)
            pending_question = _build_multimodal_query(
                [(tool_result.new_img_idx, _prepare_image_data_url(image, gpt_image_max_area, png_max_area))],
                f"\n\n{hint}",
                image_detail,
            )

    if is_last_round:
        if isinstance(pending_question, list):
            pending_question = list(pending_question)
            pending_question.append({"type": "text", "text": "\n\n" + prompts.LAST_ROUND_HINT})
        else:
            pending_question = [{"type": "text", "text": prompts.LAST_ROUND_HINT}]

    attempt = 0
    while attempt < int(max_round_retries):
        try:
            messages_out, response = await query_api(
                query=pending_question,
                model=model,
                client=client,
                image_url=None,
                image_url_extra_settings={"detail": image_detail},
                context=current_messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                **({"stop": TERMINAL_STOP_SEQUENCES} if enable_stop else {}),
            )
            updated_messages = messages_out + [response.choices[0].message]
        except Exception as e:
            logger.warning(f"query_api failed on {model} (attempt {attempt + 1} of {max_round_retries}): {e}")
            updated_messages = None

        if not updated_messages or not isinstance(updated_messages[-1], ChatCompletionMessage):
            failure_reasons.append("query_api_failed")
            attempt += 1
            continue

        current_messages = updated_messages
        assistant_msg: ChatCompletionMessage = current_messages[-1]
        normalized_assistant_msg = _normalize_assistant_message(assistant_msg)
        current_messages[-1] = normalized_assistant_msg
        content_value = normalized_assistant_msg.get("content")
        content: str = content_value if isinstance(content_value, str) else ""
        finish_reason = response.choices[0].finish_reason

        tool_call = _parse_tool_call(content)
        answer = _parse_boxed_answer(content)

        if is_last_round:
            success = answer is not None and tool_call is None
            if not success:
                if tool_call is not None:
                    failure_reasons.append("tool_call_budget_exceeded")
                failure_reasons.append(f"no_answer_in_last_round({finish_reason})")
        else:
            success = tool_call is not None or answer is not None
            if not success:
                failure_reasons.append(f"no_tool_call_nor_answer({finish_reason})")

        if success:
            region_description = None
            img_idx = None
            if tool_call is not None:
                region_description, img_idx = tool_call
            return GPTVisualSearchRequest(
                success=True,
                messages=current_messages,
                region_description=region_description,
                img_idx=img_idx,
                answer=answer,
                is_last_round=is_last_round,
                display_text=content,
                failure_reasons=failure_reasons,
            )

        attempt += 1
        pending_question = [{"type": "text", "text": prompts.FORMAT_REPAIR_HINT}]

    return GPTVisualSearchRequest(
        success=False,
        messages=updated_messages or current_messages,
        is_last_round=is_last_round,
        failure_reasons=failure_reasons,
    )
