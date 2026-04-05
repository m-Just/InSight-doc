import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from openai.types.chat import ChatCompletionMessage
from PIL import Image

from insight_o3.utils.api import create_async_openai_client, query_api  # pyright: ignore[reportMissingImports]

import verl.utils.vreasoner_v2_prompt as prompts

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

client = create_async_openai_client()
TERMINAL_STOP_SEQUENCES = ["</action>", "</response>"]
STRICT_REPLY_RE = re.compile(
    r"""
    ^\s*
    <observation>\s*(?P<observation>.*?)\s*</observation>\s*
    <state>\s*(?P<state>.*?)\s*</state>\s*
    <plan>\s*(?P<plan>.*?)\s*</plan>\s*
    (?:
        <action>\s*(?P<action>.*?)\s*</action>
        |
        <response>\s*(?P<response>.*?)\s*</response>
    )
    \s*$
    """,
    re.DOTALL | re.VERBOSE,
)
OBSERVATION_PLANNING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi will\b",
        r"\bi'll\b",
        r"\bi plan to\b",
        r"\bmy plan is to\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\bi want to\b",
        r"\bnext i\b",
        r"\bawait(?:ing)?\b",
        r"\bzoom in\b",
        r"\bcall the tool\b",
        r"\buse the tool\b",
        r"\btool call\b",
        r"\bi requested\b",
        r"\bi am going to\b",
    ]
]


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


def _extract_section(text: str, tag: str) -> Optional[str]:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
        return None
    return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()


def _extract_terminal_section(text: str, tag: str) -> Optional[str]:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text:
        return None
    tail = text.split(start_tag, 1)[1]
    if end_tag in tail:
        return tail.split(end_tag, 1)[0].strip()
    # Allow EOF-terminated content when generation stops on the closing tag.
    return tail.strip() or None


def _normalize_terminal_content(text: str) -> str:
    stripped = text.rstrip()
    if "<response>" in stripped and "</response>" not in stripped and "<action>" not in stripped:
        return stripped + "\n</response>"
    if "<action>" in stripped and "</action>" not in stripped and "<response>" not in stripped:
        return stripped + "\n</action>"
    return text


def _normalize_assistant_message(message: ChatCompletionMessage) -> dict:
    normalized = message.to_dict()
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = _normalize_terminal_content(content)
    return normalized


def _extract_action_content(text: str) -> Optional[str]:
    action = _extract_terminal_section(text, "action")
    if action is None or not action:
        return None
    return action


def _parse_action(text: str) -> Optional[tuple[Optional[str], int]]:
    action = _extract_action_content(text)
    if action is None:
        return None
    try:
        payload = json.loads(action)
    except json.JSONDecodeError:
        return None
    if payload.get("name") != "image_zoom_in_tool":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    img_idx = arguments.get("img_idx")
    if not isinstance(img_idx, int):
        return None
    region_description = arguments.get("region_description")
    if region_description is None:
        return None, img_idx
    if not isinstance(region_description, str):
        return None
    region_description = region_description.strip()
    return (region_description or None), img_idx


def _parse_response(text: str) -> Optional[str]:
    response = _extract_terminal_section(text, "response")
    if response is None:
        return None
    response = response.strip()
    return response or None


def _has_required_sections(text: str) -> bool:
    return STRICT_REPLY_RE.fullmatch(text) is not None


def _observation_is_semantically_valid(text: str) -> bool:
    observation = _extract_section(text, "observation")
    if observation is None:
        return False
    lowered = observation.strip()
    if not lowered:
        return False
    return not any(pattern.search(lowered) for pattern in OBSERVATION_PLANNING_PATTERNS)


def _build_multimodal_query(
    labeled_images: list[tuple[int, str]],
    trailing_text: str,
    image_detail: str,
    wrapper_tag: str | None = None,
    format_hint: str | None = None,
) -> list[dict]:
    content: list[dict] = []
    if wrapper_tag is not None:
        content.append({"type": "text", "text": f"<{wrapper_tag}>\n"})
    for i, (img_idx, url) in enumerate(labeled_images):
        if i > 0:
            content.append({"type": "text", "text": prompts.IMAGE_SEPARATOR})
        content.append({"type": "text", "text": f"Image {img_idx}:"})
        content.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    if wrapper_tag is not None:
        content.append({"type": "text", "text": f"\n</{wrapper_tag}>"})
    trailing_suffix = ""
    if format_hint is not None:
        trailing_suffix = f"\n\n{format_hint}"
    content.append({"type": "text", "text": trailing_text + trailing_suffix})
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
            if role == "assistant" and isinstance(content, str) and _parse_action(content) is not None:
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
            (
                f"\n\nUser question: {initial_question}\n\n"
                f"{prompts.INITIAL_QUERY_HINT}"
            ),
            image_detail,
        )
        current_messages = [{"role": "system", "content": prompts.VSEARCH_SYS_PROMPT}]
    else:
        current_messages = out_messages
        if tool_result is None:
            raise RuntimeError("tool_result is required after the initial round")
        if tool_result.status == "error":
            hint = prompts.build_tool_result_fail_hint(tool_result.requested_img_idx)
            tool_error = tool_result.error_message or "ERROR: The previous zoom request did not produce a usable result."
            pending_question = [
                {
                    "type": "text",
                    "text": (
                        f"<tool_response>\n{tool_error}\n</tool_response>\n\n"
                        f"{hint}\n\n"
                        f"{prompts.FOLLOWUP_FORMAT_HINT}"
                    ),
                }
            ]
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
                hint,
                image_detail,
                wrapper_tag="tool_response",
                format_hint=prompts.FOLLOWUP_FORMAT_HINT,
            )

    if is_last_round:
        if isinstance(pending_question, list):
            pending_question = list(pending_question)
            pending_question.append({"type": "text", "text": "\n\n" + prompts.LAST_ROUND_HINT})
        else:
            pending_question = [{"type": "text", "text": prompts.LAST_ROUND_HINT}]

    attempt = 0
    while attempt < int(max_round_retries):
        retry_hint = prompts.FORMAT_REPAIR_HINT
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

        action = _parse_action(content)
        answer = _parse_response(content)
        has_sections = _has_required_sections(content)
        observation_ok = _observation_is_semantically_valid(content)

        if is_last_round:
            success = has_sections and observation_ok and answer is not None and action is None
            if not success:
                if has_sections and not observation_ok:
                    failure_reasons.append("invalid_observation_semantics")
                    retry_hint = prompts.OBSERVATION_REPAIR_HINT
                if action is not None:
                    failure_reasons.append("tool_call_budget_exceeded")
                failure_reasons.append(f"invalid_last_round({finish_reason})")
        else:
            success = has_sections and observation_ok and (action is not None or answer is not None)
            if not success:
                if has_sections and not observation_ok:
                    failure_reasons.append("invalid_observation_semantics")
                    retry_hint = prompts.OBSERVATION_REPAIR_HINT
                failure_reasons.append(f"invalid_format_or_empty({finish_reason})")

        if success:
            region_description = None
            img_idx = None
            if action is not None:
                region_description, img_idx = action
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
        pending_question = [{"type": "text", "text": retry_hint}]

    return GPTVisualSearchRequest(
        success=False,
        messages=updated_messages or current_messages,
        is_last_round=is_last_round,
        failure_reasons=failure_reasons,
    )
