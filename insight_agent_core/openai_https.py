from __future__ import annotations

import base64
import asyncio
import copy
import io
import json
import os
import time
from typing import Any

from PIL import Image

from .runtime import CoreGenerationOutput


_MISSING = object()


def _image_to_data_url(image: Any, *, image_format: str) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL.Image.Image for HTTPS image payload, got {type(image).__name__}")
    buffer = io.BytesIO()
    fmt = image_format.upper()
    save_image = image
    if fmt in {"JPEG", "JPG"} and image.mode not in {"RGB", "L"}:
        save_image = image.convert("RGB")
    save_image.save(buffer, format=fmt)
    mime = "jpeg" if fmt == "JPG" else fmt.lower()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _coerce_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        return copy.deepcopy(schema)
    return {"type": "function", "function": copy.deepcopy(schema)}


def _arguments_to_json_object(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return arguments
    return arguments


def _format_structured_tool_calls(tool_calls: Any) -> str:
    formatted: list[str] = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        arguments = getattr(function, "arguments", None) if function is not None else None
        if not name and isinstance(tool_call, dict):
            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
        if not name:
            continue
        payload = {"name": name, "arguments": _arguments_to_json_object(arguments)}
        formatted.append(
            "<tool_call>\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "</tool_call>"
        )
    return "\n".join(formatted)


def _message_content_to_openai_parts(
    content: Any,
    *,
    image_iter,
    image_format: str,
    image_detail: str | None,
) -> Any:
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        return str(content)

    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            converted.append({"type": "text", "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            converted.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if item_type in {"image", "image_url"}:
            image_value = next(image_iter, _MISSING)
            if image_value is _MISSING:
                image_value = item.get("image")
            if image_value is None and item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and image_url.get("url"):
                    image_url_payload = copy.deepcopy(image_url)
                    if image_detail:
                        image_url_payload.setdefault("detail", image_detail)
                    converted.append({"type": "image_url", "image_url": image_url_payload})
                    continue
                if isinstance(image_url, str):
                    image_url_payload = {"url": image_url}
                    if image_detail:
                        image_url_payload["detail"] = image_detail
                    converted.append({"type": "image_url", "image_url": image_url_payload})
                    continue
            if image_value is None:
                image_value = next(image_iter)
            if isinstance(image_value, str):
                image_url_payload = {"url": image_value}
                if image_detail:
                    image_url_payload["detail"] = image_detail
                converted.append({"type": "image_url", "image_url": image_url_payload})
            else:
                image_url_payload = {"url": _image_to_data_url(image_value, image_format=image_format)}
                if image_detail:
                    image_url_payload["detail"] = image_detail
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": image_url_payload,
                    }
                )
            continue
        converted.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
    return converted


def convert_messages_to_openai_chat(
    messages: list[dict[str, Any]],
    *,
    images: list[Any] | None,
    image_format: str,
    image_detail: str | None,
    coerce_tool_role_to_user: bool,
) -> list[dict[str, Any]]:
    image_values = iter(list(images or []))
    converted_messages: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if coerce_tool_role_to_user and role == "tool":
            role = "user"
        converted = {
            "role": role,
            "content": _message_content_to_openai_parts(
                message.get("content"),
                image_iter=image_values,
                image_format=image_format,
                image_detail=image_detail,
            ),
        }
        if "name" in message:
            converted["name"] = message["name"]
        if "tool_call_id" in message:
            converted["tool_call_id"] = message["tool_call_id"]
        converted_messages.append(converted)
    return converted_messages


class OpenAIChatEndpointPool:
    """OpenAI-compatible HTTPS chat-completion generation backend.

    This keeps the standalone agent loop shared, but uses a remote chat API for
    generation. If the endpoint returns vLLM-style token_ids, they are used
    directly; otherwise generated text is encoded with the local tokenizer so the
    existing tool parser can run unchanged.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        tokenizer: Any,
        timeout: float,
        max_retries: int,
        image_format: str = "PNG",
        image_detail: str | None = "high",
        reasoning_effort: str | None = "high",
        send_tool_schema: bool = True,
        coerce_tool_role_to_user: bool = False,
    ) -> None:
        from insight_o3.utils import api as insight_api

        if insight_api.log_chat_completion is None:
            try:
                from insight_o3.utils.api_logger import log_chat_completion
            except Exception as exc:
                raise RuntimeError("API logging is required for HTTPS generation but api_logger is unavailable") from exc
            insight_api.log_chat_completion = log_chat_completion

        client = insight_api.create_async_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.client = client.with_options(max_retries=max_retries)
        self.complete_chat_and_maybe_log = insight_api.complete_chat_and_maybe_log
        self.api_logging_enabled = insight_api.log_chat_completion is not None
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.image_format = image_format
        self.image_detail = image_detail
        self.reasoning_effort = reasoning_effort
        self.send_tool_schema = send_tool_schema
        self.coerce_tool_role_to_user = coerce_tool_role_to_user
        print(
            "https_openai_chat api_stack_ready "
            f"base_url={base_url} model={model} api_logging_enabled={self.api_logging_enabled} "
            f"image_detail={self.image_detail} reasoning_effort={self.reasoning_effort}",
            flush=True,
        )

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CoreGenerationOutput:
        if videos:
            raise NotImplementedError("HTTPS OpenAI-compatible generation does not support video payloads")
        if messages is None:
            raise ValueError("HTTPS OpenAI-compatible generation requires current chat messages")

        start = time.perf_counter()
        print(
            "https_openai_chat request_prepare_start "
            f"pid={os.getpid()} request_id={request_id} "
            f"prompt_tokens={len(prompt_ids)} images={len(images or [])} messages={len(messages)}",
            flush=True,
        )
        request_messages = convert_messages_to_openai_chat(
            messages,
            images=images,
            image_format=self.image_format,
            image_detail=self.image_detail,
            coerce_tool_role_to_user=self.coerce_tool_role_to_user,
        )
        payload_chars = len(json.dumps(request_messages, ensure_ascii=False))
        print(
            "https_openai_chat request_prepare_done "
            f"pid={os.getpid()} request_id={request_id} "
            f"elapsed_s={time.perf_counter() - start:.3f} payload_chars={payload_chars}",
            flush=True,
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "max_tokens": sampling_params.get("max_tokens"),
            "reasoning_effort": self.reasoning_effort,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if self.send_tool_schema and tools:
            kwargs["tools"] = [_coerce_openai_tool(schema) for schema in tools]
        print(
            "https_openai_chat request_send_start "
            f"pid={os.getpid()} request_id={request_id} timeout_s={self.timeout}",
            flush=True,
        )
        try:
            response = await asyncio.wait_for(
                self.complete_chat_and_maybe_log(
                    messages=request_messages,
                    model=self.model,
                    client=self.client,
                    show_detailed_error_message=True,
                    **{key: value for key, value in kwargs.items() if key not in {"model", "messages"}},
                ),
                timeout=self.timeout + 5.0,
            )
        except Exception as exc:
            print(
                "https_openai_chat request_error "
                f"pid={os.getpid()} request_id={request_id} "
                f"elapsed_s={time.perf_counter() - start:.3f} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        print(
            "https_openai_chat request_done "
            f"pid={os.getpid()} request_id={request_id} elapsed_s={time.perf_counter() - start:.3f}",
            flush=True,
        )
        if not response.choices:
            return CoreGenerationOutput(token_ids=[])

        choice = response.choices[0]
        token_ids = getattr(choice, "token_ids", None)
        if token_ids is None and getattr(choice, "model_extra", None):
            token_ids = choice.model_extra.get("token_ids")
        if token_ids is not None:
            return CoreGenerationOutput(token_ids=list(token_ids))

        message = choice.message
        content = message.content or ""
        structured_tool_text = _format_structured_tool_calls(getattr(message, "tool_calls", None))
        if structured_tool_text:
            content = f"{content}\n{structured_tool_text}" if content else structured_tool_text
        return CoreGenerationOutput(
            token_ids=self.tokenizer.encode(content, add_special_tokens=False),
        )
