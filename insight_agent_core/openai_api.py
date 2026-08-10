from __future__ import annotations

import os
from pprint import pformat
from typing import Any

import httpx
from openai import AsyncOpenAI, DEFAULT_TIMEOUT
from openai._types import NOT_GIVEN, NotGiven, Timeout
from openai._utils import is_given
from openai.types.chat import ChatCompletion, ChatCompletionMessage


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _api_proxy_http_client(timeout: float | Timeout | None | NotGiven) -> httpx.AsyncClient | None:
    http_proxy = os.getenv("API_HTTP_PROXY")
    https_proxy = os.getenv("API_HTTPS_PROXY")
    if not http_proxy and not https_proxy:
        return None

    transport_kwargs: dict[str, Any] = {}
    if _env_flag_enabled("API_DISABLE_KEEPALIVE"):
        transport_kwargs["limits"] = httpx.Limits(max_keepalive_connections=0)

    mounts = {}
    if http_proxy:
        mounts["http://"] = httpx.AsyncHTTPTransport(proxy=http_proxy, **transport_kwargs)
    if https_proxy:
        mounts["https://"] = httpx.AsyncHTTPTransport(proxy=https_proxy, **transport_kwargs)
    httpx_timeout = DEFAULT_TIMEOUT if not is_given(timeout) else timeout
    return httpx.AsyncClient(mounts=mounts, timeout=httpx_timeout, trust_env=False)


def create_async_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
) -> AsyncOpenAI:
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")
    if not is_given(timeout):
        timeout = os.getenv("OPENAI_CLIENT_TIMEOUT", NOT_GIVEN)
        if isinstance(timeout, str):
            timeout = float(timeout)

    http_client = _api_proxy_http_client(timeout)
    if http_client is None:
        return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client)


def prune_non_text_content(message: dict[str, Any] | ChatCompletionMessage) -> dict[str, Any]:
    if isinstance(message, ChatCompletionMessage):
        message = message.to_dict()
    message_pruned: dict[str, Any] = {}
    for key, value in message.items():
        if key != "content":
            message_pruned[key] = value
            continue
        if value is None or isinstance(value, str):
            message_pruned[key] = value
        else:
            message_pruned[key] = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    message_pruned[key].append(item)
                elif isinstance(item, dict):
                    item_type = item.get("type", "unknown")
                    message_pruned[key].append({"type": item_type, item_type: "[pruned]"})
                else:
                    message_pruned[key].append(str(item))
    return message_pruned


def _format_error_message(
    err: Exception,
    model: str,
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    show_detailed_error_message: bool = False,
) -> str:
    error_message = f'failed to query "{model}" at {client.base_url}: {err!r}'
    if show_detailed_error_message:
        error_message += (
            "\n  Client information:"
            f"\n    client.timeout={client.timeout!r}"
            f"\n    client.max_retries={client.max_retries!r}"
        )
        formatted_input_messages = pformat([prune_non_text_content(message) for message in messages])
        formatted_input_messages = formatted_input_messages.replace("\n", "\n    ")
        error_message += f"\n  Input messages:\n    {formatted_input_messages}"
    return error_message


async def complete_chat_and_maybe_log(
    messages: list[dict[str, Any]],
    model: str,
    client: AsyncOpenAI,
    show_detailed_error_message: bool = False,
    **chat_completion_kwargs: Any,
) -> ChatCompletion:
    if not isinstance(client, AsyncOpenAI):
        raise TypeError("client must be an instance of AsyncOpenAI")
    try:
        return await client.chat.completions.create(
            messages=messages,
            model=model,
            **chat_completion_kwargs,
        )
    except Exception as err:
        raise RuntimeError(_format_error_message(err, model, client, messages, show_detailed_error_message)) from err


async def query_api(
    query: str | list[dict[str, Any]],
    model: str,
    client: AsyncOpenAI,
    image_url: str | None = None,
    image_urls: list[str] | None = None,
    image_url_extra_settings: dict[str, Any] | None = None,
    context: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], ChatCompletion]:
    if isinstance(query, str):
        query = [{"type": "text", "text": query}]

    if image_url is not None and image_urls is not None:
        raise ValueError("Provide either image_url or image_urls, not both")
    if image_urls is None and image_url is not None:
        image_urls = [image_url]
    if image_urls is not None:
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": current_image_url, **(image_url_extra_settings or {})},
            }
            for current_image_url in image_urls
        ]
        query = [*image_content, *query]

    messages = [*context] if context else []
    messages.append({"role": "user", "content": query})
    return messages, await complete_chat_and_maybe_log(
        messages=messages,
        model=model,
        client=client,
        **kwargs,
    )
