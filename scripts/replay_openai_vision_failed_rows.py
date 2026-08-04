#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from PIL import Image


def cap_size_by_area(size: tuple[int, int], max_area: int) -> tuple[int, int]:
    width, height = size
    if not max_area or width * height <= max_area:
        return size
    scale = (max_area / float(width * height)) ** 0.5
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_dims_by_factor(size: tuple[int, int], factor: float) -> tuple[int, int]:
    width, height = size
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


def image_to_data_url(image: Image.Image, *, image_format: str, quality: int) -> tuple[str, int]:
    buffer = io.BytesIO()
    fmt = image_format.upper()
    save_image = image
    save_kwargs: dict[str, Any] = {}
    if fmt in {"JPEG", "JPG"}:
        if image.mode not in {"RGB", "L"}:
            save_image = image.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    save_image.save(buffer, format=fmt, **save_kwargs)
    raw = buffer.getvalue()
    mime = "jpeg" if fmt in {"JPEG", "JPG"} else fmt.lower()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/{mime};base64,{encoded}", len(raw)


def load_and_resize_image(path: Path, *, rescale: float, max_area: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if rescale != 1.0:
        image = image.resize(resize_dims_by_factor(image.size, rescale), Image.LANCZOS)
    capped = cap_size_by_area(image.size, max_area)
    if capped != image.size:
        image = image.resize(capped, Image.LANCZOS)
    return image


def build_messages(
    row: dict[str, Any],
    *,
    bundle_dir: Path,
    image_format: str,
    image_detail: str | None,
    jpeg_quality: int,
    rescale: float,
    max_area: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    raw_image_bytes = 0
    image_dims = []
    for rel_path in row["images"]:
        image = load_and_resize_image(bundle_dir / rel_path, rescale=rescale, max_area=max_area)
        image_dims.append(list(image.size))
        url, raw_bytes = image_to_data_url(image, image_format=image_format, quality=jpeg_quality)
        raw_image_bytes += raw_bytes
        payload = {"url": url}
        if image_detail:
            payload["detail"] = image_detail
        parts.append({"type": "image_url", "image_url": payload})
    parts.append({"type": "text", "text": row["question"]})
    messages = [{"role": "user", "content": parts}]
    stats = {
        "num_images": len(row["images"]),
        "raw_image_bytes": raw_image_bytes,
        "image_dims": image_dims,
        "payload_chars": len(json.dumps(messages, ensure_ascii=False)),
    }
    return messages, stats


async def run_one(
    row: dict[str, Any],
    *,
    client: AsyncOpenAI,
    args: argparse.Namespace,
    bundle_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    messages, payload_stats = build_messages(
        row,
        bundle_dir=bundle_dir,
        image_format=args.image_format,
        image_detail=None if args.image_detail.lower() in {"", "none", "null"} else args.image_detail,
        jpeg_quality=args.jpeg_quality,
        rescale=args.rescale,
        max_area=args.max_area,
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.reasoning_effort.lower() not in {"", "none", "null"}:
        kwargs["reasoning_effort"] = args.reasoning_effort
    try:
        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content if response.choices else ""
        return {
            "ok": True,
            "sample_index": row["sample_index"],
            "data_source": row["data_source"],
            "uid": row["uid"],
            "elapsed_s": time.perf_counter() - started,
            "response": text,
            **payload_stats,
        }
    except Exception as exc:
        response_obj = getattr(exc, "response", None)
        status_code = getattr(response_obj, "status_code", None)
        response_text = getattr(response_obj, "text", None)
        return {
            "ok": False,
            "sample_index": row["sample_index"],
            "data_source": row["data_source"],
            "uid": row["uid"],
            "elapsed_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "status_code": status_code,
            "response_text": response_text[:2000] if isinstance(response_text, str) else response_text,
            **payload_stats,
        }


async def main_async(args: argparse.Namespace) -> None:
    bundle_dir = Path(args.bundle_dir).resolve()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env}")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise SystemExit("missing --base-url or OPENAI_BASE_URL")

    rows = [json.loads(line) for line in (bundle_dir / "failed_rows.jsonl").read_text().splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    output_path = Path(args.output).resolve() if args.output else bundle_dir / "replay_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(
                "start",
                row["sample_index"],
                row["data_source"],
                f"nimg={len(row['images'])}",
                flush=True,
            )
            result = await run_one(row, client=client, args=args, bundle_dir=bundle_dir)
            print(
                "done",
                row["sample_index"],
                "ok" if result["ok"] else result.get("error_type"),
                f"elapsed={result['elapsed_s']:.1f}s",
                f"payload_mb={result['payload_chars'] / 1e6:.2f}",
                flush=True,
            )
            return result

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for coro in asyncio.as_completed([guarded(row) for row in rows]):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay packed failed multimodal OpenAI-compatible requests.")
    parser.add_argument("--bundle-dir", default=".", help="Directory containing failed_rows.jsonl and images/")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--proxy", default=None, help="Optional HTTP(S) proxy URL to set for this process.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-format", default="png", choices=["png", "jpeg", "jpg"])
    parser.add_argument("--image-detail", default="high")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--rescale", type=float, default=0.5)
    parser.add_argument("--max-area", type=int, default=12_250_000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="null")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
