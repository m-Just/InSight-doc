from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
from IPython.display import HTML, Markdown, display
from PIL import Image

REWRITE_METADATA_KEY = "unanswerable_final_answer_rewrite"
DEFAULT_MAX_IMAGE_SIDE = 1200


def load_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def _load_reference_rows(reference_dir_str: str) -> dict[str, dict[str, Any]]:
    reference_dir = Path(reference_dir_str)
    rows: dict[str, dict[str, Any]] = {}
    for name in ("valid_veqa_qas.json", "valid_multivisual_group_qas.json"):
        path = reference_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        source_rows = data.get("qas", []) if isinstance(data, dict) else data
        for row in source_rows:
            qa_id = row.get("qa_id")
            if isinstance(qa_id, str):
                rows[qa_id] = row
    manifest_path = reference_dir / "manifest.jsonl"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qa_id = row.get("question_id")
                if isinstance(qa_id, str):
                    rows.setdefault(qa_id, row)
    return rows


def question_id(record: dict[str, Any]) -> str:
    extra = record.get("extra_info")
    if isinstance(extra, dict):
        return str(extra.get("question_id") or "")
    return ""


def final_answer_payload(record: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return None
    for idx in range(len(conversation) - 1, -1, -1):
        message = conversation[idx]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, dict) and ("answer" in content or "think" in content):
            return idx, content
    return None


def final_answer_text(record: dict[str, Any]) -> str:
    payload = final_answer_payload(record)
    if payload is None:
        return ""
    _, content = payload
    answer = str(content.get("answer") or "").strip()
    think = str(content.get("think") or "").strip()
    if answer:
        return answer
    return think


def base_question_id(question_id_value: str) -> str:
    if "__drop_" in question_id_value:
        return question_id_value.split("__drop_", 1)[0]
    return question_id_value


def dropped_target(question_id_value: str) -> str | None:
    marker = "__drop_relevant_"
    if marker not in question_id_value or not question_id_value.endswith("_unanswerable"):
        return None
    suffix = question_id_value.split(marker, 1)[1]
    return suffix[: -len("_unanswerable")]


def _path_from_ref(ref: dict[str, Any]) -> Path:
    path_value = ref.get("path")
    if isinstance(path_value, str) and path_value:
        return Path(path_value)
    uri = ref.get("uri")
    if isinstance(uri, str) and uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(unquote(parsed.path))
    raise FileNotFoundError(f"Could not resolve image path from ref: {ref}")


@lru_cache(maxsize=1024)
def _load_original_image(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")


def _resize_for_display(image: Image.Image, max_side: int | None = DEFAULT_MAX_IMAGE_SIDE) -> Image.Image:
    if max_side is None or max_side <= 0:
        return image
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image
    scale = max_side / float(current_max)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size)


def rendered_presented_image(record: dict[str, Any], presented_idx: int) -> Image.Image:
    refs = record.get("image_references") or {}
    presented = refs.get("presented_images") or []
    input_images = refs.get("input_images") or []
    meta = next(item for item in presented if item.get("presented_img_idx") == presented_idx)
    source_idx = int(meta.get("source_original_img_idx", meta.get("presented_img_idx", 0)))
    source_ref = input_images[source_idx]
    source_path = _path_from_ref(source_ref)
    image = _load_original_image(str(source_path)).copy()
    bbox = meta.get("bbox_on_original")
    if isinstance(bbox, list) and len(bbox) == 4:
        image = image.crop(tuple(int(v) for v in bbox))
    display_size = meta.get("display_size")
    if isinstance(display_size, list) and len(display_size) == 2:
        image = image.resize((int(display_size[0]), int(display_size[1])))
    return _resize_for_display(image)


def _infer_reference_image_root(record: dict[str, Any]) -> Path | None:
    refs = record.get("image_references") or {}
    input_images = refs.get("input_images") or []
    if not input_images:
        return None
    try:
        sample_path = _path_from_ref(input_images[0])
    except Exception:
        return None
    parts = sample_path.parts
    if "pdf_image" not in parts:
        return None
    pdf_idx = parts.index("pdf_image")
    return Path(*parts[:pdf_idx])


def dropped_evidence_page(record: dict[str, Any], reference_dir: str | Path | None) -> dict[str, Any] | None:
    if reference_dir is None:
        return None
    qid = question_id(record)
    target = dropped_target(qid)
    if not target:
        return None
    rows = _load_reference_rows(str(Path(reference_dir).resolve()))
    row = rows.get(base_question_id(qid))
    if not isinstance(row, dict):
        return None
    visuals = row.get("visuals")
    if not isinstance(visuals, dict):
        return None
    target_info = visuals.get(target)
    if not isinstance(target_info, dict):
        return None
    visual_box = target_info.get("visual") if isinstance(target_info.get("visual"), dict) else None
    caption_box = target_info.get("caption") if isinstance(target_info.get("caption"), dict) else None
    page_id = None
    if isinstance(visual_box, dict) and visual_box.get("page_id") is not None:
        page_id = int(visual_box["page_id"])
    elif isinstance(caption_box, dict) and caption_box.get("page_id") is not None:
        page_id = int(caption_box["page_id"])
    if page_id is None:
        return None
    doc_id = row.get("document_id")
    if not isinstance(doc_id, str) or not doc_id:
        return None
    image_root = _infer_reference_image_root(record)
    if image_root is None:
        return None
    page_path = image_root / "pdf_image" / Path(doc_id) / f"{page_id:06d}.png"
    if not page_path.exists():
        return None
    return {
        "target": target,
        "page_id": page_id,
        "page_path": page_path,
        "visual_bbox": visual_box.get("bbox") if isinstance(visual_box, dict) else None,
        "caption_bbox": caption_box.get("bbox") if isinstance(caption_box, dict) else None,
    }


def render_dropped_evidence_page(
    record: dict[str, Any],
    reference_dir: str | Path | None,
    max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
) -> tuple[dict[str, Any], Image.Image] | None:
    info = dropped_evidence_page(record, reference_dir)
    if info is None:
        return None
    image = Image.open(info["page_path"]).convert("RGB")
    return info, _resize_for_display(image, max_image_side)


def _render_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    if message.get("role") == "assistant":
        think = str(content.get("think") or "").strip()
        answer = str(content.get("answer") or "").strip()
        tool_call = content.get("tool_call")
        parts: list[str] = []
        if think:
            parts.append(f"Think:\n{think}")
        if isinstance(tool_call, dict):
            parts.append(
                "Tool call:\n"
                + json.dumps(
                    {
                        "img_idx": tool_call.get("img_idx"),
                        "region_description": tool_call.get("region_description"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if answer:
            parts.append(f"Answer:\n{answer}")
        return "\n\n".join(parts) if parts else json.dumps(content, ensure_ascii=False, indent=2)
    if "question" in content:
        return str(content.get("question") or "")
    if "hint" in content:
        return str(content.get("hint") or "")
    return json.dumps(content, ensure_ascii=False, indent=2)


def build_case_index(original_dir: str | Path, rewritten_dir: str | Path) -> pd.DataFrame:
    original_dir = Path(original_dir)
    rewritten_dir = Path(rewritten_dir)
    rows: list[dict[str, Any]] = []
    for rewritten_path in sorted(rewritten_dir.glob("*.json")):
        record = load_record(rewritten_path)
        metadata = record.get(REWRITE_METADATA_KEY)
        if not isinstance(metadata, dict) or not metadata.get("rewrite_safe"):
            continue
        original_path = original_dir / rewritten_path.name
        if not original_path.exists():
            continue
        original_record = load_record(original_path)
        extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
        rows.append(
            {
                "file_name": rewritten_path.name,
                "question_id": question_id(record),
                "document_id": str(extra.get("document_id") or ""),
                "subset": str(extra.get("subset") or ""),
                "tool_calls": sum(
                    1
                    for message in record.get("conversation", [])
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and isinstance(message.get("content"), dict)
                    and isinstance(message["content"].get("tool_call"), dict)
                ),
                "original_final_answer": final_answer_text(original_record),
                "rewritten_final_answer": final_answer_text(record),
            }
        )
    return pd.DataFrame(rows)


def show_case(
    original_dir: str | Path,
    rewritten_dir: str | Path,
    file_name: str,
    reference_dir: str | Path | None = None,
    max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
) -> None:
    original_path = Path(original_dir) / file_name
    rewritten_path = Path(rewritten_dir) / file_name
    original_record = load_record(original_path)
    rewritten_record = load_record(rewritten_path)
    metadata = rewritten_record.get(REWRITE_METADATA_KEY) or {}
    extra = rewritten_record.get("extra_info") if isinstance(rewritten_record.get("extra_info"), dict) else {}
    reward = rewritten_record.get("reward") if isinstance(rewritten_record.get("reward"), dict) else {}

    display(
        Markdown(
            "\n".join(
                [
                    f"### {file_name}",
                    f"- `question_id`: `{question_id(rewritten_record)}`",
                    f"- `document_id`: `{extra.get('document_id', '')}`",
                    f"- `subset`: `{extra.get('subset', '')}`",
                    f"- `question`: {extra.get('question', '')}",
                    f"- `ground_truth`: {reward.get('ground_truth', '')}",
                    f"- `reference_answer`: {metadata.get('reference_answer', '')}",
                ]
            )
        )
    )

    table_html = f"""
    <table style="width:100%; border-collapse:collapse;">
      <tr>
        <th style="text-align:left; border-bottom:1px solid #ccc; width:50%;">Original final answer</th>
        <th style="text-align:left; border-bottom:1px solid #ccc; width:50%;">Rewritten final answer</th>
      </tr>
      <tr>
        <td style="vertical-align:top; white-space:pre-wrap; padding:8px;">{html.escape(final_answer_text(original_record))}</td>
        <td style="vertical-align:top; white-space:pre-wrap; padding:8px;">{html.escape(final_answer_text(rewritten_record))}</td>
      </tr>
    </table>
    """
    display(HTML(table_html))

    evidence = render_dropped_evidence_page(rewritten_record, reference_dir, max_image_side=max_image_side)
    if evidence is not None:
        info, image = evidence
        display(
            Markdown(
                "\n".join(
                    [
                        "#### Dropped Evidence Page",
                        f"- target: `{info['target']}`",
                        f"- page_id: `{info['page_id']}`",
                        f"- source image: `{info['page_path']}`",
                    ]
                )
            )
        )
        display(image)
    elif reference_dir is not None:
        display(Markdown("#### Dropped Evidence Page\nNo recoverable dropped-evidence page metadata for this case."))

    conversation = original_record.get("conversation") or []
    for idx, message in enumerate(conversation):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        if role == "assistant" and idx == final_answer_payload(original_record)[0]:
            display(Markdown("#### Final assistant turn"))
            continue
        display(Markdown(f"#### Turn {idx} `{role}`"))
        display(Markdown(f"```text\n{_render_message_text(message)}\n```"))
        content = message.get("content")
        if isinstance(content, dict):
            indices = content.get("presented_img_indices")
            if isinstance(indices, list) and indices:
                for presented_idx in indices:
                    try:
                        img = rendered_presented_image(original_record, int(presented_idx))
                    except Exception as exc:
                        display(Markdown(f"- image `{presented_idx}` unavailable: `{exc}`"))
                        continue
                    if max_image_side is not None:
                        img = _resize_for_display(img, max_image_side)
                    display(Markdown(f"- image `{presented_idx}`"))
                    display(img)

    display(Markdown("#### Final assistant turn"))
    display(Markdown("**Original**"))
    display(Markdown(f"```text\n{_render_message_text(original_record['conversation'][final_answer_payload(original_record)[0]])}\n```"))
    display(Markdown("**Rewritten**"))
    display(Markdown(f"```text\n{_render_message_text(rewritten_record['conversation'][final_answer_payload(rewritten_record)[0]])}\n```"))
