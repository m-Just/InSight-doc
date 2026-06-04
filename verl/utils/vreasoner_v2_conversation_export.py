import base64
import hashlib
import json
import os
import re
import socket
import tempfile
from io import BytesIO
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.request import urlopen

from PIL import Image

import verl.utils.vreasoner_v2_prompt as prompts


_EXPORT_ID_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify_export_component(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    text = _EXPORT_ID_SLUG_RE.sub("-", text).strip("._-")
    return text[:96] or fallback


def build_root_conversation_export_id(
    *,
    extra_info: Mapping[str, Any] | None,
    data_source: Any = None,
    validate: bool,
    val_trial_idx: int | None = None,
) -> str:
    extra_info = extra_info or {}
    question_id = extra_info.get("question_id")
    if question_id is None or not str(question_id).strip():
        raise ValueError("conversation export resume requires extra_info['question_id']")
    identity_payload = _json_safe(
        {
            "question_id": question_id,
            "data_source": data_source,
            "validate": validate,
            "val_trial_idx": val_trial_idx if validate else None,
        }
    )
    digest = hashlib.sha1(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    parts = [
        _slugify_export_component(data_source, fallback="unknown"),
        "val" if validate else "train",
    ]
    if validate and val_trial_idx is not None:
        parts.append(f"trial{int(val_trial_idx)}")
    parts.append(_slugify_export_component(question_id, fallback="question"))
    return "-".join(parts + [digest])


def build_repeated_conversation_export_id(base_export_id: str, repeat_idx: int) -> str:
    if repeat_idx < 0:
        raise ValueError(f"repeat_idx must be non-negative, got {repeat_idx}")
    if repeat_idx == 0:
        return base_export_id
    return f"{base_export_id}--repeat{repeat_idx}"


def build_child_conversation_export_id(parent_export_id: str, child_idx: int) -> str:
    if child_idx < 0:
        raise ValueError(f"child_idx must be non-negative, got {child_idx}")
    return f"{parent_export_id}--child{child_idx}"


def build_conversation_export_path(export_dir: str, export_id: str) -> str:
    return os.path.join(export_dir, f"{export_id}.json")


def build_conversation_export_index_path(export_dir: str, *, global_step: Any, split: str) -> str:
    step_component = f"global_step_{_slugify_export_component(global_step, fallback='unknown')}"
    split_component = _slugify_export_component(split, fallback="unknown")
    hostname = _slugify_export_component(socket.gethostname(), fallback="unknown-host")
    pid = os.getpid()
    return os.path.join(
        export_dir,
        "index",
        step_component,
        split_component,
        f"worker_{hostname}_{pid}.jsonl",
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Image.Image):
        return {
            "__type__": "PIL.Image",
            "size": list(value.size),
            "mode": value.mode,
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _text_after_last_label(parts: list[dict[str, Any]]) -> str:
    text_fragments: list[str] = []
    for part in parts:
        if part["kind"] == "text":
            text_fragments.append(part["text"])
    return "".join(text_fragments).strip()


def _extract_tag_content(text: str, tag: str) -> str | None:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
        return None
    return text.rsplit(start_tag, 1)[1].split(end_tag, 1)[0].strip()


def _extract_tag_counts(text: str, tag: str) -> dict[str, int]:
    return {
        "open": text.count(f"<{tag}>"),
        "close": text.count(f"</{tag}>"),
    }


def _extract_unwrapped_prefix_as_think(text: str, terminal_tag: str) -> str | None:
    start_tag = f"<{terminal_tag}>"
    if start_tag not in text:
        return None
    prefix = text.split(start_tag, 1)[0].strip()
    if not prefix:
        return None
    # Only synthesize think when the prefix contains no think tags at all.
    # If there are any <think> / </think> markers here, they must have been
    # parsed explicitly; otherwise the message is malformed and should stay
    # as "others" rather than being normalized silently.
    if "<think>" in prefix or "</think>" in prefix:
        return None
    return prefix


def parse_assistant_message(text: str) -> dict[str, Any]:
    think = _extract_tag_content(text, "think")
    tool_call = _extract_tag_content(text, "tool_call")
    answer = _extract_tag_content(text, "answer")
    tag_counts = {
        "think": _extract_tag_counts(text, "think"),
        "tool_call": _extract_tag_counts(text, "tool_call"),
        "answer": _extract_tag_counts(text, "answer"),
    }

    parsed_tool_call = None
    tool_call_parse_error = None
    if tool_call is not None:
        try:
            parsed_tool_call = json.loads(tool_call)
        except json.JSONDecodeError as exc:
            tool_call_parse_error = str(exc)

    synthesized_think = think
    if synthesized_think is None:
        if tool_call is not None and answer is None:
            synthesized_think = _extract_unwrapped_prefix_as_think(text, "tool_call")
        elif answer is not None and tool_call is None:
            synthesized_think = _extract_unwrapped_prefix_as_think(text, "answer")

    if synthesized_think is not None and tool_call is not None and answer is None:
        message_type = "tool_call"
        content = {
            "think": synthesized_think,
            "tool_call": parsed_tool_call if parsed_tool_call is not None else tool_call,
        }
    elif synthesized_think is not None and answer is not None and tool_call is None:
        message_type = "answer"
        content = {
            "think": synthesized_think,
            "answer": answer,
        }
    else:
        message_type = "others"
        content = {"text": text}
        extracted_tags = {}
        if think is not None:
            extracted_tags["think"] = think
        if tool_call is not None:
            extracted_tags["tool_call"] = parsed_tool_call if parsed_tool_call is not None else tool_call
        if answer is not None:
            extracted_tags["answer"] = answer
        if extracted_tags:
            content["extracted_tags"] = extracted_tags

    out = {
        "type": message_type,
        "content": content,
        "tag_counts": tag_counts,
    }
    if tool_call_parse_error is not None:
        out["tool_call_parse_error"] = tool_call_parse_error
    return out


def parse_assistant_answer_revision_message(text: str) -> dict[str, Any]:
    answer = _extract_tag_content(text, "answer")
    tag_counts = {
        "think": _extract_tag_counts(text, "think"),
        "tool_call": _extract_tag_counts(text, "tool_call"),
        "answer": _extract_tag_counts(text, "answer"),
    }
    if answer is None:
        return {
            "type": "others",
            "content": {"text": text},
            "tag_counts": tag_counts,
        }
    return {
        "type": "answer_revision",
        "content": {
            "think": "",
            "answer": answer,
        },
        "tag_counts": tag_counts,
    }


def parse_user_message(content: Any, *, initial_question: str) -> dict[str, Any]:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped == prompts.FORMAT_REPAIR_HINT.strip():
            return {"type": "format_repair_hint", "content": {"hint": stripped}}
        if stripped == prompts.LAST_ROUND_HINT.strip():
            return {"type": "last_round_hint", "content": {"hint": stripped}}
        if stripped == initial_question.strip():
            return {"type": "query", "content": {"question": stripped}}
        chunks = content.split("\n\n", 1)
        if len(chunks) == 2 and "Please adjust or refine your region description" in chunks[1]:
            return {
                "type": "tool_result_fail_hint",
                "content": {
                    "error_message": chunks[0].strip(),
                    "hint": chunks[1].strip(),
                },
            }
        return {"type": "others", "content": {"text": content}}

    if not isinstance(content, Sequence):
        return {"type": "others", "content": {"text": content}}

    parts: list[dict[str, Any]] = []
    pending_label: tuple[int, str] | None = None
    for item in content:
        if not isinstance(item, Mapping):
            parts.append({"kind": "item", "item_type": type(item).__name__, "value": _json_safe(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
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
        else:
            parts.append({"kind": "item", "item_type": item_type, "value": _json_safe(item)})

    main_text_parts = [
        part["text"]
        for part in parts
        if part["kind"] == "text" and part["text"].strip() != prompts.LAST_ROUND_HINT.strip()
    ]
    main_text = "".join(main_text_parts).strip()
    image_indices = [part["presented_img_idx"] for part in parts if part["kind"] == "image_ref"]
    secondary_types: list[str] = []
    if any(part["kind"] == "text" and part["text"].strip() == prompts.LAST_ROUND_HINT.strip() for part in parts):
        secondary_types.append("last_round_hint")

    if image_indices and main_text == initial_question.strip():
        message_type = "query"
        parsed = {"question": initial_question.strip(), "presented_img_indices": image_indices}
    elif image_indices:
        message_type = "tool_result"
        parsed = {
            "hint": main_text,
            "presented_img_indices": image_indices,
        }
    elif main_text == prompts.FORMAT_REPAIR_HINT.strip():
        message_type = "format_repair_hint"
        parsed = {"hint": main_text}
    elif main_text == prompts.LAST_ROUND_HINT.strip():
        message_type = "last_round_hint"
        parsed = {"hint": main_text}
    elif "Please adjust or refine your region description" in main_text:
        chunks = main_text.split("\n\n", 1)
        message_type = "tool_result_fail_hint"
        parsed = {
            "error_message": chunks[0].strip(),
            "hint": chunks[1].strip() if len(chunks) > 1 else "",
        }
    else:
        message_type = "others"
        parsed = {"text": _text_after_last_label(parts)}

    out = {
        "type": message_type,
        "content": parsed,
        "parts": parts,
    }
    if secondary_types:
        out["secondary_types"] = secondary_types
    return out


def parse_answer_verification_hint_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"type": "answer_verification_hint", "content": {"hint": content.strip()}}

    parts: list[str] = []
    if isinstance(content, Sequence):
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
    hint = "".join(parts).strip()
    return {"type": "answer_verification_hint", "content": {"hint": hint}}


def build_input_image_refs(
    raw_prompt: list[dict[str, Any]],
    original_images: list[Image.Image],
    preserved_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if preserved_refs:
        refs = [_json_safe(ref) for ref in preserved_refs]
        for idx, ref in enumerate(refs):
            if idx < len(original_images):
                ref.setdefault("original_img_idx", idx)
                ref.setdefault("original_size", list(original_images[idx].size))
        return refs

    refs: list[dict[str, Any]] = []
    for message_idx, message in enumerate(raw_prompt):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_idx, item in enumerate(content):
            item_type = item.get("type")
            if item_type not in ("image", "image_url"):
                continue
            ref: dict[str, Any] = {
                "source_message_idx": message_idx,
                "source_content_idx": content_idx,
                "message_role": message.get("role"),
            }
            if item_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                ref.update({"source_type": "image_url", "value": url})
            else:
                image_value = item.get("image")
                if isinstance(image_value, str):
                    ref.update({"source_type": "path_or_url", "value": image_value})
                else:
                    ref.update({"source_type": "unavailable", "value": None})
            refs.append(ref)

    for idx, ref in enumerate(refs):
        if idx < len(original_images):
            ref["original_size"] = list(original_images[idx].size)
            ref["original_img_idx"] = idx
    return refs


def build_export_conversation(
    messages_api: list[dict[str, Any]],
    *,
    initial_question: str,
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for idx, raw_message in enumerate(messages_api):
        message = raw_message.to_dict() if hasattr(raw_message, "to_dict") else raw_message
        role = message.get("role")
        export_type = message.get("export_type")
        if role == "system":
            exported.append(
                {
                    "message_idx": idx,
                    "role": "system",
                    "type": "system_prompt",
                    "content": {"text": message.get("content", "")},
                }
            )
        elif role == "assistant":
            if export_type == "answer_revision":
                parsed = parse_assistant_answer_revision_message(message.get("content") or "")
            else:
                parsed = parse_assistant_message(message.get("content") or "")
            exported.append({"message_idx": idx, "role": "assistant", **parsed})
        elif role == "user":
            if export_type == "answer_verification_hint":
                parsed = parse_answer_verification_hint_message(message.get("content"))
            else:
                parsed = parse_user_message(message.get("content"), initial_question=initial_question)
            exported.append({"message_idx": idx, "role": "user", **parsed})
        else:
            exported.append(
                {
                    "message_idx": idx,
                    "role": role,
                    "type": "others",
                    "content": {"value": _json_safe(message.get("content"))},
                }
            )
    return exported


def build_export_record(
    *,
    job_id: str,
    parent_job_id: str | None,
    root_job_id: str,
    validate: bool,
    initial_question: str,
    messages_api: list[dict[str, Any]],
    raw_prompt: list[dict[str, Any]],
    original_images: list[Image.Image],
    presented_image_refs: list[dict[str, Any]],
    request_params: dict[str, Any],
    loop_params: dict[str, Any],
    sampling_params: dict[str, Any],
    tools_kwargs: dict[str, Any],
    extra_info: dict[str, Any],
    failure_events: list[dict[str, Any]],
    critical_failure: bool,
    final_failure_reasons: list[str] | None,
) -> dict[str, Any]:
    sanitized_extra_info = {
        key: _json_safe(value)
        for key, value in extra_info.items()
        if key != "image_ori"
    }
    return {
        "schema_version": "vreasoner_v2_conversation_v2",
        "agent_name": "vreasoner_v2",
        "job": {
            "job_id": job_id,
            "parent_job_id": parent_job_id,
            "root_job_id": root_job_id,
            "validate": validate,
        },
        "status": {
            "critical_failure": critical_failure,
            "final_failure_reasons": _json_safe(final_failure_reasons),
        },
        "parameters": {
            "loop": _json_safe(loop_params),
            "request": _json_safe(request_params),
            "sampling": _json_safe(sampling_params),
        },
        "tools_kwargs": _json_safe(tools_kwargs),
        "extra_info": sanitized_extra_info,
        "image_references": {
            "input_images": build_input_image_refs(
                raw_prompt,
                original_images,
                preserved_refs=extra_info.get("original_image_refs"),
            ),
            "presented_images": _json_safe(presented_image_refs),
        },
        "conversation": build_export_conversation(messages_api, initial_question=initial_question),
        "failures": _json_safe(failure_events),
        "reward": None,
    }


def write_json_atomic(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory or None, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_conversation_export_index(
    export_dir: str,
    *,
    export_id: str,
    export_path: str,
    record: dict[str, Any],
    index_metadata: Mapping[str, Any] | None = None,
) -> str:
    metadata = dict(index_metadata or {})
    job = record.get("job") if isinstance(record, Mapping) else {}
    if isinstance(job, Mapping):
        metadata = {**job, **metadata}
    extra_info = record.get("extra_info") if isinstance(record, Mapping) else {}
    if not isinstance(extra_info, Mapping):
        extra_info = {}

    validate = bool(metadata.get("validate", False))
    split = str(metadata.get("split") or ("val" if validate else "train"))
    global_step = metadata.get("global_step", metadata.get("step"))
    index_path = build_conversation_export_index_path(export_dir, global_step=global_step, split=split)
    index_dir = os.path.dirname(index_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    entry = {
        "schema_version": "conversation_export_index_v1",
        "export_id": export_id,
        "path": os.path.abspath(export_path),
        "relative_path": os.path.relpath(export_path, export_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_step": _json_safe(global_step),
        "split": split,
        "validate": validate,
        "worker": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
        "job_id": metadata.get("job_id"),
        "parent_job_id": metadata.get("parent_job_id"),
        "root_job_id": metadata.get("root_job_id"),
        "run_name": metadata.get("run_name"),
        "trial_name": metadata.get("trial_name"),
        "trajectory_sample_index": metadata.get("trajectory_sample_index"),
        "rollout_n": metadata.get("rollout_n"),
        "data_source": extra_info.get("data_source"),
        "question_id": extra_info.get("question_id"),
        "document_id": extra_info.get("document_id"),
        "index": extra_info.get("index"),
        "subset": extra_info.get("subset"),
        "conversation_export_repeat_idx": extra_info.get("conversation_export_repeat_idx"),
    }
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(entry), ensure_ascii=False, sort_keys=True))
        f.write("\n")
    return index_path


def is_conversation_export_complete(export_path: str) -> bool:
    if not export_path or not os.path.exists(export_path):
        return False
    try:
        with open(export_path, encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return False
    reward = record.get("reward")
    if reward is None:
        return False
    if reward.get("compute_score_success") is True:
        return True
    score = reward.get("score")
    if isinstance(score, dict) and score.get("compute_score_success") is True:
        return True
    return False


def export_conversation(
    export_dir: str,
    record: dict[str, Any],
    *,
    job_id: str,
    export_id: str | None = None,
    index_metadata: Mapping[str, Any] | None = None,
) -> str:
    actual_export_id = export_id or job_id
    path = build_conversation_export_path(export_dir, actual_export_id)
    write_json_atomic(path, record)
    write_conversation_export_index(
        export_dir,
        export_id=actual_export_id,
        export_path=path,
        record=record,
        index_metadata=index_metadata,
    )
    return path


def append_reward_info(export_path: str, reward_info: dict[str, Any]) -> None:
    if not export_path:
        return
    with open(export_path, encoding="utf-8") as f:
        record = json.load(f)
    record["reward"] = _json_safe(reward_info)
    write_json_atomic(export_path, record)


def load_exported_conversation(export_path: str) -> dict[str, Any]:
    with open(export_path, encoding="utf-8") as f:
        return json.load(f)


def _load_image_from_input_ref(ref: dict[str, Any] | None) -> Image.Image | None:
    if ref is None:
        return None
    source_type = ref.get("source_type")
    if source_type == "path":
        path = ref.get("path")
        if not path:
            return None
        return Image.open(path).copy()
    if source_type == "url":
        url = ref.get("url")
        if not url:
            return None
        with urlopen(url) as response:
            return Image.open(BytesIO(response.read())).copy()
    if source_type == "data_url":
        url = ref.get("url")
        if not url or "base64," not in url:
            return None
        _, b64data = url.split("base64,", 1)
        return Image.open(BytesIO(base64.b64decode(b64data))).copy()
    if source_type == "path_or_url":
        value = ref.get("value")
        if not value:
            return None
        if isinstance(value, str) and value.startswith("file://"):
            path = value[len("file://") :]
            return Image.open(path).copy()
        if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
            with urlopen(value) as response:
                return Image.open(BytesIO(response.read())).copy()
        if isinstance(value, str):
            return Image.open(value).copy()
    return None


def _render_presented_image(
    original_images: list[Image.Image | None],
    presented_ref: dict[str, Any],
) -> Image.Image | None:
    source_idx = presented_ref.get("source_original_img_idx")
    if not isinstance(source_idx, int) or not (0 <= source_idx < len(original_images)):
        return None
    source_image = original_images[source_idx]
    if source_image is None:
        return None

    bbox = presented_ref.get("bbox_on_original")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = map(int, bbox)
    cropped = source_image.crop((x1, y1, x2, y2))

    display_size = presented_ref.get("display_size")
    if isinstance(display_size, list) and len(display_size) == 2:
        target_size = (int(display_size[0]), int(display_size[1]))
        if cropped.size != target_size:
            cropped = cropped.resize(target_size, Image.LANCZOS)
    return cropped


def restore_presented_images(record: dict[str, Any]) -> list[dict[str, Any]]:
    input_refs = record.get("image_references", {}).get("input_images", [])
    original_images = [_load_image_from_input_ref(ref) for ref in input_refs]

    restored: list[dict[str, Any]] = []
    for presented_ref in sorted(
        record.get("image_references", {}).get("presented_images", []),
        key=lambda item: item.get("presented_img_idx", -1),
    ):
        restored.append(
            {
                **presented_ref,
                "image": _render_presented_image(original_images, presented_ref),
            }
        )
    return restored


def _assistant_message_to_text(message: dict[str, Any]) -> str:
    content = message.get("content", {})
    if message.get("type") == "tool_call":
        payload = content.get("tool_call")
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return f"<think>{content.get('think', '')}</think>\n<tool_call>{payload}</tool_call>"
    if message.get("type") in ("answer", "answer_revision"):
        return f"<think>{content.get('think', '')}</think>\n<answer>{content.get('answer', '')}</answer>"
    return content.get("text", "")


def _user_message_to_contents(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    if isinstance(parts, list):
        contents: list[dict[str, Any]] = []
        for part in parts:
            kind = part.get("kind")
            if kind in ("text", "separator"):
                contents.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_ref":
                label = part.get("label")
                if label:
                    contents.append({"type": "text", "text": f"{label}:"})
                contents.append({"type": "image", "presented_img_idx": part.get("presented_img_idx")})
        if contents:
            return contents

    content = message.get("content", {})
    if message.get("type") == "query":
        return [{"type": "text", "text": content.get("question", "")}]
    if message.get("type") == "tool_result":
        return [{"type": "text", "text": content.get("hint", "")}]
    if message.get("type") == "tool_result_fail_hint":
        return [{"type": "text", "text": f"{content.get('error_message', '')}\n\n{content.get('hint', '')}".strip()}]
    if message.get("type") in ("format_repair_hint", "last_round_hint", "answer_verification_hint"):
        return [{"type": "text", "text": content.get("hint", "")}]
    return [{"type": "text", "text": content.get("text", "")}]


def restore_exported_conversation(record_or_path: dict[str, Any] | str) -> dict[str, Any]:
    record = load_exported_conversation(record_or_path) if isinstance(record_or_path, str) else record_or_path
    presented_images = restore_presented_images(record)
    presented_idx_to_image = {
        item.get("presented_img_idx"): item.get("image")
        for item in presented_images
    }

    messages: list[dict[str, Any]] = []
    for message in record.get("conversation", []):
        role = message.get("role")
        if role == "system":
            messages.append({"role": "system", "content": message.get("content", {}).get("text", "")})
            continue
        if role == "assistant":
            messages.append({"role": "assistant", "content": _assistant_message_to_text(message)})
            continue
        if role == "user":
            restored_contents = []
            for content in _user_message_to_contents(message):
                if content.get("type") == "image":
                    restored_contents.append(
                        {
                            "type": "image",
                            "image": presented_idx_to_image.get(content.get("presented_img_idx")),
                            "presented_img_idx": content.get("presented_img_idx"),
                        }
                    )
                else:
                    restored_contents.append(content)
            messages.append({"role": "user", "content": restored_contents})
            continue
        messages.append({"role": role, "content": message.get("content", {})})

    return {
        "record": record,
        "messages": messages,
        "presented_images": presented_images,
        "reward": record.get("reward"),
    }


def restore_conversation_for_visualization(record_or_path: dict[str, Any] | str) -> dict[str, Any]:
    restored = restore_exported_conversation(record_or_path)
    ordered_images: list[Image.Image | None] = []
    notebook_messages: list[dict[str, Any]] = []

    for message in restored["messages"]:
        if message["role"] != "user" or not isinstance(message["content"], list):
            notebook_messages.append(message)
            continue

        contents: list[dict[str, Any]] = []
        for content in message["content"]:
            if content.get("type") == "image":
                ordered_images.append(content.get("image"))
                contents.append({"type": "image"})
            else:
                contents.append(content)
        notebook_messages.append({"role": message["role"], "content": contents})

    return {
        **restored,
        "messages": notebook_messages,
        "multi_modal_data": {"images": ordered_images},
    }
