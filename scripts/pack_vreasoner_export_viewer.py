#!/usr/bin/env python3
"""Pack exported VReasoner/Insight conversations into a portable notebook bundle.

The exported conversation JSONs usually contain image references pointing at
absolute paths on the cluster. This script copies only the locally referenced
source images, rewrites those references to bundle-relative paths, and writes a
small standalone viewer notebook plus helper module. The resulting bundle does
not require the full verl checkout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PORTABLE_VIEWER_PY = r'''"""Standalone viewer for packed VReasoner/Insight exported conversations."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
from IPython.display import Markdown, Pretty, display
from PIL import Image


BUNDLE_ROOT = Path(__file__).resolve().parent


def load_exported_conversation(export_path):
    with open(export_path, encoding="utf-8") as f:
        return json.load(f)


def _is_url(value):
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _resolve_local_path(value, bundle_root=BUNDLE_ROOT):
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.scheme == "file":
            value = parsed.path
        else:
            value = value[len("file://") :]
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = bundle_root / path
    if candidate.exists():
        return candidate
    return path


def _load_image_from_input_ref(ref, bundle_root=BUNDLE_ROOT):
    if ref is None:
        return None
    source_type = ref.get("source_type")

    if source_type == "data_url":
        url = ref.get("url") or ref.get("value")
        if not url or "base64," not in url:
            return None
        _, b64data = url.split("base64,", 1)
        return Image.open(BytesIO(base64.b64decode(b64data))).copy()

    if source_type in ("url", "image_url"):
        value = ref.get("url") or ref.get("value")
        if not value:
            return None
        if _is_url(value):
            with urlopen(value) as response:
                return Image.open(BytesIO(response.read())).copy()
        path = _resolve_local_path(value, bundle_root=bundle_root)
        return Image.open(path).copy() if path else None

    if source_type in ("path", "path_or_url"):
        value = ref.get("path") or ref.get("value")
        if not value:
            return None
        if _is_url(value):
            with urlopen(value) as response:
                return Image.open(BytesIO(response.read())).copy()
        path = _resolve_local_path(value, bundle_root=bundle_root)
        return Image.open(path).copy() if path else None

    value = ref.get("value") or ref.get("path") or ref.get("url")
    if isinstance(value, str) and value:
        if _is_url(value):
            with urlopen(value) as response:
                return Image.open(BytesIO(response.read())).copy()
        path = _resolve_local_path(value, bundle_root=bundle_root)
        return Image.open(path).copy() if path else None
    return None


def _render_presented_image(original_images, presented_ref):
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


def restore_presented_images(record, bundle_root=BUNDLE_ROOT):
    input_refs = record.get("image_references", {}).get("input_images", [])
    original_images = [_load_image_from_input_ref(ref, bundle_root=bundle_root) for ref in input_refs]

    restored = []
    for presented_ref in sorted(
        record.get("image_references", {}).get("presented_images", []),
        key=lambda item: item.get("presented_img_idx", -1),
    ):
        restored.append({**presented_ref, "image": _render_presented_image(original_images, presented_ref)})
    return restored


def _assistant_message_to_text(message):
    content = message.get("content", {})
    if message.get("type") == "tool_call":
        payload = content.get("tool_call")
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return f"<think>{content.get('think', '')}</think>\n<tool_call>{payload}</tool_call>"
    if message.get("type") in ("answer", "answer_revision"):
        return f"<think>{content.get('think', '')}</think>\n<answer>{content.get('answer', '')}</answer>"
    return content.get("text", "")


def _user_message_to_contents(message):
    parts = message.get("parts")
    if isinstance(parts, list):
        contents = []
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


def restore_exported_conversation(record_or_path, bundle_root=BUNDLE_ROOT):
    record = load_exported_conversation(record_or_path) if isinstance(record_or_path, (str, Path)) else record_or_path
    presented_images = restore_presented_images(record, bundle_root=bundle_root)
    presented_idx_to_image = {item.get("presented_img_idx"): item.get("image") for item in presented_images}

    messages = []
    for message in record.get("conversation", []):
        role = message.get("role")
        if role == "system":
            content = message.get("content", {})
            messages.append({"role": "system", "content": content.get("text", "") if isinstance(content, dict) else content})
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

    return {"record": record, "messages": messages, "presented_images": presented_images, "reward": record.get("reward")}


def restore_conversation_for_visualization(record_or_path, bundle_root=BUNDLE_ROOT):
    restored = restore_exported_conversation(record_or_path, bundle_root=bundle_root)
    ordered_images = []
    notebook_messages = []

    for message in restored["messages"]:
        if message["role"] != "user" or not isinstance(message["content"], list):
            notebook_messages.append(message)
            continue

        contents = []
        for content in message["content"]:
            if content.get("type") == "image":
                ordered_images.append(content.get("image"))
                contents.append({"type": "image"})
            else:
                contents.append(content)
        notebook_messages.append({"role": message["role"], "content": contents})

    return {**restored, "messages": notebook_messages, "multi_modal_data": {"images": ordered_images}}


def load_records(export_dir=BUNDLE_ROOT / "exported_conversations"):
    export_dir = Path(export_dir)
    export_paths = sorted(export_dir.rglob("*.json"))
    records = [load_exported_conversation(path) for path in export_paths]
    return export_paths, records


def _reward_value(record, key):
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return None
    return reward.get(key)


def build_summary_df(export_paths, records):
    rows = []
    for idx, (path, record) in enumerate(zip(export_paths, records)):
        extra = record.get("extra_info") or {}
        rows.append(
            {
                "idx": idx,
                "subset": extra.get("subset"),
                "file": Path(path).name,
                "job_id": record.get("job_id"),
                "agent_name": record.get("agent_name"),
                "question_id": extra.get("question_id"),
                "question": extra.get("question"),
                "critical_failure": record.get("critical_failure"),
                "reward": record.get("reward"),
                "score": _reward_value(record, "score"),
                "format_reward": _reward_value(record, "format_reward"),
                "accuracy_reward": _reward_value(record, "accuracy_reward"),
                "n_valid_tool_calls": _reward_value(record, "n_valid_tool_calls"),
                "extracted_answer": record.get("extracted_answer"),
                "ground_truth": record.get("ground_truth"),
                "question_type": extra.get("question_type"),
            }
        )
    return pd.DataFrame(rows)


def display_restored_conversation(restored_payload, *, omit_system_prompt=False):
    images = list(restored_payload.get("multi_modal_data", {}).get("images", []))
    image_idx = 0
    for msg_idx, message in enumerate(restored_payload["messages"]):
        role = message.get("role")
        if omit_system_prompt and role == "system":
            continue
        display(Markdown(f"### {msg_idx}: {role}"))
        content = message.get("content")
        if isinstance(content, str):
            display(Markdown("```text\n" + content.replace("```", "~~~") + "\n```"))
            continue
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    display(Markdown("```text\n" + str(part.get("text", "")).replace("```", "~~~") + "\n```"))
                elif part.get("type") == "image":
                    image = images[image_idx] if image_idx < len(images) else None
                    image_idx += 1
                    if image is None:
                        display(Markdown("`[image unavailable from reference]`"))
                    else:
                        display(image)
                else:
                    display(Pretty(part))
            continue
        display(Pretty(content))


def display_conversation_by_index(idx, export_paths, records, summary_df=None, *, omit_system_prompt=False, bundle_root=BUNDLE_ROOT):
    if summary_df is not None:
        display(summary_df.iloc[[idx]])
    display(Markdown(f"**File:** `{Path(export_paths[idx]).name}`"))
    restored = restore_conversation_for_visualization(records[idx], bundle_root=bundle_root)
    display_restored_conversation(restored, omit_system_prompt=omit_system_prompt)
    return restored
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, help="Directory containing exported conversation JSON files.")
    parser.add_argument("--output-dir", required=True, help="Destination bundle directory. Must not be non-empty.")
    parser.add_argument("--output-tar", help="Optional .tar.gz path to create after writing the bundle directory.")
    parser.add_argument("--max-records", type=int, help="Optional cap for smoke-testing a subset.")
    parser.add_argument("--skip-missing-images", action="store_true", help="Keep missing image refs instead of failing.")
    return parser.parse_args()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_http_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def ref_value(ref: dict) -> tuple[str | None, str | None]:
    source_type = ref.get("source_type")
    if source_type == "path":
        return source_type, ref.get("path")
    if source_type in ("path_or_url", "image_url"):
        return source_type, ref.get("value") or ref.get("url")
    if source_type in ("url", "data_url"):
        return source_type, ref.get("url") or ref.get("value")
    return source_type, ref.get("value") or ref.get("path") or ref.get("url")


def local_path_from_ref(ref: dict) -> Path | None:
    _source_type, value = ref_value(ref)
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:") or is_http_url(value):
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        value = parsed.path if parsed.scheme == "file" else value[len("file://") :]
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return path.resolve()


def target_image_path(src: Path, image_dir: Path) -> tuple[Path, str]:
    digest = sha1_file(src)[:16]
    suffix = src.suffix or ".img"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in src.stem)[:96] or "image"
    filename = f"{digest}_{safe_stem}{suffix}"
    dst = image_dir / filename
    return dst, f"images/{filename}"


def copy_and_rewrite_record(
    record: dict,
    *,
    image_dir: Path,
    image_map: dict[str, str],
    missing_images: list[dict],
    nonlocal_refs: list[dict],
    json_name: str,
    skip_missing_images: bool,
) -> dict:
    rewritten = copy.deepcopy(record)
    input_refs = rewritten.get("image_references", {}).get("input_images", [])
    if not isinstance(input_refs, list):
        return rewritten

    for idx, ref in enumerate(input_refs):
        if not isinstance(ref, dict):
            continue
        source_type, value = ref_value(ref)
        if source_type == "data_url" or (isinstance(value, str) and value.startswith("data:")):
            continue
        if is_http_url(value):
            nonlocal_refs.append({"json": json_name, "input_image_idx": idx, "source_type": source_type, "value": value})
            continue

        src = local_path_from_ref(ref)
        if src is None:
            nonlocal_refs.append({"json": json_name, "input_image_idx": idx, "source_type": source_type, "value": value})
            continue
        if not src.exists():
            item = {"json": json_name, "input_image_idx": idx, "source_type": source_type, "value": value, "path": str(src)}
            missing_images.append(item)
            if not skip_missing_images:
                raise FileNotFoundError(f"Missing image for {json_name} input_image_idx={idx}: {src}")
            continue

        src_key = str(src)
        if src_key not in image_map:
            dst, rel = target_image_path(src, image_dir)
            if not dst.exists():
                shutil.copy2(src, dst)
            image_map[src_key] = rel

        ref.clear()
        ref.update(
            {
                "source_type": "path_or_url",
                "value": image_map[src_key],
                "packed_original_source_type": source_type,
                "packed_original_value": value,
            }
        )
    return rewritten


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_notebook(path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Portable VReasoner/Insight Export Viewer\n\nThis notebook is generated by `pack_vreasoner_export_viewer.py` and expects to run from the bundle directory.",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "from pathlib import Path\nfrom portable_vreasoner_export_viewer import load_records, build_summary_df, display_conversation_by_index\n\nEXPORT_DIR = Path('exported_conversations')\nexport_paths, records = load_records(EXPORT_DIR)\nsummary_df = build_summary_df(export_paths, records)\nlen(records)",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "summary_df",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "summary_df[['subset', 'critical_failure', 'accuracy_reward', 'n_valid_tool_calls']].describe(include='all')",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "idx = 0\nrestored = display_conversation_by_index(idx, export_paths, records, summary_df, omit_system_prompt=False)",
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(path: Path, *, export_dir: Path, num_records: int, num_images: int, output_tar: str | None) -> None:
    tar_line = f"\nTransfer `{output_tar}` and unpack it with `tar -xzf {Path(output_tar).name}`." if output_tar else ""
    write_text(
        path,
        f"""# Portable Conversation Export Viewer

Source export directory:

```text
{export_dir}
```

Packed records: {num_records}

Packed local source images: {num_images}
{tar_line}

## Usage on another machine

```bash
cd <bundle-dir>
python -m pip install -r requirements.txt
jupyter lab visualize_vreasoner_v2_export_portable.ipynb
```

The notebook uses only files in this bundle. Crops are reconstructed from the
packed source images and the `image_references.presented_images` metadata in the
JSON conversations.
""",
    )


def create_tarball(bundle_dir: Path, output_tar: Path) -> None:
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not export_dir.is_dir():
        raise NotADirectoryError(export_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    conversation_dir = output_dir / "exported_conversations"
    image_dir = output_dir / "images"
    conversation_dir.mkdir()
    image_dir.mkdir()

    export_paths = sorted(export_dir.rglob("*.json"))
    if args.max_records is not None:
        export_paths = export_paths[: args.max_records]

    image_map: dict[str, str] = {}
    missing_images: list[dict] = []
    nonlocal_refs: list[dict] = []
    copied_records = 0

    for src_json in export_paths:
        rel_json = src_json.relative_to(export_dir)
        dst_json = conversation_dir / rel_json
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        with src_json.open(encoding="utf-8") as f:
            record = json.load(f)
        rewritten = copy_and_rewrite_record(
            record,
            image_dir=image_dir,
            image_map=image_map,
            missing_images=missing_images,
            nonlocal_refs=nonlocal_refs,
            json_name=str(rel_json),
            skip_missing_images=args.skip_missing_images,
        )
        dst_json.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied_records += 1

    write_text(output_dir / "portable_vreasoner_export_viewer.py", PORTABLE_VIEWER_PY)
    write_notebook(output_dir / "visualize_vreasoner_v2_export_portable.ipynb")
    write_text(output_dir / "requirements.txt", "pillow\npandas\nipython\njupyter\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_export_dir": str(export_dir),
        "num_records": copied_records,
        "num_local_images_copied": len(image_map),
        "image_map": image_map,
        "missing_images": missing_images,
        "nonlocal_refs_kept": nonlocal_refs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_tar = None
    if args.output_tar:
        output_tar = str(Path(args.output_tar).expanduser().resolve())
        create_tarball(output_dir, Path(output_tar))

    write_readme(
        output_dir / "README.md",
        export_dir=export_dir,
        num_records=copied_records,
        num_images=len(image_map),
        output_tar=output_tar,
    )

    print(f"Wrote bundle: {output_dir}")
    print(f"Records: {copied_records}")
    print(f"Local source images copied: {len(image_map)}")
    if missing_images:
        print(f"Missing images kept unresolved: {len(missing_images)}")
    if nonlocal_refs:
        print(f"Nonlocal/data refs kept as-is: {len(nonlocal_refs)}")
    if output_tar:
        print(f"Wrote tarball: {output_tar}")


if __name__ == "__main__":
    main()
