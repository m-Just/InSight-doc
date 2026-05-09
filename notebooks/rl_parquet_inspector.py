from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.vreasoner_v2_conversation_export import (
    load_exported_conversation,
    restore_conversation_for_visualization,
)


ARXIV_POSTPROCESS_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess")
O3_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
GENERATED_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_python(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    return value


def _maybe_parse_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    if value[0] in "[{":
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value)
            except Exception:
                continue
    return value


def load_rl_parquet(parquet_path: str | Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).expanduser().resolve()
    df = pd.read_parquet(parquet_path)
    return df


def dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        sample = None
        if len(df) > 0:
            sample = type(df.iloc[0][col]).__name__
        rows.append({"column": col, "dtype": str(df[col].dtype), "sample_type": sample})
    return pd.DataFrame(rows)


def normalize_row(row_like: pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = dict(row_like)
    row = _to_python(row)
    if isinstance(row.get("extra_info"), dict):
        for key in ("question_involved_visuals", "question_involved_visual_details", "question_type"):
            row["extra_info"][key] = _maybe_parse_serialized(row["extra_info"].get(key))
    return row


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info") or {}
    reward_model = row.get("reward_model") or {}
    relevant_page_ids = extract_relevant_page_ids(row)
    return {
        "question_id": extra.get("question_id"),
        "subset": extra.get("subset"),
        "question_type": extra.get("question_type"),
        "document_id": extra.get("document_id"),
        "ground_truth": reward_model.get("ground_truth"),
        "n_images": len(extract_image_paths(row)),
        "relevant_page_ids": relevant_page_ids if relevant_page_ids else None,
    }


def extract_image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images") or []
    out = []
    for item in images:
        if isinstance(item, dict) and "image" in item:
            out.append(str(item["image"]))
        else:
            out.append(str(item))
    return out


def _strip_file_uri(path_or_uri: str) -> Path:
    if path_or_uri.startswith("file://"):
        return Path(path_or_uri[7:])
    return Path(path_or_uri)


def _page_id_from_image_path(path_or_uri: str) -> int | None:
    path = _strip_file_uri(path_or_uri)
    try:
        return int(path.stem)
    except ValueError:
        return None


def original_manifest_specs() -> list[tuple[str, Path, Path]]:
    specs = [
        (
            "O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            O3_ROOT / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        (
            "O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            O3_ROOT / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        (
            "O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            O3_ROOT / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        (
            "O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            O3_ROOT / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        (
            "O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40",
            O3_ROOT / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            O3_ROOT / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        (
            "arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40",
            ARXIV_POSTPROCESS_ROOT
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            ARXIV_POSTPROCESS_ROOT
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image",
        ),
        (
            "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0",
            ARXIV_POSTPROCESS_ROOT
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "manifest.jsonl",
            ARXIV_POSTPROCESS_ROOT
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "pdf_image",
        ),
        (
            "arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40",
            ARXIV_POSTPROCESS_ROOT
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            ARXIV_POSTPROCESS_ROOT
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image",
        ),
    ]
    return specs


def export_conversation_roots() -> list[Path]:
    return [
        GENERATED_ROOT / "O3_data_0424" / "train_part1" / "medium" / "raw",
        GENERATED_ROOT / "O3_data_0424" / "train_part2a" / "medium" / "raw",
        GENERATED_ROOT / "O3_data_0424" / "train_part2b" / "medium" / "raw",
        GENERATED_ROOT / "O3_data_0424" / "train_part2c" / "medium" / "raw",
        GENERATED_ROOT / "O3_data_0424" / "dude_poster_unanswerable" / "medium" / "raw",
        GENERATED_ROOT / "arxiv" / "train_part1" / "medium" / "raw",
        GENERATED_ROOT / "arxiv" / "train_part2" / "medium" / "raw",
        GENERATED_ROOT / "arxiv" / "train_part3" / "medium" / "raw",
        GENERATED_ROOT / "arxiv" / "spanning_train_part1" / "medium" / "raw",
    ]


def build_original_manifest_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for group_name, manifest_path, pdf_image_root in original_manifest_specs():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = str(row["question_id"])
                index[qid] = {
                    "group_name": group_name,
                    "manifest_path": manifest_path,
                    "pdf_image_root": pdf_image_root,
                    "row": row,
                }
    return index


def build_export_conversation_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for root in export_conversation_roots():
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                record = load_exported_conversation(str(path))
            except Exception:
                continue
            extra = record.get("extra_info") or {}
            qid = extra.get("question_id")
            if not qid:
                continue
            qid = str(qid)
            if qid in index:
                existing = index[qid]
                existing.setdefault("duplicate_paths", []).append(str(path))
                continue
            index[qid] = {
                "path": path,
                "root": root,
                "agent_name": record.get("agent_name"),
                "question": extra.get("question"),
                "subset": extra.get("subset"),
                "reward": record.get("reward"),
                "record": record,
            }
    return index


def extract_relevant_page_ids(row: dict[str, Any]) -> list[int]:
    extra = row.get("extra_info") or {}
    details = _maybe_parse_serialized(extra.get("question_involved_visual_details"))
    if not isinstance(details, list):
        return []
    page_ids: set[int] = set()
    for item in details:
        if not isinstance(item, dict):
            continue
        visual = item.get("visual")
        if isinstance(visual, dict) and isinstance(visual.get("page_id"), int):
            page_ids.add(int(visual["page_id"]))
        captions = item.get("caption")
        if isinstance(captions, list):
            for caption in captions:
                if isinstance(caption, dict) and isinstance(caption.get("page_id"), int):
                    page_ids.add(int(caption["page_id"]))
    return sorted(page_ids)


def relevant_image_indices(row: dict[str, Any]) -> list[int]:
    target_pages = set(extract_relevant_page_ids(row))
    if not target_pages:
        return []
    out: list[int] = []
    for idx, image_uri in enumerate(extract_image_paths(row)):
        page_id = _page_id_from_image_path(image_uri)
        if page_id is not None and page_id in target_pages:
            out.append(idx)
    return out


def original_row_lookup(row: dict[str, Any], manifest_index: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if manifest_index is None:
        manifest_index = build_original_manifest_index()
    qid = str((row.get("extra_info") or {}).get("question_id"))
    return manifest_index.get(qid)


def export_conversation_lookup(
    row: dict[str, Any], export_index: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    if export_index is None:
        export_index = build_export_conversation_index()
    qid = str((row.get("extra_info") or {}).get("question_id"))
    return export_index.get(qid)


def original_relevant_page_paths(
    row: dict[str, Any], manifest_index: dict[str, dict[str, Any]] | None = None
) -> list[tuple[int, Path]]:
    lookup = original_row_lookup(row, manifest_index=manifest_index)
    if lookup is None:
        return []
    page_ids = extract_relevant_page_ids(row)
    pdf_image_root = lookup["pdf_image_root"]
    out: list[tuple[int, Path]] = []
    for page_id in page_ids:
        image_path = pdf_image_root / str(lookup["row"]["document_id"]).replace(".pdf", "") / f"{page_id:06d}.png"
        if not image_path.exists():
            image_path = pdf_image_root / str(lookup["row"]["document_id"]).replace(".pdf", "") / f"{page_id:06d}.jpg"
        if image_path.exists():
            out.append((page_id, image_path))
    return out


def _resized_for_display(image: Image.Image, max_side: int = 1200) -> Image.Image:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


def _display_restored_export_conversation(restored_payload: dict[str, Any], max_image_side: int = 1200) -> None:
    image_idx = 0

    def display_next_image() -> None:
        nonlocal image_idx
        image = restored_payload["multi_modal_data"]["images"][image_idx]
        display(Markdown(f"`presented_image_idx={image_idx}`"))
        if image is None:
            display(Markdown("_image unavailable from stored reference_"))
        else:
            display(_resized_for_display(image, max_side=max_image_side))
        image_idx += 1

    for message_idx, message in enumerate(restored_payload["messages"]):
        role = message.get("role", "unknown")
        display(Markdown(f"**{message_idx}. {role}**"))
        contents = message["content"] if isinstance(message.get("content"), list) else [message.get("content")]
        for content in contents:
            if isinstance(content, str):
                content = {"type": "text", "text": content}
            if not isinstance(content, dict):
                display(Markdown(f"```text\n{str(content)}\n```"))
                continue
            if content.get("type") == "text":
                text = content.get("text", "")
                if text:
                    display(Markdown(f"```text\n{text}\n```"))
            elif content.get("type") == "image":
                display_next_image()
            else:
                display(Markdown(f"```json\n{json.dumps(content, ensure_ascii=False, indent=2)}\n```"))


def display_prompt(row: dict[str, Any]) -> None:
    prompt = row.get("prompt") or []
    prompt = _to_python(prompt)
    for turn in prompt:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        display(Markdown(f"**{role}**\n\n```text\n{content}\n```"))


def display_images(
    row: dict[str, Any],
    max_images: int | None = None,
    start: int = 0,
    max_image_side: int = 1200,
    relevant_only: bool = False,
) -> None:
    paths = extract_image_paths(row)
    indexed_paths = list(enumerate(paths))
    if relevant_only:
        wanted = set(relevant_image_indices(row))
        indexed_paths = [(idx, path) for idx, path in indexed_paths if idx in wanted]
    indexed_paths = indexed_paths[start:]
    if max_images is not None:
        indexed_paths = indexed_paths[:max_images]
    for idx, image_uri in indexed_paths:
        image_path = _strip_file_uri(image_uri)
        page_id = _page_id_from_image_path(image_uri)
        page_suffix = f" (page_id={page_id})" if page_id is not None else ""
        display(Markdown(f"**Image {idx}**{page_suffix}  \n`{image_path}`"))
        display(_resized_for_display(Image.open(image_path), max_side=max_image_side))


def display_original_relevant_pages(
    row: dict[str, Any],
    manifest_index: dict[str, dict[str, Any]] | None = None,
    max_image_side: int = 1200,
) -> None:
    page_paths = original_relevant_page_paths(row, manifest_index=manifest_index)
    if not page_paths:
        display(Markdown("### Original Relevant Pages\n- not recoverable for this row"))
        return
    display(Markdown("### Original Relevant Pages"))
    for page_id, image_path in page_paths:
        display(Markdown(f"**page_id={page_id}**  \n`{image_path}`"))
        display(_resized_for_display(Image.open(image_path), max_side=max_image_side))


def display_export_conversation(
    row: dict[str, Any],
    export_index: dict[str, dict[str, Any]] | None = None,
    max_image_side: int = 1200,
) -> None:
    lookup = export_conversation_lookup(row, export_index=export_index)
    if lookup is None:
        display(Markdown("### Corresponding `vreasoner_v2` Exported Conversation\n- not found"))
        return
    record = lookup["record"]
    reward = record.get("reward") or {}
    score = reward.get("score") or {}
    display(Markdown("### Corresponding `vreasoner_v2` Exported Conversation"))
    display(
        pd.DataFrame(
            [
                {
                    "export_path": str(lookup["path"]),
                    "agent_name": record.get("agent_name"),
                    "subset": (record.get("extra_info") or {}).get("subset"),
                    "question_id": (record.get("extra_info") or {}).get("question_id"),
                    "accuracy_reward": score.get("accuracy_reward"),
                    "n_valid_tool_calls": score.get("n_valid_tool_calls"),
                    "extracted_answer": reward.get("extracted_answer"),
                    "ground_truth": reward.get("ground_truth"),
                }
            ]
        )
    )
    restored = restore_conversation_for_visualization(record)
    _display_restored_export_conversation(restored, max_image_side=max_image_side)


def display_row(
    df: pd.DataFrame,
    row_index: int,
    max_images: int | None = 8,
    image_start: int = 0,
    max_image_side: int = 1200,
    relevant_only: bool = False,
    show_original_manifest_row: bool = True,
    show_original_relevant_pages_only: bool = True,
    manifest_index: dict[str, dict[str, Any]] | None = None,
    show_export_conversation: bool = True,
    export_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row = normalize_row(df.iloc[row_index])
    brief = row_brief(row)
    display(Markdown(f"## Row {row_index}"))
    display(pd.DataFrame([brief]))
    if show_original_manifest_row:
        lookup = original_row_lookup(row, manifest_index=manifest_index)
        if lookup is None:
            display(Markdown("### Original Manifest Row\n- not found"))
        else:
            original_row = lookup["row"]
            display(Markdown("### Original Manifest Row"))
            display(
                pd.DataFrame(
                    [
                        {
                            "group_name": lookup["group_name"],
                            "manifest_path": str(lookup["manifest_path"]),
                            "question_id": original_row.get("question_id"),
                            "document_id": original_row.get("document_id"),
                            "subset": original_row.get("subset"),
                            "question_type": original_row.get("question_type"),
                            "answer": original_row.get("answer"),
                        }
                    ]
                )
            )
    rel_pages = extract_relevant_page_ids(row)
    rel_indices = relevant_image_indices(row)
    if rel_pages:
        display(Markdown(f"### Recovered Relevant Pages\n- page_ids: `{rel_pages}`\n- matching image indices: `{rel_indices}`"))
    else:
        display(Markdown("### Recovered Relevant Pages\n- not present in parquet metadata for this row"))
    if show_original_relevant_pages_only:
        display_original_relevant_pages(row, manifest_index=manifest_index, max_image_side=max_image_side)
    display(Markdown("### Question"))
    display(Markdown(row.get("extra_info", {}).get("question", "")))
    display(Markdown("### Ground Truth"))
    display(Markdown(f"```text\n{row.get('reward_model', {}).get('ground_truth', '')}\n```"))
    display(Markdown("### Prompt"))
    display_prompt(row)
    display(Markdown("### Images"))
    display_images(
        row,
        max_images=max_images,
        start=image_start,
        max_image_side=max_image_side,
        relevant_only=relevant_only,
    )
    if show_export_conversation:
        display_export_conversation(row, export_index=export_index, max_image_side=max_image_side)
    return row


def subset_counts(df: pd.DataFrame) -> pd.Series:
    return df["extra_info"].map(lambda x: (x or {}).get("subset")).value_counts(dropna=False)


def question_type_counts(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    def _qt(value: Any) -> str:
        extra = value or {}
        qt = _maybe_parse_serialized(extra.get("question_type"))
        return json.dumps(qt, ensure_ascii=False) if isinstance(qt, (list, dict)) else str(qt)

    return df["extra_info"].map(_qt).value_counts(dropna=False).head(top_n)


def sample_rows(df: pd.DataFrame, n: int = 10, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)
