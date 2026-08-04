#!/usr/bin/env python3
"""Build deterministic arXiv structural QA rows similar to LongDocURL tasks.

The builder uses existing rendered page images from arXiv training parquets and
MinerU line JSONs for the same documents. It does not call any LLM. Answers are
copied exactly from MinerU title/table/figure-caption text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


QWEN3_VL_ANALYSIS_PROMPT = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""


DEFAULT_ARXIV_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample")
DEFAULT_INSIGHT_DOC_SRC = Path("/scratch/ywxzml3j/likaican/src/InSight-doc")
DEFAULT_OUTPUT_DIR = Path(
    "notes/generated/arxiv_struct_longdocurl_like_20260721"
)
DEFAULT_SOURCE_PARQUETS = [
    DEFAULT_ARXIV_ROOT
    / "parquets/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-insight_qwen_agent_default_sys_0426_train_part1.parquet",
    DEFAULT_ARXIV_ROOT
    / "parquets/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-insight_qwen_agent_default_sys_0426_train_part2.parquet",
    DEFAULT_ARXIV_ROOT
    / "parquets/veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0-insight_qwen_agent_default_sys_0426_train_part5.parquet",
]
DEFAULT_TASK_TARGETS_1K = {
    "summary2title": 300,
    "summary2tab": 300,
    "extract_figure_other_figures": 120,
    "extract_table_other_tables": 120,
    "extract_figure2table": 70,
    "extract_table2figure": 50,
    "topic2title": 40,
}
DEFAULT_TASK_TARGETS_1K_UNIFORM_MULTI = {
    "topic2title": 200,
    "summary2tab": 250,
    "extract_figure_other_figures": 75,
    "extract_table_other_tables": 75,
    "extract_figure2table": 75,
    "extract_table2figure": 75,
    "summary2title": 250,
}
DEFAULT_MULTI_ANSWER_TARGETS_1K_UNIFORM = {
    "topic2title": 140,
    "summary2tab": 150,
    "extract_figure_other_figures": 38,
    "extract_table_other_tables": 38,
    "extract_figure2table": 37,
    "extract_table2figure": 37,
}
STOPWORDS = {
    "about",
    "above",
    "after",
    "against",
    "all",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "can",
    "case",
    "each",
    "for",
    "from",
    "has",
    "have",
    "here",
    "into",
    "more",
    "most",
    "not",
    "only",
    "other",
    "our",
    "over",
    "paper",
    "present",
    "results",
    "section",
    "show",
    "shown",
    "shows",
    "such",
    "table",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "various",
    "were",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class Item:
    kind: str
    text: str
    page_id: int
    block_id: int


@dataclass(frozen=True)
class Candidate:
    task_type: str
    document_id: str
    question: str
    answer: str
    answer_items: tuple[str, ...]
    page_ids: tuple[int, ...]
    anchor: str | None
    source_block_ids: tuple[int, ...]


def normalize_ws(text: str | None) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int, min_punct_cut: int = 90) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text.rstrip(" ,;:")
    cut = max(text.rfind(".", 0, max_chars), text.rfind(";", 0, max_chars), text.rfind(",", 0, max_chars))
    if cut >= min_punct_cut:
        return text[: cut + 1].rstrip(" ,;:")
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")


def join_descriptions(descs: list[str]) -> str:
    return "; ".join(truncate_text(desc, 160).rstrip(" .?,;:") for desc in descs if desc)


def strip_number_prefix(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"^\s*(?:section\s+)?(?:[A-Z]?\d+(?:\.\d+)*|[IVXLCDM]+)\s*[\).:-]?\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:figure|fig\.?|table)\s+[A-Z]?\d+(?:\.\d+)*\s*[:.-]?\s*", "", text, flags=re.I)
    return normalize_ws(text)


def normalize_for_match(text: str) -> str:
    text = strip_number_prefix(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return normalize_ws(text)


def answer_body_leaks(question: str, answer_items: list[str]) -> bool:
    question_norm = normalize_for_match(question)
    for item in answer_items:
        answer_norm = normalize_for_match(item)
        if len(answer_norm) >= 10 and answer_norm in question_norm:
            return True
    return False


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", normalize_for_match(text))
    return {word for word in words if word not in STOPWORDS and len(word) >= 4}


def content_overlap_count(left: str, right: str) -> int:
    return len(content_tokens(left) & content_tokens(right))


VISUAL_LABEL_RE = re.compile(
    r"\b(?:tables?|tabs?\.?|figs?\.?|figures?)\s*[\(\[]?\s*[A-Za-z]?\d+(?:[.\-]\s*[A-Za-z0-9]+)*[A-Za-z]?\b",
    flags=re.I,
)


def normalize_visual_label(label: str) -> str:
    label = label.lower()
    label = re.sub(r"\b(?:figs?|figures?)\.?\b", "figure", label)
    label = re.sub(r"\b(?:tabs?|tables?)\.?\b", "table", label)
    return normalize_ws(label).strip(" .:-")


def visual_labels(text: str) -> set[str]:
    return {normalize_visual_label(match.group(0)) for match in VISUAL_LABEL_RE.finditer(str(text))}


def visual_label_leaks(question: str, answer_items: list[str]) -> bool:
    question_labels = visual_labels(question)
    if not question_labels:
        return False
    answer_labels = set()
    for item in answer_items:
        answer_labels.update(visual_labels(item))
    return bool(question_labels & answer_labels)


def remove_visual_label_mentions(text: str) -> str:
    text = re.sub(r"\b[Tt]ables?\s*(?:[A-Za-z]?\d+(?:[.\-][A-Za-z0-9]+)*[A-Za-z]?\s*(?:,|and|&)\s*)+[A-Za-z]?\d+(?:[.\-][A-Za-z0-9]+)*[A-Za-z]?\b", "these tables", text)
    text = VISUAL_LABEL_RE.sub("these tables", text)
    text = re.sub(r"\bthese tables\s*(?:,|and|&)\s*these tables\b", "these tables", text, flags=re.I)
    return normalize_ws(text)


def stable_choice(options: list[str], *parts: Any) -> str:
    key = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def caption_prefix(kind: str) -> str:
    return "figure" if kind == "figure" else "table"


def is_placeholder_or_noise(text: str) -> bool:
    low = text.lower()
    if re.search(r"<<[^>]+>>", text):
        return True
    if len(text) < 6 or len(text) > 260:
        return True
    if sum(ch.isalpha() for ch in text) < 4:
        return True
    if low in {"abstract", "references", "acknowledgements", "acknowledgments", "appendix"}:
        return True
    # Very equation-heavy captions make brittle exact-answer supervision.
    if text.count("$") >= 8 or text.count("\\") >= 10:
        return True
    return False


def math_or_ocr_noise_score(text: str) -> int:
    text = normalize_ws(text)
    score = 0
    score += text.count("$") * 2
    score += text.count("\\")
    score += len(re.findall(r"[_^{}]|[=<>≤≥±∑∫√∞≈≠]", text))
    score += len(re.findall(r"\b[A-Za-z]\s*(?:_|\\_|\^|\\\^)", text))
    score += len(re.findall(r"\b(?:mathbb|mathrm|mathbf|mathcal|operatorname|frac|sum|prod|int)\b", text))
    # Penalize fragmented OCR such as "fea- tures" only lightly; a few cases are recoverable.
    score += len(re.findall(r"\b[A-Za-z]{2,}-\s+[A-Za-z]{2,}\b", text))
    return score


def has_inline_math(text: str) -> bool:
    return bool(
        "$" in text
        or "\\" in text
        or re.search(r"\b[A-Za-z]\s*(?:_|\\_|\^|\\\^)\s*\{?[A-Za-z0-9]", text)
    )


def is_toc_like(text: str) -> bool:
    text = normalize_ws(text)
    toc_hits = re.findall(
        r"\b(?:[A-Z]?\d+(?:\.\d+)+|[A-Z]\.\d+)\s+[^.;:]{5,90}\s+\d{1,3}\b",
        text,
    )
    return len(toc_hits) >= 3


def is_noisy_context(text: str, *, max_score: int = 10, reject_inline_math: bool = False) -> bool:
    text = normalize_ws(text)
    if len(text) < 55:
        return True
    if reject_inline_math and has_inline_math(text):
        return True
    if is_toc_like(text):
        return True
    if math_or_ocr_noise_score(text) > max_score:
        return True
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < max(30, len(text) * 0.45):
        return True
    if len(re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", text)) < 10:
        return True
    return False


def is_valid_title(text: str) -> bool:
    if is_placeholder_or_noise(text):
        return False
    if re.match(r"^\W*(?:fig\.?|figure|table)\s+[A-Za-z]?\d+", text, flags=re.I):
        return False
    body = strip_number_prefix(text)
    low = body.lower().strip(" .:")
    if low in {
        "abstract",
        "introduction",
        "references",
        "bibliography",
        "conclusion",
        "conclusions",
        "appendix",
        "preliminaries",
        "acknowledgements",
        "acknowledgments",
        "code availability",
        "data availability",
    }:
        return False
    if len(body) < 8:
        return False
    return True


def is_valid_caption(kind: str, text: str) -> bool:
    if is_placeholder_or_noise(text):
        return False
    if math_or_ocr_noise_score(text) > 10:
        return False
    if re.search(r"\bcontinued\b", text, flags=re.I):
        return False
    if len(strip_number_prefix(text)) < 8:
        return False
    if kind == "figure":
        if re.match(r"^\W*(?:fig\.?|figure)\s+[A-Za-z]?\d+", text, flags=re.I) is None:
            return False
        return len(re.findall(r"\b(?:fig\.?|figure)\s+[A-Za-z]?\d+", text, flags=re.I)) == 1
    if re.match(r"^\W*table\s+[A-Za-z]?\d+", text, flags=re.I) is None:
        return False
    return len(re.findall(r"\btable\s+[A-Za-z]?\d+", text, flags=re.I)) == 1


def answer_string(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return json.dumps(items, ensure_ascii=False)


def title_topic_question(topic: str, document_id: str, title: str) -> str:
    template = stable_choice(
        [
            'Which section discusses "{topic}"? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.',
            'Which section is about "{topic}"? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.',
            'Where can we find discussion of "{topic}"? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.',
            'Which section title best corresponds to "{topic}"? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.',
        ],
        "topic2title",
        document_id,
        title,
    )
    return template.format(topic=topic)


def summary_title_question(desc: str, document_id: str, title: str) -> str:
    template = stable_choice(
        [
            "Which section best matches the following description: <description>{desc}</description> Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Find the section title that corresponds to this description: <description>{desc}</description> Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Where in the document is the following content discussed? <description>{desc}</description> Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Which title names the section summarized here: <description>{desc}</description> Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
        ],
        "summary2title",
        document_id,
        title,
        desc[:80],
    )
    return template.format(desc=desc)


def table_topic_question(topic: str, document_id: str, table: str) -> str:
    template = stable_choice(
        [
            "Which tables provide information on {topic}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Which table names are most relevant to {topic}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "From which tables can we learn about {topic}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Find the tables that discuss {topic}. Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
        ],
        "summary2tab",
        document_id,
        table,
    )
    return template.format(topic=topic)


def table_context_question(desc: str, document_id: str, table: str) -> str:
    template = stable_choice(
        [
            "Which tables are relevant to the following context: <description>{desc}</description> Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Find the table names that best match this surrounding evidence: <description>{desc}</description> Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "From which tables can we verify the following information? <description>{desc}</description> Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Which tables would you consult for the following described content? <description>{desc}</description> Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
        ],
        "summary2tab_context",
        document_id,
        table,
        desc[:80],
    )
    return template.format(desc=desc)


def multi_title_topic_question(topics: list[str], document_id: str, titles: list[str]) -> str:
    joined = join_descriptions(topics)
    template = stable_choice(
        [
            "Which sections discuss the following topics: {joined}? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Where can we find sections about these topics: {joined}? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Which section titles cover these pieces of content: {joined}? Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
            "Find the section titles corresponding to these descriptions: {joined}. Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
        ],
        "topic2title_multi",
        document_id,
        " || ".join(titles),
    )
    return template.format(joined=joined)


def multi_table_context_question(descs: list[str], document_id: str, tables: list[str]) -> str:
    joined = join_descriptions(descs)
    template = stable_choice(
        [
            "Which tables provide information about these described contents: {joined}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Find the table names that match these pieces of evidence: {joined}. Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "From which tables can we verify these kinds of information: {joined}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
            "Which tables would you consult for these topics: {joined}? Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
        ],
        "summary2tab_multi",
        document_id,
        " || ".join(tables),
    )
    return template.format(joined=joined)


def same_kind_question(kind: str, anchor: str, document_id: str, page_id: int) -> str:
    name = caption_prefix(kind)
    template = stable_choice(
        [
            'List names of the other {name}s at the page which contains a {name} whose name is "{anchor}".',
            'What are the names of the other {name}s on the same page as the {name} named "{anchor}"?',
            'Find the other {name} names from the page containing the {name} named "{anchor}".',
            'On the page with the {name} named "{anchor}", list the names of the other {name}s.',
        ],
        f"extract_{kind}_other_{kind}s",
        document_id,
        page_id,
        anchor,
    )
    return template.format(name=name, anchor=anchor)


def cross_kind_question(anchor_kind: str, target_kind: str, anchor: str, document_id: str, page_id: int) -> str:
    anchor_name = caption_prefix(anchor_kind)
    target_name = caption_prefix(target_kind)
    template = stable_choice(
        [
            'List names of the {target_name}s at the page which contains a {anchor_name} whose name is "{anchor}".',
            'What are the names of the {target_name}s on the same page as the {anchor_name} named "{anchor}"?',
            'Find the {target_name} names from the page containing the {anchor_name} named "{anchor}".',
            'On the page with the {anchor_name} named "{anchor}", list the names of the {target_name}s.',
        ],
        f"extract_{anchor_kind}2{target_kind}",
        document_id,
        page_id,
        anchor,
    )
    return template.format(anchor_name=anchor_name, target_name=target_name, anchor=anchor)


def safe_doc_id_for_qid(document_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", document_id).strip("_")


def get_text_for_line_types(block: Any, line_types: set[str]) -> str:
    return normalize_ws(" ".join(line.text or "" for line in block.lines if line.type in line_types))


def first_known_page(block: Any, line_types: set[str] | None = None) -> int | None:
    for line in block.lines:
        if line_types is not None and line.type not in line_types:
            continue
        if line.page_id is not None:
            return int(line.page_id)
    for line in block.lines:
        if line.page_id is not None:
            return int(line.page_id)
    return None


def block_text(block: Any, line_types: set[str] | None = None) -> str:
    return normalize_ws(
        " ".join(
            line.text or ""
            for line in block.lines
            if line.text and (line_types is None or line.type in line_types)
        )
    )


def page_for_block(block: Any) -> int | None:
    return first_known_page(block)


def image_page_id(image: Any) -> int | None:
    if isinstance(image, dict):
        image = image.get("image") or image.get("path") or image.get("image_path") or str(image)
    match = re.search(r"/([0-9]{6})\.(?:png|jpe?g)$", str(image), flags=re.I)
    if not match:
        return None
    return int(match.group(1))


def build_doc_image_map(source_parquets: list[Path], max_images: int) -> dict[str, list[dict[str, str]]]:
    doc_to_images: dict[str, list[dict[str, str]]] = {}
    for parquet in source_parquets:
        if not parquet.exists():
            raise FileNotFoundError(f"source parquet not found: {parquet}")
        df = pd.read_parquet(parquet, columns=["images", "extra_info"])
        for _, row in df.iterrows():
            extra = row["extra_info"] or {}
            document_id = extra.get("document_id")
            if not document_id:
                continue
            images = list(row["images"])
            if not images:
                continue
            images = images[:max_images]
            old = doc_to_images.get(document_id)
            if old is None or len(images) > len(old):
                doc_to_images[document_id] = images
    return doc_to_images


def available_image_page_ids(images: list[dict[str, str]]) -> set[int]:
    return {page_id for image in images if (page_id := image_page_id(image)) is not None}


def extract_items(document: Any) -> tuple[list[Item], list[Item], list[Item]]:
    titles: list[Item] = []
    figures: list[Item] = []
    tables: list[Item] = []

    for block in document.blocks:
        if block.type == "title":
            text = get_text_for_line_types(block, {"title"})
            page_id = first_known_page(block, {"title"})
            if page_id is not None and is_valid_title(text):
                # Page-0 unnumbered titles are usually paper titles, not section headings.
                if page_id == 0 and re.match(r"^\s*(?:[A-Z]?\d+(?:\.\d+)*|[IVXLCDM]+)\s*[\).:-]?\s+", text, flags=re.I) is None:
                    continue
                titles.append(Item("title", text, page_id, int(block.id)))
            continue

        if block.type == "image":
            text = get_text_for_line_types(block, {"image_caption"})
            page_id = first_known_page(block, {"image_caption"}) or first_known_page(block)
            if page_id is not None and is_valid_caption("figure", text):
                figures.append(Item("figure", text, page_id, int(block.id)))
            continue

        if block.type == "table":
            text = get_text_for_line_types(block, {"table_caption"})
            page_id = first_known_page(block, {"table_caption"}) or first_known_page(block)
            if page_id is not None and is_valid_caption("table", text):
                tables.append(Item("table", text, page_id, int(block.id)))

    # Deduplicate repeated merged captions/titles.
    def dedup(items: list[Item]) -> list[Item]:
        seen: set[tuple[str, int]] = set()
        out: list[Item] = []
        for item in items:
            key = (item.text.lower(), item.page_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    return dedup(titles), dedup(figures), dedup(tables)


def section_description(document: Any, title: Item, max_chars: int = 420) -> str | None:
    blocks = list(document.blocks)
    try:
        idx = next(i for i, block in enumerate(blocks) if int(block.id) == title.block_id)
    except StopIteration:
        return None

    parts: list[str] = []
    for block in blocks[idx + 1 :]:
        if block.type == "title":
            break
        if block.type not in {"text", "list"}:
            continue
        text = normalize_ws(" ".join(line.text or "" for line in block.lines if line.text))
        if not text:
            continue
        parts.append(text)
        if sum(len(p) for p in parts) >= max_chars:
            break
    desc = normalize_ws(" ".join(parts))
    if len(desc) < 80:
        return None
    if visual_labels(desc):
        return None
    if is_noisy_context(desc, max_score=12, reject_inline_math=True):
        return None
    if re.search(r"\b(?:i do not|i cannot|i can't)\s+(?:assist|help|provide)\b", desc, flags=re.I):
        return None
    return truncate_text(desc, max_chars).rstrip()


def split_description_units(text: str) -> list[str]:
    text = normalize_ws(text)
    units = re.split(r"(?<=[.!?])\s+|;\s+|:\s+", text)
    out: list[str] = []
    for unit in units:
        unit = normalize_ws(unit)
        if len(unit) < 55:
            continue
        if not re.match(r"^[A-Za-z\"'“‘]", unit):
            continue
        if len(content_tokens(unit)) < 8:
            continue
        if is_noisy_context(unit, max_score=6, reject_inline_math=True):
            continue
        out.append(truncate_text(unit, 220).rstrip())
    return out


def section_topic_from_description(desc: str, title: str, document_id: str) -> str | None:
    answer_items = [title]
    units = [
        unit
        for unit in split_description_units(desc)
        if not answer_body_leaks(unit, answer_items) and not visual_labels(unit)
    ]
    if not units:
        return None
    return stable_choice(units, "section_topic", document_id, title)


def table_context_description(document_id: str, document: Any, table: Item, max_chars: int = 320) -> str | None:
    blocks = list(document.blocks)
    try:
        idx = next(i for i, block in enumerate(blocks) if int(block.id) == table.block_id)
    except StopIteration:
        return None

    parts: list[tuple[str, str]] = []
    table_block = blocks[idx]
    footnote = block_text(table_block, {"table_footnote"})
    if footnote:
        parts.append(("footnote", footnote))

    # Nearby same-page prose is usually the strongest non-caption evidence for a table.
    for offset in (-3, -2, -1, 1, 2, 3):
        j = idx + offset
        if j < 0 or j >= len(blocks):
            continue
        block = blocks[j]
        if block.type not in {"text", "list"}:
            continue
        if page_for_block(block) != table.page_id:
            continue
        text = block_text(block)
        if not text:
            continue
        parts.append(("nearby_text", text))

    units = []
    for source, part in parts:
        for unit in split_description_units(part):
            if answer_body_leaks(unit, [table.text]):
                continue
            # For single-table semantic lookup, avoid any explicit visual labels:
            # even non-target labels often make the question ambiguous.
            if visual_labels(unit):
                continue
            if source != "footnote" and content_overlap_count(unit, table.text) < 2:
                continue
            units.append(unit)
    if not units:
        return None
    # Use up to two stable units to make the query specific without copying the caption.
    seed = int.from_bytes(hashlib.sha1(f"table_context||{document_id}||{table.block_id}".encode("utf-8")).digest()[:4], "big")
    rng = random.Random(seed)
    rng.shuffle(units)
    desc = normalize_ws(" ".join(units[:2]))
    if len(desc) < 60:
        return None
    if visual_label_leaks(desc, [table.text]):
        return None
    return desc[:max_chars].rstrip()


def compact_description(desc: str, max_chars: int = 150) -> str:
    """Keep combined multi-answer prompts readable without model rewriting."""
    units = split_description_units(desc)
    if units:
        desc = units[0]
    desc = remove_visual_label_mentions(desc)
    desc = re.sub(r"\([^)]{35,}\)", "", desc)
    desc = normalize_ws(desc)
    desc = truncate_text(desc, max_chars)
    if desc.split() and desc.split()[-1].strip(".,;:").lower() in {
        "and",
        "or",
        "of",
        "for",
        "to",
        "with",
        "by",
        "due",
        "the",
        "a",
        "an",
    }:
        return ""
    return desc


SECTION_PREFIX_RE = re.compile(
    r"^\s*((?:[A-Z]\.)?\d+(?:\.\d+)*|[A-Z](?:\.\d+)+|[IVXLCDM]+(?:\.\d+)*)\s*[\).:-]?\s+",
    flags=re.I,
)


def title_group_keys(title: Item) -> list[tuple[str, str]]:
    text = normalize_ws(title.text)
    keys: list[tuple[str, str]] = []
    match = SECTION_PREFIX_RE.match(text)
    if match:
        prefix = match.group(1).strip(".")
        parts = prefix.split(".")
        if len(parts) > 1:
            keys.append(("section_parent", ".".join(parts[:-1]).lower()))
    keys.append(("page", str(title.page_id)))
    return keys


def multi_title_topic_candidates(
    document_id: str,
    title_infos: list[tuple[Item, str, str]],
) -> list[Candidate]:
    groups: dict[tuple[str, str], list[tuple[Item, str, str]]] = defaultdict(list)
    seen_per_group: set[tuple[tuple[str, str], int]] = set()
    for info in title_infos:
        title, _, topic = info
        if answer_body_leaks(topic, [title.text]) or is_noisy_context(topic, max_score=6, reject_inline_math=True):
            continue
        for key in title_group_keys(title):
            seen_key = (key, title.block_id)
            if seen_key in seen_per_group:
                continue
            seen_per_group.add(seen_key)
            groups[key].append(info)

    candidates: list[Candidate] = []
    for key, infos in groups.items():
        if len(infos) < 2:
            continue
        infos = sorted(infos, key=lambda x: (x[0].page_id, x[0].block_id))
        # Consecutive sections preserve local coherence better than arbitrary global grouping.
        for start in range(0, len(infos) - 1):
            window = infos[start : start + 3]
            if len(window) < 2:
                continue
            if len(window) > 2 and key[0] == "page":
                window = window[:2]
            titles = [item.text for item, _, _ in window]
            topics = [compact_description(topic, 135) for _, _, topic in window]
            if len({normalize_for_match(topic) for topic in topics}) != len(topics):
                continue
            if any(not topic or is_noisy_context(topic, max_score=6, reject_inline_math=True) for topic in topics):
                continue
            question = multi_title_topic_question(topics, document_id, titles)
            if answer_body_leaks(question, titles):
                continue
            add_candidate(
                candidates,
                "topic2title",
                document_id,
                question,
                titles,
                [item.page_id for item, _, _ in window],
                " || ".join(titles),
                [item.block_id for item, _, _ in window],
            )
    return candidates


def multi_table_context_candidates(
    document_id: str,
    document: Any,
    tables: list[Item],
    available_page_ids: set[int],
) -> list[Candidate]:
    by_page: dict[int, list[Item]] = defaultdict(list)
    for table in tables:
        if table.page_id in available_page_ids and visual_labels(table.text):
            by_page[table.page_id].append(table)

    candidates: list[Candidate] = []
    blocks = list(document.blocks)
    for page_id, page_tables in by_page.items():
        if len(page_tables) < 2:
            continue
        label_to_table: dict[str, Item] = {}
        for table in page_tables:
            for label in visual_labels(table.text):
                label_to_table[label] = table
        if len(label_to_table) < 2:
            continue

        for block in blocks:
            if block.type not in {"text", "list"}:
                continue
            if page_for_block(block) != page_id:
                continue
            for unit in split_description_units(block_text(block)):
                matched_tables: dict[str, Item] = {
                    label: label_to_table[label]
                    for label in visual_labels(unit)
                    if label in label_to_table
                }
                answer_tables = list({table.text: table for table in matched_tables.values()}.values())
                if len(answer_tables) < 2:
                    continue
                desc = remove_visual_label_mentions(unit)
                if len(desc) < 65 or len(content_tokens(desc)) < 8:
                    continue
                answer_items = [table.text for table in answer_tables]
                if answer_body_leaks(desc, answer_items) or visual_label_leaks(desc, answer_items):
                    continue
                question = table_context_question(desc[:320].rstrip(), document_id, " || ".join(answer_items))
                if answer_body_leaks(question, answer_items) or visual_label_leaks(question, answer_items):
                    continue
                add_candidate(
                    candidates,
                    "summary2tab",
                    document_id,
                    question,
                    answer_items,
                    [page_id],
                    " || ".join(answer_items),
                    [table.block_id for table in answer_tables] + [int(block.id)],
                )
    return candidates


def multi_table_semantic_candidates(
    document_id: str,
    document: Any,
    tables: list[Item],
    available_page_ids: set[int],
) -> list[Candidate]:
    by_page: dict[int, list[Item]] = defaultdict(list)
    for table in tables:
        if table.page_id in available_page_ids and visual_labels(table.text):
            by_page[table.page_id].append(table)

    candidates: list[Candidate] = []
    seen_answer_sets: set[tuple[str, ...]] = set()

    def clean_table_infos(table_list: list[Item]) -> list[tuple[Item, str]]:
        infos: list[tuple[Item, str]] = []
        for table in sorted(table_list, key=lambda item: (item.page_id, item.block_id)):
            desc = table_context_description(document_id, document, table)
            if not desc:
                continue
            desc = compact_description(desc, 145)
            if visual_labels(desc) or answer_body_leaks(desc, [table.text]) or is_noisy_context(desc, max_score=6, reject_inline_math=True):
                continue
            infos.append((table, desc))
        return infos

    def add_multi_table_candidate(window: list[tuple[Item, str]]) -> None:
        if len(window) < 2:
            return
        tables_out = [table.text for table, _ in window]
        key = tuple(sorted(normalize_for_match(table) for table in tables_out))
        if key in seen_answer_sets:
            return
        descs = [desc for _, desc in window]
        if len({normalize_for_match(desc) for desc in descs}) != len(descs):
            return
        question = multi_table_context_question(descs, document_id, tables_out)
        if answer_body_leaks(question, tables_out) or visual_label_leaks(question, tables_out):
            return
        seen_answer_sets.add(key)
        add_candidate(
            candidates,
            "summary2tab",
            document_id,
            question,
            tables_out,
            [table.page_id for table, _ in window],
            " || ".join(tables_out),
            [table.block_id for table, _ in window],
        )

    for page_id, page_tables in by_page.items():
        if len(page_tables) < 2:
            continue
        infos = clean_table_infos(page_tables)
        if len(infos) < 2:
            continue

        for start in range(0, len(infos) - 1):
            window = infos[start : start + 3]
            add_multi_table_candidate(window)

    # LongDocURL summary2tab often asks for a small set of related tables across
    # a page range, not necessarily on the exact same page. Consecutive clean
    # table-contexts within a short page span are a reliable deterministic proxy.
    all_infos = clean_table_infos(
        [table for table in tables if table.page_id in available_page_ids and visual_labels(table.text)]
    )
    for start in range(0, len(all_infos) - 1):
        for size in (2, 3):
            window = all_infos[start : start + size]
            if len(window) < size:
                continue
            pages = [table.page_id for table, _ in window]
            if max(pages) - min(pages) > 6:
                continue
            add_multi_table_candidate(window)
    return candidates


def add_candidate(
    candidates: list[Candidate],
    task_type: str,
    document_id: str,
    question: str,
    answer_items: list[str],
    page_ids: list[int],
    anchor: str | None,
    block_ids: list[int],
) -> None:
    if not answer_items:
        return
    candidates.append(
        Candidate(
            task_type=task_type,
            document_id=document_id,
            question=question,
            answer=answer_string(answer_items),
            answer_items=tuple(answer_items),
            page_ids=tuple(sorted(set(page_ids))),
            anchor=anchor,
            source_block_ids=tuple(sorted(set(block_ids))),
        )
    )


def build_candidates_for_doc(document_id: str, document: Any, available_page_ids: set[int]) -> list[Candidate]:
    titles, figures, tables = extract_items(document)
    candidates: list[Candidate] = []
    title_infos: list[tuple[Item, str, str]] = []

    for title in titles:
        if title.page_id not in available_page_ids:
            continue
        desc = section_description(document, title)
        if desc:
            topic = section_topic_from_description(desc, title.text, document_id)
            if topic:
                title_infos.append((title, desc, topic))
                question = title_topic_question(topic, document_id, title.text)
                if not answer_body_leaks(question, [title.text]):
                    add_candidate(
                        candidates,
                        "topic2title",
                        document_id,
                        question,
                        [title.text],
                        [title.page_id],
                        title.text,
                        [title.block_id],
                    )
            question = summary_title_question(desc, document_id, title.text)
            if not answer_body_leaks(question, [title.text]):
                add_candidate(
                    candidates,
                    "summary2title",
                    document_id,
                    question,
                    [title.text],
                    [title.page_id],
                    title.text,
                    [title.block_id],
                )

    candidates.extend(multi_title_topic_candidates(document_id, title_infos))

    for table in tables:
        if table.page_id not in available_page_ids:
            continue
        desc = table_context_description(document_id, document, table)
        if not desc:
            continue
        question = table_context_question(desc, document_id, table.text)
        if answer_body_leaks(question, [table.text]) or visual_label_leaks(question, [table.text]):
            continue
        add_candidate(
            candidates,
            "summary2tab",
            document_id,
            question,
            [table.text],
            [table.page_id],
            table.text,
            [table.block_id],
        )

    candidates.extend(multi_table_context_candidates(document_id, document, tables, available_page_ids))
    candidates.extend(multi_table_semantic_candidates(document_id, document, tables, available_page_ids))

    by_page: dict[int, dict[str, list[Item]]] = defaultdict(lambda: {"figure": [], "table": []})
    for item in figures + tables:
        if item.page_id in available_page_ids:
            by_page[item.page_id][item.kind].append(item)

    for page_id, grouped in by_page.items():
        for kind in ("figure", "table"):
            same = grouped[kind]
            if len(same) >= 2:
                for anchor in same:
                    others = [x for x in same if x.text != anchor.text]
                    answer_items = [x.text for x in others]
                    question = same_kind_question(kind, anchor.text, document_id, page_id)
                    if answer_body_leaks(question, answer_items) or visual_label_leaks(question, answer_items):
                        continue
                    add_candidate(
                        candidates,
                        f"extract_{kind}_other_{kind}s",
                        document_id,
                        question,
                        answer_items,
                        [page_id],
                        anchor.text,
                        [x.block_id for x in others] + [anchor.block_id],
                    )

        for anchor_kind, target_kind in (("figure", "table"), ("table", "figure")):
            anchors = grouped[anchor_kind]
            targets = grouped[target_kind]
            if not anchors or not targets:
                continue
            for anchor in anchors:
                answer_items = [x.text for x in targets]
                question = cross_kind_question(anchor_kind, target_kind, anchor.text, document_id, page_id)
                if answer_body_leaks(question, answer_items) or visual_label_leaks(question, answer_items):
                    continue
                add_candidate(
                    candidates,
                    f"extract_{anchor_kind}2{target_kind}",
                    document_id,
                    question,
                    answer_items,
                    [page_id],
                    anchor.text,
                    [x.block_id for x in targets] + [anchor.block_id],
                )

    return candidates


def make_parquet_row(candidate: Candidate, images: list[dict[str, str]], index: int, initial_rescale: float) -> dict[str, Any]:
    image_tokens = "<image>" * len(images)
    qid = f"arxiv_struct_{candidate.task_type}_{safe_doc_id_for_qid(candidate.document_id)}_{index:06d}"
    return {
        "images": images,
        "data_source": f"arxiv_struct_{candidate.task_type}_answerable",
        "prompt": [
            {"role": "system", "content": QWEN3_VL_ANALYSIS_PROMPT},
            {"role": "user", "content": f"{image_tokens}{candidate.question}"},
        ],
        "reward_model": {"ground_truth": candidate.answer, "style": "rule"},
        "extra_info": {
            "document_id": candidate.document_id,
            "index": index,
            "initial_rescale": initial_rescale,
            "initial_rescale_dpi": int(round(200 * initial_rescale)),
            "initial_rescale_source": "arxiv_struct_longdocurl_like_deterministic",
            "prompt_style": "insight_qwen_agent",
            "question": candidate.question,
            "question_id": qid,
            "question_involved_visual_details": None,
            "question_involved_visuals": list(candidate.page_ids),
            "question_type": candidate.task_type,
            "split": "all",
            "subset": candidate.task_type,
            "answer_items": list(candidate.answer_items),
            "anchor": candidate.anchor,
            "source_block_ids": list(candidate.source_block_ids),
            "builder": "build_arxiv_struct_longdocurl_like_parquet.py",
        },
        "agent_name": "insight_qwen_agent",
    }


def balanced_sample(candidates: list[Candidate], max_rows: int, max_per_type: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_type[cand.task_type].append(cand)
    selected: list[Candidate] = []
    for task_type in sorted(by_type):
        rows = by_type[task_type]
        rng.shuffle(rows)
        selected.extend(rows[:max_per_type])
    rng.shuffle(selected)
    return selected[:max_rows]


def shuffle_with_multi_preference(rows: list[Candidate], rng: random.Random, prefer_multi_answer: bool) -> list[Candidate]:
    rows = list(rows)
    rng.shuffle(rows)
    if not prefer_multi_answer:
        return rows
    multi = [row for row in rows if len(row.answer_items) > 1]
    single = [row for row in rows if len(row.answer_items) <= 1]
    return multi + single


def target_mix_sample(
    candidates: list[Candidate],
    targets: dict[str, int],
    seed: int,
    max_selected_rows_per_doc: int = 0,
    prefer_multi_answer: bool = False,
    multi_answer_targets: dict[str, int] | None = None,
) -> list[Candidate]:
    rng = random.Random(seed)
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_type[cand.task_type].append(cand)

    selected: list[Candidate] = []
    selected_per_doc: Counter[str] = Counter()
    shortfall = 0

    def can_select(cand: Candidate) -> bool:
        return max_selected_rows_per_doc <= 0 or selected_per_doc[cand.document_id] < max_selected_rows_per_doc

    def add_selected(cand: Candidate) -> None:
        selected.append(cand)
        selected_per_doc[cand.document_id] += 1

    def select_from_rows(rows: list[Candidate], target: int, multi_target: int) -> list[Candidate]:
        picked: list[Candidate] = []
        picked_ids: set[int] = set()
        local_per_doc: Counter[str] = Counter()
        rows = list(rows)
        rng.shuffle(rows)

        def can_pick(cand: Candidate) -> bool:
            return (
                max_selected_rows_per_doc <= 0
                or selected_per_doc[cand.document_id] + local_per_doc[cand.document_id] < max_selected_rows_per_doc
            )

        def pick(cand: Candidate) -> None:
            picked.append(cand)
            picked_ids.add(id(cand))
            local_per_doc[cand.document_id] += 1

        if multi_target > 0:
            multi_rows = [row for row in rows if len(row.answer_items) > 1]
            rng.shuffle(multi_rows)
            for cand in multi_rows:
                if len(picked) >= multi_target:
                    break
                if id(cand) in picked_ids or not can_pick(cand):
                    continue
                pick(cand)

        remaining_target = target - len(picked)
        if remaining_target <= 0:
            return picked

        if multi_target > 0:
            # Prefer single-answer rows after the requested multi-answer quota so
            # high-quality extraction tasks do not dominate the multi-answer mix.
            fill_rows = [row for row in rows if len(row.answer_items) <= 1 and id(row) not in picked_ids]
            fill_rows += [row for row in rows if len(row.answer_items) > 1 and id(row) not in picked_ids]
        elif prefer_multi_answer:
            fill_rows = shuffle_with_multi_preference(
                [row for row in rows if id(row) not in picked_ids],
                rng,
                True,
            )
        else:
            fill_rows = [row for row in rows if id(row) not in picked_ids]

        for cand in fill_rows:
            if len(picked) >= target:
                break
            if not can_pick(cand):
                continue
            pick(cand)
        return picked

    multi_answer_targets = multi_answer_targets or {}
    for task_type, target in targets.items():
        rows = by_type.get(task_type, [])
        picked = select_from_rows(rows, target, multi_answer_targets.get(task_type, 0))
        before = len(selected)
        for cand in picked:
            add_selected(cand)
        shortfall += target - (len(selected) - before)

    if shortfall > 0:
        selected_ids = {id(cand) for cand in selected}
        leftovers = [cand for cand in candidates if id(cand) not in selected_ids]
        leftovers = shuffle_with_multi_preference(leftovers, rng, prefer_multi_answer)
        for cand in leftovers:
            if shortfall <= 0:
                break
            if can_select(cand):
                add_selected(cand)
                shortfall -= 1

    rng.shuffle(selected)
    return selected


def parse_task_targets(spec: str) -> dict[str, int]:
    if spec == "longdocurl_struct_1k":
        return dict(DEFAULT_TASK_TARGETS_1K)
    if spec == "longdocurl_struct_1k_uniform_multi":
        return dict(DEFAULT_TASK_TARGETS_1K_UNIFORM_MULTI)
    path = Path(spec)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            values = json.load(f)
        return {str(k): int(v) for k, v in values.items()}
    targets: dict[str, int] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        targets[key.strip()] = int(value)
    return targets


def parse_multi_answer_targets(spec: str) -> dict[str, int]:
    if spec == "longdocurl_struct_1k_uniform_multi":
        return dict(DEFAULT_MULTI_ANSWER_TARGETS_1K_UNIFORM)
    return parse_task_targets(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-root", type=Path, default=DEFAULT_ARXIV_ROOT)
    parser.add_argument("--insight-doc-src", type=Path, default=DEFAULT_INSIGHT_DOC_SRC)
    parser.add_argument("--source-parquet", type=Path, action="append", dest="source_parquets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--max-per-type", type=int, default=500)
    parser.add_argument(
        "--task-targets",
        default="",
        help=(
            "Optional weighted target mix. Use 'longdocurl_struct_1k', "
            "'longdocurl_struct_1k_uniform_multi', a JSON file, "
            "or comma-separated task=count pairs. Overrides --max-rows/--max-per-type."
        ),
    )
    parser.add_argument(
        "--multi-answer-targets",
        default="",
        help=(
            "Optional per-task multi-answer quotas. Use 'longdocurl_struct_1k_uniform_multi', "
            "a JSON file, or comma-separated task=count pairs."
        ),
    )
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all docs found in source parquets")
    parser.add_argument("--max-rows-per-doc", type=int, default=6)
    parser.add_argument(
        "--max-selected-rows-per-doc",
        type=int,
        default=0,
        help="0 disables the final selected-output per-document cap",
    )
    parser.add_argument(
        "--prefer-multi-answer",
        action="store_true",
        help="Prefer multi-answer candidates when applying per-doc and per-task caps.",
    )
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--initial-rescale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_parquets = args.source_parquets or DEFAULT_SOURCE_PARQUETS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["INSIGHT_DOC_DATA_STORAGE_ROOT_DIR"] = str(args.arxiv_root)
    sys.path.insert(0, str(args.insight_doc_src))
    from insight_doc.mineru_postproc.middle_to_lines import get_document_by_id

    doc_to_images = build_doc_image_map(source_parquets, args.max_images)
    doc_ids = sorted(doc_to_images)
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)
    if args.max_docs > 0:
        doc_ids = doc_ids[: args.max_docs]

    all_candidates: list[Candidate] = []
    skipped = Counter()
    docs_with_candidates = 0
    for doc_id in doc_ids:
        try:
            document = get_document_by_id(doc_id)
        except Exception as exc:
            skipped[type(exc).__name__] += 1
            continue
        available_page_ids = available_image_page_ids(doc_to_images[doc_id])
        if not available_page_ids:
            skipped["no_parseable_image_page_ids"] += 1
            continue
        candidates = build_candidates_for_doc(doc_id, document, available_page_ids)
        if not candidates:
            skipped["no_candidates"] += 1
            continue
        candidates = shuffle_with_multi_preference(candidates, rng, args.prefer_multi_answer)
        all_candidates.extend(candidates[: args.max_rows_per_doc])
        docs_with_candidates += 1

    task_targets = parse_task_targets(args.task_targets) if args.task_targets else {}
    multi_answer_targets = {}
    if args.multi_answer_targets:
        multi_answer_targets = parse_multi_answer_targets(args.multi_answer_targets)
    elif args.task_targets == "longdocurl_struct_1k_uniform_multi":
        multi_answer_targets = dict(DEFAULT_MULTI_ANSWER_TARGETS_1K_UNIFORM)
    if task_targets:
        selected = target_mix_sample(
            all_candidates,
            task_targets,
            args.seed,
            max_selected_rows_per_doc=args.max_selected_rows_per_doc,
            prefer_multi_answer=args.prefer_multi_answer,
            multi_answer_targets=multi_answer_targets,
        )
    else:
        selected = balanced_sample(all_candidates, args.max_rows, args.max_per_type, args.seed)
    rows = [
        make_parquet_row(candidate, doc_to_images[candidate.document_id], i, args.initial_rescale)
        for i, candidate in enumerate(selected)
    ]

    output_rows_label = sum(task_targets.values()) if task_targets else args.max_rows
    parquet_path = args.output_dir / f"arxiv_struct_longdocurl_like_deterministic_{output_rows_label}-insight_qwen_agent.parquet"
    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "summary.json"

    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row, cand in zip(rows, selected):
            f.write(json.dumps({
                "question_id": row["extra_info"]["question_id"],
                "data_source": row["data_source"],
                **asdict(cand),
            }, ensure_ascii=False) + "\n")

    summary = {
        "output_parquet": str(parquet_path),
        "manifest": str(manifest_path),
        "source_parquets": [str(p) for p in source_parquets],
        "arxiv_root": str(args.arxiv_root),
        "rows": len(rows),
        "candidate_rows_before_sampling": len(all_candidates),
        "docs_with_images": len(doc_to_images),
        "docs_scanned": len(doc_ids),
        "docs_with_candidates": docs_with_candidates,
        "skipped": dict(skipped),
        "data_source_counts": dict(Counter(row["data_source"] for row in rows)),
        "task_type_counts": dict(Counter(row["extra_info"]["question_type"] for row in rows)),
        "initial_rescale": args.initial_rescale,
        "max_images": args.max_images,
        "max_rows_per_doc": args.max_rows_per_doc,
        "max_selected_rows_per_doc": args.max_selected_rows_per_doc,
        "max_per_type": args.max_per_type,
        "task_targets": task_targets,
        "multi_answer_targets": multi_answer_targets,
        "prefer_multi_answer": args.prefer_multi_answer,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
