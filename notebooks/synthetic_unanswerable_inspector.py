from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import Markdown, display
from PIL import Image

DEFAULT_MAX_IMAGE_SIDE = 900
DEFAULT_MAX_GENERATION_IMAGES = 6
DEFAULT_MAX_VERIFICATION_IMAGES = 8
INSIGHT_DOC_DATA_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc")
_IMAGE_PATH_CACHE: dict[str, Path] = {}
_KNOWN_PDF_IMAGE_ROOTS: list[Path] | None = None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_verified_rows(verified_jsonl: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(verified_jsonl)


def build_overview_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        candidate = item.get("candidate") or {}
        verification = item.get("verification") or {}
        source_row = candidate.get("source_row") or {}
        records.append(
            {
                "row_index": idx,
                "candidate_id": candidate.get("candidate_id"),
                "source_question_id": candidate.get("source_question_id"),
                "subset": source_row.get("subset"),
                "mutation_type": candidate.get("mutation_type"),
                "label": verification.get("label"),
                "question": candidate.get("candidate_question"),
                "final_answer": verification.get("final_answer"),
            }
        )
    return pd.DataFrame.from_records(records)


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _reconstruct_marked_text(preamble: str, labels: list[str], parts: list[str]) -> str:
    prefix = f"{preamble.strip()} " if preamble.strip() else ""
    joined = " ".join(f"({label}) {_normalize_text(part)}" for label, part in zip(labels, parts))
    return (prefix + joined).strip()


def _full_original_question(candidate: dict[str, Any]) -> str:
    return (
        _normalize_text(candidate.get("seed_question_full"))
        or _normalize_text((candidate.get("source_row_full") or {}).get("question"))
        or _normalize_text((candidate.get("source_row") or {}).get("question"))
    )


def _full_original_answer(candidate: dict[str, Any]) -> str:
    return (
        _normalize_text(candidate.get("seed_answer_full"))
        or _normalize_text((candidate.get("source_row_full") or {}).get("answer"))
        or _normalize_text((candidate.get("source_row") or {}).get("answer"))
    )


def _full_final_answer(candidate: dict[str, Any], verification: dict[str, Any]) -> str:
    multipart = candidate.get("multipart_metadata")
    final_answer = _normalize_text(verification.get("final_answer"))
    if not isinstance(multipart, dict) or not multipart.get("is_multipart"):
        return final_answer or _full_original_answer(candidate)
    answer_parts = list(multipart.get("answer_parts_original") or [])
    labels = [str(x) for x in (multipart.get("part_labels") or [])]
    idx = int(multipart.get("selected_part_index", 0))
    if final_answer and 0 <= idx < len(answer_parts):
        answer_parts[idx] = final_answer
    return _reconstruct_marked_text(
        _normalize_text(multipart.get("preamble_answer")),
        labels,
        [str(x) for x in answer_parts],
    )


def _manifest_image_root(source_manifest_path: str | Path) -> Path:
    return Path(source_manifest_path).expanduser().resolve().parent / "pdf_image"


def _known_pdf_image_roots() -> list[Path]:
    global _KNOWN_PDF_IMAGE_ROOTS
    if _KNOWN_PDF_IMAGE_ROOTS is not None:
        return _KNOWN_PDF_IMAGE_ROOTS

    roots: list[Path] = []
    patterns = [
        "O3_data_0424/*/dpi200_aug_noaug_maxp40/pdf_image",
        "arxiv_0307_sample/qa_gen/postprocess/*/*/pdf_image",
    ]
    for pattern in patterns:
        for path in INSIGHT_DOC_DATA_ROOT.glob(pattern):
            if path.is_dir():
                roots.append(path)
    _KNOWN_PDF_IMAGE_ROOTS = roots
    return roots


def _resolve_image_path(source_manifest_path: str | Path, rel_path: str) -> Path:
    if rel_path in _IMAGE_PATH_CACHE:
        return _IMAGE_PATH_CACHE[rel_path]

    manifest_root = _manifest_image_root(source_manifest_path)
    direct = manifest_root / rel_path
    if direct.exists():
        _IMAGE_PATH_CACHE[rel_path] = direct
        return direct

    for root in _known_pdf_image_roots():
        candidate = root / rel_path
        if candidate.exists():
            _IMAGE_PATH_CACHE[rel_path] = candidate
            return candidate

    fallback = INSIGHT_DOC_DATA_ROOT / rel_path
    _IMAGE_PATH_CACHE[rel_path] = fallback
    return fallback


def _resize_for_display(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)


def _display_images(title: str, image_paths: list[Path], max_image_side: int, max_images: int | None = None) -> None:
    if not image_paths:
        display(Markdown(f"**{title}**: none"))
        return
    shown_paths = image_paths[:max_images] if max_images is not None else image_paths
    suffix = f" (showing {len(shown_paths)} of {len(image_paths)})" if len(shown_paths) != len(image_paths) else ""
    display(Markdown(f"**{title}**{suffix}"))
    for image_path in shown_paths:
        display(Markdown(f"`{image_path.name}`"))
        with Image.open(image_path) as image:
            display(_resize_for_display(image.convert("RGB"), max_image_side))


def _prioritize_verification_images(
    rel_paths: list[str],
    selected_pages: list[int],
    verifier_ref_pages: list[int],
) -> list[str]:
    if not rel_paths or not verifier_ref_pages:
        return rel_paths
    by_page = {page: rel for page, rel in zip(selected_pages, rel_paths)}
    prioritized: list[str] = []
    seen: set[str] = set()
    for page in verifier_ref_pages:
        rel = by_page.get(page)
        if rel and rel not in seen:
            prioritized.append(rel)
            seen.add(rel)
    for rel in rel_paths:
        if rel not in seen:
            prioritized.append(rel)
    return prioritized


def show_case(
    rows: list[dict[str, Any]],
    row_index: int,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    max_generation_images: int = DEFAULT_MAX_GENERATION_IMAGES,
    max_verification_images: int = DEFAULT_MAX_VERIFICATION_IMAGES,
) -> None:
    item = rows[row_index]
    candidate = item.get("candidate") or {}
    verification = item.get("verification") or {}
    source_row = candidate.get("source_row") or {}

    source_manifest_path = Path(candidate.get("source_manifest_path")).expanduser().resolve()
    image_root = _manifest_image_root(source_manifest_path)

    generation_rel_paths = candidate.get("generation_relevant_images", [])
    verification_rel_paths = item.get("verification_selected_images", [])
    verification_rel_paths = _prioritize_verification_images(
        verification_rel_paths,
        item.get("verification_selected_pages", []),
        verification.get("verifier_ref_pages", []),
    )

    generation_images = [_resolve_image_path(source_manifest_path, rel) for rel in generation_rel_paths]
    verification_images = [_resolve_image_path(source_manifest_path, rel) for rel in verification_rel_paths]

    display(Markdown(f"### Row {row_index}"))
    display(
        Markdown(
            "\n".join(
                [
                    f"- `source_question_id`: `{candidate.get('source_question_id')}`",
                    f"- `subset`: `{source_row.get('subset')}`",
                    f"- `mutation_type`: `{candidate.get('mutation_type')}`",
                    f"- `label`: `{verification.get('label')}`",
                ]
            )
        )
    )
    display(Markdown("**Original Question**"))
    display(Markdown(_full_original_question(candidate)))
    display(Markdown("**Original Answer**"))
    display(Markdown(_full_original_answer(candidate)))
    if isinstance(candidate.get("multipart_metadata"), dict) and candidate["multipart_metadata"].get("is_multipart"):
        selected_label = candidate["multipart_metadata"].get("selected_part_label")
        display(Markdown(f"**Selected Sub-question** `{selected_label}`"))
        display(Markdown(_normalize_text(source_row.get("question"))))
        display(Markdown("**Mutated Full Question**"))
        display(Markdown(_normalize_text(candidate.get("candidate_question"))))
    else:
        display(Markdown("**Candidate Question**"))
        display(Markdown(_normalize_text(candidate.get("candidate_question"))))
    display(Markdown("**Verifier Reason**"))
    display(Markdown(verification.get("reason", "")))
    ref_pages = verification.get("verifier_ref_pages", [])
    if ref_pages:
        display(Markdown(f"**Verifier Reference Pages**: `{ref_pages}`"))
    display(Markdown("**Final Answer**"))
    display(Markdown(_normalize_text(verification.get("final_answer"))))
    if isinstance(candidate.get("multipart_metadata"), dict) and candidate["multipart_metadata"].get("is_multipart"):
        display(Markdown("**Final Multipart Answer**"))
        display(Markdown(_full_final_answer(candidate, verification)))

    _display_images(
        "Generation Images",
        generation_images,
        max_image_side=max_image_side,
        max_images=max_generation_images,
    )
    _display_images(
        "Verification Images",
        verification_images,
        max_image_side=max_image_side,
        max_images=max_verification_images,
    )
