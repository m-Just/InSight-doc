import argparse
import base64
import json
import random
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from pprint import pp

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from PIL import Image


QWEN3_VL_ANALYSIS_PROMPT = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""


PROMPTS = {
    "vreasoner": {
        "system": "Dummy system prompt. This will be replaced during rollout.",
        "user_template": "Dummy prompt. This will be replaced during rollout. <image>{question}",
    },
    "insight_qwen_agent": {
        "system": QWEN3_VL_ANALYSIS_PROMPT,
        "user_template": "<image>{question}",
    },
    "vreasoner_qwen3_vl": {
        "system": QWEN3_VL_ANALYSIS_PROMPT,
        "user_template": "<image>{question}\nPut your final answer inside <answer>...</answer>.",
    },
    "vsearcher_qwen2_5_vl": {
        "system": 'You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n**Example**:  \n<tool_call>  \n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n</tool_call>',
        "user_template": (
            "<image>\nLocate {target}."
            "\nThink first, call **image_zoom_in_tool** if needed, then answer with the bbox coordinates in [x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can't locate it). "
            "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  "
            "<answer>[x1, y1, x2, y2]</answer> (otherwise)"
        ),
    },
    "vsearcher_qwen3_vl": {
        "system": QWEN3_VL_ANALYSIS_PROMPT,
        "user_template": (
            "<image>\nLocate {target}."
            "\nThink first, call **image_zoom_in_tool** if needed, then answer with the bbox coordinates in [x1, y1, x2, y2] format (or [0, 0, 0, 0] if you can't locate it). "
            "Put your answer inside <answer>...</answer>."
        ),
    },
}


class VerlFormatDataset(ABC):
    DATA_SOURCE: str  # unique identifier for the dataset
    SPLITS: list[str]  # list of available splits
    DEFAULT_PROMPT_TEMPLATE: str | None = None

    def __init__(self, data_root, **extra_options):
        self.data_root = data_root

    def get_raw_hf_dataset(self, split_name, sample_size=None) -> Dataset:
        """Helper function to create a HuggingFace dataset for the specified split."""
        assert split_name in self.SPLITS, f"Invalid split name: {split_name}. Available splits: {self.SPLITS}"
        data = self._load_raw_data(split_name)

        # If data is already a Dataset, just handle sampling
        if isinstance(data, Dataset):
            if sample_size:
                return data.shuffle(seed=42).select(range(sample_size))
            return data

        # If data is a list, convert to Dataset and handle sampling
        if sample_size:
            random.seed(42)
            data = random.sample(data, k=sample_size)
        return Dataset.from_list(data)

    @abstractmethod
    def _load_raw_data(self, split_name) -> Dataset | list[dict]: ...

    @abstractmethod
    def get_question(self, example) -> str: ...

    @abstractmethod
    def get_answer(self, example) -> str: ...

    @abstractmethod
    def get_image_objs_or_urls(self, example) -> list[Image.Image | str]: ...

    def get_extra_info(self, example) -> dict:
        return {
            "question": self.get_question(example),
        }

    def get_question_id(self, example) -> str:
        extra_info = self.get_extra_info(example)
        question_id = extra_info.get("question_id")
        if question_id is None:
            raise ValueError(
                f"{self.__class__.__name__} does not expose question_id in get_extra_info(), "
                "so it cannot be filtered by question-id file."
            )
        return str(question_id)


class O3Bench(VerlFormatDataset):
    DATA_SOURCE = "o3_bench"
    SPLITS = ["test"]

    def _load_raw_data(self, split_name):
        df = pd.read_json(Path(self.data_root, "test", "metadata.jsonl"), lines=True)
        data = df.to_dict(orient="records")
        data_new = []
        for i, d in enumerate(data):
            d_new = {
                "question_id": str(i),
                "image": d["file_name"],
                "question": d["question"],
                "options": d["options"],
                "answer": d["answer"].strip(),
                "subset": d["subset"],
                "data_source": f"{self.DATA_SOURCE}/{d['subset']}",
            }
            data_new.append(d_new)
        return data_new

    def get_question(self, example):
        question = f"{example['question']}\n{example['options']}"
        question = question.replace("\\n", "\n")
        return question

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        img_path = Path(self.data_root, "test", example["image"])
        assert img_path.exists(), f"Image file does not exist: {img_path}"
        return [f"file://{img_path}"]

    def get_extra_info(self, example):
        return {
            **VerlFormatDataset.get_extra_info(self, example),
            "question_id": str(example["question_id"]),
            "subset": example["subset"],
        }


class VStarBench(VerlFormatDataset):
    DATA_SOURCE = "vstar_bench"
    SPLITS = ["test"]

    def _load_raw_data(self, split_name):
        df = pd.read_json(Path(self.data_root, "vstar_bench.jsonl"), lines=True)
        return df.to_dict(orient="records")

    def get_question(self, example):
        question = f"{example['question']}\n{example['options']}"
        question = question.replace("\\n", "\n")
        return question

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        img_path = Path(self.data_root, example["image"])
        assert img_path.exists(), f"Image file does not exist: {img_path}"
        return [f"file://{img_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": str(example["question_id"]),
            "category": example["category"],
            "bboxes": self.get_bboxes(example),
        }

    def get_bboxes(self, example):
        with Path(self.get_image_objs_or_urls(example)[0].replace("file://", "")).with_suffix(".json").open("rb") as f:
            attributes = json.load(f)
        bboxes = []
        for bbox_xywh in attributes["bbox"]:
            bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
            bboxes.append(bbox_xyxy)
        return bboxes


class MME_RealWorld_Lite(VerlFormatDataset):
    DATA_SOURCE = "mme_realworld_lite"
    SPLITS = ["test"]

    def _load_raw_data(self, split_name):
        df = pd.read_json(Path(self.data_root, "MME_RealWorld_Lite.jsonl"), lines=True)
        return df.to_dict(orient="records")

    def get_question(self, example):
        question = f"{example['question']}\n{example['options']}"
        question = question.replace("\\n", "\n")
        return question

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        img_path = Path(self.data_root, "imgs", example["image"])
        assert img_path.exists(), f"Image file does not exist: {img_path}"
        return [f"file://{img_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": str(example["question_id"]),
            "category": example["category"],
        }


class VisualProbeHard(VerlFormatDataset):
    DATA_SOURCE = "visual_probe_hard"
    SPLITS = ["test"]

    def _load_raw_data(self, split_name):
        df = pd.read_json(Path(self.data_root, "visual_probe_hard.jsonl"), lines=True)
        return df.to_dict(orient="records")

    def get_question(self, example):
        return example["question"]

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        img_path = Path(self.data_root).parent / example["image"]
        assert img_path.exists(), f"Image file does not exist: {img_path}"
        return [f"file://{img_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": str(example["question_id"]),
            "category": example["category"],
        }


class VisCoT_VStar_Collage(VerlFormatDataset):
    DATA_SOURCE = "viscot_vstar_collage"
    SPLITS = ["train"]

    def _load_raw_data(self, split_name):
        def add_index(example, idx):
            example["index"] = idx
            return example

        dataset = load_dataset(self.data_root, split="train")
        dataset = dataset.map(add_index, with_indices=True)

        self.images_dir = Path(self.data_root, "train", "images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dataset loaded with packed images. Images will be extracted to {self.images_dir}")

        return dataset

    def get_question(self, example):
        return example["question"]

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        image_path = self.images_dir / f"{example['index']}.jpg"
        example["image"].save(image_path)
        return [f"file://{image_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": str(example["index"]),
            "qa_pairs": example["qa_pairs"],
            "n_layouts": example["n_layouts"],
            "layout_types": example["layout_types"],
            "core_layout_coord": example["core_layout_coord"],
            "core_target_bbox": example["core_target_bbox"],
            "has_ref": True,
        }


class InfoVQA_RegionLocalization(VerlFormatDataset):
    DATA_SOURCE = "info_vqa_region_localization"
    SPLITS = ["train"]

    def _load_raw_data(self, split_name):
        dataset = load_dataset(self.data_root, split="train")
        extract_img_flags = []
        seen_imgs = set()
        for question_id in dataset["question_id"]:
            img_id, _ = question_id.split("_")
            if img_id not in seen_imgs:
                seen_imgs.add(img_id)
                extract_img_flags.append(True)
            else:
                extract_img_flags.append(False)
        dataset = dataset.add_column("extract_img_flag", extract_img_flags)

        self.images_dir = Path(self.data_root, "train", "images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dataset loaded with packed images. Images will be extracted to {self.images_dir}")

        return dataset

    def get_question(self, example):
        return example["region_description"]

    def get_search_target(self, example):
        return example["region_description"]

    def get_answer(self, example):
        return str(example["bbox"])

    def get_image_objs_or_urls(self, example):
        img_id, _ = example["question_id"].split("_")
        image_path = self.images_dir / f"{img_id}.jpg"
        if example["extract_img_flag"]:
            example["image"].save(image_path)
        return [f"file://{image_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": str(example["question_id"]),
            "search_target": example["region_description"],
            "bboxes": [example["bbox"]],
        }


class InSightDoc(VerlFormatDataset):
    """Document QA dataset from gather_qa_outputs output.

    Expects the directory layout:
        <data_root>/
        ├── qa_samples.jsonl
        └── images/
            └── <document_id>/
                ├── 000000.jpg
                └── ...

    Each sample may reference multiple page images.  When ``document_slice``
    is set, only the sliced pages are included.
    """

    DATA_SOURCE = "insight_doc"
    SPLITS = ["all"]
    DEFAULT_PROMPT_TEMPLATE = "<image>{question}"

    def _load_raw_data(self, split_name):
        jsonl_path = Path(self.data_root) / "qa_samples.jsonl"
        assert jsonl_path.exists(), f"qa_samples.jsonl not found at {jsonl_path}"
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def get_question(self, example):
        return example["question"]

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        document_id = example["document_id"]
        images_dir = Path(self.data_root) / "images" / document_id
        assert images_dir.is_dir(), f"Images directory not found: {images_dir}"

        all_pages = sorted(images_dir.glob("*.jpg"))
        assert all_pages, f"No page images found in {images_dir}"

        doc_slice = example.get("document_slice")
        if doc_slice is not None:
            start, end = doc_slice
            all_pages = [p for p in all_pages if start <= int(p.stem) < end]
            assert all_pages, (
                f"No page images in slice [{start}, {end}) for document {document_id}"
            )

        return [f"file://{p}" for p in all_pages]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": example["question_id"],
            "document_id": example["document_id"],
            "document_slice": example.get("document_slice"),
        }


class InSightDocRegionLocalization(VerlFormatDataset):
    """Region-localization dataset gathered from region description generation.

    Expects the directory layout:
        <data_root>/
        ├── region_desc_samples.jsonl
        └── images/
            └── <document_id>/
                ├── 000000.jpg
                └── ...
    """

    DATA_SOURCE = "insight_doc_region_localization"
    SPLITS = ["all"]

    def _load_raw_data(self, split_name):
        jsonl_path = Path(self.data_root) / "region_desc_samples.jsonl"
        assert jsonl_path.exists(), f"region_desc_samples.jsonl not found at {jsonl_path}"
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def get_question(self, example):
        return self.get_search_target(example)

    def get_search_target(self, example):
        desc = example["region_description"].replace("<image>", "< image >")  # workaround for <image> placeholder in prompt
        return desc

    def get_answer(self, example):
        return str(example["bbox"])

    def get_image_objs_or_urls(self, example):
        page_image_path = example.get("page_image_path")
        if page_image_path is not None:
            image_path = Path(self.data_root) / page_image_path
        else:
            image_path = (
                Path(self.data_root) / "images" / example["document_id"]
                / f"{int(example['page_id']):06d}.jpg"
            )
        assert image_path.exists(), f"Image file does not exist: {image_path}"
        return [f"file://{image_path}"]

    def get_extra_info(self, example):
        return {
            **super().get_extra_info(example),
            "question_id": example.get("question_id", example["sample_id"]),
            "sample_id": example["sample_id"],
            "document_id": example["document_id"],
            "page_id": example["page_id"],
            "search_target": example["region_description"],
            "bboxes": [example["bbox_image"]],
            "description_mode": example.get("description_mode"),
            "block_types": example.get("block_types"),
            "num_blocks": example.get("num_blocks"),
        }


class InSightDocBase(VerlFormatDataset):
    """Document QA dataset from postprocess Stage-4 augmented output.

    Expects the directory layout produced by ``run_postprocess_stage4_png_from_json.py``::

        <data_root>/                        # = <postprocess_dir>/<augment_id>/
        ├── manifest.jsonl
        └── pdf_image/
            └── <document_stem>/
                ├── 000000.jpg
                └── ...

    Each line in ``manifest.jsonl`` contains ``question_id``, ``document_id``,
    ``question``, ``answer``, ``images`` (relative to ``pdf_image/``), ``subset``
    (``veqa`` | ``mveqa``), and optional visual detail fields.
    """

    DATA_SOURCE = "insight_doc_base"
    SPLITS = ["all"]
    DEFAULT_PROMPT_TEMPLATE = "<image>{question}"

    def __init__(self, data_root, **extra_options):
        super().__init__(data_root, **extra_options)
        self.manifest_file = extra_options.get("manifest_file", "manifest.jsonl")

    def _load_raw_data(self, split_name):
        jsonl_path = Path(self.data_root) / self.manifest_file
        assert jsonl_path.exists(), f"{self.manifest_file} not found at {jsonl_path}"
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for key in ("question_id", "question", "answer", "document_id", "subset", "data_source"):
                    value = record.get(key)
                    if value is not None and not isinstance(value, str):
                        record[key] = str(value)
                if "images" in record and isinstance(record["images"], list):
                    record["images"] = [str(item) for item in record["images"]]
                question_type = record.get("question_type")
                if isinstance(question_type, (list, dict)):
                    record["question_type"] = json.dumps(question_type, ensure_ascii=False)
                elif question_type is not None:
                    record["question_type"] = str(question_type)
                data.append(record)
        return data

    def get_question(self, example):
        return example["question"]

    def get_answer(self, example):
        return example["answer"]

    def get_image_objs_or_urls(self, example):
        pdf_image_dir = Path(self.data_root) / "pdf_image"
        images = example.get("images", [])
        assert images, f"No images for question {example.get('question_id')}"
        urls = []
        for rel in images:
            img_path = pdf_image_dir / rel
            assert img_path.exists(), f"Image file does not exist: {img_path}"
            urls.append(f"file://{img_path}")
        return urls

    def get_extra_info(self, example):
        visual_details = example.get("question_involved_visual_details")
        if visual_details is not None:
            visual_details = json.dumps(visual_details, ensure_ascii=False)
        extra_info = {
            **super().get_extra_info(example),
            "question_id": str(example["question_id"]),
            "document_id": example.get("document_id"),
            "subset": example.get("subset"),
            "question_involved_visuals": example.get("question_involved_visuals"),
            "question_involved_visual_details": visual_details,
            "question_type": example.get("question_type"),
        }
        optional_extra_info_keys = ("initial_rescale", "initial_rescale_source", "initial_rescale_dpi")
        for key in optional_extra_info_keys:
            value = example.get(key)
            if value is None:
                continue
            extra_info[key] = float(value) if key == "initial_rescale" else value
        return extra_info


class InSightDoc0352(InSightDocBase):
    DATA_SOURCE = "insight_doc_0352"


class InSightDocMixed(InSightDocBase):
    DATA_SOURCE = "insight_doc_mixed"


class InSightDocMixedWithArxiv(InSightDocBase):
    DATA_SOURCE = "insight_doc_mixed_with_arxiv"


class InSightEvalDude200(InSightDocBase):
    DATA_SOURCE = "dude200"


class InSightEvalLongDocURL200(InSightDocBase):
    DATA_SOURCE = "longdocurl200"


class InSightEvalMMLite200(InSightDocBase):
    DATA_SOURCE = "mmlite200"

    def __init__(self, data_root, **extra_options):
        extra_options = {"manifest_file": "mme_manifest_sample200.jsonl", **extra_options}
        super().__init__(data_root, **extra_options)


class InSightEvalMMLongBench200(InSightDocBase):
    DATA_SOURCE = "mmlongbench200"


class InSightEvalMPDocVQA200(InSightDocBase):
    DATA_SOURCE = "mpdocvqa200"


class InSightEvalO3Bench(InSightDocBase):
    DATA_SOURCE = "o3bench0502"

    def __init__(self, data_root, **extra_options):
        extra_options = {"manifest_file": "o3bench_manifest.json", **extra_options}
        super().__init__(data_root, **extra_options)


class InSightDocRL(InSightDocBase):
    DATA_SOURCE = "insight_doc_rl"


def get_image_obj(url_or_path: str | Path) -> Image.Image:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        response = requests.get(url_or_path, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif url_or_path.startswith("file://"):
        image_obj = Image.open(url_or_path[7:])
    elif url_or_path.startswith("data:image"):
        if "base64," in url_or_path:
            _, base64_data = url_or_path.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(url_or_path)
    return image_obj


def make_map_fn(dataset, split_name, prompt_style, agent_name=None, validate_images=False):
    def process_fn(example, idx):
        prompts = PROMPTS[prompt_style]
        if "{target}" in prompts["user_template"]:
            user_prompt = prompts["user_template"].format(target=dataset.get_search_target(example))
        elif "{question}" in prompts["user_template"]:
            user_prompt = prompts["user_template"].format(question=dataset.get_question(example))
        else:
            raise ValueError(f"Invalid prompt template: {prompts['user_template']}")

        image_objs_or_urls = dataset.get_image_objs_or_urls(example)
        num_images = len(image_objs_or_urls)
        assert num_images >= 1, "Expected at least one image"

        # Expand a single <image> placeholder to match the actual number of images.
        num_tags = user_prompt.count("<image>")
        if num_tags == 1 and num_images > 1:
            user_prompt = user_prompt.replace("<image>", "<image>" * num_images, 1)
            num_tags = num_images
        assert num_tags == num_images, (
            f"Prompt has {num_tags} <image> tags but dataset returned {num_images} images."
        )

        for img in image_objs_or_urls:
            assert isinstance(img, str | Image.Image), f"Invalid image object or URL: {img}"
        if validate_images:
            for img in image_objs_or_urls:
                if isinstance(img, str):
                    image_obj = get_image_obj(img)
                    try:
                        image_obj.load()
                    except Exception:
                        raise
                    finally:
                        image_obj.close()

        extra_info = dataset.get_extra_info(example)
        for reserved_key in ("split", "index", "prompt_style"):
            assert reserved_key not in extra_info

        prompt_messages = []
        if prompts.get("system") is not None:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": prompts["system"],
                }
            )
        prompt_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        data = {
            "data_source": example.get("data_source", dataset.DATA_SOURCE),
            "prompt": prompt_messages,
            "images": [{"image": img} for img in image_objs_or_urls],
            "reward_model": {
                "style": "rule",
                "ground_truth": dataset.get_answer(example),
            },
            "extra_info": {
                "split": split_name,
                "index": idx,
                "prompt_style": prompt_style,
                **extra_info,
            },
        }

        if agent_name:
            data["agent_name"] = agent_name

        example.clear()
        return data

    return process_fn


if __name__ == "__main__":
    """Example usages:

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset O3Bench \
        --data_root data/O3-Bench \
        --split test \
        --prompt vreasoner \
        --output_path data/O3-Bench/test.parquet \
        --agent_name vreasoner

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset O3Bench \
        --data_root data/O3-Bench \
        --split test \
        --prompt vreasoner_qwen3_vl \
        --output_path data/O3-Bench/test-vreasoner_qwen3vl.parquet \
        --agent_name vreasoner_qwen3_vl

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset VStarBench \
        --data_root data/vstar_bench \
        --split test \
        --prompt vreasoner \
        --output_path data/vstar_bench/test.parquet \
        --agent_name vreasoner

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset VStarBench \
        --data_root data/vstar_bench \
        --split test \
        --prompt vreasoner_qwen3_vl \
        --output_path data/vstar_bench/test-vreasoner_qwen3vl.parquet \
        --agent_name vreasoner_qwen3_vl

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset VisualProbeHard \
        --data_root data/VisualProbe_Hard \
        --split test \
        --prompt vreasoner \
        --output_path data/VisualProbe_Hard/test.parquet \
        --agent_name vreasoner

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset MME_RealWorld_Lite \
        --data_root data/MME-RealWorld-Lite \
        --split test \
        --prompt vreasoner \
        --output_path data/MME-RealWorld-Lite/test.parquet \
        --agent_name vreasoner

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset VisCoT_VStar_Collage \
        --data_root data/VisCoT_VStar_Collage \
        --split train \
        --prompt vreasoner \
        --output_path data/VisCoT_VStar_Collage/train.parquet \
        --agent_name vreasoner \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset VisCoT_VStar_Collage \
        --data_root data/VisCoT_VStar_Collage \
        --split train \
        --prompt vreasoner_qwen3_vl \
        --output_path data/VisCoT_VStar_Collage/train-vreasoner_qwen3vl.parquet \
        --agent_name vreasoner_qwen3_vl \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InfoVQA_RegionLocalization \
        --data_root data/InfoVQA_RegionLocalization \
        --split train \
        --prompt vsearcher_qwen2_5_vl \
        --output_path data/InfoVQA_RegionLocalization/train-vsearcher.parquet \
        --agent_name vsearcher \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InfoVQA_RegionLocalization \
        --data_root data/InfoVQA_RegionLocalization \
        --split train \
        --prompt vsearcher_qwen3_vl \
        --output_path data/InfoVQA_RegionLocalization/train-vsearcher_qwen3vl.parquet \
        --agent_name vsearcher_qwen3_vl \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDocRegionLocalization \
        --data_root /scratch/ywxzml3j/likaican/data/InSightDocRegionLocalization \
        --split all \
        --prompt vsearcher_qwen3_vl \
        --output_path /scratch/ywxzml3j/likaican/data/InSightDocRegionLocalization/all-vsearcher_qwen3vl.parquet \
        --agent_name vsearcher_qwen3_vl \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc \
        --data_root /scratch/ywxzml3j/likaican/temp/0207select234 \
        --split all \
        --prompt vreasoner \
        --output_path /scratch/ywxzml3j/likaican/temp/0207select234-vreasoner.parquet \
        --agent_name vreasoner \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc \
        --data_root /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50 \
        --split all \
        --prompt vreasoner \
        --output_path /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50-vreasoner_v2.parquet \
        --agent_name vreasoner_v2 \
        --num_workers 32

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc0352 \
        --data_root /home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40 \
        --split all \
        --prompt insight_qwen_agent \
        --output_path /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.parquet \
        --agent_name insight_qwen_agent \
        --num_workers 32 \
        --extra_options "{\"manifest_file\": \"manifest.jsonl\"}" \
        --test_size 0.2

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc0352 \
        --data_root /home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40 \
        --split all \
        --prompt insight_qwen_agent \
        --output_path /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_sample_50_maxp40-insight_qwen_agent.parquet \
        --agent_name insight_qwen_agent \
        --num_workers 32 \
        --extra_options "{\"manifest_file\": \"manifest_sample_50.jsonl\"}"

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc0352 \
        --data_root /home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40 \
        --split all \
        --prompt vreasoner \
        --output_path /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-vreasoner_v2.parquet \
        --agent_name vreasoner_v2 \
        --num_workers 32 \
        --extra_options "{\"manifest_file\": \"manifest.jsonl\"}" \
        --test_size 0.2

    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset InSightDoc0352 \
        --data_root /home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40 \
        --split all \
        --prompt vreasoner \
        --output_path /scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_sample_50_maxp40-vreasoner_v2.parquet \
        --agent_name vreasoner_v2 \
        --num_workers 32 \
        --extra_options "{\"manifest_file\": \"manifest_sample_50.jsonl\"}"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset class name")
    parser.add_argument("--data_root", type=str, help="Root directory for the dataset")
    parser.add_argument("--split", type=str, required=True, help="Split name")
    parser.add_argument(
        "--extra_options", type=json.loads, default={}, help="Extra options to pass to the dataset class."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the parquet file. Two splits will be saved if --test_size is specified.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["default", *PROMPTS.keys()],
        default="default",
        help="Prompt template to use. If not specified, the dataset's default (if there is one) will be used.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Random sample size for the dataset. If not specified, the full dataset will be used.",
    )
    parser.add_argument(
        "--test_size", type=float, help="Force train-test split with a test set of this size (between 0 and 1)"
    )
    parser.add_argument("--agent_name", type=str, help="Name of the agent intended for the data. (default: None)")
    parser.add_argument(
        "--validate_images",
        action="store_true",
        help="Ensure that all images can be accessed and opened as PIL images. This may take a while.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for parallel processing.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run the dataset creation process. Do not save the dataset to parquet.",
    )
    parser.add_argument(
        "--question_id_file",
        type=str,
        default=None,
        help="Optional text file with one question_id per line. Only those questions will be included.",
    )
    args = parser.parse_args()

    if args.dataset not in globals():
        raise ValueError(f"Dataset class {args.dataset} not found.")
    dataset_class = globals()[args.dataset]

    data_root = str(Path(args.data_root).resolve())
    dataset = dataset_class(data_root, **args.extra_options)
    hf_dataset = dataset.get_raw_hf_dataset(args.split, args.sample_size)

    if args.question_id_file:
        question_id_path = Path(args.question_id_file).expanduser().resolve()
        assert question_id_path.exists(), f"question_id file not found: {question_id_path}"
        with question_id_path.open("r", encoding="utf-8") as f:
            selected_question_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(selected_question_ids)} question_ids from {question_id_path}")

        filter_kwargs = {
            "desc": f"Filtering to {len(selected_question_ids)} selected question_ids",
        }
        if args.num_workers > 1:
            filter_kwargs["num_proc"] = args.num_workers
        hf_dataset = hf_dataset.filter(
            lambda example: dataset.get_question_id(example) in selected_question_ids,
            **filter_kwargs,
        )
        print(f"Filtered dataset len: {len(hf_dataset)}")

    if dataset.DEFAULT_PROMPT_TEMPLATE:
        PROMPTS["default"] = {
            "system": None,
            "user_template": dataset.DEFAULT_PROMPT_TEMPLATE,
        }

    def show_example_data(hf_dataset):
        example = hf_dataset[0]

        # Omit image bytes (if any)
        for img in example["images"]:
            if isinstance(img["image"], dict) and img["image"].get("bytes", None):
                img["image"]["bytes"] = "..."

        print("\nData example:")
        pp(example, width=100)

    if not args.test_size:
        map_fn = make_map_fn(dataset, args.split, args.prompt, args.agent_name, args.validate_images)
        map_kwargs = {
            "function": map_fn,
            "with_indices": True,
        }
        if args.num_workers > 1:
            map_kwargs["num_proc"] = args.num_workers
        hf_dataset = hf_dataset.map(**map_kwargs)
        if not args.dry_run:
            hf_dataset.to_parquet(args.output_path)
        show_example_data(hf_dataset)
        full_path = str(Path(args.output_path).resolve())
        print(f"\nFinished {args.split} split with {len(hf_dataset)} samples.\nSaved to {full_path}")
    else:
        print(f"Original split: {args.split} split with {len(hf_dataset)} samples.")
        hf_dataset_dict = hf_dataset.train_test_split(test_size=args.test_size, seed=42)
        output_paths = {}
        for split_name in hf_dataset_dict:
            print("--------------------------------")
            map_fn = make_map_fn(dataset, split_name, args.prompt, args.agent_name, args.validate_images)
            split_map_kwargs = {
                "function": map_fn,
                "with_indices": True,
            }
            if args.num_workers > 1:
                split_map_kwargs["num_proc"] = args.num_workers
            hf_dataset_dict[split_name] = hf_dataset_dict[split_name].map(**split_map_kwargs)
            output_paths[split_name] = args.output_path.replace(".parquet", f".{split_name}.parquet")
            if not args.dry_run:
                hf_dataset_dict[split_name].to_parquet(output_paths[split_name])
            show_example_data(hf_dataset_dict[split_name])
            full_path = str(Path(output_paths[split_name]).resolve())
            print(f"\nNew {split_name} split: {len(hf_dataset_dict[split_name])} samples.\nSaved to: {full_path}")
