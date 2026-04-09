import importlib
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.vsearch import resize_bbox


def _import_insight_o3_agent_loop():
    api_module = types.ModuleType("insight_o3.utils.api")
    api_module.create_async_openai_client = lambda *args, **kwargs: None
    api_module.query_api = lambda *args, **kwargs: None
    sys.modules.setdefault("insight_o3", types.ModuleType("insight_o3"))
    sys.modules.setdefault("insight_o3.utils", types.ModuleType("insight_o3.utils"))
    sys.modules["insight_o3.utils.api"] = api_module
    return importlib.import_module("verl.experimental.agent_loop.insight_o3_agent_loop")


def test_processed_bbox_translation_stays_valid_when_display_projection_collapses():
    module = _import_insight_o3_agent_loop()
    parent_bbox = (7, 11, 307, 211)
    displayed_size = (6, 4)
    processed_size = (84, 56)
    model_bbox = (0, 0, 1, 7)
    parent = module.PresentedImage(
        image=Image.new("RGB", displayed_size),
        source_original_img_idx=0,
        bbox_on_original=parent_bbox,
        display_size=displayed_size,
    )

    bbox_on_original = module._translate_processed_bbox_to_original(parent, model_bbox, processed_size)
    expected = (
        parent_bbox[0] + round(model_bbox[0] * (parent_bbox[2] - parent_bbox[0]) / processed_size[0]),
        parent_bbox[1] + round(model_bbox[1] * (parent_bbox[3] - parent_bbox[1]) / processed_size[1]),
        parent_bbox[0] + round(model_bbox[2] * (parent_bbox[2] - parent_bbox[0]) / processed_size[0]),
        parent_bbox[1] + round(model_bbox[3] * (parent_bbox[3] - parent_bbox[1]) / processed_size[1]),
    )
    assert bbox_on_original == expected

    with pytest.raises(ValueError):
        resize_bbox(model_bbox, processed_size, displayed_size)

    bbox_on_presented = module._resize_bbox_by_rounding(model_bbox, processed_size, displayed_size)
    assert bbox_on_presented == (0, 0, 0, 0)
    assert bbox_on_original == (7, 11, 11, 36)


def test_processed_bbox_translation_matches_full_image_resize_path():
    module = _import_insight_o3_agent_loop()
    original_size = (4032, 3024)
    displayed_size = (1008, 756)
    processed_size = (1036, 784)
    model_bbox = (251, 139, 806, 642)
    parent = module.PresentedImage(
        image=Image.new("RGB", displayed_size),
        source_original_img_idx=0,
        bbox_on_original=(0, 0, *original_size),
        display_size=displayed_size,
    )

    bbox_on_original = module._translate_processed_bbox_to_original(parent, model_bbox, processed_size)
    assert bbox_on_original == resize_bbox(model_bbox, processed_size, original_size)
