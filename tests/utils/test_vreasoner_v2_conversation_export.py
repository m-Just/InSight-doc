import json

from PIL import Image

from verl.utils.vreasoner_v2_conversation_export import (
    append_reward_info,
    build_export_record,
    export_conversation,
    parse_assistant_message,
    parse_user_message,
)


def test_parse_assistant_message_answer():
    parsed = parse_assistant_message("<think>reasoning</think><answer>final</answer>")
    assert parsed["type"] == "answer"
    assert parsed["content"]["think"] == "reasoning"
    assert parsed["content"]["answer"] == "final"


def test_parse_user_message_tool_result():
    parsed = parse_user_message(
        [
            {"type": "text", "text": "Image 3:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx", "detail": "high"}},
            {"type": "text", "text": "\n\nBased on your description, here is the zoomed-in image (Image 3).\n"},
        ],
        initial_question="What is shown?",
    )
    assert parsed["type"] == "tool_result"
    assert parsed["content"]["presented_img_indices"] == [3]


def test_export_and_append_reward(tmp_path):
    record = build_export_record(
        job_id="job-1",
        parent_job_id=None,
        root_job_id="job-1",
        validate=False,
        initial_question="What is shown?",
        messages_api=[
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 0:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx", "detail": "high"}},
                    {"type": "text", "text": "\n\nWhat is shown?"},
                ],
            },
            {"role": "assistant", "content": "<think>reasoning</think><answer>cat</answer>"},
        ],
        raw_prompt=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
            }
        ],
        original_images=[Image.new("RGB", (32, 24))],
        presented_image_refs=[
            {
                "presented_img_idx": 0,
                "kind": "initial",
                "source_original_img_idx": 0,
                "bbox_on_original": [0, 0, 32, 24],
                "display_size": [16, 12],
            }
        ],
        request_params={"model": "gpt-test"},
        loop_params={"initial_rescale": 0.5},
        sampling_params={"temperature": 1.0},
        tools_kwargs={},
        extra_info={"question": "What is shown?", "image_ori_wh": [(32, 24)]},
        failure_events=[],
        critical_failure=False,
        final_failure_reasons=None,
    )
    export_path = export_conversation(str(tmp_path), record, job_id="job-1")
    append_reward_info(export_path, {"reward": 1.0, "score": {"accuracy_reward": 1.0}})

    with open(export_path, encoding="utf-8") as f:
        written = json.load(f)

    assert written["conversation"][2]["type"] == "answer"
    assert written["reward"]["reward"] == 1.0
