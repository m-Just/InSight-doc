from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _dedup_vllm_multimodal_placeholder_tokens,
    _qwen2_5_vl_dedup_image_tokens,
)


def _processor_with_image_processor(image_processor_name: str):
    image_processor_cls = type(image_processor_name, (), {})
    processor_cls = type("Processor", (), {})
    processor = processor_cls()
    processor.image_processor = image_processor_cls()
    processor.image_token_id = 10
    processor.video_token_id = 20
    return processor


def test_dedup_vllm_multimodal_placeholder_tokens_supports_qwen2vl_fast_processor():
    processor = _processor_with_image_processor("Qwen2VLImageProcessorFast")

    prompt_ids = [1, 10, 10, 10, 2, 20, 20, 3, 10, 4]

    assert _dedup_vllm_multimodal_placeholder_tokens(prompt_ids, processor) == [1, 10, 2, 20, 3, 10, 4]


def test_dedup_vllm_multimodal_placeholder_tokens_supports_qwen3vl_processor_prefixes():
    processor = _processor_with_image_processor("Qwen3VLImageProcessorFast")

    prompt_ids = [1, 10, 10, 2]

    assert _dedup_vllm_multimodal_placeholder_tokens(prompt_ids, processor) == [1, 10, 2]


def test_dedup_vllm_multimodal_placeholder_tokens_leaves_other_processors_unchanged():
    processor = _processor_with_image_processor("OtherImageProcessor")
    prompt_ids = [1, 10, 10, 2]

    assert _dedup_vllm_multimodal_placeholder_tokens(prompt_ids, processor) == prompt_ids


def test_dedup_vllm_multimodal_placeholder_tokens_does_not_match_non_prefix_names():
    processor = _processor_with_image_processor("WrappedQwen2VLImageProcessor")
    prompt_ids = [1, 10, 10, 2]

    assert _dedup_vllm_multimodal_placeholder_tokens(prompt_ids, processor) == prompt_ids


def test_qwen2_5_vl_dedup_alias_keeps_existing_imports_working():
    assert _qwen2_5_vl_dedup_image_tokens is _dedup_vllm_multimodal_placeholder_tokens
