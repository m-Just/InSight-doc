# Standalone API Model Config Cleanup

Status: resolved by the split standalone rollout path.

Current behavior: `standalone_eval/rollout.py --model-config ...` supports
`backend: https_openai_chat` without a local tokenizer or processor path. The
API model name is the top-level `model` field in the model config, for example
`model: gpt-5-nano`.

Local tokenizer/processor loading is only used by `backend: ray_vllm`, where it
defaults to the same top-level HF model path.
