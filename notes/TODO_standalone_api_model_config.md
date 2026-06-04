# TODO: Standalone API Model Config Cleanup

Current issue: `evaluate.py --generation-backend https_openai_chat` still requires
`--model-path` for the local tokenizer/processor, even when generation is done by
an API model such as `gpt-5-nano`.

This is confusing because `--model-path` is not the generation model in this
mode. It is the local tokenizer/processor reference used by the dataset and
agent pipeline.

Follow-up cleanup:
- Split generation model identity from local processor/tokenizer identity.
- Make API model configs explicit, e.g. `--api-model gpt-5-nano` plus an
  optional `--processor-model Qwen/Qwen3-VL-8B-Instruct`.
- Avoid requiring a local HF model path for API-only evals unless the processor
  path is truly needed.
