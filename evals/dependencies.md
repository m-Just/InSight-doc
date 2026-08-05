# Evaluation Dependencies

The evaluator is intended to run against an already-available model endpoint or a local model served through the included Ray/vLLM helper.

## Local Source Dependencies

- This repository must be on `PYTHONPATH`.
- If `insight_agent_core` / InSight-o3 utilities are checked out separately, set `INSIGHT_O3_ROOT=/path/to/InSight-o3`.
- If Qwen-Agent utilities are required by your environment, set `QWEN_AGENT_ROOT=/path/to/Qwen-Agent`.

## Python Package Dependencies

- `transformers`, `omegaconf`, `numpy`, `pandas`, `Pillow`, `tqdm`, and `pyarrow`.
- Ray/vLLM packages are required only for the `ray_vllm` backend.
- An OpenAI-compatible client stack is required only for HTTPS model or judge backends.

## Runtime Inputs

- `MODEL_PATH` for local Ray/vLLM evaluation, or `MODEL_NAME` plus `OPENAI_BASE_URL`/`OPENAI_API_KEY` for HTTPS evaluation.
- `VAL_FILES` pointing to one or more evaluation parquet files.
- Agent/tool configuration YAMLs under `recipe/vsearch/config/`.
