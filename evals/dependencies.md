# Evaluation Dependencies

The evaluator is intended to run against an already-available model endpoint or a local model served through the included Ray/vLLM helper.

## Local Source Dependencies

- This repository must be on `PYTHONPATH`.
- The pinned `verl/` submodule must be initialized for Ray/vLLM evaluation.
- No external companion source checkout is required.

## Python Package Dependencies

- `transformers`, `omegaconf`, `numpy`, `pandas`, `Pillow`, `tqdm`, and `pyarrow`.
- `qwen-agent>=0.0.31` is required for the legacy VERL tool-agent training path.
- Ray/vLLM packages are required only for the `ray_vllm` backend.
- The `openai` Python SDK is required for HTTPS model or judge backends.

## Runtime Inputs

- `MODEL_PATH` for local Ray/vLLM evaluation, or `MODEL_NAME` plus `OPENAI_BASE_URL`/`OPENAI_API_KEY` for HTTPS evaluation.
- `VAL_FILES` pointing to one or more evaluation parquet files.
- Agent/tool configuration YAMLs under `recipe/vsearch/config/`.
