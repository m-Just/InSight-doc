# Evaluation Dependencies

The evaluator is intended to run against an already-available model endpoint or a local model served through the included Ray/vLLM helper.

## Local Source Dependencies

- This repository must be on `PYTHONPATH`.
- The pinned `verl/` submodule must be initialized for Ray/vLLM evaluation.
- No external companion source checkout is required.

## Python Package Dependencies

Install the package from the repository root:

```bash
pip install -e .
pip install -e ./verl
```

The base package includes the dependencies needed by the released evaluation,
training, conversion, and viewer utilities. CUDA-sensitive packages such as
`torch`, `vllm`, and `flash-attn` must still match your local driver/runtime.

## Runtime Inputs

- `MODEL_PATH` for local Ray/vLLM evaluation, or `MODEL_NAME` plus `OPENAI_BASE_URL`/`OPENAI_API_KEY` for HTTPS evaluation.
- `VAL_FILES` pointing to one or more evaluation parquet files.
- Agent/tool configuration YAMLs under `recipe/vsearch/config/`.
