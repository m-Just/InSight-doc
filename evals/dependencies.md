# Evaluation Dependencies

The evaluator is intended to run against an already-available model endpoint or a local model served through the included Ray/vLLM helper.

## Local Source Dependencies

- This repository must be on `PYTHONPATH`.
- The pinned `verl/` submodule must be initialized for Ray/vLLM evaluation.
- No external companion source checkout is required.

## Python Package Dependencies

Install the evaluation extra from the repository root:

```bash
pip install -e ".[eval]"
pip install -e ./verl
```

The `eval` extra includes `transformers`, `omegaconf`, `numpy`, `pandas`,
`Pillow`, `tqdm`, `pyarrow`, `openai`, `qwen-vl-utils`, `qwen-agent`, Ray,
and vLLM. CUDA-sensitive packages must still match your local driver/runtime.

## Runtime Inputs

- `MODEL_PATH` for local Ray/vLLM evaluation, or `MODEL_NAME` plus `OPENAI_BASE_URL`/`OPENAI_API_KEY` for HTTPS evaluation.
- `VAL_FILES` pointing to one or more evaluation parquet files.
- Agent/tool configuration YAMLs under `recipe/vsearch/config/`.
