• The standalone eval is not fully standalone yet. It still depends on several local repos/modules.

  Local Repo Dependencies

  - verl-qwen3-vl itself:
      - verl.utils.dataset.rl_dataset.RLHFDataset
      - verl.experimental.agent_loop.tool_parser.ToolParser
      - verl.utils.vreasoner_v2_conversation_export
      - verl.trainer.ppo.metric_utils
      - verl.utils.reward_score.vsearch_batch.compute_score_batch
      - Ray/vLLM server wrappers in verl.workers.rollout.*
  - /scratch/ywxzml3j/likaican/src/InSight-o3:
      - insight_agent_core
      - insight_agent_core.images
      - insight_o3.utils.api
      - insight_o3.utils.api_logger
  - /scratch/ywxzml3j/likaican/src/Qwen-Agent:
      - Added to sys.path; current standalone files do not directly import a module from it, but it may be needed
        transitively by insight_agent_core or old Qwen tooling.

  Python Package Dependencies

  - transformers
  - omegaconf
  - numpy
  - pandas
  - Pillow
  - tqdm
  - pyarrow preferred for parquet row-count metadata, with pandas.read_parquet fallback
  - Ray/vLLM stack for the ray_vllm backend
  - OpenAI-compatible async client stack through insight_o3.utils.api

  Model/Data Dependencies

  - Ray/vLLM backend needs a local/HF model path usable by AutoTokenizer and AutoProcessor.
  - Ray/vLLM backend needs an external Ray server manifest from scripts/serve_ray_vllm.py.
  - HTTPS backend needs OPENAI_BASE_URL or model config base_url, plus API key env.
  - Both backends need parquet eval files and image paths referenced inside them.

  The biggest remaining coupling is to verl for dataset loading, export format, tool parsing, reward judging, metrics, and
  Ray/vLLM server wrappers.

## 2026-07-03 Agent Core / Demo Runtime Refactor

This memo records the eval-pipeline-facing changes from the portable demo/runtime refactor.

### Summary

- `insight_agent_core` is now the shared portable scaffold for the agent loop, runtime protocol, standalone runtime, image helpers, config loading, and tool parsing.
- `agent_demo_runtime` depends on `insight_agent_core` and no longer imports `standalone_eval`.
- `standalone_eval` keeps the old public import paths for eval compatibility through wrappers:
  - `standalone_eval/config/agent.py` re-exports from `insight_agent_core.config`.
  - `standalone_eval/core/tool_parser.py` re-exports from `insight_agent_core.tool_parser`.
- Eval-specific generation transports were moved out of `insight_agent_core`:
  - `insight_agent_core/ray_vllm.py` -> `standalone_eval/backends/ray_vllm_endpoint.py`
  - `insight_agent_core/openai_https.py` -> `standalone_eval/backends/openai_https.py`
- `standalone_eval/backends/ray_vllm.py` now imports `RayVLLMEndpointPool` from `standalone_eval.backends.ray_vllm_endpoint`.
- `standalone_eval/backends/https_openai_chat.py` now imports `_image_to_data_url` from `standalone_eval.backends.openai_https`.
- The default agent config is packaged under `insight_agent_core/configs/`, but eval and demo code can still pass explicit agent config paths.
- Package metadata now includes `insight_agent_core/configs/*.yaml`.

### Dependency Boundary After Refactor

- Active demo export path:
  - `agent_demo_runtime/`
  - `insight_agent_core/`
- The active demo export path has no imports of:
  - `verl`
  - `standalone_eval`
  - `insight_o3`
  - hard-coded `/scratch` paths
- Eval-only backend dependencies remain inside `standalone_eval/backends/`:
  - Ray/vLLM transport still depends on `verl`.
  - HTTPS/OpenAI-compatible transport still depends on `insight_o3`.
- This refactor did not intentionally modify any `verl/` files.

### Validation Run

Commands run after the refactor:

```bash
python -m py_compile standalone_eval/backends/ray_vllm_endpoint.py standalone_eval/backends/openai_https.py standalone_eval/backends/ray_vllm.py standalone_eval/backends/https_openai_chat.py insight_agent_core/*.py agent_demo_runtime/*.py
python -m pytest tests/agent_demo_runtime/test_agent_demo_runtime.py -q
```

Additional checks:

- Importing `agent_demo_runtime` and `insight_agent_core` loaded no `verl`, `standalone_eval`, or `insight_o3` modules.
- Importing eval backend entry points resolved to the moved modules:
  - `standalone_eval.backends.ray_vllm_endpoint.RayVLLMEndpointPool`
  - `standalone_eval.backends.openai_https._image_to_data_url`
- Focused demo/runtime tests passed: `14 passed`.

### Backup

A backup archive was created before the refactor and copied into the workspace:

```text
artifacts/backups/verl_qwen3_vl_eval_pipeline_backup_20260703_agent_core_refactor.tar.gz
```
