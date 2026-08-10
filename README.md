<p align="center">
  <img src="assets/insight_doc_logo.png" alt="InSight-doc" width="720">
</p>

<p align="center">
  <a href="https://example.com/insight-doc-paper"><img src="https://img.shields.io/badge/Paper-coming%20soon-blue" alt="Paper"></a>
  <a href="https://example.com/insight-doc-model"><img src="https://img.shields.io/badge/Model-coming%20soon-5c7cfa" alt="Model checkpoint"></a>
  <a href="https://example.com/insight-doc-data"><img src="https://img.shields.io/badge/Datasets-coming%20soon-00a884" alt="Datasets"></a>
  <a href="https://example.com/insight-doc-demo"><img src="https://img.shields.io/badge/Demo-coming%20soon-purple" alt="Demo"></a>
</p>

# InSight-doc

InSight-doc is a document-understanding agent for high-resolution, long-context visual question answering. It combines a Qwen3-VL policy with an image zoom-in tool so the model can first inspect compressed document pages, then crop relevant regions before answering.

This repository packages the release code for:

- SFT and RL training of the InSight-doc agent.
- Evaluation with local Ray/vLLM serving or OpenAI-compatible endpoints.
- Conversation export, visualization, and trajectory-quality analysis.
- Shared agent logic in `insight_agent_core/`, with a pinned VERL backend in `verl/`.

## Resources

| Resource | Link |
| --- | --- |
| Paper | [Coming soon](https://example.com/insight-doc-paper) |
| Project page | [Coming soon](https://example.com/insight-doc-project) |
| Demo | [Coming soon](https://example.com/insight-doc-demo) |
| Model checkpoint | [Coming soon](https://example.com/insight-doc-model) |
| SFT dataset | [Coming soon](https://example.com/insight-doc-sft-data) |
| RL dataset | [Coming soon](https://example.com/insight-doc-rl-data) |

## What Is Included

```text
InSight-doc/
|-- insight_agent_core/     # Shared agent runtime, image handling, tool parsing, prompt-length logic
|-- evals/                  # Rollout, vLLM/Ray backends, judging, metrics, and export code
|-- recipe/                 # Release training configs and dataset construction utilities
|-- scripts/                # Public launchers and analysis/conversion utilities
|-- notebooks/              # Notebook viewers for SFT/RL rows and exported conversations
|-- assets/                 # Logo and lightweight repository assets
`-- verl/                   # Pinned VERL backend submodule with InSight-doc integration patches
```

The top-level package is the intended public interface. The `verl/` submodule contains the modified training backend and is pinned to the commit used by this release.

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/m-Just/InSight-doc.git
cd InSight-doc
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

Install the Python packages in your CUDA/vLLM environment:

```bash
pip install -e .
pip install -e ./verl
```

The launchers assume that `torchrun`, `ray`, `vllm`, `transformers`, `qwen-vl-utils`, `pyarrow`, `omegaconf`, and `openai` are available. Depending on your setup, you may also need optional companion code roots:

```bash
export INSIGHT_O3_ROOT=/path/to/InSight-o3        # optional, needed for legacy InSight-o3 paths
export QWEN_AGENT_ROOT=/path/to/Qwen-Agent        # optional, needed if not installed as a package
```

## Quick Evaluation

Evaluate a local Hugging Face checkpoint with the release Ray/vLLM backend:

```bash
export MODEL_PATH=/path/to/hf_checkpoint
export VAL_FILES='/path/to/longdocurl.parquet,/path/to/mmlongbench.parquet'
export RESCALES='0.25 0.35 0.5'
export EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://.../v1

bash scripts/evaluate_insight_doc.sh
```

By default this uses:

- `evals/model_configs/release_ray_vllm.yaml` for local Ray/vLLM serving.
- `evals/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml` for the agent/tool configuration.
- `gpt-5-nano` through an OpenAI-compatible judge endpoint.

Generated conversations, scores, and sweep summaries are written under `outputs/eval/` unless `OUTPUT_ROOT` is set.

## SFT Training

The release SFT launcher trains Qwen3-VL with full-parameter SFT, a frozen vision tower, and the same data fields used by the released SFT dataset.

```bash
TRAIN_FILES='[/path/to/sft_train.parquet]' \
VAL_FILES='[/path/to/sft_val.parquet]' \
OUTPUT_ROOT=/path/to/runs/sft \
EXP_NAME=insight_doc_sft_qwen3vl8b \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/train_sft_qwen3vl_insight_doc.sh
```

The default key settings are max sequence length 65,536, sequence parallel size 4, global batch size 32, cosine LR `5e-6 -> 5e-7`, and two epochs. Checkpoints are exported to:

```text
$OUTPUT_ROOT/$EXP_NAME/sft_checkpoints/global_step_*/huggingface
```

## RL Training

The RL launcher starts from an SFT checkpoint and trains the InSight Qwen agent loop with the image zoom-in tool and weighted refill source sampling.

```bash
MODEL_PATH=/path/to/sft_hf_checkpoint \
TRAIN_FILES='[/path/to/rl_train.parquet]' \
VAL_FILES='[/path/to/eval_a.parquet,/path/to/eval_b.parquet]' \
WORK_DIR=/path/to/runs/rl \
EXP_NAME=insight_doc_rl_qwen3vl8b \
OPENAI_API_KEY=... \
OPENAI_BASE_URL=https://.../v1 \
bash scripts/train_rl_qwen3vl_insight_doc.sh
```

The release sampling weights are stored in:

```text
recipe/vsearch/config/insight_doc_rl_sampling_weights_release.yaml
```

See `docs/insight_doc_release.md` for more detailed training and evaluation settings.

## Data Format

The released SFT and RL datasets will be published as Hugging Face datasets. The training code expects parquet rows with the standard VERL-compatible fields used in this repository, including multimodal `messages`, embedded `images`, `tools` where applicable, `data_source`, and task metadata in `extra_info`.

For inspection and debugging, use:

```bash
jupyter lab notebooks/visualize_converted_sft_parquet.ipynb
jupyter lab notebooks/visualize_rl_parquet.ipynb
```

## Evaluation Outputs

Evaluation produces both machine-readable metrics and inspectable conversations:

- `samples*.jsonl`: rollout records with prompts, responses, tool calls, timing, and score fields.
- `exported_conversations/`: conversation JSON files with image references for manual inspection.
- `eval_summary*.tsv`: benchmark-level and trial-level summary tables.

Useful utilities:

```bash
python scripts/export_conversation_image_source_bundle.py --help
python scripts/evaluate_exported_conversation_trajectory_quality.py --help
python scripts/evaluate_sft_trajectory_quality.py --help
```

## Development Notes

- The top-level `insight_agent_core/` is shared by the eval path and the newer VERL wrapper agent.
- The released RL launcher still defaults to the legacy VERL `insight_qwen_agent` rollout path to preserve the checkpoint training setup.
- The `insight_qwen_agent_core` VERL wrapper is included as a migration path for tighter training/evaluation alignment.
- The `verl/` submodule should be treated as backend infrastructure; release-facing scripts and configs live at the repository root.

## Citation

```bibtex
@article{insightdoc2026,
  title   = {InSight-doc: Tool-Augmented Document Understanding with High-Resolution Visual Search},
  author  = {TODO},
  journal = {TODO},
  year    = {2026},
  url     = {https://example.com/insight-doc-paper}
}
```

## License

License information for the InSight-doc release will be added before public release. The bundled VERL backend is provided as a submodule; see `verl/LICENSE` for the VERL license.
