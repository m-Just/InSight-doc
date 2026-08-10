<p align="center">
  <img alt="InSight-doc" src="assets/insight_doc_logo.png" width="650" style="max-width: 100%;">
</p>

<p align="center">
  <strong>Tool-Augmented Visual Search for High-Resolution Long-Document Understanding</strong>
</p>

<div align="center">

🤗 **[Models and Datasets](https://example.com/insight-doc-hf-collection)** |
📄 **[Paper](https://example.com/insight-doc-paper)** |
🚀 **[Demo](https://example.com/insight-doc-demo)** |
🌐 **[Project Page](https://example.com/insight-doc-project)**

</div>

## What's new

- [x] [2026/08/10] Initial release repository prepared with training, evaluation, visualization, and data-conversion utilities.
- [ ] Paper link coming soon.
- [ ] Model checkpoints coming soon.
- [ ] SFT/RL datasets coming soon.
- [ ] Interactive demos coming soon.

---

**Long documents are information-dense, high-resolution, and often too large to inspect at full fidelity in a single model call.**
InSight-doc is a document-understanding agent that combines a Qwen3-VL policy with an image zoom-in tool. The agent first reads compressed document pages, then selectively crops high-resolution regions before producing an answer.

In this repository, we release the code for:

- **Training** InSight-doc with supervised fine-tuning (SFT) and reinforcement learning (RL).
- **Evaluating** checkpoints with a local Ray/vLLM backend or OpenAI-compatible endpoints.
- **Exporting and visualizing** tool-use conversations for manual inspection.
- **Analyzing trajectory quality**, including crop counts, evidence-page hits, overlap, stuck-rate, and crop-area metrics when evidence annotations are available.

The code is organized as a lightweight release wrapper around a pinned [VERL](https://github.com/volcengine/verl) backend submodule. Release-facing agent and evaluation code lives at the repository root; backend training infrastructure lives under `verl/`.

## Repository layout

```text
InSight-doc/
|-- insight_agent_core/     # Shared agent runtime, image handling, tool parsing, prompt-length logic
|-- evals/                  # Rollout, Ray/vLLM backends, judging, metrics, and export code
|-- recipe/                 # Release training configs and dataset construction utilities
|-- scripts/                # Public launchers and analysis/conversion utilities
|-- notebooks/              # Notebook viewers for SFT/RL rows and exported conversations
|-- assets/                 # Logo and lightweight repository assets
`-- verl/                   # Pinned VERL backend submodule with InSight-doc integration patches
```

## Installation

Clone the repository with the pinned backend submodule:

```sh
git clone --recurse-submodules https://github.com/m-Just/InSight-doc.git
cd InSight-doc
```

If you cloned without submodules, initialize them with:

```sh
git submodule update --init --recursive
```

Install the top-level package and the VERL backend in your CUDA/vLLM environment:

```sh
pip install -e .
pip install -e ./verl
```

The launchers assume the following packages are available: `torch`, `ray`, `vllm`, `transformers`, `qwen-vl-utils`, `pyarrow`, `omegaconf`, and `openai`.
Depending on your environment, you may also need to expose optional companion code roots:

```sh
export INSIGHT_O3_ROOT=/path/to/InSight-o3        # optional, needed for legacy InSight-o3 paths
export QWEN_AGENT_ROOT=/path/to/Qwen-Agent        # optional, needed if Qwen-Agent is not installed
```

## Evaluation

Evaluate a local Hugging Face checkpoint with the release Ray/vLLM backend:

```sh
export MODEL_PATH=/path/to/hf_checkpoint
export VAL_FILES='/path/to/longdocurl.parquet,/path/to/mmlongbench.parquet'
export RESCALES='0.25 0.35 0.5'
export EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://.../v1

bash scripts/evaluate_insight_doc.sh
```

By default, evaluation uses:

- `evals/model_configs/release_ray_vllm.yaml` for local Ray/vLLM serving.
- `evals/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml` for the agent/tool configuration.
- `gpt-5-nano` through an OpenAI-compatible judge endpoint.

Outputs are written under `outputs/eval/` unless `OUTPUT_ROOT` is set. The evaluator exports both machine-readable scores and inspectable conversations:

- `samples*.jsonl`: rollout records with prompts, responses, tool calls, timing, and score fields.
- `exported_conversations/`: conversation JSON files with image references for manual inspection.
- `eval_summary*.tsv`: benchmark-level and trial-level summary tables.

For more detailed evaluation settings, see [`docs/insight_doc_release.md`](docs/insight_doc_release.md).

## Training

We provide separate launchers for SFT and RL. Both expect VERL-compatible parquet files and use the top-level wrapper scripts rather than scripts inside the `verl/` submodule.

### Supervised fine-tuning

The SFT launcher trains Qwen3-VL with full-parameter SFT, a frozen vision tower, and the same data fields used by the released SFT dataset.

```sh
TRAIN_FILES='[/path/to/sft_train.parquet]' \
VAL_FILES='[/path/to/sft_val.parquet]' \
OUTPUT_ROOT=/path/to/runs/sft \
EXP_NAME=insight_doc_sft_qwen3vl8b \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/train_sft_qwen3vl_insight_doc.sh
```

Default key settings: max sequence length 65,536, sequence parallel size 4, global batch size 32, cosine LR `5e-6 -> 5e-7`, and two epochs. Checkpoints are exported to:

```text
$OUTPUT_ROOT/$EXP_NAME/sft_checkpoints/global_step_*/huggingface
```

### Reinforcement learning

The RL launcher starts from an SFT checkpoint and trains the InSight Qwen agent loop with the image zoom-in tool and weighted refill source sampling.

```sh
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

For additional training details, see [`docs/insight_doc_release.md`](docs/insight_doc_release.md).

## Data

The released SFT and RL datasets will be published on Hugging Face:

- SFT dataset: [coming soon](https://example.com/insight-doc-sft-data)
- RL dataset: [coming soon](https://example.com/insight-doc-rl-data)

The training code expects parquet rows with VERL-compatible fields used in this repository, including multimodal `messages`, embedded `images`, `tools` where applicable, `data_source`, and task metadata in `extra_info`.

To inspect released or locally constructed data, open:

```sh
jupyter lab notebooks/visualize_converted_sft_parquet.ipynb
jupyter lab notebooks/visualize_rl_parquet.ipynb
```

## Useful utilities

```sh
python scripts/export_conversation_image_source_bundle.py --help
python scripts/evaluate_exported_conversation_trajectory_quality.py --help
python scripts/evaluate_sft_trajectory_quality.py --help
python scripts/eval_sweep.py --help
```

The notebook [`notebooks/visualize_vreasoner_v2_export.ipynb`](notebooks/visualize_vreasoner_v2_export.ipynb) can be used to browse exported conversations.

## Development notes

- `insight_agent_core/` is shared by the evaluation path and the newer VERL wrapper agent.
- The released RL launcher still defaults to the legacy VERL `insight_qwen_agent` rollout path to preserve the original checkpoint training setup.
- The `insight_qwen_agent_core` VERL wrapper is included as a migration path for tighter training/evaluation alignment.
- `verl/` should be treated as backend infrastructure. Release-facing scripts and configs live at the repository root.

## Citation

If you find this repository useful, please consider citing our paper:

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

License information for the InSight-doc release will be added before public release. The bundled VERL backend is provided as a submodule; see [`verl/LICENSE`](verl/LICENSE) for the VERL license.
