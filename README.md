<p align="center">
  <img alt="InSight-doc" src="assets/insight_doc_logo.png" width="720" style="max-width: 100%;">
</p>

<h3 align="center">Agentic Visual Perception for Long-Document Understanding</h3>

<p align="center">
  <a href="https://example.com/insight-doc-paper"><b>Paper</b></a> |
  <a href="https://example.com/insight-doc-project"><b>Project Page</b></a> |
  <a href="https://example.com/insight-doc-8b"><b>Model</b></a> |
  <a href="https://example.com/insight-doc-sft-18k"><b>SFT Data</b></a> |
  <a href="https://example.com/insight-doc-rl-19k"><b>RL Data</b></a> |
  <a href="https://example.com/insight-doc-demo"><b>Demo</b></a>
</p>

<p align="center">
  <b>InSight-doc answers long-document questions by reading low-resolution pages first and zooming into high-resolution evidence regions only when needed.</b>
</p>

---

## Overview

Long-document VQA is expensive when every page is processed at high resolution, but aggressive downsampling can remove the evidence needed for fine-grained answers. **InSight-doc** is a retriever-free document VQA agent that treats visual resolution as an adaptive test-time resource: it starts from low-resolution page views, issues structured zoom-in tool calls, observes high-resolution crops, and answers from the accumulated evidence.

<p align="center">
  <img alt="InSight-doc teaser" src="assets/teaser.png" width="900" style="max-width: 100%;">
</p>

## Highlights

- Adaptive visual resolution through model-issued zoom-in calls.
- End-to-end SFT and RL recipes for Qwen3-VL document agents.
- Released artifacts: `InSight-doc-8B`, `InSight-doc-SFT-18k`, and `InSight-doc-RL-19k` links coming soon.
- Paper results: `InSight-doc-8B` improves document VQA accuracy by **4.3%-16.4%**, reduces long-document hallucination by **40%+**, and lowers latency by **41%-68%**.

<p align="center">
  <img alt="Long-document efficiency comparison" src="assets/longdoc_efficiency.png" width="650" style="max-width: 100%;">
</p>

## Install

```sh
git clone --recurse-submodules https://github.com/m-Just/InSight-doc.git
cd InSight-doc

pip install -e .
pip install -e ./verl
```

If you cloned without submodules:

```sh
git submodule update --init --recursive
```

The release was tested with Python 3.12, PyTorch 2.9.1, vLLM 0.14.0rc2, Transformers 4.57.6, Ray 2.53.0, flash-attn 2.8.3, and qwen-agent 0.0.31. CUDA-sensitive packages should be installed with wheels compatible with your cluster.

## Quick Evaluation

```sh
export MODEL_PATH=/path/to/InSight-doc-8B
export VAL_FILES='/path/to/longdocurl.parquet,/path/to/mmlongbench.parquet'
export RESCALES='0.25 0.35 0.5'
export EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://.../v1

bash scripts/evaluate_insight_doc.sh
```

Evaluation writes rollout records, exported conversations, and summary tables under `outputs/eval/` unless `OUTPUT_ROOT` is set. See [`docs/insight_doc_release.md`](docs/insight_doc_release.md) for the full configuration details.

## Quick Training

SFT:

```sh
TRAIN_FILES='[/path/to/InSight-doc-SFT-18k/train.parquet]' \
VAL_FILES='[/path/to/validation.parquet]' \
OUTPUT_ROOT=/path/to/runs/sft \
EXP_NAME=insight_doc_sft_qwen3vl8b \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/train_sft_qwen3vl_insight_doc.sh
```

RL:

```sh
MODEL_PATH=/path/to/sft_hf_checkpoint \
TRAIN_FILES='[/path/to/InSight-doc-RL-19k/train.parquet]' \
VAL_FILES='[/path/to/eval_a.parquet,/path/to/eval_b.parquet]' \
WORK_DIR=/path/to/runs/rl \
EXP_NAME=insight_doc_rl_qwen3vl8b \
OPENAI_API_KEY=... \
OPENAI_BASE_URL=https://.../v1 \
bash scripts/train_rl_qwen3vl_insight_doc.sh
```

Default SFT settings use max length 65,536, sequence parallel size 4, global batch size 32, frozen vision tower, cosine LR `5e-6 -> 5e-7`, and two epochs. The RL launcher starts from an SFT checkpoint and uses weighted refill source sampling with weights in `recipe/vsearch/config/insight_doc_rl_sampling_weights_release.yaml`.

## Repository Layout

```text
InSight-doc/
|-- insight_agent_core/  # Shared agent runtime, image handling, tools, prompt-length logic
|-- evals/               # Rollout backends, judging, metrics, resume, and export code
|-- recipe/              # Training configs and dataset construction utilities
|-- scripts/             # Public launchers and conversion/analysis utilities
|-- notebooks/           # Dataset and conversation viewers
|-- assets/              # README figures
`-- verl/                # Pinned VERL backend submodule with InSight-doc patches
```

`insight_agent_core/` is shared by the standalone evaluation path and the newer VERL wrapper agent. The released RL launcher still defaults to the legacy VERL `insight_qwen_agent` path to preserve the original checkpoint training setup; the `insight_qwen_agent_core` wrapper is included as a migration path for tighter training/evaluation alignment.

## License

Recommended release plan: use **Apache-2.0** for code. Model and dataset licenses should be finalized separately on Hugging Face after source-data constraints are checked.

## Citation

BibTeX will be added with the paper link.

```bibtex
@misc{insightdoc2026,
  title        = {InSight-doc: Agentic Visual Perception for Long-Document Understanding},
  author       = {TODO},
  year         = {2026},
  eprint       = {TODO},
  archivePrefix= {arXiv},
  url          = {https://example.com/insight-doc-paper}
}
```
