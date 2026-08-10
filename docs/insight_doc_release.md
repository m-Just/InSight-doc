# InSight-Doc Release Recipes

This repository is a small InSight-Doc release wrapper around a pinned `verl/`
submodule. The public surface is intentionally small: one SFT launcher, one RL
launcher, one evaluation launcher, and a few inspection/export tools.

## Environment

Install the release package, then install the pinned VERL backend submodule:

```bash
pip install -e .
pip install -e ./verl
```

The launchers assume that CUDA-compatible `torch`, `ray`, `vllm`, `flash-attn`,
`transformers`, `qwen-vl-utils`, `pyarrow`, `omegaconf`, `openai`, and
`qwen-agent` are available. The release uses a local OpenAI-compatible helper
and does not require external companion source checkouts.

```bash
export OPENAI_API_KEY=...                         # required for judge/reward
export OPENAI_BASE_URL=https://.../v1             # OpenAI-compatible endpoint
```

## SFT Training

The released SFT checkpoint `sft_v2_ckpt1118` was trained as full-parameter SFT
from `Qwen/Qwen3-VL-8B-Instruct`, with the vision tower frozen, sequence
parallelism 4, max sequence length 65,536, global batch size 32, cosine LR
`5e-6 -> 5e-7`, and two epochs.

Use released SFT-format parquet files with at least `messages` and `tools`
columns. If the parquet has a `message_loss_mask` column, the launcher uses it
by default to match the original training path; set `MESSAGE_LOSS_MASK_KEY=` to
train on all assistant messages.

```bash
TRAIN_FILES='[/path/to/sft_train.parquet]' \
VAL_FILES='[/path/to/sft_val.parquet]' \
OUTPUT_ROOT=/path/to/runs/sft \
EXP_NAME=insight_doc_sft_qwen3vl8b \
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/train_sft_qwen3vl_insight_doc.sh
```

The resulting HF checkpoint is written under:

```text
$OUTPUT_ROOT/$EXP_NAME/sft_checkpoints/global_step_*/huggingface
```

## RL Training

The released RL checkpoint
`rl_v4_unans014_mc_false_e05_arxiv_struct1k_legacy_prompt_v2_ckpt800` starts
from the released SFT checkpoint and trains with the InSight Qwen agent loop,
the image zoom-in tool, weighted refill source sampling, temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5, and 2,000 total RL steps. The final
sampling weights are in:

```text
recipe/vsearch/config/insight_doc_rl_sampling_weights_release.yaml
```

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

Checkpoints are written under:

```text
$WORK_DIR/ckpts/insight_doc/$EXP_NAME
```

## Evaluation

The evaluator assumes the model is served through the included
Ray/vLLM server wrapper or an OpenAI-compatible HTTPS endpoint. For local HF
checkpoints, set `MODEL_PATH` and use the default release model config.

```bash
MODEL_PATH=/path/to/hf_checkpoint \
VAL_FILES='/path/to/dude.parquet,/path/to/longdocurl.parquet,/path/to/mmlongbench.parquet' \
RESCALES='0.25 0.35 0.5' \
EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENAI_API_KEY=... \
OPENAI_BASE_URL=https://.../v1 \
bash scripts/evaluate_insight_doc.sh
```

Important vLLM defaults are in `evals/model_configs/release_ray_vllm.yaml`:
4 replicas, 1 GPU per replica, `max_model_len=262144`, `max_num_seqs=64`,
chunked prefill enabled, prefix caching enabled, and the same sampling settings
used by RL validation.

## Caveats and Future Work

The evaluator uses the extracted `insight_agent_core` runner by
default. The RL reward/judge path is shared with evaluation through
`verl/utils/reward_score/vsearch_batch.py`, but the released RL launcher still
uses the legacy VERL rollout agent loop (`insight_qwen_agent`) by default. A
VERL wrapper for the extracted core agent (`insight_qwen_agent_core`) is included,
but fully aligning RL rollout execution with evaluation requires
switching both `actor_rollout_ref.rollout.agent.default_agent_loop` and the RL
parquet `agent_name` values to `insight_qwen_agent_core`. This migration is left
as future work so the released training recipe preserves the checkpoint's
original training path.

## Useful Utilities

- `scripts/pack_exported_conversation_viewer.py`: packs exported
  conversations with source images and a portable viewer notebook.
- `scripts/evaluate_exported_conversation_trajectory_quality.py`: computes crop
  count, evidence-page/region hits, overlap, stuck-rate, and crop-area metrics
  when evidence metadata is available.
- `scripts/evaluate_sft_trajectory_quality.py`: computes analogous metrics for
  SFT parquets.
- `notebooks/visualize_exported_conversations.ipynb`: browse exported eval
  conversations.

Historical one-off experiment launchers are intentionally not part of the public
API. Use the release launchers above for reproducible runs.
