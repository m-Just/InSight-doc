
- the previous SFT got past the first epoch with MAX_LENGTH=32768 without any error (there are samples exceeding
  this length). how?
    -  Answer: with pad_mode=no_padding, truncation=error is ignored

- TODO(export/conversion): fix runtime-hint typing bugs in vreasoner/vsearch conversation export.
    - Bug 1: malformed assistant turn + repair prompt is sometimes exported as assistant=others followed by user=others instead of user=format_repair_hint, which prevents stitch_runtime_hints from removing the repair scaffolding.
    - Bug 2: tool failure hint is sometimes exported as user=others instead of user=tool_result_fail_hint.
    - Current impact from /scratch/ywxzml3j/likaican/mms1_rl/exported_conversations: about 2620 / 318363 files for Bug 1 (~0.82%), and 3 / 318363 files for Bug 2 (~0.001%).
    - Safe near-term plan: fix exporter typing for future runs; add converter-side recovery for Bug 1 on old exports. Keep Bug 2 conservative unless we explicitly decide to salvage failed-tool episodes.

- TODO(training): debug activation offload for Qwen3-VL LoRA SFT.
    - Symptom: `model.enable_activation_offload=True` crashes the 40k LoRA medium data-ablation sweep with `KeyError: 62` in `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/verl/utils/activation_offload.py`, inside `synchronize_on_group_commit_forward` when indexing `self.layer_window_map[self.offloaded_group_count]`.
    - Failed run root: `/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_40k_4gpu_lanes_wandb_actoff_continue_20260513_204803`. All four experiments failed; `CONTINUE_ON_FAILURE=1` worked but activation offload itself is invalid for this path.
    - Likely cause: offload group bookkeeping reaches `offloaded_group_count == num_offload_group` during an extra wrapped forward commit. Need a small guarded patch plus a smoke test before using activation offload in sweeps.

- TODO(training): debug `data.use_dynamic_bsz=True` for jagged multimodal SFT batches.
    - Symptom: the 48k LoRA medium data-ablation sweep with dynamic batching failed before step 1 in `rearrange_micro_batches -> tu.index_select_tensor_dict -> tensor.unbind()`, with a PyTorch nested-tensor `Expected cond to be True` runtime error.
    - Failed run root: `/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_48k_4gpu_lanes_wandb_dynbsz_20260513_201206`.
    - Likely cause: dynamic micro-batch repartitioning does not correctly handle 3D jagged `position_ids` / nested tensors from the Qwen3-VL no-padding multimodal dataset. Need to special-case jagged tensors similarly to the existing `chunk_tensordict` workaround before relying on dynamic batch sizing.

- TODO(data/training): decide how to handle O3 high-DPI samples longer than 40k tokens.
    - Symptom: the 40k no-activation-offload sweep failed `O3_w_higher_dpi` after a few steps with `ValueError: sequence_length=41752 is larger than self.max_length=40960` from `verl/utils/dataset/multiturn_sft_dataset.py`.
    - Run root: `/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_40k_4gpu_lanes_wandb_noactoff_continue_20260513_205620`.
    - Options: keep 48k/64k for O3 high-DPI mixtures, filter/drop overlength rows for 40k runs, or add a deterministic downsampling/truncation policy. This is separate from OOM and activation offload.

- TODO(infra/rollout): fix multi-GPU vLLM rollout startup on non-contiguous `CUDA_VISIBLE_DEVICES` when tensor parallelism > 1.
    - Symptom: 32B filtering runs fail during vLLM engine init on lanes like `CUDA_VISIBLE_DEVICES=0,5,6,7` with either `invalid device ordinal`, `DP adjusted local rank 5 is out of bounds`, or `local_world_size (4) must be less than or equal to the number of visible devices (1)` depending on whether `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` is enabled.
    - Root cause: the rollout path mixes physical GPU ids, Ray accelerator ids, and torch/vLLM visible-device ordinals. With `tensor_model_parallel_size > 1`, the vLLM server actor must see all TP GPUs, which effectively requires `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`; but in that mode the code currently assumes Ray accelerator ids can be used directly as local ranks.
    - Relevant paths: `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/verl/single_controller/base/worker.py`, and `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/construct_synthetic_unanswerable_cot_dataset.sh`.
    - Safe fix: centralize physical-id -> visible-ordinal mapping from `CUDA_VISIBLE_DEVICES`, apply it anywhere Ray accelerator ids are converted into torch/vLLM local ranks, and automatically require no-set mode for TP>1 rollout actors.

- TODO(ans eval): the judge is often too strict on unanswerable questions

- TODO(ans eval): align answer-correctness judging between InSight `evaluate.py` and VERL reward scoring.
    - Current mismatch: `/scratch/ywxzml3j/likaican/src/InSight-doc/insight_o3/scripts/evaluate.py` special-cases MCQ by exact option-letter match after extraction, while `/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/verl/utils/reward_score/vsearch_batch.py` routes extracted answers through generic normalized/LLM correctness judging.
    - Current mismatch: VERL normalizes extraction-judge output by unwrapping `<answer>...</answer>`, `\boxed{...}`, and backticks; InSight eval mostly uses extraction output as-is.
    - Current mismatch: VERL can fall back to normalized exact/substring matching when no judge endpoint is configured, and retries judge failures in batch; InSight eval marks extraction/judge failures as invalid records.
    - Current mismatch: VERL checks literal `<correct>` in the judge response; InSight eval checks `"correct" in judge_content.lower()`, which is more permissive and can false-positive if the judge returns unexpected text such as `incorrect`.
    - Decide the intended canonical behavior, then share the extraction normalization and correctness parser so offline eval metrics and training rewards agree.

- TODO(training): clean up PEFT LoRA load warnings under FSDP meta initialization.
    - Symptom: nonzero ranks emit many warnings like `copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op` for LoRA weights during `PeftModel.from_pretrained(...)`.
    - Current assessment: likely benign for PPO resume. Nonzero ranks initialize the base model on `meta`; PEFT adapter loading into those meta LoRA params warns; then FSDP wraps with `sync_module_states=True`, and resume loads the sharded actor checkpoint over the PEFT-wrapped module.
    - Potential cleanup: pass `low_cpu_mem_usage=True` to `PeftModel.from_pretrained(...)` so PEFT uses `load_state_dict(..., assign=True)` for meta tensors. Validate first with a short LoRA RL resume smoke test and by checking no LoRA params remain on `meta` after FSDP init/checkpoint load.


› I currently have two inference+eval stacks:
  - one is scattered inside the verl codebase. e.g., some inference code is at verl/experimental/agent_loop/qwen_agent_loop.py and
  some eval code is at verl/utils/reward_score/vsearch_batch.py (example launch script: /scratch/ywxzml3j/likaican/src/InSight-doc/
  verl/scripts/run_iq_ft_eval_default_sampling_rl15360.sh)
  - one is a standalone stack under /scratch/ywxzml3j/likaican/src/InSight-doc/scripts (example launch script: /scratch/ywxzml3j/
  likaican/src/InSight-doc/scripts/qwen3_vl_test_0501.sh)

  I want to make sure they are aligned as much as possible in terms of:
  - Image preprocessing
  - API request handling
  - Evaluation judge logic


› the thing i want is: say i have trained a model with verl under a certain configuration (e.g., /scratch/ywxzml3j/likaican/src/
  InSight-doc/verl/recipe/vsearch/train_insight_qwen_agent_rl.t0_7.insight_doc_rl_balanced_dude_reduced.new_data.sh) and i want to
  evaluate the model via the standalone path, i want most of the config stays the same as training but i don't want to respecify them.
