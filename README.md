# InSight-Doc Release

This repository is a compact release wrapper for InSight-Doc training and evaluation. It keeps the relatively independent InSight agent/evaluation code in the main module and vendors the modified VERL training backend as a pinned `verl/` git submodule.

## Layout

- `insight_agent_core/`: shared InSight Qwen agent runtime, prompt construction, tool parsing, image handling, and prompt-length utilities.
- `evals/`: evaluation runner, model/backend configs, Ray/vLLM serving helpers, judging, resume/export, and metric aggregation.
- `recipe/`: release training configs and parquet-construction utilities used by the launchers.
- `scripts/`: public launchers plus inspection, conversion, packing, and quality-analysis utilities.
- `notebooks/`: lightweight notebooks for inspecting SFT/RL rows and exported conversations.
- `verl/`: pinned VERL submodule containing the modified trainer, dataset, rollout, and reward infrastructure.

## Clone

```bash
git clone --recurse-submodules <repo-url> InSight-doc
cd InSight-doc
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The wrapper scripts default to `VERL_ROOT=$REPO_ROOT/verl`. Override `VERL_ROOT` only if you intentionally want to run against a different local VERL checkout.

## Main Entry Points

- `scripts/train_sft_qwen3vl_insight_doc.sh`: full-parameter Qwen3-VL SFT launcher.
- `scripts/train_rl_qwen3vl_insight_doc.sh`: InSight-Doc RL launcher with weighted refill source sampling.
- `scripts/evaluate_insight_doc.sh`: rollout plus judge evaluation over one or more eval parquets.

See `docs/insight_doc_release.md` for the concrete environment variables and example commands.

## Submodule URL

The `verl/` submodule is pinned by git commit and currently points to `git@github.com:m-Just/verl-public.git`. If the backend release moves to a different repository, update `.gitmodules` before publishing the wrapper repository.
