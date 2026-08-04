# Initial Rescale Sweep: `longdocurl200_uncapped` + `mmlongbench200_uncapped` (`base`, no tool use)

Sweep outputs:
- [status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/status.tsv)
- [lane0.queue.log](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/lane0.queue.log)
- [lane1.queue.log](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/lane1.queue.log)

Input parquets:
- [longdocurl200_uncapped-insight_qwen_agent_no_tool.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool/longdocurl200_uncapped-insight_qwen_agent_no_tool.parquet)
- [mmlongbench200_uncapped-insight_qwen_agent_no_tool.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool/mmlongbench200_uncapped-insight_qwen_agent_no_tool.parquet)

Setup:
- model: `Qwen/Qwen3-VL-8B-Instruct`
- tool schema: disabled via `QWEN_TOOL_LIST=[]`
- prompt change: system message replaced with empty string
- benchmarks:
  - `longdocurl200_uncapped`
  - `mmlongbench200_uncapped`

Notes:
- scores below are validation `reward/mean@1`
- `avg` is the simple mean of the two benchmark scores
- this note summarizes the completed runs so far
- `0.7` is still running at the time of this snapshot

## Completed Results

| initial_rescale | longdocurl200 | mmlongbench200 | avg | longdocurl core_time_s | mmlongbench core_time_s | avg_core_time_s | prompt_tokens_mean | response_tokens_generated_mean | run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.175` | `0.295` | `0.240` | `0.2675` | `10.99` | `9.31` | `10.15` | `7989.48` | `320.24` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `0.25` | `0.375` | `0.250` | `0.3125` | `14.16` | `12.14` | `13.15` | `15274.79` | `247.57` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `0.35` | `0.510` | `0.360` | `0.4350` | `26.57` | `22.27` | `24.42` | `30112.62` | `191.43` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `0.5` | `0.555` | `0.415` | `0.4850` | `63.38` | `42.68` | `53.03` | `60404.81` | `174.52` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |

## Ranking So Far

| rank | initial_rescale | longdocurl200 | mmlongbench200 | avg |
|---:|---:|---:|---:|---:|
| 1 | `0.5` | `0.555` | `0.415` | `0.4850` |
| 2 | `0.35` | `0.510` | `0.360` | `0.4350` |
| 3 | `0.25` | `0.375` | `0.250` | `0.3125` |
| 4 | `0.175` | `0.295` | `0.240` | `0.2675` |

## Behavior Summary

Confirmed from the completed runs:
- `n_valid_tool_calls = 0.0` on both datasets for every completed row
- `response_tokens_tool = 0.0` on every completed row
- `acc/with_tool_use = nan`, as expected
- `acc/direct = reward`
- `num_turns = 2.0`

Per-slice answerability metrics:

| initial_rescale | longdocurl acc/answerable | longdocurl acc/not_answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable |
|---:|---:|---:|---:|---:|
| `0.175` | `0.2915` | `1.0` | `0.2468` | `0.2174` |
| `0.25` | `0.3719` | `1.0` | `0.2727` | `0.1739` |
| `0.35` | `0.5075` | `1.0` | `0.3961` | `0.2391` |
| `0.5` | `0.5528` | `1.0` | `0.4805` | `0.1957` |

## Current Read

- Accuracy is improving monotonically through `0.5` on both datasets.
- Latency is also increasing sharply with `initial_rescale`.
- Prompt length scales up quickly:
  - `~8k` at `0.175`
  - `~15k` at `0.25`
  - `~30k` at `0.35`
  - `~60k` at `0.5`
- Generated response length is going down as the initial resolution increases.

So the partial pattern is:
- higher initial resolution is helping direct-answer performance
- but the latency cost is large
- the current best completed no-tool point is `0.5`

## Pending

Still running at the time of this note:
- `base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07`

Once that finishes, this note should be updated with:
- the final `0.7` row
- the final ranking
- the final best setting for the no-tool regime
