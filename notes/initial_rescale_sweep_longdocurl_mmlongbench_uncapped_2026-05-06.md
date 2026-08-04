# Initial Rescale Sweep: `longdocurl200_uncapped` + `mmlongbench200_uncapped`

Sweep outputs:
- [status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/status.tsv)
- [summary.txt](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/summary.txt)

Input parquets:
- [longdocurl200_uncapped-insight_qwen_agent.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/longdocurl200_uncapped-insight_qwen_agent.parquet)
- [mmlongbench200_uncapped-insight_qwen_agent.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/mmlongbench200_uncapped-insight_qwen_agent.parquet)

Scope:
- models:
  - `base`
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch`
  - `base_no_tool`
  - `base_no_tool_no_system`
  - `base_no_tool_no_system_answer_only`
- `initial_rescale`:
  - `0.175`
  - `0.25`
  - `0.35`
  - `0.5`
  - `0.7`
- benchmarks:
  - `longdocurl200_uncapped`
  - `mmlongbench200_uncapped`

Notes:
- scores below are validation `reward/mean@1`
- `avg` is the simple mean of the two benchmark scores
- original sweep had `8 / 10` successes
- the two `0.7` cells were rerun after adding initial-prompt shrink fallback
- final result set below is now complete at `10 / 10`
- additional no-tool variants are merged below and clearly labeled

## Best By Model

| model | best initial_rescale | longdocurl200 | mmlongbench200 | avg | run |
|---|---:|---:|---:|---:|---|
| `base_no_tool` | `0.7` | `0.590` | `0.455` | `0.5225` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base_no_tool_no_system` | `0.5` | `0.575` | `0.440` | `0.5075` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool_no_system_answer_only` | `0.7` | `0.580` | `0.390` | `0.4850` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base` | `0.7` | `0.545` | `0.460` | `0.5025` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07_r2) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.560` | `0.455` | `0.5075` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |

## Overall Ranking

Completed rows only.

| rank | model | initial_rescale | longdocurl200 | mmlongbench200 | avg | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | `base_no_tool` | `0.7` | `0.590` | `0.455` | `0.5225` | `success` |
| 2 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.560` | `0.455` | `0.5075` | `success` |
| 3 | `base_no_tool_no_system` | `0.5` | `0.575` | `0.440` | `0.5075` | `success` |
| 4 | `base_no_tool_no_system` | `0.7` | `0.565` | `0.445` | `0.5050` | `success` |
| 5 | `base` | `0.7` | `0.545` | `0.460` | `0.5025` | `success` |
| 6 | `base_no_tool` | `0.5` | `0.555` | `0.415` | `0.4850` | `success` |
| 7 | `base_no_tool_no_system_answer_only` | `0.7` | `0.580` | `0.390` | `0.4850` | `success` |
| 8 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.550` | `0.400` | `0.4750` | `success` |
| 9 | `base` | `0.5` | `0.470` | `0.475` | `0.4725` | `success` |
| 10 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.510` | `0.415` | `0.4625` | `success` |
| 11 | `base_no_tool_no_system_answer_only` | `0.5` | `0.500` | `0.390` | `0.4450` | `success` |
| 12 | `base_no_tool_no_system` | `0.35` | `0.505` | `0.380` | `0.4425` | `success` |
| 13 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.500` | `0.375` | `0.4375` | `success` |
| 14 | `base_no_tool` | `0.35` | `0.510` | `0.360` | `0.4350` | `success` |
| 15 | `base_no_tool_no_system_answer_only` | `0.35` | `0.475` | `0.295` | `0.3850` | `success` |
| 16 | `base` | `0.35` | `0.390` | `0.355` | `0.3725` | `success` |
| 17 | `base_no_tool_no_system` | `0.25` | `0.420` | `0.290` | `0.3550` | `success` |
| 18 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.400` | `0.255` | `0.3275` | `success` |
| 19 | `base_no_tool` | `0.25` | `0.375` | `0.250` | `0.3125` | `success` |
| 20 | `base_no_tool_no_system_answer_only` | `0.25` | `0.380` | `0.245` | `0.3125` | `success` |
| 21 | `base` | `0.25` | `0.310` | `0.310` | `0.3100` | `success` |
| 22 | `base_no_tool` | `0.175` | `0.295` | `0.240` | `0.2675` | `success` |
| 23 | `base_no_tool_no_system` | `0.175` | `0.295` | `0.225` | `0.2600` | `success` |
| 24 | `base` | `0.175` | `0.185` | `0.280` | `0.2325` | `success` |
| 25 | `base_no_tool_no_system_answer_only` | `0.175` | `0.260` | `0.185` | `0.2225` | `success` |

## Core Inference Time Comparison

`avg_core_time_s` below is the simple mean of:
- `longdocurl200 core_inference_time`
- `mmlongbench200 core_inference_time`

### Best-Scoring Config Per Model

| model | best initial_rescale | avg | avg_core_time_s | longdocurl core_time_s | mmlongbench core_time_s |
|---|---:|---:|---:|---:|---:|
| `base_no_tool` | `0.7` | `0.5225` | `112.44` | `121.68` | `103.20` |
| `base_no_tool_no_system` | `0.5` | `0.5075` | `53.05` | `59.20` | `46.89` |
| `base_no_tool_no_system_answer_only` | `0.7` | `0.4850` | `102.35` | `113.72` | `90.99` |
| `base` | `0.7` | `0.5025` | `171.35` | `178.93` | `163.77` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.5075` | `87.16` | `89.66` | `84.67` |

### Fastest Config Per Model

| model | fastest initial_rescale | avg | avg_core_time_s |
|---|---:|---:|---:|
| `base_no_tool` | `0.175` | `0.2675` | `10.15` |
| `base_no_tool_no_system` | `0.175` | `0.2600` | `10.91` |
| `base_no_tool_no_system_answer_only` | `0.175` | `0.2225` | `8.25` |
| `base` | `0.175` | `0.2325` | `32.92` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.3275` | `18.00` |

### Practical Tradeoff

- On the uncapped datasets, the strongest settings are now split across five regimes:
  - `base_no_tool`: `0.7`
  - `base_no_tool_no_system`: `0.5`
  - `base_no_tool_no_system_answer_only`: `0.7`
  - `base`: `0.7`
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch`: `0.5`
- Removing the empty system block helps the no-tool baseline at moderate-to-high rescale.
- Adding `answer_only` drastically reduces response length, but costs accuracy on these two benchmarks.
- Best overall successful run remains:
  - `base_no_tool @ 0.7`
  - `avg = 0.5225`
  - `avg_core_time_s = 112.44`
- Best no-system run:
  - `base_no_tool_no_system @ 0.5`
  - `avg = 0.5075`
  - `avg_core_time_s = 53.04`
- Best no-system + answer-only run:
  - `base_no_tool_no_system_answer_only @ 0.7`
  - `avg = 0.4850`
  - `avg_core_time_s = 102.35`

## Full Results

| model | initial_rescale | longdocurl200 | mmlongbench200 | avg | longdocurl acc/answerable | longdocurl acc/not_answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable | run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base_no_tool` | `0.175` | `0.295` | `0.240` | `0.2675` | `0.2915` | `1.0` | `0.2468` | `0.2174` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool` | `0.25` | `0.375` | `0.250` | `0.3125` | `0.3719` | `1.0` | `0.2727` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool` | `0.35` | `0.510` | `0.360` | `0.4350` | `0.5075` | `1.0` | `0.3961` | `0.2391` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool` | `0.5` | `0.555` | `0.415` | `0.4850` | `0.5528` | `1.0` | `0.4805` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool` | `0.7` | `0.590` | `0.455` | `0.5225` | `0.5879` | `1.0` | `0.5325` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base_no_tool_no_system` | `0.175` | `0.295` | `0.225` | `0.2600` | `0.2915` | `1.0` | `0.2208` | `0.2391` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool_no_system` | `0.25` | `0.420` | `0.290` | `0.3550` | `0.4171` | `1.0` | `0.2987` | `0.2609` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool_no_system` | `0.35` | `0.505` | `0.380` | `0.4425` | `0.5025` | `1.0` | `0.4351` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool_no_system` | `0.5` | `0.575` | `0.440` | `0.5075` | `0.5729` | `1.0` | `0.5065` | `0.2174` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool_no_system` | `0.7` | `0.565` | `0.445` | `0.5050` | `0.5628` | `1.0` | `0.5195` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base_no_tool_no_system_answer_only` | `0.175` | `0.260` | `0.185` | `0.2225` | `0.2613` | `0.0` | `0.2013` | `0.1304` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool_no_system_answer_only` | `0.25` | `0.380` | `0.245` | `0.3125` | `0.3819` | `0.0` | `0.2922` | `0.0870` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool_no_system_answer_only` | `0.35` | `0.475` | `0.295` | `0.3850` | `0.4774` | `0.0` | `0.3377` | `0.1522` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool_no_system_answer_only` | `0.5` | `0.500` | `0.390` | `0.4450` | `0.5025` | `0.0` | `0.4481` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool_no_system_answer_only` | `0.7` | `0.580` | `0.390` | `0.4850` | `0.5829` | `0.0` | `0.4481` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base` | `0.175` | `0.185` | `0.280` | `0.2325` | `0.1809` | `1.0` | `0.2273` | `0.4565` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base` | `0.25` | `0.310` | `0.310` | `0.3100` | `0.3065` | `1.0` | `0.2922` | `0.3696` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base` | `0.35` | `0.390` | `0.355` | `0.3725` | `0.3920` | `0.0` | `0.3636` | `0.3261` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base` | `0.5` | `0.470` | `0.475` | `0.4725` | `0.4673` | `1.0` | `0.5130` | `0.3478` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base` | `0.7` | `0.545` | `0.460` | `0.5025` | `0.5427` | `1.0` | `0.5130` | `0.2826` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07_r2) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.400` | `0.255` | `0.3275` | `0.4020` | `0.0` | `0.2922` | `0.1304` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.500` | `0.375` | `0.4375` | `0.5025` | `0.0` | `0.4221` | `0.2174` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.550` | `0.400` | `0.4750` | `0.5528` | `0.0` | `0.4675` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.560` | `0.455` | `0.5075` | `0.5578` | `1.0` | `0.5584` | `0.1087` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.510` | `0.415` | `0.4625` | `0.5075` | `1.0` | `0.4870` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07_r2) |

## Operational Metrics

`assistant_resp_tokens` is `response_tokens_generated/mean`.
`tool_resp_tokens` is `response_tokens_tool/mean`.
For older runs that predate token-length logging, these cells are `n/a`.
For rows backfilled later, the token columns come from the corresponding `_tokenbackfill` reruns; the score, tool-call, and core-time columns remain from the original sweep rows.

| model | initial_rescale | longdocurl tool_calls | mmlongbench tool_calls | longdocurl core_time_s | mmlongbench core_time_s | assistant_resp_tokens | tool_resp_tokens | run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base_no_tool` | `0.175` | `0.000` | `0.000` | `10.99` | `9.31` | `320.24` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool` | `0.25` | `0.000` | `0.000` | `14.16` | `12.14` | `247.57` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool` | `0.35` | `0.000` | `0.000` | `26.57` | `22.27` | `191.43` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool` | `0.5` | `0.000` | `0.000` | `63.38` | `42.68` | `174.52` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool` | `0.7` | `0.000` | `0.000` | `121.68` | `103.20` | `201.45` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_0506/base_no_tool_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base_no_tool_no_system` | `0.175` | `0.000` | `0.000` | `13.60` | `8.21` | `359.18` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool_no_system` | `0.25` | `0.000` | `0.000` | `15.08` | `15.58` | `370.67` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool_no_system` | `0.35` | `0.000` | `0.000` | `27.85` | `21.33` | `225.45` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool_no_system` | `0.5` | `0.000` | `0.000` | `59.20` | `46.89` | `175.48` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool_no_system` | `0.7` | `0.000` | `0.000` | `123.60` | `106.77` | `180.39` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base_no_tool_no_system_answer_only` | `0.175` | `0.000` | `0.000` | `11.41` | `5.08` | `263.26` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base_no_tool_no_system_answer_only` | `0.25` | `0.000` | `0.000` | `11.02` | `7.33` | `56.89` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base_no_tool_no_system_answer_only` | `0.35` | `0.000` | `0.000` | `22.46` | `16.39` | `54.50` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base_no_tool_no_system_answer_only` | `0.5` | `0.000` | `0.000` | `55.06` | `34.99` | `15.78` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base_no_tool_no_system_answer_only` | `0.7` | `0.000` | `0.000` | `113.72` | `90.99` | `16.64` | `0.00` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507/base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07) |
| `base` | `0.175` | `6.250` | `4.520` | `34.91` | `30.93` | `959.35` | `485.59` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `base` | `0.25` | `4.830` | `3.955` | `44.02` | `35.77` | `738.94` | `537.68` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `base` | `0.35` | `3.620` | `3.525` | `60.19` | `53.46` | `724.28` | `610.38` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `base` | `0.5` | `1.670` | `1.940` | `103.15` | `82.77` | `712.23` | `620.04` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `base` | `0.7` | `1.080` | `1.455` | `178.93` | `163.77` | `538.83` | `826.52` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/base_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07_r2) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `2.860` | `4.305` | `16.31` | `19.69` | `n/a` | `n/a` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale0175) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `1.845` | `2.670` | `21.05` | `20.80` | `332.56` | `509.20` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale025) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `1.315` | `2.280` | `36.75` | `37.46` | `259.39` | `667.03` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale035) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `1.150` | `1.955` | `89.66` | `84.67` | `233.17` | `1087.76` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `1.045` | `1.925` | `203.31` | `220.67` | `229.24` | `1554.90` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_uncapped_mmlongbench200_uncapped_rescale07_r2) |

## `0.7` Rerun Notes

Original `0.7` runs failed from prompt overflow on the uncapped page counts.
After adding the initial-prompt shrink fallback, both were rerun successfully:
- presented image area is shrunk by `50%` per retry
- at most `4` retries
- unresolved samples would return an empty answer instead of crashing validation

Rerun results:
- base:
  - `initial_prompt_fit_succeeded = 1.0`
  - `initial_prompt_shrink_applied = 0.0375`
  - max prompt before shrink: `365882`
  - max prompt after shrink: `255786`
- `freeze_vt_bs32_tool_arg_order_medium_only_epoch`:
  - `initial_prompt_fit_succeeded = 1.0`
  - `initial_prompt_shrink_applied = 0.0375`
  - max prompt before shrink: `365882`
  - max prompt after shrink: `255786`

## Takeaways

- Uncapped page counts materially changed the sweep outcome.
- The best `initial_rescale` is no longer the same for the two models:
  - `base_no_tool`: `0.7`
  - `base`: `0.7`
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch`: `0.5`
- Compared with the capped sweep:
  - scores are generally lower
  - core inference time is much higher
  - prompt overflow becomes a real constraint at high rescale
- Best overall successful run:
  - `base_no_tool @ 0.7`
  - `longdocurl200 = 0.590`
  - `mmlongbench200 = 0.455`
  - `avg = 0.5225`
- Best base-model run:
  - `base @ 0.7`
  - `longdocurl200 = 0.545`
  - `mmlongbench200 = 0.460`
  - `avg = 0.5025`

Tradeoff pattern:
- as `initial_rescale` rises from `0.175 -> 0.7`:
  - scores improve
  - tool calls drop
  - core inference time rises sharply
- with shrink fallback, `0.7` fits, but only at a large latency cost.

## Eval Inference Stack

This is the current eval inference stack used by the launcher scripts for these runs.

### Topology

- GPUs per eval job: `4`
- vLLM tensor parallel size: `1`
- vLLM data parallel size: `1`
- vLLM pipeline parallel size: `1`
- vLLM replicas per eval job: `4`
- agent-loop workers per eval job: `32`

So the actual structure is:
- `32` logical agent-loop workers
- load-balanced across `4` vLLM instances

In practice, with the usual eval batch size:
- validation batch size: `32`
- rollout samples per prompt: `1`
- typical active conversations per vLLM instance: about `8`

The hard engine cap is much larger:
- `max_num_seqs = 1024` per vLLM instance

### Per-vLLM Instance Settings

Base async rollout config:
- backend: `vllm`
- mode: `async`
- dtype: `bfloat16`
- `enforce_eager = True`
- `enable_chunked_prefill = False`
- `free_cache_engine = True`
- `enable_prefix_caching = True`
- `disable_log_stats = True`
- `max_num_batched_tokens = 32768`
- `max_num_seqs = 1024`

Current eval overrides commonly used in these sweeps:
- `actor_rollout_ref.rollout.max_model_len = 262144`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len = 262144`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.gpu_memory_utilization = 0.9`

### Sampling / Multi-turn Settings

Validation sampling:
- `temperature = 0.7`
- `top_p = 0.8`
- `top_k = 20`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`
- `rollout.n = 1`

Multi-turn agent settings used in these sweeps:
- `max_user_turns = 10`
- `max_assistant_turns = 11`
- tool list: `image_zoom_in_tool_qwen3vl`

### Why This Matters

The main practical bottlenecks in these evals are not `max_num_seqs`.
They are:
- multimodal prompt length
- `max_model_len`
- GPU memory utilization

That is why the uncapped `0.7` runs failed:
- not from concurrency limits
- but from prompt overflow beyond `262144` before the shrink fallback was added
