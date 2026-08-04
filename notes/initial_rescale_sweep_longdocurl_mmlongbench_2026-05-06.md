# Initial Rescale Sweep: `longdocurl200` + `mmlongbench200`

Sweep outputs:
- [status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/status.tsv)
- [summary.txt](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/summary.txt)

Scope:
- models:
  - `base`
  - `freeze_vt_bs32_epoch2`
  - `freeze_vt_bs8_epoch5`
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch`
- `initial_rescale`:
  - `0.175`
  - `0.25`
  - `0.35`
  - `0.5`
  - `0.7`
- benchmarks:
  - `longdocurl200`
  - `mmlongbench200`

Notes:
- scores below are validation `reward/mean@1`
- `avg` is the simple mean of the two benchmark scores
- all `20 / 20` runs succeeded

## Best By Model

| model | best initial_rescale | longdocurl200 | mmlongbench200 | avg | run |
|---|---:|---:|---:|---:|---|
| `base` | `0.7` | `0.600` | `0.500` | `0.550` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale07) |
| `freeze_vt_bs32_epoch2` | `0.7` | `0.590` | `0.445` | `0.5175` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale07) |
| `freeze_vt_bs8_epoch5` | `0.5` | `0.625` | `0.435` | `0.530` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.625` | `0.465` | `0.545` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale07) |

## Overall Ranking

| rank | model | initial_rescale | longdocurl200 | mmlongbench200 | avg |
|---:|---|---:|---:|---:|---:|
| 1 | `base` | `0.7` | `0.600` | `0.500` | `0.550` |
| 2 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.625` | `0.465` | `0.545` |
| 3 | `freeze_vt_bs8_epoch5` | `0.5` | `0.625` | `0.435` | `0.530` |
| 4 | `freeze_vt_bs8_epoch5` | `0.7` | `0.605` | `0.440` | `0.5225` |
| 5 | `freeze_vt_bs32_epoch2` | `0.7` | `0.590` | `0.445` | `0.5175` |
| 6 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.605` | `0.425` | `0.515` |
| 7 | `freeze_vt_bs32_epoch2` | `0.5` | `0.625` | `0.395` | `0.510` |
| 8 | `base` | `0.5` | `0.555` | `0.430` | `0.4925` |
| 9 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.580` | `0.390` | `0.485` |
| 10 | `freeze_vt_bs32_epoch2` | `0.35` | `0.570` | `0.375` | `0.4725` |
| 11 | `freeze_vt_bs8_epoch5` | `0.35` | `0.560` | `0.370` | `0.465` |
| 12 | `freeze_vt_bs8_epoch5` | `0.25` | `0.555` | `0.355` | `0.455` |
| 13 | `base` | `0.35` | `0.520` | `0.370` | `0.445` |
| 14 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.565` | `0.320` | `0.4425` |
| 15 | `freeze_vt_bs32_epoch2` | `0.25` | `0.510` | `0.285` | `0.3975` |
| 16 | `base` | `0.25` | `0.495` | `0.245` | `0.370` |
| 17 | `freeze_vt_bs32_epoch2` | `0.175` | `0.465` | `0.260` | `0.3625` |
| 18 | `freeze_vt_bs8_epoch5` | `0.175` | `0.430` | `0.285` | `0.3575` |
| 19 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.420` | `0.250` | `0.335` |
| 20 | `base` | `0.175` | `0.325` | `0.240` | `0.2825` |

## Core Inference Time Comparison

`avg_core_time_s` below is the simple mean of:
- `longdocurl200 core_inference_time`
- `mmlongbench200 core_inference_time`

### Best-Scoring Config Per Model

| model | best initial_rescale | avg | avg_core_time_s | longdocurl core_time_s | mmlongbench core_time_s |
|---|---:|---:|---:|---:|---:|
| `base` | `0.7` | `0.550` | `64.86` | `62.70` | `67.01` |
| `freeze_vt_bs32_epoch2` | `0.7` | `0.5175` | `56.03` | `50.73` | `61.32` |
| `freeze_vt_bs8_epoch5` | `0.5` | `0.530` | `24.63` | `21.16` | `28.11` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.545` | `60.24` | `53.96` | `66.52` |

### Fastest Config Per Model

| model | fastest initial_rescale | avg | avg_core_time_s |
|---|---:|---:|---:|
| `base` | `0.175` | `0.2825` | `17.91` |
| `freeze_vt_bs32_epoch2` | `0.175` | `0.3625` | `10.94` |
| `freeze_vt_bs8_epoch5` | `0.175` | `0.3575` | `9.68` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.3350` | `11.55` |

### Practical Tradeoff

- The highest-scoring settings are not the fastest.
- `base @ 0.7` is the best overall score, but it is also the slowest winning config:
  - `avg = 0.550`
  - `avg_core_time_s = 64.86`
- `freeze_vt_bs32_tool_arg_order_medium_only_epoch @ 0.7` is close in score and similarly expensive:
  - `avg = 0.545`
  - `avg_core_time_s = 60.24`
- `freeze_vt_bs8_epoch5 @ 0.5` is the strongest score/time compromise in this sweep:
  - `avg = 0.530`
  - `avg_core_time_s = 24.63`
- Moving from `0.5` to `0.7` usually gives only a modest score gain or none, but a very large core-time increase.

## Full Results

| model | initial_rescale | longdocurl200 | mmlongbench200 | avg | longdocurl tool_calls | mmlongbench tool_calls | longdocurl core_time_s | mmlongbench core_time_s | longdocurl acc/answerable | longdocurl acc/not_answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable | run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | `0.175` | `0.325` | `0.240` | `0.2825` | `3.990` | `3.825` | `16.96` | `18.86` | `0.3216` | `1.0` | `0.2273` | `0.2826` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale0175) |
| `base` | `0.25` | `0.495` | `0.245` | `0.3700` | `3.345` | `3.665` | `17.22` | `19.99` | `0.4925` | `1.0` | `0.2208` | `0.3261` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale025) |
| `base` | `0.35` | `0.520` | `0.370` | `0.4450` | `2.575` | `2.500` | `23.52` | `23.25` | `0.5176` | `1.0` | `0.3506` | `0.4348` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale035) |
| `base` | `0.5` | `0.555` | `0.430` | `0.4925` | `1.955` | `1.785` | `32.01` | `36.39` | `0.5578` | `0.0` | `0.4481` | `0.3696` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale05) |
| `base` | `0.7` | `0.600` | `0.500` | `0.5500` | `1.610` | `1.495` | `62.70` | `67.01` | `0.5980` | `1.0` | `0.5390` | `0.3696` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/base_longdocurl200_mmlongbench200_rescale07) |
| `freeze_vt_bs32_epoch2` | `0.175` | `0.465` | `0.260` | `0.3625` | `2.550` | `3.850` | `9.08` | `12.81` | `0.4623` | `1.0` | `0.2987` | `0.1304` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale0175) |
| `freeze_vt_bs32_epoch2` | `0.25` | `0.510` | `0.285` | `0.3975` | `1.935` | `3.000` | `10.28` | `13.85` | `0.5075` | `1.0` | `0.3247` | `0.1522` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale025) |
| `freeze_vt_bs32_epoch2` | `0.35` | `0.570` | `0.375` | `0.4725` | `1.240` | `2.315` | `13.22` | `17.90` | `0.5678` | `1.0` | `0.4481` | `0.1304` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale035) |
| `freeze_vt_bs32_epoch2` | `0.5` | `0.625` | `0.395` | `0.5100` | `0.975` | `2.110` | `23.48` | `31.71` | `0.6231` | `1.0` | `0.4545` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale05) |
| `freeze_vt_bs32_epoch2` | `0.7` | `0.590` | `0.445` | `0.5175` | `0.850` | `1.825` | `50.73` | `61.32` | `0.5930` | `0.0` | `0.5260` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_epoch2_longdocurl200_mmlongbench200_rescale07) |
| `freeze_vt_bs8_epoch5` | `0.175` | `0.430` | `0.285` | `0.3575` | `2.135` | `3.565` | `7.78` | `11.58` | `0.4322` | `0.0` | `0.3182` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale0175) |
| `freeze_vt_bs8_epoch5` | `0.25` | `0.555` | `0.355` | `0.4550` | `1.405` | `2.340` | `8.23` | `12.01` | `0.5528` | `1.0` | `0.4156` | `0.1522` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale025) |
| `freeze_vt_bs8_epoch5` | `0.35` | `0.560` | `0.370` | `0.4650` | `1.260` | `1.960` | `13.59` | `16.99` | `0.5628` | `0.0` | `0.4351` | `0.1522` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale035) |
| `freeze_vt_bs8_epoch5` | `0.5` | `0.625` | `0.435` | `0.5300` | `0.850` | `1.545` | `21.16` | `28.11` | `0.6281` | `0.0` | `0.5065` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale05) |
| `freeze_vt_bs8_epoch5` | `0.7` | `0.605` | `0.440` | `0.5225` | `0.710` | `1.490` | `47.41` | `59.72` | `0.6030` | `1.0` | `0.5260` | `0.1522` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs8_epoch5_longdocurl200_mmlongbench200_rescale07) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.175` | `0.420` | `0.250` | `0.3350` | `2.820` | `4.090` | `9.71` | `13.40` | `0.4221` | `0.0` | `0.2922` | `0.1087` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale0175) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.565` | `0.320` | `0.4425` | `1.710` | `2.995` | `9.58` | `13.88` | `0.5628` | `1.0` | `0.3636` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale025) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.580` | `0.390` | `0.4850` | `1.300` | `2.160` | `13.06` | `17.47` | `0.5779` | `1.0` | `0.4545` | `0.1739` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale035) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.605` | `0.425` | `0.5150` | `1.125` | `2.020` | `25.45` | `32.01` | `0.6030` | `1.0` | `0.4935` | `0.1957` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.7` | `0.625` | `0.465` | `0.5450` | `0.935` | `1.810` | `53.96` | `66.52` | `0.6231` | `1.0` | `0.5325` | `0.2391` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_initial_rescale_sweep_0506/freeze_vt_bs32_tool_arg_order_medium_only_epoch_longdocurl200_mmlongbench200_rescale07) |

## Takeaways

- Higher `initial_rescale` helped almost everywhere in this sweep.
- The best setting was usually `0.5` or `0.7`.
- Best overall run:
  - `base` at `0.7`
  - `longdocurl200 = 0.600`
  - `mmlongbench200 = 0.500`
  - `avg = 0.550`
- Best SFT run:
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch` at `0.7`
  - `avg = 0.545`
- Best speed/quality compromise among SFT runs:
  - `freeze_vt_bs8_epoch5` at `0.5`
  - `avg = 0.530`

Tradeoff pattern:
- as `initial_rescale` rises:
  - tool calls go down
  - core inference time goes up sharply
  - scores generally go up

Examples:
- `base`
  - `longdocurl200` tool calls: `3.990 -> 1.610`
  - `longdocurl200` core time: `16.96s -> 62.70s`
- `freeze_vt_bs8_epoch5`
  - `mmlongbench200` tool calls: `3.565 -> 1.490`
  - `mmlongbench200` core time: `11.58s -> 59.72s`
