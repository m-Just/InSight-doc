# Initial Rescale Sweep: `longdocurl_highpage_0507` + `mmlongbench_highpage_0507`

Sweep outputs:
- [status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/status.tsv)
- [summary.txt](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/summary.txt)

Additional single-model sweep appended here:
- [status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/status.tsv)
- [summary.txt](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/summary.txt)

Input parquets:
- [longdocurl_highpage_0507-insight_qwen_agent.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet)
- [mmlongbench_highpage_0507-insight_qwen_agent.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet)
- [longdocurl_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/longdocurl_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet)
- [mmlongbench_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/mmlongbench_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet)

Scope:
- models:
  - `base`
  - `base_no_tool_no_system`
  - `freeze_vt_bs32_tool_arg_order_medium_only_epoch`
  - `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2`
  - `rl_ckpt425_actor_merged_hf`
- `initial_rescale`:
  - `0.25`
  - `0.35`
  - `0.5`
- benchmarks:
  - `longdocurl_highpage_0507`
  - `mmlongbench_highpage_0507`

Dataset page-count stats:
- `longdocurl_highpage_0507`: `189` rows, mean page count `106.49`, top-40 mean `135.5`
- `mmlongbench_highpage_0507`: `178` rows, mean page count `71.03`, top-40 mean `163.55`
- pooled top-40-each-benchmark mean page count: `149.525`

Notes:
- scores below are validation `reward/mean@1`
- `avg` is the simple mean of the two benchmark scores
- the original 4-model sweep finished at `12 / 12` successful runs
- `base_no_tool_no_system` uses the no-tool, no-system parquets
- `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2 @ 0.35` completed successfully in `val_heldout.log`; the original launcher marked it failed due to a post-run shell syntax error, so the metrics below use the completed eval output

## Best By Model

| model | best initial_rescale | longdocurl | mmlongbench | avg | run |
|---|---:|---:|---:|---:|---|
| `base` | `0.5` | `0.492` | `0.371` | `0.4314` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_highpage_0507_rescale05) |
| `base_no_tool_no_system` | `0.5` | `0.524` | `0.410` | `0.4670` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_no_tool_no_system_highpage_0507_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.545` | `0.433` | `0.4888` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/freeze_vt_bs32_tool_arg_order_medium_only_epoch_highpage_0507_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.5` | `0.540` | `0.399` | `0.4693` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/freeze_vt_tool_arg_order_medium_only_p45_sp2_highpage_0507_rescale05) |
| `rl_ckpt425_actor_merged_hf` | `0.5` | `0.587` | `0.427` | `0.5071` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/rl_ckpt425_actor_merged_hf_highpage_0507_rescale05) |

## Overall Ranking

| rank | model | initial_rescale | longdocurl | mmlongbench | avg | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | `rl_ckpt425_actor_merged_hf` | `0.5` | `0.587` | `0.427` | `0.5071` | `success` |
| 2 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.545` | `0.433` | `0.4888` | `success` |
| 3 | `rl_ckpt425_actor_merged_hf` | `0.35` | `0.556` | `0.421` | `0.4885` | `success` |
| 4 | `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.5` | `0.540` | `0.399` | `0.4693` | `success` |
| 5 | `base_no_tool_no_system` | `0.5` | `0.524` | `0.410` | `0.4670` | `success` |
| 6 | `base_no_tool_no_system` | `0.35` | `0.519` | `0.388` | `0.4531` | `success` |
| 7 | `rl_ckpt425_actor_merged_hf` | `0.25` | `0.513` | `0.382` | `0.4476` | `success` |
| 8 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.524` | `0.348` | `0.4361` | `success` |
| 9 | `base` | `0.5` | `0.492` | `0.371` | `0.4314` | `success` |
| 10 | `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.35` | `0.534` | `0.326` | `0.4301` | `success` |
| 11 | `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.508` | `0.309` | `0.4085` | `success` |
| 12 | `base` | `0.35` | `0.434` | `0.337` | `0.3855` | `success` |
| 13 | `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.25` | `0.466` | `0.298` | `0.3817` | `success` |
| 14 | `base_no_tool_no_system` | `0.25` | `0.402` | `0.270` | `0.3359` | `success` |
| 15 | `base` | `0.25` | `0.280` | `0.270` | `0.2750` | `success` |

## Core Inference Time Comparison

`avg_core_time_s` below is the simple mean of:
- `longdocurl_highpage_0507 core_inference_time`
- `mmlongbench_highpage_0507 core_inference_time`

| model | best initial_rescale | avg | avg_core_time_s | longdocurl core_time_s | mmlongbench core_time_s |
|---|---:|---:|---:|---:|---:|
| `base` | `0.5` | `0.4314` | `119.92` | `121.95` | `117.90` |
| `base_no_tool_no_system` | `0.5` | `0.4670` | `71.47` | `77.91` | `65.03` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.4888` | `126.14` | `117.23` | `135.04` |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.5` | `0.4693` | `106.77` | `97.69` | `115.85` |
| `rl_ckpt425_actor_merged_hf` | `0.5` | `0.5071` | `134.65` | `127.88` | `141.42` |

## Chart

Average accuracy vs average core inference time over the two high-page benchmarks:
- [PNG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_avg_core_time_vs_avg_accuracy_2026-05-11.png)
- [SVG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_avg_core_time_vs_avg_accuracy_2026-05-11.svg)
- [CSV](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_avg_core_time_vs_avg_accuracy_2026-05-11.csv)

![High-page avg core-time vs avg accuracy](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_avg_core_time_vs_avg_accuracy_2026-05-11.png)

## Full Results

| model | initial_rescale | longdocurl | mmlongbench | avg | longdocurl acc/answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable | run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | `0.25` | `0.280` | `0.270` | `0.2750` | `0.2804` | `0.2138` | `0.5152` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_highpage_0507_rescale025) |
| `base` | `0.35` | `0.434` | `0.337` | `0.3855` | `0.4339` | `0.2966` | `0.5152` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_highpage_0507_rescale035) |
| `base` | `0.5` | `0.492` | `0.371` | `0.4314` | `0.4921` | `0.3586` | `0.4242` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_highpage_0507_rescale05) |
| `base_no_tool_no_system` | `0.25` | `0.402` | `0.270` | `0.3359` | `0.4021` | `0.2552` | `0.3333` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_no_tool_no_system_highpage_0507_rescale025) |
| `base_no_tool_no_system` | `0.35` | `0.519` | `0.388` | `0.4531` | `0.5185` | `0.3793` | `0.4242` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_no_tool_no_system_highpage_0507_rescale035) |
| `base_no_tool_no_system` | `0.5` | `0.524` | `0.410` | `0.4670` | `0.5238` | `0.4138` | `0.3939` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/base_no_tool_no_system_highpage_0507_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `0.508` | `0.309` | `0.4085` | `0.5079` | `0.3172` | `0.2727` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/freeze_vt_bs32_tool_arg_order_medium_only_epoch_highpage_0507_rescale025) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `0.524` | `0.348` | `0.4361` | `0.5238` | `0.4000` | `0.1212` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/freeze_vt_bs32_tool_arg_order_medium_only_epoch_highpage_0507_rescale035) |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `0.545` | `0.433` | `0.4888` | `0.5450` | `0.4897` | `0.1818` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/freeze_vt_bs32_tool_arg_order_medium_only_epoch_highpage_0507_rescale05) |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.25` | `0.466` | `0.298` | `0.3817` | `0.4656` | `0.3379` | `0.1212` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/freeze_vt_tool_arg_order_medium_only_p45_sp2_highpage_0507_rescale025) |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.35` | `0.534` | `0.326` | `0.4301` | `0.5344` | `0.3655` | `0.1515` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/freeze_vt_tool_arg_order_medium_only_p45_sp2_highpage_0507_rescale035) |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.5` | `0.540` | `0.399` | `0.4693` | `0.5397` | `0.4414` | `0.2121` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_freeze_vt_tool_arg_order_medium_only_p45_sp2_05011/freeze_vt_tool_arg_order_medium_only_p45_sp2_highpage_0507_rescale05) |
| `rl_ckpt425_actor_merged_hf` | `0.25` | `0.513` | `0.382` | `0.4476` | `0.5132` | `0.3793` | `0.3939` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/rl_ckpt425_actor_merged_hf_highpage_0507_rescale025) |
| `rl_ckpt425_actor_merged_hf` | `0.35` | `0.556` | `0.421` | `0.4885` | `0.5556` | `0.4276` | `0.3939` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/rl_ckpt425_actor_merged_hf_highpage_0507_rescale035) |
| `rl_ckpt425_actor_merged_hf` | `0.5` | `0.587` | `0.427` | `0.5071` | `0.5873` | `0.4483` | `0.3333` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_initial_rescale_sweep_0507/rl_ckpt425_actor_merged_hf_highpage_0507_rescale05) |

## Operational Metrics

| model | initial_rescale | avg_tool_calls | longdocurl tool_calls | mmlongbench tool_calls | avg_core_time_s | assistant_resp_tokens | tool_resp_tokens | prompt_tokens | initial_prompt_fit_time_s | shrink_applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | `0.25` | `4.608` | `5.058` | `4.157` | `49.27` | `821.76` | `495.17` | `20680.95` | `13.12` | `0.0000` |
| `base` | `0.35` | `3.546` | `3.593` | `3.500` | `80.38` | `793.92` | `620.38` | `40468.01` | `18.08` | `0.0000` |
| `base` | `0.5` | `1.854` | `1.534` | `2.174` | `119.92` | `553.40` | `565.14` | `80243.52` | `26.54` | `0.0027` |
| `base_no_tool_no_system` | `0.25` | `0.000` | `0.000` | `0.000` | `17.47` | `317.78` | `0.00` | `20222.95` | `13.10` | `0.0000` |
| `base_no_tool_no_system` | `0.35` | `0.000` | `0.000` | `0.000` | `32.57` | `248.35` | `0.00` | `40010.01` | `18.29` | `0.0000` |
| `base_no_tool_no_system` | `0.5` | `0.000` | `0.000` | `0.000` | `71.47` | `227.29` | `0.00` | `79785.52` | `26.54` | `0.0027` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.25` | `2.609` | `2.095` | `3.124` | `28.01` | `363.13` | `587.68` | `20680.95` | `13.46` | `0.0000` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.35` | `2.085` | `1.418` | `2.753` | `51.38` | `300.88` | `867.52` | `40468.01` | `18.21` | `0.0000` |
| `freeze_vt_bs32_tool_arg_order_medium_only_epoch` | `0.5` | `1.680` | `1.164` | `2.197` | `126.14` | `250.09` | `1226.10` | `80243.52` | `27.26` | `0.0027` |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.25` | `2.323` | `1.550` | `3.096` | `24.76` | `317.13` | `520.26` | `20680.95` | `13.17` | `0.0000` |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.35` | `1.930` | `1.185` | `2.674` | `48.29` | `274.11` | `778.23` | `40468.01` | `17.99` | `0.0000` |
| `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` | `0.5` | `1.226` | `0.677` | `1.775` | `106.77` | `191.39` | `804.34` | `80243.52` | `26.69` | `0.0027` |
| `rl_ckpt425_actor_merged_hf` | `0.25` | `3.042` | `2.265` | `3.820` | `32.00` | `410.43` | `916.43` | `20680.95` | `13.49` | `0.0000` |
| `rl_ckpt425_actor_merged_hf` | `0.35` | `2.349` | `1.783` | `2.916` | `55.94` | `330.61` | `1257.92` | `40468.01` | `18.19` | `0.0000` |
| `rl_ckpt425_actor_merged_hf` | `0.5` | `2.039` | `1.270` | `2.809` | `134.65` | `296.43` | `1695.95` | `80243.52` | `26.94` | `0.0027` |


## LoRA Follow-Up Results

Additional LoRA high-page sweep outputs:
- [lora_basic status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514/status.tsv)
- [lora higher-dpi chain status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/status.tsv)

Notes:
- `lora_basic_40k_freeze_vt_medium_only @ 0.35` uses the successful retry run from the higher-dpi chain; its original 0.35 log did not contain a completed final metric line.
- The higher-dpi chain status file contains stale failed rows from a bad launcher attempt. The later appended `success` rows are the valid runs.
- `judge_fallback_used` is near zero; the Gemini fallback only triggered for a small number of longdocurl samples in a few runs.

### LoRA Best By Model

| model | best initial_rescale | longdocurl | mmlongbench | avg | avg_core_time_s |
|---|---:|---:|---:|---:|---:|
| `lora_basic_40k_freeze_vt_medium_only` | `0.35` | `0.497` | `0.371` | `0.4341` | `26.73` |
| `lora_arxiv_w_higher_dpi` | `0.5` | `0.519` | `0.393` | `0.4559` | `71.80` |
| `lora_O3_w_higher_dpi` | `0.5` | `0.550` | `0.382` | `0.4661` | `81.33` |

### LoRA Full Results

| model | initial_rescale | longdocurl | mmlongbench | avg | longdocurl acc/answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable | run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lora_basic_40k_freeze_vt_medium_only` | `0.25` | `0.481` | `0.292` | `0.3868` | `0.4815` | `0.3172` | `0.1818` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514/lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale025) |
| `lora_basic_40k_freeze_vt_medium_only` | `0.35` | `0.497` | `0.371` | `0.4341` | `0.4974` | `0.3931` | `0.2727` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_basic_40k_freeze_vt_medium_only_retry_highpage_0507_rescale035) |
| `lora_basic_40k_freeze_vt_medium_only` | `0.5` | `0.503` | `0.348` | `0.4255` | `0.5026` | `0.4000` | `0.1212` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514/lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale05) |
| `lora_arxiv_w_higher_dpi` | `0.25` | `0.476` | `0.281` | `0.3785` | `0.4762` | `0.3103` | `0.1515` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale025) |
| `lora_arxiv_w_higher_dpi` | `0.35` | `0.534` | `0.337` | `0.4357` | `0.5344` | `0.3793` | `0.1515` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale035) |
| `lora_arxiv_w_higher_dpi` | `0.5` | `0.519` | `0.393` | `0.4559` | `0.5185` | `0.4345` | `0.2121` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_arxiv_w_higher_dpi_highpage_0507_rescale05) |
| `lora_O3_w_higher_dpi` | `0.25` | `0.492` | `0.275` | `0.3837` | `0.4921` | `0.3034` | `0.1515` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale025) |
| `lora_O3_w_higher_dpi` | `0.35` | `0.524` | `0.360` | `0.4417` | `0.5238` | `0.4138` | `0.1212` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale035) |
| `lora_O3_w_higher_dpi` | `0.5` | `0.550` | `0.382` | `0.4661` | `0.5503` | `0.4483` | `0.0909` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514/lora_O3_w_higher_dpi_highpage_0507_rescale05) |

### LoRA Operational Metrics

| model | initial_rescale | avg_tool_calls | longdocurl tool_calls | mmlongbench tool_calls | avg_core_time_s | assistant_resp_tokens | tool_resp_tokens | prompt_tokens | initial_prompt_fit_time_s | shrink_applied | judge_fallback_used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_basic_40k_freeze_vt_medium_only` | `0.25` | `2.717` | `2.063` | `3.371` | `14.18` | `377.25` | `588.67` | `20680.95` | `12.70` | `0.0000` | `0.0000` |
| `lora_basic_40k_freeze_vt_medium_only` | `0.35` | `2.188` | `1.651` | `2.725` | `26.73` | `311.02` | `862.54` | `40468.01` | `17.45` | `0.0000` | `0.0026` |
| `lora_basic_40k_freeze_vt_medium_only` | `0.5` | `1.748` | `0.989` | `2.506` | `77.28` | `255.88` | `1121.21` | `80243.52` | `25.78` | `0.0027` | `0.0000` |
| `lora_arxiv_w_higher_dpi` | `0.25` | `2.722` | `2.095` | `3.348` | `14.09` | `371.13` | `592.54` | `20680.95` | `12.86` | `0.0000` | `0.0000` |
| `lora_arxiv_w_higher_dpi` | `0.35` | `1.831` | `1.196` | `2.466` | `26.17` | `268.38` | `681.13` | `40468.01` | `17.56` | `0.0000` | `0.0026` |
| `lora_arxiv_w_higher_dpi` | `0.5` | `1.493` | `0.862` | `2.124` | `71.80` | `224.09` | `1066.57` | `80243.52` | `26.29` | `0.0027` | `0.0026` |
| `lora_O3_w_higher_dpi` | `0.25` | `2.534` | `1.810` | `3.258` | `13.82` | `355.14` | `529.25` | `20680.95` | `12.68` | `0.0000` | `0.0000` |
| `lora_O3_w_higher_dpi` | `0.35` | `2.054` | `1.354` | `2.753` | `26.92` | `296.26` | `759.76` | `40468.01` | `17.33` | `0.0000` | `0.0026` |
| `lora_O3_w_higher_dpi` | `0.5` | `1.731` | `1.085` | `2.376` | `81.33` | `253.68` | `1075.04` | `80243.52` | `26.08` | `0.0027` | `0.0026` |

### LoRA vs Previous Strongest

| initial_rescale | previous best model | previous best avg | best LoRA model | best LoRA avg | delta |
|---:|---|---:|---|---:|---:|
| `0.25` | `rl_ckpt425_actor_merged_hf` | `0.4476` | `lora_basic_40k_freeze_vt_medium_only` | `0.3868` | `-0.0608` |
| `0.35` | `rl_ckpt425_actor_merged_hf` | `0.4885` | `lora_O3_w_higher_dpi` | `0.4417` | `-0.0468` |
| `0.5` | `rl_ckpt425_actor_merged_hf` | `0.5071` | `lora_O3_w_higher_dpi` | `0.4661` | `-0.0410` |

## Matched Base Reruns Under Current LoRA-Setting Eval

Additional matched base rerun outputs:
- [base current-setting status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/status.tsv)
- [base no-tool/no-system current-setting status.tsv](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_no_system_current_lora_setting_20260515/status.tsv)

Notes:
- These runs use the same high-page parquets, rescale values, max-length settings, console logging, and judge fallback used for the recent LoRA follow-up evals.
- `base_current_lora_setting` is the unsfted base Qwen model with the normal tool-use prompt/schema.
- `base_no_tool_no_system_current_lora_setting` is the unsfted base Qwen model with absent system message and no tool schema.
- The older `base` and `base_no_tool_no_system` rows above are kept unchanged for historical comparison.

### Matched Base Best By Model

| model | best initial_rescale | longdocurl | mmlongbench | avg | avg_core_time_s |
|---|---:|---:|---:|---:|---:|
| `base_current_lora_setting` | `0.5` | `0.534` | `0.343` | `0.4385` | `116.50` |
| `base_no_tool_no_system_current_lora_setting` | `0.5` | `0.545` | `0.416` | `0.4804` | `71.83` |

### Matched Base Full Results

| model | initial_rescale | longdocurl | mmlongbench | avg | longdocurl acc/answerable | mmlongbench acc/answerable | mmlongbench acc/not_answerable | run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base_current_lora_setting` | `0.25` | `0.317` | `0.258` | `0.2879` | `0.3175` | `0.2414` | `0.3333` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale025) |
| `base_current_lora_setting` | `0.35` | `0.423` | `0.343` | `0.3830` | `0.4233` | `0.2966` | `0.5455` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale035) |
| `base_current_lora_setting` | `0.5` | `0.534` | `0.343` | `0.4385` | `0.5344` | `0.3448` | `0.3333` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_current_lora_setting_20260515/base_before_sft_current_lora_setting_highpage_0507_rescale05) |
| `base_no_tool_no_system_current_lora_setting` | `0.25` | `0.392` | `0.242` | `0.3166` | `0.3915` | `0.2345` | `0.2727` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_no_system_current_lora_setting_20260515/base_no_tool_no_system_current_lora_setting_highpage_0507_rescale025) |
| `base_no_tool_no_system_current_lora_setting` | `0.35` | `0.503` | `0.348` | `0.4255` | `0.5026` | `0.3448` | `0.3636` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_no_system_current_lora_setting_20260515/base_no_tool_no_system_current_lora_setting_highpage_0507_rescale035) |
| `base_no_tool_no_system_current_lora_setting` | `0.5` | `0.545` | `0.416` | `0.4804` | `0.5450` | `0.4138` | `0.4242` | [run](/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_no_system_current_lora_setting_20260515/base_no_tool_no_system_current_lora_setting_highpage_0507_rescale05) |

### Matched Base Operational Metrics

| model | initial_rescale | avg_tool_calls | longdocurl tool_calls | mmlongbench tool_calls | avg_core_time_s | assistant_resp_tokens | tool_resp_tokens | prompt_tokens | initial_prompt_fit_time_s | shrink_applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_current_lora_setting` | `0.25` | `4.657` | `4.905` | `4.410` | `49.33` | `873.40` | `588.60` | `20680.95` | `12.75` | `0.0000` |
| `base_current_lora_setting` | `0.35` | `3.707` | `3.937` | `3.478` | `76.47` | `812.45` | `603.71` | `40468.01` | `17.24` | `0.0000` |
| `base_current_lora_setting` | `0.5` | `1.694` | `1.444` | `1.944` | `116.50` | `505.04` | `565.72` | `80243.52` | `25.88` | `0.0027` |
| `base_no_tool_no_system_current_lora_setting` | `0.25` | `0.000` | `0.000` | `0.000` | `16.06` | `232.94` | `0.00` | `20222.95` | `12.58` | `0.0000` |
| `base_no_tool_no_system_current_lora_setting` | `0.35` | `0.000` | `0.000` | `0.000` | `33.04` | `284.64` | `0.00` | `40010.01` | `17.18` | `0.0000` |
| `base_no_tool_no_system_current_lora_setting` | `0.5` | `0.000` | `0.000` | `0.000` | `71.83` | `280.25` | `0.00` | `79785.52` | `25.55` | `0.0027` |

### Matched Base Vs Previous Baseline Rows

| model family | initial_rescale | previous avg | current-setting avg | delta | previous avg_core_time_s | current avg_core_time_s |
|---|---:|---:|---:|---:|---:|---:|
| `base` | `0.25` | `0.2750` | `0.2879` | `+0.0129` | `49.27` | `49.33` |
| `base` | `0.35` | `0.3855` | `0.3830` | `-0.0025` | `80.38` | `76.47` |
| `base` | `0.5` | `0.4314` | `0.4385` | `+0.0071` | `119.92` | `116.50` |
| `base_no_tool_no_system` | `0.25` | `0.3359` | `0.3166` | `-0.0193` | `17.47` | `16.06` |
| `base_no_tool_no_system` | `0.35` | `0.4531` | `0.4255` | `-0.0276` | `32.57` | `33.04` |
| `base_no_tool_no_system` | `0.5` | `0.4670` | `0.4804` | `+0.0134` | `71.47` | `71.83` |

### LoRA And Matched Base Chart

Average accuracy vs average core inference time for the three LoRA runs plus the two matched base reruns:
- [PNG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_2026-05-15.png)
- [SVG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_2026-05-15.svg)
- [CSV](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_2026-05-15.csv)

![LoRA and matched base high-page chart](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_2026-05-15.png)

### LoRA And Matched Base Median-Core Chart

Same accuracy values as above, but the x-axis is the average of the per-benchmark median `core_inference_time` values (`(median_longdocurl + median_mmlongbench) / 2`). The CSV also includes each benchmark median separately.
- [PNG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_median_core_2026-05-15.png)
- [SVG](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_median_core_2026-05-15.svg)
- [CSV](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_median_core_2026-05-15.csv)

![LoRA and matched base high-page median-core chart](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/highpage_lora_and_matched_base_current_setting_median_core_2026-05-15.png)

## Takeaways

- Best overall run by avg remains `rl_ckpt425_actor_merged_hf` at `0.5`, with `avg = 0.5071`.
- Best `base` run is still `0.5`, with `avg = 0.4314`.
- Best `base_no_tool_no_system` run is still `0.5`, with `avg = 0.4670`.
- Best `freeze_vt_bs32_tool_arg_order_medium_only_epoch` run is still `0.5`, with `avg = 0.4888`.
- The new `freeze_vt_bs32_tool_arg_order_medium_only_p45_sp2` variant peaks at `0.5`, with `avg = 0.4693`.
- Compared with the previous `freeze_vt_bs32_tool_arg_order_medium_only_epoch` run, the `p45_sp2` variant is worse at every tested rescale:
  - `0.25`: `0.4085 -> 0.3817` `(-0.0268)`
  - `0.35`: `0.4361 -> 0.4301` `(-0.0060)`
  - `0.5`: `0.4888 -> 0.4693` `(-0.0195)`
- The `p45_sp2` variant is faster than the earlier `freeze_vt_bs32_tool_arg_order_medium_only_epoch` run at all three rescales, but the speedup came with lower accuracy.
- On this high-page sample, every model still improves monotonically from `0.25 -> 0.5` in average accuracy.
- Higher `initial_rescale` also raises prompt size and core inference time sharply.

- New LoRA follow-up runs did not beat the previous strongest `rl_ckpt425_actor_merged_hf` results at any tested rescale.
- Among the new LoRA runs, `lora_O3_w_higher_dpi` is best at `0.35` and `0.5`; `lora_basic_40k_freeze_vt_medium_only` is slightly best at `0.25`.
- The best new LoRA result is `lora_O3_w_higher_dpi @ 0.5`, with `avg = 0.4661`, below `rl_ckpt425_actor_merged_hf @ 0.5` by `0.0410`.
- The LoRA runs are substantially faster than the previous strongest tool-use runs at comparable rescale, but the speedup comes with lower accuracy.

- In the matched current-setting base reruns, `base_current_lora_setting @ 0.5` reaches `avg = 0.4385`, slightly above the older `base @ 0.5` row (`0.4314`).
- In the matched current-setting no-tool/no-system reruns, `base_no_tool_no_system_current_lora_setting @ 0.5` reaches `avg = 0.4804`, above the older `base_no_tool_no_system @ 0.5` row (`0.4670`) and still below `rl_ckpt425_actor_merged_hf @ 0.5` (`0.5071`).
