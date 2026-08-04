# Highpage Full-SFT Epoch-1 vs Epoch-2 Comparison

Date: 2026-05-20

This note compares the full-weight SFT runs trained on `both_higher_dpi_unans02503505` against the LoRA `both_higher_dpi_plus_unans02503505_e1_64k_sp2` result from [`highpage_lora_unanswerable_medium_comparison_2026-05-18.md`](/scratch/ywxzml3j/likaican/src/InSight-doc/verl/notes/highpage_lora_unanswerable_medium_comparison_2026-05-18.md).

The highpage comparison uses `longdocurl0507_highpage` and `mmlongbench0507_highpage` at rescale `0.25`, `0.35`, and `0.5`.

## Runs

| Group | Run | Output root |
|---|---|---|
| LoRA reference | `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_unanswerable02503505_20260517` |
| Full-SFT epoch 1 | `full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_epoch1_fp32_scratch` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519_retry` |
| Full-SFT epoch 2 | `full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_epoch2_fp32_scratch` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260520_fallback_restart` |

Notes:

- The LoRA reference metrics are copied from the 2026-05-18 comparison note.
- The two full-SFT runs are full-weight tuning with frozen vision tower, fp32 model precision, max length 64k, `sp=4`, `bs=32`.
- The epoch-2 evals used fallback judge config, but the logged samples I checked still mostly used the primary judge; fallback was only a safety path for content-filter failures.

## Full Highpage Metrics

| Run | Rescale | Longdoc acc | MMLong acc | Avg acc | Core time s | Avg tool calls | Avg turns | Gen tokens | Tool tokens | MMLong unanswerable acc | MMLong answerable acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.25 | 0.4392 | 0.3258 | 0.3825 | 15.00 | 3.09 | 8.15 | 407.56 | 647.79 | 0.3333 | 0.3241 |
| `full_sft_epoch1_fp32_sp4` | 0.25 | 0.4603 | 0.3202 | 0.3903 | 14.61 | 3.10 | 8.16 | 403.80 | 657.10 | 0.4242 | 0.2966 |
| `full_sft_epoch2_fp32_sp4` | 0.25 | 0.4550 | 0.2865 | 0.3708 | 13.85 | 2.72 | 7.39 | 373.76 | 541.75 | 0.4242 | 0.2552 |
| `lora_both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.35 | 0.5238 | 0.3989 | 0.4613 | 26.31 | 1.96 | 5.87 | 278.13 | 748.47 | 0.4242 | 0.3931 |
| `full_sft_epoch1_fp32_sp4` | 0.35 | 0.5026 | 0.3764 | 0.4395 | 27.26 | 2.36 | 6.66 | 318.48 | 943.54 | 0.4545 | 0.3586 |
| `full_sft_epoch2_fp32_sp4` | 0.35 | 0.4815 | 0.3090 | 0.3952 | 26.75 | 2.01 | 5.99 | 287.94 | 813.06 | 0.3030 | 0.3103 |
| `lora_both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.5 | 0.5556 | 0.4551 | 0.5053 | 73.02 | 1.68 | 5.32 | 249.19 | 1115.38 | 0.5758 | 0.4276 |
| `full_sft_epoch1_fp32_sp4` | 0.5 | 0.5344 | 0.4101 | 0.4723 | 74.18 | 1.77 | 5.51 | 253.66 | 1174.27 | 0.4545 | 0.4000 |
| `full_sft_epoch2_fp32_sp4` | 0.5 | 0.5608 | 0.3652 | 0.4630 | 77.08 | 1.88 | 5.71 | 272.56 | 1224.30 | 0.3636 | 0.3655 |

## Delta vs LoRA Reference

Positive accuracy deltas are improvements over `lora_both_higher_dpi_plus_unans02503505_e1_64k_sp2`. Positive core-time/tool-call/turn deltas mean slower or more tool-use-heavy.

| Run | Rescale | Avg acc delta | Longdoc delta | MMLong delta | Core time delta s | Tool-call delta | Turn delta | MMLong unanswerable delta | MMLong answerable delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_sft_epoch1_fp32_sp4` | 0.25 | +0.0078 | +0.0211 | -0.0056 | -0.39 | +0.01 | +0.01 | +0.0909 | -0.0275 |
| `full_sft_epoch2_fp32_sp4` | 0.25 | -0.0117 | +0.0158 | -0.0393 | -1.15 | -0.38 | -0.76 | +0.0909 | -0.0689 |
| `full_sft_epoch1_fp32_sp4` | 0.35 | -0.0218 | -0.0212 | -0.0225 | +0.95 | +0.40 | +0.79 | +0.0303 | -0.0345 |
| `full_sft_epoch2_fp32_sp4` | 0.35 | -0.0661 | -0.0423 | -0.0899 | +0.44 | +0.05 | +0.12 | -0.1212 | -0.0828 |
| `full_sft_epoch1_fp32_sp4` | 0.5 | -0.0330 | -0.0212 | -0.0450 | +1.17 | +0.09 | +0.19 | -0.1213 | -0.0276 |
| `full_sft_epoch2_fp32_sp4` | 0.5 | -0.0423 | +0.0052 | -0.0899 | +4.06 | +0.20 | +0.39 | -0.2122 | -0.0621 |

## Epoch-2 Delta vs Epoch-1

Positive accuracy deltas mean epoch 2 improved over epoch 1. Positive core-time/tool-call/turn deltas mean epoch 2 is slower or more tool-use-heavy.

| Rescale | Avg acc delta | Longdoc delta | MMLong delta | Core time delta s | Tool-call delta | Turn delta | Gen-token delta | Tool-token delta | MMLong unanswerable delta | MMLong answerable delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | -0.0195 | -0.0053 | -0.0337 | -0.76 | -0.38 | -0.77 | -30.04 | -115.36 | +0.0000 | -0.0414 |
| 0.35 | -0.0443 | -0.0211 | -0.0674 | -0.51 | -0.34 | -0.68 | -30.54 | -130.49 | -0.1515 | -0.0483 |
| 0.5 | -0.0093 | +0.0264 | -0.0449 | +2.90 | +0.10 | +0.20 | +18.89 | +50.03 | -0.0909 | -0.0345 |

## Broad Eval

I did not find a matched epoch-1 broad eval for the same `full_sft_epoch1_fp32_sp4` run. The available epoch-2 broad results are:

| Run | Rescale | Broad avg acc | Core time s | Avg tool calls | Avg turns | Gen tokens | Tool tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_sft_epoch2_fp32_sp4` | 0.25 | 0.4634 | 7.47 | 3.06 | 8.87 | 491.64 | 821.89 |
| `full_sft_epoch2_fp32_sp4` | 0.35 | 0.4809 | 8.02 | 2.45 | 7.65 | 414.81 | 1022.12 |
| `full_sft_epoch2_fp32_sp4` | 0.5 | 0.5041 | 11.39 | 2.11 | 6.82 | 361.83 | 1341.53 |

Per-benchmark epoch-2 broad accuracy:

| Rescale | DUDE | LongDocURL | MMLite | MMLongBench | MPDocVQA | O3Bench | Avg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.5250 | 0.5550 | 0.4350 | 0.3600 | 0.7400 | 0.1652 | 0.4634 |
| 0.35 | 0.5500 | 0.5250 | 0.4650 | 0.3850 | 0.7400 | 0.2203 | 0.4809 |
| 0.5 | 0.5750 | 0.5350 | 0.4950 | 0.4450 | 0.7050 | 0.2696 | 0.5041 |

## Takeaways

- Full-SFT epoch 1 is the best of the full-SFT variants on highpage. Epoch 2 reduces average accuracy at all three rescale settings.
- The epoch-2 regression is mostly from `mmlongbench0507_highpage`. At rescale `0.5`, epoch 2 improves LongDocURL by `+0.0264` over epoch 1, but MMLongBench drops by `-0.0449`.
- The LoRA reference remains strongest overall at rescale `0.35` and `0.5`, mainly because it preserves better MMLongBench accuracy and much better MMLong unanswerable accuracy.
- Full-SFT epoch 1 only beats the LoRA reference at rescale `0.25` by a small amount (`+0.0078` avg acc). At `0.35` and `0.5`, it is lower by `-0.0218` and `-0.0330`.
- Epoch 2 also appears less tool-use-heavy at `0.25` and `0.35` than epoch 1, but that reduction does not translate to better accuracy.
- For broad eval, epoch-2 accuracy increases with rescale from `0.4634` to `0.5041`, but there is no matched epoch-1 broad result in the located outputs, so this is not an epoch comparison.
