# Highpage LoRA Unanswerable-Medium Comparison

Date: 2026-05-18

This note compares the LoRA runs with added synthetic unanswerable medium data against their matched medium-only baselines on `longdocurl0507_highpage` and `mmlongbench0507_highpage`.

## Runs

| Group | Run | Output root |
|---|---|---|
| Basic baseline | `lora_basic_medium_only_epoch1_32k_sp1` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_medium_only_epoch1_32k_sp1_20260517` |
| Basic + unanswerable | `lora_basic_unanswerable025_len32768_sp1_epoch1` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_unanswerable025_20260517` |
| Both higher-DPI baseline | `lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_epoch1_sp2_64k_retry_no_expandable_20260517` |
| Both higher-DPI + unanswerable | `lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_unanswerable02503505_20260517` |
| Both higher-DPI + unanswerable, epoch 2 | `lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch2_retry1` | `/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_unanswerable02503505_epoch2_retry1_20260518` |

Unanswerable-data additions:

| New run | Added data |
|---|---|
| `lora_basic_unanswerable025_len32768_sp1_epoch1` | synthetic unanswerable medium base-model-order data at rescale `0.25` |
| `lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1` | synthetic unanswerable medium base-model-order data at rescale `0.25`, `0.35`, and `0.5` |
| `lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch2_retry1` | same data as epoch 1, trained for 2 epochs |

## Full Metrics

| Run | Rescale | Longdoc acc | MMLong acc | Avg acc | Core time s | Avg tool calls | Avg turns | Gen tokens | Tool tokens | MMLong unanswerable acc | MMLong answerable acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `basic_medium_only_e1_32k_sp1` | 0.25 | 0.4921 | 0.2303 | 0.3612 | 13.71 | 2.56 | 7.07 | 349.31 | 559.19 | 0.1212 | 0.2552 |
| `basic_plus_unans025_e1_32k_sp1` | 0.25 | 0.4709 | 0.3090 | 0.3899 | 14.99 | 2.62 | 7.20 | 362.36 | 533.76 | 0.3939 | 0.2897 |
| `basic_medium_only_e1_32k_sp1` | 0.35 | 0.5079 | 0.3258 | 0.4169 | 26.66 | 2.11 | 6.17 | 301.37 | 810.90 | 0.2121 | 0.3517 |
| `basic_plus_unans025_e1_32k_sp1` | 0.35 | 0.5291 | 0.3652 | 0.4471 | 27.65 | 2.04 | 6.04 | 292.84 | 743.89 | 0.4545 | 0.3448 |
| `basic_medium_only_e1_32k_sp1` | 0.5 | 0.5291 | 0.3483 | 0.4387 | 72.23 | 1.59 | 5.13 | 239.83 | 1014.77 | 0.2424 | 0.3724 |
| `basic_plus_unans025_e1_32k_sp1` | 0.5 | 0.5661 | 0.4101 | 0.4881 | 77.23 | 1.79 | 5.54 | 267.47 | 1165.38 | 0.4545 | 0.4000 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | 0.25 | 0.4709 | 0.2753 | 0.3731 | 14.04 | 2.70 | 7.35 | 381.00 | 546.29 | 0.0606 | 0.3241 |
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.25 | 0.4392 | 0.3258 | 0.3825 | 15.00 | 3.09 | 8.15 | 407.56 | 647.79 | 0.3333 | 0.3241 |
| `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.25 | 0.4180 | 0.2978 | 0.3579 | 14.41 | 2.60 | 7.16 | 360.31 | 564.80 | 0.2727 | 0.3034 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | 0.35 | 0.5344 | 0.3371 | 0.4357 | 26.89 | 2.10 | 6.14 | 296.90 | 793.99 | 0.1212 | 0.3862 |
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.35 | 0.5238 | 0.3989 | 0.4613 | 26.31 | 1.96 | 5.87 | 278.13 | 748.47 | 0.4242 | 0.3931 |
| `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.35 | 0.4762 | 0.3202 | 0.3982 | 27.09 | 1.93 | 5.84 | 284.25 | 807.80 | 0.3333 | 0.3172 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | 0.5 | 0.5344 | 0.3820 | 0.4582 | 67.97 | 1.58 | 5.11 | 241.86 | 1074.07 | 0.0909 | 0.4483 |
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.5 | 0.5556 | 0.4551 | 0.5053 | 73.02 | 1.68 | 5.32 | 249.19 | 1115.38 | 0.5758 | 0.4276 |
| `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.5 | 0.5079 | 0.3708 | 0.4394 | 82.20 | 1.75 | 5.47 | 256.92 | 1175.38 | 0.3030 | 0.3862 |

## Delta Versus Matched Baselines

Positive accuracy deltas are improvements. Positive core-time/tool-call/turn deltas mean slower or more tool-use-heavy.

| Baseline | New run | Rescale | Avg acc delta | Longdoc delta | MMLong delta | Core time delta s | Tool-call delta | Turn delta | MMLong unanswerable delta | MMLong answerable delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `basic_medium_only_e1_32k_sp1` | `basic_plus_unans025_e1_32k_sp1` | 0.25 | +0.0287 | -0.0212 | +0.0787 | +1.28 | +0.06 | +0.13 | +0.2727 | +0.0345 |
| `basic_medium_only_e1_32k_sp1` | `basic_plus_unans025_e1_32k_sp1` | 0.35 | +0.0302 | +0.0212 | +0.0393 | +0.99 | -0.07 | -0.14 | +0.2424 | -0.0069 |
| `basic_medium_only_e1_32k_sp1` | `basic_plus_unans025_e1_32k_sp1` | 0.5 | +0.0494 | +0.0370 | +0.0618 | +5.00 | +0.20 | +0.41 | +0.2121 | +0.0276 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.25 | +0.0094 | -0.0317 | +0.0506 | +0.96 | +0.40 | +0.81 | +0.2727 | +0.0000 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.35 | +0.0256 | -0.0106 | +0.0618 | -0.58 | -0.14 | -0.27 | +0.3030 | +0.0069 |
| `both_higher_dpi_medium_only_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | 0.5 | +0.0471 | +0.0212 | +0.0730 | +5.05 | +0.10 | +0.22 | +0.4848 | -0.0207 |

## Epoch-2 Retry1 Versus Epoch-1

Positive accuracy deltas are improvements for epoch 2. Positive core-time/tool-call/turn deltas mean slower or more tool-use-heavy.

| Epoch-1 run | Epoch-2 run | Rescale | Avg acc delta | Longdoc delta | MMLong delta | Core time delta s | Tool-call delta | Turn delta | MMLong unanswerable delta | MMLong answerable delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.25 | -0.0246 | -0.0212 | -0.0281 | -0.59 | -0.49 | -0.99 | -0.0606 | -0.0207 |
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.35 | -0.0631 | -0.0476 | -0.0787 | +0.78 | -0.02 | -0.03 | -0.0909 | -0.0759 |
| `both_higher_dpi_plus_unans02503505_e1_64k_sp2` | `both_higher_dpi_plus_unans02503505_e2_retry1_64k_sp2` | 0.5 | -0.0659 | -0.0476 | -0.0843 | +9.19 | +0.07 | +0.14 | -0.2728 | -0.0414 |

## Takeaways

- Both epoch-1 unanswerable-data runs improve average accuracy at all tested rescale values.
- The improvement is driven mainly by `mmlongbench0507_highpage`, especially its unanswerable subset.
- The best new result is `both_higher_dpi_plus_unans02503505_e1_64k_sp2` at rescale `0.5`: avg acc `0.5053`, longdoc `0.5556`, mmlong `0.4551`, core time `73.02s`.
- The epoch-2 retry1 run is worse than epoch 1 at all tested rescales. The largest drop is at rescale `0.5`: avg acc `-0.0659`, driven by MMLong `-0.0843` and MMLong unanswerable `-0.2728`.
- The basic `32k/sp1` epoch-1 unanswerable run has a cleaner gain profile than the higher-DPI epoch-1 run at `0.25`: it improves avg acc by `+0.0287`, while the higher-DPI variant improves only `+0.0094` and loses `-0.0317` on longdoc.
- At rescale `0.5`, both epoch-1 unanswerable runs get large gains with about `+5s` extra average core inference time.
- The unanswerable additions do not simply suppress tool use. Tool calls and turns are roughly stable, with small increases at `0.5`; the main behavior change appears to be better unanswerable handling on MMLong.
- The regenerated plot with epoch-2 retry1 is at [`highpage_lora_epoch1_unanswerable_vs_base_2026-05-18.png`](/scratch/ywxzml3j/likaican/src/InSight-doc/verl/notes/generated/highpage_lora_epoch1_unanswerable_vs_base_2026-05-18.png).
