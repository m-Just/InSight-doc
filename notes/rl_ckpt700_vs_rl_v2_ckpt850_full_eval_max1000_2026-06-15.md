# rl_ckpt700 vs rl_v2_ckpt850 Full-Benchmark Sweep

Comparison uses the standalone full-5 benchmark evals with `max_samples=1000`, `num_trials=3`, Ray/vLLM backend, and rescale `0.25/0.35/0.5`.

Sources:

- `rl_ckpt700`: `workspace/standalone_full_eval_max1000_rl_ckpt700_rescale025_035_05_gpu0_3_conc8x4_20260614_183300/sweep_summary.tsv`
- `rl_v2_ckpt850`: `workspace/standalone_full_eval_max1000_rl_v2_ckpt850_rescale025_035_05_gpu4_7_conc8x4_20260614_190000/sweep_summary.tsv`

The `rl_v2_ckpt850` `rescale=0.5` row was repaired by rerunning failed rows with Ray/vLLM `gpu_memory_utilization=0.8`; all rows are now scored.

## Macro Average

Macro average is over `dude`, `longdocurl`, `mmlite`, `mmlongbench`, and `mpdocvqa`.

| model | rescale | valid | acc | core_s | prompt_tok | total_tok | resp_tok | valid_tool_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rl_ckpt700 | 0.25 | 1.000 | 0.5978 | 6.83 | 3448 | 4263 | 815 | 2.44 |
| rl_v2_ckpt850 | 0.25 | 1.000 | 0.6227 | 9.68 | 3448 | 4575 | 1127 | 2.56 |
| rl_ckpt700 | 0.35 | 1.000 | 0.6377 | 7.23 | 6319 | 7174 | 855 | 1.92 |
| rl_v2_ckpt850 | 0.35 | 1.000 | 0.6553 | 9.35 | 6319 | 7535 | 1216 | 2.14 |
| rl_ckpt700 | 0.5 | 1.000 | 0.6468 | 9.96 | 12173 | 13226 | 1053 | 1.68 |
| rl_v2_ckpt850 | 0.5 | 1.000 | 0.6408 | 11.33 | 12173 | 13608 | 1435 | 1.89 |

## Delta: rl_v2_ckpt850 - rl_ckpt700

| rescale | delta_acc | delta_core_s | delta_resp_tok | delta_valid_tool_calls |
|---:|---:|---:|---:|---:|
| 0.25 | +0.0249 | +2.85 | +312 | +0.12 |
| 0.35 | +0.0176 | +2.13 | +361 | +0.22 |
| 0.5 | -0.0060 | +1.37 | +382 | +0.22 |

## Per-Benchmark Accuracy Delta

| rescale | benchmark | rl_ckpt700 | rl_v2_ckpt850 | delta |
|---:|---|---:|---:|---:|
| 0.25 | dude | 0.6703 | 0.6849 | +0.0146 |
| 0.25 | longdocurl | 0.5227 | 0.5733 | +0.0507 |
| 0.25 | mmlite | 0.5212 | 0.5485 | +0.0273 |
| 0.25 | mmlongbench | 0.4718 | 0.5077 | +0.0359 |
| 0.25 | mpdocvqa | 0.8030 | 0.7990 | -0.0040 |
| 0.35 | dude | 0.6886 | 0.7151 | +0.0265 |
| 0.35 | longdocurl | 0.6187 | 0.6187 | +0.0000 |
| 0.35 | mmlite | 0.5455 | 0.5576 | +0.0121 |
| 0.35 | mmlongbench | 0.4821 | 0.5385 | +0.0564 |
| 0.35 | mpdocvqa | 0.8537 | 0.8468 | -0.0070 |
| 0.5 | dude | 0.6895 | 0.7032 | +0.0137 |
| 0.5 | longdocurl | 0.6560 | 0.6240 | -0.0320 |
| 0.5 | mmlite | 0.5212 | 0.5606 | +0.0394 |
| 0.5 | mmlongbench | 0.4974 | 0.4564 | -0.0410 |
| 0.5 | mpdocvqa | 0.8697 | 0.8597 | -0.0100 |

## Takeaway

`rl_v2_ckpt850` is better at lower and medium rescale, especially on `longdocurl`/`mmlongbench` at `0.25` and `mmlongbench` at `0.35`. At `0.5`, it gives up the macro lead because `longdocurl` and `mmlongbench` drop relative to `rl_ckpt700`.

The v2 checkpoint is consistently slower and more verbose: it adds about `+1.37` to `+2.85` seconds macro core time, `+312` to `+382` response tokens, and `+0.12` to `+0.22` valid tool calls per sample.
