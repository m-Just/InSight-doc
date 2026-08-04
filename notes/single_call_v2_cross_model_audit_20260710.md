# single_call_v2 Cross-Model Audit

Date: 2026-07-10

## Improvement Over Previous Judges

Manual expected labels are from text-level inspection of the original 90-row audit sampled from `base_no_tool_no_system_clean_20260616`.

| Judge | Agreement With Manual Labels | Errors | False Positives | False Negatives |
|---|---:|---:|---:|---:|
| Legacy judge | 87/90 | 3 | 0 | 3 |
| `single_call_v1` | 85/90 | 5 | 3 | 2 |
| `single_call_v2` prompt-v2 | 89/90 | 1 | 0 | 1 |
| `single_call_v2` tightened prompt | 89/90 | 1 | 0 | 1 |

Observed improvement:

- Versus legacy: +2/90 manually correct labels; observed text-level error count reduced from 3 to 1.
- Versus `single_call_v1`: +4/90 manually correct labels; observed text-level error count reduced from 5 to 1.
- The tightened prompt has zero label changes versus accepted prompt-v2 on the original 90-row audit.
- The largest `single_call_v1` issue was MC false positives caused by accepting a mentioned GT option even when the model selected a different final option.

## Current Judge

`single_call_v2` keeps the deterministic side minimal:

- High-confidence MC label/concise-option handling.
- Exact normalized full-answer match.

Everything semantic is judged by `gpt-5-nano`; fallback judge is disabled in these audits.

For names/identifiers, the tightened prompt requires exact essential identifiers and ignores only non-essential source/location qualifiers. It does not accept spelling/OCR variants, missing characters, or extra characters inside identifiers unless GT explicitly provides aliases or the source image/document is verified. It still accepts full-name expansions like `Ruth C. Cohn` for GT `Ruth` when the GT appears as an exact token/string and the answer does not identify a different entity.

Focused tests:

```bash
PYTHONPATH=/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:. \
  pytest -q tests/utils/reward_score/test_vsearch_batch_mc_judge_on_cpu.py
```

Result: `16 passed`.

## Base No-Tool Audits

Main 90-row audit:

- Audit dir: `workspace/single_call_v2_audit_base_no_tool_no_system_clean_20260616_rescale05_20260710`
- Accepted score dir: `scores_single_call_v2_tight_prompt_v2_bashrc`
- Row audit: `single_call_v2_row_audit_minimal_prompt_v2.md` (label-equivalent to current tightened prompt)
- Text-level manual judge reliability estimate: `89/90`

Heldout 90-row audit:

- Audit dir: `workspace/single_call_v2_audit_base_no_tool_no_system_clean_20260616_rescale05_heldout_20260710`
- Accepted score dir: `scores_single_call_v2_minimal_prompt_v2_bashrc`
- Row audit: `single_call_v2_heldout_row_audit.md`
- No obvious text-level judge false positives/false negatives found.

## RL Audit: rl_ckpt700

Source response run:

`workspace/rl_ckpt700_broad_standalone_rescale025_035_05_3trials_wc4_overlap_20260608_171726_wc4_overlap/rescale05_trial0`

Audit:

- Audit dir: `workspace/single_call_v2_audit_rl_ckpt700_broad_rescale05_trial0_20260710`
- Sampling: 50 `mmlite200`, 10 each from `dude200`, `longdocurl200`, `mmlongbench200`, `mpdocvqa200`.
- Accepted score dir: `scores_single_call_v2_minimal_prompt_v2_bashrc`
- Row audit: `single_call_v2_rl_row_audit.md`

Model-answer scores on this sampled subset:

| Score Source | Correct / 90 |
|---|---:|
| Source run score | 57 |
| `single_call_v2` prompt-v2 | 59 |

Manual inspection of changed rows:

- `reason/monitoring/calculate/0115`: source `1`, v2 `0`; model selected `E` while GT is `D`, so v2 correctly removes a false positive.
- `dude_0c1b3fa0dc2f09d04fcf4e3e118ec72b_c73c26514c7cc7394996a91ad345682b`: source `0`, v2 `1`; answer gives `Volunteer Expenses`, matching GT.
- `longdocurl_free_gpt4o_4083860_40_69_16`: source `0`, v2 `1`; answer gives `61.8%`, matching GT.
- `longdocurl_summary2tab_4112595_94_95_7`: source `0`, v2 `1`; answer gives both requested table names.

Rejected prompt-v4 would additionally score `longdocurl_free_gpt4o_4114758_43_72_2` as correct because it allowed `Donn Schlote` as a one-character variant of GT `"Donn Schlotec"`. That is now considered too permissive for names/identifiers without explicit aliases or source verification, so prompt-v4 is not accepted.

With the current tightened prompt, the same `Donn Schlote` row was sanity-checked directly and remains wrong (`accuracy_reward=0.0`).

No remaining obvious v2/tightened-prompt judge issue was found from text-level inspection of changed rows and compact review of non-mechanically checked rows.

## RL Audit: rl_v3_unans014_judge_single_call_v1_ckpt1k

Source response run:

`workspace/standalone_full_eval_full_rl_v3_unans014_judge_single_call_v1_ckpt1k_rescale025_035_05_1trial_gpu0_3_rerun_20260704/rl_v3_unans014_judge_single_call_v1_ckpt1k/full5_tool/rescale05`

Audit:

- Audit dir: `workspace/single_call_v2_audit_rl_v3_unans014_ckpt1k_full5_rescale05_20260710`
- Sampling: 50 `mmlite`, 10 each from `dude`, `longdocurl`, `mmlongbench`, `mpdocvqa`.
- Accepted score dir: `scores_single_call_v2_minimal_prompt_v2_bashrc`
- Row audit: `single_call_v2_rl_row_audit.md`

Model-answer scores on this sampled subset:

| Score Source | Correct / 90 |
|---|---:|
| Source run score | 48 |
| `single_call_v2` prompt-v2 | 46 |

Manual inspection of changed rows:

- `Reasoning/Autonomous_Driving/Attention_TrafficSignal/0015`: source `1`, v2 `0`; model says the question is unanswerable/no signal, but GT is `D` (`Crosswalk`), so v2 correctly removes a false positive.
- `reasoning/diagram_and_table/diagram/0120`: source `1`, v2 `0`; model selects `E`, while GT is `C`, so v2 correctly removes a false positive.

No remaining obvious v2 judge issue was found from text-level inspection of changed rows and compact review of non-mechanically checked rows.

## Caveat

All manual checks are text-level checks over question, GT, final answer, and judge score. They do not reopen the underlying images/documents for visual verification.
