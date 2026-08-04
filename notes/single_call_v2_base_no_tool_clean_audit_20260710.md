# single_call_v2 Audit On base_no_tool_no_system_clean_20260616

Date: 2026-07-10

## Setup

- Source run: `workspace/standalone_full_eval_num_trials1_base_no_tool_no_system_rescale035_05_localmodel_20260616/base_no_tool_no_system_local/full5_no_tool_no_system/rescale05`
- Audit subset: `workspace/single_call_v2_audit_base_no_tool_no_system_clean_20260616_rescale05_20260710`
- Sampling: seed `20260710`; 50 `mmlite` rows and 10 rows each from `dude`, `longdocurl`, `mmlongbench`, and `mpdocvqa`.
- Judge mode: `single_call_v2`
- Judge model: `gpt-5-nano` only; fallback judge disabled.
- API/proxy environment: loaded from `/home/ywxzml3j/ywxzml3juser40/.bashrc`
- Current accepted score dir: `scores_single_call_v2_tight_prompt_v2_bashrc`
- Current accepted row audit table: `single_call_v2_row_audit_minimal_prompt_v2.md` (label-equivalent to current tightened prompt)

## Current Design

`single_call_v2` keeps only simple deterministic guards before querying `gpt-5-nano`:

- High-confidence multiple-choice handling: parse A-E options, extract an explicit final selected label, or match a concise answer to exactly one option.
- Exact normalized full-answer match against GT.

The brittle semantic shortcuts are not used by `single_call_v2`: title/section matching, list matching, entity-token matching, and regex unanswerable matching are legacy `single_call_v1` / raw-final-answer fallback behavior only.

The accepted prompt adds generic guidance for document QA equivalence:

- Accept essential-equivalent labels, titles, and extracted text while ignoring harmless source/location qualifiers.
- For names, labels, organizations, IDs, option letters, dates, numbers, and extracted text, require the exact essential identifier.
- Do not accept spelling/OCR variants, missing characters, or extra characters inside identifiers unless GT explicitly lists aliases.
- Accept a full-name or expanded-name answer for a shorter GT name only when it contains the GT name as an exact token/string and does not identify a different entity.
- Accept paragraph/title/section answers that identify or quote the essential GT text, even with location/context.
- Require exact requested item sets for list/multi-part questions.
- For unanswerable GT, accept clear statements that the answer cannot be determined, the requested page/table/field/item is absent, no matching item exists, or none satisfy the question.

Focused regression tests:

```bash
PYTHONPATH=/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:. \
  pytest -q tests/utils/reward_score/test_vsearch_batch_mc_judge_on_cpu.py
```

Result: `16 passed`.

## 90-Row Audit Results

These are model-answer scores produced by the judge, not judge-reliability percentages.

| Benchmark | Rows | Correct | Accuracy | Previous minimal correct |
|---|---:|---:|---:|---:|
| `dude` | 10 | 8 | 0.800 | 7 |
| `longdocurl` | 10 | 8 | 0.800 | 7 |
| `mmlite` | 50 | 30 | 0.600 | 30 |
| `mmlongbench` | 10 | 6 | 0.600 | 6 |
| `mpdocvqa` | 10 | 9 | 0.900 | 9 |
| **overall** | **90** | **61** | **0.678** | **59** |

Rows improved versus the previous minimal prompt:

- `dude_e30bed1bc3dfd1d55ef351b1aedd5b1a_12c864c6125dea55ef130550cd199544`: GT unanswerable; answer says no table is present and no information is given in a table. Now judged correct.
- `longdocurl_free_gpt4o_4091310_43_59_9`: GT `A Reparation instead of punishment`; answer quotes `Reparation instead of punishment` and gives the slide/paragraph context. Now judged correct.

Known unresolved text-level issue:

- `dude_fa7dc867be18acbc3c5b90e79495b467_3943631265c1d2fff3557b73d10d1acf`: GT unanswerable; answer mentions visible typewritten text on another image, but concludes there is no page 5 and no page-5 typewriter text can be identified. Prompt-v2 still judges this wrong.

## Judge-Side Improvement

Manual expected labels were derived from the text-level inspection of the 90 audited rows.

| Judge | Agreement With Manual Labels | Errors | False Positives | False Negatives |
|---|---:|---:|---:|---:|
| Legacy judge | 87/90 | 3 | 0 | 3 |
| `single_call_v1` | 85/90 | 5 | 3 | 2 |
| `single_call_v2` prompt-v2 | 89/90 | 1 | 0 | 1 |
| `single_call_v2` tightened prompt | 89/90 | 1 | 0 | 1 |

Relative to the legacy judge, the tightened prompt fixes 2/90 rows and reduces observed text-level errors from 3 to 1. Relative to `single_call_v1`, it fixes 4/90 rows and removes the observed MC false positives.

Manual inspection:

- The changed `dude` and `longdocurl` rows above look correct from question/GT/answer text.
- The remaining wrong non-`mmlite` rows were checked from question/GT/answer text and appear correctly judged wrong.
- The `mmlite` rows with explicit selected labels are mechanically consistent with selected-label-versus-GT scoring.
- The `mmlite` wrong rows without a mechanically extracted final label were manually checked; their conclusions are wrong against the GT label.

Estimated judge reliability on this text-level audit is therefore `89/90`. This is an estimate because the audit did not reopen source images/documents for visual verification.

## Rejected Prompt Experiment

I tried a broader unanswerable clarification:

> judge by final conclusion; related visible content is okay if not given as the requested answer

That variant was scored in `scores_single_call_v2_minimal_prompt_v3_bashrc`. It did not fix the page-5 row and flipped the no-table row back to wrong, reducing the score to `60/90`. I reverted this prompt change.

I also tried prompt-v4, which explicitly allowed one-character OCR/transcription variants when the distinctive name/text is otherwise the same. This is now rejected: in the RL audit it flips `Donn Schlote` versus GT `"Donn Schlotec"` from wrong to correct, but for names/identifiers the GT should remain authoritative unless aliases are explicit or the source image/document is verified.

I then tried an overly strict simplification in `scores_single_call_v2_tight_prompt_bashrc`. This is rejected because it introduced two false negatives relative to prompt-v2:

- `dude_e30bed1bc3dfd1d55ef351b1aedd5b1a_12c864c6125dea55ef130550cd199544`: GT unanswerable; answer clearly says no table is present.
- `longdocurl_free_gpt4o_4152351_46_69_12`: GT `Ruth`; answer gives `Ruth C. Cohn`, which is a valid exact-token full-name expansion.

The accepted tightened prompt was rescored in `scores_single_call_v2_tight_prompt_v2_bashrc`. It has zero label changes versus accepted prompt-v2 on the 90-row audit, keeps the same `89/90` text-level reliability estimate, and the motivating `Donn Schlote` vs GT `"Donn Schlotec"` row remains wrong (`accuracy_reward=0.0`).

## Heldout Audit

To reduce overfitting risk, I sampled another 90 rows from the same source run, excluding all rows in the first audit subset.

- Heldout subset: `workspace/single_call_v2_audit_base_no_tool_no_system_clean_20260616_rescale05_heldout_20260710`
- Sampling: seed `20260711`; 50 `mmlite` rows and 10 rows each from `dude`, `longdocurl`, `mmlongbench`, and `mpdocvqa`.
- Score dir: `scores_single_call_v2_minimal_prompt_v2_bashrc`
- Row audit table: `single_call_v2_heldout_row_audit.md`

Heldout model-answer scores:

| Benchmark | Rows | Correct | Accuracy |
|---|---:|---:|---:|
| `dude` | 10 | 9 | 0.900 |
| `longdocurl` | 10 | 10 | 1.000 |
| `mmlite` | 50 | 25 | 0.500 |
| `mmlongbench` | 10 | 7 | 0.700 |
| `mpdocvqa` | 10 | 9 | 0.900 |
| **overall** | **90** | **60** | **0.667** |

Heldout manual inspection:

- 41 rows were mechanically checkable MC rows where the extracted selected label agreed with the judge score.
- 49 rows needed text-level review; I inspected their question, GT, final answer, and judge score.
- No obvious judge false positives/false negatives were found from the text-level heldout review.
- Examples of judged-wrong rows looked genuinely wrong: missing/mismatched MC choices in `mmlite`, wrong numeric/date answer in `mpdocvqa`, incomplete list answer in `mmlongbench`, and unanswerable GT answered with a concrete phone number in `dude`.

## Caveat

This audit verifies judge behavior against the question text, model answer, extracted/final answer, and GT. It does not visually verify the original source images/documents.
