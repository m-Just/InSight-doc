# single_call_v2 MC Judge Reliability Audit

Date: 2026-07-10

## Change Summary

The fixed judge behavior is exposed as `single_call_v2`. The historical
`single_call_v1` behavior is kept unchanged for provenance and reproducibility,
including its known MC false-positive behavior when an answer mentions the GT
option while rejecting it.

The MC part of `single_call_v2` is conservative and high precision.

- Parse MC options only when labels form a consecutive sequence starting at `A`.
- Support both `(A)` / `(A).` and `A.` option-label styles.
- Reject abbreviation false positives such as `Carol A. Tozzi`, `Ph.D.`, `D.C.No`, and `B.A.`.
- Extract the selected option only from final-selection cues, not from any mention of `(A)`-`(E)`.
- Treat direct concise option-text answers as deterministic correct only when they match the GT option.
- Treat rejection / unanswerable answers as option `E` only when option `E` is itself a rejection option.
- Otherwise defer to the `gpt-5-nano` single-call judge prompt.

Important API setting: `gpt-5-nano` judge calls need a sufficiently large completion budget. A small budget can be exhausted by reasoning tokens and return empty visible content. The judge path uses `max_tokens=2048`; this produced explicit `<correct>` / `<wrong>` tags in the audit.

## Tests

Command:

```bash
source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest
PYTHONPATH=/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:. \
  pytest -q tests/utils/reward_score/test_vsearch_batch_mc_judge_on_cpu.py
```

Result after provenance split and the base-no-tool audit fixes: `16 passed`.

Covered cases:

- GT option mentioned but explicitly rejected, final selected option differs.
- Final explicit selection from `answer`, `option`, `choice`, `correct choice`, and punctuated tail labels.
- OCR answers starting with `A`-`E`, e.g. `B HH 30H`, not mistaken for option `B`.
- `A.` / `B.` longdocurl-style options.
- Abbreviation false positives rejected.
- Count-zero non-`E` answers not mistaken as unanswerable option `E`.
- Ambiguous option-comparison paragraphs defer to `gpt-5-nano`.

## Benchmark MC Coverage

Representative file:

`workspace/standalone_full_eval_full_rl_v3_unans014_judge_single_call_v1_ckpt1k_rescale025_035_05_1trial_gpu0_3_rerun_20260704/rl_v3_unans014_judge_single_call_v1_ckpt1k/full5_tool/rescale05/scores/samples.jsonl`

Parsed MC rows after the fix:

| Benchmark | Rows | Parsed MC Rows | GT `A`-`E` Rows | Parsed MC with GT Option |
|---|---:|---:|---:|---:|
| `mmlite` | 1919 | 1919 | 1919 | 1919 |
| `longdocurl` | 2207 | 102 | 103 | 101 |
| `dude` | 6260 | 0 | 2 | 0 |
| `mmlongbench` | 1089 | 0 | 1 | 0 |
| `mpdocvqa` | 5109 | 0 | 0 | 0 |

Conclusion: `mmlite` is the only fully MC benchmark. `longdocurl` has a small real MC subset. The other benchmarks have occasional `A`-`E` ground truths or abbreviations, but they are not parsed as MC after the consecutive-label guard.

## Rule Agreement Snapshot

For parsed MC rows in the representative file:

| Benchmark | Patched Rule | Existing Score | Count |
|---|---:|---:|---:|
| `mmlite` | correct | 1 | 894 |
| `mmlite` | wrong | 0 | 885 |
| `mmlite` | defer | 0 | 81 |
| `mmlite` | defer | 1 | 23 |
| `mmlite` | wrong | 1 | 36 |
| `longdocurl` | correct | 1 | 88 |
| `longdocurl` | wrong | 0 | 9 |
| `longdocurl` | defer | 0 | 3 |
| `longdocurl` | defer | 1 | 2 |

The `wrong` / existing-score-1 rows are mostly legacy false positives where the answer selected another option, often `E`, while mentioning the GT in reasoning.

## API Audit

Fresh audit artifact:

`notes/generated/single_call_v1_mc_patch_api_audit_20260710_fresh.jsonl`

This artifact was generated from the patched behavior before the mode was
renamed; it corresponds to `single_call_v2`, not legacy `single_call_v1`.

Settings:

- API/proxy values imported from `/home/ywxzml3j/ywxzml3juser40/.bashrc`.
- Judge model: `gpt-5-nano`.
- `max_tokens=2048`.
- Sample size: 24 rows.
- API-backed rows: 18.
- API errors: 0.

Outcome summary:

| Previous Score | Patched Rule | Patched/API Score | Count |
|---:|---|---:|---:|
| 0 | defer | 0 | 12 |
| 1 | defer | 1 | 5 |
| 1 | correct | 1 | 4 |
| 1 | wrong | 0 | 1 |
| 0 | wrong | 0 | 1 |
| 0 | defer | 1 | 1 |

Manual inspection highlights:

- `Reasoning/Autonomous_Driving/Prediction_Intention_Ego/0241`: GT `A`, answer explicitly rejects `A` and selects `E`; patched rule marks wrong. This fixes a legacy false positive.
- `reason/monitoring/calculate/0016`: GT `D`, answer says no relevant vehicles and final count is `0 (D)`; patched rule marks correct instead of treating it as option `E`.
- `perception/ocr_cc/license/0663`: GT `A`, answer is `B HH 30H`; patched rule marks correct and does not confuse leading `B` with option `B`.
- `perception/ocr_cc/text_recog/0345`: GT `C`, answer text matches GT string but says it matches option `D`; patched rule marks wrong because the final selected option is `D`.
- `longdocurl_free_gpt4o_4060934_85_114_5`: longdocurl `A.`/`B.` option style; `gpt-5-nano` marks the deferred answer correct.
- `perception/remote_sensing/color/0824`: GT `E`, answer says the bottom-right boat/color cannot be determined; `gpt-5-nano` marks correct, fixing a legacy false negative.

## Reliability Assessment

The updated judge is reliable enough for the current MC failure mode:

- It no longer passes an answer merely because it mentions the GT option.
- It handles the two MC formats present in the evals.
- It avoids false MC detection in non-MC benchmarks.
- Ambiguous cases are deferred to `gpt-5-nano` instead of over-fitting deterministic rules.

Residual risks:

- The manual audit checks consistency between question, GT, model final answer, and judge output; it does not re-verify visual ground truth from images.
- Some paraphrased MC answers still rely on the LLM fallback. This is intentional to keep deterministic rules high precision.
- Future endpoints should keep the 2048-token judge budget unless the GPT-5-compatible API exposes a separate reasoning-token control.
