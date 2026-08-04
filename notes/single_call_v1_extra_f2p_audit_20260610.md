# single_call_v1 Extra F2P Candidate Audit

Date: 2026-06-10

This audit mines additional candidate fail-to-pass examples for the `single_call_v1` judge beyond the original 40-per-benchmark fixture.

## Setup

- Candidate source: legacy-scored `rl_ckpt700` broad/highpage standalone runs.
- Candidate filter: legacy `accuracy_reward = 0.0`, `compute_score_success = true`.
- Candidate ranking: lexical overlap between GT and raw final answer, plus small boosts for title/section/entity/list/unanswerable-style questions.
- Rescored subset: top 80 candidates from `notes/generated/single_call_v1_more_f2p_candidates_20260610.jsonl`.
- Judge mode: `single_call_v1`.
- Judge model: `gpt-5-nano`.
- Result file: `notes/generated/single_call_v1_more_f2p_candidates_scored_top80_20260610.json`.

## Aggregate Result

| Metric | Value |
|---|---:|
| Legacy-fail candidates rescored | 80 |
| `single_call_v1` passes | 47 |
| Unique passed question ids | 30 |
| Compute failures | 0 |
| Likely correct unique F2P examples after manual inspection | 26 / 30 |
| Suspicious unique passes after manual inspection | 4 / 30 |

## Likely Correct F2P Examples

These look like genuine cases where the legacy extraction/judge path was too strict or extracted only a partial answer, while the raw final answer is correct.

| Source | Question id | Legacy extracted | GT / expected answer | Raw final answer summary |
|---|---|---|---|---|
| `longdocurl200` | `longdocurl_topic2title_4073993_5_6_1` | `A large cooperative banking group` | Two full section titles ending `(1/2)` and `(2/2)` | Gives both full titles. |
| `dude200` | `dude_f138f858022fc6aa0bf6bd55fbaa1dbf_eda20c5a0360f96ce1a8f620188043ec` | `Photo copy of photo ID` | `Photo copy of photo ID with address` | Includes the full phrase in context. |
| `longdocurl200` | `longdocurl_free_gpt4o_4192639_34_58_8` | `A Pomegranate in the Hand` | `A Pomegranate in the Hand of God` | Gives the full subtitle. |
| `longdocurl200` | `longdocurl_summary2title_4165048_2_31_6` | `Which information for conservation researchers?` | `2.1.3 Which information for conservation researchers?` | Gives the section title with section number context. |
| `mpdocvqa200` | `mpdocvqa_32348` | `Future Options` | Full title `FUTURE OPTIONS FOR THE HOWARD HEINZ ENDOWMENT HEALTH GRANTS PROGRAM` | Gives the full title. |
| `dude200` | `dude_a311c5a0ad91b9471b8763b8e5af7847_dacfbc141114dd45f1a906c86228f0a2` | `Why co-operative? How well?` | Two full questions under the section | Gives both full questions. |
| `longdocurl200` | `longdocurl_free_gemini15_pro_4165048_2_31_5` | `Mind the gap` | Full thesis title | Gives the full thesis title. |
| `longdocurl200` | `longdocurl_summary2title_4056089_89_115_8` | `#1 Designating Limited Common Property` | `SPECIAL RESOLUTION: #1:` | Gives `SPECIAL RESOLUTION: #1` plus descriptive qualifier. |
| `longdocurl0507_highpage` | `longdocurl_free_gpt4o_4003168_4_33_5` | `Cecilia Berg` | Four chapter authors | Gives all four authors, with `D. G. Joakim Larsson` variant. |
| `longdocurl200` | `longdocurl_free_gpt4o_4174181_15_44_5` | `Five domains with weights` | Five CompTIA PenTest+ domains and weights | Gives all five domains with percentages. |
| `longdocurl200` | `longdocurl_summary2tab_4112595_94_95_7` | `Strategic Goals; Key Upgrade Goals` | Two full table names | Gives both full table names. |
| `longdocurl200` | `longdocurl_extract_fig2tab_4076912_11_40_13` | `Figure 13` | Full Figure 13 title | Gives full figure title. |
| `longdocurl200` | `longdocurl_summary2tab_4098000_62_63_2` | `Missing Headers; Upcoming Headers` | Four table names | Gives all four table names. |
| `longdocurl200` | `longdocurl_free_gemini15_pro_4182674_55_84_2` | `Highland and Fife 11 each` | Full list of locations and counts | Gives all listed locations/counts. |
| `longdocurl200` | `longdocurl_free_gemini15_pro_4009036_81_110_12` | Age sizes only | Size codes `90/100`, `110/120`, `130/140`, `150/160` | Gives both age ranges and exact size codes. |

## Suspicious Passes

These are useful stress cases. They suggest `single_call_v1` can still be too permissive on some title/list questions where the model gives extra or wrong-level sections.

| Source | Question id | Why suspicious |
|---|---|---|
| `longdocurl0507_highpage` | `longdocurl_topic2title_4054021_53_71_7` | GT is `["2 SCOPE OF WORK", "Civil Engineering Work"]`, but answers mention broad sections like `Section 6` / `Section 8` and extra titles. |
| `longdocurl0507_highpage` | `longdocurl_topic2title_4051057_83_94_8` | GT is `["3.3.2 Adaptive Traits", "3.4.2 Adaptive Traits"]`, but answer includes extra sections such as `3.1.1 Adaptive Tree Traits` and `3.2.1 Trait Measurements`. |
| `longdocurl0507_highpage` | `longdocurl_summary2title_4032453_65_92_1` | GT is `BUTTERFLY MILKWEED`, but answer selects the broader container `Appendix 1 - Plant portraits...` and only mentions the page containing Butterfly milkweed. |
| `longdocurl0507_highpage` | `longdocurl_topic2title_4112595_8_10_1` | GT asks for two specific sections, but answer gives related broader/summary sections and omits one exact GT title. |

## Takeaway

The additional mining supports that `single_call_v1` fixes many genuine legacy false negatives, especially cases where legacy extraction returned a shortened answer but the raw final answer was complete.

However, the suspicious passes show that title/list questions remain the main risk area. The current fixture F2F set did not catch these exact highpage-style over-inclusive cases. If we want to harden the judge further, the next targeted improvement should be a stricter list/title verifier or additional F2F negative fixtures for over-inclusive title-list answers.
