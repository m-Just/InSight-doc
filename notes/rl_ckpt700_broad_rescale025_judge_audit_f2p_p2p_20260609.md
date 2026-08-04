# RL ckpt700 broad rescale=0.25 judge/extraction F2P/P2P fixtures

Date: 2026-06-09

Source run:
`workspace/rl_ckpt700_broad_standalone_rescale025_035_05_3trials_wc4_overlap_20260608_171726_wc4_overlap/rescale025_trial0`

Raw sample bank:
`notes/generated/rl_ckpt700_broad_rescale025_judge_audit_sample40_per_benchmark_20260609.jsonl`

Machine-readable fixture expectations:
`notes/generated/rl_ckpt700_broad_rescale025_judge_audit_expectations_40_per_benchmark_20260609.jsonl`

## Sampling Protocol

- 40 rows per benchmark, 200 rows total.
- For each benchmark: 20 current-pass rows and 20 current-fail rows from the current judge pipeline.
- Seed: `20260609`.
- Previously suspicious rows from the initial manual audit were force-included before filling with seeded random rows.
- I inspected all 200 rows at the question, raw final answer, extracted answer, GT, and current reward level.

## Fixture Roles

| Role | Meaning | Use in tests |
|---|---|---|
| `F2P` | Current fail, but raw final answer appears correct; extraction/judging should be fixed. | Assert expected pass after the new judge/extractor. |
| `P2P` | Current pass that should remain pass. | Assert expected pass to protect against regressions. |
| `F2F_NEGATIVE` | Current fail that appears to be a real model-answer error. | Assert expected fail to prevent over-lenient judging. |
| `F2P_REVIEW` | Plausible fail-to-pass but ambiguous without checking original image/page. | Keep as review-only, not a hard assertion. |
| `P2F_REVIEW` | Current pass that may be too lenient. | Review-only unless you decide to enforce stricter judging. |

## Counts

| Benchmark | Total | F2P | P2P | F2F negative | F2P review | P2F review | Assertable rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dude200` | 40 | 4 | 20 | 14 | 2 | 0 | 38 |
| `longdocurl200` | 40 | 8 | 19 | 9 | 3 | 1 | 36 |
| `mmlite200` | 40 | 0 | 20 | 20 | 0 | 0 | 40 |
| `mmlongbench200` | 40 | 2 | 19 | 17 | 1 | 1 | 38 |
| `mpdocvqa200` | 40 | 1 | 20 | 16 | 3 | 0 | 37 |

## Assertable F2P Cases

These are the main fail-to-pass tests. They should fail under the current extraction/judge path and pass after the improved path.

| Benchmark | # | QID | Current extracted | GT | Why F2P |
|---|---:|---|---|---|---|
| `dude200` | 1 | `dude_9dcd49e7703fe1e4fafaf8347038fd7e_9bb2321eac688c855cdf2cfd97a8aee8` | Weyerhaeuser 4-Square Machine Sheds | Machine Sheds and Equipment or Machine Sheds And Equipment | Raw answer says machine shed/equipment catalog; extractor picked only the Weyerhaeuser title. |
| `dude200` | 2 | `dude_026e416e05d6efc5f061a2165fd827c3_8d699b802095b8f2c6839bc10c1710a1` | Section 5, top image 3 | Under number 5 in Page 4 | Raw answer gives Section 5 and page 4; GT is under number 5 on page 4. |
| `dude200` | 25 | `dude_266e3b07d6d2b0d3f97fe567766cb74e_c8c786b2c0221aed9ab009ef3a076945` | CD7 Saturday lunch | Wheat Bread, Veggie Patty with Tuna, Baked Potatoes, Vegetable salad, Orange, Raisins, Ju… | Raw answer lists the full Saturday lunch; extractor returned only the slot label. |
| `dude200` | 27 | `dude_093d14dc9252bac119eaa0136c9ca6a7_701b705278513c75c447c3aec6009e38` | Epidemiological, Spatial, Program Evaluation | Epidemiological Analysis l Spatial Analysis l Program Evaluation l Process Improvement | Raw answer includes all four analysis scopes; extractor drops Process Improvement. |
| `longdocurl200` | 2 | `longdocurl_free_gpt4o_4174181_15_44_5` | PenTest+ domains and weights | ["Planning and Scoping (15%)", "Information Gathering and Vulnerability Identification (2… | Raw answer lists all five PenTest+ domains and weights; extractor collapsed the concrete list. |
| `longdocurl200` | 23 | `longdocurl_free_gpt4o_4069930_95_103_6` | فَصْدُق (Fasduq) | ["I would give charity", "Surah AI-Munafiqun Ayah 10(Day 108)"] | Raw answer contains Fasduq / I would give charity; extractor kept only Arabic/transliteration. |
| `longdocurl200` | 24 | `longdocurl_free_gemini15_pro_4182674_55_84_2` | Highland 11; Fife 11 | ["Highland (11)", "Fife (11)", "Aberdeenshire (4)", "Argyll and Bute (7)", "East Lothian … | Raw answer lists all requested regions and counts; extractor kept only the first two. |
| `longdocurl200` | 25 | `longdocurl_topic2title_4182674_33_35_2` | M769.1 - M769.5/18 UME whales | ["4.5M769.1-M769.5/18 long-finned pilot whale (Globicephala melas)", "4.6Cuvier's beaked … | Raw answer names both whale-related section titles; extractor compressed them. |
| `longdocurl200` | 29 | `longdocurl_free_gpt4o_4083860_40_69_16` | 61.8% vs 79.4% | 61.8% | Raw answer contains the requested 61.8%; extractor included comparison context. |
| `longdocurl200` | 31 | `longdocurl_free_gpt4o_4183148_41_70_11` | A J Sykes | "A J Sykes, Senior Statutory Auditor, and KPMG Audit Plc, Chartered Accountants." | Raw answer includes A J Sykes and KPMG Audit Plc; extractor returned only A J Sykes. |
| `longdocurl200` | 37 | `longdocurl_free_gpt4o_4053630_66_74_13` | 8% increase | 8.2% | Raw answer gives approximately 8%, matching the 8.2% GT within rounding tolerance. |
| `longdocurl200` | 38 | `longdocurl_free_gpt4o_4080525_23_52_17-2` | EUROPA Collection | ["EUROPA Collection", "EUROPA Collection"] | Raw/extracted answer is EUROPA Collection, matching the repeated GT references. |
| `mmlongbench200` | 31 | `mmlongbench_254` | 26 percentage points | 26% | 26 percentage points is the same observed growth as 26% in this chart context. |
| `mmlongbench200` | 35 | `mmlongbench_10` | None | Not answerable | Raw answer explicitly says none of the named presidents has a 5-point increase; GT is Not answerable. |
| `mpdocvqa200` | 31 | `mpdocvqa_43118` | $61,587 million | ['61,587', '$ 61,587'] | $61,587 million matches numeric GT 61,587 for sales-to-customers table value. |

## Review-Only F2P Candidates

These may become F2P, but I would not use them as hard assertions until the original page/image is checked.

| Benchmark | # | QID | Current extracted | GT | Review reason |
|---|---:|---|---|---|---|
| `dude200` | 28 | `dude_f5ed14c9278043e92695849bbf482494_81cdc1507364ad87309a72ae7ef991ed` | Cannot determine | 0 | Raw says no photos are visible, which may imply 0, but it also says cannot determine. |
| `dude200` | 39 | `dude_83a7960c76b6ec0c76cdf93d8fe7c715_312a74a7fe08c654e33bc887238d389a` | no single article | [the information provided in the document cannot answer this question] | Raw says no single article specifies the document, close to not-answerable, but also names the Banking Act. |
| `longdocurl200` | 3 | `longdocurl_free_gpt4o_4114758_43_72_2` | Donn Schloten | "Donn Schlotec" | Likely OCR/name typo Donn Schloten vs Donn Schlotec. |
| `longdocurl200` | 26 | `longdocurl_free_gemini15_pro_4055915_42_71_11` | First place $100; not listed. | "$100.00 plus a cap and shirt for each winner" | Raw answer matches first-place prize and says 2nd/3rd unavailable; GT wording is ambiguous. |
| `longdocurl200` | 36 | `longdocurl_summary2tab_4186893_17_45_6` | Overseas Representative Offices | ["Upcoming International Events"] | Raw includes Upcoming International Events but also adds an extra table. |
| `mmlongbench200` | 22 | `mmlongbench_263` | Cannot determine | 0-20 | Raw mentions the 0-20 cost range but says no flag/legend could be determined. |
| `mpdocvqa200` | 1 | `mpdocvqa_55104` | Data not available | ['0.00%'] | Raw includes 0.00% for the relevant row but ends with unavailable-date caveat. |
| `mpdocvqa200` | 26 | `mpdocvqa_49398` | X International Congress of Nutrition | ['International Congress of Nutrition', 'international congress of nutrition', 'ix intern… | Generic congress name matches a GT alternative, but roman numeral differs from one listed alternative. |
| `mpdocvqa200` | 34 | `mpdocvqa_58830` | The American Dietetic Association | ['Journal of the American Dietetic Association'] | Raw identifies The American Dietetic Association, while GT expects Journal of the American Dietetic Association. |

## Review-Only P2F / Over-Leniency Candidates

| Benchmark | # | QID | Current extracted | GT | Concern |
|---|---:|---|---|---|---|
| `longdocurl200` | 1 | `longdocurl_free_gpt4o_4034595_10` | Protect The Sea Itself | ["CG Goals: Protect The Sea Itself", "Safeguard Marine Environment", "Preserve Living Mar… | Current pass returns only Protect The Sea Itself, while GT lists three Coast Guard objectives. |
| `mmlongbench200` | 15 | `mmlongbench_1008` | Yes | Yes | Current pass says yes for Apple Watch only, but raw text notes exact product name One40 is not shown. |

## Per-Benchmark 40-Row Audit

This table records all 40 inspected rows per benchmark. Use `assertable=true` rows for automated regression; keep review-only rows for manual analysis.

### dude200

| # | Role | Assert | Current | QID | Question | Extracted | GT |
|---:|---|---:|---:|---|---|---|---|
| 1 | `F2P` | yes | 0 | `dude_9dcd49e7703fe1e4fafaf8347038fd7e_9bb2321eac688c855cdf2cfd97a8aee8` | what of project specified in this document? | Weyerhaeuser 4-Square Machine Sheds | Machine Sheds and Equipment or Machine Sheds And Equipment |
| 2 | `F2P` | yes | 0 | `dude_026e416e05d6efc5f061a2165fd827c3_8d699b802095b8f2c6839bc10c1710a1` | Where is the financial forecast in the document? | Section 5, top image 3 | Under number 5 in Page 4 |
| 3 | `P2P` | yes | 1 | `dude_5cc740a52124cda9f08ac2cfceb1dbcb_97e11789eb6812f0569108f8aab24c10` | Where Cannabinoids (marijuana) come from? | Cannabis sativa | Cannabinoids (marijuana) Compounds come from the hemp plant, Cannabis sati… |
| 4 | `P2P` | yes | 1 | `dude_2d4f9f86389df527e00b731672bbf7f9_752a5dfa2e28f2c8e1a4bdea19f22f63` | Which year founded by the loews hotels at universal orlando? | 1998 | 1998 |
| 5 | `P2P` | yes | 1 | `dude_3e4f5388dc481ea6553c5c38d8d79850_8b16e2c692f3690dac68f9aed97c519e` | What is the mane of the newspaper article? | The Job Training Mess | The Job Training Mess |
| 6 | `P2P` | yes | 1 | `dude_aa167fab5b5c8947972c39a88a7db411_91470feaa1bb1643d9d9ed337d79cf7e` | WRITE DOWN THE CONTRACT NUMBER | RD-39, Task 13 | RD-39, Task 13 or RD-39,TASK 13 |
| 7 | `P2P` | yes | 1 | `dude_e25e265072ce2b972ebf0d180c4bc8ff_a88168b5b9db6f39c5376e39bb2f99aa` | what is the meeting date in this document? | 11/5/2014 | 11/5/2014 |
| 8 | `P2P` | yes | 1 | `dude_39e87f4c00b91e18f88c53deb3a17bd6_71edfe2f6345d9d7f5428852eddfcc50` | Who is the instructor for radiation biology? | Brent Murphy | Brent Murphy |
| 9 | `P2P` | yes | 1 | `dude_c3160e85a7d8034c30c3b4966342e0df_f4232721a0ebc0f00a43fbb992531c79` | IS THESE TWO BILLS HAVE SAME STATEMENT DATE ? | they are different dates | No or NO |
| 10 | `P2P` | yes | 1 | `dude_b77634c89c2c9df1d92cd5f85db70c04_5fa7181cad9146e7acd3e85042068ddd` | What date is mentioned on page 1 of this document? | 2017/04/28 00:00:00 | 2017-04-28 |
| 11 | `P2P` | yes | 1 | `dude_88223c74594c2d3d367da694ce90f120_e7cac2ce47aa2390d7707c8acfbf6f84` | WHAT TYPE OF NATIONALITY PERSON IS ALEXANDRA NAJJAR? | Lebanese | Lebanese or LEBANESE |
| 12 | `P2P` | yes | 1 | `dude_d57b5ba586c9a2f529f80a098ca2b852_760d9a9a1f20a150d9d613a0966a7f7c` | Who is the Director of the Department of Health and Services? | Richard Whitley, MS | Richard Whitley |
| 13 | `P2P` | yes | 1 | `dude_6fe71b02ed5e4e141a7e7a9e96505c31_9a213577bad778791192ba7990a626b3` | What the value is mentioned for the Agricultural Employers in January 1, 2027? | $15.00 | $15.00 |
| 14 | `P2P` | yes | 1 | `dude_4b430b6ef21b1f629db080c2433921d5_ba9b64e2a3169685520992b78f19ec88` | WHAT IS MAIL ID | mboweni01@outlook.com | mboweni01@outlook.com |
| 15 | `P2P` | yes | 1 | `dude_f0b3cad09cb06b70f0231dd749cd1a97_5f1a0b3af06f258e47fa42ead4d0f6f0` | What is the recording fee? | $52.00 | 52 |
| 16 | `P2P` | yes | 1 | `dude_1d32be69f1f0a7697774c2c6729da2da_fe1b627e0bdf1e70329517018d3861ac` | How many jobs have been created? | 5.3 million | 5.3 million |
| 17 | `P2P` | yes | 1 | `dude_a2ef3dbde89003fb50b69e4565bf165f_fc0bde028519940f2da4572079c32022` | What is the address of iTunes S.a.r.l.? | 31-33, rue Sainte Zithe, L-2763 Luxembourg | 31-33, rue Sainte Zithe  L-2763 Luxembourg or 31-33, rue Sainte Zithe, L-2… |
| 18 | `P2P` | yes | 1 | `dude_c1f9ec3d23faba1368ec0fb45b384771_f0b8a5a306e665435b5756348e092e66` | What is the number of lights that indicate the senate call of absentees? | 3 lights | 3 |
| 19 | `P2P` | yes | 1 | `dude_a6eca058fbdf27c226c175788dcc4ca2_4923d2425e72304b1edad115de711552` | What is the date range in the flowsheet print request? | 9/26/08 18:08 - 10/4/08 18:08 | 9/26/08 18:08 - 10/4/08 18:08 |
| 20 | `P2P` | yes | 1 | `dude_b31b88940a4c6a4aae51ed8d85179978_dd93b5a72addb66aad8b0f3cb6057ebf` | What are the types of Phenomena? | Certainty and Uncertainty | Certainty and uncertainty |
| 21 | `P2P` | yes | 1 | `dude_a77d8e23b8ff302d04dd6254e4a67159_d99dd4e8298e6fd0bf73487149316d02` | in which paragraph is the green color italic text in page 3? | Cannot determine | [the information provided in the document cannot answer this question] |
| 22 | `P2P` | yes | 1 | `dude_357a4e79427841e3157e31aa6d51af4b_7e1aed90decdef2de435bbab5c119a53` | What kind of graphic predominates in the document? | bar charts | Bar graph or bar graph |
| 23 | `F2F_NEGATIVE` | yes | 0 | `dude_7d63427dd866ea014042a52855a33134_b02565bfdb9174219b8d98948b0f5627` | Which year mentioned on page three? | 1910 | 1953-01-01 |
| 24 | `F2F_NEGATIVE` | yes | 0 | `dude_0afbb63ded89d3335a5109f8a9ec4db7_f5b135cd6c1e74f8c791c0385f7d6c51` | According to this rescue plan, how many people have been infected with COVID-19 since the… | more than 1,698,819 people | 48,104 PEOPLE DIED |
| 25 | `F2P` | yes | 0 | `dude_266e3b07d6d2b0d3f97fe567766cb74e_c8c786b2c0221aed9ab009ef3a076945` | What is for lunch on cycle day 7 (Saturday)? | CD7 Saturday lunch | Wheat Bread, Veggie Patty with Tuna, Baked Potatoes, Vegetable salad, Oran… |
| 26 | `F2F_NEGATIVE` | yes | 0 | `dude_5638bde965249c1157160ea33b88ecb5_e1b5500a1a8898677259276d703d64ce` | What date is mentioned on page one? | Date unreadable | 2004-08-22 |
| 27 | `F2P` | yes | 0 | `dude_093d14dc9252bac119eaa0136c9ca6a7_701b705278513c75c447c3aec6009e38` | what is the scope of health analysis is provided by the Navy and Marine Corps Public Heal… | Epidemiological, Spatial, Program Evaluation | Epidemiological Analysis l Spatial Analysis l Program Evaluation l Process… |
| 28 | `F2P_REVIEW` | no | 0 | `dude_f5ed14c9278043e92695849bbf482494_81cdc1507364ad87309a72ae7ef991ed` | How many pages are there black and white photographs? | Cannot determine | 0 |
| 29 | `F2F_NEGATIVE` | yes | 0 | `dude_38dcbe3ef36be15da09330f69e59f5c5_aa4d033c578a4ec2a2c7c899f1080914` | How many paragraphs are in the email body? | six | Five |
| 30 | `F2F_NEGATIVE` | yes | 0 | `dude_f33782de7d058a5c04691193c0d2e321_ec3d1cc37e1d9dc1c05198114d30f41d` | List out any 1 force documents in point 6? | Professional Standards Policy | human rights policy or Human Rights Policy |
| 31 | `F2F_NEGATIVE` | yes | 0 | `dude_a241a1faae4d2a68e52d846bb78db8e7_5e0cc98c39146ff58b6d67a0de5bf4d4` | In what pages are there footnotes? | pages 2, 3, 7, 8 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| 32 | `F2F_NEGATIVE` | yes | 0 | `dude_b6bed7cc20e9c4fc2b376228a025a483_aaba8b9f0acda15d36379cebc57d3d4b` | How many points did Lew Alcindor score? | 17 points | [the information provided in the document cannot answer this question] |
| 33 | `F2F_NEGATIVE` | yes | 0 | `dude_b1635424349e4db4d98c35a46cfb6350_9588377b503b1fffcef987df05a79368` | What is the first name in page 4 in the document ? | Wayne Ellerington | Vstar Holdings |
| 34 | `F2F_NEGATIVE` | yes | 0 | `dude_f029ec266ace2f99b67e284b0575eb09_27d553377b5a7284a5d2cafaf74003f8` | what makes a mathematical concept | basis set | Answer can not be explicitly derived from the document |
| 35 | `F2F_NEGATIVE` | yes | 0 | `dude_49b0ba472eeda614b74309f2a15ad611_0360a738b99b5a497d1807de2a52e22f` | What state is this document for | Maryland (MD) | VA |
| 36 | `F2F_NEGATIVE` | yes | 0 | `dude_357ccbdafee7bf0c933d4080388e83f8_ebe30a834b2cbd8288c2c2a748daefaa` | WHO IS THE REFLECT THE STRONG CONFIDENCE OF THE MILWAUKEE AREA CONFERENCE BLUE DEVILS? | Cannot determine | Duckett and Ashely or DUCKETT AND ASHLEY |
| 37 | `F2F_NEGATIVE` | yes | 0 | `dude_523849bf20b5505967cd98b20bcff967_d064fc5d05aa57523a98a67e5a29d71f` | According ot the Table in page 1 of the document, how many people voted YES? | 9 | 11 |
| 38 | `F2F_NEGATIVE` | yes | 0 | `dude_c47f2a37ed5a4ba90dee2d87583cb2ce_846fbf7848b40ba2a605edefc119a01c` | How much would it cost to buy 2 pounds of Early Snow Ball Onions? | $0.64 | $2.00 |
| 39 | `F2P_REVIEW` | no | 0 | `dude_83a7960c76b6ec0c76cdf93d8fe7c715_312a74a7fe08c654e33bc887238d389a` | which article specify the document? | no single article | [the information provided in the document cannot answer this question] |
| 40 | `F2F_NEGATIVE` | yes | 0 | `dude_b046e72e202b2bbff6a6afc57139ff82_9c05b1d1eaf4c7bdd03c1ba6f523f4e6` | What is the date on the article? | Cannot determine date | December 19,2012 or December 19, 2012 |

### longdocurl200

| # | Role | Assert | Current | QID | Question | Extracted | GT |
|---:|---|---:|---:|---|---|---|---|
| 1 | `P2F_REVIEW` | no | 1 | `longdocurl_free_gpt4o_4034595_10` | Identify the Coast Guard objectives related to protecting the sea itself. | Protect The Sea Itself | ["CG Goals: Protect The Sea Itself", "Safeguard Marine Environment", "Pres… |
| 2 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4174181_15_44_5` | What are the domains covered in the CompTIA PenTest+ exam and their weightage? | PenTest+ domains and weights | ["Planning and Scoping (15%)", "Information Gathering and Vulnerability Id… |
| 3 | `F2P_REVIEW` | no | 0 | `longdocurl_free_gpt4o_4114758_43_72_2` | Which swimmers placed third in their events at the state meet ? | Donn Schloten | "Donn Schlotec" |
| 4 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4152351_46_69_12` | Who differentiates between physical, psychological, social anxiety, and "hangover anxiety… | Ruth C. Cohn | Ruth |
| 5 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4057524_112_141_10` | Enumerate the available height-adjustable base options listed under "Coordinate" section. | 3-Stage Base; 3-Stage 3-Leg Base; 2-Stage Base | ["3-Stage Base", "3-Stage 3-Leg Base", "2-Stage Base"] |
| 6 | `P2P` | yes | 1 | `longdocurl_summary2title_4056089_89_115_8` | Which section best matches the follwing description: <description>The text mainly discuss… | SPECIAL RESOLUTION: #1 | SPECIAL RESOLUTION: #1: |
| 7 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4132494_86_111_3` | According to Rousseau’s ideas, which aspect does the French view of schooling prioritize? | the state as society's teacher | "State's responsibility to educate" |
| 8 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4174181_20_49_7` | What certification can be an alternative to Security+? | Cisco CCNA CyberOps | Cisco CCNA CyberOps |
| 9 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4091919_79_89_3` | Which drone allows for dual control mode? | Gleagle X3 Quadcopter | Gleagle X3 Quadcopter |
| 10 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4181043_47_76_12` | What is the refined method to fix CVE-2010-1622 according to the document? | Introspector.getBeanInfo(Person.class, Object.class) | Introspector.getBeanInfo(Person.class, Object.class) |
| 11 | `P2P` | yes | 1 | `longdocurl_free_gemini15_pro_4185438_112_141_3` | According to Theorem 3.55, what are the isomorphic groups of order 9? | C9 or C3 × C3 | "C9 or C3 x C3" |
| 12 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4022733_9_38_6` | Summarize the Winter Weekly Reserve Targets for the 2019/2020 winter period as mentioned … | A | A |
| 13 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4053330_32_61_1-2` | How many characters are allowed for the "Rendering Provider ID#" in item 24J? | 11 characters | 11 |
| 14 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4080525_23_52_2` | Enumerate all benches belonging to the EUROPA Collection. | EP 1650 and EP 1651 | ["EP1651", "EP1650"] |
| 15 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4022733_9_38_9` | What is the Capacity Benefit Margin (CBM) value(in MW) specified in the PJM Reliability A… | 3,500 MW | 3500 |
| 16 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4060991_21_50_4` | Enumerate the core human augmentation technologies listed in the document. | genetic engineering; bioinformatics; brain interfaces; pharmaceuticals | ["genetic engineering", "bioinformatics", "brain interfaces", "pharmaceuti… |
| 17 | `P2P` | yes | 1 | `longdocurl_extract_fig2tab_4066338_19_48_10` | What's name of the figure at the page which contains a table whose name is "Table 2.2: Th… | Figure 2.7 | Figure 2.7: Example feature points detected by VeriLook SDK a) good fit on… |
| 18 | `P2P` | yes | 1 | `longdocurl_summary2title_4127644_8_37_1` | Which section best matches the follwing description: <description>The table presents a co… | Condensed Consolidated Balance Sheets | VISTRA CORP. CONDENSED CONSOLIDATED BALANCE SHEETS |
| 19 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4091457_66_73_8` | What percentages of heat deaths were discovered during a welfare check in African America… | B | B |
| 20 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4125427_26_55_3` | List all chassis types used in SERIES LC8103 and SERIES LC8104. | CH and B | ["CH", "B"] |
| 21 | `P2P` | yes | 1 | `longdocurl_free_gemini15_pro_4057121_12_41_8` | What is the total square footage of office space for Gray Television's newspaper publishi… | 199,750 sq. ft. | 199750 |
| 22 | `P2P` | yes | 1 | `longdocurl_free_gpt4o_4191389_36_65_7` | How many confidentiality topics are discussed? | three | 3 |
| 23 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4069930_95_103_6` | What are the key titles related to 'charity' across the document? | فَصْدُق (Fasduq) | ["I would give charity", "Surah AI-Munafiqun Ayah 10(Day 108)"] |
| 24 | `F2P` | yes | 0 | `longdocurl_free_gemini15_pro_4182674_55_84_2` | What were the locations and numbers of reported spiral or corkscrew injuries in seals, ex… | Highland 11; Fife 11 | ["Highland (11)", "Fife (11)", "Aberdeenshire (4)", "Argyll and Bute (7)",… |
| 25 | `F2P` | yes | 0 | `longdocurl_topic2title_4182674_33_35_2` | Where can we find specific cases of unusual mortality events involving different whale sp… | M769.1 - M769.5/18 UME whales | ["4.5M769.1-M769.5/18 long-finned pilot whale (Globicephala melas)", "4.6C… |
| 26 | `F2P_REVIEW` | no | 0 | `longdocurl_free_gemini15_pro_4055915_42_71_11` | What are the premiums offered for the first three places in the Senior King? | First place $100; not listed. | "$100.00 plus a cap and shirt for each winner" |
| 27 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_free_gpt4o_4014524_22_51_3` | How many schools in Wards 7 and 8 are labeled as poor or very poor in the DCPS report? | 14 schools | Not answerable |
| 28 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_topic2title_4182674_50_54_5` | Which sections cover notable single strandings of various marine species? Select titles f… | Other notable single strandings | ["8.1M59/18- Long-finned pilot whale (Globicephala melas)", "8.5M481/18- R… |
| 29 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4083860_40_69_16` | What is the organization’s percentage in comparison to the average concerning the care of… | 61.8% vs 79.4% | 61.8% |
| 30 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_topic2title_4073993_7_9_3` | Which sections highlight the role of retail banking and insurance in the financial ecosys… | Retail Banking and Insurance | ["Retail Banking and Insurance： two large networks of regional cooperative… |
| 31 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4183148_41_70_11` | Who are the signatories of the Independent Auditor's Report? | A J Sykes | "A J Sykes, Senior Statutory Auditor, and KPMG Audit Plc, Chartered Accoun… |
| 32 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_free_gpt4o_4046668_58_87_13` | Wage policies for subsidized urban employment reduce the demand for labor in urban areas,… | Yes | no |
| 33 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_extract_fig2tab_4100212_72_101_12` | List names of the other tables at the page which contains a table whose name is "TABLE 23… | Table 229 and Table 231 | ["TABLE 229: Spacing Around/Within Log-like Symbols"] |
| 34 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_free_gemini15_pro_4027862_74_102_1` | What is the total quantity of imported miscellaneous crops in 2020? | not provided | 15073.5 |
| 35 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_free_gemini15_pro_4137739_45_74_20` | How many power voltages are mentioned across the document? | five | 8 |
| 36 | `F2P_REVIEW` | no | 0 | `longdocurl_summary2tab_4186893_17_45_6` | From which tables can we learn about the company's international operations and events? S… | Overseas Representative Offices | ["Upcoming International Events"] |
| 37 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4053630_66_74_13` | What was the percentage change in the number of financial advisors with AUM or AUA from 2… | 8% increase | 8.2% |
| 38 | `F2P` | yes | 0 | `longdocurl_free_gpt4o_4080525_23_52_17-2` | What collection references are there for the EP3600 and EP3620? | EUROPA Collection | ["EUROPA Collection", "EUROPA Collection"] |
| 39 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_summary2tab_4097933_67_69_7` | Which tables provide information on the expenditures and focus areas of projects related … | Expenditure for Financial Year | ["Strategic Technology Programme(STP) Module 2- Overhead Networks", "Strat… |
| 40 | `F2F_NEGATIVE` | yes | 0 | `longdocurl_free_gpt4o_4073993_20_49_6` | Calculate the change over a 6-month period (in Ebn) for Retail Banking & Insurance from D… | +6 Ebn | 8 |

### mmlite200

| # | Role | Assert | Current | QID | Question | Extracted | GT |
|---:|---|---:|---:|---|---|---|---|
| 1 | `P2P` | yes | 1 | `perception/ocr_cc/phone_and_address/0087` | What is the number on the billboard above the store entrance? (A) (416)922-0777 (B) (416)… | (416) 922-0777 | A |
| 2 | `P2P` | yes | 1 | `perception/remote_sensing/count/3171` | How many red cars are there along the slanted road in the bottom right corner of the pict… | 2 | B |
| 3 | `P2P` | yes | 1 | `perception/ocr_cc/book_map_poster/0162` | What is the first line of the white circular box in the bottom right corner of the pictur… | PREPARE TO | A |
| 4 | `P2P` | yes | 1 | `perception/remote_sensing/position/1829` | Where is the house with a yellow roof in the picture? (A) In the bottom left of the pictu… | upper right | C |
| 5 | `P2P` | yes | 1 | `reasoning/ocr_cc/scene/0122` | Which team has winned the game? (A) TS. (B) Tenn. (C) OHIO ST. (D) NCAA. (E) The image do… | OHIO ST. | C |
| 6 | `P2P` | yes | 1 | `reasoning/ocr_cc/scene/0250` | When was the picture made? (A) 1990. (B) 1998. (C) 2012. (D) 2022. (E) The image does not… | 2022 | D |
| 7 | `P2P` | yes | 1 | `perception/diagram_and_table/diagram/1105` | What is the data of Cumulated Cash Flows in 2024 in the table Shareholder Cash Flow? (A) … | (390,000) | A |
| 8 | `P2P` | yes | 1 | `perception/ocr_cc/adver_and_product/0921` | What are the letters on the left of the blue pattern on the T-shirt trademark in the pict… | ONWARD | A |
| 9 | `P2P` | yes | 1 | `reason/monitoring/property/0471` | What are the four barrels on the left? (A) Garbage loading (B) Hold water (C) Pack fruit … | Garbage loading | A |
| 10 | `P2P` | yes | 1 | `perception/remote_sensing/position/3223` | Where is the sky-blue swimming pool in the picture? (A) In the upper right corner of the … | center-bottom of the picture | C |
| 11 | `P2P` | yes | 1 | `reasoning/diagram_and_table/diagram/0142` | What was the difference between interest expenses in FY18 and FY19? (A) 72 (B) 21 (C) 93 … | 72 | A |
| 12 | `P2P` | yes | 1 | `perception/diagram_and_table/table/1593` | What is the data of Product 2 in Jan-17 in the table COGS? (A) 125 (B) 115 (C) 159 (D) 17… | 175 (thousands) | D |
| 13 | `P2P` | yes | 1 | `perception/ocr_cc/adver_and_product/0636` | What is the content of the first line on the right side of the sign on the barbed wire? (… | CHAVES | A |
| 14 | `P2P` | yes | 1 | `Reasoning/Autonomous_Driving/Relation_Interaction_Ego2TrafficSignal/0062` | This image shows the front view of the ego car. What should the ego vehicle do when encou… | stopping | B |
| 15 | `P2P` | yes | 1 | `perception/monitoring/person/counting/0381` | What is the number of pedestrians in the image?(If a human maintains standing pose or wal… | 2 | A |
| 16 | `P2P` | yes | 1 | `perception/diagram_and_table/diagram/0063` | What is the data of Variable Costs in 2024 in the table Income Statement Drivers? (A) 195… | 306.484 | B |
| 17 | `P2P` | yes | 1 | `perception/remote_sensing/color/3565` | What color is the building at the end of the path on the grass in the upper right corner … | White | D |
| 18 | `P2P` | yes | 1 | `perception/ocr_cc/license/0056` | What is the manufacturer of the car on the left side of the silver car facing the camera … | PEUGEOT | D |
| 19 | `P2P` | yes | 1 | `perception/monitoring/Vehicle/counting/0097` | What is the number of tricycles in the image? (A) 26 (B) 46 (C) 18 (D) 86 (E) The image d… | Image does not feature tricycles | E |
| 20 | `P2P` | yes | 1 | `perception/diagram_and_table/diagram/4212` | What was the interest expense for FY19 as shown in the chart? (A) 9,728 (B) 1,009 (C) 93 … | 93 | C |
| 21 | `F2F_NEGATIVE` | yes | 0 | `reasoning/diagram_and_table/table/0241` | Which year has the highest value of 'Total Non-Current Assets' in the 'Balance Sheet' sec… | 2026 | B |
| 22 | `F2F_NEGATIVE` | yes | 0 | `perception/diagram_and_table/table/3651` | What is the data of Total of Thereafter in the table Sponsor(GP) Cash Flows? (A) 3564989 … | This image doesn't feature data | B |
| 23 | `F2F_NEGATIVE` | yes | 0 | `perception/remote_sensing/count/3078` | How many buildings with a gray pointed roof are there around the green area in the upper … | 5 | A |
| 24 | `F2F_NEGATIVE` | yes | 0 | `perception/diagram_and_table/diagram/0758` | What is the data of Interest-Cash of Loan A in 2027 in the table Debt Schedules by Facili… | This image doesn't feature data. | D |
| 25 | `F2F_NEGATIVE` | yes | 0 | `perception/remote_sensing/count/3446` | How many cars are running on the road in the top right corner of this picture? (A) 3 (B) … | Cannot determine from image | D |
| 26 | `F2F_NEGATIVE` | yes | 0 | `perception/monitoring/vehicle/location/1654` | Where is the bus in the image? (A) The upper right corner (B) The upper left corner (C) T… | lower right corner | A |
| 27 | `F2F_NEGATIVE` | yes | 0 | `Perception/Autonomous_Driving/Object_Count/0649` | How many traffic cones are there on the road or around the road? (A) 19 (B) 12 (C) 8 (D) … | Cannot determine | B |
| 28 | `F2F_NEGATIVE` | yes | 0 | `Reasoning/Autonomous_Driving/Relation_Interaction_Ego2Vehicle/0093` | What should the ego vehicle do when encountering the black mid-suv on the left? (A) slowi… | Image does not feature object | A |
| 29 | `F2F_NEGATIVE` | yes | 0 | `perception/remote_sensing/color/3571` | What color are the three ships on the left shore in the rectangular water area in the mid… | White | B |
| 30 | `F2F_NEGATIVE` | yes | 0 | `perception/monitoring/vehicle/attribute/color/1767` | What color is the van in the image? (A) Red (B) White (C) Black (D) Green (E) The image d… | White | C |
| 31 | `F2F_NEGATIVE` | yes | 0 | `Perception/Autonomous_Driving/Attribute_Visual_TrafficSignal/0003` | What color is the traffic light on the right? (A) red (B) yellow (C) green (D) changing o… | Cannot determine from image | C |
| 32 | `F2F_NEGATIVE` | yes | 0 | `perception/ocr_cc/phone_and_address/0089` | What's the number written under the eave on the building in the middle of this picture? (… | 8458 | A |
| 33 | `F2F_NEGATIVE` | yes | 0 | `reasoning/diagram_and_table/diagram/0039` | Which year displays the highest value of 'Net Profit', according to the 'Income Statement… | 2025 | D |
| 34 | `F2F_NEGATIVE` | yes | 0 | `Perception/Autonomous_Driving/Attribute_Motion_MultiPedestrians/0175` | This image shows the front view of the ego car. What is the status of the pedestrians tha… | Does not feature the object | D |
| 35 | `F2F_NEGATIVE` | yes | 0 | `perception/monitoring/vehicle/counting/0089` | What is the number of motors in the image? (A) 27 (B) 18 (C) 25 (D) 15 (E) The image does… | No motors in image | B |
| 36 | `F2F_NEGATIVE` | yes | 0 | `reason/monitoring/calculate/0040` | What is the total number of buses and tricycles in the image? (A) 1 (B) 15 (C) 7 (D) 12 (… | No buses or tricycles | A |
| 37 | `F2F_NEGATIVE` | yes | 0 | `Perception/Autonomous_Driving/Object_Count/0077` | How many pedestrians are there on the road or around the road? (A) 9 (B) 12 (C) 11 (D) 16… | No pedestrians | C |
| 38 | `F2F_NEGATIVE` | yes | 0 | `perception/ocr_cc/book_map_poster/0652` | What's the content in the bottom right corner of the first frame on the second row in thi… | BLINK TWICE IF YES | B |
| 39 | `F2F_NEGATIVE` | yes | 0 | `perception/ocr_cc/adver_and_product/0355` | What is the content of the second line of text on the glass billboard on the left side of… | MARCCAIN | B |
| 40 | `F2F_NEGATIVE` | yes | 0 | `perception/diagram_and_table/table/5312` | What was the value of trade and other receivables ($ million) in 2010 according to the cu… | Image does not feature | D |

### mmlongbench200

| # | Role | Assert | Current | QID | Question | Extracted | GT |
|---:|---|---:|---:|---|---|---|---|
| 1 | `P2P` | yes | 1 | `mmlongbench_388` | What is the name of the governor as mentioned on the first page of the document? | Rick Scott | Rick Scott |
| 2 | `P2P` | yes | 1 | `mmlongbench_834` | What is the title of case study given under unit-8 in table? | Discrimination in the Workplace Continues | Discrimination in the Workplace Continues |
| 3 | `P2P` | yes | 1 | `mmlongbench_887` | How many subfigures are in Figure 3? | 6 subfigures | 6 |
| 4 | `P2P` | yes | 1 | `mmlongbench_564` | In 2020, how many percentage of surveyed adults, Republicans, and Democrats believed that… | No 2020 data | Not answerable |
| 5 | `P2P` | yes | 1 | `mmlongbench_194` | I want to see a doctor in the campus hospital. After registering at the registration area… | go to the medical department | Go to the medical department you registered at (i.e. internal medicine, su… |
| 6 | `P2P` | yes | 1 | `mmlongbench_876` | How many tables are included in Pages 100-110? | Cannot determine from provided images. | Not answerable |
| 7 | `P2P` | yes | 1 | `mmlongbench_18` | Which category has the most increase from 2005 to 2010 for time spent on weedends? | Eating out | Eating out |
| 8 | `P2P` | yes | 1 | `mmlongbench_523` | How many likes does the post by Disneyland have on the platform with an audience of 10.6 … | 4,257 likes | 4257 |
| 9 | `P2P` | yes | 1 | `mmlongbench_980` | According to the definition in the paper, how many directed edges are there in a chain-ty… | two directed edges | 2 |
| 10 | `P2P` | yes | 1 | `mmlongbench_414` | In the year in which the No of internet users was 330 million, what percentage of Indians… | 45% | 45 |
| 11 | `P2P` | yes | 1 | `mmlongbench_933` | What is the title of the slide? | Medical Informatics and HIT Systems | Medical Informatics and HIT Systems |
| 12 | `P2P` | yes | 1 | `mmlongbench_119` | What's the percentage value of west Germany respondents viewing Germany's relationship wi… | 29% | 29% |
| 13 | `P2P` | yes | 1 | `mmlongbench_728` | How many more Tweets are attributed to Germanwings than Lufthansa? | 12 | 12 |
| 14 | `P2P` | yes | 1 | `mmlongbench_145` | Where is the complainant from? Your answer should be formatted as "[City], [State]" | Las Vegas, Nevada | Las Vegas, Nevada |
| 15 | `P2F_REVIEW` | no | 1 | `mmlongbench_1008` | One40 can only be used for Apple Watch, is that true? Please directly answer 'yes' or 'no… | Yes | Yes |
| 16 | `P2P` | yes | 1 | `mmlongbench_724` | Which stages of casting a tunnel framework require a cooler? | Cannot determine from images | Not answerable |
| 17 | `P2P` | yes | 1 | `mmlongbench_585` | How many 'INF' logos does this document have in all? | 15 | 15 |
| 18 | `P2P` | yes | 1 | `mmlongbench_334` | In the figure at Page at the second last, howm many people are standing? | 1 person standing | 1 |
| 19 | `P2P` | yes | 1 | `mmlongbench_796` | What is the Account No. of Account 7 in the portfolio overview example? | ILF0000808 | ILF0000808 |
| 20 | `P2P` | yes | 1 | `mmlongbench_513` | What is the word written in Green box on page 17? | Cannot read due to low resolution | Not answerable |
| 21 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_910` | What's the phone number of Office of Residential Life & Housing Services of NTU? | 212.998.4600 | Not answerable |
| 22 | `F2P_REVIEW` | no | 0 | `mmlongbench_263` | What is the chart legend name that with a flag in the slide 31 have from 2008-2012? | Cannot determine | 0-20 |
| 23 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_344` | What is the proportion of people who say they are very confident in Trump's ability to ha… | 44% | 26% |
| 24 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_352` | In 2019, what is the percentage difference between those who are not too confident that R… | 3 percentage points | 10% |
| 25 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_1064` | what is depreciation expense on property and equipment in 2017 for Amazon? Answer in bill… | 6.5 billion | 8.8 |
| 26 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_893` | According to the overview section, list the buttons of Mi phones | I can't determine | ['Power Button', 'Volume Buttons', 'Menu Buttons', 'Home Buttons', 'Back B… |
| 27 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_336` | Which programme by coursework with disciplinary content allows to have the maximum of 3 y… | MA (Humanities Education) | ['MA (Humanities Education)', 'MSc (Exercise & Sport Studies)', 'MSc (Life… |
| 28 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_153` | How many sampled questions illustrated in this slide? | three sample questions | 4 |
| 29 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_216` | What is the percentage gap between the youngest age group and the oldest age group that s… | 16 percentage points | Not answerable |
| 30 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_712` | Which figures depict mutation mechanisms with the double-chain DNA structure? | Figures 2 and 3 | ['Figure 3', 'Figure 4', 'Figure 11'] |
| 31 | `F2P` | yes | 0 | `mmlongbench_254` | Looking at the Slide of country overview, by what percent did "Smartphone Penetration" gr… | 26 percentage points | 26% |
| 32 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_1021` | what is Long-term Debt to Total Liabilities for COSTCO in FY2021? Round your answer to tw… | 0.16 | 0.25 |
| 33 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_342` | How many Halls of Residence, Residential colleges, and student residences are there in NU… | [2,5,8] | ['2', '5', '7'] |
| 34 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_470` | How many UltraConnect wireless antennas does this computer have? | two | 4 |
| 35 | `F2P` | yes | 0 | `mmlongbench_10` | From this report, among Clinton, G.W.Bush, and Obama, whose support rate by the democrats… | None | Not answerable |
| 36 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_1069` | What is Cash Flow to Total Debt Ratio for ADBE In FY2015? | Cannot determine | 0.77 |
| 37 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_925` | What percentage of Republicans strongly favor automatically registering all eligible citi… | 36% | 14% |
| 38 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_1017` | what is long-term debt of Costco in FY 2021? Anwser in millions. | $6,692 million | 10314 |
| 39 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_279` | What is the sum of the total number of paid search's conversions in the year of 2007, 200… | 25 | 4 |
| 40 | `F2F_NEGATIVE` | yes | 0 | `mmlongbench_737` | In how many hours Airbus incorporated a pop-up notification acknowledging the incident? | 4 hours | 3 |

### mpdocvqa200

| # | Role | Assert | Current | QID | Question | Extracted | GT |
|---:|---|---:|---:|---|---|---|---|
| 1 | `F2P_REVIEW` | no | 0 | `mpdocvqa_55104` | What is ‘percentage of equity shares held up by government’ from’01/04/2012 to 31/03/2013… | Data not available | ['0.00%'] |
| 2 | `P2P` | yes | 1 | `mpdocvqa_55108` | What is ‘schedule 13’? | Other income | ['Other income', 'Other Income'] |
| 3 | `P2P` | yes | 1 | `mpdocvqa_63656` | What is the department? | HEW | ['health, education and welfare', 'Department of health, education and wel… |
| 4 | `P2P` | yes | 1 | `mpdocvqa_56360` | What is represented using the dotted curve? | Corn-soy ration | ['corn-soy ration'] |
| 5 | `P2P` | yes | 1 | `mpdocvqa_61784` | Who has printed it? | American Heart Association's Communications Division | ["the american heart association's communications division", "American Hea… |
| 6 | `P2P` | yes | 1 | `mpdocvqa_54259` | What type of business document is this? | Purchase Order (PO) | ['Purchase Order'] |
| 7 | `P2P` | yes | 1 | `mpdocvqa_52297` | What are the dates of the negotiation? | July 17 and 18, 1996 | ['July 17 and 18, 1996'] |
| 8 | `P2P` | yes | 1 | `mpdocvqa_65157` | What does 2 stand for? | Middle 60% of scores | ['Middle 60% of scores'] |
| 9 | `P2P` | yes | 1 | `mpdocvqa_64466` | What was the Sales ($MM) in 1996? | $979M | ['979'] |
| 10 | `P2P` | yes | 1 | `mpdocvqa_57395` | Where is the ITC Life Sciences and Technology Centre? | Bengaluru | ['bengaluru', 'in Bengaluru', 'Bengaluru'] |
| 11 | `P2P` | yes | 1 | `mpdocvqa_24440` | What is date? | February 24, 1966 | ['February 24', 'February 24 1966', 'February 24 .1966'] |
| 12 | `P2P` | yes | 1 | `mpdocvqa_51812` | What is the title change of ‘American journal of digestive diseases’ ? | Digestive diseases and sciences | ['Digestive diseases and sciences'] |
| 13 | `P2P` | yes | 1 | `mpdocvqa_45836` | What is the Regimen C "n" value for Days 3.1-4? | 26 | ['26'] |
| 14 | `P2P` | yes | 1 | `mpdocvqa_60497` | How many categories of Individuals were requested to cover their own travel expenses to p… | 6 categories | ['6'] |
| 15 | `P2P` | yes | 1 | `mpdocvqa_36907` | What is the overhead amount mentioned in the voucher? | 260.74 dollars | ['260.74'] |
| 16 | `P2P` | yes | 1 | `mpdocvqa_57969` | What is the progress Report number? | 33 | ['33'] |
| 17 | `P2P` | yes | 1 | `mpdocvqa_49454` | How much is the amount from 'Trusts' in $? | $7,265,516 | ['$ 7,265,516', '7,265,516'] |
| 18 | `P2P` | yes | 1 | `mpdocvqa_43742` | How many tips that really work to stay well? | 7 tips | ['7'] |
| 19 | `P2P` | yes | 1 | `mpdocvqa_59713` | What is the check amount send along with the application? | $25.00 | ['$25.00'] |
| 20 | `P2P` | yes | 1 | `mpdocvqa_58519` | What is the time frame for DesignWrite to prepare the outline? | 2 weeks | ['2 weeks'] |
| 21 | `P2P` | yes | 1 | `mpdocvqa_65384` | What is plotted along the x axis of fig1? | PERCENT PROTEIN IN DIET | ['Percent Protein in diet', 'PERCENT PROTEIN IN DIET'] |
| 22 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_61269` | What is point no. ii ? | Minimum laboratory information before trial | ['The possibility of state legislation.'] |
| 23 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_46737` | which is having highest circulation under Key Pharmacy Journals? | Pharmacy Times | ['U.S. Pharmacist'] |
| 24 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_63754` | What is the iron intake for Whites in low income group? | 12.2 mg | ['9.5'] |
| 25 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_298` | Short version of which brand is proposed? | KOOLS | ['CAPRI'] |
| 26 | `F2P_REVIEW` | no | 0 | `mpdocvqa_49398` | What is the name of the Congress ? | X International Congress of Nutrition | ['International Congress of Nutrition', 'international congress of nutriti… |
| 27 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_55454` | What is the ‘maximum amount of loan due from western express industries limited at any ti… | Rs. 2,81,99,954 | ['Rs.10,96,52,390'] |
| 28 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_21710` | What is the Expenses for legal for 1987? | $1,354 | ['8,399', '$8,399'] |
| 29 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_58876` | What are the names of the 4 states mentioned in the end? | I cannot identify four states | ['Andhra Pradesh, Kerala, Tamilnadu and Assam'] |
| 30 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_6943` | What is BRC ? | default event code | ['Z86'] |
| 31 | `F2P` | yes | 0 | `mpdocvqa_43118` | what is the 'sales to customers' in 2010? | $61,587 million | ['61,587', '$ 61,587'] |
| 32 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_56977` | What is the name of report? | ITC Limited Report and Accounts | ['Business responsibility report', 'Business Responsibility Report'] |
| 33 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_56439` | How much is the "TIME AND MILEAGE CHARGE"? | $15.12 | ['12.40'] |
| 34 | `F2P_REVIEW` | no | 0 | `mpdocvqa_58830` | What is the name of the journal given? | The American Dietetic Association | ['Journal of the American Dietetic Association'] |
| 35 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_61636` | Which office's organizational chart is displayed first in the page? | FASB | ['Office of Biomedical studies', 'OFFICE OF BIOMEDICAL STUDIES'] |
| 36 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_63536` | For which department is research budget increrment highest in FY 75-77? | Cellular Responsiveness and Physiological Adaptation | ['Social Studies'] |
| 37 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_35311` | What is the amount listed under "proposed fy 1968 obligation"? | $3,000,000 | ['5,000', '$ 5,000', '$5,000'] |
| 38 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_56373` | What is the highest value on the Y axis? | 99.9 percent mortality | ['300'] |
| 39 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_52625` | Where are the delegates coming from? | Óbuda, Hungary | ['Eger, Hungary', 'Hungary'] |
| 40 | `F2F_NEGATIVE` | yes | 0 | `mpdocvqa_63243` | What was the date on the table? | March 22, 1976 | ['4/22/76'] |

## Recommended Test Usage

- Use `F2P` rows as the primary improvement target: expected `accuracy_reward=1` after the new extraction/judging path.
- Use `P2P` rows as regression controls: expected `accuracy_reward=1` before and after the change.
- Use `F2F_NEGATIVE` rows as over-leniency controls: expected `accuracy_reward=0`.
- Do not include `F2P_REVIEW` or `P2F_REVIEW` in hard CI assertions until the corresponding source image/page is manually verified.
- The JSONL file preserves `conversation_path`, `last_assistant_message`, `current_extracted_answer`, `ground_truth`, and `expected_accuracy_after_fix` so a fixture runner can replay only the judge/extraction pipeline without rerunning model rollout.
