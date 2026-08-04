# RL Prompt Length Note

Target script:
- [train_insight_qwen_agent_rl.t0_7.insight_doc_rl_balanced_dude_reduced.sh](/scratch/ywxzml3j/likaican/src/InSight-doc/verl/recipe/vsearch/train_insight_qwen_agent_rl.t0_7.insight_doc_rl_balanced_dude_reduced.sh)

## Assumptions

- Prompt text tokens were estimated with the local `Qwen3-VL-8B-Instruct` chat template tokenizer.
- Image tokens were estimated after:
  - `InSightQwenAgentLoop` initial presentation resize
  - `initial_rescale = 0.25`
  - `gpt_image_max_area = 3500 * 3500`
  - processor-side cap from [\_base.sh](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/recipe/vsearch/_base.sh):
    - `MAX_IMG_TOKENS_TRAIN = 4K`
    - `MAX_IMG_TOKENS_VAL = 4K`
    - `MAX_PIXELS = 4096 * 28 * 28 = 3,211,264`
- Image token estimate:
  - `ceil(width / 32) * ceil(height / 32)`

## Estimated Max Prompt Tokens

| Split | Dataset | Rows | Max est. prompt tokens | Text | Image | Max images |
|---|---:|---:|---:|---:|---:|---:|
| train | `insight_doc_rl` | 5446 | 13482 | 282 | 13200 | 40 |
| val | `arxiv_test` | 102 | 10311 | 231 | 10080 | 40 |
| val | `dude200` | 200 | 6612 | 190 | 6422 | 20 |
| val | `mmlongbench200` | 200 | 10331 | 251 | 10080 | 40 |
| val | `o3bench0502` | 345 | 3483 | 291 | 3192 | 1 |

Implication:
- the train parquet still contains rows above the script's `data.max_prompt_length=12288`

## 11k Threshold Filter

Input parquet:
- [/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent.parquet](/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent.parquet)

Filtered parquet:
- [insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.parquet](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.parquet)

Summary JSON:
- [insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.summary.json](/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.summary.json)

Filter result:
- total rows: `5446`
- rows removed for estimated prompt length `> 11000`: `32`
- rows kept: `5414`

Worst row:
- `row_idx = 1504`
- `question_id = docvqa_merged_twopass_38688_39907`
- `document_id = docvqa/qsnc0227.pdf`
- `n_images = 40`
- `text_tokens = 282`
- `image_tokens = 13200`
- `total_tokens = 13482`

The detailed top removed rows are stored in the summary JSON.
