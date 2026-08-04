# Full-SFT LR And Regularization Memo

Date: 2026-05-20

Context: the full-weight SFT run on `both_higher_dpi_unans02503505` improves MMLongBench unanswerable accuracy but loses enough answerable accuracy that the aggregate highpage result does not clearly improve over the older full-SFT baseline. LoRA benefits more from the new data, likely because the constrained update preserves more of the old answerable-question behavior.

## Current Read

This looks more like a data-mix / behavior-shift issue than a pure epoch-count issue. The added unanswerable data teaches useful abstention behavior, but full-SFT appears to absorb that behavior globally and over-applies it to answerable cases.

Lower LR is the cleaner first control than larger weight decay. Increasing weight decay is a blunt regularizer and may suppress useful visual/tool adaptation. Qwen3-VL's official finetuning guidance uses much smaller LRs than our current `5e-6`; the README example uses `2e-7` and suggests a range around `1e-6` to `2e-7`, with cosine schedule, warmup, and `weight_decay=0.01`.

References:

- Qwen3-VL finetuning README: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md
- SFT data-composition effects: https://huggingface.co/papers/2310.05492
- Catastrophic forgetting in continual instruction tuning: https://arxiv.org/abs/2308.08747

## Recommended Next Experiment

Keep the same data and major training setup as the current full-SFT run, but reduce LR:

| knob | setting |
|---|---|
| model | `Qwen/Qwen3-VL-8B-Instruct` |
| tuning | full-weight SFT, vision tower frozen |
| data | current `both_higher_dpi_unans02503505` mix |
| context / SP / batch | `64k / sp=4 / global bs=32` |
| precision | same fp32 model precision path as the old stronger full-SFT baseline |
| LR | `1e-6` |
| min LR | `1e-7` |
| scheduler | cosine |
| warmup | `0.03` to `0.05`; current `0.05` is acceptable |
| weight decay | keep `0.01` |
| epochs | 1 epoch first |
| checkpoints | save/eval around `0.5x`, `0.75x`, and `1.0x` epoch |

If `1e-6` underfits, try `2e-6`. If `1e-6` still over-abstains on answerable MMLongBench, try `5e-7`.

## Decision Rule

The lower-LR run is promising if it preserves most of the unanswerable gain while recovering answerable MMLongBench and LongDoc accuracy. If answerable accuracy still drops at `1e-6`, the next fix should be data reweighting or replay, not larger weight decay.

Suggested data-side follow-up: train on the higher-DPI medium data with synthetic unanswerable shards removed or downweighted, then compare against the current full-SFT. This directly tests whether the synthetic unanswerable mix is causing the answerable degradation.
