---
title: 2B Rollout FN-Factor Results After Hard-16 Prefix Repair
date: 2026-03-17
status: complete
topics: [stage2, 2b, fn-analysis, sampling, prefix, sequence-length, train-vs-val]
tags: [2b, diagnostics, rollout, fn, prefix, sampling, hard-subset]
summary: Updated 2B FN-factor read after the repaired Hard-16 prefix rerun completed. Deterministic A-only gains and sampling recoverability remain, extended length is still null, and the repaired prefix intervention now yields valid continuation-only recovery rather than universal parse collapse.
---

# 2B Rollout FN-Factor Results After Hard-16 Prefix Repair (2026-03-17)

This note records the current best read on the fixed 2B FN-factor study for:

- original:
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
- A-only:
  `output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

with:

- train:
  `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- val:
  `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

and A-only config provenance:

- `output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml`

This note supersedes the earlier same-day interpretation that treated prefix as a global null result. A repaired Hard-16 prefix rerun completed later and rewrote the Hard-16 artifact.

Primary artifacts:

- refreshed Hard-16 full-factor report:
  `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/report.md`
- Hard-32 extension report:
  `output/analysis/rollout-fn-factor-2b-hard32-extension-20260317/report.md`
- Hard-16 recovery summaries:
  `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/recovery/*.summary.json`

## Bottom Line

Five conclusions are now safe.

1. `A-only` is still genuinely better than original in deterministic greedy rollout on both train and val hard subsets.
2. Many remaining FN are still decode-selection misses rather than proven incapacity, because same-prompt union-of-`K` recovers a substantial additional block of GT objects.
3. The repaired Hard-16 prefix intervention is now valid and yields nonzero continuation-only recovery.
4. The repaired prefix result does **not** show a strong winner between train-order and random-order prefixing.
5. Extended-length rollout is still a null result: `length_bias_miss = 0` everywhere.

The main nuance remains split-dependent:

- on train, `A-only` is better both greedily and after sampling
- on val, `A-only` is better greedily, but original still retains slightly more sampling-recoverable latent coverage on the hardest held-out scenes

## What Changed Relative to the Earlier Read

The original completed Hard-16 artifact had a broken prefix-control path and therefore reported:

- `24 / 24` prefix-order cells invalid
- `prefix_sensitive_miss = 0`

After the prefix-control repair and rerun, the refreshed Hard-16 report now shows:

- all `24 / 24` clean prefix-order cells are `health_valid=True`
- all switched / broken stress cells still fail with `parse_invalid`
- nonzero `prefix_sensitive_miss` on every checkpoint × split table

So the repaired result is:

- prefix is no longer a dead intervention
- but stress/broken-prefix probes still collapse

## Stable Findings That Did Not Change

### Deterministic greedy rollout still improves with A-only

Hard-16 deterministic hits:

- train:
  A-only `252 / 838` vs original `225 / 838`
- val:
  A-only `208 / 651` vs original `185 / 651`

Hard-32 deterministic hits:

- train:
  A-only `563 / 1667` vs original `514 / 1667`
- val:
  A-only `449 / 1210` vs original `402 / 1210`

This is still the cleanest stable result in the study.

### Sampling still matters a lot

Hard-16 recoverable coverage under image-only controls
defined as `deterministic_hit + decode_selection_miss + length_bias_miss`:

- train:
  A-only `357` vs original `333`
- val:
  A-only `305` vs original `323`

Hard-32 recoverable coverage:

- train:
  A-only `800` vs original `761`
- val:
  A-only `618` vs original `625`

Interpretation did not change:

- train hard scenes:
  `A-only` wins both greedily and after same-prompt sampling
- val hard scenes:
  original still keeps a slightly larger pool of sampling-recoverable misses

### Longer rollout length still does not explain the FN

Across the refreshed Hard-16 report and the unchanged Hard-32 extension:

- `length_bias_miss = 0` for every checkpoint × split table

So the simple “the model knows the object but needs a longer default decode budget” explanation is still unsupported here.

## New Prefix Read From the Repaired Hard-16 Rerun

The repaired Hard-16 recovery summaries now report:

- train / A-only:
  `prefix_sensitive_miss = 30`
- train / original:
  `34`
- val / A-only:
  `15`
- val / original:
  `15`

So the repaired prefix intervention is recovering a real, nontrivial block of GT objects that image-only deterministic rollout and image-only union-of-`K` sampling did not recover first.

That means the strongest updated claim is:

- some FN are blocked by continuation state, not only by raw object incapacity or decode randomness

## Does Prefix Order Matter?

Yes, but not in the clean “training order wins” way we originally hypothesized.

Across the `94` true `prefix_sensitive_miss` rows in the refreshed Hard-16 recovery tables:

- train-order recovered `60`
- random-order recovered `60`
- self-prefix recovered `40`

Recovery overlap is mixed:

- recovered by both train-order and random-order:
  `39`
- recovered only by self-prefix:
  `26`
- recovered by all three:
  `13`
- recovered only by train-order:
  `8`
- recovered only by random-order:
  `7`

Interpretation:

- there is no strong train-order-over-random-order advantage in the repaired Hard-16 rerun
- order-sensitive continuation effects do exist, but they are not dominated by the training serialization prior alone
- self-prefix recovers a smaller total set than train-order or random-order, but it still has a meaningful unique recovery block

So the updated prefix conclusion is more nuanced:

- prefix state matters
- but “sorted training order is the main blocking prior” is not supported strongly by this repaired result

## Train vs Val Read

Your train-vs-val concern still looks important and is still partially encouraging.

On train hard scenes:

- `A-only` has higher deterministic hit count
- `A-only` has higher union-of-`K` recoverable coverage
- `A-only` has fewer persistent unrecovered GT objects

But train is not solved. Even after repaired prefix recovery, Hard-16 still leaves:

- train / A-only persistent unrecovered:
  `451 / 838`
- train / original persistent unrecovered:
  `471 / 838`

On val, the split asymmetry remains:

- `A-only` is more deterministic
- original still preserves slightly more sampling-recoverable latent alternatives
- repaired prefix recovery is equal across checkpoints on val (`15` each)

So the present 2B read is:

- `A-only` improves stable greedy behavior
- original retains a slightly broader sampled alternative set on the hardest held-out scenes
- repaired prefix continuation partially closes the gap, but does not eliminate the large persistent-unrecovered block

## Caveats

### The refreshed Hard-16 report has duplicated Hard-32 bookkeeping rows

The repaired rerun was launched with:

- `hard16_size = 16`
- `hard32_size = 16`

so the refreshed Hard-16 `report.md` contains `Hard-32` rows that are bookkeeping duplicates of the 16-image cohort, not a true independent 32-image extension.

Use the separate Hard-32 extension run for real `32`-image numbers:

- `output/analysis/rollout-fn-factor-2b-hard32-extension-20260317/report.md`

### Stress-prefix probes still fail

The cleaned prefix-order cells are now valid, but the intentionally broken / switched prefix stress probes still collapse with `parse_invalid`. So the repaired result should not be read as “all prefix intervention problems are solved.”

### Random-order Stage-1 smoke is still pending

The 2B random-order Stage-1 config surface exists and is aligned to the `1024` JSONLs, but the smoke launcher has not yet executed end-to-end. So this note does **not** yet provide training evidence for the random-order SFT ablation.

## Practical Read

The updated 2B FN-factor story is now:

- original Stage-1 2B is already strong
- `A-only` makes greedy rollout better and more stable
- same-prompt sampling still recovers many misses
- repaired prefix continuation now recovers additional misses and therefore matters
- but the remaining prefix signal does not strongly support “training order is the key cause”
- and longer sequence length still does not explain the residual FN

If you want the operator-oriented artifact summary next, read:

- `progress/diagnostics/2026-03-17_stage2_2b_fn_factor_artifact_guide.md`

If you want the repair and pending random-order follow-up state, read:

- `progress/diagnostics/2026-03-17_stage2_2b_prefix_random_order_followup.md`
