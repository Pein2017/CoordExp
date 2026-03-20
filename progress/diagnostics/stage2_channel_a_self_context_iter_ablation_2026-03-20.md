---
doc_id: progress.diagnostics.stage2-channel-a-self-context-iter-ablation-2026-03-20
layer: progress
doc_type: diagnostic-study
status: completed-diagnostic
domain: research-history
summary: Stage-level ablation record for Channel-A-only self-context iteration on the 1024-res COCO path, comparing iter_1, iter_2, iter_2 decode_st, iter_4, stop-signal damping, and an A1-only sorted-order control.
updated: 2026-03-20
---

# Stage-2 Channel-A Self-Context Iteration Ablation (2026-03-20)

Date: 2026-03-20
Last updated: 2026-03-20
Supersedes: `progress/diagnostics/stage2_channel_a_random_order_iter4_2026-03-19.md`
Primary question: does increasing Channel-A self-context iteration beyond `n_softctx_iter=1` improve rollout quality enough to justify the added complexity and compute, or do current artifacts support keeping only `A1` and dropping later passes for now?

This note is a stage-level synthesis, not a single-run diary. It combines the March 2026 random-order ablations with earlier Channel-A self-context diagnostics to answer the current decision question as carefully as the available artifacts allow.

## 1. Executive Summary

- Recommendation: keep `n_softctx_iter=1` for the current Channel-A-only direction and retire later self-context passes (`A2`, and by extension `A3`/`A4`) from active configs for now.
- Confidence: medium-high for the practical decision above; medium for the stronger causal claim that iteration count alone is the driver, because the `iter_2` and `iter_4` runs are not pure iter-only ablations.
- Direct evidence against `n_iter > 1` in the current regime:
  - At shared eval checkpoints, random-order `iter_2`, `iter_2 decode_st`, and `iter_4` all underperform the random-order `iter_1` baseline on rollout quality despite mostly valid parsing.
  - At `step 600`, random-order `iter_1` reaches `mAP=0.4466`, `F1=0.6486`, `matched=2437`, while random-order `iter_2` reaches `mAP=0.4016`, `F1=0.4765`, `matched=1910`, and random-order `iter_4` reaches `mAP=0.3216`, `F1=0.3325`, `matched=1313`.
  - In every logged `n_iter > 1` training row, the final self-context pass (`A2` telemetry in the current trainer) remains worse than the anchor pass (`A1`) on coord diagnostics:
    - random-order `iter_2`: `0 / 88` rows where `coord_diag/A2/acc_top5 > coord_diag/A1/acc_top5`, and `0 / 88` rows where `coord_diag/A2/expected_bin_mae < coord_diag/A1/expected_bin_mae`
    - random-order `iter_2 decode_st`: `0 / 37` and `0 / 37`
    - random-order `iter_4`: `0 / 74` and `0 / 74`
- The recurring failure mode is not primarily parse invalidity. It is parse-valid but non-atomic geometry:
  - large semantic-region boxes instead of one box per instance
  - repeated narrow book strips / duplicate-heavy shelves
  - crowd-scene collapse with weak matching despite `sample_valid_pred_rate = 1.0`
- Direct evidence covers `iter_2` and `iter_4`; the recommendation to drop `A3`/`A4` is an inference from that evidence plus the current training contract:
  - the current trainer supervises `A1` plus the final self-context pass only
  - `iter_4` already pays the full deeper-loop cost while giving worse rollout quality than `iter_2` and much worse quality than `iter_1`
  - there is no direct artifact here showing independently useful later-pass behavior beyond the already weak final-pass behavior

## 2. Artifact Inventory

| Run | Artifact root | Order regime | `n_softctx_iter` | Notable differences in stored config | Completion status | Eval/monitor coverage |
| --- | --- | --- | ---: | --- | --- | --- |
| Random-order `iter_1` baseline | `output/a_only/random_order/epoch-1-iter_1-eff_bs_128/v0-20260318-094911` | random | 1 | `struct_ce_weight=1.0`; `bbox={smoothl1 1.0, ciou 0.5}`; `coord_ce_weight=0.04`; stop-signal damping disabled | last train row `910/915` (`99.45%`) | eval/monitor at `300`, `600`, `900`; `class_summary.json` at `vis_step_000900`; checkpoint `README.md` at `600`, `900` |
| Random-order `iter_1` + stop-signal damping | `output/a_only/random_order-stop_signal_damping/epoch-1-iter_1-eff_bs_128/v0-20260318-044005` | random | 1 | baseline objective plus `stop_signal_damping.enabled=true` with `{min_weight 0.3, max_weight 0.75, branch_temperature 1.25, curve_gamma 2.0, detach_gate true}` | last train row `550/915` (`60.11%`) | eval/monitor at `300` only; no checkpoint README present |
| Random-order `iter_2` | `output/a_only/random_order-iter_2/epoch-1-eff_bs_128/v0-20260319-133905` | random | 2 | `struct_ce_weight=0.1`; `bbox={smoothl1 2.0, ciou 0.2}`; `coord_ce_weight=0.02`; same `coord_decode_mode=exp` | last train row `870/915` (`95.08%`) | eval/monitor at `300`, `600`; checkpoint `README.md` at `300`, `600` |
| Random-order `iter_2` + `decode_st` | `output/a_only/random_order-iter_2-decode_st/epoch-1-eff_bs_128/v0-20260320-024125` | random | 2 | same as random-order `iter_2` except `coord_decode_mode=st` | last train row `360/915` (`39.34%`) | eval/monitor at `300` only; `class_summary.json` at `vis_step_000300`; checkpoint `README.md` at `300` |
| Random-order `iter_4` | `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914` | random | 4 | same altered `iter > 1` objective bundle as random-order `iter_2`; `coord_decode_mode=exp` | last train row `730/915` (`79.78%`); incomplete | eval/monitor at `300`, `600`; `class_summary.json` at `vis_step_000300` and `vis_step_000600`; no checkpoint README present |
| Sorted-order `ciou_bbox_area` control | `output/a_only/ciou_bbox_area/4b-softce_w1-coco80/epoch_1-eff_size_128-n_softctx_iter_1-a_only/v1-20260316-105736` | sorted | 1 | different base model (`output/stage1/...`); resolved config shows `struct_ce_weight=0.1`, `coord_w1_weight=0.0`, `bbox_log_area_weight=0.0` | last train row `910/915` (`99.45%`) | eval/monitor at `300`, `600`, `900`, `915`; checkpoint `README.md` at `900`, `915` |

Checkpoint README note:
- Checkpoint `README.md` files are present for some runs, but they are generic autogenerated model cards and do not contain run-specific checkpoint metrics. They were useful only for confirming checkpoint presence.

## 3. Methodology And Comparison Caveats

- This is not a pure iter-count ablation.
  - Moving from the random-order `iter_1` baseline to random-order `iter_2` / `iter_4` changes `n_softctx_iter`, but also changes the objective bundle:
    - `token_ce.config.struct_ce_weight: 1.0 -> 0.1`
    - `bbox_geo.config.smoothl1_weight: 1.0 -> 2.0`
    - `bbox_geo.config.ciou_weight: 0.5 -> 0.2`
    - `coord_reg.config.coord_ce_weight: 0.04 -> 0.02`
- `iter_2 decode_st` is only a decode-mode ablation inside the altered `iter_2` objective bundle; it does not restore the original `iter_1` weights.
- The stop-signal damping run is a control for count/termination behavior inside the `iter_1` regime; it does not answer the self-context iteration question by itself.
- The sorted-order `ciou_bbox_area` control is supportive context, not a causal comparator:
  - it uses a different base model path
  - it uses sorted ordering rather than random ordering
  - the stored `resolved_config.json` shows `bbox_log_area_weight=0.0`, so the path name should not be over-read as proof that an active bbox-area term was present in the final resolved run state
- The random-order `iter_4` run is partial/incomplete. The fairest comparisons are at shared checkpoints (`300`, `600`), not at its truncated endpoint.
- Numeric claims below come from:
  - `logging.jsonl` for train/eval dynamics and runtime
  - `monitor_dumps/*.json` for sample-level and aggregate rollout evidence
  - `monitor_dumps/vis_step_*/class_summary.json` where available
- Historical overlap used only for interpretation, not for March 2026 metrics:
  - `progress/diagnostics/stage2_channel_a_coord_loss_2026-02-25.md`
  - `progress/diagnostics/stage2_channel_a_visual_audit_2026-02-25.md`
  - `progress/diagnostics/stage2_softctx_discretization_vs_stage1_bbox_2026-02-22.md`

## 4. Per-Run Evidence

### 4.1 Random-order `iter_1` baseline

Artifact root:
- `output/a_only/random_order/epoch-1-iter_1-eff_bs_128/v0-20260318-094911`

Training dynamics from `logging.jsonl`:
- total loss improves from `1.1386` to best `0.8016`, ending at `0.8136`
- `coord_diag/A1/acc_top5`: `0.3381 -> 0.4672 -> 0.4481`
- `coord_diag/A1/expected_bin_mae`: `74.4 -> 27.6 -> 32.4`

Eval at logged checkpoints:
- `step 300`: `mAP=0.4417`, `F1=0.6445`, `precision=0.6146`, `recall=0.6773`, `pred_objects=4030`, `matched=2477`
- `step 600`: `mAP=0.4466`, `F1=0.6486`, `precision=0.6317`, `recall=0.6664`, `pred_objects=3858`, `matched=2437`
- `step 900`: `mAP=0.4533`, `F1=0.6775`, `precision=0.6887`, `recall=0.6667`, `pred_objects=3540`, `matched=2438`

Runtime:
- mean `time/forward_s = 6.78`
- mean `train_speed(iter/s) = 0.0360`

Monitor-dump evidence:
- `step_000600.json` sample summary:
  - average `pred/gt` ratio `1.76`
  - `1 / 10` samples with `f1 < 0.15`
  - `1 / 10` truncated samples
- `vis_step_000900/class_summary.json`:
  - `fp_total=77`, `fn_total=43`
  - top FP classes: `person=40`, `book=32`
  - top FN classes: `person=19`, `book=10`, `bench=7`
- Qualitative read:
  - the bookshelf sample is still duplicate-prone
  - but compared with later-pass runs, boxes remain much more instance-shaped and matching stays substantially higher

### 4.2 Random-order `iter_1` with stop-signal damping

Artifact root:
- `output/a_only/random_order-stop_signal_damping/epoch-1-iter_1-eff_bs_128/v0-20260318-044005`

Training dynamics:
- total loss improves from `1.1372` to best `0.7985`, ending at `0.8257`
- `coord_diag/A1/acc_top5`: `0.3381 -> 0.4701 -> 0.4421`
- `coord_diag/A1/expected_bin_mae`: `74.4 -> 27.8 -> 30.0`

Eval and rollout behavior at `step 300`:
- `mAP=0.4209`, `F1=0.4330`, `precision=0.3119`, `recall=0.7077`
- `pred_objects=8297`, `matched=2588`, `fp_total=5709`, `fn_total=1069`
- `sample_valid_pred_rate=1.0`, but `parse_truncated_rate=0.0742`

Runtime:
- mean `time/forward_s = 6.75`
- mean `train_speed(iter/s) = 0.0350`

Monitor-dump evidence:
- `step_000300.json` sample summary:
  - average `pred/gt` ratio `2.87`
  - `3 / 10` truncated samples
  - `2 / 10` samples with `f1 < 0.15`
  - three samples hit `pred >= 64`

Interpretation:
- this control does not rescue rollout quality
- the dominant problem here is over-generation / stopping, not improved atomicity
- it therefore does not explain away the later-pass failures

### 4.3 Random-order `iter_2`

Artifact root:
- `output/a_only/random_order-iter_2/epoch-1-eff_bs_128/v0-20260319-133905`

Training dynamics:
- total loss improves from `1.0057` to best `0.7138`, ending at `0.7323`
- final-pass losses improve materially early in training:
  - `loss/A2_coord/coord_soft_ce`: `0.5371 -> 0.4013 -> 0.4189`
  - `loss/A2_coord/coord_token_ce`: `0.1061 -> 0.0785 -> 0.0821`
  - `loss/A2_coord/bbox_smoothl1`: `0.0393 -> 0.0120 -> 0.0157`
  - `loss/A2_coord/bbox_ciou`: `0.1620 -> 0.1032 -> 0.1249`
- But the final pass never becomes better than the anchor pass:
  - `coord_diag/A2/acc_top5`: `0.2640 -> 0.3762 -> 0.3500`
  - `coord_diag/A2/expected_bin_mae`: `108.5 -> 51.2 -> 60.3`
  - `coord_diag/A1/acc_top5` first/last: `0.3381 -> 0.4081`
  - `coord_diag/A1/expected_bin_mae` first/last: `74.4 -> 46.2`
  - `0 / 88` rows where `A2` beats `A1` on either `acc_top5` or `expected_bin_mae`

Eval at shared checkpoints:
- `step 300`: `mAP=0.3989`, `F1=0.4789`, `precision=0.4321`, `recall=0.5371`, `pred_objects=4545`, `matched=1964`
- `step 600`: `mAP=0.4016`, `F1=0.4765`, `precision=0.4382`, `recall=0.5223`, `pred_objects=4359`, `matched=1910`

Runtime:
- mean `time/forward_s = 13.75`
- mean `train_speed(iter/s) = 0.0194`
- relative to random-order `iter_1`, this is about `2.03x` slower per forward and about `1.86x` slower in iteration throughput

Monitor-dump evidence:
- `step_000300.json` sample summary:
  - average `pred/gt` ratio `2.57`
  - `3 / 10` truncated samples
  - `3 / 10` samples with `f1 < 0.15`
- `step_000600.json` sample summary:
  - average `pred/gt` ratio `2.55`
  - `3 / 10` truncated samples
  - `3 / 10` samples with `f1 < 0.15`
- worst crowded samples stay severely over-generated:
  - sample `87140591468898` at `step 600`: `gt=29`, `pred=128`, `matched=2`, `f1=0.025`
  - sample `87140591468831` at `step 600`: `gt=21`, `pred=128`, `matched=3`, `f1=0.040`

Interpretation:
- scalar losses improve, but rollout matching does not
- the added pass behaves like a degraded geometry context rather than a useful refinement pass

### 4.4 Random-order `iter_2` with `coord_decode_mode=st`

Artifact root:
- `output/a_only/random_order-iter_2-decode_st/epoch-1-eff_bs_128/v0-20260320-024125`

Training dynamics:
- total loss improves from `1.0094` to best `0.7303`, ending at `0.7474`
- final-pass losses also improve:
  - `loss/A2_coord/coord_soft_ce`: `0.5371 -> 0.4152 -> 0.4412`
  - `loss/A2_coord/coord_token_ce`: `0.1061 -> 0.0814 -> 0.0867`
  - `loss/A2_coord/bbox_smoothl1`: `0.0481 -> 0.0204 -> 0.0295`
  - `loss/A2_coord/bbox_ciou`: `0.1536 -> 0.1007 -> 0.1211`
- But `A2` again never beats `A1`:
  - `coord_diag/A2/acc_top5`: `0.2640 -> 0.3648 -> 0.3182`
  - `coord_diag/A2/expected_bin_mae`: `108.5 -> 64.5 -> 88.0`
  - `0 / 37` rows where `A2` beats `A1` on either metric

Eval at `step 300`:
- `mAP=0.4057`, `F1=0.4696`, `precision=0.4255`, `recall=0.5239`
- versus random-order `iter_2` at the same step:
  - `mAP` is slightly higher (`0.4057` vs `0.3989`)
  - `F1` is slightly lower (`0.4696` vs `0.4789`)
- versus random-order `iter_1` at the same step:
  - `mAP` is still lower by `0.0360`
  - `F1` is still lower by `0.1749`

Runtime:
- mean `time/forward_s = 13.76`
- mean `train_speed(iter/s) = 0.0206`

Monitor-dump evidence:
- `step_000300.json` sample summary:
  - average `pred/gt` ratio `2.53`
  - `2 / 10` truncated samples
  - `2 / 10` samples with `f1 < 0.15`
- `vis_step_000300/class_summary.json`:
  - `fp_total=363`, `fn_total=77`
  - top FP classes: `person=172`, `book=163`, `chair=19`
  - top FN classes: `person=35`, `book=21`, `bench=7`

Interpretation:
- `decode_st` does not remove the non-atomic crowd/bookshelf regime
- it modestly changes the balance of errors, but does not change the stage-level conclusion

### 4.5 Random-order `iter_4`

Artifact root:
- `output/a_only/random_order-iter_4/epoch-1-eff_bs_128/v0-20260318-173914`

Training dynamics:
- total loss improves from `1.1606` to best `0.7831`, ending at `0.7973`
- final-pass losses improve substantially before partially regressing:
  - `loss/A2_coord/coord_soft_ce`: `0.6348 -> 0.4361 -> 0.4756`
  - `loss/A2_coord/coord_token_ce`: `0.1258 -> 0.0858 -> 0.0939`
  - `loss/A2_coord/bbox_smoothl1`: `0.0564 -> 0.0193 -> 0.0250`
  - `loss/A2_coord/bbox_ciou`: `0.1841 -> 0.1306 -> 0.1697`
- Yet the final pass remains worse than the anchor pass throughout:
  - `coord_diag/A2/acc_top5`: `0.2015 -> 0.3220 -> 0.2460`
  - `coord_diag/A2/expected_bin_mae`: `144.3 -> 75.2 -> 92.6`
  - `coord_diag/A1/acc_top5` first/last: `0.3381 -> 0.2997`
  - `coord_diag/A1/expected_bin_mae` first/last: `74.4 -> 76.1`
  - `0 / 74` rows where `A2` beats `A1` on either metric

Eval at shared checkpoints:
- `step 300`: `mAP=0.3170`, `F1=0.2751`, `precision=0.2259`, `recall=0.3519`, `pred_objects=5698`, `matched=1287`
- `step 600`: `mAP=0.3216`, `F1=0.3325`, `precision=0.3096`, `recall=0.3590`, `pred_objects=4241`, `matched=1313`
- Parseability is not the dominant issue:
  - `sample_valid_pred_rate=1.0` at both eval checkpoints
  - `parse_truncated_rate` falls from `0.0215` at `300` to `0.0020` at `600`

Runtime:
- mean `time/forward_s = 27.49`
- mean `train_speed(iter/s) = 0.0106`
- relative to random-order `iter_1`, this is about `4.05x` slower per forward and about `3.40x` slower in iteration throughput

Monitor-dump evidence:
- `step_000300.json` sample summary:
  - average `pred/gt` ratio `2.86`
  - `5 / 10` samples with `f1 < 0.15`
  - `3 / 10` truncated samples
- `step_000600.json` sample summary:
  - average `pred/gt` ratio improves to `1.67`
  - `5 / 10` samples still have `f1 < 0.15`
  - only `1 / 10` sample remains truncated
- `vis_step_000600/class_summary.json`:
  - `fp_total=218`, `fn_total=90`
  - top FP classes: `book=147`, `person=60`
  - top FN classes: `person=41`, `book=24`, `bench=7`
- Visualization-side evidence:
  - `monitor_dumps/vis_step_000600/step_000600_s07_base000354.png` shows row-spanning `person` boxes covering crowd regions and large court-area strips rather than one box per spectator/player
  - `monitor_dumps/vis_step_000600/step_000600_s09_base000002.png` shows shelf-covering and repeated vertical `book` boxes rather than stable per-book partitioning
- Same-sample comparison against random-order `iter_1` at `step 600`:
  - sample `87140591468561`: `iter_1 pred=10, matched=9, f1=0.750` vs `iter_4 pred=15, matched=2, f1=0.138`
  - sample `87140591468591`: `iter_1 pred=21, matched=12, f1=0.632` vs `iter_4 pred=22, matched=2, f1=0.103`
  - sample `87140591468602`: `iter_1 pred=11, matched=9, f1=0.783` vs `iter_4 pred=13, matched=6, f1=0.480`
- Mixed-evidence note:
  - sample `87140591468546` is already bad in random-order `iter_1` (`pred=128`, `matched=5`, `f1=0.069`), so not every crowded-scene failure is uniquely introduced by `iter_4`
  - however, the fixed-sample summary still shifts clearly in the wrong direction: random-order `iter_1` has `1 / 10` samples with `f1 < 0.15` at `step 600`, while random-order `iter_4` has `5 / 10`

Interpretation:
- this is the clearest direct evidence that deeper Channel-A self-context passes are learning a semantically plausible but atomically wrong geometry regime

### 4.6 Sorted-order `ciou_bbox_area` control

Artifact root:
- `output/a_only/ciou_bbox_area/4b-softce_w1-coco80/epoch_1-eff_size_128-n_softctx_iter_1-a_only/v1-20260316-105736`

Role in this note:
- this is a supporting control, not a direct causal comparator to the random-order March runs

Stored resolved-config facts:
- `object_ordering=sorted`
- `n_softctx_iter=1`
- base model differs from the random-order March family
- `token_ce.struct_ce_weight=0.1`
- `coord_reg.w1_weight=0.0`
- `bbox_size_aux.log_area_weight=0.0`

Training dynamics:
- total loss improves from `0.7037` to best `0.6578`, ending at `0.6632`
- `coord_diag/A1/acc_top5`: `0.4632 -> 0.5184 -> 0.4884`
- `coord_diag/A1/expected_bin_mae`: `19.7 -> 15.3 -> 16.6`

Eval:
- `step 300`: `mAP=0.4383`, `F1=0.7277`
- `step 600`: `mAP=0.4393`, `F1=0.7314`
- `step 900`: `mAP=0.4417`, `F1=0.7385`
- `step 915`: `mAP=0.4395`, `F1=0.7357`
- all logged eval checkpoints have:
  - `sample_valid_pred_rate=1.0`
  - `parse_truncated_rate=0.0`

Runtime:
- mean `time/forward_s = 7.03`
- mean `train_speed(iter/s) = 0.0367`

Monitor-dump evidence:
- `step_000600.json` sample summary:
  - average `pred/gt` ratio `0.94`
  - `0 / 10` truncated samples
  - `0 / 10` samples with `f1 < 0.15`
- dense bookshelf sample `87140591468898` is still hard (`pred=40`, `matched=8`, `f1=0.232` at `step 600`), so this control does not solve dense small-object detection
- But the catastrophic large-region collapse seen in random-order `iter_4` is absent

Interpretation:
- this control does not prove that sorted order is the answer
- it does show that a stable A1-only Channel-A rollout regime is achievable without the later-pass failure pattern

## 5. Cross-Run Synthesis

### 5.1 Does extra self-context improve scalar losses?

Yes, but only in the limited sense that the altered `iter > 1` objective bundles optimize their own scalar losses and improve their own final-pass coord diagnostics relative to their starts.

Direct evidence:
- random-order `iter_2` total loss improves `1.0057 -> 0.7138`
- random-order `iter_4` total loss improves `1.1606 -> 0.7831`
- final-pass bbox/coord losses also improve materially early in both runs

But that improvement is not enough to justify the later passes, because the current trainer never turns those final-pass metrics into a better-than-anchor geometry state:
- random-order `iter_2`: `A2` beats `A1` on neither coord diagnostic in any logged row
- random-order `iter_2 decode_st`: same
- random-order `iter_4`: same

So the direct training-side evidence is:
- the later pass is not pure noise
- but it is also not a superior refinement pass

### 5.2 Does that scalar improvement translate to better eval rollout?

No.

At shared checkpoint `300`:
- random-order `iter_1`: `mAP=0.4417`, `F1=0.6445`
- random-order `iter_2`: `mAP=0.3989`, `F1=0.4789`
- random-order `iter_2 decode_st`: `mAP=0.4057`, `F1=0.4696`
- random-order `iter_4`: `mAP=0.3170`, `F1=0.2751`

At shared checkpoint `600`:
- random-order `iter_1`: `mAP=0.4466`, `F1=0.6486`
- random-order `iter_2`: `mAP=0.4016`, `F1=0.4765`
- random-order `iter_4`: `mAP=0.3216`, `F1=0.3325`

Matched-object counts move in the same direction:
- `step 600` matched count: `2437` (`iter_1`) vs `1910` (`iter_2`) vs `1313` (`iter_4`)

So the current evidence does not support the claim that more self-context iterations improve rollout quality.

### 5.3 What failure mode appears most consistently?

The most consistent failure mode is:
- valid parse
- semantically plausible class choice
- poor object matching because boxes stop behaving atomically

Direct evidence for that read:
- `sample_valid_pred_rate` remains `1.0` in the `iter_2`, `iter_2 decode_st`, and `iter_4` eval rows
- the problem shows up in `matched`, `fp_total`, `fn_total`, fixed-sample `f1`, and the overlays
- the visual artifacts are exactly the kind described in the older February diagnostics:
  - long, row-like crowd boxes
  - shelf-covering / repeated book strips
  - large semantic-region boxes instead of one box per instance

This aligns with the older historical interpretation in:
- `stage2_channel_a_coord_loss_2026-02-25.md`
- `stage2_channel_a_visual_audit_2026-02-25.md`
- `stage2_softctx_discretization_vs_stage1_bbox_2026-02-22.md`

The March random-order runs reproduce that same failure pattern under a different order regime rather than overturning it.

### 5.4 What do the supporting controls contribute?

Stop-signal damping:
- shows that `n_iter=1` can still fail by count explosion if stopping is perturbed
- does not provide positive evidence for later self-context passes

`iter_2 decode_st`:
- shows that swapping `coord_decode_mode` from `exp` to `st` does not rescue the `iter_2` rollout regime

Sorted-order `ciou_bbox_area` control:
- shows that an A1-only regime can remain rollout-stable and parse-clean
- does not prove that sorted order or any single config knob is the cause, because the init and objective surface differ

## 6. Decision

Decision:
- Keep only `A1` (`n_softctx_iter=1`) in the current Channel-A-only direction.
- Drop `A2` / `A3` / `A4` from active near-term configs and discussions unless a cleaner matched-weight ablation is explicitly being run.

Why this is strong enough for a practical decision:
- all direct `n_iter > 1` artifacts here underperform the random-order `iter_1` baseline on rollout quality
- the final self-context pass never beats the anchor pass on logged coord diagnostics
- `decode_st` does not change the conclusion
- runtime cost scales sharply upward (`~2x` for `iter_2`, `~4x` for `iter_4` in forward time) while rollout quality degrades
- the observed failure is not a minor metric wobble; it is a concrete object-level failure mode that makes the boxes less useful for detection

Confidence statement:
- medium-high confidence that later self-context passes should be retired for now in the current Channel-A-only implementation regime
- medium confidence that iteration count itself is the entire cause, because the `iter_2` and `iter_4` runs also changed the loss bundle

Inference boundary:
- direct evidence exists for `iter_2` and `iter_4`
- dropping `A3`/`A4` is an inference, not a direct measurement of independently supervised later passes
- that inference is still reasonable here because the current trainer supervises only the final pass, `iter_4` already performs worse than `iter_2`, and there is no evidence that extra unsupervised intermediate passes are buying anything useful

## 7. Open Questions And Minimum Follow-Up

Only a small follow-up surface is justified if this topic remains active.

1. Run one matched-weight rerun that changes only `n_softctx_iter` (`1` vs `2`) from the same init/seed.
   - Purpose: separate the iteration question from the altered `struct_ce` / bbox / coord-loss bundle.
2. If self-context remains interesting, test it only together with an explicit object-level corrective signal rather than deeper Channel-A-only passes.
   - The current evidence says the failure is object partitioning, not raw syntax.
3. Keep the fixed-sample monitor gate for any revisit.
   - Minimum gate metrics: count of samples with `f1 < 0.15`, count of samples with `pred >= 64`, and qualitative review of the bookshelf / crowd overlays at shared checkpoints.
