# V2 result: tiny dev annotation-F1 gain, no net owner recovery

Status: lead-accepted bounded result after explicit evaluation-only
recovery. No promotion or further package work. Frozen plan remained unchanged.

## Natural outcome after recovery

| Panel | Arm | Owners / GT | Valid predictions | Recall | Annotation-relative F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev64 | Source | 295 / 498 | 574 | .592369 | .550373 |
| dev64 | immediate v1 | 293 / 498 | 583 | .588353 | .542091 |
| dev64 | downstream v1 | 295 / 498 | 575 | .592369 | .549860 |
| dev64 | completed F1 v2 | 295 / 498 | 571 | .592369 | .551918 |
| train16 | Source | 93 / 124 | 130 | .750000 | .732283 |
| train16 | immediate v1 | 94 / 124 | 131 | .758065 | .737255 |
| train16 | downstream v1 | 93 / 124 | 131 | .750000 | .729412 |
| train16 | completed F1 v2 | 92 / 124 | 131 | .741935 | .721569 |

- Dev versus Source/downstream: gain `1987423` in image `070033`, lose
  `1656471` in image `131580`, retain294; unchanged total owners, not owner-set
  parity. Valid predictions decrease3/4, respectively. Annotation-relative F1
  increases .001544546/.002057475.
- Dev versus immediate: gain `1987423`/`070033` and `1194977`/`255904`, lose0,
  retain293; predictions decrease12, F1 increases .009827023.
- Train versus Source/downstream: gain0, lose `1926376` in image `532132`,
  retain92. Versus immediate also lose `1926253` in the same image. F1 decreases
  .010714837/.007843137/.015686275 versus Source/downstream/immediate.
- All80 fresh rows terminate `im_end`, zero caps, zero strict geometry repeats.
  Train has0 parser drops; dev has3 drops in2 images, same as Source/immediate
  (downstream had2). Dev unmatched valid predictions are276 versus279/290/280
  in Source/immediate/downstream. These are annotation-relative counts.
- Category-consistent dev owners decrease287→286 versus Source; corresponding
  F1 .535079514 versus .535447761. The small primary F1 gain is not a category-
  consistent gain and not evidence of physical false-object removal.

Interpretation: this reward-only successor shows a small development-panel
annotation-burden improvement with owner turnover, but no net owner/recall
gain and a training regression. It does not achieve the requested net owner
recovery. The result does not support robust reward superiority or promotion.
Root's independently verified v1 transition census shows actual coordinate and
geometry movement despite owner-set parity, so total behavioral immobility is
not a good explanation of v1. Credit assigned only to coordinates after a
description has already been chosen remains a possible owner-ineffective-credit
mechanism, not established by this trial and not authority for another launch.

Root freshly recomputed the full v2 reduction, including historical v1 controls:
`lead-results.json` is identical to `results.json` with the SHA256 below. Root
also replayed the real saved-adapter consumer test and verified recovery exits,
the unchanged checkpoint and GPU release.

### Root contextual visual check

Root inspected the original processed images `000000070033.jpg` and
`000000131580.jpg` under
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/`.
The changed GT-owner regions contain real book/media spines in shelf context;
they are not analogous to the previously judged informationless 1x7 action.
This does not certify every neighboring prediction as a distinct owner.
In 070033, prediction10 changes pixel box `[926,349,952,439]` to
`[926,362,946,439]` and newly matches owner1987423 at IoU0.530779, without adding
a prediction. In 131580, Source prediction9 matched owner1656471 at IoU0.594043;
the F1 continuation has no assigned match to that owner and one fewer prediction.
These gains/losses are consistent with local geometric assignment changes,
not evidence of new semantic object discovery or hallucination removal.

## Recovery and final evidence

Root authorized a consumer-arm repair and evaluation-only recovery under
`root-owner-v2-f1-eval-recovery-20260908-a`. A CPU test against the actual saved
F1 receipt reproduced the failure below (RED), then passed after the evaluator
checked the plan-authorized arm (v1 still defaults to its original pair).
Wrong-plan and wrong-arm tests still reject. Nine train/F1 plus eight evaluator
tests passed; no other two-arm gate was found on this actual new-arm path.

`run_trial.py --f1-eval-recovery` pinned the already saved checkpoint below,
asserted the eval output absent/empty, launched exactly eight evaluator children
and no training, and completed PASS in126.308852 seconds (300-second budget).
All eight children exit0. The original failed44.965040-second invocation remains
preserved: combined GPU-controller intervals171.273891 seconds, including the
one useful saved training update. Recovery never repeated that update.

The final CPU `f1.py reduce` exited0/PASS with six comparisons. It freshly
recomputed v1 and required exact equality with the historical sealed result,
then validated all F1 train/checkpoint/cold-load/row/media/metric identities.
The three controls are explicitly historical, not new-plan executions.

Final artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v2-f1/`

- `results.json` SHA256:
  `94f9f4ee13aa1d0ce4c092d4c688efc807362419b2bb8037f6d35afcc310d97c`.
- `eval-recovery-execution.json`: separate recovery grant, code/checkpoint hashes,
  linkage to original failure receipt, eight child exits and126.308852-second time.
- `eval-recovery-commands.json`, `eval-recovery-controller.log`,
  `eval-recovery-logs/`: exact commands and raw logs; original logs unchanged.
- `recovery-red.log`, `recovery-green.log`: real consumer regression evidence.
- `f1/eval/`:80 new rows and eight cold-loaded identity receipts.

Training counters are64 forwards/backwards,256 coordinate tokens,16 images,
eight ranks, one optimizer step,588 changed tensors; parameter delta L2
.010587818, raw gradient norm1.72709024, clipped norm approximately1.0.
Protected confirmation512 and old controls were never regenerated or trained on.

## Preserved initial failure receipt

Initial status: concrete mechanical blocker returned to root, before recovery.
Grant `root-owner-v2-f1-20260908-a`; stopped after 44.965040 seconds. No retry.
Frozen protocol and plan: `protocol-v2-f1.md` and its recorded SHA256.

The eight-rank F1 update completed exit0 in the first phase (~39.13 seconds).
Saved adapter SHA256:
`8b43d5d2966e1d6a6bc29ca13b32c33a77f7495a08deba67d025cbec7bb19e2a`.
Original checkpoint, optimizer, training receipt and all rank logits remain at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v2-f1/f1/`.

Cold evaluation failed before model loading: `evaluate.py:_adapter_identity`
still accepts only `immediate`/`downstream` training receipt arms, rejecting the
new persisted `f1` arm with `ValueError: updated adapter training arm is invalid`.
The lead worker missed this consumer-specific whitelist while reusing the
qualified evaluator; the existing CPU tests did not exercise the new real arm.
This is an implementation failure, not evidence against F1 credit.

The controller failed on rank0 exit1 and stopped/reaped the remaining owned
children. Its original `execution.json`, `controller.log` and raw rank logs
are preserved under the new trial root. No historical control changed; no
new natural owner/recall/F1 comparison is available.

Cheapest proposed repair (not authorized/executed by this record): bind the
evaluator's allowed arm to the sealed plan, add a real saved-F1-receipt CPU
RED/GREEN test, and request one evaluation-only recovery of this exact already
saved checkpoint. Do not retrain, overwrite the original execution receipt,
relax checkpoint/Source/plan identity checks, or automatically extend the grant.

Team facts: v2 implementation used one inherited Astra worker and no L2 agents.
Inherited effort was not independently exposed/verified in the worker tools.
Rough L1 time was about15 minutes from initial v2 implementation through final
recovery/reduction, including root readiness/recovery coordination; L2 time0.
There was no prelaunch CPU failure, but one missed evaluator-arm consumer seam
caused this abort and requires a root recovery ruling. Root provided the frozen
scope, independently repeated CPU readiness and owned the single GPU grant.
This is a rough collaboration receipt, not a price or model benchmark.
