# Trial v1 result: real learning executed, no dev natural benefit

Status: lead-accepted bounded negative result. Frozen protocol: `protocol-v1.md`.
Grant: `root-owner-v1-20260908-a`. No retry, sweep, extra update or promotion.

## Observation

Primary category-agnostic pixel-IoU>=0.50 one-to-one pooled outcomes:

| Panel | Arm | Owners / GT | Predictions | Recall | Annotation-relative F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| train16 | Source | 93 / 124 | 130 | .750000 | .732283 |
| train16 | immediate | 94 / 124 | 131 | .758065 | .737255 |
| train16 | downstream | 93 / 124 | 131 | .750000 | .729412 |
| dev64 | Source | 295 / 498 | 574 | .592369 | .550373 |
| dev64 | immediate | 293 / 498 | 583 | .588353 | .542091 |
| dev64 | downstream | 295 / 498 | 575 | .592369 | .549860 |

- Immediate versus Source gains train owner `1926253` in image `532132` and
  loses none; dev gains none and loses `1656471` in image `131580` and `1194977`
  in image `255904`.
- Downstream versus Source preserves exactly the train and dev owner sets:
  zero gained and zero lost. It emits one additional valid prediction in each
  panel, slightly reducing annotation-relative F1. Downstream versus immediate
  recovers the two dev owners but loses the single train gain.
- All 160 new trajectories terminate `im_end`; zero token-cap rows. Train has
  no parser drops or strict geometry repeats. Dev Source/immediate/downstream
  have 3/3/2 parser-dropped predictions respectively; strict IoU>0.95 later
  prediction counts are 0/0/1. Valid unmatched dev predictions are 279/290/280.
  These are annotation-relative unmatched counts, not physical false-object
  labels; no automatic grounding claim is made.
- Category-consistent dev owners are 287/284/287, respectively. The primary
  conclusion does not depend on relabeling category-agnostic results.

## Interpretation and stop

Neither arm improves the decision-bearing natural dev owner/F1 endpoint over
Source. Immediate's small training gain does not transfer; downstream's better
dev outcome relative to immediate is avoidance of a regression, not an absolute
gain. This one-step, low-dose, reused-development-panel result does not show
that downstream credit cannot work, nor that forced-prefix preferences improved:
the latter was not measured after the update. It closes only this frozen trial.
No claim of robust generalization or statistical superiority is warranted.
The protected confirmation512 was not used. Any new dose, fresh bank, objective
or trial requires a new root decision; no automatic continuation is scheduled.

## Technical evidence

Controller PASS in 334.300861 seconds (budget 600 seconds); all 18 child
processes exit 0. Two independent Source-initialized eight-rank updates each
execute 64 action forwards/backwards, 256 coordinate tokens, 16 images and
exactly one AdamW step. Each changes all 588 permitted DoRA tensors.
Immediate/downstream raw gradient norms are 1.459983/1.708164, clipped norms
approximately 1.0, and parameter delta L2 is .010587843/.010588718. This is
actual update/save evidence, not a learning-benefit proxy.

Cold evaluation validates the requested saved adapter and original selected
embedding identities at runtime, with eight rank receipts and complete train16
plus dev64 for each arm. Fresh CPU reduction from persisted rows exits 0 with
`status=PASS` and six paired comparisons. Root independently verified controller
completion, counters, changed tensors and persisted checkpoint hashes.

Controller cleanup was corrected before launch to use one shared five-second
TERM grace period across all children (not five seconds per child); an
eight-stubborn-process CPU test verified the shared bound. Runtime code hashes
are bound in the execution receipt; the scientific plan remained unchanged.

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v1/`

- `plan.json`: SHA256 `dcb5a334933a22faea5030d47adb5089f0c62036a36a70bdc93700e87c47e9ea`.
- `execution.json`: grant, code hashes, commands, per-child exit statuses and time.
- `results.json`: identity-checked totals, all paired image/owner changes, source
baseline hashes, raw input references and arm receipt hashes.
- `lead-results.json`: root's fresh CPU reduction, identical to `results.json`
  (SHA256 `0043c13fe067f13dc1a52c130a70eb2a02b1cdbea8a439d01f543af9cc45b3a9`).
- `immediate/train-receipt.json`, `downstream/train-receipt.json`: counters,
  gradients, parameter changes, optimizer and rank evidence.
- `immediate/adapter/adapter_model.safetensors`: SHA256
  `01ab78211ec2c0c777fdfe307e29c5ac8414894d2983f66314a457ddf3344cc7`.
- `downstream/adapter/adapter_model.safetensors`: SHA256
  `848ede9b1bee50e46da8c2a8803e4a8dab4dfee3202f0a00a8c89eb63e9e7f8b`.
- Arm `eval/` directories retain raw natural outputs and cold-load receipts.

Reproduction (CPU, run from `/data/CoordExp/.worktrees/dora-prox-linear-n2`):

```sh
python /data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/learning/reduce.py --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v1/plan.json --trial-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v1 --output /tmp/owner-outcome-v1-recheck.json
```
