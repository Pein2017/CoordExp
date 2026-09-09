# Authorized eight-GPU recovery round

User authorized no time/GPU-hour upper bound and explicitly requested direct
launch without occupancy-based waiting. No unrelated process is stopped.
Scientific stop remains16 additional updates per arm, refresh after8, final
holdout512, no sweep or independent confirmation.

## Fixed execution

- Four ranks per arm: control GPUs0/1/2/3, dedup GPUs4/5/6/7.
- Same Rweak64 adapter AND optimizer, global32, actual microbatch2, seed20260908.
- Initial exact train256 replay is sealed at
  `round-v1/replay/initial.json` under the experiment output root. All256 rows,
  504 eligible later rows on12 images, initial checkpoint identity verified.
- New trainer uses existing trajectory scorer, per-component Rweak losses,
  scheduler, adapter loader and atomic checkpoint publisher. The only old
  trainer change is an optional identity-gated owner-recovery recipe field;
  legacy evaluation code is unchanged. New evaluator has a separate identity.
- Replay backwards are `no_sync` before supervised work. Every rank then uses
  the same final supervised synchronization. Zero-repeat images retain their
  denominator and analytically zero penalty without a wasted replay forward.
  The control bypasses all zero-coefficient replay backwards.
- Two eight-update stages:64→72, own-model train256 refresh,72→80; then final
  natural holdout512. Each arm's durable chain is independent of the other arm's
  wall time. No checkpoint or run root is overwritten on a fresh attempt.

## Production-shaped slice

The32-image training smoke includes the longest canonical teacher344 tokens,
longest correction teacher3315 tokens and the dominant generated-repeat images
351017/417044. It retains global32/microbatch2/four-rank shape. Control checks
its padded gradient against an independent scalar legacy Rweak replay at the
same weights; thresholds are inherited gradient relative L2<=1e-4 and component
loss absolute difference<=4e-6. Dedup must produce a finite nonzero gradient
from actual generated repeats. Frozen parameters, complete checkpoint and
separate-process eight-image/two-rank native cold decode are required.

Smoke-v1 failed after the treatment update because the frozen-version snapshot
was taken BEFORE DDP construction. A local one-rank DDP reproduction changes a
frozen bias version1→2 while preserving its value exactly: constructor broadcast
was being mistaken for a training mutation. Move the snapshot after DDP, as in
the original trainer. The failed treatment artifacts are preserved; the same
known-invalid control smoke was stopped by its verified torchrun PID2364363,
without touching unrelated processes. No completed checkpoint was accepted.

Smoke-v2 launched in `coco-recovery-smoke-v2`, pane/driver PID2384351.
Its `round-v1/launch-smoke-v2/` owns code hashes, per-arm logs and terminal exits.
The one-line guard placement correction does not change the loss, optimizer,
sample, model, dose or comparison. A successful slice is mechanics evidence,
not a scientific result.

## Launch gate and continuation

`run-round.sh` owns the complete fixed round: two concurrent stage1→refresh→
stage2→final chains, then all-image reduction. It requires a lead-written
`round-v1/launch-approval.json` after actual smoke checkpoint/cold-consumer
acceptance. The script has no wall-time budget, automatic scientific extension
or retry loop. `round-v1/launch-round-v1/driver.exit` and `results.json` are the
terminal execution and reduction artifacts; lead acceptance remains separate.

## Mechanics disposition and full launch

Smoke-v2 treatment completed its real update:33.66s, peak allocated26.64GB,
reserved33.59GB and host RSS12.62GB. The rank-local dedup gradient norms were
0 /1.49466 /0.27256 /0, covering277 and158 eligible rows on two ranks and no
eligible rows on two ranks. Frozen parameters and save/cold-consume passed.

The added scalar (microbatch1) control diagnostic reported a rank-local loss
threshold failure before other ranks reached the next collective. Its asymmetric
exception caused a ten-minute NCCL timeout. Smoke-v3 now gathers parity receipts
before a common decision, and propagates gradient/frozen checks collectively.
The original failed logs remain under launch-smoke-v2; this was failure-path
handling, not a silent accepted update.

Smoke-v3 exposes the exact numbers: scalar-versus-padded gradient relative L2
3.3690e-5 on every rank (passes the1e-4 bound); largest component-loss difference
4.1723e-6 (fails the strict4e-6 scalar diagnostic). The scalar diagnostic remains
recorded as failed, not relabeled passing or assigned a tuned tolerance.
Lead removed this off-production-batch diagnostic from the critical path:
production remains the previously qualified padded microbatch2 in BOTH arms,
using the unchanged Rweak component implementation, and the actual gradient
comparison passes. No model, objective, denominator, LR, batch or dose changed.

The corrected treatment update completed again in33.58s, followed by a fresh
eight-image/two-rank native cold decode under the exact current producer:
`smoke-v3/cold-eval/dedup-step65-smoke8-b4`. The current checkpoint ID is
`03d240c63e156ece6e546b3d0e0702ae6ac0505f659af1ee3625cdbdd91181a4`.
The older/current smoke tensors were not bit-identical (max2.3842e-7 difference),
so the earlier cold run was NOT substituted as an exact-payload proof. The new
actual cold run completed and all eight rows/checkpoint identities were checked.

`round-v1/launch-approval.json` records this scoped lead ruling, exact current
code identities and real consumer evidence. It does not accept a scientific
effect. The launch script's first attempted invocation correctly exited before
launch when that receipt was absent; no scientific run was created by it.

**Full fixed round launched at2026-09-08T02:57:54Z**, tmux
`coco-owner-recovery-round-v1`, actual driver PID2428465. Both four-rank stage1
jobs are running from the original Rweak64, NOT any smoke checkpoint. The
durable driver owns refresh, stage2, final evaluation and reduction; no manual
phase restart is required. Current scientific disposition: **pending**.

## Second-update collective-order repair

Control completed updates65–72 and entered its natural train256 refresh.
Dedup completed65 (checkpoint `0501cb8a3f8fa689c1d611acfb45ec20b896ab7cebc74fab5e7f1b8d65dda82f`)
but stalled at66: the scalar gradient check between conditional replay forwards
and mandatory supervised forwards crossed DDP's lazy second-update bucket
rebuild in different orders on repeat/no-repeat ranks. A real two-rank,
two-update CPU test reproduced the timeout and passed after moving the check
after all mandatory supervised backwards. Objective, batch, optimizer and
scientific dose are unchanged. Focused trainer/consumer tests:6 passed.

Only the verified stuck dedup torchrun PID2428497 was stopped; its stage1 exit1
and all artifacts are retained. Control remains owned by the original driver.
`recover-dedup.sh`, tmux `coco-owner-recovery-dedup-resume-v1`, driver PID2445478,
resumes dedup from65 through72, refresh,80 and final holdout. Its receipts live
under `round-v1/launch-recovery-dedup-v1/`.

The evaluator accepts the exact preserved v3 producer hash only for control65–72
and dedup65, besides the current producer; unknown hashes and old-producer
dedup72/80 fail closed. `round-v1/trainer-v3.py` preserves that producer.
The original driver will exit1 because of the initial dedup failure, so the
lead must run the final reducer manually after BOTH final panels complete.
Recovery stage1 exited0: updates66–72 completed, including the formerly failing
second-forward boundary; all65–72 completed-update manifests were checked.
Dedup72 ID: `bf84753d736a33b7c5a453664957f2fbe8cf30374f04fd22a21975166fd118b6`.
Both arms now run natural train256 refresh. This retires the observed mechanics
failure, not the scientific question; final holdout evidence remains pending.

## Terminal lead acceptance

Both final evaluations exited0. Original control stage1/refresh/seal/stage2/final
all exited0; original driver exit1 is solely the preserved initial dedup-stage1
failure. The separate dedup recovery stage1/refresh/seal/stage2/final and driver
all exited0. No further launch is needed.

Wake registration failed before arming (`thread delivery requires an exact
delivery-capable daemon before arming`); the lead remained active and joined
both existing driver PIDs without runtime-service changes or relaunches.

Lead ran the final reducer manually: exit0, `RECOVERY_REDUCTION_COMPLETED`,
`round-v1/results.json`. All32 completed-update manifests65–80, both final cold
checkpoint identities, both256-image refresh identities and source hashes, and
all512 final images/3759 owners per arm were checked. Refreshed eligible rows:
control625 on11 images, treatment435 on10 (initial504 on12).
Fresh loss/recount/reducer/evaluator/trainer tests:12 passed.

Execution and bounded readout are **lead-accepted**. Scientific finding is
reduced FP/repeat burden with uncertain owner improvement (net+5, diagnostic
bootstrap95% interval[-11,+27]); see [results](results.md). No new user-accepted
architecture or publication claim is implied. Fixed stop rule reached; failed
artifacts retained, unrelated work untouched, no commit or cleanup performed.
