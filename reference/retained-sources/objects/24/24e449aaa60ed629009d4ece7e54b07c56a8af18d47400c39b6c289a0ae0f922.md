# Training Infra Optimization Handoff

## Objective

Reduce end-to-end time-to-training and operational risk for the current
CoordExp-Swift stack without changing experiment semantics. Preserve data,
geometry, prompts, object order, model identity, loss, EBS24, typegate,
scheduler, and eight-epoch meaning.

This handoff is only for infrastructure optimization. The step-2765 eval
collective incident was fixed and full-eval-smoked in the originating session;
do not restart that diagnosis from this document.

## Current Infra Decision

Optimize in this order:

1. real resumable checkpoints;
2. single-owner cache materialization and rank-local admission;
3. sharded, exactly-once distributed eval;
4. live progress / ETA receipts;
5. measured CPU preparation and GPU overlap.

The next session should profile and repair one item at a time. Do not launch a
new full training run merely to collect broad performance evidence.

Work package 1 is now implemented and representative-smoked. Work package 2,
cache ownership and admission, is the next unresolved item; do not reopen the
resume design unless new evidence violates the recorded boundary below.

## Evidence And Bottlenecks

- Worktree: `/data/CoordExp/.worktrees/8-coords-bbox`
- Inspected branch/head: `codex/8-coords-bbox` /
  `e7d3724b32599e2377c285dd819dc18fcbd09336`
- Production config:
  `configs/coordexp_swift/experiments/8_coords_bbox/prod_sorted.yaml`
- Historical failed production run directory (deleted after the failure facts
  were distilled into the investigation owner):
  `outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_xy_quad_clockwise_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1`
- Config fingerprint:
  `4fac90a4a5538be9ac540691d307a3b0884c0b9021e2e15a536b8c92b0bddeb1`
- Train cache fingerprint:
  `e8874aa887e249b85373d88d18938c75084260d5f891bd33a910c79853baef5b`
- Eval cache fingerprint:
  `4cbebe54b1e2a2991e4349ed319a3801c10c1901763dc7a5ec32f762c7c7a3b8`
- Observed cache construction was roughly 40 minutes and 14 GiB. Re-measure;
  this is an observation, not a target or acceptance threshold.
- Eight-rank startup spent roughly 10 additional minutes rescanning/admitting
  materialized state before useful GPU work.
- Current eval loads the complete 701-pack tuple on every rank. That duplicates
  model forward work rather than sharding the 4,952 examples.
- During a long eval, `logging.jsonl` and `run.json` expose no pack-level
  heartbeat. Operators cannot distinguish slow progress from a hang or produce
  a reliable live ETA.
- Image materialization is synchronous in the consumption path. The actual
  CPU-decode / transfer / GPU-forward critical path has not yet been measured.
- The historical failed run's `step-1659` adapter contained adapter and
  special-token weights only. It had no optimizer, scheduler, RNG, sampler, or
  trainer state, so an interruption could not resume exactly. Those failed-run
  artifacts were deleted at the user's request after this evidence was
  distilled.

## Work Packages And Acceptance

### 1. Resumable Checkpoints

Persist adapter/model state plus optimizer, scheduler, scaler if applicable,
RNG/sampler state, epoch/global step, pack position, and compatibility
fingerprints. Acceptance is an interrupt/resume smoke whose continuation
matches an uninterrupted control at the next optimizer update and scheduler
state. Fail closed on incompatible config, cache, world-size, or data identity.

Status: completed on 2026-08-02 under OpenSpec change
`add-resumable-training-checkpoints`. The implementation keeps inference
adapter/delta payloads independent and atomically adds a hash-attested
`training_state/` directory with optimizer, scheduler, optional scaler,
per-rank RNG, logical pack position, and compatibility fingerprints. Resume is
an explicit `--resume-from` CLI input outside the config fingerprint, restores
into a new run, and begins at K+1 without replaying old eval/checkpoint events.

Acceptance evidence used the exact eight-coordinate sorted/typegate/DoRA/EBS24
route with a three-step, four-rank diagnostic control. Resuming its step-2
checkpoint executed only step 3 and reproduced the complete train row exactly,
including loss `4.957846729084849` and zero terminal learning rates. The
checkpoint metadata matched at completed step 3 with optimizer, scheduler, and
zero-grad counts all equal to 3; config and world-size mutations were rejected
as `resume.compatibility_mismatch` before model setup. An independent CPU
subprocess control also matched the next dropout-bearing update, AdamW state,
and scheduler exactly. Real BF16/DDP payloads showed bounded numerical
nondeterminism despite identical train metrics: maximum absolute tensor
difference was about `9.38e-5` for adapter state and `5.54e-5` for optimizer
state.

Measured cost for the diagnostic checkpoint was 241,451,653 bytes total. The
main new cost was `optimizer.bin` at 160,992,458 bytes; scheduler state was
1,525 bytes and each of four RNG files was 15,601 bytes. The adapter and
selected-token inference payloads remained about 72.1 MB and 8.2 MB. These
numbers characterize this diagnostic configuration, not the full production
run's wall-clock or storage budget.

The final independent audit found that the first diagnostic manifest hashed
only trainer state, not the sibling adapter/delta payloads. The implementation
was hardened after the GPU smoke so new manifests also seal every learned-
payload file, resume-load outcomes are reduced across ranks before training,
and pre-trainer failures close resume hooks and the control group. A CPU/file
probe rebuilt a manifest over a hard-linked copy of the real step-3 payload and
loaded all six state files plus five inference files successfully; a valid,
shape-preserving safetensor mutation is rejected by tests. The original smoke
directories remain immutable schema-v1 pre-hardening evidence and are not valid
schema-v2 resume sources. No additional GPU job was launched for this hardening,
per the operator stop boundary.

The durable comparison receipt is
`2026-08-02-resume-equivalence-receipt.json`. File mtimes bound the observed
payload-emission window to about 1.10 seconds for step 2 and 0.52 seconds for
the final step, but this is not an instrumented end-to-end checkpoint latency.

### 2. Cache Ownership And Admission

Map the current materialization path before changing workers. Prove from
receipts whether the default 16-worker builder runs once globally or is
recreated independently by multiple ranks. Target one manifest/materialization
owner, followed by deterministic rank-local reads and a bounded readiness
barrier. A complete directory is not sufficient; every admitted payload must
be independently loadable and match the semantic fingerprint.

Acceptance must report cold and warm time-to-first-GPU, peak host memory, bytes
written/read, worker count by PID/rank, and cache reuse identity. Preserve
strict paired admission; do not solve startup time by weakening validation.

### 3. Distributed Eval Sharding

Shard eval packs across ranks and reduce sufficient statistics exactly once.
Acceptance must record the unique example IDs processed by each rank and prove:

- disjoint rank-local coverage;
- union equals all 4,952 eval examples;
- no missing or duplicate examples;
- reduced metrics match the current redundant full-eval reference within the
  metric's declared numerical tolerance.

Do not change loss normalization or model mode while sharding.

### 4. Heartbeat And ETA

Add a lightweight owner artifact or atomic `run.json` heartbeat containing the
current phase, planned step, eval pack count, total packs, last-progress time,
and rolling phase throughput. Acceptance is that a read-only monitor can tell
startup, training, eval, checkpointing, completion, and failure apart without
attaching to tmux or scraping terminal control codes.

### 5. CPU/GPU Overlap

Profile end-to-end pack latency before adding concurrency. Separate image read
and decode, processor work, host memory preparation, H2D transfer, forward,
backward, and optimizer time. Then test a bounded producer/consumer path that
prepares the next pack while the GPU consumes the current pack. Acceptance is
lower wall-clock and GPU idle time at equal outputs and bounded host memory;
isolated worker throughput is not sufficient.

## Stop Boundary

- Do not change experiment data, prompt, ordering, model, objective, EBS,
  schedule, or claim scope as an infra optimization.
- Do not pre-cache the entire future horizon by default; optimize
  time-to-primary-observation and bounded overlap.
- Do not delete or rewrite the current production cache/run evidence.
- Do not restart production training until the selected work package passes its
  representative smoke and the user authorizes the material cost.
- Do not stage, commit, or push unrelated dirty worktree changes.
- Do not take ownership of the step-2765 eval collective fix from this handoff;
  reverify its final receipt in the investigation owner instead.

## Minimum Reading Order

1. `research/investigations/eight-coordinate-bbox-supervision/README.md`
2. `configs/coordexp_swift/experiments/8_coords_bbox/prod_sorted.yaml`
3. Current checkpoint writer and pack-cache materialization/admission owners
4. Eval micro-step loading and metric reduction owners
5. This handoff last; it transports scope and evidence but is not authority

## Volatile Facts To Reverify

Recheck worktree/head/diff, active tmux and Python processes, all GPU owners,
cache manifests, checkpoint inventory, and any concurrent edits before action.
The GPU/process snapshot in the originating session is not durable state.
