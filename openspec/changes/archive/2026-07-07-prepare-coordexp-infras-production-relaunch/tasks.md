# Tasks

## 1. Governance And Docs

- [x] 1.1 Add this follow-on OpenSpec change without modifying the completed
  rebuild task ledger.
- [x] 1.2 Add a Superpowers production-relaunch roadmap.
- [x] 1.3 Mark the old Swift source-rebuild roadmap as historical/completed or
  superseded.
- [x] 1.4 Update the execution charter phase and clarify that supervised
  packing-cache reuse is approved infrastructure.
- [x] 1.5 Validate this change and the completed rebuild baseline.

## 2. Runtime Seed Control

- [x] 2.1 Add a runtime-owned seed helper based on
  `transformers.trainer_utils.set_seed`.
- [x] 2.2 Apply the configured seed before Qwen load and fresh DoRA setup.
- [x] 2.3 Reapply the seed at runtime setup without introducing a public config
  knob for deterministic algorithms.
- [x] 2.4 Emit and manifest-link a runtime seed-control receipt.
- [x] 2.5 Add tests proving seed order and receipt materialization.
- [x] 2.6 Emit and manifest-link a scheduler receipt proving derived warmup
  steps.

## 3. Relaunch Configs

- [x] 3.1 Add the r16/a32 EBS64 warmup0p1 production config.
- [x] 3.2 Add the matching 8-GPU two-step eval-forward patchproof smoke config.
- [x] 3.3 Verify both configs load strictly and leave backend accumulation
  derived.

## 4. Verification And Launch

- [x] 4.1 Run the targeted unit/config test gate.
- [x] 4.2 Trace the production config and verify resolved values.
- [x] 4.3 Run the cheap distributed eval-forward smoke.
- [x] 4.4 Run the final 8-GPU r16/a32 EBS64 smoke.
- [x] 4.5 Relaunch production in tmux only after smoke acceptance and safe GPU
  occupancy.
- [x] 4.6 Monitor early production artifacts for expected schedule, seed,
  runtime batch, pack-cache, FA2, finite metrics, eval cadence, and checkpoint
  evidence.
