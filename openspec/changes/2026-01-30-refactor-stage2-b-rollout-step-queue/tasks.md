## 1. Proposal Validation Prep
- [x] Run `openspec validate 2026-01-30-refactor-stage2-b-rollout-step-queue --strict`

## 2. Spec Deltas
- [x] Add delta spec updates under `specs/stage2-ab-training/spec.md` for step-budgeted Channel-B and the in-step 1-step queue pipeline.

## 3. Stage2-AB Channel-B Refactor
- [x] Implement step-budgeted Channel-B in `src/trainers/stage2_ab_training.py`
  - [x] Collect `rollouts_per_step` raw samples for the optimizer step.
  - [x] Generate rollouts in decode micro-batches (preferably batch size 2).
  - [x] Pack produced segments into variable packs under `global_max_length`.
  - [x] Run forward/backward once per pack and accumulate gradients.
  - [x] Preserve one optimizer update per optimizer step (no extra optimizer stepping inside the trainer).

## 4. Tests
- [x] Add unit test that Channel-B step-budgeted mode executes only on the last micro-step of the accumulation window.
- [x] Add regression test for stable ordering + deterministic seed derivation under step-budgeted mode.

## 5. Run Checks
- [x] `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- [x] `openspec validate 2026-01-30-refactor-stage2-b-rollout-step-queue --strict`
