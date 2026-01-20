## 1. Implementation
- [x] 1.1 Add a YAML knob under `custom.extra.rollout_matching` to select post-rollout packing scope (micro vs window).
- [x] 1.2 Implement window-local accumulation of post-rollout segments across a *full* gradient-accumulation window.
- [x] 1.3 Implement window-aware packing selection that preserves GA semantics:
  - still executes exactly `gradient_accumulation_steps` forward/backward micro-steps per optimizer step
  - does not change loss scaling/normalization
- [x] 1.4 Ensure buffering semantics remain unchanged (E-step generate, M-step reuse). No cross-step segment carry.
- [x] 1.5 Ensure eval/predict keep existing behavior.

## 2. Instrumentation & Validation
- [x] 2.1 Add logging counters for: segments-per-window, packed sequences per window, post-rollout fill, and time spent in pack/forward.
- [x] 2.2 Add a small deterministic test or debug harness to compare old vs new scheduling on a fixed set of rollouts (greedy) and assert identical labels masks and comparable loss.

## 3. Docs
- [x] 3.1 Update `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` to document the new knob and intended use.
