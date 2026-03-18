## 1. Config And Runtime Deprecation

- [x] 1.1 Reject any authored `token_ce.config.stop_signal_damping` presence for Stage-2 and rollout-aligned pipelines with a fail-fast deprecation error.
- [x] 1.2 Remove token CE stop-signal runtime weighting and all related objective / diagnostic emission from active training code.

## 2. Live Surface Cleanup

- [x] 2.1 Scrub checked-in YAML profiles, smoke configs, and reference configs that still advertise stop-signal damping.
- [x] 2.2 Remove stable training-doc references to `stop_signal_ce`, `stop_signal/*`, and dedicated stop-signal smoke/reference workflows.
- [x] 2.3 Convert positive stop-signal tests into deprecation/fail-fast coverage and delete feature-only tests that no longer represent supported behavior.

## 3. Governance

- [x] 3.1 Add this rollback/deprecation change to record the removal decision.
- [x] 3.2 Archive the original `adaptive-stop-signal-damping` change without syncing its delta specs into `openspec/specs/`.
