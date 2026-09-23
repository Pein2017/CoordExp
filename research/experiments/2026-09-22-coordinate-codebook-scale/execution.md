# Scale execution record

Status: launch preparation,2026-09-22. Authority: [unit.md](unit.md). Worker922-worker, Astra/low. The accepted predecessor and immutable preparationv3 remain unchanged.

The fixed question is natural training-panel fit after32epochs, with paired format/duplication limits on1024training/256validation. The strongest alternative is ordinary refitting/memorization; no component causal or transfer claim is sought. Epoch32 is definitive, sentinel epochs4/16 diagnostic only. Reused32source cells remain reused evidence.

Mechanical changes are limited to the maintained probe package: explicit new root/cache/reserve arguments in `execute.py`, task-specific `scale_execute.py`, first-production instrumentation in `scale_train.py`, and saved-output scale reduction. The maintained trainer, model assembly, objective, optimizer and evaluator are reused. Source captures are retained before new process launch; captures are never executed.

All four production queues were copied byte-exactly into separate condition directories. A CPU regression reproduces the shared-parent collision and verifies isolation. The supervisor preserves old defaults for existing callers and uses the new ledger/cache with900-second reserve here. Global hard limits remain8wall-hours/64GPU-hours, counting in-flight allocations and loading from first process launch. Four training ranks overlap four evaluator workers; all eight serve fixed evaluation queues after training. No GPU/model work has begun at this note's preparation boundary.

First-two-update instrumentation validates real rank/pack identities, segment denominator, optimizer coverage, gradients and frozen tensors without extra model forwards. Valid updates count toward1968; prior qualification is reused rather than rerun. CPU preparation and failures are retained under the new output root. Final records will distinguish executed, reused, missing/HOLD cells and include costs and terminal producer evidence.

## Preserved first-update instrumentation failure and repair

The first GPU/model-entry launch set wall_start=1790076495.1041024. Attempt `fit-seed1729` reached one optimizer call on each rank but failed the newly added unconditional nonzero-parameter-delta assertion before a training log row or checkpoint was published. This is not zero optimizer calls: the zero-LR call can advance Adam moments. Four source evaluators continued independently. All first-production rank receipts, failed run files, log, source captures and costs remain under their original paths; `first-production-failure-v1.json` binds the failure.

The maintained constant_with_warmup10 scheduler initializes all group rates to zero. The probe now validates rates before the call and requires parameter movement only after a positive-LR call; all-zero LR is allowed only at call1. Call2 must have positive LR and movement. The actual CPU before/step/after regression demonstrates Adam state advancement at zero LR, rejects the premature pre-step delta check, and rejects a positive-LR optimizer no-op. Seven focused CPU tests pass; an initial CPU fixture failure (missing required optimizer betas/epsilon) is retained in repair-tests-v2.txt, followed by repair-tests-v3.txt.

`launch-v2.json` freezes the corrected probe and collision-free retry. Only `run.output_dir` changes in its config; source, seed, cache,1968-update exposure, optimizer and all scientific settings are identical to v3. The fresh retry uses `scale1024-seed1729-32epoch-repair-v2` and writes separate rank receipts. Supervision continues the existing ledger and reserves900seconds; it recognizes the still-owned original source producers and never reclaims their GPUs or resets the clock. The original coordinator is retained until its source jobs join. This is a mechanical retry, not an outcome-selected fit or qualification replay.

## Terminal execution

The corrected run completed1968updates and all2720new evaluation cells;32source cells were reused. Both coordinators were joined and all22producer intervals are terminal. See [candidate-results.md](candidate-results.md) for the frozen endpoint, failed guardrails, costs and replay commands. No successor was launched.
