---
name: debug-feedback-loop
description: Use when diagnosing or fixing a reproducible CoordExp code, config, CLI, runtime, integration, flake, or performance failure that needs a tight feedback loop; route model-quality, rollout, metric, or optimization symptoms to model-diagnosis.
---

# Debug Feedback Loop

Build a fast signal for the exact failure, establish the cause, and close the
loop against the original symptom.

## Mode and routing

- `diagnose`: reproduce, minimize, identify the cause, and recommend the fix;
  do not change production files without fix authorization.
- `fix`: retain a regression check, make the smallest correction, and rerun the
  original symptom.
- `performance`: establish a repeatable baseline and measurement signal before
  changing code.

Route model-quality, rollout, metric, repetition, train/eval divergence, or
optimization symptoms to `model-diagnosis`; pre-launch mechanism/loss/eval trust
questions to `model-innovation-risk-audit`; and contract review to
`audit-review`.

## Loop

1. **Reproduce.** Name one command that detects the reported symptom. Prefer a
   focused test, config load, CLI fixture, artifact replay, known-good/bad
   differential, or measured profiler. Use `/tmp/` only for disposable probes.
2. **Minimize.** Remove one input, config, environment, caller, distributed, or
   step dimension at a time. Preserve both the minimized and original
   reproducer.
3. **Discriminate.** Rank only plausible hypotheses and give each an observable
   prediction. Probe one variable at a time with boundary assertions, receipts,
   targeted logging, or runtime counters.
4. **Classify.** Separate implementation defects, missing test/interface seams,
   contract mismatches, and user-owned research-semantic decisions. Stop for
   the user before changing algorithms, data meaning, objectives, metrics, or
   statistical claims.
5. **Fix when authorized.** Observe the regression check fail, apply the bounded
   correction, observe it pass, then rerun the original reproducer and adjacent
   checks selected by the impact radius.
6. **Clean up.** Remove temporary instrumentation and report the causal chain,
   falsified alternatives, verification scope, and residual risk.

The feedback signal must be symptom-specific, repeatable or probability
measured, fast enough to iterate, agent-runnable, and scope-labeled (`unit`,
`CLI`, `artifact replay`, `tiny`, and so on). If no such signal can be built,
report the missing artifact/access and the smallest capture that would unblock
it; do not call a hunch the root cause.

If the correct regression seam is missing, record that architecture finding
and route it to `improve-codebase-architecture` after the immediate failure is
understood. Do not add a shallow test that creates false confidence.

## Output

Lead with the root cause or current blocker, then give the feedback command,
minimized evidence, correction or next probe, verification of both minimized
and original symptoms, and any user-owned semantic decision.
