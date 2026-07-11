---
name: debug-feedback-loop
description: Use when diagnosing or fixing a reproducible CoordExp code, config, CLI, runtime, integration, flake, or performance failure that needs a tight feedback loop; route model-quality, rollout, metric, or optimization symptoms to model-diagnosis.
---

# Debug Feedback Loop

Build a tight signal for the exact failure, minimize it, test falsifiable
hypotheses, and close the loop against the original symptom. Do not patch from a
plausible theory that has never been made capable of failing.

## Role And Mode

Choose the mode from the user's request:

- `diagnose`: reproduce, minimize, identify the root cause, and specify the
  correction. Do not modify production code/config/docs unless the user asks
  for a fix. Prefer `/tmp/` for disposable harnesses.
- `fix`: perform the diagnosis loop, retain a regression check at the correct
  seam, implement the bounded correction, and verify the original symptom.
- `performance`: establish a repeatable baseline and profiler/measurement
  signal before changing code.

Route elsewhere when appropriate:

- metric drops, FP/FN shifts, malformed model outputs, duplication,
  length/stop/repetition behavior, train/eval divergence, or optimization
  symptoms -> `model-diagnosis`;
- planned or newly wired model mechanisms, losses, objectives, or eval paths
  before trustworthy results -> `model-innovation-risk-audit`;
- approval, branch review, or implementation-vs-contract questions ->
  `audit-review`.

## Phase 1: Build The Feedback Loop

The feedback loop is the central artifact. Name one command that has already
been run and is capable of detecting the user's exact symptom.

Prefer, roughly in this order:

1. a focused failing `pytest` at the real seam;
2. a deterministic config/schema load with a minimal YAML or mapping;
3. a CLI invocation with a small fixture and asserted stdout/stderr/artifact;
4. replay of a captured JSONL, encoded sample, manifest, trace, or artifact;
5. a differential run across known-good/bad configs, commits, or versions;
6. a disposable harness under `/tmp/` that invokes the smallest real path;
7. a seeded property/stress loop for intermittent behavior;
8. `git bisect run` when known-good and known-bad revisions exist;
9. a measured profiler or timing harness for performance regressions.

Tighten the loop until it is:

- **symptom-specific**: it fails on the reported behavior, not merely a nearby
  exception or nonzero exit;
- **repeatable**: deterministic, or a measured high reproduction rate for a
  flake;
- **fast enough**: narrow enough to run repeatedly;
- **agent-runnable**: no hidden manual step;
- **scope-honest**: `unit`, `integration`, `CLI`, `artifact replay`, `tiny`, or
  another exact label.

If no loop can be built, stop and report what was tried, the missing access or
artifact, and the smallest instrumentation or capture that would unblock it.
Do not promote a hunch to a root cause.

## Phase 2: Reproduce And Minimize

Run the loop and confirm it exhibits the exact reported symptom. Shrink one
dimension at a time: input, data, config, callers, environment, distributed
shape, or execution steps. Re-run after every cut.

Stop minimizing when every remaining element is load-bearing or when another
cut would leave the real failure path. Preserve both:

- the minimized reproducer for reasoning and regression coverage;
- the original reproducer for end-to-end closure.

## Phase 3: Hypothesize

Generate 3-5 ranked hypotheses before testing the first plausible idea. Each
hypothesis must predict an observable change:

```text
If <cause> is responsible, then <single discriminating change or observation>
will make <specific signal> improve, disappear, or worsen.
```

Discard explanations that cannot make a falsifiable prediction. Use code and
runtime evidence to rank hypotheses. Inform the user when their domain context
could materially re-rank them, but do not turn ordinary technical diagnosis
into a questionnaire.

## Phase 4: Instrument One Variable At A Time

Choose the smallest observation that separates the leading hypotheses:

- debugger or REPL inspection;
- targeted boundary logging;
- config/runtime receipt;
- shape, dtype, mask, token, gradient, or artifact assertions;
- profiler, timing baseline, query/trace plan, or resource counters;
- known-good/known-bad differential.

Tag temporary instrumentation with a unique searchable prefix such as
`[DEBUG-<id>]`. Do not log everything and search afterward. Change one variable
per probe so the result remains interpretable.

## Phase 5: Protect Research Meaning

Classify the cause before choosing the correction:

- implementation defect with intended semantics already clear;
- missing or misplaced interface/test seam;
- config/runtime/artifact contract mismatch;
- algorithm, forward-pass, data, loss, optimization, metric, or statistical
  decision that requires user ownership.

For the last category, stop before patching. Explain the alternatives at the
architectural level, give a recommendation and evidence target, then use
`grill-me` to ask exactly one decision question. The user's lack of coding
expertise is not permission for the agent to choose scientific semantics.

## Phase 6: Fix And Regress

Run this phase only in `fix` mode or after explicit fix authorization.

1. Turn the minimized reproducer into a failing test at the correct seam when
   such a seam exists.
2. Observe the red result before the fix.
3. Apply the smallest correction that addresses the established cause.
4. Observe the regression test turn green.
5. Re-run the original, unminimized feedback loop.
6. Run adjacent contract checks selected by the actual impact radius.

If no correct test seam exists, do not add a shallow test that creates false
confidence. Record the missing seam as an architecture finding and route it to
`codebase-design` or `improve-codebase-architecture` after the immediate defect
is understood.

## Phase 7: Clean Up And Explain

Before declaring completion:

- re-run the original reproducer;
- verify the retained regression check;
- remove all tagged debug instrumentation;
- delete or clearly quarantine disposable harnesses;
- state the causal chain and which hypotheses were falsified;
- distinguish `fixes the observed symptom` from `improves the underlying model
  or research method`;
- name the architectural prevention, if one exists.

## Output

```text
Mode And Exact Symptom
Feedback Loop Command And Result
Minimized Reproducer
Ranked Hypotheses And Falsification Evidence
Root Cause
User-Owned Semantic Decision (if any)
Correction Or Fix Direction
Regression And Original-Loop Verification
Cleanup And Residual Risk
```

## Philosophy

Debugging is not inspired guessing. It is the construction of a feedback loop
that lets reality disagree with the agent. A fast, specific failure signal is
more valuable than a sophisticated explanation that cannot be falsified.

The agent should absorb the mechanical burden of reproduction, instrumentation,
and implementation evidence. The user should retain control wherever the fix
changes the algorithm, data, objective, statistical claim, or experiment. Good
debugging makes that boundary visible.
