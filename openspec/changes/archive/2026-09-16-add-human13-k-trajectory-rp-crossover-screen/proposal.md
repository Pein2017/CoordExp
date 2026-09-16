## Why

The prior Human-13 one-update probes can add K-hit owners to clean-greedy
decode, but every proposal also displaced Source-visible owners and only some
selected targets compiled into greedy behavior.  The next bounded question is
whether whole-trajectory sampling credit, a sparse greedy compiler, and direct
proposal preservation can separate gain from owner exchange under both
repetition-penalty policies relevant to this project.

## What Changes

- Add an experiment-local RP-aware K16 acquisition and replay contract for
  `repetition_penalty` values `1.0` and `1.10`, including processed chosen-token
  likelihood evidence and a live no-update parity gate.  Exact score-function
  replay and the score-function gradient forward run on the HF fp32/SDPA
  exact-history surface; this is an execution-surface correction after the
  quantified v3/v4 live parity failures, and the BF16/FA2 packed forward
  carries no score-function evidence.
- Add a one-image (1584) K16 parity-only v5 qualification per RP contract
  that must pass the unchanged sealed gate on the exact surface before any
  witness, dose, update, or owner analysis; failure retires the
  exact-on-policy route instead of tuning tolerance, and a pass hands the
  existing vertical unchanged to a fresh full-panel successor root.
- Add detached first-hit owner accounting and signed row-level trajectory
  credit, with legacy K-miss owners neutral and terminal STOP one-sided.
- Add a sparse Source-boundary greedy compiler over the frozen metric-valid
  native alias bank.
- Add exact fresh-AdamW proposal reconstruction and an owner-wise preservation
  projection over the actual parameter delta.
- Add a qualification-only, predeclared AdamW learning-rate dose ray that may
  replace the default `3e-6` once using mechanics-only gates, then freezes one
  global learning rate for both RP contracts, every arm, and every matrix seed.
- Add a bounded matrix runner that evaluates every private one-update proposal
  under clean greedy at both RP values and then restores Source exactly.
- Add immutable receipts and an analyzer for the paired `trajectory`,
  `trajectory+compiler`, and `trajectory+compiler+preservation` contrasts.
- Keep the screen deliberately narrow: no K-miss supervision, multi-update
  continuation, accepted checkpoint, production default change, or validation
  claim.

## Capabilities

### New Capabilities

- `coordexp-infras-human13-k-trajectory-rp-crossover-screen`: Experiment-local
  contracts for RP-aware sampled-policy evidence, row-level trajectory credit,
  sparse greedy compilation, exact proposal preservation, dual-RP evaluation,
  exact rollback, and bounded Human-13 matrix execution.

### Modified Capabilities

None.  Existing production inference defaults, model assembly, packing,
optimizer, parser, matcher, checkpoint, and evaluation contracts remain
unchanged and are reused through experiment-local adapters.

## Impact

- New Human-13 research helpers, leaf configs, typed receipts, focused tests,
  one production-shaped dose-qualified vertical, and one bounded eighteen-
  proposal artifact tree.
- Reuse of the sealed Human-13 manifest and alias bank, language-only DoRA
  surface, HF fp32/SDPA batch-one evaluator and exact-history surface (which
  also owns score-function replay and gradients), owner matcher, and full
  training-state transaction.  The no-padding packed forward remains only for
  non-score-function plumbing proven mathematically identical.
- The owning research unit defines the cohort, utility, success rule, matrix,
  and scientific claim.  This OpenSpec change owns only the bounded
  implementation and execution behavior.
- No new external dependency, owner architecture, generic RL trainer,
  production API, deployment checkpoint, or population-level conclusion.

## Closeout disposition (2026-08-24)

This change is closed as verified negative evidence at its predeclared
sampler-versus-replay admission gate. The exact-on-policy route did not admit
an update, so Task 4.5 and Tasks 6.4--7.4 remain intentionally unchecked and
will not be executed. The owning evidence is the
[bounded result](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/results.md).

The later N=13 K4/K8 standalone result is a different, simplified treatment
and does not retroactively complete this change's exact-policy, compiler,
projection, dose-ray, or eighteen-cell contracts. Its bounded conclusion is
recorded in the
[continuity note](../../../memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md):
under that exact cohort and four-update dual-RP screen, K8 is not supported
over K4. Neither result supports generalization, deployment, checkpoint
promotion, or resuming this retired matrix.
