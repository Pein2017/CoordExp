## Why

The original goal of this change was never just to show that a teacher-forced
proxy can separate clean positives from synthetic negatives.
The real question is the one stated at the start of this study:

- when a rollout proposal is unmatched to GT, is it a dead / duplicate / bad
  proposal,
- or is it often a real visually grounded object that annotations miss or
  underspecify?

That question is deployment-facing, because it determines whether unmatched
proposals can be promoted later as soft pseudo-labels.

The current implementation already established something useful:

- on the clean GT-vs-hard-negative slice,
  `counterfactual` is stronger than plain `commitment`.

But that is not yet a trustworthy end conclusion for pseudo-label promotion,
because the current evidence chain is incomplete:

- successful runs mainly validate the clean teacher-forced slice,
- rollout-facing unmatched analysis is not yet a reliable primary benchmark,
- temperature effects cannot be trusted when rollout collection itself is not
  first screened for validity,
- and there is no small manual audit layer anchoring the meaning of
  high-scoring unmatched proposals.

In other words, the current study can say:

- encode-side verification has signal,

but it cannot yet safely say:

- unmatched rollout proposals are reliable pseudo-label candidates,
- nor how rollout temperature changes that conclusion.

So this change now needs to be reframed around a more trustworthy evidence
stack rather than a single monolithic run.

## What Changes

This change should be tightened into an authority-first offline study with
three explicit evidence layers:

1. `Layer A: clean verifier benchmark`
2. `Layer B: rollout proposal benchmark`
3. `Layer C: small manual audit benchmark`

### Layer A: clean verifier benchmark

Keep the existing GT positives and deterministic GT-derived hard negatives.

Purpose:

- answer whether `commitment`, `counterfactual`, and their combination have
  real grounding signal in the cleanest possible setting,
- isolate verifier quality from decode randomness, stop behavior, and parser
  instability.

This layer remains necessary, but it is no longer sufficient by itself.

### Layer B: rollout proposal benchmark

This is the core deployment-facing layer and must become primary for the final
recommendation.

It should be split into two stages:

- `B1. collection validity gate`
- `B2. rollout proposal scoring`

`B1` exists because temperature comparisons are not trustworthy unless the
proposal population is first shown to be healthy enough for analysis.

Each checkpoint × temperature run must first record collection-health metrics
such as:

- total prediction count,
- non-empty prediction image rate,
- matched count,
- unmatched count,
- ignored count,
- parse / invalid-rollout diagnostics,
- duplicate-like rate.

Only runs that pass an explicit validity gate should enter the main
temperature-comparison tables.

`B2` then scores the rollout proposals with the existing teacher-forced verifier
path and compares:

- matched vs unmatched separation,
- unmatched top-k proposal quality,
- score distributions by temperature,
- and whether `counterfactual` adds useful signal beyond `commitment`.

### Layer C: small manual audit benchmark

Add a small manually auditable unmatched subset.

Purpose:

- ground the meaning of high-scoring unmatched proposals,
- distinguish real visible objects from duplicates, wrong-location proposals,
  and dead / hallucinated proposals,
- provide the strongest trust anchor before any pseudo-label promotion claim.

This manual audit should remain small and lightweight, but it should be treated
as required for the final recommendation, not optional decoration.

## Temperature Scope

The authoritative temperature sweep should be reduced to four explicit values:

- `0.0`
- `0.3`
- `0.5`
- `0.7`

Reason:

- `0.0` is the greedy baseline,
- `0.3 / 0.5 / 0.7` cover low / medium / high decode stochasticity,
- extra points such as `0.05`, `0.1`, or `1.0` add runtime cost without clearly
  improving interpretability in the current study.

The temperature sweep should be interpreted primarily through rollout proposal
collection and unmatched quality, not through the clean GT slice alone.

## Study Contract Changes

The study should move from a single monolithic “run everything and report” shape
to a staged workflow:

- prepare subset and GT tables,
- collect rollouts,
- validate collection health,
- score only collection-valid rollout runs,
- aggregate reports,
- run a small manual audit.

This change is still intentionally narrow:

- no new detector head,
- no architectural changes,
- no retraining,
- no DETR-style branch,
- no upstream HF model edits.

## Impact

This revised framing should produce a conclusion that is substantially more
trustworthy for the original pseudo-label question:

- strongest single proxy on the clean slice,
- whether the proxy remains useful on real rollout unmatched proposals,
- how decode temperature changes proposal quality,
- and whether unmatched promotion is justified at all.

The key shift is:

- clean GT evidence remains necessary,
- rollout validity and manual unmatched audit become mandatory for a final
  pseudo-label recommendation.
