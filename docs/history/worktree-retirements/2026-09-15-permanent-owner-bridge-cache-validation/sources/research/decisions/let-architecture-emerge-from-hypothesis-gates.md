---
id: decision.let-architecture-emerge-from-hypothesis-gates
type: decision
status: active
updated: 2026-07-12
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-canonical-commit-depth/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md
  - docs/history/worktree-cleanup/2026-07-12-pvci-research-worktree-recycle/README.md
relations:
  supports: []
  narrows: []
  supersedes: []
---
# Let Architecture Emerge from Hypothesis Gates

## Decision

Do not select slots, a persistent ledger, a cursor renderer, or a final forward
pass in advance. Promote only the smallest mechanism whose defining causal
signature passes a bounded experiment on canonical CoordExp-Swift.

## Evidence

- Native commit probes found a real but order-conditioned transition and
  explicitly did not authorize slots or memory.
- The proposal screen found a reusable representation handle, while its tested
  behavioral bridge failed target-specificity and safety gates.
- Temporary worktrees yielded durable evidence but were retired as
  implementation bases; canonical infrastructure now owns future probes.

## Belief Update

The research graph should retain failed routes and narrowed claims without
turning them into permanent code. Architecture is a posterior over the latest
passed and failed discriminators, not a roadmap that experiments are expected
to confirm.

## Next Discriminator

Before each new implementation unit, state one hypothesis, its strongest
alternative, a minimal intervention, a falsifier, and the route change for
each outcome. A failed gate must narrow or retire a decision before more scale,
more controls, or a larger module is added.
