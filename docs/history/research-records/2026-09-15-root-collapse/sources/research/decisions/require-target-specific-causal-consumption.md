---
id: decision.require-target-specific-causal-consumption
type: decision
status: active
updated: 2026-07-12
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/heldout-representation-results-2026-07-12.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md
relations:
  supports:
    - decision.separate-selection-transcription-commit-stop
  narrows:
    - decision.visual-designation-causal-teacher
  supersedes: []
---
# Require Target-Specific Causal Consumption

## Decision

Do not promote an auxiliary representation because it is decodable,
correlated, or improves a probe loss. Promotion requires a controlled
intervention in which the correct representation changes the intended row and
source-swapped, token-permuted, position-only, and norm-matched controls lose
that effect.

## Evidence

- The proposal readout carried held-out target-specific visual information.
- The tested own-prefix bridge raised labeled recall only slightly while
  causing precision collapse, duplication, malformed output, prediction-count
  inflation, and closure failure.
- Another-image and token-permuted controls reproduced the harmful signature,
  making a generic continuation or traversal pulse more plausible than a
  target-specific visual bridge.

## Belief Update

Representation availability and behavioral use are separate hypotheses. A
bridge must demonstrate source-specific causal consumption before its scale,
lifetime, or downstream architecture is optimized.

## Next Discriminator

Run a one-row target-specific transition at a fixed legal prefix. Compare the
correct proposal with within-image permutation, another-image, position-only,
norm-matched random, and bridge-off controls. Score the first distinguishing
phrase token, complete phrase, every coordinate slot, full-row binding, and
valid closure. Failure retires this delivery route rather than triggering a
larger rollout or magnitude sweep.
