## Why

Qwen3 Vision-Language (Qwen3-VL) dense enumeration can expose verified objects through stochastic
same-prefix sampling that ordinary greedy rollout misses, while other states
enter a duplicate, premature terminal, or geometrically incorrect branch. The
approved research unit now needs a bounded training path that acts directly on
those rollout-derived decision sites without adding an inference-time
architecture or mixing canonical supervised-fine-tuning sequences into the
pilot.

The scientific design is owned by the
[Own-Prefix Entity-Transition and Coordinate-Boundary Calibration Training Screen](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/unit.md).
This change supplies only the minimum reusable implementation needed to run
that screen.

## What Changes

- Add a frozen, exact-token rollout state-bank input path with physical-entity,
  review, provenance, split-group, candidate, and geometry eligibility data.
- Add same-prefix grouped candidate scoring for verified uncovered positives
  versus the actual harmful greedy branch.
- Add first-wrong-coordinate supervision against a reviewed discrete set of
  acceptable coordinate tokens.
- Apply a small token-type gate at every rollout-derived decision site with a
  defined intended token type.
- Add transition-only, coordinate-boundary-only, and joint research profiles,
  with independent eligibility masks, normalization, metrics, and compact run
  receipts.
- Reuse the current full image-and-prefix Qwen forward, Weight-Decomposed
  Low-Rank Adaptation setup, optimizer/runtime, checkpoint writer, and
  inference pipeline.
- Add the one-event and 8-to-16-state smoke paths required before the formal
  screen.
- Do not add canonical supervised-fine-tuning replay, Kullback-Leibler
  divergence anchoring, Gaussian coordinate smoothing, online state-bank
  refresh, new model heads, or inference-time control modules.
- **BREAKING for the new research profile only:** permit a declared
  rollout-calibration run to omit protected full-row base cross-entropy while
  retaining a positive-weight token-type gate over its selected research
  sites. Existing supervised-training profiles and defaults remain unchanged.

## Capabilities

### New Capabilities

- `coordexp-swift-own-prefix-calibration-training`: Defines frozen exact-prefix
  state-bank replay, entity-transition and first-wrong-coordinate objectives,
  rollout-site token-type gating, research-arm configuration, diagnostics, and
  smoke evidence.

### Modified Capabilities

- `coordexp-swift-supervision-losses`: Allows one explicit rollout-calibration
  research profile to omit full-row base cross-entropy while requiring a
  positive-weight token-type gate on every selected site with an intended
  token type; ordinary supervised-training behavior is unchanged.

## Impact

Expected owner surfaces are `src/config/`, `src/data/` or a narrow research
state-bank loader, `src/supervision/`, `src/losses/`,
`src/training/pipeline.py`, and existing training artifact/logging paths.
Inference code is reused for state collection and post-training evaluation but
its public model-forward and decoding behavior do not change. No new external
model or detector dependency is introduced.
