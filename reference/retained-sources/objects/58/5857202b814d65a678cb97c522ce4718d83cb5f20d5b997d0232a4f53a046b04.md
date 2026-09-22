## Why

The first static IoU50-count RLOO update produced verified train-248 gains but
lost annotated-owner counts on the image-disjoint screen while slightly
improving localization geometry.  The smallest train-first successor is to
make higher-IoU quality weakly credit-bearing without allowing it to outrank
IoU50 coverage.

## What Changes

- Extend the existing frozen-plan materializer with one explicit
  `90% IoU50 + 5% IoU60 + 5% IoU80` reward profile.
- Reuse the exact frozen eight-image K4 actions, C anchor, eight-rank one-step
  runner, optimizer dose, language-DoRA surface, adapter-only persistence, and
  natural evaluation path.
- Keep the original binary-IoU50 plan and receipts reproducible.
- Record train-panel and train-248 behavior descriptively; train improvement is
  evidence worth retaining rather than a production promotion gate.
- Exclude resampling, QP projection, PPO/critic/KL, multi-step training, merged
  checkpoints, and reward-weight sweeps.

## Capabilities

### New Capabilities

- `coverage-graded-static-rloo-update`: Canonical construction and one-step
  execution of a coverage-dominant, higher-IoU-aware static RLOO reward while
  preserving the existing binary plan.

### Modified Capabilities

None.

## Impact

This is probe-local.  It minimally extends the current plan and runner, adds
one focused compatibility check and one cold-inference config, and creates a
new research unit.  It adds no dependency or production default and persists
only `base + unmerged shared DoRA`.
