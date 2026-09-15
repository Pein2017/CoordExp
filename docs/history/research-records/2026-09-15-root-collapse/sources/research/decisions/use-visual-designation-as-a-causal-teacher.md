---
id: decision.visual-designation-causal-teacher
type: decision
status: active
updated: 2026-07-12
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-08-pvci-step2-anticopy-snap-radius-results/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-09-pvci-identity-conflict/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-09-pvci-post-scatter-image-token-equivalence/unit.md
relations:
  supports:
    - decision.let-architecture-emerge-from-hypothesis-gates
  narrows: []
  supersedes: []
---
# Use Visual Designation as a Causal Teacher

## Decision

Use painted marks and their post-scatter equivalents as privileged probes and
teachers for object-conditioned generation. Do not promote pixel painting,
ground-truth boxes, or a particular cursor renderer as the final inference
interface.

## Evidence

- Anti-copy training showed that corrupted same-object marks can become a
  proposal from which the model recovers tighter object geometry.
- Identity-conflict panels showed that a visual mark can dominate the scheduled
  textual target at row level.
- Post-scatter image-token deltas reproduced the earlier image-feature
  intervention, locating a non-pixel entry surface into the language model.

## Belief Update

The useful fact is not that rectangles are a good product interface. It is that
Qwen3-VL has a causally effective visual designation route that can be used to
teach or falsify an internal control mechanism. Claims remain bounded by row,
checkpoint, and intervention scope.

## Next Discriminator

Test whether a clean-image, learned or synthesized internal intervention can
reproduce target-specific full-row phrase and geometry behavior under source
swap and wrong-object controls. Failure retires the cheap internalization route,
not the observed painted-cue mechanism.
