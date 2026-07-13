---
title: Qwen3 Vision-Language Dense Enumeration Bottleneck
description: Competing mechanisms and belief updates for masked spatial policy, accepted-row prefix policy, and dense-scene enumeration failures.
type: investigation
status: active
topic: qwen3-vl-dense-enumeration
updated: 2026-07-13
---

# Qwen3 Vision-Language Dense Enumeration Bottleneck

## Terminology and Name Registry

- **Qwen3-VL — Qwen3 Vision-Language**: the pretrained multimodal model family
  under investigation.
- **COCO-80 — Common Objects in Context 80-category ontology**: the closed set
  of reportable object categories used by this investigation.
- **STOP — terminal no-more-objects decision**: the model action that ends
  object enumeration; it is not assumed to be a calibrated coverage verifier.
- **Full-Image Single Rollout (`FULL_SINGLE`)**: one sampled autoregressive
  rollout over the complete image using an independent baseline seed that is
  not reused in the multi-call arms.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: `K` independent
  full-image rollouts aggregated with the same merge policy as the spatial
  arms; `K` is the number of spatial cells. It matches total call count, not
  sampled opportunities per object or total computation.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: `K`
  full-size canvases, each exposing one spatial region, decoded from a fresh
  prompt.
- **Full-Canvas Masked Region with Cumulative Accepted-Row Prefix
  (`MASK_CUMULATIVE`)**: the same masked canvases decoded while retaining rows
  admitted and serialized by a frozen prompt-state transition.
- **Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)**: unresized image
  tiles decoded independently.
- **Owning-seed per-object comparison**: compare an object's raw owning-cell
  outcome with the raw full-image outcome using the same seed, thereby matching
  one sampled opportunity without claiming equal input.

## Question

Why does object-level capability remain strong while a single full-image
autoregressive rollout becomes short, conservative, duplicate-prone, or
invalid as scene density and sequence length grow?

The working decomposition is:

```text
static image evidence
  -> choose a spatial scope
  -> identify one reportable object
  -> bind phrase and geometry
  -> emit and commit the row
  -> recover the next uncovered object
  -> stop only when no supported object remains
```

The present investigation does not assume that a ledger, slot, cursor, or
specialized detector is required. It asks which part of this loop first needs
an external intervention.

## First-Principles Mechanism Candidates and Invariants

### Masked-Input Spatial Restriction Utility (`HYPOTHESIS_MASKED_SPATIAL_POLICY`)

Exposing one core-plus-halo region on a full-size masked canvas may recover
objects more reliably than one seed-matched full-image opportunity and may add
final utility beyond matched-call full-image bagging. This is a policy-level
hypothesis: masking changes pixels and visual activations before the vision
tower.

The deeper **Post-Vision Spatial Candidate Competition
(`HYPOTHESIS_POST_VISION_COMPETITION`)** mechanism proposes that simultaneous
object candidates in one fixed full-image representation suppress valid
continuations. The active unit does not test it directly. A positive masked
policy result only authorizes a later same-feature post-vision discriminator.

### Accepted-Row Prefix Policy Interference (`HYPOTHESIS_ACCEPTED_ROW_PREFIX_POLICY`)

The declared cumulative accepted-row prompt policy may harm later rescue
relative to per-region reset. This comparison bundles prompt length, semantic
content, correctness, order, visibility consistency, and state reconstruction;
it estimates a total policy effect rather than pure memory horizon.

The deeper **History-Horizon Instability (`HYPOTHESIS_HISTORY_HORIZON`)**
mechanism claims that length alone harms retrieval under matched content,
correctness, order, visual input, and decode processing. It requires a later
controlled prefix panel.

### Tile-Local Computation Explanation (`HYPOTHESIS_TILE_LOCAL_COMPUTATION`)

Native-scale tiles may help because they contain fewer image tokens or because
the vision tower encodes a local frame differently, not because the language
decoder received a better search scope. Full-size masked-canvas arms are needed
to preserve the nominal full image-token grid while restricting visible
content, but they do not preserve the encoded visual features.

### No-Resize Scale Invariant (`INVARIANT_NO_RESIZE_SCALE`)

Conventional resized tiles improve small-object recognition by changing scale.
This is a protocol invariant rather than a causal hypothesis: the investigation
forbids resizing so that any tile benefit cannot be attributed to object
magnification.

### Partition and Boundary Context (`HYPOTHESIS_CONTEXT_BOUNDARY`)

Hard tiles can cut objects or remove needed context. Core-plus-halo inputs with
unique core ownership separate useful local context from naive union and
non-maximum-suppression gain.

### Geometry-Sorted Traversal Prior (`HYPOTHESIS_GEOMETRY_ORDER`)

The adapter was trained with geometry-sorted rows. Raster cell order can align
with that prior, especially when outputs accumulate in the prefix. Multiple
counterbalanced cumulative orders plus reset-order controls can establish order
sensitivity; causal attribution to training order still requires a
random-order-trained checkpoint.

### Dense-Scene Annotation Gaps (`HYPOTHESIS_ANNOTATION_GAPS`)

COCO annotations are incomplete in dense scenes. Structured decomposition may
recover real COCO-80 objects that official evaluation counts as false
positives. A blinded, manually audited subset is required before interpreting
precision loss as hallucination.

### Residual Recognition or Control-State Limit (`HYPOTHESIS_RESIDUAL_LIMIT`)

Objects that remain missed under matched-call full-image bagging, no-resize tiles,
masked scope, and short history raise the posterior for a genuine recognition
limit or an inability to compile object-specific visual evidence into decoder
control. This unit cannot distinguish those two residual explanations.

## Belief-Update Rules

| Observation | Belief update |
|---|---|
| Owning-seed raw superiority and safe post-merge masked superiority both pass | Supports masked-input spatial-policy utility; raises but does not establish post-vision competition. |
| Owning-seed raw masked advantage is positive, but bagging catches up post-merge | Local restriction helps one opportunity, while repeated full-image opportunities recover comparable final utility. |
| Post-merge masked advantage appears without an owning-seed raw advantage | Aggregation, merge, ownership, or opportunity allocation remains the leading explanation. |
| `MASK_RESET` safely exceeds `MASK_CUMULATIVE` under neutral repetition penalty | Supports harm from the declared cumulative accepted-row prefix policy; does not isolate pure history length or a missing ledger. |
| `TILE_RESET` exceeds `MASK_RESET` while `MASK_RESET` matches `FULL_BAG_K` | Raises fewer-image-token, local-frame, or vision-side explanations. |
| Matched-call bagging matches structured arms, owning-seed differences are equivalent, and mask retention passes | No incremental masked-input utility is detected under this protocol; generic sampling is a sufficient policy comparator but is not thereby proven to be the mechanism. |
| Mask retention fails | A null masked comparison is inconclusive because intervention harm may cancel scope benefit. |
| Official precision falls but blinded manual precision holds | Supports annotation incompleteness rather than hallucination. |
| The same visible objects remain missed by every arm | Raises residual recognition/control-synthesis explanations. |

## Claim Boundary

No tile or masked-canvas result by itself proves prior full-image recognition,
a pure language-side disease, post-vision candidate competition, history-length
failure, or a missing ledger. Both spatial interventions alter visual
computation. Matched call count does not match per-object opportunities. Reset
versus cumulative estimates a total accepted-row prefix-policy effect. A later
same-feature spatial intervention and a separate controlled-prefix panel are
required for stronger localization.

## Current Evidence Update

The completed [Masked Spatial Policy and Accepted-Row Prefix Policy
Disentanglement unit](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md)
executed two independently derived seed roots on the sealed Dense-Union-51
cohort. [Its results](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md)
support a reproducible local input-level spatial-restriction effect in one
seed-matched owning opportunity. They do not show a final masked-policy benefit
over equal-call full-image bagging, and the masked policy fails retention,
mask-harm, manual-precision, and prediction-count safety gates.

The same evidence shows that the complete cumulative accepted-row prefix policy
is harmful relative to reset and that native-scale tiling is worse than
full-canvas masking. Because the cumulative intervention bundles length,
content, correctness, order, and visibility consistency, pure history-horizon
instability remains unresolved. Post-vision competition, a language-only root
cause, and the need for a ledger or architecture change also remain unresolved.

The narrow next discriminator is input-level pixel masking versus spatial
restriction after one fixed full-image visual encoding. A separately controlled
prefix panel should follow; no architecture is promoted from the completed
unit.

## Active Units

| Unit | Status | Purpose |
|---|---|---|
| [Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md) | complete; dual-root evidence verified; [results](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md); architecture not promoted | Localized a reproducible one-opportunity input-mask effect and a harmful cumulative prompt-policy effect while rejecting safe final-policy promotion. |
