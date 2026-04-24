---
title: Qwen3-VL Instance Binding Mechanism Findings
date: 2026-04-24
status: canonical-mechanism-summary
owner: codex
depends_on:
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report/report.md
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report/summary.json
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/binding_probe/summary.json
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/merge_multimodality/summary.json
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/merge_patching/summary.json
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/donor_patching/summary.json
  - output/analysis/qwen3-vl-instance-binding-mechanism-20260424/rollout_failure_split/summary.json
  - output/analysis/qwen3-vl-instance-binding-core-diagnosis-20260424/core_diagnosis/report.md
  - output/analysis/qwen3-vl-instance-binding-core-diagnosis-20260424/core_diagnosis/summary.json
---

# Qwen3-VL Instance Binding Mechanism Findings

## Why This Note Exists

This is the canonical progress note for the fixed-checkpoint Qwen3-VL
instance-binding mechanism study. It preserves the first-pass evidence loop,
the core-diagnosis addendum, and the final closure decision.

The note is intentionally not a benchmark record. It is a mechanism diagnosis
for one fixed CoordExp checkpoint:

`output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

Primary artifact root:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424`

Implementation worktree used for the study:

`/data/CoordExp/.worktrees/qwen3-vl-instance-binding`

Closure state:

- promoted to `main`
- worktree/branch can be removed after merge verification
- no broader benchmark claim is made here

## Current Decision

Final closure conclusion:

`converged_mixed_partial_pre_x1_binding_with_pre_coordinate_readout`

The best-supported mechanism read is the mixed view:

- weak or partial pre-`x1` instance binding exists
- pre-`x1` state is not strong enough to call the instance fully selected in
  difficult same-desc scenes
- `x1/y1` remains the hard local coordinate-basin split
- late schema/pre-coordinate states act as a readout or carrier for partial
  binding, but punctuation/schema tokens are not proven to be the original
  storage site of instance identity

This rejects the strongest H0 form, where there is no meaningful binding before
`x1`. It also rejects the strongest H1 form, where a mostly selected instance is
already cleanly represented before the coordinate burst.

## Evidence

Study subset:

- `64` total cases
- `56` priority repeated/same-desc cases
- `8` sparse single-instance controls
- dominant descs in the mined subset: `person` and `book`
- target ordinal spans `0..12`, so the probe is not only a first-same-desc
  heuristic

Position-wise probe:

- best pre-`x1` probe accuracy: `0.421875`
- best post-`x1` or post-`y1` probe accuracy: `0.59375`

Interpretation:

- instance identity is weakly decodable before coordinates
- the jump after coordinate generation is still large enough to make `x1/y1`
  the practical hardening boundary

Pre-`x1` multimodality:

- `x1` target strict-best rate: `0.546875`
- mean target neighborhood mass: `0.167873`
- mean other-candidate mass: `0.145206`

Interpretation:

- the target often has some pre-`x1` advantage
- the distribution is still close enough across same-desc candidates to remain
  a multi-modal identity posterior rather than a committed instance choice

Causal attenuation:

- schema-context mean absolute margin delta: `0.067651`
- schema-context top-candidate flip rate: `0.375`
- current-desc mean absolute margin delta: `0.001123`
- previous-geometry mean absolute margin delta: `0.001629`

Interpretation:

- the tested schema-context span is causally high-impact for the next `x1`
  distribution
- direct desc content and previous geometry spans are much weaker under this
  attenuation test

Donor activation patching:

- donor-eligible repeated-object cases: `56`
- total donor-patching rows: `224`
- schema-context mean donor-mass delta: `+0.029070`
- schema-context mean target-mass delta: `-0.063707`
- schema-context changed-to-donor rate: `0.196429`
- current-desc mean donor-mass delta: `-0.000059`
- previous-geometry mean donor-mass delta: `-0.000597`

Interpretation:

- copying schema-context hidden states from the highest competing same-desc
  donor pulls mass toward that donor and away from the target
- copying current-desc or previous-geometry spans does not show the same donor
  transfer effect
- this strengthens the claim that schema-context states participate in real
  instance routing or binding

Rollout contrast:

- healthy labels: `38`
- failure-like labels: `26`
- duplicate-collapse-like: `13`
- near-duplicate-like: `9`
- wrong-or-missing target desc: `4`

Interpretation:

- the curated subset contains both good and bad same-desc behavior, so the
  mechanism conclusion is not based only on teacher-forced good cases
- bad-basin proxy rows are enriched for duplicate-collapse-like rollout labels,
  but the proxy is not perfect and should not be treated as a replacement for
  rollout-derived labels

## Mechanism Interpretation

The model seems to carry a soft same-desc instance preference before `x1`, but
the representation is not hard enough to eliminate alternative objects. The
first coordinate token then behaves like a local-basin commitment step: once
`x1/y1` starts, the autoregressive state has much more information about which
object it is continuing.

Schema/pre-coordinate states are the most suspicious pre-`x1` mechanism site,
but the core-diagnosis addendum makes the causal reading narrower than the
first-pass result. Broad schema context is high-impact, yet the strongest
localized site is the bracket / immediate-pre-`x1` slot. Desc-closing quote and
field-delimiter states are nearly inert.

The final read is therefore not "schema punctuation stores identity." The safer
mechanism is that late schema/pre-coordinate residual states act as a readout or
carrier for a partial same-desc instance preference, while `x1/y1` performs the
hard coordinate-basin commitment.

## Core Diagnosis Addendum

Artifact root:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-core-diagnosis-20260424`

Executed addendum:

- `5040` hidden-state patch rows over the same `56` donor-eligible repeated
  object sites
- `98` previous-geometry token-edit rows
- donor policies: same-image best competitor, same-image random same-desc,
  wrong-image same-desc, wrong-image any-desc, and self-noop
- spans: broad schema context, desc closing quote, field delimiter, `bbox_2d`
  key/colon/bracket vicinity, immediate pre-`x1`, current desc, and previous
  `x1/y1`

Chance-normalized probe context:

- pre-`x1`: top1 `0.421875`, mean chance `0.259943`, lift `+0.161932`
- post-`x1`: top1 `0.593750`, mean chance `0.259943`, lift `+0.333807`

Carrier-vs-cause controls:

- same-image best-competitor schema donor delta: `+0.028810`
- same-image random same-desc schema donor delta: `+0.020564`
- same-image best-competitor bbox-open-bracket donor delta: `+0.051062`
- wrong-image same-desc schema target delta: `-0.052666`
- wrong-image any-desc schema target delta: `-0.054127`
- wrong-image same-desc bbox-open-bracket target delta: `-0.114017`
- self-noop schema KL: `0.0`

Fine-grained localization:

- `bbox_open_bracket` and `immediate_pre_x1` are equivalent under the current
  token inventory, so treat them as one site rather than two independent
  confirmations.
- bracket / immediate pre-`x1` is the strongest causal carrier:
  aggregate donor delta `+0.044136`, target delta `-0.088192`, KL `1.255485`.
- broad `schema_context` is still high-impact but weaker:
  aggregate donor delta `+0.010846`, target delta `-0.034504`, KL `0.467019`.
- desc closing quote and field delimiter are nearly inert:
  desc-closing KL `0.000071`, field-delimiter KL `0.000065`, and zero
  flip-to-donor rate.
- previous `x1/y1` hidden-state patch is also near zero:
  aggregate donor delta `+0.000019`, target delta `+0.000279`, KL `0.000154`.

Token-content edit vs hidden-state patch:

- previous `x1/y1` hidden-state patch donor delta: `+0.000005`
- previous `x1/y1` token edit to competitor source delta: `-0.017340`
- previous `x1/y1` token edit target delta: `-0.031328`
- token edit top-candidate flip rate: `0.591837`
- token edit flip-to-source rate: `0.020408`

Updated decision:

The comment's proposed extension was worth doing, and it makes the conclusion
more conservative. Weak/partial pre-`x1` binding still exists, and late
schema-adjacent states can carry that partial binding. But the strongest causal
site is the bracket / immediate-pre-`x1` slot, not desc-ending punctuation or
field delimiters. Same-image same-desc donors can transfer mass toward another
candidate, but wrong-image replacements at the same syntax sites also strongly
disrupt the target distribution. So the safest mechanism read is:

`partial pre-x1 binding exists; late schema/pre-coordinate states act as a
readout or carrier for that partial binding; x1/y1 remains the main hard
commitment step; punctuation/schema tokens should not yet be called the original
storage site of instance identity.`

This supersedes the stronger wording above that schema-context tokens are
"causally important, not inert JSON punctuation" only if read as an origin
claim. They are causally important as late readout/carrier sites; the addendum
does not prove they are where binding is first formed.

## Open Uncertainty

Do not overclaim these results:

- Donor activation patching can create out-of-distribution residual states.
- The core addendum fixed the most important donor-control gap, but wrong-image
  disruption means schema/pre-coordinate effects remain control-sensitive.
- The current rollout contrast is only `64` curated cases, not a broad eval.
- The sparse controls verify that the machinery is sane, but the main
  conclusion is about difficult repeated-object scenes.

## Closure Decision

Close this fixed-checkpoint mechanism loop.

The original research question has enough causal evidence for a decision-facing
mechanism answer. Additional hidden-state patch variants would probably refine
the language but are unlikely to change the conclusion for this checkpoint and
curated subset.

What would reopen the question:

- A fresh same-desc subset shows a qualitatively different pre-`x1` versus
  post-`x1` probe gap.
- Another nearby checkpoint shows strong pre-`x1` binding without a coordinate
  hardening jump.
- Manual review finds that the current high-effect bracket/pre-`x1` patch rows
  are dominated by candidate-label or visual-ambiguity artifacts.
- A broader rollout-derived duplicate-collapse cohort contradicts the
  same-desc hardening pattern.

Recommended next loop, if needed, should be a generality check rather than
another same-surface mechanism probe:

- fresh hard-case subset or manually audited duplicate-collapse cohort
- same minimal probe/patch metrics
- no new benchmark sweep unless the goal changes from mechanism diagnosis to
  checkpoint evaluation
