---
title: Qwen3-VL Instance Binding Mechanism Temporary Conclusion
date: 2026-04-24
status: temporary-diagnostic
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

# Qwen3-VL Instance Binding Mechanism Temporary Conclusion

## Why This Note Exists

This is a local temporary progress note for the fixed-checkpoint Qwen3-VL
instance-binding mechanism study. It preserves the first-pass conclusion and
sets up the next diagnosis loop before the result is promoted to a canonical
progress finding.

The note is intentionally not a benchmark record. It is a mechanism diagnosis
for one fixed CoordExp checkpoint:

`output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

Primary artifact root:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424`

Implementation worktree:

`/data/CoordExp/.worktrees/qwen3-vl-instance-binding`

## Current Decision

First-pass conclusion:

`converged_first_pass_mixed_soft_pre_x1_coordinate_hardening`

The best-supported mechanism read is the mixed view:

- weak or partial pre-`x1` instance binding exists
- pre-`x1` state is not strong enough to call the instance fully selected in
  difficult same-desc scenes
- `x1/y1` remains the hard local coordinate-basin split
- schema-context tokens are causally important, not inert JSON punctuation

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

Schema-context states are the most suspicious pre-`x1` mechanism site. They are
not merely delimiters that help parse JSON. Under attenuation, they change
target-vs-distractor margins much more than desc or previous-geometry spans.
Under donor copying, they can shift mass toward a same-desc donor while
non-schema spans remain nearly inert.

The current evidence is still not enough to say whether schema-context states
directly carry identity, or whether they are a routing/readout interface that
pulls geometry-bearing information from broader residual state and visual
context. That distinction is the next core diagnosis.

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

Do not overclaim these results yet:

- Donor activation patching can create out-of-distribution residual states.
- The donor copy currently uses the highest competing same-desc candidate, so
  it needs randomized-donor and wrong-image controls.
- The schema-context span is still broad; it includes tokens between desc end
  and pre-`x1`, not a fine-grained quote/key/bracket decomposition.
- The current rollout contrast is only `64` curated cases, not a broad eval.
- The sparse controls verify that the machinery is sane, but the main
  conclusion is about difficult repeated-object scenes.

## Next Diagnosis Plan

Goal:

Separate three possibilities inside the current mixed conclusion:

1. schema-context states directly carry intended instance identity
2. schema-context states are a routing/readout site for geometry-bearing state
3. schema-context donor effects are partly positional or serialization
   disruption artifacts

### Diagnosis 1: Randomized Donor Controls

Run donor patching again on the same `56` donor-eligible cases with additional
donor definitions:

- `same_image_best_competitor`: current donor policy
- `same_image_random_same_desc`: random same-desc non-target donor
- `wrong_image_same_desc`: same desc from another image
- `wrong_image_any_desc`: position-matched object from another image
- `self_noop`: target span copied into itself

Primary metrics:

- donor mass delta
- target mass delta
- changed-to-donor rate
- patched top-is-donor rate
- coordinate-distribution KL from baseline

Decision read:

- If only same-image same-desc donors transfer, schema-context states are likely
  image-grounded and identity-relevant.
- If wrong-image donors transfer similarly, the result may be positional or
  serialization disruption rather than instance binding.
- If self-noop changes margins, the patching mechanism is too intrusive and
  must be repaired before interpretation.

### Diagnosis 2: Fine-Grained Schema Localization

Split `schema_context` into narrower roles:

- desc closing quote
- desc comma / field delimiter
- `bbox_2d` key tokens
- colon after key
- opening bracket
- immediate pre-`x1` token

Run attenuation and donor copying over these spans, with the last several
decoder layers separated instead of bundled.

Decision read:

- A narrow high-effect bracket or pre-`x1` site supports a coordinate-basin
  readout mechanism.
- A broader effect across quote/key/bracket supports schema-context routing.
- A desc-closing-only effect would suggest identity is already being packed
  immediately after semantic description.

### Diagnosis 3: Layer-Wise Causal Boundary

Repeat donor patching for the implicated schema spans across late-layer bands:

- early/middle sanity anchors
- layer `-8` through final layer
- final two layers separately

Primary metrics:

- first layer where donor mass delta becomes non-trivial
- first layer where changed-to-donor events appear
- alignment with probe-layer onset

Decision read:

- If causal donor transfer appears only in very late layers, late language
  layers are probably reading out/routing a partially bound instance.
- If it appears earlier and grows smoothly, identity may be represented before
  the final coordinate readout path.

### Diagnosis 4: Coordinate-Basin Chain

Measure not only pre-`x1` effects but also the immediate chain:

- pre-`x1` distribution
- post-`x1` distribution over `y1`
- post-`y1` distribution over `x2`
- whether donor/schema patching changes the local basin after the first
  coordinate is fixed

Decision read:

- If schema patching changes pre-`x1` but `x1` still dominates downstream
  recovery, the boundary remains coordinate-first.
- If schema patching flips the whole box trajectory before any coordinate is
  emitted, schema-context binding is stronger than the current mixed read.

### Diagnosis 5: Wider Same-Desc Rollout Split

After the causal controls, widen only the rollout-derived contrast, not a full
benchmark sweep:

- target `128..192` repeated/same-desc object sites
- preserve sparse controls
- keep generation batch size `8`
- run 8 tmux shards
- label healthy, duplicate-collapse-like, near-duplicate-like, and wrong-desc
  cases as before

Decision read:

- If the good/bad contrast and donor/schema effects stay stable, promote this
  note to a canonical diagnostic.
- If the effect collapses under randomized controls or wider rollout, keep this
  as a first-pass artifact and revise the mechanism conclusion.

## Immediate Implementation Recommendation

Implement Diagnosis 1 and Diagnosis 2 first. They are the cheapest and most
decisive:

1. Add donor-policy support to the donor-patching stage.
2. Add fine-grained schema span extraction.
3. Run both over the existing `56` donor-eligible cases on 8 GPUs.
4. Merge into a new artifact root:
   `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-core-diagnosis-20260424`
5. Update this progress note or replace it with a canonical diagnostic only
   after the randomized controls are interpreted.
