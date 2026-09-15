---
title: Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query Spatial-Key Eligibility Hybrid
description: One-image crossed-region test of whether semantic and geometric row ownership follow earlier visual support, current-row visual support, or agreement between both.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query Spatial-Key Eligibility Hybrid

## Closure

Execution and bounded interpretation are complete. The conclusion-owning
evidence, audit state, limitations, and sole retained later candidate are in
[results.md](results.md). No architecture, training, rollout, or follow-up
implementation is authorized by this closure.

## Question

For image `139`, does object ownership during canonical-row scoring follow the
regional image keys available before row scoring, the regional image keys
available while scoring the row, or agreement between those two query
intervals?

The completed earlier-versus-row factorial found that:

- earlier-query-only restriction favored the scheduled `vase` row under both
  the vase and clock regions;
- row-scoring-query-only restriction carried strong regional geometry; and
- strong cross-category semantic discrimination appeared only when earlier
  and row-scoring restrictions used the same region.

This unit adds only the two missing cross-region cells. It does not split
layers, add training, generate text, or expand the image panel.

## Frozen Evidence and Scope

The conclusion-owning parent is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/
  image139-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
e819216b4b4c012bf55c332c2f6d773a2c694b1440347ade446c024c5a7d3441
```

The parent binds the completed row-scoring-query-only receipt and all-query
hard endpoint with these respective `SHA-256` digests:

```text
4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4
6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

Reuse only image `139`, target annotation `1669970` (`vase`), competitor
annotation `1666628` (`clock`), their exact twenty-key supports, both canonical
rows, prompt, prefix, input identifiers, explicit position identifiers, image
grid, merge size, and frozen full-image visual features.

Execution uses Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter, Institute of
Electrical and Electronics Engineers 754 32-bit floating point (`float32`),
Scaled Dot Product Attention (`SDPA`), teacher-forced scoring, physical batch
size one, no cache, and no repetition penalty. The image is encoded once and
the exact primary and DeepStack features are replayed for every new arm.

## Exact Cross-Region Panel

Let `P` be prefix length and `K` the canonical-row length. Earlier queries are
`0` through `P-2`. Row-scoring queries are `P-1` through `P+K-2`. Query
`P+K-1` is excluded because its logit does not score a row token.

| Earlier-query eligible region | Row-scoring-query eligible region | Arm | Evidence source |
|---|---|---|---|
| vase target | vase target | Target-Earlier and Target-Row matched arm | frozen parent |
| vase target | clock competitor | Target-Earlier and Competitor-Row hybrid | new execution |
| clock competitor | vase target | Competitor-Earlier and Target-Row hybrid | new execution |
| clock competitor | clock competitor | Competitor-Earlier and Competitor-Row matched arm | frozen parent |

The first region name always describes queries before row scoring. The second
always describes queries whose logits score the canonical row. This naming is
the complete operational meaning; no undeclared two-letter arm codes are used.

## Competing Hypotheses and Predictions

For each phase, define owner margin as the mean selected-token log probability
of the vase row minus that of the clock row.

### Earlier-region semantic state

Description should follow the earlier region. Target-Earlier and
Competitor-Row should favor vase description, while Competitor-Earlier and
Target-Row should favor clock description. This supports an object-specific
earlier state only if the preferred owner row is not destructively lowered.

### Direct row-region ownership

Description or geometry should follow the row-scoring region. Both arms whose
row region is vase should favor vase; both arms whose row region is clock
should favor clock. This does not imply that earlier computation is irrelevant.

### Stage-specific semantic-to-geometry handoff

The two hybrid arms should become phrase-geometry chimeras:

- Target-Earlier and Competitor-Row: vase description with clock geometry;
- Competitor-Earlier and Target-Row: clock description with vase geometry.

Opposite, effect-bearing owner margins must occur in description and geometry.
A complete-row average cannot establish this signature by itself.

### Mismatch-sensitive hard-routing compatibility

The matched arms should retain clear ownership while both hybrids lose owner
confidence or become incompatible with both canonical rows. This is bounded
evidence that the hard-routed states require agreement. It cannot distinguish
native coherence-sensitive binding from disruption caused by switching hard
key eligibility at the phase boundary.

### Generic earlier perturbation with row control

Changing the earlier-region label should have little effect at a fixed row
region, while row-region identity should determine the direction. Together
with the completed earlier-only cells, this supports a generic early
perturbation or gain followed by a regional row controller.

## Primary Observation

For each of the four region assignments, both canonical rows, and every
available row phase, record:

- raw selected-token log probabilities, ranks, and top-token identifiers;
- phase sum, mean, and token count;
- vase-row minus clock-row owner margin;
- each row's change relative to unrestricted scoring; and
- each hybrid's change relative to the row-only and matched arm sharing its
  row-scoring region.

Let the unrestricted vase-row and clock-row phase scores be
`S_vase_unrestricted` and `S_clock_unrestricted`. For every hybrid, define the
two primary owner changes only relative to that unrestricted arm:

\[
\Delta_{\text{vase}}
= S_{\text{vase}}^{\text{hybrid}} - S_{\text{vase}}^{\text{unrestricted}},
\]

\[
\Delta_{\text{clock}}
= S_{\text{clock}}^{\text{hybrid}} - S_{\text{clock}}^{\text{unrestricted}}.
\]

These two deltas are the only fields called owner release. Comparisons with a
row-only or matched arm are secondary diagnostics and must use distinct field
names.

Primary phases are:

1. first differing description token;
2. complete description;
3. complete geometry and each of `x1`, `y1`, `x2`, and `y2`; and
4. complete row.

Use the inherited effect floor:

```text
maximum(10 times no-op drift, 0.01)
```

Treat absolute owner margin below `0.10` as weak. Treat a preferred owner row
more than `0.05` natural-log units per token below unrestricted as
destructive. Record positive owner actuation only when its release is at least
`+0.05`. Mechanically:

- constructive vase activation requires owner margin at least `+0.10` and
  `Delta_vase` at least `+0.05`;
- constructive clock activation requires owner margin at most `-0.10` and
  `Delta_clock` at least `+0.05`;
- vase-side suppression requires a vase-favoring margin, `Delta_vase` below
  `+0.05`, and `Delta_clock` at most `-0.05`;
- clock-side suppression requires a clock-favoring margin, `Delta_clock` below
  `+0.05`, and `Delta_vase` at most `-0.05`; and
- if both the preferred owner increases by at least `+0.05` and the alternative
  row falls by at least `0.05`, classify the movement as mixed activation and
  suppression rather than pure activation.

The implementation must test deterministic synthetic examples for vase
activation, clock activation, pure alternative-row suppression, destructive
preferred-owner lowering, and mixed movement.

## Trust Gate

Before behavioral interpretation:

1. validate the frozen factorial, row-scoring-query-only, and hard-parent
   receipt digests before model loading;
2. require exact source, resolved-configuration, audit-ledger, annotation,
   canonical-row, regional-key, image-grid, merge-size, input, explicit-position,
   and visual-feature identities;
3. reproduce all-image-allowed selected-token log probabilities within `1e-4`
   of the frozen factorial receipt with exact selected-token ranks;
4. prove for Target-Earlier and Competitor-Row that the changed earlier-query
   cells exactly equal the frozen target earlier-only cells and the changed
   row-scoring cells exactly equal the frozen competitor row-only cells;
5. prove the exact reverse equality for Competitor-Earlier and Target-Row;
6. require earlier and row changed-cell sets to be disjoint and their union to
   equal the constructed hybrid mask on every scored-query and key cell;
7. require zero changes to non-image keys, off-scope queries, and causal
   future-key blocking;
8. require both regional supports to retain exactly twenty eligible image keys;
   and
9. bind the new runner, imported scorer, normalized argument vector, resolved
   configuration, base model, and adapter identifiers in the receipt.

Any failed gate invalidates the execution. A zero behavioral effect after all
gates pass is valid negative evidence.

## Minimal Implementation and Stop Rule

Create one experiment-local runner and one focused test file. Reuse the
completed factorial loader, feature replay, canonical-row scoring, region
supports, phase aggregation, and frozen receipts. Add only the hybrid mask,
exact structural checks, raw comparison table, and provenance fields required
above. Do not add a shared source module or general-purpose interface.

Run exactly one `float32` image-`139` execution on one graphics processing unit
and stop after one immutable receipt plus independent artifact and scientific
audits. Do not add another image, layer, head, soft transition, same-object
support switch, generation replay, training screen, or architecture module in
this unit.

Logical artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/
  image139-float32-20260715a/
```

If both hybrids collapse, a same-object support-switch or non-hard transition
control may be proposed as a later discriminator. It must not be silently
folded into this execution.

## Not Claimed

This unit cannot distinguish native coherence-sensitive binding from a hard
mask phase-switch artifact. It cannot localize earlier computation to image
tokens, prefix text, a layer, or a head. It cannot establish native instance
binding, commit, coverage, stopping, free-rollout utility, Average Precision,
recall gain, population prevalence, a trainable objective, or a final
architecture.
