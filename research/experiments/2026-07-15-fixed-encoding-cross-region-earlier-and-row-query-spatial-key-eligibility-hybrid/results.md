---
title: Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query Spatial-Key Eligibility Hybrid Results
description: One-image crossed-region evidence that earlier and row-scoring visual regions jointly shape the first semantic token, while geometry follows the row-scoring region and only one direction yields a pairwise phrase-geometry chimera.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query Spatial-Key Eligibility Hybrid Results

## Verdict

Accept the one-image 32-bit floating-point (`float32`) cross-region hybrid as
valid and informative, then close the unit without architecture or training
promotion.

Both the earlier-query region and the row-scoring-query region affect the first
semantic decision. The earlier region is directional but insufficient: a
clock-region earlier interval followed by a vase-region row interval makes
`clock` the top first description token, whereas the reverse hybrid makes
`person`, not `vase`, the top token. Geometry-margin sign follows the
row-scoring region in both directions, but constructive owner activation is
asymmetric.

Only Clock-Earlier and Vase-Row forms the prespecified pairwise chimera: the
first description token favors and predicts `clock`, while aggregate geometry
favors the vase row. Vase-Earlier and Clock-Row does not form the symmetric
chimera because its top first description token is `person`; a pairwise
vase-versus-clock margin cannot substitute for the actual top token. Both
hybrids nearly neutralize the complete-row owner margin. The matched hard-
eligibility endpoints remain suppression-heavy, and the hard phase switch
between two twenty-key regions remains an unresolved artifact explanation.

## Evidence Boundary

The conclusion-owning receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/
  image139-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
a50820cbc8dde6d92b964cfe9233e0e897da54e14d817dc48bd79787c8380bec
```

The receipt binds the earlier-query factorial, row-scoring-query-only, and
all-query hard-parent evidence with respective `SHA-256` digests:

```text
e819216b4b4c012bf55c332c2f6d773a2c694b1440347ade446c024c5a7d3441
4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4
6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

The panel contains only image `139`, target annotation `1669970` (`vase`),
competitor annotation `1666628` (`clock`), their frozen twenty-key supports,
and the two canonical rows.

## Executed Contract and Trust Gates

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: `float32`, physical batch size one, and Scaled Dot Product
  Attention (`SDPA`).
- Scoring: teacher-forced canonical rows, no cache, and no repetition penalty.
- Visual state: one fixed full-image encoding with exact primary and DeepStack
  feature replay.
- New interventions: vase-region earlier queries plus clock-region row-scoring
  queries, and the exact reverse assignment.
- Query intervals: earlier queries are `0` through `P-2`; row-scoring queries
  are `P-1` through `P+K-2`, where `P` is prefix length and `K` is canonical-
  row length. The final unscored query is excluded.

The artifact audit approved the execution. All structural receipts passed;
both regional supports contained exactly twenty image keys; the earlier and
row-scoring changed-cell sets were disjoint and matched their independently
constructed masks; no non-image, future, or off-scope causal cells changed.
Input, explicit-position, canonical-row, feature, and frozen-parent identities
passed. Frozen unrestricted baseline drift was exactly zero. Maximum live
no-op drift was `5.2452e-5`, below the `1e-4` ceiling, with exact selected-token
ranks.

## Primary Observations

For each phase, gamma is the vase-row mean selected-token log probability minus
the clock-row mean selected-token log probability. Owner release is the named
canonical row's score under the hybrid minus its unrestricted score. Values are
natural-log units per token.

### First differing description token

This is the primary semantic observation.

| Hybrid | Vase score | Clock score | Gamma | Vase owner release | Clock owner release | Actual top token |
|---|---:|---:|---:|---:|---:|---|
| Clock-Earlier and Vase-Row | `-4.817` | `-0.283` | `-4.534` | `-1.017` | `+0.110` | `clock` |
| Vase-Earlier and Clock-Row | `-2.574` | `-1.399` | `-1.175` | `+1.226` | `-1.006` | `person` |

The first hybrid expresses a clock phrase decision with modest constructive
clock release and vase suppression. In the reverse hybrid, the pairwise margin
still favors the clock canonical token, but neither canonical owner is the top
token. The vase row is released relative to unrestricted scoring while the
clock row is suppressed, showing that the earlier vase region is directional
without being sufficient to make `vase` the semantic decision.

### Geometry and complete row

| Hybrid | Geometry gamma | Vase geometry release | Clock geometry release | Complete-row gamma | Vase complete-row release | Clock complete-row release |
|---|---:|---:|---:|---:|---:|---:|
| Clock-Earlier and Vase-Row | `+0.271` | `+0.395` | `-2.259` | `-0.046` | `+0.135` | `-1.493` |
| Vase-Earlier and Clock-Row | `-0.415` | `+0.021` | `-1.947` | `-0.130` | `+0.135` | `-1.410` |

Geometry sign follows the row-scoring region in both directions. Only the
vase-row direction constructively releases its preferred geometry owner. The
clock-row direction favors clock geometry only because the vase-versus-clock
pairwise balance changes while the preferred clock geometry itself falls by
`-1.947` relative to unrestricted scoring. Both complete-row margins are close
to zero despite large phase-specific changes.

### Complete-description mean is secondary

The complete-description mean gamma is `-2.126` for Clock-Earlier and Vase-Row
and `+0.112` for Vase-Earlier and Clock-Row. These averages are secondary
because the vase canonical description has two tokens while the clock
description has one. In particular, the reverse hybrid's positive mean does
not override its first-token fact: the actual top token is `person`, so the arm
does not instantiate a vase phrase.

### Matched endpoints are suppression-heavy

Under matched Vase-Earlier and Vase-Row hard eligibility, the first-token gamma
is `+5.986`; however, the vase token changes by `-0.681` relative to
unrestricted scoring while the clock token falls by approximately `-10.074`.
Under matched Clock-Earlier and Clock-Row eligibility, gamma is `-8.864`; the
clock token improves by only `+0.386` while the vase token falls by
approximately `-5.070`. The strongest matched semantic margins therefore arise
mostly from alternative-owner destruction, not symmetric constructive owner
activation.

## Supported

- Earlier and row-scoring image-key regions jointly affect the first semantic
  token; neither interval can be treated as irrelevant in this anchor.
- Earlier-region influence is directional but not sufficient to determine the
  semantic owner under a mismatched row-scoring region.
- Aggregate geometry-margin sign follows the row-scoring region in both crossed
  assignments.
- The Clock-Earlier and Vase-Row arm gives bounded pairwise evidence for a
  clock-phrase and vase-geometry chimera.
- Hard regional compatibility is strongly asymmetric and suppression-heavy.

## Ruled Out

- A symmetric phrase-geometry handoff in which semantic owner always follows
  the earlier region and geometry owner always follows the row-scoring region.
- The reverse Vase-Earlier and Clock-Row hybrid as a vase-phrase and clock-
  geometry chimera: its top first description token is `person`.
- A simple constructive interpretation of both geometry reversals: the clock-
  row geometry preference is destructive relative to unrestricted scoring.
- Either hybrid preserving the large complete-row ownership margin of the
  matched endpoints.

## Unresolved

- Whether the asymmetry reflects native phrase-region compatibility or a
  discontinuity manufactured by hard image-key exclusion and the abrupt phase
  switch.
- Whether the earlier influence arises from image-token queries, prefix-text
  queries, or their interaction.
- Whether the crossed behavior generalizes beyond this one selected image,
  checkpoint, pair of categories, and teacher-forced rows.
- Whether an all-keys-readable selective operator can preserve global evidence
  while retaining the directional phase interaction.

## Not Claimed

This unit does not establish a native object pointer, stable object binding,
free-rollout improvement, commit, coverage, stopping, Average Precision,
recall, population prevalence, a trainable loss, or a final architecture. It
does not show that attention weights are explanations. The pairwise chimera is
a teacher-forced selected-token likelihood result under one hard mask, not a
generated row.

## Later Candidate

At most one follow-up is retained: a prespecified finite soft regional
attention-bias cross transition that keeps all image keys readable while
crossing earlier-query and row-scoring-query regional preferences. It would
test whether the directional interaction survives without a hard phase-switch
discontinuity. That unit is not authorized or executed here.
