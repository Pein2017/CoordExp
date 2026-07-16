---
title: Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility Factorial Results
description: One-image factorial evidence that direct row reads dominate geometry routing while cross-category semantic discrimination requires matched earlier-state and row-read restriction.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility Factorial Results

## Verdict

Accept the one-image 32-bit floating-point (`float32`) factorial as valid and
close the unit. Direct current-row image-key reads dominate geometry routing
and most complete-row regional discrimination in image `139`. Earlier-query-
only restriction is not a region-specific object compiler: both earlier-only
regional conditions favor the target `vase` row, and changing the earlier-only
region produces only `+0.343` description, `-0.043` geometry, and `+0.024`
complete-row crossover.

The strong cross-category semantic effect appears only when the earlier and
current-row query intervals use the same regional restriction. Under target-
region all-query eligibility, the target description-row score changes by only
`-0.340` relative to unrestricted scoring while the competing `clock` row
falls by `-10.074`. The complete-row changes are `+0.057` for the target and
`-2.689` for the competitor. The matched semantic phenotype is therefore
mainly competitive non-owner suppression, not constructive target-owner
activation.

The bounded mechanism is phase-separated and interactive: direct row reads
carry most geometry routing, while compatible earlier state and row reads are
required for the selected cross-category semantic exclusion. This does not
localize a mediator, identify the earlier-query token type, prove useful native
binding, or justify architecture or training promotion.

## Evidence Boundary

The conclusion-owning receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/
  image139-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
e819216b4b4c012bf55c332c2f6d773a2c694b1440347ade446c024c5a7d3441
```

The receipt binds the completed row-scoring-query-only receipt with digest
`4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4`
and the all-query hard-parent receipt with digest
`6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee`.
It contains exactly image `139`, target annotation `1669970` (`vase`), and
competitor annotation `1666628` (`clock`).

## Executed Contract and Trust Gates

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: `float32`, physical batch size one, and Scaled Dot Product
  Attention (`SDPA`).
- Scoring: teacher-forced canonical rows, no cache, and no repetition penalty.
- Visual state: one fixed full-image encoding with exact primary and DeepStack
  feature replay.
- New factor: target- or competitor-region hard image-key eligibility only for
  queries `0` through `P-2`, where `P` is the prefix token count. Current-row
  scoring queries retain unrestricted image-key access.
- Factorial sources: unrestricted and row-only cells come from the bound
  query-only receipt; all-query cells come from the bound hard parent; only the
  earlier-only cells are newly scored.

All trust gates passed. Cross-receipt unrestricted baselines had zero selected-
token log-probability drift and identical selected-token ranks. Maximum live
no-op drift was `5.2452e-5`, below the frozen `1e-4` ceiling. Feature, canonical-
row, explicit-position, parent-continuity, and structural-mask checks passed.
For each row and region, the earlier-only and row-only changed-cell sets were
disjoint and their union exactly equaled the all-query changed-cell set over
the declared scoring domain. Independent arithmetic recomputation matched the
receipt exactly.

## Observed

For each phase, gamma is the target-row minus competitor-row mean log
likelihood under one regional condition. Values are natural-log units per
token. The four columns are Full-Image Unrestricted, Earlier-Query-Only
Regional Eligibility, Row-Scoring-Query-Only Regional Eligibility, and All-
Query Regional Eligibility. The factorial interaction is:

\[
\text{all-query} - \text{earlier-only} - \text{row-only} + \text{unrestricted}.
\]

It is an output-scale difference-in-differences contrast, not mediator
localization.

### Target-region eligibility

| Phase | Unrestricted gamma | Earlier-only gamma | Row-only gamma | All-query gamma | Factorial interaction |
|---|---:|---:|---:|---:|---:|
| Description | `-1.507` | `+1.482` | `-3.486` | `+8.226` | `+8.723` |
| Geometry | `-2.382` | `-0.143` | `+0.564` | `+0.181` | `-2.623` |
| Complete row | `-1.675` | `+0.250` | `-0.078` | `+1.071` | `-0.776` |

### Competitor-region eligibility

| Phase | Unrestricted gamma | Earlier-only gamma | Row-only gamma | All-query gamma | Factorial interaction |
|---|---:|---:|---:|---:|---:|
| Description | `-1.507` | `+1.139` | `-2.391` | `-4.429` | `-4.684` |
| Geometry | `-2.382` | `-0.099` | `-4.292` | `-1.780` | `+0.229` |
| Complete row | `-1.675` | `+0.226` | `-2.970` | `-1.754` | `-0.685` |

### Regional crossover

Regional crossover is target-region gamma minus competitor-region gamma.

| Phase | Earlier-only crossover | Row-only crossover | All-query crossover |
|---|---:|---:|---:|
| Description | `+0.343` | `-1.095` | `+12.655` |
| Geometry | `-0.043` | `+4.856` | `+1.961` |
| Complete row | `+0.024` | `+2.892` | `+2.826` |

Earlier-only changes the common row preference but barely distinguishes which
region was eligible. Row-only strongly separates geometry and complete-row
ownership. Only matched all-query restriction produces the large cross-
category description crossover.

The all-query target-region owner-score decomposition relative to unrestricted
scoring is:

| Phase | Target-owner change | Competitor-owner change |
|---|---:|---:|
| Description | `-0.340` | `-10.074` |
| Complete row | `+0.057` | `-2.689` |

Thus the `+12.655` semantic crossover is dominated by removal of the non-owner
row. The target description itself does not receive constructive release.

## Supported

- Direct current-row image-key reads are the dominant tested route for
  geometry and most complete-row regional discrimination in this anchor.
- Earlier-query restriction alone changes the common semantic state but does
  not compile a region-specific `vase`-versus-`clock` identity.
- The matched all-query semantic phenotype is compatible with an earlier-state
  by current-row-read interaction and competitive exclusion of an incompatible
  non-owner row.
- The mechanism is phase-separated rather than a uniform regional gain: the
  largest semantic non-additivity and the geometry-routing effect occupy
  different factorial patterns.

## Ruled Out

- Earlier-query-only regional eligibility is not sufficient for region-
  specific identity compilation in image `139`.
- Row-scoring-query-only eligibility is not sufficient for the complete
  cross-category semantic-plus-geometry phenotype.
- The factorial does not support one uniform synergy shared by description,
  geometry, and complete-row phases.
- The result does not justify another bias dose, an architecture module, or a
  training screen.

## Unresolved

- Whether the necessary earlier computation resides in image-token queries,
  prefix-text queries, or their interaction.
- Whether the strong non-owner suppression reflects useful phrase-region
  compatibility or an artifact of hard key exclusion.
- Whether the phase-separated pattern generalizes beyond this selected image,
  checkpoint, and teacher-forced row pair.
- Free rollout, autonomous selection, commit, coverage, stopping, and detection
  utility.

## Not Claimed

This one-image factorial does not establish a native object pointer, a causal
mediator, attention as explanation, population prevalence, Average Precision,
recall improvement, a trainable bridge, or a final architecture. The
factorial interaction is descriptive output-scale non-additivity only.

## Receipt Limitation

The top-level runner file and normalized command-line arguments are bound by
the receipt. Imported scorer files and every resolved imported path are not
fully self-contained in that binding. The accepted raw scores, frozen receipt
links, feature fingerprints, structural masks, and trust gates are sufficient
for this bounded verdict. This is a provenance-hardening limitation, not a
reason to rerun the completed unit.

## Next Discriminator

Before any architecture or training work, run one cross-region hybrid
factorial on image `139`:

1. earlier queries restricted to the `vase` region while row-scoring queries
   are restricted to the `clock` region; and
2. earlier queries restricted to the `clock` region while row-scoring queries
   are restricted to the `vase` region.

The discriminator is whether description ownership follows the earlier region,
the current-row read region, only matched region pairs, or a directional
compatibility relation. Reuse the exact checkpoint, canonical rows, fixed
encoding, masks, `float32` scoring, and trust gates. This is an experiment seed,
not Graphics Processing Unit (`GPU`) launch or implementation authorization.
