---
title: Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility Factorial
description: One-anchor two-factor causal test separating earlier-query regional restriction from current-row direct regional reads.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility Factorial

## Question

For the frozen image-`139` vase-versus-clock transition, is the semantic and
complete-row owner switch produced by regional image-key restriction before
the current row, by direct regional reads while scoring the current row, or by
an interaction between both query intervals?

This is the smallest complement to the completed row-scoring-query-only unit.
It adds exactly one missing factorial cell: **Earlier-Query-Only Regional
Eligibility**. Together with the existing unrestricted, row-only, and
all-query arms, it forms a two-factor, two-level factorial over:

1. regional restriction before current-row scoring; and
2. regional restriction during current-row scoring.

## Competing Explanations

### Earlier-state compilation

Regional restriction before the row constructs an object-specific semantic
state that later full-image row queries can use. Earlier-query-only restriction
should recover the vase-versus-clock description and complete-row owner switch.

### Nonlinear earlier-state by row-read interaction

Earlier restriction prepares the semantic state, but restricted current-row
reads are also required to preserve it. Neither isolated arm should recover
the parent phenotype, while their combination should.

### Direct current-row geometry routing

Current-row restricted reads primarily route coordinate evidence. Earlier-only
restriction may recover description identity without reproducing the strong
geometry crossover, while the completed row-only arm retains geometry but not
description ownership.

### Generic regional suppression

Earlier restriction may only destroy nonregional candidates or perturb a
generic row mode. It can increase a margin without releasing the intended
owner row. Owner-row likelihood and target-versus-competitor gamma must
therefore be reported separately.

## Frozen Evidence and Scope

Reuse image `139`, target annotation `1669970` (`vase`), competitor annotation
`1666628` (`clock`), exact canonical rows, object-centered key sets, source,
audit ledger, model, prompt, and full-image feature encoding from:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/
  cohort-four-float32-20260715a/receipt.json
```

Required Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4
```

The all-query hard endpoint remains owned by its immutable parent receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/
  cohort-six-float32-20260715b/receipt.json

6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

Execution uses Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter, Institute of
Electrical and Electronics Engineers 754 32-bit floating point (`float32`),
Scaled Dot Product Attention (`SDPA`), physical batch size one, teacher-forced
row scoring, no cache, and no repetition penalty. The image is encoded once;
the exact primary and DeepStack features are replayed for every newly executed
arm. Explicit Multimodal Rotary Position Embedding (`M-RoPE`) identifiers are
derived from the ordinary two-dimensional attention mask.

No image, object, row, support region, halo, prefix, model parameter, pixel,
resolution, generation policy, or decoding variable may change.

## Exact Factorial

Let `P` be the prefix token count and `K` the canonical row token count.

| Earlier queries `0` through `P-2` restricted | Row-scoring queries `P-1` through `P+K-2` restricted | Arm | Evidence source |
|---|---|---|---|
| no | no | Full-Image Unrestricted | completed receipt |
| no | yes | Row-Scoring-Query-Only Regional Eligibility | completed receipt |
| yes | no | Earlier-Query-Only Regional Eligibility | new execution |
| yes | yes | All-Query Regional Eligibility | immutable hard parent |

The new earlier-only mask must leave every row-scoring query unrestricted over
all causally visible image keys. It must leave non-image keys and causal
future-key blocking unchanged. Target and competitor rows are scored
independently, although their earlier-query range is the same.

Separately for each canonical row and each target or competitor region, the
changed-cell sets of the earlier-only and row-only masks must be disjoint. Over
every mask cell with query index `q=0` through `P+K-2` and every key column,
their union must exactly equal the changed-cell set of the all-query hard
mask. Query `P+K-1` is explicitly excluded because its logit does not score a
canonical-row token. This is a structural equality check, not a behavioral
arm.

## Primary Observation

For complete description, complete geometry, and complete row, report under
target-region and competitor-region eligibility:

- target-row minus competitor-row gamma;
- target-owner and competitor-owner release relative to unrestricted scoring;
- crossover between target-region and competitor-region gamma; and
- owner-row change relative to the all-query parent endpoint.

For each region and phase, compute the descriptive factorial interaction on
owner margin:

\[
I = \gamma_{\text{all-query}}
  - \gamma_{\text{earlier-only}}
  - \gamma_{\text{row-only}}
  + \gamma_{\text{unrestricted}}.
\]

Here `I` is the output-scale difference-in-differences interaction contrast,
not an abbreviation for image. Always show the four raw gamma values beside
it. Also report the corresponding four-cell contrast separately for the
target-row score and competitor-row score so that intended-owner release is
not confused with non-owner destruction. This interaction is descriptive; it
does not by itself localize a mediator or prove that one query interval causes
the effect through the other.

Use the case effect floor inherited from the completed query-only unit:

\[
\operatorname{effect\_floor}
=
\max(10\times\operatorname{no\_op\_drift},\ 0.01).
\]

The result is interpreted by pattern, not promoted by one aggregate label:

- **Earlier-state compilation support**: earlier-only produces description,
  geometry, and complete-row owner reversal without destructive owner-row
  loss.
- **Interaction support**: neither isolated arm produces the complete
  semantic-plus-geometry phenotype, the all-query arm does, and the raw
  difference-in-differences contrasts show the non-additive output change.
- **Geometry-route separation support**: row-only retains geometry while
  earlier-only preferentially restores description identity.
- **Generic suppression support**: margin movement is dominated by non-owner
  loss without intended-owner release.

These are bounded descriptions of image `139`; they are not population
mechanisms.

## Trust Gate

Before interpretation:

1. validate the pinned query-only and hard-parent receipt digests before model
   loading;
2. require exact source, config, audit-ledger, annotation, canonical-row,
   regional-key, image-grid, merge-size, and feature identities; record the
   freshly constructed target and competitor input-token and explicit-position
   identifier hashes, and require the target explicit-position hash to match
   the hard parent receipt;
3. reproduce the unrestricted explicit-position and all-image-allowed no-op
   paths within maximum absolute selected-token log-probability drift `1e-4`,
   with identical selected-token ranks; additionally, for both canonical rows,
   require the new all-image-allowed selected-token log-probability arrays to
   match the completed query-only receipt's `all_allowed_4d` arrays within
   `1e-4`, with exact selected-token rank equality, before mixing factorial
   cells across receipts;
4. emit an exact structural mask receipt proving the changed cells are exactly
   the blocked image-key columns on queries `0` through `P-2`, with zero changed
   row-scoring query cells, non-image cells, or causal future-key cells; and
5. separately for each canonical row and each target or competitor region,
   prove that the earlier-only and row-only changed-cell sets are disjoint and
   that their union equals the all-query changed-cell set for every key column
   on queries `0` through `P+K-2`; explicitly exclude query `P+K-1` from this
   equality because it does not score the row.

A zero behavioral effect after these gates pass is valid negative evidence.
Any trust-gate failure invalidates the run and does not support a mechanism.

## Minimal Implementation and Stop Rule

Create one experiment-local runner and focused test file. Reuse the completed
query-only loader, feature replay, row scoring, phase metrics, frozen receipts,
and image-`139` contract. Add only the earlier-query mask, exact structural
checks, and factorial arithmetic.

Run one real `float32` image-`139` execution on one graphics processing unit.
Stop after one immutable receipt and one independent scientific audit. Do not
add another image, layer, head, bias dose, generation replay, training screen,
architecture module, or attention visualization inside this unit.

Logical artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/
  image139-float32-20260715a/
```

Expected cost is one model load and two newly scored regional arms for each of
the target and competitor canonical rows.

## Not Claimed

This unit cannot identify whether the useful earlier computation belongs to
image-token queries, prefix-text queries, or their interaction. It cannot
establish native instance binding, commit, coverage, stopping, free-rollout
utility, Average Precision, recall gain, population prevalence, a trainable
mechanism, or a final architecture.

## Closure

Execution and bounded interpretation are complete. See the verified
[results](results.md). The architecture remains unpromoted; the next research
seed is the cross-region hybrid of earlier-query and row-scoring-query regional
eligibility recorded in the result.
