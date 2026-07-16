---
title: Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query and Row-Scoring-Query Spatial Bias Results
description: Verified one-image negative result in which the finite soft operator failed its matched-owner gate, closing crossed-arm adjudication for this operator.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query and Row-Scoring-Query Spatial Bias Results

## Verdict

Accept the one-image 32-bit floating-point (`float32`) execution as valid and
informative negative evidence, then close the count-balanced finite soft cross-
region operator.

The matched-control adjudication gate failed in both object directions. Under
Vase-Earlier and Vase-Row, the actual first description token was `clock`, not
`vase`, and complete geometry favored the clock canonical row. Under Clock-
Earlier and Clock-Row, the semantic token remained `clock`, but the preferred
clock geometry owner was materially suppressed relative to unrestricted
scoring. The only allowed scientific classification is therefore:

```text
no_adjudication_close_count_balanced_soft_cross_operator
```

The crossed arms were executed and retained as raw receipt evidence, but they
are not interpreted. This result does not adjudicate whether the earlier hard-
eligibility crossed behavior reflected phase-dependent regional routing or a
hard phase-switch artifact.

## Evidence Boundary

The conclusion-owning receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/
  image139-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
c2a1b65ce7c40402cf19427dafd0c332ce5158d44aa8b0faede1429d9ced2877
```

The panel contains only image `139`, target annotation `1669970` (`vase`),
competitor annotation `1666628` (`clock`), their frozen twenty-key regional
supports, and the two canonical rows. The additive bias was the single
cardinality-derived value `3.9060049057006836` as represented in `float32`.

## Executed Contract and Trust Gates

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: `float32`, physical batch size one, Scaled Dot Product Attention
  (`SDPA`), and no cache.
- Scoring: direct teacher-forced canonical-row forward passes with raw model
  logits, no logits processor, and no repetition penalty.
- Visual state: one fixed full-image encoding with exact primary and DeepStack
  feature replay for every arm.
- Operator: every causally visible key remained readable; selected regional
  image keys received the finite additive bias only in the declared earlier-
  query or row-scoring-query interval.

The execution trust gate passed. Parent receipt, source, configuration,
canonical-row, regional-key, input, explicit-position, and visual-feature
identities matched. Both supports contained exactly twenty keys among `1014`
image keys. Structural mask checks passed, the executed attention masks were
bound to the masks whose changed cells were verified, and runtime attestation
confirmed `SDPA`, `torch.float32`, `use_cache=False`, and direct raw forward
scoring. Frozen parent continuity had zero selected-token log-probability
drift. Maximum all-image-allowed no-operation drift was `5.245208740234375e-5`,
below the `1e-4` ceiling, with exact selected-token ranks.

## Primary Observations

Gamma is the vase canonical-row mean selected-token log probability minus the
clock canonical-row mean selected-token log probability. Owner release is the
named canonical owner's score under the matched soft arm minus its unrestricted
score. Values are natural-log units per token.

| Matched soft arm | Actual first description token | First-token gamma | Semantic owner release | Geometry gamma | Geometry owner release | Matched gate |
|---|---|---:|---:|---:|---:|---|
| Vase-Earlier and Vase-Row | `clock` (token identifier `20666`; expected `vase`, token identifier `85`) | `-2.424568` | vase `+0.389964` | `-2.201999` | vase `+0.170055` | failed |
| Clock-Earlier and Clock-Row | `clock` (token identifier `20666`) | `-6.009457` | clock `+0.369137` | `-1.475941` | clock `-1.023263` | failed |

The vase-matched arm failed two independent requirements: its actual semantic
top token was the competitor token, and its geometry gamma had the wrong sign.
Its positive vase owner releases therefore do not constitute vase ownership.

The clock-matched arm passed its semantic token, semantic-release, and geometry-
sign requirements, but failed non-destructive geometry ownership: the clock
geometry score fell by `1.023263` natural-log units per token relative to
unrestricted scoring. Its clock-favoring geometry margin is therefore
suppression-dependent rather than a valid matched-owner control.

## Supported

- The count-balanced finite soft operator is causally active but cannot recover
  both prespecified matched semantic-plus-geometry owner controls on this
  anchor.
- A correct pairwise owner-margin sign is insufficient when the actual top
  semantic token is wrong or the preferred geometry owner is materially
  suppressed.
- The matched-control gate prevented a crossed-arm story from being inferred
  from an operator that did not preserve its own matched endpoints.

## Ruled Out or Closed

- This single cardinality-derived dose is closed as an adjudicator of the hard
  crossed-region phase interaction.
- The crossed arms cannot support a phrase-geometry chimera, row-region
  control, earlier-region control, mismatch-only effect, or mixed phase-routing
  classification because their prerequisite matched controls failed.
- No dose rerun, additional dose, image expansion, graphics processing unit
  execution, architecture module, or training screen is authorized by this
  result.

## Unresolved

- Whether the hard crossed-region asymmetry depends on hard exclusion, abrupt
  phase switching, or another nonlinear compatibility effect.
- Whether any different all-keys-readable operator can preserve matched owner
  behavior while separating earlier and row-scoring regional contributions.

## Not Claimed

This unit does not establish or falsify native object binding, an object
pointer, causal interpretation of attention magnitude, free-rollout utility,
commit, coverage, stopping, Average Precision, recall gain, population
prevalence, a trainable objective, or a final architecture. It does not
interpret either crossed arm. It does not close the previously observed hard-
eligibility behavior; it closes only this count-balanced finite soft operator
as its adjudicator.

## Rejected Continuation and Stop

Do not pool the historical additive bias values `0`, `0.5`, `1`, and `2` with
the current `3.9060049` point into one dose trajectory. The historical finite-
dose operator applies uniform regional bias over every causally visible query
row. The current operator applies bias only over separately declared earlier-
query and row-scoring-query slices. Shared model, data, visual features, and
unrestricted baseline provenance do not make these intervention operators
exchangeable. A pooled curve would confound dose with query scope.

The unit therefore retains no immediate successor. A future study could
prespecify a fresh `3.9060049` endpoint under the exact historical uniform
operator, but that would be a new execution requiring separate authorization;
it is not authorized here. No model forward pass, graphics processing unit
work, dose rerun, architecture work, or training follows from this closure.
