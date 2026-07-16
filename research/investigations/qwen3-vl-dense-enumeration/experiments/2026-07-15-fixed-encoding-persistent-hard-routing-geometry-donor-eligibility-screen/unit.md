---
title: Fixed-Encoding Persistent Hard-Routing Geometry-Donor Eligibility Screen
description: Eligibility-first screen of whether persistent row-scoring-query hard image-key routing can produce tight instance-owned geometry on resolution-qualified same-description pairs.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-16
---

# Fixed-Encoding Persistent Hard-Routing Geometry-Donor Eligibility Screen

Execution is complete. See [results.md](results.md) for the bounded verdict and
the frozen residual-portability successor decision.

## Question

On a frozen panel containing four resolution-qualified same-description object
pairs and two resolution or crowding stress controls, does the already verified
row-scoring-query-only hard image-key eligibility operator produce at least one
valid, tight, instance-owned geometry path?

This is an eligibility screen for a possible donor state. It is not a residual
transport experiment and it does not propose a final architecture.

## Motivation And Competing Mechanisms

The prior image-`632` experiment established strong coordinate actuation but no
eligible book-owned geometry: persistent hard routing released the generated
coordinate path while donor Intersection over Union (`IoU`) remained `0.000`
or `0.018`. Both books were narrower than one merged visual-grid cell, so that
single failure cannot distinguish two mechanisms:

1. **Instance-owned geometry compilation**: repeated object-region routing can
   compile a tight box state when the designated object is spatially resolved.
2. **Support-envelope actuation**: repeated hard restriction mainly
   renormalizes visual evidence or copies a coarse eligible-support footprint;
   it need not compile the designated object's boundaries.

The screen resolves only this operator-specific uncertainty. Zero qualifying
clean donors means that this exact persistent hard-routing operator is not a
qualified geometry teacher. It does not mean that Qwen3-VL cannot form an
instance-owned geometry state through another native or oracle route.

## Frozen Cohort

The exact machine-readable cohort is [cohort.json](cohort.json). It is a
curated mechanism panel, not a random sample or a prevalence estimator. It was
frozen from Common Objects in Context (`COCO`) annotations, actual Qwen
processor grid metadata, and manual visibility review before any new hard-
routing output was inspected. Its externally bound Secure Hash Algorithm
256-bit (`SHA-256`) digest is:

```text
1f7d44bf8291f7b3e708372f97ed51ac85eebef3b97fa3d83456adbe7f1060bf
```

The runner must embed this digest and refuse any cohort mutation. The panel can
show that an eligible donor exists and can provide a bounded operational stop
for this research line. A zero result is not a population-level estimate of
how often the operator might work.

The conclusion-owning stratum is:

| Frozen order | Image | Identical description | Target annotation | Paired annotation | Why resolution-qualified |
|---:|---:|---|---:|---:|---|
| 1 | `7818` | wine glass | `664730` | `661523` | Two large, separated foreground glasses. |
| 2 | `12576` | cup | `678023` | `678923` | Two large, separated foreground cups. |
| 3 | `2157` | knife | `696182` | `696112` | Two large, separated knives. |
| 4 | `13923` | chair | `109275` | `108331` | Two visible, separated chairs; all-object attribution handles nearby chairs. |

Each clean pair spans at least two merged cells on each axis, has pairwise box
`IoU` at most `0.10`, and has center separation of at least two merged cells.

The descriptive-only stress stratum is:

| Frozen order | Image | Identical description | Target annotation | Paired annotation | Stress reason |
|---:|---:|---|---:|---:|---|
| 5 | `632` | book | `1661908` | `1989419` | One or both books are below one merged-cell width. |
| 6 | `12639` | person | `543629` | `1215138` | Crowded scene; the paired person is below two merged cells in width. |

Stress cases cannot promote the mechanism or overturn a clean-stratum result.
They calibrate expected failure under granularity and crowding.

## Intervention Contract

For each image, encode the unmodified full image exactly once. Keep the image
tokens, visual feature streams, position identifiers, model weights, prompt,
and canonical row prefix fixed. Beginning at the existing `BOX_START` token,
greedily generate exactly four coordinate tokens followed by `BOX_END`.

At every partial-row step, the complete row-scoring-query range may attend only
to one frozen subset of already-computed image-token keys. Pre-row queries and
all non-image keys retain ordinary causal access. Run three equal-area arms:

1. target-object support;
2. paired-object support, created by exact translation of the target support;
3. unrelated-location support, created by a prespecified disjoint translation.

The third arm is deliberately named **unrelated location**, not background.
Some images contain unannotated or unrelated objects at that location. Its
purpose is to test routing specificity, not background recognition.

The run must use full-model 32-bit floating-point (`float32`) arithmetic,
Scaled Dot Product Attention (`SDPA`), repetition penalty `1.0`, no logits
processor, no key-value cache, and complete recomputation after every generated
token. The full-image visual encoding must remain byte-identical across arms.

## Trust Gates

Before interpreting an image:

- verify the frozen config, source JavaScript Object Notation Lines (`JSONL`),
  audit-ledger, and externally bound cohort `SHA-256` digests;
- verify actual processor grid metadata and merged-grid dimensions against the
  frozen cohort;
- verify the target, paired, and unrelated supports have equal token counts and
  the declared pairwise disjointness;
- verify that only declared row queries lose declared image-key access;
- reproduce the unrestricted canonical row under implicit, explicit, and
  all-image-keys-allowed execution with identical top tokens and selected-token
  ranks and maximum absolute log-probability drift at most `1e-4`;
- require every generated suffix to contain exactly four coordinate tokens and
  a natural final `BOX_END`.

A failed gate invalidates that image; it is not a negative mechanism result.

## Attribution And Donor Eligibility

Decode the four coordinate bins into normalized `xyxy` geometry. Attribute
each generated box against:

- the designated donor annotation;
- the paired annotation;
- every other object whose frozen audit-ledger `final_state` is exactly
  `accepted`; `crowd` entries are excluded from this competing-object gate;
- the eligible-support envelope in normalized image coordinates.

For a target or paired route to become an eligible donor, all conditions must
hold:

1. exact valid suffix and natural closure;
2. normalized coordinate Manhattan distance to the donor is strictly lower
   than distance to the paired object;
3. donor `IoU` is at least `0.30`;
4. donor `IoU` minus the highest competing accepted-object `IoU` is at least
   `0.15`;
5. the generated coordinate path's mean teacher-forced log-probability under
   hard routing minus unrestricted execution is at least `-0.05` natural-log
   units per coordinate;
6. donor `IoU` minus the unrelated-location route's `IoU` to that same donor is
   at least `0.15`.

Two additional conditions prevent an exact or lightly perturbed support
envelope from masquerading as tight object geometry:

7. donor `IoU` must exceed the eligible-support envelope's `IoU` to that donor
   by at least `0.05`; and
8. the generated box's normalized coordinate Manhattan distance to the donor
   must be at most `75%` of the eligible-support envelope's distance to the
   donor.

The exact support envelope must fail eligibility in a synthetic contract test,
while the exact donor box must pass these two geometry-refinement conditions.
If generated geometry systematically follows the support envelope rather than
the donor, classify the symptom as support-envelope actuation even if
coordinate release is large.

## Decision Rules And Stop Conditions

- **Zero eligible clean donors**: classify
  `no_resolution_qualified_geometry_donor_in_curated_panel`; close this hard-
  routing geometry-teacher line operationally. Do not add images, layers, bias
  doses, attention heads, or transplant variants. Do not interpret this as a
  population-level or model-capability impossibility result.
- **One or more eligible clean donors**: freeze the first at most two eligible
  donors in cohort order and open a separate residual-portability successor.
  Do not execute residual transplantation inside this screen.
- **Stress-only eligibility**: report descriptively; do not promote or open the
  portability successor.
- **Correct and unrelated locations act similarly**: classify generic regional
  restriction or renormalization rather than object-specific routing.
- **Generated geometry belongs to a third object**: classify selection
  ambiguity, not donor ownership.

If a successor is opened, it may test only the frozen late layer `23` and the
prespecified early negative-control layer `13`, on the first at most two clean
donors. A separate oracle painted or post-scatter visual-delta donor may be
considered later only as a new research unit; it is not an automatic fallback.

## Minimal Implementation And Cost

Create one experiment-local runner under `scripts/research/`. Reuse the
existing model loader, fixed-encoding feature replay, query-scoped hard-mask,
coordinate parser, scoring, and box-attribution helpers. Generalize only the
hardcoded image-`632` assumptions that prevent the frozen six-case panel.

Run one clean case as a real smoke, then place the remaining cases on separate
graphics processing units. This is a bounded inference probe, not training.
The expected cost is six image encodings and eighteen five-token generation
arms plus teacher-forced scoring.

Artifacts resolve under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/
  <run-id>/
```
