---
title: Fixed-Encoding Conditional Downstream Layer-Output Residual-State Portability Results
description: Verified semantic-only portability result plus an ineligible geometry control showing strong coordinate actuation without donor-owned box geometry.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-fixed-encoding-downstream-residual-state-portability-gate
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Conditional Downstream Layer-Output Residual-State Portability Results

## Verdict

Accept bounded one-sided semantic portability for the eligible image-`139`
clock donor at language-decoder layer `23`. Do not promote geometry portability
or a combined phase-specific semantic-and-geometry claim.

The prespecified image-`632` geometry discriminator executed correctly, but
neither persistent hard-routing donor generated donor-owned geometry. Its only
allowed classification is:

```text
no_eligible_donor_control_only
```

This is a donor-qualification failure, not evidence that a valid geometry
state cannot be portable. It shows that strong coordinate-path release under
regional hard routing can represent coarse spatial or coordinate-mode control
without representing the designated object's tight bounding box.

## Evidence Handles

The accepted image-`139` semantic receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/
  image139-float32-smoke-20260715b/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
cfd576c17df0e3f3bea7286ecf5c6d22d2e5d52a3d91d396caf88b6ce0194e
```

The conclusion-owning image-`632` geometry receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/
  image632-geometry-float32-20260715a/receipt.json
```

Its `SHA-256` digest is:

```text
40e2db7587ad5d169a5b70998b419808bff34f4568235db476a9f3ea7f1662b8
```

The earlier image-`139` `20260715a` receipt is superseded by `20260715b`
because it lacked required provenance and used over-broad labels. It remains
an immutable diagnostic artifact and does not own a conclusion.

## Preregistered-Scope Deviation and Early Close

The original unit named image `12639` and image-`139` layer `20` as later
primary-anchor executions after the first smoke. They were not run. This is an
explicit early-stop deviation from the frozen panel, not a claim that every
preregistered anchor executed.

The deviation does not alter the accepted image-`139` layer-`23` semantic
claim or the image-`632` donor-ineligibility verdict. Image `12639` was frozen
as a one-sided negative anchor and could not establish a positive geometry
teacher. Image-`139` layer `20` could only replicate or weaken the already
accepted layer-`23` semantic result; it could not repair the missing eligible
geometry donor required for a combined phase-specific claim. The unit therefore
stopped without spending additional graphics processing unit work merely to
complete the original table.

## Execution Trust

Both conclusion-owning receipts use the geometry-sorted step-4887
Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter, Institute of Electrical
and Electronics Engineers 754 32-bit floating point (`float32`), Scaled Dot
Product Attention (`SDPA`), repetition penalty `1.0`, and no logits processor.

For image `632`:

- the exact parent receipt `SHA-256`, canonical rows, annotation identifiers,
  regional supports, and mapped row-query-only hard arms matched;
- hard parent reproduction had zero selected-token log-probability drift;
- unrestricted reproduction drift was at most `4.9591e-5`, below `1e-4`, with
  identical selected-token ranks and top-token identifiers;
- dynamic generation restricted only the row-scoring query range, growing
  from queries `1319` through `1323` to `1319` through `1327` as coordinates
  were appended;
- every pre-row query, non-image key, eligible regional image key, and causal
  future-key relation remained unchanged;
- layers `23` and `13` passed separate-forward unrestricted self-state
  no-operation checks with zero drift, identical teacher-forced top-token
  identifiers, and identical generated suffixes; and
- donor caches were discarded, recipients used fresh unrestricted prefill,
  and only one returned full-block layer-output state at the pre-`x1`
  boundary was replaced.

## Image-139 Semantic Result

The clock hard-routing donor was the only eligible semantic donor. Its
persistent realized-description release was `+0.386305` natural-log units per
token. Layer-`23` replacement recovered `+0.380103`, or approximately `98.4%`
of the persistent release, while the layer-`13` negative control recovered
only `+0.109754` and remained below the preregistered half-persistent threshold.

The unrestricted baseline already greedily produced the clock row. The result
therefore establishes donor-conditioned clock-path confidence portability,
not an object-choice switch, universal object identity, geometry transfer,
selection, commit, coverage, or enumeration.

## Image-632 Persistent Geometry Eligibility

Both donors shared the exact token history:

```text
<|object_ref_start|>book<|object_ref_end|><|box_start|>
```

Their persistent row-query-only hard paths closed naturally and had large
coordinate-only release, but failed the absolute donor Intersection over Union
(`IoU`) floor of `0.30`.

| Donor | Generated normalized box | Coordinate-only release | Strictly closer to donor | Donor `IoU` | Eligible |
|---|---|---:|---|---:|---|
| Target book annotation `1661908` | `[0.783, 0.400, 0.811, 0.475]` | `+2.328892` | yes | `0.000` | no |
| Competitor book annotation `1989419` | `[0.746, 0.098, 1.000, 0.176]` | `+2.544239` | yes | `0.018` | no |

The target path was shifted just beyond the narrow target box. The competitor
path was vertically aligned with its tiny book but expanded almost to the
right image edge. Relative proximity therefore reflected coarse location, not
acceptable bounding-box ownership.

## Image-632 Replacement Diagnostics

Because no persistent donor was eligible, no replacement arm could support a
portability claim. The layer-`23` diagnostics independently remained below the
preregistered gate:

| Layer-`23` donor | Coordinate-only replacement release | Half-persistent floor | Donor `IoU` | Geometry ownership |
|---|---:|---:|---:|---|
| Target | `+0.121292` | `1.164446` | `0.000` | failed; output was closer to the paired book |
| Competitor | `+0.240385` | `1.272120` | `0.072` | failed absolute `IoU` floor |

Layer `13` also failed geometry ownership: target release was `+0.071023` with
donor `IoU` `0`, while competitor release was `-0.023184` with donor `IoU`
`0.100`. No corresponding-donor negative-control veto fired.

## Supported

- A one-time layer-`23` returned-state replacement can preserve most of one
  eligible clock donor's description-path confidence after hard routing is
  removed.
- Row-scoring-query-only regional hard routing strongly and reproducibly
  changes the coordinate distribution on image `632`.
- Large coordinate-token likelihood release is not sufficient evidence of
  instance binding, tight localization, or donor-owned geometry.
- The image-`632` failure begins at the first coordinate decision under the
  canonical teacher-forced prefix; later self-generated coordinate exposure
  is not a sufficient explanation.

## Not Supported

- No image-`632` donor qualifies as a geometry-state teacher.
- Geometry-state portability, a whole-row current-object state, and combined
  phase-specific semantic-and-geometry portability are not established.
- The result does not identify a final bridge layer, selector, cursor, slot,
  ledger, coverage field, stopping mechanism, training loss, or architecture.
- It does not improve or evaluate Average Precision, recall, long rollout, or
  dense-scene enumeration.

## Belief Update and Stop

Close layer, dose, attention-head, and transplant expansion inside this unit.
Do not run the prespecified one-sided image-`12639` control or image-`139`
layer-`20` replication: neither can repair the missing positive geometry donor,
and image-`139` already has an accepted layer-`23` semantic result.

The highest-information successor, if authorized as a new unit, is an
eligibility-first geometry panel over a small prespecified cohort. It should
run only persistent row-query hard routing, require natural closure, strict
donor-versus-paired proximity, donor `IoU >= 0.30`, and coordinate-only release
at least `-0.05`, and compare generated geometry with both the object box and
regional-support footprint. Residual replacement should execute only after at
least one donor qualifies. This is a new discriminator, not an extension or a
post hoc search for a passing image.
