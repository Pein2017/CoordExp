---
title: Fixed-Encoding Persistent Hard-Routing Geometry-Donor Eligibility Screen Results
description: Verified six-case result establishing resolution-qualified instance-owned geometry donors for a bounded residual-state portability successor.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen
status: complete
evidence_status: verified
updated: 2026-07-16
---

# Fixed-Encoding Persistent Hard-Routing Geometry-Donor Eligibility Screen Results

## Verdict

The frozen panel passes the preregistered donor gate. Three of four clean
resolution-qualified images contain at least one eligible donor, and images
`7818`, `12576`, and `2157` each produce eligible target and paired-object
paths. The panel classification is:

```text
clean_geometry_donor_found_requires_portability_successor
```

This falsifies the broad claim that persistent row-scoring-query hard routing
can only copy a regional support envelope. On the successful clean cases, the
same description prefix can be routed to two different instances and the
generated boxes are substantially tighter and closer to the designated object
than the eligible-support envelope.

The result does not establish residual-state portability, autonomous object
selection, a reusable controller, or an architecture. It opens only the
prespecified layer-`23` versus layer-`13` residual-portability successor on the
first two eligible donors in frozen cohort order.

## Evidence And Runtime Trust

The merged receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/
  cohort-six-float32-20260715a/merged/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
2046d5784030f255bc039b9252b2e63770e9f9b069219cfa9478a6e94c8f59b2
```

All six images passed the canonical execution trust gate. Every run used
full-model 32-bit floating-point (`float32`) arithmetic, Scaled Dot Product
Attention (`SDPA`), no key-value cache, no logits processor, repetition
penalty `1.0`, and exactly five generated suffix tokens. Implicit, explicit,
and all-image-keys-allowed no-op paths preserved top token identifiers and
ranks within the `1e-4` log-probability tolerance. No image was excluded for a
trust failure.

## Frozen Panel Result

`IoU` below means Intersection over Union with the designated donor. Release
is hard-routing minus unrestricted mean natural-log probability over the four
realized coordinate tokens. Envelope `IoU` is the donor overlap obtained by
the eligible-support envelope itself.

| Image | Stratum | Route | Donor `IoU` | Envelope `IoU` | Coordinate release | Eligible |
|---:|---|---|---:|---:|---:|---|
| `7818` | clean | target | `0.840` | `0.544` | `+0.438` | yes |
| `7818` | clean | paired | `0.807` | `0.519` | `+1.055` | yes |
| `12576` | clean | target | `0.935` | `0.402` | `+5.295` | yes |
| `12576` | clean | paired | `0.847` | `0.553` | `+1.973` | yes |
| `2157` | clean | target | `0.960` | `0.370` | `+1.382` | yes |
| `2157` | clean | paired | `0.944` | `0.553` | `+4.694` | yes |
| `13923` | clean | target | `0.041` | `0.370` | `+2.015` | no |
| `13923` | clean | paired | `0.957` | `0.774` | `-0.060` | no |
| `632` | resolution stress | target | `0.000` | `0.055` | `+2.329` | no |
| `632` | resolution stress | paired | `0.018` | `0.029` | `+2.544` | no |
| `12639` | crowding and resolution stress | target | `0.195` | `0.335` | `+3.061` | no |
| `12639` | crowding and resolution stress | paired | `0.729` | `0.134` | `+2.296` | stress-only |

Every unrelated-location arm closed naturally but failed donor eligibility.
On the clean successful paths, donor ownership also exceeded the strongest
accepted competing object by the required margin. The stress-only image
`12639` result is descriptive and cannot promote the mechanism.

## Interpretation

The result distinguishes three phenomena that were previously conflated:

1. Repeated hard routing can release a coordinate path without owning useful
   geometry, as image `632` demonstrates.
2. At sufficient spatial resolution, the same operator can compile an
   instance-specific tight geometry path rather than merely echoing its coarse
   support, as images `7818`, `12576`, and `2157` demonstrate.
3. High donor `IoU` alone is insufficient. The image-`13923` paired route is
   tight but fails the preregistered non-destructive coordinate-release floor,
   so it is not admitted as a portability teacher.

The strongest remaining discriminator is therefore no longer whether a valid
geometry donor exists. It is whether the donor-owned returned residual state
is a portable causal state or whether tight geometry requires continued access
to the hard-routed visual support.

## Successor Decision

Open one separate bounded unit using only image `7818`, because it is first in
the frozen cohort and supplies two eligible same-description donors. Capture
each donor state at the shared pre-`x1` boundary and test only decoder layer
`23` against the prespecified layer-`13` negative control. Do not sweep layers,
add images, train a bridge, or promote an architecture from this eligibility
result.
