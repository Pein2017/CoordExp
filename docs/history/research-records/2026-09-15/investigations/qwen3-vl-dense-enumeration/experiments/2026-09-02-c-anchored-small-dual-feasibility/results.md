---
title: C-Anchored Small-Dual Local Feasibility Result
description: The evidence-linked C-prefix gain direction improves all three surviving lesion margins to first order, so active-set projection is unnecessary on the frozen support.
status: complete; mechanically valid; GO_C_ANCHORED_SIMPLE_REFRESHED_VERTICAL; production held
---

# C-Anchored Small-Dual Local Feasibility Result

## Outcome

The authoritative zero-update probe returns
`GO_C_ANCHORED_SIMPLE_REFRESHED_VERTICAL`.

The equal-image complete-row gain direction at the current C prefixes has
strictly positive directional change on all three surviving, behavior-linked
incumbent decisions.  It is already inside the registered preservation cone:
the small dual has no active constraint, the projected direction equals the
unconstrained direction, and projection has no demonstrated role on this
support.

This is a local first-order result.  It authorizes only a separately contracted
one-update simple refreshed vertical slice.  It does not authorize projected
training, a COCO run, screen-2 promotion, architecture promotion, or
production.  `HOLD_PRODUCTION` remains active.

## Immutable evidence

- plan:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-small-dual-feasibility/plan-v1.json`,
  SHA-256
  `f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`;
- non-authoritative mechanics sentinel:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-small-dual-feasibility/sentinel-v1/receipt.json`,
  SHA-256
  `393ed07404761488a496304160a80315cff7992bd0f63a34e78d5cfec06084c5`;
- authoritative receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-small-dual-feasibility/authoritative-v1/receipt.json`,
  SHA-256
  `7d88edc9e47a10ae47ea6d9f5cd54a8ab88964836c50b6185057f83c8dbcba80`;
- live runner commit: `fc8f8aff8bb861d0b15093233182740c1e1b4edb`;
- C adapter fingerprint:
  `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`;
- inherited step-2444 embedding-delta fingerprint:
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`;
- resolved C inference-config fingerprint:
  `5fc24774d3c9fec9d832ffc8a6667ffb8e1d2525b11dd59d170ca88d74931eed`.

The reference remained the immutable universal base plus one shared unmerged C
DoRA adapter.  No adapter was merged or exported.

## Frozen support

The CPU plan reproduced the registered evidence funnel:

| Stage | Count |
|---|---:|
| D0-over-Source positive-target uptake owners | 63 |
| still missed naturally by C | 20 |
| C-prefix reachability class 1 / 2 / 3 | 1 / 12 / 7 |
| admitted class-1/2 complete-row candidates | 13 |
| one-per-image gain actions after latest-boundary selection | 11 |
| Phase-A behavior-linked lesions | 5 |
| lesions still realized by C | 3 |
| frozen weakest-decision constraints, including ties | 3 |

There were no weakest-position or runner-up ties within `1e-7`; each surviving
incumbent contributed one coordinate-token decision.  The other two historical
lesions were absent from C and were not replaced by newly low-margin owners.

The eleven field-balanced gain losses have mean `1.731361` and range
`[1.245909, 2.671331]`.  Their mean coordinate component is `3.251849`, versus
`0.210873` for schema plus description.  This is descriptive of the frozen
gain objective, not a natural-policy improvement measurement.

## Direction and incumbent results

The full 588-tensor, 18,006,016-scalar shared language-DoRA gain gradient has
Euclidean norm `4.487137584389178`.  For the normalized unconstrained gain
direction `d0`, the table reports the dot product with each normalized
incumbent gradient:

| C incumbent owner | anchor raw-logit margin | normalized directional change | threatened? |
|---|---:|---:|---|
| `101636:coco_ann:1760455` | 0.02978325 | +0.05697910 | no |
| `347671:coco_ann:1799145` | 0.08687401 | +0.00593656 | no |
| `359310:coco_ann:1172698` | 0.03891754 | +0.00883489 | no |

All changes exceed the frozen `1e-9` threat tolerance.  Consequently:

- threatened constraints: `0 / 3`;
- active constraints: `0 / 3`;
- projected-direction norm: `0.999999999999874`;
- predicted-gain fraction retained: `0.9999999999997153`;
- minimum projected constraint slack: `0.005936557290548926`;
- positive-row rescaling direction difference: `0.0`;
- cone-projection lemma residual: `1.47e-13`;
- primal-dual gap, stationarity, complementarity, and solver residual: `0.0`.

The operative falsification is therefore simple: the known D0 preservation
lesions do not imply a local conflict between refreshed C-prefix positive-row
learning and those same incumbents at the C anchor.

## Mechanical validity and bounded repair

The first sentinel attempt stopped before any forward pass because the HF
composition left only
`model.language_model.embed_tokens.shared_embed_delta` marked
`requires_grad=true`.  It produced no scientific receipt, gradient, update, or
checkpoint.  The runner was repaired to freeze every loaded parameter first
and then re-enable exactly the 588 named DoRA tensors.  A fresh sentinel and
the authoritative run then passed from the unchanged plan and model artifacts.

The authoritative execution recorded:

- 14 graph forwards, 11 gain backwards, and 3 incumbent VJPs;
- adapter state SHA-256 identical before and after;
- parameter updates `0`, optimizer constructed `false`, checkpoints `0`, DDP
  collectives `0`;
- peak CUDA allocated/reserved `24.73 / 26.31 GiB`;
- peak host RSS `11.29 GiB`;
- live elapsed time `24.53 s`.

The residual embedding-delta `requires_grad` flag is an inference-composition
implementation detail exposed by graph mode; the explicit frozen-surface gate
prevents it from entering this result.  This unit does not change the shared
inference loader.

## Interpretation and next boundary

The active-set hypothesis has failed its cheapest discriminator: the only
evidence-supported C incumbents are not threatened by the refreshed gain
direction.  Adding a dual projection, constraint bank, or distributed solver
to the next vertical slice would therefore be mechanism without evidence.

The next permissible research unit is the simpler one-update C-anchored
refreshed-positive vertical slice.  It must separately freeze the optimizer or
step-size semantics and test finite unmerged readback plus natural greedy
gain/preservation.  Until that finite bridge exists, no claim is made about
actual owner uptake, finite-step retention, held-out transfer, missing-label
behavior, DDP scalability, or COCO-scale learning.
