---
title: Fixed-Encoding Persistent Hard-Routing Geometry-State Portability on Image 7818 Results
description: Verified one-way owner-specific x1 geometry-basin portability at decoder layer 23 with a valid layer-13 negative control.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-16
---

# Fixed-Encoding Persistent Hard-Routing Geometry-State Portability on Image 7818 Results

## Verdict

Accept a bounded, one-way, owner-specific `x1` geometry-basin portability
result on image `7818`.

Under the identical prefix

```text
<|object_ref_start|>wine glass<|object_ref_end|><|box_start|>
```

the unrestricted recipient generated the geometry of annotation `664730`.
Replacing exactly one returned residual state at the shared pre-`x1` boundary
after decoder block `23` with the hard-routed paired-object donor state switched
the generated owner to annotation `661523` with Intersection over Union (`IoU`)
`0.800259`. The corresponding decoder-block-`13` donor state passed every
execution and no-operation trust gate but did not switch the owner.

The frozen panel classification is:

```text
bounded_owner_specific_geometry_state_portability
```

Scientifically, the narrower description is more accurate: the positive
teacher-forced effect is concentrated almost entirely at `x1`, after which the
ordinary autoregressive path completes the paired object's box. The result does
not show that one residual vector explicitly stores all four coordinates.

## Evidence And Execution Trust

The conclusion-owning receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/
  image7818-layer23-13-float32-20260716a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
73805bc237276de1595c6d0ea071db047a6498a97b87d1f0d2cdcee9e56e0ec4
```

The run used the geometry-sorted step-`4887` Weight-Decomposed Low-Rank
Adaptation (`DoRA`) adapter, full-model 32-bit floating-point (`float32`)
arithmetic, Scaled Dot Product Attention (`SDPA`), repetition penalty `1.0`, no
logits processor, no-cache teacher-forced scoring, and exactly five cached
continuation tokens.

Both persistent-hard and unrestricted parent paths reproduced their selected
token identifiers, ranks, and per-slot raw log-probabilities with zero drift.
At both decoder blocks, self-state replacement had zero maximum absolute
log-probability drift, identical selected ranks and top tokens, and identical
generated suffixes. Donor key-value caches were discarded. Each recipient used
a fresh unrestricted prefill, exactly one returned-state replacement, and no
later feature replay. The layer-`13` negative control was valid and trusted.

## Frozen Result

Release is replacement minus unrestricted mean natural-log probability on the
same four frozen donor coordinate tokens. Owner and `IoU` come from a separate
five-token free continuation after the one-time replacement.

| Decoder block | Donor | Replacement release | Generated coordinate bins | Generated owner | Donor `IoU` | Portability gate |
|---:|---|---:|---|---:|---:|---|
| `23` | paired annotation `661523` | `+1.084567` | `[519, 371, 644, 739]` | `661523` | `0.800259` | passed |
| `23` | target annotation `664730` | `+0.001565` | `[209, 348, 359, 868]` | `664730` | `0.769340` | failed release floor |
| `13` | paired annotation `661523` | `+0.095656` | `[212, 348, 359, 861]` | `664730` | `0.000000` | failed |
| `13` | target annotation `664730` | `+0.019251` | `[203, 348, 359, 868]` | `664730` | `0.739750` | failed |

The unrestricted baseline owner was annotation `664730`. The layer-`23`
paired-donor arm therefore satisfies the preregistered owner-switch rule rather
than merely strengthening the baseline path. The target direction is not a
second positive: it retained the baseline owner but did not meet the absolute
or half-persistent release floors.

## The Effect Is A First-Coordinate Basin Switch

For the positive layer-`23` paired-donor arm, the four per-coordinate
log-probability changes were:

| Coordinate | Replacement minus unrestricted log-probability |
|---|---:|
| `x1` | `+4.335076` |
| `y1` | `+0.002125` |
| `x2` | `+0.001132` |
| `y2` | `-0.000065` |

Approximately `99.9%` of the summed positive teacher-forced effect is at the
first coordinate. The strongest warranted mechanism description is therefore:

```text
hard-routed visual support
-> late pre-x1 returned residual state
-> first-coordinate and spatial-owner basin switch
-> native autoregressive completion of the box
```

This is stronger than generic coordinate confidence actuation: the intervention
switches between two same-description instances while image, phrase, prefix,
recipient encoding, and decode policy remain fixed. It is weaker than a proof
of a persistent four-coordinate object file: the emitted `x1` can itself route
the later autoregressive decisions.

## Supported

- A hard-routing-induced state after decoder block `23` can causally switch the
  generated geometry owner between two same-description instances on one fixed
  image and prefix.
- The same paired donor after block `13` is insufficient under a valid matched
  control, localizing sufficiency to computation later than that control seam.
- The native decoder can complete a tight paired-object box after the late
  state selects the paired object's first-coordinate basin.
- A late decision bottleneck is a viable explanation for at least this local
  spatial-owner competition event.

## Not Supported

- The result is one image, one pair, one recipient prefix, and one successful
  donor direction. It does not establish population generality or bidirectional
  donor specificity.
- It does not show that decoder block `23` is the unique formation layer; the
  relevant computation may arise anywhere after block `13` and before or at
  block `23`.
- It does not establish phrase identity control, a complete object-state
  representation, clean-image endogenous state synthesis, autonomous
  selection, commit, coverage, stopping, or dense enumeration.
- It does not justify a trainable bridge, slot, cursor, ledger, architecture,
  or Average Precision claim.
- The portable quantity may be horizontal or first-coordinate route state
  rather than an abstract instance identity variable.

## Belief Update And Stop

The previous geometry-donor screen showed that hard post-vision routing can
compile tight instance-owned geometry. This unit now shows that one such
compiled late state can survive removal of the hard visual route and causally
launch the other same-description instance's geometry trajectory.

Close this unit. Do not automatically sweep layers, add images, train a bridge,
or promote an architecture. A separately authorized highest-information
successor would mediate the result through `x1`: compare forcing the paired
`x1` without residual replacement against residual replacement while forcing
the baseline `x1`. That would distinguish a persistent downstream geometry
state from a state whose primary role is selecting the first-coordinate basin.
