---
title: Object-Specific Geometry Transport, Decision Phase, and Cross-Row Influence Horizon Results
description: Wave One stopped after two valid executions found no admitted common-support donor or grammar contrast; forced coordinate paths remain descriptively grammar-compatible off support.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: stopped_after_wave_one_common_support_failure
updated: 2026-07-17
---

# Object-Specific Geometry Transport, Decision Phase, and Cross-Row Influence Horizon Results

## Verdict

Stop this unit after Wave One. Wave Two decision-phase decomposition and Wave
Three cross-row influence-horizon experiments are not admitted.

Two full-model 32-bit floating-point (`float32`) executions were valid, but
neither supplied the common support required by the frozen causal questions:

1. on image `7818`, the target and paired residual donors strongly preferred
   different early `x1,y1` histories, so no common mediator was admitted; and
2. on image `19432`, only one of four fixed `x1,y1` histories passed the
   predeclared 10-percent support floor, so no real-versus-synthetic grammar
   contrast was admitted.

The image-`19432` forced paths nevertheless form a coherent descriptive
pattern: expected legal `x2` moves monotonically with `x1`, translated width
distributions are much more similar than absolute `x2` distributions, and all
four greedy boxes close naturally. This is off-support extrapolative
compatibility with box-translation grammar. It is not an admitted causal
grammar handle and cannot identify object ownership, background, coverage, or
the absence of a physical visual boundary.

## Evidence Identity

The image-`7818` donor-state receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/
  image7818-donor-state-late-coordinate-float32-20260717a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
923a232fffc2723628255caa61719ac9b33038072c21f00cb4112d745d12f38c
```

The image-`19432` non-boundary grammar receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/
  image19432-nonboundary-box-grammar-float32-20260717a/receipt.json
```

Its `SHA-256` digest is:

```text
14aeb6493247c930bfb101dff54a8cabb22ea7f476985406704584688c580696
```

Both runs used the geometry-sorted Gaussian-coordinate Weight-Decomposed
Low-Rank Adaptation (`DoRA`) adapter at checkpoint step `4887`, full-model
`float32`, Scaled Dot Product Attention (`SDPA`), physical batch size one, and
repetition penalty `1.0`. The second run reproduced the canonical
`1152`-by-`864` image digest, exact 1,441-token pre-`x1` prompt digest, frozen
fixture digest, configuration digest, source-data digest, model identity, and
tokenizer identity. The conclusion-owning runner digest recorded by that
receipt is:

```text
152126b675748b4e175a8f6d879b80bd77c0f7c6516429eef5209b546c7c3bd4
```

## Image 7818: Donor-State Late Transport Is Unidentified

The intended estimand held `x1,y1` byte-identical and changed only the trusted
block-`23` donor state. The two predeclared common histories were each tested
under both donors before any late-coordinate effect was admitted.

| Forced common history | Incompatible donor | Native-minus-candidate log score | Candidate/native probability ratio | Admitted under both donors |
|---|---|---:|---:|---|
| paired `x1,y1` | target donor | `8.154105` | `0.000288` | no |
| target `x1,y1` | paired donor | `4.593848` | `0.010114` | no |

The frozen tolerance was `log(10)=2.302585`. No history passed both donor
branches, so the receipt correctly reports:

```text
no_branch_supported_common_history
```

This is a positivity or common-support failure, not a null donor effect. It
supports only the narrower observation that the two donor states sharply
separate their preferred early-coordinate basins. It does not show that donor
state disappears after `x1,y1`, that early coordinates fully mediate the
state, or that no late object-specific state exists.

## Image 19432: Grammar Pattern Is Coherent but Off Support

Before opening new outcome logits, the dense-chair image was reviewed and the
misleading `object-free` label was retired. The four arms used two real
annotated chair left edges and two coordinates at which no frozen annotated
chair left edge begins. All arms forced the same source-native `y1=123`.

Support used the joint full-vocabulary score of forced `x1` followed by forced
`y1`, relative to the best-supported real edge. The frozen floor was `0.10`.

| Arm | `x1` | Relative history probability | Support admitted | Legal-`x2` mass | Conditional expected `x2` | Conditional expected width | Greedy `x2,y2` |
|---|---:|---:|---|---:|---:|---:|---|
| real target left edge | `537` | `1.000000` | yes | `0.999883` | `672.749939` | `135.749939` | `[664,354]` |
| synthetic inter-edge non-boundary | `608` | `0.072125` | no | `0.999861` | `759.265076` | `151.265076` | `[749,352]` |
| real adjacent left edge | `643` | `0.028363` | no | `0.999770` | `760.624451` | `117.624451` | `[749,354]` |
| synthetic right dense-field non-boundary | `700` | `0.042127` | no | `0.998618` | `819.900391` | `119.900391` | `[829,354]` |

Only one arm is support-admitted; therefore no contrast is identified. The
receipt classification is:

```text
unidentified_control_support_failure
```

The forced outcomes are still useful descriptive evidence:

- all four legal-right-mass gates pass;
- all four bounded greedy suffixes produce `x2,y2,<|box_end|>` naturally;
- real adjacent minus target conditional expected `x2` is `87.874512` bins;
- Kendall rank correlation between `x1` and conditional expected `x2` is
  `1.0`;
- the fitted conditional-expected-`x2` slope is `0.868485`;
- expected-width range is `33.640625` bins;
- median absolute-`x2` Jensen-Shannon divergence is `0.475838`; and
- median translated-width Jensen-Shannon divergence is only `0.101729`.

Every outcome-side translation-grammar gate passes. The conclusion remains
unadmitted because the intervention histories fail the earlier support gate.
The floor cannot be relaxed after seeing this pattern.

## Combined Mechanism Update

The strongest bounded synthesis is:

```text
late donor state and early coordinate history are tightly coupled
  +
forced x1 histories extrapolate into coherent translated x2 distributions
  !=
an identified on-support donor or grammar mechanism
```

This raises confidence that an emitted `x1` can route later box completion,
including away from the most natural current history. It does not determine
whether the on-support native process uses only numeric box grammar, a visual
boundary, a geometry-sorted successor state, or an object-specific carrier.
The current interventions are too branch-selective to separate those
explanations.

The common-support failures are themselves mechanistically informative. A
simple hard mediator clamp assumes the donor and coordinate history can be
crossed while remaining on the model's native manifold. These cases reject
that assumption. Future causal work must either select donor/history pairs for
overlap before opening downstream outcomes, or use a graded on-manifold
intervention. This is a requirement for a new unit, not permission to repair
this completed one post hoc.

## Supported

- Trusted block-`23` donors on image `7818` occupy sharply separated early
  coordinate basins.
- Forced `x1` values on image `19432` produce valid, naturally closing boxes
  and a coherent off-support translation pattern in complete `x2` logits.
- Legal `x2` completion is not coordinate-independent and can remain stable
  under synthetic non-boundary histories.
- Common-support validation is conclusion-critical for these hard-clamp
  experiments; outcome coherence alone is insufficient.

## Not Supported

- No late donor-state transport claim is admitted.
- No on-support box-grammar-specific causal handle is admitted.
- No physical-object owner, object file, background, boundary-independence,
  commit, covered set, cross-row horizon, or terminal completeness claim is
  admitted.
- No decision-phase factorial, cross-row coverage experiment, training loss,
  architecture, slot, ledger, cursor, or detector is promoted.
- The result does not justify replacing the cues, relaxing the support floor,
  or searching new anchors inside this unit.

## Stop-Rule Application

Wave One required at least one admitted donor-state or grammar-specific handle.
It produced neither. Therefore:

1. do not enter Wave Two;
2. do not enter Wave Three;
3. preserve both failed support panels and the off-support curves;
4. close this unit as `stopped_after_wave_one_common_support_failure`; and
5. leave architecture and training unpromoted.

## Verification

The new image-`19432` runner and its focused pure-function checks are:

```text
scripts/research/run_fixed_prefix_nonboundary_box_grammar_image19432.py
tests/analysis/test_fixed_prefix_nonboundary_box_grammar_image19432.py
```

The focused test slice passed `9/9`; Python compilation and `git diff --check`
also passed before graphics-processing-unit execution. A separate read-only
contract audit approved the launch, and a post-run model diagnosis independently
applied the frozen stop rule.
