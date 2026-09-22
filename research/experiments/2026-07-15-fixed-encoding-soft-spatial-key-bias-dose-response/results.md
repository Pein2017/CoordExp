---
title: Fixed-Encoding Soft Spatial-Key Bias Dose-Response Results
description: Four-anchor evidence that finite decoder-wide positive regional key bias produces graded spatial reweighting but no semantic or geometric object-owner reversal.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Soft Spatial-Key Bias Dose-Response Results

## Verdict

Accept the executed artifact and close the tested **uniform decoder-wide
positive soft spatial-key-bias operator**.

The intervention is real, spatially differential, and usually dose responsive.
It changes later phrase and coordinate likelihoods by far more than numerical
no-op drift. However, none of the four valid anchors at any frozen bias value
switches complete-row or geometry ownership. No shared dose produces two
complete cases, and no description-only or geometry-only reversal recurs.

The result therefore weakens the hypothesis that mild context-preserving
attention reweighting is sufficient to release a valid object from full-image
competition. It does **not** show that regional evidence is irrelevant: finite
bias moves several owner margins substantially, but remains below the owner-
basin crossover reached by hard eligibility.

This unit does not authorize free-row replay, phase-specific escalation,
training, a layer sweep, architecture promotion, more doses, or a larger
cohort.

## Conclusion-Owning Artifact

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/
  cohort-four-float32-20260715a/receipt.json
```

Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a
```

The prerequisite hard-eligibility receipt and digest are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/
  cohort-six-float32-20260715b/receipt.json

6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

The conclusion-owning resolved-config, source JSON Lines (`JSONL`), and audit-
ledger digests are respectively:

```text
config: 80aa7fdca59dae988d4dfbe55277bd2c39f11fd9990017a17815eff6b11f9381
source: 9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4
ledger: 52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df
```

The accepted image-`139` real smoke is retained at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/
  smoke-image139-float32-20260715a/receipt.json
```

Its `SHA-256` digest is:

```text
24ee4566b5e8f925baf64cea6fd7368aa5d261e0c78b58bde975c1e14ce68525
```

## Executed Contract and Trust Gates

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Model execution: full Institute of Electrical and Electronics Engineers 754
  32-bit floating point (`float32`), physical batch size one, and Scaled Dot
  Product Attention (`SDPA`).
- Scoring: teacher-forced, no cache, and no repetition penalty.
- Visual execution: one full-image encoding per case; exact primary and all
  DeepStack features were replayed for every arm.
- Position execution: explicit Multimodal Rotary Position Embedding
  (`M-RoPE`) identifiers were reused across arms.
- Intervention: add a positive finite bias of `0.5`, `1.0`, or `2.0` to one
  frozen object-centered image-key set while preserving every other causal
  image and non-image key.
- Scope: the same bias applies across decoder layers, heads, phases, and causal
  queries. It affects image-token-to-image-token as well as later text-token-
  to-image-token reads.

All four cases pass feature continuity, parent-arm continuity, selected-token
rank parity, and the frozen `1e-4` no-op threshold. Maximum no-op drift ranges
from `2.38e-5` to `5.25e-5`. At bias `0.5`, maximum selected-token log-
probability movement ranges from `0.380` to `0.965`, so the null result is not
an execution no-op.

The frozen runner, tests, and pre-execution unit-contract digests recorded for
the conclusion-owning execution are:

```text
runner: 46ec0740cbafb98c60d9302f04c7b7577d9e25c64b06b5d45cf5fac1a0614f00
tests:  505957b5d48ef3b0c666cc24dba1e8bd336b795de56e090c5191c7e586989879
unit:   b08b4826ad5cb30a2de80669488dab11e8c28c391be5bbc20d5b1cf709cd00f3
```

Focused verification passed:

```text
13 passed
```

An independent read-only scientific adjudication recomputed every stored
gamma, crossover, and release field from the raw arm scores with maximum error
zero and found no conclusion-critical receipt or classifier defect.

## Panel Result

For one phase, `gamma target` is target-row minus competitor-row mean log
likelihood under target-region bias. `Gamma competitor` is the same contrast
under competitor-region bias. A complete owner switch requires the former to
be positive and the latter negative for both complete row and geometry.

### Complete-row target ownership across dose

| Image | Target to competitor | Zero-bias gamma | Bias `0.5` gamma target | Bias `1.0` gamma target | Bias `2.0` gamma target | Hard-eligibility gamma target | Case verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| `139` | vase to clock | `-1.675` | `-1.555` | `-1.518` | `-1.473` | `+1.071` | graded movement, no reversal |
| `632` | book to book | `-0.337` | `-0.338` | `-0.339` | `-0.351` | `-0.380` | target stationary; differential suppression only |
| `12120` | person to tennis racket | `-1.025` | `-0.799` | `-0.318` | `-0.066` | `+1.340` | strongest near-threshold movement; no reversal |
| `12639` | person to person | `-1.505` | `-1.447` | `-1.404` | `-1.497` | `-0.011` | movement peaks at `1.0` and regresses |

Geometry target gamma follows the same decisive pattern: it remains negative
for every image and finite dose. The panel classifier reports:

```text
close_uniform_soft_spatial_key_bias
```

Every one of the twelve case-dose combinations is classified as
`inconclusive`; there are zero complete owner switches and zero phase-specific
reversals.

### Dose response anatomy

| Image | Full-row crossover at `0.5 / 1.0 / 2.0` | Target-row release at `0.5 / 1.0 / 2.0` | Interpretation |
|---:|---:|---:|---|
| `139` | `0.195 / 0.308 / 0.494` | `+0.117 / +0.152 / +0.192` | graded target release plus non-owner suppression |
| `632` | `0.032 / 0.061 / 0.123` | `+0.001 / +0.002 / +0.006` | almost purely differential suppression |
| `12120` | `0.193 / 0.614 / 0.614` | `+0.200 / +0.659 / +0.884` | strong geometry-oriented target release, then saturation |
| `12639` | `0.162 / 0.304 / 0.421` | `+0.067 / +0.121 / +0.038` | target release is non-monotonic; competitor separation keeps growing |

The candidate rows share the row-entry token, so row-entry owner gamma is zero
by construction. Absolute row-entry-versus-terminal margins move modestly in
some arms, but the later phrase and geometry effects are larger and case-
specific. The response is therefore not reducible to a generic instruction to
continue one more row.

## Hard Endpoint Is Not a Smooth Finite-Bias Continuation

Images `139` and `12120` reveal the main discontinuity. Moving from bias `2.0`
to hard target eligibility does not improve the target row materially:

| Image | Target score at bias `2.0` to hard | Competitor score at bias `2.0` to hard | Consequence |
|---:|---:|---:|---|
| `139` | `-2.438` to `-2.572` | `-0.965` to `-3.644` | hard reversal is driven by non-owner collapse |
| `12120` | `-1.827` to `-1.825` | `-1.761` to `-3.165` | target is unchanged while non-owner collapses |

The hard operator therefore contributes a nonlinear exclusion or explaining-
away effect that finite positive bias does not reproduce. This also explains
why a large hard crossover cannot be read as evidence that the target row was
robustly recovered.

## Supported

1. Already-computed regional visual evidence causally changes phrase and
   geometry likelihood under a fixed full-image encoding.
2. Finite spatial bias produces graded, dose-dependent regional reweighting.
3. Both target release and non-owner suppression contribute, with their balance
   varying by image.
4. The hard endpoint includes a qualitatively stronger exclusion-driven
   effect, rather than merely the next point on a smooth low-dose curve.

## Weakened or Closed

1. Mild attention dilution is not sufficient to explain the owner-basin
   failures on this frozen panel.
2. One uniform decoder-wide positive regional bias is not a reliable complete-
   row object-binding operator.
3. Adding larger doses or expanding the same operator to more cases is not the
   next information-gaining move.

## Still Unresolved

- Whether a selective redistribution operator can suppress a dominant
  competitor while preserving unrelated global context.
- Whether hard eligibility succeeds because of text-time candidate selection
  or because it also changes image-token-to-image-token computation.
- Whether an object-specific intervention at another causal seam can bind the
  complete phrase and geometry row.
- Population prevalence, free-rollout utility, training learnability, commit,
  coverage, and termination remain untested here.

The object-centered windows are not object-pure, and the visual features were
already globally contextualized before intervention. No result may be promoted
to a pure instance pointer.

## Route Decision

Stop the frozen unit. Do not add doses, tune one value per case, run a free-row
replay, or launch a training screen.

Two independent design reviews selected one
[row-scoring-query-only hard-eligibility
counterfactual](../2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/unit.md)
as the only next routing discriminator. It preserves image-token-to-image-token
and unscored-prefix computation while restricting only the query positions
whose logits predict the canonical row. It tests whether direct row-read
restriction is sufficient. A negative would show dependence on earlier query
computation without uniquely attributing that dependence to image-token state
recompilation.

Balanced target-plus/competitor-minus bias is not next. It adds explicit anti-
competitor force while retaining the same all-query confound and could
manufacture a stronger crossover without identifying the native bottleneck.
