---
title: Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test
description: One-case pixel-space erasure test of whether the exact successor transition depends on raw visual evidence inside the committed left-cup bounding box.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test

The unit is complete. See [the verified results and bounded verdict](results.md).

## Decision Question

At the exact Common Objects in Context validation image `12576` **Prefix
State 56**, meaning the legal assistant continuation containing 56 generated
tokens, plus the coherent left-cup description-and-geometry row, does the
right-cup successor depend on raw visual evidence inside the committed
left-cup bounding box after the complete visual tower recomputes its
representation?

The preceding post-vision feature-substitution unit found no detectable
dependence on selected local encoded support. This final discriminator moves
the intervention before visual contextualization. It is an erasure test, not
a mechanism identifier: even a selective effect establishes only that the
target-region pixels contribute causally somewhere through the recomputed
visual path.

## Frozen Three-Condition Panel

All conditions start from lossless in-memory unsigned 8-bit red-green-blue
arrays and use the standard no-resize Qwen3 Vision-Language (`Qwen3-VL`)
image-materialization and full visual forward pass.

1. **Clean Pixel Replay**: pass an exact copy of the recipient pixels through
   the compositor without selecting any pixels.
2. **Target Raw-Bounding-Box Donor Patch**: replace only the recipient pixels
   in the annotated left-cup bounding box with donor pixels at the same
   coordinates.
3. **Equal-Shape Unrelated-Region Donor Patch**: apply the identical operation
   to a manually reviewed, disjoint non-candidate region with the same width,
   height, area, and exposed perimeter.

No resizing, alpha blending, interpolation, cached visual-feature replay,
feature hook, or hidden-state intervention is allowed.

### Target region

The receipt-space left-cup box is:

```text
[251.6757, 360.9369, 391.7838, 623.8559]
```

The frozen half-open integer mask is:

```text
x = [251, 392)
y = [360, 624)
shape = 141 x 264
pixel count = 37,224
```

The operation is exact same-position assignment:

```text
recipient[360:624, 251:392, :]
  <- donor[360:624, 251:392, :]
```

### Equal-shape control region

The manually reviewed control contains wall and door structure rather than a
reportable immediate candidate. It is disjoint from both cups, the target
pizza, and the target region; it does not touch an image boundary.

```text
x = [150, 291)
y = [20, 284)
shape = 141 x 264
pixel count = 37,224
```

```text
recipient[20:284, 150:291, :]
  <- donor[20:284, 150:291, :]
```

The donor-recipient perturbation is stronger in the control region than in
the target region: the pre-run review measured mean absolute red-green-blue
difference of `148.3046` versus `52.4598`, root mean squared difference of
`152.8315` versus `70.7676`, and boundary-seam change of `3.9790 -> 137.3889`
versus `9.7276 -> 77.6309`. A target-selective loss therefore cannot be
attributed to greater target perturbation magnitude or seam discontinuity.

### Why the previous halo is excluded

The previous 86-cell post-vision mask spans approximately `x=[192,448)` and
`y=[320,672)`. In pixel space it changes 88,064 pixels, erases roughly 78
percent of neighboring pizza annotation `1571233`, and changes part of a fork
plus surrounding table context. That halo was a conservative encoded-support
estimate; it is not an object-specific pixel intervention. This unit uses the
raw cup box only.

## Frozen Evidence Identity

- Worktree: `/data/CoordExp/.worktrees/research-probes`.
- Recipient image: Common Objects in Context image `12576`, red-green-blue,
  `864 x 1,152`.
- Donor image: Common Objects in Context image `17436`, the same dimensions
  and color mode.
- Inference configuration:
  `/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml`.
- Checkpoint step: `4,887`.
- Recipient prompt-token hash:
  `11f6c79dcd2b570cf5794238994aa319f1bb23f8a580ffef850dafb6bc59b864`.
- Coherent-row token hash:
  `0e65d82cfbcc12af3c357266848c5f7bc4bf94eab5ba3ea806c953ec073d6355`.
- Maximum generation length: `512` tokens.
- Sampled decoding: temperature `0.4`, top-p nucleus threshold `0.95`,
  repetition penalty `1.0`, and this ordered vector of eight paired seeds:

  ```text
  602711627830374173
  5641984501295450920
  8306462649179848189
  8458627694586881429
  4423663331540486457
  1604446646505700360
  6481035254874347622
  6885534711486411115
  ```

  The receipt must match the ordered vector exactly.
- Greedy decoding, tokenizer, prompt template, parser, first-action
  classifier, model, adapter, and request attribution remain unchanged.

## Primary Estimand and Symptom Taxonomy

`first_action_label` is the classifier output for the first attempted free
action immediately after the frozen prefix. Terminal or malformed first
actions remain terminal or invalid; a later valid row does not replace them in
the estimand. For condition `c`:

```text
R(c) = fraction of the eight paired sampled requests whose first_action_label
       is right_cup
```

The primary contrast is:

```text
Pixel-Support Effect
  = R(equal-shape unrelated-region donor patch)
    - R(target raw-bounding-box donor patch)
```

For every request, retain the paired primitive and classify the first free
action as right cup, left-cup revisit, target-pizza fallback, another supported
object, phrase-geometry chimera, valid row with unmatched geometry, terminal
action, or invalid output. Valid alternatives must not be merged with terminal
or malformed failures.

For a paired seed, define:

```text
target-selective loss:
  clean first_action_label = right_cup
  control first_action_label = right_cup
  target first_action_label != right_cup

reverse-selective loss:
  clean first_action_label = right_cup
  target first_action_label = right_cup
  control first_action_label != right_cup
```

## Predeclared Interpretation

### Positive target-region dependence

Require all of the following:

- clean and control retain the right cup in at least seven of eight requests;
- target patch loses the right cup in at least six of eight requests;
- at least six paired seeds show `clean=right cup`, `control=right cup`, and
  `target!=right cup`;
- zero seeds show the reverse-selective pattern;
- if `T` is the target-selective count, at least `floor(T/2)+1` of those
  failures have labels in exactly `left_cup_revisit`,
  `target_pizza_fallback`, or `another_supported_object`.

Phrase-geometry chimera, valid row with unmatched geometry, terminal, and
invalid labels do not count as supported valid-object alternatives.

Supported statement: raw pixels in the committed-object target region
contribute causally somewhere through the fully recomputed visual path to this
exact successor transition.

This outcome does not establish a visual ledger, committed-object
revalidation, object-specific binding, or a persistent commit state.

### Strong null

Require clean, target, and control each to retain the right cup in at least
seven of eight requests, at most one target-selective loss, at most one
reverse-selective loss, and no asymmetric invalidity collapse.

Supported statement: the complete raw left-cup bounding-box pixels are not
detectably necessary for this exact high-margin successor transition under
same-position donor-patch erasure. This materially strengthens the
text-mediated transition explanation but does not prove visual independence.

### Inconclusive

Apply the verdicts in strict order: first evaluate the positive criteria, then
evaluate the strong-null criteria, and otherwise classify the result as
inconclusive. A trust-gate failure is always inconclusive and prevents causal
interpretation. Examples include clean or control retention below seven of
eight, similar target/control degradation that satisfies neither prior branch,
or an intermediate selective count. Do not tune donor, mask, halo, alpha, or
intervention magnitude after observing the result.

## Claim Boundaries and Confounds

Even a selective target effect remains compatible with several mechanisms:

- the raw cup box removes some overlapping pizza and table pixels;
- full visual re-encoding propagates a local perturbation globally;
- the patch may alter scene compatibility, geometry-sorted traversal, or the
  right-cup representation instead of revalidating the committed cup;
- hard donor boundaries form an out-of-distribution composite;
- target and control donor contents are location-specific rather than
  content-matched;
- exact Prefix State 56 is a high-margin teacher-derived state rather than a
  native own-rollout commitment.

Pixel patching is therefore a strong raw-evidence erasure test but a weak
identifier of the internal commit mechanism.

## Minimum Trust Gate

The compact receipt must establish:

- source image paths and file Secure Hash Algorithm 256-bit hashes;
- exact recipient and donor red-green-blue array hashes;
- target and control half-open bounds, mask hashes, shape, count, and
  disjointness;
- exact equality between each composed selected slice and the corresponding
  donor slice;
- exact equality between recipient and composed pixels outside the selected
  mask;
- float32 pixel-difference summaries for target and control;
- `do_resize=false`, image grid `[1,72,54]`, 972 merged visual tokens,
  processor-output shapes, dtypes, and hashes;
- distinct condition-specific processor hashes;
- one complete visual-tower call per condition and primary plus all three
  DeepStack multi-level visual-feature stream hashes demonstrating fresh
  recomputation;
- standard clean versus empty-mask compositor equality for raw pixels,
  processor tensors, grid, greedy generated tokens, and canonical float32
  score traces;
- one sampled-seed clean no-op token and score-trace parity;
- raw outputs, parser status, first-action label, seed, and condition identity.

A three-condition greedy smoke precedes the eight-seed panel. Any failure in
no-op parity, mask alignment, no-resize materialization, fresh visual
recomputation, or request attribution stops interpretation.

## Execution and Stop Rule

1. Reuse the accepted model loader, exact composed-row builder, no-resize
   image processor, generation backend, request-scoped sampler, parser,
   first-action classifier, and compact artifact writer.
2. Add only an experiment-local unsigned 8-bit red-green-blue compositor and
   the receipt evidence required above.
3. Run focused deterministic tests, then the greedy real-model smoke.
4. After a GPT-5.6-Sol model-family trust audit approves the smoke, run the
   frozen eight-seed panel once.
5. A GPT-5.6-Sol model-family agent independently recounts the immutable
   receipt and owns the scientific verdict.

After this panel, close the committed-object-support branch:

- strong null: retire the branch and study how native rollout creates coherent
  object events and selects the next object;
- selective positive: record only pre-vision target-region causal dependence
  and require one independent exact-prefix replication before method design;
- generic or intermediate outcome: close as inconclusive without mask or donor
  sweeps.

No training, OpenSpec change, architecture promotion, population metric,
attention study, residual replay, or second image is authorized in this unit.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/
  paired-panel-20260715a/receipt.json
```

The receipt Secure Hash Algorithm 256-bit (`SHA-256`) is
`ebe7e20cde77932ac7a41c0b5da9757792703a94312550e8aa813d852f948d5b`.
