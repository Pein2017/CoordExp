---
title: Single-Target Visual-Feature Replay into Homogeneous Batch Four
description: Causal one-recipient test that holds the Qwen3-VL vision output fixed while changing downstream physical batch shape.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Single-Target Visual-Feature Replay into Homogeneous Batch Four

## Question

Does the verified Brain Floating Point 16-bit (`bfloat16`) batch-shape split
already exist in the output of Qwen3-VL's `get_image_features`, or is it created
after that seam by multimodal scatter, DeepStack consumption, or the language
decoder?

The predecessor [batch coordinate-logit invariance
result](../2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
showed that:

- one coherent target recipient selects the white-bowl coordinate distribution;
- four byte-identical coherent target recipients select the orange-bowl
  coordinate distribution;
- semantic neighbors and target batch position are inactive; and
- full-model Institute of Electrical and Electronics Engineers 754 32-bit
  floating-point (`float32`) execution is practically batch invariant.

This unit performs one causal intervention. It does not scan decoder layers.

## Frozen State

Use the same Common Objects in Context image `7574`, source prompt, coherent
target-bowl row, Qwen3-VL 2B base model, step-4887 DoRA adapter, special-token
embedding delta, tokenizer, Scaled Dot-Product Attention (`SDPA`), repetition
penalty `1.0`, and maximum 16 generated tokens as the predecessor.

The target prompt-token Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
d24725c7eca4b74b39043d2e6f52501decef760a0f95c270907745c7d741b351
```

The first five naturally generated tokens must be:

```text
<|object_ref_start|> b owl <|object_ref_end|> <|box_start|>
```

with token identifiers:

```text
[151646, 65, 9605, 151647, 151648]
```

## Causal Operation

Let `F1` be the complete visual feature bundle produced when
`get_image_features` processes the image as one physical request. It contains:

1. the primary merged image-feature stream; and
2. all three DeepStack image-feature streams.

Let `D4` be the downstream computation for four homogeneous target recipients.
The intervention is:

```text
do(visual features supplied to D4 = four exact copies of F1)
```

Pixel values, prompts, image grids, positions, model weights, decoder batch
shape, and generation policy remain unchanged. The existing
`FeatureReplayController` replaces only the return value of
`get_image_features` for one model call and restores the method afterward.

## Arms

Run all arms in one loaded-model process.

| Arm | Vision feature source | Downstream physical batch | Role |
|---|---|---:|---|
| Natural Single Target | live single-target vision forward | 1 | White-bowl reference. |
| Natural Homogeneous Batch Four | live four-image vision forward | 4 | Orange-bowl reference. |
| Replayed Single Target | captured `F1` | 1 | Exact replay no-op control. |
| Single-Target Features Replayed into Homogeneous Batch Four | four exact copies of captured `F1` | 4 | Conclusion-owning causal intervention. |

Execute both ordinary natural cached generation and direct full-prefix scoring.
Natural cached generation owns the primary behavioral comparison. Direct
full-prefix scoring is a matched secondary path because the predecessor found
that cached and direct logits differ numerically even though both exhibit the
batch-shape basin switch.

## Hypotheses

### Hypothesis 1: Vision-output ownership

The batch-dependent coordinate state is already encoded in primary or
DeepStack visual features.

Prediction:

- replaying `F1` into batch four moves the complete coordinate distribution
  toward Natural Single Target, ideally recovering the white-bowl basin.

Falsification:

- the replayed batch-four distribution remains equal or much closer to Natural
  Homogeneous Batch Four.

### Hypothesis 2: Post-vision ownership

The visual feature bundle is not sufficient to transfer the batch split. The
material divergence arises during image-token scatter, DeepStack injection, or
language decoding under a different physical batch shape.

Prediction:

- replaying `F1` into batch four leaves the orange-bowl distribution and the
  broad coordinate-vector shift intact.

Falsification:

- substantial recovery toward Natural Single Target.

### Hypothesis 3: Mixed ownership

Both the vision output and downstream batch-shaped computation contribute.

Prediction:

- replay produces a stable partial movement between the two natural
  references on the complete coordinate vector and white-minus-orange margin.

## Primary Measurements

For cached and direct paths, record:

1. the complete 1,000-coordinate raw-logit vector as Central Processing Unit
   float32 artifact data;
2. coordinate-conditional and full-vocabulary probabilities;
3. selected coordinate, top-coordinate margin, and frozen white-bowl and
   orange-bowl window masses;
4. centered root-mean-square difference, centered maximum absolute difference,
   outside-window root-mean-square difference, and Jensen-Shannon divergence;
5. distance to Natural Single Target and Natural Homogeneous Batch Four;
6. recovery fraction:

```text
1 - distance(replayed batch four, Natural Single Target)
    / distance(Natural Homogeneous Batch Four, Natural Single Target)
```

7. primary and DeepStack feature shapes, dtypes, float32 fingerprints, and
   equality across every repeated homogeneous copy; and
8. controller call count, grid validation, restoration, and replay fingerprints.

A recovery fraction near one supports vision-output ownership; near zero
supports post-vision ownership. Values between them support mixed ownership.
The full vector remains conclusion owning when exact `bfloat16` coordinate
logits are tied.

For the receipt's predeclared label only, recovery of at least `0.9` while
closer to Natural Single Target is labeled vision-output ownership. Recovery of
at most `0.1` while closer to Natural Homogeneous Batch Four is labeled post-
vision ownership. Every other result is labeled mixed or unresolved ownership;
the continuous distances remain the scientific evidence.

## Trust Gates

Stop interpretation if:

- source model, adapter, special-token embedding delta, tokenizer, image,
  prompt, generation policy, attention implementation, or model dtype changes;
- execution is not `bfloat16`, evaluation mode, and inference mode;
- Natural Single Target does not reproduce cached coordinate-logit vector hash
  `0dadea7e1b4b0835946d5b33fef2a28a83eee448ef153acc787d6c6aab6f3c42`;
- Natural Homogeneous Batch Four does not reproduce cached coordinate-logit
  vector hash
  `a7fbe8f169ae67c6b912d9b110f70ecdb71a3392f1da8f1c25aa3b2a61c81fbe`;
- either target does not naturally generate the declared five-token suffix;
- the captured feature bundle does not contain one primary and three DeepStack
  streams with the expected image grid;
- the four live homogeneous primary streams or per-image DeepStack slices are
  not mutually equal;
- Replayed Single Target differs from Natural Single Target on the complete
  coordinate vector;
- a replay controller does not apply exactly once or fails to restore the
  original method; or
- any serialized value is non-finite or cannot reproduce its reported summary.

The causal intervention has no required outcome. A white, orange, or partial
distribution is evidence once all trust gates pass.

## Decision Rules

- If replayed batch four recovers the Natural Single Target vector on cached
  and direct paths, localize the batch split to `get_image_features` or earlier.
- If replayed batch four remains at Natural Homogeneous Batch Four on both
  paths, localize the split after `get_image_features`; only then consider a
  post-scatter replay.
- If replay is intermediate on both paths, conclude mixed vision and downstream
  ownership and quantify the recovered fraction.
- If cached and direct paths produce different ownership verdicts, hold a
  single-source conclusion and treat cached decoder dynamics as an independent
  contributor.

No outcome authorizes training, architecture promotion, a population claim,
or a precision change in production inference.

## Closeout

The unit completed with all trust gates passing. See [results.md](results.md)
for the verified post-`get_image_features` verdict and its interpretation
boundary.

## Minimal Implementation

Add one research-local runner and focused deterministic tests. Reuse:

- the current batch coordinate-logit runner for model loading, prompt
  construction, natural cached/direct scoring, complete coordinate summaries,
  and identity gates; and
- `capture_feature_bundle` plus `FeatureReplayController` from
  `src/analysis/visual_support_counterfactual/intervention.py`.

Do not modify shared inference infrastructure, add a generic hook framework,
or implement a layer sweep.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/
  <immutable-run-identifier>/
```

## Non-Goals

- no pixel intervention, crop, mask, tile, or image resize;
- no primary-versus-DeepStack component ablation in this unit;
- no post-scatter, decoder-layer, attention-head, or residual patch;
- no cached-versus-direct equivalence claim;
- no second image, dense-scene prevalence estimate, or detection metric;
- no training, reinforcement learning, pseudo-labeling, ledger, slot, cursor,
  or final architecture.
