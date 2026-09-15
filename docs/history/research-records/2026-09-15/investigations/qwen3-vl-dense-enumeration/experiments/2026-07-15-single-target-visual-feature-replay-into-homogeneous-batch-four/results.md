---
title: Single-Target Visual-Feature Replay into Homogeneous Batch Four Results
description: Causal evidence that the exact bfloat16 batch-shape split arises after Qwen3-VL get_image_features rather than in the vision-tower output.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Single-Target Visual-Feature Replay into Homogeneous Batch Four Results

## Verdict

The exact Brain Floating Point 16-bit (`bfloat16`) white-bowl versus orange-bowl
batch split arises after Qwen3-VL's `get_image_features` seam.

The batch-one and homogeneous batch-four primary image features and all three
DeepStack feature streams were elementwise identical. Replaying the exact
batch-one feature bundle into downstream physical batch four changed neither
the cached nor direct coordinate distribution: both remained exactly equal to
Natural Homogeneous Batch Four. The recovered fraction of the batch-one state
was `0.0` on both paths.

This causally closes the vision-tower-output branch for one exact recipient. It
does not yet distinguish image-token scatter, DeepStack consumption, or the
language decoder, and it does not establish population prevalence.

## Conclusion-Owning Artifact

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/
  source-bfloat16-20260715a/receipt.json
```

Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
3a7ebcf8c601aeed29dbcf7aa4420953e6e9a64d2e02ae1c9663e9da6e19a7fc
```

All ten trust checks passed. The natural baselines reproduced the frozen
coordinate-vector hashes from the predecessor, each replay controller applied
exactly once and restored the original method, and replaying the captured
features back into batch one was an exact no-op for cached and direct paths.

## Visual Feature Equality

The image grid contains `972` merged visual tokens per image. Comparing Natural
Single Target with the first image from Natural Homogeneous Batch Four found:

| Stream | Root-mean-square difference | Maximum absolute difference | Relative L2 difference | Exact equality |
|---|---:|---:|---:|---|
| Primary merged image features | `0.0` | `0.0` | `0.0` | yes |
| DeepStack stream 0 | `0.0` | `0.0` | `0.0` | yes |
| DeepStack stream 1 | `0.0` | `0.0` | `0.0` | yes |
| DeepStack stream 2 | `0.0` | `0.0` | `0.0` | yes |

All four images inside the homogeneous batch were also exactly equal on every
primary and DeepStack stream. The batch split therefore cannot be attributed
to a different visual-tower output being computed for batch one and batch four.

## Causal Replay Result

| Arm | Cached first coordinate | Direct first coordinate | Complete-vector role |
|---|---|---|---|
| Natural Single Target | `181` | `181` | White-bowl reference. |
| Natural Homogeneous Batch Four | `406`, all four copies | `406`, all four copies | Orange-bowl reference. |
| Replayed Single Target | `181` | `181` | Exact no-op; both vectors equal Natural Single Target. |
| Single-Target Features Replayed into Homogeneous Batch Four | `406`, all four copies | `406`, all four copies | Both vectors equal Natural Homogeneous Batch Four. |

For natural batch one versus natural batch four, centered coordinate-vector
root-mean-square difference was:

- cached path: `1.166623`;
- direct path: `0.968180`.

After replaying batch-one visual features into batch four, distance to Natural
Homogeneous Batch Four was exactly `0.0` for every reported vector statistic on
both paths. Distance to Natural Single Target remained the complete natural
separation. Therefore:

```text
cached recovery fraction = 0.0
direct recovery fraction = 0.0
cross-path verdict = post-vision ownership
```

## Supported

1. **The batch split is not produced by `get_image_features` or earlier.** The
   executed primary and DeepStack tensors are exactly equal across physical
   batch shapes.
2. **Holding the complete visual feature bundle fixed is causally
   insufficient.** Downstream batch four recreates the full orange-bowl vector
   exactly even when supplied with batch-one visual features.
3. **Cached and direct paths agree on causal ownership.** Their numeric vectors
   differ, but neither recovers any batch-one state under visual-feature replay.
4. **The immediate locus is after the visual feature return seam.** Remaining
   candidates are image-token scatter, construction or consumption of
   DeepStack inputs, multimodal position preparation, and language-decoder
   execution.

## Held or Rejected

Held:

- the first downstream operation or decoder layer at which `bfloat16` batch
  shape becomes numerically different;
- whether the same effect occurs at other recipients or explains any material
  fraction of dense-scene low recall; and
- whether production inference benefits from a precision change.

Rejected for this recipient:

- vision-tower batch-shape instability;
- primary image-feature differences;
- DeepStack feature-source differences; and
- a vision-output state that can transfer the batch-one white-bowl basin into
  downstream batch four.

## Boundary on Interpretation

The supported label is **post-`get_image_features` ownership**, not yet
"language tower root cause." The deterministic scatter and language-model
entry tensors have not been directly attested in this unit. One image and one
coordinate boundary also cannot establish a general detector limitation.

## Independent Arbitration and Next Step

Independent Sol model-diagnosis review found no conclusion-blocking defect and
recomputed the exact feature equality, no-op replay, controller receipts, and
zero recovery on cached and direct paths.

The next unit should not immediately perform a layer sweep on this one numeric
edge case. First run a four-to-eight selected-recipient **Batch-Precision
Prevalence Screen** comparing physical batch one, four homogeneous copies, and
a bounded `float32` control. If the effect recurs, language-model entry and
per-layer onset become decision relevant. If it does not, close this numeric
branch as a local execution confound and return to the enumeration mechanism.
