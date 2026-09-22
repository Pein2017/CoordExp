# Val14439 Layer-Transition Residual Patch

Date: 2026-06-27

Scope: single-sample causal residual replacement panel for val image `14439`, receiver forced x1 `729`, target y1 `38`, contrast/lower-basin coordinate `385`. This continues the x1 basin-key finding and tests whether replacing the receiver state with earlier or donor states can prevent the layer-27 lower-y overwrite.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

Base row source:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_basin_key_sweep_route/v1_dry_run/coord_basin_key_route_plan_rows.jsonl
```

Patch artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_layer_transition_patch/v2_val14439_receiver729_donors729_48_readout_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_layer_transition_patch/v2_val14439_receiver729_donors385_390_readout_gpu5
```

## Setup

Receiver:

- image `14439`
- forced x1 `729`
- baseline y1 top1 `385`
- baseline target rank `924`
- baseline target-vs-contrast margin `-10.1875`

Donor x1 values:

- `729`: true x1 receiver prefix itself
- `48`: near target-y token used as a helpful synthetic control
- `385`, `390`: lower-y/baseline-basin controls that previously had strong layer-24 routes

Donor hidden layers:

```text
8,16,20,24
```

Receiver patch layers:

```text
20,24,27
```

Patch operation: replace the receiver decoder layer output at the post-x1/pre-y1 decision token with the donor hidden vector, with interpolation alphas:

```text
0,0.25,0.5,0.75,1
```

This is not an additive readout-direction bridge. It is a full-forward residual replacement at one token position; downstream layers recompute from the patched state.

## Integrity

- donor split `729,48`: `row_count=120`, `ok_row_count=120`, `patch_status_counts={"ok":120}`
- donor split `385,390`: `row_count=120`, `ok_row_count=120`, `patch_status_counts={"ok":120}`
- combined rows: `240`
- flips to target top1: `0`
- continuation was intentionally skipped in this first pass because no patch made target y1 the first token.

## Main Result

Replacing the receiver state can strongly weaken or remove the original lower-y `385` attractor, but it does not create a true target-y `38` attractor.

Best rank improvements:

| donor x1 | donor layer | patch layer | alpha | top1 | target rank | margin target-vs-385 | rank delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 385 | 8 | 24 | 1.0 | 292 | 94 | 0.0 | -830 |
| 48 | 8 | 20 | 1.0 | 917 | 145 | +0.375 | -779 |
| 390 | 8 | 24 | 1.0 | 292 | 165 | -0.875 | -759 |
| 729 | 8 | 20 | 1.0 | 917 | 193 | +0.5625 | -731 |
| 390 | 8 | 27 | 1.0 | 843 | 222 | +3.4921875 | -702 |
| 385 | 8 | 20 | 1.0 | 917 | 240 | +0.09375 | -684 |

Best margin improvements:

| donor x1 | donor layer | patch layer | alpha | top1 | target rank | margin target-vs-385 | margin delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 729 | 8 | 27 | 1.0 | 811 | 337 | +4.4453125 | +14.6328125 |
| 48 | 8 | 27 | 1.0 | 818 | 302 | +4.068359375 | +14.255859375 |
| 390 | 8 | 27 | 1.0 | 843 | 222 | +3.4921875 | +13.6796875 |
| 385 | 8 | 27 | 1.0 | 845 | 294 | +2.9111328125 | +13.0986328125 |
| 385 | 20 | 24 | 1.0 | 234 | 295 | +1.75 | +11.9375 |
| 390 | 20 | 24 | 1.0 | 234 | 291 | +1.75 | +11.9375 |

The patch can move target rank from `924` to `94`, and target-vs-385 margin from `-10.1875` to positive. However, the top1 remains far from target: `292`, `917`, `843`, `234`, `811`, etc.

## Top-Bin Patterns

The best-rank patch, donor `385` layer 8 into receiver layer 24:

```text
top1=292, rank=94, margin=0.0
top bins: 292,291,792,293,791,294,793,290
```

Helpful donor layer-8 states patched into layer 20:

```text
donor 48 -> patch 20: top1=917, rank=145, margin=+0.375
donor 729 -> patch 20: top1=917, rank=193, margin=+0.5625
donor 385 -> patch 20: top1=917, rank=240, margin=+0.09375
donor 390 -> patch 20: top1=917, rank=236, margin=+0.25
```

Donor layer-8 states patched directly into layer 27 reproduce the donor-direct early-state basin:

```text
donor 729 -> patch 27: top1=811, rank=337, margin=+4.4453125
donor 48  -> patch 27: top1=818, rank=302, margin=+4.068359375
donor 390 -> patch 27: top1=843, rank=222, margin=+3.4921875
donor 385 -> patch 27: top1=845, rank=294, margin=+2.9111328125
```

This validates the replacement hook and reaffirms the earlier readout result: early states contain target-vs-385 evidence, but the target coordinate is not the global top coordinate inside the 1000-bin coordinate manifold.

## Interpretation

This panel refines the late-overwrite hypothesis.

The original claim remains true: by layer 27, the receiver state has collapsed into a lower-y attractor around `385/390`, and replacing the residual state can causally remove that attractor.

But the stronger result is negative: removing the `385` attractor is not enough. The model does not recover target y1 `38`; instead it jumps to other wrong basins:

- layer-20 replacements often go to a high-y basin near `917`;
- layer-24 replacements often go to mid-y basins near `292` or `234`;
- layer-27 donor-layer-8 replacement preserves high-y donor basins near `811/843/845`.

Therefore the failure has at least two components:

1. **Late attractor overwrite:** target-vs-lower-basin evidence is destroyed by late layers, especially before y1 emission.
2. **Missing target sharpening:** even when the lower-y antagonist is removed or target-vs-385 margin becomes positive, the residual state does not sharpen around the actual target y1 `38`.

This is deeper than a one-head or one-token bug. It suggests the model's coordinate manifold has multiple competing scene/position basins, and the target object state is too weak to select the exact y1 coordinate after x1.

## Consequences

Single-head suppression and x1 guidance are unlikely to fully repair this val case. They may move rank or margin, but they do not establish the target-specific object-span state.

The next surgery should target the source of target sharpening, not just the lower-y antagonist:

1. Patch visual/object-state representations from a true target-object donor, not just synthetic x1-coordinate donors.
2. Add prefix-history ablations that remove lower-scene/prior-object coordinate attractors before the y1 decision.
3. Patch a full object-span state across y1/x2/y2 rather than only the post-x1 y1 decision token.
4. Compare against a train sample where local y basin exists, to isolate why train has target-near sharpening but val does not.

## Bottom Line

The causal patch panel supports a two-stage mechanism. Val `14439` has early target-vs-lower-basin evidence, then late layers overwrite the state into the lower-y attractor. Residual replacement can erase that specific attractor, but target y1 still does not become top1. The deeper missing mechanism is not merely "escape from 385"; it is formation of an object-specific coordinate basin centered near y1 `38`.
