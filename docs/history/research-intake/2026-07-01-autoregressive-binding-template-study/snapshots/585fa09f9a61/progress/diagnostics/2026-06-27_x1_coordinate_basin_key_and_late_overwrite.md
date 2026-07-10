# X1 Coordinate Basin-Key And Late Overwrite Probe

Date: 2026-06-27

Scope: readout-only and route-only mechanistic probes over two curated sample bases from the prefix-denoising sorted checkpoint. This is not a population metric and should not be interpreted as validation over normal or well-learned images.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

Sample bases:

- Train `418535`, target `person [795,48,806,83]`; previous forced-x1 continuation was `[795,46,794,99]`.
- Val `14439`, target `person [729,38,737,85]`; previous forced-x1 continuation was `[729,385,935,671]`.

## Question

Does the final x1 coordinate token act as a discrete key into the y1 coordinate basin, and where does the model lose or overwrite target-y evidence before emitting y1?

The probe fixes image, prompt, object text, prior assistant prefix, and box scaffold, then sweeps only the final forced x1 token at `post_x1_pre_y1`. It pairs a cheap layerwise readout sweep with full attention/value route capture.

## Artifacts

Route-compatible y1 alias plan:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route_plan/v3_train418535_val14439_y1_legacy_y2_alias_matching_route_keys
```

This plan intentionally uses `target_y2_bin/generated_y2_bin/original_y2_bin` as compatibility aliases for `target_next_coord_slot=y1`, because several existing tools are still y2-named. The row-level semantic slot remains y1.

Tensor-flow reruns after route-key alias repair:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_tensor_flow/v4_train418535_y1_alias_routekey_scored_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_tensor_flow/v4_val14439_y1_alias_routekey_scored_gpu7
```

Readout basin-key sweep:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_basin_key_readout/v1_train418535_layers0_8_16_20_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_basin_key_readout/v1_val14439_layers0_8_16_20_24_27_gpu1
```

Full route basin-key sweep:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_basin_key_sweep_route/v1_train418535_layers16_20_24_27_allheads_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_basin_key_sweep_route/v1_val14439_layers16_20_24_27_allheads_gpu3
```

## Artifact Integrity

- Tensor-flow v4 train: `group_spec_count=8`, `row_count=384`, all `readout_status=ok`, `error_count=0`.
- Tensor-flow v4 val: `group_spec_count=8`, `row_count=384`, all `readout_status=ok`, `error_count=0`.
- Readout train: `row_count=96`, `error_count=0`.
- Readout val: `row_count=90`, `error_count=0`.
- Route sweep train: `plan_row_count=16`, `row_count=5120`, `ok_row_count=5120`, `error_count=0`.
- Route sweep val: `plan_row_count=15`, `row_count=4800`, `ok_row_count=4800`, `error_count=0`.

The earlier tensor-flow v3 roots with `group_spec_count=0` were handle-mismatch artifacts: the alias-suffixed plan keys did not match pre-alias route row `source_state_key` values. Do not use those zero-row roots as model evidence.

## Tensor-Flow Finding

The repaired tensor-flow runs confirm a sharp train/val split at the y1 slot.

Train `418535`:

- Forced target x1 has late target-y1 readability: layer 27 best rank reaches `7` under current negative suppression / signed combo groups.
- Natural wrong x1 also reaches layer 27 best rank `7`.
- The late top bins remain near target-y: mostly `46`, `54`, `56`, `62`, `64`. This is a local basin, not a far spatial collapse.

Val `14439`:

- Forced target x1 remains far from target at late layers: layer 27 best ranks stay around `893` to `927` depending group.
- Natural wrong x1 is similarly poor: layer 27 best ranks around `912` to `937`.
- Late top bins live in a lower-y basin: mostly `353`, `382`, `383`, `385`, `390`.

Interpretation: the val failure is not just one bad route head. The late residual/readout state has already moved into a different y basin.

## X1 Sweep Readout Finding

### Train 418535

Layer 27 is surprisingly invariant: most synthetic x1 tokens still end in the local `46` basin, near target y1 `48`.

Layer-27 examples:

| forced x1 | y1 top1 | target rank | margin target-vs-contrast 46 | top bins |
|---:|---:|---:|---:|---|
| 47 | 46 | 4 | -0.75 | 46,45,49,43 |
| 302 | 54 | 7 | -0.25 | 54,55,56,52 |
| 729 | 46 | 6 | -0.625 | 46,45,49,52 |
| 795 | 46 | 14 | -0.875 | 46,45,55,54 |
| 385 | 60 | 26 | -0.5 | 62,60,46,52 |

Best target rank by layer:

| layer | best x1 | best rank |
|---:|---:|---:|
| 0 | 47 | 269 |
| 8 | 385 | 335 |
| 16 | 794 | 867 |
| 20 | 385 | 527 |
| 24 | 302 | 24 |
| 27 | 47 | 4 |

This says the train sample develops the correct local y-range late. The exact x1 token modulates the basin but does not determine it alone; the whole prefix/image context pulls toward the correct y band.

### Val 14439

Layer 27 remains wrong for every tested synthetic x1. The x1 token does select among discrete wrong basins, but none reaches target y1 `38`.

Layer-27 examples:

| forced x1 | y1 top1 | target rank | margin target-vs-contrast 385 | top bins |
|---:|---:|---:|---:|---|
| 38 | 298 | 641 | -5.375 | 298,299,296,295 |
| 48 | 298 | 588 | -4.8125 | 298,296,295,300 |
| 385 | 226 | 725 | -1.4375 | 226,223,225,228 |
| 390 | 226 | 718 | -1.5625 | 226,228,223,225 |
| 729 | 385 | 924 | -10.1875 | 385,388,390,391 |
| 795 | 388 | 937 | -10.25 | 390,388,391,387 |

Best target rank by layer:

| layer | best x1 | best rank |
|---:|---:|---:|
| 0 | 38 | 662 |
| 8 | 390 | 222 |
| 16 | 795 | 398 |
| 20 | 38 | 673 |
| 24 | 48 | 775 |
| 27 | 48 | 588 |

Important layer trajectory for true x1 `729`:

| layer | top1 | target rank | margin target-vs-contrast 385 |
|---:|---:|---:|---:|
| 8 | 811 | 337 | +4.4453125 |
| 16 | 224 | 445 | +1.16015625 |
| 20 | 442 | 770 | -1.75 |
| 24 | 383 | 907 | -7.46875 |
| 27 | 385 | 924 | -10.1875 |

This is the most important finding in this round: target-vs-contrast evidence is present and positive around layers 8/16, but the model later overwrites it into the wrong lower-y basin. The failure is therefore better described as late attractor transition / overwrite, not absence of early visual or slot evidence.

## Route Sweep Finding

Train late routes:

- Positive layer-27 routes are mainly L27H1/H3 and use prior-object/recent coordinate context.
- High-x synthetic tokens near the target x1 range (`794`, `795`, `796`, `806`) produce positive L27H1 projections through previous object coordinates such as `<|coord_664|>`, `<|coord_637|>`, `<|coord_108|>`.
- L27H2 and L27H14 can be strongly negative, especially for x1 `302` and high-x tokens, but the overall state remains a near-target local y basin.

Val late routes:

- Layer 24 has a strong positive L24H0 current-coordinate route for x1 `385/390` and moderate positive route for x1 `38/48`:
  - x1 `390`, L24H0, current/recent regions: projection about `+9.3`.
  - x1 `385`, L24H0, current/recent regions: projection about `+9.2`.
  - x1 `38/48`, L24H0, current/recent regions: projection about `+4.8`.
- Layer 27 introduces very strong negative L27H15 current-coordinate/current-object routes:
  - x1 `54`: about `-17.5`.
  - x1 `38`: about `-16.9`.
  - x1 `48`: about `-16.1`.
  - x1 `46`: about `-15.7`.
  - x1 `302`: about `-15.5`.
  - x1 `729`: minimum around `-8.0` across current/recent regions.

For selected val x1 values, current/recent route projections show the transition:

| x1 | layer | region family | max proj | min proj | mean proj |
|---:|---:|---|---:|---:|---:|
| 48 | 24 | current/recent | +4.847 | -2.021 | +0.143 to +0.147 |
| 48 | 27 | current/recent | +0.910 | -16.082 | about -1.1 |
| 385 | 24 | current/recent | +9.315 | -0.434 | about +0.55 |
| 385 | 27 | current/recent | +4.441 | -7.887 | about -0.12 |
| 390 | 24 | current/recent | +9.413 | -0.401 | about +0.56 |
| 390 | 27 | current/recent | +3.531 | -7.565 | about -0.17 |
| 729 | 24 | current/recent | +0.301 | -0.623 | near 0 |
| 729 | 27 | current/recent | +1.542 | -7.956 | about -0.4 |

This route profile matches the readout trajectory: layer 24 can still contain positive target-vs-lower-basin routes, but layer 27 contains a stronger negative current-coordinate route that pushes the residual state away from target y1.

## Refined Mechanism Hypothesis

The current best hypothesis is:

1. The model does perceive or at least encode enough target-y evidence early. Val `14439` has positive target-vs-lower-basin margins at layers 8/16 under true x1 `729`.
2. The final x1 token is a partial basin key: changing only that token moves val among discrete wrong y basins (`298`, `226`, `383/385/390`), while train mostly remains in a near-target basin (`46/54/60`).
3. x1 token identity is not sufficient to recover the correct object tail. For val, no tested x1 token reaches target y1 `38` at layer 27; even helpful synthetic x1 tokens only move to intermediate wrong basins.
4. The core failure is a late layer attractor transition. Around layers 20-27, the target-y signal is overwritten by current-coordinate and object-prefix routes, especially val L27H15. This produces a lower-y basin before visible y1 emission.
5. Train vs val differ in attractor basin quality: train converges into a local y basin near target and can be locally nudged; val converges into a far lower-y basin and needs representation-level surgery, not one-token guidance.

This also explains why forcing correct x1 was not enough: the x1 token repairs one coordinate, but the model has not committed to the target object-span latent state across `[y1,x2,y2]`.

## Immediate Next Experiments

1. Layer-transition residual surgery:
   - For val true x1 `729`, patch or preserve residual state from layers 8/16 into later layers and observe layer-27 y1 rank/top1.
   - Separately patch across `16 -> 20`, `20 -> 24`, and `24 -> 27` to locate the attractor transition.
   - Score full continuation `[y1,x2,y2,box_end]`, not only y1.

2. Late-head interaction surgery:
   - Test combined suppression of val L27H15 current/recent negative routes with amplification or preservation of layer-24 positive routes.
   - Do not over-interpret single-head suppression, because prior exact L27H15 suppression improved rank only weakly and did not flip top1.

3. Prefix-history ablation:
   - For val `14439`, remove or coordinate-scramble prior lower-scene boxes while keeping target image/object text fixed.
   - If the lower-y attractor weakens, the failure origin is object-list/history attraction rather than pure x1 token geometry.

4. Full-span object-state patch:
   - Patch donor target-object states across post-x1, post-y1, and post-x2.
   - Compare y1-only repair against whole `[y1,x2,y2]` repair to distinguish local coordinate calibration from object-span binding.

## Bottom Line

This round gives a more precise mechanism than "the model misses the object" or "x1 guidance is fragile." In the val failure case, early target-y evidence exists, but the late autoregressive coordinate machinery overwrites it into a lower-y attractor. The current x1 token participates in basin selection, but the basin is also shaped by prior object/coordinate context and late attention/value routes. The deepest next question is therefore the transition mechanism between layers 16/20/24/27: where the target-object state loses control to the coordinate/history attractor, and whether preserving that state repairs the whole object span.
