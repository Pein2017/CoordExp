# Cross-Sample FN-Guidance Prefix-History Panel

Date: 2026-06-27

Scope: three-case mechanism panel for sorted prefix-denoising checkpoint `908`, using existing val200 free-rollout artifacts. This is not a population metric. The panel asks whether selected false negatives are visual non-perception failures or whether true language-side guidance can form a target-near coordinate basin once harmful rollout history is removed or edited.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

Rollout/eval source:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

## Question

For missing GT objects, if we append the missing object's description and true x1 after the model's own free-rollout object history:

```text
...free rollout object history...
<|object_ref_start|>{missing desc}<|object_ref_end|><|box_start|><|coord_true_x1|>
```

does the post-x1/pre-y1 hidden state form the target y1 basin, or does prior generated history still pull it into a wrong coordinate basin?

This is the false-negative analogue of the val `14439` prefix-history ablation. It probes guidance sufficiency, not rollout quality.

## Samples

Selected from the current val200 sorted prefix-denoising rollout:

| image | target | target bins `[x1,y1,x2,y2]` | free-rollout context | contrast anchor |
|---:|---|---|---|---|
| `632` | `book` gt0 | `[776,113,784,172]` | 19 predictions, mostly books | nearest generated book `[788,96,802,164]` |
| `5586` | `person` gt0 | `[425,100,461,183]` | 15 predictions, 14 persons plus tennis racket | nearest generated person `[439,21,456,87]` |
| `18380` | `wine glass` gt9 | `[516,287,557,426]` | 44 dense mixed predictions | partial generated wine glass `[525,281,538,313]` |

For route analysis, contrast was reset from the nearest generated anchor to the actual full-history layer-27 failure top1:

| image | target y1 | full-history L27 failure top1 |
|---:|---:|---:|
| `632` | `113` | `593` |
| `5586` | `100` | `710` |
| `18380` | `287` | `946` |

## Artifacts

Plan rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val_cases_632_5586_18380_plan
```

Layer readout:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val_cases_632_5586_18380_readout_layers0_8_16_20_24_27_gpu2
```

Route contrast plan:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val_cases_632_5586_18380_route_plan_orig_l27_contrast
```

Focused route shards:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val632_route_orig_l27_contrast_layers16_20_24_27_gpu5
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val5586_route_orig_l27_contrast_layers16_20_24_27_gpu6
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val18380_route_orig_l27_contrast_layers16_20_24_27_gpu7
```

## Integrity

- plan rows: `30` = 3 cases x 10 prefix-history variants
- readout rows: `180`, errors `0`
- route contrast plan rows: `15` = 3 cases x 5 selected variants
- route rows: `4800`, errors `0`
- route coverage per image: 5 variants x 4 layers x 16 heads x 5 value-source regions
- route direction: target y1 minus original full-history L27 failure top1

## Prefix Variants

The readout panel used:

```text
original
no_history
keep_recent8_only
drop_recent8
drop_same_desc_history
keep_same_desc_only
coord_scramble_same_desc_to_target_band
coord_scramble_recent8_to_target_band
coord_scramble_all_to_target_band
coord_scramble_same_desc_to_offset_band
```

The route panel focused on:

```text
original
no_history
drop_recent8
drop_same_desc_history
coord_scramble_all_to_target_band
```

## Readout Finding

### Image 632, Missing Book

Target y1 is `113`. Original full history collapses to a far lower basin:

| variant | L24 top/rank/margin vs anchor 96 | L27 top/rank/margin vs anchor 96 | best |
|---|---:|---:|---:|
| `original` | `592/899/-1.5625` | `593/798/-1.5625` | `L20 436/401/+1.1875` |
| `no_history` | `96/71/-2.625` | `93/41/-2.5` | `L27 93/41/-2.5` |
| `drop_recent8` | `343/723/-0.96875` | `319/739/+0.875` | `L20 436/249/+1.375` |
| `drop_same_desc_history` | `341/875/-1.15625` | `592/875/-0.75` | `L20 442/384/+0.5625` |
| `coord_scramble_all_to_target_band` | `868/569/+1.03125` | `999/710/+2.0` | `L20 489/276/+1.6875` |

Removing all history moves the late top1 from `593` to `93` and improves target rank from `798` to `41`, but exact target `113` still loses to nearby/anchor coordinates.

### Image 5586, Missing Person

Target y1 is `100`. This case does not recover cleanly even with no history:

| variant | L24 top/rank/margin vs anchor 21 | L27 top/rank/margin vs anchor 21 | best |
|---|---:|---:|---:|
| `original` | `716/870/+1.3203125` | `710/619/+4.84375` | `L16 224/193/+2.1953125` |
| `no_history` | `281/215/+0.03125` | `281/336/-1.6875` | `L16 724/171/+2.8828125` |
| `drop_recent8` | `2/30/+2.0625` | `2/247/-1.0` | `L24 2/30/+2.0625` |
| `drop_same_desc_history` | `281/547/+1.171875` | `282/795/-0.9375` | `L8 843/255/+3.0` |
| `coord_scramble_all_to_target_band` | `716/868/+1.140625` | `716/776/+4.5625` | `L16 225/201/+2.515625` |

This is the cautionary case. The model can be pushed away from the full-history failure top1, but it does not form a target-near y1 basin. It demonstrates missing target sharpening even when target-vs-contrast margins are positive.

### Image 18380, Missing Wine Glass

Target y1 is `287`. This is the clearest cross-sample replication of the val `14439` mechanism:

| variant | L24 top/rank/margin vs anchor 281 | L27 top/rank/margin vs anchor 281 | best |
|---|---:|---:|---:|
| `original` | `936/917/-0.5078125` | `946/949/-0.5` | `L16 194/128/+1.203125` |
| `no_history` | `280/36/-2.0` | `281/9/-0.75` | `L27 281/9/-0.75` |
| `drop_recent8` | `664/875/-0.96875` | `704/926/-0.4375` | `L16 193/153/+1.359375` |
| `drop_same_desc_history` | `936/859/-0.578125` | `946/939/-0.4375` | `L16 194/128/+1.2578125` |
| `coord_scramble_all_to_target_band` | `414/141/-0.5` | `545/150/+0.25` | `L16 194/129/+1.1875` |

Removing all history moves L27 from `946/rank949` to `281/rank9`, right next to target `287`. That is difficult to reconcile with pure visual non-perception. The image and target scaffold are fixed; history removal is the intervention.

## Route Finding

Route directions use target y1 minus the full-history failure top1. Positive means "toward target over the failure basin"; negative means "toward the failure basin over target."

### Image 18380 Replicates The L27H15 Sign Flip

The late L27H15 current-object route flips almost exactly like val `14439`:

| variant | L27 strongest positive heads | L27 strongest negative heads |
|---|---|---|
| `original` | H2 `+9.445`, H14 `+7.726`, H3 `+7.641` | H15 `-18.223`, H13 `-6.983`, H1 `-1.974` |
| `no_history` | H15 `+27.909`, H3 `+17.507`, H13 `+15.712` | H1 `-2.125`, H14 `-2.029`, H2 `-1.415` |
| `drop_recent8` | H14 `+6.544`, H3 `+5.853`, H5 `+5.498` | H15 `-20.632`, H1 `-3.170`, H12 `-0.697` |
| `drop_same_desc_history` | H2 `+11.356`, H3 `+5.586`, H14 `+3.058` | H15 `-29.692`, H1 `-2.053`, H13 `-1.608` |
| `coord_scramble_all_to_target_band` | H2 `+9.038`, H15 `+5.820`, H3 `+4.053` | H1 `-1.389`, H12 `-0.824`, H7 `-0.504` |

Top L27H15 current routes:

```text
original:
  current_forced_coords  -6.132
  current_partial_object -6.125
  recent_16              -6.105

no_history:
  current_forced_coords  +9.297
  current_partial_object +9.289
  recent_16              +9.288

drop_same_desc_history:
  current_forced_coords  -9.818
  current_partial_object -9.875
  recent_16              -9.854
```

This is not only same-desc history. Dropping wine-glass prior objects while keeping the rest of the dense history makes L27H15 more negative. The harmful state is distributed in the broader generated object history.

### Image 632 Shows A Competing-Head Version

Image `632` has a positive L27H3 target-vs-failure route even in the original prefix, but L27H15 is a strong antagonist:

| variant | L27 strongest positive heads | L27 strongest negative heads |
|---|---|---|
| `original` | H3 `+29.202`, H14 `+3.687`, H5 `+2.041` | H15 `-25.865`, H2 `-3.371`, H10 `-1.309` |
| `no_history` | H3 `+21.563`, H11 `+1.663`, H0 `+0.975` | H14 `-13.940`, H1 `-8.165`, H2 `-5.042` |
| `drop_recent8` | H15 `+18.485`, H3 `+9.578`, H11 `+6.885` | H14 `-12.564`, H2 `-4.643`, H6 `-3.305` |
| `drop_same_desc_history` | H3 `+18.138`, H0 `+2.337`, H5 `+1.779` | H15 `-23.317`, H2 `-3.909`, H14 `-1.795` |
| `coord_scramble_all_to_target_band` | H15 `+44.243`, H3 `+38.057`, H5 `+3.012` | H14 `-12.735`, H2 `-2.324`, H13 `-1.503` |

The no-history readout is target-near (`93/rank41`), but not exact. The original state has both strong positive and strong negative target-vs-failure routes, and the final basin loses.

### Image 5586 Is A Missing-Sharpening Case

Image `5586` does not follow the "negative H15 causes failure" template:

| variant | L27 strongest positive heads | L27 strongest negative heads |
|---|---|---|
| `original` | H14 `+46.132`, H8 `+2.095`, H12 `+2.016` | H15 `-3.292`, H13 `-2.590`, H7 `-0.391` |
| `no_history` | H15 `+28.681`, H2 `+5.695`, H3 `+3.488` | H6 `-0.457`, H10 `-0.287`, H14 `-0.286` |
| `drop_recent8` | H15 `+33.593`, H2 `+29.740`, H0 `+4.497` | H3 `-5.951`, H14 `-2.757`, H6 `-0.362` |
| `drop_same_desc_history` | H3 `+7.004`, H8 `+1.577`, H14 `+1.153` | H15 `-7.271`, H13 `-3.218`, H7 `-0.212` |
| `coord_scramble_all_to_target_band` | H14 `+25.313`, H8 `+2.018`, H1 `+1.817` | H15 `-14.436`, H13 `-1.364`, H7 `-0.550` |

Original has a very strong positive H14 target-vs-710 route, yet top1 remains `710` and target rank is `619`. This means "route away from the failure top1" is not equivalent to exact target formation. The model may know that `710` is wrong under some directions, but it still fails to make `100` locally sharp.

## Refined Mechanism

The cross-sample panel separates three mechanisms that were partially entangled in val `14439`.

1. **Guidance can be sufficient when history is removed.** For `18380`, true desc+x1 with no prior generated object history gives L27 target rank `9` and top1 `281`, near target `287`. For `632`, no history gives rank `41` and top1 `93`, near target `113`. These are not solved, but they are much closer than full-history states.

2. **Generated prefix history can change the route meaning of the current coordinate token.** In `18380`, L27H15 current/recent routes flip from about `-6.1` under full history to about `+9.3` under no history, along the target-vs-failure direction. This directly replicates the val `14439` sign-flip mechanism on a different class and scene.

3. **Exact coordinate sharpening is a separate bottleneck.** `632` and `5586` show that escaping the failure basin or having positive target-vs-failure routes does not guarantee target top1. The hidden state often lands in a nearby coordinate basin (`93`, `281`) or an unrelated basin (`2`, `716`) rather than the exact target y1.

4. **The harmful history is not always same-desc.** For `18380`, dropping same-desc wine-glass history makes L27H15 more negative; the harmful state is carried by the broader dense object list. For `632`, dropping same-desc book history leaves L27H15 strongly negative. Same-class repetition is one handle, not the whole mechanism.

5. **Coordinate-band rewriting can move basins without solving binding.** Rewriting all prior y coordinates to target bands flips some target-vs-failure routes positive, but it can send top1 to `999` (`632`) or `545` (`18380`). This mirrors val `14439`: the coordinate manifold can be redirected without forming the correct object-specific coordinate basin.

## Consequence For False Negatives

The answer to "does the model really not perceive the missing object, or does it need language-side guidance?" is mixed but now sharper:

- For `18380` and likely `632`, the model has enough visual/language capacity to form a target-near basin when the harmful prefix history is removed. These look more like contextual binding / exposure-history failures than pure visual non-perception.
- For `5586`, guidance plus history removal is not enough. This case may involve weak visual evidence, crowded person ambiguity, or missing target-specific sharpening after x1.

So false negatives should not be treated as one bucket. They split into at least:

1. **history-poisoned recoverable FNs**: no-history target scaffold recovers a near-target basin;
2. **history-poisoned but unsharp FNs**: routes improve, but exact target y1 is not selected;
3. **possibly visual-weak FNs**: even no-history guidance cannot establish a useful target-local basin.

## Next Experiments

1. Run full-span continuation after the no-history and original variants for `632` and `18380`. If no-history emits plausible `[y1,x2,y2]`, the FN is strongly language/history recoverable.
2. Patch history state from `no_history` into `original` at layers 20/24/27 for `18380` to test whether the L27H15 sign flip is caused by current-token hidden state, prior-object keys/values, or both.
3. Add a duplicate-specific stop-vs-repeat probe for `5586`, `632`, and `19432`; do not force every duplicate case into the y1-target frame.
4. Compare the same FN-guidance rows against the random prefix-denoising checkpoint and a pure sorted CE baseline to isolate whether prefix-denoising changes history sensitivity or only shifts the basin geometry.

## Bottom Line

This panel generalizes the val `14439` result without claiming population proof. The deepest current picture is that prefix-denoising did not remove autoregressive history attraction. In multiple hard samples, the model can be guided toward a missing object only when the generated prefix history is removed or heavily altered. The internal mechanism is not just "attention to previous boxes"; prior object history changes the late route meaning of the current object/coordinate state, especially through L27H15 in `18380` and val `14439`. But exact coordinate selection remains a second bottleneck, visible in `632` and `5586`.
