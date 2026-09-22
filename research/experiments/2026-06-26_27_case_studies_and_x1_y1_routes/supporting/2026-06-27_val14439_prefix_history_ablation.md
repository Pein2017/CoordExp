# Val14439 Prefix-History Ablation

Date: 2026-06-27

Scope: single-sample prefix-history surgery for val image `14439`, sorted prefix-denoising checkpoint, post-x1/pre-y1 coordinate decision. This is a causal mechanism probe over one high-value failure case, not a population metric.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

Question: is the late lower-y basin for target `person [729,38,737,85]` mainly produced by absent visual evidence / final x1 ambiguity, or by autoregressive prefix-history attraction from the prior object list?

The probe keeps image, object scaffold, current target description, and forced current x1 `<|coord_729|>` fixed. It edits only the already-completed assistant object history before the current scaffold:

```text
<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_729|>
```

## Artifacts

Plan rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_plan/prefix_history_ablation_rows.jsonl
```

Plan summary:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_plan/prefix_history_ablation_summary.json
```

Layer readout:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_readout_layers0_8_16_20_24_27_gpu7
```

Value-route full sweep:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_route_layers16_20_24_27_allheads_gpu0
```

Do not use these redundant split-route scratch roots for evidence; they were intentionally interrupted after the all-row route sweep completed:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_route_original_layers16_20_24_27_allheads_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_route_keep_target_first_only_layers16_20_24_27_allheads_gpu6
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_prefix_history_ablation/v1_val14439_receiver729_route_drop_first_target_keep_tail_layers16_20_24_27_allheads_gpu7
```

## Integrity

- plan rows: `8`
- readout rows: `48`, errors `0`
- route rows: `2560`, ok rows `2560`, errors `0`
- route coverage: `8` ablations x `4` layers x `16` heads x `5` value-source regions
- layers: `16,20,24,27` for routes; `0,8,16,20,24,27` for readouts
- route direction: effective output direction for target y1 `38` minus contrast/original lower-y coordinate `385`

## Original Prefix Structure

The original prefix has `23` completed prior objects before the current forced-x1 scaffold:

| idx | desc | box |
|---:|---|---|
| 0 | person | `[729,38,737,85]` |
| 1 | person | `[0,62,24,104]` |
| 2 | person | `[584,92,625,200]` |
| 3 | person | `[598,98,617,201]` |
| 4 | person | `[552,105,585,240]` |
| 5 | person | `[164,108,204,234]` |
| 6 | person | `[290,111,311,154]` |
| 7 | person | `[492,96,539,198]` |
| 8 | person | `[555,114,592,259]` |
| 9 | kite | `[252,199,971,854]` |
| 10 | person | `[502,210,508,241]` |
| 11 | person | `[503,244,501,247]` |
| 12 | person | `[519,242,526,231]` |
| 13 | person | `[624,222,796,704]` |
| 14 | person | `[801,281,931,671]` |
| 15 | chair | `[116,302,163,386]` |
| 16 | backpack | `[1,328,40,370]` |
| 17 | handbag | `[0,335,41,371]` |
| 18 | backpack | `[0,336,27,388]` |
| 19 | backpack | `[0,342,42,396]` |
| 20 | backpack | `[0,351,43,408]` |
| 21 | backpack | `[100,372,170,431]` |
| 22 | backpack | `[113,382,178,421]` |

This history has two important properties:

1. The first completed object is target-shaped and exactly carries the target y band.
2. The tail contains many lower-scene objects with y coordinates around `302-431`, close to the model's bad late basin `382/385/390`.

## Ablations

| variant | prior objects after edit | edit |
|---|---:|---|
| `original` | 23 | unmodified prefix history |
| `drop_lower_nonperson_tail` | 14 | drop prior non-person objects with `y1>=300` or `y2>=360` |
| `drop_lower_tail_all_desc` | 12 | drop all prior objects with `y1>=280` or `y2>=360` |
| `keep_target_first_only` | 1 | keep only the first completed target-shaped person |
| `drop_first_target_keep_tail` | 22 | drop the completed target-shaped first object while keeping the lower tail |
| `coord_scramble_lower_nonperson_to_target_band` | 23 | rewrite lower non-person y coordinates to the target y band, preserving x and labels |
| `coord_scramble_lower_tail_to_target_band` | 23 | rewrite all lower-tail y coordinates to the target y band, preserving x and labels |
| `coord_scramble_lower_nonperson_to_mid_band` | 23 | rewrite lower non-person y coordinates to a middle band as a non-target control |

## Readout Result

Layer readout reports the coordinate-only y1 distribution at the post-x1/pre-y1 state. `margin` means `target y1=38` logit minus original contrast `385` logit.

| variant | L24 top1/rank/margin | L27 top1/rank/margin | best layer top1/rank/margin |
|---|---:|---:|---:|
| `coord_scramble_lower_nonperson_to_mid_band` | `372/581/-2.34375` | `121/722/-3.6875` | `L8 811/327/+4.53125` |
| `coord_scramble_lower_nonperson_to_target_band` | `288/81/+0.46875` | `300/259/0.0` | `L24 288/81/+0.46875` |
| `coord_scramble_lower_tail_to_target_band` | `288/96/+0.46875` | `300/279/-0.125` | `L24 288/96/+0.46875` |
| `drop_first_target_keep_tail` | `383/797/-7.09375` | `382/868/-9.0` | `L8 811/328/+4.3671875` |
| `drop_lower_nonperson_tail` | `282/607/-3.1875` | `282/382/-2.0625` | `L27 282/382/-2.0625` |
| `drop_lower_tail_all_desc` | `251/760/-4.140625` | `241/393/-0.8125` | `L8 811/393/+4.1015625` |
| `keep_target_first_only` | `62/13/+7.09375` | `52/18/+11.0` | `L24 62/13/+7.09375` |
| `original` | `383/907/-7.46875` | `385/924/-10.1875` | `L8 811/337/+4.4453125` |

Main readout result:

- Original late state is the known failure: layer 27 top1 `385`, target rank `924`, margin `-10.1875`.
- Keeping only the first target-shaped prior object changes the layer 27 state to top1 `52`, target rank `18`, margin `+11.0`.
- Dropping that first target-shaped prior object while preserving the lower tail leaves the failure essentially intact: layer 27 top1 `382`, target rank `868`, margin `-9.0`.
- Rewriting lower-tail y coordinates into the target band helps substantially at layer 24 (`rank 81/96`) but still does not make the true target `38` top1.
- Rewriting lower-tail y coordinates into a middle band moves the late top1 to `121`, a useful control: coordinate history can redirect the basin away from `385`, but not necessarily toward the true object coordinate.

## Route Result

The route rows decompose attention/value contributions along the `target 38 - contrast 385` direction. Positive projection supports target y1 over the lower-y contrast; negative projection supports the lower-y contrast over target.

The decisive route is layer 27 head 15. Its sign flips under prefix-history surgery:

| variant | prompt_non_image | assistant_prefix_prior_objects | current_forced_coords | current_partial_object | recent_16 |
|---|---:|---:|---:|---:|---:|
| `original` | `-0.125` | `+0.393` | `-7.956` | `-7.813` | `-7.696` |
| `drop_first_target_keep_tail` | `-0.096` | `+0.513` | `-7.887` | `-7.671` | `-7.514` |
| `keep_target_first_only` | `-0.131` | `+0.549` | `+8.230` | `+8.229` | `+8.782` |
| `drop_lower_nonperson_tail` | `-0.082` | `+0.091` | `-9.847` | `-9.811` | `-9.765` |
| `coord_scramble_lower_nonperson_to_target_band` | `-0.081` | `+0.498` | `+6.025` | `+6.213` | `+6.391` |
| `coord_scramble_lower_nonperson_to_mid_band` | `-0.095` | `+0.597` | `+6.933` | `+7.069` | `+7.171` |

Attention mass alone does not explain the sign flip:

| variant | current_forced_coords attention | current_partial_object attention | recent_16 attention |
|---|---:|---:|---:|
| `original` | `0.490` | `0.506` | `0.519` |
| `drop_first_target_keep_tail` | `0.461` | `0.486` | `0.502` |
| `keep_target_first_only` | `0.328` | `0.338` | `0.385` |
| `drop_lower_nonperson_tail` | `0.645` | `0.648` | `0.673` |
| `coord_scramble_lower_nonperson_to_target_band` | `0.629` | `0.652` | `0.667` |
| `coord_scramble_lower_nonperson_to_mid_band` | `0.484` | `0.501` | `0.511` |

This means the current forced `<|coord_729|>` route is not merely attended more or less. Its value-direction meaning changes with prefix history. In the original and `drop_first_target_keep_tail` histories, the same current x1 token is routed as evidence for the bad lower-y basin. In `keep_target_first_only`, it is routed as evidence against that basin and toward the target-near y band.

Top route patterns:

- `original`: largest absolute route is L27H15 current/recent, all negative (`-7.956`, `-7.813`, `-7.696`), top tokens led by current `<|coord_729|>` and current box scaffold.
- `drop_first_target_keep_tail`: same negative L27H15 pattern (`-7.887`, `-7.671`, `-7.514`), top1 basin `390`.
- `keep_target_first_only`: same L27H15 families become positive (`+8.230`, `+8.229`, `+8.782`), top tokens are led by current `<|coord_729|>` and the now-short local box history; a separate positive L27H3 prior-object route attends to target-shaped coordinates `<|coord_737|>`, `<|coord_38|>`, `<|coord_85|>`.
- `drop_lower_nonperson_tail`: L27H15 becomes even more negative (`-9.847`, `-9.811`, `-9.765`), despite removing obvious lower non-person tail objects. This warns that the attractor is not only "backpack tail tokens"; broader object-history layout still matters.
- `coord_scramble_lower_nonperson_to_target_band`: L27H15 flips positive (`+6.025`, `+6.213`, `+6.391`) but late top1 remains `300`, rank `259`; positive against `385` is not sufficient for exact target sharpening.
- `coord_scramble_lower_nonperson_to_mid_band`: L27H15 also flips positive against `385`, but late top1 moves to `121`, target rank `722`; this is a control proving the route direction is contrast-specific, not a full target-coordinate proof.

Summed over source regions, the late head profile is:

| variant | layer | strongest positive heads | strongest negative heads |
|---|---:|---|---|
| `original` | 27 | H6 `+4.587`, H14 `+4.375`, H2 `+2.110` | H15 `-23.198`, H11 `-5.296`, H9 `-0.827` |
| `keep_target_first_only` | 27 | H15 `+25.659`, H3 `+8.876`, H2 `+1.898` | H10 `-2.462`, H14 `-1.522`, H4 `-1.242` |
| `drop_first_target_keep_tail` | 27 | H14 `+4.682`, H6 `+4.550`, H2 `+2.201` | H15 `-22.655`, H11 `-5.585`, H9 `-1.061` |

## Interpretation

This ablation refines the previous two-component failure model.

Earlier probes showed that val `14439` has early target-vs-385 evidence at layers 8/16, then late layers overwrite the state into a lower-y coordinate basin. Residual replacement could weaken the lower-y attractor but did not create exact target y1 `38`.

This probe identifies a likely origin of that late overwrite: the autoregressive object-history state conditions what the current coordinate token means to late route heads.

The current `<|coord_729|>` token is not an invariant geometry key. Under the original full prefix, L27H15 reads the current forced x1/current object as evidence for the wrong lower-y basin. Under `keep_target_first_only`, L27H15 reads the same current forced x1/current object as evidence for target-vs-385. The input image and current x1 did not change. The object-history prefix changed the latent binding state that late attention/value routes operate on.

The most important causal contrast is:

```text
original:
  L27 top1=385, target rank=924, margin=-10.1875
  L27H15 current/recent route about -7.7 to -8.0

keep_target_first_only:
  L27 top1=52, target rank=18, margin=+11.0
  L27H15 current/recent route about +8.2 to +8.8

drop_first_target_keep_tail:
  L27 top1=382, target rank=868, margin=-9.0
  L27H15 current/recent route about -7.5 to -7.9
```

This strongly supports a prefix-history binding mechanism:

1. A target-shaped previous object can anchor a near-target y basin when the rest of history is removed.
2. The lower/mixed object history can flip the late current-coordinate route into a lower-y attractor.
3. The final x1 token alone does not specify the next coordinate; it is interpreted through a contextual object-history state.
4. Removing or rewriting the lower tail can move the basin, but exact y1 sharpening remains missing. So the model needs both escape from the wrong attractor and formation of an exact object-specific coordinate basin.

## What This Does Not Prove

- It does not prove population frequency, because this is one curated hard val sample.
- It does not prove visual non-perception. The image was fixed; this probe manipulates only text-side assistant history.
- It does not prove that the first completed object should normally be kept or repeated. The first prior object is target-shaped in this constructed prefix, and the result is a causal handle, not an inference-time recipe.
- It does not prove L27H15 alone is sufficient. It is the largest sign-flipping route, but other heads and the residual stream determine final top1. Target-vs-385 positivity can coexist with top1 `121` or `300`.

## Next Experiments

1. Run the same prefix-history ablation on at least two additional hard val samples with different false-negative/duplicate contexts to test whether late current-coordinate sign flip is a general mechanism or a val14439-specific pathology.
2. Add object-history donor patching: transplant only prior-object hidden states from `keep_target_first_only` into the original prefix at layers 20/24/27, then test whether L27H15 and the layer-27 readout flip.
3. Probe the full object span `[y1,x2,y2,box_end]` after `keep_target_first_only` and `coord_scramble` variants. The current result reaches a near-target y band but not exact y1 top1.
4. Compare sorted prefix-denoising against pure sorted CE and random denoising on the same ablation rows to separate prefix-denoising's learned history use from the base sorted-template autoregressive bias.

## Bottom Line

The best current explanation for val `14439` is no longer "the model cannot see the target object" or "x1 guidance is fragile" in a simple way. The deeper failure is contextual binding: prior object history changes the value-route meaning of the current x1 coordinate token. In the full prefix, late L27H15 routes current `<|coord_729|>` into the lower-y basin; in the target-only prefix, the same token routes toward the target-near basin. Prefix-denoising has therefore not removed exposure-bias-like autoregressive history attraction. It has given us a sharper internal surface where the attraction can be measured.
