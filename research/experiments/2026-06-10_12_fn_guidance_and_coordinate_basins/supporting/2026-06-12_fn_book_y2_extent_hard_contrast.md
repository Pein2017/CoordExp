# FN Book Y2 Extent Hard Contrast

Date: 2026-06-12

## Scope

This note records a paired probe for the `hard_no_rescue` book false negative
on COCO val image `139`, GT object `17`.

Question:

- If the model is given the correct object text and the first three box
  coordinates `x1,y1,x2`, can either the no-aligner parent or aux checkpoint
  recover the missing object's `y2` extent from prefix context?

Answer: no. Both checkpoints remain in a short-box coordinate basin. The true
`y2=826` stays very low rank, while top coordinate mass remains near
`730..739`. Full-prefix hidden-state patching gives only tiny rank recovery and
does not move the top coordinate toward the real lower edge.

## Case

Target object:

```text
image_id = 139
gt_idx = 17
desc = book
target = [944, 716, 967, 826]
```

Decode evidence from the guidance panel:

| checkpoint | prefix | forced tier | decoded box | IoU |
| --- | --- | --- | --- | ---: |
| no-aligner parent | `0` | `desc_x1_y1_x2` | `[944,716,967,736]` | 0.1818 |
| no-aligner parent | `all` | `desc_x1_y1_x2` | `[944,716,967,739]` | 0.2091 |
| aux checkpoint | `0` | `desc_x1_y1_x2` | `[944,716,967,730]` | 0.1273 |
| aux checkpoint | `all` | `desc_x1_y1_x2` | `[944,716,967,730]` | 0.1273 |

This makes the case specifically about box extent, not just object-name access
or origin localization.

## Artifacts

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_book17_y2_extent_full_to_empty_l24_27
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/aux_latest_ckpt32_book17_y2_extent_full_to_empty_l24_27
```

Each root contains:

```text
fn_coordslot_logit_condition_rows.jsonl
fn_coordslot_logit_layer_rows.jsonl
fn_coordslot_hidden_delta_rows.jsonl
fn_coordslot_direction_rows.jsonl
fn_coordslot_residual_patch_rows.jsonl
phase4_fn_coordslot_logit_probe_summary.json
phase4_fn_coordslot_logit_probe_report.md
```

## Probe Design

Same-next-slot extent patch:

```text
source = all,desc_x1_y1_x2
target = 0,desc_x1_y1_x2
next slot = y2
target bin = 826
layers = 24,25,26,27
sites = decoder_layer,mlp,self_attn
```

The source has full-prefix context, while the target has an empty object prefix.
If the full-prefix state contained a portable hidden-state repair for the true
book extent, residual patching should lift the target `y2=826` basin or at
least move top coordinate mass toward it.

## Condition Readout

| checkpoint | prefix | target rank | target prob | top1 | top1 distance | top bins |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no-aligner parent | `0` | 131 | 0.000696 | 737 | 89 | `[737,736,739,743,741,740,730,734]` |
| no-aligner parent | `all` | 130 | 0.000650 | 739 | 87 | `[739,734,737,743,736,730,740,741]` |
| aux checkpoint | `0` | 161 | 0.000219 | 730 | 96 | `[730,731,722,734,723,718,724,725]` |
| aux checkpoint | `all` | 155 | 0.000278 | 734 | 92 | `[734,731,730,736,723,722,725,726]` |

The full prefix barely helps the no-aligner parent and only modestly helps the
aux checkpoint. In both cases the top coordinate remains roughly 90 bins above
the true lower edge.

## Residual Patch Result

Best no-aligner parent patch rows:

| site | layer | patched rank | rank recovery | patched top1 | top1 distance | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 26 | 125 | 6 | 739 | 87 | +0.000110 |
| `self_attn` | 24 | 126 | 5 | 743 | 83 | -0.000010 |
| `decoder_layer` | 24 | 127 | 4 | 739 | 87 | +0.000011 |
| `decoder_layer` | 26 | 130 | 1 | 739 | 87 | -0.000080 |
| `decoder_layer` | 27 | 130 | 1 | 739 | 87 | -0.000046 |
| `mlp` | 27 | 130 | 1 | 739 | 87 | -0.000018 |

Best aux checkpoint patch rows:

| site | layer | patched rank | rank recovery | patched top1 | top1 distance | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decoder_layer` | 24 | 154 | 7 | 730 | 96 | +0.000094 |
| `decoder_layer` | 25 | 154 | 7 | 734 | 92 | +0.000072 |
| `decoder_layer` | 26 | 155 | 6 | 734 | 92 | +0.000042 |
| `decoder_layer` | 27 | 155 | 6 | 734 | 92 | +0.000059 |
| `mlp` | 27 | 156 | 5 | 730 | 96 | +0.000044 |
| `mlp` | 24 | 158 | 3 | 730 | 96 | +0.000075 |

These are not meaningful repairs. The largest rank recovery is only seven
positions, and every patched top1 remains in the same short-box basin.

## Direction Readout

The no-aligner parent has essentially no target-specific extent direction in
the tested late layers:

| site | layer | target delta at 826 | baseline-top1 delta | target margin |
| --- | ---: | ---: | ---: | ---: |
| `self_attn` | 27 | -0.0312 | -0.0938 | +0.0625 |
| `mlp` | 25 | -0.3301 | -0.3320 | +0.0020 |
| `decoder_layer` | 24 | +0.1250 | +0.1875 | -0.0625 |
| `self_attn` | 25 | +0.2500 | +0.3125 | -0.0625 |

The aux checkpoint has small positive target margins in some whole-layer rows,
but they are too weak to overcome the basin:

| site | layer | target delta at 826 | baseline-top1 delta | target margin |
| --- | ---: | ---: | ---: | ---: |
| `decoder_layer` | 27 | +0.2500 | -0.1250 | +0.3750 |
| `decoder_layer` | 25 | +0.3125 | +0.1250 | +0.1875 |
| `decoder_layer` | 26 | +0.2812 | +0.1250 | +0.1562 |
| `mlp` | 27 | +0.1250 | +0.0000 | +0.1250 |

This aux signal is directionally interesting, but the causal patch still leaves
the true coordinate around rank `154..156`.

## Mechanism Read

This is a hard residual FN / extent-basin failure:

1. The object can be named and the first three coordinates can be forced, yet
   the model still chooses a short `y2`.
2. Full-prefix context does not contain a portable y2 repair comparable to the
   earlier positive no-aligner vase same-slot repair.
3. The no-aligner parent and aux checkpoint agree on the failure family even
   though their exact basins differ: no-aligner top1 around `737..739`, aux
   top1 around `730..734`.
4. The aux checkpoint is worse by raw target rank/probability, despite showing
   slightly larger small positive direction margins in whole-layer rows.
5. This pushes the false-negative taxonomy toward at least two mechanisms:
   prefix-state repairable misses, such as the no-aligner vase case, and hard
   extent misses where language guidance and nearby prefix state do not expose
   a usable coordinate basin.

The case does not prove visual blindness. It says the currently tested language
and residual-state interventions cannot make the y2 extent accessible. The next
mechanistic hook should therefore move closer to visual/source evidence for the
book lower edge: image-region attention, visual-token contribution, or a
source-region patch tied to the target box extent.

## Verification

Completed counts for both artifact roots:

```text
condition_row_count = 6
layer_row_count = 174
hidden_delta_row_count = 58
direction_row_count = 12
patch_row_count = 12
```

Verification checked:

- required artifact files exist in both roots;
- condition rows include `desc_x1_y1_x2` y2 target bin `826`;
- target rank remains greater than `100` for both prefixes and both
  checkpoints;
- top1 distance remains at least `87` in the unpatched condition rows;
- max residual-patch rank recovery is only `6` for no-aligner and `7` for aux;
- every patched top1 coordinate remains more than `80` bins from the target.
