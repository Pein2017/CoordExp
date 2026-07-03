---
doc_id: progress.diagnostics.image19432_x2_route_group_span_binding
date: 2026-06-27
scope: sample-base mechanistic probe, image_id=19432, same generated prefix, x2 transition after forced target x1/y1
status: partial mechanism evidence
---

# Image 19432 X2 Route-Group Span Binding

## Question

The y1 route-group study showed a split between basin entry and span binding:
selected route heads can move y1 toward the target coordinate, but exact y1
does not reliably produce the remaining target-object box. This note asks the
next transition question: after forcing target x1 and y1, can x2 route surgery
bind the span, or does the model still fall into the duplicate y2 corridor?

This slice focuses on gt5 because it gives a clean x2 contrast across all three
models:

```text
target box: [351,122,458,348]
forced prefix: <|coord_351|><|coord_122|>
target x2: 458
baseline x2 top1: 467
route direction: coord458 - coord467
```

For gt8, x2 is already top-1 correct in random, and the more relevant failure
is y2/closure. That case belongs to the y2 closure lane.

## Artifacts

Derived x2 plan rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v5_sameprefix_19432_random_x2_gt5_contrast467
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v5_sameprefix_19432_sorted_x2_gt5_contrast467
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v5_sameprefix_19432_purece_x2_gt5_contrast467
```

Route attribution outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v6_sameprefix_19432_random_x2_gt5_contrast467_layers22_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v6_sameprefix_19432_sorted_x2_gt5_contrast467_layers22_27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v6_sameprefix_19432_purece_x2_gt5_contrast467_layers22_27_gpu2
```

Route reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v6_sameprefix_19432_x2_gt5_route_reduction.md
```

Route-group intervention outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v6_sameprefix_19432_random_x2_gt5_group_top2_bridge_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v6_sameprefix_19432_sorted_x2_gt5_group_top2_bridge_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v6_sameprefix_19432_purece_x2_gt5_group_top2_bridge_gpu2
```

Intervention reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v6_sameprefix_19432_x2_gt5_group_reduction.md
```

All route and intervention runs completed with `error_count=0`.

## Route Structure

Mean route contribution by region:

| model | prompt non-image | current forced coords | current partial object | recent 16 | all prefix |
|---|---:|---:|---:|---:|---:|
| random | -0.2084 | +0.0721 | +0.0654 | +0.0645 | -0.1591 |
| sorted | -0.3172 | +0.0799 | +0.0791 | +0.0813 | -0.2567 |
| pureCE | -0.1658 | -0.0175 | -0.0214 | -0.0219 | -0.2216 |

Strong route heads:

- random:
  - positives: `L25H13`, `L25H6`, `L26H12`, `L26H5`, `L24H11`
  - negatives: `L24H6`, `L25H7`, `L22H1`, `L24H14`
- sorted:
  - positives: `L26H5`, `L25H6`, `L25H13`, `L26H12`, `L23H6`
  - negatives: `L24H6`, `L25H10`, `L25H7`, `L24H10`
- pureCE:
  - huge positive current route: `L26H12`
  - strong current negatives: `L27H14`, `L27H2`
  - prompt/template negatives include `L24H6`

Interpretation: x2 has clearer target direction than y1. Random and sorted
show positive current-coordinate evidence but strong prompt/template opposition.
PureCE is more current-route dominated: a large `L26H12` positive opposes
localized late negatives at `L27H14` / `L27H2`.

## Group Surgery

Protocol:

- one gt5 plan row per model
- top-2 positive and top-2 negative route heads from current forced coords and
  prompt/template regions
- positive scale `2.0`; negative scale `0.0`
- continuation steps: 5
- bridge alphas: `0.05,0.1,0.2`

### Direct rows

| model | direct rows | exact x2 flips | within radius 1 | rank improved | main behavior |
|---|---:|---:|---:|---:|---|
| random | 7 | 0 | 1 | 5 | best row moves `467 -> 459`, rank `19 -> 2` |
| sorted | 7 | 0 | 0 | 5 | best rows move `467 -> 460`, rank `21 -> 2` |
| pureCE | 7 | 0 | 0 | 6 | best row moves `467 -> 461`, rank `30 -> 9` |

Representative direct rows:

```text
random all-signed combo:
<|coord_459|><|coord_342|><|box_end|>

sorted template/all-signed combo:
<|coord_460|><|coord_337|><|box_end|>

pureCE all-signed combo:
<|coord_461|><|coord_343|><|box_end|>
```

Direct surgery moves x2 near the target and often improves target rank sharply,
but it does not exactly flip to `458`.

### Bridge rows

| model | bridge rows | exact x2 flips | within radius 1 | continuation labels |
|---|---:|---:|---:|---|
| random | 21 | 0 | 18 | near-first-token only |
| sorted | 21 | 2 | 19 | 2 first-token-only, otherwise near-first-token |
| pureCE | 21 | 0 | 21 | near-first-token only |

Sorted exact x2 bridge flips:

```text
<|coord_458|><|coord_342|><|box_end|>
```

These are first-token-only repairs. The next coordinate remains the duplicate
y2 corridor (`342`), not target y2 `348`.

## Mechanistic Read

The x2 transition confirms the y1 finding and makes the mechanism picture more
modular:

1. **Coordinate entry is locally movable.** x2 is much easier than y1:
   direct surgery reaches near-target coordinates and rank-2 readouts in
   random/sorted.
2. **Exact coordinate readout still has a quantized barrier.** Even when the
   target direction is strong, direct attention route surgery usually lands at
   neighboring bins (`459`, `460`, `461`) rather than exact `458`.
3. **Span binding is downstream of coordinate entry.** Exact or near-exact x2
   does not force correct y2. The model still emits `342/343` and closes the
   box, reproducing the duplicate corridor.

This is evidence against a single "missing object perception" explanation.
The model can expose local coordinate directions, but the autoregressive
object-span program is not recomputed globally after a local coordinate repair.
It behaves more like a locally editable coordinate cursor plus a sticky
continuation program.

## Updated Hypothesis

For image 19432, the duplicate-chair failure appears to decompose into:

- **onset basin**: y1 strongly attracted to coord0 / old local anchor;
- **coordinate cursor**: x2/y2 coordinates have local target-direction evidence
  and can be moved near/exact by surgery;
- **span program**: after a coordinate is moved, the next coordinates often
  remain controlled by the original duplicate corridor.

The next most valuable experiment is a chained or conditional surgery:

1. force or bridge x1/y1/x2 to the target path,
2. inspect the y2 route immediately afterward,
3. patch only the y2 closure route and ask whether box close plus object-ref
   continuation becomes coherent.

The existing y2 closure repairs suggest this may be where prefix-denoising
actually helps: sorted-denoise has more coherent late closure repairs even
though y1/x2 entry still shows strong basin barriers.
