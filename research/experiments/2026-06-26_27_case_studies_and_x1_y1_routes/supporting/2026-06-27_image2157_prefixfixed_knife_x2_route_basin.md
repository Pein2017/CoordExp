# Image 2157 Prefix-Fixed Knife x2 Basin Probe

## Scope

This note records the corrected image 2157 knife `x2` probe after forcing the
prefix to include target `x1=338,y1=729`. The target continuation is:

```text
<|coord_639|><|coord_991|><|box_end|>
```

The evidence scope is one high-value image-base/object slice:

- image id: `2157`
- target object: `knife`
- target bbox: `[338,729,639,991]`
- random force-target-x1-y1 generated box: `[338,729,467,987]`
- sorted force-target-x1-y1 generated box: `[338,729,641,987]`
- pureCE force-target-x1-y1 generated box: `[338,729,638,989]`

## Tooling Correction

The initial v7/v8 x2 plan rows were mispositioned. They set metadata such as
`forced_coord_bins=[338,729]` and `target_next_coord_slot=x2`, but the actual
`trajectory_prefix_text` and `assistant_prefix_text` still ended at:

```text
<|object_ref_start|>knife<|object_ref_end|><|box_start|>
```

Therefore v7/v8 tested a box-start decision with target coord639, not a true
post-y1/pre-x2 state. Those artifacts are useful only as a cautionary
mispositioned control, not as x2 mechanism evidence.

Corrected v9 plan rows materialized the forced prefix text:

```text
<|object_ref_start|>knife<|object_ref_end|><|box_start|><|coord_338|><|coord_729|>
```

Corrected route plan paths:

- random:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v9_2157_random_knife_x2_after_x1y1_prefixfixed_contrast467/position_plan_rows.jsonl`
- sorted:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v9_2157_sorted_knife_x2_after_x1y1_prefixfixed_contrast641/position_plan_rows.jsonl`
- pureCE:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v9_2157_purece_knife_x2_after_x1y1_prefixfixed_contrast638/position_plan_rows.jsonl`

The group-intervention builder was also widened from only
`current_forced_coords` to:

```python
("current_box_start", "current_forced_coords", "current_partial_object")
```

This matters for cross-slot probes because important local route evidence can
live on the current box marker or current partial-object span, not only on
already forced coordinate tokens.

## Corrected v9 Route Evidence

Route reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v9_2157_knife_x2_after_x1y1_prefixfixed_route_reduction.md
```

Random, target `639` vs failed contrast `467`:

- Mean current forced-coordinate route is slightly anti-target:
  `current_forced_coords = -0.0379`, `current_partial_object = -0.0590`,
  `recent_16 = -0.0757`.
- Mean prompt/template route is positive: `prompt_non_image = +0.1401`.
- Strongest positive head-region is mostly template/prompt:
  `L26H5 prompt_non_image = +10.3113`, sourced heavily from prior
  `<|box_end|>` tokens.
- Strongest local negative head-region is `L23H6`, sourced from the forced
  y1 token:
  `current_forced_coords = -4.0782`, top token `<|coord_729|>`.

Sorted, target `639` vs near-target contrast `641`:

- Mean current evidence is mildly positive:
  `current_forced_coords = +0.0438`, `current_partial_object = +0.0410`,
  `recent_16 = +0.0425`.
- Strong positive current heads include `L23H6` and `L27H15`, both reading
  the forced y1 token.

PureCE, target `639` vs near-target contrast `638`:

- Mean current evidence strongly favors the near-neighbor contrast:
  `current_forced_coords = -0.5417`, `current_partial_object = -0.5400`.
- The dominant anti-target route is `L26H12`, almost entirely from
  `<|coord_729|>`:
  `current_forced_coords = -20.7541`.
- This is not necessarily a bad-mechanism signal; contrast `638` is already
  one bin from the target `639`.

## Corrected v9 Group/Surgery Evidence

Group reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v9_2157_knife_x2_after_x1y1_prefixfixed_routecontrib_group_reduction.md
```

This run selected sites using `value_source_contribution_projection` and used
top-2 signed route groups with readout bridge alphas `0.05,0.1,0.2`.

Random:

- Direct route-group surgery never fixes x2. All seven direct rows preserve a
  valid box tail and y2 near target (`987`, distance 4 from `991`), but x2
  remains wrong:
  - best direct template rows move `467 -> 528`, still 111 bins away.
  - current rows can worsen to `371/369`.
- Bridge surgery exposes the missing x2 as a readout/basin gating issue:
  - `current_negative_suppress_top2 alpha=0.2` emits
    `<|coord_639|><|coord_987|><|box_end|>` exactly at x2, with x2 rank 1.
  - `template_positive_amplify_top2 alpha=0.1`,
    `template_signed_combo_top2 alpha=0.1`, and
    `template_signed_combo_top2 alpha=0.2` also emit exact x2 `639`.
  - Across bridge rows: 4 exact x2 repairs, 12 within one bin, all 21 keep y2
    within 4 bins and emit `<|box_end|>` as the third token.

Sorted:

- Baseline/non-noop first token is already near target: `<|coord_641|>`.
- Direct current positive can emit `<|coord_638|><|coord_987|><|box_end|>`,
  while current signed/all signed direct combos can destabilize into
  non-coordinate text (`ChangeLane...`), showing that combining positive and
  negative current sites is not automatically benign.
- Bridge rows mostly converge to x2 `638` with y2 `987`.

PureCE:

- Baseline/non-noop first token is already near target: `<|coord_638|>`.
- Direct rows mostly keep `638/989` or shift to `628/989`; all keep a valid
  box tail.
- Bridge rows can flip exactly to `639/989/box_end`, especially when
  suppressing the strong current negative `L26H12` route.

## Interpretation

For image 2157, the random checkpoint's knife failure after forcing x1/y1 is
not explained by inability to carry y2 or terminate a box. The y2 continuation
is stable near `987` across direct and bridge interventions. The fragile part
is the x2 slot basin: the model has accessible target-compatible evidence, but
the unassisted route groups do not lift it over the local readout/basin
threshold. Small readout bridges can switch x2 exactly to `639` without
destroying the y2/box-end tail.

This makes image 2157 a complementary case to image 19432:

- 19432 showed that coordinate entry can be nudged but the downstream duplicate
  corridor can remain sticky.
- 2157 shows the reverse: the downstream y2/tail is stable once x2 is fixed,
  and the main failure is x2 basin entry/readout gating.

Together, these support a more granular mechanism picture: false negatives and
duplicate-like failures are not a single visual-perception failure mode. The
model can have target object and tail information available while the current
coordinate slot is controlled by a small number of late route/readout gates
that decide whether the span enters the correct coordinate basin.

## Follow-Up Tooling

After this note's first draft, the route-group intervention tooling was updated
with a slot-aware continuation annotation helper. For `x1`, `y1`, and `x2`
route probes, it now derives expected follow-up tokens from `target_bbox` and
`target_next_coord_slot` instead of relying on y2-specific labels. Future group
rows should explicitly classify:

- first token: target slot coordinate, here x2;
- second token: follow-up coordinate, here y2;
- third token: expected wrapper, here `<|box_end|>`.

A tiny smoke run verified the new fields on the corrected random v9 plan:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v10_smoke_2157_random_knife_x2_prefixfixed_slotaware_onegroup_gpu0/anchor_escape_y2_route_group_intervention_rows.jsonl
```

The smoke rows contain:

- `continuation_probe_slot = "x2"`
- `expected_followup_token_texts = ["<|coord_991|>", "<|box_end|>"]`
- `continuation_repair_label = "first_token_not_repaired"` for the sampled
  non-repair rows
