---
doc_id: progress.diagnostics.image19432_y1_route_group_basin_entry
date: 2026-06-27
scope: sample-base mechanistic probe, image_id=19432, same generated prefix, y1 onset route-group surgery
status: partial mechanism evidence
---

# Image 19432 Y1 Route-Group Basin Entry

## Question

The staged patch panel in
`progress/diagnostics/2026-06-27_image19432_staged_prefix_patch_and_hidden_flow.md`
showed that the hard failure is earlier than the final y2 closure: after x1 is
forced, all three models still prefer the `coord_0` y1 basin. This note asks
whether the y1 failure is causally carried by a small set of attention
value-route heads, or whether those heads are only correlated with a deeper
readout / state-machine barrier.

The setting is same-prefix image `19432`, target y1 `122`, contrast coord `0`,
across:

- `random_denoise`
- `sorted_denoise`
- `pure_ce_sorted_natadj`

The target chairs remain:

- gt5: `[351,122,458,348]`
- gt8: `[537,122,651,348]`

## Tooling Fixes

Two generic-coordinate fixes were needed before running this as a y1 probe:

1. `anchor_escape_y2_route_probe.py` now accepts generic route target/contrast
   coordinates via `route_target_coord_bin` and `route_contrast_coord_bin`,
   falling back to the legacy `target_y2_bin` / `generated_y2_bin` fields.
2. `anchor_escape_y2_route_group_intervention.py` now uses the same generic
   route target/contrast fields for readout, plan matching, and exact-target
   flip detection. v5 rows also preserve `source_position_state_key`, so the
   results can be split by staged prefix mode instead of blurred across rows.

Tests:

```text
pytest -q tests/analysis/test_prefix_denoising_anchor_escape_y2_route_group_intervention.py tests/analysis/test_prefix_denoising_anchor_escape_y2_route_probe.py tests/analysis/test_prefix_denoising_position_plan_adapter.py
```

Result:

```text
18 passed
```

## Route Evidence

Y1 route rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v4_sameprefix_19432_random_y1_contrast0_layers22_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v4_sameprefix_19432_sorted_y1_contrast0_layers22_27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v4_sameprefix_19432_purece_y1_contrast0_layers22_27_gpu2
```

Reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v4_sameprefix_19432_y1_route_reduction.md
```

Mean value contribution toward target y1 versus coord0 by model / staged mode:

| model | mode | current forced coords | current partial object | recent 16 | all prefix |
|---|---|---:|---:|---:|---:|
| random | target-x1 | +0.0598 | +0.0773 | +0.0770 | +0.2376 |
| random | stale/original-x1 | +0.1033 | +0.1030 | +0.0975 | +0.2732 |
| sorted | target-x1 | -0.0358 | -0.0378 | -0.0728 | +0.0877 |
| sorted | stale/original-x1 | -0.0201 | -0.0223 | -0.0797 | -0.1327 |
| pureCE | target-x1 | -0.3078 | -0.3365 | -0.3503 | -0.3432 |
| pureCE | stale/original-x1 | -0.2666 | -0.3011 | -0.3332 | -0.3152 |

Strong heads:

- random has target-carrying current heads `L27H3` and `L24H8`; the strongest
  prompt/template positive is `L24H13`.
- sorted has current positives around `L27H2` / `L23H7`, current negatives
  `L25H8` / `L25H0`, and large prompt positives `L25H13`, `L26H4`, `L26H5`.
- pureCE has a very strong local anti-target route in `L27H15` over current
  forced / partial / recent regions, with `L23H6` as a second negative route.
  Positive current heads include `L25H9`, `L25H0`, and `L26H9`.

Interpretation: y1 has measurable precursor route structure, but the three
models organize it differently. PureCE's y1 anti-target basin is the most
localized and strongest; prefix-denoise models distribute the opposition more
across current and prompt/template routes.

## Group Surgery

Derived y1 plan rows with explicit coord0 contrast:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention_plan/v1_sameprefix_19432_random_y1_contrast0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention_plan/v1_sameprefix_19432_sorted_y1_contrast0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention_plan/v1_sameprefix_19432_purece_y1_contrast0
```

v5 surgery outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v5_sameprefix_19432_random_y1_group_top2_bridge_statekey_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v5_sameprefix_19432_sorted_y1_group_top2_bridge_statekey_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v5_sameprefix_19432_purece_y1_group_top2_bridge_statekey_gpu2
```

Reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v5_sameprefix_19432_y1_group_statekey_reduction.md
```

Protocol:

- 4 y1 plan rows per model: gt5/gt8 crossed with target-x1 and
  stale/original-x1.
- 28 route-group specs per model.
- Direct group perturbation:
  - amplify top positive current/template heads with scale `2.0`
  - suppress top negative current/template heads with scale `0.0`
  - include signed current/template/all combos.
- Optional post-group readout bridge alphas: `0.05,0.1,0.2`.
- Continuation steps: 5.

All three runs completed with `error_count=0`, `row_count=116`.

## Findings

### Direct route-group surgery does not cross exact y1

Direct non-bridge rows:

| model | mode | gt | n | exact y1 flips | within radius 1 | rank improved | mean margin delta |
|---|---|---|---:|---:|---:|---:|---:|
| random | target-x1 | gt5 | 7 | 0 | 1 | 6 | +0.821 |
| random | target-x1 | gt8 | 7 | 0 | 0 | 6 | +0.393 |
| random | stale/original-x1 | gt5 | 7 | 0 | 0 | 6 | +0.375 |
| random | stale/original-x1 | gt8 | 7 | 0 | 0 | 6 | +0.375 |
| sorted | target-x1 | gt5 | 7 | 0 | 0 | 5 | +0.446 |
| sorted | target-x1 | gt8 | 7 | 0 | 0 | 7 | +0.964 |
| sorted | stale/original-x1 | gt5 | 7 | 0 | 0 | 7 | +1.179 |
| sorted | stale/original-x1 | gt8 | 7 | 0 | 0 | 7 | +1.179 |
| pureCE | target-x1 | gt5 | 7 | 0 | 0 | 6 | +0.964 |
| pureCE | target-x1 | gt8 | 7 | 0 | 0 | 1 | +0.929 |
| pureCE | stale/original-x1 | gt5 | 7 | 0 | 0 | 7 | +0.554 |
| pureCE | stale/original-x1 | gt8 | 7 | 0 | 0 | 7 | +0.554 |

Direct surgery moves rank and target-vs-coord0 margin, but exact y1 remains
behind a stronger basin/readout barrier. The largest direct movement is
near-target for random target-x1 gt5:

```text
<|coord_121|><|coord_467|><|coord_342|><|box_end|>
```

This is one bin below the target y1 and then continues through the duplicate
corridor. It is not a coherent object-span repair.

### Bridge rows expose basin-entry but not span binding

Post-group readout bridge rows:

| model | mode | exact y1 flips | within radius 1 | dominant continuation label |
|---|---|---:|---:|---|
| random | target-x1 | 0 | 30 / 42 | near first token only |
| random | stale/original-x1 | 0 | 42 / 42 | near first token only |
| sorted | target-x1 | 0 | 37 / 42 | near first token only |
| sorted | stale/original-x1 | 2 | 24 / 42 | mostly near first token, 2 first-token-only |
| pureCE | target-x1 | 0 | 24 / 42 | near first token or not repaired |
| pureCE | stale/original-x1 | 10 | 26 / 42 | 10 first-token-only |

The exact y1 bridge flips happen only for stale/original-x1 controls in sorted
and pureCE. The real target-x1 rows do not exact-flip under the bridge, although
many rows land at `coord_121`.

Example pureCE stale/original-x1 exact first-token flip:

```text
<|coord_122|><|coord_93|><|coord_333|><|box_end|>
```

Example sorted stale/original-x1 exact first-token flip:

```text
<|coord_122|><|coord_86|><|coord_290|><|box_end|>
```

These are first-token-only repairs. The next coordinate is not the target x2,
so the object span is not bound.

### Mechanistic Read

The evidence separates three mechanisms that were easy to conflate:

1. **Visual / object availability**: the target chair is not absent. Later y2
   patching and x2 readouts show target geometry is accessible.
2. **Y1 basin entry**: attention value-route heads carry measurable target-y1
   direction. Group surgery improves margins and often moves the top coordinate
   from coord0 to coord121/near-target after a bridge.
3. **Autoregressive span binding**: even when the first y1 token is forced or
   bridged to target, the following coordinates remain a duplicate/stale local
   program. Correct y1 alone does not instantiate the full target-object span.

This supports a two-stage failure picture for image 19432:

- the model can perceive enough geometry to expose the target direction;
- y1 onset is trapped in a coord0 / local-anchor basin;
- after y1 is crossed, a second transition must bind the new object span into
  x2/y2. That transition is not solved by local y1 route surgery.

Prefix denoising appears to reshape where the y1 evidence and opposition live,
but not to fully solve the onset-to-span-binding transition:

- sorted-denoise has more distributed route structure and better late y2
  closure repairs, but target-x1 y1 still does not exact-flip.
- pureCE has the clearest localized anti-target route (`L27H15`), and
  suppressing it helps margin, but exact y1 repair mainly appears in stale
  controls under an additional readout bridge.
- random can be moved to coord121 in target-x1 gt5 direct combo surgery, but
  still falls into the duplicate y2 corridor.

## Next Probe

The next high-value slice is not another aggregate metric. It should trace the
transition after a successful or near-successful y1:

1. Use `force_target_x1_y1_pre_x2` rows to run generic x2 route attribution
   and route-group intervention with contrast equal to the observed/stale x2
   basin.
2. Compare rows where y1 is naturally coord0, directly moved to coord121, or
   bridged/forced to coord122.
3. Ask whether x2 failure is a separate span-binding gate or simply the same
   local-anchor basin propagated one token later.

This is now the more promising path than repeatedly trying to flip y1 alone.
