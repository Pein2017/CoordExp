# Auxiliary Checkpoint Negative-Wide Control-Anchor Rows

Date: 2026-06-11

## Scope

This note drills into the row-level cases queued by the cross-checkpoint coordinate
band-flow contrast. The target is `aux_latest_ckpt32`, record `33`,
`post_y1/pre_x2`, where several components had patched top-1 in the `control_anchor`
band but negative wide-radius target mass deltas.

Source rows:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl`

The central question is whether these "control-like" patched top-1 moves are actually
semantically correct basin relocations, or whether they are nearby coordinate attractors
that compete with exact target mass.

## Counts

Scope: `aux_latest_ckpt32`, record `33`, phase `post_y1/pre_x2`.

| component | rows | negative wide-radius rows | negative-wide + control-top1 rows |
|---|---:|---:|---:|
| duplicate_basin | 12 | 7 | 4 |
| visual_near_ring | 12 | 6 | 2 |
| visual_far_background | 12 | 7 | 4 |
| non_region_complement | 12 | 7 | 2 |
| whole_head | 12 | 10 | 7 |

`whole_head` is the clearest abnormal case: it is top-1 control-like in 9/12 rows, but
7 of those control-like rows have negative wide-radius target mass.

## Distance Summary

For `whole_head` negative-wide + control-top1 rows:

- rows: `7`
- mean patched top1 minus target: `-6.143`
- mean control top1 minus target: `-6.286`
- mean masked top1 minus target: `1.000`
- patched top1 below target: `7/7`
- masked top1 exact target: `2/7`
- mean wide-radius delta: `-0.068069`
- mean control-anchor delta: `0.081415`
- mean old-anchor delta: `-0.081394`

For `duplicate_basin` negative-wide + control-top1 rows:

- rows: `4`
- mean patched top1 minus target: `-7.500`
- mean control top1 minus target: `-9.250`
- mean masked top1 minus target: `0.500`
- patched top1 below target: `4/4`
- masked top1 exact target: `2/4`
- mean wide-radius delta: `-0.084096`
- mean control-anchor delta: `0.132236`
- mean old-anchor delta: `-0.114923`

The signature is therefore an undershoot: the patch moves mass into the lower control
anchor while losing target-near mass. The masked state is often already closer to the
target than the patched/control top-1 state.

## Key Rows

### Row 20, target `165`, generated `<|coord_339|>`, next `<|coord_165|>`

`duplicate_basin`:

- masked top1: `165`, exact target
- patched top1: `158`, control anchor
- control top1: `151`, control anchor
- patched top1 distance: `7`
- control top1 distance: `14`
- wide-radius delta: `-0.034275`
- exact-target delta: `-0.003108`
- control-anchor delta: `0.160052`
- old-anchor delta: `-0.154486`

`whole_head`:

- masked top1: `165`, exact target
- patched top1: `158`, control anchor
- control top1: `151`, control anchor
- patched top1 distance: `7`
- control top1 distance: `14`
- wide-radius delta: `-0.079195`
- exact-target delta: `-0.037540`
- control-anchor delta: `0.192730`
- old-anchor delta: `-0.154486`

Interpretation: patching actively moves away from an already-correct masked top1 into a
lower control basin.

### Row 21, target `170`, generated `<|coord_343|>`, next `<|coord_170|>`

`duplicate_basin` and `whole_head` have the same top-1 pattern:

- masked top1: `170`, exact target
- patched top1: `158`, control anchor
- control top1: `158`, control anchor
- patched/control top1 distance: `12`
- wide-radius delta: `-0.183812`
- exact-target delta: `-0.036757`
- old-anchor delta: `-0.147055`
- control-anchor delta: `0.219922` for `duplicate_basin`, `0.216171` for `whole_head`

Interpretation: this is the cleanest example that "control anchor" is not synonymous
with correct next-coordinate recovery. The control basin itself is lower than the exact
target.

### Row 23, target `163`, generated `<|coord_354|>`, next `<|coord_163|>`

`whole_head`:

- masked top1: `170`, old anchor, distance `7`
- patched top1: `158`, control anchor, distance `5`
- control top1: `158`, control anchor, distance `5`
- wide-radius delta: `-0.101951`
- exact-target delta: `-0.026207`
- control-anchor delta: `0.082937`
- old-anchor delta: `-0.086063`

Interpretation: patched top1 is slightly closer by absolute distance, but the patch still
loses exact/wide target-near mass. It is a competing lower attractor, not a full target
commitment.

### Row 24, target `157`, generated `<|coord_350|>`, next `<|coord_157|>`

`whole_head`:

- masked top1: `170`, old anchor, distance `13`
- patched top1: `151`, control anchor, distance `6`
- control top1: `158`, control anchor, distance `1`
- wide-radius delta: `-0.020229`
- exact-target delta: `0.005225`
- control-anchor delta: `0.053911`
- old-anchor delta: `-0.109798`

Interpretation: this is mixed. The patch improves top1 distance and exact-target mass,
but still lowers wide-radius mass because it concentrates into lower control/bridge bins.
This should not be collapsed with the row-20/21 damage pattern.

### Row 28, target `156`, generated `<|coord_351|>`, next `<|coord_156|>`

`whole_head`:

- masked top1: `151`, control anchor, distance `5`
- patched top1: `151`, control anchor, distance `5`
- control top1: `150`, control anchor, distance `6`
- wide-radius delta: `-0.078994`
- exact-target delta: `0.000000`
- control-anchor delta: `0.037943`
- old-anchor delta: `-0.072087`

Interpretation: top1 band identity does not change, but mass is pulled out of the wide
target neighborhood into the lower control-anchor/bridge side.

## Mechanism Update

For the auxiliary-loss checkpoint, the "control-like" band-flow pattern is not a clean
repair. It often means:

1. old/duplicate mass is removed;
2. top-1 snaps into a lower control anchor;
3. exact or wide target-near mass decreases;
4. masked top1 may already be closer to the next target coordinate than the patched
   top1.

This is a distinct mechanism from `none_latest_ckpt32`, where `whole_head` simultaneously
removed old-anchor mass and increased exact/wide/control-neighborhood mass. In
`aux_latest_ckpt32`, the auxiliary loss appears to reshape the coordinate basin so that
the control state can itself be an undershooting attractor.

The practical warning is that band labels must be interpreted row-locally: `control_anchor`
means "near the control-state top-k basin", not automatically "correct target basin."
For auxiliary-loss mechanics, control-anchor movement can be damaging when the control
state is offset below the next coordinate.

## Next Deterministic Step

For the auxiliary checkpoint, split the `control_anchor` band into below-target and
above-target sub-bands. Then report mass flow separately for:

- lower-control undershoot;
- upper-control overshoot;
- exact target;
- target-near bridge.

That would remove the current ambiguity where control-like top-1 movement hides
directional coordinate errors. This can be done post-hoc from the existing enriched
band-flow rows.

## Verification

- Read the committed 410-row enriched band-flow JSONL.
- Filtered to `aux_latest_ckpt32`, record `33`, `post_y1/pre_x2`.
- Counted negative wide-radius rows and negative-wide + control-top1 rows by component.
- Computed top1-minus-target distance summaries for `duplicate_basin` and `whole_head`.
- No new code or generated artifacts were required for this note.
