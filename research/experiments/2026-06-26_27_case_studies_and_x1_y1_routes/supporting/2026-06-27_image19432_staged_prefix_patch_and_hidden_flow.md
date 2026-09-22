---
doc_id: progress.diagnostics.image19432_staged_prefix_patch_and_hidden_flow
date: 2026-06-27
scope: sample-base mechanistic probe, image_id=19432, same generated prefix, staged coord slots
status: partial mechanism evidence
---

# Image 19432 Staged Prefix Patch And Hidden Flow

## Question

The previous image-19432 y2 route work showed that the duplicate-heavy chair
corridor can be repaired at the final y2 boundary by target-bridge or selected
value-region interventions. This slice asks whether that stiffness already
exists at earlier coordinate slots, or whether the main failure is a late
coordinate-closure program.

The comparison is same-prefix across:

- `random_denoise`
- `sorted_denoise`
- `pure_ce_sorted_natadj`

The two target chairs are:

- gt5: `[351,122,458,348]`
- gt8: `[537,122,651,348]`

## Adapter Contract Fix

Initial activation-patch attempt:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v18_19432_sameprefix_random_l23_l24_l27_components_cont5_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v18_19432_sameprefix_sorted_l23_l24_l27_components_cont5_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v18_19432_sameprefix_purece_l23_l24_l27_components_cont5_gpu2
```

This surfaced a plan-row contract issue. `activation_patch_continuation`
deduplicates selected rows by `state_key`, while staged slot-scaffold plan rows
reuse the same semantic `state_key` across multiple forced-prefix modes. The
v18 run selected only two rows per model, so it is a collapsed x1/onset probe,
not the intended staged panel.

Added helper:

```text
src/analysis/prefix_denoising_surgery_probing/position_plan_adapter.py
scripts/analysis/run_prefix_denoising_position_plan_adapter.py
tests/analysis/test_prefix_denoising_position_plan_adapter.py
```

The adapter promotes `plan_id` into a unique `state_key`, preserves the original
semantic state as `source_state_key`, and infers `probe_position` from staged
mode. This prevents silent row collapse in atlas/activation tools.

Adapted position-row artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan_adapter/v1_19432_sameprefix_random_slot_scaffold
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan_adapter/v1_19432_sameprefix_sorted_slot_scaffold
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan_adapter/v1_19432_sameprefix_purece_slot_scaffold
```

Each has 8 rows:

- 2 target chairs
- `force_original_x1_pre_y1`
- `force_target_x1_pre_y1`
- `force_target_x1_y1_pre_x2`
- `force_target_x1_y1_x2_pre_y2`

## Causal Patch Panel

Activation-patch outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v19_19432_sameprefix_random_staged_l23_l24_l27_components_cont5_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v19_19432_sameprefix_sorted_staged_l23_l24_l27_components_cont5_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v19_19432_sameprefix_purece_staged_l23_l24_l27_components_cont5_gpu2
```

Protocol:

- source rows: adapted slot-scaffold rows above
- layers: 23, 24, 27
- patch sites: `layer_output`, `layer_input`, `self_attn`, `mlp`
- alphas: `0,0.02,0.05,0.1,0.2`
- continuation steps: 5
- target: staged next coordinate token
- scope: 480 rows per model

Aggregate:

| model | rows | rank improved | first-token target | coherent tail |
|---|---:|---:|---:|---:|
| random_denoise | 480 | 299 | 53 | 2 |
| sorted_denoise | 480 | 372 | 31 | 7 |
| pure_ce_sorted_natadj | 480 | 353 | 38 | 3 |

`coherent tail` means the patched first coordinate is the target and the short
continuation matches the expected coordinate tail / box close.

### Slot Findings

#### y1 after forced x1

All three models remain trapped around the `coord_0` y1 basin.

For target-x1 y1 rows (`force_target_x1_pre_y1`):

- random: 0 first-token target, 0 coherent tails
- sorted: 0 first-token target, 0 coherent tails
- pureCE: 0 first-token target, 0 coherent tails

For stale/original-x1 y1 control rows:

- random: 0 first-token target
- sorted: 0 first-token target
- pureCE: 8 first-token targets across gt5/gt8, but 0 coherent tails; patched
  continuations go to wrong local geometry such as
  `<|coord_122|><|coord_79|><|coord_333|><|box_end|>...`

Interpretation: y1 is the hard early vertical-anchor basin. The models often
rank-improve under perturbation, but direct target-direction perturbation does
not reliably cross to the correct y1 top-1, even when x1 is forced to the target.

#### x2 after forced x1,y1

For gt8 x2 (`target=651`), first-token target is easy to expose:

- random: 51 first-token target rows, 0 coherent tails
- sorted: 24 first-token target rows, 0 coherent tails
- pureCE: 27 first-token target rows, 0 coherent tails

For gt5 x2 (`target=458`), all models get rank improvement but no first-token
target:

- random: 44 rank-improved / 60, 0 first-token target
- sorted: 44 rank-improved / 60, 0 first-token target
- pureCE: 43 rank-improved / 60, 0 first-token target

Interpretation: x2 has available target direction, especially for gt8, but
first-token correctness does not imply a coherent object-span program. The next
y2 token remains a nearby duplicate-corridor basin (`342`, `347`, `349`, etc.).

#### y2 after forced x1,y1,x2

This is the only slot where coherent repairs appear.

Coherent tail rows:

| model | gt | repaired rows | route examples |
|---|---:|---:|---|
| random_denoise | 5 | 2 | L27 `layer_output` alpha 0.2; L27 `layer_input` alpha 0.2 |
| sorted_denoise | 5 | 2 | L24 `self_attn` alpha 0.2; L27 `layer_input` alpha 0.05 |
| sorted_denoise | 8 | 5 | L23/L24/L27 `layer_output` or `layer_input`, mostly alpha 0.2 |
| pure_ce_sorted_natadj | 5 | 3 | L27 `layer_output` alpha 0.2; L27 `layer_input` alpha 0.1; L27 `mlp` alpha 0.1 |
| pure_ce_sorted_natadj | 8 | 0 | no coherent tail |

Every coherent repair is a local-basin switch from `340` or `342` to target
`348`, followed by `<|box_end|><|object_ref_start|>chair...`.

Interpretation: the previous y2-bridge result generalizes within the staged
panel. The object is not visually absent at closure. The model can be pushed out
of the local y2 basin into the correct box-close continuation, but this is a
late slot-specific corridor escape, not a full onset repair.

## Component-Flow Labels

Component-flow reducer outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v3_19432_sameprefix_random_staged_v19_matched
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v3_19432_sameprefix_sorted_staged_v19_matched
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v3_19432_sameprefix_purece_staged_v19_matched
```

Object-flow labels:

| model | exact tail program | first-token-only / stale-tail | hard nonrepair |
|---|---:|---:|---:|
| random_denoise | 2 | 51 | 427 |
| sorted_denoise | 7 | 24 | 449 |
| pure_ce_sorted_natadj | 3 | 35 | 442 |

Sorted-denoise is not best by first-token flips; it is best by exact-tail
programs. This is a useful distinction: it suggests prefix denoising may have
changed the late closure program availability more than raw coordinate-token
access.

## Hidden-Vector Flow

Hidden-flow outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v9_19432_sameprefix_random_staged_l23_l24_l27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v9_19432_sameprefix_sorted_staged_l23_l24_l27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v9_19432_sameprefix_purece_staged_l23_l24_l27_gpu2
```

Protocol:

- selected rows: 8 per model
- output rows: 144 per model
- layers: 23, 24, 27
- components: `layer_input_state`, `attention_update`,
  `after_attention_state`, `mlp_update`, `layer_output_state`,
  `layer_delta_update`
- readout direction: target coordinate token minus antagonist coordinate token

Mean target-antagonist margin by slot:

| model | slot | layer/output state | attention update | mlp update | main read |
|---|---|---:|---:|---:|---|
| random | y1 | -1.224 | +2.012 | -1.301 | attention has some target direction, residual/MLP state opposes it |
| sorted | y1 | -0.676 | +0.782 | -0.061 | weaker opposition than pureCE, but still not enough to cross |
| pureCE | y1 | -3.390 | -1.381 | -0.984 | strongest y1 anti-target basin |
| random | x2 | +4.092 | +2.095 | +2.997 | x2 target direction is available |
| sorted | x2 | +4.878 | +1.741 | +2.888 | x2 target direction is available |
| pureCE | x2 | +3.149 | +2.085 | +3.244 | x2 target direction is available |
| random | y2 | +3.607 | +1.801 | -1.057 | closure positive in state, MLP opposes |
| sorted | y2 | +3.793 | +4.010 | +0.559 | attention route strongly supports closure |
| pureCE | y2 | +9.305 | +5.321 | +6.246 | raw y2 direction large, but causal repair less distributed |

Key hidden-flow interpretation:

1. y1 is the true early basin. Across checkpoints, the final state for y1 is
   still negative or weak in the target-vs-antagonist readout direction. This
   explains why y1 rank can improve without a top-1 crossing.
2. x2 is not the main information bottleneck. It has positive readout direction
   in all models, and gt8 can be made first-token correct often, but the
   continuation falls into the wrong y2 basin.
3. y2 closure separates raw direction magnitude from usable route structure.
   PureCE has the largest y2 readout margins, yet sorted-denoise has more
   coherent causal repairs and repairs both chairs. Sorted-denoise's y2 repairs
   include L24 self-attn and L23/L24/L27 state routes, suggesting a broader
   closure bridge rather than merely larger coordinate logits.

## Mechanistic Picture After This Slice

The same-prefix corridor contains enough object and coordinate information for
late repair. The failure is not "the model cannot perceive the chair."

The deeper split is:

- early y1: target vertical anchor is trapped behind an anti-target/coord0 basin;
- x2: coordinate identity is accessible, especially for gt8, but not sufficient
  to bind a full tail;
- y2: local closure basin (`340/342/347/349` vs target `348`) is causally
  movable, and sorted prefix-denoising exposes the most coherent repair routes.

So the duplication corridor looks less like a single missing-object failure and
more like a staged autoregressive program failure:

1. repeated chair history makes the next object's vertical anchor default to
   a stale/local y1 basin;
2. later slots can recover partial coordinate identity;
3. final closure chooses among nearby y2 basins and may need a route-specific
   bridge, not just more coordinate-token mass.

## Next Work

1. Run route attribution for y1 and x2 using the adapted staged rows. y1 should
   be treated as the main onset-basin target, not merely a precursor to y2.
2. For y1, compare prompt/non-image, current forced coords, recent objects, and
   visual-token value routes. The goal is to locate the source of the coord0
   attractor.
3. For y2, compare sorted-denoise repair routes against pureCE high-margin but
   lower-repair routes. The key question is why raw target readout can be large
   without a robust causal closure bridge.
4. Repeat this staged panel on at least one training image and one non-chair
   duplication corridor to test whether the y1-onset/y2-closure split is local
   to image 19432 or a general repeated-object mechanism.

Verification:

```text
pytest -q tests/analysis/test_prefix_denoising_position_plan_adapter.py
# Pytest: 2 passed
```
