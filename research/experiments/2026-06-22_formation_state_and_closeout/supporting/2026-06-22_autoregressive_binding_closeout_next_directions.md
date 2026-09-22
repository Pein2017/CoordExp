---
doc_id: progress.diagnostics.autoregressive_binding_closeout_next_directions_2026_06_22
layer: progress
doc_type: diagnostic-closeout
status: active-branch-steering
evidence_scope: selected-case-hidden-patch-continuation-and-posthoc-taxonomy
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Autoregressive Binding Closeout And Next Directions

## Purpose

This note closes the current continuation/projection probe sequence and revises
the next few implementation directions. It supersedes the older framing where
the main unknown was a missing final descriptor logit or a static descriptor
axis. It does not supersede stable docs or OpenSpec contracts.

The current best framing is:

```text
post-box and pre-x1 row state can fall into the wrong object-binding, cursor, or
coordinate-basin regime. Late layers can translate parts of that state into
descriptor logits, but descriptor repair is not the same as selected-instance
binding, current-box repair, or next-row routing.
```

## Current Process Status

The continuation taxonomy aggregation is complete and recorded.

Code/report commits already in the branch include:

```text
c16658f2 Add continuation taxonomy batch contrast
29f50827 Add continuation taxonomy for pre-x1 probes
d27a5d49 Record pre-x1 component localization
a541b703 Record pre-x1 coordinate surface falsifier
b5cc55da Add pre-x1 basin choice probe
294f4fb8 Add continuation coordinate score trace
3dbaa99d Add path averaged transported projection
7292ddef Add continuation emitted basin labels
f5327f15 Add transported readout projection probe
```

The main completed aggregation artifacts are:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_batch/hidden_patch_continuations_after_29f50827_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_contrast/hidden_patch_continuations_after_29f50827_v1
```

Recorded counts:

```text
batch rows: 1258
paired contrast rows: 1162
source continuation artifacts: 20
```

Read-only audit caveat: `continuation_taxonomy_batch_summary.json` reports
`case_count=3`, while a row scan found two `post_box_boundary_case_id` values.
The progress note and contrast summary also describe two cases. Treat this as a
summary-counter caveat, not as evidence of an unfinished run.

No existing artifact found so far answers the formation-time question over
descriptor onset, descriptor end, `object_ref_end`, `box_start`, pre-x1,
post-x1, box close, and next-object onset. Existing artifacts cover final
post-box or final pre-x1 states and vary layer/component/site, not token
formation time with suffix replay.

## Evidence Now Settled

1. **Descriptor transport is real but insufficient.**

Transported and path-averaged readout directions can isolate a live
descriptor-repair axis. Static descriptor projection is misleading, especially
at earlier late sites. But descriptor-span repair still emits the wrong
same-description spatial basin in the backpack case.

The transported-readout machinery is already implemented as component
projection bases in the hidden causal patch stages. Keep it as a reusable
readout/control surface; do not spend the next cycle on more descriptor-only
variants unless a new geometry or formation result makes that necessary.

Key notes:

```text
progress/diagnostics/2026-06-22_transported_readout_projection_findings.md
progress/diagnostics/2026-06-22_path_avg_transport_projection_findings.md
```

2. **Simple target-bbox output directions are falsified for the descriptor
boundary.**

`target_bbox_tokens_minus_completed_box_tokens` at the post-box descriptor
boundary is tiny and inert in the selected backpack flip. It neither repairs
the descriptor nor moves the generated object toward the selected target box.

Key note:

```text
progress/diagnostics/2026-06-22_geometry_basis_falsifier_findings.md
```

3. **Coordinate logits already choose the wrong family at generated coordinate
slots.**

Score traces show descriptor-repairing rows are not merely suffering from
downstream sampling noise. At x1/x2/y2, the wrong backpack family can be
coordinate top-1 while selected target bins are hundreds of coordinate ranks
away.

4. **Pre-x1 has a positive but state-conditioned basin handle.**

For the backpack case `post_box_boundary_ec16feebe5d087b0`, the paired
baseline-minus-current hidden delta can move target x1 support before x1
emission. The effect is not a generic coordinate-token output direction:
`coord_target_minus_bin:874` and `coord_target_minus_mean:874+862` improve rank
but expose neighboring wrong ridges such as `862`, `858`, or `358`.

Key note:

```text
progress/diagnostics/2026-06-22_pre_x1_basin_choice_probe_findings.md
```

5. **Late isolated `self_attn` and `mlp` deltas are not local sufficient
writers.**

For the selected backpack pre-x1 handle, `layer_input` reproduces most of the
target-rank movement, while isolated `self_attn` and `mlp` outputs do not. This
supports a residual cursor/state carrier interpretation, not a simple "one late
head writes binding" story.

Key note:

```text
progress/diagnostics/2026-06-22_component_basin_label_findings.md
```

6. **Current-box repair and next-row routing are distinct surfaces.**

In open-box pre-x1 continuations, `full_output_delta` first-step can repair the
current open box enough to produce target overlap in one nearest-previous
backpack row. `layer_input_delta` does not locally repair the current open box
there; its positive signal appears later as next-object route movement under
all-steps patching.

Key note:

```text
progress/diagnostics/2026-06-22_continuation_taxonomy_batch_contrast_findings.md
```

## Revised Next Directions

### P0: Formation-Time Layer-Input Map

Use the positive backpack pre-x1 handle as the first anchor. Patch earlier
meaningful token positions and replay the suffix so later layer-input states are
recomputed. Do not patch an already-cached early hidden state without replay;
that would test a local readout lens rather than formation-time causality.

Priority positions:

```text
descriptor onset
descriptor end
object_ref_end
box_start
pre-x1
post-x1
box close
next-object onset
```

Primary question:

```text
when does the late layer-input carrier become a coordinate-basin or next-row
routing state, and why does it not act as a current-coordinate repair state in
the open-box layer-input continuation rows?
```

Readouts:

```text
target rank/prob
wrong-family rank/prob
target-vs-wrong-family margin
top1 bin
overshoot bin
current/open-box taxonomy label
next/generated-object taxonomy label
IoU bands: >0, >=0.1, >=0.5
parse and malformed counters
```

### P0/P1: Path Mediation, Not Isolated Component Blame

The current late component evidence demotes isolated `self_attn` and `mlp`
output sweeps. The next path experiment should patch layer input and then
recompute or clamp attention and MLP outputs, or use a small interventional
Shapley over attention/MLP paths.

The useful contrast is:

```text
patch layer input, recompute attention + MLP
patch layer input, clamp attention output to failed/active
patch layer input, clamp MLP output to failed/active
patch layer input, clamp both
```

Success should be scored by x1 family movement and coherent object span, not by
descriptor exactness alone.

### P1: Family-Aware Coordinate Basin Reducer

Future coordinate analysis should score target versus a wrong family, not a
single wrong bin. For the backpack pre-x1 handle, the local family currently
includes:

```text
target: 420
wrong/neighbor family: 874,862,858,893,426,387,358
```

The reducer should report:

```text
target-vs-family margin
top1 family identity
family substitution such as 874 -> 862/858
exact-bin recovery
near-target overshoot
target-loss versus paired baseline
```

Single wrong-bin coordinate directions are now controls, not the main
mechanism.

### P1: Span-Aware Continuation For Positive Pre-X1 Handles

Rank-1 x1 or first-token rescue is not object-binding repair. Continue to score
both the current open object and the following generated object.

Required taxonomy:

```text
first-token rescue only
descriptor-span rescue
descriptor + valid object box
descriptor + target-overlapping box
wrong same-description family
completed-box basin
next-row transition rescue
malformed or incomplete span
target loss versus paired baseline
```

Keep `first_step` and `all_steps` separate. `all_steps` is a sustained steering
stress test, not natural boundary unfolding evidence.

### P2: Regime Labels Before Low-Rank Or Transfer Claims

Do not run broad low-rank SVD or cross-checkpoint transfer across mixed cases
yet. Current evidence already has different regimes:

```text
backpack: layer-input steerable but current-box/next-row split
tiny object: same broad delta can harm or fail to improve
descriptor-boundary rows: descriptor-only repair
coordinate-output basis rows: wrong-family substitution
value/route panels: local basin perturbation without object-span sufficiency
```

Only after labels stabilize should we test whether any low-rank subspace or
cross-template transfer is real.

## Directions To Demote

Stop spending near-term effort on:

```text
missing-final-descriptor-logit framing
descriptor-token rescue as object-binding repair
static descriptor cosine as semantic information percentage
more descriptor-only transported/path-averaged gradient variants
post-box descriptor-boundary target-bbox output embedding as the missing lever
single wrong-bin coordinate directions as the main explanation
isolated late self-attn or MLP output sweeps as writer localization
downstream argmax/sampling as the main cause after coordinate logits are formed
all-steps continuation as natural boundary unfolding evidence
mean target IoU across mixed baseline/flip roles
low-rank SVD over mixed regimes
broad training intervention from current evidence
```

## Training Gate

Do not launch a training intervention from the current evidence alone. Before
training, require:

1. A replicated pre-x1 or formation-time causal handle beyond one selected
   backpack row, or an explicit case-specific scope.
2. Span-aware continuation evidence that repairs a full valid object, not only
   descriptor or x1 rank.
3. Paired baseline contrast with target gains, target losses, malformed
   regressions, and wrong-family substitutions.
4. Family-aware coordinate-basin readouts.
5. Control families: noop, too-late, wrong-region, wrong-head/same-region,
   norm-matched random, projection-norm random, and repetition-penalty controls
   where relevant.
6. Regime labels before aggregation.
7. Exact artifact provenance: checkpoint, config, selected rows, case ids,
   parse/drop counters, emitted-basin labels, IoU bands, and selector/filter
   behavior.

Dynamic adjustment remains welcome: if a probe exposes a more promising route to
the final mechanism picture, the roadmap should bend around that evidence
rather than preserve a stale checklist.

## Next Implementation Plan

Use:

```text
docs/superpowers/plans/2026-06-22-binding-state-formation-roadmap.md
```

as the next local implementation roadmap. It should be treated as the active
near-term plan for the branch, while older Phase 4 bridge plans remain useful
history.
