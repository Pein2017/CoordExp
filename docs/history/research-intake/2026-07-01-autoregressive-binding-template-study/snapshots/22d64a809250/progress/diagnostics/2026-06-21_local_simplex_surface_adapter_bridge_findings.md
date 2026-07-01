---
doc_id: progress.diagnostics.local_simplex_surface_adapter_bridge_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-three-state-y2-local-simplex-surface-summary
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Local Simplex Surface Adapter Bridge Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_adapter_surface_probe_decomposition_findings.md
progress/diagnostics/2026-06-21_source36_combined_value_region_coord_probe_findings.md
```

The prior adapter-surface probe showed that source36/object14/y2 has an
adapter-amplified local ridge, while source123/object5/y2 and
source145/object20/y2 are different failure modes. The missing operational
piece was a compact post-hoc reducer that scores the target against the
selected local or contrastive coordinate family, not just against full-vocab
rank or one highlighted competitor.

Evidence remains tiny: three selected y2 states, layers `20,24,28`, and
post-hoc reduction of existing hidden-state readout rows. No model
perturbation or training was run for this note.

## Helper Added

The coordinate threshold hidden-state probe now has a post-hoc reducer:

```text
scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py \
  --stage summarize-local-simplex-surfaces \
  --hidden-state-readout-rows <trajectory_hidden_state_readout_rows.jsonl> \
  --output-root <summary_root>
```

It consumes rows that already contain per-bin probe fields from
`trajectory-hidden-state-readout` and writes:

```text
local_simplex_surface_summary.json
local_simplex_surface_summary.md
```

For each state and layer, it records target-vs-competitor-family margins for:

```text
layer_coord logits
pre-adapter base surface scores
token_embeddings_adapter effective surface scores
adapter delta scores
```

The main fields are:

```text
*_top_bin
*_target_rank_family
*_target_minus_max_competitor
*_target_minus_mean_competitor
effective_minus_base_target_minus_max
effective_minus_base_target_minus_mean
```

This is deliberately post-hoc, so we can repeatedly change the family scoring
without rerunning GPU hidden-state extraction when the needed probe bins were
already captured.

## Verification

```text
python -m pytest tests/analysis/test_coordinate_threshold_hidden_state_probe.py -q
  13 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/coordinate_threshold_hidden_state_probe.py scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py tests/analysis/test_coordinate_threshold_hidden_state_probe.py
  exit 0
```

## Artifacts

Input hidden-state readouts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source36_object14_y2_layers20_24_28_surface_probe_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source123_object5_y2_layers20_24_28_surface_probe_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source145_object20_y2_layers20_24_28_surface_probe_bins_v1
```

Post-hoc local-simplex summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_local_simplex_surface_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_local_simplex_surface_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_local_simplex_surface_summary_v1
```

Each summary reports:

```text
row_count: 3
state_count: 1
readout_status_counts: {"ok": 3}
model_perturbation_ran: false
training_ran: false
```

## Final-Layer Local-Family Comparison

| state | target | competitor family | layer top | layer target rank | layer target-minus-max | base top | base target rank | base target-minus-max | effective top | effective target rank | effective target-minus-max | adapter delta top | adapter delta target rank | adapter delta target-minus-max | effective-minus-base margin shift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source36/object14/y2 | 996 | 994,995,997,998,999 | 999 | 5 | -2.250 | 998 | 4 | -4.408 | 999 | 5 | -56.986 | 999 | 6 | -55.256 | -52.579 |
| source123/object5/y2 | 270 | 188,254,259,266,274 | 259 | 3 | -0.625 | 188 | 6 | -135.481 | 188 | 5 | -101.887 | 270 | 1 | +18.992 | +33.594 |
| source145/object20/y2 | 146 | 142,144,145,150,154,158 | 154 | 4 | -0.250 | 158 | 6 | -38.610 | 158 | 7 | -46.866 | 158 | 4 | -8.256 | -8.256 |

## Mechanistic Reading

The local-simplex view sharpens the previous three-way split:

```text
source36/object14/y2:
  The target is already not the layer-local winner, but the final adapter
  surface is the decisive amplifier. At layer 28, target 996 is only -4.408
  below the best base-surface local competitor, but after adapter offsets it is
  -56.986 below local winner 999. The adapter delta alone prefers 999 over 996
  by -55.256, and the effective-minus-base margin shift is -52.579.

source123/object5/y2:
  The adapter is target-helpful, not target-harmful. It gives target 270 the
  strongest local adapter delta and improves the target-minus-max margin by
  +33.594, but the base surface is so far from the target basin that the
  effective surface still prefers 188 by -101.887. This is upstream weak or
  misbound target evidence rather than an adapter-created attractor.

source145/object20/y2:
  The adapter slightly worsens the wrong local anchor, but the failure is
  already present in the base/effective surface family around 158. The final
  hidden layer itself is only weakly off target, while both output surfaces
  strongly prefer 158. This looks like a mixed off-target surface basin rather
  than the extreme source36 adapter cliff.
```

This means the same visible symptom, a wrong coordinate slot near the correct
region, has at least three distinct internal causes:

```text
adapter-amplified local attractor:
  source36

target-helpful adapter unable to rescue weak upstream evidence:
  source123

off-target base/effective surface basin with mild adapter reinforcement:
  source145
```

## Why This Matters

This reducer turns the coordinate-token concern into a reusable measurement:

```text
target-vs-family margin before adapter
target-vs-family margin after adapter
adapter-only target-vs-family margin
effective-minus-base margin shift
```

That measurement is closer to the core mechanism than full-rank summaries,
because the model does not fail by choosing an arbitrary token. It fails inside
small coordinate families that are locally plausible, history-conditioned, and
adapter-shaped.

For source36, this supports a stronger next hypothesis:

```text
The dangerous basin is not a single coordinate token and not a single attention
route. It is a local coordinate simplex whose final winner is selected by the
interaction of late hidden-state evidence and token_embeddings_adapter geometry.
```

For source123, the same measurement warns against over-focusing on the adapter:

```text
When the adapter is target-helpful, the next probe should move earlier, toward
visual-binding, object identity, and history-side evidence formation.
```

## Next High-Value Steps

1. Run this reducer over any future hidden-state readout panel with selected
   local/contrastive bins before deciding whether to intervene on heads,
   residual directions, or adapter geometry.
2. For source36, test whether adapter-surface neutralization over the local
   family changes the first emitted coordinate without suppressing all
   coordinate-token mass.
3. For source123, run an earlier visual/object-binding probe rather than a
   stronger adapter ablation: compare target-object image-region evidence,
   semantic token preparation, and coordinate-slot emergence before y2.
4. For source145, probe whether the base surface around 158 is inherited from
   nearby object/history anchors or from coordinate embedding geometry alone.
