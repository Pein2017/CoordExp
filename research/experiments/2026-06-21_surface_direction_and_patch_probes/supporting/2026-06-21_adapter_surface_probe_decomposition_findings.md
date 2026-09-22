---
doc_id: progress.diagnostics.adapter_surface_probe_decomposition_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-three-state-y2-adapter-surface-probe-decomposition
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Adapter Surface Probe Decomposition Findings

## Scope

This note follows the coordinate direction patch findings:

```text
progress/diagnostics/2026-06-21_coord_target_direction_causal_patch_findings.md
progress/diagnostics/2026-06-21_coord_target_mean_direction_causal_patch_findings.md
```

The previous result showed that source36/object14/y2 is not single-axis
steerable but is local-simplex steerable. This follow-up asks where that local
ridge comes from:

```text
pre-adapter base output surface
token_embeddings_adapter effective output surface
final hidden-state / LM-head coord distribution
```

## Helper Added

The trajectory hidden-state readout now accepts:

```text
--probe-coord-bins
```

for `--stage trajectory-hidden-state-readout`, not only causal patch stages.
For each requested bin, rows now include:

```text
layer_coord_probe_<bin>_{logit,prob_full_vocab,prob_coord_only,rank_coord_only,top1_distance}
surface_base_probe_<bin>_{score,prob,rank,top1_distance}
surface_effective_probe_<bin>_{score,prob,rank,top1_distance}
surface_adapter_probe_<bin>_delta
```

A post-hoc reducer was also added:

```text
scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py \
  --stage summarize-adapter-surface-probes
```

It writes:

```text
adapter_surface_probe_summary.json
adapter_surface_probe_summary.md
```

Verification:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/coordinate_threshold_hidden_state_probe.py scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py tests/analysis/test_coordinate_threshold_hidden_state_probe.py src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
  exit 0
python -m pytest tests/analysis/test_coordinate_threshold_hidden_state_probe.py -q
  11 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
  370 passed
git diff --check
  exit 0
```

Subagent review gates:

```text
spec compliance review: approved
artifact/note numeric consistency review: approved after scope-wording edits
code-quality review: approved after early probe-bin validation, scoped
  max_adapter_delta_bin_among_probes naming, bin-0 tie handling, and
  zero-delta handling fixes
```

## Artifacts

All model-backed readouts used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
layers: 20,24,28
```

Hidden-state readouts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source36_object14_y2_layers20_24_28_surface_probe_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source123_object5_y2_layers20_24_28_surface_probe_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source145_object20_y2_layers20_24_28_surface_probe_bins_v1
```

Post-hoc summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_adapter_surface_probe_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_adapter_surface_probe_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_adapter_surface_probe_summary_v1
```

## Final-Layer Comparison

| state | target | layer top1 | base top1 | effective top1 | target base rank | target effective rank | target adapter delta rank among probes | max adapter-delta bin among probes | effective best probe bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source36/object14/y2 | 996 | 999 | 998 | 999 | 4 | 7 | 6 | 999 | 999 |
| source123/object5/y2 | 270 | 259 | 252 | 252 | 193 | 157 | 1 | 270 | 188 |
| source145/object20/y2 | 146 | 154 | 158 | 158 | 20 | 17 | 4 | 158 | 158 |

Selected final-layer probe details:

```text
source36 target 996:
  adapter delta target 996: +38.565
  adapter delta 997: +52.639
  adapter delta 999: +93.821
  base ranks: 998 rank1, 997 rank2, 999 rank3, 996 rank4
  effective ranks: 999 rank1, 997 rank2, 998 rank3, 995 rank4, 996 rank7

source123 target 270:
  target 270 receives the largest local adapter delta: +47.911
  but target remains weak in the surface: base rank193 -> effective rank157
  final hidden/LM-head top1 is 259, not target 270

source145 target 146:
  adapter delta target 146: +61.760
  adapter delta 154: +67.045
  adapter delta 158: +70.016
  base and effective surfaces both prefer 158
  final hidden/LM-head top1 is 154
```

## Mechanistic Reading

The three y2 states separate into different failure modes:

```text
source36:
  The adapter surface readout is consistent with an active contributor to the
  wrong local attractor, especially when read together with the prior local-
  simplex causal patch.
  Before adapter offsets, target 996 is already local-rank 4 while 998/997/999
  are nearby. After adapter offsets, 999 receives an extreme positive boost
  and target falls to local-rank 7. This matches the local-simplex causal patch:
  the final error appears compatible with a local ridge distorted by
  adapter/new-token geometry.

source123:
  The adapter is not the main antagonist. It gives the target 270 the largest
  local boost, but the hidden state is poorly aligned with the target basin.
  This looks more like weak/diffuse visual-binding or history-binding evidence
  before the adapter surface.

source145:
  The adapter and base surface both support an off-target local anchor around
  158, while the realized LM-head distribution peaks at 154. The target is
  recoverable by a target-minus-158 causal direction, but the natural surface
  does not choose it. This is a mixed surface-plus-hidden-state offset.
```

This gives a deeper split than "coordinate basin problem":

```text
adapter-distorted local ridge:
  source36

weak upstream target evidence despite helpful adapter boost:
  source123

off-target local anchor shared by base/effective surface and hidden readout:
  source145
```

## Implications

For the user's coordinate-token concern:

```text
Within these tiny probed states, the token_embeddings_adapter is not only a
smoothness-preservation surface. It can selectively amplify local ridge members.
In source36, the adapter heavily amplifies 999 relative to target 996, turning a
near-target base surface into a ceiling-side attractor candidate.
```

This also offers a candidate explanation for why the local-simplex patch worked:

```text
Single-bin suppression exposes another boosted local member.
Suppressing the local attractor family cancels the adapter-shaped ridge enough
for the target to win.
```

## Next Probe

The next high-value mechanism step is head/value routing scored by local-ridge
movement:

```text
For source36 y2, run targeted head/value contribution or intervention rows
around layers 24 -> 28, but score not only p996. Track:
  p996
  p999
  max local family member
  target adapter-delta rank proxy from the surface probe
  local-simplex recoverability threshold
```

The goal is to identify which attention/value route writes the hidden-state
component that the adapter surface converts into the 999 ridge.
