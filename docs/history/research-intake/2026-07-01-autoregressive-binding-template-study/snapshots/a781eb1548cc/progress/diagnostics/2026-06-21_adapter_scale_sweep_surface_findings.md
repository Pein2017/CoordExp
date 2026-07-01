---
doc_id: progress.diagnostics.adapter_scale_sweep_surface_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-three-state-y2-posthoc-adapter-scale-sweep
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Adapter Scale Sweep Surface Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_local_simplex_surface_adapter_bridge_findings.md
```

The previous local-simplex summary showed that source36/object14/y2 has a
large token_embeddings_adapter-amplified local coordinate attractor. This
follow-up asks a sharper counterfactual question without rerunning model
forwards:

```text
If the same hidden state were scored by base + scale * adapter_delta over the
already-probed local coordinate family, what adapter scale would make the
target win?
```

This is not a full autoregressive decode with hooks disabled. It is a read-only
surface-space counterfactual over existing `surface_base_probe_*_score` and
`surface_effective_probe_*_score` fields.

## Helper Added

The coordinate threshold hidden-state probe now has:

```text
scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py \
  --stage summarize-adapter-scale-sweep \
  --hidden-state-readout-rows <trajectory_hidden_state_readout_rows.jsonl> \
  --output-root <summary_root>
```

It writes:

```text
adapter_scale_sweep_summary.json
adapter_scale_sweep_summary.md
```

Default sampled scales:

```text
-1,-0.5,0,0.25,0.5,0.75,1,1.25
```

For each layer, it records sampled target-vs-family margins and an exact
linear target-win interval. The interval solves:

```text
base_target + scale * (effective_target - base_target)
  > base_competitor + scale * (effective_competitor - base_competitor)
```

for every probed competitor.

## Verification

```text
python -m pytest tests/analysis/test_coordinate_threshold_hidden_state_probe.py -q
  17 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/coordinate_threshold_hidden_state_probe.py scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py tests/analysis/test_coordinate_threshold_hidden_state_probe.py
  exit 0

git diff --check
  exit 0
```

## Artifacts

Post-hoc summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_adapter_scale_sweep_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_adapter_scale_sweep_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_adapter_scale_sweep_v1
```

Each summary reports:

```text
row_count: 3
state_count: 1
readout_status_counts: {"ok": 3}
model_perturbation_ran: false
training_ran: false
```

## Final-Layer Results

| state | target | family | exact target-win adapter-scale interval | active constraint | scale 0 top/rank/margin | scale 1 top/rank/margin | best sampled scale/top/rank/margin |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| source36/object14/y2 | 996 | 994,995,996,997,998,999 | `scale < -1.03957160748` | upper bound from 998 | 998 / 4 / -4.408 | 999 / 5 / -56.986 | -1 / 998 / 2 / -0.168 |
| source123/object5/y2 | 270 | 188,254,259,266,270,274 | `scale > 4.83895920484` | lower bound from 254 | 188 / 6 / -135.481 | 188 / 5 / -101.887 | 1.25 / 188 / 5 / -93.488 |
| source145/object20/y2 | 146 | 142,144,145,146,150,154,158 | empty | incompatible lower 142 vs upper 144 | 158 / 6 / -38.610 | 158 / 7 / -46.866 | 0 / 158 / 6 / -38.610 |

## Mechanistic Reading

This separates the three cases further:

```text
source36:
  The adapter is a large harmful amplifier, but not the only problem. Removing
  the adapter path to scale 0 still leaves target 996 local-rank 4 behind 998.
  Even scale -1 almost recovers the target but remains slightly behind 998.
  Exact target victory requires over-subtracting the adapter contribution to
  scale < -1.0396. This is a local-simplex ridge: adapter amplification
  creates the visible 999 cliff, but the base surface already has a residual
  off-target 998/997 structure.

source123:
  The adapter direction is target-helpful. Increasing adapter scale improves
  the target margin, but the base target evidence is so weak that the target
  would need scale > 4.839 to win. This is stronger evidence that source123 is
  upstream weak/misbound evidence, not an adapter-created coordinate basin.

source145:
  No single scale along the adapter_delta direction can make target 146 beat
  all local competitors. The constraints are incompatible: one competitor
  requires scale > 40.78, another requires scale < -37.14. This points away
  from adapter scalar magnitude and toward a multi-direction surface or hidden
  state misbinding.
```

The important update to the source36 hypothesis is:

```text
Adapter neutralization alone is not expected to rescue the coordinate slot.
The adapter creates the final steep 999 attractor, but a base local ridge
already remains. A useful causal experiment should neutralize or project the
local coordinate family, not merely disable the adapter globally.
```

## Next Step

For source36, the strongest next causal probe is local-family surface
orthogonalization or targeted competitor suppression:

```text
1. keep coordinate-token mass available;
2. remove the adapter_delta components that distinguish 997/998/999 from 996;
3. measure whether the next-token local winner changes without collapsing the
   slot schema.
```

For source123, prioritize earlier visual/object-binding probes. Adapter-side
work is unlikely to explain the failure unless it can be shown to amplify the
right evidence by an unrealistically large factor.
