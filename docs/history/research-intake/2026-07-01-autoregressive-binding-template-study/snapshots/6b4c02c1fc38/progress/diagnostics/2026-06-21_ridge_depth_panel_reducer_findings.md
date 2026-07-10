---
doc_id: progress.diagnostics.ridge_depth_panel_reducer_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-four-state-y2-posthoc-causal-patch-ridge-panel
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Ridge-Depth Panel Reducer Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_source123_145_onset_ridge_depth_probe.md
```

The previous note ended with a hand-built four-state ridge-depth panel. This
follow-up turns that panel into a reusable post-hoc reducer over existing
`trajectory_hidden_causal_activation_patch_rows.jsonl` artifacts.

This is selected-state activation-patch evidence. It does not run new model
forwards, does not decode, does not train, and should not be interpreted as
population validation.

## Helper Added

New CPU-only reducer:

```text
scripts/analysis/run_autoregressive_binding_ridge_depth_panel.py \
  --run-specs <run_specs.json> \
  --output-root <summary_root>
```

The run-spec JSON has shape:

```json
{"runs":[{"label":"...","state_label":"...","layer_pair":"20->24","rows_path":"..."}]}
```

It writes:

```text
ridge_depth_panel_summary.json
ridge_depth_panel_summary.md
ridge_depth_panel_manifest.json
```

For each run and `direction_patch_basis_key`, it records the target coordinate
bin, parsed antagonist bins, first sampled strength where the target becomes
coord top1, best sampled target probability, and the max-strength top1 bin.

The reducer now explicitly validates that a direction group is one semantic
state/target/layer identity. If a multi-state artifact reuses the same
direction key, the reducer raises instead of silently mixing rows.

## Artifact

Four-state y2 panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/ridge_depth_panel/four_state_y2_onset_late_v1
```

Inputs:

```text
run_specs.json
```

Generated outputs:

```text
ridge_depth_panel_summary.json
ridge_depth_panel_summary.md
ridge_depth_panel_manifest.json
```

Manifest:

```text
run_count: 10
input_row_count: 258
panel_row_count: 29
ignored_missing_basis_key_count: 30
ignored_missing_patch_scale_count: 0
```

The ignored rows are baseline/no-direction rows with
`direction_patch_basis_key = null`.

## Compact Results

| state | layer pair | direction | target | first target top1 | best target prob | best top1/rank | max-strength top1 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| source33/object11/y2 onset | 20->24 | target - 413 | 417 | 256 | 0.176198 at 256 | 417 / 1 | 823 |
| source33/object11/y2 onset | 20->24 | target - mean(413,414,415,416,418,419) | 417 | 256 | 0.189793 at 256 | 417 / 1 | 105 |
| source33/object11/y2 late | 24->28 | target - 413 | 417 | 256 | 0.316600 at 512 | 417 / 1 | 417 |
| source33/object11/y2 late | 24->28 | target - mean(415,418) | 417 | 256 | 0.293351 at 512 | 417 / 1 | 417 |
| source33/object11/y2 late | 24->28 | target - full local mean | 417 | 256 | 0.378010 at 1024 | 417 / 1 | 417 |
| source36/object14/y2 onset | 20->24 | target - 995 | 996 | never | 0.049927 at 0 | 999 / 5 | 999 |
| source36/object14/y2 onset | 20->24 | target - 997 | 996 | never | 0.074628 at 128 | 999 / 5 | 994 |
| source36/object14/y2 onset | 20->24 | target - 999 | 996 | never | 0.182745 at 512 | 995 / 2 | 995 |
| source36/object14/y2 onset | 20->24 | target - mean(999,995) | 996 | 224 | 0.367505 at 512 | 996 / 1 | 996 |
| source36/object14/y2 onset | 20->24 | target - full local mean | 996 | 320 | 0.369526 at 512 | 996 / 1 | 996 |
| source36/object14/y2 late | 24->28 | target - mean(999,995) | 996 | 512 | 0.661797 at 1024 | 996 / 1 | 996 |
| source36/object14/y2 late | 24->28 | target - full local mean | 996 | 256 | 0.560125 at 1024 | 996 / 1 | 996 |
| source123/object5/y2 onset | 20->24 | target - 188 | 270 | 64 | 0.303104 at 512 | 270 / 1 | 270 |
| source123/object5/y2 onset | 20->24 | target - 259 | 270 | 64 | 0.200626 at 512 | 270 / 1 | 270 |
| source123/object5/y2 onset | 20->24 | target - mean(188,259) | 270 | 64 | 0.310896 at 512 | 270 / 1 | 270 |
| source145/object20/y2 onset | 20->24 | target - 154 | 146 | never | 0.229949 at 512 | 145 / 2 | 145 |
| source145/object20/y2 onset | 20->24 | target - 158 | 146 | 256 | 0.458186 at 512 | 146 / 1 | 146 |
| source145/object20/y2 onset | 20->24 | target - mean(154,158) | 146 | 256 | 0.412720 at 512 | 146 / 1 | 146 |
| source145/object20/y2 late | 24->28 | target - 154 | 146 | never | 0.209807 at 512 | 145 / 3 | 145 |
| source145/object20/y2 late | 24->28 | target - 158 | 146 | 128 | 0.550767 at 512 | 146 / 1 | 146 |

## Mechanistic Reading

The machine-readable panel preserves the four-way distinction from the earlier
hand table:

```text
source123:
  easy/broad steerability. Both visible candidate antagonists and their mean
  select the target by strength 64 at onset.

source145:
  specific-antagonist steerability. The target is recoverable through the 158
  antagonist, but not through the visible 154 side in the tested directions.

source33:
  low-side history-anchor steerability. The 413 anchor and full local mean can
  recenter the target, but high strength at onset overshoots out of the local
  family to unrelated bins.

source36:
  multi-anchor local-ridge steerability. Single-bin onset directions do not
  select the target in the tested scale range. Local means that include the
  999/995 ridge do select the target.
```

The most useful update is methodological: ridge-depth should be tracked as a
state-and-direction table, not a prose-only observation. That lets future
false-negative and duplication cases be compared by the depth and specificity
of the intervention needed to make the intended coordinate win.

## Verification

```text
python -m pytest tests/analysis/test_ridge_depth_panel.py -q
  7 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/ridge_depth_panel.py scripts/analysis/run_autoregressive_binding_ridge_depth_panel.py
  exit 0

git diff --check -- src/analysis/autoregressive_binding_template_ablation/ridge_depth_panel.py scripts/analysis/run_autoregressive_binding_ridge_depth_panel.py tests/analysis/test_ridge_depth_panel.py
  exit 0

python scripts/analysis/run_autoregressive_binding_ridge_depth_panel.py \
  --run-specs /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/ridge_depth_panel/four_state_y2_onset_late_v1/run_specs.json \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/ridge_depth_panel/four_state_y2_onset_late_v1
  run_count=10 input_row_count=258 panel_row_count=29
```

## Next Step

Use this reducer as the common comparison table for the next selected cases:

```text
1. false-negative object states, to test whether guidance can make missing
   object coordinates reachable;
2. source36-like local-ridge states, to find which attention/value routes
   inject the multi-anchor basin before y2 emission;
3. source145-like specific-antagonist states, to separate visible hidden top1
   from the actual steering antagonist.
```
