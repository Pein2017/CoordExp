---
doc_id: progress.diagnostics.coord_target_mean_direction_causal_patch_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source36-y2-multi-competitor-coordinate-direction-causal-patch
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Coord-Target Mean Direction Causal Patch Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_coord_target_direction_causal_patch_findings.md
```

The single-coordinate contrast probe found that source36/object14/y2 could not
be centered on target bin `996` by adding any tested single direction:

```text
embedding(<|coord_996|>) - embedding(<|coord_k|>)
```

The best single-bin contrast raised target probability but landed in nearby
bin `995`.

This follow-up asks whether the stubborn basin is actually a local attractor
family that needs simultaneous suppression. It adds a dynamic direction basis:

```text
coord_target_minus_mean:<bin>+<bin>[+...]
```

For a coordinate target token, the patch direction is:

```text
embedding(target coord token) - mean(embedding(<|coord_<bin>|>), ...)
```

## Implementation Surface

Extended the existing causal activation patch direction parser in:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Validation/behavior:

```text
coord_target_minus_mean:<bins>
  bins are plus-separated because comma separates direction bases
  at least two distinct bins are required
  each bin must be canonical integer 0..999
  direction_patch_basis_key keeps the literal key
  direction_patch_basis is artifact-safe, e.g. coord_target_minus_mean_999_998_output_embedding
```

Verification:

```text
RED:
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "coord_target_minus_mean" -q
  7 passed, 2 failed before implementation; failures were unsupported dynamic mean basis

GREEN:
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "coord_target_minus_mean" -q
  9 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "trajectory_hidden_causal_activation_patch or coord_target_minus" -q
  40 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
  367 passed
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
  exit 0
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py --help | grep coord_target_minus
  help includes coord_target_minus_bin and coord_target_minus_mean
```

## Artifacts

All model-backed runs used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
state: source36/object14/y2
target coord: 996
layer pair: 24 -> 28
patch_alphas: 0
```

Coarse mean sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s24_t28_coord_target_minus_mean_local_bins_v1
```

Threshold sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s24_t28_coord_target_minus_mean_local_threshold_v1
```

Run counters:

| artifact | direction bases | strengths | realized direction rows | row count |
| --- | --- | --- | ---: | ---: |
| mean_local_bins_v1 | 4 | 0,16,32,64,128,256,512,1024 | 32 | 35 |
| mean_local_threshold_v1 | 2 | 128,160,192,224,256,320,384,448,512 | 18 | 21 |

## Result

The source36/object14/y2 basin is multi-axis recoverable.

Coarse sweep:

| direction basis | first target top1 | best target prob | best top1 | interpretation |
| --- | ---: | ---: | ---: | --- |
| target-minus-mean(995,994) | never | 0.059204 | 999 | low-side suppression alone is insufficient |
| target-minus-mean(999,998,997) | never | 0.152020 | 995 | ceiling-side suppression alone moves mass but lands on 995 |
| target-minus-mean(999,995) | 512 | 0.661797 | 996 | suppressing both ceiling and 995 anchors works |
| target-minus-mean(999,998,997,995,994) | 256 | 0.560125 | 996 | full local-attractor suppression works earlier |

Threshold sweep:

| direction basis | strength | top1 | target rank | p996 | p995 | p997 | p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean(999,995) | 256 | 997 | 2 | 0.247303 | 0.192599 | 0.317543 | 0.048697 |
| mean(999,995) | 320 | 996 | 1 | 0.318738 | 0.193324 | 0.318738 | 0.020376 |
| mean(999,995) | 512 | 996 | 1 | 0.445337 | 0.238372 | 0.210362 | 0.001104 |
| mean(999,998,997,995,994) | 224 | 997 | 2 | 0.233964 | 0.233964 | 0.265115 | 0.067032 |
| mean(999,998,997,995,994) | 256 | 996 | 1 | 0.270818 | 0.238996 | 0.270818 | 0.047061 |
| mean(999,998,997,995,994) | 512 | 996 | 1 | 0.422333 | 0.328914 | 0.155368 | 0.000719 |

The source36 update is therefore:

```text
single-bin coordinate contrasts:
  target evidence can be increased, but the basin remains displaced.

multi-bin local-attractor contrasts:
  simultaneous suppression of the 999/998/997/995/994 family recenters the
  final coordinate surface onto 996.
```

The apparent tie rows are useful, not noise. At the flip threshold, p996 often
equals or nearly equals a neighboring bin such as 997, which suggests the local
basin has a ridge-like structure. Tiny changes in the hidden direction can move
the argmax among adjacent bins without destroying locality.

## Mechanistic Update

The previous note framed source36 as a "stubborn displaced basin." The refined
reading is more specific:

```text
source36 is not single-axis steerable, but it is local-simplex steerable.
```

This points away from a pure visual-perception failure. The model has enough
local geometric evidence for the correct y2 region. The failure is in how the
final coordinate slot resolves a local attractor family under autoregressive
history:

```text
1. The target coordinate evidence is present in the hidden/readout state.
2. Multiple adjacent/ceiling bins jointly form the emission basin.
3. Suppressing only one bin exposes another neighbor.
4. Suppressing the local family lets the target become top1.
```

This is a concrete version of the coordinate-token hypothesis:

```text
CE supervision preserved geometry locality, but the coordinate surface is not
smooth enough for the right local peak to win reliably under repeated bottom-
edge snowboard history.
```

## Next Useful Probes

The highest-value next step is adapter-space decomposition:

```text
Compare source36 against steerable source123/source145 at the same y2 slot:
  pre-adapter hidden projection onto coord tokens
  token_embeddings_adapter delta contribution
  post-adapter/final LM-head coordinate distribution
  response to single-bin vs local-mean patch directions
```

If source36's pre-adapter state is already target-centered but the adapter
surface shifts it into the local attractor family, the origin is the learned
new-token adapter geometry. If the pre-adapter state is also ridge-like, the
origin is earlier autoregressive binding/history dynamics.

After that, the natural causal path is:

```text
head/value route tomography around the same y2 state, with patches scored by
local-simplex basin movement instead of only target-bin probability.
```
