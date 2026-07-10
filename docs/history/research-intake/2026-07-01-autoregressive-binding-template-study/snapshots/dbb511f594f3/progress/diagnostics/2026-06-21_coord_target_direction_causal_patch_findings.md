---
doc_id: progress.diagnostics.coord_target_direction_causal_patch_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-three-state-y2-coordinate-direction-causal-patch
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Coord-Target Direction Causal Patch Findings

## Scope

This note extends the source-36 coordinate threshold readout:

```text
progress/diagnostics/2026-06-20_source36_coordinate_threshold_hidden_state_readout.md
```

The new probe is deliberately tiny. It tests whether late coordinate-basin
centering can be steered by adding output-embedding directions of the form:

```text
coord_target_minus_bin:<bin>
```

For a coordinate target token, this adds the unit direction:

```text
embedding(target coord token) - embedding(<|coord_<bin>|>)
```

to the selected raw decoder-layer source hidden state before running the
remaining model blocks. This is not a training intervention and not population
evidence.

## Implementation Surface

Added support for dynamic coordinate direction bases in:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Validation/behavior:

```text
coord_target_minus_bin:<bin>
  bin must be canonical integer 0..999
  requires target_spec.target_next_kind == coord
  direction_patch_basis_key keeps the literal key, e.g. coord_target_minus_bin:999
  direction_patch_basis is artifact-safe, e.g. coord_target_minus_bin_999_output_embedding
```

Verification:

```text
RED:
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "coord_target_minus_bin" -q
  0 passed, 8 failed before implementation; failures were unsupported dynamic basis

GREEN:
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "coord_target_minus_bin" -q
  8 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "trajectory_hidden_causal_activation_patch" -q
  25 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
  358 passed
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
  exit 0
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py --help | grep coord_target_minus_bin
  help includes coord_target_minus_bin:<0..999>
```

## Artifacts

All model-backed runs used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
patch_strengths: 0,8,16,32,64,128,256,512
patch_alphas: 0
```

Main three-state sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s24_t28_coord_target_minus_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source123_object5_y2_s24_t28_coord_target_minus_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source145_object20_y2_s24_t28_coord_target_minus_bins_v1
```

Source-36 local-neighbor follow-ups:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s24_t28_coord_target_minus_local_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s20_t24_coord_target_minus_local_bins_v1
```

Run counters:

| state | layer pair | direction bases | realized direction rows | row count |
| --- | --- | --- | ---: | ---: |
| source36/object14/y2 | 24 -> 28 | 999,997,234 | 24 | 27 |
| source123/object5/y2 | 24 -> 28 | 259,188 | 16 | 19 |
| source145/object20/y2 | 24 -> 28 | 154,158 | 16 | 19 |
| source36/object14/y2 | 24 -> 28 | 994,995,998,999 | 32 | 35 |
| source36/object14/y2 | 20 -> 24 | 995,997,999 | 24 | 27 |

## Main Result

The three comparator states split cleanly:

```text
source123 and source145 can be steered to the target coordinate bin by a
single target-minus-competitor coordinate direction.

source36 cannot be steered to target bin 996 by any tested single coordinate
contrast. The target probability rises, but the top coordinate shifts into a
neighbor basin around 995 instead of centering on 996.
```

Compact readout:

| state | target | baseline coord top1 | baseline target rank | best tested direction | first literal target top1 | best target prob | best target rank | best top1 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| source36/object14/y2 | 996 | 999 | 5 | target-minus-999 @ 512 | never | 0.251963 | 2 | 995 |
| source123/object5/y2 | 270 | 259 | 12 | target-minus-188 @ 512 | strength 64 | 0.350768 | 1 | 270 |
| source145/object20/y2 | 146 | 154 | 6 | target-minus-158 @ 512 | strength 128 | 0.550767 | 1 | 146 |

Source-36 local-neighbor sweep:

| direction | best strength | best top1 | best target rank | best p996 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| target-minus-994 | 128 | 999 | 4 | 0.064706 | weakly improves rank only |
| target-minus-995 | 0 | 999 | 5 | 0.049927 | moving away from 995 makes p996 worse |
| target-minus-998 | 256 | 995 | 3 | 0.128863 | improves p996 but lands on 995 |
| target-minus-999 | 512 | 995 | 2 | 0.251963 | strongest improvement, still not target |

The repeated source-36 `20 -> 24` layer-pair check showed the same qualitative
stubbornness: `target-minus-999 @ 512` improved p996 to 0.182745 and rank 2,
but top1 remained 995.

## Interpretation

This is a sharper mechanism split than the earlier structural-direction probe:

```text
1. Some late coordinate basin errors are linearly steerable in the output
   embedding coordinate-contrast basis. Source123 and source145 are examples.

2. Source36 is not merely "the wrong competitor logit is too high." Removing
   pressure from 999, 998, 997, 995, 994, or the far 234 contrast does not make
   996 the emitted center. The surface falls into an adjacent 995 basin.

3. For source36, p996 can be raised substantially without making 996 the peak.
   That suggests the basin center is set by a hidden-state / adapter-surface
   interaction, not by a single output-embedding axis.

4. The source145 contrast is especially informative: target-minus-158 flips to
   target 146, while target-minus-154 does not. The visible final top1 is not
   necessarily the causal antagonist; an upstream or side-lobe coordinate basis
   can be a better steering direction.
```

Current mechanistic update:

```text
The coordinate slot has at least two regimes:

linearly steerable basin errors:
  target coordinate is already available as a recoverable peak direction, and
  a single coordinate contrast can reveal it.

stubborn displaced basin errors:
  target evidence is present but the peak is anchored in a nearby attractor.
  Single coord-token output-embedding contrasts move mass locally but do not
  recenter the basin.
```

This connects to the user's prior coordinate-token observation:

```text
CE supervision can preserve broad geometry locality while destroying smoothness.
```

Source36 looks like exactly that failure mode: locality is strong
(`radius4_mass` is high), but the local coordinate surface is nonsmooth and the
emission peak prefers 995/999 over the target 996.

## Next Useful Probes

The next high-value bridge is not another single-axis contrast. Prefer:

```text
1. Multi-competitor coordinate simplex directions:
   target embedding minus mean(999,998,997,995,994), to test whether source36
   needs simultaneous suppression of the local attractor family.

2. Adapter-space direction decomposition:
   compare pre-adapter hidden readout, token_embeddings_adapter delta, and
   final LM-head coord distribution for source36 vs steerable source123/source145.

3. Head/value route gating into the stubborn basin:
   reuse the value-region and attention-tomography machinery to identify which
   late route moves source36 from "near target" to "995/999 peak."

4. Tiny hypothesis-training check:
   a limited coordinate-smoothness regularizer or local simplex contrast on
   only selected coord slots could test whether the stubborn displaced basin is
   caused by nonsmooth coord-token embedding geometry rather than visual
   perception failure.
```
