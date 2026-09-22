# Pre-Onset All-Slot Edge Residual Asymmetry

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_history_feature_join.md`.
It expands the pre-onset residual-patch target selection across all four
coordinate slots:

- `box_start/pre_x1`;
- `post_x1/pre_y1`;
- `post_y1/pre_x2`;
- `post_x2/pre_y2`.

The purpose is to check whether the residual direction asymmetry observed in
the earlier `post_y1/pre_x2` panel is slot-specific, edge-band-specific, or a
broader pre-onset property.

This is still an analysis of selected windows, not a population-level rollout
claim.

## Implementation Change

The selector and history reducer now carry an explicit `coord_slot` field
derived from phase:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_patch_selector.py
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_history_features.py
```

Tests were extended to assert that selected and materialized rows preserve
`coord_slot`.

## Run

Selector:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py \
  --coord-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_onset_precursor_panel/onset_precursor_coord_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-00-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-01-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-02-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-03-of-04/phase3_masking/masking_delta_rows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-00-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-01-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-02-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-03-of-04/token_windows.jsonl \
  --region-rows-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1 \
  --max-targets 16 \
  --rank-band 2,10 \
  --probability-drift-threshold 0.01 \
  --phases box_start/pre_x1,post_x1/pre_y1,post_y1/pre_x2,post_x2/pre_y2 \
  --patch-layers 16,20,24,27 \
  --patch-sites mlp,post_attention_residual \
  --patch-directions masked_to_control,control_to_masked \
  --target-layer 20
```

Residual patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_residual_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/pre_onset_patch_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/pre_onset_patch_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/residual_patch_selected_cases \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --patch-layers 16,20,24,27 \
  --patch-sites mlp,post_attention_residual \
  --patch-directions masked_to_control,control_to_masked \
  --top-k 8 \
  --target-top-k 16
```

History feature join:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py \
  --selector-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1 \
  --residual-patch-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/residual_patch_selected_cases \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/history_features
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/residual_patch_selected_cases
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/history_features
```

## Selection Boundary

The expanded selector requested both rank-band and probability-drift targets,
but all selected rows were `top1_probability_drift`:

```text
target_count=16
selection_kind_counts={"top1_probability_drift": 16}
```

Therefore this pass mostly tests pre-onset probability calibration, edge-basin
fragility, and residual direction asymmetry. It does not sufficiently cover the
rank-band rescue surface.

Selected slots and edge bands:

| group | count |
| --- | ---: |
| `x1` | 2 |
| `y1` | 5 |
| `x2` | 5 |
| `y2` | 4 |
| `high_edge` | 6 |
| `low_edge` | 5 |
| `interior` | 5 |

## Summary

History feature row count: `16`.

Taxonomy:

| taxonomy | count |
| --- | ---: |
| `top1_stable_calibration` | 13 |
| `rank_moving_repair` | 2 |
| `patch_damage_fragile_basin` | 1 |

Taxonomy by slot:

| slot | `top1_stable_calibration` | `rank_moving_repair` | `patch_damage_fragile_basin` |
| --- | ---: | ---: | ---: |
| `x1` | 2 | 0 | 0 |
| `y1` | 4 | 1 | 0 |
| `x2` | 3 | 1 | 1 |
| `y2` | 4 | 0 | 0 |

Patch directions:

| direction counter | count |
| --- | ---: |
| best repair `masked_to_control` | 9 |
| best repair `control_to_masked` | 7 |
| worst damage `masked_to_control` | 13 |
| worst damage `control_to_masked` | 3 |

Worst damage by slot:

| slot | `masked_to_control` | `control_to_masked` |
| --- | ---: | ---: |
| `x1` | 1 | 1 |
| `y1` | 4 | 1 |
| `x2` | 5 | 0 |
| `y2` | 3 | 1 |

Checkpoint mix:

| checkpoint label | rows |
| --- | ---: |
| `no_aligner_parent_ckpt3668` | 7 |
| `aligner_parent_ckpt1824` | 5 |
| `none_latest_ckpt32` | 2 |
| `aux_latest_ckpt32` | 2 |

## Row-Level Join

Direction abbreviations:

- MTC: `masked_to_control`;
- CTM: `control_to_masked`.

Coordinate reuse columns are counts of prior generated rows containing the
selected target bin exactly, within radius `4`, or within radius `16` in any box
slot.

| class | slot | checkpoint | record | desc | target | edge | prior rows | same desc | coord exact/r4/r16 | bbox iou>=0.5 | best repair | worst damage |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| `top1_stable_calibration` | `x2` | `no_aligner_parent_ckpt3668` | 88 | `cow` | 999 | `high_edge` | 2 | 0 | 1/1/1 | 0 | MTC `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `x1` | `aligner_parent_ckpt1824` | 37 | `dining table` | 0 | `low_edge` | 3 | 0 | 0/0/1 | 1 | CTM `mlp L16` | CTM `mlp L16` |
| `top1_stable_calibration` | `y2` | `aligner_parent_ckpt1824` | 36 | `person` | 999 | `high_edge` | 0 | 0 | 0/0/0 | 0 | MTC `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `x2` | `aligner_parent_ckpt1824` | 37 | `bowl` | 999 | `high_edge` | 1 | 0 | 0/0/0 | 0 | MTC `mlp L27` | MTC `mlp L27` |
| `top1_stable_calibration` | `x1` | `none_latest_ckpt32` | 114 | `person` | 0 | `low_edge` | 5 | 1 | 3/4/4 | 1 | MTC `mlp L20` | MTC `mlp L20` |
| `top1_stable_calibration` | `y1` | `no_aligner_parent_ckpt3668` | 47 | `person` | 0 | `low_edge` | 3 | 2 | 2/2/2 | 0 | CTM `mlp L24` | CTM `mlp L24` |
| `top1_stable_calibration` | `y2` | `none_latest_ckpt32` | 36 | `person` | 999 | `high_edge` | 0 | 0 | 0/0/0 | 0 | MTC `mlp L20` | MTC `mlp L20` |
| `top1_stable_calibration` | `y1` | `aligner_parent_ckpt1824` | 54 | `person` | 0 | `low_edge` | 0 | 0 | 0/0/0 | 0 | MTC `mlp L24` | MTC `mlp L24` |
| `rank_moving_repair` | `y1` | `aligner_parent_ckpt1824` | 37 | `carrot` | 170 | `interior` | 2 | 0 | 0/0/0 | 0 | CTM `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `y2` | `no_aligner_parent_ckpt3668` | 88 | `sheep` | 999 | `high_edge` | 1 | 0 | 0/0/0 | 0 | CTM `mlp L20` | CTM `mlp L20` |
| `top1_stable_calibration` | `y2` | `no_aligner_parent_ckpt3668` | 36 | `person` | 999 | `high_edge` | 0 | 0 | 0/0/0 | 0 | MTC `mlp L20` | MTC `mlp L20` |
| `top1_stable_calibration` | `y1` | `no_aligner_parent_ckpt3668` | 54 | `person` | 0 | `low_edge` | 0 | 0 | 0/0/0 | 0 | MTC `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `x2` | `aux_latest_ckpt32` | 79 | `bottle` | 579 | `interior` | 9 | 0 | 0/0/1 | 1 | CTM `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `y1` | `no_aligner_parent_ckpt3668` | 88 | `person` | 157 | `interior` | 3 | 1 | 0/0/0 | 0 | CTM `mlp L24` | MTC `mlp L16` |
| `patch_damage_fragile_basin` | `x2` | `no_aligner_parent_ckpt3668` | 48 | `person` | 33 | `interior` | 0 | 0 | 0/0/0 | 0 | CTM `mlp L24` | MTC `mlp L16` |
| `top1_stable_calibration` | `x2` | `aux_latest_ckpt32` | 50 | `person` | 395 | `interior` | 7 | 3 | 0/0/2 | 0 | MTC `mlp L16` | MTC `mlp L16` |

## Read

The main asymmetry survived the all-slot expansion, but the mechanistic
payload is uneven across the selected panel.

1. Worst damage remains strongly MTC-skewed: `13/16` rows have worst damage
   under `masked_to_control`. This is stronger than the mixed best-repair
   direction count (`9` MTC, `7` CTM), and it supports preserving direction
   rather than aggregating patch magnitude.
2. The clearest repair-style rows are still interior-coordinate cases. The two
   `rank_moving_repair` rows are `y1` and `x2`, both with CTM best repair and
   MTC worst damage. The only `patch_damage_fragile_basin` row is also `x2`,
   interior, with CTM best repair and MTC worst damage.
3. Edge rows are mostly calibration basins. All `high_edge` and `low_edge`
   selected rows landed in `top1_stable_calibration`, even when prior local
   history shows repeated coordinate anchors. That suggests edge slots are
   often stable attractors whose probability can drift without rank-level
   recoverability.
4. Local autoregressive history is not a single necessary condition in this
   selected panel. Some high-history rows are stable calibration cases
   (`none_latest_ckpt32` record 114, `aux_latest_ckpt32` record 50), while a
   zero-history `x2` row is the only fragile-basin case. The better current
   hypothesis is not "history alone causes duplication"; it is "history,
   coordinate-basin attraction, and masked/control residual direction interact,
   with slot and edge band determining whether the evidence appears as stable
   calibration or rank-moving fragility."

The most important refinement is that `x2` remains special after adding all
slots: every `x2` row's worst damage direction is MTC, and `x2` contains both
the only `patch_damage_fragile_basin` row and one of the two
`rank_moving_repair` rows. This does not prove that `x2` is the origin of
duplication, but it is a strong reason to keep `x2` as a priority site for the
next hidden-state and attention-level probes.

## Next Step

The next deterministic probe should rebalance selection toward
rank-moving/onset-instability targets instead of allowing probability-drift
targets to fill the panel:

- select a small fixed quota per coordinate slot;
- separate edge bins (`0`, `999`) from interior bins;
- require rank drift, top-k identity drift, or onset-instability evidence for
  at least part of each slot quota;
- then rerun residual patch and history join.

That would test whether the MTC-damage / CTM-repair asymmetry is genuinely a
rank-moving mechanism or mainly an artifact of the easier probability-drift
surface.

## Verification

Code checks:

```bash
python -m pytest \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py \
  -q
```

The repo-local pytest wrapper printed its usual compact `No tests collected`
label but exited `0`.

Direct test invocation:

```bash
python - <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
for test_path in [
 'tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py',
 'tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py',
]:
 spec=spec_from_file_location('mod', test_path)
 assert spec and spec.loader
 mod=module_from_spec(spec)
 spec.loader.exec_module(mod)
 for name in sorted(n for n in dir(mod) if n.startswith('test_')):
  fn=getattr(mod,name)
  if 'tmp_path' in fn.__code__.co_varnames:
   with tempfile.TemporaryDirectory() as tmp:
    fn(Path(tmp))
  else:
   fn()
print('direct_selector_and_history_tests_passed')
PY
```

Compile check:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_patch_selector.py \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_history_features.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py
```

Artifact check:

```bash
find /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_all_slots_v1/history_features \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

Observed files:

```text
pre_onset_history_feature_report.md 2960 bytes
pre_onset_history_feature_rows.jsonl 18361 bytes
pre_onset_history_feature_summary.json 1555 bytes
```
