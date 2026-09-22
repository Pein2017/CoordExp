# Pre-Onset Balanced Rank-Onset Selector

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_all_slot_edge_residual_asymmetry.md`.
The previous all-slot selector was technically balanced across phases, but the
selected panel was swallowed by `top1_probability_drift` rows:

```text
top1_probability_drift=16/16
rank_moving_repair=2/16
top1_stable_calibration=13/16
```

That was useful for calibration and edge-basin readout, but too weak for the
main mechanism question. This slice adds and runs a balanced selector intended
to stress pre-onset rank movement and onset instability.

## Implementation Change

`phase4_pre_onset_patch_selector.py` now supports:

- `selection_strategy=priority`, preserving the old global-priority behavior;
- `selection_strategy=balanced_rank_onset`, which:
  - admits `onset_instability` rows when masking changes top-1 or moves target
    rank by at least `rank_instability_threshold`;
  - selects a fixed quota per coordinate slot;
  - can reserve edge candidates per slot before filling from interior rows.

The CLI exposes:

```text
--selection-strategy priority|balanced_rank_onset
--rank-instability-threshold
--slot-quota
--edge-quota-per-slot
```

`phase4_pre_onset_history_features.py` now preserves `mask_top1_changed` and
prints `selection_kind` plus `mask_target_rank_delta/mask_top1_changed` in the
report table. This matters because the selected panel is explicitly about
onset instability.

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
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1 \
  --max-targets 16 \
  --rank-band 2,10 \
  --probability-drift-threshold 0.01 \
  --rank-instability-threshold 10 \
  --selection-strategy balanced_rank_onset \
  --slot-quota 4 \
  --edge-quota-per-slot 1 \
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
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/residual_patch_selected_cases \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --patch-layers 16,20,24,27 \
  --patch-sites mlp,post_attention_residual \
  --patch-directions masked_to_control,control_to_masked \
  --top-k 8 \
  --target-top-k 16
```

History join:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py \
  --selector-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1 \
  --residual-patch-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/residual_patch_selected_cases \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/history_features
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/residual_patch_selected_cases
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/history_features
```

## Selector Panel

The selected panel is exactly slot-balanced:

| slot | rows |
| --- | ---: |
| `x1` | 4 |
| `y1` | 4 |
| `x2` | 4 |
| `y2` | 4 |

Selection kinds:

| selection kind | rows |
| --- | ---: |
| `onset_instability` | 14 |
| `rank_band_precursor` | 1 |
| `top1_probability_drift` | 1 |

Edge bands:

| edge band | rows |
| --- | ---: |
| `interior` | 12 |
| `low_edge` | 2 |
| `high_edge` | 2 |

This is a deliberately different slice from the previous all-slot panel. It is
not a random sample; it is a target-rich stress panel for hidden-state patching.

## Residual Patch And History Summary

Residual patch run:

```text
checkpoint_count=3
source_replay_case_count=8
targeted_replay_case_count=8
residual_patch_row_count=256
```

History taxonomy:

| taxonomy | rows |
| --- | ---: |
| `rank_moving_repair` | 11 |
| `top1_stable_calibration` | 2 |
| `no_effect_hard_or_stable` | 2 |
| `patch_damage_fragile_basin` | 1 |

Taxonomy by slot:

| slot | rank moving | stable calibration | hard/stable | fragile basin |
| --- | ---: | ---: | ---: | ---: |
| `x1` | 4 | 0 | 0 | 0 |
| `y1` | 3 | 0 | 1 | 0 |
| `x2` | 3 | 1 | 0 | 0 |
| `y2` | 1 | 1 | 1 | 1 |

Patch direction:

| direction counter | rows |
| --- | ---: |
| best repair `control_to_masked` | 10 |
| best repair `masked_to_control` | 6 |
| worst damage `masked_to_control` | 14 |
| worst damage `control_to_masked` | 2 |

Worst damage by slot:

| slot | MTC worst | CTM worst |
| --- | ---: | ---: |
| `x1` | 4 | 0 |
| `y1` | 4 | 0 |
| `x2` | 3 | 1 |
| `y2` | 3 | 1 |

Taxonomy by edge:

| edge band | rank moving | stable calibration | hard/stable | fragile basin |
| --- | ---: | ---: | ---: | ---: |
| `low_edge` | 1 | 0 | 1 | 0 |
| `high_edge` | 0 | 1 | 1 | 0 |
| `interior` | 10 | 1 | 0 | 1 |

## Row-Level Join

Direction abbreviations:

- MTC: `masked_to_control`;
- CTM: `control_to_masked`.

| class | slot | select | checkpoint | record | desc | target | mask rank/top1 | prior rows | same desc | coord exact/r4/r16 | bbox iou>=0.5 | best repair | worst damage |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| `rank_moving_repair` | `x1` | `onset_instability` | `none_latest_ckpt32` | 114 | `backpack` | 0 | 17/True | 3 | 0 | 2/3/3 | 0 | CTM `mlp L24` | MTC `mlp L24` |
| `rank_moving_repair` | `x1` | `onset_instability` | `aligner_parent_ckpt1824` | 37 | `carrot` | 467 | 4/True | 2 | 0 | 0/0/0 | 0 | MTC `mlp L27` | MTC `mlp L16` |
| `rank_moving_repair` | `x1` | `onset_instability` | `aux_latest_ckpt32` | 50 | `person` | 328 | 75/True | 8 | 4 | 0/0/1 | 0 | CTM `mlp L27` | MTC `mlp L16` |
| `rank_moving_repair` | `x1` | `onset_instability` | `none_latest_ckpt32` | 50 | `surfboard` | 354 | 69/True | 2 | 0 | 0/0/1 | 0 | CTM `mlp L16` | MTC `mlp L24` |
| `no_effect_hard_or_stable` | `y1` | `rank_band_precursor` | `aligner_parent_ckpt1824` | 37 | `dining table` | 2 | 0/False | 3 | 0 | 0/0/1 | 1 | MTC `mlp L24` | MTC `mlp L24` |
| `rank_moving_repair` | `y1` | `onset_instability` | `aligner_parent_ckpt1824` | 37 | `carrot` | 170 | 407/True | 2 | 0 | 0/0/0 | 0 | CTM `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `y1` | `onset_instability` | `aux_latest_ckpt32` | 47 | `scissors` | 420 | 397/True | 1 | 0 | 0/0/1 | 0 | CTM `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `y1` | `onset_instability` | `none_latest_ckpt32` | 114 | `person` | 123 | 160/True | 5 | 1 | 0/0/1 | 1 | CTM `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `x2` | `top1_probability_drift` | `no_aligner_parent_ckpt3668` | 88 | `cow` | 999 | 0/False | 2 | 0 | 1/1/1 | 0 | MTC `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `x2` | `onset_instability` | `aux_latest_ckpt32` | 79 | `bottle` | 579 | 70/True | 9 | 0 | 0/0/1 | 1 | CTM `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `x2` | `onset_instability` | `aux_latest_ckpt32` | 47 | `person` | 218 | 162/True | 3 | 2 | 0/0/0 | 0 | CTM `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `x2` | `onset_instability` | `none_latest_ckpt32` | 114 | `backpack` | 323 | -119/True | 3 | 0 | 0/0/0 | 0 | MTC `mlp L20` | CTM `mlp L16` |
| `no_effect_hard_or_stable` | `y2` | `onset_instability` | `aux_latest_ckpt32` | 47 | `person` | 999 | -95/True | 3 | 2 | 0/0/0 | 0 | MTC `mlp L20` | CTM `mlp L20` |
| `rank_moving_repair` | `y2` | `onset_instability` | `aux_latest_ckpt32` | 47 | `scissors` | 902 | 517/True | 1 | 0 | 0/0/0 | 0 | CTM `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `y2` | `onset_instability` | `no_aligner_parent_ckpt3668` | 88 | `person` | 826 | 0/True | 3 | 1 | 0/0/0 | 0 | MTC `mlp L24` | MTC `mlp L27` |
| `patch_damage_fragile_basin` | `y2` | `onset_instability` | `no_aligner_parent_ckpt3668` | 50 | `cell phone` | 479 | 0/True | 3 | 0 | 0/0/0 | 0 | CTM `mlp L16` | MTC `mlp L20` |

## Read

This panel strongly separates the easy probability-drift surface from the
rank-moving/onset-instability surface.

1. Selecting instability changes the patch taxonomy. The previous all-slot
   probability-drift panel had `rank_moving_repair=2/16`; this balanced
   instability panel has `rank_moving_repair=11/16`.
2. The MTC-damage asymmetry persists and strengthens: `14/16` rows have worst
   damage under MTC. In the repair direction, CTM is now favored (`10/16` best
   repair), which is the cleanest version so far of the "masked-side residual
   is unsafe to insert into control; control-side residual often repairs
   masked instability" hypothesis.
3. Interior coordinates carry most of the rank-moving mechanism:
   `10/12` interior rows are `rank_moving_repair`, while high-edge rows are not
   rank-moving in this panel. This supports separating coordinate edge basins
   from interior onset dynamics.
4. `x1` and `y1` are no longer peripheral. All `x1` rows are rank-moving
   repairs, and `y1` has three strong rank-moving repairs. Earlier x2-focused
   evidence remains important, but this panel says the hidden onset mechanism
   is not exclusively an `x2` phenomenon.
5. Local history remains heterogeneous. Strong repairs appear with dense prior
   same-description history (`x1` person: 8 prior rows, 4 same-desc) and sparse
   history (`y2` scissors: 1 prior row, 0 same-desc). So local history is a
   modulator and candidate amplifier, not a necessary condition by itself.

Working mechanism update:

```text
Duplication precursors are better read as a slot-local instability state than
as simple top-1 probability drift. The most reproducible intervention signature
is directional: MTC often damages, while CTM often repairs masked instability.
Interior coord slots expose this signature most clearly; edge coord basins can
look unstable under masking while remaining rank-stable or hard to patch.
```

## Next Step

The most promising next path is a hidden-state/attention localization pass over
the high-yield rank-moving rows:

- prioritize the `x1` and `y1` rank-moving rows, because they make the mechanism
  broader than the previous x2-only story;
- keep `x2` bottle/person rows as bridge cases to the earlier x2 evidence;
- compare CTM-best/MTC-worst rows against the two CTM-worst exceptions
  (`x2` backpack and `y2` high-edge person);
- inspect attention/logit concentration around the pre-coordinate hidden states
  at L16/L20/L24/L27, starting with the frequent `mlp L16` repair/damage site.

This is the path most likely to affect the final picture, because it can
separate "autoregressive local history loop" from "slot-local residual
instability that history sometimes amplifies."

## Verification

Code checks:

```bash
python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py -q

python - <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
path=Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py')
spec=spec_from_file_location('selector_tests', path)
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
print('direct_pre_onset_patch_selector_tests_passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_patch_selector.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py
```

Artifact check:

```bash
find /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1 \
  -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
```

Observed key artifact sizes:

```text
history_features/pre_onset_history_feature_report.md 3460 bytes
history_features/pre_onset_history_feature_rows.jsonl 18760 bytes
history_features/pre_onset_history_feature_summary.json 1744 bytes
pre_onset_patch_selected_rows.jsonl 18543 bytes
pre_onset_patch_selector_summary.json 3631 bytes
pre_onset_patch_target_manifest.json 22582 bytes
residual_patch_selected_cases/phase4_residual_patch_summary.json 93802 bytes
residual_patch_selected_cases/residual_patch_rows.jsonl 1331834 bytes
```
