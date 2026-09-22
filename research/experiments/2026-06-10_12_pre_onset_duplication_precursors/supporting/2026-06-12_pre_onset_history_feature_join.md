# Pre-Onset History Feature Join

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_multicase_residual_taxonomy.md`.
It does not run the model. It joins the eight selected pre-onset residual-patch
targets back to generated rollout history features:

- prior generated rows before the selected row;
- prior same-description count;
- prior coordinate-token reuse near the selected target bin;
- prior high-overlap bbox count;
- target edge/saturation band;
- best repair direction and worst damage direction from the residual-patch rows.

The purpose is to test whether the residual taxonomy is visibly connected to
autoregressive local history rather than only to isolated coordinate logits.

## Implementation

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_history_features.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py
```

Test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py
```

The reducer parses compact generated token rows directly from
`pred_token_trace.jsonl`, joins them with selected target rows and residual patch
rows, and writes:

- `pre_onset_history_feature_rows.jsonl`
- `pre_onset_history_feature_summary.json`
- `pre_onset_history_feature_report.md`

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py \
  --selector-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1 \
  --residual-patch-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/residual_patch_selected_cases \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/history_features
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/history_features
```

Counts:

- history rows: `8`;
- taxonomy classes: `patch_damage_fragile_basin=3`,
  `rank_moving_repair=2`, `top1_stable_calibration=3`;
- target edge bands: `high_edge=2`, `interior=6`;
- best repair directions: `control_to_masked=5`, `masked_to_control=3`;
- worst damage directions: `masked_to_control=7`, `control_to_masked=1`.

## Row-Level Join

| class | checkpoint | record | desc | target | prior rows | same desc | coord exact/r4/r16 | bbox iou>=0.5 | best repair | worst damage |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| `top1_stable_calibration` | `no_aligner_parent_ckpt3668` | `88` | `cow` | `999` | `2` | `0` | `1/1/1` | `0` | MTC `mlp L16` | MTC `mlp L16` |
| `top1_stable_calibration` | `aligner_parent_ckpt1824` | `37` | `bowl` | `999` | `1` | `0` | `0/0/0` | `0` | MTC `mlp L27` | MTC `mlp L27` |
| `rank_moving_repair` | `aux_latest_ckpt32` | `79` | `bottle` | `579` | `9` | `0` | `0/0/1` | `1` | CTM `mlp L16` | MTC `mlp L16` |
| `patch_damage_fragile_basin` | `no_aligner_parent_ckpt3668` | `48` | `person` | `33` | `0` | `0` | `0/0/0` | `0` | CTM `mlp L24` | MTC `mlp L16` |
| `top1_stable_calibration` | `aux_latest_ckpt32` | `50` | `person` | `395` | `7` | `3` | `0/0/2` | `0` | MTC `mlp L16` | MTC `mlp L16` |
| `rank_moving_repair` | `no_aligner_parent_ckpt3668` | `36` | `person` | `407` | `0` | `0` | `0/0/0` | `0` | CTM `mlp L16` | MTC `mlp L16` |
| `patch_damage_fragile_basin` | `aligner_parent_ckpt1824` | `37` | `carrot` | `513` | `2` | `0` | `0/0/0` | `0` | CTM `mlp L20` | CTM `mlp L24` |
| `patch_damage_fragile_basin` | `no_aligner_parent_ckpt3668` | `50` | `cell phone` | `178` | `3` | `0` | `0/0/0` | `0` | CTM `mlp L20` | MTC `mlp L20` |

Direction abbreviations:

- MTC: `masked_to_control`;
- CTM: `control_to_masked`.

Coordinate reuse columns are counts of prior generated rows containing the
selected target bin exactly, within radius `4`, or within radius `16` in any box
slot.

## Read

The strongest post-hoc signal is directional:

1. Both `rank_moving_repair` cases have best repair direction
   `control_to_masked` and worst damage direction `masked_to_control`.
   This supports the view that masked replay can expose a damaged basin, while
   injecting the unmasked/control-side residual direction repairs it.
2. Most worst-damage rows are `masked_to_control` (`7/8`). That is a useful
   asymmetry: the duplicate-basin-masked state often contains a direction that
   is unsafe to insert into the control stream, even when it is not enough to
   change top-1 rank.
3. The two high-edge `<|coord_999|>` targets are both `top1_stable_calibration`.
   They are high-confidence edge/saturation basins rather than hidden-target
   recovery cases. One has prior exact/radius reuse of `999`; one does not.
4. Prior local history is heterogeneous. The `aux_latest_ckpt32` bottle
   rank-moving repair has a dense previous history (`9` prior rows), one
   radius-16 coordinate neighbor, and one high-IoU prior box. The no-aligner
   person rank-moving repair is row `0`, with no prior rows. So repeated local
   history is sufficient in at least one strong repair case, but not necessary
   in this selected panel.

This refines the current mechanism picture: autoregressive local history is
part of the story, but the more stable cross-case feature in this small panel is
the residual direction asymmetry. The wrong masked-side residual direction is
often damaging, while control-side insertion can repair rank-moving failures.

## Next Step

The next deterministic probe should split the residual direction asymmetry by
coordinate slot and edge band:

- separate edge targets such as `<|coord_999|>` from interior targets;
- compare `x1`, `y1`, `x2`, and `y2` phases instead of only `post_y1/pre_x2`;
- check whether the MTC-damage / CTM-repair asymmetry persists in other slots.

This can start as a selector expansion over the existing precursor panel before
another GPU run.

## Verification

Code checks:

```bash
python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py -q

python - <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
path=Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py')
spec=spec_from_file_location('history_feature_tests', path)
assert spec and spec.loader
mod=module_from_spec(spec)
spec.loader.exec_module(mod)
mod.test_parse_generated_compact_rows_merges_description_pieces()
mod.test_build_history_feature_rows_joins_history_and_patch_taxonomy()
with tempfile.TemporaryDirectory() as tmp:
    mod.test_materialize_pre_onset_history_features_writes_rows_and_summary(Path(tmp))
print('direct_pre_onset_history_feature_tests_passed=3')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_history_features.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_history_features.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_history_features.py
```

The repo-local pytest wrapper printed its usual compact `No tests collected`
label but exited `0`; direct invocation ran all three test functions and passed.

Artifact check:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/history_features')
for name in [
    'pre_onset_history_feature_rows.jsonl',
    'pre_onset_history_feature_summary.json',
    'pre_onset_history_feature_report.md',
]:
    path=root/name
    assert path.is_file() and path.stat().st_size > 0, path
    print(name, path.stat().st_size)
print('pre-onset history feature artifacts exist')
PY
```
