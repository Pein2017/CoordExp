# Pre-Onset Rank-Moving Site Localization

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_balanced_rank_onset_selector.md`.
That panel made the main residual-patch signature much sharper:

```text
rank_moving_repair=11/16
worst_damage masked_to_control=14/16
best_repair control_to_masked=10/16
```

This slice performs a post-hoc site/layer localization over those patch rows.
It does not run the model again. It reduces the existing residual patch rows
into target-level best-repair and worst-damage sites, then summarizes the
rank-moving subset by slot, edge band, site, and layer.

## Implementation

New reducer:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_site_localization.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_site_localization.py
```

New test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_site_localization.py
```

The reducer joins:

- `history_features/pre_onset_history_feature_rows.jsonl`;
- `residual_patch_selected_cases/residual_patch_rows.jsonl`.

It writes:

- `pre_onset_site_localization_rows.jsonl`;
- `pre_onset_site_localization_summary.json`;
- `pre_onset_site_localization_report.md`.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_site_localization.py \
  --history-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/history_features/pre_onset_history_feature_rows.jsonl \
  --residual-patch-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/residual_patch_selected_cases/residual_patch_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization
```

Observed artifact sizes:

```text
pre_onset_site_localization_report.md 4171 bytes
pre_onset_site_localization_rows.jsonl 16287 bytes
pre_onset_site_localization_summary.json 12293 bytes
```

## Summary

Target-level direction signatures:

| signature | all rows | rank-moving rows |
| --- | ---: | ---: |
| `ctm_repair_mtc_damage` | 9 | 9 |
| `mtc_repair_ctm_damage` | 2 | 1 |
| `same_direction_masked_to_control` | 5 | 1 |

Rank-moving best repair site/layer:

| site/layer | rows |
| --- | ---: |
| `mlp L16` | 7 |
| `mlp L20` | 2 |
| `mlp L24` | 1 |
| `mlp L27` | 1 |

Rank-moving worst damage site/layer:

| site/layer | rows |
| --- | ---: |
| `mlp L16` | 9 |
| `mlp L24` | 2 |

Rank-moving slot summary:

| slot | rows | signature | mean best repair | mean worst damage | best site/layer | worst site/layer |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `x1` | 4 | CTM/MTC 3, same-MTC 1 | 45.75 | 42.75 | L16/L20/L24/L27 split | L16 2, L24 2 |
| `x2` | 3 | CTM/MTC 2, MTC/CTM 1 | 80.33 | 71.00 | L16 2, L20 1 | L16 3 |
| `y1` | 3 | CTM/MTC 3 | 318.67 | 248.67 | L16 3 | L16 3 |
| `y2` | 1 | CTM/MTC 1 | 539.00 | 530.00 | L16 1 | L16 1 |

Rank-moving edge summary:

| edge band | rows | signature | best site/layer | worst site/layer |
| --- | ---: | --- | --- | --- |
| `interior` | 10 | CTM/MTC 8, MTC/CTM 1, same-MTC 1 | L16 7, L20 2, L27 1 | L16 9, L24 1 |
| `low_edge` | 1 | CTM/MTC 1 | L24 1 | L24 1 |

Patch-group totals over the 11 rank-moving rows:

| patch group | mean repair | sum repair | mean damage | sum damage |
| --- | ---: | ---: | ---: | ---: |
| `control_to_masked mlp L16` | 159.09 | 1750 | -3.09 | -34 |
| `control_to_masked mlp L20` | 132.27 | 1455 | 23.73 | 261 |
| `control_to_masked mlp L24` | 129.36 | 1423 | 26.64 | 293 |
| `control_to_masked mlp L27` | 101.09 | 1112 | 54.91 | 604 |
| `masked_to_control mlp L16` | 17.18 | 189 | 138.82 | 1527 |
| `masked_to_control mlp L20` | 53.45 | 588 | 102.55 | 1128 |
| `masked_to_control mlp L24` | 86.64 | 953 | 69.36 | 763 |
| `masked_to_control mlp L27` | 129.27 | 1422 | 26.73 | 294 |

The `post_attention_residual` groups have identical values to the `mlp` groups
in this artifact, so this reducer localizes the mechanism to the layer
boundary, not yet to a unique sub-block inside that boundary.

## Read

The strongest localization is layer-16 residual boundary:

1. For rank-moving targets, the best repair site/layer is `mlp L16` in `7/11`
   rows, and the worst damage site/layer is `mlp L16` in `9/11` rows.
2. At the patch-group level, CTM L16 has the largest total rank recovery
   (`1750`) and MTC L16 has the largest total rank damage (`1527`).
3. The signature is strongest in interior coordinates: `10/11` rank-moving rows
   are interior or low-edge, with `10/12` interior rows from the previous
   selector summary already classified as rank-moving repair. High-edge rows do
   not carry this rank-moving site-localization signal.
4. `y1` is the cleanest slot: all three rank-moving `y1` rows are
   CTM-repair/MTC-damage, all best repairs are L16, and all worst damages are
   L16. This is now at least as important as the earlier x2 bridge cases.
5. The exceptions are useful, not noise:
   - `x2` backpack, record `114`, target `323`, is MTC-repair/CTM-damage;
   - `x1` carrot, record `37`, target `467`, is same-direction MTC with weak
     rank movement.

Working mechanism update:

```text
The balanced onset-instability panel points to an early residual-boundary
instability, concentrated around layer 16. For most rank-moving interior coord
targets, control-side state at L16 repairs masked instability, while masked-side
state at L16 damages the control stream. This makes L16 the first-priority
attention/logit routing layer for the next probe.
```

## Next Step

The next high-yield probe should move from residual-site localization to
attention/logit routing at L16:

- use the CTM-repair/MTC-damage `y1` rows as the clean positive set;
- keep `x2` bottle/person as bridge cases to previous x2 evidence;
- use the two exception rows as contrastive controls;
- inspect whether L16 attention/source rows concentrate on prior same-desc
  rows, prior coordinate anchors, or the current row's visual region.

This is the most direct route to separate:

```text
history-loop explanation
vs.
slot-local residual instability with history as an amplifier
```

## Verification

Code checks:

```bash
python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_site_localization.py -q

python - <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
path=Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_site_localization.py')
spec=spec_from_file_location('site_tests', path)
assert spec and spec.loader
mod=module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    fn=getattr(mod, name)
    if 'tmp_path' in fn.__code__.co_varnames:
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    else:
        fn()
print('direct_pre_onset_site_localization_tests_passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_site_localization.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_site_localization.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_site_localization.py
```

Artifact check:

```bash
find /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/site_localization \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```
