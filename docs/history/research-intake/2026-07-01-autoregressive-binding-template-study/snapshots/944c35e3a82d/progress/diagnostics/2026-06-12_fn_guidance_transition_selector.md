# FN Guidance Transition Selector

Date: 2026-06-12

## Scope

This phase converts the cross-checkpoint FN guidance decode panels into a
hidden-state target list.

The previous decode panels showed that many selected false negatives are
recoverable only after adding target coordinate context. This selector extracts
case-level transitions rather than relying on manual report reading.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_fn_guidance_transition_selector.py
scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_transition_selector.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_guidance_transition_selector.py
```

## Inputs

Decode roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aligner_parent_ckpt1824_prefix_depth_coordslot_cap1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aux_latest_ckpt32_guarded_prefix_depth_coordslot_cap1
```

## Output

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/cross_checkpoint_transition_selector
```

Files:

```text
fn_guidance_transition_rows.jsonl
phase4_fn_guidance_transition_summary.json
phase4_fn_guidance_transition_report.md
```

## Transition Classes

The selector assigns each `(checkpoint, image, gt_idx, prefix)` case to one
mechanistic transition class:

- `coord_slot_unlock`: `desc_x1` fails, but `desc_x1_y1` or `desc_x1_y1_x2`
  succeeds.
- `hard_no_rescue`: no tested tier succeeds, including `desc_x1_y1_x2`.
- `x1_specific_unlock`: `desc_x1` succeeds while wrong-control x1 does not.
- `semantic_context_accessible`: `desc_only` already succeeds.
- `wrong_control_ambiguous`: target x1 and wrong-control x1 both succeed.
- `partial_or_unclassified`: fallback for incomplete or mixed cases.

## Result

Rows:

```text
row_count = 36
hidden_state_target_count = 11
```

Class counts:

| transition class | count |
| --- | ---: |
| `coord_slot_unlock` | 7 |
| `hard_no_rescue` | 4 |
| `semantic_context_accessible` | 4 |
| `wrong_control_ambiguous` | 10 |
| `x1_specific_unlock` | 11 |

Checkpoint split:

| checkpoint | coord-slot unlock | hard no-rescue | semantic accessible | wrong-control ambiguous | x1-specific unlock |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | 1 | 2 | 0 | 4 | 5 |
| `aligner_parent_ckpt1824` | 2 | 0 | 3 | 3 | 4 |
| `aux_latest_ckpt32_guarded` | 4 | 2 | 1 | 3 | 2 |

## Recommended Hidden-State Targets

| checkpoint | image | gt | prefix | desc | class | best tier | desc-x1 IoU | unlock IoU |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| `aligner_parent_ckpt1824` | 1761 | 6 | `0` | person | `coord_slot_unlock` | `desc_x1_y1_x2` | 0.157 | 0.571 |
| `aligner_parent_ckpt1824` | 1761 | 6 | `all` | person | `coord_slot_unlock` | `desc_x1_y1_x2` | 0.167 | 0.571 |
| `aux_latest_ckpt32_guarded` | 139 | 7 | `0` | vase | `coord_slot_unlock` | `desc_x1_y1_x2` | 0.256 | 0.769 |
| `aux_latest_ckpt32_guarded` | 139 | 13 | `0` | chair | `coord_slot_unlock` | `desc_x1_y1` | 0.471 | 0.641 |
| `aux_latest_ckpt32_guarded` | 1353 | 3 | `0` | person | `coord_slot_unlock` | `desc_x1_y1` | 0.032 | 0.631 |
| `aux_latest_ckpt32_guarded` | 1353 | 3 | `all` | person | `coord_slot_unlock` | `desc_x1_y1` | 0.000 | 0.601 |
| `no_aligner_parent_ckpt3668` | 139 | 7 | `all` | vase | `coord_slot_unlock` | `desc_x1_y1` | 0.108 | 0.615 |
| `aux_latest_ckpt32_guarded` | 139 | 17 | `0` | book | `hard_no_rescue` | none | 0.079 | 0.127 |
| `aux_latest_ckpt32_guarded` | 139 | 17 | `all` | book | `hard_no_rescue` | none | 0.065 | 0.127 |
| `no_aligner_parent_ckpt3668` | 139 | 17 | `0` | book | `hard_no_rescue` | none | 0.106 | 0.182 |
| `no_aligner_parent_ckpt3668` | 139 | 17 | `all` | book | `hard_no_rescue` | none | 0.103 | 0.209 |

## Mechanism Read

This makes the next hidden-state analysis deterministic:

1. `coord_slot_unlock` rows test whether adding one more coordinate token
   redirects the residual stream from a wrong local basin into the target basin.
2. `hard_no_rescue` rows provide contrast cases where even x1/y1/x2 guidance
   cannot recover the object, suggesting missing extent/visibility evidence or
   a different failure mode.
3. The aux checkpoint contributes more coord-slot unlocks and hard contrasts
   than the parent panels in this small sample. That makes it especially useful
   for separating changed mechanics from simple perception failure.
4. The parent vase case remains the cleanest bridge to the prior residual patch
   and direction-decomposition evidence.

## Next Hook

Use this selector output as the source of truth for a hidden-state patch panel:

- source condition: successful tier, usually `desc_x1_y1` or
  `desc_x1_y1_x2`;
- target condition: failed `desc_x1`;
- contrast: `hard_no_rescue` book rows under the same checkpoint and prefix
  depths;
- readout: next missing coordinate slot and coordinate-basin margin.

## Verification

Unit and syntax checks:

```bash
python - <<'PY'
import importlib.util
path='tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_guidance_transition_selector.py'
spec=importlib.util.spec_from_file_location('transition_tests', path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import tempfile
from pathlib import Path
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    fn = getattr(mod, name)
    if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    else:
        fn()
print('direct fn guidance transition selector harness passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_fn_guidance_transition_selector.py \
  scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_transition_selector.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_guidance_transition_selector.py
```

Both passed.

Repo pytest wrapper note:

```bash
python -m pytest -q tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_guidance_transition_selector.py
```

returned `Pytest: No tests collected`, consistent with the local test wrapper
behavior observed in this mechanism worktree.

Materialization command:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_transition_selector.py \
  --decode-root no_aligner_parent_ckpt3668=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1 \
  --decode-root aligner_parent_ckpt1824=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aligner_parent_ckpt1824_prefix_depth_coordslot_cap1 \
  --decode-root aux_latest_ckpt32_guarded=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aux_latest_ckpt32_guarded_prefix_depth_coordslot_cap1 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/cross_checkpoint_transition_selector
```
