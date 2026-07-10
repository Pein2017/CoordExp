# FN Guidance Probe-Family Selector

Date: 2026-06-12

## Scope

This update refines the FN guidance transition selector so future hidden-state
probes do not mix incompatible causal questions.

The previous same-slot and cross-slot vase panels showed that:

- same-slot patches can diagnose whether hidden state repairs a missing
  coordinate basin;
- cross-slot coord forcing tests downstream coordinate accessibility and is not
  a valid repair vector for the earlier missing slot.

The selector now labels each transition row with a `causal_probe_family`.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_fn_guidance_transition_selector.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_guidance_transition_selector.py
```

The CLI is unchanged:

```text
scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_transition_selector.py
```

## Artifact

Regenerated root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/cross_checkpoint_transition_selector
```

Files:

```text
fn_guidance_transition_rows.jsonl
phase4_fn_guidance_transition_summary.json
phase4_fn_guidance_transition_report.md
```

## New Row Fields

Each row now includes:

- `causal_probe_family`
- `causal_probe_priority`
- `same_slot_desc_x1_success_prefixes`
- `same_slot_alternative_prefixes`
- `has_same_slot_alternative`
- `same_slot_probe_source_prefix`
- `same_slot_probe_target_prefix`

Family definitions:

- `same_slot_alternative_available`: the row is a `coord_slot_unlock`, and
  another prefix for the same checkpoint/image/GT succeeds at `desc_x1`. This
  is a valid same-next-slot hidden-state patch target.
- `cross_slot_only_coord_forcing`: the row is a `coord_slot_unlock`, but no
  alternate prefix succeeds at `desc_x1` in this panel. Treat it as downstream
  coordinate accessibility, not same-slot repair.
- `hard_no_rescue`: no tested tier succeeds.
- `direct_x1_accessible`: `desc_x1` succeeds and wrong-control x1 does not.
- `semantic_context_accessible`: `desc_only` succeeds.
- `wrong_control_ambiguous`: target x1 and wrong-control x1 both succeed.

## Regenerated Counts

Rows:

```text
row_count = 36
hidden_state_target_count = 11
```

Transition classes remain unchanged:

| transition class | count |
| --- | ---: |
| `coord_slot_unlock` | 7 |
| `hard_no_rescue` | 4 |
| `semantic_context_accessible` | 4 |
| `wrong_control_ambiguous` | 10 |
| `x1_specific_unlock` | 11 |

New causal probe families:

| causal probe family | count |
| --- | ---: |
| `same_slot_alternative_available` | 3 |
| `cross_slot_only_coord_forcing` | 4 |
| `hard_no_rescue` | 4 |
| `direct_x1_accessible` | 11 |
| `semantic_context_accessible` | 4 |
| `wrong_control_ambiguous` | 10 |

Checkpoint split:

| checkpoint | same-slot alt | cross-slot only | hard no-rescue | direct x1 | semantic | wrong-control ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | 1 | 0 | 2 | 5 | 0 | 4 |
| `aligner_parent_ckpt1824` | 0 | 2 | 0 | 4 | 3 | 3 |
| `aux_latest_ckpt32_guarded` | 2 | 2 | 2 | 2 | 1 | 3 |

## Hidden-State Target Routing

The 11 hidden-state target rows now split into:

### Same-Slot Alternatives

These are valid same-next-slot patch targets:

| checkpoint | image | gt | prefix | desc | source prefix | best tier |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `aux_latest_ckpt32_guarded` | 139 | 7 | `0` | vase | `all` | `desc_x1_y1_x2` |
| `aux_latest_ckpt32_guarded` | 139 | 13 | `0` | chair | `all` | `desc_x1_y1` |
| `no_aligner_parent_ckpt3668` | 139 | 7 | `all` | vase | `0` | `desc_x1_y1` |

### Cross-Slot-Only Coord Forcing

These should not be interpreted as same-slot repair unless a new same-slot
source condition is found:

| checkpoint | image | gt | prefix | desc | best tier |
| --- | ---: | ---: | ---: | --- | --- |
| `aligner_parent_ckpt1824` | 1761 | 6 | `0` | person | `desc_x1_y1_x2` |
| `aligner_parent_ckpt1824` | 1761 | 6 | `all` | person | `desc_x1_y1_x2` |
| `aux_latest_ckpt32_guarded` | 1353 | 3 | `0` | person | `desc_x1_y1` |
| `aux_latest_ckpt32_guarded` | 1353 | 3 | `all` | person | `desc_x1_y1` |

### Hard No-Rescue Contrasts

These remain candidate visibility/extent contrast rows:

| checkpoint | image | gt | prefix | desc |
| --- | ---: | ---: | ---: | --- |
| `aux_latest_ckpt32_guarded` | 139 | 17 | `0` | book |
| `aux_latest_ckpt32_guarded` | 139 | 17 | `all` | book |
| `no_aligner_parent_ckpt3668` | 139 | 17 | `0` | book |
| `no_aligner_parent_ckpt3668` | 139 | 17 | `all` | book |

## Mechanism Read

This refinement makes the next hidden-state work cleaner:

1. The parent vase and aux vase/chair rows are same-slot patch targets. They
   can test whether a prefix condition contains a portable hidden-state repair
   for the missing `y1` basin.
2. The aligner person and aux person rows are cross-slot-only in the current
   panel. They show conditional accessibility after target coordinate forcing,
   but should not be treated as hidden same-slot repair without another source.
3. Book remains a hard no-rescue contrast, especially for y2/extent.

This keeps the causal vocabulary aligned with the evidence: same-slot repair,
downstream coordinate accessibility, direct x1 accessibility, and hard
visibility/extent failure are now separate artifact-level labels.

## Verification

Checks:

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

returned `Pytest: No tests collected`, consistent with the local wrapper
behavior in this worktree.

Materialization:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_transition_selector.py \
  --decode-root no_aligner_parent_ckpt3668=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1 \
  --decode-root aligner_parent_ckpt1824=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aligner_parent_ckpt1824_prefix_depth_coordslot_cap1 \
  --decode-root aux_latest_ckpt32_guarded=/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aux_latest_ckpt32_guarded_prefix_depth_coordslot_cap1 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/cross_checkpoint_transition_selector
```
