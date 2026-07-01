# Phase 4 Basin Split Candidate Findings

Date: 2026-06-11

Scope: generate a concrete candidate menu for the next basin-specificity causal
split. This is a post-hoc artifact builder, not a model-forward run. It turns
the manual-review basin-specificity rows into candidate sub-basins/envelopes
that can later be promoted one at a time to `region_kind=duplicate_basin` for
GPU patching.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_basin_split_candidates.py
scripts/analysis/run_autoregressive_duplication_phase4_basin_split_candidates.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_split_candidates.py
```

## Artifacts

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_split_candidates_layer17_head1_constructive_destructive
```

Files:

```text
basin_split_candidate_rows.jsonl
phase4_basin_split_candidates_summary.json
phase4_basin_split_candidates_report.md
```

Inputs:

```text
phase4_basin_specificity_review_layer17_head1_constructive_destructive/basin_specificity_review_rows.jsonl
phase2_region_rows.jsonl
```

Summary:

```text
candidate_row_count = 226
target_desc_sub_envelope candidates = 14
non_target_desc_sub_envelope candidates = 14
original_duplicate_basin candidates = 14
target_row_bbox candidates = 14
```

Every candidate row has:

- `candidate_id`
- target row identity and current mechanism scores
- `candidate_kind`
- `candidate_bbox_norm1000_xyxy`
- included source rows and descriptor counts
- target-desc fraction
- IoU with target bbox
- IoU with the original duplicate basin
- promotion note for the later causal run

## Key Candidate Contrasts

### Constructive Record 33

Representative target: `record=33 row=25 desc=wine glass`.

| candidate | bbox | source rows | target-desc frac | IoU target | IoU original dup |
|---|---|---|---:|---:|---:|
| `original_duplicate_basin` | `[124,339,159,415]` | row 23 wine glass | 1.0 | 0.639506 | 1.0 |
| `target_desc_sub_envelope` | `[105,339,170,425]` | rows 23,25,26,27,28,29,30,31 wine glass | 1.0 | 0.474419 | 0.475850 |
| `non_target_desc_sub_envelope` | `[188,474,264,632]` | rows 20,21,22,24 bottle | 0.0 | 0.0 | 0.0 |

This creates the direct constructive causal menu:

```text
tiny original wine-glass basin
vs wider wine-glass envelope
vs local bottle-only envelope
```

The cleanest next run should test whether the tiny basin is uniquely strong or
whether the wider target-desc envelope substitutes.

### Destructive Record 114

Representative target: `record=114 row=4 desc=clock`.

| candidate | bbox | source rows | target-desc frac | IoU target | IoU original dup |
|---|---|---|---:|---:|---:|
| `original_duplicate_basin` | `[431,157,895,989]` | row 6 person | 0.0 | 0.006994 | 1.0 |
| `component_row_6_person` | `[431,157,895,988]` | row 6 person | 0.0 | 0.007002 | 0.998798 |
| `target_row_bbox` / `target_desc_sub_envelope` | `[749,489,803,539]` | row 4 clock | 1.0 | 1.0 | 0.006994 |

This is the strongest destructive origin handle so far. The original
`duplicate_basin` for the clock target is effectively a large person box, not a
clock-local basin. That explains why attention mass or route strength can be
large while the coordinate target-centered value projection is destructive.

For `record=114 row=7 desc=handbag`, the same original duplicate basin is row 6
person, while the target-local handbag candidate is:

```text
component_row_7_handbag = [519,320,753,748]
```

### Destructive Contrast Record 50

Representative target: `record=50 row=9 desc=person`.

| candidate | bbox | source rows | target-desc frac | IoU target | IoU original dup |
|---|---|---|---:|---:|---:|
| `original_duplicate_basin` | `[531,514,758,871]` | row 5 person | 1.0 | 0.208885 | 1.0 |
| `component_row_9_person` | `[592,228,769,682]` | row 9 person | 1.0 | 1.0 | 0.208885 |

This remains the best contrast case for separating route strength from target
alignment. The original basin is same-desc but spatially offset from the target
person, unlike record `114` where it is also semantically mismatched.

## Mechanistic Read

The candidate menu makes the next causal experiment concrete:

```text
record 33:
  original tiny wine-glass basin vs wider wine-glass envelope vs bottle-only envelope

record 114:
  original person basin vs target-local clock/handbag basin

record 50/9:
  offset same-desc person basin vs target-local person basin
```

This directly attacks the origin question:

```text
Does layer17/head1 repair depend on a small route-selected visual basin that is
both spatially and semantically aligned with the current coordinate slot?
```

The most important new evidence is that record `114` is not just "diffuse." Its
destructive original basin is concretely a large person candidate used while the
next coordinate target is clock/handbag. That turns an abstract
constructive/destructive split into a testable basin-misalignment mechanism.

## Promotion Boundary

The current GPU patchers still assume that the masked intervention and source
membership are named `duplicate_basin`. Therefore these candidate rows are not
directly runnable as a multi-candidate batch.

For a causal run, promote exactly one candidate at a time:

1. take its `candidate_bbox_norm1000_xyxy`;
2. create a temporary region row for the same target/case with
   `region_kind=duplicate_basin`;
3. rerun the existing masked/control patcher;
4. compare target probability, radius-4/radius-8 mass, expected absolute error,
   and donor/target basin movement.

This avoids changing the semantics of existing Phase 2 region artifacts.

## Verification

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_basin_split_candidates.py \
  scripts/analysis/run_autoregressive_duplication_phase4_basin_split_candidates.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_split_candidates.py

direct importlib execution of:
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_split_candidates.py

artifact assertions:
  candidate_row_count = 226
  target_desc_sub_envelope = 14
  non_target_desc_sub_envelope = 14
  record114/row4 original_duplicate_basin source row = row 6 person
  record114/row4 original_duplicate_basin target-desc fraction = 0.0
```

The local `python -m pytest ...` wrapper still reports `Pytest: No tests
collected`, matching the known worktree behavior.
