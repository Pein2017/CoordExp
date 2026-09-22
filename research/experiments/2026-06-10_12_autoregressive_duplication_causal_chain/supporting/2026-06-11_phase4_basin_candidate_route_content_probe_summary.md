# Phase 4 Basin Candidate Route/Content Probe Summary

Date: 2026-06-11

Scope: run the promoted basin-split candidates through the existing paired
route/content patcher at layer 17 head 1. This is a causal GPU probe over the
promoted `duplicate_basin` region rows, not a new training or rollout result.

GPU scope: 4 A100 GPUs, devices `0,1,2,3`.

## Artifact

Promotion root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Aggregate output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/candidate_route_content_probe_summary
```

Files:

```text
candidate_route_content_probe_summary_rows.jsonl
candidate_route_content_probe_summary.json
candidate_route_content_probe_summary_report.md
```

Per-candidate route/content outputs:

```text
promoted_cases/<candidate_id>/route_content_probe_layer17_head1/
  paired_route_content_probe_rows.jsonl
  phase4_paired_route_content_probe_summary.json
```

Coverage:

```text
candidate_count = 10
probe_row_count = 30
effect kinds = total_delta, route_delta_masked_values, value_delta_control_route
```

Every candidate produced exactly three probe rows.

Reproducible reducer:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_probe_summary.py \
  --source-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Code surface:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_basin_candidate_probe_summary.py
scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_probe_summary.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_probe_summary.py
```

## Total-Delta Recovery

| candidate | rec | row | kind | desc-frac | masked p | patched p | recovery p | r4 rec | r8 rec | exp-err rec |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `record33_target25_non_target_desc_sub_envelope` | 33 | 25 | `non_target_desc_sub_envelope` | 0.000 | 0.02417 | 0.02439 | 0.00022 | -0.00192 | -0.00094 | 0.07771 |
| `record33_target25_original_duplicate_basin` | 33 | 25 | `original_duplicate_basin` | 1.000 | 0.01292 | 0.04616 | 0.03324 | 0.20738 | 0.35381 | -13.27989 |
| `record33_target25_target_desc_sub_envelope` | 33 | 25 | `target_desc_sub_envelope` | 1.000 | 0.03186 | 0.02698 | -0.00488 | -0.02130 | -0.01685 | -3.00580 |
| `record50_target9_component_row_9_person` | 50 | 9 | `component_row_9_person` | 1.000 | 0.00001 | 0.00003 | 0.00002 | 0.00013 | 0.00026 | -29.34826 |
| `record50_target9_original_duplicate_basin` | 50 | 9 | `original_duplicate_basin` | 1.000 | 0.02549 | 0.03333 | 0.00784 | 0.05785 | 0.09508 | -7.05569 |
| `record114_target4_component_row_6_person` | 114 | 4 | `component_row_6_person` | 0.000 | 0.01253 | 0.02632 | 0.01380 | 0.07771 | 0.14212 | -13.89566 |
| `record114_target4_original_duplicate_basin` | 114 | 4 | `original_duplicate_basin` | 0.000 | 0.00761 | 0.02874 | 0.02113 | 0.13598 | 0.24657 | -26.54542 |
| `record114_target4_target_row_bbox` | 114 | 4 | `target_row_bbox` | 1.000 | 0.02933 | 0.03159 | 0.00225 | 0.00796 | 0.01874 | -0.71326 |
| `record114_target7_component_row_7_handbag` | 114 | 7 | `component_row_7_handbag` | 1.000 | 0.03691 | 0.00562 | -0.03128 | -0.22006 | -0.39632 | 68.28273 |
| `record114_target7_original_duplicate_basin` | 114 | 7 | `original_duplicate_basin` | 0.000 | 0.00262 | 0.00529 | 0.00267 | 0.01737 | 0.03446 | -57.61958 |

## Mechanistic Read

This run does not support a simple rule that target-local or target-desc
candidates are always better causal regions for layer 17 head 1. The strongest
total-delta recovery is usually from the original route-selected basin:

```text
record33 row25: original_duplicate_basin best, target_desc_sub_envelope negative
record50 row9: original_duplicate_basin best, target-local row9 person near zero
record114 row4: original person-like basin best, target-local clock weak
record114 row7: original person-like basin weak-positive, target-local handbag strongly harmful
```

The best current interpretation is more specific:

```text
Layer17/head1 appears to carry a route-selected basin contribution, but that
basin is not guaranteed to be the semantically target-local object. When the
route-selected basin is misaligned, the head can still strongly move coordinate
probability and radius mass; swapping to target-local boxes does not
automatically repair the slot and can damage it.
```

That matters for the final picture: route strength is not equivalent to
correct object binding. The next deeper probe should ask why the query chooses
the route-selected basin in these rows, not only whether values from a
human-plausible target box contain useful coordinate content.

## Route vs Value Notes

- Record 33 row 25 original basin is route-dominant:
  `route_delta_masked_values` recovery p is `0.03401`, while
  `value_delta_control_route` is approximately zero.
- Record 114 row 4 original basin has meaningful route and value components:
  route recovery p is `0.00355`, value recovery p is `0.00907`, and total
  recovery p is `0.02113`.
- Record 114 row 7 target-local handbag is actively harmful on both route and
  value surfaces: total recovery p is `-0.03128`, route recovery p is
  `-0.03190`, and value recovery p is `-0.02268`.

## Verification

Run coverage checks:

```text
missing route/content outputs = []
per-candidate row counts = 3 for all 10 candidates
aggregate probe_row_count = 30
```

The per-candidate runner emitted successful summaries for every promoted case.

Reducer verification:

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_basin_candidate_probe_summary.py \
  scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_probe_summary.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_probe_summary.py

direct importlib execution:
  test_materialize_basin_candidate_probe_summary
  test_summarize_candidate_probe_rows_requires_completed_outputs

real artifact reducer run:
  candidate_count = 10
  probe_row_count = 30
```

The local `python -m pytest ...` wrapper still reports `Pytest: No tests
collected`, consistent with the known worktree behavior.
