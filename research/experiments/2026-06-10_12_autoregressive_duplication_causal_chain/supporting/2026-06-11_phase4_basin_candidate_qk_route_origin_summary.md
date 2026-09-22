# Phase 4 Basin Candidate Q/K Route-Origin Summary

Date: 2026-06-11

Scope: run promoted basin-split candidates through the existing Q/K
route-origin probe at layer 17 head 1. This is a causal-adjacent score-origin
readout over the same promoted `duplicate_basin` region rows used by the
candidate route/content probes.

GPU scope: 4 A100 GPUs, devices `0,1,2,3`.

## Artifact

Promotion root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Q/K prep files:

```text
qk_prep_index.jsonl
phase4_basin_candidate_qk_prep_summary.json
promoted_cases/<candidate_id>/qk_token_windows.jsonl
```

Per-candidate Q/K outputs:

```text
promoted_cases/<candidate_id>/qk_route_origin_layer17_head1/
  qk_route_origin_rows.jsonl
  phase4_qk_route_origin_summary.json
```

Aggregate output:

```text
candidate_qk_route_origin_summary/
  candidate_qk_route_origin_summary_rows.jsonl
  candidate_qk_route_origin_summary.json
  candidate_qk_route_origin_summary_report.md
```

Coverage:

```text
candidate_count = 10
qk_row_count = 30
patch components = duplicate_basin, visual_near_ring, visual_far_background
missing_candidates = []
```

Every promoted candidate produced exactly three Q/K rows.

## Duplicate-Basin Q/K Terms

| candidate | rec | row | kind | route p rec | attn delta | score mean delta | key/control-query | query/masked-keys | lse delta |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `record33_target25_non_target_desc_sub_envelope` | 33 | 25 | `non_target_desc_sub_envelope` | 0.00022 | 0.00011 | 0.92191 | 0.93346 | -0.01155 | 0.72929 |
| `record33_target25_original_duplicate_basin` | 33 | 25 | `original_duplicate_basin` | 0.03324 | 0.57199 | 3.19067 | 6.21188 | -3.02121 | 3.19067 |
| `record33_target25_target_desc_sub_envelope` | 33 | 25 | `target_desc_sub_envelope` | -0.00488 | 0.53955 | 3.72939 | 3.60606 | 0.12332 | 3.72939 |
| `record50_target9_component_row_9_person` | 50 | 9 | `component_row_9_person` | 0.00002 | 0.86539 | 4.96628 | 3.91422 | 1.05206 | 6.84599 |
| `record50_target9_original_duplicate_basin` | 50 | 9 | `original_duplicate_basin` | 0.00784 | 0.12291 | -0.37891 | 1.07029 | -1.44920 | 2.22717 |
| `record114_target4_component_row_6_person` | 114 | 4 | `component_row_6_person` | 0.01380 | 0.93253 | 2.48314 | 0.45153 | 2.03161 | 13.99394 |
| `record114_target4_original_duplicate_basin` | 114 | 4 | `original_duplicate_basin` | 0.02113 | 0.84757 | 2.41282 | 0.48719 | 1.92563 | 13.53032 |
| `record114_target4_target_row_bbox` | 114 | 4 | `target_row_bbox` | 0.00225 | -0.01953 | 3.77618 | 4.41622 | -0.64004 | 3.77618 |
| `record114_target7_component_row_7_handbag` | 114 | 7 | `component_row_7_handbag` | -0.03128 | -0.30703 | -1.90779 | 0.33821 | -2.24600 | -5.12276 |
| `record114_target7_original_duplicate_basin` | 114 | 7 | `original_duplicate_basin` | 0.00267 | 0.45275 | 5.15198 | 1.33322 | 3.81876 | 11.19453 |

## Mechanistic Read

The promoted-candidate Q/K probe refines the route/content finding:

```text
Q/K routing strength is not sufficient for useful coordinate-slot repair.
```

Concrete examples:

- `record33_target25_target_desc_sub_envelope` has strong duplicate-basin
  attention delta (`0.53955`) and score delta (`3.72939`), but route/content
  total probability recovery is negative (`-0.00488`).
- `record50_target9_component_row_9_person` has the largest record-50
  attention delta (`0.86539`) and strong score delta (`4.96628`), but
  route/content total probability recovery is near zero (`0.00002`).
- `record114_target4_target_row_bbox` has positive score delta (`3.77618`) but
  negative duplicate-basin attention delta (`-0.01953`) and weak route/content
  recovery (`0.00225`).
- `record114_target7_component_row_7_handbag` is negative on both Q/K and
  route/content surfaces: attention delta `-0.30703`, score delta `-1.90779`,
  route/content recovery `-0.03128`.

The original route-selected basins remain the most useful downstream candidates
in the route/content probe, but their Q/K signatures differ by case:

- record 33 original basin is strongly key-side under the control query:
  `key/control-query = 6.21188`, `query/masked-keys = -3.02121`;
- record 114 row 4 original basin is more mixed, with both query-side and
  key-side positive terms;
- record 50 original basin has weak attention delta and negative mean score
  delta, yet still beats the target-local person component downstream.

This points to a two-stage mechanism rather than a one-scalar routing story:

```text
1. Q/K chooses or fails to choose a visual basin.
2. The selected basin's value/content contribution must still align with the
   current coordinate slot.
```

Large attention movement can select a basin whose values are not helpful for
the target coordinate. Conversely, a smaller route-selected basin can be more
useful downstream if its value projection is better aligned with the coordinate
basin.

## Next Probe

The next high-leverage probe is to connect this Q/K score origin with the value
basin projection for the same promoted candidates:

- compare duplicate-basin Q/K attention delta with `total_delta` and
  `route_delta_masked_values` recovery;
- add value-basin projection rows for the same promoted candidate boxes if they
  are not already available;
- prioritize cases where Q/K and route/content disagree:
  - record 33 target-desc envelope;
  - record 50 target-local person;
  - record 114 target-local clock;
  - record 114 target-local handbag.

## Verification

Q/K prep:

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_basin_candidate_qk_prep.py \
  scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_qk_prep.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_qk_prep.py

direct importlib execution:
  test_materialize_basin_candidate_qk_prep_copies_matching_token_window
  test_materialize_basin_candidate_qk_prep_rejects_missing_anchor

real prep artifact:
  candidate_count = 10
  qk_prep_index_row_count = 10
```

Q/K GPU run:

```text
per-candidate qk_route_origin_row_count = 3
candidate_count = 10
aggregate qk_row_count = 30
missing_candidates = []
```

The local `python -m pytest ...` wrapper still reports `Pytest: No tests
collected`, consistent with the known worktree behavior.
