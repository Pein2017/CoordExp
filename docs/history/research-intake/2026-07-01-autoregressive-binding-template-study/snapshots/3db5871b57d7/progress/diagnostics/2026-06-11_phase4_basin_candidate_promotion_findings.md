# Phase 4 Basin Candidate Promotion Findings

Date: 2026-06-11

Scope: convert selected basin-split candidate boxes into one-candidate
`duplicate_basin` region artifacts for the existing paired route/content GPU
patcher. This is a CPU artifact bridge, not a model-forward result.

## Why This Exists

The paired route/content probe loads token traces by the real `record_idx`, and
region rows are keyed by `(checkpoint_label, record_idx)`. Therefore candidate
promotion should not synthesize record ids. Each candidate is emitted as its own
subdirectory with one paired manifest row and one promoted region row.

This keeps the existing intervention contract unchanged:

```text
region_kind = duplicate_basin
patch_components = duplicate_basin
```

Only the basin bbox changes.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_basin_candidate_promotion.py
scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_promotion.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_promotion.py
```

The charter was also updated to include the false-negative visibility/guidance
probe as a deterministic pre-dessert panel before treating missing objects as a
visual-capacity limit.

## Artifact

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Files:

```text
promotion_index.jsonl
phase4_basin_candidate_promotion_summary.json
phase4_basin_candidate_promotion_report.md
promoted_cases/<candidate_id>/paired_manifest_rows.jsonl
promoted_cases/<candidate_id>/region_rows.jsonl
```

Summary:

```text
candidate_count = 10
record_counts = {"33": 3, "50": 2, "114": 5}
```

Candidate kinds:

| candidate_kind | count |
|---|---:|
| `original_duplicate_basin` | 4 |
| `component_row_6_person` | 1 |
| `component_row_7_handbag` | 1 |
| `component_row_9_person` | 1 |
| `non_target_desc_sub_envelope` | 1 |
| `target_desc_sub_envelope` | 1 |
| `target_row_bbox` | 1 |

## Shortlist

| candidate_id | record | row | kind | target-desc-frac |
|---|---:|---:|---|---:|
| `record33_target25_non_target_desc_sub_envelope` | 33 | 25 | `non_target_desc_sub_envelope` | 0.000 |
| `record33_target25_original_duplicate_basin` | 33 | 25 | `original_duplicate_basin` | 1.000 |
| `record33_target25_target_desc_sub_envelope` | 33 | 25 | `target_desc_sub_envelope` | 1.000 |
| `record50_target9_component_row_9_person` | 50 | 9 | `component_row_9_person` | 1.000 |
| `record50_target9_original_duplicate_basin` | 50 | 9 | `original_duplicate_basin` | 1.000 |
| `record114_target4_component_row_6_person` | 114 | 4 | `component_row_6_person` | 0.000 |
| `record114_target4_original_duplicate_basin` | 114 | 4 | `original_duplicate_basin` | 0.000 |
| `record114_target4_target_row_bbox` | 114 | 4 | `target_row_bbox` | 1.000 |
| `record114_target7_component_row_7_handbag` | 114 | 7 | `component_row_7_handbag` | 1.000 |
| `record114_target7_original_duplicate_basin` | 114 | 7 | `original_duplicate_basin` | 0.000 |

## Contract Assertion

Checked `record114_target4_original_duplicate_basin`:

```text
paired manifest row count = 1
region row count = 1
manifest record_idx = 114
manifest row_idx = 4
region record_idx = 114
region_kind = duplicate_basin
bbox_norm1000_xyxy = [431, 157, 895, 989]
candidate_target_desc_fraction = 0.0
```

This is the intended destructive clock test: run the original person basin,
target-local clock basin, and row-6 person component through the same
route/content patcher and compare the coordinate target probability, radius
mass, expected absolute error, and donor/target basin movement.

## Verification

Syntax:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_basin_candidate_promotion.py \
  scripts/analysis/run_autoregressive_duplication_phase4_basin_candidate_promotion.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_promotion.py
```

Direct tests passed:

```text
test_build_basin_candidate_promotions_preserves_record_idx_and_promotes_bbox
test_load_paired_manifest_by_target_rejects_duplicate_keys
test_parse_candidate_ids_accepts_repeated_and_csv_values
test_select_candidate_rows_filters_and_reports_missing_ids
```

Artifact assertions passed for the 10-candidate shortlist.

Repo pytest wrapper note: `python -m pytest
tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_candidate_promotion.py
-q` returned `Pytest: No tests collected`, consistent with the known local
wrapper behavior in this worktree.
