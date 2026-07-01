# Phase 4 Basin Specificity Review Findings

Date: 2026-06-11

Scope: manual-review-ready post-hoc join for the layer-17 head-1 paired
constructive/destructive cases. This slice follows the paired attention
composition reducer and adds concrete row/image/box handles plus geometry
specificity metrics before any new GPU causal run.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_basin_specificity_review.py
scripts/analysis/run_autoregressive_duplication_phase4_basin_specificity_review.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_specificity_review.py
```

## Artifacts

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_specificity_review_layer17_head1_constructive_destructive
```

Files:

```text
basin_specificity_review_rows.jsonl
phase4_basin_specificity_review_summary.json
phase4_basin_specificity_review_report.md
```

Inputs:

```text
phase4_paired_attention_composition_layer17_head1_constructive_destructive/paired_attention_composition_rows.jsonl
shards/shard-*-of-08/token_windows.jsonl
phase2_region_rows.jsonl
```

Join coverage:

```text
row_count = 14
missing_token_row_count = 0
missing_duplicate_region_count = 0
```

Specificity class counts:

```text
needle_basin_in_coherent_cluster = 8
diffuse_or_mismatched_basin = 2
mixed_specificity = 4
```

## Role Summary

| role | n | source tokens | target-desc fraction | dup/same-desc area | target IoU dup | target distance dup | target delta | qk key-side mean | route prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constructive core | 8 | 1.000000 | 0.666667 | 0.056548 | 0.225616 | 35.713005 | 0.588599 | 6.542107 | 0.027665 |
| destructive core | 6 | 50.333333 | 0.307407 | 0.242052 | 0.068224 | 290.132839 | -0.096862 | 0.951147 | 0.003059 |

## Candidate Read

The constructive core is the record `33` wine-glass burst in
`images/val2017/000000002685.jpg`.

```text
rows = 23,25,26,27,28,29,30,31
duplicate basin = [124, 339, 159, 415]
same-desc/spatial envelope = [105, 339, 265, 633]
source tokens = 1
dup/same-desc envelope area ratio = 0.05655
target-desc fraction in selected component = 0.66667
```

The selected component is not semantically pure: it contains `8` wine-glass rows
and `4` bottle rows. That matters. The constructive signature is not "all
same-desc tokens"; it is a small, specific visual basin inside a mostly
target-relevant local cluster.

The destructive core separates into two families:

```text
record=114 image=000000011699 rows=4 clock, 7 handbag
source tokens = 119
duplicate basin = [431, 157, 895, 989]
same-desc/spatial envelope = [0, 123, 895, 989]
target-desc fraction = 0.2
class = diffuse_or_mismatched_basin
```

```text
record=50 image=000000005193 rows=3 cell phone, 5 person, 9 person, 10 surfboard
source tokens = 16
duplicate basin = [531, 514, 758, 871]
same-desc/spatial envelope = [112, 150, 959, 989]
target-desc fraction ~= 0.36
class = mixed_specificity
```

## Interpretation

This sharpens the origin hypothesis again. The constructive route is not just a
large visual-attention event. It is a specific basin-selection event:

```text
small duplicate-basin source
  -> mostly target-relevant local cluster
  -> strong key-side Q/K advantage
  -> route-dominant coordinate-basin repair
```

The destructive rows show what breaks:

```text
large or mixed basin
  -> lower target-desc purity
  -> weaker key-side advantage
  -> target coordinate basin is negative or near-flat
```

The fact that record `33` includes bottles inside the same selected component is
important for the final picture. The mechanism does not require a perfectly
semantic source cluster; it appears to require a tight route-selected visual
basin whose value content still aligns with the current coordinate slot. The
record `114` case is the opposite: large basin, low descriptor purity, and
destructive target-centered value-basin projection.

## Next Probe Decision

Do not fan out to unrelated heads yet. The next high-leverage branch is a
basin-specificity causal split:

1. For record `33`, compare the one-token duplicate basin against the wider
   same-desc/spatial envelope. This tests whether the tiny basin is the real
   carrier or whether the envelope can substitute.
2. For record `114`, split the 119-token basin into tighter spatial/category
   sub-basins before patching. This tests whether a constructive sub-basin is
   hidden inside the destructive aggregate.
3. Keep record `50/9` as the contrast case: it is route-dominant by probability
   but target-destructive, useful for separating route strength from target
   alignment.

Manual visual review should inspect the three images/box sets before launching
that GPU wave.

## Verification

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_basin_specificity_review.py \
  scripts/analysis/run_autoregressive_duplication_phase4_basin_specificity_review.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_specificity_review.py

direct importlib execution of:
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_basin_specificity_review.py

python scripts/analysis/run_autoregressive_duplication_phase4_basin_specificity_review.py ...
```

The local `python -m pytest ...` wrapper still reports `Pytest: No tests
collected`, matching the known worktree behavior.
