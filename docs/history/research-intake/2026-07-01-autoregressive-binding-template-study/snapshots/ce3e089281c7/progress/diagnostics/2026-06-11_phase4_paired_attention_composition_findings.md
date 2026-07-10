# Phase 4 Paired Attention Composition Findings

Date: 2026-06-11

Scope: post-hoc origin analysis for the constructive/destructive paired case
manifest. This slice does not run a new model forward. It joins the existing
14 paired rows to broad attention-category evidence, duplicate-basin Q/K
route-origin rows, value-basin stratification rows, and route/content patch
rows.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_paired_attention_composition.py
scripts/analysis/run_autoregressive_duplication_phase4_paired_attention_composition.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_paired_attention_composition.py
```

## Artifacts

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_attention_composition_layer17_head1_constructive_destructive
```

Files:

```text
paired_attention_composition_rows.jsonl
phase4_paired_attention_composition_summary.json
phase4_paired_attention_composition_report.md
```

Inputs:

```text
phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
phase4_attention_source_category_layer17_head1_control_vs_mask_top4_allshards_shard-*/attention_source_rows.jsonl
phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_rows.jsonl
phase4_value_basin_stratification_layer17_head1_duplicate_top4_allshards/value_basin_stratification_rows.jsonl
phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_rows.jsonl
```

Join coverage:

```text
row_count = 14
missing_attention_category_evidence_count = 0
missing_qk_evidence_count = 0
missing_value_basin_evidence_count = 0
missing_route_content_evidence_count = 0
```

## Role Summary

| role | n | target-centered delta | source tokens | qk attention delta | qk score mean delta | qk key-side mean | qk query-side mean | route prob | value prob | route r4 | value r4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constructive core | 8 | 0.588599 | 1.000000 | 0.632790 | 4.706530 | 6.542107 | -1.835577 | 0.027665 | 0.000763 | 0.199002 | 0.008015 |
| destructive core | 6 | -0.096862 | 50.333333 | 0.260780 | 1.351156 | 0.951147 | 0.400009 | 0.003059 | 0.000887 | 0.022954 | 0.004795 |

## Broad Attention Categories

Broad visual attention is not the discriminative variable by itself:

| role | visual control mass | visual masked mass | visual control-minus-masked |
|---|---:|---:|---:|
| constructive core | 0.956673 | 0.961370 | -0.004697 |
| destructive core | 0.993632 | 0.928560 | 0.065072 |

The destructive rows can have high total visual mass and positive visual-mass
delta while still producing negative value-basin target-centered deltas. That
rules out a shallow "more visual attention fixes the row" explanation.

The sharper split is at the duplicate-basin composition level:

- Constructive rows use a one-token duplicate basin, dense same-description
  neighborhood, and strong key-side route score differences.
- Destructive rows use diffuse or mismatched basins. Record `114` has a
  119-token duplicate basin and large attention mass, but the value-basin delta
  is destructive. Record `50` has weaker 16-token basins and heterogeneous
  hints.

## Candidate-Level Read

Constructive wine-glass rows are uniform:

```text
record=33 rows=23,25,26,27,28,29,30,31
source_tokens=1
qk_attention_delta=0.378357..0.839233
qk_score_mean_delta=3.71056..6.11532
route_prob_recovery=0.02178..0.03599
value_prob_recovery=-0.00172..0.00190
```

Destructive rows are heterogeneous:

```text
record=114 row=4 clock source_tokens=119 qk_attention_delta=0.887152 route_prob=0.004517 value_prob=0.006319
record=114 row=7 handbag source_tokens=119 qk_attention_delta=0.432721 route_prob=0.000704 value_prob=0.002531
record=50 row=9 person source_tokens=16 qk_attention_delta=0.132070 route_prob=0.013189 value_prob=-0.002416
record=50 row=3 cell phone source_tokens=16 qk_attention_delta=0.000004 route_prob=0.000000 value_prob=0.000000
```

## Interpretation

The origin variable is not simply route strength or visual attention mass. The
constructive rows look like a coherent, specific basin-selection event:

```text
small duplicate basin
  -> strong key-side Q/K score advantage
  -> route-dominant attention-output repair
  -> positive coordinate-basin target-centered value delta
```

The destructive rows look like failed specificity:

```text
diffuse or mismatched duplicate basin
  -> attention can still be large
  -> score/value effects are mixed
  -> coordinate-basin target-centered delta is negative or near-flat
```

This supports the current "specific basin vector" picture from the cross-row
swap probe. Constructive route vectors are not generic repairs because they
encode a target-specific basin. When the selected source basin is diffuse or
does not align with the current coordinate slot, the same layer/head family can
increase visual mass without producing the right coordinate attractor.

## Next Decision

The next deep path should inspect basin specificity more directly, not another
global attention-mass intervention:

1. For record `33`, perturb or compare the one-token basin against the
   surrounding same-description envelope to see why the one-token route is so
   clean.
2. For record `114`, split the 119-token basin into tighter spatial/category
   sub-basins and ask whether one sub-basin is constructive while the aggregate
   basin is destructive.
3. Treat record `50/9` as the contrast case: route-dominant but destructive,
   useful for testing whether route strength can be target-misaligned.

Manual review should inspect the associated images/boxes before any expensive
new causal run, because the likely mechanism now depends on whether the source
basin is visually/semantically coherent.

## Verification

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_paired_attention_composition.py \
  scripts/analysis/run_autoregressive_duplication_phase4_paired_attention_composition.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_paired_attention_composition.py

direct importlib execution of:
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_paired_attention_composition.py

python scripts/analysis/run_autoregressive_duplication_phase4_paired_attention_composition.py ...
```

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_paired_attention_composition.py`
still reports the local `Pytest: No tests collected` wrapper issue. The direct
importlib test harness is the meaningful deterministic unit check for this
slice.
