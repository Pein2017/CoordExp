# Phase 4 Value Basin Constructive/Destructive Stratification

## Scope

This note records an offline stratification of the layer-17 head-1
duplicate-basin value-basin projection rows.

The goal is to separate three cases that were blended in the first projection
summary:

- `flat_or_zero`: no usable duplicate-basin value delta.
- `constructive`: active value delta with positive target-centered
  coordinate-basin projection.
- `destructive`: active value delta with negative target-centered
  coordinate-basin projection.

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Input rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_projection_layer17_head1_duplicate_top4_allshards_rows.jsonl
```

Output artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_stratification_layer17_head1_duplicate_top4_allshards/value_basin_stratification_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_stratification_layer17_head1_duplicate_top4_allshards/phase4_value_basin_stratification_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_basin_stratification_layer17_head1_duplicate_top4_allshards/phase4_value_basin_stratification_report.md
```

The join used all eight token-window shards, all eight selected-window shards,
and `phase2_region_rows.jsonl`. It found `0` missing prediction cases.

## Global Counts

Across all value-basin projection rows:

- Total rows: `1256`
- `constructive`: `459`
- `destructive`: `273`
- `flat_or_zero`: `524`

This confirms the value-basin carrier is mixed. It is not enough to say the
duplicate-basin value channel "has" a coordinate signal; the useful mechanism is
conditional on which basin was selected and how coherent that basin is.

## Primary Slice

Primary diagnostic slice:

```text
checkpoint_label=none_latest_ckpt32
phase=post_y1/pre_x2
patch_component=duplicate_basin
```

Counts:

- Total rows: `62`
- `constructive`: `20`
- `destructive`: `6`
- `flat_or_zero`: `36`

Constructive mean feature profile:

- Target-centered coordinate-basin delta: `0.256328`
- Source component token count: `22.45`
- Window same-description row count: `5.15`
- Duplicate-like neighbor count: `3.3`
- Same-description duplicate-like neighbor count: `2.9`
- Onset bbox IoU: `0.298068`
- Onset bbox center distance: `157.353`
- Row offset from onset: `1.4`

Destructive mean feature profile:

- Target-centered coordinate-basin delta: `-0.096862`
- Source component token count: `50.33`
- Window same-description row count: `2.5`
- Duplicate-like neighbor count: `1.67`
- Same-description duplicate-like neighbor count: `0.67`
- Onset bbox IoU: `0.257212`
- Onset bbox center distance: `215.426`
- Row offset from onset: `1.0`

## Representative Rows

The strongest constructive rows are all from record `33`, image `2685`, the
wine-glass burst. They have a one-token source component and a dense coherent
same-description neighborhood:

```text
record=33 image=2685 desc=wine glass rows=23,25,26,27,28,29,30,31
source_tokens=1
same_desc_rows=8
duplicate_neighbors=4..7
target_centered_delta=0.35..0.78
```

The strongest destructive rows include record `114` and record `50`. They have
larger source components, weaker or mismatched same-description neighborhoods,
and larger distance from the onset box:

```text
record=114 image=11699 row=4 desc=clock source_tokens=119 target_centered_delta=-0.362870
record=114 image=11699 row=7 desc=handbag source_tokens=119 target_centered_delta=-0.152514
record=50 image=5193 row=5 desc=person source_tokens=16 target_centered_delta=-0.036301
record=50 image=5193 row=9 desc=person source_tokens=16 target_centered_delta=-0.0294295
```

## Interpretation

The primary `post_y1/pre_x2` coordinate-basin repair signal appears strongest
when the duplicate-basin source is small and semantically/geometrically coherent
around the local burst. Destructive rows look less like "the same circuit points
the wrong way" and more like "the route selected a diffuse or mismatched basin,
so the value content no longer aligns with the coordinate slot's target basin."

This narrows the next mechanism question:

```text
Is constructive vs destructive behavior decided mostly by:
1. key-side route selection into the right visual basin,
2. value-side content within the selected basin,
3. the coordinate-slot residual state before layer-17 head-1,
4. or an interaction between route/content and the coordinate-slot basin?
```

## Next Probe

Recommended next deterministic-to-causal sequence:

1. Build a compact paired case set from the primary slice:
   - constructive: record `33`, rows `23,25,26,27,28,29,30,31`
   - destructive: record `114`, rows `4,7`; record `50`, rows `5,9`
2. For those rows, compare source-basin route/content decomposition with the
   existing Q/K and value-source artifacts.
3. If the split remains clean, run a targeted causal probe that swaps either:
   - route scores while keeping value content fixed, or
   - value content while keeping route scores fixed,
   between constructive and destructive rows.

This is the highest-leverage follow-up because it attacks the origin of the
coordinate-basin signal instead of only confirming that the value channel can
carry it.
