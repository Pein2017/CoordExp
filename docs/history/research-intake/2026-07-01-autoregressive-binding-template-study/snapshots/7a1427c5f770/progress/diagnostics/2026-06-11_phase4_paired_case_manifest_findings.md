# Phase 4 Paired Case Manifest Findings

## Scope

This note records the compact paired case manifest used to bridge the
constructive/destructive value-basin stratification to the next targeted causal
probe.

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Manifest outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/phase4_paired_case_manifest_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/phase4_paired_case_manifest_report.md
```

Inputs joined by stable row identity:

- `phase4_value_basin_stratification_layer17_head1_duplicate_top4_allshards/value_basin_stratification_rows.jsonl`
- `phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_rows.jsonl`
- `phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_rows.jsonl`

Join coverage:

- Manifest rows: `14`
- Constructive candidates: `8`
- Destructive candidates: `6`
- Missing Q/K evidence rows: `0`
- Missing route-content-patch evidence rows: `0`

## Candidate Set

The manifest keeps the highest-magnitude primary slice rows:

```text
checkpoint_label=none_latest_ckpt32
phase=post_y1/pre_x2
patch_component=duplicate_basin
```

Constructive core:

```text
record=33 image=2685 desc=wine glass rows=23,25,26,27,28,29,30,31
```

Destructive core:

```text
record=114 image=11699 rows=4 clock, 7 handbag
record=50 image=5193 rows=3 cell phone, 5 person, 9 person, 10 surfboard
```

## Route/Value Hints

The manifest attaches masked-to-control route/content-patch readouts for each
candidate and assigns a launch hint:

- `route_dominant`: `9`
- `value_dominant`: `3`
- `mixed_route_value`: `1`
- `no_recovery_signal`: `1`

All eight constructive wine-glass rows are `route_dominant`.

Representative constructive row:

```text
record=33 row=25 desc=wine glass
target_centered_delta=0.544964
source_tokens=1
route_prob_recovery=0.035992
value_prob_recovery=0.001902
total_prob_recovery=0.035560
hint=route_dominant
```

Representative destructive rows:

```text
record=114 row=4 desc=clock
target_centered_delta=-0.362870
source_tokens=119
route_prob_recovery=0.004517
value_prob_recovery=0.006319
total_prob_recovery=0.018298
hint=mixed_route_value

record=114 row=7 desc=handbag
target_centered_delta=-0.152514
source_tokens=119
route_prob_recovery=0.000704
value_prob_recovery=0.002531
total_prob_recovery=0.003057
hint=value_dominant

record=50 row=9 desc=person
target_centered_delta=-0.029430
source_tokens=16
route_prob_recovery=0.013189
value_prob_recovery=-0.002416
total_prob_recovery=0.008955
hint=route_dominant
```

## Interpretation

The paired manifest makes the next fork sharper:

1. Constructive examples are not just high positive coordinate-basin deltas.
   They are coherent route-dominant cases where the existing route-delta patch
   explains most of masked-to-control probability recovery.
2. Destructive examples are heterogeneous:
   - record `114` has a large `119`-token source basin and value/mixed hints,
     consistent with diffuse or mismatched basin content.
   - record `50` contains weaker destructive rows, including a route-dominant
     person row and one no-recovery row.

This means a single global "route vs value" answer would be too coarse. The
next causal probe should explicitly compare:

- constructive route-dominant wine-glass rows;
- destructive value/mixed large-basin rows from record `114`;
- destructive route-dominant row `50/9` as a contrast case.

## Next Probe

Recommended GPU probe:

```text
Phase 4 paired route/content swap
layer=17
head=1
checkpoint_label=none_latest_ckpt32
phase=post_y1/pre_x2
patch_component=duplicate_basin
candidate manifest:
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
```

Minimum causal contrasts:

1. Route-only swap: keep value content fixed, replace duplicate-basin route
   scores/mass from constructive vs destructive cases.
2. Value-only swap: keep route fixed, replace duplicate-basin value contribution
   from constructive vs destructive cases.
3. Joint route+value swap: verify whether composition is additive or nonlinear.

Primary readouts:

- target coordinate probability;
- radius-4 and radius-8 coordinate mass;
- target-centered coordinate-basin projection;
- top-1 coordinate distance;
- expected absolute coordinate error;
- whether the output moves toward the constructive row's target basin or merely
  increases generic coordinate-token confidence.

This is now a narrow enough GPU probe to run without exploratory sprawl.
