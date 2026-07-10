# Phase 4 Basin Candidate Three-Gate Taxonomy

Date: 2026-06-11

Scope: materialize a compact route/QK/value taxonomy for the promoted
basin-candidate panel. This is a CPU summary over the already completed
route/content, Q/K route-origin, and value-basin projection probes.

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/candidate_three_gate_taxonomy
```

Files:

```text
candidate_three_gate_taxonomy_rows.jsonl
candidate_three_gate_taxonomy_summary.json
candidate_three_gate_taxonomy_report.md
```

## Classes

| candidate | class | route p rec | qk attn | value target delta |
|---|---|---:|---:|---:|
| `record33_target25_non_target_desc_sub_envelope` | `mixed_or_weak` | 0.00022 | 0.00011 | -0.00003 |
| `record33_target25_original_duplicate_basin` | `aligned_route_and_value` | 0.03324 | 0.57199 | 0.53303 |
| `record33_target25_target_desc_sub_envelope` | `routed_positive_value_but_downstream_harm` | -0.00488 | 0.53955 | 0.50520 |
| `record50_target9_component_row_9_person` | `route_without_value` | 0.00002 | 0.86539 | -0.13835 |
| `record50_target9_original_duplicate_basin` | `route_with_negative_target_projection` | 0.00784 | 0.12291 | -0.02742 |
| `record114_target4_component_row_6_person` | `route_with_negative_target_projection` | 0.01380 | 0.93253 | -0.36625 |
| `record114_target4_original_duplicate_basin` | `route_with_negative_target_projection` | 0.02113 | 0.84757 | -0.35870 |
| `record114_target4_target_row_bbox` | `value_without_route` | 0.00225 | -0.01953 | 0.04432 |
| `record114_target7_component_row_7_handbag` | `value_without_route` | -0.03128 | -0.30703 | 0.21489 |
| `record114_target7_original_duplicate_basin` | `route_without_value` | 0.00267 | 0.45275 | -0.14487 |

Class counts:

```text
aligned_route_and_value = 1
mixed_or_weak = 1
route_with_negative_target_projection = 3
route_without_value = 2
routed_positive_value_but_downstream_harm = 1
value_without_route = 2
```

## Read

This taxonomy gives the next probe panel a clean shape:

- `aligned_route_and_value`: record 33 original duplicate basin.
- `route_without_value`: record 50 target-local person and record 114 original
  basin for handbag.
- `value_without_route`: record 114 target-local clock and target-local
  handbag.
- `route_with_negative_target_projection`: record 50 original basin and the two
  record 114 clock person basins.
- `routed_positive_value_but_downstream_harm`: record 33 target-desc envelope.

The most important implication is that Q/K, value projection, and downstream
route/content recovery can disagree. The next causal probe should therefore
not patch only one scalar. It should target the disagreement cases and compare
duplicate-basin versus near-ring/source-bucket competition.

## Verification

Artifact assertion:

```text
candidate_count = 10
required classes present:
  aligned_route_and_value
  route_without_value
  value_without_route
  route_with_negative_target_projection
```
