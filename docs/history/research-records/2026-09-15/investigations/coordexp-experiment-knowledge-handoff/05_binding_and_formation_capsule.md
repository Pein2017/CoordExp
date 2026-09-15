---
type: investigation
title: Binding And Formation Capsule
description: Selected historical evidence showing heterogeneous layer, site, onset, and geometry behavior during object-row formation.
role: historical-evidence-capsule
authority: non_normative_research
status: historical-synthesis
domain: autoregressive-binding-formation-and-continuation
updated: 2026-07-18
---

# Binding And Formation Capsule

## Question

When a staged-slot intervention changes coordinate rank or geometry, does it repair object binding and continuation, or merely steer a donor/origin basin?

## Source scope

The source family `outputs/analysis/autoregressive_binding_template_ablation/` contains 1,065 paths and 973 unique contents. The evidence spans formation ownership, staged-slot failure modes, coordinate-value geometry, and object-step continuation bridges. Reducer labels and rows are post-hoc summaries over selected train/validation panels.

## Verdict

- **BIND-001 — formation ownership is mixed, not a single repair path.** The component reducer has 340 rows (170 MLP and 170 self-attention across layers 17/18). Role counts are layer-18 donor snap 108, origin collapse 120, persistent worse/escape 56, rank-only target repair 24, and stable target geometry 32. Overall target-like rate is 0.1971 and target-geometry rate 0.10; worse-like rate is 0.3735.
- **BIND-002 — site/layer effects are heterogeneous.** In the component panel, L17 self-attention is donor-like on 0.7882 of rows, while L18 MLP is origin-like on 0.3765; stable-target-geometry rows are a small labeled subset, not a population rate. The transition summary is even more conditional: one clean simple-control panel has 605 rows, 57 receivers, and 121 sequences, with target-geometry rate 0 and donor/origin-like rates above 0.99.
- **BIND-003 — failure modes are dominated by boundary and onset conditions.** The staged failure reducer has 472 rows, split train 216/val 256. Failure loci: x1-onset anchor failure 204, premature boundary 132, tail no-rescue/delayed evidence 64, small-object weak evidence 60, and wrapper-mode router competition 12. Intrusive transitions are 245/472; strict clean local repairs are 61/472.
- **BIND-004 — geometry checks are valid but do not imply binding repair.** The staged coordinate-value geometry summary has 313 input/output rows and 0 invalid geometry rows. It labels 52 clean transitions and 261 intrusive transitions; intrusive transitions have rank improvement on 0.4559 of rows and slot intrusion rate 1.0. Geometry validity is a data-quality check, not a causal success metric.
- **BIND-005 — continuation repairs rarely alter stopping.** Object-step value-region decompositions contain many `max_steps` outcomes; core continuation tables report final stop changed 0 for the highlighted repairs/harmful patches. Rank or coordinate movement therefore does not establish termination or object-count repair.

## Negative controls and overclaim guards

- Preserve train/validation, regime, patch site, layer, baseline token mode, and failure locus. Aggregating them hides wrapper-mode competition and onset-specific failures.
- Distinguish coordinate rank improvement, target geometry, instance binding, and continuation/termination. They are separate endpoints.
- `invalid_geometry_row_count=0` means the reducer parsed its geometry rows; it does not validate the scientific interpretation.
- The simple-control transition panel is a negative control against treating donor/origin movement as target repair.

## Not claimed

This capsule does not claim that a donor state is a target state, that coordinate rank repair repairs object identity, that staged patches improve free decoding, or that any layer/site is a universal binding locus.

## Limitations

Formation labels are reducer-defined and often selected. Several panels are teacher-forced or local interventions. Continuation experiments include intervention-skipped and `max_steps` outcomes, so stop behavior is not a clean endpoint. The source family mixes model generations and research branches.

## Continuation seeds

1. Re-evaluate staged patches on held-out free rollouts with separate identity, geometry, count, and stop metrics.
2. Stratify by x1-onset, wrapper mode, object size, and termination tail before choosing a formation intervention.
3. Require a clean transition and unchanged control behavior before interpreting a rank gain as binding repair.

## Exact recovery snapshot handles

- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/b6382ca47e50/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_reducer/v1_component_layers17_18_panel_merged/formation_ownership_component_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/98a3a1dc4ddc/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v1_v24_clean_prex1_compatible_layers17_21/formation_ownership_transition_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/8617c6b42cd6/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_failure_mode_reducer/v2_v30_layers17_18_selfattn_mlp_failure_modes/staged_slot_failure_mode_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/03afdd68f5b4/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v2_v5_v18_v7_prevslot_geometry/staged_slot_coord_value_geometry_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/00b2cfbea6cd/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation_bridge_summary/v1_harmful_repair_fullvec_dirproj_regions3_scales012_steps6/bridge_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/a22fedbca486/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_decomposition_summary/v2_span_decomp_and_independent_dirproj/decomposition_summary.md`
