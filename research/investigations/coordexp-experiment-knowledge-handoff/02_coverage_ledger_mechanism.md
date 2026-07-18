---
type: investigation
title: Coverage Ledger Mechanism Capsule
description: Bounded historical evidence about history-conditioned coordinate and continuation signals without claiming a durable object ledger.
role: historical-evidence-capsule
authority: non_normative_research
status: historical-synthesis
domain: coverage-ledger-mechanistic-probing
updated: 2026-07-18
---

# Coverage Ledger Mechanism Capsule

## Question

Do coverage-ledger interventions expose a reusable coordinate/binding mechanism, or only selected hidden-state routes that can be read out differently at the model output?

## Source scope

The source family is `outputs/analysis/coverage_ledger_mechanistic_probing/` in the byte-preserving 2026-07-17 output union: 1,524 source paths, 1,499 unique contents. Evidence is mostly selected checkpoints, objects, slots, and intervention panels. The counts below are rows in named post-hoc aggregates, not population sample sizes.

## Verdict

- **COV-001 — source-history signal is reproducible within the bridge panel.** The merged source-history localization aggregate has 4 pairs, 3,024 bridge rows, and 0 errors. Removing the sorted-pureCE source history reduced final-logit contrast by 10.03 at layer 27 and 6.31 at layer 24 for one highlighted source; aggregate layer-27/layer-24 reductions were about 6.23/4.77. This supports a source-history contribution, not a unique coordinate-locality explanation.
- **COV-002 — hidden accessibility is not final-output causality.** The bridge explicitly compares `hidden_direct_lm_head`, `hidden_final_norm_lm_head`, and actual `final_logits`. A patch can produce a very large direct-hidden delta while the actual final-logit delta is small or zero. For example, selected self-attention/source-history rows show hidden direct deltas in the tens to roughly 114 while final deltas are around 0–4.625. Claim hidden accessibility/readout only unless the `final_logits` row moves.
- **COV-003 — route-state transfer is available but heterogeneous.** The route-state transfer summary contains 45 pairs, 1,440 rows, and 0 errors. The source-minus-target patch often moves the target toward an available coordinate bin, but sign, layer, slot, and alpha behavior vary; it is not a route-equivalence proof.
- **COV-004 — coordinate rowspace is accessible but not canonical.** The coordinate-rowspace aggregate has 3,024 rows, three 864-row panels, and three 144-row dose panels. It supports accessible, heterogeneous coordinate directions (for example, selected layer-27 effective deltas of +3.625 and +3.5), not a single universal coordinate subspace or route.
- **COV-005 — visual bridge and source-transplant effects remain selected-case evidence.** Visual-bridge mediation and source-transplant summaries can show recovery or rank movement on anchors, but extreme single cases are unstable and do not establish population mediation.

## Negative controls and overclaim guards

- Keep all three readouts separate: direct hidden LM-head, final-normal hidden LM-head, and actual final logits.
- Do not collapse `coord_parallel`, `coord_residual`, and `full` patches; their hidden norms and output deltas can disagree in sign and scale.
- Do not infer route equivalence from a successful source/target transplant. Transfer signs and alpha responses are heterogeneous.
- Treat selected visual anchors and post-hoc bridge reductions as mechanism-generating probes, not validation metrics.
- A zero-error artifact join proves execution and alignment for that panel; it does not prove generalization.

## Not claimed

This capsule does not claim that the coverage ledger is a universal causal mechanism, that coordinate locality is exact, that hidden-state readout equals generated behavior, or that any intervention improves population detection or termination.

## Limitations

Evidence is selected-panel, checkpoint-specific, and intervention-dependent. Many tables report teacher-forced or local readouts rather than free rollouts. Bridge rows preserve multiple axes, but the union does not turn them into an independent population estimate. Final-logit effects can be attenuated by later normalization, residual, or continuation paths.

## Continuation seeds

1. Re-run the 45-pair route-state panel with stratified slots and checkpoints, reporting all three readouts and free continuation.
2. Separate source-history removal from coordinate-direction replacement with matched controls and pre-registered final-logit endpoints.
3. Repeat visual-bridge/source-transplant probes on held-out anchors; require artifact validity and a non-selected denominator before mechanism promotion.

## Exact recovery snapshot handles

- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/9b41553678cf/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zzzzzbh_source_history_localization/merged_source_history_localization/rollout_route_state_final_norm_bridge_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/9b5582fe7c90/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zzzzzbb_slot_span_group_panel/route_state_transfer_core_merged/rollout_coord_route_state_transfer_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/1e53a88d4833/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zzzzzr_coord_rowspace_direction_patch/aggregate_seed_panel/phase_zzzzzr_seed_panel_aggregate_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/2264dc8a0993/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zr_val_failure_visual_bridge_mediation/selected_val_anchors_summary/visual_bridge_residual_mediation_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/cab50bfad66f/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zd_head_attention_transport/current_bbox_allheads_summary/head_attention_route_summary.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/9284033bcf78/outputs/analysis/coverage_ledger_mechanistic_probing/ledger_vs_nonledger_v1/phase_zzzzk_source_transplant/h17_same_image_person_ledger_prefix_all_directed/forced_span_source_transplant_summary.md`
