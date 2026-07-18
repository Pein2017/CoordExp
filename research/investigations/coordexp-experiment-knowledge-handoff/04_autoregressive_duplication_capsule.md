---
type: investigation
title: Autoregressive Duplication Mechanism Capsule
description: Selected historical causal probes separating attention routing, value contribution, and actual duplicate-output recovery.
role: historical-evidence-capsule
authority: non_normative_research
status: historical-synthesis
domain: autoregressive-duplication-mechanism
updated: 2026-07-18
---

# Autoregressive Duplication Mechanism Capsule

## Question

Do attention, Q/K score routing, value sources, or downstream residual sites provide a causal explanation for selected duplicate-coordinate basins?

## Source scope

The source family `outputs/analysis/autoregressive_duplication_mechanism/` contains 184 paths and 180 unique contents. The reports combine selected checkpoints, records, phases, heads, and post-hoc reductions. They are mechanism probes, not population estimates.

## Verdict

- **DUP-001 — a selected pre-onset routing signal is strong.** The phase-2 attention report has 17,584 source-attention rows, 1,496 zero-token rows skipped, and 240 head-contrast rows. In the highlighted `none_latest_ckpt32`, record 33, L17H1, `post_y1/pre_x2` case, duplicate density is 0.42748358; comparable no-align and aux cases are 0.39479090 and 0.35513485. The report calls these candidate routing heads, not causal heads.
- **DUP-002 — onset is phase- and checkpoint-dependent.** The pre-onset panel has 1,256 coordinate rows, 192 coordinate summaries, 384 attention summaries, and 192 masking summaries. Example alignment probability/rank: aligner `post_y1/pre_x2`, offset −1, 0.182718/rank 5.4; no-align same phase, offset −2, 0.061681/rank 17.2. These are selected-window precursors.
- **DUP-003 — Q/K route changes dominate a selected case, but output recovery is small.** In record 33 `post_y1/pre_x2`, Q/K rows show attention-mass delta +0.415218 and score delta +4.15748. Score-bias proxy recovery is only +0.010221 probability; direct key+value state patch +0.0100274; value-only −0.0000683. Downstream self-attention output recovery is +0.0127848 total or +0.0143284 route-only, while post-attention normalization is negative (−0.0053063 total).
- **DUP-004 — the projected duplicate direction is not a clean target basis.** In the selected projection cancellation case, duplicate-basin projected target fraction is 15.9077 but projected target cosine is only 0.0342665. Large projection magnitude therefore cannot be read as target alignment.
- **DUP-005 — broad masking does not establish a universal causal mask.** Phase-3 mean mask probability deltas are negative for all listed checkpoints: aligner −0.00062, aux −0.00142, no-align −0.00185, and none −0.00292. Rank changes are mixed. The visual-causal score is a prioritization heuristic, not an independent causal estimator.

## Negative controls and overclaim guards

- Attention contrast is density contrast, not raw mass, and the source report explicitly says attention contrast alone is not causal evidence.
- The visual-causal score is `(duplicate-minus-rest attention density) * max(-mask target-prob delta, 0)`; it identifies follow-up cases only.
- Keep route, key, value, downstream site, and final-logit effects separate. A route change can be real while value-only and final output recovery remain near zero.
- The aggregate rows are selected records and phases; do not pool them as independent examples.

## Not claimed

This capsule does not claim L17H1 is a universal duplication head, that visual masking is a general cure, that projected duplicate directions are target geometry, or that selected probability/rank recovery transfers to free decoding or population metrics.

## Limitations

Most probes are teacher-forced and selected. Head and phase labels are model- and checkpoint-specific. Downstream normalization and continuation can attenuate or reverse local effects. Masking changes the visual input and may alter more than one semantic source.

## Continuation seeds

1. Replicate candidate heads on a stratified held-out panel with pre-registered final-logit and free-rollout endpoints.
2. Pair Q/K, value, and downstream-site interventions with matched random and non-duplicate controls.
3. Report density, mass, final-logit, rank, and continuation outcomes in one row schema before any causal summary.

## Exact recovery snapshot handles

- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/f48da88cbd6d/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_attention_suspect_heads_report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/fc646a196b9b/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_onset_precursor_panel/phase1_onset_precursor_panel_report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/a7030fe602b7/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_qk_score_bias_synthesis_layer17_head1/targeted_qk_score_bias_synthesis_report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/932f66e19d15/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_state_patch_synthesis_layer17_head1/targeted_state_patch_synthesis_report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/94107d4c5555/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_downstream_site_synthesis_layer17_head1/targeted_downstream_site_synthesis_report.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/f7a9640fbaf2/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_projected_direction_cancellation_synthesis_layer17_head1/targeted_projected_direction_cancellation_synthesis_report.md`
