---
type: investigation
title: Prefix History And Route Capsule
description: Historical prefix-route evidence with an explicit supersession boundary for the invalid v7 prefix-position arm.
role: historical-evidence-capsule
authority: non_normative_research
status: historical-synthesis
domain: prefix-denoising-surgery-and-route-attribution
updated: 2026-07-18
---

# Prefix History And Route Capsule

## Question

Can prefix history, forced coordinate slots, or route-group interventions be treated as stable keys for later coordinate basins and termination behavior?

## Source scope

The source family `outputs/analysis/prefix_denoising_surgery_probing/` contains 626 paths and 547 unique contents. Evidence includes image-specific route reductions, same-prefix group interventions, visual-value reducers, donor transport, and termination-boundary reductions. Most results are selected, post-hoc, or case-level.

## Verdict

- **PFX-001 — route attribution is prefix- and model-dependent.** The v7 image-2157 knife x2 reduction reported random `all_prefix +0.3525`, `current_box_start +0.2399`, and `current_partial +0.2289`, while sorted gave `all_prefix +0.0312` and `current_box_start +0.0675`, and pure CE gave `all_prefix −0.1097` and `current_box_start −0.1403`. These incompatible signs falsify a universal box-start anchor.
- **PFX-002 — v7 is superseded by a prefix-position error.** The v7 route map ended at `<|box_start|>` rather than after forced x1/y1. The v9 prefix-fixed reduction is the usable comparison: random `prompt_non_image +0.1401`, `current_box_start −0.0104`, `current_forced_coords −0.0379`, `all_prefix +0.0235`; sorted `all_prefix −0.0143`, `current_box_start +0.0003`, `current_forced_coords +0.0438`. Any v7 causal wording must be marked superseded.
- **PFX-003 — same-prefix bridge movement is mostly near-target steering, not exact repair.** In image 19432 x2 GT>5 group intervention, random bridge rows are 21 with 0 flips, 18/21 within one coordinate bin, and 21/21 rank and margin improved; sorted has 2/21 flips and 19/21 within one; pure CE has 0/21 flips and 21/21 within one. Direct 7-row probes have no flips. This is strong local steering evidence with sparse exact correction.
- **PFX-004 — visual-value reduction is descriptive and weakly generalizing.** The reducer changes train 121→120 rows with 10 top-1 flips and 58 conflicts; validation 361→360 with 0 top-1 flips and 105 conflicts. It is explicitly post-hoc, not a learned or causal visual-value rule.
- **PFX-005 — termination signatures are real but not stop causality.** The termination reducer compresses 102 source rows to 7 signatures: 3 late-stop flips, 2 continue basins, and 2 knife-edge boundaries. Example final stop margins are +1.875 and 0.0, but this atlas does not show that the intervention caused stopping; the broader binding continuation tables often remain `max_steps`.

## Negative controls and overclaim guards

- Preserve the exact prefix tail, forced-slot state, model, image, object, and contrast coordinate. Prefix-fixed v9 must not be pooled with superseded v7.
- Separate direct from bridge interventions; bridge effects often move rank/margin without changing top-1.
- Treat “within one bin” as near-target steering, not exact coordinate repair or object-binding repair.
- Keep train/validation and selected-anchor denominators visible. Conflicts and zero flips are evidence against broad generalization.
- Termination margins and late-stop flips are local signatures, not a stopping policy.

## Not claimed

This capsule does not claim that prefix history is a universal basin key, that forced x1/y1 repairs later coordinates across models, that visual-value components generalize, or that late-stop margins establish termination causality.

## Limitations

Route reductions are case-level and often use 96-row region summaries or 7/21-row interventions. Some route rows omit the prefix tail; v7 had an actual prefix-position error. Post-hoc reducers reuse selected rows and do not independently sample a population.

## Continuation seeds

1. Re-run v9-style prefix-fixed attribution with stored prefix tails and matched random/sorted/pureCE controls.
2. Expand same-prefix bridge groups across images and checkpoints, reporting exact flips, distance-to-target, rank, margin, and continuation separately.
3. Test termination interventions with valid free rollouts and an explicit stop-change endpoint rather than margin-only signatures.

## Exact recovery snapshot handles

- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/99b4ccc3dc56/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v7_2157_knife_x2_after_x1y1_route_reduction.md` (superseded)
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/ec851fc88a68/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v9_2157_knife_x2_after_x1y1_prefixfixed_route_reduction.md` (prefix-fixed)
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/c1d32dc97c4a/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v6_sameprefix_19432_x2_gt5_group_reduction.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/e85506a6b3c2/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v1_train_418535_l16_h12h13_modes6_gpu0/fn_visual_value_component_basin_reduce.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/26333517369c/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v2_val_14439_l16_h8h12h13_modes6_gpu0/fn_visual_value_component_basin_reduce.md`
- `816fbb0a1:docs/history/output-markdown-union/2026-07-17/snapshots/66197e8eab33/outputs/analysis/prefix_denoising_surgery_probing/termination_boundary_reduce/v1_underexplored_termination_fn_box_end/termination_boundary_reduce.md`
