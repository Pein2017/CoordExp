# Coordinate-input continuity results

**Status:** candidate; CPU reduction complete, parent acceptance pending.  
**Unit:** 2026-09-19-coordinate-input-continuity  
**Question:** within the frozen native source histories, does changing the latest feasible coordinate input by one vocabulary bin leave a measurable effect at the immediate target and at a fixed later target?

## Estimand and denominator

The paired contrast is each saved delta-1 or delta+1 capture minus its same-state native capture. The input edit is the latest feasible coordinate selected by the frozen plan. Both score sites are target-only: the first row_delay=0 target and the first positive-delay target. The batch, companions, media, positions, source order, and intervening native suffix remain fixed.

The denominator is 16 states and 32 feasible +/-1 contrasts, yielding 64 site observations: 8 failure and 8 proxy states, split evenly between tied and untied models. Every selected state and variant is role y2; this is the observed scope of the latest-coordinate rule. It does not support claims about x1, y1, or x2.

The CPU reducer is at probes/training_set_completion/coordinate_continuity/reduce.py. It loads only the saved tensors and reconstructs input deltas, layer changes, centered coordinate logits, ranks, winner changes, probability distributions, norm/head changes, and the verification ledger.

## Runtime and integrity

The candidate producer completed 96 planned captures and preserved one failed parser attempt, for 97 observed model forwards across 9 jobs. Total elapsed job time was 286.834 seconds and allocated serial GPU time was 0.0796762 GPU-hours. The saved candidate captures occupy 188175792 bytes.

Source replay parity passed 32/32 checks, with maximum raw top-two absolute difference 4.00543213e-05; winners and token identities matched. Hook/no-hook parity passed 48/48 checks with maximum logits and coordinate-logit differences 0. Input mutation checks passed 32/32. Nine missing image_plan records were repaired only through the accepted natural-producer planning seam; all nine replayed input identities matched the saved source receipt. The original holds remain under the runtime archive. The untied scaleout receipt has a documented reconstruction because its original receipt bytes are unavailable; this is an archival limitation, not a relaunch or scientific replacement.

The saved source and hook gates therefore support a candidate scientific readout. The reducer itself made zero model calls.

## Paired input and readout changes

Values are medians within each group and site. centered Delta-infinity / native margin compares the largest centered change over the native full-vocabulary top-two margin. pair Delta is the change in the native-coordinate versus replacement-coordinate logit gap. coord L1 is the L1 change in the softmax distribution over all 1,000 coordinate logits. rank is the native winner's rank under the variant; replacement rank is the replaced coordinate's rank under the variant. Winner flips are full-vocabulary flips, with the coordinate-winner flip count equal in this cohort.

| stratum / model | site | n | input Delta-h L2 | centered Delta-infinity | centered Delta-infinity / native margin | pair Delta | coord L1 | native rank | replacement rank | winner flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| failure / tied | immediate | 8 | 0.2201 | 0.01937 | 0.8376 | -0.000254 | 0.00565 | 1 | 395.5 | 0/8 |
| failure / tied | later | 8 | 0.2201 | 0.01251 | 0.3327 | -5.91e-05 | 0.00312 | 1 | 413 | 0/8 |
| failure / untied | immediate | 8 | 0.2405 | 0.04654 | 0.6676 | 6.58e-05 | 0.00882 | 1 | 624 | 1/8 |
| failure / untied | later | 8 | 0.2405 | 0.03195 | 0.825 | -8.3e-05 | 0.0109 | 1 | 616.5 | 0/8 |
| proxy / tied | immediate | 8 | 0.2209 | 0.01246 | 0.1309 | -0.000259 | 0.00304 | 1 | 304.5 | 0/8 |
| proxy / tied | later | 8 | 0.2209 | 0.02244 | 0.4463 | 5.44e-05 | 0.00431 | 1 | 304.5 | 0/8 |
| proxy / untied | immediate | 8 | 0.2385 | 0.026 | 0.1206 | -0.00143 | 0.00609 | 1 | 272 | 0/8 |
| proxy / untied | later | 8 | 0.2385 | 0.01107 | 0.0292 | -8.11e-05 | 0.00207 | 1 | 273 | 0/8 |

The target embedding perturbation is substantial and model-dependent: median input Delta-h L2 is 0.220 for failure/tied, 0.240 for failure/untied, 0.221 for proxy/tied, and 0.238 for proxy/untied. The untied input rows therefore show a larger input perturbation than tied rows in these selected states. The readout response is not a simple monotone failure/proxy effect: failure rows are larger at both sites for untied, while proxy/tied has the larger later-site margin-normalized median (0.446 versus 0.333 for failure/tied).

The immediate site had one full and coordinate winner flip, in d-07-failure-untied-train-269858-failure, delta=+1. Its native winner was token 152020 and the variant winner was 152018; the native margin fell from 0.01358 to 0.00398, and the replacement coordinate remained rank 724 under the variant. The other 31 immediate sites and all 32 later sites retained the native full winner. The median native-winner rank was 1 in every group/site; the immediate minimum was 2 only for the flipped contrast. Thus hidden-state changes can be large relative to the native margin without usually changing greedy choice in this conditional cohort.

EOS probability changes were numerically negligible: the maximum absolute change was 4.17e-11. Coordinate-family probability change maxima were 8.8e-05 immediate and 1.95e-05 later; their pooled medians were about 3e-8. The main observed movement is within the coordinate logit distribution: the pooled coordinate-distribution L1 median is 0.00530 immediate and 0.00351 later.

## Per-layer residual path

The hook captures clone each layer input and output. The residual value below is the L2 difference of (layer_output - layer_input) between variant and native; it is a recorded residual-branch change, not an attribution to attention or MLP. Values are medians across the group.

| stratum / model | site | residual Delta L2 checkpoints | final norm/head Delta L2 |
|---|---:|---|---:|
| failure / tied | immediate | L0=0.0461 L10=0.0808 L15=0.211 L20=0.886 L24=2.83 L27=5.41 | norm=8.54; head=0.203 |
| failure / tied | later | L0=0.0083 L10=0.0325 L15=0.12 L20=0.546 L24=1.59 L27=5.58 | norm=6.86; head=0.126 |
| failure / untied | immediate | L0=0.0679 L10=0.127 L15=0.39 L20=1.36 L24=4.99 L27=11.9 | norm=16.1; head=0.376 |
| failure / untied | later | L0=0.017 L10=0.0388 L15=0.213 L20=0.802 L24=2.99 L27=8.91 | norm=11.1; head=0.267 |
| proxy / tied | immediate | L0=0.0271 L10=0.0645 L15=0.172 L20=0.454 L24=1.26 L27=4.25 | norm=5.9; head=0.113 |
| proxy / tied | later | L0=0.00801 L10=0.0391 L15=0.115 L20=0.77 L24=2.73 L27=7.67 | norm=9.35; head=0.2 |
| proxy / untied | immediate | L0=0.0354 L10=0.106 L15=0.296 L20=1.17 L24=3.91 L27=9.19 | norm=10.7; head=0.301 |
| proxy / untied | later | L0=0.0108 L10=0.0402 L15=0.163 L20=0.521 L24=1.44 L27=4.33 | norm=6.66; head=0.145 |

Across all 32 contrasts, the median residual-branch change grows from 0.035 at layer 0 to 8.00 at layer 27 at the immediate site, and from 0.0100 to 5.93 at the later site. The corresponding median block-output change grows from 0.035 to 10.74 and from 0.0100 to 8.11. Untied failure states have the largest late residual changes (layer-27 medians 11.92 immediate and 8.91 later); proxy/tied is smaller immediately (4.25) but larger than failure/tied later (7.67 versus 5.58). This is evidence of progressive state separation under the fixed native suffix, not identification of a particular causal module.

## Source-panel strata

The three source panels contribute 4, 10, and 18 contrasts. This table keeps source provenance visible rather than treating the 32 contrasts as exchangeable.

| source panel | site | n | input Delta-h L2 | centered Delta-infinity / native margin | coord L1 | winner flips |
|---|---:|---:|---:|---:|---:|---:|
| 2026-09-18-numerical-recurrence-feedback | immediate | 4 | 0.2385 | 0.1126 | 0.017 | 0/4 |
| 2026-09-18-numerical-recurrence-feedback | later | 4 | 0.2385 | 0.08171 | 0.00712 | 0/4 |
| 2026-09-18-untied-highconfidence18-natural | immediate | 10 | 0.2353 | 0.1258 | 0.00466 | 0/10 |
| 2026-09-18-untied-highconfidence18-natural | later | 10 | 0.2353 | 0.02315 | 0.00151 | 0/10 |
| 2026-09-19-recurrence-distribution-census | immediate | 18 | 0.2249 | 0.6287 | 0.00579 | 1/18 |
| 2026-09-19-recurrence-distribution-census | later | 18 | 0.2249 | 0.5047 | 0.0088 | 0/18 |

The recurrence-distribution census supplies 18/32 contrasts and has margin-normalized medians 0.629 immediate and 0.505 later. The high-confidence natural panel supplies 10/32 with 0.126 and 0.023. The numerical-recurrence panel supplies 4/32 with 0.113 and 0.082. These are source-panel strata, not a claim that source provenance causes the response.

## CPU row-geometry provenance

The actual initializer is natural_adjacent with 1,000 coordinate bins, 8 frequencies, scale 0.02, and seed 0. The initializer reconstructs exactly from the recorded digit mean and projected features (maximum absolute reconstruction error 0). Its highest-band adjacent phase is 0.8050527721 radians (46.126 degrees); periodic features coincide at bins 0 and 999 up to floating-point rounding while the linear feature separates the endpoints.

The trained-row audit is separate from that initializer fact. Mean adjacent input-row distance is 0.2282 for tied and 0.2471 for untied; untied output rows have mean adjacent distance 0.2269, and untied input/output row distance has mean 0.1804. A same-anchor input witness at anchor 0 has d4 versus d499 distances 0.5940 versus 0.5885 for tied and 0.6216 versus 0.6081 for untied, and d4 exceeds d999 (0.4311 tied; 0.4469 untied). These describe non-monotone geometry and do not establish a training cause, physical identity, or greedy instability mechanism.

## Scope, confounds, and disposition

- The panel was frozen as 8 failures and 8 proxies. It is a conditional mechanism probe, not a natural-history prevalence estimate.
- All selected states are y2 because the latest-coordinate rule was the only role rule used. No other role was added to fill coverage.
- The strict geometry/order validity flags were recorded, not used to exclude states. In this selected denominator, all 16 source states and all 48 variants were valid and ordered, and no edit changed either flag. The result therefore does not measure validity-changing transitions.
- The five plan exclusions were boundaries without both required target sites, not an invalid-geometry filter.
- Companions were kept for exact batch semantics and are not independent outcome observations.
- A one-bin edit changes the model context. Layer and logit differences are paired observations; they do not prove a causal layer, training-origin, annotation, or physical-object identity explanation.
- Runtime closure is candidate-complete and the CPU reducer is candidate-complete. Parent integration and scientific acceptance remain pending; no terminal or package event is written here.

## Reproduction

    python3 probes/training_set_completion/coordinate_continuity/reduce.py \
      --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-input-continuity

The command loads runtime/**/capture.pt and the frozen plan/panel, performs no model calls, and writes reduction.json, verification.json, and result.json.
