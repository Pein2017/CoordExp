---
title: PVCI Held-out Representation Results
description: Executed v2 held-out representation-gate result for the matched A/B/C proposal screen.
type: idea
role: research-result
authority: non_normative_research
status: accepted-for-follow-up
unit_id: 2026-07-11-pvci-causal-proposal-bridge
updated: 2026-07-12
---

# Held-out representation results — 2026-07-12

## Scope

This note records the completed **teacher-forced representation** screen only.
It is not a rollout, detection, recall, duplication, STOP, or causal-bridge
result. The corresponding unit remains `promotion_status: not_promoted` for
causal behavior.

## Executed artifacts

Matched 512-step runs:

- A: `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_A-pvci512-A-20260711T194750Z`
- B: `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_B-pvci512-B-20260711T220028Z`
- C: `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_C-pvci512-C-20260712T002318Z`

In-sample v2 gate root:
`/data/CoordExp/outputs/probes/coordexp_swift/pvci_in_sample_learnability_ABC_512_v2_20260712`

Held-out v2 gate root:
`/data/CoordExp/outputs/probes/coordexp_swift/pvci_heldout_representation_gate_ABC_512_v2_20260712`

The held-out decision artifact is `representation_gate_decision.json`, SHA256
`94e3fee01514eaedb55bed44cd9f62f18296d102e5684dfdc048d5e3caa010ec`.
The primary probe receipt is `representation_probe.receipt.json`, SHA256
`b9d82bcdff6a97703eb3650611a5a7b39d3d9d40531d9821a821060a0f980529`.
The in-sample v2-compatible output decision retained the exact prior decision
SHA `f98bf3805c077e1dc1d700483037e4b089fb1cf2f1b40e5db3275d3ab5e1b956`.

The held-out cohort contains 320 images, 906 paired views, and 3,639 proposal
rows. Its source/view/cache identity is bound by the preflight artifacts cited
in the unit. Both B and C passed all 29 applicable held-out checks. The frozen
candidate priority `(C,B)` selected C; this is a deterministic selection rule,
not evidence of C-over-B superiority.

## Gate result

The held-out representation decision is `promote`, and the independent
promotion audit is `GO` for the following narrow scope:

> The trained proposal representation carries target-specific visual
> information beyond the declared position, prefix-depth, image-mean, and
> content-to-cell-shuffle controls on the frozen held-out cohort.

Representative C-versus-A paired image-bootstrap effects (10,000 resamples,
seed `20260711`) include:

| Check | Effect | 95% CI |
|---|---:|---:|
| Area-adjusted target mass | +0.2183976 | [0.1975562, 0.2393482] |
| Hit@4 | +0.3514312 | [0.3169497, 0.3857557] |
| Specificity margin | +0.1666023 | [0.1415834, 0.1941060] |
| Image-content shuffle failure/drop | +0.2236660 | [0.2019732, 0.2461192] |
| Same-class specificity beyond chance | +0.1027011 | [0.0796942, 0.1271625] |
| Crowded specificity beyond chance | +0.1013350 | [0.0861810, 0.1175964] |
| Combined noncanonical area gain | +0.1759615 | [0.1528052, 0.1992161] |
| Combined noncanonical Hit@4 gain | +0.3430034 | [0.2849829, 0.3993174] |
| Top16Lift | +0.59916 | lower bound 0.58306 |

All 29 applicable checks passed for both B and C. The cohort-wide all-token
saturation check was zero. These values support the representation gate only;
they do not establish that proposal feedback changes generated rows.

## Tie-aware support strata

Support strata use deterministic value-based nearest-rank assignment. Observed
cutpoints were `[972, 972, 1014]`. The duplicate cutpoint means nominal q2 is
unattainable without splitting equal support values; q1, q3, and q4 are
attainable and were required. The gate records attainable/unattainable strata
explicitly. Treating q2 as missing evidence would have been a stratification
contract error, not a model failure.

For candidate C, the frozen native support counts are q1=`2029`, q3=`1415`,
q4=`195`; q2 is not an eligible stratum. The q4 count is small (19 images),
so no independent per-quartile efficacy claim is made.

## Audit caveats

- The resolved configuration hash is retained in provenance but was not
  independently replayed during the promotion audit.
- The primary receipt SHA is a receipt binding and does not constitute a full
  independent replay of every producer tensor.
- No independent full A/B/C replay was performed.
- C was selected by frozen priority, while B is near-tied in several checks;
  the result must not be reported as C superiority.
- This is one matched seed and a teacher-forced representation panel. It says
  nothing yet about own-prefix enumeration, commit/coverage, STOP, COCO
  unmatched objects, or detection metrics.

## Authorized next step

Proceed to the natural own-prefix causal/safety panel on the same 320-image
cohort, with C-on/C-off and the declared semantic controls under RP1.10 primary
and RP1.00 secondary decoding. Keep the representation promotion and causal
behavior decisions separate; a causal claim requires its own paired rollout
gate and safety audit.
