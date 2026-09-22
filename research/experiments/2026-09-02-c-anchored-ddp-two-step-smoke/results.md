---
title: C-Anchored Two-Rank Two-Step DDP Result
description: Canonical DDP and unmerged persistence pass, but the second update replaces a protected Source owner at the target insertion boundary.
type: experiment-result
status: completed
disposition: SCIENTIFIC_STOP_C_ANCHORED_DDP_TWO_STEP
date: 2026-09-02
---

# C-Anchored Two-Rank Two-Step DDP Result

## Decision

`SCIENTIFIC_STOP_C_ANCHORED_DDP_TWO_STEP`.

The canonical trainer, two-rank reduction, two finite updates, checkpoint
writer, and cold unmerged readback all pass. Step 1 also passes its behavioral
gate. Step 2 learns a reliable selected `umbrella`, but at the same
autoregressive boundary it replaces the protected `handbag`; protected
retention falls from `3 / 3` to `2 / 3`. The frozen stop rule therefore forbids
scaling this first-crossing objective to more images or GPUs.

## Executed contract

- start: the frozen C adapter, not the accepted one-update adapter;
- supervision: 11 annotated complete rows conditioned on their actual C
  prefixes; unmatched and unannotated rows remain unknown;
- StateBank: 24 records in two equal-credit 12-event windows;
- runtime: world size 2, effective batch size 12, six microsteps per rank and
  update, BF16;
- optimizer: language-DoRA-only AdamW, LR `2.5e-6` for both updates, no weight
  decay, clip 1;
- persistence: step-1 and step-2 unmerged adapters plus an identity-copy of the
  frozen step-2444 selected-token embedding delta; no merged checkpoint.

Both updates used global eligible denominator 12 and passed the all-rank
finite-gradient gate:

| Update | total loss | complete-row loss | site gate | grad norm |
|---:|---:|---:|---:|---:|
| 1 | 1.728020 | 1.727798 | 0.000222 | 4.416628 |
| 2 | 1.710933 | 1.710712 | 0.000221 | 4.447212 |

Each checkpoint contains exactly 588 DoRA tensors and 18,006,016 scalars.
Both embedding payloads are tensor-identical to the input payload. Fresh HF
processes report `merged_adapters=[]`, `requires_grad=false`, 588 saved and 588
materialized keys, zero runtime dtype casts, and exact saved-to-materialized
tensor equality.

## Cold natural result

IoU matching is category-consistent global one-to-one. IoU50 is
decision-owning; IoU60/80 are diagnostics.

| Read | IoU50 | IoU60 | IoU80 | predictions | selected uptake | protected | duplicates | invalid | natural EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 137 | 127 | 77 | 204 | 0 / 11 | 3 / 3 | 1 | 0 | 12 / 12 |
| step 1 | 138 | 128 | 78 | 207 | 1 / 11 | 3 / 3 | 1 | 0 | 12 / 12 |
| step 2 | 140 | 129 | 78 | 209 | 2 / 11 | 2 / 3 | 2 | 0 | 12 / 12 |

All three reads have one unchanged dropped prediction, no malformed row, no
cap, and the same six rows / 14 natural ordering violations. Ordering remains
a monitor, not an eligibility gate.

At IoU50:

- step 1 gains only `270570:coco_ann:1140374` (`book`) and loses none;
- step 2 gains that book, selected
  `359310:coco_ann:1426509` (`umbrella`), and two non-target cars on image
  398214;
- step 2 loses protected `359310:coco_ann:1172698` (`handbag`).

The density-conditioned annotation monitor keeps all 233 owners in the
denominator and all training weights unchanged. It marks only the visually
reviewed dense pairs `90862/cup`, `270570/book`,
`273317/{banana,broccoli,carrot}`, and `575627/cup`; sparse instances of those
families stay reliable. This yields 23 dense-category-uncertain owners. The
step-1 uptake is only in that uncertain stratum. The step-2 umbrella is outside
it, although visually small; the lost handbag is visually uncertain. Neither
observation relaxes the protected-owner stop.

## Mechanism finding

All 11 registered target sites are insertions before an existing C-natural
row; none is an end append. On image 359310 the frozen prefix ends after three
Source rows. C and step 1 next emit:

```text
handbag [151, 372, 186, 425]
```

Step 2 instead emits the supervised target:

```text
umbrella [149, 328, 206, 356]
```

and then returns to the remaining C suffix. The umbrella reaches IoU50 while
the handbag row disappears. Owner credit is still assigned by the global
matcher, but the raw transcript makes a same-prefix substitution—not a pure
matcher reassignment—the strongest explanation.

This exposes a structural limitation of isolated-row CE. If target token
`u` and incumbent token `a` first differ under the same prefix, the target CE
gradient directly increases the logit margin `z_u-z_a`; it cannot encode
“emit `u`, then recover `a`.” Adding both as immediate next-token labels only
makes them compete for probability mass. Preservation must instead move to a
different conditional boundary.

## Artifacts

- canonical analysis:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-ddp-two-step-smoke/analysis-v1.json`
  (`95b68d8c11a291491541e953dceb7fc157ae02d3f78d7dccdfe9d25c2076ced0`);
- completed run receipt:
  `train/runs/qwen3_vl_2b_c_anchored_annotated_complete_row_ddp2_ebs12_2step_lr2p5e6/run.json`
  (`57c7e4059b9154e859ecd64d6e8f5d6932cc988bcb6bb40c935792c85dac4e04`);
- frozen train config SHA-256:
  `be2f26c61fe15f69c07b6bf028be964dd3bd5d0ca13267edd999b0be19f69073`;
- step-1 adapter fingerprint:
  `c95feffad64b8efed2d99a0bab0133d2485bc6bf771738c07852647da7a2e483`;
- step-2 adapter fingerprint:
  `fea8c9af55a25657a0684a05ac1771f887ffd617e7fb24874c3c08bbd6bb85a3`;
- frozen embedding-delta fingerprint:
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`.

Two pre-update fail-closed attempts are retained separately: the current
worktree initially lacked the already-validated DoRA and special-token source
gate receipts. Both stopped at step 0 with zero consumed packs; neither is
scientific evidence.

## Bounded conclusion and next discriminator

The result proves canonical DDP mechanics and dose-dependent multi-image
uptake, not generalization. It falsifies scaling the present isolated
first-crossing row objective unchanged: a second update can exchange a missing
owner for an incumbent at their shared boundary.

The shortest successor is **tail-boundary annotated-row CE**: condition each
missing row on the complete C-natural transcript immediately before EOS, so it
competes directly with EOS rather than with an existing owner. This reuses the
same StateBank family, loss, trainer, DDP path, and unmerged DoRA persistence.
It remains a new experiment and is not part of this stopped unit.
