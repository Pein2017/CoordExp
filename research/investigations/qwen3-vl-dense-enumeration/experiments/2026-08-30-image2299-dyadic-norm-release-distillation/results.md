---
title: Image2299 dyadic norm-release distillation results
type: investigation-result
role: research-result
unit_id: 2026-08-30-image2299-dyadic-norm-release-distillation
status: complete
evidence_status: fresh_cold_augmented_greedy_38_person_success
updated: 2026-08-30
---

# Image2299 dyadic norm-release distillation results

## Disposition

**Complete: fresh cold augmented-greedy success.** The constrained release
produced a clean strict owner gain on the frozen Image2299 route and passed
warm/cold exactness. This remains single-image, augmented-model,
ordinary-greedy evidence.

## Authoritative receipt

Receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-dyadic-norm-release-distillation/20260830T-image2299-dyadic-norm-release-distillation-v1/receipt.json`

Receipt SHA-256: `9811bdb632c5c564607537fb80892c38b66a584b31a09da97fa887b6fc90a5e6`.
Recorded status: `cold_greedy_38_person_success`.

## Decision-bearing outcome

- The minimum normalized norm was `1.0972734315870298`, below the cap
  `9/8 = 1.125`; all `108/108` constraints were feasible.
- The exact warm and cold route has SHA-256
  `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e` and
  length `370`.
- Both executions emit `38` persons and `3` ties, giving `41` strict owners.
  The exact gained-person set is `gt:2299:{0,1,14,18,20,32,34,35}`.
- The eight gained persons are added to the retained parent owner set; there
  is no owner exchange. Parser and global matcher are accepted, natural EOS is
  present, and every hard counter/debt field is zero.
- Warm and cold both pass all declared parity checks: route, selected IDs,
  ledger, basis, payload, protected hidden state, frozen surface, gate, and
  successor schema are all exact/true.

## Payload, controls, and resources

The release used 12 positive states, one solve, and a protected null of 664
states. The output contains nine FP64 residual rows of shape `9 x 2048`:

- payload SHA-256: `10455e0587bfd103bf3c539cdc655f21b1ee778a02ef27be5f80a6e9f8c34704`;
- residual-row SHA-256: `f85c0022306c426d5cc752d9ac383ea313c1af6fc737f42bee058b8891b1b1ec`;
- controlled route SHA-256: `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`.

The run used two model loads, one solve, and one warm candidate. Wall time was
`840.6706967055798 s`; peak CUDA reserved memory was `10,815,012,864` bytes
(warm) and `10,588,520,448` bytes (cold), within the declared 17,179,869,184
byte bound. Artifact bytes before the receipt were `1,082,056`.

## Claim boundary

This result supports only one Image2299 frozen-r32 augmented-model ordinary
greedy parameterization. It is not evidence for base-r32 greedy behavior,
transfer, general enumeration learning, or recovery of the eight ties.
