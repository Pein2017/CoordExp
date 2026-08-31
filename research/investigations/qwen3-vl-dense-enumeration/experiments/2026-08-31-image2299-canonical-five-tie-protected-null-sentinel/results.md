---
unit_id: 2026-08-31-image2299-canonical-five-tie-protected-null-sentinel
status: complete
evidence_status: fresh_cold_augmented_greedy_46_owner_success
receipt_sha256: e475bd7e9a3f21f5a8f83c77fd0fd2fd0161b94a2e7da92546eb109cd44e06a7
---

# Results

## Decision

The mandatory canonical five-tie sentinel passed. One composed output residual
made the canonical suffix ordinarily greedy under the augmented model, so the
separate 120-order screen is not required by this sentinel's stop rule.

## Receipt and identities

- Receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-five-tie-protected-null-sentinel/20260831T-image2299-canonical-five-tie-protected-null-sentinel-v1/receipt.json`.
- Parent route: 370 tokens, SHA-256 `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`; parent payload SHA-256 `10455e0587bfd103bf3c539cdc655f21b1ee778a02ef27be5f80a6e9f8c34704`.
- Candidate/canonical route: 415 tokens, SHA-256 `9491c9c4027cdf050caacc0d41d261418bbdc21d3676227f53fb7c4ccc0c4572`.
- Canonical order: `gt:2299:45`, `gt:2299:11`, `gt:2299:9`, `gt:2299:8`, `gt:2299:43`.
- Composed payload SHA-256: `64f55e3d69f39dcb93f3b4b7a58ae090061640427e4ca3aa713cd27602128f98`; one wrapper, 28 x 2048 FP64 rows. Parent, child, and composed identities are separately bound.

## Solve and protected null

The target-only solve was feasible: 21 positive states, projected rank 21,
21 child rows, 441 constraints/variables, and minimum normalized child norm
`0.8405848820284767` (below cap `9/8`; lower bound
`0.8405848820283546`). The protected span contains 763 states; maximum
protected correction is `5.6830651296024826e-12` against tolerance `1e-10`.
Full-vocabulary recheck passed. The solve count was one.

## Warm and fresh-cold ordinary greedy gates

Both gates passed with owner-equivalent exact route and ledger parity:

- 38 persons, 8 ties, 46 unique strict predictions/owners.
- Zero hard counters and zero debt; natural row-aligned EOS.
- Gained ties: `gt:2299:8`, `gt:2299:9`, `gt:2299:11`, `gt:2299:43`, `gt:2299:45`.
- Warm/cold identities, protected states, frozen surfaces, selected rows, and
  route/ledger were exact; `warm_cold` all passed.
- One GPU, world size 1, two model loads, one warm candidate, one solve;
  wall time 544.204732 s and peak reserved CUDA memory 11,146,362,880 bytes,
  within all declared bounds.

## Claim boundary

This is one Image2299 composed-augmented-model ordinary-greedy 46-owner result
only. It is not base-model, transfer, or general enumeration evidence.
