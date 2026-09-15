---
title: Image2299 canonical-G QP norm-release sentinel results
type: investigation-result
role: research-result
authority: non_normative_research
unit_id: 2026-08-31-image2299-canonical-g-qp-norm-release-sentinel
status: complete
evidence_status: fresh_cold_exact_canonical_g_41_owner_success
updated: 2026-08-31
---

# Image2299 canonical-G QP norm-release sentinel results

## Disposition

**Success.** Releasing only the admission cap from `1.125` to `28.0` allowed
the already frozen G+QP v5 minimum-norm candidate
`27.98494582667317` to enter decoding. It required no new solve and no
first-divergence iteration.

## Decision evidence

| Witness | Route | Ledger | Gate |
|---|---|---|---|
| ordinary-greedy process | exact canonical G, 370 tokens, SHA `e238e67122aa46b54cf9290d11093b07a7490746d6730d6f843f8d4ee61d677d` | SHA `d0e5b1a07b97a0b5b987ca1b7568ef018145b9df2d798b359a67b62a8de5a388` | pass; empty debt; no first divergence |
| fresh-cold process | exact canonical G, 370 tokens, same SHA | same ledger SHA | pass; empty debt; no first divergence |

The exact output is 38 persons plus ties `gt10/gt12/gt44`: 41 unique strict
owners, zero duplicate/unmatched/unsupported/malformed/ambiguity/unknown debt,
and natural terminal-only EOS. The payload contains the same 36 selected FP64
output rows as G+QP v5, shape `36 x 2048`, with SHA-256
`d96f25707f44ee96d524370828fb3792ff91a35d429ff22e023014d6104c00e6`.

Authoritative receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-g-qp-norm-release-sentinel/20260831T-image2299-canonical-g-qp-norm-release-v2/receipt.json`,
SHA-256 `290e0fc30cfaa6de55ad08b1f6a1eae865ac75d1c060931174514e77d5979718`.
The v1 launch is retained only as technical-invalid provenance: it stopped
before model loading because its subprocess lacked the repository
`PYTHONPATH`.

## Interpretation

**Observation.** Under the original fixed `1.125` cap, M+QP alone passed while
G+QP was cap-gated. Once the exact G solution was admitted, both independent
processes reproduced canonical G exactly. M and G require normalized norms
`1.0972734315870412` and `27.98494582667317`, respectively: a `25.5041x` norm
ratio and `650.458x` ratio in the quadratic minimum-norm objective.

**Inference.** Canonical G is not merely statically valid or solver-feasible;
it is behaviorally compilable by the same protected-null output-only QP
mechanism. Sequence choice changes intervention cost dramatically, but the
matched experiment no longer supports a claim that only a model-emitted route
can succeed. Combined with both frozen CE negatives, the decision-grade result
is: QP can compile either route on this specimen when its required norm is
admitted, whereas the fixed four-LR, 50-update CE recipe compiled neither.

**Speculation.** The `25.5x` cost gap may reflect how far canonical token/box
realizations lie from the model's preferred route-conditioned logits. This unit
does not identify an internal representation or establish that the same cost
geometry holds across images.

## Further G46 successor

The later [canonical-G46 global-QP sentinel](../2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md)
solves all 46 canonical rows jointly from frozen r32. One 51-row residual at
normalized norm `28.282770681462463` exact-replays 46/46 warm and fresh-cold.
It uses no G41 or completed-46 payload, but retains G41's owner order before the
five-tie suffix.

## Claim boundary

This is single-image oracle-route, augmented-output-head evidence. The large
36-row residual is not a scalable or transferable learning result. It does not
establish multi-image protected-null capacity, held-out gains, parameter
efficiency, base-model behavior, or a deployable algorithm. The prior four-cell
table remains correct under its frozen `1.125` cap; this successor answers the
previously unexecuted norm-release question.
