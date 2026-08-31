---
title: Image2299 canonical-G46 global-QP protected-null sentinel
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: fresh_cold_direct_canonical_g46_success
updated: 2026-08-31
---

# Image2299 canonical-G46 global-QP protected-null sentinel

## Question and contrast

Can one global protected-null output-only QP compile a fully canonical 46-row
Image2299 route directly from frozen r32, without using a model-emitted M route
or composing the already successful five-tie child payload?

The target is G41's frozen owner order followed by the accepted missing-tie
suffix `[gt45,gt11,gt9,gt8,gt43]`; every owner contributes its complete
canonical nine-token `target.json` row, followed by one EOS. This contrasts with
the completed 46/46 mainline, whose first 41 rows are model-emitted and whose
last five rows are a second composed QP stage.

## Frozen target and inputs

- Owner order:
  `30,23,29,16,7,38,44,31,6,40,28,21,3,24,37,26,13,27,41,25,2,19,39,10,4,17,42,33,5,12,15,36,22,20,1,0,35,32,14,34,18,45,11,9,8,43`.
- Exact route: 46 canonical rows, 415 tokens, pre-EOS SHA-256
  `d5e7dc703727f2f3db265df848f4c64fbc67e9fb8417eca23d0ae0c8e69c8dbb`,
  route SHA-256
  `96c2cbe15fdb4c09822d6a472f2510605c5701cfafcc9e33324f14ce90e0cb49`.
- Target library SHA-256:
  `22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988`.
- Frozen r32 checkpoint, prompt, image, parser/global matcher, FP32 HF runtime,
  and output-only tied-head handling are identical to the completed matched
  ablation.

The CPU static gate is already witnessed as 46 accepted predictions, 38
persons, 8 ties, 46 unique owners, zero hard debt, and row-aligned terminal EOS.
It must be replayed by the production runner before model optimization.

## One global QP

Teacher-force G46 at zero residual exactly once. Freeze all positions whose
target is not strict top-1 and set `S` to the sorted unique target IDs at those
positive positions. Build one protected-null basis against the base ordinary
route and every G46 nonpositive state. Optimize one FP64 output residual:

`D_s = ||W_s|| X_s B^T`, for `s in S`.

The sole QP minimizes `1/2 ||X||^2` while making every positive target beat all
other movable rows and the strongest fixed-vocabulary competitor by margin
`0.01`. Require finite feasibility, numerical slack, protected-null tolerance
`<=1e-10`, and exhaustive full-vocabulary recheck. There is deliberately no
norm cap: record the minimum norm and admit that one solution without a sweep,
iteration, or second solve.

## Acceptance and stop rule

Apply exactly one output-head residual. One warm ordinary greedy and one fresh
subprocess cold replay must both equal the exact G46 route and ledger: 38
persons, 8 ties, 46 strict owners, zero debt, natural terminal EOS, and exact
frozen surface identities. Tied input embeddings, DoRA, existing embedding
delta, aligner, vision tower, and every nonselected output row remain frozen.

Stop after this single solve and its warm/cold decision. A certified
infeasibility, projected-rank collapse, solver/resource HOLD, warm divergence,
or cold mismatch is recorded without changing route, solver, surface, margin,
or architecture. The completed 46/46 artifact remains an immutable external
control and is never an input payload.

## Claim boundary

Success would establish single-image direct canonical-G46 compilation on an
augmented output head. It would not establish parameter efficiency,
multi-image generalization, scalable training, base-model behavior, or
transfer. Failure would reject only this frozen one-shot surface/program.

## Completed evidence

The authoritative v3 run used one global QP and one output-head residual. Warm
and fresh-cold ordinary greedy exactly reproduced the 415-token G46 route with
38 persons, 8 ties, 46 strict owners, zero debt, and natural EOS. See
[results](results.md). v1 is numerical-solver lineage and v2 is label-only
ledger-parity lineage; neither is scientific negative evidence.
