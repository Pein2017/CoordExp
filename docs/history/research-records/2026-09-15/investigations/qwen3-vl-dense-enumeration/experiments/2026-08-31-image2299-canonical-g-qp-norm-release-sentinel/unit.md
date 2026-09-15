---
title: Image2299 canonical-G QP norm-release sentinel
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-31-image2299-canonical-g-qp-norm-release-sentinel
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: fresh_cold_exact_canonical_g_41_owner_success
updated: 2026-08-31
---

# Image2299 canonical-G QP norm-release sentinel

## Question and decision

Does the exact solver-feasible canonical-G QP candidate rejected only by the
matched ablation's `1.125` norm cap produce the complete canonical G route under
ordinary greedy decoding when that single cap is released?

The decision-owning outcome is exact ordinary-greedy and fresh-process parity
with route SHA-256
`e238e67122aa46b54cf9290d11093b07a7490746d6730d6f843f8d4ee61d677d`:
38 persons plus ties `gt10/gt12/gt44`, 41 unique strict owners, zero hard debt,
and natural terminal-only EOS. The strongest alternative is that the static
teacher-forced QP certificate does not survive runtime application.

## Frozen candidate

- Source G+QP v5 receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-g-qp-v5/receipt.json`,
  SHA-256 `a3944066e8b692a43c8b0fdfdb2ebaa6100ca96ada5758220f6a55046dd9fc7b`.
- Immutable runner snapshot SHA-256:
  `e8efaea02fc6840106ed0ef07616f5e88bd13ad782948b3f72710a33b51b9c43`.
- Preflight v6 SHA-256:
  `89002288f0485ea1c5f3421e9100c65eebd9265e0589b94dbba8eafd166c1996`.
- Candidate: fixed `S_union=36`, G rank 38, full-vocabulary pass, minimum
  normalized norm `27.98494582667317`, minimum slack
  `-2.41229258790554e-12`.
- New sentinel admission cap: exactly `28.0`. This cap admits the already
  frozen solution; the sentinel does not re-solve or search.

All checkpoint, prompt/image, routes, rows, basis/protected identities, margin,
output-only parameterization, matcher, and decoding settings remain frozen.
The existing four-cell result and completed 46/46 mainline remain immutable
external evidence.

## Execution and stop rule

Materialize the exact G+QP v5 residual once. Run the immutable verifier twice
in independent model processes: one ordinary-greedy witness and one fresh-cold
witness. Both must pass the exact route, owner/debt/EOS gate and have identical
route and ledger hashes. Record the first divergence on failure.

This one-shot sentinel stops on success, scientific negative, or technical
identity/runtime HOLD. If it is scientifically negative, a separate successor
may iteratively add only the observed first-divergence cut; no such iteration
is silently folded into this sentinel. There is no CE, sampling, controller,
training, embedding/aligner/vision change, cap sweep, or new QP solve here.

## Claim boundary

Success would establish only single-image canonical-G compilation by one large
output-only residual. It would not establish parameter efficiency, multi-image
generalization, scalable learning, base-model behavior, or transfer. Failure
would reject this exact frozen residual at runtime, not QP in general.

## Completed evidence

The exact frozen candidate succeeded without any iterative cut. Two independent
model processes produced the exact 370-token canonical G route, route SHA-256
`e238e67122aa46b54cf9290d11093b07a7490746d6730d6f843f8d4ee61d677d`,
ledger SHA-256
`d0e5b1a07b97a0b5b987ca1b7568ef018145b9df2d798b359a67b62a8de5a388`,
empty debt, and no first divergence. See [results](results.md).
