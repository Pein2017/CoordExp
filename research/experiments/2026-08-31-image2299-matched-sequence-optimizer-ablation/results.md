---
title: Image2299 matched sequence-optimizer ablation results
type: investigation-result
role: research-result
authority: non_normative_research
unit_id: 2026-08-31-image2299-matched-sequence-optimizer-ablation
status: complete
evidence_status: conditional_single_instance_matched_ablation
updated: 2026-08-31
---

# Image2299 matched sequence-optimizer ablation results

## Disposition

**Complete.** On the frozen augmented-r32 Image2299 instance, only M+QP passed
the ordinary-greedy 41-owner gate and fresh-subprocess cold verification. The
other three cells are frozen-recipe negatives, not technical HOLDs.

## Authoritative evidence

| Receipt | SHA-256 |
|---|---|
| [preflight v6](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-preflight-v6/preflight.json) | `89002288f0485ea1c5f3421e9100c65eebd9265e0589b94dbba8eafd166c1996` |
| [M+QP v5](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-m-qp-v5/receipt.json) | `105b4c4045da3711892830226b1f03781bb4656832c16e6f4e50b15f8ec714d5` |
| [M+CE v5](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-m-ce-v5/receipt.json) | `35dc7754c0cc43c930035eeb7cfd9bdbb7fcea8edce920b4642cf9a3ba997e85` |
| [G+QP v5](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-g-qp-v5/receipt.json) | `a3944066e8b692a43c8b0fdfdb2ebaa6100ca96ada5758220f6a55046dd9fc7b` |
| [G+CE v5](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/20260831T-image2299-matched-ablation-g-ce-v5/receipt.json) | `1f5cb3b300c50ab15c6f7cd711ff9c3e7c77e15f77bd2222bfba1768a09553d9` |
| [final v1](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-matched-sequence-optimizer-ablation/final/20260831T-image2299-matched-ablation-final-v1/receipt.json) | `4b4b223f39d3db4881e1edec67f749bb3bee131351cefe6874004d48bc80a51e` |

The common admission is static-G accepted, `S_union=36` with SHA-256
`883bed1bf2e67b812a8ee459e4f8d0e25c19c607f229c202568191f750998129`.
M has 12 positives/rank 12/664 protected states; G has 38/rank 38/638.

## Four-cell outcome

| Sequence + optimizer | Observed outcome | Gate-relevant evidence |
|---|---|---|
| M + QP | **cold greedy 41-owner success** | Feasible minimum normalized norm `1.0972734315870412 <= 1.125`; warm and cold both exactly reproduce route `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`, with 41 owners. |
| M + CE | **frozen-recipe negative** | All `4 x 50` updates and all 32 scheduled warm milestones (`0,1,2,4,8,16,32,50` per LR) fail; final norms are `0.0006998`, `0.0013996`, `0.0027990`, `0.0055973`. No warm payload entered cold verification. |
| G + QP | **cap-gated frozen-recipe negative** | The QP is feasible, but its minimum normalized norm is `27.98494582667317 > 1.125`; the unchanged cap blocks a warm or cold candidate. This is not QP infeasibility. |
| G + CE | **frozen-recipe negative** | All `4 x 50` updates and all 32 scheduled warm milestones fail; final norms are `0.0013442`, `0.0026872`, `0.0053696`, `0.0107193`. No warm payload entered cold verification. |

## Observation, inference, speculation

**Observation.** Under the fixed single-instance admission, M+QP alone reaches
the cold ordinary-greedy 41-owner gate. The real-autograd CE panels produce no
warm candidate; G+QP is solver-feasible but cannot enter decoding under the
frozen cap.

**Inference.** M+QP success against M+CE failure supports QP over this frozen
four-LR, 50-update CE recipe. M+QP success against cap-gated G+QP supports a
model-manifold-heavy, sequence-conditioned contribution under the same selected
`S_union`, margin, norm cap, and QP program. M and G necessarily retain their
own route-conditioned bases and protected sets, so this is not an identical
hidden-state surface comparison.

**Speculation.** The canonical-G failure may arise from its hidden-state
trajectory, its sequence-conditioned basis, or another route-conditioned
factor. This unit does not distinguish those mechanisms.

## Successor norm-release result

The separate [canonical-G QP norm-release sentinel](../2026-08-31-image2299-canonical-g-qp-norm-release-sentinel/results.md)
subsequently admitted the exact same G+QP v5 rows at cap `28.0`. Both an
ordinary-greedy process and a fresh-cold process exactly reproduced canonical G
with 41 owners, zero debt, natural EOS, and no first divergence. Thus the
original four-cell table remains correct under cap `1.125`, while the successor
shows that G is behaviorally QP-compilable at `27.98494582667317`; sequence
choice changes cost, not demonstrated feasibility.

## Claim boundary and lineage

This is not a universal CE-impossibility, owner-visibility, owner-
representation, base-r32, transfer, cross-image, training, or 46/46 claim.
Static G owner-equivalence is an admission control, not natural model owner
visibility evidence. Preflight v1--v5 and cell v1--v4 are retained only as
technical-invalid provenance; all scientific statements above use v6/v5 and
the final receipt.
