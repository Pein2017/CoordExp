# Image2299 canonical-QP compilation phase closeout

Date: 2026-08-31

## Why this checkpoint exists

The user requested a phase closeout after the Image2299 route moved from a long
owner-exchange/grounding investigation to two cold-verified 46/46
augmented-output-head constructions. This note preserves the decision logic and
technical lineage that would otherwise be easy to misread from individual
receipts. Formal evidence remains in the linked research results and artifacts.

## Evidence chain

1. [Dyadic norm release](../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-30-image2299-dyadic-norm-release-distillation/results.md)
   first made the 38-person/3-tie M route ordinary-greedy and fresh-cold at
   normalized norm `1.09727`.
2. [Canonical five-tie composition](../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-31-image2299-canonical-five-tie-protected-null-sentinel/results.md)
   added the five missing ties and reached composed 46/46, receipt `e475bd7e...`.
3. [Matched M/G x QP/CE](../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-31-image2299-matched-sequence-optimizer-ablation/results.md)
   found M+QP cold success under cap `1.125`, both fixed CE schedules negative,
   and G+QP solver-feasible but cap-gated at norm `27.98495`.
4. [Canonical-G41 norm release](../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-31-image2299-canonical-g-qp-norm-release-sentinel/results.md)
   applied the exact frozen G solution and obtained two independent cold exact
   G41 replays, receipt `290e0fc3...`.
5. [Direct canonical-G46 global QP](../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md)
   restarted from frozen r32, solved all 57 positive states jointly over 51
   output rows, and obtained exact warm/fresh-cold 46/46 with one residual,
   receipt `ff04300c...`.

## Decision logic

The data no longer support “the model must supply a sampled row manifold before
QP can succeed.” Canonical rows are directly compilable on this specimen. The
model-manifold route remains valuable because its intervention is about 25.8x
smaller in normalized norm, not because canonical G is infeasible.

The result also narrows what the failed CE cells mean. They do not show that
backward learning is impossible. They show that the registered four-rate,
50-update averaged target-token CE program did not realize the simultaneous
hard set constraints that QP satisfied.

Direct canonical G46 does not yet remove every sampling dependency. Its rows
are fully canonical, but the owner order is G41's M-derived order plus the five
accepted ties. A deterministic-order sentinel is therefore a cleaner next
discriminator than another temperature/K sweep.

## Technical lineage that must not become science

- G41 v1 stopped before model loading because the verifier subprocess lacked
  repository `PYTHONPATH`; v2 is authoritative.
- G46 v1 had a feasible HiGHS point and a nearly exact SLSQP primal but the old
  helper treated SLSQP status 8 as a HOLD. The repaired runner admitted it only
  after primal feasibility and an absolute `2.05e-8` primal-dual gap certificate.
- G46 v2 was exact 46/46 in both processes. It remained a technical HOLD only
  because warm/cold evaluator labels were embedded in otherwise corresponding
  row and trajectory IDs. v3 uses one shared label and is authoritative.
- None of these technical corrections changed route, target rows, margin,
  selected surface, solve count, or behavioral acceptance.

## Stop and continuation boundary

The fixed Image2299 objective and this mechanism phase are complete. Do not
resume the retired DoRA/CE corridors, enlarge K, add a norm sweep, or infer
multi-image learning from the single image. If research resumes, freeze the new
estimand first: deterministic order or a small multi-image constrained-QP
cohort, with natural greedy retained/gained owners, debt, EOS, rank, norm, and
held-out behavior as decision evidence.
