## 0. Authorization gate

- [x] 0.1 Obtain explicit implementation authorization before starting any
  code, config, test, or runtime task in this change; completed planning
  artifacts do not grant that authority.

## 1. RP policy evidence vertical

- [x] 1.1 Add focused failing tests for the exact RP transform, temperature,
  chosen-token processed log probability, full prompt/generated-history
  convention, cap/stop rules, sealed parity tolerances, and deterministic
  policy fingerprints at `rp=1.0` and `rp=1.10`.
- [x] 1.2 Implement the experiment-local policy-evidence records and pure replay
  transform without changing production inference defaults.
- [x] 1.3 Add the batch-four K16 acquisition adapter, immutable request/output
  receipts, and no-update sampler-versus-replay parity check.
- [x] 1.4 Gate Wave 1 with focused tests, a zero-action dry run, format/lint,
  residue checks, and separate standards and research-intent reviews.

## 2. Detached trajectory credit

- [x] 2.1 Add failing tests for frozen trusted-owner weights, first-hit credit,
  burden precedence, legacy-M direct-token masking, natural-STOP versus cap
  sign, row returns, tied groups, and leave-one-out advantages.
- [x] 2.2 Implement the pure trajectory ledger and immutable row/token advantage
  artifact with complete Source/manifest/policy lineage.
- [x] 2.3 Implement the packed score-function loss over replayed processed
  chosen-token likelihoods; apply the logical `N*K` denominator once and verify
  gradients are absent from all selectors.
- [x] 2.4 Gate Wave 2 with mathematical counterexamples, tensor gradient tests,
  arbitrary pack/gradient-accumulation invariance, artifact round trips, and
  independent standards and intent reviews.

## 3. Sparse greedy compiler

- [x] 3.1 Add failing tests for frozen alias membership, owner/alias
  normalization, log-mean-exp bounds, realized-bad-child binding, absent
  boundaries, RP-processed greedy logits without temperature division, and no
  fresh-support leakage.
- [x] 3.2 Implement Source-boundary compiler materialization and the separately
  receipted image-mean compiler loss using the existing compact-logit pack path.
- [x] 3.3 Verify the nested trajectory versus trajectory-plus-compiler packs use
  byte-identical acquisition and credit artifacts and differ only by the named
  compiler sites.
- [x] 3.4 Gate Wave 3 with focused tests, packing parity, bounded memory/token
  receipts, and independent standards and intent reviews.

## 4. Exact AdamW proposal preservation

- [x] 4.1 Add failing tests that compare captured parameter deltas with an actual
  fresh-AdamW step, including denominator, trust-radius, and state restoration.
- [x] 4.2 Implement private exact-proposal capture over the existing full
  training-state transaction and bind the gradient, optimizer, delta, and
  metric evidence.
- [x] 4.3 Add failing tests for feasible, active-set, infeasible, non-finite, and
  trust-radius projection cases plus predicted and realized witness changes.
- [x] 4.4 Implement the bounded owner-wise projection and fail-closed applied-
  delta receipt without defining projected optimizer-moment continuation;
  retain finite realized witness degradation for dual-RP behavioral audit.
- [ ] 4.5 Gate Wave 4 with finite differences, exact applied-parameter hashes,
  rollback reproduction, measured CPU/GPU memory bounds, and independent
  standards and research-intent reviews.

## 5. Bounded matrix runtime and analyzer

- [x] 5.1 Add six descriptive leaf configs for the three nested arms under the
  two training RP contracts, plus the three frozen seed groups and dual-RP
  evaluation contract.
- [x] 5.2 Implement a dry-run-first matrix materializer that shares acquisition
  only within one RP/seed group, allocates independent Source/fresh-optimizer
  proposals, caps concurrency at eight GPUs, and forbids adaptive settings.
- [x] 5.3 Implement the private one-update runtime using the existing Human-13
  assembly, packed forward, HF fp32/SDPA evaluator, transaction, and rollback
  seams.
- [x] 5.4 Implement the dual-RP analyzer with separate trusted gains, named
  baseline losses, historical G/H/M identities, M incidental recovery, output
  burdens, same-seed contract-local/RP-robust rules, and exact complete-matrix
  admission.
- [x] 5.5 Gate Wave 5 with zero-action dry runs, forged-lineage and incomplete-
  matrix tests, full focused CPU integration, strict OpenSpec validation, and
  independent standards and intent reviews.

## 6. Production-shaped vertical

- [x] 6.1 Obtain explicit model/GPU execution authorization and freeze a new
  immutable vertical root; do not infer this authority from completed docs.
- [x] 6.2 Implement exact score-function replay and the score-function
  gradient forward on the existing HF fp32/SDPA exact-history batch-one
  surface with focused CPU tests on real frozen shapes; forbid BF16/FA2
  packed score-function evidence and admit packed materialization only for
  non-score-function plumbing with proof of mathematical identity.  This is
  an execution-surface correction: keep A/B/C arms, both RP contracts, the
  sealed 0.02/0.002 gates, exact histories/processed semantics, estimand,
  optimizer, and owner gates unchanged.
- [x] 6.3 Run the reserved, confirmed-absent v5 root as a one-image (1584)
  K16 parity-only qualification: acquisition plus exact-surface replay at
  rp=1.0 and then rp=1.10, each against the unchanged sealed gate, before any
  witness, dose, update, or owner analysis.  If either contract fails, retire
  the exact-on-policy route on that recorded result without tolerance
  revision; if both pass, reserve a fresh full-panel successor root for the
  remaining vertical.

  Executed-negative disposition: `rp=1.0` native acquisition passed, but exact
  fp32/SDPA replay failed the unchanged gate (`max=0.1675825`,
  `mean=0.0021683`, `22/1573` tokens over `0.02`).  The first-failure rule
  retired the exact-on-policy route before `rp=1.10` or any update.  Tasks
  6.4--7.4 remain intentionally unexecuted because no admitted vertical exists.
- [ ] 6.4 On the passing successor root, run one real batch-four K16
  acquisition per RP on the disjoint qualification seed group and require
  exact request coverage plus sampler/replay numeric parity on both
  contracts.
- [ ] 6.5 Evaluate the sealed qualification-only AdamW dose ray
  `{3e-7,1e-6,3e-6,1e-5,3e-5}` from fresh Source state under both RP
  contracts; run at most one independent C proposal per attempted `(RP,dose)`,
  quarantine owner outcomes, apply the declared mechanical floor/ceiling, and
  publish exactly one content-bound global learning-rate choice.
- [ ] 6.6 Materialize all three nested objectives, then admit the selected ray
  proposal per training RP only after backward, exact AdamW capture,
  projection, private apply, both clean-greedy audits, and exact rollback
  reproduction are complete.
- [ ] 6.7 Publish vertical wall time, peak memory, decode/packed tokens,
  forward/backward counts, witness counts, artifact sizes, and a bounded
  admission verdict; exclude owner outcomes from matrix disposition and stop on
  any unresolved P0 or P1.

## 7. Fixed Human-13 mechanism screen

- [ ] 7.1 Materialize and prevalidate the exact eighteen-cell matrix against the
  admitted vertical and frozen research-unit identities.
- [ ] 7.2 Execute the matrix without adaptive retries, accepted checkpoints,
  acquisition refresh, or cross-cell optimizer/model reuse.
- [ ] 7.3 Analyze paired A-to-B-to-C changes within every RP/seed group and the
  crossover behavior under both evaluation RP surfaces.
- [ ] 7.4 Publish immutable artifacts and independent standards and scientific-
  interpretation reviews; do not promote a checkpoint or generalization claim.

## 8. Closure and continuity

- [x] 8.1 Write bounded results, disposition, exact artifact/config/checkpoint
  links, negative findings, claim boundary, and the next user-owned decision.
- [x] 8.2 Update the research graph and repository-local project memory with the
  accepted evidence and continuation point.
- [x] 8.3 Run strict OpenSpec verification and residue checks; archive only after
  implementation, execution evidence, and both independent reviews agree.

  Strict validation, the focused 105-test slice, canonical content-hash checks,
  prohibited-artifact residue checks, process cleanup, and the independent
  evidence review all pass for the bounded negative closure.  This change is
  not archived: Task 4.5 and Tasks 6.4--7.4 are intentionally unexecuted and
  retired rather than falsely completed.

## Closeout disposition (2026-08-24)

No incomplete checkbox is changed during closeout. Direct evidence in
[`results.md`](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/results.md)
shows that RP1.0 parity failed before backward/update, so Task 4.5 and Tasks
6.4--7.4 have no admitted prerequisite and remain intentionally unexecuted.
The later standalone N=13 K4/K8 artifact is a separate treatment and cannot be
used to mark these exact-policy tasks complete.
