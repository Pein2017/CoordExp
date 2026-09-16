## 1. Shared-surface contracts and CPU math

- [x] 1.1 Freeze one failure-mode matrix for surface identity, request/history
  lineage, processor order, parity, objective denominator, optimizer delta,
  private audit, and rollback; name one executable owner and one minimal
  counterexample for each invariant.
- [x] 1.2 Add typed shared-surface identity, policy, group-plan, sampled-group,
  replay-group, parity, and close receipts with one admission choke point used
  by constructors, loaders, publishers, and runtime consumers.
- [x] 1.3 Add CPU/tensor TDD for four-by-four K16 planning, no-cache enforcement,
  exact seed/request coverage, active-batch histories, sign-aware RP processing,
  chosen-token gathering, and `0.02/0.002` parity admission.
- [x] 1.4 Add a pure resource estimator and dry-run receipt for image/prompt
  forwards, replay forwards, backward count, expected token cap, required GPU
  roles, output roots, and zero model/GPU actions.
- [x] 1.5 Run one independent standards/intent review of Wave 1, apply at most
  one bundled P0/P1 correction, rerun only localized checks, and stop CPU tamper
  hardening when no P0/P1 remains.

## 2. All-HF sampling and replay integration

- [x] 2.1 Implement an experiment-local `HFSharedSurfaceSession` that loads one
  Source BF16/FA2 language-only DoRA model, keeps it in eval mode, owns the
  tokenizer/processor and parameter layout, and closes all model/session/cache
  state on every exit.
- [x] 2.2 Implement no-cache, full-history, stepwise HF sampling for four active
  requests with exact histories, processed chosen log probabilities, RNG and
  shape receipts; execute four groups for K16 without vLLM or prefix-cache use.
- [x] 2.3 Implement grad-enabled sampler-step-aligned teacher-forced replay
  for one completed group: reproduce each recorded active batch and causal
  history length, select each sampled action's causal logit, apply the same
  policy processor, and retain bounded checkpointed replay tensors only after
  parity admission.
- [x] 2.4 Prove through injected production-shaped tests that sampling and replay
  use the same model-object identity, unchanged parameters, adapter/delta,
  BF16/FA2 surface, model mode, prompt/image/tokenizer, and RP order, and fail
  before backward on any substitution.
- [x] 2.5 Gate Wave 2 with focused tests, Ruff/compile/Serena checks, strict
  OpenSpec validation, resource-bound assertions, and one bounded review plus
  one correction bundle.

## 3. Complete one-update composition

- [x] 3.1 Adapt the admitted replay rows to the existing sealed
  `TrajectoryCreditLedger` and global `N*K` numerator/denominator without
  changing first-hit, burden, STOP, RLOO, or legacy-M semantics.
- [x] 3.2 Materialize the existing sparse compiler on a BF16-native canonical
  Source projection from the same live BF16/FA2 session, including
  absent-site zero semantics, frozen alias binding, `kappa=1`, margin `1e-4`,
  coefficient `1.0`, and remaining-owner state that excludes H owners already
  covered by the BF16 Source baseline.  Do not feed fp32/SDPA token paths into
  the compiler or preservation witness bank.
- [x] 3.3 Reuse fresh AdamW at fixed `3e-6` and the existing actual-delta
  owner-wise preservation projection; require all three components and forbid
  CE, unprojected, or missing-component fallback.
- [x] 3.4 Wrap backward, proposal apply, private checkpoint, and failure paths in
  the complete `TrainingStateTransaction`, including gradients, counters,
  optimizer/scheduler, CPU/CUDA RNG, and exact Source restore.
- [x] 3.5 Add objective/component/resource receipts and tests showing exactly
  one update, unchanged global normalization across group boundaries, finite
  gradient/delta diagnostics, exact projected apply, and exact rollback.
- [x] 3.6 Gate Wave 3 with the focused and adjacent Human-13 suites, strict
  OpenSpec validation, one production-shaped zero-action entry, and the bounded
  review/correction discipline from Wave 1.

## 4. Dual-RP behavioral audit and live entry

- [x] 4.1 Add one leaf config and guarded entry for image 1584, disjoint seeds
  `35001..35016`, K16 batch-four, training RP 1.0, BF16/FA2 shared surface,
  fixed AdamW `3e-6`, and a new confirmed-absent immutable output root.
- [x] 4.2 Bind GPU 0 to the shared training session and GPU 1 to the established
  HF fp32/SDPA batch-one audit path; fail before model load when two distinct
  cards, output roots, or authority receipts are unavailable.
- [ ] 4.3 Build independent canonical baselines: BF16-native Source
  projection/witness/compiler inputs on GPU0 and fp32/SDPA Source clean-greedy
  baselines at RP 1.0 and RP 1.10 on GPU1.  Publish H gained, G lost,
  incidental M gained, net unique owners, duplicate/unmatched/malformed
  burdens, stop/cap, rows, and tokens under the canonical parser/matcher for
  the fp32 Source-versus-proposal audit.  Keep cross-surface token/row/owner
  differences as `diagnostic_only` evidence; enforce strict identity fields and
  protected BF16 G presence, but do not use coordinate equality or owner-set
  equality as an admission gate.  BF16/FA2 sampler-to-replay parity remains
  strict.
- [x] 4.4 Implement the exact continuation gate and a full-panel entry that
  remains model/GPU-inert unless it receives the content hash of a passing
  one-image terminal; add fail-closed tests for every missing condition.
- [x] 4.5 Add immutable phase receipts, private-proposal cleanup, Source
  reproduction, resource telemetry, and one terminal that distinguishes
  parity failure, update failure, completed-null/unsafe result, and passing
  one-image result.
- [ ] 4.6 Run the final prelaunch smoke review against the real public CLI,
  config, model assembly, BF16-native Source projection/witness/compiler,
  sampler/replay, backward, private checkpoint, paired fp32 audit, analyzer,
  rollback, and consumer interfaces; verify the shared independent-baseline
  choke point, diagnostic-only divergence receipt, unchanged internal replay
  parity, and the post-prepare AdamW runtime ownership/pre-acquisition
  admission boundary; resolve only conclusion-changing P0/P1 findings before
  execution.  This ownership correction is production admission evidence,
  not algorithm evidence; the phase must also persist the ownership receipt
  digest and reject missing backend hooks, non-cosine schedulers, or live
  param-group hyperparameter drift before K16.
- [x] 4.6a Add the strict canonical CUDA logical-device identity receipt and
  revalidation, plus append-only training/audit/evaluator action-attempt
  accounting with explicit legacy terminal schema dispatch. This is CPU-only
  production-admission evidence; it does not mark the real prelaunch or any
  scientific Task-5 execution complete.
- [x] 4.6b Replace opaque replay graph-owner rejection with one shared typed,
  content-addressed graph-owner attribution at HF replay creation and CUDA
  adapter admission; attest non-model inputs before K16, persist value-free
  receipt lineage/bounded foreign-leaf details and full foreign count, preserve
  typed rejection across evidence-publication failure, and cover the
  production-shaped constrained CPU-injected sentinel separately from K16
  counters. This is CPU/injected diagnostic evidence only and does not
  complete 4.6 or scientific Task 5.2+.

## 5. Bounded execution and closure

- [x] 5.1 Inspect live GPU/process/artifact state, reserve two suitable cards and
  the immutable one-image root, and run the guarded no-update image-1584
  preflight: freeze the BF16-native Source witness/compiler inputs and fp32
  Source baselines, publish cross-surface divergence as diagnostic-only, and
  keep K16 sample/replay/update counters at zero.
- [ ] 5.2 If independent BF16/fp32 baseline admission and strict BF16 internal
  parity both pass, continue in the same declared run to one complete private
  update, both fp32 clean-greedy audits, exact rollback, and Source
  reproduction; if a protected BF16 G owner is missing or parity fails,
  publish the typed implementation HOLD with zero update and do not tune
  tolerances or force cross-surface agreement.
- [ ] 5.3 Independently audit artifact hashes, surface lineage, request/token
  coverage, one-update count, gained/lost arithmetic, prohibited-path absence,
  private-byte lifecycle, and rollback before interpreting the result.
- [ ] 5.4 Execute one 13-image K16 shared-surface update only when the exact
  one-image continuation gate passes; otherwise record it as intentionally
  unexecuted rather than incomplete infrastructure.
- [x] 5.5 Publish bounded results/review, update the research graph and project
  memory, run strict OpenSpec and residue checks, and leave validation,
  scalable vLLM/off-policy work, checkpoint promotion, and archive to a new
  user-owned decision.

## Closeout disposition (2026-08-24)

The closeout changes only the two checkboxes whose exact task text is now
satisfied by durable execution or documentary closure:

- Task 3.2 is complete as a materialization task. The durable K16 receipt at
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-all-hf-shared-surface-trajectory-credit-vertical/one-image-scientific-fresh-primary-6aa30b7-v1/receipts/017-k16_acquisition_replay.json`
  (content SHA-256
  `f6cf74ffd752589c63c2075c921a8fe13d812845f5b4dabdec35cf13ff885989`)
  records 463 sampling plus 463 replay forwards and binds compiler ledger
  SHA-256
  `55c7794cf64c7f5be909471fc1eb119f39ab05738622b10d87c07bc353f99072`.
  This does not imply that the compiler objective, backward, or update was
  admitted.
- Task 4.3 remains incomplete. Four canonical Source baselines were durably
  admitted, but no proposal audit or gained/lost publication completed the
  compound task.
- Task 4.6 remains incomplete; 4.6a and 4.6b are bounded CPU/injected
  diagnostics and do not constitute the real prelaunch smoke.
- Tasks 5.2--5.4 remain incomplete. No official all-HF attempt completed
  backward, an optimizer step, proposal audit, rollback after update, or the
  conditional 13-image continuation.
- Task 5.5 is complete only as the bounded documentary closeout of this
  partially executed, retired route: the owning unit and research index are
  aligned, project memory records the successor decision, strict OpenSpec and
  residue checks pass, and an independent closure audit reviewed the claim
  boundary. It is not completion of Tasks 5.2--5.4, an algorithm result,
  checkpoint promotion, or archive.

These tasks are intentionally retired, not pending implementation. The later
standalone N13 K4/K8 matrix belongs to a simplified treatment and provides no
direct completion evidence for this change. Its strict claim boundary and
current program decision are recorded in
[`memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md`](../../../memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md).
