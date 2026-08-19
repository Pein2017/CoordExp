# Wave 6 Final Verification Receipt (6.1-6.4 + matrix reconciliation)

Date: 2026-08-19. Lead: Claude Fable session. State under verification:
HEAD `161160592094b5b9493dacc8ab9f546f231830ab` with the evidence-matrix
reconciliation staged on top; production `src/`, `scripts/`, and `configs/`
are byte-identical from Attempt-8's bound commit
`f492f36874f036ad4145a45c7b03cd2e0b2fd049` through this state
(`git diff --stat -- src scripts configs` empty).

## 6.1 Focused suites and directories

`conda run -n ms python -m pytest -q tests/config tests/artifacts
tests/training tests/inference` → **2088 passed, 0 failed, 0 skipped**
(837s). Zero skips means no unexpected-skip investigation was required.
Earlier same-day focused runs at intermediate commits: 489+3+15 (Wave-4
gate, clean bytecode), 179 (Wave-5 gate), 246 (pre-execution insurance).

## 6.2 Frozen-verifier evidence from the final implementation state

The frozen verifier's writer path is immutable by design
(`reconcile_probe.receipt_exists` fail-closed on an existing
`terminal-receipt.json`), so a destructive re-run against the `-r8` root is
not permitted by the contract itself. 6.2 is satisfied by the conjunction:

1. **Input identity**: production `src/scripts/configs` bytes at the final
   state equal the bytes at `f492f3687` where the frozen verifier executed —
   the executed verifier IS the final implementation state's verifier.
2. **Read-only re-authentication** (script preserved in the session job
   directory; result PASS): all eight signed digests recomputed and valid
   (outer terminal receipt `verified`, inner terminal receipt `verified`
   with 5/5 required comparisons, zero missing inputs, zero bounded
   mismatches; pre-cost review `READY`; prepare receipt `prepared`; pack
   cache receipt `completed`; rank-failure `converged_failure`;
   interruption `converged_partial_state`); durable run states re-checked
   (control `completed`, parent `initialized` with null `completed_at`,
   child `completed`); step-2 objective-row equality re-derived from
   durable `logging.jsonl`: 21/21 compared fields exactly equal.

## 6.3 Strict validation and conflict scan

`openspec validate reconcile-coordexp-swift-training-contracts --strict` →
valid (re-run after every wave close and after the matrix reconciliation).
Zero merge-conflict markers under `openspec/`. All four delta specs carry
scenarios for every requirement (config-runtime 2/7,
pack-cache-semantic-identity 3/12, training-artifacts 2/9, training-resume
5/12).

## 6.4 Residue scans

- This session changed zero bytes under `src/`, `scripts/`, `configs/`
  (all six commits since `4525a0f73` are receipts/tests/docs/openspec).
- No production cache campaign: the Wave-4 probe and Attempt-8 setup used
  only private dot-prefixed cache roots; no production pack-cache target
  was created or mutated.
- Documentation promotion scan (Wave-5 gate): no canonical page promotes
  changed-order packing, speculative efficiency, logging enhancement,
  loss-objective/RL behavior, or architecture work; categorical denials
  replaced only with the bounded contract language.
- No dependency upgrade, orchestration refactor, historical migration
  shim, cross-world-size resume, or mid-accumulation promise appears in
  the change's diff or receipts; the exact-resume claim everywhere stays
  bounded to same-world-size optimizer-step boundaries at two-rank
  smoke scale.

## 6.6 (first half) Evidence-matrix reconciliation

`evidence-matrix.md` re-pinned with the original pin preserved as history;
the four Wave-3 `gap` rows are now `accepted` citing exact Attempt-8
receipt fields (comparisons `boundary_step1_control_vs_parent`,
`post_update_step2_control_vs_child`,
`post_update_objective_step2_control_vs_child`, `rank_failure_arm`,
`interruption_arm`; zero `bounded_mismatches`), with owner columns
corrected to the module that actually executed
(`reconcile_exact_resume_probe.py:verify_artifacts`). Wave-4/5 nodes and
receipts folded into their owning rows; the inference-ignore row's claim
boundary widened from equality-level to file-access-level at the
payload-reader layer, still excluding `src/inference/pipeline.py`
composition. All newly cited nodes and owners grep-confirmed to exist.

## Dispositions carried to 6.5

- P3: pre-publish payload-validation failure surfaces as plain
  `ValueError("packing cache chunk payload is unreadable")` without a
  `validation_category`, unlike sibling branches (fail-closed semantics
  intact; taxonomy inconsistency only).
- Accepted-by-ordering: optimizer/scheduler non-construction on cache
  failure is transitive (model assembly precedes optimizer construction;
  cache failure precedes model assembly with executed proof).
- Cosmetic: executor error string "not the fixed Attempt-6 input" while
  constants are correct.
- Incident (resolved): poisoned `.pyc` from a same-length same-second
  mutation-kill probe; diagnosed via `co_consts` vs in-process compile,
  purged, and all affected gates fully re-run.
