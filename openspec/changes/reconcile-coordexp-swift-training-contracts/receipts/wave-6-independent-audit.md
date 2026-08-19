# Wave 6 Independent Final Audit (Task 6.5)

Auditor: independent Claude session, not the implementer. Date: 2026-08-19.

**Audited commit: `6ede480b0a86f1c83b91ae0de94cebdaf6468c11`** (verified by
`git rev-parse HEAD`; branch `coordexp-swift`, tree clean). Note: `tasks.md`
and `evidence-matrix.md` pin "final HEAD `161160592`". `6ede480b0` is one
later commit that adds only `receipts/wave-6-final-verification-receipt.md`
and the `evidence-matrix.md` reconciliation (`git diff --stat
161160592..HEAD` = 2 files, docs/openspec only). This is disclosed by the
Wave-6 receipt itself ("with the evidence-matrix reconciliation staged on
top"), so it is a documentation lag, not undisclosed drift.

Runtime: `conda run -n ms`. No GPU/model command was executed. No production
file was modified; this receipt is the auditor's only repo write.

---

## VERDICT A — STANDARDS / CODE QUALITY: **PASS-WITH-DISPOSITIONS**

No P0. One P1-adjacent proof-strength issue recorded as **P2** (does not
invalidate any claim). Remaining items are P3.

### P2-1 — `Path.open` interception is one-sided; a non-`Path.open` reader mechanism could silently pass

`tests/artifacts/test_checkpoint_payload_identity.py:330-364` proves the
inference reader never opens a `training_state/` path by monkeypatching
`pathlib.Path.open` (`:331-337`) and asserting
`not any("training_state" in path for path in opened_paths)` (`:364`).

The interception does not observe every reader mechanism actually used:

- `src/artifacts/checkpoint_payload.py:192,211` calls into
  `src/adapters/dora.py:inspect_dora_adapter_payload` and
  `src/qwen/special_token_embeddings.py:inspect_special_token_embedding_delta_payload`.
- Those modules read tensors through the native safetensors loader —
  `src/adapters/dora.py:1229` and
  `src/qwen/special_token_embeddings.py:397`, both
  `safe_open(str(tensor_path), framework="pt", device="cpu")`.
- Empirically verified by this audit (Python 3.12.11): `safe_open` records
  **zero** opens under a `Path.open` interception, while `Path.read_text`
  does route through `Path.open`.

The positive assertions at `:357` and `:362`
(`adapter_model.safetensors`, `special_token_embeddings.safetensors` must
appear among the opens) therefore do **not** close the hole: those paths are
captured because `_sha256_file`
(`src/artifacts/checkpoint_payload.py:357-362`, `path.open("rb")`) hashes
every inventoried file, not because the safetensors reads are observed. A
hypothetical reader that touched `training_state/` via `safe_open`,
`torch.load`, `os.open`, or builtin `open()` would be invisible to
`opened_paths` and the negative assertion would still pass.

Why P2 and not P1 — the underlying claim is independently true:

- `training_state` has **zero** references under `src/inference/`,
  `src/adapters/`, and `src/qwen/` (grep confirmed by this audit).
- The evidence-matrix claim boundary (row "Inference reads a checkpoint with
  exact state") explicitly names the mechanism —
  "`pathlib.Path.open` interception ... capturing 45 real opens" — so the
  matrix statement is literally accurate and appropriately hedged.

Disposition: the generalization in
`receipts/wave-5-historical-docs-receipt.md:34-37` — "The interception
matches the reader's actual mechanism" — is accurate for
`checkpoint_payload.py` only and is **overstated** for the DoRA and
special-token readers reached through it. Recommended (non-blocking): either
add `safetensors` `safe_open`/builtin-`open` interception, or narrow that
sentence to `checkpoint_payload.py`.

### P3-1 — `ValueError` taxonomy (pre-existing disposition) — **confirmed below P1**

`src/training/pack_cache.py:987` (read failure) and `:1000` (restricted
unpickler rejection) raise the *identical* plain
`ValueError("packing cache chunk payload is unreadable: ...")` with no
`validation_category`. Fail-closed semantics are intact: the sampled test
asserts the target is absent and no stage residue survives
(`tests/training/test_pack_cache.py:2195-2196`).

Additional precision note found by this audit: because both branches share
the same message, the test's `pytest.raises(ValueError, match="unreadable")`
(`:2182`) cannot discriminate *which* guard fired. A **digest** mismatch is
correctly excluded, because that branch raises a different message
("expected ... got ...") at `src/training/pack_cache.py:989-993`, and the
digest check runs *before* the unpickler (`:989` precedes `:997`). The test
is therefore sound for "fails closed at publication" and merely imprecise for
"fails specifically on the forbidden global". P3 stands.

### P3-2 — cosmetic executor string (pre-existing disposition) — **confirmed below P1**

`scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py:566`
raises "base config identity is not the fixed Attempt-6 input" while the
compared constants are correct:
`BASE_CONFIG_SHA256 = "44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f"`
(`:62`), which this audit matched against both the Attempt-8 frozen manifest
and the on-disk base config. Message text only. P3 stands.

### P3-3 — substring-marker exclusion list in the objective comparison

`scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:1777-1780`
excludes keys by *substring* markers `("duration", "resource", "wall",
"timing")` plus three explicit keys. A future objective-bearing key
containing one of those substrings would be silently excluded. Empirically
benign for the `-r8` run: the 9 excluded keys are
`input_build_seconds`, `input_wait_seconds`, `per_rank_measurement`,
five `resource/*` keys, and `step_duration_seconds` — all timing/resource,
none objective-bearing. The full policy is recorded inside the signed
terminal receipt (`comparison_policy.post_update_objective_step2_control_vs_child`),
which is good practice. P3.

### P3-4 — "21 of 21 compared objective fields" is an unlabeled subset count

`receipts/wave-6-final-verification-receipt.md:37` and `tasks.md:207` state
"21 of 21 compared objective fields exactly equal". Independently re-derived
from durable `logging.jsonl` by this audit: under the probe's own documented
policy (union of keys minus exclusions) the step-2 train rows compare **26**
fields, all equal, zero mismatches; the control and child key sets are
identical (no key present on only one side). **21** is exactly the count of
*numeric-scalar* compared fields. The claim is true and in fact stronger than
stated; only the figure is unlabeled. P3.

### Seams verified sound (no finding)

- **Forbidden-global test is genuinely digest-valid on the real writer
  path.** `tests/training/test_pack_cache.py:2164-2196` monkeypatches
  `pickle.dump` globally; the real writer
  (`src/training/pack_cache.py:539-546`) dumps through it into
  `chunk_path.open("wb")`, and `:551` computes `_file_sha256(chunk_path)`
  over the bytes actually written. The malicious chunk therefore carries a
  *valid* digest, and the failure is the restricted unpickler
  (`_RestrictedCacheUnpickler.find_class`, `:1200-1206`), not a digest
  mismatch. The test uses the real `write_micro_step_cache` (imported alias,
  `tests/training/test_pack_cache.py:38`).
- **Objective-row comparison is union-based, not intersection-based.**
  `reconcile_exact_resume_probe.py:2135` iterates
  `sorted(set(left) | set(right))`, so a key missing on one side yields a
  mismatch rather than being skipped. This is the classic fail-open pattern
  and it is *absent* here.
- **Signature verification precedes trust.**
  `reconcile_exact_resume_probe.py:286-303` (`_load_signed_receipt`) rejects
  unsigned receipts and payload-digest mismatches before any field is read.
- **`verify_artifacts` is fail-closed.** Missing required inputs return
  `status: "failed"` (`:2222-2231`, `:2279-2288`); commit drift raises
  (`:2199-2205`); `status` is constrained to `verified|failed` (`:2159`).
- **GPU-evidence join is strict.**
  `packet_executor.py:2363-2440` requires `world_size == 2`, exactly two
  launcher attestations, and `set(by_rank) == {0, 1}` (`:2403-2407`), with
  per-rank `rank`/`local_rank`/`logical_cuda_device` binding;
  `_validate_gpu_run_state:2442-2533` requires exact run status, completed
  steps, consumed packs, and checkpoint-event inventory, and enforces
  `completed_at is None` for `resumed_parent`. All mismatches raise; no
  default-swallowing branch found. Receipt digests are verified via
  `_verify_signed` before use.

### Observation (not a finding)

The probe's `_signed` is an **unkeyed** SHA-256 checksum: it provides
integrity/tamper-evidence, not authenticity. `tasks.md` 3.5 requires
rejecting "re-signed-tampered" receipts, which a digest alone cannot deliver
(a re-signer recomputes it). That property is instead supplied by
defence-in-depth: the executor independently binds expected digests from the
frozen command manifest, and `verify_artifacts` re-derives outcomes from
durable `run.json` / `logging.jsonl` / manifests. Adequate as built.

---

## VERDICT B — INTENT / CONTRACT: **PASS-WITH-DISPOSITIONS**

No P0, no P1. Every sampled requirement is backed by evidence that actually
exists and actually executed. Four findings (2× P2, 2× P3) are **citation /
traceability** defects inside `evidence-matrix.md` — the evidence exists but
is cited on the wrong row or under-cited. None invalidates a claim.

### B1. Spec→matrix mapping — structure verified, two rows under-cite

Structural mapping is exact: **12 requirements + 40 scenarios = 52 spec
items** across the four delta specs against **52 matrix rows**
(`evidence-matrix.md:132-183`), a verified 1:1 correspondence with zero
unmapped spec items, zero orphan rows, zero duplicates, and **zero
requirements lacking a scenario** (config-runtime 2/7,
pack-cache-semantic-identity 3/12, training-artifacts 2/9, training-resume
5/12 — matching the 6.3 claim exactly). All test nodes cited on the eight
sampled requirement chains exist on disk, and all cited receipts exist.

#### P2-2 — row 175 "Resume Admission Is Fail Closed" omits the owner of its inference-payload clause

`specs/coordexp-swift-training-resume/spec.md:94-98` requires that admission
authenticate, among other things, **"the sibling inference payload
identity"**. The matrix row (`evidence-matrix.md:175`) names only
`src/artifacts/training_state.py:admit_training_state` as source owner and
`tests/artifacts/test_training_state.py:test_admission_rejects_every_identity_mismatch_before_mutation`
as test owner. That test parametrizes over `REQUIRED_IDENTITY_KINDS`
(`src/artifacts/training_state.py:58-67`), which is exactly eight kinds —
`base_model, cache, dependencies, policy, resolved_config,
resume_compatibility, topology, trainable_surface` — **none** of which is an
inference-payload identity; `grep -c "inference" src/artifacts/training_state.py`
returns **0**.

Escalation was considered and **rejected on verification**: the clause *is*
implemented and *is* enforced, in a module the row does not name.
`src/artifacts/run_writer.py:admit_exact_resume_checkpoint_publication`
(`:1165`) calls `admit_inference_checkpoint_payload_identity` at `:1278` and
raises `ValueError("inference payload identity is not the committed
identity")` at `:1282-1283`. This audit confirmed the call is **unconditional**
on that path — the straight-line validation sequence at `:1242-1277` has no
`None`-guard, and `:1252` rejects any event without
`exact_training_state_enabled is True`. The function is exercised by
`tests/training/test_pipeline_exact_resume.py`.

The requirement's stated boundary — "**Before mutable state is restored**" —
was then verified explicitly, since the downgrade depends on it. The call
site is `src/training/pipeline.py:5735`, inside
`_read_only_admit_pipeline_exact_resume` (`:5640`), which also performs
`admit_training_state` (`:5675`). In the pipeline flow that whole read-only
admission runs at `:4118`, while the mutable restore
(`restore_distributed_exact_resume`) runs only afterwards at `:4148`, with
`:4163` cross-checking that the restored manifest digest equals the
read-only admission's digest. The clause is therefore satisfied **at exactly
the boundary the requirement names**. Conclusion: **no coverage gap and no
boundary gap; a traceability gap only.** Row 175 should also name
`run_writer.py:admit_exact_resume_checkpoint_publication`.

#### P2-3 — row 136's runtime clause is backed by Attempt-8 but cites only Wave 2

`specs/coordexp-swift-config-runtime/spec.md:35-43`, scenario "Exact state
publication is selected without continuation", has two THENs. The first
(resolved config preserves exact mode + null path) is exactly covered by
`tests/config/test_train_config.py:577`. The second — "**runtime MUST publish
exact training state without attempting restore** so the run can serve as an
uninterrupted control or interrupted parent" — has no cited node at the
`run_training_pipeline` level, although the row names
`src/training/pipeline.py:run_training_pipeline` as a live source owner. The
row's other test owner, `tests/training/test_wave7_exact_resume_config_bundle.py`,
is a module citation with no node and tests a config-bundle *author* script,
not pipeline publish-without-restore behavior.

Escalation was considered and **rejected on verification**: the clause was
executed on GPU. This audit read the frozen `-r8` role configs directly —
`uninterrupted_control.yaml` and `resumed_parent.yaml` both carry
`resume.mode: exact_same_world_size` with `resume.checkpoint_dir: null`
(the publish-only shape), and Attempt 8 ran both to completion, publishing
authenticated exact training state with no restore attempt. The evidence is
real; row 136 simply does not cite
`receipts/wave-3-attempt-8-outer-terminal-receipt.json` alongside its Wave-2
receipt. **Traceability gap, not an evidence gap.**

#### P3-5 — row 153 cross-cites the eager-eval resolver for a preparation-scoped scenario

`specs/coordexp-swift-pack-cache-semantic-identity/spec.md:113` scopes
"Preparation verifies payloads" to "the single-process preparation command",
but row 153's third owner
(`test_resolve_eval_pack_cache_hardcodes_payloads_verification_level`)
covers the eager-eval resolver. The direct preparation proof
(`assert train_verification_levels == ["payloads"]`,
`tests/training/test_pipeline_assembly.py:1465`) is cited on **row 145**
instead. Row 153 also omits `receipts/wave-4-cache-probe-receipt.json`,
though that real-CLI probe exercises the preparation entrypoint. Claim true,
citation misrouted.

#### P3-6 — row 151's third THEN is guarded only positively

The scenario's "the error MUST NOT claim that rerunning preparation can
overwrite it" is tested only as a positive token assertion
(`automatic_recovery=unavailable`,
`tests/training/test_pack_cache.py:2345-2356`); there is no negative
assertion. The two fixed message templates
(`src/training/pack_cache.py:580-586`, `:1882-1886`) contain no
rerun/overwrite language, so the contract holds today but is unguarded
against future message drift.

### B4. The 6.2 satisfaction argument is **sound**; no real verification gap

Both legs independently re-verified by this audit rather than accepted:

1. **Input identity — verified.** `git diff --stat f492f3687..HEAD -- src
   scripts configs` is **empty**. The full `f492f3687..HEAD` diff touches
   only `docs/`, `tests/`, `openspec/` (25 files). The verifier that executed
   at `f492f3687` is byte-identically the final state's verifier.
2. **Read-only re-authentication — independently reproduced**, not taken on
   faith from the implementer's script:
   - Inner terminal receipt
     (`outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/terminal-receipt.json`):
     payload digest **recomputed and valid**; `status: verified`;
     `commit: f492f36874f036ad4145a45c7b03cd2e0b2fd049`;
     `missing_inputs: []`; `bounded_mismatches: []`; all 5 required
     comparisons present.
   - Durable run states re-read from `run.json`: `uninterrupted_control`
     `completed`; `resumed_parent` `initialized` with `completed_at: None`;
     `resumed_child` `completed`. Exactly as claimed.
   - Step-2 objective equality re-derived from raw `logging.jsonl`:
     26/26 policy-compared fields equal, zero mismatches (see P3-4).

**Strengthening correction to the receipt's reasoning.** The Wave-6 receipt
attributes the impossibility of a re-run solely to the
`reconcile_probe.receipt_exists` guard. That guard is real
(`reconcile_exact_resume_probe.py:2177-2181`) but fires *late*, at terminal
receipt write. A re-run at the audited HEAD would in fact fail **earlier and
harder**: `verify_artifacts:2199-2205` compares `_current_commit()` against
the prepare receipt and raises `reconcile_probe.commit_drift`, because
`6ede480b0 != f492f3687`. The verifier is commit-bound by contract and
cannot execute at any state other than the one it was frozen at. The
receipt's conclusion is therefore correct and in fact over-defended; only its
stated mechanism is incomplete.

**Residual boundary (assessed, acceptable):** byte-identity covers tracked
code, not the Conda environment. This is acceptable because every matrix
claim remains pinned to `f492f3687`, the artifacts are digest-stable, and the
re-authentication above is environment-independent. Test files did change
after `f492f3687`, but additively; the 6.1 sweep covers the final state.

### B2. Claim boundaries are honest — confirmed

- All four Wave-3 matrix rows carry the identical explicit bound: "2-rank
  smoke-scale target-bound run at `f492f36874f` ... **not a general N-rank or
  production-scale claim**" (`evidence-matrix.md:171,173,178,180`).
- Inference-ignore scoped correctly: "payload-reader level ... still
  explicit-path consumption only, **NOT** coverage through
  `src/inference/pipeline.py` composition" (`evidence-matrix.md:170`).
- No efficiency claim anywhere. A word-boundary grep over the four delta
  specs for
  `production|N-rank|multi-node|throughput|speedup|efficiency|efficient|faster|performance|scalab|at scale|large-scale|wall-clock|GPU-hour`
  returns **zero matches**. (A looser substring grep yields six apparent hits
  that are all false positives from `scal` inside **"scaler"** — the
  mixed-precision scaler state — at `training-resume/spec.md:13,50,57,78,89`
  and `training-artifacts/spec.md:37`.) Across `proposal.md` and `design.md`
  such words appear only as **non-goals**: `design.md:42` ("No performance
  claim, throughput gate, speculative optimization") and `proposal.md:13`.
  The Wave-4 probe receipt self-declares `claim_boundary: "... no production
  cache published; no efficiency claim"`. All scale-bounding language lives
  in the matrix's claim-boundary column, stated as a *limit*.
- Wave-5 canonical docs stay bounded: `docs/ARTIFACTS.md` describes the
  sibling as "opt-in and disabled by default", supporting continuation "only
  from an optimizer-step save boundary at the same world size and rank map",
  with fail-closed admission. No scale or performance promotion.

### B3. `tasks.md` checkboxes are backed by named evidence — 10 sampled

| Box | Independent check | Result |
|---|---|---|
| 3.4 | Recomputed SHA-256 of **all 16** attempt-3..8 manifests/packets/reviews/markers/receipts against the digests recorded in 3.4 | **16/16 exact match** |
| 3.5 | Frozen representatives vs. signed receipt `comparison_policy` | Exact: rank-failure `expected_ranks [0,1]`, `serialized [0,1]`, `published [0]`, `error_code training_state.incomplete_rank_set`; interruption `serialized/published []`, `rank_state_boundary_reached false` |
| 3.6 | Attempt-8 frozen manifest inspected directly | `implementation_commit f492f3687…`; `-r8` target; base config SHA `44c2cd2a…`; per-role `effective_batch_size 2` + `resolved_grad_accum_steps 1`; `world_size 2`; `gpu_measurement_source torch_allocator_high_water`; three **distinct** role digests (`8dd5fbe1…`, `e5803d8d…`, `99b0671b…`) — **not** the Attempt-4/5 values that caused the Attempt-5 P0-1, confirming "no copied identity" |
| 3.6 (render) | Re-hashed the three on-disk `-r8` role configs | **All 3 MATCH** their frozen `expected_sha256` |
| 3.7 | Executor GPU-join code + outer/inner receipts | Verified (see Verdict A); outer receipt digest `c7581e8e…` matches 3.4 |
| 4.3 | `wave-4-cache-probe-receipt.json` | Arms `absent_target_built` / `valid_existing_target_hit` / `invalid_target_fail_closed`; `no_model_proof` key present; `production_cache_untouched: true`; private cache root; `claim_boundary` states model-free, no production cache, no efficiency claim |
| 4.4 / 4.5 | `wave-4-provenance-receipt.json` | `repository_state: dirty`; `no_environment_section: true`; recursive scan finds **no** `environment`/`env`/`environ` section anywhere; `resume_identity_projection_boundary` present |
| 5.2 | Node executed by this audit | Passes; mechanism analysed in P2-1 |
| 6.1 | `pytest --collect-only tests/config tests/artifacts tests/training tests/inference` | **2088 tests collected** — exactly the claimed 2088 |
| 6.3 | `openspec validate … --strict` re-run by this audit | **valid**, exit 0 |

Sampled test nodes executed by this audit (bytecode-safe,
`PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`): the two Wave-5 nodes and
three Wave-4 nodes — `test_inference_reader_ignores_real_committed_training_state_without_opening_it`,
`test_unknown_schema_value_is_rejected`,
`test_publication_fails_closed_on_hash_matching_forbidden_global_payload`,
`test_environment_secrets_never_enter_provenance`,
`test_resolve_eval_pack_cache_hardcodes_payloads_verification_level` →
**5 passed**.

Observation (not a finding): boxes 6.1-6.4 and 6.6 remain `[ ]` although
receipted. This is correct sequencing discipline — 6.5 is this audit, and
6.6 closes after it.

### B5. Known dispositions — each confirmed below P1

| Disposition | Confirmation |
|---|---|
| `ValueError` taxonomy (P3) | **Confirmed P3.** Fail-closed intact; digest-mismatch branch correctly distinct. See P3-1. |
| Transitive optimizer/scheduler ordering | **Confirmed accepted-by-ordering.** Backed by executed assertions, not narrative: `tests/training/test_pipeline_cache_preflight.py:415-416` collects `model_load_calls` / `accelerator_calls`, and the Wave-4 receipt records `model_load_calls == [False]`, `accelerator_calls == []`. |
| Cosmetic executor string | **Confirmed P3.** Constants correct. See P3-2. |
| Poisoned-`.pyc` incident (resolved) | **Confirmed resolved.** Audited 130 cached bytecode files under `src/` and `scripts/`: **0 stale or mismatched** (embedded source mtime/size vs. on-disk). Sampled tests additionally run with bytecode writing disabled. |

### Integrity checks required by the task — all confirmed

- **Attempts 1-7 receipts unmodified.** 16/16 digests match `tasks.md` 3.4
  (spot-checks demanded: attempt 3, 4, 5, 6, 7 all exact). `git log` shows
  exactly **one** commit per attempt receipt file (created once, never
  amended) — e.g. attempt-3 manifest `345fc55d6`, attempt-5 pre-cost review
  `8d013224f`, attempt-7 packet `f492f3687`.
- **No stable spec edited pre-sync.** `git log 4525a0f73..HEAD --
  openspec/specs/` is **empty**.
- **Archived changes cited only as history.** The four delta specs contain
  **zero** `archive/` references. `proposal.md:3` refers to the "superseded
  broad infrastructure change"; `docs/COORDEXP_SWIFT.md:225` says completed
  rebuild changes "are preserved under `openspec/changes/archive/`". Both are
  provenance framing, never authority.

---

## What this audit did NOT examine

- **Attempts 1 and 2 receipt contents** (only their existence and the
  narrative in `tasks.md`); 3.4 records no digests for them.
- **Full re-run of the 2088-node suite.** Node count was independently
  confirmed by collection; only the 5 sampled nodes were executed. The
  claimed pass/fail/skip counts for 6.1 are accepted from the Wave-6 receipt.
- **Any GPU/model execution.** The Attempt-8 GPU run itself is accepted from
  its signed receipts plus this audit's read-only re-derivation of the
  durable artifacts; it was not re-executed.
- **Non-sampled `src/` modules** — `training_state.py`, `run_writer.py`,
  `checkpoints.py`, `pipeline.py` internals were read only where a sampled
  seam reached them.
- **Non-sampled matrix rows.** The matrix has **52** rows. All were read and
  their structural 1:1 mapping to spec items was verified programmatically,
  but only **8 requirement chains** were traced end-to-end
  (spec text → row → node on disk → receipt), per the audit brief. The
  remaining rows' test nodes were not individually re-executed.
- **Structural spec→matrix enumeration was delegated** to a read-only
  sub-agent; its two most serious findings (P2-2, P2-3) were then
  **re-verified independently by this auditor** against source and against
  the `-r8` role configs before being classified, and both were down-graded
  from the sub-agent's stated severity on that evidence.
- **Wave-0/1/2 receipts** were read as context, not re-executed.
- **The archive sync itself** (the four delta specs merging into
  `openspec/specs/`), which happens after this audit.
- **`design.md` in full** — read for claim-boundary and non-goal language
  only.

---

## Bottom line

- **Standards / code quality: PASS-WITH-DISPOSITIONS** — no P0; one P2
  (`Path.open` interception is one-sided, with the Wave-5 receipt's
  "matches the reader's actual mechanism" sentence overstated); four P3s.
  The three previously-logged dispositions are correctly classified.
- **Intent / contract: PASS-WITH-DISPOSITIONS** — the assembled evidence
  supports the claims at their declared boundaries; the 6.2 argument is sound
  and was independently reproduced; claim boundaries are consistently and
  honestly bounded to 2-rank smoke scale at `f492f3687`; spec→matrix mapping
  is an exact 1:1 over 52 items; sampled `tasks.md` boxes are backed by named,
  verifiable evidence. Two P2 and two P3 findings are **citation/traceability**
  defects in `evidence-matrix.md`, each verified to have real underlying
  evidence in a module or receipt the row does not name.
- **No P0 or P1 finding. Nothing blocks archive.** Two candidate P1s were
  investigated and deliberately **down-graded to P2 after verification**:
  the "sibling inference payload identity" clause is enforced unconditionally
  at `run_writer.py:1278-1283`, and the publish-only runtime clause was
  actually executed by Attempt 8 with `checkpoint_dir: null` on both the
  control and parent roles.

### Recommended (non-blocking) follow-ups

1. Add `run_writer.py:admit_exact_resume_checkpoint_publication` as a source
   owner on matrix row 175 (P2-2).
2. Add `receipts/wave-3-attempt-8-outer-terminal-receipt.json` to matrix row
   136's Receipt column (P2-3).
3. Narrow the Wave-5 receipt sentence "the interception matches the reader's
   actual mechanism" to `checkpoint_payload.py`, or extend the interception
   to `safetensors.safe_open` (P2-1).
4. Label the "21 of 21" figure as the numeric-scalar subset, or restate as
   26/26 under the probe's own policy (P3-4).
5. Re-route row 153's preparation citation and add the cache-probe receipt
   (P3-5).
