## 1. Establish The Current Contract Evidence

- [x] 1.1 Record the exact implementation commit, worktree status, supported runtime, and relevant source/test owners for config resolution, checkpoint payloads, exact training state, run lineage, inference loading, cache admission, and execution provenance in a change-local evidence matrix.
- [x] 1.2 Map every requirement and scenario in the four delta specs to current source, current focused tests, an executable verification command, and live receipt status; label each item `accepted`, `gap`, or `remove`, and treat the superseded archive only as a risk index.
- [x] 1.3 Record the provider authority classification from the stable `coordexp-swift-packing-forward` requirement: only explicit `synchronous` and `overlapped` are supported; `legacy_fused` and `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` are unsupported live residue, MUST NOT enter these deltas, and are handed to the later decomposition change for deletion after this change's exact implementation commit.
- [x] 1.4 Run the existing config, checkpoint-payload, training-state, exact-resume, run-artifact, pack-cache, pipeline-assembly, provenance, and inference checkpoint-reader tests; preserve the exact command/results and classify every failure before changing production code.
- [x] 1.5 Scan stable specs and canonical docs for conflicts with the deltas, especially categorical no-resume claims, inference/exact-payload conflation, mutable-cache recovery, unsupported runtime promises, and any text that admits a provider other than `synchronous|overlapped`; narrow or remove any delta language that lacks live evidence.
- [x] 1.6 Gate Wave 1 with a read-only contract audit: no implementation may begin while a proposed requirement has no named owner, test/probe path, or bounded disposition, while provider authority is ambiguous, or while any P0/P1 scope finding remains unresolved.

## 2. Close Config, Payload, And Admission Gaps

- [x] 2.1 Add or complete the strict config matrix for omitted/disabled resume, disabled-with-path rejection, exact publish-only control/parent mode with a null path, YAML-relative continuation path resolution, unknown mode/field rejection, strict replay requirement, and launcher determinism failure before CUDA/model setup.
- [x] 2.2 Add a disabled-mode checkpoint test proving that normal inference-payload publication and aliases are unchanged and no `training_state/`, exact-state manifest, callback, identity, or exact publication event is written.
- [x] 2.3 Add an enabled-mode sibling test proving that the self-authenticating inference payload commits independently, exact state is typed under `training_state/`, selectors/events wait for phase two, and inference readers ignore the sibling while loading only explicit adapter/delta payloads.
- [x] 2.4 Add exact-admission tests for same-world-size and rank-map compatibility, optimizer-step save-boundary enforcement, restored next planned step/pack cursor/optimizer/scheduler/scaler/RNG state, and fail-closed rejection of world-size drift, unsupported accumulation position, identity mismatch, and inference-only payloads before mutable restore. These interface tests do not close exactness without the matched first post-resume forward/update probe in Wave 3.
- [x] 2.5 Preserve current source behavior: `ResumeConfig._disabled_mode_has_no_checkpoint` rejects only disabled mode with a path; exact mode with a null path is the required publish-only control/parent branch. Replace the over-broad RED with a positive resolved-config test and rerun config-bundle, input-attestation, and Wave-7 exact-resume consumer suites. No `src/` edit is authorized.
- [x] 2.6 Run the focused config, artifact, inference-reader, and exact-resume suites and publish a change-local Wave 2 receipt binding command, source state, test counts, failures, and claim boundary.
> Task 3 status (`be720f58d`, receipt `receipts/wave-2-contract-qualification.md`):
> 2.1 through 2.7 are backed by executed passing nodes. The initial
> exact-without-path RED was withdrawn because it contradicted the publish-only
> control/parent contract; it is replaced by the positive resolved-config test,
> and the affected config consumers pass `311` tests. The archive move at
> `71dab9772` intentionally changed `wave7_exact_resume_request.py` without
> refreshing its frozen identity, so only the sequence controller's exact frozen
> request-producer SHA-256 was updated to the observed current source hash; the
> strict identity check is unchanged. No `src/` edit was made.

- [x] 2.7 Gate Wave 2 with the focused executable config/payload/admission/historical-reader matrix; resolve every failure or zero-test selection before distributed qualification without adding another independent audit layer.

## 3. Qualify Distributed Atomic Publication

- [ ] 3.1 Add deterministic multi-rank control-plane coverage for successful exact-state publication with one complete, unique contribution from every expected rank and aliases/events committed only after the authenticated manifest.
- [ ] 3.2 Add injected one-rank failure plus missing, duplicate, malformed, and corrupt rank-contribution cases; prove all live ranks converge the same bounded failure without hanging and no selector or completed exact-state event references the step.
- [ ] 3.3 Add interruption tests at each externally visible boundary: before inference commit, after inference commit but before exact-state manifest commit, and after manifest staging but before authoritative event/alias commit; prove partial state is inadmissible and any surviving inference payload remains inference-only.
> Task 4 preparatory-tooling status (commit `test(training): add two-rank
> exact-resume qualifier`): the Wave-7 controller
> (`scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py`) is fixed to
> eight ranks (`payload["world_size"] != 8`) and its grammar cannot express the
> current `world_size=2` packet, so a bounded, independent CLI --
> `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`, covered by
> `tests/training/test_reconcile_exact_resume_probe.py` (45 focused nodes) --
> reuses `src.artifacts.training_state`, `src.artifacts.checkpoint_payload`,
> and `src.config.loader` to provide `prepare`/`success-control`/
> `success-resumed`/`rank-failure`/`interruption`/`verify` commands. `prepare`
> and `rank-failure`/`interruption` are real and model-free (no GPU/model
> import); `success-control`/`success-resumed` build the real
> `torch.distributed.run -m src.train` argv but execute through an injectable
> `launch` seam so unit tests never touch a GPU. `rank-failure` and
> `interruption` exercise real production admission code
> (`begin_training_state_contributions`, `publish_rank_training_state_contribution`,
> `commit_training_state_contributions`) and durably converge the same
> missing/duplicate/malformed/corrupt and three-boundary scenarios 3.2/3.3
> describe; they are offered as supporting evidence, not a substitute for
> dedicated coverage in `tests/training/test_exact_resume.py` if the task
> owner wants 3.1-3.3 checked directly against that file. This preparatory
> commit does not launch a model/GPU and does not check 3.1-3.7; it only
> equips the frozen command manifest that Task 4 Step 2 will bind by argv.
>
> Corrections (commits `fix(training): make resume qualifier fail closed` and
> `fix(training): make resume probe launch-compatible`):
> fixed-target review found the initial cut fail-open and non-fail-closed in
> three places, all resolved without launching a model/GPU: (1) `verify` now
> fails closed -- it admits both checkpoint boundaries via the real
> `admit_training_state`/`TrainingStateExpectations` primitives (a small,
> `world_size=2`-bounded comparator modeled on
> `wave7_exact_resume_compare.py`'s `_compare_decoded_rank`, not a copy of
> that 8-rank-fixed module) and requires exactly one durable `logging.jsonl`
> train row at step 2 on both branches, instead of comparing invented
> `run.json` fields that production never writes; any missing input or
> mismatch now yields `status: "failed"`, never `"verified"`. (2) `prepare`
> now runs the real, model-free `src.prepare_train_cache` entrypoint once
> (`load_model=False`) against a private `COORDEXP_SWIFT_PACK_CACHE_ROOT`,
> behind an injectable seam, instead of only reserving an absent cache root.
> (3) `success-control`/`success-resumed` now require `--commit` and check
> argument == prepare receipt == current `HEAD` immediately before launch,
> reverify each config file's hash/fingerprint, and verify every receipt's
> signature digest before trusting it. `_role_overrides` also now sets
> `uninterrupted_control`/`resumed_parent` to `resume.mode:
> exact_same_world_size` with a null path (the Task 2.5 publish-only
> contract) instead of `disabled`, which would have skipped exact-state
> publication entirely; eval forward is disabled in every role so the
> declared numeric forward ceiling (control=2/rank, resumed_parent=1/rank,
> resumed_child=1/rank) counts only train-split forwards. Parent and child
> intentionally share `training.max_steps: 2`, `checkpoint.steps: [1, 2]`,
> and `save_final: true`, so their resume-compatibility projections match;
> the parent supplies the step-1 comparison boundary and is then stopped by a
> bounded process-group interruption controller before step 2, while the child
> restores step 1 and executes only step 2. The controller authenticates the
> authoritative step-1 event and manifest before termination, re-authenticates
> unchanged one-step parent progress afterward, and records the controlled exit
> without treating it as normal parent completion.
> `rank-failure`/
> `interruption` no longer call `torch.cuda.current_device()` through
> `rng_snapshot=None`; they build CPU-only RNG state explicitly and are
> covered by a `CUDA_VISIBLE_DEVICES=''` subprocess test.
- [ ] 3.4 Keep attempt 3 frozen at implementation commit `037ab6683f9eeeb99157960f9fcf5bb3176a7044`, manifest SHA-256 `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`, and packet SHA-256 `70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02`; record its pre-launch `HOLD` without editing either frozen file. Close only its two P1s test-first: (a) in `reconcile_exact_resume_probe.py`, route only the resumed parent through a synchronous held-parent wrapper that calls the real pipeline checkpoint handler and then blocks after committed step 1 while control/child remain `src.train`; (b) add the separate experiment-local packet executor and schema-v2 `execution_contract`. Preserve production `src/`, strict admission, and resume-compatibility behavior.
- [ ] 3.5 Re-freeze the successor packet and schema-v2 manifest against the post-repair commit, absent target, exact configs/commands, and original `world_size=2`/two-GPU/three-arm numeric bounds. Run the independent pre-cost/distributed-qualification audit and resolve every P0/P1. If the exact frozen packet is `READY`, the lead-only executor proceeds under current goal authority without another user prompt; any mutation invalidates `READY` and requires re-freeze/re-review, not a repeated authorization prompt.
- [ ] 3.6 Execute only through `reconcile_exact_resume_packet_executor.py` after `READY`. It MUST claim the attempt marker with `O_EXCL`, run exactly once in order `setup`, `success.uninterrupted_control`, `success.resumed_child`, `rank_failure`, `interruption`, `verification`, stop on the first failure with no retry, and publish one signed outer receipt binding implementation/manifest/packet hashes, exact argv observations, launcher/runtime identity, world size, artifact trees, process/GPU/resource maxima, stop outcome, and the inner verifier receipt when reached. The success evidence MUST bind the matched branch pair and compare next input/pack identity; pre-forward trainable/optimizer/scheduler/scaler/RNG/cursor state; objective/loss fields; and resulting trainable parameters after the first post-resume optimizer update under the declared exact policy.
- [ ] 3.7 Gate Wave 3 only when the successor packet has a `READY` pre-cost review, the signed outer receipt proves exact six-command order and marker ownership, the matched success-pair and failure/interruption inner receipts are target-bound, every required rank and branch is accounted for, focused artifact verification passes, and the supported claim stays limited to same-world-size optimizer-step-boundary continuation; publication/admission alone is insufficient.

> **Wave 3 attempt 1 stopped before GPU (2026-08-13):** the freshly
> authorized frozen setup command exited `1` before cache publication because
> it set `CUDA_VISIBLE_DEVICES` to the empty string. The production strict
> single-rank launcher mapping accepts an absent variable but rejects an empty
> device entry. The artifact root, private cache, and cache receipt were all
> absent after cleanup; no later command, model, torchrun, GPU work, or retry
> ran. The immutable attempt-1 manifest, packet, and terminal receipt are under
> `receipts/wave-3-attempt-1-*`. A successor packet MUST use
> `/usr/bin/env -u CUDA_VISIBLE_DEVICES`, a new absent target, a new commit and
> manifest digest, and independent pre-cost `READY`; under the current goal the
> lead-only executor then proceeds without another authorization prompt.

> **Wave 3 attempt 2 stopped at child read-only admission (2026-08-13):**
> corrected CPU setup, two-step control, and two-step parent all completed
> within bounds. The child failed closed on both ranks before state apply or a
> forward with `run_writer.exact_resume_publication_invalid`: the parent had
> completed both step-1 and step-2 events, so step 1 was no longer the latest
> completed publication and the parent top-level progress was correctly bound
> to step 2. A direct durable admission check rejected parent step 1 and
> admitted parent step 2. No failure/interruption/verify command or retry ran;
> GPU memory returned to zero. Immutable evidence is under
> `receipts/wave-3-attempt-2-*`. The successor qualification tool MUST keep
> parent/child resume-compatible while stopping the parent after its step-1
> checkpoint becomes authoritative, then prove that interrupted parent record
> admits step 1 before launching the child. This requires a test-first bounded
> parent-interruption controller; changing admission to accept a stale event is
> forbidden.

> **Wave 3 attempt 3 held before launch (2026-08-13):** pre-cost review bound
> implementation commit `037ab6683f9eeeb99157960f9fcf5bb3176a7044`, manifest
> SHA-256 `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`,
> and packet SHA-256
> `70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02`
> and returned `HOLD` on two P1s: the parent step-1 observation-to-signal TOCTOU
> can allow step 2 to begin, and no outer executor guarantees an attempt-level
> signed receipt. No attempt-3 command, GPU/model work, cache preparation, or
> artifact-target mutation executed. The manifest and packet remain immutable;
> `receipts/wave-3-attempt-3-pre-cost-review.md` authorizes only the bounded
> test-first held-parent and packet-executor successor implementation.

## 4. Reconcile Cache And Provenance Dependencies

- [ ] 4.1 Trace the live cached-payload determinant registry, its independent completeness test, post-build determinant revalidation, immutable absent-target publication, and pre-model train/eval admission; remove any cache delta claim not supported by current source and focused tests.
- [ ] 4.2 Add or tighten focused tests only for demonstrated gaps in content/producer invalidation, immutable-current-target collision, full preparation validation, rank-required eager validation, and the guarantee that cache failure occurs before model/optimizer/runtime construction.
- [ ] 4.3 Execute a model-free cache preparation/admission probe that records fingerprints, manifest identities, verification levels, absent/valid/invalid target outcomes, and proof that no model weights were loaded; do not publish a production cache or make an efficiency claim.
- [ ] 4.4 Verify execution-provenance behavior for clean/dirty repository state and safely unavailable dependency identity, including stable non-secret local-state identity and explicit unavailable reasons; test that credentials and raw secret environment values cannot enter the artifact.
- [ ] 4.5 Publish a bounded provenance receipt from the current worktree/runtime and verify that exact-resume compatibility consumes only the declared identity projection rather than undocumented environment state.
- [ ] 4.6 Gate Wave 4 with focused executable cache-contract and provenance/privacy checks; remove unsupported normative language instead of adding another independent audit layer or inheriting unfinished archive scope.

## 5. Historical Reading And Canonical Documentation

- [ ] 5.1 Build a historical-reader fixture matrix covering a current committed exact checkpoint, current inference-only checkpoint, older adapter-plus-delta payload with extra metadata, incomplete current publication, and unknown resume-like historical files.
- [ ] 5.2 Prove inference accepts compatible explicit inference payloads without reading `training_state/`, while exact admission accepts only the current committed typed schema and never upgrades historical artifacts from names, extra files, or archived claims.
- [ ] 5.3 Update `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and `docs/ARTIFACTS.md` to describe the minimal inference payload and opt-in exact-state sibling separately, including disabled-mode behavior, supported boundary, failure semantics, and non-goals; preserve concurrent unrelated user edits.
- [ ] 5.4 Update any smaller canonical router or artifact inventory that still denies the accepted bounded capability, but keep schema detail in stable specs and keep receipts/historical explanation out of evergreen docs.
- [ ] 5.5 Run a repository-wide stable-vs-delta, docs-vs-source, and terminology conflict scan; resolve every live contradiction around `resume`, `training_state`, inference payloads, selectors/events, cache mutability, and exactness boundaries without rewriting historical evidence.
- [ ] 5.6 Gate Wave 5 with the focused executable documentation/authority conflict scan confirming that archived changes are provenance only and no canonical page promotes unfinished packing, efficiency, logging, loss, RL, or architecture work; do not add another independent audit layer.

## 6. Final Verification And Disposition

- [ ] 6.1 Run all focused suites named by the evidence matrix plus the relevant config, artifact, training, cache, inference, and distributed test directories through the repository runtime; record exact pass/fail/skip counts and investigate every unexpected skip.
- [ ] 6.2 Re-run only the already-reviewed frozen verifier against the target-bound matched success branch pair and one distributed failure/interruption path from the final implementation state; verify next input/pack identity, declared pre-forward state, first-update objective/loss fields and resulting trainable parameters, manifest/event/selector identity, and historical-reader outcomes from durable artifacts, not console text alone. If implementation or execution inputs changed, do not relaunch: re-freeze and obtain `READY` pre-cost review, after which the lead-only executor proceeds under current goal authority without another prompt.
- [ ] 6.3 Run `openspec validate reconcile-coordexp-swift-training-contracts --strict` and a stable-spec/delta merge-conflict scan; verify every modified requirement copies the full owning stable block and every new requirement has executable scenarios.
- [ ] 6.4 Run residue scans proving the change introduced no changed-order packing promotion, speculative efficiency claim, production cache campaign, logging enhancement, loss-objective/RL behavior, dependency upgrade, orchestration refactor, historical migration shim, cross-world-size resume, or mid-accumulation promise.
- [ ] 6.5 Obtain one independent final audit against the exact implementation commit and receipts, reporting both standards/code-quality and intent/contract verdicts; resolve all P0/P1 findings and record lower-priority dispositions explicitly.
- [ ] 6.6 Reconcile the evidence matrix to final source/tests/receipts, mark tasks complete only from executed evidence, and sync/archive only when every gate passes; otherwise keep the change active or archive it explicitly incomplete without changing stable specs or claiming exact-resume support.
