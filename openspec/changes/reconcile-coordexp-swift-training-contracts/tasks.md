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

- [x] 3.1 Add deterministic multi-rank control-plane coverage for successful exact-state publication with one complete, unique contribution from every expected rank and aliases/events committed only after the authenticated manifest.
- [x] 3.2 Add injected one-rank failure plus missing, duplicate, malformed, and corrupt rank-contribution cases; prove all live ranks converge the same bounded failure without hanging and no selector or completed exact-state event references the step.
- [x] 3.3 Add interruption tests at each externally visible boundary: before inference commit, after inference commit but before exact-state manifest commit, and after manifest staging but before authoritative event/alias commit; prove partial state is inadmissible and any surviving inference payload remains inference-only.
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
>
> Owner ruling (2026-08-19): 3.1-3.3 are closed by the combination of (a)
> the two-rank control-plane suite in `tests/training/test_exact_resume.py`
> (publication with control-only gathers committing once, pre-publication
> and rank-local failure convergence, Gloo pre-manifest aborts, restore,
> rollback, identity/topology rejection), (b) the 45-node probe suite
> exercising production admission with missing/duplicate/malformed/corrupt
> injections and the three interruption boundaries, and (c) the executed
> Attempt-8 arm receipts validated by the frozen verifier. Per the
> no-extra-audit-layer constraint, no duplicate dedicated layer is added.
- [x] 3.4 Keep all attempt 1-7 evidence immutable. Attempt 3 remains bound to implementation `037ab6683f9eeeb99157960f9fcf5bb3176a7044`, manifest `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`, and packet `70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02` with pre-launch `HOLD`. Attempt 4 remains bound to implementation `5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e`, manifest `b3538fb6167186cd5063f343a447ef1ed8f3024dc07f96284c7628c1ab08d3a9`, and packet `45afe6475e2aa3cae6e106bc446725de4b60197b77ba3d0a3a8ff45263242a87`: setup and control returned zero with confirmed cleanup, but the signed outer receipt stopped at `packet_executor.missing_gpu_rank` and ran no later command/retry because host NVML PIDs cannot be verifiably joined to the executor's container `/proc` namespace. Replace only that GPU evidence gate test-first: preserve the physical-index-to-UUID and occupancy preflight, make NVML process rows optional observations, and require target-bound receipt + `run.json` + canonical `logging.jsonl` GPU coverage. Preserve production `src/`, CPU measurement modes, strict admission, resume compatibility, the public `execute`/CLI surface, and the six-command order. Attempt 5 remains bound to implementation `9ef726085d2118dca90b373ccde4a88d068bf516`, manifest `ee9227fdb310f7e9a2d629000df137ac5dbc2b472461a83f729df56909d58c24`, packet `48b7bb924f960bb5f65a87dfa4be4b9f5289b79b9e02b5375b253dd7b4caf897`, target root `reconcile_exact_resume_2026-08-13-r5`, and a signed independent pre-cost `HOLD` at `receipts/wave-3-attempt-5-pre-cost-review.json`; it executed no command, marker, cache, model, `torchrun`, GPU allocation, or artifact-target mutation and MUST NOT be edited, re-signed, or reused. Its two P0s are (a) all three role-config `expected_sha256` values were copied forward from the Attempt-4 target although the rendered role configs embed `run.artifact_root` and the child's `resume.checkpoint_dir`, and (b) the executor bound the interrupted `resumed_parent` `run.json` to status `running`, which production `RunWriter` never writes because a `SIGTERM`-terminated held parent never reaches `finalize()`. P0 (b) was fixed test-first in `0f9ce01f3ee09c8862b34a34c72b719a86b1d48a` by binding the real controlled-exit status `initialized` with a null `completed_at`, keeping the exact progress, checkpoint-event, and topology checks and rejecting `completed`, `failed`, and `running`. Attempt 6 remains bound to implementation `1fbd9d68be4cf5dee1be1b09886f7c8a7b84205e`, manifest `4dff3f0bc81ca0e55cf614f3ef676430bddb1a236f24b36d97e948d5eafbe1ae`, packet `28e79a1d850a2308980e172089b4f03823703be0c004bb25e373527818afb0b2`, and signed independent `READY` review `8b23ae8838ad453ea5fb430f17e244759c67f382e67473810e2a0f680dd76a43` with payload digest `1f8705c595bc5771cb34e791fb6859483dd8ddbc1a344c169d61f8567d06c78f`. The review passed at its exact bound implementation, but concurrent tests-only commit `27abc8087e6c8591257463b19c8a83a63fd554e1` changed tracked HEAD before marker, command, target, cache, model, `torchrun`, GPU, or artifact mutation, invalidating the exact-commit/tracked-clean binding. Attempt 6 executed nothing and MUST NOT be launched, edited, re-signed, or reused. Attempt 7 remains bound to implementation `4525a0f73dedc1f31bb88c59ed18f95d22930531`, manifest `c04545163857f227db5652d19b8d6955538e10d2f62181ca2f136b1ced43990f`, packet `c1b592f2485f8d97a31835970387da825e1fb7635076b59b1421110551d03d0d`, and a signed independent pre-cost `HOLD` review at `receipts/wave-3-attempt-7-pre-cost-review.json` (file SHA-256 `74c843afa218d8da27cff9c9feda0b4e704b46667d34244402a5f06fbe26651e`): every packet-shape check passed, including the deterministic `-r7` role-config re-render with no copied identity, but concurrent docs-only commit `0b0f554e42421333a31497ebf766bee0d49c91b4` changed tracked HEAD during the independent review, so the reviewer signed `HOLD` (its receipt discloses and supersedes an earlier defective `READY` that had been written after the drift and read by no consumer) and Attempt 7 was retired unexecuted: no marker, command, cache preparation, model, `torchrun`, GPU allocation, or artifact-target mutation. Attempt 7 MUST NOT be launched, edited, re-signed, or reused. Attempt 8 executed the full contract on 2026-08-19: implementation `f492f36874f036ad4145a45c7b03cd2e0b2fd049`, manifest `22d10b3956888e0fc5aeb77f88332114750a4196e08a49d23d0445233309c0f6`, packet `0833860a79603e898a93d2565f928bb23e5f052e9033daa4b874e253b01aa2bf`, independent signed `READY` review (file SHA-256 `7a7e758c787b0016b767c31ffc607bbace241a5dcadcccbc3e36555fe458621d`, payload `6ec080d04fcbe58f9557af40b9ca97528079cfc59d929d9c55c9c2e1f574cf8c`), marker SHA-256 `8f3bb6dff645309235a0200cdfb1b72a5382cfacee9ebc87b9fc28f61231abae`, and signed outer terminal receipt SHA-256 `c7581e8e2ef2b93df73ee349be8c58b5e573bb9735d8be812b3d0a0eff3841b3` with `status: verified` after all six frozen commands returned zero within bounds; the signed inner verifier receipt is `verified` at the same commit. Attempt 8's receipts are likewise immutable.
- [x] 3.5 Preserve and verify the existing experiment-local rank-failure and interruption schema-v2 receipts and exhaustive semantic tests. Every rank-failure injection MUST record exact `expected_ranks: [0, 1]`, `serialized_ranks: [0, 1]`, injection-derived `published_ranks`, and its exact production `error_code`; the frozen representative missing rank 1 has published `[0]` and `error_code: training_state.incomplete_rank_set`. Every interruption boundary MUST record exact `expected_ranks: [0, 1]`, serialized/published ranks, and `rank_state_boundary_reached`; frozen `stop_after=1` has serialized/published `[]` and boundary false. The verifier MUST compare these exact fields and reject missing, stale-schema, re-signed-tampered, and digest-tampered receipts.
- [x] 3.6 Freeze successor Attempt 8 only after this planning/evidence commit, against the exact clean then-current `git rev-parse HEAD` and `git rev-parse HEAD^{tree}`, a new absent `outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8` target, and new `wave-3-attempt-8-{command-manifest.json,launch-packet.md,pre-cost-review.json,attempt-marker.json,outer-terminal-receipt.json}` paths; do not reuse Attempt 6's `-r6` or Attempt 7's `-r7` targets, files, reviews, or identities. Preserve the exact six-command order, `world_size=2`, two-GPU/three-arm numeric bounds, per-rank CPU coverage for both success commands, command-tree aggregate CPU coverage for the four model-free commands, `gpu_measurement_source: torch_allocator_high_water`, exact GPU ranks `[0, 1]` through signed success receipt + `run.json` + `logging.jsonl` joins, the physical-index-to-UUID map and occupancy preflight, and observational-only NVML process rows. Deterministically re-render all three role configs against `-r8`; bind their exact bytes/SHA-256, resolved fingerprints/digests, the absolute base-config path `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml` and SHA-256 `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`, with no copied digest from an earlier attempt. Load every role through production `load_train_config` and require `effective_batch_size: 2`; bind `world_size = 2` only from setup/two-rank launch argv and the manifest; invoke production `resolve_effective_batch_runtime(config, world_size=2)` and require `resolved_grad_accum_steps: 1`, yielding consumed progress step 1 -> 1 and step 2 -> 2. Retain resumed-parent `status: initialized` with null `completed_at`, all config/resource/accumulation/GPU/receipt semantics, and the experiment-local one-pack-per-step scope. Record exact new HEAD/tree/manifest/packet/config identities and obtain a new independent signed `READY` review bound to them; any drift requires a fresh successor freeze/review, not reuse or another prompt.
- [x] 3.7 Execute Attempt 8 only through `reconcile_exact_resume_packet_executor.py` after the new signed review validates the exact clean HEAD/tree/manifest/packet/config/target bindings. The lead-only executor keeps `O_EXCL`, exact six-command order, zero retries, one owned process group, bounded cleanup, the frozen CPU modes and artifact summaries, and stops at the first drift or failure. Before marker creation and again in setup validation, revalidate the bound base/role config bytes and SHA-256, `load_train_config` effective batch size `2`, argv/manifest `world_size = 2`, and production `resolved_grad_accum_steps: 1`. After each success command returns zero, cleanup is confirmed, and artifact summary completes, validate the signed receipt + `run.json` + `logging.jsonl` artifact-GPU join: control has completed world-size-2 step 2 with exact train steps 1 and 2; resumed merges authenticated parent step 1 in durable `initialized`/null-`completed_at` state with completed child step 2. Reject wrong receipt digests, state/topology/rows/steps/ranks/fields, non-finite/boolean/negative/non-integer-valued metrics, missing owned CPU/GPU evidence, or bound excess; fake NVML rows cannot satisfy artifacts. The signed outer receipt retains source labels, artifact paths/SHA-256, row/step inventory, raw allocated/reserved and conservative maxima, rank-to-local/logical/physical/UUID mapping, cleanup, and the bound inner receipt. Gate Wave 3 only when both receipts prove the full contract; exit zero alone is insufficient, and no automatic retry or repeated authorization prompt is permitted.

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

> **Wave 3 attempt 4 stopped at artifact-evidence reconciliation
> (2026-08-13):** implementation commit
> `5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e`, manifest SHA-256
> `b3538fb6167186cd5063f343a447ef1ed8f3024dc07f96284c7628c1ab08d3a9`,
> and packet SHA-256
> `45afe6475e2aa3cae6e106bc446725de4b60197b77ba3d0a3a8ff45263242a87`
> are immutable. Setup and uninterrupted control returned zero, their cleanup
> was confirmed, and the signed outer receipt then stopped fail-closed with
> `packet_executor.missing_gpu_rank`; no resumed, failure, interruption,
> verification, or retry command ran. NVML/`nvidia-smi` exposed host PIDs while
> the executor's `/proc` observations used a container PID namespace, with no
> verifiable mapping. The next successor therefore required a new absent target
> and re-freeze/re-review of an artifact-derived torch-allocator GPU contract; the
> existing blanket authority does not require another user prompt.

> **Wave 3 attempt 5 held before launch (2026-08-13):** implementation commit
> `9ef726085d2118dca90b373ccde4a88d068bf516`, manifest SHA-256
> `ee9227fdb310f7e9a2d629000df137ac5dbc2b472461a83f729df56909d58c24`,
> and packet SHA-256
> `48b7bb924f960bb5f65a87dfa4be4b9f5289b79b9e02b5375b253dd7b4caf897`
> are immutable, and the independent signed pre-cost review
> `receipts/wave-3-attempt-5-pre-cost-review.json`
> (reviewer `cc-opus5-xhigh-attempt5-20260813`) returned `HOLD` on two P0s. No
> attempt-5 command, marker, cache preparation, model, `torchrun`, GPU
> allocation, or artifact-target mutation executed. P0-1: the three
> `config_files` digests `08eae66e...`, `7e24e6bf...`, and `2daf76ee...` were
> Attempt-4 renders, but the role configs embed `run.artifact_root` (and the
> child's `resume.checkpoint_dir`), so setup would have failed closed at
> `packet_executor.setup_config_mismatch` after the marker, `-r5` target, and
> cache were already consumed. P0-2: the executor required the interrupted
> `resumed_parent` `run.json` to carry status `running`, which production
> `RunWriter` never writes; real interrupted parents under
> `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-3` and
> `-core-4` record `initialized` with `completed_at: null`. P0-2 is fixed
> test-first in the executor; P0-1 is closed only by a successor freeze.
> Attempt 6 subsequently froze correct `-r6` role-config identities and obtained
> an independent signed `READY`, but tests-only commit `27abc808...` changed
> tracked HEAD before marker or execution and invalidated that exact binding.
> Attempt 6 executed nothing and is immutable. Attempt 7 then froze correct
> fresh `-r7` identities and passed every packet-shape review check, but
> concurrent docs-only commit `0b0f554e4...` changed tracked HEAD during its
> independent review; the reviewer signed `HOLD` and Attempt 7 was retired
> unexecuted and is immutable. Attempt 8 MUST use a new absent `-r8` target and
> new paths, and MUST deterministically re-render and compare all three
> role-config byte streams, digests, and resolved fingerprints against that
> root before freezing; no identity may be copied forward. Attempts 1-7 and
> their receipts stay immutable.

> **Wave 3 attempt 8 verified (2026-08-19):** all six frozen commands
> returned zero within bounds at implementation `f492f3687...`. The control
> run completed world-size-2 training with exact train steps 1 and 2; the
> held parent durably recorded `status: initialized` with null
> `completed_at` after its authenticated step-1 publication; the child
> admitted parent step 1 and executed only train step 2. The frozen
> verifier admitted both checkpoint boundaries and signed `verified`; an
> independent spot-check of the step-2 train logging rows found 21 of 21
> compared objective fields exactly equal between control and child. The
> rank-failure arm recorded `converged_failure` with expected `[0, 1]`,
> serialized `[0, 1]`, published `[0]`, and
> `error_code: training_state.incomplete_rank_set`; the interruption arm
> recorded `converged_partial_state` with serialized/published `[]` and
> boundary false — both matching the 3.5 frozen representatives. GPU
> evidence is target-bound `torch_allocator_high_water` for ranks `[0, 1]`
> joined through signed receipts, `run.json`, and canonical
> `logging.jsonl`. Immutable evidence: `receipts/wave-3-attempt-8-*`;
> artifact root `outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8`.

## 4. Reconcile Cache And Provenance Dependencies

- [x] 4.1 Trace the live cached-payload determinant registry, its independent completeness test, post-build determinant revalidation, immutable absent-target publication, and pre-model train/eval admission; remove any cache delta claim not supported by current source and focused tests.
- [x] 4.2 Add or tighten focused tests only for demonstrated gaps in content/producer invalidation, immutable-current-target collision, full preparation validation, rank-required eager validation, and the guarantee that cache failure occurs before model/optimizer/runtime construction.
- [x] 4.3 Execute a model-free cache preparation/admission probe that records fingerprints, manifest identities, verification levels, absent/valid/invalid target outcomes, and proof that no model weights were loaded; do not publish a production cache or make an efficiency claim.
- [x] 4.4 Verify execution-provenance behavior for clean/dirty repository state and safely unavailable dependency identity, including stable non-secret local-state identity and explicit unavailable reasons; test that credentials and raw secret environment values cannot enter the artifact.
- [x] 4.5 Publish a bounded provenance receipt from the current worktree/runtime and verify that exact-resume compatibility consumes only the declared identity projection rather than undocumented environment state.
- [x] 4.6 Gate Wave 4 with focused executable cache-contract and provenance/privacy checks; remove unsupported normative language instead of adding another independent audit layer or inheriting unfinished archive scope.

> **Wave 4 closed (2026-08-19, receipt `receipts/wave-4-cache-provenance-receipt.md`):**
> the determinant-registry/admission/publication trace found no unsupported
> cache delta claim, so nothing was removed. Three demonstrated coverage gaps
> were closed test-first in commit `4e4c5f5d7` (eval preparation `payloads`
> level, hash-matching forbidden-global payload at publication, environment
> secrets excluded from provenance). The model-free three-arm probe receipt is
> `receipts/wave-4-cache-probe-receipt.json` (absent->built, valid->hit,
> invalid->fail-closed with untouched bytes); the bounded provenance receipt is
> `receipts/wave-4-provenance-receipt.json` with the exact-resume identity
> projection verified to consume only the declared resolved-config projection.
> Gate: 489 + 3 + 15 focused nodes green on clean bytecode. Dispositions
> logged for 6.5: pre-publish payload-validation error taxonomy (plain
> `ValueError` vs categorized `PackingCacheInvalidError`, P3) and the
> transitive optimizer/scheduler ordering argument. A poisoned-pyc incident
> during verification was diagnosed, purged, and the gate fully re-run.

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
