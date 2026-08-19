## 1. Wave 0 - Prerequisite and Compatibility Baseline

- [x] 1.1 Verify `reconcile-coordexp-swift-training-contracts` is fully verified, synchronized, archived, and committed. Record and checkout that exact predecessor SHA; confirm its authority record says stable `coordexp-swift-packing-forward` supports only `synchronous|overlapped` and classifies `legacy_fused` plus `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` as unsupported residue. Stop on an incomplete archive, moving branch tip, dirty predecessor ownership, or contrary stable contract.
- [x] 1.2 Freeze a change-local exact test-command manifest against that SHA. Give each command an ID and record full argv, cwd, `conda` environment, bounded environment selectors, node list/order, fixture roots, timeout, CPU-only/launch-bearing classification, and expected receipt path; later gates invoke IDs, and any revision requires a recorded baseline rerun.
- [x] 1.3 Add pre-move characterization tests for the compatibility-ledger surfaces: facade result mapping, ordered pipeline phases/collectives, cache preparation and admission receipts, old `SupervisedMicroStep` pickle bytes/module path and decoded values, causal logits selection, generic identity success/failure cases, completed-step rows, and complete representative RunWriter bytes.
- [x] 1.4 Add `tests/training/test_training_module_boundaries.py` with the intended one-way import graph and a production-import inventory that will fail while `src/prepare_train_cache.py`, `src/training/input_attestation.py`, or training assembly still depends on `src.qwen.parity`.
- [x] 1.5 Run the frozen baseline manifest against the untouched implementation and retain only expectations derived from stable specs or exact current bytes; do not normalize away an unexplained field, error, phase, collective, pickle module path, or byte difference.
- [x] 1.6 Gate Wave 0 with the frozen entry manifest, strict OpenSpec validation, residue checks, and separate read-only standards and intent-contract audits; do not start Wave 1 with an unresolved P0/P1.
- [x] 1.7 Commit the Wave-0 fixtures/manifest as one scoped baseline commit and prove reverting that commit returns exactly to the predecessor without touching caches or unrelated files.

> **Wave 0 closed (2026-08-19, baseline commit `2ee6c4959`):** predecessor
> pinned at `eb2dc97ab` (reconcile archive commit == decompose baseline,
> legitimately equal); 16 production-derived fixtures under
> `tests/fixtures/training_orchestration/`; characterization 272 passed with
> the intended 14-node RED boundary set; frozen test-command manifest with
> per-wave gates, declared flips (26), harness seam repoints (6), and
> per-command plan-line provenance. Both 1.6 read-only audits initially
> found six P1s (gate-scheduling: characterization module and
> `tests/runtime/test_rank_report_collective.py` unwired from wave gates,
> wave-3 assembly file missing; infra: `_reap` exception-path process leak,
> package-`__init__` relative-import resolution); all were fixed in one
> bundled correction round and the baseline was re-recorded. Revert proof:
> reverting `2ee6c4959` restores the predecessor tree exactly, caches
> untouched. Recorded dispositions (below P1, not fixed by design):
> TYPE_CHECKING imports not distinguished by the boundary scanner (loud
> failure mode), no SIGKILL escalation in `_reap`, no `_reap` unit seam,
> `_free_port` TOCTOU (inherited convention), `pipeline_result.json`
> initialized-training fields are declared-scope stub values, no pre-move
> ordered trace for the initialized-training region (CPU-infeasible;
> mitigated by Wave-5-authored tests plus the Wave-8 GPU smoke), and the
> 21-helper ownership inventory has no completeness guard (parity inventory
> does).

## 2. Wave 1 - Leaf Semantic and Identity Owners

- [x] 2.1 Add failing tests for `src.artifacts.identity` that require byte/digest/schema/bound/error equivalence with the characterized strict JSON, absent-target publication, base-model weight, repository, and source-owner identity behavior, including historical `src.qwen.parity` imports.
- [x] 2.2 Move the production identity implementation into `src/artifacts/identity.py`; switch `src/prepare_train_cache.py` and `src/training/input_attestation.py` to it, retain import-only re-exports from `src/qwen/parity.py`, and delete the moved duplicate implementation from parity.
- [x] 2.3 Add failing `TokenSequence.causal_logits_positions()` tests for sorted uniqueness, causal shift, empty atoms, and exact Qwen `logits_to_keep` inputs.
- [x] 2.4 Implement `TokenSequence.causal_logits_positions()`, migrate trainer and provider callers, and delete `_logits_positions_to_keep` plus the private cross-module import from `forward_input_provider.py`.
- [x] 2.5 Add failing canonical-owner tests for `src.training.micro_steps.SupervisedMicroStep`, requiring exact fields/order/annotations/defaults/frozen status and readable compatibility imports from `src.training` and `src.training.supervised_trainer`.
- [x] 2.6 Move `SupervisedMicroStep` and `supervised_micro_step_schema_identity()` into `src/training/micro_steps.py`; update current production imports and restricted cache allowlists while keeping import-only compatibility re-exports for historical readers.
- [x] 2.7 Gate Wave 1 with its frozen manifest IDs, import/residue checks, strict OpenSpec validation, exact old-byte load plus new canonical pickle-module assertions, and decoded-value equality; resolve every unexplained ledger difference, commit the wave as one scoped independently revertible unit, and prove its revert restores the Wave-0 commit without cache mutation.

> **Wave 1 closed (2026-08-19, commit `0583938cb`, parent `3279d14bc`):**
> gate exact (10 intended RED, 563 passed; zero unexpected fails/passes/
> skips); wave-0 baseline replay 10 RED / 276 passed with the four wave-1
> obligations cleared and no fixture byte modified; the three declared
> flips green under manifest-rule revisions; `ParityContractError` and all
> `qwen.parity.*` codes preserved through same-object re-exports (AST test
> forbids leftover bodies); dual-path restricted pickle allowlist proven
> against the frozen legacy chunk. Revert of `0583938cb` restores its
> parent byte-exactly. Disclosures: (a) docs-only commit `3279d14bc`
> (AGENTS.md calibration) interleaved between the wave-0 and wave-1
> commits — no owner-surface overlap, ordering noise only; (b) the D7 move
> broke the wave-7 probe frozen source pins
> (`wave7_exact_resume_request.py` owner pin, its own producer pin in the
> sequence controller, the parallel identity-owner and input-attestation
> pins, and one test path expectation) — refreshed to observed post-move
> hashes per the reconcile-era precedent at `71dab9772`, strict check
> mechanisms untouched, 237 nodes green; (c) three test files outside the
> plan's Task-2 list repointed under the manifest's harness_seam_repoints
> rule with fixtures replaying unchanged; (d)
> `causal_logits_positions()` keeps the deleted helper's
> `getattr(self, "atoms", None)` tolerance because the frozen
> `missing_atoms_attribute` characterization requires it — tasks.md/D6 fix
> only signature and return contract and are controlling over the plan's
> literal body.

## 3. Wave 2 - Model-Free Plan and Rank Control Plane

- [x] 3.1 Add failing tests for frozen `TrainingExecutionPlan` construction: exact resolved config/fingerprint, copied measurement context, launcher identity and entry evidence, plus assertions that plan construction performs no collective, filesystem publication, cache materialization, Accelerator construction, tokenizer/model load, or callback registration.
- [x] 3.2 Implement `src/training/execution_plan.py` with the exact design interface and migrate only facade-owned static resolution into it; reject a mutable service-locator or live model/runtime/cache field.
- [x] 3.3 Add failing `RankControlPlane` contract tests that replay the characterized single-/multi-rank success and failure reports, fixed frame limits, rank ordering, resource convergence, receipt sinks, pre/post-Accelerator identity binding, close idempotence, and ordered collective calls.
- [x] 3.4 Implement `src/training/control_plane.py` by moving the bounded rank-report transport, validation, normalization, phase convergence, and cleanup without changing phase names, report bytes, timeout/size bounds, error choice, or collective order.
- [x] 3.5 Switch `run_training_pipeline(...)` model-free preflight and post-Accelerator binding to `TrainingExecutionPlan` and `RankControlPlane`, keep initialized training in its current owner, then delete the replaced pipeline transport/validation helpers rather than forwarding through them.
- [x] 3.6 Gate Wave 2 with its frozen execution-plan, phase-convergence, cache-preflight, determinism, pipeline-entry, and mocked-transport manifest IDs plus import/residue and strict OpenSpec checks; commit one scoped independently revertible wave and prove revert restores Wave 1 without cache mutation.

> **Wave 2 closed (2026-08-19, commit `505a14b36`, parent `912a57219`):** `src/training/execution_plan.py` owns the frozen
> `TrainingExecutionPlan` and `build_training_execution_plan(...)` at design
> decision 2's exact interface; `src/training/control_plane.py` owns the moved
> rank-report transport, frame/header helpers, report validation and
> normalization, phase convergence, resource convergence, and cleanup behind
> `RankControlPlane.open/converge/bind_accelerator/close`.  Helper bodies were
> moved verbatim; `run_training_pipeline(...)` now builds the plan and drives
> the plane, initialized training stays in `pipeline.py` until wave 5, and every
> replaced pipeline helper is deleted with no forwarding wrapper.
>
> Evidence: wave-2 gate = exactly its 8 expected RED nodes, 253 passed, 0 skips;
> all 10 declared flips at or below wave 2 green under manifest-rule revisions;
> wave-0 baseline replay and wave-1 gate replay each fail exactly that same
> 8-node wave-2 subset (278 / 565 passed); `phase_order.json`,
> `collective_order.json`, `pipeline_result.json`, and `cache_admission.json`
> replay byte-unchanged through the repointed two-rank gloo characterization;
> `git status -- tests/fixtures/` empty; `openspec validate ... --strict` valid;
> `ruff check` clean on every touched file.
>
> Disclosures: (a) moved helpers keep their private Wave-0 names in the new
> owner modules, so the harness repoints are pure module swaps -- tasks.md 3.4
> ("moving ... without changing") is controlling over the plan document's
> optional de-underscoring suggestion; (a2) `RankControlPlane.bind_accelerator`
> replaces the transport only -- the Accelerate rank/world-size identity check
> stays inside the facade's converged body, because the characterized two-rank
> contract requires a mismatch to converge as
> `runtime.distributed_phase_failed` on every live rank rather than raise
> locally on the mismatching rank; (b) the cache-preflight rank-diagnostic
> projection/normalization/reconstruction helpers and their bounded constants
> moved with `_run_rank_converged_phase`, which needs them and may not import
> the facade -- they carry no declared flip and no test referenced them;
> (c) three harness seams were repointed under the manifest's
> `harness_seam_repoints` rule with every expected value unchanged:
> `tests/runtime/test_rank_report_collective.py` imports,
> `tests/training/test_orchestration_compatibility.py` phase/collective spies,
> and `tests/training/test_pipeline_phase_convergence.py` /
> `test_pipeline_cache_preflight.py` / `test_pipeline_assembly.py` patch
> targets; the frozen `collective_order.json` transport provenance string is
> deliberately left naming the pre-move owner; (d) `load_train_config` patch
> sites split by entry point -- `run_training_pipeline` callers repoint to
> `execution_plan`, `prepare_training_pack_caches` callers keep patching
> `pipeline` until wave 3 moves that owner; (e)
> `scripts/probes/coordexp_swift/smoke_rank_report_collective.py` still imports
> `_build_rank_report_gatherer` from `src.training.pipeline` and is now broken;
> it is outside this agent's write scope and is referenced by no test, doc, or
> other script.  Task 3.6's commit and revert proof are unexecuted: this wave
> was produced under an explicit no-commit instruction.

> Close-out: the lead independently re-ran the wave-2 gate (8 failed /
> 253 passed, failure set byte-equal to the expected wave-2 RED nodes,
> zero skips) and proved revertibility -- a staged revert of `505a14b36`
> is zero lines different from `912a57219`. Additional disclosure: the
> orphan `scripts/probes/coordexp_swift/smoke_rank_report_collective.py`
> (referenced by no test, doc, or script; no frozen hash pin) had its one
> import repointed to the control-plane owner in the wave commit.
> Convention note: the manifest's `expected_receipt_path` entries remain
> declarative; per the wave-1 precedent, gate records live in these
> tasks.md notes, and revising the manifest just to register receipt
> files would trigger its revision rule for no evidentiary gain.

## 4. Wave 3 - Cache Contract and Workflow

- [x] 4.1 Add failing tests for `micro_step_runtime_config_identity(config)` and determinant-owner mutation. Require the final determinant diff to contain at least the four declared source changes—`supervision_tokens`, `micro_step_runtime_config`, `micro_step_schema`, and `cache_serializer`—while reporting/session/facade edits add no determinant; any missing declared change or unexplained additional change blocks the wave.
- [x] 4.2 Implement `src/training/cache_contract.py`; update `PACKING_CACHE_DETERMINANT_OWNERS` to bind `micro_step_runtime_config` and `micro_step_schema` to the two narrow owners and prove the semantic content projections remain exact.
- [x] 4.3 Add failing cache-workflow and CLI tests for one-process preparation, worker resolution, train/eval split aggregation, absent-target publication, model-free fingerprint/admission, rank-local hydration, image-processor attachment, and bounded actionable failures. The tests MUST require `src.prepare_train_cache --require-all-hit` to validate existing train/eval targets, succeed on two valid hits, and fail on any miss/invalid target before any render/tokenize/pack/build/temporary-publication/immutable-publication call.
- [x] 4.4 Move `prepare_training_pack_caches(...)`, multi-worker render/tokenize/pack orchestration, and its receipt projection into `src/training/cache_workflow.py`; keep the ordinary CLI behavior and `src.training.pipeline` compatibility re-export exact. Add `--require-all-hit` as an in-workflow fail-before-build mode with no fallback construction path; it may only fingerprint, admit, validate both existing targets, and publish its named verification receipt.
- [x] 4.5 Move model-free fingerprint/admission and rank/eval hydration orchestration into `cache_workflow.py` using only the bounded `CachePreflight`/`HydratedTrainingInputs` records from the design; keep low-level identity, serialization, immutable publication, safe load, and manifest validation in `pack_cache.py`.
- [x] 4.6 Switch pipeline callers to the cache workflow, delete replaced cache helpers and worker globals from `pipeline.py`, and enforce that `cache_workflow.py` imports neither `session.py` nor model-loading with `load_model=True`.
- [x] 4.7 Prove determinant semantic projections match the baseline except for the four declared source entries and resulting aggregate/fingerprint. Prove old immutable pickle bytes still restricted-load, new serialization names `src.training.micro_steps`, decoded micro-steps are equal, and no other protected bytes differ; do not invoke production cache materialization in this wave.
- [x] 4.8 Gate Wave 3 with its frozen cache/preflight/rebuild/CLI/input-attestation/packing manifest IDs, import/residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 2 without publishing or mutating a cache.

> **Wave 3 closed (2026-08-19, commit `ca669017c`, parent `d44b691b5`;
> per an explicit no-commit instruction):** `src/training/cache_contract.py`
> owns `micro_step_runtime_config_identity(config)` at design decision 4's exact
> three-field projection and imports neither the facade, the workflow, nor
> `pack_cache.py`. `PACKING_CACHE_DETERMINANT_OWNERS` rebinds
> `micro_step_runtime_config -> src/training/cache_contract.py` and
> `micro_step_schema -> src/training/micro_steps.py`; every other owner entry
> and `registry_schema_version` are unchanged.
> `src/training/cache_workflow.py` owns `prepare_training_pack_caches(...)`,
> the absent-target build orchestration, the multi-worker
> render/tokenize/pack materialization, split aggregation, model-free
> fingerprint/admission, rank/eval hydration, image-processor attachment, and
> the bounded actionable failures; `pack_cache.py` keeps determinant
> construction, fingerprinting, immutable publication, restricted
> deserialization, manifest validation, and payload loading.
> `src.prepare_train_cache --require-all-hit` is an in-workflow
> fail-before-build mode with its own verification receipt schema.
>
> Determinant diff (4.1/4.7 proof, `build_packing_cache_determinants` over
> `tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml` with a fixed
> scratch root, before vs after this wave): the semantic payload is
> byte-identical for all 31 determinants (empty diff, including both rebound
> entries); exactly three registry entries changed —
> `micro_step_runtime_config` and `micro_step_schema` (`owner` +
> `owner_source_identity` only, `content_identity` equal) and
> `cache_serializer` (`owner_source_identity` only) — plus `code_identity` for
> those same three and the aggregate fingerprint. The declared fourth source,
> `supervision_tokens`, already turned over in Wave 1
> (`src/supervision/tokens.py` is unmodified by this wave); measured against
> the Wave-0 baseline `eb2dc97ab`, the determinant-owner source files that
> differ are exactly `src/supervision/tokens.py`, `src/training/pack_cache.py`
> and the two rebound owners' old and new paths — no fifth owner. A dedicated
> node also proves that synthesizing new hashes for both narrow owners changes
> no `content_identity`.
>
> Evidence: wave-3 gate (manifest argv verbatim) = exactly its 6 expected RED
> nodes, 556 passed, zero skips, zero unexpected passes; wave-0 baseline
> replay 6 RED / 280 passed, wave-1 gate replay 6 RED / 567 passed, wave-2
> gate replay 6 RED / 255 passed — each failing exactly the same wave-3
> 6-node subset; all 20 declared flips at or below wave 3 green under
> manifest-rule revisions; `git status -- tests/fixtures/` empty and the
> fixture tree hash unchanged at `2fe137ccf`; `.cache/coordexp_swift/packing`
> byte-for-byte unchanged (snapshot diff before/after the whole wave, empty);
> `openspec validate ... --strict` valid; `ruff check` clean on every touched
> file.
>
> Disclosures: (a) the one sanctioned compatibility surface is implemented as
> a delegating `def pipeline.prepare_training_pack_caches(config_path, *,
> require_all_hit=False)`, not the plan document's
> `from src.training.cache_workflow import ...` re-export, because two
> undeclared Wave-0 nodes
> (`test_prepare_training_pack_caches_remains_importable_from_the_facade`,
> `test_wave0_facade_signature_and_compatibility_reexport_are_frozen`) freeze
> `__module__ == "src.training.pipeline"`; an import re-export would break two
> preserved surfaces, and tasks.md 4.4 ("keep the ... compatibility re-export
> exact") is controlling over the plan snippet. (b) Every other moved name is
> referenced from `pipeline.py` module-qualified as `cache_workflow.<name>`,
> never re-imported, so each symbol keeps exactly one patch point; this drags
> non-cache helpers the moved code needs (`_utc_now`, `_file_sha256`,
> `_sha256_json`, `_environment_selector_source` and the receipt env-selector
> constants, `TRAIN_SPLIT`, `_launcher_device_mapping`,
> `_establish_converged_runtime_determinism`,
> `_runtime_determinism_run_policy`, `_resolve_eval_reduction_receipt`,
> `_resolve_converged_eval_reduction_receipt`, `_resolve_pack_cache_root`,
> `_packing_policy_receipt`) into the workflow owner, because
> `cache_workflow.py` may not import the facade and duplicating them would
> create a second owner. This mirrors Wave-2 disclosure (b). (c)
> `HydratedTrainingInputs` is deliberately NOT introduced this wave: design
> decision 5 permits the frozen records "only for the exact bundles currently
> returned as untyped mappings", and no current bundle carries its four fields
> without loss (train and eval hydration happen in different phases and
> `total_ordinal_count` has no field). Its natural consumer is task 6.1's
> "cache hydration inputs" session surface, so the record should land with the
> wave that creates it; changing its fields first requires a design update,
> which is user-owned. `CachePreflight` IS implemented and used, and its five
> plan-declared fields fit the `--require-all-hit` bundle exactly. (d) The
> Wave-6 pack-plan probe pinned pipeline ownership of the production encoder
> seam. `scripts/probes/coordexp_swift/wave6_pack_plan_comparison.py` and
> `tests/packing/test_wave6_pack_plan_probe.py` were repointed to
> `src.training.cache_workflow` under the manifest's `harness_seam_repoints`
> rule: an import-only re-export would have left the probe's fixture
> `_render_and_encode_example` monkeypatch inert (Python resolves the moved
> function's globals in its new module), silently running the real encoder.
> The recorded `production_encoder_seam` string and the
> `pipeline_symbols[...]["path"]` assertion now name the new owner; the probe
> has no frozen self-hash pin (`_source_owners`/quiescence are computed live
> per run) and the published `outputs/probes/.../2026-08-10-r2/plan.json`
> artifact is historical evidence replayed by no test. `tests/packing` is
> green (34 probe nodes) after the repoint. (e) Two parametrized node IDs in
> `test_pack_cache_determinant_registry.py` are renamed as a pure consequence
> of the declared rebinding —
> `test_every_unique_declared_owner_source_edit_changes_fingerprint[pipeline]`
> and `[supervised_trainer]` become `[cache_contract]` and `[micro_steps]` —
> and `test_pipeline_micro_step_constructor_owner_is_explicit_and_source_bound`
> is renamed to `test_micro_step_runtime_config_owner_is_explicit_and_source_bound`;
> the obligation count is unchanged (one node per unique owner path) and all
> were passing before and after. (f) Test files outside the plan's Task-4 list
> were repointed under the same seam rule:
> `test_orchestration_compatibility.py`, `test_pipeline_assembly.py`,
> `test_pipeline_cache_preflight.py`, `test_pipeline_pack_cache_rebuild.py`,
> `test_pipeline_phase_convergence.py`, and
> `test_pack_cache_runtime_constructor.py` (the last two are wave-8-matrix
> files; Wave-2 disclosure (c) is the precedent). Names that BOTH owners still
> import (`load_qwen_components`, `build_token_vocabulary_groups`,
> `resolve_qwen_runtime_controls`, `resolve_planned_step_schedule`,
> `load_rank_micro_steps_from_cache`, `collect_execution_provenance`,
> `require_pinned_runtime_baseline`) are now replaced on both modules through
> a local `_patch_shared_cache_import` helper (or chained assignment in the
> gloo worker bodies) so no seam silently reaches production through the other
> owner; no expected value was changed anywhere. (g) `--require-all-hit` is
> fail-closed when the config declares no evaluation split: it raises
> `training.pack_cache_verification_split_undeclared` before admitting the
> train target rather than verifying one split and reporting success. Nothing
> in the design, tasks, plan, or manifest pins this case; the choice is
> derived from 4.3's "succeed on two valid hits", D5's "validate both existing
> targets", and D13's two-target cache packet, and it keeps `CachePreflight`'s
> non-optional `eval_*` fields honest. It is a user-reviewable semantic choice,
> not a discovered fact. (h) Consumers of `pipeline.py` outside the wave-3
> gate argv were executed once to prove the 1,995-line trim broke no unlisted
> importer: `test_pipeline_exact_resume.py`,
> `test_checkpoint_handler_identity.py`, `tests/eval/test_forward_eval.py`,
> `test_reconcile_exact_resume_probe.py`, and
> `tests/losses/test_wave3_zero_weight_probe_contract.py` — 267 passed. The
> remaining pipeline-referencing files were AST-checked for stale moved
> symbols (`wave2_packed_parity.py`, `wave3_zero_weight_gpu.py`,
> `wave5_provider_benchmark.py`, `wave6_pack_plan_comparison.py`,
> `src/train.py`, `tests/helpers/training_architecture_fixture_builder.py` —
> all clean). `tests/test_objective_profile_resolution.py` and
> `tests/test_training_pipeline_registry.py` still fail collection on
> `src.training.pipeline_registry` and `ConfigLoader`, both absent at the
> wave-0 baseline and untouched by this wave: pre-existing, not a regression.
> (i) Task 4.8's commit and revert proof are unexecuted: this wave was
> produced under an explicit no-commit instruction.

> Close-out: the lead independently re-ran the wave-3 gate (6 FAILED, all
> in the boundary file, byte-equal to the expected wave-3 RED set, zero
> skips) and proved revertibility -- a staged revert of `ca669017c` is
> zero lines different from `d44b691b5`.

## 5. Wave 4 - Existing-Schema Reporting and RunWriter Internals

- [x] 5.1 Add failing `CompletedStepReporter` tests that exact-compare lifecycle mutation, loss/LR/timing/resource extraction, reduction requests, accuracy validation, rank-resource accounting, every-completed-step row bytes, append outcome broadcast, warmup accounting, and first-step phase behavior with the baseline handler.
- [x] 5.2 Implement `src/training/reporting.py`, switch the initialized trainer callback to `CompletedStepReporter`, and delete `_train_logging_handler` plus its moved pipeline-only helpers without adding cadence, sinks, fields, ETA, TensorBoard, or a metric registry.
- [x] 5.3 Expand pre-move RunWriter fixtures to cover initialization, strict `logging.jsonl`, policy/materialization/schedule binding, phase success/failure/not-run, checkpoint publication, best/final files, continuation, warning bounds, failed finalization, and successful finalization as exact bytes and errors.
- [x] 5.4 Move pure normalization/serialization and bounded field validation into `src/artifacts/run_schema.py`; leave every read, append, fsync, link/replace, collision, and atomic-write operation in the `RunWriter` facade.
- [x] 5.5 Move pure run-state transitions and exact-resume checkpoint-publication admission into `src/artifacts/run_state.py`; keep `RunWriter` and `admit_exact_resume_checkpoint_publication` import paths, signatures, return values, and I/O sequencing unchanged.
- [x] 5.6 Delete replaced helpers from `run_writer.py`, prohibit a new store/repository interface, and exact-compare the full fixture tree and failure outcomes against Wave 0.
- [x] 5.7 Gate Wave 4 with its frozen reporting/RunWriter/checkpoint/resource/training-state/logging manifest IDs, artifact-byte diff, import/residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 3 without cache mutation.

> **Wave 4 closed (2026-08-19, commit `4986836b0`, parent `5e4e3e4bb`):**
> `CompletedStepReporter` plus the step-row helpers own reporting in
> `src/training/reporting.py`; pure normalization/validation and pure
> run-state/admission logic split into `src/artifacts/run_schema.py` and
> `run_state.py` with all I/O and the single mockable payload-identity
> binding retained in `run_writer.py`. Central proof: the frozen
> `run_writer/` byte tree, `completed_step_rows.json`, and the full
> two-rank pipeline characterization replay byte-unchanged; gate exactly
> its 3 expected RED nodes with 412 passed and zero skips; wave-0
> baseline replay 3/282; frozen exact-resume verifiers 115 green; lead
> independently re-ran the gate and proved a staged revert of
> `4986836b0` is zero lines different from `5e4e3e4bb`. Disclosures:
> (a) a first-draft split created a second import binding of
> `admit_inference_checkpoint_payload_identity` and broke two exact-resume
> monkeypatch tests -- restructured into two pure `run_state` functions
> bookending the one `run_writer` I/O call, which also made `run_state.py`
> genuinely I/O-free; (b) `_train_logging_handler`'s
> `WAVE0_PIPELINE_OWNED_HELPERS` entry was deleted rather than flipped
> because design decision 9 replaces the factory with the differently
> named `CompletedStepReporter` class -- a collected-node-set diff proves
> exactly that one parametrized node was removed and nothing else; (c)
> `reporting.py` carries a private byte-faithful duplicate of the
> first-optimizer-step phase-finish (importing the facade back would be a
> forbidden reverse edge), proven by the unchanged pipeline
> characterization; (d) provenance blind spot recorded for the Wave-6
> residue sweep: `wave2_packed_parity.py` SOURCE_OWNERS and
> `wave5_provider_benchmark.py` EXECUTION_OWNER_PATHS fingerprint
> `pipeline.py`/`run_writer.py` as complete behavior owners but do not
> yet list `reporting.py`/`run_schema.py`/`run_state.py`.

## 6. Wave 5 - Training Session and Thin Facade

- [x] 6.1 Add failing `TrainingSession` tests for the fixed phase order, owned lifecycle/resource close behavior, model/runtime assembly boundary, cache hydration inputs, trainer handler wiring, primary-exception preservation, best-effort failure publication, success finalization, exact facade result mapping, and idempotent close.
- [x] 6.2 Implement the `TrainingSession` constructor and move the model/adapter/selected-token/loss/optimizer/runtime assembly plus lifecycle state into `src/training/session.py` without introducing phase subclasses, registries, or alternate backends.
- [x] 6.3 Move exact-resume admission/restoration/publication wiring, eval/checkpoint/final handlers, forward-input-provider lifetime, profile-sync reset, and success/failure finalization into the session while preserving their literal collective and event order.
- [x] 6.4 Reduce `run_training_pipeline(...)` to build the immutable plan, open/bind the control plane, initialize the current run owner, admit the cache workflow, construct exactly one session, return its result, and close resources in `finally`.
- [x] 6.5 Delete all replaced initialized-training, handler, lifecycle, cache, control-plane, and identity code from `pipeline.py`; retain only the public facade and documented compatibility re-export, then enforce the one-way import graph and absence of pass-through helper layers.
- [x] 6.6 Gate Wave 5 with its frozen pipeline/preflight/convergence/trainer/runtime/eval/checkpoint/resume/cache/entrypoint manifest IDs, protected row/artifact/ordered-call comparisons, residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 4 without cache mutation.

> **Wave 5 closed (2026-08-19, parent `8239be128`; produced under an explicit
> no-commit instruction):** `src/training/session.py` owns the model-bearing
> lifetime -- `TrainingSession.__init__/run/fail/close`, `RunIdentity`,
> `_run_initialized_training`, `_checkpoint_handler`, `_eval_forward_handler`,
> `_final_handler`, `_build_accelerator`, the phase-lifecycle helpers, the
> pre-model policy admission, exact-resume admission/restoration/publication,
> the forward-input-provider lifetime, and the profile-sync reset.
> `src/training/pipeline.py` is 156 lines: `run_training_pipeline(...)` plus the
> documented `prepare_training_pack_caches` compatibility surface, and nothing
> else (an AST node asserts the module defines exactly those two names and
> carries no re-export assignment). The whole import-boundary suite is GREEN for
> the first time (20 passed): `src/training/session.py` exists, and the wave-5
> parity edge cleared by repointing `base_model_weight_identity` to the Wave-1
> owner `src/artifacts/identity.py`, so **no** production module imports
> `src.qwen.parity` any more.
>
> Evidence: wave-5 gate (manifest argv verbatim, `expected_red_nodes: []`) =
> **696 passed, 0 failed, 0 skipped**; wave-0 baseline replay 285 passed / 0
> failed (all 14 original RED nodes converged); wave-2 gate replay 260 passed;
> wave-3 gate replay 561 passed; wave-4 gate replay 415 passed; frozen
> exact-resume verifiers (`tests/artifacts/test_training_state.py`,
> `tests/training/test_exact_resume.py`,
> `tests/training/test_pipeline_exact_resume.py`) 144 passed; 23 of the 24
> `declared_flips_in_scope` node IDs run green as one invocation and the 24th
> (`[_train_logging_handler]`) is legitimately absent under wave-4 disclosure
> (b)'s recorded revision. The wave-1 gate replay is the sole red one -- 199
> failed / 374 passed -- and its failure set is exactly
> `tests/qwen/test_packed_parity.py`'s 199 failures (that file alone reports
> 199 failed / 88 passed), i.e. entirely the disclosure (g) script residue and
> nothing in this wave's owner surfaces. The frozen two-rank characterization
> replays `pipeline_result.json`, `phase_order.json`, `collective_order.json`,
> and `cache_admission.json` byte-unchanged; `git status -- tests/fixtures/`
> empty; `openspec validate ... --strict` valid; `ruff check` clean on every
> touched file; Serena `get_diagnostics_for_file` reports zero findings on
> `pipeline.py` and on `tests/training/test_training_session.py`, and the eight
> Pyright findings on `session.py` all sit inside verbatim-moved bodies
> (`_initialize_artifact_owner`, `_run_initialized_training`,
> `_checkpoint_handler`, `_eval_forward_handler`,
> `_resolve_shared_run_directory`) and are pre-existing. A `difflib` proof over
> the moved region reports **3007 of 3023 moved lines verbatim with zero
> deletions** -- the only changed lines are
> `_initialize_model_free_run_owner`'s return annotation and return statement,
> which now build `RunIdentity`.
>
> Disclosures:
> (a) `TrainingSession.__init__` takes design decision 8's six keywords plus a
> seventh, `lifecycle`. D8's prose says "the session owns the current lifecycle
> counters", but the counters must exist before the session: the facade's
> pre-model `config_provenance_resolution` and `cache_admission` phases already
> record into them, and D8 places session construction *after* the cache
> preflight. Threading the mutable mapping through the frozen `RunIdentity` or
> through `admitted_policies` would have hidden mutable state inside a record
> whose whole point is that it is frozen, so the parameter is explicit.
> (b) `cache_preflight` is annotated `Mapping[str, Any] | None`, not
> `CachePreflight`. The `CachePreflight` record implemented in Wave 3 carries the
> `--require-all-hit` bundle (two fingerprints, two targets, one receipt);
> `_run_initialized_training` consumes the model-free *training* preflight
> mapping (`rank`, `world_size`, `cache_root`, `cache_root_receipt`,
> `components`, `vocab_groups`, `schedule`, `train_cache`, `eval_cache`,
> `train_micro_steps`, `eval_reduction`, `phase_trace`). These are different
> bundles; reusing the name would require changing D5's record, which is
> user-owned.
> (c) `HydratedTrainingInputs` is still NOT introduced, and Wave 3's deferral to
> this task resolves as "cannot land without semantic loss". D5 permits the
> record "only for the exact bundles currently returned as untyped mappings", and
> the session has no such bundle: `cache_workflow._hydrate_eval_micro_steps_from_cache`
> returns `tuple[tuple[SupervisedMicroStep, ...], str, int]`, whose third value
> (`eval_pack_count_for_consensus`) has no field in the plan's four-field record
> and is consumed by `runtime.validate_eval_reduction_consensus`; dropping it
> would break a live collective. `train_micro_steps` is hydrated in a different
> phase, is rebound twice (`_attach_image_processors_to_micro_steps`,
> `_apply_fa2_branch_proof_policy`) and split into
> `full_train_micro_steps`/`trainer_micro_steps` before eval hydration starts, so
> the two sequences never coexist as one returned bundle; and the eval
> hydration `receipt` is published through
> `_phase_receipt_sink(lifecycle, "evaluation_hydration")` rather than returned,
> while train hydration has no hydration receipt at all. Adding a fifth field or
> reordering the phases would change the design or the frozen event order.
> (d) The facade retains Accelerator construction and binding
> (D8: "binds Accelerator through the control plane, then constructs exactly one
> session"; tasks 6.4 "open/bind the control plane"), while `_build_accelerator`
> and the `validate_accelerator_runtime` import move to the session owner per the
> manifest's wave-5 `harness_seam_repoints`. The session reads the live
> Accelerator back from `RankControlPlane.accelerator`, which
> `bind_accelerator` already stores, so no seventh resource slot was added.
> (e) The three declared wave-5 harness repoints landed as pure module swaps:
> `pipeline._build_accelerator`, `pipeline.validate_accelerator_runtime`, and
> `pipeline._run_initialized_training` become `session.*`; the
> `_run_initialized_training` stub keeps its `**kwargs` shape and its
> `writer`/`run_directory`/`run_id`/`resolved_config` reads because the session
> calls the moved function with its Wave-0 keyword names. Six test files were
> repointed under the same rule (`test_pipeline_assembly.py`,
> `test_pipeline_cache_preflight.py`, `test_pipeline_phase_convergence.py`,
> `test_pipeline_exact_resume.py`, `test_checkpoint_handler_identity.py`,
> `test_orchestration_compatibility.py`), and `_patch_shared_cache_import` now
> replaces shared imports on `(session, cache_workflow)`. A normalized diff proves
> every removed line has an identical added counterpart differing only by the
> module name: **no expected value changed anywhere**.
> (f) `WAVE0_PIPELINE_OWNED_HELPERS` flips its four wave-5 entries
> (`_run_initialized_training`, `_checkpoint_handler`, `_eval_forward_handler`,
> `_final_handler`) from `(5, None)` to `(5, session)`; the node then proves the
> session owns each helper and the facade deleted it.
> (g) **Blocking residue, not fixed here (scripts/ is outside this wave's write
> scope):** three probe scripts still reach into the pre-move owner and now fail
> at import/attribute time, taking their contract tests with them ---
> `scripts/probes/coordexp_swift/wave2_packed_parity.py:119`
> (`from src.training.pipeline import _build_accelerator, enable_training_memory_savers`),
> `scripts/probes/coordexp_swift/wave3_zero_weight_gpu.py:83`
> (`from src.training.pipeline import enable_training_memory_savers`), and
> `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:1177-1185`
> (reads/writes `training_pipeline._checkpoint_handler`). Each is a one-symbol
> repoint to `src.training.session`; no `SOURCE_OWNERS`/`EXECUTION_OWNER_PATHS`
> entry needs to change, because `src/training/pipeline.py` still exists and those
> lists are hashed live (stale, not failing). Wave-4 disclosure (d) already
> recorded a Wave-6 disposition for two of these owner lists.
> Measured fallout (the first three files were measured green at parent
> `8239be128` -- 417 passed in one invocation; the fourth's two failures are
> `AttributeError` on the moved `_checkpoint_handler`, so they are caused by
> this wave):
> `tests/qwen/test_packed_parity.py` 199 failed / 88 passed,
> `tests/qwen/test_wave2_v3_probe.py` 15 failed / 1 passed,
> `tests/losses/test_wave3_zero_weight_probe_contract.py` 113 failed / 1 passed,
> `tests/training/test_reconcile_exact_resume_probe.py` 2 failed / 94 passed --
> 329 nodes total. `scripts/probes/coordexp_swift/wave5_provider_benchmark.py`
> is unaffected (it imports only `run_training_pipeline`, which the facade
> keeps): its `EXECUTION_OWNER_PATHS` fingerprint of `pipeline.py` is stale, not
> failing. `tests/training/test_reconcile_exact_resume_probe.py:639-712` carries
> the mirror-image `training_pipeline._checkpoint_handler` seam and needs the
> same repoint, but fixing it alone would not turn those two nodes green while
> the script it exercises still patches the facade, so it was left with its
> script.
> (h) Task 6.6's commit and revert proof are unexecuted: this wave was produced
> under an explicit no-commit instruction.

> Lead close-out (2026-08-19): wave commit is `32867ef18`; a staged
> revert of it is zero lines different from `8239be128`. The blocking
> scripts/ residue was resolved by the lead with one-symbol repoints to
> the session owner (`wave2_packed_parity.py`, `wave3_zero_weight_gpu.py`,
> `reconcile_exact_resume_probe.py` held-parent seam split into
> session/facade aliases so `_checkpoint_handler` patches land on the
> owner while `run_training_pipeline` stays on the facade, mirrored in
> its test); all affected suites re-verified green at 710 passed, zero
> skips, fixtures untouched. git itself detected `session.py` as an 89%
> copy of the pre-wave `pipeline.py` -- independent confirmation of
> verbatim movement.

## 7. Wave 6 - Retire Legacy Provider Selection

- [x] 7.1 Add failing strict-config and provider tests requiring `legacy_fused`, `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`, unknown aliases, and environment-driven replacement of strict config to be rejected or ignored exactly as established by the prerequisite contract, while `synchronous` and explicit `overlapped` remain accepted.
- [x] 7.2 Narrow `ForwardInputProviderMode` to `synchronous|overlapped`; remove the deprecated environment resolver/source receipt, selector allowlist, convergence branch, and all legacy config fixtures or compatibility defaults.
- [x] 7.3 Delete the `legacy_fused` provider disposition and trainer-owned device-direct build branch; make the provider interface non-optional and preserve synchronous/overlapped causal positions, CPU-build/device-transfer split, depth bound, validation, error timing, cancellation, and close behavior.
- [x] 7.4 Update RunWriter/provider policy validation and exact-resume policy construction for the supported modes without changing unrelated run/checkpoint schemas or current supported-mode bytes.
- [x] 7.5 Update canonical operator docs and implementation maps to name strict config as the sole selector, `synchronous` as reference, `overlapped` as experimental, and the new code owners; route any needed legacy explanation to history rather than retaining live aliases.
- [x] 7.6 Run a repository residue search for `legacy_fused`, `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`, deprecated provider-source receipts, production `src.qwen.parity` imports, pipeline-private causal helpers, and forbidden reverse imports; allow only explicitly labeled historical evidence.
- [x] 7.7 Gate Wave 6 with its frozen config/provider/trainer/pipeline/resume/artifact/historical-reader manifest IDs and strict OpenSpec validation. Commit one scoped independently revertible wave, prove revert restores Wave 5 without cache mutation, then obtain the single pre-cost standards/overdesign/intent audit; no cache/GPU action may be authorized with an unresolved P0/P1.

> **Wave 6 closed (2026-08-19, parent `6760a6a8c`; produced under an explicit
> no-commit instruction):** `legacy_fused` and
> `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` are deleted.
> `ForwardInputProviderMode` is `Literal["synchronous", "overlapped"]`,
> `resolve_forward_input_provider_mode` reads only strict config,
> `ForwardInputProviderModeSource` is `Literal["strict_config"]`,
> `build_forward_input_provider` returns a non-optional provider, the session's
> disposition-agreement check is gone, `run_writer.bind_forward_input_provider_mode`
> accepts two modes, and `cache_workflow._RECEIPT_ENVIRONMENT_SELECTORS` no longer
> allowlists the retired variable (no caller passed it, so no receipt byte moved).
> Gate: the manifest `wave6-legacy-selector-gate` argv ran verbatim at 547 passed,
> 0 failed, 0 skipped (all 24 declared flips green, `expected_red_nodes` empty);
> the wave-0 baseline argv replayed fully green at 283 passed; `openspec validate
> --strict` valid; ruff clean on every touched file with no new format debt;
> `tests/fixtures/**` untouched and the characterization fixtures replayed
> unchanged. The RED step recorded 12 failed / 179 passed before any source edit.
> Collected-node accounting: 8 nodes removed from the gate argv and 18 added, each
> removal a legacy-mode parametrize case or a renamed/rewritten legacy test; the
> wave-0 argv moved 285 -> 283 by the same accounting.
> (a) `to_receipt_dict()` keeps all nine keys, including the now-always-`None`
> `environment_variable` and the now-always-`False` `is_semantic_override`.
> Dropping either would have changed supported-mode run.json and exact-resume
> policy bytes, which this wave is not allowed to do.
> (b) The `config_provenance_resolution` rank-convergence phase is retained per
> design decision 11 ("the direct config value may still be rank-converged");
> only the environment branch inside the resolver was deleted. Its two-rank
> divergence test now diverges by patching the resolver on rank 1 instead of by
> setting the retired variable, and every other assertion replays unchanged.
> (c) "Trainer-owned device-direct build branch" was read as the *default
> wiring*, not the function: `self.qwen_forward = qwen_forward or
> _default_qwen_forward` becomes `self.qwen_forward = qwen_forward`, and
> construction now fails closed with `trainer.forward_input_source_required`
> unless exactly one of `forward_input_provider` / `qwen_forward` is supplied.
> `_default_qwen_forward` itself is retained because `src/eval/forward.py`
> imports it as its own production default; it is now trainer-dead and
> eval-only, so moving its ownership is a later-wave residual. The alternative
> reading -- requiring the provider and deleting the `qwen_forward` seam --
> would have rewritten 28 trainer-suite call sites and changed what that suite
> asserts, colliding with "supported-mode tests keep passing unchanged".
> (d) Wave-4 disclosure (d) is discharged: `wave2_packed_parity.py`
> `SOURCE_OWNERS` gains `src/artifacts/identity.py` plus the six post-decomposition
> owners of `pipeline.py`, and `wave5_provider_benchmark.py`
> `EXECUTION_OWNER_PATHS` gains `execution_plan`, `control_plane`, `session`,
> `cache_workflow`, `cache_contract`, `reporting`, `micro_steps`, `run_schema`,
> and `run_state`. Both lists hash live, so this changes receipts only at the next
> run; `tests/qwen/test_packed_parity.py` (287 passed) and
> `tests/training/test_wave5_provider_benchmark.py` stay green.
> (e) **STOP AND REPORT, unresolved by design:**
> `scripts/probes/coordexp_swift/wave5_provider_benchmark.py` still declares arm
> `"D"` (`legacy_fused`) and is the only live legacy reference outside strict
> rejection tests and this change's own authority artifacts. Removing it is not
> separable with bounded effort: `TRIAD_ORDERS` is a 3-arm Latin square and
> `CANDIDATE_ARMS`/`MIN_PAIRED_OBSERVATIONS = 3`/`expected_arm_executions = 9`
> would all need new values, which is a user-owned measurement-design decision.
> The frozen r2 evidence does not block removal -- `load_historical_r2_plan`
> authenticates by file SHA256, schema string, and `plan_sha256` only, never by
> arms, and that run terminated `resource-stop` / `complete_non_promoting` with
> zero observations. Arm `"D"` is now unrunnable (`_write_arm_config` strict-loads
> the config it writes, which fails validation), but no test exercises that path,
> so the suite is green at 111 passed.
> (f) Task 7.7's commit, revert proof, and pre-cost standards/overdesign/intent
> audit are unexecuted: this wave was produced under an explicit no-commit
> instruction. Gate evidence is recorded in
> `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-6-gate.json`.


> Lead disposition (2026-08-19), benchmark arm D: retained as an
> unrunnable, fail-closed historical probe. The three-arm Latin-square
> measurement contract (TRIAD_ORDERS, MIN_PAIRED_OBSERVATIONS=3,
> expected_arm_executions=9) is frozen research design owned by the user;
> re-choosing it for two arms is a new design decision, and no promoted
> claim depends on the r2 run (terminated unmeasurable/non-promoting).
> Any attempt to execute arm D now fails closed at strict config, which
> is the correct behavior for a deleted mode. The wave-6-gate.json
> receipt the builder drafted was dropped per the waves-0..5 precedent
> (gate records live in these notes).

## 8. Wave 7 - Freeze Owners and Perform the Single Cache Transition

- [ ] 8.1 Freeze the final determinant-owner inventory and record old/new train and eval determinant projections for `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`; prove only declared owner/source identities changed and both old immutable targets remain untouched.
- [ ] 8.2 Prepare a fresh cache-action packet bound to the exact Wave-6 commit that freezes two full argv vectors and two absent receipt paths: the single build-capable `conda run -n ms python -m src.prepare_train_cache ...` invocation and the later identical invocation with `--require-all-hit` and a distinct receipt. Record config, absent train/eval targets, numeric worker, wall-time, CPU-RSS, new-byte, free-disk, and split-count bounds; prove no intermediate fingerprint exists and obtain fresh user authorization. The second argv receives no cache-materialization authority. Planning/implementation approval or prior launch authorization does not satisfy this task.
- [ ] 8.3 Invoke `conda run -n ms python -m src.prepare_train_cache` exactly once only under that authorization, allowing the invocation to publish its train and eval split targets. Stop without retry on command/commit drift, occupied targets, insufficient headroom, timeout, or a declared-bound exceedance; retain the terminal receipt, fingerprints, manifests, worker policy, timing, RSS, bytes, and stop outcome.
- [ ] 8.4 Invoke only the packet-frozen `--require-all-hit` argv with its distinct absent receipt. Require the command itself—not a prior preflight—to fail before render/tokenize/pack/build/temporary-publication/immutable-publication if either target became missing or invalid; on two valid hits, prove it only fingerprints/admits/validates the train/eval targets and writes its named receipt. Verify training admission consumes the new targets while old targets and historical evidence are unchanged; any fallback build is blocking.
- [ ] 8.5 Gate Wave 7 with its frozen cache determinant/manifest/payload/full-digest manifest IDs, receipt validation, disk inventory, and strict OpenSpec validation; a second build or unexplained fingerprint is blocking. Commit one scoped wave whose code revert returns to Wave 6 while both immutable old/new cache targets remain untouched evidence.

## 9. Wave 8 - Vertical Smoke, Residue, and Completion Audit

- [ ] 9.1 Run the full relevant CPU test matrix under `conda run -n ms pytest`, including config, data/template/Qwen/packing/supervision/loss/runtime/training/artifact/eval suites, and record exact command, commit, pass/fail count, duration, and skipped tests.
- [ ] 9.2 Prepare a fresh GPU-launch packet bound to the exact Wave-7 commit, full command, named 1-step config, new cache, and absent artifact root. Fix `world_size=2`, at most two GPUs, and one planned/applied optimizer step; record numeric config-derived ceilings for per-rank model forwards/collectives, wall time, CPU RSS, GPU high-water mark, artifact bytes, and free disk. Obtain fresh user authorization, then run only that production-shaped BF16 smoke. Stop without retry on drift, occupied output, insufficient headroom, timeout, OOM, or bound exceedance; retain all resource, counter, status, and artifact receipts.
- [ ] 9.3 Exact-compare protected result mappings, completed-step/eval rows, run/checkpoint/final/best files, phase order, collective order, and consumer behavior with the compatibility ledger; record the cache identity turnover and legacy-selector removal as the only intentional differences and make no speedup claim.
- [ ] 9.4 Run final architecture residue checks: no reverse imports, no production parity dependency, no duplicated moved implementation, no `legacy_fused`/provider environment override outside history, no event bus/registry/container/FSDP/backend abstraction, and no cache determinant bound to `pipeline.py` or `supervised_trainer.py`.
- [ ] 9.5 Update `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, and `docs/IMPLEMENTATION_MAP.md` only where accepted owners changed; keep stable behavioral requirements in existing specs and do not advertise an efficiency improvement.
- [ ] 9.6 Run the frozen final manifest and `openspec validate decompose-coordexp-swift-training-orchestration --strict`, inspect the complete diff and task receipts, and obtain the single final read-only standards, overdesign, and intent-contract audit set; resolve all P0/P1, disposition lower findings, commit the final scoped wave, and prove its independent code revert leaves immutable cache/evidence untouched before marking the change complete.
