# Wave 0 evidence matrix

This matrix was originally bound to `71dab9772983a4680dfcea742aee949c79960560`
and the receipt beside it. `accepted` means named source plus focused
unit/control-plane coverage exists; it does not promote the conditional delta.
`gap` identifies a later bounded qualification, never an inferred behavior.
That original pin and the Wave 1/Wave 2 prose below it are preserved as
history; they are superseded, not erased, by the final reconciliation.

**Final reconciliation (Task 6.6, 2026-08-19).** Every row below was
re-verified at the final implementation state. Production `src/`, `scripts/`,
and `configs/` are byte-identical from the Wave-3 Attempt-8 execution commit
`f492f36874f036ad4145a45c7b03cd2e0b2fd049` through final HEAD
`161160592094b5b9493dacc8ab9f546f231830ab`
(`git diff --stat f492f3687..1611605920 -- src scripts configs` is empty);
only test files, receipts, docs, and this change's own openspec artifacts
(`tasks.md`) changed across that span (confirmed by
the full `git diff --stat` between the two commits). Rows are therefore
accepted from tests executed at the commit named in each row's Receipt
column, not necessarily re-run again at `161160592`, because the production
source those tests exercise has not moved since. See "Wave 3, 4, and 5
evidence (Task 6.6 final reconciliation)" below the Wave 2 section for the
detail behind every status change.

## Wave 1 focused execution (Task 2)

`receipts/wave-1-focused-baseline.md` reran the frozen 16-file focused suite at
the current docs-only descendant of the pinned implementation commit.  It
executes every named test module on an `accepted` row; the receipt remains
bounded to unit/control-plane behavior and its artifact assertions.  The run
has no failures or skips, so it demonstrates no production-source gap.

The `gap` rows below remain deliberately unaccepted: they name the missing
Wave 2, 3, or 5 qualification rather than a failing current owner.  They do
not justify a `src/` edit, a delta narrowing, or an execution-promotion claim.
They do authorize Task 3 test-first, scenario-specific qualification; if a
failing test demonstrates a production-source gap, stop and re-plan before any
`src/` edit.  Task 2's single decision is therefore **continue with
qualification** under the existing plan; do not treat this baseline as an
exact-continuation or production-launch receipt.

*(Historical, superseded by Task 6.6: the "Wave 2, 3, or 5 qualification"
gap this paragraph describes is now executed for the four Wave-3 rows named
below; see the Wave 3/4/5 section after Wave 2.)*

## Wave 2 contract qualification (Task 3)

`receipts/wave-2-contract-qualification.md` executed the four planned Task 3
commands at `be720f58d82bca79c526bbb3ec1e6dbfa4c95240` with no `src/` change.
All four pass (`7`, `127`, `105`, and `94` selected nodes; no failures, no
skips, no zero-test selection).  They close the disabled-mode publication,
enabled-mode typed-sibling, inference-sibling-ignore, and historical-reader
rows below with cited nodes, all bounded to CPU unit and artifact-fixture
behavior.

The first Wave-2 gate attempt stopped on an over-broad RED that treated exact
mode with a null checkpoint path as invalid. Fixed-target review proved this is
the required publish-only control/parent shape: exact mode enables exact-state
publication, while a non-null checkpoint path independently selects restore.
The config delta is corrected accordingly, no `src/` edit is made, and the
positive publish-only config/consumer regression is the remaining Wave-2 gate.

Rows requiring Wave 3 distributed or GPU evidence remain `gap` and were not
touched.

*(Historical, superseded by Task 6.6 for the four Wave-3 rows only: those
rows now have executed distributed evidence and are `accepted`. The
pack-cache-semantic-identity and training-artifacts rows in this table were
never `gap`; Wave 4 and Wave 5 only added or strengthened their evidence.)*

## Wave 3, 4, and 5 evidence (Task 6.6 final reconciliation)

`receipts/wave-3-attempt-8-outer-terminal-receipt.json` is the executed,
target-bound Wave-3 distributed exact-resume probe at implementation commit
`f492f36874f036ad4145a45c7b03cd2e0b2fd049` (2-rank, GPUs 6/7,
`configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`,
runner `scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py:execute`
driving `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`). All
six frozen commands (`setup`, `success.uninterrupted_control`,
`success.resumed_child`, `rank_failure`, `interruption`, `verification`)
completed with `returncode: 0`; outer `status: verified`. The inner receipt
it validates (`.../reconcile_exact_resume_2026-08-13-r8/terminal-receipt.json`,
schema `coordexp-swift-reconcile-resume-probe-terminal-receipt-v1`, produced
by `reconcile_exact_resume_probe.py:verify_artifacts`) also reports
`status: verified` with `bounded_mismatches: []` and `missing_inputs: []`
across all five required comparisons: `boundary_step1_control_vs_parent`,
`post_update_step2_control_vs_child`,
`post_update_objective_step2_control_vs_child`, `rank_failure_arm`, and
`interruption_arm`. This closes the four Wave-3 `gap` rows below to
`accepted`, bounded strictly to this one 2-rank smoke-scale target-bound run
at `f492f36874f`; it is not a general N-rank or production-scale claim, and
for those four rows it supersedes the Wave 1/Wave 2 "remain gap" prose above.
The pre-existing `scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py`
and `wave7_exact_resume_interrupt.py` probe scripts (and their unit tests)
did not produce this evidence and are retired as row owners in favor of the
packet executor that actually ran.

`receipts/wave-4-cache-provenance-receipt.md` (tests-only commit `4e4c5f5d7`
atop `582a19ff3`, no `src/` change; `489 passed` gate) adds three focused
tests that close previously-untested clauses of already-`accepted` rows, plus
two real-execution receipts: `receipts/wave-4-cache-probe-receipt.json`
(three arms of the real `src.prepare_train_cache` CLI entrypoint against a
private `COORDEXP_SWIFT_PACK_CACHE_ROOT`: arm `a` absent-target build
`built`/`built`, arm `b` valid-target byte-preserving `hit`/`hit`, arm `c`
invalid-target `training.pack_cache_immutable_collision` fail-closed with the
corrupted bytes proven untouched after the failed run) and
`receipts/wave-4-provenance-receipt.json` (a real dirty-worktree provenance
snapshot: `repository_state: dirty`, `no_environment_section: true`). The
Wave-4 scout also flagged three pre-existing tests in
`tests/training/test_pipeline_pack_cache_rebuild.py` and
`tests/training/test_pipeline_assembly.py` that were never cited in this
matrix; they are folded into their matching rows below as additional owners,
not new rows, since the scenarios they cover already have rows.

`receipts/wave-5-historical-docs-receipt.md` (close commit `161160592`, i.e.
final HEAD) adds a file-access-level proof that inference readers never open
`training_state/` — a real committed inference payload paired with a real
committed `training_state/` sibling, with `pathlib.Path.open` interception
around the three inference-reader call sites capturing 45 real opens
(manifest, adapter config/weights, embedding delta) and zero paths
containing `training_state` — and a schema-rejection test for an unknown
schema family string and an unknown future `schema_version`. Both are folded
into their matching rows below. This wave also reconciled `docs/COORDEXP_SWIFT.md`,
`docs/SYSTEM_OVERVIEW.md`, `docs/ARTIFACTS.md`, `docs/PROJECT_CONTEXT.md`,
`docs/IMPLEMENTATION_MAP.md`, `docs/AGENT_INDEX.md`, and `docs/catalog.yaml`
to the accepted bounded contract, removing the categorical exact-resume
denials the "Stable/document conflicts" section below still quotes as
history; editing those docs is out of scope for this matrix.

| Delta spec | Requirement / scenario | Live source owner | Focused test owner | Verification command | Receipt | Disposition | Claim boundary |
|---|---|---|---|---|---|---|---|
| config-runtime | Exact Resume Configuration Is Strict And Opt-In | `src/config/models.py:ResumeConfig` | `tests/config/test_train_config.py:test_exact_resume_defaults_disabled_and_is_persisted` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py` | `receipts/wave-0-baseline.md` | accepted | strict config only |
| config-runtime | Resume configuration is omitted | `src/config/models.py:ResumeConfig` | `tests/config/test_train_config.py:test_exact_resume_defaults_disabled_and_is_persisted` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py` | `receipts/wave-0-baseline.md` | accepted | config, not publication |
| config-runtime | Exact resume path is authored | `src/config/loader.py:load_train_config` | `tests/config/test_train_config.py:test_exact_resume_accepts_same_world_mode_and_resolves_checkpoint_path` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py` | `receipts/wave-0-baseline.md` | accepted | config only |
| config-runtime | Resume fields are incompatible | `src/config/models.py:ResumeConfig` | `tests/config/test_train_config.py:test_exact_resume_rejects_unknown_or_incompatible_controls` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py` | `receipts/wave-2-contract-qualification.md` | accepted | disabled-with-path and unknown controls fail before runtime |
| config-runtime | Exact state publication is selected without continuation | `src/config/models.py:ResumeConfig`; `src/training/pipeline.py:run_training_pipeline` | `tests/config/test_train_config.py:test_exact_publish_only_mode_preserves_null_checkpoint_path`; `tests/training/test_wave7_exact_resume_config_bundle.py` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py tests/training/test_wave7_exact_resume_config_bundle.py tests/training/test_input_attestation.py tests/training/test_wave7_exact_resume_sequence.py` | `receipts/wave-2-contract-qualification.md`; `receipts/wave-3-attempt-8-outer-terminal-receipt.json` (frozen `-r8` `uninterrupted_control.yaml` and `resumed_parent.yaml` both resolve `resume.mode: exact_same_world_size` with null `checkpoint_dir` and executed publish-only, committing exact state without any restore) | accepted | 311-pass config-consumer gate; publish-only control/parent executed at runtime in Attempt 8, not continuation |
| config-runtime | Exact Resume Requires Strict Deterministic Replay Admission | `src/runtime/seeding.py:seed_training_runtime`; `src/runtime/seeding.py:_apply_strict_determinism_policy` | `tests/runtime/test_train_runtime.py:test_strict_determinism_requires_launcher_environment_and_sets_torch_cudnn_policy` | `conda run -n ms python -m pytest -q tests/runtime/test_train_runtime.py` | `receipts/wave-0-baseline.md` | accepted | unit control plane |
| config-runtime | Exact resume selects the compatibility runtime policy | `src/config/models.py:TrainConfig` | `tests/config/test_train_config.py:test_exact_resume_requires_strict_cuda_replay_determinism` | `conda run -n ms python -m pytest -q tests/config/test_train_config.py` | `receipts/wave-0-baseline.md` | accepted | config admission |
| config-runtime | A launcher prerequisite is absent | `src/runtime/seeding.py:seed_training_runtime`; `src/runtime/seeding.py:_apply_strict_determinism_policy` | `tests/runtime/test_train_runtime.py:test_strict_determinism_requires_launcher_environment_and_sets_torch_cudnn_policy` | `conda run -n ms python -m pytest -q tests/runtime/test_train_runtime.py` | `receipts/wave-0-baseline.md` | accepted | mocked launcher env |
| config-runtime | Rank runtime identities disagree | `src/training/pipeline.py:_establish_converged_runtime_determinism` | `tests/training/test_pipeline_cache_preflight.py:test_pinned_runtime_baseline_admission_rejects_a_drifted_peer` | `conda run -n ms python -m pytest -q tests/training/test_pipeline_cache_preflight.py` | `receipts/wave-0-baseline.md` | accepted | control-plane rank admission |
| pack-cache-semantic-identity | Realized Cache Payload Identity Is Complete | `src/training/pack_cache.py:build_packing_cache_determinants` | `tests/training/test_pack_cache_determinant_registry.py:test_same_path_frontend_asset_edit_changes_fingerprint` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache_determinant_registry.py` | `receipts/wave-0-baseline.md` | accepted | focused cache controls |
| pack-cache-semantic-identity | A transitive cached-payload producer changes | `src/training/pack_cache.py:build_packing_cache_determinants` | `tests/training/test_pack_cache.py:test_packing_cache_registry_exposes_source_owner_identities` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md` | accepted | cache identity unit test |
| pack-cache-semantic-identity | A front-end asset changes at the same path | `src/training/pack_cache.py:build_packing_cache_fingerprint` | `tests/training/test_pack_cache_determinant_registry.py:test_same_path_frontend_asset_edit_changes_fingerprint` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache_determinant_registry.py` | `receipts/wave-0-baseline.md` | accepted | cache identity unit test |
| pack-cache-semantic-identity | Determinants drift during preparation | `src/training/pack_cache.py:write_micro_step_cache`; `src/training/pipeline.py:_resolve_or_build_pack_cache` | `tests/training/test_pack_cache.py:test_publication_revalidates_determinants_immediately_before_install`; `tests/training/test_pipeline_pack_cache_rebuild.py:test_preparation_revalidates_determinants_after_build_before_publication` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py tests/training/test_pipeline_pack_cache_rebuild.py` | `receipts/wave-4-cache-provenance-receipt.md` | accepted | cache publication unit test; pipeline-level determinant revalidation before install now also covered (previously-untested, Wave-4 scout flag) |
| pack-cache-semantic-identity | Cache Admission Precedes Expensive Training Setup | `src/training/pipeline.py:_resolve_pack_cache_root`; `src/training/pipeline.py:prepare_training_pack_caches` | `tests/training/test_pipeline_cache_preflight.py:test_each_required_cache_failure_is_before_model_and_accelerator`; `tests/training/test_pipeline_assembly.py:test_prepare_training_pack_caches_is_model_free_and_covers_train_and_eval` | `conda run -n ms python -m pytest -q tests/training/test_pipeline_cache_preflight.py`; `conda run -n ms python -m pytest -q tests/training/test_pipeline_assembly.py -k prepare_training_pack_caches` | `receipts/wave-4-cache-provenance-receipt.md` | accepted | preflight control plane; positive model-free path covering both train and eval splits now also covered (previously-untested, Wave-4 scout flag) |
| pack-cache-semantic-identity | A required cache is absent | `src/training/pack_cache.py:load_cache_manifest` | `tests/training/test_pipeline_cache_preflight.py:test_each_required_cache_failure_is_before_model_and_accelerator` | `conda run -n ms python -m pytest -q tests/training/test_pipeline_cache_preflight.py` | `receipts/wave-0-baseline.md` | accepted | preflight control plane |
| pack-cache-semantic-identity | A required payload is corrupt | `src/training/pack_cache.py:_iter_validated_chunks` | `tests/training/test_pack_cache.py:test_micro_step_cache_rejects_corrupt_required_chunk` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md` | accepted | cache payload unit test |
| pack-cache-semantic-identity | Required caches are admitted | `src/training/pack_cache.py:load_rank_micro_steps_from_cache` | `tests/training/test_pipeline_cache_preflight.py:test_two_rank_preflight_success_tears_down_gloo_before_accelerator_transition`; `tests/training/test_pipeline_cache_preflight.py:test_cache_rank_report_projects_only_bounded_allowlisted_context_into_artifact` | `conda run -n ms python -m pytest -q tests/training/test_pipeline_cache_preflight.py` | `receipts/wave-0-baseline.md` | accepted | two-rank preflight/context projection |
| pack-cache-semantic-identity | Cache Payload Is Current-Version-Only | `src/training/pack_cache.py:cache_is_complete` | `tests/training/test_pack_cache.py:test_public_cache_apis_reject_alternate_well_shaped_root_before_consumption` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md` | accepted | cache validation unit test |
| pack-cache-semantic-identity | Old cache version is discovered | `src/training/pack_cache.py:cache_dir_for_fingerprint` | `tests/training/test_pack_cache.py:test_cache_dir_for_fingerprint_uses_v3_namespace` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md` | accepted | version namespace only |
| pack-cache-semantic-identity | Expected current target is invalid | `src/training/pack_cache.py:_publish_micro_step_cache` | `tests/training/test_pack_cache.py:test_existing_invalid_target_fails_closed_without_byte_changes`; `tests/training/test_pipeline_pack_cache_rebuild.py:test_rank_zero_rejects_occupied_invalid_cache_without_rebuilding` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py tests/training/test_pipeline_pack_cache_rebuild.py` | `receipts/wave-4-cache-provenance-receipt.md`; `receipts/wave-4-cache-probe-receipt.json` | accepted | immutable collision unit test; real CLI arm `c` (`invalid_target_fail_closed`, `training.pack_cache_immutable_collision`, corrupted bytes proven untouched) adds non-fixture confirmation |
| pack-cache-semantic-identity | Existing current target is valid | `src/training/pack_cache.py:cache_is_complete` | `tests/training/test_pack_cache.py:test_valid_existing_target_is_byte_preserving_hit` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md`; `receipts/wave-4-cache-probe-receipt.json` | accepted | cache reuse unit test; real CLI arm `b` (`valid_existing_target_hit`, train/eval both `hit`/`complete`) adds non-fixture confirmation |
| pack-cache-semantic-identity | Preparation verifies payloads | `src/training/pack_cache.py:_iter_validated_chunks`; `src/training/pipeline.py:_resolve_eval_pack_cache` | `tests/training/test_pack_cache.py:test_micro_step_cache_rejects_corrupt_required_chunk`; `tests/training/test_pack_cache.py:test_publication_fails_closed_on_hash_matching_forbidden_global_payload`; `tests/training/test_pipeline_assembly.py:test_resolve_eval_pack_cache_hardcodes_payloads_verification_level` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py`; `conda run -n ms python -m pytest -q tests/training/test_pipeline_assembly.py -k hardcodes_payloads` | `receipts/wave-4-cache-provenance-receipt.md` | accepted | payload validation unit test; a digest-valid payload that decodes to a forbidden pickle global now proven to fail closed at publication, and the real eager-eval call site now proven to hardcode `verification_level="payloads"` (previously masked because all call sites monkeypatched the resolver) |
| pack-cache-semantic-identity | Distributed train resolution avoids a duplicate payload pass | `src/training/pack_cache.py:load_rank_micro_steps_from_cache` | `tests/training/test_pack_cache.py:test_manifest_admission_defers_payload_work_to_one_eager_rank_load` | `conda run -n ms python -m pytest -q tests/training/test_pack_cache.py` | `receipts/wave-0-baseline.md` | accepted | manifest/eager-load unit test |
| pack-cache-semantic-identity | Required payload changes after admission | `src/training/pack_cache.py:load_rank_micro_steps_from_cache` | `tests/training/test_pipeline_cache_preflight.py:test_two_rank_rank_one_required_payload_failure_converges_and_tears_down` | `conda run -n ms python -m pytest -q tests/training/test_pipeline_cache_preflight.py` | `receipts/wave-0-baseline.md` | accepted | two-rank control plane |
| training-artifacts | Run Artifacts Record Executed Environment Provenance | `src/artifacts/provenance.py:collect_execution_provenance` | `tests/artifacts/test_provenance.py:test_full_provenance_is_deterministic_strict_json_and_non_secret`; `tests/artifacts/test_provenance.py:test_environment_secrets_never_enter_provenance` | `conda run -n ms python -m pytest -q tests/artifacts/test_provenance.py` | `receipts/wave-4-cache-provenance-receipt.md` | accepted | fixture provenance; four secret-shaped environment variables with sentinel values now proven absent from the serialized receipt, with the top-level key set unchanged and no `environment`/`env` key anywhere in the nested structure |
| training-artifacts | A dirty checkout starts training | `src/artifacts/provenance.py:collect_repository_provenance` | `tests/artifacts/test_provenance.py:test_repository_provenance_for_tracked_dirty_checkout` | `conda run -n ms python -m pytest -q tests/artifacts/test_provenance.py` | `receipts/wave-0-baseline.md`; `receipts/wave-4-provenance-receipt.json` | accepted | fixture repo; real dirty-worktree collection confirmed at Wave 4 (`repository_state: dirty`, `no_environment_section: true`), non-fixture |
| training-artifacts | A dependency identity is unavailable | `src/artifacts/provenance.py:collect_dependency_provenance` | `tests/artifacts/test_provenance.py:test_dependency_failures_are_per_component_and_do_not_crash` | `conda run -n ms python -m pytest -q tests/artifacts/test_provenance.py` | `receipts/wave-0-baseline.md` | accepted | fixture dependency |
| training-artifacts | Minimal Inference Checkpoint Payloads | `src/artifacts/checkpoints.py:CheckpointWriter.write_checkpoint` | `tests/artifacts/test_checkpoint_writer.py:test_adapter_only_atomic_checkpoint_and_safe_peft_arguments` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | writer control plane |
| training-artifacts | Adapter-only checkpoint is saved | `src/artifacts/checkpoints.py:CheckpointWriter.write_checkpoint` | `tests/artifacts/test_checkpoint_writer.py:test_adapter_only_atomic_checkpoint_and_safe_peft_arguments` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | mocked PEFT |
| training-artifacts | Adapter and selected embeddings are saved | `src/artifacts/checkpoints.py:CheckpointWriter.write_checkpoint` | `tests/artifacts/test_checkpoint_writer.py:test_adapter_plus_compact_selected_token_delta` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | writer fixture |
| training-artifacts | Exact state is disabled | `src/artifacts/checkpoints.py:CheckpointWriter.write_checkpoint` | `tests/artifacts/test_checkpoint_writer.py:test_disabled_exact_state_publishes_only_the_inference_payload_and_aliases` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-2-contract-qualification.md` | accepted | writer fixture; no sibling, normal aliases |
| training-artifacts | Exact state is enabled | `src/artifacts/checkpoints.py:_run_exact_training_state_callback` | `tests/artifacts/test_checkpoint_writer.py:test_exact_state_callback_runs_after_durable_payload_and_before_aliases` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | callback choreography |
| training-artifacts | Exact-state publication fails on one rank | `src/artifacts/checkpoints.py:_gather_callback_statuses` | `tests/artifacts/test_checkpoint_writer.py:test_exact_state_failure_converges_and_retains_unaliased_inference_payload` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | in-process peers |
| training-artifacts | Rank-zero inference payload save fails | `src/artifacts/checkpoints.py:_raise_checkpoint_save_failed` | `tests/artifacts/test_checkpoint_writer.py:test_rank_zero_failure_is_shared_with_peer_without_post_save_barrier` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `receipts/wave-0-baseline.md` | accepted | in-process peers |
| training-artifacts | Existing checkpoint is used | `src/artifacts/checkpoint_payload.py:load_inference_checkpoint_payload_manifest`; `src/adapters/dora.py:inspect_dora_adapter_payload`; `src/qwen/special_token_embeddings.py:inspect_special_token_embedding_delta_payload` | `tests/artifacts/test_checkpoint_payload_identity.py:test_historical_payload_without_a_current_manifest_stays_inference_loadable`; `tests/artifacts/test_checkpoint_payload_identity.py:test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_payload_identity.py` | `receipts/wave-2-contract-qualification.md` | accepted | fixture payload readers; no live model load |
| training-resume | Exact Training State Is An Opt-In Typed Sibling | `src/artifacts/training_state.py:publish_training_state` | `tests/artifacts/test_training_state.py:test_manifest_strict_roundtrip_and_deterministic_digest` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py` | `receipts/wave-0-baseline.md` | accepted | artifact schema |
| training-resume | Exact training state is disabled | `src/artifacts/checkpoints.py:CheckpointWriter.write_checkpoint`; `src/artifacts/run_writer.py:RunWriter.record_checkpoint_publication_event` | `tests/artifacts/test_checkpoint_writer.py:test_disabled_exact_state_publishes_only_the_inference_payload_and_aliases`; `tests/artifacts/test_run_artifacts.py:test_inference_only_publication_event_cannot_carry_exact_state_identity` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py`; `conda run -n ms python -m pytest -q tests/artifacts/test_run_artifacts.py` | `receipts/wave-2-contract-qualification.md` | accepted | writer and event fixtures |
| training-resume | Exact training state is enabled | `src/artifacts/training_state.py:commit_training_state_contributions`; `src/artifacts/checkpoints.py:_run_exact_training_state_callback` | `tests/artifacts/test_training_state.py:test_distributed_rank_contributions_commit_one_complete_atomic_state`; `tests/artifacts/test_checkpoint_writer.py:test_exact_state_callback_runs_after_durable_payload_and_before_aliases`; `tests/artifacts/test_checkpoint_payload_identity.py:test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_checkpoint_payload_identity.py` | `receipts/wave-2-contract-qualification.md` | accepted | in-process publication/ordering fixtures |
| training-resume | Inference reads a checkpoint with exact state | `src/artifacts/checkpoint_payload.py:admit_inference_checkpoint_payload_identity` | `tests/artifacts/test_checkpoint_payload_identity.py:test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata`; `tests/artifacts/test_checkpoint_payload_identity.py:test_inference_reader_ignores_real_committed_training_state_without_opening_it` | `conda run -n ms python -m pytest -q tests/artifacts/test_checkpoint_payload_identity.py` | `receipts/wave-5-historical-docs-receipt.md` | accepted | payload-reader level, now proven at file-access level: a real committed inference payload paired with a real committed `training_state/` sibling, with `pathlib.Path.open` interception around the three inference-reader call sites capturing 45 real opens (manifest, adapter config/weights, embedding delta) and zero paths containing `training_state`; still explicit-path consumption only, NOT coverage through `src/inference/pipeline.py` composition |
| training-resume | Exact Resume Is Same-World-Size And Optimizer-Boundary Only | `src/artifacts/training_state.py:admit_training_state`; `src/training/pipeline.py:run_training_pipeline`; `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:verify_artifacts` | `tests/training/test_exact_resume.py:test_mid_accumulation_serialization_is_rejected_explicitly`; `tests/training/test_pipeline_exact_resume.py:test_restored_cursor_seeds_absolute_counters_and_slices_only_trainer_input`; `tests/training/test_reconcile_exact_resume_probe.py:test_verify_admits_and_compares_both_boundaries_and_reports_verified` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py tests/training/test_reconcile_exact_resume_probe.py`; `conda run -n ms python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8` | `receipts/wave-3-attempt-8-outer-terminal-receipt.json` (`status: verified`; comparison `boundary_step1_control_vs_parent` matched with zero `bounded_mismatches`) | accepted | 2-rank smoke-scale target-bound run at `f492f36874f`; boundary comparison covers identities/cursor/trainable-model/optimizer/scheduler/scaler/rng/cuda-topology/structure-signature at the restore boundary; not a general N-rank or production-scale claim |
| training-resume | Matching same-world-size continuation is admitted | `src/artifacts/training_state.py:admit_training_state` | `tests/training/test_exact_resume.py:test_admit_restore_returns_authenticated_next_cursor_position`; `tests/training/test_pipeline_exact_resume.py:test_read_only_admission_builds_lineage_from_the_authenticated_manifest` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py` | `receipts/wave-0-baseline.md` | accepted | admission/restore only |
| training-resume | First post-resume update matches the uninterrupted branch | `src/training/pipeline.py:run_training_pipeline`; `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:verify_artifacts` | `tests/training/test_reconcile_exact_resume_probe.py:test_objective_comparison_ignores_input_timing_but_not_training_semantics`; `tests/training/test_reconcile_exact_resume_probe.py:test_verify_fails_closed_on_missing_or_mismatched_train_step2_row` | `conda run -n ms python -m pytest -q tests/training/test_reconcile_exact_resume_probe.py`; `conda run -n ms python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8` | `receipts/wave-3-attempt-8-outer-terminal-receipt.json` (`status: verified`; comparisons `post_update_step2_control_vs_child` and `post_update_objective_step2_control_vs_child` both matched with zero `bounded_mismatches`) | accepted | 2-rank smoke-scale target-bound run at `f492f36874f`; step-2 runtime state and the training-semantic objective (loss row, timing/duration/resource keys excluded) both matched child vs. control; not a general N-rank or production-scale claim. Superseded owner `scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py` did not execute this evidence |
| training-resume | World size or save boundary differs | `src/artifacts/training_state.py:admit_training_state` | `tests/training/test_exact_resume.py:test_identity_and_topology_mismatch_fail_before_runtime_mutation` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py` | `receipts/wave-0-baseline.md` | accepted | control-plane rejection |
| training-resume | Resume Admission Is Fail Closed | `src/artifacts/training_state.py:admit_training_state`; `src/artifacts/run_writer.py:1278-1283` (unconditional sibling inference-payload identity authentication); `src/training/pipeline.py:4118` (read-only admission) preceding `:4148` mutable restore with `:4163` cross-check | `tests/artifacts/test_training_state.py:test_admission_rejects_every_identity_mismatch_before_mutation` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py` | `receipts/wave-0-baseline.md`; `receipts/wave-6-independent-audit.md` (clause-owner verification) | accepted | artifact admission; the sibling inference-payload identity clause is enforced at the run-writer/pipeline layer cited here, verified by the independent audit rather than the parametrized identity-kind test |
| training-resume | Required state is incomplete or corrupt | `src/artifacts/training_state.py:_admit_training_state_at` | `tests/artifacts/test_training_state.py:test_corrupt_and_missing_components_are_rejected_before_callback` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py` | `receipts/wave-0-baseline.md` | accepted | artifact admission |
| training-resume | Inference-only checkpoint is selected | `src/artifacts/training_state.py:_open_training_state_at` | `tests/artifacts/test_training_state.py:test_model_only_and_uncommitted_checkpoints_are_rejected` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py` | `receipts/wave-0-baseline.md` | accepted | reader admission |
| training-resume | Exact-State Publication Is Atomic Across Ranks | `src/artifacts/training_state.py:commit_training_state_contributions`; `src/artifacts/training_state.py:abort_training_state_contributions`; `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:rank_failure`; `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:interruption` | `tests/training/test_exact_resume.py:test_two_rank_publication_uses_control_only_gathers_and_commits_once`; `tests/training/test_reconcile_exact_resume_probe.py:test_interruption_after_manifest_staging_before_event_commit`; `tests/training/test_reconcile_exact_resume_probe.py:test_rank_failure_converges_without_admitting_a_manifest` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py tests/training/test_reconcile_exact_resume_probe.py` | `receipts/wave-3-attempt-8-outer-terminal-receipt.json` (`status: verified`; comparisons `rank_failure_arm` and `interruption_arm` both matched with zero `bounded_mismatches`) | accepted | 2-rank smoke-scale target-bound run at `f492f36874f`; a mid-publication rank failure (`manifest_admitted: false`, `published_ranks: [0]`, `residue: []`) and a pre-commit interruption (`published_ranks: []`, `exact_state_present: false`) both converge without a partial or admitted manifest; not a general N-rank or production-scale claim |
| training-resume | One rank fails during exact-state publication | `src/artifacts/training_state.py:abort_training_state_contributions` | `tests/training/test_exact_resume.py:test_rank_local_contribution_failure_converges_without_commit` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py` | `receipts/wave-0-baseline.md` | accepted | gloo control plane |
| training-resume | Publication is interrupted before commit | `src/artifacts/training_state.py:abort_training_state_contributions`; `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py:interruption` | `tests/training/test_exact_resume.py:test_two_rank_gloo_pre_manifest_failures_abort_owned_stage_without_residue`; `tests/training/test_reconcile_exact_resume_probe.py:test_interruption_before_inference_commit_leaves_no_artifacts`; `tests/training/test_reconcile_exact_resume_probe.py:test_interruption_after_manifest_staging_before_event_commit` | `conda run -n ms python -m pytest -q tests/training/test_exact_resume.py tests/training/test_reconcile_exact_resume_probe.py`; `conda run -n ms python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/arms/interruption --stop-after 1 --world-size 2` | `receipts/wave-3-attempt-8-outer-terminal-receipt.json` (`status: verified`; comparison `interruption_arm` matched with zero `bounded_mismatches`: `status: converged_partial_state`, `published_ranks: []`, `exact_state_present: false`, `rank_state_boundary_reached: false`) | accepted | 2-rank smoke-scale target-bound run at `f492f36874f`; stopping after step 1 leaves no exact-state residue and no published ranks; not a general N-rank or production-scale claim. Superseded owner `scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py` did not execute this evidence |
| training-resume | Resume Lineage Is Append Only | `src/artifacts/run_writer.py:RunWriter.initialize` | `tests/artifacts/test_run_artifacts.py:test_initialize_exact_child_records_lineage_without_mutating_parent` | `conda run -n ms python -m pytest -q tests/artifacts/test_run_artifacts.py` | `receipts/wave-0-baseline.md` | accepted | fixture writer |
| training-resume | A continuation run starts | `src/artifacts/run_writer.py:RunWriter.initialize` | `tests/artifacts/test_run_artifacts.py:test_continuation_lineage_is_bound_only_after_parent_admission` | `conda run -n ms python -m pytest -q tests/artifacts/test_run_artifacts.py` | `receipts/wave-0-baseline.md` | accepted | artifact lineage |
| training-resume | Historical artifacts contain extra metadata | `src/artifacts/training_state.py:load_training_state_manifest`; `src/artifacts/training_state.py:_open_training_state_at` | `tests/artifacts/test_checkpoint_payload_identity.py:test_historical_payload_without_a_current_manifest_stays_inference_loadable`; `tests/artifacts/test_training_state.py:test_model_only_and_uncommitted_checkpoints_are_rejected`; `tests/artifacts/test_training_state.py:test_schema_v1_manifest_is_explicitly_unsupported`; `tests/artifacts/test_training_state.py:test_undeclared_file_is_rejected_before_callback`; `tests/artifacts/test_training_state.py:test_unknown_schema_value_is_rejected` | `conda run -n ms python -m pytest -q tests/artifacts/test_training_state.py tests/artifacts/test_checkpoint_payload_identity.py` | `receipts/wave-5-historical-docs-receipt.md` | accepted | reader fixtures; no historical upgrade or migration; an unknown schema family string and an unknown future `schema_version` are now both proven rejected with `training_state.unsupported_schema` before digest validation, alongside the existing schema-v1 rejection |

## Stable/document conflicts

- The stable packing-forward requirement at
  `openspec/specs/coordexp-swift-packing-forward/spec.md` supports only
  `synchronous|overlapped`. Live legacy owners are
  `src/config/models.py:ForwardInputProviderMode`,
  `src/training/forward_input_provider.py:resolve_forward_input_provider_mode`,
  `src/training/pipeline.py:_FORWARD_INPUT_PROVIDER_MODE_ENV`, and
  `src/artifacts/run_writer.py:RunWriter.bind_forward_input_provider_mode`; they
  additionally expose `legacy_fused` and/or the environment override. This is
  later-decomposition residue, not a delta capability.
- `docs/ARTIFACTS.md:47-57` and `docs/SYSTEM_OVERVIEW.md:123-127` deny an
  accepted exact-resume feature; `docs/COORDEXP_SWIFT.md:101-102` and the
  `openspec/specs/coordexp-swift-vertical-smoke/spec.md:130-133` non-goal do
  likewise. They remain canonical until qualification;
  `tests/artifacts/test_training_state.py:test_inference_minimal_artifact_type_is_not_exact_state`
  is only the current focused reader control.
  *(Historical, superseded by Task 6.6 for the canonical docs named here
  except the vertical-smoke spec non-goal: qualification happened, and Wave 5
  removed the categorical exact-resume denials from `docs/ARTIFACTS.md`,
  `docs/SYSTEM_OVERVIEW.md`, and `docs/COORDEXP_SWIFT.md`
  (`receipts/wave-5-historical-docs-receipt.md`); the cited line numbers are
  therefore stale. The `vertical-smoke` non-goal is smoke-scoped, not a
  repository-wide denial, and was left as-is per that same receipt. Editing
  these docs is out of scope for this matrix.)*
