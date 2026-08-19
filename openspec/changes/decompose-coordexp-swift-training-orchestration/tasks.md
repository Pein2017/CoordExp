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

- [ ] 2.1 Add failing tests for `src.artifacts.identity` that require byte/digest/schema/bound/error equivalence with the characterized strict JSON, absent-target publication, base-model weight, repository, and source-owner identity behavior, including historical `src.qwen.parity` imports.
- [ ] 2.2 Move the production identity implementation into `src/artifacts/identity.py`; switch `src/prepare_train_cache.py` and `src/training/input_attestation.py` to it, retain import-only re-exports from `src/qwen/parity.py`, and delete the moved duplicate implementation from parity.
- [ ] 2.3 Add failing `TokenSequence.causal_logits_positions()` tests for sorted uniqueness, causal shift, empty atoms, and exact Qwen `logits_to_keep` inputs.
- [ ] 2.4 Implement `TokenSequence.causal_logits_positions()`, migrate trainer and provider callers, and delete `_logits_positions_to_keep` plus the private cross-module import from `forward_input_provider.py`.
- [ ] 2.5 Add failing canonical-owner tests for `src.training.micro_steps.SupervisedMicroStep`, requiring exact fields/order/annotations/defaults/frozen status and readable compatibility imports from `src.training` and `src.training.supervised_trainer`.
- [ ] 2.6 Move `SupervisedMicroStep` and `supervised_micro_step_schema_identity()` into `src/training/micro_steps.py`; update current production imports and restricted cache allowlists while keeping import-only compatibility re-exports for historical readers.
- [ ] 2.7 Gate Wave 1 with its frozen manifest IDs, import/residue checks, strict OpenSpec validation, exact old-byte load plus new canonical pickle-module assertions, and decoded-value equality; resolve every unexplained ledger difference, commit the wave as one scoped independently revertible unit, and prove its revert restores the Wave-0 commit without cache mutation.

## 3. Wave 2 - Model-Free Plan and Rank Control Plane

- [ ] 3.1 Add failing tests for frozen `TrainingExecutionPlan` construction: exact resolved config/fingerprint, copied measurement context, launcher identity and entry evidence, plus assertions that plan construction performs no collective, filesystem publication, cache materialization, Accelerator construction, tokenizer/model load, or callback registration.
- [ ] 3.2 Implement `src/training/execution_plan.py` with the exact design interface and migrate only facade-owned static resolution into it; reject a mutable service-locator or live model/runtime/cache field.
- [ ] 3.3 Add failing `RankControlPlane` contract tests that replay the characterized single-/multi-rank success and failure reports, fixed frame limits, rank ordering, resource convergence, receipt sinks, pre/post-Accelerator identity binding, close idempotence, and ordered collective calls.
- [ ] 3.4 Implement `src/training/control_plane.py` by moving the bounded rank-report transport, validation, normalization, phase convergence, and cleanup without changing phase names, report bytes, timeout/size bounds, error choice, or collective order.
- [ ] 3.5 Switch `run_training_pipeline(...)` model-free preflight and post-Accelerator binding to `TrainingExecutionPlan` and `RankControlPlane`, keep initialized training in its current owner, then delete the replaced pipeline transport/validation helpers rather than forwarding through them.
- [ ] 3.6 Gate Wave 2 with its frozen execution-plan, phase-convergence, cache-preflight, determinism, pipeline-entry, and mocked-transport manifest IDs plus import/residue and strict OpenSpec checks; commit one scoped independently revertible wave and prove revert restores Wave 1 without cache mutation.

## 4. Wave 3 - Cache Contract and Workflow

- [ ] 4.1 Add failing tests for `micro_step_runtime_config_identity(config)` and determinant-owner mutation. Require the final determinant diff to contain at least the four declared source changes—`supervision_tokens`, `micro_step_runtime_config`, `micro_step_schema`, and `cache_serializer`—while reporting/session/facade edits add no determinant; any missing declared change or unexplained additional change blocks the wave.
- [ ] 4.2 Implement `src/training/cache_contract.py`; update `PACKING_CACHE_DETERMINANT_OWNERS` to bind `micro_step_runtime_config` and `micro_step_schema` to the two narrow owners and prove the semantic content projections remain exact.
- [ ] 4.3 Add failing cache-workflow and CLI tests for one-process preparation, worker resolution, train/eval split aggregation, absent-target publication, model-free fingerprint/admission, rank-local hydration, image-processor attachment, and bounded actionable failures. The tests MUST require `src.prepare_train_cache --require-all-hit` to validate existing train/eval targets, succeed on two valid hits, and fail on any miss/invalid target before any render/tokenize/pack/build/temporary-publication/immutable-publication call.
- [ ] 4.4 Move `prepare_training_pack_caches(...)`, multi-worker render/tokenize/pack orchestration, and its receipt projection into `src/training/cache_workflow.py`; keep the ordinary CLI behavior and `src.training.pipeline` compatibility re-export exact. Add `--require-all-hit` as an in-workflow fail-before-build mode with no fallback construction path; it may only fingerprint, admit, validate both existing targets, and publish its named verification receipt.
- [ ] 4.5 Move model-free fingerprint/admission and rank/eval hydration orchestration into `cache_workflow.py` using only the bounded `CachePreflight`/`HydratedTrainingInputs` records from the design; keep low-level identity, serialization, immutable publication, safe load, and manifest validation in `pack_cache.py`.
- [ ] 4.6 Switch pipeline callers to the cache workflow, delete replaced cache helpers and worker globals from `pipeline.py`, and enforce that `cache_workflow.py` imports neither `session.py` nor model-loading with `load_model=True`.
- [ ] 4.7 Prove determinant semantic projections match the baseline except for the four declared source entries and resulting aggregate/fingerprint. Prove old immutable pickle bytes still restricted-load, new serialization names `src.training.micro_steps`, decoded micro-steps are equal, and no other protected bytes differ; do not invoke production cache materialization in this wave.
- [ ] 4.8 Gate Wave 3 with its frozen cache/preflight/rebuild/CLI/input-attestation/packing manifest IDs, import/residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 2 without publishing or mutating a cache.

## 5. Wave 4 - Existing-Schema Reporting and RunWriter Internals

- [ ] 5.1 Add failing `CompletedStepReporter` tests that exact-compare lifecycle mutation, loss/LR/timing/resource extraction, reduction requests, accuracy validation, rank-resource accounting, every-completed-step row bytes, append outcome broadcast, warmup accounting, and first-step phase behavior with the baseline handler.
- [ ] 5.2 Implement `src/training/reporting.py`, switch the initialized trainer callback to `CompletedStepReporter`, and delete `_train_logging_handler` plus its moved pipeline-only helpers without adding cadence, sinks, fields, ETA, TensorBoard, or a metric registry.
- [ ] 5.3 Expand pre-move RunWriter fixtures to cover initialization, strict `logging.jsonl`, policy/materialization/schedule binding, phase success/failure/not-run, checkpoint publication, best/final files, continuation, warning bounds, failed finalization, and successful finalization as exact bytes and errors.
- [ ] 5.4 Move pure normalization/serialization and bounded field validation into `src/artifacts/run_schema.py`; leave every read, append, fsync, link/replace, collision, and atomic-write operation in the `RunWriter` facade.
- [ ] 5.5 Move pure run-state transitions and exact-resume checkpoint-publication admission into `src/artifacts/run_state.py`; keep `RunWriter` and `admit_exact_resume_checkpoint_publication` import paths, signatures, return values, and I/O sequencing unchanged.
- [ ] 5.6 Delete replaced helpers from `run_writer.py`, prohibit a new store/repository interface, and exact-compare the full fixture tree and failure outcomes against Wave 0.
- [ ] 5.7 Gate Wave 4 with its frozen reporting/RunWriter/checkpoint/resource/training-state/logging manifest IDs, artifact-byte diff, import/residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 3 without cache mutation.

## 6. Wave 5 - Training Session and Thin Facade

- [ ] 6.1 Add failing `TrainingSession` tests for the fixed phase order, owned lifecycle/resource close behavior, model/runtime assembly boundary, cache hydration inputs, trainer handler wiring, primary-exception preservation, best-effort failure publication, success finalization, exact facade result mapping, and idempotent close.
- [ ] 6.2 Implement the `TrainingSession` constructor and move the model/adapter/selected-token/loss/optimizer/runtime assembly plus lifecycle state into `src/training/session.py` without introducing phase subclasses, registries, or alternate backends.
- [ ] 6.3 Move exact-resume admission/restoration/publication wiring, eval/checkpoint/final handlers, forward-input-provider lifetime, profile-sync reset, and success/failure finalization into the session while preserving their literal collective and event order.
- [ ] 6.4 Reduce `run_training_pipeline(...)` to build the immutable plan, open/bind the control plane, initialize the current run owner, admit the cache workflow, construct exactly one session, return its result, and close resources in `finally`.
- [ ] 6.5 Delete all replaced initialized-training, handler, lifecycle, cache, control-plane, and identity code from `pipeline.py`; retain only the public facade and documented compatibility re-export, then enforce the one-way import graph and absence of pass-through helper layers.
- [ ] 6.6 Gate Wave 5 with its frozen pipeline/preflight/convergence/trainer/runtime/eval/checkpoint/resume/cache/entrypoint manifest IDs, protected row/artifact/ordered-call comparisons, residue checks, and strict OpenSpec validation; commit one scoped independently revertible wave and prove revert restores Wave 4 without cache mutation.

## 7. Wave 6 - Retire Legacy Provider Selection

- [ ] 7.1 Add failing strict-config and provider tests requiring `legacy_fused`, `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`, unknown aliases, and environment-driven replacement of strict config to be rejected or ignored exactly as established by the prerequisite contract, while `synchronous` and explicit `overlapped` remain accepted.
- [ ] 7.2 Narrow `ForwardInputProviderMode` to `synchronous|overlapped`; remove the deprecated environment resolver/source receipt, selector allowlist, convergence branch, and all legacy config fixtures or compatibility defaults.
- [ ] 7.3 Delete the `legacy_fused` provider disposition and trainer-owned device-direct build branch; make the provider interface non-optional and preserve synchronous/overlapped causal positions, CPU-build/device-transfer split, depth bound, validation, error timing, cancellation, and close behavior.
- [ ] 7.4 Update RunWriter/provider policy validation and exact-resume policy construction for the supported modes without changing unrelated run/checkpoint schemas or current supported-mode bytes.
- [ ] 7.5 Update canonical operator docs and implementation maps to name strict config as the sole selector, `synchronous` as reference, `overlapped` as experimental, and the new code owners; route any needed legacy explanation to history rather than retaining live aliases.
- [ ] 7.6 Run a repository residue search for `legacy_fused`, `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`, deprecated provider-source receipts, production `src.qwen.parity` imports, pipeline-private causal helpers, and forbidden reverse imports; allow only explicitly labeled historical evidence.
- [ ] 7.7 Gate Wave 6 with its frozen config/provider/trainer/pipeline/resume/artifact/historical-reader manifest IDs and strict OpenSpec validation. Commit one scoped independently revertible wave, prove revert restores Wave 5 without cache mutation, then obtain the single pre-cost standards/overdesign/intent audit; no cache/GPU action may be authorized with an unresolved P0/P1.

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
