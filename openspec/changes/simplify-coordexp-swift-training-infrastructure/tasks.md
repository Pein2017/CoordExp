This OpenSpec change is the Program. Each numbered task group is a Wave, each
checkbox is a dependency-ordered Slice, and each Wave ends with a verification
plus independent-audit Gate. Wave 1 is the first active implementation wave;
later Waves state bounded outcomes and MUST be refined with
`openspec-update-change` only after the preceding Gate closes. No Wave may
advance with unresolved P0/P1 findings, and no gate may leave both old and new
production ownership paths active.

## 1. Wave 1 — Accelerate Process And Runtime Foundation

- [ ] 1.1 Refine Wave 1 into file-owned slices and add failing config/runtime tests for Accelerate world size one, replicated multi-GPU DDP, Accelerator-owned rank/world/device identity, and pre-prepare rejection of externally selected FSDP, DeepSpeed, tensor-parallel, or other unsupported distributed types.
- [ ] 1.2 Initialize one Accelerator process context early enough for rank zero to resolve collision policy once, broadcast one run descriptor/path to every rank, and prove identical `fail`/`timestamp` collision outcomes without creating rank-suffixed directories.
- [ ] 1.3 Collapse the embedded `runtime.backend` branches in `RuntimeConfig`, `TrainRuntime`, pipeline Accelerator construction, and status helpers into the concrete Accelerate path; remove DeepSpeed plugin/config/status code and the implicit `single` fallback without inventing replacement backend interfaces.
- [ ] 1.4 Derive effective accumulation and scheduler/runtime setup from Accelerator world size while preserving all-rank denominator/finite decisions, gradient clipping, optimizer order, CoordExp-owned planned-step scheduler order, barriers, and neutral Accelerate accumulation.
- [ ] 1.5 Migrate every checked-in active production and smoke profile away from backend/DeepSpeed fields, delete DeepSpeed helper/smoke configs, and compare old/new resolved mappings after removing only the approved backend-field allowlist; every other value MUST match.
- [ ] 1.6 Gate Wave 1 with config/runtime/schedule tests, a public-entrypoint one-process Accelerate smoke, an unsupported-wrapper negative subprocess probe, strict OpenSpec validation, `git diff --check`, backend residue searches, and separate engineering-standards plus intent/contract audits with no unresolved P0/P1.

## 2. Wave 2 — Atomic Replacement Of The Artifact/Event/Checkpoint Graph

- [ ] 2.1 Refine Wave 2 into file-owned slices and add failing interface tests for the final run inventory, one typed completed-step observation, bounded trainer result, wide train/eval rows, NaN/Inf-to-null normalization with `non_finite_fields`, bounded warning counters, immutable materialization handles, safe best selection, and one canonical checkpoint alias owner.
- [ ] 2.2 Replace generic trainer event-name dispatch and full `step_results` retention with one typed `on_completed_step(observation)` callback, direct scheduled handlers, compact counters/latest state, and release of step-local tensors/receipts after consumers finish.
- [ ] 2.3 Refactor `ForwardEvalRunner` to return counts, triggers, and one wide reduced scalar mapping without importing artifact-writer or metric-event types; make its rank-zero `logging.jsonl` row the only durable eval scalar schema and remove separate eval-summary metric files.
- [ ] 2.4 Implement one concrete rank-zero run writer for atomic `run.json`, one self-contained `resolved_config.json`, direct single-writer `logging.jsonl`, and atomic checkpoint aliases; bind cache format/fingerprint/determinant digest in `run.json` without copying cache payloads, presentations, or selector state.
- [ ] 2.5 Implement replicated-DDP checkpoint choreography with a pre-save barrier, one rank-zero staged PEFT save using safe serialization/the configured adapter/`save_embedding_layers=False`, optional compact selected-token delta, required LoRA A/B/DoRA and forbidden full-tensor validation, atomic step commit, and a failure-aware success/error collective that makes every rank continue or raise together while cleaning staging residue.
- [ ] 2.6 Make `final.json` and optional `best.json` the sole checkpoint selector owners; test that update-skipped, unsafe, or non-finite steps cannot advance `best.json` by default even with a better eval value.
- [ ] 2.7 Remove new handoff generation, neighboring-handoff and `checkpoint-final` metadata resolution, readiness gates, and downstream inference identity/provenance/merge fields/tests that only propagate those metadata objects; migrate supported inference configs/smokes to explicit adapter and optional delta paths, preserve actual loaded base/adapter/delta provenance, and document/test the narrower standard-PEFT identity boundary plus the stronger embedding-delta hash boundary.
- [ ] 2.8 Delete `RunArtifactManager`, `MetricStreamEvent`, `TrainingArtifactBridge`, per-step Qwen files, per-rank progress streams, receipt/report registries, resolved YAML, durable schedule receipt, full-history `training_result.json`, and every production import/caller/test of those schemas in the same Wave; do not leave a compatibility facade or dual writer.
- [ ] 2.9 Migrate all active configs/tests off `training.logging` cadence and old artifact/debug receipt controls; compare old/new resolved mappings after removing only the complete approved infrastructure-deletion allowlist so seeds, order/augmentation, precision/attention, adapter/embedding sources, cache semantics, losses, optimizer/scheduler, batch, and eval/checkpoint cadence remain identical.
- [ ] 2.10 Gate Wave 2 with artifact/trainer/eval/checkpoint/inference/config tests, a public-entrypoint five-step one-process Accelerate smoke with exactly five train and two eval rows, real adapter-only and adapter-plus-delta load round trips, a two-rank injected rank-zero save-failure subprocess proving shared failure/no hang/no alias/no partial commit, file/byte inventory assertions, removed-surface residue searches, strict OpenSpec validation, and separate standards plus intent/contract audits with no unresolved P0/P1.

## 3. Wave 3 — Rebuild-Only Packing Cache

- [ ] 3.1 Refine Wave 3 into file-owned slices and add failing direct-reader tests for old-version, partial, corrupt, mismatched expected fingerprint, invalid chunk hash, chunk gap, and count mismatch behavior.
- [ ] 3.2 Increment the cache format version and require every manifest/rank/all-step load API to accept or derive the expected semantic fingerprint and verify current version, matching fingerprint, complete status, contiguous chunks, declared counts, and chunk hashes before returning payloads.
- [ ] 3.3 Delete legacy manifest tolerance, optional old provenance handling, cache migrations, and old payload readers; make deterministic rebuild the sole recovery path for every invalid cache state.
- [ ] 3.4 Preserve semantic determinants across data content, template/order/augmentation, seeds, tokenizer/processor, packing budget, supervision, and Qwen position/FA2/forward sources while keeping worker count operational and order-neutral.
- [ ] 3.5 Keep compact augmentation/materialization provenance in the cache manifest, bind only the format/fingerprint/determinant digest into `run.json`, and prove cache deletion after a run does not erase its consumed-materialization identity.
- [ ] 3.6 Gate Wave 3 with cache/augmentation/packing tests, deterministic cross-worker comparison, mismatch/corruption rebuild execution probes, strict OpenSpec validation, and separate standards plus intent/contract audits with no unresolved P0/P1.

## 4. Wave 4 — Multi-Rank Evidence And Accepted Snapshot

- [ ] 4.1 Refine Wave 4 with exact smoke configs, supported hardware/distributed type, representative existing checkpoint root, expected final file inventory, disk-measurement command, and stop conditions.
- [ ] 4.2 Run the public-entrypoint multi-rank Accelerate smoke and verify Accelerator-owned rank identity, identical run descriptor on every rank, symmetric checkpoint barriers, synchronized update decisions, exactly one logical run directory/logging stream/checkpoint tree, and no rank-suffixed duplicates.
- [ ] 4.3 Load a representative existing adapter and optional selected-token delta plus the newly written multi-rank payload through the real inference engine using explicit payload paths and without consulting `checkpoint-final`, `checkpoint.json`, or `checkpoint_handoff.json`; verify exact standard PEFT and embedding-delta identity guarantees claimed by the specs.
- [ ] 4.4 Compare file count and non-checkpoint bytes against a representative current run family and confirm absence of `metrics/`, `eval/` scalar summaries, `receipts/`, per-step Qwen files, progress streams, full-history training results, handoff manifests, and duplicate rank trees.
- [ ] 4.5 Run targeted tests followed by the relevant broader training/runtime/artifact/config/inference suite; record exact commands, passes, skipped hardware/evidence, and artifact roots in the change verification evidence.
- [ ] 4.6 Update current `docs/` architecture, implementation map, artifact/operator guidance, and catalog only after implementation evidence passes; do not add dynamic proposal/progress material under `docs/architecture/`.
- [ ] 4.7 Run final strict OpenSpec validation, documentation link/catalog checks, `git diff --check`, independent fixed-point engineering and intent/contract audits, and resolve or explicitly gate every P0/P1.
- [ ] 4.8 Present the completed change for user acceptance; only after acceptance sync delta specs, archive the OpenSpec change, promote accepted commits to `main`, and refresh the development worktree according to repository policy.
