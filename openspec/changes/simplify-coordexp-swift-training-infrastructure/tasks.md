This OpenSpec change is the Program. Each numbered task group is a Wave, each
checkbox is a dependency-ordered Slice, and each Wave ends with a verification
plus independent-audit Gate. Wave 1 is the first active implementation wave;
later Waves state bounded outcomes and MUST be refined with
`openspec-update-change` only after the preceding Gate closes. No Wave may
advance with unresolved P0/P1 findings, and no gate may leave both old and new
production ownership paths active.

## 1. Wave 1 — Accelerate Process And Runtime Foundation

Wave 1 execution ownership is split into three non-overlapping implementation
lanes: `src/runtime/train_runtime.py` plus runtime-facing tests; early process
context and shared-run assembly in `src/training/pipeline.py` plus pipeline
assembly tests; and runtime schema/profile migration in `src/config/`, active
`configs/coordexp_swift/{prod,smoke}/`, and config/schedule tests. The Wave gate
and task ledger remain parent-owned so no worker can self-approve its slice.

- [x] 1.1 Refine Wave 1 into file-owned slices and add failing config/runtime tests for Accelerate world size one, replicated multi-GPU DDP, Accelerator-owned rank/world/device identity, and pre-prepare rejection of externally selected FSDP, DeepSpeed, tensor-parallel, or other unsupported distributed types.
- [x] 1.2 Initialize one Accelerator process context early enough for rank zero to resolve collision policy once, broadcast one run descriptor/path to every rank, and prove identical `fail`/`timestamp` collision outcomes without creating rank-suffixed directories.
- [x] 1.3 Collapse the embedded `runtime.backend` branches in `RuntimeConfig`, `TrainRuntime`, pipeline Accelerator construction, and status helpers into the concrete Accelerate path; remove DeepSpeed plugin/config/status code and the implicit `single` fallback without inventing replacement backend interfaces.
- [x] 1.4 Derive effective accumulation and scheduler/runtime setup from Accelerator world size while preserving all-rank denominator/finite decisions, gradient clipping, optimizer order, CoordExp-owned planned-step scheduler order, barriers, and neutral Accelerate accumulation.
- [x] 1.5 Migrate every checked-in active production and smoke profile away from backend/DeepSpeed fields, delete DeepSpeed helper/smoke configs, and compare old/new resolved mappings after removing only the approved backend-field allowlist; every other value MUST match.
- [x] 1.6 Gate Wave 1 with config/runtime/schedule tests, a public-entrypoint one-process Accelerate smoke, an unsupported-wrapper negative subprocess probe, strict OpenSpec validation, `git diff --check`, backend residue searches, and separate engineering-standards plus intent/contract audits with no unresolved P0/P1.

Wave 1 gate evidence (2026-07-11): the integrated runtime/config/pipeline/
trainer/source-gate slice passed 194 tests; the public one-process entrypoint
completed one planned step, two micro-steps, eval, checkpoint, and finalization
at `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step-20260711T122737Z`;
an ambient `ACCELERATE_MIXED_PRECISION=fp16` probe still constructed and
validated the configured `bf16` mode; and a real two-rank `--use_fsdp` launch
failed on both ranks at `runtime.distributed_type_unsupported` before run or
model mutation. Strict OpenSpec validation and `git diff --check` passed. The
only backend residue is intentional deleted-field migration/rejection evidence.
Independent engineering and intent/contract re-audits both approved with no
remaining P0/P1 findings.

## 2. Wave 2 — Atomic Replacement Of The Artifact/Event/Checkpoint Graph

Wave 2 executes in two implementation phases without an intermediate
production commit. Phase 2A has five disjoint owners: the trainer contract in
`src/training/supervised_trainer.py`; the pure eval contract in
`src/eval/forward.py`; the concrete run writer and legacy manager/metric-stream
deletion in `src/artifacts/`; the training logging-schema/profile migration in
`src/config/`, `src/training/schedule.py`, and active train configs; and the
explicit-path inference migration in `src/inference/` plus inference tests.
Phase 2B starts only after those interfaces pass their focused tests: one
checkpoint owner rewrites `src/artifacts/checkpoints.py`, and one integration
owner rewires `src/training/pipeline.py`, deletes every remaining legacy
caller, and owns the full inventory/smoke gate. `src/artifacts/__init__.py` is
integration-owned until exports can switch atomically. No worker may commit or
self-approve; the parent owns the task ledger, cross-lane integration, real
subprocess probes, independent audits, and the single Wave 2 commit.

- [x] 2.1 Refine Wave 2 into file-owned slices and add failing interface tests for the final run inventory, one typed completed-step observation, bounded trainer result, wide train/eval rows, NaN/Inf-to-null normalization with `non_finite_fields`, bounded warning counters, immutable materialization handles, safe best selection, and one canonical checkpoint alias owner.
- [x] 2.2 Replace generic trainer event-name dispatch and full `step_results` retention with one typed `on_completed_step(observation)` callback, direct scheduled handlers, compact counters/latest state, and release of step-local tensors/receipts after consumers finish.
- [x] 2.3 Refactor `ForwardEvalRunner` to return counts, triggers, and one wide reduced scalar mapping without importing artifact-writer or metric-event types; make its rank-zero `logging.jsonl` row the only durable eval scalar schema and remove separate eval-summary metric files.
- [x] 2.4 Implement one concrete rank-zero run writer for atomic `run.json`, one self-contained `resolved_config.json`, direct single-writer `logging.jsonl`, and atomic checkpoint aliases; bind cache format/fingerprint/determinant digest in `run.json` without copying cache payloads, presentations, or selector state.
- [x] 2.5 Implement replicated-DDP checkpoint choreography with a pre-save barrier, one rank-zero staged PEFT save using safe serialization/the configured adapter/`save_embedding_layers=False`, optional compact selected-token delta, required LoRA A/B/DoRA and forbidden full-tensor validation, atomic step commit, and a failure-aware success/error collective that makes every rank continue or raise together while cleaning staging residue.
- [x] 2.6 Make `final.json` and optional `best.json` the sole checkpoint selector owners; test that update-skipped, unsafe, or non-finite steps cannot advance `best.json` by default even with a better eval value.
- [x] 2.7 Remove new handoff generation, neighboring-handoff and `checkpoint-final` metadata resolution, readiness gates, and downstream inference identity/provenance/merge fields/tests that only propagate those metadata objects; migrate supported inference configs/smokes to explicit adapter and optional delta paths, preserve actual loaded base/adapter/delta provenance, and document/test the narrower standard-PEFT identity boundary plus the stronger embedding-delta hash boundary.
- [x] 2.8 Delete `RunArtifactManager`, `MetricStreamEvent`, `TrainingArtifactBridge`, per-step Qwen files, per-rank progress streams, receipt/report registries, resolved YAML, durable schedule receipt, full-history `training_result.json`, and every production import/caller/test of those schemas in the same Wave; do not leave a compatibility facade or dual writer.
- [x] 2.9 Migrate all active configs/tests off `training.logging` cadence and old artifact/debug receipt controls; compare old/new resolved mappings after removing only the complete approved infrastructure-deletion allowlist so seeds, order/augmentation, precision/attention, adapter/embedding sources, cache semantics, losses, optimizer/scheduler, batch, and eval/checkpoint cadence remain identical.
- [x] 2.10 Gate Wave 2 with artifact/trainer/eval/checkpoint/inference/config tests, a public-entrypoint five-step one-process Accelerate smoke with exactly five train and two eval rows, real adapter-only and adapter-plus-delta load round trips, a two-rank injected rank-zero save-failure subprocess proving shared failure/no hang/no alias/no partial commit, file/byte inventory assertions, removed-surface residue searches, strict OpenSpec validation, and separate standards plus intent/contract audits with no unresolved P0/P1.

Wave 2 gate evidence (2026-07-11): the final runtime/training/eval/artifact/
config/inference slice passed 393 tests. The public entrypoint completed five
planned steps and ten micro-steps at
`outputs/smoke/wave2_training_infra/wave2-training-infra-5step-postaudit`,
producing exactly five `train` and two `eval` rows. Its fixed inventory is ten
files: 45,511,096 bytes of inference-required checkpoint payload and 13,270
bytes of run, logging, config, and alias state. Real inference assembly loaded
both its adapter-only payload and its adapter-plus-selected-token-delta payload
from explicit paths. Real two-rank Accelerate probes made both ranks receive
the same `checkpoint.save_failed` after an injected rank-zero staging failure,
with no hang, step directory, staging residue, `final.json`, or `best.json`; a
second probe likewise shared `runtime.logging_append_failed` after injected
rank-zero logging I/O failure. Removed-surface searches, strict OpenSpec
validation, and `git diff --check` passed. Independent engineering-standards
and intent/contract fixed-point audits approved with no unresolved P0/P1; both
non-blocking P2 cleanup requests (replicated same-dataset eval coverage and
direct trainer scheduled-handler dispatch) were also completed before commit.

## 3. Wave 3 — Rebuild-Only Packing Cache

Wave 3 has two non-overlapping implementation owners. The cache-format owner
owns `src/training/pack_cache.py` and direct cache tests: version/API changes,
strict manifest and payload validation, transactional directory publication,
and removal of legacy tolerance. After that interface passes, the integration
owner owns cache callers in `src/training/pipeline.py`, the cache binding in
`src/artifacts/run_writer.py`, and their focused tests: rank-zero reuse/rebuild
selection, waiter behavior, expected-fingerprint propagation, and the durable
identity boundary. Neither owner edits this ledger or commits; the parent owns
the execution probes, cross-worker determinism comparison, independent audits,
task closure, and the single Wave 3 commit.

- [x] 3.1 Refine Wave 3 into the file-owned slices above and add failing direct-reader tests for old-version, partial/incomplete, corrupt JSON/pickle, mismatched expected fingerprint, invalid declared or actual chunk hash, missing/gapped chunks, per-chunk count mismatch, and total-count mismatch behavior.
- [x] 3.2 Increment the cache format version and require every manifest/rank/all-step load API to accept an explicit expected semantic fingerprint and verify current version, matching canonical determinant fingerprint, complete status, mandatory materialization/augmentation provenance, contiguous safe chunk paths, declared per-chunk/total counts, chunk hashes, and current tuple payload shape through an exact-class restricted unpickler before returning.
- [x] 3.3 Delete legacy manifest tolerance, optional old provenance handling, cache migrations, unrestricted pickle loading, and old payload readers; serialize same-root writers with a persistent POSIX advisory lock, publish writes through an isolated sibling staging directory with the manifest last, and make a rank-zero deterministic rebuild plus bounded shared Accelerate outcome followed by strict peer reads the sole recovery path for every invalid cache state.
- [x] 3.4 Preserve semantic determinants across data content, template/order/augmentation, seeds, tokenizer/processor, packing budget, supervision, and Qwen position/FA2/forward sources while keeping worker count operational and order-neutral.
- [x] 3.5 Require compact augmentation/materialization provenance in the current manifest, bind only format/fingerprint/determinant digest (not cache path, manifest path/hash, chunk hashes, or payload metadata) into `run.json`, and prove cache deletion after a run does not erase its consumed-materialization identity.
- [x] 3.6 Gate Wave 3 with direct cache plus augmentation/packing suites, deterministic one-worker versus multi-worker sequence/fingerprint comparison, real mismatch/corruption rebuild and waiter probes, partial-publication residue checks, removed-legacy residue searches, strict OpenSpec validation, `git diff --check`, and separate standards plus intent/contract audits with no unresolved P0/P1.

Wave 3 gate evidence (2026-07-11): 66 focused direct-cache and distributed
resolver tests passed, including malicious-reduce rejection without executing
its side effect, mandatory provenance checks, cache deletion after durable run
binding, and spawned-process overlap at both staging and live-root backup. The
broader runtime/training/eval/artifact/config/inference/augmentation/packing
suite passed 476 tests with one upstream sparse-CSR warning. Real rich-Qwen
materialization with one and two workers produced the same ordered payload and
semantic digest
`d1427121da12a7a35d7050ddbafd840e2d7e969ad3b5db0857d659bce170718b`.
Fresh two-rank Accelerate probes rebuilt both fingerprint-mismatched and
corrupted caches with rank zero reporting `built`, its peer reporting `waited`,
and both consuming pack IDs `[10, 11]`. Injected materialization, staged
validation, and publication failures produced the same
`training.pack_cache_resolution_failed` descriptor on both ranks without a
hang; each clean retry succeeded and left no stage/backup residue. Legacy
reader/path residue searches, strict OpenSpec validation, and `git diff
--check` passed. Independent engineering and intent/contract fixed-point
audits approved with no unresolved P0/P1. The retained double integrity pass
on cache hits and possible duplicate pre-publication work during simultaneous
independent cold starts are deliberate correctness-first P2 trade-offs; they
remain measurement-triggered optimizations rather than new coordination
machinery.

## 4. Wave 4 — Multi-Rank Evidence And Accepted Snapshot

- [ ] 4.1 Refine Wave 4 with exact smoke configs, supported hardware/distributed type, representative existing checkpoint root, expected final file inventory, disk-measurement command, and stop conditions.
- [ ] 4.2 Run the public-entrypoint multi-rank Accelerate smoke and verify Accelerator-owned rank identity, identical run descriptor on every rank, symmetric checkpoint barriers, synchronized update decisions, exactly one logical run directory/logging stream/checkpoint tree, and no rank-suffixed duplicates.
- [ ] 4.3 Load a representative existing adapter and optional selected-token delta plus the newly written multi-rank payload through the real inference engine using explicit payload paths and without consulting `checkpoint-final`, `checkpoint.json`, or `checkpoint_handoff.json`; verify exact standard PEFT and embedding-delta identity guarantees claimed by the specs.
- [ ] 4.4 Compare file count and non-checkpoint bytes against a representative current run family and confirm absence of `metrics/`, `eval/` scalar summaries, `receipts/`, per-step Qwen files, progress streams, full-history training results, handoff manifests, and duplicate rank trees.
- [ ] 4.5 Run targeted tests followed by the relevant broader training/runtime/artifact/config/inference suite; record exact commands, passes, skipped hardware/evidence, and artifact roots in the change verification evidence.
- [ ] 4.6 Update current `docs/` architecture, implementation map, artifact/operator guidance, and catalog only after implementation evidence passes; do not add dynamic proposal/progress material under `docs/architecture/`.
- [ ] 4.7 Run final strict OpenSpec validation, documentation link/catalog checks, `git diff --check`, independent fixed-point engineering and intent/contract audits, and resolve or explicitly gate every P0/P1.
- [ ] 4.8 Present the completed change for user acceptance; only after acceptance sync delta specs, archive the OpenSpec change, promote accepted commits to `main`, and refresh the development worktree according to repository policy.
