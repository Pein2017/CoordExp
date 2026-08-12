## Why

The current 6,588-line `src/training/pipeline.py` owns model-free preflight, rank convergence, cache admission and hydration, model/runtime assembly, exact-resume choreography, reporting, evaluation, checkpoints, and finalization, while the pack-cache determinant registry hashes that whole file and `src/training/supervised_trainer.py` for two narrow cached-payload concerns.  After `reconcile-coordexp-swift-training-contracts` settles the supported contracts, this change gives those responsibilities explicit owners so unrelated orchestration edits stop causing cache churn and production code no longer depends on generic identity helpers embedded in the 10,136-line Wave-2 parity module.

## What Changes

- Sequence this work after `reconcile-coordexp-swift-training-contracts` is fully verified, synchronized, archived, and committed. Bind this change's baseline and every characterization fixture to that exact predecessor commit; the preceding change owns contract correction, while this change owns only decomposition and removal of residue that predecessor classified as unsupported.
- Keep `run_training_pipeline(...)` as the training facade, but have it assemble and execute an immutable, model-free execution plan rather than carry mutable orchestration state through one module.
- Extract a bounded `RankControlPlane` for pre-model rank convergence and failure receipts, and a `TrainingSession` for model/runtime resources, lifecycle, exact-resume, evaluation, checkpoint, and finalization choreography.  Neither abstraction introduces a second backend or changes collective ordering.
- Extract a `CompletedStepReporter` from the current train logging callback.  It MUST retain the existing completed-step row schema, reduction behavior, rank-zero append handshake, lifecycle counters, and failure semantics; new ETA, TensorBoard, logging cadence, or metric fields belong to `add-coordexp-swift-training-observability`, not this change.
- Keep `RunWriter` as the public artifact facade.  Internal collaboration may be decomposed, but existing callers, file names, schemas, strict serialization, atomic publication, and rank-zero ownership remain intact.
- Separate cache preparation/admission/hydration orchestration from training-session assembly, and replace the determinant registry's whole-file ownership of `pipeline.py` and `supervised_trainer.py` with narrow owners for the cached micro-step runtime projection and immutable micro-step schema. The final fingerprint is expected to turn over for at least four declared determinant-source changes: `supervision_tokens`, `micro_step_runtime_config`, `micro_step_schema`, and `cache_serializer`. Moving `SupervisedMicroStep` also intentionally changes new pickle module-path bytes to `src.training.micro_steps`; old immutable payload bytes remain untouched and readable. After all determinant owners settle, one separately authorized offline multi-worker materialization may publish one new absent train/eval target pair; no intermediate fingerprint is built.
- Move canonical JSON/content, model-weight, repository, and source-owner identity helpers out of `src/qwen/parity.py` into a domain-neutral owner.  Production callers use the neutral owner, while compatibility re-exports keep historical parity probes and artifact readers able to consume prior evidence without rewriting immutable history.
- Move canonical causal-logit-position selection to `TokenSequence`, whose atoms already own `causal_logits_position`; trainer and forward-input-provider callers consume that one domain result without changing selected positions or their order.
- **BREAKING only for unsupported residue:** remove the deprecated `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` override and non-current `legacy_fused` path after the exact predecessor commit records that stable `coordexp-swift-packing-forward` supports only `synchronous|overlapped`. Keep `synchronous` as the supported reference path and `overlapped` as an explicit experimental selection with its existing depth-one, CPU-only semantics.
- Preserve supported data order, rendered/tokenized content, packing, decoded cached micro-step values, Qwen forward inputs, loss computation and reduction, optimizer/scheduler/DDP behavior, exact-resume behavior, collective order, eval/checkpoint schedule, and training artifact contracts. Artifact serialization remains byte-identical except for the declared determinant/manifest turnover, new canonical `SupervisedMicroStep` pickle module-path bytes, and receipts that necessarily reflect removal of the legacy selector.
- Do not adopt TorchTitan, TorchTune, Megatron, VERL, Transformers Trainer, or any other external framework as the execution owner.  An event bus, plugin registry, dependency-injection container, FSDP, new telemetry, loss-semantic changes, runtime-tokenization changes, and performance claims are explicitly out of scope.

## Capabilities

### New Capabilities

- None. This is an internal architecture refactor plus deletion of behavior already classified as unsupported by the exact predecessor commit. Stable `coordexp-swift-packing-forward` already limits supported provider modes to `synchronous|overlapped`; therefore this change does not alter a stable capability and keeps `skip_specs: true`.

### Modified Capabilities

- None. Existing stable requirements remain authoritative, and `reconcile-coordexp-swift-training-contracts` owns any contract correction needed before this refactor begins.

## Impact

- **Training orchestration:** `src/training/pipeline.py`, `src/training/supervised_trainer.py`, `src/training/forward_input_provider.py`, and focused modules introduced under `src/training/`.
- **Cache identity and workflow:** `src/training/pack_cache.py`, its determinant-owner registry, cache preparation/admission/hydration tests, and one deliberately new immutable cache target after source ownership stabilizes.
- **Semantic ownership:** `src/supervision/tokens.py` for causal-logit position selection, plus a domain-neutral identity owner replacing production imports from `src/qwen/parity.py`.
- **Compatibility surfaces:** `src/qwen/parity.py` retains historical-reader shims; `src/artifacts/run_writer.py` retains its public facade and serialized contracts.  Tests must prove equivalent supported-mode payloads and artifacts rather than bless broad snapshots generated after the fact.
- **Operations:** planning or implementation approval is not cache/GPU launch authority. No production cache materialization may occur between determinant-owner edits. Once owners settle, the one cache build and later GPU vertical smoke each require a fresh exact-commit/command/target authorization packet with quantitative stop bounds.
- **Dependencies and performance:** no new runtime dependency, backend, or user-visible performance guarantee is introduced.  Any efficiency improvement must be measured separately and is not an acceptance claim for this change.
