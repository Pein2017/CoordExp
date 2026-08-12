## Context

See [proposal.md](proposal.md) for motivation and scope. This change starts only from the exact commit at which `reconcile-coordexp-swift-training-contracts` is verified, synchronized, archived, and committed. That commit must establish that stable `coordexp-swift-packing-forward` supports only explicit `synchronous|overlapped`, while `legacy_fused` and the provider environment override are unsupported live residue. An incomplete archive or a moving branch tip is not a valid predecessor.

The current implementation has several real seams, but most of their orchestration is co-located in `src/training/pipeline.py`:

- `run_training_pipeline(...)` spans model-free launch discovery, rank-converged preflight, run initialization, Accelerator construction, model/runtime assembly, cache hydration, exact-resume choreography, trainer callback construction, execution, and terminal publication.
- `_run_rank_converged_phase(...)` plus the bounded rank-report gatherers already form a control-plane protocol, but the protocol is expressed as free functions and mutable dictionaries inside the pipeline.
- `prepare_training_pack_caches(...)` and model-free cache admission use the same cache primitives as training, but live in the training assembly owner.
- `_train_logging_handler(...)` is already one completed-step callback boundary with a stable reduction and append handshake.
- `RunWriter` is the correct public rank-zero facade, but pure schema validation and state-transition logic share its filesystem module.
- `PACKING_CACHE_DETERMINANT_OWNERS` binds `micro_step_runtime_config` to all 6,588 lines of `pipeline.py` and `micro_step_schema` to all of `supervised_trainer.py`, although the content identities are only a three-field runtime projection and one frozen dataclass schema.
- `TokenAtom` owns `causal_logits_position`, but `supervised_trainer.py` derives the sequence-wide selected positions and `forward_input_provider.py` imports that private trainer helper.
- production code imports strict JSON, model-weight, and artifact-target identity helpers from the 10,136-line historical Wave-2 `src/qwen/parity.py` module.

All strict config models are frozen. The existing trainer callback interfaces, `SupervisedMicroStep` record, rank convergence boundary, cache preparation CLI, and `RunWriter` facade are therefore sufficient seams; this design does not add an event system, registry, container, or backend abstraction.

## Goals / Non-Goals

**Goals:**

- Make `src/training/pipeline.py` a small compatibility facade that builds an immutable model-free plan, establishes a bounded control plane, and runs one training session.
- Give rank convergence, cache workflow, cached micro-step identity, completed-step reporting, session lifecycle, generic content identity, and RunWriter's pure internals one explicit owner each.
- Preserve supported-mode collective order, failures, config values, cache payloads, training results, logging rows, run/checkpoint files, and public import paths, except for the declared removal of legacy provider selection and the one intentional cache identity turnover.
- Make cache invalidation proportional to cached-payload semantics rather than unrelated orchestration edits.
- Use characterization tests and interface-level TDD before every move, then delete the replaced implementation and compatibility residue.

**Non-Goals:**

- No telemetry fields, logging cadence, TensorBoard sink, ETA, loss terms, loss weights, tokenizer strategy, packing policy, optimizer behavior, DDP semantics, checkpoint contract, or exact-resume contract expansion.
- No TorchTitan/TorchTune/Megatron/VERL/Transformers Trainer adoption, FSDP, event bus, plugin registry, generic lifecycle framework, dependency-injection container, or alternative training backend.
- No claimed speedup. Reduced cache churn and smaller ownership surfaces are maintainability outcomes; any runtime improvement requires a separate controlled measurement.
- No wholesale breakup of cohesive long modules such as the packing planner, loss runner, Qwen forward implementation, or training-state serializer.

## Decisions

### 1. Preserve one facade and use a one-way import graph

The canonical import direction will be:

```text
src/train.py
  -> src/training/pipeline.py              # public facade only
       -> execution_plan.py                # frozen model-free decisions
       -> control_plane.py                 # rank convergence and close
       -> cache_workflow.py                 # prepare/admit/hydrate orchestration
       -> session.py                        # model/runtime/trainer lifecycle
            -> reporting.py                # completed-step callback
            -> supervised_trainer.py
            -> exact_resume.py
            -> artifacts/RunWriter + CheckpointWriter

micro_steps.py -> supervision/tokens.py + losses/vocab.py
pack_cache.py  -> cache_contract.py + micro_steps.py
qwen/parity.py -> artifacts/identity.py    # compatibility re-exports
```

Leaf/domain modules MUST NOT import `pipeline.py` or `session.py`. `control_plane.py`, `execution_plan.py`, `cache_contract.py`, `micro_steps.py`, `reporting.py`, and `artifacts/identity.py` MUST NOT import `pipeline.py`. `cache_workflow.py` may depend on cache/data/Qwen-frontend primitives and the control-plane interface, but MUST NOT load a model or import `session.py`. `session.py` may compose all lower owners but MUST NOT be imported by them.

`src/training/pipeline.py` retains `run_training_pipeline(...)` and a compatibility re-export of `prepare_training_pack_caches(...)`; `src/train.py` therefore remains unchanged at the public call site. New internal callers import the owning module directly.

Alternative considered: retain a large pipeline and add many forwarding helpers. Rejected because it preserves hidden ownership and makes dependency cycles likely. A framework-style phase registry was also rejected because the phase order is fixed, compatibility-sensitive, and already explicit.

### 2. Define a small immutable model-free execution plan

`src/training/execution_plan.py` will own:

```python
@dataclass(frozen=True)
class TrainingExecutionPlan:
    resolved_config: ResolvedTrainConfig
    repo_root: Path
    launch_rank: int
    launch_world_size: int
    measurement_context: Mapping[str, Any]
    entry_started_at: str
    entry_started_monotonic: float
    entry_resources: Mapping[str, Any]

def build_training_execution_plan(
    config_path: str | Path,
    *,
    measurement_context: Mapping[str, Any] | None,
) -> TrainingExecutionPlan: ...
```

Construction loads and freezes strict config, resolves the repository root and model-free launcher identity, validates/copies the bounded measurement context, and captures entry evidence. It MUST NOT construct Accelerator, load model weights, create a run directory, build a cache, tokenize data, or perform a collective. The plan is a value passed to the next owners, not a service locator; it contains no callback registry or mutable lifecycle state.

Alternative considered: a large `PipelineContext` containing writer, model, runtime, cache, callbacks, and mutable counters. Rejected because it merely renames the current implicit bag of state and permits every phase to depend on everything.

### 3. Make rank convergence a bounded `RankControlPlane`

`src/training/control_plane.py` will own the current fixed-frame CPU rank-report transport, report validation/normalization, phase convergence, resource convergence, and gatherer cleanup:

```python
class RankControlPlane:
    @classmethod
    def open(cls, *, rank: int, world_size: int) -> "RankControlPlane": ...

    def converge(
        self,
        phase: str,
        body: Callable[[], T],
        *,
        local_details: Callable[[], Mapping[str, Any] | None] | None = None,
        receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> T: ...

    def bind_accelerator(self, accelerator: Any) -> None: ...
    def close(self) -> None: ...
```

`open` establishes only the existing model-free gatherer. `bind_accelerator` validates the admitted rank/world-size pair and replaces the transport only at the current post-Accelerator boundary. `converge` preserves current phase names, report schemas, resource snapshots, timeout/size bounds, rank ordering, exception selection, and receipt sinks. `close` is idempotent and called from the facade's `finally` path.

The control plane owns transport and failure convergence, not the meaning or order of training phases. `pipeline.py`, `cache_workflow.py`, and `TrainingSession` call it in the current order. This keeps collective ordering visible in ordinary code and prevents a generic event dispatcher from hiding it.

Alternative considered: pass a bare `gather_status` callable into every function. Rejected because lifetime, pre/post-Accelerator rebinding, validation, and close remain split across callers. A generic distributed backend interface was rejected because Accelerate replicated DDP remains the only backend.

### 4. Give cached micro-steps and determinant projections narrow owners

`src/training/micro_steps.py` becomes the canonical owner of `SupervisedMicroStep` and a pure `supervised_micro_step_schema_identity()` function. `src/training/supervised_trainer.py` and `src/training/__init__.py` re-export `SupervisedMicroStep` so existing supported and historical imports continue to resolve. The record fields, order, annotations, defaults, and frozen status are protected. Old immutable pickle bytes remain untouched and readable through an explicit historical module-path allowlist; new publications use canonical module path `src.training.micro_steps`, so their pickle bytes are intentionally different and must never be characterized as byte-identical to the old payload.

`src/training/cache_contract.py` owns only:

```python
def micro_step_runtime_config_identity(config: TrainConfig) -> dict[str, Any]: ...
```

It returns the current exact three-field projection for `fa2_model_dtype`, `capture_fa2_branch`, and `require_fa2_branch_proof`. It must not import the pipeline or cache workflow.

The determinant registry changes:

```text
micro_step_schema         -> src/training/micro_steps.py
micro_step_runtime_config -> src/training/cache_contract.py
```

No other owner is narrowed merely to reduce hash churn. Each determinant must still bind the complete source that can change its semantic content. The new aggregate/fingerprint is expected to differ for at least four explicit determinant-source changes:

1. `supervision_tokens`, because `src/supervision/tokens.py` gains canonical causal-position selection;
2. `micro_step_runtime_config`, because its owner path/hash moves from `pipeline.py` to `cache_contract.py` while its semantic projection stays equal;
3. `micro_step_schema`, because its owner path/hash moves from `supervised_trainer.py` to `micro_steps.py` while its field schema stays equal; and
4. `cache_serializer`, because `pack_cache.py` changes the determinant registry and restricted historical/canonical pickle allowlist.

Registry schema, semantic content identities, manifest validation, and immutable publication remain unchanged. New cached payload bytes are allowed to differ only at the characterized pickle class module path; decoded micro-step values and all non-pickle protected content remain equivalent. Any additional changed determinant or byte difference is blocking until explained in this design.

Alternative considered: hash individual symbols by AST/source extraction. Rejected because symbol extraction is brittle across decorators/imported types and would create a second source-identity mechanism. Small explicit owner files use the existing file hashing contract.

### 5. Extract cache workflow without weakening `pack_cache.py`

`src/training/pack_cache.py` remains the low-level owner of determinant construction, fingerprinting, immutable publication, safe restricted deserialization, manifest validation, and rank/eval payload loading. `src/training/cache_workflow.py` owns the higher-level operations currently in `pipeline.py`:

- `prepare_training_pack_caches(config_path)` and its preparation receipt assembly;
- absent-target build orchestration, multi-worker render/tokenize/pack materialization, and split aggregation;
- model-free train/eval cache fingerprint resolution and admission;
- rank-local train/eval hydration and image-processor attachment;
- conversion of low-level cache failures into the current actionable preparation command and bounded rank diagnostic.

The workflow may use small frozen result records (`CachePreflight`, `HydratedTrainingInputs`) only for the exact bundles currently returned as untyped mappings. These records expose named fields and `to_receipt_dict()` projections; they do not abstract storage backends or cache versions.

Training remains fail-closed on a missing/invalid cache. Cache preparation remains a separate single-process command and MUST NOT be hidden inside distributed/model startup. The standard build remains one offline multi-worker tokenization/materialization pass.

The cache CLI additionally exposes `src.prepare_train_cache --require-all-hit`.
The flag is a fail-before-build mode, not a preflight hint: it may only resolve
the train/eval fingerprints and targets, admit and fully validate both existing
targets, and publish the named verification receipt. If either target is
missing or invalid, the invocation MUST fail before entering any render,
tokenize, pack, absent-target build, temporary-publication, or immutable-
publication path. A successful discovery check performed before the process
starts is insufficient because the target can change between discovery and
use; enforcement therefore lives inside the invoked cache workflow. The
ordinary no-flag route remains the sole build-capable route.

Alternative considered: combine cache preparation and training into a streaming runtime data service. Rejected because it changes memory, worker, determinism, failure, and performance semantics and conflicts with the agreed one-time multi-worker preparation route.

### 6. Move causal position selection to `TokenSequence`

`TokenSequence` will expose:

```python
def causal_logits_positions(self) -> tuple[int, ...] | None: ...
```

It returns the current sorted unique `TokenAtom.causal_logits_position` values, or `None` for an empty atom set. `supervised_trainer.py` and `forward_input_provider.py` call this domain method. The private `_logits_positions_to_keep` trainer helper and the provider's private import of it are deleted.

The atom property, ordering, uniqueness, empty behavior, forward arguments, and artifact representation do not change. This move happens before determinant owners are frozen because `src/supervision/tokens.py` is correctly a cache determinant owner.

Alternative considered: move the helper into Qwen forward. Rejected because selected causal positions are derived entirely from supervision atoms, not from a model-specific forward implementation.

### 7. Move production identity machinery out of historical parity ownership

`src/artifacts/identity.py` becomes the domain-neutral owner for the existing generic operations used by production:

- strict canonical JSON bytes and JSON/file SHA256;
- absent-target validation and atomic strict-JSON publication used by `src/prepare_train_cache.py`;
- bounded base-model weight identity, validation, equality, and hash execution policy;
- repository and source-owner identity.

The move preserves current payload schemas, sorting/encoding, byte bounds, file-stability checks, symlink rejection, aggregate digests, worker bounds, exception codes, and artifact bytes. `src/training/input_attestation.py`, `src/training/session.py`, and `src/prepare_train_cache.py` import the neutral owner.

`src/qwen/parity.py` imports and re-exports the moved public symbols under their existing names. Historical probe scripts and immutable artifacts are not rewritten. Parity-only comparison, tolerance, plan, receipt, gradient, and Qwen attestation logic remains in `qwen/parity.py`. The compatibility test must prove the old and new import paths resolve to behaviorally identical functions and preserve caught contract errors used by historical readers.

Alternative considered: duplicate the helpers and gradually allow implementations to diverge. Rejected because two content-identity implementations would silently break evidence comparison. Moving every parity function was also rejected because numerical parity remains a cohesive historical domain.

### 8. Encapsulate the initialized run in `TrainingSession`

`src/training/session.py` will own the mutable, model-bearing lifecycle after the control plane has admitted model-free inputs:

```python
class TrainingSession:
    def __init__(
        self,
        *,
        plan: TrainingExecutionPlan,
        control_plane: RankControlPlane,
        writer: RunWriter | None,
        run_identity: RunIdentity,
        cache_preflight: CachePreflight,
        admitted_policies: Mapping[str, Any],
    ) -> None: ...

    def run(self) -> dict[str, Any]: ...
    def fail(self, error: BaseException) -> None: ...
    def close(self) -> None: ...
```

The session owns the current lifecycle counters, Accelerator/model/adapter/embedding/loss/optimizer/runtime assembly, cache hydration, exact-resume admission/restoration/publication callbacks, eval/checkpoint/final handlers, provider lifetime, run finalization, and profile-sync policy reset. `run` returns the existing facade result mapping. `fail` preserves best-effort failed-phase/eval-summary/finalization behavior without swallowing the primary exception. `close` is idempotent and never performs a success finalization.

The facade creates the writer and pre-model receipt ownership at the same current point, binds Accelerator through the control plane, then constructs exactly one session. Session phase calls remain in a literal ordered method body. There is no subclassing or pluggable phase list.

Alternative considered: one class per phase. Rejected as pass-through layering with no independent invariant. The selected session boundary has a real resource lifetime and terminal success/failure protocol.

### 9. Use `CompletedStepReporter` for the existing logging callback only

`src/training/reporting.py` owns:

```python
class CompletedStepReporter:
    def __init__(
        self,
        *,
        writer: RunWriter | None,
        lifecycle: MutableMapping[str, Any],
        runtime: Any,
        resource_collector: Callable[[], Mapping[str, Any]],
    ) -> None: ...

    def __call__(self, observation: CompletedStepObservation) -> None: ...
```

It moves the current `_train_logging_handler` behavior verbatim: lifecycle counters, loss/scheduler/timing/resource scalar extraction, runtime gather, accuracy-stat validation, rank resource projection, current row keys, rank-zero append and outcome broadcast, warmup-qualified measurement accumulation, and first-step phase completion. The existing row remains written after every completed planned step. This owner exposes no configurable sinks or metric registry.

`add-coordexp-swift-training-observability` may later extend reporting from this seam, but this change does not pre-build that feature or change an artifact byte.

Alternative considered: a general observer/event bus shared by train/eval/checkpoint. Rejected because the three events have different collective and failure semantics; generalization would obscure rather than simplify them.

### 10. Keep `RunWriter` public and extract only pure internals

`src/artifacts/run_writer.py` remains the import and I/O facade for `RunWriter` and `admit_exact_resume_checkpoint_publication`. Its method names, arguments, paths, rank-zero ownership, append behavior, atomic replacement, collision behavior, and returned values remain unchanged.

Two private implementation owners are justified by existing code shape:

- `src/artifacts/run_schema.py` owns strict JSON normalization/serialization, bounded detail validation, timestamp/number/lineage validation, measurement payload construction, and logging-row normalization.
- `src/artifacts/run_state.py` owns pure run-state transitions and exact-resume checkpoint-publication admission. Functions receive a mapping and return a new validated mapping; they perform no filesystem I/O.

`RunWriter` remains the only owner that reads/writes run files and sequences transitions. The facade re-exports the existing standalone exact-resume admission function. Tests compare complete bytes for representative initialization, logging, phase, checkpoint, best/final, failure, and finalization paths.

Alternative considered: introduce a repository/store interface and inject it into `RunWriter`. Rejected because only the local filesystem exists and such an interface would be speculative. Keeping every helper in `run_writer.py` was rejected because it prevents pure state/schema tests and leaves the facade unnecessarily broad.

### 11. Delete legacy provider selection, retain explicit experimental overlap

After the prerequisite change establishes the supported contract:

- `ForwardInputProviderMode` becomes `Literal["synchronous", "overlapped"]`;
- `legacy_fused`, its trainer-owned direct-build branch, `None` provider disposition, config fixtures, run-writer acceptance, and tests are deleted;
- `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`, its allowlisting, rank-local resolution, deprecated source receipt, and environment-driven semantic override are deleted;
- strict config is the only selector, `synchronous` remains the default/reference implementation, and `overlapped` remains an explicit experimental config value with the existing CPU-only depth-one and bounded shutdown behavior.

The direct config value may still be rank-converged as part of the existing config/provenance phase, but no environment value may replace it. Unsupported historical configs fail strict validation rather than silently falling back.

Alternative considered: retain hidden aliases indefinitely. Rejected because they keep three code paths and permit runtime semantics to differ from the resolved config. Removing `overlapped` was also rejected because it is an explicit, bounded, already-tested experimental comparison arm.

### 12. Freeze a compatibility ledger before implementation

The implementation gate uses this ledger; “preserved” means exact for supported inputs unless the row declares an intentional exception.

| Surface | Disposition | Required evidence |
| --- | --- | --- |
| `src.train -> run_training_pipeline(config_path)` and result mapping | Preserve | entrypoint and pipeline assembly characterization tests |
| `src.training.pipeline.prepare_training_pack_caches` | Preserve as compatibility re-export; canonical owner moves | CLI plus direct-import tests |
| `SupervisedMicroStep` fields/order/defaults/frozen/pickle allowlist | Preserve values and historical readability; canonical owner moves, and new pickle bytes intentionally encode `src.training.micro_steps` | schema identity, old-byte restricted load, new-byte module-path assertion, and decoded-value equality through both import paths |
| `TokenSequence` artifacts and selected causal positions | Preserve; method owner moves | exact position/forward-input tests including empty atoms |
| rank phase order, report frames, resource receipts, error convergence, collective calls | Preserve | single-/multi-rank control-plane contract tests and ordered-call assertions |
| cache determinant semantic payload, manifest, payload serialization, immutability | Preserve except the four declared determinant-source changes, aggregate/fingerprint, and canonical pickle module-path bytes | exact four-entry determinant diff, old/new pickle characterization, decoded equality, plus new-target admission tests |
| current cache target | Intentionally not reused after owner move; never mutated/deleted | old target remains untouched; exactly one new absent fingerprint is published |
| `RunWriter` public methods/imports and run/checkpoint/log bytes | Preserve | golden byte comparisons from pre-move characterization fixtures |
| completed-step `logging.jsonl` rows and append-failure broadcast | Preserve | reporter vs current-handler exact row/error tests |
| generic identity schemas/digests/bounds/errors and `src.qwen.parity` imports | Preserve; production owner moves and old path re-exports | cross-import equality, malformed-input, and historical reader tests |
| supported `synchronous`/`overlapped` provider behavior and receipts | Preserve | provider equivalence and bounded-shutdown tests |
| `legacy_fused` and provider environment override | Delete intentionally after reconcile | strict rejection and residue search; no fallback |
| loss, optimizer/scheduler, eval/checkpoint cadence, exact-resume implementation, artifact schemas | Preserve; no new promise | targeted suites and production-shaped vertical smoke |

Any unexplained ledger difference stops the wave. Tests must not update expected bytes merely because the refactor produced new bytes; only the declared determinant/manifest turnover, canonical pickle module-path bytes, and legacy-selector receipt absence may change.

### 13. Make waves independently revertible and freeze verification once

Wave 0 freezes one exact test-command manifest against the predecessor commit.
Each entry records an ID, full argv, cwd, `conda` environment, relevant bounded
environment selectors, test node list/order, expected fixture roots, timeout,
and whether it is CPU-only or launch-bearing. Later wave gates invoke manifest
IDs; an added or changed command requires an explicit manifest revision and
baseline rerun, not an ad hoc suite expansion.

Each implementation wave starts from the exact prior wave commit, changes only
its declared owner surfaces, deletes replaced code in the same wave, and ends
in one independently testable commit. Reverting that commit must restore the
prior wave without source edits from a later wave or cache mutation. Cross-wave
half-moves and squash-only rollback are prohibited. Independent review happens
at three decision points only: entry (Wave 0), immediately before the first
costly cache/GPU action (Wave 6), and final completion (Wave 8). Ordinary wave
gates use the frozen manifest, characterization diffs, residue checks, and
strict OpenSpec validation without duplicating two broad audits each time.

Costly actions are separate authority surfaces. Wave 7 cache materialization
and Wave 8 GPU smoke each need fresh user authorization bound to the exact
commit, full command, immutable absent target/artifact root, and a numeric
resource packet. The cache packet freezes two separate full argv vectors and
two absent receipt paths: one build-capable preparation invocation and one
`--require-all-hit` verification invocation. It declares workers, wall timeout,
CPU RSS, new bytes, free disk, and split count. The second invocation receives
no cache-materialization authority: it may validate the already published
train/eval targets and write only its named receipt. A successful preflight
cannot authorize fallback construction, so target drift between the two
invocations fails before render/tokenize/pack/publication instead of causing a
second build. The GPU packet fixes `world_size=2`, at most two GPUs,
one planned/applied optimizer step, and declares config-derived upper bounds
for model forwards, collective count, wall time, CPU RSS, GPU high-water mark,
and artifact bytes. Command/commit drift, occupied targets, inadequate
headroom, OOM, timeout, or any bound exceedance stops without retry and requires
a new packet.

## Risks / Trade-offs

- **[Collective order changes while moving control flow]** → Capture an ordered phase/collective trace before extraction, keep literal call order, and require single-rank plus multi-rank control-plane tests before deleting old helpers.
- **[Frozen dataclass moves break pickle compatibility]** → Keep compatibility re-exports, explicitly allow both historical and canonical module paths while old immutable caches are readable for diagnosis, and prove old-byte load plus new canonical-module bytes before publishing the new cache. Never expect old/new pickle byte equality.
- **[Generic identity move changes exception type, code, or digest]** → Characterize malformed and successful cases first; re-export one implementation rather than wrap/duplicate logic; retain schemas and error codes in this change.
- **[RunWriter split changes JSON ordering or atomicity]** → Keep all filesystem operations in `RunWriter`; move only pure logic and compare exact file bytes plus collision/failure behavior.
- **[The session becomes another oversized object]** → Limit it to the model-bearing resource lifetime and fixed phase choreography. Move cache, reporting, control-plane, identity, and artifact-pure logic out; do not create phase subclasses.
- **[Narrow determinant owners omit a real semantic dependency or hide expected churn]** → Narrow only the two evidenced overbroad entries, retain every other owner, and require an exact determinant diff containing at least the four declared source changes; any fifth change or missing declared change blocks publication pending explanation.
- **[Multiple refactor commits create several unusable cache fingerprints]** → Do not materialize during owner migration. Freeze and test the final determinant registry first, then perform one absent-target build.
- **[A preflight-to-hit TOCTOU race silently triggers a second build]** → Implement and test `--require-all-hit` as an in-workflow fail-before-build gate, freeze its argv and separate receipt in the same authorization packet as the build argv, and give the verification invocation no materialization authority.
- **[Deleting the environment override surprises an operator]** → The reconcile change must mark it unsupported first; strict config is persisted and discoverable, and unsupported values fail early with no silent translation.
- **[Compatibility shims become permanent duplicate authority]** → Shims contain imports/re-exports only, are listed in the ledger, and are excluded from production imports by a residue test.
- **[A passing smoke is misreported as an efficiency result]** → Acceptance claims only behavior preservation and executed integration; timing/resource rows diagnose regressions but do not support a speedup claim.

## Migration Plan

1. Require `reconcile-coordexp-swift-training-contracts` to be verified, synchronized, archived, and committed. Record and checkout that exact predecessor commit; an incomplete archive or later branch tip is not a substitute.
2. Freeze compatibility characterization fixtures, the ordered execution/collective trace, and the exact test-command manifest before moving code. Add import-direction and production-import residue checks.
3. Move generic identity helpers, `SupervisedMicroStep`, and `TokenSequence.causal_logits_positions()` first. Keep old import paths as re-exports and run their focused suites.
4. Extract `RankControlPlane` and `TrainingExecutionPlan`; switch the facade to them while leaving initialized training behavior in place. Delete the replaced free-function transport only after order/error equivalence passes.
5. Extract `cache_contract.py` and `cache_workflow.py`; update determinant ownership and prove semantic payload equivalence. Do not materialize a cache yet.
6. Extract `CompletedStepReporter` and RunWriter's pure schema/state internals, preserving the public facade and byte fixtures.
7. Introduce `TrainingSession`, move the fixed initialized-run choreography, and reduce `pipeline.py` to facade assembly. Delete replaced pipeline helpers instead of leaving forwarding layers.
8. Remove `legacy_fused` and `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`; retain strict `synchronous` and explicit experimental `overlapped`. Run config/provider/resume/artifact compatibility gates and residue searches.
9. Freeze the final owner graph and obtain the fresh cache-action authorization described in Decision 13. For `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`, compute the old and new determinant identities, assert each required new split target is absent and each old target is untouched, and freeze both the build argv/receipt and the separate `--require-all-hit` argv/receipt. Run exactly one authorized multi-worker preparation invocation. That invocation may publish the config's train and eval targets, but no intermediate owner-graph fingerprint may be materialized. Then invoke only the frozen `--require-all-hit` command, which has no materialization authority and must fail before render/tokenize/pack/publication on any missing or invalid target. Record both receipts and both split identities; do not repair or delete an occupied target.
10. Run the frozen final CPU manifest, strict OpenSpec validation, then obtain separate fresh authorization for a production-shaped two-rank vertical smoke covering cache admission, one applied step, eval/checkpoint/finalization, and artifact consumers. Compare protected artifacts/rows against the ledger and obtain the final independent overdesign/residue audit.

Rollback is code-only until step 9: revert the implementation while leaving existing caches untouched. After the new cache is published, rollback continues to use the old code/old fingerprint; the new immutable target remains unreferenced evidence and is not rewritten or deleted by rollback. No rollback path mutates either cache target or historical artifacts.

## Open Questions

None. File ownership, import direction, compatibility exceptions, migration order, and acceptance evidence are fixed by this design; discoveries that would alter them require updating this change before implementation proceeds.
