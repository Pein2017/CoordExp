# CoordExp-Swift Training Orchestration Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the behavior-preserving orchestration decomposition owned by OpenSpec change [`decompose-coordexp-swift-training-orchestration`](../../../openspec/changes/decompose-coordexp-swift-training-orchestration/proposal.md) while preserving supported training semantics and making cache ownership proportional to cached content.

**Architecture:** OpenSpec is the sole authority for scope, compatibility, and completion: read its [`proposal.md`](../../../openspec/changes/decompose-coordexp-swift-training-orchestration/proposal.md), [`design.md`](../../../openspec/changes/decompose-coordexp-swift-training-orchestration/design.md), and [`tasks.md`](../../../openspec/changes/decompose-coordexp-swift-training-orchestration/tasks.md) before every task. This document supplies only an executable RED-GREEN-REFACTOR order, exact files/interfaces, verification commands, and commit boundaries; any conflict is resolved by updating OpenSpec first, never by treating this plan as a second contract.

**Tech Stack:** Python 3.12 in Conda environment `ms`, PyTorch, Accelerate replicated DDP, pytest, strict YAML config, immutable pickle/JSON cache artifacts, OpenSpec CLI, Git.

## Global Constraints

- Work only from `/data/CoordExp/.worktrees/CoordExp-swift`; every Python and pytest invocation uses the exact `conda run -n ms` commands written below.
- Start only from the exact committed result of `reconcile-coordexp-swift-training-contracts`; record that commit and stop if it does not establish the prerequisite required by the authoritative design.
- Preserve user-owned dirty files, especially `docs/catalog.yaml`, `docs/data/PACKING.md`, `docs/history/README.md`, and `docs/history/cache-retirement/2026-08-12.md`; every `git add` below names only task-owned paths.
- Each task is one independently reviewable, committable, and revertible wave. Do not begin the next task with an unresolved P0/P1 or unexplained compatibility-ledger difference.
- Do not introduce an event bus, plugin registry, dependency-injection container, phase subclass hierarchy, storage abstraction, alternate backend, FSDP, new telemetry, loss change, tokenizer strategy, packing-policy change, or performance claim.
- During Tasks 1-7, never run `src.prepare_train_cache` and never publish any production cache target. Freeze the final owner graph before the single Task 8 materialization.
- Planning or implementation permission is not authority to build caches or launch GPU work. Task 8 cache materialization and Task 9 two-rank smoke each stop for a fresh packet naming the exact commit, command, config, output targets, disk/RSS/worker or GPU/time bounds, and stop conditions. The Task-8 packet freezes separate build and `--require-all-hit` argv/receipt pairs; the latter has no materialization authority and must enforce fail-before-build inside the invoked workflow.
- Existing immutable caches and historical artifacts are read-only: never delete, repair, overwrite, rename, or rewrite them. A rollback changes code only and leaves old and newly published immutable targets intact.
- The only accepted cache transition is the one declared in the authoritative design: the four determinant-source changes and canonical new pickle module path; decoded payloads and all other protected semantics remain equal.
- Commit messages below are proposed future checkpoints. Inspect `git diff --check`, the staged diff, and `git status --short` before each commit; do not stage unrelated files.

---

### Task 1: Pin the predecessor and freeze the compatibility ledger

**OpenSpec owner:** Tasks 1.1-1.5 and Design decisions 1, 12.

**Files:**
- Create: `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/reconcile-baseline.json`
- Create: `tests/training/test_training_module_boundaries.py`
- Create: `tests/training/test_orchestration_compatibility.py`
- Create: `tests/fixtures/training_orchestration/supervised_micro_step_legacy.pkl`
- Modify: `tests/training/test_pipeline_assembly.py`
- Modify: `tests/training/test_pipeline_phase_convergence.py`
- Modify: `tests/artifacts/test_run_artifacts.py`
- Modify: `tests/training/test_pack_cache_determinant_registry.py`

**Interfaces:**
- Consumes: `src.train.main(argv: list[str] | None = None) -> int`, `src.training.pipeline.run_training_pipeline(config_path: str | Path, *, measurement_context: Mapping[str, Any] | None = None) -> dict[str, Any]`, and the compatibility ledger in the authoritative design.
- Produces: byte fixtures and ordered-call traces consumed by Tasks 2-9; test helpers `assert_allowed_training_imports(repo_root: Path) -> None`, `exercise_characterized_pipeline() -> CharacterizedPipeline`, `load_fixture(name: str) -> Any`, and `load_binary_fixture_tree(name: str) -> Mapping[str, bytes]`; a receipt containing `reconcile_commit`, `decompose_baseline_commit`, and clean prerequisite verdicts.

- [ ] **Step 1: Prove and record the exact predecessor commit**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
test ! -d openspec/changes/reconcile-coordexp-swift-training-contracts
reconcile_archive="$(find openspec/changes/archive -maxdepth 1 -type d -name '*-reconcile-coordexp-swift-training-contracts' -print -quit)"
test -n "$reconcile_archive"
reconcile_commit="$(git log -1 --format=%H -- "$reconcile_archive")"
test -n "$reconcile_commit"
git show "$reconcile_commit:openspec/specs/coordexp-swift-packing-forward/spec.md" | rg -n 'synchronous|overlapped'
git show "$reconcile_commit:openspec/specs/coordexp-swift-packing-forward/spec.md" | rg -n 'legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE' && exit 1 || true
```

Expected: the archive and commit exist, supported-mode text is present, and unsupported live selectors are absent. If any assertion fails, stop and repair the predecessor through its own OpenSpec lifecycle.

- [ ] **Step 2: Write the failing characterization and import-boundary tests**

Add tests whose public spine is explicit:

```python
def test_training_import_graph_has_no_reverse_edges() -> None:
    assert_allowed_training_imports(Path.cwd())


def test_pipeline_characterization_matches_frozen_result_and_order() -> None:
    observed = exercise_characterized_pipeline()
    assert observed.result == load_fixture("pipeline_result.json")
    assert observed.phase_order == load_fixture("phase_order.json")
    assert observed.collective_order == load_fixture("collective_order.json")


def test_run_writer_characterization_matches_exact_bytes(tmp_path: Path) -> None:
    observed = exercise_characterized_run_writer(tmp_path)
    assert observed == load_binary_fixture_tree("run_writer")
```

The boundary test must parse imports and reject these exact reverse edges: any leaf/domain owner importing `src.training.pipeline` or `src.training.session`; `cache_workflow` importing `session` or model-loading with `load_model=True`; and production imports of `src.qwen.parity` from `src/prepare_train_cache.py`, `src/training/input_attestation.py`, or final session assembly.

- [ ] **Step 3: Run RED without changing production code**

Run:

```bash
conda run -n ms pytest tests/training/test_training_module_boundaries.py tests/training/test_orchestration_compatibility.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_phase_convergence.py tests/artifacts/test_run_artifacts.py tests/training/test_pack_cache_determinant_registry.py -q
```

Expected: existing behavior characterization passes; intended-owner/import tests fail because the new owner modules and import graph do not exist. Any failure in a characterization assertion is a baseline defect, not a snapshot-update request.

- [ ] **Step 4: Publish the exact baseline receipt, not generated implementation outputs**

Create strict JSON with this fixed shape, populated by the exact command rather than handwritten hashes:

```bash
conda run -n ms python -c 'import json, pathlib, re, sys; values=sys.argv[1:]; assert len(values)==2 and all(re.fullmatch(r"[0-9a-f]{40,64}", value) for value in values); payload={"schema":"coordexp-swift-orchestration-decomposition-baseline-v1","reconcile_commit":values[0],"decompose_baseline_commit":values[1],"prerequisite_status":"verified_committed","characterization_status":"passing","intended_import_boundary_status":"expected_red"}; pathlib.Path("openspec/changes/decompose-coordexp-swift-training-orchestration/receipts").mkdir(parents=True, exist_ok=True); pathlib.Path("openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/reconcile-baseline.json").write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")' "$reconcile_commit" "$(git rev-parse HEAD)"
```

Read the file back with a second `conda run -n ms python -c` validator that requires exact keys, lowercase Git object IDs, and the three literal statuses. Do not store branch names as identity.

- [ ] **Step 5: Gate, review, and commit Wave 0**

Run:

```bash
openspec validate decompose-coordexp-swift-training-orchestration --strict
git diff --check
git status --short
git add openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/reconcile-baseline.json tests/training/test_training_module_boundaries.py tests/training/test_orchestration_compatibility.py tests/fixtures/training_orchestration/supervised_micro_step_legacy.pkl tests/training/test_pipeline_assembly.py tests/training/test_pipeline_phase_convergence.py tests/artifacts/test_run_artifacts.py tests/training/test_pack_cache_determinant_registry.py
git diff --cached --check
git diff --cached --stat
git commit -m "test(training): freeze orchestration compatibility ledger"
```

Expected: strict OpenSpec validation passes; only the listed receipt/tests are staged; the commit is independently revertible.

### Task 2: Move leaf semantics and generic identity to their final owners

**OpenSpec owner:** Tasks 2.1-2.7 and Design decisions 4, 6, 7.

**Files:**
- Create: `src/artifacts/identity.py`
- Create: `src/training/micro_steps.py`
- Modify: `src/qwen/parity.py`
- Modify: `src/prepare_train_cache.py`
- Modify: `src/training/input_attestation.py`
- Modify: `src/training/supervised_trainer.py`
- Modify: `src/training/forward_input_provider.py`
- Modify: `src/training/__init__.py`
- Modify: `src/training/pack_cache.py`
- Modify: `src/supervision/tokens.py`
- Create: `tests/artifacts/test_identity_compatibility.py`
- Modify: `tests/qwen/test_packed_parity.py`
- Modify: `tests/supervision/test_tokens.py`
- Modify: `tests/training/test_supervised_trainer.py`
- Modify: `tests/training/test_forward_input_provider.py`
- Modify: `tests/training/test_pack_cache.py`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: characterized identity functions currently exported by `src.qwen.parity`; `TokenAtom.causal_logits_position`; current `SupervisedMicroStep` field/schema contract.
- Produces: `src.artifacts.identity` as the one implementation owner with compatibility re-exports from `src.qwen.parity`; `TokenSequence.causal_logits_positions(self) -> tuple[int, ...] | None`; canonical `src.training.micro_steps.SupervisedMicroStep`; `supervised_micro_step_schema_identity() -> dict[str, Any]`.

- [ ] **Step 1: Write RED tests for cross-import identity equivalence**

The new test must import both paths and exercise success plus malformed inputs:

```python
@pytest.mark.parametrize(
    "name",
    (
        "canonical_json_bytes",
        "sha256_json",
        "sha256_file",
        "assert_absent_artifact_target",
        "write_strict_json_atomic",
        "base_model_weight_identity",
        "base_model_weight_identity_with_execution_policy",
        "validate_model_weight_identity",
        "assert_model_weight_identity_equal",
        "repo_identity",
        "source_owner_identity",
    ),
)
def test_historical_and_neutral_identity_imports_share_one_object(name: str) -> None:
    assert getattr(parity_identity, name) is getattr(neutral_identity, name)
```

Run:

```bash
conda run -n ms pytest tests/artifacts/test_identity_compatibility.py tests/qwen/test_packed_parity.py -q
```

Expected: FAIL because `src.artifacts.identity` does not exist.

- [ ] **Step 2: Move one identity implementation and retain import-only shims**

Move the exact generic implementations and their private dependencies to `src/artifacts/identity.py`. In `src/qwen/parity.py`, use direct imports only:

```python
from src.artifacts.identity import (
    assert_absent_artifact_target,
    assert_model_weight_identity_equal,
    base_model_weight_identity,
    base_model_weight_identity_with_execution_policy,
    canonical_json_bytes,
    repo_identity,
    sha256_file,
    sha256_json,
    source_owner_identity,
    validate_model_weight_identity,
    write_strict_json_atomic,
)
```

Switch production imports in `src/prepare_train_cache.py` and `src/training/input_attestation.py`. Keep parity-only comparison and Qwen attestation code in `src/qwen/parity.py`; wrappers or duplicate bodies fail the identity-object test.

- [ ] **Step 3: Write RED tests for the token and micro-step owner moves**

Add exact tests:

```python
def test_causal_logits_positions_are_sorted_unique_or_none() -> None:
    assert token_sequence(at_positions=(5, 3, 5)).causal_logits_positions() == (2, 4)
    assert token_sequence(at_positions=()).causal_logits_positions() is None


def test_micro_step_has_one_canonical_owner_and_compatibility_exports() -> None:
    assert public_training.SupervisedMicroStep is micro_steps.SupervisedMicroStep
    assert supervised_trainer.SupervisedMicroStep is micro_steps.SupervisedMicroStep
    assert micro_steps.SupervisedMicroStep.__module__ == "src.training.micro_steps"
```

Also retain a checked-in historical pickle fixture generated at the Wave-0 commit and assert restricted loading of its `src.training.supervised_trainer.SupervisedMicroStep` path; assert newly pickled bytes contain `src.training.micro_steps` and decoded values compare field-by-field equal.

Run:

```bash
conda run -n ms pytest tests/supervision/test_tokens.py tests/training/test_supervised_trainer.py tests/training/test_forward_input_provider.py tests/training/test_pack_cache.py -q
```

Expected: FAIL on missing domain method/canonical module owner while all unrelated baseline assertions remain green.

- [ ] **Step 4: Implement the final leaf interfaces and remove private cross-imports**

Use these exact signatures:

```python
class TokenSequence:
    def causal_logits_positions(self) -> tuple[int, ...] | None:
        positions = tuple(
            sorted({int(atom.causal_logits_position) for atom in self.atoms})
        )
        return positions or None


def supervised_micro_step_schema_identity() -> dict[str, Any]:
    schema_fields: list[dict[str, Any]] = []
    for item in fields(SupervisedMicroStep):
        has_default = item.default is not MISSING
        schema_fields.append(
            {
                "name": item.name,
                "annotation": str(item.type),
                "has_default": has_default,
                "default": json.loads(
                    json.dumps(item.default, allow_nan=False, sort_keys=True)
                )
                if has_default
                else None,
            }
        )
    return {
        "class": "SupervisedMicroStep",
        "frozen": bool(SupervisedMicroStep.__dataclass_params__.frozen),
        "fields": schema_fields,
    }
```

Move the current frozen dataclass unchanged, re-export it from both old supported paths, update the restricted unpickler to allow exactly the historical and canonical module/class pairs, switch both trainer/provider callers to `micro_step.token_sequence.causal_logits_positions()`, and delete `_logits_positions_to_keep` and its private import.

- [ ] **Step 5: Run GREEN, residue checks, and refactor review**

Run:

```bash
conda run -n ms pytest tests/artifacts/test_identity_compatibility.py tests/qwen/test_packed_parity.py tests/supervision/test_tokens.py tests/training/test_supervised_trainer.py tests/training/test_forward_input_provider.py tests/training/test_pack_cache.py tests/training/test_training_module_boundaries.py -q
rg -n 'from src\.qwen\.parity import' src/prepare_train_cache.py src/training/input_attestation.py && exit 1 || true
rg -n '_logits_positions_to_keep|_logits_to_keep_positions' src tests/training tests/supervision && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS; historical and new pickle paths both load, new bytes carry only the canonical module path, and production no longer imports the moved identity owner from parity.

- [ ] **Step 6: Commit the leaf-owner wave**

```bash
git diff --check
git add src/artifacts/identity.py src/qwen/parity.py src/prepare_train_cache.py src/training/input_attestation.py src/training/micro_steps.py src/training/supervised_trainer.py src/training/forward_input_provider.py src/training/__init__.py src/training/pack_cache.py src/supervision/tokens.py tests/artifacts/test_identity_compatibility.py tests/qwen/test_packed_parity.py tests/supervision/test_tokens.py tests/training/test_supervised_trainer.py tests/training/test_forward_input_provider.py tests/training/test_pack_cache.py tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): assign leaf semantic owners"
```

### Task 3: Extract the immutable execution plan and bounded control plane

**OpenSpec owner:** Tasks 3.1-3.6 and Design decisions 2, 3.

**Files:**
- Create: `src/training/execution_plan.py`
- Create: `src/training/control_plane.py`
- Modify: `src/training/pipeline.py`
- Create: `tests/training/test_execution_plan.py`
- Create: `tests/training/test_control_plane.py`
- Modify: `tests/training/test_pipeline_cache_preflight.py`
- Modify: `tests/training/test_pipeline_phase_convergence.py`
- Modify: `tests/training/test_pipeline_assembly.py`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: `ResolvedTrainConfig`, existing fixed-frame rank report protocol, phase names/order, and pre/post-Accelerator binding point captured in Task 1.
- Produces: `TrainingExecutionPlan`, `build_training_execution_plan(config_path: str | Path, *, measurement_context: Mapping[str, Any] | None) -> TrainingExecutionPlan`; `RankControlPlane.open(*, rank: int, world_size: int) -> RankControlPlane`, `.converge(phase: str, body: Callable[[], T], *, local_details: Callable[[], Mapping[str, Any] | None] | None = None, receipt_sink: Callable[[Mapping[str, Any]], None] | None = None) -> T`, `.bind_accelerator(accelerator: Any) -> None`, `.close() -> None`.

- [ ] **Step 1: Write RED tests for pure plan construction**

```python
def test_build_execution_plan_is_frozen_and_model_free(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = forbid_calls(
        "Accelerator",
        "load_qwen_components",
        "prepare_training_pack_caches",
        "RunWriter.initialize",
        "all_gather",
    )
    plan = build_training_execution_plan(CONFIG, measurement_context={"run": "unit"})
    assert dataclasses.is_dataclass(plan)
    assert plan.__dataclass_params__.frozen is True
    assert plan.measurement_context == {"run": "unit"}
    assert forbidden.calls == ()
```

Run: `conda run -n ms pytest tests/training/test_execution_plan.py -q`

Expected: FAIL because `src.training.execution_plan` does not exist.

- [ ] **Step 2: Implement the exact immutable plan interface**

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
) -> TrainingExecutionPlan:
    resolved_config = load_train_config(config_path)
    rank, world_size = resolve_model_free_launch_identity()
    return TrainingExecutionPlan(
        resolved_config=resolved_config,
        repo_root=Path.cwd().resolve(),
        launch_rank=rank,
        launch_world_size=world_size,
        measurement_context=MappingProxyType(dict(measurement_context or {})),
        entry_started_at=utc_now(),
        entry_started_monotonic=time.monotonic(),
        entry_resources=MappingProxyType(dict(collect_resource_snapshot())),
    )
```

Name formerly private helpers without leading underscores only when the new module owns them; do not add writer/model/cache/callback fields.

- [ ] **Step 3: Write RED tests for `RankControlPlane`**

Require single-/multi-rank success, selected failure, rank ordering, fixed frame limits, resource convergence, receipt sinks, identity mismatch on bind, exact collective trace, and double close:

```python
def test_control_plane_preserves_order_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedRankTransport(success_reports())
    monkeypatch.setattr(
        control_plane_module,
        "build_model_free_gatherer",
        lambda world_size: transport,
    )
    plane = RankControlPlane.open(rank=0, world_size=2)
    assert plane.converge("preflight", lambda: "ok") == "ok"
    plane.close()
    plane.close()
    assert transport.events == expected_control_plane_events()
```

Run: `conda run -n ms pytest tests/training/test_control_plane.py tests/training/test_pipeline_phase_convergence.py -q`

Expected: FAIL on missing owner; baseline phase-convergence tests remain green.

- [ ] **Step 4: Move the existing protocol behind the exact bounded interface**

```python
class RankControlPlane:
    @classmethod
    def open(cls, *, rank: int, world_size: int) -> "RankControlPlane":
        return cls(rank=rank, world_size=world_size, gatherer=build_model_free_gatherer(world_size))

    def converge(
        self,
        phase: str,
        body: Callable[[], T],
        *,
        local_details: Callable[[], Mapping[str, Any] | None] | None = None,
        receipt_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> T:
        return run_rank_converged_phase(
            phase,
            body,
            rank=self.rank,
            world_size=self.world_size,
            gatherer=self.gatherer,
            local_details=local_details,
            receipt_sink=receipt_sink,
        )

    def bind_accelerator(self, accelerator: Any) -> None:
        self.gatherer = bind_accelerator_gatherer(
            accelerator,
            expected_rank=self.rank,
            expected_world_size=self.world_size,
        )

    def close(self) -> None:
        close_rank_gatherer_once(self.gatherer)
```

The actual moved helpers retain current frame sizes, timeouts, schemas, rank order, resource receipts, exception selection, and calls. The facade constructs the plan, opens the plane, binds at the current point, and closes in `finally`; delete replaced free-function bodies only after exact ordered-call tests pass.

- [ ] **Step 5: Run GREEN and import/residue checks**

```bash
conda run -n ms pytest tests/training/test_execution_plan.py tests/training/test_control_plane.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pipeline_phase_convergence.py tests/training/test_pipeline_assembly.py tests/training/test_training_module_boundaries.py tests/training/test_wave7_determinism_preflight.py -q
rg -n 'from src\.training\.pipeline import|import src\.training\.pipeline' src/training/execution_plan.py src/training/control_plane.py && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS with the Task-1 phase/collective trace unchanged.

- [ ] **Step 6: Commit the plan/control-plane wave**

```bash
git diff --check
git add src/training/execution_plan.py src/training/control_plane.py src/training/pipeline.py tests/training/test_execution_plan.py tests/training/test_control_plane.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pipeline_phase_convergence.py tests/training/test_pipeline_assembly.py tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): extract execution control plane"
```

### Task 4: Narrow cache determinants and extract the model-free cache workflow

**OpenSpec owner:** Tasks 4.1-4.8 and Design decisions 4, 5.

**Files:**
- Create: `src/training/cache_contract.py`
- Create: `src/training/cache_workflow.py`
- Modify: `src/training/pack_cache.py`
- Modify: `src/training/pipeline.py`
- Modify: `src/prepare_train_cache.py`
- Create: `tests/training/test_cache_contract.py`
- Create: `tests/training/test_cache_workflow.py`
- Modify: `tests/training/test_pack_cache_determinant_registry.py`
- Modify: `tests/training/test_pipeline_pack_cache_rebuild.py`
- Modify: `tests/training/test_pipeline_cache_preflight.py`
- Modify: `tests/training/test_prepare_train_cache_cli.py`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: final `SupervisedMicroStep` owner from Task 2, `RankControlPlane` from Task 3, low-level `pack_cache.py` identity/publication/load APIs.
- Produces: `micro_step_runtime_config_identity(config: TrainConfig) -> dict[str, Any]`; frozen `CachePreflight`; frozen `HydratedTrainingInputs`; `prepare_training_pack_caches(config_path: str | Path, *, require_all_hit: bool = False) -> dict[str, Any]` owned by `cache_workflow.py` and re-exported by `pipeline.py`; CLI flag `src.prepare_train_cache --require-all-hit`.

- [ ] **Step 1: Write RED determinant mutation tests**

Require these exact owner entries after the move:

```python
assert PACKING_CACHE_DETERMINANT_OWNERS["micro_step_runtime_config"] == "src/training/cache_contract.py"
assert PACKING_CACHE_DETERMINANT_OWNERS["micro_step_schema"] == "src/training/micro_steps.py"
```

Mutation tests must prove edits to either new owner change the aggregate while edits to `reporting.py`, `session.py`, and `pipeline.py` do not; all other registry entries must equal the Task-1 fixture.

Run: `conda run -n ms pytest tests/training/test_cache_contract.py tests/training/test_pack_cache_determinant_registry.py -q`

Expected: FAIL on missing owner and old registry paths.

- [ ] **Step 2: Implement the narrow runtime projection and registry paths**

```python
def micro_step_runtime_config_identity(config: TrainConfig) -> dict[str, Any]:
    return {
        "fa2_model_dtype": config.training.precision,
        "capture_fa2_branch": config.model.fa2_branch_proof == "every_forward",
        "require_fa2_branch_proof": (
            config.model.fa2_branch_proof == "every_forward"
        ),
    }
```

Before committing, compare this function's keys and values against the exact current pipeline projection; if actual config ownership differs at the pinned baseline, retain the characterized access path while keeping exactly these three output keys. Update only the two registry owner paths here.

- [ ] **Step 3: Write RED workflow tests around named result records**

```python
@dataclass(frozen=True)
class CachePreflight:
    train_fingerprint: str
    eval_fingerprint: str
    train_target: Path
    eval_target: Path
    receipt: Mapping[str, Any]

    def to_receipt_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True)
class HydratedTrainingInputs:
    train_micro_steps: Sequence[SupervisedMicroStep]
    eval_micro_steps: Sequence[SupervisedMicroStep]
    eval_reduction_mode: str
    receipt: Mapping[str, Any]

    def to_receipt_dict(self) -> dict[str, Any]:
        return dict(self.receipt)
```

Tests cover one-process preparation, worker resolution, train/eval aggregation, absent-target publication, all-hit behavior, admission, rank-local hydration, image-processor attachment, and exact actionable failures. These record fields must be adjusted only by first updating the authoritative design if the extracted current bundle cannot fit without semantic loss.

Add RED CLI/workflow tests that replace every render/tokenize/pack/build and
publication callable with a sentinel that raises `AssertionError`. With
`require_all_hit=True`, two valid existing targets must validate successfully;
a missing train target, missing eval target, or invalid target must raise the
bounded cache error while every sentinel remains uncalled. This proves the
gate is enforced after argv parsing and before any build-capable branch.

Run: `conda run -n ms pytest tests/training/test_cache_workflow.py tests/training/test_pipeline_pack_cache_rebuild.py tests/training/test_pipeline_cache_preflight.py -q`

Expected: FAIL because orchestration still resides in `pipeline.py`.

- [ ] **Step 4: Move orchestration, not low-level storage semantics**

Move preparation, absent-target multi-worker materialization, split aggregation, model-free admission, rank-local hydration, attachment, and bounded diagnostics to `cache_workflow.py`. Leave determinant construction, fingerprints, restricted deserialization, manifests, immutable publication, and payload loading in `pack_cache.py`. Update:

```python
# src/training/pipeline.py
from src.training.cache_workflow import prepare_training_pack_caches

__all__ = ["prepare_training_pack_caches", "run_training_pipeline"]

# src/prepare_train_cache.py
from src.training.cache_workflow import prepare_training_pack_caches
```

Parse `--require-all-hit` and pass it to the workflow. In that branch, resolve
both fingerprints/targets and validate both existing targets before returning;
do not call the absent-target builder or catch a miss/invalid result as a build
request. No workflow path may call model loading with `load_model=True`, import
`session.py`, or materialize during distributed startup.

- [ ] **Step 5: Prove the exact four determinant-source changes and pickle exception**

Create a comparison helper in the tests that reports changed determinant names. Assert:

```python
assert changed_determinants == {
    "supervision_tokens",
    "micro_step_runtime_config",
    "micro_step_schema",
    "cache_serializer",
}
assert decode_old_payload(old_bytes) == decode_new_payload(new_bytes)
assert b"src.training.supervised_trainer" in old_bytes
assert b"src.training.micro_steps" in new_bytes
```

No production cache command is allowed in this task; use temporary directories and the checked-in historical fixture only.

- [ ] **Step 6: Run GREEN and the no-materialization gate**

```bash
conda run -n ms pytest tests/training/test_cache_contract.py tests/training/test_cache_workflow.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_pack_cache.py tests/training/test_pipeline_pack_cache_rebuild.py tests/training/test_pipeline_cache_preflight.py tests/training/test_prepare_train_cache_cli.py tests/training/test_input_attestation.py tests/training/test_training_module_boundaries.py tests/packing tests/supervision -q
rg -n 'load_model\s*=\s*True|from src\.training\.session import' src/training/cache_workflow.py && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS; the four-entry diff is exact; no cache target outside pytest temporary directories is created.

- [ ] **Step 7: Commit the cache-owner wave**

```bash
git diff --check
git add src/training/cache_contract.py src/training/cache_workflow.py src/training/pack_cache.py src/training/pipeline.py src/prepare_train_cache.py tests/training/test_cache_contract.py tests/training/test_cache_workflow.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_pipeline_pack_cache_rebuild.py tests/training/test_pipeline_cache_preflight.py tests/training/test_prepare_train_cache_cli.py tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): isolate pack cache workflow"
```

### Task 5: Extract completed-step reporting and pure RunWriter internals

**OpenSpec owner:** Tasks 5.1-5.7 and Design decisions 9, 10.

**Files:**
- Create: `src/training/reporting.py`
- Create: `src/artifacts/run_schema.py`
- Create: `src/artifacts/run_state.py`
- Modify: `src/training/pipeline.py`
- Modify: `src/artifacts/run_writer.py`
- Create: `tests/training/test_reporting.py`
- Create: `tests/artifacts/test_run_schema.py`
- Create: `tests/artifacts/test_run_state.py`
- Modify: `tests/artifacts/test_run_artifacts.py`
- Modify: `tests/training/test_pipeline_assembly.py`
- Modify: `tests/training/test_pipeline_exact_resume.py`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: `CompletedStepObservation`, Task-1 exact row/error/tree fixtures, existing `RunWriter` public methods and the callable `admit_exact_resume_checkpoint_publication` import.
- Produces: `CompletedStepReporter(writer: RunWriter | None, lifecycle: MutableMapping[str, Any], runtime: Any, resource_collector: Callable[[], Mapping[str, Any]])`; pure schema functions in `run_schema.py`; pure state transitions in `run_state.py`; unchanged `RunWriter` public I/O facade.

- [ ] **Step 1: Write RED reporter parity tests**

```python
def test_completed_step_reporter_matches_baseline_handler_exactly() -> None:
    old = exercise_baseline_logging_handler()
    new = exercise_completed_step_reporter()
    assert new.lifecycle == old.lifecycle
    assert new.reduction_requests == old.reduction_requests
    assert new.logging_bytes == old.logging_bytes
    assert new.broadcasts == old.broadcasts
```

Cover finite/skipped steps, accuracy validation, rank resources, append failure, warmup accounting, and first-step phase completion.

Run: `conda run -n ms pytest tests/training/test_reporting.py tests/training/test_pipeline_assembly.py -q`

Expected: FAIL because `CompletedStepReporter` does not exist.

- [ ] **Step 2: Move the callback behavior into the fixed reporter seam**

```python
class CompletedStepReporter:
    def __init__(
        self,
        *,
        writer: RunWriter | None,
        lifecycle: MutableMapping[str, Any],
        runtime: Any,
        resource_collector: Callable[[], Mapping[str, Any]],
    ) -> None:
        self._writer = writer
        self._lifecycle = lifecycle
        self._runtime = runtime
        self._resource_collector = resource_collector

    def __call__(self, observation: CompletedStepObservation) -> None:
        report_completed_step(
            observation=observation,
            writer=self._writer,
            lifecycle=self._lifecycle,
            runtime=self._runtime,
            resource_collector=self._resource_collector,
        )
```

`report_completed_step` contains the moved existing body. Add no cadence, sinks, fields, ETA, TensorBoard, registry, or changed collective.

- [ ] **Step 3: Write RED pure-schema/state tests from complete byte fixtures**

Require pure calls to return new mappings without filesystem I/O and exact-compare initialization, strict logging rows, policy/schedule binding, phase outcomes, checkpoint events, best/final, continuation, warnings, failed finalization, and success finalization. Monkeypatch `Path.open`, `os.replace`, `os.fsync`, and `Path.read_bytes` to fail inside `run_schema.py` and `run_state.py` tests.

Run: `conda run -n ms pytest tests/artifacts/test_run_schema.py tests/artifacts/test_run_state.py tests/artifacts/test_run_artifacts.py -q`

Expected: FAIL because pure owners do not exist.

- [ ] **Step 4: Move only pure logic and keep filesystem sequencing in `RunWriter`**

`run_schema.py` owns strict JSON normalization/serialization, bounded detail checks, timestamp/number/lineage validation, measurement construction, and logging-row normalization. `run_state.py` owns mapping-to-new-mapping transitions plus exact-resume checkpoint publication admission. `run_writer.py` retains all read, append, fsync, link/replace, collision, and atomic-write operations and re-exports:

```python
from src.artifacts.run_state import admit_exact_resume_checkpoint_publication

__all__ = ["RunWriter", "admit_exact_resume_checkpoint_publication"]
```

Delete each moved body from `run_writer.py`; do not leave wrappers except the required public re-export.

- [ ] **Step 5: Run GREEN and exact-byte review**

```bash
conda run -n ms pytest tests/training/test_reporting.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_state.py tests/artifacts/test_run_artifacts.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_checkpoint_payload_identity.py tests/artifacts/test_resources.py tests/artifacts/test_training_state.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_exact_resume.py tests/training/test_training_module_boundaries.py -q
rg -n 'open\(|os\.replace|os\.fsync|read_bytes|write_bytes' src/artifacts/run_schema.py src/artifacts/run_state.py && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS and exact Task-1 bytes/errors remain equal.

- [ ] **Step 6: Commit the reporting/artifact-pure wave**

```bash
git diff --check
git add src/training/reporting.py src/artifacts/run_schema.py src/artifacts/run_state.py src/training/pipeline.py src/artifacts/run_writer.py tests/training/test_reporting.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_state.py tests/artifacts/test_run_artifacts.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_exact_resume.py tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): isolate reporting and run state"
```

### Task 6: Encapsulate the model-bearing lifetime in `TrainingSession`

**OpenSpec owner:** Tasks 6.1-6.6 and Design decisions 1, 8.

**Files:**
- Create: `src/training/session.py`
- Modify: `src/training/pipeline.py`
- Modify: `src/training/cache_workflow.py`
- Modify: `src/training/reporting.py`
- Create: `tests/training/test_training_session.py`
- Modify: `tests/training/test_pipeline_assembly.py`
- Modify: `tests/training/test_pipeline_exact_resume.py`
- Modify: `tests/training/test_pipeline_cache_preflight.py`
- Modify: `tests/training/test_checkpoint_handler_identity.py`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: `TrainingExecutionPlan`, `RankControlPlane`, `CachePreflight`, `RunWriter`, `RunIdentity`, `CompletedStepReporter`, current exact-resume/eval/checkpoint/final handler order.
- Produces: `TrainingSession.__init__(*, plan, control_plane, writer, run_identity, cache_preflight, admitted_policies)`, `.run() -> dict[str, Any]`, `.fail(error: BaseException) -> None`, `.close() -> None`; thin `run_training_pipeline(config_path: str | Path, *, measurement_context: Mapping[str, Any] | None = None) -> dict[str, Any]` facade.

- [ ] **Step 1: Write RED lifetime and fixed-order tests**

```python
def test_training_session_preserves_fixed_order_and_primary_failure() -> None:
    session, events = build_scripted_session(fail_at="evaluation")
    with pytest.raises(ExpectedPrimaryError):
        session.run()
    session.fail(ExpectedPrimaryError("primary"))
    session.close()
    session.close()
    assert events == expected_session_failure_events()


def test_pipeline_builds_exactly_one_session_and_closes_in_finally() -> None:
    result, events = exercise_pipeline_facade()
    assert result == load_fixture("pipeline_result.json")
    assert events == ["plan", "control.open", "writer", "cache.admit", "session.init", "session.run", "session.close", "control.close"]
```

Run: `conda run -n ms pytest tests/training/test_training_session.py tests/training/test_pipeline_assembly.py -q`

Expected: FAIL because `TrainingSession` does not exist.

- [ ] **Step 2: Add the exact session boundary**

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
    ) -> None:
        self.plan = plan
        self.control_plane = control_plane
        self.writer = writer
        self.run_identity = run_identity
        self.cache_preflight = cache_preflight
        self.admitted_policies = MappingProxyType(dict(admitted_policies))
        self._closed = False

    def run(self) -> dict[str, Any]:
        return run_initialized_training_session(self)

    def fail(self, error: BaseException) -> None:
        publish_training_session_failure(self, error)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_training_session_resources(self)
```

The three module-level functions receive only the session and contain the moved fixed choreography; no subclass, registry, phase list, callback container, or alternate backend is introduced.

- [ ] **Step 3: Move initialized-run ownership in coherent slices**

Move model/adapter/selected-token/loss/optimizer/runtime assembly first; then cache hydration/provider lifetime; then exact-resume/eval/checkpoint/final handlers and profile-sync reset. After each slice run:

```bash
conda run -n ms pytest tests/training/test_training_session.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_exact_resume.py tests/training/test_checkpoint_handler_identity.py -q
```

Expected after every slice: PASS with exact phase, collective, result, and artifact fixtures unchanged.

- [ ] **Step 4: Reduce the facade and delete replaced implementation**

The final facade has one literal lifetime:

```python
def run_training_pipeline(
    config_path: str | Path,
    *,
    measurement_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = build_training_execution_plan(
        config_path,
        measurement_context=measurement_context,
    )
    control_plane = RankControlPlane.open(
        rank=plan.launch_rank,
        world_size=plan.launch_world_size,
    )
    session: TrainingSession | None = None
    try:
        initialized = initialize_pipeline_owners(plan, control_plane)
        session = TrainingSession(**initialized.session_arguments())
        return session.run()
    except BaseException as error:
        if session is not None:
            session.fail(error)
        raise
    finally:
        if session is not None:
            session.close()
        control_plane.close()
```

`initialize_pipeline_owners` must remain a small facade-owned assembly result, not a state bag; if it cannot be bounded to pre-model writer/cache ownership in the authoritative design, inline the fixed calls instead. Delete moved handlers/helpers and retain only `run_training_pipeline` plus the documented cache-preparation compatibility re-export.

- [ ] **Step 5: Run GREEN, boundary enforcement, and completion gate**

```bash
conda run -n ms pytest tests/training/test_training_session.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pipeline_phase_convergence.py tests/training/test_pipeline_exact_resume.py tests/training/test_checkpoint_handler_identity.py tests/training/test_supervised_trainer.py tests/runtime tests/artifacts tests/training/test_training_module_boundaries.py -q
rg -n 'from src\.training\.(pipeline|session) import|import src\.training\.(pipeline|session)' src/training/execution_plan.py src/training/control_plane.py src/training/cache_contract.py src/training/micro_steps.py src/training/reporting.py src/artifacts/identity.py && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS; reverse imports and pass-through copies are absent; the Task-1 ordered trace and bytes are unchanged.

- [ ] **Step 6: Commit the session/thin-facade wave**

```bash
git diff --check
git add src/training/session.py src/training/pipeline.py src/training/cache_workflow.py src/training/reporting.py tests/training/test_training_session.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_exact_resume.py tests/training/test_pipeline_cache_preflight.py tests/training/test_checkpoint_handler_identity.py tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): isolate training session lifetime"
```

### Task 7: Remove unsupported provider residue and freeze final owners

**OpenSpec owner:** Tasks 7.1-7.7 and Design decision 11.

**Files:**
- Modify: `src/config/models.py`
- Modify: `src/training/forward_input_provider.py`
- Modify: `src/training/supervised_trainer.py`
- Modify: `src/training/session.py`
- Modify: `src/artifacts/run_writer.py`
- Modify: `tests/config/test_train_config.py`
- Modify: `tests/training/test_forward_input_provider.py`
- Modify: `tests/training/test_supervised_trainer.py`
- Modify: `tests/training/test_pipeline_assembly.py`
- Modify: `tests/training/test_pipeline_cache_preflight.py`
- Modify: `tests/training/test_pipeline_exact_resume.py`
- Modify: `tests/training/test_wave5_provider_benchmark.py`
- Modify: `docs/COORDEXP_SWIFT.md`
- Modify: `docs/SYSTEM_OVERVIEW.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `tests/training/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: prerequisite contract pinned in Task 1; current `ForwardInputProvider` lifecycle.
- Produces: `ForwardInputProviderMode = Literal["synchronous", "overlapped"]`; strict-config-only `resolve_forward_input_provider_mode(configured_mode)`; non-optional provider assembly; frozen final import/determinant owner graph.

- [ ] **Step 1: Write RED strict-rejection and supported-equivalence tests**

```python
@pytest.mark.parametrize("mode", ("legacy_fused", "unknown"))
def test_unsupported_provider_modes_fail_strict_config(mode: str) -> None:
    with pytest.raises(ValidationError):
        load_config_with_provider_mode(mode)


def test_environment_cannot_replace_strict_provider_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", "overlapped")
    assert resolve_forward_input_provider_mode("synchronous").resolved_mode == "synchronous"
```

Retain exact synchronous/overlapped prepared-input equivalence, depth-one CPU ownership, error timing, cancellation, and close tests.

Run: `conda run -n ms pytest tests/config/test_train_config.py tests/training/test_forward_input_provider.py tests/training/test_supervised_trainer.py -q`

Expected: FAIL because legacy and the environment override are still accepted.

- [ ] **Step 2: Remove the unsupported selector and direct-build branch**

Use:

```python
ForwardInputProviderMode = Literal["synchronous", "overlapped"]


def resolve_forward_input_provider_mode(
    configured_mode: ForwardInputProviderMode,
) -> ResolvedForwardInputProviderMode:
    validated = validate_forward_input_provider_mode(configured_mode)
    return ResolvedForwardInputProviderMode(
        configured_mode=validated,
        resolved_mode=validated,
        source="strict_config",
    )
```

Delete the environment constant/source, legacy constant/disposition, optional provider path, trainer-owned fused device-direct branch, config fixtures/defaults, RunWriter acceptance, and exact-resume policy variants. Keep supported-mode artifacts byte-equal except for fields whose authoritative ledger explicitly disappears with the removed selector.

- [ ] **Step 3: Update current owner docs without touching user-owned dirty docs**

Update only the three listed canonical architecture docs to identify strict config as sole selector, synchronous as reference, overlapped as explicit experimental, and the new owner modules. Do not edit `docs/catalog.yaml`, `docs/data/PACKING.md`, `docs/history/README.md`, or cache-retirement history in this task.

- [ ] **Step 4: Run GREEN and exhaustive live-residue checks**

```bash
conda run -n ms pytest tests/config/test_train_config.py tests/training/test_forward_input_provider.py tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pipeline_exact_resume.py tests/training/test_wave5_provider_benchmark.py tests/artifacts/test_run_artifacts.py tests/training/test_training_module_boundaries.py -q
rg -n 'legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE|deprecated_environment_override|trainer_fused_device_direct' src configs/coordexp_swift tests docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md && exit 1 || true
rg -n 'from src\.qwen\.parity import' src/prepare_train_cache.py src/training src/artifacts && exit 1 || true
rg -n '"micro_step_runtime_config": "src/training/pipeline.py"|"micro_step_schema": "src/training/supervised_trainer.py"' src tests && exit 1 || true
openspec validate decompose-coordexp-swift-training-orchestration --strict
```

Expected: PASS with residue allowed only in explicitly historical directories excluded from this search.

- [ ] **Step 5: Commit and pin the frozen owner graph**

```bash
git diff --check
git add src/config/models.py src/training/forward_input_provider.py src/training/supervised_trainer.py src/training/session.py src/artifacts/run_writer.py tests/config/test_train_config.py tests/training/test_forward_input_provider.py tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pipeline_exact_resume.py tests/training/test_wave5_provider_benchmark.py docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md tests/training/test_training_module_boundaries.py
git diff --cached --check
git commit -m "refactor(training): retire legacy provider residue"
git rev-parse HEAD
```

Expected: one exact commit is now the only eligible input to Task 8 authorization.

### Task 8: Perform the single authorized cache transition

**OpenSpec owner:** Tasks 8.1-8.5 and Migration step 9.

**Files:**
- Create after authorization: an absent receipt path outside the repository named in the authorization packet.
- Create through immutable publication: exactly the absent train/eval cache targets resolved for the named config.
- Modify: no source, config, test, OpenSpec, or existing cache files.

**Interfaces:**
- Consumes: exact frozen-owner commit from Task 7 and config `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`.
- Produces: one completed preparation receipt, new train/eval fingerprints/manifests, then a distinct `--require-all-hit` receipt; no second build.

- [ ] **Step 1: Build the read-only authorization packet and stop**

Record exact commit, both commands, resolved cache root, old/new determinant projections, absent new targets, existing old targets, expected workers, free disk, RSS ceiling, wall-time bound, and stop conditions. Run only read-only discovery commands. The packet must freeze both exact argv vectors and both absent receipt paths:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
decompose_commit="$(git rev-parse HEAD)"
cache_receipt="/data/CoordExp/outputs/cache-preparation-receipts/decompose-coordexp-swift-training-orchestration-${decompose_commit}.json"
hit_receipt="/data/CoordExp/outputs/cache-preparation-receipts/decompose-coordexp-swift-training-orchestration-${decompose_commit}-require-all-hit.json"
test ! -e "$cache_receipt"
test ! -e "$hit_receipt"
conda run -n ms python -m src.prepare_train_cache \
  --config configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml \
  --receipt "$cache_receipt"

conda run -n ms python -m src.prepare_train_cache \
  --config configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml \
  --receipt "$hit_receipt" \
  --require-all-hit
```

Both computed receipt paths are part of the packet and must be absent when the packet is submitted and immediately before their respective executions. The second argv receives no cache-materialization authority. Stop here until the user explicitly approves that exact packet.

- [ ] **Step 2: Revalidate the approved packet immediately before execution**

Before the build, require: `git rev-parse HEAD` equals the approved commit; both new targets and both receipts are absent; both old targets remain unchanged; resolved workers/disk/RSS bounds still fit; no intermediate fingerprint exists. Before the hit command, recheck the same commit/argv, require the build receipt and both new targets to exist and validate, and require the hit receipt to remain absent. Any mismatch invalidates authorization and stops the task; the hit command may not inherit fallback materialization permission.

- [ ] **Step 3: Run exactly one multi-worker materialization invocation**

Run only the approved command once. Expected: exit 0, one terminal receipt with `terminal_status=completed`, and exactly the declared train/eval targets. On failure, stop; do not retry, repair, delete, or overwrite.

- [ ] **Step 4: Run the frozen fail-before-build hit verifier**

Run only the packet-frozen command below; do not substitute the ordinary preparation argv:

```bash
conda run -n ms python -m src.prepare_train_cache \
  --config configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml \
  --receipt "$hit_receipt" \
  --require-all-hit
```

The command itself must fingerprint, admit, and validate both existing splits. If either target is missing or invalid, it must fail before render/tokenize/pack/build/temporary-publication/immutable-publication; a prior successful preflight does not permit fallback. On success, assert the distinct terminal receipt, the four changed determinant names, old/new pickle module paths, decoded-value equality, manifest/full digests, immutable old targets, worker policy, timing, RSS, and no third target.

- [ ] **Step 5: Gate without committing cache artifacts**

```bash
conda run -n ms pytest tests/training/test_cache_contract.py tests/training/test_cache_workflow.py tests/training/test_pack_cache.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_pipeline_pack_cache_rebuild.py tests/training/test_prepare_train_cache_cli.py -q
openspec validate decompose-coordexp-swift-training-orchestration --strict
git status --short
```

Expected: PASS; repository status contains no cache outputs and no task-owned source change. Do not commit external cache/receipt artifacts.

### Task 9: Run final CPU proof, obtain GPU authorization, and close the change

**OpenSpec owner:** Tasks 9.1-9.6 and Migration step 10.

**Files:**
- Modify only if owner documentation is still stale: `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`
- Update after evidence exists: `openspec/changes/decompose-coordexp-swift-training-orchestration/tasks.md`
- Create after execution: bounded receipts under `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/`

**Interfaces:**
- Consumes: committed code through Task 7, immutable cache evidence from Task 8, Task-1 compatibility fixtures.
- Produces: full CPU receipt, separately authorized two-rank smoke receipt, exact compatibility comparison, final residue/overdesign/intent verdicts, and completed OpenSpec task evidence.

- [ ] **Step 1: Run the full relevant CPU matrix and record exact receipts**

Run:

```bash
conda run -n ms pytest tests/config tests/data tests/templates tests/qwen tests/packing tests/supervision tests/losses tests/runtime tests/training tests/artifacts tests/eval -q
```

Expected: exit 0. Record exact commit, command, pass/fail/skip counts, duration, and environment identity. A skipped test is reported, not silently treated as executed coverage.

- [ ] **Step 2: Build the GPU authorization packet and stop**

The packet names the exact commit, the one-step config, `conda run -n ms accelerate launch` command with exactly two processes, selected GPU IDs, new cache fingerprints, absent run root, wall-time/GPU-memory/disk bounds, one finite applied-step requirement, eval/checkpoint/finalization consumers, and stop conditions for OOM, non-finite values, unexpected cache build, collective hang, or artifact collision. Stop until the user explicitly approves the exact packet.

- [ ] **Step 3: Run the approved two-rank BF16 vertical smoke once**

Expected: model-free admission, admitted two-rank Accelerator identity, one finite applied step on synchronous provider, eval, checkpoint, finalization, and downstream readers all complete. This is integration evidence only; do not claim throughput or efficiency improvement.

- [ ] **Step 4: Compare every protected surface and run final residue checks**

```bash
conda run -n ms pytest tests/training/test_orchestration_compatibility.py tests/training/test_training_module_boundaries.py tests/artifacts/test_identity_compatibility.py tests/artifacts/test_run_artifacts.py tests/training/test_pack_cache_determinant_registry.py -q
rg -n 'legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE|deprecated_environment_override|trainer_fused_device_direct' src configs/coordexp_swift tests docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md && exit 1 || true
rg -n 'from src\.qwen\.parity import' src/prepare_train_cache.py src/training src/artifacts && exit 1 || true
rg -n 'from src\.training\.(pipeline|session) import|import src\.training\.(pipeline|session)' src/training/execution_plan.py src/training/control_plane.py src/training/cache_contract.py src/training/micro_steps.py src/training/reporting.py src/artifacts/identity.py && exit 1 || true
rg -n 'event_bus|plugin_registry|dependency_injection|phase_registry|FSDP' src/training src/artifacts && exit 1 || true
```

Expected: PASS; only the authoritative ledger's declared cache identity/pickle transition and removed legacy selector differ.

- [ ] **Step 5: Obtain independent standards, overdesign, and intent-contract verdicts**

Give reviewers the exact baseline commit, final commit, OpenSpec links, CPU/smoke receipts, cache transition receipt, and compatibility diff. Require each verdict to distinguish blocking P0/P1 from lower findings; resolve or disposition every finding in OpenSpec before completion.

- [ ] **Step 6: Mark tasks only from receipts, validate, and commit closeout**

```bash
openspec validate decompose-coordexp-swift-training-orchestration --strict
git diff --check
git status --short
git add openspec/changes/decompose-coordexp-swift-training-orchestration/tasks.md openspec/changes/decompose-coordexp-swift-training-orchestration/receipts docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs(training): close orchestration decomposition evidence"
```

Expected: only evidence-backed task checkboxes, bounded receipts, and any necessary final owner-doc corrections are staged. Do not archive or sync the change until `openspec-verify-change` independently confirms implementation, contract, and task coherence.

## Plan self-review checklist

- [ ] Every OpenSpec wave 0-8 maps to exactly one Task 1-9 above.
- [ ] The fixed owner/import graph is enforced before and after moves, with no reverse imports or new framework layer.
- [ ] The cache proof requires exactly the four declared determinant-source changes: `supervision_tokens`, `micro_step_runtime_config`, `micro_step_schema`, and `cache_serializer`.
- [ ] Historical pickle bytes remain readable through the old module-path allowlist; new bytes use `src.training.micro_steps`; decoded values are equal.
- [ ] No intermediate cache target is materialized; the single final build and read-only hit proof occur only after fresh exact-packet authorization.
- [ ] The two-rank smoke has independent fresh authorization and bounded stop conditions.
- [ ] All future staging commands name task-owned paths and preserve known user-owned dirty files.
- [ ] OpenSpec remains the sole authority; this plan neither adds nor changes compatibility requirements.
