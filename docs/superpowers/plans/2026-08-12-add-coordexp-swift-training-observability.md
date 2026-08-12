# Add CoordExp-Swift Training Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add truthful every-step/eval training observations with explicit distributed reduction, JSONL-first rank-zero presentation, approximate progress/ETA, and failure-isolated TensorBoard output.

**Architecture:** OpenSpec is the sole authority for behavior, compatibility, and completion; this document only sequences test-first implementation of its [proposal](../../../openspec/changes/add-coordexp-swift-training-observability/proposal.md), [design](../../../openspec/changes/add-coordexp-swift-training-observability/design.md), [tasks](../../../openspec/changes/add-coordexp-swift-training-observability/tasks.md), and [delta specs](../../../openspec/changes/add-coordexp-swift-training-observability/specs/). Implementation begins only after three predecessor changes are synced and archived at pinned commits, then uses four fixed owners: typed reduction in `src/runtime/metrics.py`, row construction in `src/training/reporting.py`, JSONL-first derived publication in `src/artifacts/observation_publisher.py`, and lifecycle composition in `src/training/session.py`.

**Tech Stack:** Python 3.12, PyTorch/Accelerate replicated DDP, CUDA AMP/GradScaler, Pydantic/YAML, pytest, two-process Gloo, TensorBoard `SummaryWriter` and event accumulator, OpenSpec CLI, Git, and the `ms` Conda environment.

## Global Constraints

- Work only in `/data/CoordExp/.worktrees/CoordExp-swift`; execute Python and pytest through `conda run -n ms`.
- `reconcile-coordexp-swift-training-contracts`, `decompose-coordexp-swift-training-orchestration`, and `standardize-coordexp-swift-supervised-losses` MUST be implementation-complete, synced, and archived before this plan starts. Record one exact implementation commit for each; do not accept an active directory, unchecked tasks, or an archive-only date as an implementation pin.
- Rebase this change's complete `Wide-Step Logging Stream` delta against the stable requirement produced by the supervised-loss predecessor before implementation. Also rebase the complete `Optimizer-Step Order`, `Non-Finite Loss And Gradient Gates`, and `Planned Step Schedule` requirements. Preserve every predecessor paragraph/scenario and make only the explicit optimizer-boundary truth/lifecycle refinements owned here.
- `observability.steps` is required, has no default, accepts only positive integers, and controls only console/TensorBoard presentation. It never samples the canonical one-row-per-completed-planned-step/eval `logging.jsonl` stream.
- Every canonical train/eval row requires planned `step`; `logging.jsonl` remains the sole authoritative durable scalar stream.
- Consume prerequisite loss telemetry only from `loss/<term>/raw`, `loss/<term>/weighted`, configured weight, and already-produced counts/denominators. Do not reconstruct loss math, restore `loss/<term>`, or emit an omitted optional term.
- One runtime-owned `AppliedUpdateReceipt` represents every outcome. The post-backward all-rank gate either selects one closed normal `optimizer_boundary_action`—`apply`, `scaler_skip`, or `not_attempted`—or converges one pre-wrapper terminal unsafe receipt. Supported `not_attempted` returns known false booleans and null LRs. `scaler_skip` is allowed only for an all-rank-confirmed current fp16 GradScaler overflow candidate and calls the wrapper once solely to finalize the scaler-owned skip. A previous-step skip flag is never current overflow evidence.
- Every real fp16 wrapper call converges immediate post-call booleans as `all_skipped`, `none_skipped`, or `mixed` before scheduler or normal row handling. `apply` accepts only none-skipped and `scaler_skip` only all-skipped. Mixed truth uses nullable global applied/skip fields; unanimous contradictory truth remains known. All contradictions produce one terminal row then common failed finalization without scheduler/eval/checkpoint/exact-resume/selector/success-final progression.
- `optimizer_step_count` retains its existing meaning: increment once for every completed optimizer-wrapper invocation, including accepted scaler skip and post-wrapper terminal outcomes, and do not increment before a wrapper. Actual application lives only in the receipt/status; do not add or redefine a durable applied counter. The scheduler advances once only for accepted completed `apply`, `scaler_skip`, or supported `not_attempted` boundaries.
- Under fp16, `TrainRuntime.post_backward` calls `accelerator.unscale_gradients(optimizer)` exactly once, then owns current-gradient finiteness/pre-clip norm and the global action. `apply` clips already-unscaled gradients with a non-unscaling primitive such as `torch.nn.utils.clip_grad_norm_`; `scaler_skip` performs no clip before wrapper finalization. No branch calls `accelerator.clip_grad_norm_`. The required post-wrapper consensus is an optimizer-correctness collective, not a metric sink.
- Every durable distributed metric has an explicit reducer. There is no implicit mean, key/suffix inference, mutable registry, callback bus, per-rank durable stream, metrics DB, W&B sink, MFU, TFLOPS, energy estimate, or normal detailed trace.
- Rank zero MUST successfully append and converge the canonical JSONL row before console or TensorBoard consumes it. The first TensorBoard import/init/add/flush/close failure emits at most one bounded warning, latches TensorBoard off, and never invalidates JSONL or fails otherwise-valid training.
- ETA is approximate, segment-local, in-memory presentation state. It is not persisted, restored, or used as scheduling evidence.
- Timing must name an accurately completed scope. Never add a CUDA synchronization solely for observation, relabel enqueue time as GPU execution, reset process-lifetime peak allocator state, or fabricate zero for unavailable metrics.
- `observability` and its presentation/timing/resource state MUST remain outside exact-resume, input-attestation, cache, data, packing, forward, objective, optimizer, scheduler, and RNG semantic identity. The pack-cache determinant payload and fingerprint must remain byte-for-byte unchanged when only `observability.steps` changes.
- Every GPU-backed acceptance action—including each genuine CUDA fp16 finite/overflow arm and the final production-shaped smoke—requires fresh user authorization immediately before launch, bound to the exact commit, command manifest, config, devices, absent artifact roots, and quantitative limits. Approval of this plan is not launch authority.
- Preserve unrelated work. Never reset, clean, broad-stage, overwrite concurrent edits, rebuild a production cache, or mutate an existing cache target.
- Commits below are future execution checkpoints. Stage only named paths, inspect `git diff --cached`, and omit unchanged paths.

---

## Fixed File and Interface Map

- `src/runtime/metrics.py` owns immutable `MetricReducer`, `ScalarMetricSample`, `RatioMetricSample`, `MetricBatch`, `ReducedMetricBatch`, and `reduce_metric_batch(batch: MetricBatch, *, gather_reports: Callable[[MetricBatch], Sequence[MetricBatch]]) -> ReducedMetricBatch`. No other module selects reducers from names.
- `src/runtime/finite_gates.py` owns `OptimizerBoundaryAction`, `Fp16WrapperConsensus`, current-gradient and post-wrapper rank reports, and their all-rank reductions to `apply`, `scaler_skip`, `not_attempted`, one common terminal unsafe result, or `all_skipped|none_skipped|mixed`. It admits `scaler_skip` only when every rank has an active scaler and a non-finite current unscaled gradient; it does not read a previous-step backend skip flag.
- `src/runtime/train_runtime.py` owns the sole `AppliedUpdateReceipt` type, its normal/terminal constructors, fp16 unscale, action-specific norm/clip ordering, pre-call optimizer-group LR sampling, the optimizer wrapper call, immediate post-call skip report, and invocation of the all-fp16-wrapper consensus reducer. Its `optimizer_step(*, planned_step_id: int, expected_action: Literal["apply", "scaler_skip"]) -> AppliedUpdateReceipt` never raises from rank-local post-call truth.
- `src/training/supervised_trainer.py` carries `AppliedUpdateReceipt`, finite-gate diagnostics, exact work counts, and honest timing receipts into `CompletedStepObservation`; a terminal receipt suppresses scheduler and scheduled-handler eligibility but does not format artifact keys or finalize the run.
- `src/training/reporting.py` retains `CompletedStepReporter` and owns `build_train_metric_batch(observation: CompletedStepObservation) -> MetricBatch`, `build_eval_metric_batch(observation: Mapping[str, Any]) -> MetricBatch`, `build_train_row(observation: CompletedStepObservation, reduced: ReducedMetricBatch) -> dict[str, Any]`, and `build_eval_row(observation: Mapping[str, Any], reduced: ReducedMetricBatch) -> dict[str, Any]`. It consumes prerequisite loss artifacts without recomputation.
- `src/artifacts/observation_publisher.py` owns `ObservationPublisher.publish(row, terminal=False)` and `ObservationPublisher.close()`, JSONL/status convergence first, rank-zero console cadence, lazy run-local TensorBoard, approximate ETA, bounded warning, and one-way sink disablement.
- `src/training/session.py` constructs exactly one reporter/publisher pair, routes train/eval rows to it, and owns terminal optimizer-boundary choreography: clear gradients, converge one terminal row, then common failed finalization without success handlers. It closes sinks once in terminal cleanup; `src/training/pipeline.py` remains a facade.
- `src/config/models.py` owns required `ObservabilityConfig(steps: int > 0)` and `TrainConfig.observability`; `src/artifacts/training_state.py` and input-attestation/comparator owners exclude presentation-only state.
- `tests/runtime/test_metrics.py`, `tests/runtime/test_finite_gates.py`, `tests/runtime/test_train_runtime.py`, `tests/training/test_reporting.py`, `tests/artifacts/test_observation_publisher.py`, and `tests/training/test_session_observability.py` are the focused test owners.
- `scripts/probes/coordexp_swift/fp16_grad_scaler_observability.py` is the genuine one-GPU overflow probe; `scripts/probes/coordexp_swift/observability_vertical_smoke.py` verifies final two-rank artifacts without becoming a training entrypoint.

The fixed runtime types are:

```python
class MetricReducer(str, Enum):
    SUM = "sum"
    MAX = "max"
    IDENTICAL = "identical"
    BOOL_ALL = "bool_all"

OptimizerBoundaryAction = Literal["apply", "scaler_skip", "not_attempted"]
Fp16WrapperConsensus = Literal["all_skipped", "none_skipped", "mixed"]
MutationState = Literal[
    "applied",
    "unchanged",
    "optimizer_unchanged_scaler_updated",
    "divergent_or_unknown",
    "corrupted_or_unsafe",
]

@dataclass(frozen=True)
class RankFp16WrapperReport:
    planned_step_id: int
    rank: int
    world_size: int
    step_was_skipped: bool

def reduce_fp16_wrapper_reports(
    reports: Sequence[RankFp16WrapperReport],
) -> Fp16WrapperConsensus: ...

@dataclass(frozen=True)
class ScalarMetricSample:
    name: str
    reducer: MetricReducer
    value: int | float | bool | None
    required: bool
    unavailable_reason: str | None = None

@dataclass(frozen=True)
class RatioMetricSample:
    name: str
    numerator: int | float
    denominator: int | float
    required: bool

@dataclass(frozen=True)
class MetricBatch:
    planned_step_id: int
    split: Literal["train", "eval"]
    samples: tuple[ScalarMetricSample | RatioMetricSample, ...]

@dataclass(frozen=True)
class ReducedMetricBatch:
    planned_step_id: int
    split: Literal["train", "eval"]
    values: Mapping[str, int | float | bool | None]
    unavailable_fields: tuple[str, ...]
    non_finite_fields: tuple[str, ...]

@dataclass(frozen=True)
class AppliedUpdateReceipt:
    planned_step_id: int
    attempted: bool
    applied: bool | None
    step_was_skipped: bool | None
    learning_rates: tuple[float | None, ...]
    unavailable_reason: str | None
    mutation_state: MutationState
    terminal: bool

    @classmethod
    def not_attempted(
        cls,
        planned_step_id: int,
        group_count: int,
        reason: str,
    ) -> "AppliedUpdateReceipt":
        return cls(
            planned_step_id=planned_step_id,
            attempted=False,
            applied=False,
            step_was_skipped=False,
            learning_rates=tuple(None for _ in range(group_count)),
            unavailable_reason=reason,
            mutation_state="unchanged",
            terminal=False,
        )

    @classmethod
    def terminal_not_attempted(
        cls,
        planned_step_id: int,
        group_count: int,
        reason: str,
    ) -> "AppliedUpdateReceipt":
        """Record an fp16 terminal decision made after exactly-once unscale."""
        return cls(
            planned_step_id=planned_step_id,
            attempted=False,
            applied=False,
            step_was_skipped=False,
            learning_rates=tuple(None for _ in range(group_count)),
            unavailable_reason=reason,
            # Parameters and the underlying optimizer are untouched, but
            # GradScaler is already UNSCALED/unfinalized and may differ by rank.
            mutation_state="divergent_or_unknown",
            terminal=True,
        )

    @classmethod
    def terminal_post_wrapper(
        cls,
        *,
        planned_step_id: int,
        learning_rates: tuple[float, ...],
        consensus: Fp16WrapperConsensus,
        expected_action: Literal["apply", "scaler_skip"],
    ) -> "AppliedUpdateReceipt":
        if consensus == "mixed":
            applied, skipped = None, None
            mutation = "divergent_or_unknown"
        elif expected_action == "apply" and consensus == "all_skipped":
            applied, skipped = False, True
            mutation = "optimizer_unchanged_scaler_updated"
        elif expected_action == "scaler_skip" and consensus == "none_skipped":
            applied, skipped = True, False
            mutation = "corrupted_or_unsafe"
        else:
            raise ValueError("terminal constructor requires contradictory truth")
        return cls(
            planned_step_id=planned_step_id,
            attempted=True,
            applied=applied,
            step_was_skipped=skipped,
            learning_rates=(
                learning_rates
                if applied is True
                else tuple(None for _ in learning_rates)
            ),
            unavailable_reason="optimizer_boundary_contradiction",
            mutation_state=mutation,
            terminal=True,
        )
```

If a predecessor ships different exact names or signatures, stop in Task 0 and revise this plan against its pinned implementation; do not add adapters that create two owners.

### Task 0: Pin Predecessors, Rebase Authority, and Freeze the Baseline

**Files:**
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/predecessor-pins.json`
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/wave-0-baseline.md`
- Modify: `openspec/changes/add-coordexp-swift-training-observability/specs/coordexp-swift-config-runtime/spec.md`
- Modify: `openspec/changes/add-coordexp-swift-training-observability/specs/coordexp-swift-training-artifacts/spec.md`
- Modify: `openspec/changes/add-coordexp-swift-training-observability/specs/coordexp-swift-supervision-losses/spec.md`
- Modify: `openspec/changes/add-coordexp-swift-training-observability/tasks.md`

**Interfaces:**
- Consumes: three archived predecessor changes, their synced stable specs, the post-decomposition four-owner graph, the post-loss raw/weighted row schema, and stable optimizer-step/finite-gate/planned-step lifecycle contracts.
- Produces: exact predecessor commit pins, rebased complete deltas, baseline commands/results, fixed interface confirmation, and the pre-change cache determinant identity.

- [ ] **Step 1: Require a clean implementation base**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
```

Expected: status and diff-check produce no output, branch is the intended implementation branch, and HEAD is one 40-character commit. If dirty, stop for the current owner to commit/isolate; do not stash, reset, or clean.

- [ ] **Step 2: Resolve exactly one archived directory and implementation commit per predecessor**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
for change in reconcile-coordexp-swift-training-contracts decompose-coordexp-swift-training-orchestration standardize-coordexp-swift-supervised-losses; do
  test ! -d "openspec/changes/$change"
  matches=(openspec/changes/archive/*-"$change")
  test "${#matches[@]}" -eq 1
  test -f "${matches[0]}/tasks.md"
  test -z "$(rg -n '^- \[ \]' "${matches[0]}/tasks.md")"
  git log -1 --format='%H' -- "${matches[0]}"
done
```

Expected: all tests succeed and exactly three 40-character commits print. Inspect each archive's verification/disposition plus the synced stable diff; use `apply_patch` to write strict JSON mapping each change id to archive path, implementation commit, stable-sync commit, and verification receipt identity. If archive commit and implementation commit differ, record both; never substitute the archive date.

- [ ] **Step 3: Confirm the fixed post-predecessor owners and loss fields**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
test -f src/runtime/metrics.py
test -f src/training/reporting.py
test -f src/artifacts/observation_publisher.py || true
test -f src/training/session.py
rg -n "class CompletedStepReporter|class TrainingSession|class RankControlPlane" src/training src/runtime
rg -n "loss/.+/raw|loss/.+/weighted" src tests openspec/specs/coordexp-swift-training-artifacts
rg -n '"loss/[^/]+"' src tests openspec/specs/coordexp-swift-training-artifacts && exit 1 || true
```

Expected: predecessor-owned `metrics.py`, `reporting.py`, and `session.py` exist with one owner each; the ambiguous loss alias is absent. `observation_publisher.py` may be absent because this change creates it. If predecessor signatures differ from the Fixed File and Interface Map, stop and update this plan before implementation.

- [ ] **Step 4: Rebase the complete modified requirement**

Compare the stable `Wide-Step Logging Stream` block with this change's modified block. Use `apply_patch` so the delta copies the entire post-loss stable requirement and all scenarios verbatim before adding observability paragraphs/scenarios.

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
openspec validate add-coordexp-swift-training-observability --strict
```

Expected: strict validation passes and independent contract review confirms no post-loss field/scenario was dropped, renamed, or retyped.

- [ ] **Step 5: Freeze baseline tests and cache determinant evidence**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/config tests/losses tests/runtime tests/training/test_reporting.py tests/artifacts/test_run_artifacts.py tests/eval/test_forward_eval.py tests/training/test_exact_resume.py tests/training/test_input_attestation.py tests/training/test_pack_cache_determinant_registry.py
```

Expected: exact pass/fail/skip counts are recorded in `wave-0-baseline.md`. Record current train/eval cache determinant JSON and hashes through the predecessor's admitted determinant API; do not materialize or mutate a cache. Any unexpected failure or unresolved P0/P1 standards/intent finding stops Wave 0.

- [ ] **Step 6: Commit prerequisite pins and rebased authority**

```bash
git add openspec/changes/add-coordexp-swift-training-observability/receipts/predecessor-pins.json openspec/changes/add-coordexp-swift-training-observability/receipts/wave-0-baseline.md openspec/changes/add-coordexp-swift-training-observability/specs/coordexp-swift-training-artifacts/spec.md openspec/changes/add-coordexp-swift-training-observability/tasks.md
git diff --cached --check
git diff --cached
git commit -m "docs(openspec): pin observability prerequisites"
```

Expected: only predecessor pins, baseline evidence, the rebased full requirement, and evidence-backed task marks are committed.

### Task 1: Require Explicit Presentation Cadence Without Semantic Drift

**Files:**
- Modify: `src/config/models.py`
- Modify: every YAML returned by `rg --files configs/coordexp_swift/prod configs/coordexp_swift/smoke | sort`
- Modify: `tests/config/test_train_config.py`
- Modify: current accepted training fixture factories found by `rg -n 'TrainConfig\(|_minimal_config\(|schema_version: 1' tests/config tests/training tests/runtime`
- Test: `tests/training/test_pack_cache_determinant_registry.py`

**Interfaces:**
- Consumes: strict `TrainConfig`, resolved-config writer, exact-resume projection, and pinned pre-change cache determinant.
- Produces: required `ObservabilityConfig(steps: int)` with no default and an explicit positive value in every active training config, while cache fingerprint remains unchanged.

- [ ] **Step 1: Write failing strict-config tests**

Add these tests to `tests/config/test_train_config.py` using its existing `_minimal_config`, `_write_yaml`, and `load_train_config` helpers:

```python
def test_observability_steps_is_required_and_persisted(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload.pop("observability", None)
    _write_yaml(config_path, payload)
    with pytest.raises(ConfigContractError):
        load_train_config(config_path)

    payload["observability"] = {"steps": 7}
    _write_yaml(config_path, payload)
    resolved = load_train_config(config_path)
    assert resolved.config.observability.steps == 7
    assert resolved.config_dict["observability"] == {"steps": 7}


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_observability_steps_rejects_non_positive_or_non_integer(
    tmp_path: Path, value: object
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["observability"] = {"steps": value}
    _write_yaml(config_path, payload)
    with pytest.raises(ConfigContractError):
        load_train_config(config_path)
```

- [ ] **Step 2: Run tests and verify the required-field failure**

Run:

```bash
conda run -n ms pytest -q tests/config/test_train_config.py -k observability_steps
```

Expected: FAIL because `TrainConfig` does not yet own required `observability` and/or a missing block is still accepted.

- [ ] **Step 3: Implement the minimal strict model**

Add exactly:

```python
class ObservabilityConfig(StrictConfigModel):
    steps: int = Field(gt=0, strict=True)
```

and add `observability: ObservabilityConfig` to `TrainConfig` with no assignment and no assigned default or default factory. Add no enable flag, sink list, alias, or migration validator.

- [ ] **Step 4: Author active configs and fixtures explicitly**

Inventory paths with:

```bash
rg --files configs/coordexp_swift/prod configs/coordexp_swift/smoke | sort
```

Use `apply_patch` to add one explicit positive `observability.steps` value to each returned training YAML and to every current fixture builder. Leave inference, archive, Stage 1, and Stage 2 roots untouched. Choose cadence as an operator-facing presentation decision; it does not alter train/eval/checkpoint schedules.

- [ ] **Step 5: Add the cache-identity invariance test**

Add to `tests/training/test_pack_cache_determinant_registry.py`:

```python
def test_observability_cadence_does_not_change_pack_cache_identity(
    cache_inputs: _CacheInputs,
) -> None:
    baseline = _fingerprint(cache_inputs)
    baseline_determinants = _determinants(cache_inputs)
    changed_config = cache_inputs.config.model_copy(
        update={
            "observability": cache_inputs.config.observability.model_copy(
                update={"steps": cache_inputs.config.observability.steps + 1}
            )
        }
    )
    assert _fingerprint(cache_inputs, config=changed_config) == baseline
    assert _determinants(cache_inputs, config=changed_config) == baseline_determinants
```

- [ ] **Step 6: Run config, inventory, and cache gates**

Run:

```bash
conda run -n ms pytest -q tests/config/test_train_config.py tests/training/test_pack_cache_determinant_registry.py
conda run -n ms python - <<'PY'
from pathlib import Path
from src.config.loader import load_train_config

paths = sorted(Path("configs/coordexp_swift/prod").rglob("*.yaml"))
paths += sorted(Path("configs/coordexp_swift/smoke").rglob("*.yaml"))
assert paths
for path in paths:
    resolved = load_train_config(path)
    assert resolved.config.observability.steps > 0
print(len(paths))
PY
openspec validate add-coordexp-swift-training-observability --strict
rg -n "logging_steps|training:\n[[:space:]]+logging:|observability.*default" src/config tests/config configs/coordexp_swift/prod configs/coordexp_swift/smoke
```

Expected: pytest and strict validation pass; resolver prints the exact active-config count; residue scan has no accepted legacy alias or observability default; cache identity test proves determinant equality without cache materialization.

- [ ] **Step 7: Commit required cadence and identity proof**

```bash
git add src/config/models.py tests/config/test_train_config.py tests/training/test_pack_cache_determinant_registry.py configs/coordexp_swift/prod configs/coordexp_swift/smoke
git diff --cached --check
git diff --cached
git commit -m "feat(config): require observability cadence"
```

Expected: only config/schema/test changes are committed; no cache artifact is staged.

### Task 2: Replace Implicit Metric Reduction With Typed Batches

**Files:**
- Modify: `src/runtime/metrics.py`
- Modify: `src/runtime/finite_gates.py`
- Modify: `src/runtime/train_runtime.py`
- Modify: `src/eval/forward.py`
- Create: `tests/runtime/test_metrics.py`
- Modify: `tests/eval/test_forward_eval.py`

**Interfaces:**
- Consumes: the fixed metric types in this plan and `RankControlPlane`'s bounded gather transport.
- Produces: `reduce_metric_batch(batch, *, gather_reports) -> ReducedMetricBatch`, explicit producer-owned reducers, and no key-based/default reducer path.

- [ ] **Step 1: Write failing scalar/ratio reducer tests**

Create `tests/runtime/test_metrics.py` with tests that build two asymmetric `MetricBatch` values and assert:

```python
assert reduced.values["grad_norm/pre_clip_rank_max"] == 4.0
assert reduced.values["count/packs"] == 7
assert reduced.values["throughput/physical_tokens_per_second"] == pytest.approx(30 / 4.0)
assert reduced.values["lr/group_0"] == pytest.approx(2.0e-4)
assert reduced.unavailable_fields == ()
```

Add separate tests that require `RuntimeContractError` for metric-name, sample-form, reducer, or required-status disagreement; unequal `IDENTICAL` values; non-positive ratio denominator; and construction without a reducer.

- [ ] **Step 2: Run the new test and verify missing typed interfaces**

Run:

```bash
conda run -n ms pytest -q tests/runtime/test_metrics.py
```

Expected: FAIL on imports for the new fixed metric types/reducer function.

- [ ] **Step 3: Implement immutable metric types and strict reduction**

Implement the exact types from the Fixed File and Interface Map in `src/runtime/metrics.py`. Validate sorted-unique names and identical cross-rank schema before values; reduce `SUM`, `MAX`, `IDENTICAL`, and `BOOL_ALL` explicitly; reduce `RatioMetricSample` by summing numerator and denominator before division; normalize unavailable/non-finite values into sorted bounded field-name tuples. Reject any unclassified metric before durable publication.

- [ ] **Step 4: Migrate train and eval producers one family at a time**

Build typed samples at producer call sites. Accuracy uses summed integer correct/atom statistics; replicated eval uses `IDENTICAL`; disjoint eval uses `SUM` and ratios from sufficient statistics; critical-path timings and pre-clip norm use `MAX`; allocator retry/OOM deltas use `SUM`; applied LR uses `IDENTICAL`; finite flags use `BOOL_ALL`. Keep per-rank reports ephemeral.

- [ ] **Step 5: Add a real two-process CPU Gloo test**

In `tests/runtime/test_metrics.py`, use `torch.multiprocessing.spawn` plus a file-init Gloo process group. Rank 0 reports norm `3`, duration `2`, count `10`, ratio `10/2`; rank 1 reports norm `4`, duration `4`, count `20`, ratio `20/3`. Assert both ranks receive norm `4`, duration `4`, count `30`, ratio `6`, and exit cleanly. Add a second arm with reducer disagreement and assert both ranks converge the same bounded error.

- [ ] **Step 6: Run typed-reducer and eval parity gates**

Run:

```bash
conda run -n ms pytest -q tests/runtime/test_metrics.py tests/runtime/test_train_runtime.py tests/eval/test_forward_eval.py
rg -n "plain.mean|default.*mean|suffix.*reduc|_MAX_REDUCED_METRIC_KEYS|_EVAL_.*METRIC_KEY" src/runtime src/eval src/training
openspec validate add-coordexp-swift-training-observability --strict
```

Expected: tests and strict validation pass; residue scan finds no live implicit mean, name/suffix reducer table, or normal-row per-rank serialization. Obtain standards and intent-contract audits with no unresolved P0/P1.

- [ ] **Step 7: Commit typed metric reduction**

```bash
git add src/runtime/metrics.py src/runtime/train_runtime.py src/eval/forward.py tests/runtime/test_metrics.py tests/runtime/test_train_runtime.py tests/eval/test_forward_eval.py
git diff --cached --check
git diff --cached
git commit -m "refactor(runtime): type metric reduction semantics"
```

Expected: only the typed-metric owners and their focused tests are staged.

### Task 3: Capture Applied Update, Pre-clip Norm, Loss, Work, Timing, and Resources Truthfully

**Files:**
- Modify: `src/runtime/metrics.py`
- Modify: `src/runtime/finite_gates.py`
- Modify: `src/runtime/train_runtime.py`
- Modify: `src/training/supervised_trainer.py`
- Modify: `src/training/reporting.py`
- Modify: `src/artifacts/resources.py`
- Create: `tests/training/test_reporting.py`
- Modify: `tests/runtime/test_finite_gates.py`
- Modify: `tests/runtime/test_train_runtime.py`
- Create: `scripts/probes/coordexp_swift/fp16_wrapper_consensus_gloo.py`
- Create: `tests/runtime/test_fp16_wrapper_consensus_gloo.py`
- Modify: `tests/artifacts/test_resources.py`
- Modify: the exact current provider timing test path located at execution by `rg -l "input_build_seconds|input_wait_seconds|input_h2d_seconds" tests/training`

**Interfaces:**
- Consumes: current-gradient rank reports after exactly-once fp16 unscale, post-wrapper rank skip reports, `GateDecision.diagnostics["max_grad_norm"]`, prerequisite loss artifacts, optimizer param groups, runtime-owned `AppliedUpdateReceipt`, immediate post-call `accelerator.optimizer_step_was_skipped`, provider timing receipts, and CUDA allocator APIs.
- Produces: one closed `optimizer_boundary_action`, one `Fp16WrapperConsensus` when a wrapper runs, `AppliedUpdateReceipt`, enriched `CompletedStepObservation`, typed train batches, canonical train rows, and honest unavailable/non-finite fields without a second loss or gradient authority.

- [ ] **Step 1: Write failing CPU receipt, counter, and LR/off-by-one tests**

Add the full receipt matrix. A finite `apply+none_skipped` update whose optimizer begins at `2.0e-4` records that pre-call LR rather than the scheduler's next value. A supported pre-backward `not_attempted` rejection returns known false booleans with `mutation_state="unchanged"`, leaves `optimizer_step_count` unchanged, publishes null LRs, and advances the planned scheduler once. `scaler_skip+all_skipped` increments the wrapper counter, publishes null LRs, and advances the scheduler without an applied-update counter. Pre-wrapper mixed/unsupported fp16 state after exactly-once unscale returns terminal known-false truth without a wrapper; assert the parameters and underlying optimizer are untouched, GradScaler is already `UNSCALED`/unfinalized, and the composite `mutation_state` is `divergent_or_unknown`, never `unchanged`. Post-wrapper mixed returns nullable global application/skip truth; `apply+all_skipped` preserves false/true and null LRs; `scaler_skip+none_skipped` preserves true/false, the identical pre-call LRs actually applied, and corrupted/unsafe mutation state. Every terminal case clears gradients and yields terminal observation eligibility without scheduler or scheduled handlers; Task 4 proves the durable row and common failed finalization.

Run:

```bash
conda run -n ms pytest -q tests/runtime/test_train_runtime.py tests/training/test_reporting.py -k 'applied_lr or scheduler'
```

Expected: FAIL because optimizer step returns no applied-update receipt and current logging reads scheduler output.

- [ ] **Step 2: Implement the closed boundary action and post-call semantics**

Extend the current-gradient report with scaler-active and exactly-once-unscale context. Replace the pre-call `_accelerator_overflow`/previous-step skip inference with an all-rank reduction that returns `apply` only for finite gradients and norm, `scaler_skip` only when every rank has an active scaler and a current non-finite unscaled gradient, and `not_attempted` for a supported pre-backward or bf16/non-scaler rejection. Mixed-rank or unrelated unsafe fp16 state after unscale returns the same `terminal_not_attempted` receipt on every rank before any wrapper call; its parameters and underlying optimizer remain untouched, but the receipt records `mutation_state="divergent_or_unknown"` because GradScaler is already `UNSCALED`/unfinalized and may differ by rank. No rank raises locally. Change the fp16 branch of `TrainRuntime.optimizer_step(*, planned_step_id: int, expected_action: Literal["apply", "scaler_skip"]) -> AppliedUpdateReceipt` to the following shape:

```python
learning_rates = tuple(float(group["lr"]) for group in self.optimizer.param_groups)
self.optimizer.step()
self.optimizer_step_count += 1
local_skipped = bool(self.accelerator.optimizer_step_was_skipped)
local_report = RankFp16WrapperReport(
    planned_step_id=planned_step_id,
    rank=self.rank,
    world_size=self.world_size,
    step_was_skipped=local_skipped,
)
consensus = reduce_fp16_wrapper_reports(
    self._gather_rank_reports(local_report),
)
if expected_action == "apply" and consensus == "none_skipped":
    return AppliedUpdateReceipt(
        planned_step_id=planned_step_id,
        attempted=True,
        applied=True,
        step_was_skipped=False,
        learning_rates=learning_rates,
        unavailable_reason=None,
        mutation_state="applied",
        terminal=False,
    )
if expected_action == "scaler_skip" and consensus == "all_skipped":
    return AppliedUpdateReceipt(
        planned_step_id=planned_step_id,
        attempted=True,
        applied=False,
        step_was_skipped=True,
        learning_rates=tuple(None for _ in learning_rates),
        unavailable_reason="grad_scaler_overflow",
        mutation_state="optimizer_unchanged_scaler_updated",
        terminal=False,
    )
return AppliedUpdateReceipt.terminal_post_wrapper(
    planned_step_id=planned_step_id,
    learning_rates=learning_rates,
    consensus=consensus,
    expected_action=expected_action,
)
```

Increment the existing `optimizer_step_count` exactly once after every completed wrapper invocation, including terminal post-wrapper outcomes. `reduce_fp16_wrapper_reports(self._gather_rank_reports(local_report))` uses the existing bounded rank-report transport for every fp16 wrapper call and returns exactly `all_skipped`, `none_skipped`, or `mixed`; no rank-local raise precedes it. In supported `not_attempted`, do not call the wrapper or increment that counter. A terminal receipt suppresses scheduler and scheduled handlers, clears gradients, and is routed by `TrainingSession` through terminal-row convergence and failed finalization. Do not add an actual-applied counter or redefine any existing durable counter.

- [ ] **Step 3: Write and pass the fp16 ordering unit test**

Use a fake accelerator call log to assert this exact order for a safe fp16 step:

```python
assert calls == [
    "backward",
    "unscale_gradients",
    "gradient_finite_and_norm",
    "torch_clip_grad_norm",
    "optimizer_step",
    "read_optimizer_step_was_skipped",
    "all_rank_skip_consensus",
    "scheduler_step",
    "zero_grad",
]
```

Also assert this overflow order:

```python
assert calls == [
    "backward",
    "unscale_gradients",
    "gradient_finite_and_norm",
    "reduce_action_scaler_skip",
    "optimizer_step",
    "read_optimizer_step_was_skipped",
    "all_rank_skip_consensus",
    "scheduler_step",
    "zero_grad",
]
```

Implement one idempotence guard per planned step so `TrainRuntime.post_backward` invokes `accelerator.unscale_gradients(self.optimizer)` exactly once before the current-gradient report. Only `apply` then calls a non-unscaling primitive such as `torch.nn.utils.clip_grad_norm_`; `scaler_skip` performs no clip and enters the wrapper only to finalize GradScaler's recorded overflow. Add negative assertions that `accelerator.clip_grad_norm_` is never called, that no clip occurs for `scaler_skip`, and that a mixed-rank pre-call candidate reaches neither wrapper nor scheduler. Add two-rank injected post-call tests for mixed, apply+all-skipped, and scaler_skip+none-skipped: all must reach the same truthful terminal receipt/observation with no scheduler or success-handler eligibility. Task 4 owns durable terminal-row/finalization choreography. Reuse the existing pre-call rank-report transport and one bounded post-call consensus for every fp16 wrapper invocation.

- [ ] **Step 4: Extend observation and row tests without reconstructing loss**

Create a prerequisite-shaped loss artifact containing `loss/base_ce/raw=2.0`, `loss/base_ce/weighted=2.0`, configured weight `1.0`, denominator/counts, plus a deliberately different backend-local backward contribution. Assert `build_train_row(observation, reduced)` preserves `2.0`, never reads the backend contribution, retains a finite zero-weight protected gate, omits a disabled optional family completely, and never emits `loss/base_ce`.

- [ ] **Step 5: Add work/timing/resource observations**

Capture exact physical-token, supervised-atom, and pack counts before step-local tensors are released. Reduce work by sum and divide by rank-max `step_duration_seconds`. Add accurately completed H2D timing only from provider-owned CUDA-event receipts already known complete after forward consumption; otherwise omit it and name it unavailable. Extend the resource collector with current/peak allocated/reserved bytes and retry/OOM counter deltas without resetting peaks.

- [ ] **Step 6: Run focused truth and overhead gates**

Run:

```bash
conda run -n ms pytest -q tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py tests/runtime/test_fp16_wrapper_consensus_gloo.py tests/training/test_reporting.py tests/artifacts/test_resources.py tests/training/test_supervised_trainer.py
conda run -n ms python scripts/probes/coordexp_swift/fp16_wrapper_consensus_gloo.py
rg -n "scheduler.*learning_rates|raw_loss|backend_gradient_scale|_accelerator_overflow|synchronize\(|reset_peak_memory_stats|accelerator\.clip_grad_norm_" src/runtime src/training src/artifacts
openspec validate add-coordexp-swift-training-observability --strict
```

Expected: all tests pass; `_accelerator_overflow` and pre-call reads of prior wrapper-skip state have no live match, and any other matches are classified. No logging path samples post-scheduler LR, reconstructs semantic loss, forces observation-only CUDA synchronization, resets peak state, or starts another normal-path gradient collective.

- [ ] **Step 7: Commit truthful completed-step observations**

```bash
git add src/runtime/metrics.py src/runtime/finite_gates.py src/runtime/train_runtime.py src/training/supervised_trainer.py src/training/reporting.py src/artifacts/resources.py scripts/probes/coordexp_swift/fp16_wrapper_consensus_gloo.py tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py tests/runtime/test_fp16_wrapper_consensus_gloo.py tests/training/test_reporting.py tests/artifacts/test_resources.py tests/training/test_supervised_trainer.py
git diff --cached --check
git diff --cached
git commit -m "feat(training): report applied update and resource truth"
```

Expected: only fixed owners and their focused tests are committed.

### Task 4: Publish Canonical Rows Before Console and TensorBoard

**Files:**
- Create: `src/artifacts/observation_publisher.py`
- Modify: `src/training/reporting.py`
- Modify: `src/training/session.py`
- Create: `tests/artifacts/test_observation_publisher.py`
- Create: `tests/training/test_session_observability.py`
- Modify: `tests/artifacts/test_run_artifacts.py`

**Interfaces:**
- Consumes: strict canonical row mappings including terminal optimizer receipts, `RunWriter.append_logging_row`, `RunWriter.record_warning`, rank/control-plane status convergence, `observability.steps`, resolved max steps, and rank identity.
- Produces: `ObservationPublisher.publish(row, terminal=False) -> None`, `TrainingSession._publish_terminal_optimizer_boundary(observation) -> NoReturn`, and idempotent `close()`, with JSONL-first ordering, common failed finalization, and rank-zero-only derived sinks.

- [ ] **Step 1: Write failing publication-order and cadence tests**

Use spies that append operation names and assert:

```python
assert operations == [
    "jsonl_append",
    "jsonl_status_converged",
    "console",
    "tensorboard_add",
    "tensorboard_flush",
]
```

Cover required `step`, train presentation at positive cadence multiples, off-cadence terminal presentation, every eval observation, exact `step/total` console rendering, non-main silence, and canonical JSONL append on every completed train step regardless of cadence. Add the terminal exception: the current step is written exactly once with truthful nullable receipt/mutation fields, without lifecycle `completed_steps` increment, before common failed finalization. Inject append/status-convergence failure and assert finalization still converges while retaining the optimizer-boundary code as primary.

- [ ] **Step 2: Run tests and verify the owner is absent**

Run:

```bash
conda run -n ms pytest -q tests/artifacts/test_observation_publisher.py tests/training/test_session_observability.py
```

Expected: FAIL because `ObservationPublisher` and session wiring do not exist.

- [ ] **Step 3: Implement the direct publisher interface**

Implement exact public methods `ObservationPublisher.publish(self, row: Mapping[str, Any], *, terminal: bool = False) -> None` and `ObservationPublisher.close(self) -> None`. Constructor inputs are `writer`, `control_plane`, `run_dir`, `is_main_process`, `observability_steps`, and `resolved_max_steps`. Keep one direct method, no event name, registry, subscriber list, or generic sink protocol. Validate `step` and split, append/converge JSONL first, then present only on rank zero. `terminal=True` changes cadence/presentation only; it does not let the publisher decide failure semantics or count the boundary as completed.

- [ ] **Step 4: Implement bounded console/ETA and lazy TensorBoard**

Console shows `step/total`, useful available loss/LR/pre-clip norm/throughput/memory/status, and an explicitly approximate in-memory ETA. Lazily create `SummaryWriter(log_dir=run_dir / "tensorboard")`; map finite numeric row fields deterministically to `train/<canonical-key>` or `eval/<canonical-key>` at `global_step=row["step"]`. Do not emit strings, nulls, nested diagnostics, ETA, or unavailable values as scalars.

- [ ] **Step 5: Write the real TensorBoard event-reader test**

Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` on a temporary run's `tensorboard/` directory. Publish one train and one eval row at step 3, close the publisher, reload events, and assert the expected tags, values, and `event.step == 3`.

- [ ] **Step 6: Inject every TensorBoard failure boundary**

Parametrize import/factory initialization, `add_scalar`, `flush`, and `close` failures. Assert the corresponding JSONL row is already readable, `RunWriter.record_warning("tensorboard_sink_disabled")` occurs once, one bounded stderr warning is attempted, the sink stays disabled, a close failure does not recurse, and later JSONL/eval/checkpoint work continues.

- [ ] **Step 7: Wire one publisher lifecycle into TrainingSession**

Construct one publisher per session, route reporter-produced train rows and eval rows through `publish`, and call `close()` once from terminal cleanup. For `observation.applied_update_receipt.terminal`, skip scheduler and all scheduled/success handlers, clear gradients, build the terminal row without incrementing completed-step state, call `publish(..., terminal=True)`, then make every rank enter the same failed finalization and raise the same bounded optimizer-boundary error. If row publication fails, attach it as secondary context while preserving the primary optimizer-boundary code. Delete predecessor pass-through logging helpers only after all callers move; do not move behavior into `pipeline.py`.

- [ ] **Step 8: Run publisher/session gates and commit**

Run:

```bash
conda run -n ms pytest -q tests/artifacts/test_observation_publisher.py tests/training/test_session_observability.py tests/artifacts/test_run_artifacts.py tests/training/test_reporting.py
rg -n "event.bus|subscribe|registry|wandb|per_rank.*(json|event)|SummaryWriter" src/runtime src/training src/artifacts
openspec validate add-coordexp-swift-training-observability --strict
git add src/artifacts/observation_publisher.py src/training/reporting.py src/training/session.py tests/artifacts/test_observation_publisher.py tests/training/test_session_observability.py tests/artifacts/test_run_artifacts.py
git diff --cached --check
git diff --cached
git commit -m "feat(artifacts): publish JSONL before derived sinks"
```

Expected: tests and strict validation pass; `SummaryWriter` is owned only by `observation_publisher.py`; no alternate durable scalar authority or per-rank event tree exists; terminal-boundary fixtures have one row, failed `run.json`, unchanged completed/scheduler counters, and no eval/checkpoint/exact-resume/selector/success-final artifact.

### Task 5: Preserve Resume, Attestation, and Cache Semantic Identity

**Files:**
- Modify: `src/artifacts/training_state.py`
- Modify: `src/training/input_attestation.py`
- Modify: `scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py`
- Modify: `tests/artifacts/test_training_state.py`
- Modify: `tests/training/test_input_attestation.py`
- Modify: `tests/training/test_wave7_exact_resume_compare_v2.py`
- Modify: `tests/training/test_pack_cache_determinant_registry.py`

**Interfaces:**
- Consumes: exact-resume semantic projection, three-run input-attestation projection, comparator strict/observational field sets, and Task 0/1 cache determinant evidence.
- Produces: presentation-only config drift acceptance, strict scientific/optimizer field comparison, and unchanged cache determinants.

- [ ] **Step 1: Write failing projection and comparator tests**

Create parent/child resolved configs differing only in `observability.steps`, run identity, and resume path; assert semantic projection equality. Add comparator rows differing in ETA/timing/resource/unavailable fields and assert acceptance, then change raw/weighted loss, accuracy, applied LR, pre-clip norm, finite status, and optimizer-update status one at a time and assert strict rejection. Add terminal-boundary fixtures proving no exact-resume/checkpoint/best/success-final payload is emitted and the failed run retains the current step without increasing completed progress.

- [ ] **Step 2: Run the tests and verify observability still enters equality**

Run:

```bash
conda run -n ms pytest -q tests/artifacts/test_training_state.py tests/training/test_input_attestation.py tests/training/test_wave7_exact_resume_compare_v2.py -k observability
```

Expected: FAIL until projections/comparator classify the new fields explicitly.

- [ ] **Step 3: Implement the minimum exclusion sets**

Exclude top-level `observability` beside existing run/resume presentation exclusions. Classify cadence, ETA, timing, resource, and availability fields as observational; keep loss raw/weighted/total, accuracy, applied LR, pre-clip norm, finite status, optimizer-update status, forward/data/order/RNG/optimizer/scheduler semantics strict. Do not ignore all unknown row fields.

- [ ] **Step 4: Re-run cache determinant equality against pinned evidence**

Run:

```bash
conda run -n ms pytest -q tests/training/test_pack_cache_determinant_registry.py -k observability_cadence
```

Expected: PASS with identical fingerprint and determinant payload. Compare the current admitted determinant JSON/hashes to Task 0's pins; any drift stops the change. Do not build a second cache.

- [ ] **Step 5: Run resume/attestation/comparator gates and commit**

Run:

```bash
conda run -n ms pytest -q tests/artifacts/test_training_state.py tests/training/test_input_attestation.py tests/training/test_wave7_exact_resume_compare.py tests/training/test_wave7_exact_resume_compare_v2.py tests/training/test_pack_cache_determinant_registry.py
openspec validate add-coordexp-swift-training-observability --strict
git add src/artifacts/training_state.py src/training/input_attestation.py scripts/probes/coordexp_swift/wave7_exact_resume_compare_v2.py tests/artifacts/test_training_state.py tests/training/test_input_attestation.py tests/training/test_wave7_exact_resume_compare_v2.py tests/training/test_pack_cache_determinant_registry.py
git diff --cached --check
git diff --cached
git commit -m "test(training): isolate observability from semantic identity"
```

Expected: no cache payload, run artifact, or unrelated probe is staged.

### Task 6: Qualify Genuine CUDA fp16 Finite/Overflow Arms and Final Two-rank Smoke

**Files:**
- Create: `scripts/probes/coordexp_swift/fp16_grad_scaler_observability.py`
- Create: `scripts/probes/coordexp_swift/observability_vertical_smoke.py`
- Create: `tests/training/test_fp16_grad_scaler_observability_probe.py`
- Create: `tests/training/test_observability_vertical_smoke.py`
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-command-manifest.json`
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-launch-packet.md`
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-terminal-receipt.json`

**Interfaces:**
- Consumes: exact final implementation commit, real CUDA Accelerate fp16 optimizer wrapper/GradScaler, smallest current two-rank one-step+eval config, fixed publisher artifacts, and absent probe roots.
- Produces: one-GPU finite and overflow arm evidence plus two-GPU production-shaped artifact evidence; none is performance or model-quality evidence.

- [ ] **Step 1: Implement model-free validators and unit-test them**

The fp16 validator requires separate real-Accelerate CUDA finite and overflow receipts. Both prove one `accelerator.unscale_gradients(optimizer)`, then current-gradient finite/norm inspection, one post-wrapper correctness consensus, and zero `accelerator.clip_grad_norm_` calls. The finite arm proves `optimizer_boundary_action=apply`, consensus `none_skipped`, exactly one non-unscaling clip, `attempted=True`, `applied=True`, `step_was_skipped=False`, pre-call group LR, and one wrapper-count increment. The overflow arm proves `optimizer_boundary_action=scaler_skip`, consensus `all_skipped`, zero clip calls, exactly one wrapper call for scaler finalization, `attempted=True`, `applied=False`, `step_was_skipped=True`, null LRs, unchanged parameters, one wrapper-count increment, and normal planned scheduler advance. The CPU/Gloo consensus probe must already have proven the mixed and unanimous-contradictory terminal matrix. The vertical validator requires exactly one shared run tree, one train row and one eval row at the expected step, readable rank-zero TensorBoard tags, no rank-local run/event trees, strict loss/LR/norm/status fields, and bounded timing/resource availability.

Run:

```bash
conda run -n ms pytest -q tests/training/test_fp16_grad_scaler_observability_probe.py tests/training/test_observability_vertical_smoke.py
```

Expected: validators pass on complete fixtures and reject wrong skip timing, LR, order, row count, event step, rank ownership, or identity.

- [ ] **Step 2: Freeze exact GPU argv arrays without launching**

Use `apply_patch` to write strict JSON with schema `coordexp-swift-observability-gpu-command-manifest-v1`, exact implementation commit, cwd, environment, separate `fp16_finite`, `fp16_overflow`, and `two_rank_vertical` argv arrays, exact config paths and SHA256 values, selected device UUIDs, absent artifact roots, and `authorization_status: not_requested`. Each fp16 arm uses exactly one GPU and one planned step; the vertical arm uses `world_size=2`, at most two GPUs, one finite applied update, and one scheduled eval.

- [ ] **Step 3: Declare quantitative bounds and stop rules**

In `gpu-launch-packet.md`, bind the manifest SHA256 and record for each arm: max model forwards, collective count/order ceiling, per-rank CPU RSS and GPU-memory high-water ceiling, per-arm/total wall timeout, new artifact byte limit, required free disk, occupied-target check, shared-GPU baseline, and no-retry rule. `fp16_finite` and `fp16_overflow` must use the same tiny synthetic CUDA model through real Accelerate; the overflow arm induces genuine GradScaler overflow and neither loads the production model.

Run read-only preflight:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
git status --short
git rev-parse HEAD
sha256sum openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-command-manifest.json
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader
df -B1 /data/CoordExp/.worktrees/CoordExp-swift
```

Expected: commit/config/manifest bindings match, both artifact roots are absent, and declared headroom is available. Any drift or insufficient headroom stops before authorization.

- [ ] **Step 4: Obtain fresh authorization immediately before each GPU arm**

Present the exact packet. Obtain explicit user approval immediately before each one-GPU fp16 arm and separately before the two-GPU vertical arm. Earlier approval, planning approval, or approval bound to another commit/root is invalid.

- [ ] **Step 5: Run the genuine fp16 finite and overflow arms once each**

Execute each frozen argv array unchanged. Expected for both: one real Accelerate unscale, then current-gradient finite/norm authority, one post-wrapper correctness consensus, and no `accelerator.clip_grad_norm_`. The finite arm selects `apply`, converges `none_skipped`, performs exactly one non-unscaling clip, applies the update, publishes pre-call LRs, and increments `optimizer_step_count` once. The overflow arm selects `scaler_skip`, performs no clip, calls the wrapper once only to finalize GradScaler's recorded overflow, converges `all_skipped`, leaves parameters unchanged, publishes null/unavailable LRs, increments `optimizer_step_count` once for the completed wrapper, and advances the planned scheduler normally. Stop without retry on unexpected action/consensus, parameter mutation in the overflow arm, timeout, OOM, command drift, or bound exceedance.

- [ ] **Step 6: Run the two-rank production-shaped arm once**

Execute the frozen argv array unchanged. Expected: one finite update and scheduled eval produce one shared run tree, exactly one canonical train row and one canonical eval row, correct strict fields, readable rank-zero TensorBoard events, and no rank-local event/run tree. Record counters/resource maxima and make no throughput/model-quality claim.

- [ ] **Step 7: Verify durable artifacts and commit qualification code/evidence**

Run both model-free validators against the immutable output roots and write `gpu-terminal-receipt.json` with command/config/commit/device/artifact identities, every rank terminal status, resource maxima, and stop outcome.

```bash
git add scripts/probes/coordexp_swift/fp16_grad_scaler_observability.py scripts/probes/coordexp_swift/observability_vertical_smoke.py tests/training/test_fp16_grad_scaler_observability_probe.py tests/training/test_observability_vertical_smoke.py openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-command-manifest.json openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-launch-packet.md openspec/changes/add-coordexp-swift-training-observability/receipts/gpu-terminal-receipt.json
git diff --cached --check
git diff --cached
git commit -m "test(training): qualify observability GPU boundaries"
```

Expected: large generated run/probe artifacts remain outside Git; only validators, tests, and bounded receipts are committed.

### Task 7: Update Canonical Docs and Complete Final Verification

**Files:**
- Modify: `docs/COORDEXP_SWIFT.md`
- Modify: `docs/SYSTEM_OVERVIEW.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/ARTIFACTS.md`
- Create: `openspec/changes/add-coordexp-swift-training-observability/receipts/final-verification.md`
- Modify: `openspec/changes/add-coordexp-swift-training-observability/tasks.md`

**Interfaces:**
- Consumes: final emitted schemas/receipts, stable/delta specs, cache identity proof, and GPU qualification.
- Produces: concise operator docs, full regression receipt, independent audits, and an honest verify/sync/archive decision.

- [ ] **Step 1: Patch only current operator-facing truth**

Use `apply_patch` to document required `observability.steps`, every-step/eval JSONL authority, explicit reducer meanings, applied LR and pre-clip rank-max norm, run-local `tensorboard/`, approximate ETA, JSONL-first failure policy, availability semantics, and probe-only exclusions. Keep schema detail in stable specs; do not revive historical Trainer/event-bus terminology.

- [ ] **Step 2: Run focused and broader regressions**

Run:

```bash
conda run -n ms pytest -q tests/config tests/losses tests/runtime tests/artifacts tests/training tests/eval
conda run -n ms python - <<'PY'
from pathlib import Path
from src.config.loader import load_train_config

for root in (Path("configs/coordexp_swift/prod"), Path("configs/coordexp_swift/smoke")):
    for path in sorted(root.rglob("*.yaml")):
        assert load_train_config(path).config.observability.steps > 0
PY
```

Expected: exit code 0 with exact pass/fail/skip counts; every unexpected skip is investigated. Every active config resolves an explicit positive cadence.

- [ ] **Step 3: Run strict artifact, identity, and residue gates**

Run:

```bash
openspec validate add-coordexp-swift-training-observability --strict
rg -n "implicit.*mean|default.*reducer|loss/[^/]+[\"']|scheduler.*learning_rates|per_rank.*(json|event)|wandb|event.bus|MFU|TFLOPS|energy" src/runtime src/training src/artifacts docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md
conda run -n ms pytest -q tests/training/test_pack_cache_determinant_registry.py -k observability_cadence
git diff --check
```

Expected: strict validation and cache-identity test pass; residue matches are limited to explicit rejection/probe-only documentation; no alias, alternate authority, implicit reducer, scheduler-derived applied LR, or per-rank sink remains.

- [ ] **Step 4: Obtain two independent final audits**

Obtain separate standards/code-quality and intent/spec-contract verdicts over implementation, emitted JSONL/TensorBoard artifacts, GPU receipts, cache equality, resume exclusions, and docs. Resolve every P0/P1 finding and record lower-severity disposition in `final-verification.md`.

- [ ] **Step 5: Commit final docs and verification**

```bash
git add docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md openspec/changes/add-coordexp-swift-training-observability/receipts/final-verification.md openspec/changes/add-coordexp-swift-training-observability/tasks.md
git diff --cached --check
git diff --cached
git commit -m "docs(training): finalize observability contract"
```

Expected: every checked task has executed evidence and no unrelated documentation is staged.

- [ ] **Step 6: Verify, sync, and archive only when complete**

Run `openspec-verify-change`; if all gates pass, use `openspec-sync-specs`, inspect the merged full stable requirements, then use `openspec-archive-change`.

Final command:

```bash
openspec validate --all --strict
```

Expected: all stable specs and active changes validate. If any gate fails, keep the change active or archive explicitly incomplete; do not sync unsupported semantics or claim the observability feature complete.
