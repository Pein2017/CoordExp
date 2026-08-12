# Standardize CoordExp-Swift Supervised Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the SFT-only objective, zero-policy, distributed-normalization, config-migration, and loss-telemetry change owned by OpenSpec change [`standardize-coordexp-swift-supervised-losses`](../../../openspec/changes/standardize-coordexp-swift-supervised-losses/proposal.md).

**Architecture:** OpenSpec is the sole authority for scope, scientific meaning, compatibility, and completion: read its [`proposal.md`](../../../openspec/changes/standardize-coordexp-swift-supervised-losses/proposal.md), [`design.md`](../../../openspec/changes/standardize-coordexp-swift-supervised-losses/design.md), [`tasks.md`](../../../openspec/changes/standardize-coordexp-swift-supervised-losses/tasks.md), and [`delta specs`](../../../openspec/changes/standardize-coordexp-swift-supervised-losses/specs/) before every task. This plan supplies only the RED-GREEN-REFACTOR order, exact final-owner paths and interfaces, commands, expected evidence, and future commit boundaries; when it differs from OpenSpec, stop and update the plan after OpenSpec is corrected.

**Tech Stack:** Python 3.12 in Conda environment `ms`, Pydantic strict configuration, PyTorch fp32 loss math and autograd, Accelerate replicated DDP, pytest, strict JSONL run artifacts, immutable pack-cache identities, OpenSpec CLI, Git.

## Global Constraints

- Work only from `/data/CoordExp/.worktrees/CoordExp-swift`; every Python and pytest command below uses `conda run -n ms`.
- Start only after `reconcile-coordexp-swift-training-contracts` and `decompose-coordexp-swift-training-orchestration` are implementation-complete, verified, synchronized, archived, and committed. Pin both archive commits and require the checked-out base to be the exact committed decompose result.
- The final owner graph is a prerequisite, not a suggestion: loss construction remains in `src/losses/runner.py`; initialized composition uses `src/training/session.py`; completed-step projection uses `src/training/reporting.py`; do not move behavior back into `src/training/pipeline.py` or create a parallel reporter/session.
- This change is supervised training plus forward eval only. Do not add rollout, policy, value, KL, reward, hidden-state, RL-composition, public plugin, import-path, callable-config, registry-discovery, or framework behavior.
- Base CE is required at weight exactly `1.0`. The protected gate uses exactly ordered groups `desc_text,schema,coordinate,eos`; `enabled` requires weight `0.1`, and `zero_weight_ablation` requires weight `0`.
- A zero-weight protected gate remains a detached computed diagnostic and participates in fail-closed all-rank finite consensus. A zero-weight optional coordinate Gaussian/RPS auxiliary is wholly omitted from construction, denominators, execution, finite checks, metrics, and retained graph state.
- Preserve fp32 per-atom math, complete-planned-step `segment_balanced` normalization, and exactly one world-size compensation for Accelerate/DDP mean-gradient reduction. Semantic raw/weighted telemetry never includes backend compensation.
- Do not write ambiguous `loss/<term>` aliases. Computed terms use explicit raw and weighted fields; disabled optional terms emit no field family; historical JSONL and historical configs are never rewritten.
- The predecessor's cache transition is final. Loss/config/reporting/session paths must remain outside `PACKING_CACHE_DETERMINANT_OWNERS`; every task runs the hash guard, and any train/eval determinant payload or aggregate-hash change stops the change. Never materialize, repair, delete, overwrite, or publish a cache target in this plan.
- Planning or implementation approval is not DDP/GPU launch authority. Every distributed or GPU-backed action requires a fresh packet naming the exact commit, full command, devices/world size, config, artifact root, cache identities, model-forward and collective bounds, planned steps, wall time, RSS/GPU-memory ceilings, artifact-byte ceiling, zero cache passes, and stop conditions.
- Preserve unrelated dirty work. Every future `git add` below names only task-owned paths; inspect `git status --short`, `git diff --check`, and `git diff --cached` before committing.
- Commit commands are future execution checkpoints. Each task ends in one independently reviewable and revertible commit; do not stage, commit, or launch anything while merely reading this plan.

---

## File and Ownership Map

- `src/config/models.py`: strict authored SFT loss schema and migration-oriented validation.
- `src/losses/bindings.py`: closed internal `TokenLossBinding` metadata; it is not exported as a public plugin surface.
- `src/losses/runner.py`: the single deep owner of term construction, planned-step denominators, micro-step contributions, semantic finalization, and canonical loss metric fields.
- `src/runtime/finite_gates.py`: protected raw-diagnostic finite reports and all-rank fail-closed consensus.
- `src/runtime/train_runtime.py`: existing metric collective and optimizer-update boundary; no second loss reducer or collective is introduced.
- `src/training/session.py`: post-decomposition initialized-run composition; it constructs the configured `LossRunner` and passes it to trainer/eval owners.
- `src/training/reporting.py`: post-decomposition `CompletedStepReporter`; it writes the final loss mapping without reconstructing objective semantics.
- `src/eval/forward.py`: forward-eval consumer of the same `LossRunner.finalize_planned_step` schema.
- `src/artifacts/run_schema.py` and `src/artifacts/run_writer.py`: strict scalar normalization and single-writer JSONL publication; they validate rows but do not infer term meaning.
- `tests/config/test_train_config.py` and `tests/config/test_supervised_loss_config_inventory.py`: strict config and current-inventory acceptance.
- `tests/losses/test_bindings.py`, `tests/losses/test_runner.py`, `tests/losses/test_zero_policy_contract.py`, and `tests/losses/test_ddp_objective_parity.py`: closed composition, zero policies, semantic/backward separation, and DDP parity.
- `tests/runtime/test_finite_gates.py` and `tests/runtime/test_train_runtime.py`: non-finite gate consensus and cross-rank sufficient-statistic reduction.
- `tests/training/test_reporting.py`, `tests/training/test_training_session.py`, `tests/training/test_supervised_trainer.py`, and `tests/training/test_loss_cache_invariance.py`: final-owner wiring, canonical rows, and cache-hash guard.
- `tests/eval/test_forward_eval.py` and `tests/artifacts/test_run_artifacts.py`: eval/train row parity, disabled-field omission, and JSON-null normalization.
- `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and `docs/ARTIFACTS.md`: canonical operator documentation, updated only after executable acceptance.

The current supported config migration owns exactly the YAML files returned by this command at the pinned base:

```bash
rg -l '^losses:|^[[:space:]]+losses:' configs/coordexp_swift/prod configs/coordexp_swift/smoke --glob '*.yaml' | sort
```

At plan-writing time that inventory is five production files, sixteen top-level smoke files, and four `smoke/length_isolation` files. Task 1 freezes their exact paths and SHA256 values before Task 2 edits any config; a later added/removed file stops for an OpenSpec inventory review rather than being silently included.

### Task 1: Pin predecessors, freeze commands, and install the cache-identity guard

**OpenSpec owner:** Wave 0, Design context, Migration steps 1 and 6.

**Files:**
- Create: `openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/wave-0-loss-baseline.json`
- Create: `tests/training/test_loss_cache_invariance.py`
- Modify: none outside those two files.

**Interfaces:**
- Consumes: the exact archived reconcile/decompose commits; final `TrainingSession` and `CompletedStepReporter` owners; `src.training.pack_cache.PACKING_CACHE_DETERMINANT_OWNERS`; predecessor train/eval cache manifests and admission receipts.
- Produces: `assert_loss_change_is_not_cache_owned() -> None`; a strict baseline receipt containing exact predecessor/base commits, command manifest, supported-config inventory, train/eval determinant payload SHA256, aggregate hashes, immutable target paths, and cache-hit status.

- [ ] **Step 1: Prove the exact predecessor chain and final owners**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
test ! -d openspec/changes/reconcile-coordexp-swift-training-contracts
test ! -d openspec/changes/decompose-coordexp-swift-training-orchestration
reconcile_archive="$(find openspec/changes/archive -maxdepth 1 -type d -name '*-reconcile-coordexp-swift-training-contracts' -print -quit)"
decompose_archive="$(find openspec/changes/archive -maxdepth 1 -type d -name '*-decompose-coordexp-swift-training-orchestration' -print -quit)"
test -n "$reconcile_archive"
test -n "$decompose_archive"
reconcile_commit="$(git log -1 --format=%H -- "$reconcile_archive")"
decompose_commit="$(git log -1 --format=%H -- "$decompose_archive")"
test "$(git rev-parse HEAD)" = "$decompose_commit"
test -f src/training/session.py
test -f src/training/reporting.py
test -f src/training/cache_contract.py
test -f src/training/cache_workflow.py
git merge-base --is-ancestor "$reconcile_commit" "$decompose_commit"
git status --short
```

Expected: both archives and nonempty commits exist, reconcile is an ancestor of decompose, `HEAD` equals the decompose commit, all four final-owner files exist, and the worktree is clean. Any failure stops execution; do not infer a predecessor from branch names or an incomplete archive.

- [ ] **Step 2: Write the RED cache-ownership test**

Create `tests/training/test_loss_cache_invariance.py` with these exact protected paths and assertions:

```python
from src.training.pack_cache import PACKING_CACHE_DETERMINANT_OWNERS


LOSS_CHANGE_PATHS = {
    "src/config/models.py",
    "src/losses/bindings.py",
    "src/losses/runner.py",
    "src/runtime/finite_gates.py",
    "src/runtime/train_runtime.py",
    "src/training/session.py",
    "src/training/reporting.py",
    "src/eval/forward.py",
    "src/artifacts/run_schema.py",
    "src/artifacts/run_writer.py",
}


def assert_loss_change_is_not_cache_owned() -> None:
    owners = set(PACKING_CACHE_DETERMINANT_OWNERS.values())
    assert LOSS_CHANGE_PATHS.isdisjoint(owners), sorted(LOSS_CHANGE_PATHS & owners)


def test_loss_change_paths_are_not_pack_cache_determinant_owners() -> None:
    assert_loss_change_is_not_cache_owned()
```

Run:

```bash
conda run -n ms pytest tests/training/test_loss_cache_invariance.py -q
```

Expected: FAIL only if the final predecessor owner graph made a loss/config/reporting/session path cache-semantic. Such a failure is a blocking architecture conflict; do not weaken the assertion or build another cache.

- [ ] **Step 3: Freeze exact cache and config identities in a strict receipt**

Read the predecessor's completed cache-transition receipt and both admitted immutable manifests. Create `wave-0-loss-baseline.json` with exactly the keys and value types below; populate every value from executed Git, inventory, manifest, and cache-hit reads rather than example strings:

```python
class ConfigInventoryEntry(TypedDict):
    path: str
    sha256: str


class CacheBaselineEntry(TypedDict):
    target: str
    determinant_payload_sha256: str
    aggregate_fingerprint: str
    cache_status: Literal["hit"]


class LossBaselineReceipt(TypedDict):
    schema: Literal["coordexp-swift-supervised-loss-baseline-v1"]
    reconcile_commit: str
    decompose_commit: str
    loss_base_commit: str
    supported_config_inventory: list[ConfigInventoryEntry]
    train: CacheBaselineEntry
    eval: CacheBaselineEntry
    cache_materialization_passes_authorized: Literal[0]
```

The literal example path is replaced by every exact path emitted by the frozen inventory command; the schema is validated by a test in this task, not accepted by visual inspection. Add assertions that commits are 40 lowercase hex, hashes are 64 lowercase hex, paths are absolute/unique, both targets exist and are directories, both statuses equal `hit`, and the authorized pass count equals integer zero.

- [ ] **Step 4: Freeze the exact CPU command manifest and run the entry gate**

Record these commands verbatim in the receipt's adjacent OpenSpec task evidence before implementation:

```bash
conda run -n ms pytest tests/config/test_train_config.py tests/losses/test_context_and_terms.py tests/losses/test_coord_gaussian_rps.py tests/losses/test_normalizers.py tests/losses/test_runner.py tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py tests/training/test_supervised_trainer.py tests/training/test_reporting.py tests/training/test_training_session.py tests/training/test_loss_cache_invariance.py tests/eval/test_forward_eval.py tests/artifacts/test_run_artifacts.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
```

Expected: all predecessor tests pass, the cache-owner test passes, strict validation passes, and exact pass/fail/skip counts are recorded. Resolve the entry standards/intent audit and every P0/P1 before Task 2.

- [ ] **Step 5: Commit the pinned baseline**

```bash
git diff --check
git add openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/wave-0-loss-baseline.json tests/training/test_loss_cache_invariance.py
git diff --cached --check
git diff --cached
git commit -m "test(losses): pin supervised loss and cache baseline"
```

Expected: only the strict receipt and cache-invariance test are committed; the commit is independently revertible and creates no cache/artifact output.

### Task 2: Enforce the strict SFT config and migrate only current configs

**OpenSpec owner:** Wave 1; Config-runtime delta; Design decisions 1 and 6.

**Files:**
- Modify: `src/config/models.py`
- Modify: `tests/config/test_train_config.py`
- Create: `tests/config/test_supervised_loss_config_inventory.py`
- Modify: the exact 25 YAML paths frozen in Task 1 under `configs/coordexp_swift/prod/` and `configs/coordexp_swift/smoke/`; no file outside that receipt inventory.

**Interfaces:**
- Consumes: `StrictConfigModel`, `WeightedLossConfig`, `CoordGaussianRPSLossConfig`, and Task-1 config inventory.
- Produces: `BaseCELossConfig`; `TokenTypeGateLossConfig(mode, weight, groups)`; `ProtectedLossesConfig(base_ce, token_type_gate)`; `AuxiliaryLossesConfig(coord_gaussian_rps)`; `LossesConfig(normalizer, protected, auxiliary)`.

- [ ] **Step 1: Write RED tests for every strict constant and forbidden surface**

Add tests with these exact assertions:

```python
CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")


def parse_losses(
    *,
    base_weight: float,
    gate_mode: str,
    gate_weight: float,
) -> LossesConfig:
    return LossesConfig.model_validate(
        {
            "normalizer": "segment_balanced",
            "protected": {
                "base_ce": {"weight": base_weight},
                "token_type_gate": {
                    "mode": gate_mode,
                    "weight": gate_weight,
                    "groups": list(CANONICAL_GROUPS),
                },
            },
            "auxiliary": {"coord_gaussian_rps": {"weight": 0.0}},
        }
    )


def test_canonical_enabled_sft_loss_config() -> None:
    losses = parse_losses(base_weight=1.0, gate_mode="enabled", gate_weight=0.1)
    assert losses.protected.base_ce.weight == 1.0
    assert losses.protected.token_type_gate.mode == "enabled"
    assert losses.protected.token_type_gate.weight == 0.1
    assert losses.protected.token_type_gate.groups == CANONICAL_GROUPS


def test_named_gate_ablation_is_explicit() -> None:
    losses = parse_losses(
        base_weight=1.0,
        gate_mode="zero_weight_ablation",
        gate_weight=0.0,
    )
    assert losses.protected.token_type_gate.model_dump(mode="json") == {
        "mode": "zero_weight_ablation",
        "weight": 0.0,
        "groups": list(CANONICAL_GROUPS),
    }


@pytest.mark.parametrize("weight", (0.0, 0.5, 1.1))
def test_base_ce_rejects_noncanonical_weight(weight: float) -> None:
    with pytest.raises(ValidationError, match="base_ce.*1.0"):
        parse_losses(base_weight=weight, gate_mode="enabled", gate_weight=0.1)


@pytest.mark.parametrize(
    ("mode", "weight"),
    (("enabled", 0.0), ("enabled", 0.2), ("zero_weight_ablation", 0.1)),
)
def test_gate_mode_weight_pairs_fail_closed(mode: str, weight: float) -> None:
    with pytest.raises(ValidationError, match="token_type_gate"):
        parse_losses(base_weight=1.0, gate_mode=mode, gate_weight=weight)
```

Also mutate the ordered group tuple by omission, duplication, addition, and reordering; author `protected.coord_gaussian_rps`; author `implementation`, `callable`, `factory`, `registry`, an unknown loss name, and an RL loss. Every case must raise before `LossRunner.from_config` or model construction.

Run:

```bash
conda run -n ms pytest tests/config/test_train_config.py -q
```

Expected: FAIL because the current schema lacks the discriminated mode, permits noncanonical constants/groups, and still owns coordinate loss under `protected`.

- [ ] **Step 2: Implement only the term-specific strict models**

Use exact model structure and validation:

```python
CanonicalTokenType = Literal["desc_text", "schema", "coordinate", "eos"]
CANONICAL_TOKEN_TYPE_GATE_GROUPS: tuple[CanonicalTokenType, ...] = (
    "desc_text",
    "schema",
    "coordinate",
    "eos",
)


class BaseCELossConfig(WeightedLossConfig):
    @model_validator(mode="after")
    def _require_canonical_weight(self) -> "BaseCELossConfig":
        if self.weight != 1.0:
            raise ValueError("losses.protected.base_ce.weight must equal 1.0")
        return self


class TokenTypeGateLossConfig(WeightedLossConfig):
    mode: Literal["enabled", "zero_weight_ablation"]
    groups: tuple[CanonicalTokenType, ...]

    @model_validator(mode="after")
    def _require_canonical_mode_weight_and_groups(self) -> "TokenTypeGateLossConfig":
        expected_weight = 0.1 if self.mode == "enabled" else 0.0
        if self.weight != expected_weight:
            raise ValueError(
                f"losses.protected.token_type_gate mode={self.mode} requires weight={expected_weight}"
            )
        if self.groups != CANONICAL_TOKEN_TYPE_GATE_GROUPS:
            raise ValueError(
                "losses.protected.token_type_gate.groups must equal "
                "[desc_text, schema, coordinate, eos] in that order"
            )
        return self


class ProtectedLossesConfig(StrictConfigModel):
    base_ce: BaseCELossConfig
    token_type_gate: TokenTypeGateLossConfig


class AuxiliaryLossesConfig(StrictConfigModel):
    coord_gaussian_rps: CoordGaussianRPSLossConfig = Field(
        default_factory=CoordGaussianRPSLossConfig
    )


class LossesConfig(StrictConfigModel):
    normalizer: Literal["segment_balanced"]
    protected: ProtectedLossesConfig
    auxiliary: AuxiliaryLossesConfig = Field(default_factory=AuxiliaryLossesConfig)
```

Do not add field aliases, pre-validation migration, arbitrary mappings, or implementation identifiers.

- [ ] **Step 3: Migrate the frozen supported inventory and test it exactly**

For every frozen YAML:

- retain `losses.normalizer: segment_balanced` and `base_ce.weight: 1.0`;
- write `mode: enabled` plus `weight: 0.1` for enabled gate runs;
- write `mode: zero_weight_ablation` plus `weight: 0.0` for pure-CE runs;
- retain the exact ordered four-group list;
- move positive or authored-zero coordinate config to `losses.auxiliary.coord_gaussian_rps`;
- do not edit `configs/archive`, historical docs, or completed artifacts.

Create `test_supervised_loss_config_inventory.py` so it compares the current `rg` path list to the receipt list, strictly resolves every path, asserts the exact mode/weight/group invariants, and asserts Task-1 historical fixture bytes are unchanged.

Run:

```bash
conda run -n ms pytest tests/config/test_train_config.py tests/config/test_supervised_loss_config_inventory.py -q
conda run -n ms pytest tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
```

Expected: PASS; exactly the frozen current configs resolve, representative old layouts fail current validation without being modified, and the cache-owner guard remains green.

- [ ] **Step 4: Commit the schema and current-config migration**

```bash
git diff --check
git add src/config/models.py tests/config/test_train_config.py tests/config/test_supervised_loss_config_inventory.py configs/coordexp_swift/prod/*.yaml configs/coordexp_swift/smoke/*.yaml configs/coordexp_swift/smoke/length_isolation/*.yaml
git diff --cached --check
git diff --cached
git commit -m "feat(config): standardize supervised loss schema"
```

Expected: the commit contains only strict schema/tests and frozen current YAML files; its revert restores Task 1 without altering caches or historical files.

### Task 3: Implement closed composition and distinct zero policies

**OpenSpec owner:** Wave 2; Supervision-losses delta; Design decisions 2 and 3.

**Files:**
- Create: `src/losses/bindings.py`
- Modify: `src/losses/runner.py`
- Modify: `src/losses/__init__.py`
- Create: `tests/losses/test_bindings.py`
- Create: `tests/losses/test_zero_policy_contract.py`
- Modify: `tests/losses/test_runner.py`
- Modify: `tests/training/test_supervised_trainer.py`

**Interfaces:**
- Consumes: Task-2 strict config and existing `BaseTokenCE`, `TokenTypeGateLoss`, `CoordGaussianRPSLoss`, `SegmentBalancedDenominator`, `LossContext`.
- Produces: frozen `TokenLossBinding(name, role, normalizer, zero_weight_policy, token_types)`; closed `TOKEN_LOSS_BINDINGS`; `LossTermResult.backward_contribution`; unchanged `LossRunner.prepare_planned_step`, `compute_micro_step`, and `finalize_planned_step` call signatures.

- [ ] **Step 1: Write RED closed-binding tests**

Use exact metadata:

```python
def test_token_loss_binding_inventory_is_closed_and_exact() -> None:
    assert TOKEN_LOSS_BINDINGS == (
        TokenLossBinding("base_ce", "protected", "segment_balanced", "forbid", None),
        TokenLossBinding(
            "token_type_gate",
            "protected",
            "segment_balanced",
            "detached_diagnostic",
            ("desc_text", "schema", "coordinate", "eos"),
        ),
        TokenLossBinding(
            "coord_gaussian_rps",
            "auxiliary",
            "segment_balanced",
            "omit",
            ("coordinate",),
        ),
    )
```

Assert `TokenLossBinding` is frozen, the tuple is not derived from config names, `src.losses.__all__` does not expose a registry/factory/plugin API, and unknown/RL config already fails in Task 2.

Run:

```bash
conda run -n ms pytest tests/losses/test_bindings.py -q
```

Expected: FAIL because `src/losses/bindings.py` does not exist.

- [ ] **Step 2: Add the closed metadata without a factory framework**

Implement:

```python
LossRole = Literal["protected", "auxiliary"]
LossNormalizer = Literal["segment_balanced"]
ZeroWeightPolicy = Literal["forbid", "detached_diagnostic", "omit"]


@dataclass(frozen=True)
class TokenLossBinding:
    name: Literal["base_ce", "token_type_gate", "coord_gaussian_rps"]
    role: LossRole
    normalizer: LossNormalizer
    zero_weight_policy: ZeroWeightPolicy
    token_types: tuple[str, ...] | None
```

Define the exact tuple asserted by Step 1. Do not add callable fields, discovery, registration methods, entry points, lifecycle hooks, or public config identifiers. Import the metadata directly inside `runner.py`; omit it from `src.losses.__all__`.

- [ ] **Step 3: Write RED execution-path tests for both zero policies**

Instrument term construction and calls:

```python
def test_zero_coordinate_auxiliary_is_wholly_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed = 0

    def forbidden_constructor(*args: object, **kwargs: object) -> object:
        nonlocal constructed
        constructed += 1
        raise AssertionError("zero auxiliary must not construct its term")

    monkeypatch.setattr(runner_module, "CoordGaussianRPSLoss", forbidden_constructor)
    runner = LossRunner.from_config(loss_config(coord_weight=0.0))
    plan = runner.prepare_planned_step(sample_micro_steps())
    bundle = runner.compute_micro_step(sample_context(), plan, local_micro_step_index=0)
    assert constructed == 0
    assert "coord_gaussian_rps" not in plan.denominators
    assert all(term.name != "coord_gaussian_rps" for term in bundle.terms)
    assert not any("coord_gaussian_rps" in key for key in bundle.metrics)


def test_gate_ablation_is_detached_but_retains_diagnostics() -> None:
    runner = LossRunner.from_config(loss_config(gate_mode="zero_weight_ablation"))
    plan, bundle = execute_one_micro_step(runner)
    gate = bundle.term_by_name("token_type_gate")
    assert gate.raw_loss.grad_fn is None
    assert gate.raw_loss.requires_grad is False
    assert gate.weighted_loss.item() == 0.0
    assert gate.backward_contribution is None
    assert "token_type_gate" in plan.denominators
```

Also use `torch.autograd.graph.saved_tensors_hooks` to count saved tensors: the omitted auxiliary adds zero saves/calls/state relative to the same runner without an authored auxiliary; label this as correctness evidence, not a speed or peak-memory result.

Run:

```bash
conda run -n ms pytest tests/losses/test_zero_policy_contract.py tests/losses/test_runner.py -q
```

Expected: FAIL on the old protected-coordinate access and missing explicit `backward_contribution` separation.

- [ ] **Step 4: Refactor one deep runner around the zero policies**

Keep `LossRunner.from_config` explicit:

```python
coord_cfg = config.auxiliary.coord_gaussian_rps
coord_term = (
    CoordGaussianRPSLoss(
        gaussian_weight=coord_cfg.gaussian_weight,
        rps_weight=coord_cfg.rps_weight,
        temperature=coord_cfg.temperature,
        gaussian_r95_axis_fraction=coord_cfg.gaussian_r95_axis_fraction,
        gaussian_r95_cap_bins=coord_cfg.gaussian_r95_cap_bins,
        gaussian_r95_min_bins=coord_cfg.gaussian_r95_min_bins,
        gaussian_r95_fallback_bins=coord_cfg.gaussian_r95_fallback_bins,
    )
    if coord_cfg.weight > 0.0
    else None
)
return cls(
    base_ce_weight=config.protected.base_ce.weight,
    token_type_gate_mode=config.protected.token_type_gate.mode,
    token_type_gate_weight=config.protected.token_type_gate.weight,
    token_type_gate_groups=tuple(config.protected.token_type_gate.groups),
    coord_gaussian_rps_weight=coord_cfg.weight,
    coord_gaussian_rps=coord_term,
)
```

Extend `LossTermResult` with:

```python
backward_contribution: torch.Tensor | None
```

For positive-weight differentiable terms, set `raw_loss` to the local semantic contribution without backend compensation, `weighted_loss = raw_loss * weight`, and `backward_contribution = weighted_loss * plan.backend_gradient_scale`. For gate ablation, execute per-atom math under `torch.no_grad()`, retain finite raw/count diagnostics, set weighted semantic value to exact zero, set `backward_contribution=None`, and exclude it from `LossBundle.total_loss`. For zero coordinate auxiliary, create no term, denominator, call, bundle entry, finite entry, metric, or persistent state.

- [ ] **Step 5: Run GREEN, residue checks, and commit**

```bash
conda run -n ms pytest tests/losses/test_bindings.py tests/losses/test_context_and_terms.py tests/losses/test_coord_gaussian_rps.py tests/losses/test_normalizers.py tests/losses/test_runner.py tests/losses/test_zero_policy_contract.py tests/training/test_supervised_trainer.py -q
conda run -n ms pytest tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py -q
rg -n 'config\.protected\.coord_gaussian_rps|loss_registry|register_loss|importlib|entry_points|loss_factory' src/losses src/training/session.py && exit 1 || true
openspec validate standardize-coordexp-swift-supervised-losses --strict
git diff --check
git add src/losses/bindings.py src/losses/runner.py src/losses/__init__.py tests/losses/test_bindings.py tests/losses/test_zero_policy_contract.py tests/losses/test_runner.py tests/training/test_supervised_trainer.py
git diff --cached --check
git diff --cached
git commit -m "feat(losses): enforce closed supervised zero policies"
```

Expected: all tests pass; no dynamic extension residue exists; cache guards pass; the commit is independently revertible to Task 2.

### Task 4: Separate semantic values from DDP backward compensation

**OpenSpec owner:** Wave 3; Planned-step normalizer requirements; Design decision 4.

**Files:**
- Modify: `src/losses/runner.py`
- Modify: `src/losses/normalizers.py`
- Modify: `src/runtime/finite_gates.py`
- Modify: `src/runtime/train_runtime.py`
- Create: `tests/losses/test_ddp_objective_parity.py`
- Modify: `tests/losses/test_normalizers.py`
- Modify: `tests/losses/test_runner.py`
- Modify: `tests/runtime/test_finite_gates.py`
- Modify: `tests/runtime/test_train_runtime.py`

**Interfaces:**
- Consumes: Task-3 `LossTermResult.backward_contribution`, `PlannedStepLossPlan.backend_gradient_scale`, existing exact integer `accuracy_stats`, existing metric collective and all-rank scalar finite gate.
- Produces: globally normalized raw/weighted semantic values; differentiable local contribution compensated exactly once; pooled `acc_top1`/`acc_top5`; all-rank skip decision for non-finite detached gate diagnostics.

- [ ] **Step 1: Write RED single-rank arithmetic and finite-gate tests**

For each active term assert:

```python
assert term.weighted_loss == pytest.approx(term.raw_loss * term.weight)
assert term.backward_contribution == pytest.approx(
    term.weighted_loss * plan.backend_gradient_scale
)
assert finalized["metrics"][f"loss/{term.name}/raw"] == pytest.approx(global_raw)
assert finalized["metrics"][f"loss/{term.name}/weighted"] == pytest.approx(
    global_raw * term.weight
)
```

For a zero-weight non-finite gate diagnostic, create a `LossBundle` whose gate `raw_loss` is NaN, weighted value is zero, and `backward_contribution` is `None`. Assert `RankScalarFiniteReport.from_loss_bundle` records the raw gate as non-finite and the all-rank decision skips before `backward` and optimizer mutation.

Run:

```bash
conda run -n ms pytest tests/losses/test_runner.py tests/runtime/test_finite_gates.py -q
```

Expected: FAIL until semantic and backward fields are reduced separately and finite reporting examines protected raw diagnostics.

- [ ] **Step 2: Implement semantic finalization and exact sufficient-statistic reduction**

In `LossRunner.finalize_planned_step`, emit these exact fields for every computed term:

```python
metrics[f"loss/{name}/raw"] = raw_semantic_value
metrics[f"loss/{name}/weighted"] = raw_semantic_value * configured_weight
metrics[f"loss/{name}/selected_count"] = float(selected_count)
metrics[f"loss/{name}/eligible_segment_count"] = float(eligible_segment_count)
metrics[f"loss/{name}/finite"] = 1.0 if math.isfinite(raw_semantic_value) else 0.0
```

Set `loss/total` to the sum of weighted objective terms only; the zero gate contributes zero and stays present, while omitted coordinate auxiliary has no fields. Preserve integer `top1_correct`, `top5_correct`, and `atom_count` until after all-rank sum; form ratios once. Do not average ratios or multiply semantic fields by `world_size`.

- [ ] **Step 3: Add the unequal-rank DDP parity fixture**

Create one deterministic CPU/Gloo test whose rank 0 contributes one eligible segment and rank 1 contributes three. Compare its all-rank result against a world-size-one concatenated reference for:

- base CE, enabled gate, and positive coordinate auxiliary raw/weighted values;
- parameter gradients before optimizer step;
- parameters after one SGD update with fixed learning rate;
- pooled top-1/top-5 from summed integer counts;
- identical collective call order on both ranks.

Use tolerances `rtol=1e-5, atol=1e-6` for fp32 scalars, gradients, and updated parameters. The test node is exactly:

```bash
conda run -n ms pytest tests/losses/test_ddp_objective_parity.py::test_unequal_rank_objective_gradient_and_update_match_world_size_one -q
```

Expected before GREEN: FAIL on raw/backward conflation or rank-local ratio reduction.

- [ ] **Step 4: Prepare the fresh DDP authorization packet and stop**

The packet binds the exact Task-3 commit and Step-3 command. It fixes `world_size=2`, CPU/Gloo only, two worker processes, one planned optimizer step, one model-free tiny linear forward per rank, zero GPUs, zero cache/materialization passes, a 120-second wall timeout, 2 GiB RSS per process, 16 MiB artifact ceiling under a new pytest temp root, and the exact expected test node. Stop for fresh user authorization. Commit/command drift, extra forwards/collectives, timeout, RSS/artifact exceedance, cache access, or a third process invalidates the packet and stops without retry.

- [ ] **Step 5: Run the authorized DDP node, then the full CPU GREEN gate**

Run the exact authorized command once, then:

```bash
conda run -n ms pytest tests/losses/test_normalizers.py tests/losses/test_runner.py tests/losses/test_ddp_objective_parity.py tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py tests/training/test_supervised_trainer.py tests/eval/test_forward_eval.py -q
conda run -n ms pytest tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
```

Expected: authorized parity passes within declared tolerances; finite gate skips all ranks before backward for a non-finite detached diagnostic; no cache identity changes.

- [ ] **Step 6: Commit the objective/DDP wave**

```bash
git diff --check
git add src/losses/runner.py src/losses/normalizers.py src/runtime/finite_gates.py src/runtime/train_runtime.py tests/losses/test_ddp_objective_parity.py tests/losses/test_normalizers.py tests/losses/test_runner.py tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py
git diff --cached --check
git diff --cached
git commit -m "fix(losses): preserve global SFT objective under DDP"
```

Expected: only objective/reduction/finite-gate owners and their tests are committed; the commit reverts independently to Task 3.

### Task 5: Route canonical loss telemetry through final reporting and session owners

**OpenSpec owner:** Wave 4; Training-artifacts delta; Design decision 5.

**Files:**
- Modify: `src/losses/runner.py`
- Modify: `src/training/session.py`
- Modify: `src/training/reporting.py`
- Modify: `src/eval/forward.py`
- Modify: `src/artifacts/run_schema.py`
- Modify only if strict row validation requires it: `src/artifacts/run_writer.py`
- Modify: `tests/losses/test_runner.py`
- Modify: `tests/training/test_training_session.py`
- Modify: `tests/training/test_reporting.py`
- Modify: `tests/eval/test_forward_eval.py`
- Modify: `tests/artifacts/test_run_schema.py`
- Modify: `tests/artifacts/test_run_artifacts.py`

**Interfaces:**
- Consumes: `LossRunner.finalize_planned_step(micro_loss_artifacts: Sequence[dict[str, Any]], plan: PlannedStepLossPlan) -> dict[str, Any]`; post-decomposition `TrainingSession`; `CompletedStepReporter.__call__(observation: CompletedStepObservation) -> None`; existing rank-zero append handshake.
- Produces: one canonical train/eval scalar mapping with `loss/<term>/raw`, `loss/<term>/weighted`, `loss/total`, term counts/finite fields, gate-ablation retention, optional-family omission, and non-finite JSON normalization.

- [ ] **Step 1: Write RED exact-row tests before changing projection**

Use exact expected enabled-baseline keys:

```python
EXPECTED_BASELINE_LOSS_KEYS = {
    "loss/base_ce/raw",
    "loss/base_ce/weighted",
    "loss/base_ce/selected_count",
    "loss/base_ce/eligible_segment_count",
    "loss/base_ce/finite",
    "loss/token_type_gate/raw",
    "loss/token_type_gate/weighted",
    "loss/token_type_gate/selected_count",
    "loss/token_type_gate/eligible_segment_count",
    "loss/token_type_gate/finite",
    "loss/total",
}


def test_train_row_has_only_explicit_loss_names() -> None:
    row = completed_train_row(enabled_loss_result())
    assert EXPECTED_BASELINE_LOSS_KEYS <= set(row)
    assert "loss/base_ce" not in row
    assert "loss/token_type_gate" not in row


def test_zero_auxiliary_omits_its_complete_field_family() -> None:
    row = completed_train_row(zero_auxiliary_loss_result())
    assert not any(key.startswith("loss/coord_gaussian_rps/") for key in row)


def test_gate_ablation_retains_detached_diagnostic_fields() -> None:
    row = completed_eval_row(gate_ablation_loss_result())
    assert row["loss/token_type_gate/weighted"] == 0.0
    assert row["loss/token_type_gate/selected_count"] > 0
    assert row["loss/token_type_gate/finite"] == 1.0
```

Add unsafe-row assertions: raw NaN becomes JSON `null`, its exact key appears once in `non_finite_fields`, update status is skipped, and no raw NaN/Inf reaches the writer.

Run:

```bash
conda run -n ms pytest tests/losses/test_runner.py tests/training/test_reporting.py tests/eval/test_forward_eval.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_artifacts.py -q
```

Expected: FAIL on ambiguous aliases, old finite names, or incomplete optional-field omission.

- [ ] **Step 2: Keep loss meaning in runner and projection mechanics in reporter**

`LossRunner.finalize_planned_step` constructs the complete semantic mapping. `CompletedStepReporter` forwards that mapping through the existing metric collective and rank-zero append handshake; it must not derive weights, infer term inventories, average rank ratios, or recreate counts. `TrainingSession` must instantiate exactly one `LossRunner.from_config(plan.resolved_config.losses)` and pass that same runner to supervised trainer and forward eval.

Use these wiring assertions:

```python
def test_session_shares_one_loss_runner_with_train_and_eval() -> None:
    session = build_test_session()
    assert session.trainer.loss_runner is session.loss_runner
    assert session.forward_evaluator.loss_runner is session.loss_runner


def test_reporter_preserves_finalized_loss_mapping_exactly() -> None:
    finalized = canonical_finalized_loss_mapping()
    row = report_one_completed_step(finalized)
    assert {key: row[key] for key in finalized["metrics"]} == finalized["metrics"]
```

Do not add a generic event bus, metric registry, alternate sink, or second collective. Observability cadence/TensorBoard/ETA belongs to the later observability change.

- [ ] **Step 3: Normalize non-finite fields and remove current aliases/consumers**

Keep strict JSON normalization in `run_schema.py`; the writer remains a single rank-zero filesystem facade. Search and migrate current code/tests/scripts that consume `loss/base_ce`, `loss/token_type_gate`, or `loss/coord_gaussian_rps`. Historical directories and completed artifacts remain unchanged.

Run:

```bash
rg -n 'loss/(base_ce|token_type_gate|coord_gaussian_rps)(["'"'"']|$)' src tests scripts configs/coordexp_swift docs/COORDEXP_SWIFT.md docs/ARTIFACTS.md
```

Expected before cleanup: every match is classified. After cleanup, current-runtime matches are only explicit `/raw`, `/weighted`, `/selected_count`, `/eligible_segment_count`, or `/finite` fields; historical evidence is excluded from edits.

- [ ] **Step 4: Run GREEN, cache guard, and commit final-owner telemetry**

```bash
conda run -n ms pytest tests/losses/test_runner.py tests/training/test_training_session.py tests/training/test_reporting.py tests/training/test_supervised_trainer.py tests/eval/test_forward_eval.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_artifacts.py tests/runtime/test_train_runtime.py -q
conda run -n ms pytest tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
git diff --check
git add src/losses/runner.py src/training/session.py src/training/reporting.py src/eval/forward.py src/artifacts/run_schema.py src/artifacts/run_writer.py tests/losses/test_runner.py tests/training/test_training_session.py tests/training/test_reporting.py tests/eval/test_forward_eval.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_artifacts.py
git diff --cached --check
git diff --cached
git commit -m "feat(losses): emit canonical SFT loss telemetry"
```

Expected: train/eval rows share the exact schema, ambiguous aliases are absent, optional omission is complete, the cache guard passes, and the commit reverts independently to Task 4.

### Task 6: Prove final cache equality, document SFT scope, and run bounded acceptance

**OpenSpec owner:** Wave 5; Migration steps 5-7; all three delta specs.

**Files:**
- Modify: `docs/COORDEXP_SWIFT.md`
- Modify: `docs/SYSTEM_OVERVIEW.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/ARTIFACTS.md`
- Create after authorized execution: `openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/final-ddp-smoke.json`
- Update only from executed evidence: `openspec/changes/standardize-coordexp-swift-supervised-losses/tasks.md`

**Interfaces:**
- Consumes: Task-1 determinant payload/hash baseline, Task-2 config inventory, final `TrainingSession`/`CompletedStepReporter`, predecessor immutable train/eval cache targets.
- Produces: exact post-change determinant/hash equality receipt, bounded SFT-only distributed smoke receipt, canonical docs, final standards/intent/overdesign verdicts.

- [ ] **Step 1: Run the complete CPU acceptance matrix**

```bash
conda run -n ms pytest tests/config/test_train_config.py tests/config/test_supervised_loss_config_inventory.py tests/losses tests/runtime/test_finite_gates.py tests/runtime/test_train_runtime.py tests/training/test_supervised_trainer.py tests/training/test_training_session.py tests/training/test_reporting.py tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py tests/eval/test_forward_eval.py tests/artifacts/test_run_schema.py tests/artifacts/test_run_artifacts.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
```

Expected: exit 0 with exact pass/fail/skip counts recorded. Investigate every unexpected skip; do not convert a skip into executed DDP or artifact evidence.

- [ ] **Step 2: Recompute and compare production train/eval determinant identities read-only**

Use the final read-only cache-admission path from `src/training/cache_workflow.py` against the predecessor cache targets; do not call `src.prepare_train_cache`. Compare canonical determinant payload bytes, payload SHA256, aggregate fingerprint, target path, and hit status with `wave-0-loss-baseline.json`.

Expected exact assertions:

```python
assert final_train.determinant_payload == baseline_train.determinant_payload
assert final_train.aggregate_fingerprint == baseline_train.aggregate_fingerprint
assert final_train.target == baseline_train.target
assert final_train.cache_status == "hit"
assert final_eval.determinant_payload == baseline_eval.determinant_payload
assert final_eval.aggregate_fingerprint == baseline_eval.aggregate_fingerprint
assert final_eval.target == baseline_eval.target
assert final_eval.cache_status == "hit"
```

Any inequality is blocking: stop for contract review, perform zero materialization passes, and do not repair, overwrite, delete, or publish a target.

- [ ] **Step 3: Update canonical docs without duplicating the contract**

Document the current operator-facing routes and link to the synchronized stable specs. State the canonical enabled SFT baseline, named gate ablation, typed coordinate auxiliary, distinct zero policies, raw/weighted fields, and SFT-only boundary. Keep formulas and exhaustive scenarios in OpenSpec rather than copying them into docs. Do not edit `docs/catalog.yaml`, `docs/data/PACKING.md`, `docs/history/README.md`, historical configs, or completed artifacts.

- [ ] **Step 4: Prepare the fresh production-shaped DDP/GPU packet and stop**

Bind the packet to the exact Task-5 commit, predecessor cache fingerprints, config `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`, a new absent artifact root, and the exact repository launch command established by the decompose vertical-smoke receipt. Fix `world_size=2`, at most two GPUs, one planned and at most one applied optimizer step, the config-derived maximum micro-step/model-forward count, the frozen collective-count ceiling, zero cache/materialization passes, a 30-minute wall timeout, 24 GiB RSS per rank, a 24 GiB GPU-memory high-water ceiling per rank, and an 8 GiB artifact ceiling. Require train row, eval row, checkpoint, finalization, and downstream reader evidence. Stop for fresh user authorization; if selected devices do not have 24 GiB plus their existing safety headroom free, the packet is invalid rather than enlarged.

Command/commit/config/cache/output drift, occupied artifact root, any cache miss/build attempt, non-finite protected diagnostic, collective divergence/hang, OOM, timeout, or resource-bound exceedance stops without retry and invalidates the packet.

- [ ] **Step 5: Run the exact authorized smoke once and inspect durable evidence**

Expected durable assertions:

- the resolved config records base CE `1.0`, gate `zero_weight_ablation`, gate `0.0`, and the exact four groups;
- both ranks admit the predecessor cache and perform zero preparation passes;
- one train row has base raw/weighted, gate raw/weighted-zero/count/finite, no coordinate auxiliary family, pooled accuracy, and explicit finite/update status;
- the gate raw diagnostic is finite and detached; the applied update equals the base-only objective route within declared tolerance;
- eval uses the same term-field rules;
- checkpoint/finalization/readers complete without a new artifact schema family;
- the receipt reports only integration evidence, not a throughput or memory improvement.

- [ ] **Step 6: Run final residue/authority audits**

```bash
rg -n 'protected:[\s\S]*coord_gaussian_rps|token_type_gate:[\s\S]*weight: (0\.2|0\.25)' configs/coordexp_swift/prod configs/coordexp_swift/smoke
rg -n 'loss/(base_ce|token_type_gate|coord_gaussian_rps)(["'"'"']|$)' src tests scripts docs/COORDEXP_SWIFT.md docs/ARTIFACTS.md
rg -n 'rollout_loss|policy_loss|value_loss|kl_loss|reward_loss|hidden_state_loss|loss_registry|register_loss|entry_points|importlib' src/config src/losses src/training/session.py
conda run -n ms pytest tests/training/test_loss_cache_invariance.py tests/training/test_pack_cache_determinant_registry.py -q
openspec validate standardize-coordexp-swift-supervised-losses --strict
git diff --check
```

Expected: no live noncanonical config, ambiguous alias, dynamic extension, or RL-composition residue; cache guards and strict validation pass. Obtain independent standards, user-intent/contract, and overdesign verdicts against the exact final commit and receipts; resolve every P0/P1.

- [ ] **Step 7: Commit documentation and evidence-backed closeout**

```bash
git add docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/final-ddp-smoke.json openspec/changes/standardize-coordexp-swift-supervised-losses/tasks.md
git diff --cached --check
git diff --cached
git commit -m "docs(losses): close supervised loss standardization"
```

Expected: only canonical docs, immutable bounded receipt, and evidence-backed task checkboxes are committed. Do not sync/archive until `openspec-verify-change` independently confirms implementation, delta, receipt, and task coherence.

## Plan Self-Review

- [x] OpenSpec Waves 0-5 map to Tasks 1-6 without changing scope or completion authority.
- [x] Reconcile and decompose are exact committed prerequisites; final `session.py` and `reporting.py` owners are mandatory.
- [x] Base/gate constants, canonical groups, coordinate auxiliary placement, and distinct zero policies have RED and GREEN coverage.
- [x] Semantic raw/weighted values, backend contribution, finite gating, unequal-rank gradients, and one optimizer update have explicit acceptance evidence.
- [x] Train/eval row schemas, alias removal, optional omission, gate-ablation retention, and JSON-null normalization have exact tests.
- [x] Every wave runs the cache-owner/hash guard; this plan authorizes zero cache materialization passes.
- [x] Both the CPU/Gloo parity probe and production-shaped two-rank GPU smoke require fresh quantitative authorization packets.
- [x] RL and public plugin surfaces remain deferred; no event bus, registry discovery, alternate reporter/session, or performance claim is introduced.
- [x] Every future commit stages only task-owned paths and is independently reviewable/revertible.
