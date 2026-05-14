# Compact-Full Prefix Roll-in Multi-Positive Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict `compact_full`-only `prefix_rollin_et_rmp_ce` objective that unifies ground-truth prefix roll-in, balance-first entry-trie multi-positive CE, compact token-type gating, and EOS-weighted `<|im_end|>` supervision.

**Architecture:** Keep the owner surface under latest `src/detection/*`. Add typed roll-in and eos-policy containers, build prefix/suffix examples before Swift alignment, construct sparse local next-token targets from remaining objects, and compute support/balance plus type-gate losses from aligned sidecars. EOS supervision uses `<|im_end|>` from the closed Qwen chat template. E1 uses `empirical_unlabeled_poisson_v0` as an ablation-only prior; production uses `calibrated_formula_ref` only. Reject legacy configs and non-compact templates rather than preserving compatibility.

**Tech Stack:** Python, PyTorch, HuggingFace/Qwen tokenizer stack, ms-swift training integration, YAML latest-schema configs, pytest under `conda run -n ms`.

---

Date: 2026-05-06

Spec: `docs/superpowers/specs/2026-05-06-compact-full-prefix-rollin-multipositive-unification-design.md`

Status: implementation authorized by the user on 2026-05-07 and in progress.

## 2026-05-07 Audit Hardening Scope

The following post-audit items are part of this implementation pass:

- [x] Make `prefix_rollin_et_rmp_ce` config-truthful: require
  `objective.state_weighting: uniform_permutation` and
  `objective.normalization: semantic_image_bucket_balanced`.
- [x] Treat `training.effective_batch_size` as the source of truth and reject
  authored `training.gradient_accumulation_steps` whenever effective batch is
  present.
- [x] Add strict recursive CE sidecar/collated-batch validation before model
  forward.
- [x] Remove `labels` from the recursive CE model forward path while keeping
  labels available for local metrics.
- [x] Keep EOS trust on the main CE term only; do not scale type-gate loss with
  EOS trust.
- [x] Set global HF/Qwen generation ids to
  `eos_token_id=<|im_end|>` and `pad_token_id=<|endoftext|>`.
- [x] Set HF processor inference calls to `do_resize=false`.
- [x] Align vLLM inference with the same decode contract: stop on
  `<|im_end|>` only and pass local multimodal processor `do_resize=false`.
- [x] Add image-root and coord-token `xyxy` safety guards.
- [x] Record latest-detection objective identity, effective batch source,
  actual global effective batch, model path identity, Qwen generation contract,
  and compact-full token-row count in run artifacts.
- [x] Add target-mix diagnostics for tiny/smoke trend interpretation:
  `target_mix/eos_fraction`, `target_mix/non_eos_fraction`,
  `target_mix/trie_multi_positive_fraction`, role fractions, and
  `target_mix/positive_children_per_trie_target`.
- [x] Add entry/type-gate probability diagnostics:
  `entry/continue_minus_eos_margin`, `entry/valid_child_entropy`,
  `entry/valid_child_kl_to_uniform`, `type_gate_allowed_mass`, plus the
  separator append-boundary metrics
  `free_boundary/continue_minus_eos_margin` and `free_boundary/continue_mass`.
- [x] Add config-truthful append-boundary ablation weights under
  `objective.boundary` and introduce E2
  `compact_full_prefix_rollin_separator2.yaml`, which raises only
  `separator_continue_weight` to target the generated-prefix `\n` vs
  `<|im_end|>` failure.
- [x] Add the forced-prefix continue-vs-EOS probe surface:
  `src.analysis.prefix_rollin_teacher_forced_diagnostic` writes
  `forced_prefix_continue_vs_eos_v0` rows with `prefix_k`, `prefix_mode`,
  `gt_count`, `remaining_gt_count`, `continue_logsumexp`, `valid_mass`, and
  `continue_minus_eos_margin`; `--k-values every` scans the full `K=0..N`
  forced-prefix curve without changing autoregressive decoding. The probe now
  separates `*_free_boundary` (`\n` or first object token vs `<|im_end|>`) from
  `*_entry_after_separator` (`<|object_ref_start|>` vs `<|im_end|>` after a
  forced newline), and can replay free-decode artifacts via
  `--prefix-modes generated_prefix --decode-artifact ... --trace-artifact ...`.
- [ ] Add the future content-level EOS calibration artifact validator once the
  calibrated formula file schema is finalized by the user.

## Hard Guardrails

| Guardrail | Requirement |
|---|---|
| Template scope | Reject every template except `compact_full`. |
| Decode path | Preserve ordinary autoregressive generation as the main path. |
| Model architecture | Do not edit upstream Qwen/HF model files or add a detection head. |
| Config style | Put behavior under typed latest-schema sections; do not add `custom.*` knobs. |
| Legacy | Do not preserve old Stage-1 set-continuation compatibility in the new surface. |
| EOS | Treat `<|im_end|>` as the only EOS token in this variant. |
| Tokenizer stop contract | Resolve `<|im_end|>` from the active Qwen/Qwen3-VL tokenizer and never construct training targets with `<|endoftext|>` or `<|end_of_text|>`. |
| EOS formula | Implement `empirical_unlabeled_poisson_v0` as the default E1 ablation prior with log-linear missing-count penalty, while keeping production calibrated formulas artifact-backed. |
| HF generation stop/pad | All HF Qwen generation surfaces use `eos_token_id=id("<|im_end|>")` and `pad_token_id=id("<|endoftext|>")`; training EOS targets remain `<|im_end|>` only. |
| vLLM generation stop | vLLM local/server inference stops on `"<|im_end|>"` only; `<|endoftext|>` remains pad/text metadata, not a stop target. |
| Batch source of truth | `training.effective_batch_size` owns optimizer-step budget; `gradient_accumulation_steps` is derived and must not be YAML-authored when effective batch is set. |
| Packing/cache | Keep latest recursive sidecar training packing/cache disabled unless a separate offset-rewrite design exists. |
| Python navigation | Use Serena MCP for Python symbol exploration and edits after narrowing files with `rg` or `rtk`. |
| Tests | Prefer `rtk conda run -n ms python -m pytest ...` for targeted test commands. |

## Planned File Map

| Path | Role |
|---|---|
| `src/config/schema.py` | Add strict objectized config schema for `prefix_rollin_et_rmp_ce`, roll-in, target, type gate, and eos prior. |
| `src/detection/rollin.py` | New typed roll-in sampler and `RollinState` owner. |
| `src/detection/objective.py` | Extend recursive target construction to accept initial emitted/remaining state and suffix-only supervised spans. |
| `src/detection/dataset.py` | Render compact prefix+suffix examples, mask prefix labels, align sidecars to actual encoded positions. |
| `src/detection/template.py` | Use `eos` / `semantic_eos` vocabulary for the new variant. |
| `src/detection/tokenization.py` | Project Qwen chat-template `<|im_end|>` stop marker into encoded `semantic_eos` spans without tokenizer-level `<|endoftext|>` fallback. |
| `src/detection/loss.py` | Centralize support/balance, sparse soft CE identity, type gate, and eos-weight handling. |
| `src/detection/token_types.py` | Compact token-type grouping helper. |
| `src/detection/tokenizer_contract.py` | Focused compact tokenizer stop-contract helper. |
| `src/detection/runtime.py` | Canonical latest-detection runtime owner for the new variant, compact-only checks, padding/packing/cache fail-fast, and recursive CE runtime config. |
| `src/sft.py` | Delegate to latest detection runtime; avoid duplicating objective routing policy here. |
| `src/training_runtime/plan.py` | Remove/deactivate old `custom.trainer_variant=stage1_set_continuation` exposure. |
| `src/bootstrap/trainer_setup.py` | Review/update recursive CE mixin gating or incompatible-mixin composition. |
| `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml` | New ablation config for the first runnable surface. |
| `tests/test_prefix_rollin_schema.py` | Schema and reject-legacy tests. |
| `tests/test_prefix_rollin_sampler.py` | `K` distribution and emitted/remaining tests. |
| `tests/test_prefix_rollin_dataset_alignment.py` | Prefix masking and chat-token alignment tests. |
| `tests/test_entry_trie_targets.py` | Multi-positive target tests. |
| `tests/test_recursive_loss_support_balance.py` | Sparse loss identity and balance-first tests. |
| `tests/test_compact_type_gate.py` | Token-type allowed-mass tests. |
| `tests/test_compact_tokenizer_stop_contract.py` | `<|im_end|>` tokenizer/template stop-token contract tests. |
| `tests/test_eos_prior.py` | `<|im_end|>` supervision and production formula guard tests. |
| `tests/test_prefix_rollin_diagnostics.py` | Metric denominator, DDP reducer, and scalar payload tests. |
| `tests/test_prefix_rollin_artifact_contracts.py` | Synthetic research artifact schema tests. |
| `tests/test_prefix_rollin_config_materialization.py` | No-training materialization of one encoded prefix-rollin batch. |
| `tests/test_prefix_rollin_runtime.py` | Runtime owner, tokenizer padding, packing/cache, and SFT delegation tests. |
| `docs/training/STAGE1_OBJECTIVE.md` | Update current objective description after implementation passes tests. |
| `docs/training/README.md` | Route the new compact-only ablation surface after implementation. |
| `docs/ARTIFACTS.md` | Register prefix-rollin research artifacts before paper-scale claims. |

## Binding Contract Decisions

These decisions are binding for implementation:

| Area | Decision |
|---|---|
| Chat stop | Assistant payload excludes `<|im_end|>`; `apply_chat_template(add_generation_prompt=False)` supplies the single assistant stop marker. |
| Roll-in sampling | Sample one `RollinState` per dataset item and pass that exact state through render, encode, target construction, loss, and diagnostics. |
| Instance identity | Roll-in state stores `object_instance_id`; duplicate serialized compact rows are valid when instance IDs differ. |
| Suffix order | v1 uses one sampled full permutation: `prefix = pi[:K]`, `teacher_suffix = pi[K:]`. Future independent suffix-order sampling is a separate ablation. |
| EOS scope | Weighted `<|im_end|>` supervision applies whenever remaining labeled objects become empty, including `K=N` and `K<N` after suffix completion. |
| EOS zero weight | Represent the EOS target even when `eos_trust_weight=0`; it contributes zero loss but keeps diagnostics and denominators defined. |
| Label rewrite | `DetectionTrainingDataset` owns encoded-label rewrite after Swift/chat encoding; alignment checks run after rewrite. |
| Padding | v1 requires right padding or fail-fast; left-padding support requires explicit sidecar position rewrite. |
| Sidecars | Keep new Python metadata under the existing `recursive_detection_targets` top-level sidecar. |
| Schema surface | Add required latest-schema `experiment.surface` with enum values `smoke`, `ablation`, and `production` to validate production vs ablation EOS policy. |
| Legacy route | New surface rejects legacy knobs and must not route through old `stage1_set_continuation` config/runtime compatibility. Old executable source files must not remain for provenance; provenance lives in git history, progress notes, and archived artifacts only. |
| Metrics | Emit scalar finite metrics plus explicit denominator/count keys; DDP aggregation must use numerator/denominator semantics, not rank-local means for global claims. |

## Execution Order Policy

Implementation starts with cleanup, not additive wiring. The first implementation phase removes active legacy set-continuation exposure before adding the new `prefix_rollin_et_rmp_ce` schema/config/runtime. This avoids a mixed period where old and new names both appear runnable and future workers accidentally build compatibility shims.

Cleanup means deleting active config files, active routing, config discovery, docs recommendations, tests that assert legacy executability, and runtime materialization for the old set-continuation surface. Historical evidence remains in git history, archived progress notes, and benchmark artifacts; it should not remain as a current trainable path or as a renamed historical config directory.

## Test Harness Contract

Current repo tests use a flat `tests/test_*.py` layout. Keep this plan on that layout unless a separate test-organization change is explicitly approved.

Every failing-test step below must either import these helpers from the declared local test helper owner or define them in the same test file before first use:

```python
def compact_obj(instance_id: str, *, desc: str, bbox: tuple[int, int, int, int]) -> SyntheticCompactObject: ...
def fake_qwen_tokenizer() -> FakeQwenTokenizer: ...
def build_compact_prefix_rollin_example(*, objects, rollin_order, k, eos_trust_weight=1.0) -> PreparedPrefixRollinExample: ...
def load_latest_detection_config(payload: Mapping[str, object]) -> LatestDetectionTrainingConfig: ...
def build_targets(*, objects, emitted, teacher_suffix, eos_trust_weight=1.0) -> RecursiveDetectionTargets: ...
def first_multi_positive_target(targets) -> TokenTarget: ...
def final_eos_target(targets) -> TokenTarget: ...
```

Expected failure before implementation: production imports such as `sample_prefix_rollin_state`, `resolve_compact_training_stop_contract`, and `build_compact_prefix_rollin_example` are missing. Fixture helpers themselves must be executable before implementation starts, so first failures are behavior/API failures rather than `NameError` noise. Unless the implementation introduces and wires a repo-wide custom config exception, schema examples should assert the current latest-schema behavior with `ValueError`.

## Task 0: Preflight And Current Surface Snapshot

**Files:**

- Read: `docs/AGENT_INDEX.md`
- Read: `docs/training/STAGE1_OBJECTIVE.md`
- Read: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`
- Read: `src/detection/objective.py`
- Read: `src/detection/loss.py`
- Read: `src/detection/dataset.py`

- [ ] **Step 1: Check git state without changing files**

Run:

```bash
git status --short
```

Expected: unrelated user changes may exist. Do not revert them.

- [ ] **Step 2: Narrow current latest detection symbols**

Run:

```bash
rg -n "random_permutation_et_rmp_ce|recursive_detection_ce|trie_support_weight|trie_balance_weight|TokenTarget|RecursiveDetectionTargets" src configs tests
```

Expected: hits in latest detection config/schema/objective/loss/dataset tests.

- [ ] **Step 3: Use Serena MCP for Python symbol inspection**

Inspect these symbols before editing:

```text
src/config/schema.py::LatestDetectionTrainingConfig
src/detection/objective.py::build_recursive_detection_targets
src/detection/loss.py::_compute_sample_loss
src/detection/dataset.py::DetectionTrainingDataset
```

Expected: identify exact call sites and keep edits local to owners.

## Task 1: Aggressively Remove Active Legacy Continuation Surface

**Files:**

- Delete: `configs/stage1/set_continuation/`
- Delete: `src/trainers/stage1_set_continuation/`
- Delete: `src/data_collators/stage1_set_continuation_collator.py`
- Modify: `src/config/schema.py`
- Modify: `src/training_runtime/plan.py`
- Modify: `src/training_runtime/profile.py`
- Modify: `src/sft.py`
- Modify: `src/bootstrap/trainer_setup.py`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/training/README.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify or archive: `docs/training/STAGE1_ET_RMP_CE.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/catalog.yaml`
- Modify: `docs/AGENT_INDEX.md`
- Modify: `progress/index.yaml` if a historical pointer is added
- Modify/Delete: `tests/test_stage1_set_continuation_config.py`
- Modify/Delete: `tests/test_stage1_set_continuation_train_forward_config.py`
- Modify/Delete: `tests/test_stage1_set_continuation_metric_keys.py`
- Modify/Delete: other tests that only prove the old continuation path is executable
- Create: `tests/test_legacy_surface_absence.py`

- [ ] **Step 1: Write the active-legacy absence tests**

Create or update `tests/test_prefix_rollin_schema.py` with negative contract tests:

```python
def test_stage1_set_continuation_is_not_active_training_variant():
    with pytest.raises(ValueError, match="stage1_set_continuation"):
        resolve_training_runtime_plan("stage1_set_continuation")
```

```python
def test_latest_schema_rejects_legacy_set_continuation_custom_block():
    payload = {
        "custom": {"trainer_variant": "stage1_set_continuation"},
        "detection_template": {"id": "compact_full"},
    }
    with pytest.raises(ValueError, match="stage1_set_continuation"):
        load_latest_detection_config(payload)
```

```python
def test_active_docs_do_not_recommend_set_continuation_configs(repo_root):
    active_docs = [
        repo_root / "docs" / "training" / "README.md",
        repo_root / "docs" / "training" / "STAGE1_OBJECTIVE.md",
        repo_root / "docs" / "training" / "STAGE1_ET_RMP_CE.md",
        repo_root / "docs" / "training" / "METRICS.md",
        repo_root / "docs" / "IMPLEMENTATION_MAP.md",
        repo_root / "docs" / "catalog.yaml",
        repo_root / "docs" / "AGENT_INDEX.md",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in active_docs)
    forbidden = (
        "configs/stage1/set_continuation",
        "custom.trainer_variant: stage1_set_continuation",
        "stage1_set_continuation_et_rmp_ce",
        "src/trainers/stage1_set_continuation",
        "src/data_collators/stage1_set_continuation_collator.py",
        "branch_support_weight",
        "branch_balance_weight",
        "stage1_set_continuation_metrics_v3",
        "current-continuation",
    )
    for pattern in forbidden:
        assert pattern not in joined
```

- [ ] **Step 2: Run tests and confirm they fail before cleanup**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_schema.py -q
```

Expected: failures prove active legacy exposure still exists.

- [ ] **Step 3: Remove active config files and routing**

Delete `configs/stage1/set_continuation/`, `src/trainers/stage1_set_continuation/`, and `src/data_collators/stage1_set_continuation_collator.py` from the active tree. Do not move the config directory to another runnable or discoverable config directory. If reusable pure logic exists, migrate it under new latest-detection names such as `prefix_rollin_*` or `entry_trie_support_balance_*`; do not keep old `stage1_set_continuation` or `setcont` import paths as provenance.

If a historical pointer must remain, put it only in a progress/benchmark note and do not expose it through `docs/catalog.yaml`, `docs/AGENT_INDEX.md`, `docs/IMPLEMENTATION_MAP.md`, training README tables, runtime plans, or config load tests. The pointer should say `Superseded`, `Historical only`, and `not runnable current config`.

Update runtime routing so the old trainer variant cannot materialize:

```text
resolve_training_runtime_plan("stage1_set_continuation") -> ValueError
custom.trainer_variant=stage1_set_continuation -> strict config error
custom.stage1_set_continuation -> strict config error
Stage1SetContinuationConfig is not reachable from public TrainingConfig/latest-schema loading
_normalize_stage1_set_continuation_payload and old branch_* aliases are deleted or rejection-only
old continuation train-forward custom blocks -> strict config error
```

- [ ] **Step 4: Delete or rewrite legacy-executability tests**

Delete tests whose only purpose is proving the old set-continuation path still parses, routes, or runs. Rewrite only tests that still protect a reusable invariant needed by the new compact latest-detection surface.

Add a test migration matrix before deleting files:

```text
delete executable-only:
  config parsing, trainer smoke, collator executable path, train-forward legacy profile tests
migrate invariant:
  entry-trie multiplicity, unique positive ids, q normalization, hard-label-in-positives
  support/balance identity, zero denominator handling, DDP numerator/denominator aggregation
  packing/cache fail-fast, strict unknown-key behavior, artifact scope labels
replace with absence/rejection:
  old config path, old trainer variant, old custom blocks, old metrics module import
```

After cleanup, `find tests -maxdepth 1 -type f -name '*stage1_set_continuation*' -print` should return nothing except explicit absence/rejection tests if they are intentionally named that way.

- [ ] **Step 5: Run legacy absence and strict-config tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_schema.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_training_runtime_plan.py \
  tests/test_legacy_surface_absence.py \
  -q
```

Expected: legacy absence tests pass, strict unknown-key behavior still passes, and remaining runtime-plan tests no longer expect `stage1_set_continuation` to be valid.

Run the static active-surface absence scan:

```bash
rg -n "stage1_set_continuation|set_continuation|setcont/|branch_support_weight|branch_balance_weight|stage1_set_continuation_metrics_v3|current-continuation" \
  docs configs src tests \
  --glob '!docs/superpowers/**' \
  --glob '!progress/**'
```

Expected: no active config, route, docs recommendation, trainer/collator import, legacy executable test, or current catalog entry. Any remaining hit must be an explicit rejection test or historical-only reference.

- [ ] **Step 6: Commit this cleanup task when implementing**

```bash
git add configs/stage1/set_continuation src/trainers/stage1_set_continuation src/data_collators/stage1_set_continuation_collator.py src/config/schema.py src/training_runtime/plan.py src/training_runtime/profile.py src/sft.py src/bootstrap/trainer_setup.py docs/IMPLEMENTATION_MAP.md docs/training/README.md docs/training/STAGE1_OBJECTIVE.md docs/training/STAGE1_ET_RMP_CE.md docs/training/METRICS.md docs/catalog.yaml docs/AGENT_INDEX.md progress/index.yaml tests/test_prefix_rollin_schema.py tests/test_training_config_strict_unknown_keys.py tests/test_training_runtime_plan.py tests/test_legacy_surface_absence.py
git commit -m "refactor(train): remove legacy set-continuation active surface"
```

## Task 2: Add Strict Objectized Config Schema

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/sft.py`
- Test: `tests/test_prefix_rollin_schema.py`

- [ ] **Step 1: Write failing schema tests**

Create tests covering these authored YAML fragments:

```python
def test_prefix_rollin_schema_accepts_compact_full_empirical_eos_ablation():
    cfg = load_latest_detection_config({
        "detection_template": {"id": "compact_full", "coordinate_surface": "coord_token", "bbox_format": "xyxy"},
        "objective": {
            "id": "recursive_detection_ce",
            "variant": "prefix_rollin_et_rmp_ce",
            "rollin": {
                "enabled": True,
                "source": "ground_truth",
                "prefix_loss": "masked",
                "suffix_order": "same_sampled_permutation",
                "k_distribution": {"type": "uniform_inclusive", "min_k": 0, "max_k": "object_count"},
            },
            "target": {
                "type": "entry_trie_support_balance",
                "trie_scope": "object_entry",
                "q_weighting": "object_multiplicity_uniform",
                "singleton": "hard_ce",
                "control_tokens": "hard_ce",
                "support_weight": 1.0,
                "balance_weight": 2.0,
            },
            "type_gate": {
                "enabled": True,
                "mode": "allowed_type_mass",
                "weights": {"struct": 2.0, "coord": 1.0, "desc": 0.2, "eos": 0.5},
            },
            "eos": {
                "eos_token": "<|im_end|>",
                "policy": "missing_label_prior_weighted_ce",
                "eos_trust_weight": {
                    "source": "empirical_unlabeled_poisson_v0",
                    "expected_unlabeled_count": {"intercept": -0.35, "slope": 0.43, "floor": 0.0},
                    "trust_mapping": {
                        "type": "log_linear_missing_count_penalty",
                        "penalty_per_missing": 1.0,
                        "temperature": 1.0,
                        "min_weight": 0.0,
                        "max_weight": 1.0,
                    },
                },
            },
        },
        "experiment": {"surface": "ablation"},
    })
    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
```

Add rejection tests for:

```text
non-compact_full template
custom.stage1_set_continuation
branch_support_weight
branch_balance_weight
objective.support_weight
objective.balance_weight
objective.trie_support_weight
objective.trie_balance_weight
eos_trust_weight.source=deferred_user_formula with any surface
eos_trust_weight.source=constant_ablation with production surface
eos_trust_weight.source=disabled_ablation with production surface
eos_trust_weight.source=empirical_unlabeled_poisson_v0 with production surface, even if a calibration artifact ref is present
objective.eos.eos_token other than <|im_end|>
missing experiment.surface for prefix_rollin_et_rmp_ce
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_schema.py -q
```

Expected: failures because the schema does not yet know the new variant and EOS policy.

- [ ] **Step 3: Implement typed config containers**

Add focused containers for:

```python
@dataclass(frozen=True)
class PrefixRollinConfig:
    enabled: bool
    source: Literal["ground_truth"]
    prefix_loss: Literal["masked"]
    suffix_order: Literal["same_sampled_permutation"]
    k_distribution: UniformInclusiveKConfig

@dataclass(frozen=True)
class EntryTrieSupportBalanceConfig:
    type: Literal["entry_trie_support_balance"]
    trie_scope: Literal["object_entry"]
    q_weighting: Literal["object_multiplicity_uniform"]
    singleton: Literal["hard_ce"]
    control_tokens: Literal["hard_ce"]
    support_weight: float
    balance_weight: float


@dataclass(frozen=True)
class CompactTypeGateConfig:
    enabled: bool
    mode: Literal["allowed_type_mass"]
    weights: CompactTypeGateWeights

@dataclass(frozen=True)
class EosPriorConfig:
    eos_token: Literal["<|im_end|>"]
    policy: Literal["missing_label_prior_weighted_ce"]
    eos_trust_weight: EosTrustWeightConfig


@dataclass(frozen=True)
class LatestDetectionExperimentConfig:
    surface: Literal["smoke", "ablation", "production"]
    ablation_id: str | None = None
    claim_scope: Literal["none", "smoke", "paper", "production"] | None = None
```

Use the existing schema style in `src/config/schema.py`; if the repo uses Pydantic-style models instead of dataclasses in this area, implement the same fields in that style rather than adding a second schema framework. Also update the latest-detection top-level allowlist and `LatestDetectionTrainingConfig` parsing/output so `cfg.experiment.surface` is available on the latest-schema path; do not rely on the legacy `TrainingConfig.experiment` path unless that path is explicitly made safe for latest detection.

- [ ] **Step 4: Add failfast validation**

Validation rules:

```text
variant == prefix_rollin_et_rmp_ce requires detection_template.id == compact_full
rollin.k_distribution.type must equal uniform_inclusive
rollin.k_distribution.min_k must equal 0
rollin.k_distribution.max_k must equal object_count
rollin.suffix_order must equal same_sampled_permutation
support_weight must be > 0
balance_weight must be > 0
objective.target.support_weight and objective.target.balance_weight are canonical legal paths
flat objective.support_weight/objective.balance_weight are rejected
objective.trie_support_weight/objective.trie_balance_weight are rejected for this variant
branch_support_weight/branch_balance_weight are rejected for this variant
eos_trust_weight.source == deferred_user_formula is rejected as a stale source for every surface
eos_trust_weight.source == constant_ablation is rejected for production
eos_trust_weight.source == disabled_ablation is rejected for production
eos_trust_weight.source == empirical_unlabeled_poisson_v0 is ablation/smoke-only and is always rejected for production
legacy custom keys are rejected for this variant
eos.eos_token must equal <|im_end|>
experiment.surface is required for prefix_rollin_et_rmp_ce and must be one of ablation, smoke, production
production requires eos_trust_weight.source == calibrated_formula_ref plus a versioned calibration artifact reference
```

- [ ] **Step 5: Re-run schema tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_schema.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 6: Commit this task when implementing**

```bash
git add src/config/schema.py src/sft.py tests/test_prefix_rollin_schema.py
git commit -m "feat(config): add compact prefix-rollin objective schema"
```

## Task 3: Implement Uniform Inclusive Roll-in Sampler

**Files:**

- Create: `src/detection/rollin.py`
- Test: `tests/test_prefix_rollin_sampler.py`

- [ ] **Step 1: Write failing sampler tests**

Required test cases:

```python
def test_uniform_inclusive_k_sampler_can_emit_every_depth():
    objects = tuple(ObjectInstanceId(f"obj-{i}") for i in range(4))
    seen = set()
    rng = random.Random(123)
    for _ in range(1000):
        state = sample_prefix_rollin_state(objects, rng=rng)
        seen.add(state.k)
    assert seen == {0, 1, 2, 3, 4}
```

```python
def test_rollin_state_partitions_emitted_and_remaining():
    objects = (ObjectInstanceId("A"), ObjectInstanceId("B"), ObjectInstanceId("C"))
    state = make_prefix_rollin_state(objects, permutation=(ObjectInstanceId("C"), ObjectInstanceId("A"), ObjectInstanceId("B")), k=2)
    assert state.emitted == (ObjectInstanceId("C"), ObjectInstanceId("A"))
    assert state.remaining == (ObjectInstanceId("B"),)
    assert set(state.emitted).isdisjoint(state.remaining)
    assert set(state.emitted) | set(state.remaining) == set(objects)
```

```python
def test_k_equals_n_has_empty_remaining():
    state = make_prefix_rollin_state((ObjectInstanceId("A"), ObjectInstanceId("B")), permutation=(ObjectInstanceId("B"), ObjectInstanceId("A")), k=2)
    assert state.emitted == (ObjectInstanceId("B"), ObjectInstanceId("A"))
    assert state.remaining == ()
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_sampler.py -q
```

Expected: import or function-not-found failures.

- [ ] **Step 3: Implement `RollinState` and sampler**

Implement a frozen typed owner:

```python
ObjectInstanceId = NewType("ObjectInstanceId", str)


@dataclass(frozen=True)
class RollinState:
    k: int
    permutation: tuple[ObjectInstanceId, ...]
    emitted: tuple[ObjectInstanceId, ...]
    remaining: tuple[ObjectInstanceId, ...]
```

Implement:

```python
def sample_prefix_rollin_state(object_instance_ids: Sequence[ObjectInstanceId], *, rng: random.Random) -> RollinState:
    permutation = tuple(object_instance_ids)
    permutation = tuple(rng.sample(permutation, k=len(permutation)))
    k = rng.randint(0, len(permutation))
    return make_prefix_rollin_state(object_instance_ids, permutation=permutation, k=k)
```

Implement validation in `make_prefix_rollin_state`:

```text
permutation must contain each object_instance_id exactly once
k must satisfy 0 <= k <= N
emitted must equal permutation[:k]
remaining must equal permutation[k:]
duplicate serialized compact entries are valid when object_instance_id differs
duplicate object_instance_id values are invalid
```

- [ ] **Step 4: Re-run sampler tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_sampler.py -q
```

Expected: pass.

- [ ] **Step 5: Commit this task when implementing**

```bash
git add src/detection/rollin.py tests/test_prefix_rollin_sampler.py
git commit -m "feat(detection): add compact prefix roll-in sampler"
```

## Task 4: Build Compact Prefix+Suffix Examples With Prefix Masking

**Files:**

- Modify: `src/detection/dataset.py`
- Modify: `src/detection/template.py`
- Create or Modify: `src/detection/tokenizer_contract.py`
- Test: `tests/test_prefix_rollin_dataset_alignment.py`
- Test: `tests/test_compact_tokenizer_stop_contract.py`

- [ ] **Step 1: Write failing prefix-masking tests**

Use two compact objects with deterministic desc and coord tokens. Assert:

```python
def test_rollin_prefix_labels_are_masked_and_suffix_labels_active():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=1)
    prefix_positions = example.debug_spans["rollin_prefix"].token_positions
    suffix_positions = example.debug_spans["supervised_suffix"].token_positions
    assert all(example.labels[pos] == -100 for pos in prefix_positions)
    assert all(example.labels[pos] != -100 for pos in suffix_positions)
```

Also assert the first suffix token is predicted by the logit immediately before it:

```python
def test_first_suffix_token_position_after_prefix_uses_active_label():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=1)
    first_suffix = example.debug_spans["supervised_suffix"].token_positions[0]
    target = example.recursive_targets.token_targets[0]
    assert target.position == first_suffix
    assert first_suffix > 0
    assert example.labels[first_suffix] == target.teacher_token_id
```

Add an explicit causal-shift loss test:

```python
def test_shifted_target_consumes_previous_logit_after_prefix_rewrite():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=1)
    target = example.recursive_targets.token_targets[0]
    logits = logits_that_only_predict_teacher_at(
        position=target.position - 1,
        teacher_id=target.teacher_token_id,
    )
    loss = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=[example.recursive_targets],
    )
    assert loss.loss.item() < 1e-4
```

Add `K=N` dataset/chat-template integration:

```python
def test_k_equals_n_masks_all_objects_and_only_trains_weighted_im_end():
    example = build_compact_prefix_rollin_example(
        objects=[A, B],
        rollin_order=[A, B],
        k=2,
        eos_trust_weight=0.25,
    )
    object_positions = example.debug_spans["rollin_prefix"].token_positions
    assert all(example.labels[pos] == -100 for pos in object_positions)
    assert example.debug_spans["supervised_suffix"].token_positions == ()
    active_positions = tuple(i for i, label in enumerate(example.labels) if label != -100)
    assert active_positions == example.assistant_stop_token_span.token_indices()
    eos_pos = active_positions[0]
    assert example.input_ids[eos_pos] == example.stop_contract.im_end_token_id
    assert example.labels[eos_pos] == example.stop_contract.im_end_token_id
    # The per-target EOS loss weight is added in Task 8; Task 4 only proves span/label alignment.
```

Add tokenizer stop-contract tests:

```python
def test_compact_training_stop_token_is_im_end_only():
    tokenizer = FakeQwenTokenizer(
        token_to_id={
            "<|im_end|>": 7,
            "<|endoftext|>": 8,
            "<|end_of_text|>": 9,
        },
        eos_token="<|im_end|>",
    )
    contract = resolve_compact_training_stop_contract(tokenizer)
    assert contract.im_end_token_id == 7
    assert contract.training_eos_token_text == "<|im_end|>"
    assert contract.training_eos_token_ids == frozenset({7})
```

```python
def test_compact_training_ignores_text_level_tokenizer_eos():
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_end|>": 7, "<|endoftext|>": 8},
        eos_token="<|endoftext|>",
    )
    contract = resolve_compact_training_stop_contract(tokenizer)
    assert contract.im_end_token_id == 7
    assert contract.training_eos_token_ids == frozenset({7})
    assert contract.tokenizer_eos_token_text == "<|endoftext|>"
    assert contract.tokenizer_eos_token_id == 8
```

```python
def test_rendered_payload_excludes_manual_eos_and_chat_template_supplies_stop():
    example = build_compact_prefix_rollin_example(objects=[A], rollin_order=[A], k=0)
    assert "<|im_end|>" not in example.rendered_assistant.text
    assert example.assistant_stop_token_text == "<|im_end|>"
    assert "<|endoftext|>" not in example.chat_text
    assert "<|end_of_text|>" not in example.chat_text
```

```python
def test_assistant_stop_span_is_template_supplied_im_end_once_after_payload():
    example = build_compact_prefix_rollin_example(objects=[A], rollin_order=[A], k=0)
    assert "<|im_end|>" not in example.rendered_assistant.text
    stop_positions = example.assistant_stop_token_span.token_indices()
    assert len(stop_positions) == 1
    assert example.input_ids[stop_positions[0]] == example.stop_contract.im_end_token_id
    assert example.assistant_stop_char_span.start == example.assistant_char_span.end
```

```python
def test_missing_im_end_that_resolves_to_unk_is_rejected():
    tokenizer = FakeQwenTokenizer(token_to_id={}, unk_token_id=0, eos_token="<|endoftext|>")
    with pytest.raises(ValueError, match="im_end.*unk"):
        resolve_compact_training_stop_contract(tokenizer)
```

```python
def test_im_end_must_encode_as_single_special_token():
    tokenizer = FakeQwenTokenizer(
        token_to_id={"<|im_end|>": 7},
        encode_result_for_im_end=[101, 102],
    )
    with pytest.raises(ValueError, match="single token"):
        resolve_compact_training_stop_contract(tokenizer)
```

```python
def test_closed_qwen_chat_template_supplies_exactly_one_im_end_id():
    tokenizer = fake_qwen_tokenizer()
    contract = resolve_compact_training_stop_contract(tokenizer)
    encoded = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "payload"}],
        tokenize=True,
        add_generation_prompt=False,
    )
    assert encoded.count(contract.im_end_token_id) == 1
    assert encoded[-1] == contract.im_end_token_id
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_dataset_alignment.py \
  tests/test_compact_tokenizer_stop_contract.py \
  -q
```

Expected: builder or debug span support not present yet.

- [ ] **Step 3: Add a compact prefix/suffix render path**

Implement a render path that produces one assistant payload plus one chat-template stop marker:

```text
assistant_payload = entry(emitted_1) ... entry(emitted_K) entry(remaining_1) ... entry(remaining_M)
chat_template_stop = <|im_end|>
```

Mark spans with explicit roles:

```text
rollin_prefix
supervised_suffix
semantic_eos
```

Do not add any extra object-list delimiter span. Do not manually append `<|im_end|>` to `RenderedAssistantSequence.text`; align EOS targets to the assistant stop marker produced by Qwen `apply_chat_template(..., add_generation_prompt=False)`. For `prefix_rollin_et_rmp_ce`, the span/sidecar/metric vocabulary is `eos` or `semantic_eos`.

Do not append `<|endoftext|>` or `<|end_of_text|>` anywhere in the training assistant text. If the tokenizer exposes those tokens for generic LM compatibility, they remain parser/inference cleanup concerns only.

- [ ] **Step 3b: Add tokenizer stop-contract validation**

Implement the focused helper in `src/detection/tokenizer_contract.py`:

```python
@dataclass(frozen=True)
class CompactTrainingStopContract:
    im_end_token_id: int
    training_eos_token_text: Literal["<|im_end|>"]
    training_eos_token_ids: frozenset[int]
    tokenizer_eos_token_text: str | None = None
    tokenizer_eos_token_id: int | None = None


def resolve_compact_training_stop_contract(tokenizer: object) -> CompactTrainingStopContract:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("compact_full requires tokenizer.convert_tokens_to_ids")
    im_end_id = convert("<|im_end|>")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(im_end_id, int) or im_end_id < 0 or im_end_id == unk_token_id:
        raise ValueError("compact_full requires <|im_end|> to resolve to one token id")
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    if "<|im_end|>" not in vocab and "<|im_end|>" not in added_vocab and "<|im_end|>" not in special_map.values():
        raise ValueError("compact_full requires <|im_end|> in tokenizer vocab or special tokens")
    encode = getattr(tokenizer, "encode", None)
    if callable(encode) and encode("<|im_end|>", add_special_tokens=False) != [im_end_id]:
        raise ValueError("compact_full requires <|im_end|> to encode as a single token")
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token, str) and eos_token and eos_token != "<|im_end|>":
        resolved = convert(eos_token)
        if isinstance(resolved, int) and resolved >= 0:
            eos_token_id = int(resolved)
    return CompactTrainingStopContract(
        im_end_token_id=int(im_end_id),
        training_eos_token_text="<|im_end|>",
        training_eos_token_ids=frozenset({int(im_end_id)}),
        tokenizer_eos_token_text=eos_token if isinstance(eos_token, str) else None,
        tokenizer_eos_token_id=eos_token_id if isinstance(eos_token_id, int) else None,
    )
```

This helper deliberately does not include `<|endoftext|>` or `<|end_of_text|>` in `training_eos_token_ids`.

Also probe `apply_chat_template(..., add_generation_prompt=False)` on a minimal closed assistant message and fail fast unless the resolved `<|im_end|>` id appears exactly once after the assistant payload. This protects the implementation from tokenizer configs that expose the special token but whose active chat template does not use it.

- [ ] **Step 4: Preserve Swift-encoded alignment validation**

After actual chat-template encoding, validate:

```text
prepared supervised token ids match encoded supervised token ids after a constant shift
prefix span token ids are present in input_ids but labels are -100
hard label token ids equal encoded labels at supervised positions
for every target in a collated batch:
  input_ids[batch_index, target.position] == target.teacher_token_id
  labels[batch_index, target.position] == target.teacher_token_id
  loss uses logits[batch_index, target.position - 1]
```

Reject tokenizer/collator configurations with left padding unless an explicit sidecar offset rewrite is implemented. For v1, assert `tokenizer.padding_side == "right"` when the tokenizer exposes `padding_side`.

Add padding guard tests:

```python
def test_recursive_sidecar_rejects_left_padding_without_offset_rewrite():
    with pytest.raises(ValueError, match="prefix_rollin_et_rmp_ce.*right padding"):
        collate_prefix_rollin_examples([short_example, long_example], padding_side="left")


def test_right_padding_preserves_sample_local_target_positions():
    batch = collate_prefix_rollin_examples([short_example, long_example], padding_side="right")
    for batch_index, targets in enumerate(batch["recursive_detection_targets"]):
        for target in targets.token_targets:
            assert batch["input_ids"][batch_index, target.position] == target.teacher_token_id
            assert batch["labels"][batch_index, target.position] == target.teacher_token_id
```

Add real-path sidecar tests, not only helper-level padding tests:

```text
two different-length compact examples pass through DetectionTrainingDataset, the actual data collator,
and the trainer/loss extras boundary;
recursive_detection_targets is available to the loss path but is stripped from model kwargs;
padding labels remain -100 and attention_mask is 0 in padded positions;
sidecar target positions remain sample-local right-padding positions.
```

Add truncation fail-fast tests:

```text
low max_length truncates a supervised suffix entry -> ValueError mentioning truncation/encoded_entry_span
low max_length truncates the semantic_eos span -> ValueError mentioning truncation/semantic_eos
boundary case with no partial object entry and intact semantic_eos span -> passes
```

- [ ] **Step 5: Re-run alignment tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_dataset_alignment.py \
  tests/test_compact_tokenizer_stop_contract.py \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit this task when implementing**

```bash
git add src/detection/dataset.py src/detection/template.py src/detection/tokenizer_contract.py tests/test_prefix_rollin_dataset_alignment.py tests/test_compact_tokenizer_stop_contract.py
git commit -m "feat(detection): render prefix-rollin compact examples"
```

## Task 5: Extend Entry-Trie Target Builder For Initial Emitted State

**Files:**

- Modify: `src/detection/objective.py`
- Test: `tests/test_entry_trie_targets.py`

- [ ] **Step 1: Write failing target tests**

Required cases:

```python
def test_emitted_object_is_not_positive_after_rollin_prefix():
    targets = build_targets(objects=[A, B, C], emitted=[A], teacher_suffix=[B, C])
    first_branch = first_multi_positive_target(targets)
    assert A.first_distinguishing_token_id not in first_branch.positive_token_ids
```

```python
def test_shared_prefix_narrows_to_coordinate_branch():
    A = compact_obj("inst-a", desc="scratch", bbox=(10, 20, 30, 40))
    B = compact_obj("inst-b", desc="scratch", bbox=(11, 20, 30, 40))
    targets = build_targets(objects=[A, B], emitted=[], teacher_suffix=[A, B])
    branch = target_after_partial_entry_prefix(targets, [OBJ_REF_START, token("scratch"), BOX_START])
    assert branch.prob_by_token[A.x1_token_id] == pytest.approx(0.5)
    assert branch.prob_by_token[B.x1_token_id] == pytest.approx(0.5)
```

```python
def test_hard_label_is_always_inside_positive_set():
    targets = build_targets(objects=[A, B], emitted=[], teacher_suffix=[B, A])
    for target in targets.token_targets:
        if target.has_positive_candidates:
            assert target.hard_token_id in target.positive_token_ids
```

```python
def test_duplicate_serialized_rows_with_distinct_instance_ids_contribute_multiplicity():
    a1 = compact_obj("inst-a1", desc="same", bbox=(1, 2, 3, 4))
    a2 = compact_obj("inst-a2", desc="same", bbox=(1, 2, 3, 4))
    targets = build_targets(objects=[a1, a2], emitted=[], teacher_suffix=[a1, a2])
    first = first_token_target(targets)
    shared_id = first.teacher_token_id
    assert first.positive_token_ids == (shared_id,)
    assert first.q_by_token[shared_id] == pytest.approx(1.0)
    assert first.candidate_object_count == 2
    assert first.multiplicity_by_token[shared_id] == 2
    assert first.positive_size == 1
```

```python
def test_exact_duplicate_rows_remove_one_object_instance_after_teacher_entry_completion():
    a1 = compact_obj("inst-a1", desc="same", bbox=(1, 2, 3, 4))
    a2 = compact_obj("inst-a2", desc="same", bbox=(1, 2, 3, 4))
    state = advance_after_teacher_entry(objects=[a1, a2], emitted=[], teacher=a1)
    assert state.emitted == (a1.instance_id,)
    assert state.remaining == (a2.instance_id,)
```

```python
def test_shared_next_token_is_aggregated_not_duplicated():
    a = compact_obj("inst-a", desc="scratch", bbox=(10, 20, 30, 40))
    b = compact_obj("inst-b", desc="scratch", bbox=(11, 20, 30, 40))
    targets = build_targets(objects=[a, b], emitted=[], teacher_suffix=[a, b])
    branch = target_after_partial_entry_prefix(targets, [OBJ_REF_START])
    scratch_id = token("scratch")
    assert branch.positive_token_ids == (scratch_id,)
    assert branch.q_by_token[scratch_id] == pytest.approx(1.0)
    assert branch.candidate_object_count == 2
```

```python
def test_candidate_ids_are_sliced_from_encoded_chat_spans_not_standalone_tokenization():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=0)
    for object_id, entry in example.entry_by_instance_id.items():
        assert entry.encoded_entry_token_ids == tuple(
            example.input_ids[entry.encoded_entry_span.start:entry.encoded_entry_span.end]
        )
```

```python
def test_target_builder_does_not_call_standalone_tokenizer_for_candidates():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=0)
    monkeypatch.setattr("src.detection.objective.standalone_tokenize_entry", fail_if_called, raising=False)
    targets = build_recursive_detection_targets(
        initial_emitted_object_instance_ids=(),
        teacher_suffix_object_instance_ids=(A.instance_id, B.instance_id),
        supervised_token_positions=example.debug_spans["supervised_suffix"].token_positions,
        entry_by_instance_id=example.entry_by_instance_id,
        semantic_eos_span=example.assistant_stop_token_span,
        im_end_token_id=example.stop_contract.im_end_token_id,
    )
    assert targets.hard_label_in_positive_rate == pytest.approx(1.0)
```

```python
def test_entry_lookup_is_instance_id_keyed_and_distinguishes_duplicate_rows():
    a1 = compact_obj("inst-a1", desc="same", bbox=(1, 2, 3, 4))
    a2 = compact_obj("inst-a2", desc="same", bbox=(1, 2, 3, 4))
    example = build_compact_prefix_rollin_example(objects=[a1, a2], rollin_order=[a1, a2], k=0)
    assert set(example.entry_by_instance_id) == {"inst-a1", "inst-a2"}
    assert example.entry_by_instance_id["inst-a1"].source_index != example.entry_by_instance_id["inst-a2"].source_index
    for entry in example.entry_by_instance_id.values():
        assert entry.encoded_entry_token_ids == tuple(
            example.input_ids[entry.encoded_entry_span.start:entry.encoded_entry_span.end]
        )
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_entry_trie_targets.py -q
```

Expected: initial emitted-state support is missing or incomplete.

- [ ] **Step 3: Add initial state arguments to target construction**

Extend target construction with explicit inputs:

```python
initial_emitted_object_instance_ids: tuple[ObjectInstanceId, ...]
teacher_suffix_object_instance_ids: tuple[ObjectInstanceId, ...]
supervised_token_positions: tuple[int, ...]
entry_by_instance_id: Mapping[ObjectInstanceId, EncodedCompactObjectEntry]
semantic_eos_span: TokenSpan
im_end_token_id: int
```

Use one authoritative owner record:

```python
@dataclass(frozen=True)
class EncodedCompactObjectEntry:
    instance_id: ObjectInstanceId
    source_index: int
    normalized_object: NormalizedCompactObject
    rendered_entry_span: CharSpan
    encoded_entry_span: TokenSpan
    encoded_entry_token_ids: tuple[int, ...]
```

At every object entry:

```text
remaining = all object_instance_ids - emitted_so_far
trie = build over encoded compact entry token ids for remaining only
candidate token ids must be sliced from the same full closed Qwen chat-template encoding
target builder must not re-render or standalone-tokenize object entries
target builder must not need a tokenizer for candidate construction
teacher object must be in remaining
after teacher entry completes, remove exactly one object instance
```

- [ ] **Step 4: Add failfast invariants**

Fail fast if:

```text
hard label token not in positives for any positive-candidate CE target
positive_token_ids contains duplicate token ids
q probabilities do not sum to 1 within tolerance
teacher object is already emitted
remaining contains duplicate object_instance_id values
target position does not point to an active supervised suffix label
semantic_eos_span is missing, has length != 1, or input_ids[semantic_eos_span.start] != im_end_token_id
standalone tokenization is attempted for candidate object entries
```

- [ ] **Step 5: Re-run target tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_entry_trie_targets.py -q
```

Expected: pass.

- [ ] **Step 6: Commit this task when implementing**

```bash
git add src/detection/objective.py tests/test_entry_trie_targets.py
git commit -m "feat(objective): build trie targets from roll-in remaining set"
```

## Task 6: Centralize Support/Balance And Balance-First Loss

**Files:**

- Modify: `src/detection/loss.py`
- Test: `tests/test_recursive_loss_support_balance.py`

- [ ] **Step 1: Write failing numeric loss tests**

Required cases:

```python
def test_support_one_balance_one_equals_sparse_soft_ce():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    positives = torch.tensor([0, 1])
    q = torch.tensor([0.5, 0.5])
    got = support_balance_loss(logits, positives, q, support_weight=1.0, balance_weight=1.0)
    log_probs = logits.log_softmax(dim=-1)[0]
    expected = -0.5 * log_probs[0] - 0.5 * log_probs[1]
    assert got.item() == pytest.approx(expected.item())
```

```python
def test_higher_balance_penalizes_valid_conditional_collapse():
    flat_valid = torch.tensor([[3.0, 3.0, -5.0]])
    peaked_valid = torch.tensor([[6.0, 0.0, -5.0]])
    positives = torch.tensor([0, 1])
    q = torch.tensor([0.5, 0.5])
    flat_loss = support_balance_loss(flat_valid, positives, q, support_weight=1.0, balance_weight=2.0)
    peaked_loss = support_balance_loss(peaked_valid, positives, q, support_weight=1.0, balance_weight=2.0)
    assert peaked_loss > flat_loss
```

```python
def test_support_term_penalizes_low_absolute_valid_mass_with_same_conditional_distribution():
    high_mass = torch.tensor([[5.0, 5.0, -5.0]])
    low_mass = torch.tensor([[-5.0, -5.0, 5.0]])
    positives = torch.tensor([0, 1])
    q = torch.tensor([0.5, 0.5])
    assert support_balance_loss(low_mass, positives, q, support_weight=1.0, balance_weight=2.0) > support_balance_loss(
        high_mass,
        positives,
        q,
        support_weight=1.0,
        balance_weight=2.0,
    )
    assert support_balance_loss(low_mass, positives, q, support_weight=0.0, balance_weight=1.0) == pytest.approx(
        support_balance_loss(high_mass, positives, q, support_weight=0.0, balance_weight=1.0)
    )
```

```python
def test_support_balance_rejects_duplicate_positive_token_ids():
    with pytest.raises(ValueError, match="unique positive token"):
        support_balance_loss(
            logits=torch.zeros(1, 10),
            positive_token_ids=torch.tensor([7, 7]),
            q=torch.tensor([0.5, 0.5]),
            support_weight=1.0,
            balance_weight=2.0,
        )
```

```python
def test_compute_loss_uses_soft_multipositive_not_teacher_hard_ce():
    target = make_token_target(
        position=3,
        teacher_token_id=10,
        positive_token_ids=(10, 11),
        q_by_token={10: 0.5, 11: 0.5},
        support_weight=1.0,
        balance_weight=1.0,
    )
    logits = logits_that_strongly_favor(token_id=11, disfavor_token_id=10, at_position=2)
    got = compute_recursive_detection_ce_batch_loss(logits=logits, targets=[targets_with(target)])
    hard_ce = hard_ce_loss_at_position(logits, position=2, teacher_token_id=10)
    assert got.loss.item() < hard_ce.item() * 0.5
    assert got.metrics["objective/multipositive_loss_position_count"] == 1
```

```python
def test_balance_weight_changes_integrated_batch_loss_on_same_targets_and_logits():
    target = make_token_target(
        position=3,
        teacher_token_id=10,
        positive_token_ids=(10, 11),
        q_by_token={10: 0.5, 11: 0.5},
    )
    logits = logits_that_peak_inside_valid_set(peak_token_id=10, flat_valid_token_id=11, at_position=2)
    loss_balance1 = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=[targets_with(target.with_weights(support_weight=1.0, balance_weight=1.0))],
    ).loss
    loss_balance2 = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=[targets_with(target.with_weights(support_weight=1.0, balance_weight=2.0))],
    ).loss
    assert loss_balance2 > loss_balance1
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_recursive_loss_support_balance.py -q
```

Expected: helper function or balance-first behavior not yet exposed.

- [ ] **Step 3: Implement one sparse helper**

Implement one helper that consumes logits at a single causal prediction position:

```python
def support_balance_loss(
    logits: torch.Tensor,
    positive_token_ids: torch.Tensor,
    q: torch.Tensor,
    *,
    support_weight: float,
    balance_weight: float,
) -> torch.Tensor:
    if positive_token_ids.unique().numel() != positive_token_ids.numel():
        raise ValueError("positive_token_ids must contain unique vocabulary token ids")
    log_probs = logits.log_softmax(dim=-1)
    valid_log_probs = log_probs.index_select(dim=-1, index=positive_token_ids)
    log_valid_mass = torch.logsumexp(valid_log_probs, dim=-1)
    support = -log_valid_mass
    balance = -(q * (valid_log_probs - log_valid_mass.unsqueeze(-1))).sum(dim=-1)
    return support_weight * support + balance_weight * balance
```

Adapt dimensions to match the existing batch/sample loss structure.

The integrated batch loss must consume the same multi-positive sidecar emitted by `build_recursive_detection_targets`. It is not sufficient for entry-trie targets and diagnostics to exist while `compute_loss` still follows teacher-token hard CE. Required loss-component metrics:

```text
objective/support_loss_sum
objective/support_loss_count
objective/balance_loss_sum
objective/balance_loss_count
objective/multipositive_loss_position_count
objective/hard_ce_position_count
objective/multipositive_loss_fraction
```

- [ ] **Step 4: Wire config defaults**

Ensure the new ablation config resolves:

```text
support_weight = 1.0
balance_weight = 2.0
```

Do not reuse old `branch_*` config names.

- [ ] **Step 5: Re-run loss tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_recursive_loss_support_balance.py -q
```

Expected: pass.

- [ ] **Step 6: Commit this task when implementing**

```bash
git add src/detection/loss.py tests/test_recursive_loss_support_balance.py
git commit -m "feat(loss): add balance-first sparse trie loss"
```

## Task 7: Add Compact Token-Type Gate

**Files:**

- Create: `src/detection/token_types.py`
- Modify: `src/detection/objective.py`
- Modify: `src/detection/loss.py`
- Test: `tests/test_compact_type_gate.py`
- Test: `tests/test_compact_tokenizer_stop_contract.py`

- [ ] **Step 1: Write failing type-gate tests**

Required cases:

```python
def test_compact_token_groups_classify_struct_coord_desc_and_eos():
    groups = build_compact_token_type_groups(tokenizer)
    assert tokenizer.convert_tokens_to_ids("<|object_ref_start|>") in groups.struct
    assert tokenizer.convert_tokens_to_ids("<|box_start|>") in groups.struct
    assert tokenizer.convert_tokens_to_ids("<|coord_0|>") in groups.coord
    assert tokenizer.convert_tokens_to_ids("<|coord_999|>") in groups.coord
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") not in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") not in groups.eos
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") not in groups.desc
    assert tokenizer.pad_token_id not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") not in groups.desc
    assert tokenizer.convert_tokens_to_ids("<|vision_start|>") not in groups.desc
```

```python
def test_type_gate_uses_union_of_positive_child_types():
    target = make_multi_positive_target(
        positive_token_ids=[DESC_TOKEN_ID, BOX_START_TOKEN_ID],
        positive_token_types=["desc", "struct"],
    )
    allowed = allowed_type_token_ids_for_target(target, groups)
    assert DESC_TOKEN_ID in allowed
    assert BOX_START_TOKEN_ID in allowed
```

```python
def test_type_gate_weight_is_added_to_position_loss():
    got = combine_main_and_type_losses(main_loss=torch.tensor(1.25), type_loss=torch.tensor(0.5), type_weight=0.2)
    assert got.item() == pytest.approx(1.35)
```

```python
def test_positive_tokens_are_subset_of_expanded_allowed_types():
    target = make_multi_positive_target(
        positive_token_ids=[DESC_TOKEN_ID, BOX_START_TOKEN_ID],
        positive_token_types=["desc", "struct"],
    )
    allowed = allowed_type_token_ids_for_target(target, groups)
    assert set(target.positive_token_ids).issubset(allowed)
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_compact_type_gate.py -q
```

Expected: token-type helper not present yet.

- [ ] **Step 3: Implement token-type grouping**

Create a frozen grouping container:

```python
@dataclass(frozen=True)
class CompactTokenTypeGroups:
    struct: frozenset[int]
    coord: frozenset[int]
    eos: frozenset[int]
    excluded_control: frozenset[int]
    desc: frozenset[int]
```

Rules:

```text
struct = object_ref_start + box_start
coord = coord_0 through coord_999
eos = im_end
excluded_control includes padding, chat role/header sentinels, image placeholders, tokenizer-added control tokens, and text-level terminators
desc allowed mass = vocabulary minus struct minus coord minus eos minus excluded_control
text-level terminators such as <|endoftext|> and <|end_of_text|> are excluded from every compact training allowed type set, including desc
```

- [ ] **Step 4: Add type-gate sidecar or loss atoms**

For each supervised target position, attach allowed type ids or a compact type enum that loss can expand.

At multi-positive positions:

```text
allowed_types = union(type(child_token) for child_token in positives)
```

At singleton hard positions:

```text
allowed_types = type(hard_token)
```

- [ ] **Step 5: Add type-gate loss and metrics**

Compute:

```text
L_type = -logsumexp(log_probs[allowed_type_token_ids])
L_total_position = L_main + type_gate.weights[position_type] * L_type
```

Required v1 defaults:

```text
type_gate.weights.struct = 2.0
type_gate.weights.coord = 1.0
type_gate.weights.desc = 0.2
type_gate.weights.eos = 0.5
```

Log:

```text
type_gate/allowed_mass_mean
type_gate/type_violation_mass_mean
type_gate/positive_not_in_allowed_type_count
type_gate/special_control_in_desc_count
```

- [ ] **Step 6: Re-run type-gate tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_compact_type_gate.py -q
```

Expected: pass.

- [ ] **Step 7: Commit this task when implementing**

```bash
git add src/detection/token_types.py src/detection/objective.py src/detection/loss.py tests/test_compact_type_gate.py
git commit -m "feat(detection): add compact token-type gate"
```

## Task 8: Implement `<|im_end|>` EOS Prior Interface

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/detection/objective.py`
- Modify: `src/detection/loss.py`
- Test: `tests/test_eos_prior.py`

- [ ] **Step 1: Write failing EOS tests**

Required cases:

```python
def test_eos_not_positive_when_remaining_objects_exist():
    targets = build_targets(objects=[A, B], emitted=[], teacher_suffix=[A, B])
    first_state = target_at_first_object_start(targets)
    assert IM_END_ID not in first_state.positive_token_ids
```

```python
def test_k_equals_n_uses_weighted_im_end_supervision():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=2, eos_trust_weight=0.25)
    targets = example.recursive_targets
    eos_target = only_eos_target(targets)
    assert eos_target.token_id == example.stop_contract.im_end_token_id
    assert eos_target.position == example.assistant_stop_token_span.start
    assert eos_target.loss_weight == pytest.approx(0.25)
```

```python
def test_k_less_than_n_adds_weighted_im_end_after_suffix_completion():
    example = build_compact_prefix_rollin_example(objects=[A, B], rollin_order=[A, B], k=1, eos_trust_weight=0.25)
    eos_target = final_eos_target(example.recursive_targets)
    assert eos_target.token_id == example.stop_contract.im_end_token_id
    assert eos_target.position == example.assistant_stop_token_span.start
    assert eos_target.loss_weight == pytest.approx(0.25)
```

```python
def test_empirical_unlabeled_poisson_eos_trust_weight_values():
    cfg = empirical_unlabeled_poisson_v0(
        intercept=-0.35,
        slope=0.43,
        penalty_per_missing=1.0,
        temperature=1.0,
    )
    assert compute_eos_trust_weight(gt_count=0, cfg=cfg) == pytest.approx(1.0)
    assert compute_eos_trust_weight(gt_count=1, cfg=cfg) == pytest.approx(math.exp(-0.08))
    assert compute_eos_trust_weight(gt_count=2, cfg=cfg) == pytest.approx(math.exp(-0.51))
    assert compute_eos_trust_weight(gt_count=10, cfg=cfg) == pytest.approx(math.exp(-3.95))
```

```python
def test_k_equals_n_eos_trust_weight_zero_keeps_zero_weight_eos_target():
    targets = build_targets(objects=[A, B], emitted=[A, B], teacher_suffix=[], eos_trust_weight=0.0)
    eos_target = only_eos_target(targets)
    assert eos_target.token_id == IM_END_ID
    assert eos_target.loss_weight == pytest.approx(0.0)
```

```python
def test_production_rejects_missing_calibrated_eos_formula():
    with pytest.raises(ValueError):
        load_latest_detection_config(config_with_eos_trust_weight_source("empirical_unlabeled_poisson_v0", surface="production"))
```

```python
def test_empirical_eos_source_is_never_production_even_with_artifact_ref():
    with pytest.raises(ValueError, match="calibrated_formula_ref"):
        load_latest_detection_config(
            config_with_eos_trust_weight_source(
                "empirical_unlabeled_poisson_v0",
                surface="production",
                calibration_artifact_ref="eos_calibration_formula.json",
            )
        )
```

```python
def test_deferred_user_formula_is_rejected_as_stale_source_on_all_surfaces():
    for surface in ("smoke", "ablation", "production"):
        with pytest.raises(ValueError, match="deferred_user_formula"):
            load_latest_detection_config(config_with_eos_trust_weight_source("deferred_user_formula", surface=surface))
```

```python
def test_empirical_eos_trust_weight_clamps_to_configured_bounds():
    cfg = empirical_unlabeled_poisson_v0(
        intercept=-0.35,
        slope=0.43,
        penalty_per_missing=1.0,
        temperature=1.0,
        min_weight=0.1,
        max_weight=0.9,
    )
    assert compute_eos_trust_weight(gt_count=0, cfg=cfg) == pytest.approx(0.9)
    assert compute_eos_trust_weight(gt_count=20, cfg=cfg) == pytest.approx(0.1)
```

```python
def test_disabled_ablation_returns_exact_zero_eos_trust_weight():
    cfg = disabled_eos_ablation()
    assert compute_eos_trust_weight(gt_count=0, cfg=cfg) == pytest.approx(0.0)
    assert compute_eos_trust_weight(gt_count=50, cfg=cfg) == pytest.approx(0.0)
```

```python
def test_constant_ablation_returns_configured_eos_trust_weight_for_smoke_and_ablation():
    for surface in ("smoke", "ablation"):
        cfg = load_latest_detection_config(config_with_constant_eos(value=0.25, surface=surface))
        assert compute_eos_trust_weight(gt_count=0, cfg=cfg.objective.eos.eos_trust_weight) == pytest.approx(0.25)
        assert compute_eos_trust_weight(gt_count=50, cfg=cfg.objective.eos.eos_trust_weight) == pytest.approx(0.25)
```

```python
def test_constant_ablation_rejects_out_of_range_values():
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError, match="constant_ablation"):
            load_latest_detection_config(config_with_constant_eos(value=value, surface="ablation"))
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_eos_prior.py -q
```

Expected: EOS policy support is missing or too hard-coded.

- [ ] **Step 3: Implement EOS target creation**

Rules:

```text
if remaining_objects is nonempty:
    no EOS positive target is created for that state
if remaining_objects is empty:
    create hard `<|im_end|>` target at `semantic_eos_span.start` with loss_weight = eos_trust_weight
    require len(semantic_eos_span) == 1
    require input_ids[semantic_eos_span.start] == im_end_token_id
    require labels[semantic_eos_span.start] == im_end_token_id
    loss uses logits[semantic_eos_span.start - 1]
```

This empty-remaining rule applies after every supervised suffix completion, not only when the initial roll-in sample has `K=N`.

No extra object-list delimiter target exists in this variant.

- [ ] **Step 4: Implement formula-source policy and empirical EOS prior**

Accepted sources:

```text
empirical_unlabeled_poisson_v0: ablation/smoke source using the user-provided rough formula
constant_ablation: explicit numeric EOS trust weight in [0, 1], allowed only for ablation/smoke
disabled_ablation: explicit disabled EOS supervision for ablation/smoke
calibrated_formula_ref: required for production after user supplies the formula contract
```

Schema for constant ablation:

```yaml
eos_trust_weight:
  source: constant_ablation
  value: 0.25
```

Rejected stale sources:

```text
deferred_user_formula: reject on every executable surface; use calibrated_formula_ref for production and empirical_unlabeled_poisson_v0 for the first ablation
```

For `empirical_unlabeled_poisson_v0`, implement:

```python
def expected_unlabeled_count(gt_count: int, *, intercept: float = -0.35, slope: float = 0.43) -> float:
    return max(0.0, intercept + slope * float(gt_count))


def eos_trust_weight_from_expected_unlabeled_count(
    expected_count: float,
    *,
    penalty_per_missing: float = 1.0,
    temperature: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> float:
    if penalty_per_missing < 0:
        raise ValueError("EOS prior penalty_per_missing must be non-negative")
    if temperature <= 0:
        raise ValueError("EOS prior temperature must be positive")
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise ValueError("EOS prior clamp must satisfy 0 <= min_weight <= max_weight <= 1")
    raw = math.exp(
        -float(penalty_per_missing)
        * max(0.0, float(expected_count))
        / float(temperature)
    )
    return min(max(raw, min_weight), max_weight)
```

Thus:

```text
unlabel_count = max(0, -0.35 + 0.43 * GT_count)
eos_trust_weight_raw = exp(-penalty_per_missing * unlabel_count / temperature)
eos_trust_weight = clamp(eos_trust_weight_raw, min_weight, max_weight)
L_eos = eos_trust_weight * CE(<|im_end|>)
CE(<|im_end|>) = -log p_theta(<|im_end|>)
```

The default `penalty_per_missing` is `1.0`, `temperature` is `1.0`, `min_weight` is `0.0`, and `max_weight` is `1.0`.

This makes the EOS trust penalty linear in log space:

```text
-log(eos_trust_weight_raw) = penalty_per_missing * unlabel_count / temperature
```

So every additional expected unlabeled object multiplies the EOS trust weight by `exp(-penalty_per_missing / temperature)`. Keep config fields for `penalty_per_missing`, `temperature`, `min_weight`, and `max_weight` so later ablations can soften or clip the prior without changing the formula source.

For `calibrated_formula_ref`, implement only the reference plumbing and failfast shape. Production configs must use `source: calibrated_formula_ref` and point to an approved `eos_calibration_formula.json`. Do not treat the empirical v0 source as final production calibration, even when a calibration artifact ref is present.

EOS target construction must consume only the `<|im_end|>` id from the compact tokenizer stop contract. If `tokenizer.eos_token_id` corresponds to `<|endoftext|>` or another text-level terminator, keep it as diagnostic metadata only and do not include it in training EOS targets.

- [ ] **Step 5: Add EOS diagnostics**

Log:

```text
eos/eos_trust_weight_mean
eos/expected_unlabeled_count_mean
eos/eos_trust_weight_raw_by_gt_count_bucket
eos/eos_trust_weight_applied_by_gt_count_bucket
eos/eos_prob_by_gt_count_bucket
eos/eos_logit_by_prefix_depth
eos/continue_vs_eos_margin
eos/eos_target_count
eos/eos_zero_weight_target_count
eos/eos_positive_when_nonempty_violation_count
eos/nonempty_state_count
```

For the margin, use:

```text
logsumexp(logits over legal next-object-start tokens) - eos_logit
```

when legal next-object-start tokens exist.

- [ ] **Step 6: Re-run EOS tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_eos_prior.py -q
```

Expected: pass.

- [ ] **Step 7: Commit this task when implementing**

```bash
git add src/config/schema.py src/detection/objective.py src/detection/loss.py tests/test_eos_prior.py
git commit -m "feat(objective): add calibrated im-end eos prior interface"
```

## Task 9: Wire Runtime Variant After Cleanup

**Files:**

- Modify: `src/detection/runtime.py`
- Modify: `src/sft.py`
- Review/Modify: `src/bootstrap/trainer_setup.py` for recursive CE mixin gating changes
- Modify: `src/training_runtime/plan.py`
- Modify: `docs/training/README.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/catalog.yaml`
- Test: existing runtime/config tests plus `tests/test_prefix_rollin_runtime.py`

- [ ] **Step 1: Write failing runtime-route tests**

Add tests asserting:

```text
prefix_rollin_et_rmp_ce routes through latest detection dataset/loss plumbing
legacy continuation route remains absent after Task 1 cleanup
prefix_rollin_et_rmp_ce rejects left padding unless sidecar offset rewrite is implemented
prefix_rollin_et_rmp_ce rejects packing/cache reuse until sidecar offset rewriting exists
src/sft.py calls the latest-detection runtime owner after tokenizer construction and before dataset/collator construction
```

- [ ] **Step 2: Run focused runtime tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_schema.py tests/test_eos_prior.py -q
```

Expected: route-related tests fail before wiring.

- [ ] **Step 3: Wire new variant only**

In `src/detection/runtime.py`, make the latest-detection runtime the canonical owner. The runtime owner must receive tokenizer context, for example:

```python
def assert_prefix_rollin_runtime_supported(
    training_config: LatestDetectionTrainingConfig,
    *,
    tokenizer: object,
    encoded_sample_cache_cfg: object | None,
) -> None: ...
```

`src/sft.py` should call this owner after tokenizer construction and before dataset/collator construction. Do not copy this policy into multiple variant branches.

```text
if objective.variant == prefix_rollin_et_rmp_ce:
    require latest detection stack
    require compact_full
    require tokenizer compact stop contract resolves <|im_end|>
    require tokenizer.padding_side in {missing, right}
    require packing/cache disabled
    pass rollin/target/type_gate/eos configs to dataset and loss
```

`src/sft.py` should delegate to this owner instead of duplicating objective-routing policy. Do not reintroduce compatibility branches for old `custom.*` configs.

- [ ] **Step 4: Re-run route tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_schema.py \
  tests/test_eos_prior.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_latest_training_config_contract.py \
  tests/test_prefix_rollin_runtime.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit this task when implementing**

```bash
git add src/detection/runtime.py src/sft.py src/bootstrap/trainer_setup.py src/training_runtime/plan.py docs/training/README.md docs/training/STAGE1_OBJECTIVE.md docs/catalog.yaml tests/test_prefix_rollin_schema.py
git commit -m "feat(train): route compact prefix-rollin as latest objective"
```

## Task 10: Add Default E1 Ablation Config

**Files:**

- Create: `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml`
- Test: `tests/test_prefix_rollin_schema.py`

- [ ] **Step 1: Create ablation YAML from strict schema**

The default first runnable ablation is exactly `E1`. The first config must include:

```yaml
experiment:
  surface: ablation
  ablation_id: E1
  claim_scope: none

detection_template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy

objective:
  id: recursive_detection_ce
  variant: prefix_rollin_et_rmp_ce
  rollin:
    enabled: true
    source: ground_truth
    prefix_loss: masked
    suffix_order: same_sampled_permutation
    k_distribution:
      type: uniform_inclusive
      min_k: 0
      max_k: object_count
  target:
    type: entry_trie_support_balance
    trie_scope: object_entry
    q_weighting: object_multiplicity_uniform
    singleton: hard_ce
    control_tokens: hard_ce
    support_weight: 1.0
    balance_weight: 2.0
  type_gate:
    enabled: true
    mode: allowed_type_mass
    weights:
      struct: 2.0
      coord: 1.0
      desc: 0.2
      eos: 0.5
  eos:
    eos_token: <|im_end|>
    policy: missing_label_prior_weighted_ce
    eos_trust_weight:
      source: empirical_unlabeled_poisson_v0
      expected_unlabeled_count:
        intercept: -0.35
        slope: 0.43
        floor: 0.0
      trust_mapping:
        type: log_linear_missing_count_penalty
        penalty_per_missing: 1.0
        temperature: 1.0
        min_weight: 0.0
        max_weight: 1.0
```

Keep the rest of the model/data/training settings aligned with the latest compact recursive detection smoke/prod configs. The empirical EOS prior is intentionally an ablation choice, not final production calibration.

- [ ] **Step 2: Add config load test**

Assert the config loads and resolves exact objective weights:

```python
assert cfg.objective.target.support_weight == 1.0
assert cfg.objective.target.balance_weight == 2.0
assert cfg.objective.rollin.suffix_order == "same_sampled_permutation"
assert cfg.objective.type_gate.weights.struct == pytest.approx(2.0)
assert cfg.objective.type_gate.weights.coord == pytest.approx(1.0)
assert cfg.objective.type_gate.weights.desc == pytest.approx(0.2)
assert cfg.objective.type_gate.weights.eos == pytest.approx(0.5)
assert cfg.objective.eos.eos_trust_weight.source == "empirical_unlabeled_poisson_v0"
assert cfg.objective.eos.eos_trust_weight.expected_unlabeled_count.intercept == pytest.approx(-0.35)
assert cfg.objective.eos.eos_trust_weight.expected_unlabeled_count.slope == pytest.approx(0.43)
assert cfg.objective.eos.eos_trust_weight.trust_mapping.penalty_per_missing == pytest.approx(1.0)
assert cfg.objective.eos.eos_trust_weight.trust_mapping.temperature == pytest.approx(1.0)
assert cfg.objective.eos.eos_trust_weight.trust_mapping.min_weight == pytest.approx(0.0)
assert cfg.objective.eos.eos_trust_weight.trust_mapping.max_weight == pytest.approx(1.0)
assert cfg.experiment.surface == "ablation"
```

- [ ] **Step 3: Run config test**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_schema.py -q
```

Expected: pass.

- [ ] **Step 4: Commit this task when implementing**

```bash
git add configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml tests/test_prefix_rollin_schema.py
git commit -m "config: add compact prefix-rollin balance2 ablation"
```

## Task 11: Add Diagnostics And Smoke Verification

**Files:**

- Modify: `src/detection/loss.py`
- Modify: `src/trainers/metrics/recursive_detection.py`
- Modify: `src/metrics/payload_contract.py`
- Test: `tests/test_prefix_rollin_diagnostics.py`
- Test: `tests/test_prefix_rollin_metric_payload.py`

- [ ] **Step 1: Add metric aggregation keys**

Emit these keys where the current recursive CE loss metrics are collected:

```text
rollin/k_histogram
rollin/prefix_token_count_mean
rollin/suffix_object_count_mean
rollin/supervised_suffix_token_ratio
trie/hard_label_in_positive_rate
trie/q_sum_error_max
trie/positive_size_mean
trie/positive_token_ids_unique_violation_count
trie/multipositive_position_count
trie/singleton_position_count
trie/target_entropy_mean
trie/valid_mass_mean
trie/pred_valid_entropy_mean
trie/conditional_kl_q_to_pred_mean
trie/conditional_top1_matches_q_argmax_rate
trie/max_positive_prob_mean
objective/support_loss_sum
objective/support_loss_count
objective/balance_loss_sum
objective/balance_loss_count
objective/multipositive_loss_position_count
objective/hard_ce_position_count
objective/multipositive_loss_fraction
type_gate/allowed_mass_mean
type_gate/type_violation_mass_mean
type_gate/allowed_vocab_fraction_mean
type_gate/desc_struct_union_rate
type_gate/positive_not_in_allowed_type_count
type_gate/special_control_in_desc_count
type_gate/position_count
eos/eos_trust_weight_mean
eos/expected_unlabeled_count_mean
eos/eos_trust_weight_raw_by_gt_count_bucket
eos/eos_trust_weight_applied_by_gt_count_bucket
eos/eos_prob_by_gt_count_bucket
eos/eos_logit_by_prefix_depth
eos/continue_vs_eos_margin
eos/eos_target_count
eos/eos_zero_weight_target_count
eos/eos_positive_when_nonempty_violation_count
eos/nonempty_state_count
objective/contributing_sample_count
```

Histogram and bucket diagnostics must be expanded into stable scalar keys. Do not emit raw dict/list/tensor metric values to the reporter.

Every mean/rate metric must have one of:

```text
sum/count reducer pair
explicit max reducer for *_max and *_any metrics
explicit histogram bucket counters for histogram metrics
```

Do not compute paper-facing scalar means by averaging rank-local means. Metrics ending in `_count` should sum across ranks unless they are deliberately renamed as `_any`.

- [ ] **Step 2: Add no-NaN and denominator tests**

Test that a batch with:

```text
K=0
K=N
one object
two shared-prefix objects
```

emits finite metrics and does not divide by zero.

Add a DDP reducer contract test with synthetic rank payloads:

```python
def test_prefix_rollin_ddp_metrics_use_global_numerators_and_denominators():
    rank0 = metric_payload(sum_key="trie/valid_mass_sum", count_key="trie/multipositive_position_count", total=0.0, count=1)
    rank1 = metric_payload(sum_key="trie/valid_mass_sum", count_key="trie/multipositive_position_count", total=9.0, count=9)
    reduced = reduce_prefix_rollin_metric_payloads([rank0, rank1])
    assert reduced["trie/valid_mass_mean"] == pytest.approx(0.9)
```

This synthetic case must fail under the wrong rank-local mean reducer, which would produce `(0.0 + 1.0) / 2 = 0.5`.

Add payload-shape tests asserting all emitted values are finite scalar numbers and that bucket diagnostics materialize as stable named keys such as `eos/eos_trust_weight_applied_by_gt_count_bucket/gt_10_20`.

- [ ] **Step 3: Run targeted diagnostics tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests -q -k "prefix_rollin or trie or type_gate or eos"
```

Expected: pass.

- [ ] **Step 4: Run config-level smoke materialization**

Run the no-training config/batch materialization test:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_config_materialization.py -q
```

Expected:

```text
sample encodes successfully
prefix labels are masked
sidecars align to encoded positions
recursive_detection_targets survives into the loss extras but is stripped from model kwargs
right padding preserves sample-local target positions
truncation/partial-span cases fail fast
metrics are finite
```

- [ ] **Step 5: Commit this task when implementing**

```bash
git add src/detection/loss.py src/trainers/metrics/recursive_detection.py src/metrics/payload_contract.py tests/test_prefix_rollin_diagnostics.py tests/test_prefix_rollin_metric_payload.py tests/test_prefix_rollin_config_materialization.py
git commit -m "test(detection): add prefix-rollin diagnostics coverage"
```

## Task 12: Add Research Diagnostics Artifact Contracts

**Files:**

- Modify: `docs/ARTIFACTS.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/eval/CONTRACT.md`
- Modify: `docs/eval/WORKFLOW.md`
- Test: `tests/test_prefix_rollin_artifact_contracts.py`

- [ ] **Step 1: Define prefix forced-eval artifact schemas**

Document these outputs:

```text
prefix_forced_eval/summary.json
prefix_forced_eval/per_case.jsonl
```

Common artifact header fields for every JSON/JSONL family:

```text
schema_version
artifact_family
created_at_utc
git_commit
command
config_path
resolved_config_json
checkpoint_path
model_id
tokenizer_id
chat_template_id_or_hash
dataset_jsonl
image_root
dataset_split
dataset_fingerprint_or_row_hash
sample_scope
limit
coordinate_surface
bbox_format
eval_surface
decode_mode
decode_settings
parser_mode
prediction_parse_contract
stop_token_contract
artifact_root
source_run_artifacts
run_manifest_refs.resolved_config_json
run_manifest_refs.effective_runtime_json
run_manifest_refs.experiment_manifest_json
run_manifest_refs.run_metadata_json
run_manifest_refs.runtime_env_json
run_manifest_refs.train_data_provenance_json
run_manifest_refs.eval_data_provenance_json
run_manifest_refs.pipeline_manifest_json
pipeline_manifest_status
source_prediction_artifact
source_metrics_json
source_resolved_config_json
```

For Stage-1 latest compact detection runs, `pipeline_manifest_status` is usually
`not_applicable` and `run_manifest_refs.pipeline_manifest_json` should be null.
Do not fabricate an empty `pipeline_manifest.json` to satisfy a research
diagnostic schema.

For teacher-forced or forced-prefix scoring artifacts that do not free-decode, still include explicit null/not-applicable provenance:

```text
decode_mode=teacher_forced_or_forced_prefix_scoring
decode_settings=null
parser_mode=not_applicable
```

Required per-case fields:

```text
record_idx
image
gt_count
prefix_k
prefix_mode
prefix_seed
prefix_object_instance_ids
remaining_object_instance_ids
teacher_suffix_object_instance_ids
remaining_gt_count
nll_teacher_suffix_total
nll_teacher_suffix_per_token
nll_first_remaining_entry
supervised_suffix_token_count
assistant_token_count
masked_prefix_token_count
hard_label_in_positive_rate
positive_size_mean
target_entropy_mean
valid_mass_mean
valid_mass_min
type_allowed_mass_mean
eos_logit
continue_logsumexp
continue_minus_eos_margin
eos_trust_weight
row_hash
```

- [ ] **Step 2: Define permutation and decode diagnostic artifacts**

Document:

```text
permutation_nll/summary.json
prefix_jitter/summary.json
prefix_rollin_decode_diagnostics/summary.json
```

The decode summary must separate raw, scored, guarded, and scored-guarded behavior. Every view must be present or explicitly marked `not_materialized`:

```text
decode_views.raw.status
decode_views.raw.source_artifact_path
decode_views.raw.metrics_json
decode_views.raw.per_image_json
decode_views.raw.parser_mode
decode_views.raw.decode_settings
decode_views.raw.parse_drop_count
decode_views.raw.prediction_count
decode_views.raw.duplicate_rate
decode_views.raw.early_eos_rate
decode_views.raw.recall_by_gt_count_bucket
decode_views.scored.status
decode_views.scored.source_artifact_path
decode_views.scored.score_policy
decode_views.guarded.status
decode_views.guarded.source_artifact_path
decode_views.guarded.guard_policy
decode_views.scored_guarded.status
decode_views.scored_guarded.source_artifact_path
decode_views.scored_guarded.score_policy
decode_views.scored_guarded.guard_policy
```

If a view is not materialized, use `status=not_materialized` and `source_artifact_path=null`; do not silently omit the view.

- [ ] **Step 3: Define EOS calibration probe and formula artifacts**

Document:

```text
eos_calibration_probe/summary.json
eos_calibration_probe/by_gt_count.csv
eos_calibration_probe/noncollapse_fp_rows.jsonl
eos_calibration_formula.json
```

The formula artifact must include:

```text
schema_version
formula_id
formula_text_or_ref
checkpoint
config
dataset_jsonl
image_root
eval_scope=full-val
eval_artifact_root
metrics_json
matches_jsonl
duplicate_guard_report_json if used
noncollapse_fp_definition
gt_count_bucket_edges
noncollapse_fp_rate_by_gt_count
eos_trust_weight_by_gt_count_bucket or closed-form parameters
clipping/interpolation/extrapolation rules
git_commit
command
row_hash
```

- [ ] **Step 4: Add artifact-schema smoke tests**

Create tiny synthetic artifacts with two rows and validate required fields. This is a schema/materialization test only; do not run full-val in this task. Include synthetic cases for forced scoring (`decode_settings=null`, `parser_mode=not_applicable`), raw-only decode diagnostics with non-materialized scored views, and scored/scored-guarded diagnostics with all score/guard source artifact paths present.

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_artifact_contracts.py -q
```

Expected: synthetic artifact contract tests pass and `docs/ARTIFACTS.md` registers all new artifact families.

## Task 13: Add Ablation Matrix Contract

**Files:**

- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/README.md`
- Modify: `docs/catalog.yaml`

- [ ] **Step 1: Document executable ablation rows**

Use a table with columns:

```text
ablation_id
status
config_path
base_config_path
objective.variant
order_policy
rollin_policy
suffix_order_policy
target_policy
support_weight
balance_weight
type_gate_policy
eos_policy
eos_trust_weight.source
train_scope
eval_scope
dataset_jsonl
image_root
checkpoint_init
seed
replicate_id
launch_shape
compute_normalization
required_metric_keys
required_artifact_families
artifact_root
interpretation_status
```

Minimum rows:

```text
A0 canonical one-hot
B0 random permutation one-hot
C0 prefix roll-in one-hot
D0 prefix roll-in trie support=1 balance=1
E0 prefix roll-in trie support=1 balance=2 without type gate
E1 default: E0 plus type gate plus empirical_unlabeled_poisson_v0
E2 E1 with EOS disabled/zero
E3 E1 with constant EOS grid if needed
E4 E1 with production calibrated EOS artifact
```

Each minimum row must also define a concrete evidence gate. Registry validation should fail if these lists are empty or generic:

```text
A0/B0/C0/D0/E0:
  required_metric_keys:
    trie/hard_label_in_positive_rate
    trie/q_sum_error_max
    objective/contributing_sample_count
    eval metrics with explicit eval_surface
  required_artifact_families:
    resolved_config/effective_runtime/run_metadata/experiment_manifest
    pipeline_manifest_status present with `present` or `not_applicable`
    eval metrics.json/per_image.json/source gt_vs_pred family
    ablation_registry row

E1:
  all E0 keys
  type_gate/allowed_mass_mean
  type_gate/type_violation_mass_mean
  type_gate/positive_not_in_allowed_type_count
  eos/eos_trust_weight_mean
  eos/eos_target_count
  eos/eos_positive_when_nonempty_violation_count
  eos/eos_trust_weight_applied_by_gt_count_bucket/*
  objective/multipositive_loss_position_count
  prefix_forced_eval/summary.json
  prefix_forced_eval/per_case.jsonl
  prefix_rollin_decode_diagnostics/summary.json

E2:
  all E1 keys
  eos/eos_zero_weight_target_count
  eos/eos_trust_weight_mean == 0 contract evidence
  early-EOS/recall comparison artifact against E1

E3:
  all E1 keys
  constant_ablation value identifier
  per-grid eos metric bundle
  sensitivity summary artifact

E4:
  all E1 production-candidate keys
  eos_calibration_probe/summary.json
  eos_calibration_probe/by_gt_count.csv
  eos_calibration_probe/noncollapse_fp_rows.jsonl
  eos_calibration_formula.json with production_approved=true and formula_class=calibrated_production
  full-val eval artifact roots and metrics_json paths
```

- [ ] **Step 2: Define first-pass compute normalization**

Use same model/data/optimizer settings and same optimizer-step count for the first pass. Always report:

```text
world_size
gpu_count_and_model
per_device_batch_size
gradient_accumulation_steps
effective_batch_size
max_length
max_pixels
optimizer_step_budget
seen_image_count
supervised_suffix_token_count
assistant_token_count
masked_prefix_token_count
forward_count
backward_count
wall_clock_time
gpu_hours
supervised_suffix_token_delta_pct_vs_baseline
seen_image_delta_pct_vs_baseline
gpu_hours_delta_pct_vs_baseline
sensitivity_required
sensitivity_reason
same_supervised_token_sensitivity_artifact_root
same_seen_image_sensitivity_artifact_root
```

First-pass comparisons may hold optimizer steps fixed, but every claim must also report `seen_image_count` and `supervised_suffix_token_count`. If `supervised_suffix_token_count` differs by more than 5% from the selected baseline, objective-superiority claims require a same-supervised-token sensitivity run. If `seen_image_count` differs by more than 5%, data-exposure claims require a same-seen-image sensitivity run. If GPU hours differ by more than 10%, efficiency claims must report both fixed-step and fixed-compute views. The ablation registry must set `sensitivity_required=true` whenever these thresholds are exceeded and no corresponding artifact root exists.

2026-05-14 correction: this rule applies directly to the A2-vs-A3/A4
retrospective. A2 supervises the full compact-full sequence, while prefix-rollin
with `K ~ Uniform[0,N]` supervises about half the object entries per image
exposure. Therefore A3/A4 are mechanism probes against A2 unless paired with a
same-supervised-token sensitivity run. The clean EOS ablation is A2+EOS-loosen;
the fair prefix-rollin ablation is supervised-token-matched A3 or an A2/A3
mixture.

- [ ] **Step 3: Validate registry rows are linkable**

Each row must point to an existing config path, define `sample_scope` with canonical labels only (`tiny`, `val200`, `limit=200`, `first-200`, `full-val`, `proxy`, `test-dev`), and list the exact required artifact families for the row. Registry validation tests must reject E4 rows without calibration artifacts and reject any row whose required metric/artifact gates are missing.

## Research Promotion Ladder

Do not promote a result directly from a scalar training log to a durable claim. Use this ladder:

```text
implementation test pass
  -> config materialized
  -> smoke artifact exists
  -> ablation registry row is updated
  -> artifact schema validation passes
  -> registry evidence gate passes
  -> scope labels are complete
  -> run artifacts and manifests are complete
  -> progress note only if the result is useful evidence
  -> checked-in claim/decision note only with metric scope and artifact links
  -> docs update only when behavior becomes stable current guidance
  -> OpenSpec only for compatibility-sensitive stable contracts
```

The next repo-local tracking surface should keep only coarse workstream/gate
state. This super-power plan owns file-level implementation details and
verification commands.

## Task 14: Documentation Cleanup After Tests Pass

**Files:**

- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/README.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/training/STAGE1_ET_RMP_CE.md` or move it to historical-only status
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/AGENT_INDEX.md`
- Keep: `docs/superpowers/specs/2026-05-06-compact-full-prefix-rollin-multipositive-unification-design.md`
- Keep: `docs/superpowers/plans/2026-05-06-compact-full-prefix-rollin-multipositive-unification.md`

- [ ] **Step 1: Update current training docs**

Docs must state:

```text
prefix_rollin_et_rmp_ce is compact_full-only
K is sampled uniformly over [0, N]
prefix labels are masked
entry-trie targets are built from remaining objects
support=1 balance=2 is the first balance-first ablation
<|im_end|> is the only EOS in this variant
<|endoftext|> and <|end_of_text|> may be stripped by compatibility parsers but are not training template or target tokens
empirical_unlabeled_poisson_v0 is the first EOS ablation prior
production eos_trust_weight policy requires a versioned calibration artifact
old stage1_set_continuation is superseded and not a current compatibility surface
```

- [ ] **Step 2: Avoid duplicating the implementation checklist**

Training docs should link to this super-power plan rather than copying task-by-task execution details.

- [ ] **Step 3: Run routing consistency scan**

Run:

```bash
rg -n "prefix_rollin_et_rmp_ce|stage1_set_continuation|custom\.stage1_set_continuation|branch_support_weight|branch_balance_weight" docs configs src
```

Expected: active docs should not advertise legacy knobs as current for the new variant. Historical mentions must be explicitly marked `Historical only`, `Superseded`, or `archived artifact interpretation`.

- [ ] **Step 4: Commit this task when implementing**

```bash
git add docs/training/STAGE1_OBJECTIVE.md docs/training/README.md docs/AGENT_INDEX.md
git commit -m "docs(train): document compact prefix-rollin objective"
```

## Task 15: Final Verification Before Research Run

**Files:**

- No planned source edits.

- [ ] **Step 1: Run focused tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_schema.py \
  tests/test_prefix_rollin_sampler.py \
  tests/test_prefix_rollin_dataset_alignment.py \
  tests/test_entry_trie_targets.py \
  tests/test_recursive_loss_support_balance.py \
  tests/test_compact_type_gate.py \
  tests/test_compact_tokenizer_stop_contract.py \
  tests/test_eos_prior.py \
  tests/test_prefix_rollin_diagnostics.py \
  tests/test_prefix_rollin_artifact_contracts.py \
  tests/test_prefix_rollin_config_materialization.py \
  tests/test_prefix_rollin_runtime.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run the narrow latest detection suite**

Run:

```bash
rtk conda run -n ms python -m pytest tests -q
```

Expected: pass or only unrelated pre-existing failures explicitly documented with evidence.

- [ ] **Step 3: Inspect metrics from one tiny smoke batch**

Expected required observations:

```text
rollin/k_histogram contains more than one K value across repeated samples
trie/hard_label_in_positive_rate == 1.0
trie/q_sum_error_max is near 0
type_gate/allowed_mass_mean is finite
eos/eos_trust_weight_mean reflects the ablation EOS trust-weight policy
eos/eos_zero_weight_target_count is present
eos/eos_trust_weight_applied_by_gt_count_bucket keys are present
trie/conditional_kl_q_to_pred_mean is finite when multipositive positions exist
type_gate/positive_not_in_allowed_type_count == 0
```

- [ ] **Step 4: Run static contract scans**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_legacy_surface_absence.py tests/test_prefix_rollin_schema.py -q
rg -n "stage1_set_continuation|set_continuation|setcont/|branch_support_weight|branch_balance_weight|objective\\.support_weight|objective\\.balance_weight|objective\\.trie_support_weight|objective\\.trie_balance_weight" configs docs src tests --glob "!docs/superpowers/**" --glob "!progress/**"
```

Expected: first command has no hits. Second command may show historical/provenance mentions only; active latest-schema/config docs must reject these paths for `prefix_rollin_et_rmp_ce`.

- [ ] **Step 5: Record exact scope before any result claim**

Any training/eval interpretation must include:

```text
config path
checkpoint path
artifact root
sample scope such as tiny, val200, limit=200, or full-val
coordinate_surface
bbox_format
eval_surface
decode settings
parser mode
raw metric file path
dataset_jsonl and image_root
dataset fingerprint or row hash
git commit and command
```

- [ ] **Step 6: Commit final doc/config verification updates when implementing**

```bash
git status --short
git add docs/training/STAGE1_OBJECTIVE.md docs/training/README.md docs/AGENT_INDEX.md docs/ARTIFACTS.md docs/catalog.yaml
git commit -m "chore(train): verify compact prefix-rollin objective"
```

## Execution Handoff

Plan complete. Recommended execution style after user approval:

1. **Subagent-Driven**: dispatch one worker per task group, with disjoint write sets and parent review between tasks.
2. **Inline Execution**: execute tasks in this session using `superpowers:executing-plans`, with checkpoints after schema, sampler, dataset alignment, loss, and runtime routing.

The recommended first implementation checkpoint is Tasks 1-3 only: cleanup plus schema plus sampler. That gives a clean, reviewable contract before touching dataset alignment and loss behavior.
