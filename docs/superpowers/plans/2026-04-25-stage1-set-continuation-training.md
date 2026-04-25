# Stage-1 Set-Continuation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a coord-token-only Stage-1 set-continuation trainer that learns from random object-set prefixes using full-entry multi-positive supervision, structural-close control, replacement-mode PEM, branch-local aux adapters, and reproducible benchmark profiles.

**Architecture:** Add `custom.trainer_variant: stage1_set_continuation` as a true setup-path fork. The new path preserves raw sample metadata, builds indexed canonical object-entry branches from `assistant_payload.objects`, scores candidates with repeated independent forwards, computes coord-aware full-entry MP/PEM losses, and logs variant-specific mechanism metrics and benchmark provenance. Ordinary Stage-1 SFT remains unchanged.

**Tech Stack:** Python, PyTorch, Hugging Face / ms-swift trainer stack, Qwen3-VL chat-template encoding, CoordExp strict config dataclasses, CoordJSON serialization, pytest, YAML config profiles, JSON run artifacts, and `rtk conda run -n ms python -m pytest` verification.

---

## Scope Decision

Keep this as one implementation plan because the feature is a single training paradigm whose pieces are tightly coupled: config, collator metadata, canonical spans, branch encoding, loss math, metrics, artifacts, and benchmark profiles must agree on one object-state contract. Use small files and staged tests to prevent the trainer from becoming a monolith.

Do not implement prefix-cache sharing, branch attention masks, RL, pseudo-labeling, external detectors, raw-text integer coordinates, or packed set-continuation training in this change.

## File Structure

### New files

- Create: `src/trainers/stage1_set_continuation/__init__.py`
  - Public exports for the trainer, config helpers, and small helper types.
- Create: `src/trainers/stage1_set_continuation/trainer.py`
  - `Stage1SetContinuationTrainer`, loss orchestration, branch forward loop, metric emission.
- Create: `src/trainers/stage1_set_continuation/serialization.py`
  - Indexed CoordJSON object-entry renderer and structural-close span extraction.
- Create: `src/trainers/stage1_set_continuation/sampling.py`
  - Subset sampler, candidate sampler, deterministic seeded RNG helpers.
- Create: `src/trainers/stage1_set_continuation/branch_encoder.py`
  - Template-state wrapper, branch tokenization, image/input alignment, label-mask construction.
- Create: `src/trainers/stage1_set_continuation/losses.py`
  - Coord-aware candidate scores, logZ estimators, MP/PEM/structural-close loss helpers.
- Create: `src/trainers/stage1_set_continuation/aux_adapters.py`
  - Branch-local wrappers for coord soft CE/W1, bbox geo, and bbox size aux helpers.
- Create: `src/trainers/stage1_set_continuation/metrics.py`
  - Canonical metric key constants, aggregation helpers, budget and length diagnostics.
- Create: `src/data_collators/stage1_set_continuation_collator.py`
  - Dedicated raw-sample-preserving collator for this trainer variant.
- Create: `configs/stage1/set_continuation/group_a_sft.yaml`
- Create: `configs/stage1/set_continuation/group_b_sft_weak_schema_close.yaml`
- Create: `configs/stage1/set_continuation/group_c_exact_mp.yaml`
- Create: `configs/stage1/set_continuation/group_d_mp_anti_close.yaml`
- Create: `configs/stage1/set_continuation/group_e_pem_replace.yaml`
- Create: `configs/stage1/set_continuation/group_f_leave_one_out.yaml`
- Create: `tests/test_stage1_set_continuation_config.py`
- Create: `tests/test_stage1_set_continuation_cache_policy.py`
- Create: `tests/test_stage1_set_continuation_serialization.py`
- Create: `tests/test_stage1_set_continuation_sampler.py`
- Create: `tests/test_stage1_set_continuation_loss.py`
- Create: `tests/test_stage1_set_continuation_collator.py`
- Create: `tests/test_stage1_set_continuation_trainer_smoke.py`
- Create: `tests/test_stage1_set_continuation_benchmark_profiles.py`

### Existing files to modify

- Modify: `src/config/schema.py`
  - Add strict `custom.stage1_set_continuation` and top-level `benchmark` config parsing, validation, and coord-token-only guards.
- Modify: `src/sft.py`
  - Route trainer variant, set `remove_unused_columns=false`, choose collator path, reject packing before pack-plan construction, enforce encoded-cache policy, and record canonical effective-runtime artifacts.
- Modify: `src/bootstrap/trainer_setup.py`
  - Exclude ordinary one-sequence Stage-1 loss mixins for this variant.
- Modify: `src/bootstrap/experiment_manifest.py`
  - Mirror benchmark group and set-continuation provenance from effective runtime into `experiment_manifest.runtime_summary`.
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
  - Add the set-continuation objective contract.
- Modify: `docs/training/METRICS.md`
  - Add variant-specific MP/PEM/structural-close metric families.
- Modify: `docs/data/PACKING.md`
  - Add v1 packing fail-fast note.
- Modify: `docs/training/README.md`
  - Route to the new objective/metric sections.
- Modify: `docs/IMPLEMENTATION_MAP.md`
  - Add files and tests for the new trainer surface.
- Modify: `tests/test_training_config_strict_unknown_keys.py`
  - Add nested config strictness cases.
- Modify: `tests/test_stage1_static_packing_runtime_config.py`
  - Add set-continuation packing rejection coverage.
- Modify: `tests/test_encoded_sample_cache_runtime_config.py`
  - Add effective-runtime encoded-cache bypass provenance coverage.
- Modify: `tests/test_stage1_metric_key_parity.py`
  - Preserve ordinary Stage-1 parity and add variant-specific expected keys.
- Modify: `tests/test_train_batch_contract.py`
  - Reuse image alignment checks in the trainer smoke or add equivalent set-continuation coverage.

### Existing files to inspect and reuse

- Reuse: `src/utils/assistant_json.py`
  - Canonical CoordJSON rendering.
- Reuse: `src/datasets/dense_caption.py`
  - Existing preservation of `messages`, `assistant_payload`, and metadata.
- Reuse: `src/datasets/builders/jsonlines.py`
  - Source of `assistant_payload.objects`; do not treat `sample["objects"]` as object entries.
- Reuse: `src/trainers/losses/coord_soft_ce_w1.py`
  - Low-level coord aux helper.
- Reuse: `src/trainers/losses/bbox_geo.py`
  - Low-level bbox geometry helper.
- Reuse: `src/trainers/losses/bbox_size_aux.py`
  - Low-level bbox size helper.
- Reuse: `src/trainers/metrics/mixins.py`
  - Batch-contract/image-alignment validation helpers.

Use the package layout above as the implementation target. Do not create a
single-file `src/trainers/stage1_set_continuation.py`; that shape was an early
design sketch and this plan supersedes it with focused modules.

Implement a dedicated typed `benchmark` config section in `src/config/schema.py`
before authoring A-F profiles. Do not place benchmark identity in an untyped
top-level YAML key; strict config parsing must accept and validate
`benchmark.group_id`, `benchmark.control_group_id`,
`benchmark.intended_variable`, and `benchmark.comparability_label`.

---

### Task 1: Strict Config and Governance Baseline

**Files:**
- Modify: `src/config/schema.py`
- Modify: `tests/test_training_config_strict_unknown_keys.py`
- Create: `tests/test_stage1_set_continuation_config.py`

- [ ] **Step 1: Write strict config tests first**

Create `tests/test_stage1_set_continuation_config.py` with focused parser tests:

```python
import pytest

from src.config.schema import PromptOverrides, TrainingConfig


def _base_config() -> dict:
    # Mirror the current minimal strict-config fixture shape used by
    # tests/test_training_config_strict_unknown_keys.py.
    return {
        "template": {"truncation_strategy": "raise"},
        "training": {"packing": False},
        "custom": {
            "train_jsonl": "train.coord.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "coord_tokens": {"enabled": True, "skip_bbox_norm": True},
            "trainer_variant": "stage1_set_continuation",
            "stage1_set_continuation": {
                "subset_sampling": {
                    "empty_prefix_ratio": 0.30,
                    "random_subset_ratio": 0.45,
                    "leave_one_out_ratio": 0.20,
                    "full_prefix_ratio": 0.05,
                    "prefix_order": "random",
                },
                "candidates": {"mode": "exact", "max_candidates": None},
                "structural_close": {
                    "anti_close_weight": 0.0,
                    "final_close_weight": 0.0,
                },
                "positive_evidence_margin": {"mode": "disabled"},
            },
        },
    }


def test_stage1_set_continuation_config_parses() -> None:
    cfg = TrainingConfig.from_mapping(_base_config(), PromptOverrides())

    assert cfg.custom.trainer_variant == "stage1_set_continuation"
    assert cfg.custom.stage1_set_continuation.subset_sampling.prefix_order == "random"
    assert cfg.custom.stage1_set_continuation.positive_evidence_margin.mode == "disabled"


def test_stage1_set_continuation_rejects_unknown_nested_key() -> None:
    data = _base_config()
    data["custom"]["stage1_set_continuation"]["subset_sampling"]["mystery"] = 1

    with pytest.raises(ValueError, match="stage1_set_continuation.subset_sampling.mystery"):
        TrainingConfig.from_mapping(data, PromptOverrides())


def test_stage1_set_continuation_requires_coord_tokens() -> None:
    data = _base_config()
    data["custom"]["coord_tokens"]["enabled"] = False

    with pytest.raises(ValueError, match="stage1_set_continuation.*coord_tokens.enabled"):
        TrainingConfig.from_mapping(data, PromptOverrides())


def test_pem_replace_requires_exactly_one_threshold() -> None:
    data = _base_config()
    data["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "mode": "replace_mp",
        "rho": 0.90,
        "log_rho": -0.1,
    }

    with pytest.raises(ValueError, match="exactly one of.*rho.*log_rho"):
        TrainingConfig.from_mapping(data, PromptOverrides())


def test_benchmark_section_accepts_group_identity() -> None:
    data = _base_config()
    data["benchmark"] = {
        "group_id": "group_c_exact_mp",
        "control_group_id": "group_a_sft",
        "intended_variable": "full-entry MP objective",
        "comparability_label": "accuracy-comparable",
    }

    cfg = TrainingConfig.from_mapping(data, PromptOverrides())
    assert cfg.benchmark.group_id == "group_c_exact_mp"
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_config.py tests/test_training_config_strict_unknown_keys.py -q
```

Expected:

- New tests fail because `stage1_set_continuation` config does not exist yet.
- Existing strict unknown-key tests still pass except where imports expose the missing new config.

- [ ] **Step 3: Add strict config dataclasses**

Implement nested dataclasses with these fields:

```python
@dataclass
class Stage1SetContinuationSubsetSamplingConfig:
    empty_prefix_ratio: float = 0.30
    random_subset_ratio: float = 0.45
    leave_one_out_ratio: float = 0.20
    full_prefix_ratio: float = 0.05
    prefix_order: Literal["random", "dataset", "canonical"] = "random"


@dataclass
class Stage1SetContinuationCandidateConfig:
    mode: Literal["exact", "uniform_subsample"] = "exact"
    max_candidates: int | None = None


@dataclass
class Stage1SetContinuationStructuralCloseConfig:
    anti_close_weight: float = 0.0
    final_close_weight: float = 0.0


@dataclass
class Stage1SetContinuationPEMConfig:
    mode: Literal["disabled", "replace_mp"] = "disabled"
    threshold_space: Literal["full_entry_logZ"] = "full_entry_logZ"
    rho: float | None = None
    log_rho: float | None = None
    threshold_calibration: str | None = None


@dataclass
class Stage1SetContinuationConfig:
    subset_sampling: Stage1SetContinuationSubsetSamplingConfig = field(
        default_factory=Stage1SetContinuationSubsetSamplingConfig
    )
    candidates: Stage1SetContinuationCandidateConfig = field(
        default_factory=Stage1SetContinuationCandidateConfig
    )
    structural_close: Stage1SetContinuationStructuralCloseConfig = field(
        default_factory=Stage1SetContinuationStructuralCloseConfig
    )
    positive_evidence_margin: Stage1SetContinuationPEMConfig = field(
        default_factory=Stage1SetContinuationPEMConfig
    )
    metric_schema_version: str = "stage1_set_continuation_metrics_v1"


@dataclass
class BenchmarkConfig:
    group_id: str | None = None
    control_group_id: str | None = None
    intended_variable: str | None = None
    comparability_label: Literal[
        "accuracy-comparable", "throughput-comparable", "not-comparable"
    ] | None = None
```

Use the repo’s existing strict dataclass parser pattern so unknown keys fail with dotted-path diagnostics.
Add `benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)` to
`TrainingConfig` and parse the top-level `benchmark` section with the same
strict unknown-key behavior as `experiment`.

- [ ] **Step 4: Add validation helpers**

Validation rules:

- subset ratios must be non-negative and sum to `1.0` within small tolerance;
- `prefix_order` must be one of `random`, `dataset`, `canonical`;
- `candidate.mode=uniform_subsample` requires positive `max_candidates`;
- `candidate.mode=exact` ignores or rejects positive `max_candidates`, choose fail-fast for clarity;
- `structural_close.*_weight >= 0`;
- `positive_evidence_margin.mode=replace_mp` requires exactly one of `rho` or `log_rho`;
- `positive_evidence_margin.threshold_space` must be `full_entry_logZ` in v1;
- PEM benchmark configs must set `threshold_calibration` to an authored fixed-ablation note or calibration artifact id;
- `rho` must be in `(0, 1]`;
- if variant is active, `custom.coord_tokens.enabled=true` and `skip_bbox_norm=true`.

- [ ] **Step 5: Run config tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_config.py tests/test_training_config_strict_unknown_keys.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add src/config/schema.py tests/test_stage1_set_continuation_config.py tests/test_training_config_strict_unknown_keys.py
git commit -m "feat(stage1): add set-continuation config contract"
```

---

### Task 2: Setup Routing, Packing Rejection, and Mixin Exclusion

**Files:**
- Modify: `src/sft.py`
- Modify: `src/bootstrap/trainer_setup.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`
- Create: `tests/test_stage1_set_continuation_config.py`

- [ ] **Step 1: Add failing setup-routing tests**

Extend tests to assert:

```python
def test_stage1_set_continuation_rejects_training_packing() -> None:
    data = _base_config()
    data["training"]["packing"] = True

    with pytest.raises(ValueError, match="stage1_set_continuation.*packing"):
        TrainingConfig.from_mapping(data)


def test_stage1_set_continuation_excludes_ordinary_stage1_mixins() -> None:
    from src.bootstrap.trainer_setup import compose_trainer_class
    from src.trainers.metrics.mixins import CoordSoftCEW1LossMixin

    trainer_cls = compose_trainer_class(
        trainer_cls=object,
        trainer_variant="stage1_set_continuation",
        instability_monitor_cfg=None,
        token_type_cfg=None,
        bbox_geo_cfg=None,
        bbox_size_aux_cfg=None,
        coord_soft_ce_w1_cfg=type("Aux", (), {"enabled": True})(),
    )

    assert CoordSoftCEW1LossMixin not in trainer_cls.__mro__
```

- [ ] **Step 2: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_config.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected:

- FAIL for missing routing/mixin exclusion.

- [ ] **Step 3: Implement setup routing**

Implementation requirements:

- Add `elif trainer_variant == "stage1_set_continuation"` in `resolve_trainer_cls`.
- Import `Stage1SetContinuationTrainer` from the new module.
- Update trainer setup to exclude ordinary Stage-1 mixins for this variant.
- Reject `training.packing=true` immediately after packing config parse/resolution and before train/eval static packing wrappers are constructed.
- Ensure eval packing cannot sneak in through default `training.eval_packing=true`.

- [ ] **Step 4: Run focused tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_config.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sft.py src/bootstrap/trainer_setup.py tests/test_stage1_set_continuation_config.py tests/test_stage1_static_packing_runtime_config.py
git commit -m "feat(stage1): route set-continuation setup path"
```

---

### Task 3: Encoded-Cache Policy and Runtime Provenance

**Files:**
- Modify: `src/sft.py`
- Modify: `src/bootstrap/experiment_manifest.py`
- Create: `tests/test_stage1_set_continuation_cache_policy.py`
- Modify: `tests/test_encoded_sample_cache_runtime_config.py`

- [ ] **Step 1: Write encoded-cache policy tests**

Create `tests/test_stage1_set_continuation_cache_policy.py` with cases that assert:

```python
def test_set_continuation_encoded_cache_error_policy_fails_fast() -> None:
    data = _base_config()
    data["training"]["encoded_sample_cache"] = {
        "enabled": True,
        "ineligible_policy": "error",
    }

    with pytest.raises(ValueError, match="stage1_set_continuation.*encoded_sample_cache"):
        TrainingConfig.from_mapping(data, PromptOverrides())


def test_set_continuation_encoded_cache_bypass_records_runtime_reason() -> None:
    runtime = build_set_continuation_effective_runtime_payload(
        encoded_sample_cache_policy="bypass",
        encoded_sample_cache_status="bypassed",
        encoded_sample_cache_reason="stage1_set_continuation_branch_sampling",
    )

    assert runtime["encoded_sample_cache"]["status"] == "bypassed"
    assert "stage1_set_continuation" in runtime["encoded_sample_cache"]["reason"]
```

Use the actual helper names introduced during implementation; the required behavior is fail-fast for `ineligible_policy=error`, bypass for `ineligible_policy=bypass`, and machine-readable provenance in runtime artifacts.

- [ ] **Step 2: Run cache-policy tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_cache_policy.py tests/test_encoded_sample_cache_runtime_config.py -q
```

Expected:

- FAIL because set-continuation cache policy is not implemented yet.

- [ ] **Step 3: Implement cache policy and provenance**

Implementation requirements:

- Treat encoded-sample cache as ineligible for runtime branch continuations in v1.
- If `training.encoded_sample_cache.enabled=true` and `ineligible_policy=error`, fail during setup before dataset cache construction.
- If `ineligible_policy=bypass`, continue uncached and record `status=bypassed`, `policy=bypass`, and reason `stage1_set_continuation_branch_sampling` in `effective_runtime.json`.
- Add set-continuation cache status to `experiment_manifest.runtime_summary` only after it is present in the canonical effective-runtime payload.

- [ ] **Step 4: Run cache-policy tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_cache_policy.py tests/test_encoded_sample_cache_runtime_config.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sft.py src/bootstrap/experiment_manifest.py tests/test_stage1_set_continuation_cache_policy.py tests/test_encoded_sample_cache_runtime_config.py
git commit -m "feat(stage1): guard set-continuation encoded cache"
```

---

### Task 4: Indexed Serialization and Structural-Close Spans

**Files:**
- Create: `src/trainers/stage1_set_continuation/serialization.py`
- Create: `tests/test_stage1_set_continuation_serialization.py`

- [ ] **Step 1: Write fragment and closure-span tests**

Create tests for:

- empty prefix;
- non-empty prefix;
- full prefix;
- duplicate identical entries;
- same description, different bbox;
- `desc_first` and geometry-first object field order;
- first/middle/last candidate;
- structural close-start and full close-sequence spans;
- exclusion of `<|im_end|>`, `<|end_of_text|>`, EOS, and object-entry close from global structural-close targets.

Use fixtures like:

```python
OBJECT_A = {"desc": "person", "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"]}
OBJECT_B = {"desc": "person", "bbox_2d": ["<|coord_50|>", "<|coord_60|>", "<|coord_70|>", "<|coord_80|>"]}


def test_duplicate_identical_entries_have_distinct_spans() -> None:
    rendered = render_indexed_object_list([OBJECT_A, OBJECT_A])

    assert rendered.object_spans[0] != rendered.object_spans[1]
    assert rendered.text[rendered.object_spans[0].start:rendered.object_spans[0].end] == rendered.text[
        rendered.object_spans[1].start:rendered.object_spans[1].end
    ]
```

- [ ] **Step 2: Run serialization tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_serialization.py -q
```

Expected:

- FAIL because serialization helpers do not exist.

- [ ] **Step 3: Implement indexed renderer**

Implementation requirements:

- Reuse `dumps_coordjson` formatting rules.
- Emit text and typed spans by object index.
- Build prefix text for any sampled `S`.
- Build candidate entry text with correct delimiter.
- Build structural close-start token text and full close sequence from canonical serialized form.
- Never search rendered text by content to locate object entries.

- [ ] **Step 4: Run serialization tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_serialization.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation/serialization.py tests/test_stage1_set_continuation_serialization.py
git commit -m "feat(stage1): add indexed set-continuation serialization"
```

---

### Task 5: Deterministic Subset and Candidate Samplers

**Files:**
- Create: `src/trainers/stage1_set_continuation/sampling.py`
- Create: `tests/test_stage1_set_continuation_sampler.py`

- [ ] **Step 1: Write sampler tests**

Required tests:

- configured ratios validate and sum to one;
- deterministic same seed/epoch/sample/rank result;
- different epoch changes sample if intended;
- `|O| = 0` produces no MP candidates;
- `|O| = 1` renormalizes invalid intermediate subset modes;
- leave-one-out returns exactly one remaining candidate;
- full-prefix returns `R = empty`;
- `uniform_subsample` with `K` returns at most `K` candidates;
- `K <= 0` fails fast.

Example:

```python
def test_leave_one_out_has_one_remaining() -> None:
    result = sample_prefix_state(
        object_count=4,
        mode="leave_one_out",
        seed_parts=("seed", 0, "sample-1", 0, 0),
    )

    assert len(result.prefix_indices) == 3
    assert len(result.remaining_indices) == 1
    assert set(result.prefix_indices).isdisjoint(result.remaining_indices)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_sampler.py -q
```

Expected:

- FAIL because sampler helpers do not exist.

- [ ] **Step 3: Implement deterministic samplers**

Implementation requirements:

- Seed from resolved seed, epoch, sample id or base idx, rank, and microstep salt.
- Renormalize invalid mode mixtures deterministically.
- Return selected mode, configured mixture, resolved valid mixture, prefix indices, remaining indices, and candidate indices.
- Log or expose candidate scoring mode and scored-candidate fraction.

- [ ] **Step 4: Run sampler tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_sampler.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation/sampling.py tests/test_stage1_set_continuation_sampler.py
git commit -m "feat(stage1): add deterministic set-continuation sampling"
```

---

### Task 6: Raw-Sample Collator and Branch Encoder

**Files:**
- Create: `src/data_collators/stage1_set_continuation_collator.py`
- Create: `src/trainers/stage1_set_continuation/branch_encoder.py`
- Create: `tests/test_stage1_set_continuation_collator.py`

- [ ] **Step 1: Write metadata preservation tests**

Test requirements:

- collator preserves `assistant_payload.objects`;
- collator preserves `messages` or equivalent image/prompt identity;
- collator preserves `metadata`, `sample_id`, `base_idx`, dataset label;
- non-model extras are consumed by trainer or popped before `model(**inputs)`;
- `sample["objects"]` is not used as the object-entry list.

Example:

```python
def test_collator_preserves_assistant_payload_objects() -> None:
    batch = build_stage1_set_continuation_collator()(
        [
            {
                "input_ids": [1, 2],
                "labels": [-100, 2],
                "assistant_payload": {"objects": [{"desc": "cat", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]}]},
                "objects": [{"ref": "metadata-not-entry"}],
                "messages": [{"role": "user", "content": "detect"}],
                "metadata": {"image_id": 1},
                "sample_id": "sample-1",
                "base_idx": 0,
            }
        ]
    )

    assert batch["set_continuation_meta"][0]["assistant_payload"]["objects"][0]["desc"] == "cat"
    assert batch["set_continuation_meta"][0]["objects"][0]["ref"] == "metadata-not-entry"
```

- [ ] **Step 2: Write branch encoder smoke tests**

Test that the branch encoder:

- accepts one metadata item and selected prefix/candidate indices;
- returns branch tensors, labels, candidate-entry label mask, coord-position mask, close-start span, close-sequence span;
- rejects missing image/prompt identity with actionable diagnostics;
- validates effective branch rendering contains `<|coord_*|>`.

- [ ] **Step 3: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_collator.py -q
```

Expected:

- New tests fail because collator/encoder do not exist.

- [ ] **Step 4: Implement collator and branch encoder**

Implementation requirements:

- Implement a dedicated identity-style or hybrid raw-row collator; do not route
  this trainer through the ordinary Stage-1 batch-extras wrapper.
- Branch encoder owns template-state restoration.
- Branch encoder disables packing/padding-free assumptions.
- Branch encoder validates image-token/grid alignment using existing helpers where possible.
- Branch encoder exposes label masks for candidate entry, non-coord labels, coord-token labels, and structural-close spans.

- [ ] **Step 5: Run tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_collator.py tests/test_train_batch_contract.py -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data_collators/stage1_set_continuation_collator.py src/trainers/stage1_set_continuation/branch_encoder.py tests/test_stage1_set_continuation_collator.py tests/test_train_batch_contract.py
git commit -m "feat(stage1): preserve set-continuation branch metadata"
```

---

### Task 7: Loss Math and PEM Replacement Mode

**Files:**
- Create: `src/trainers/stage1_set_continuation/losses.py`
- Create: `tests/test_stage1_set_continuation_loss.py`

- [ ] **Step 1: Write synthetic-logit loss tests**

Required tests:

- coord-aware candidate score uses full-vocab non-coord and coord-vocab coord slots;
- MP exact logZ equals `-logsumexp(all_remaining_scores)`;
- sampled raw logZ omits the constant but logs estimator scope;
- uniform-importance estimated logZ adds `log(|R| / |C|)`;
- PEM replacement mode does not add MP to total;
- PEM requires exactly one threshold;
- anti-close uses close-start probability;
- weak schema close uses full teacher-forced close-sequence logprob;
- `R = empty` with zero close weight contributes no objective denominator.
- one-candidate responsibility entropy is `0.0`, one-candidate std metrics are
  `0.0`, and responsibility-vs-length correlation is skipped unless at least
  two candidates are scored with non-constant lengths.

Example:

```python
def test_pem_replace_does_not_add_mp_loss() -> None:
    scores = torch.tensor([-2.0, -3.0])
    log_z = torch.logsumexp(scores, dim=0)
    result = compute_mp_pem_losses(
        scores=scores,
        pem_mode="replace_mp",
        log_rho=torch.tensor(-1.0),
    )

    assert torch.allclose(result.loss_pem, torch.clamp(torch.tensor(-1.0) - log_z, min=0.0))
    assert torch.allclose(result.total_objective, result.loss_pem)
    assert torch.allclose(result.loss_mp_diagnostic, -log_z)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_loss.py -q
```

Expected:

- FAIL because loss helpers do not exist.

- [ ] **Step 3: Implement loss helpers**

Implementation requirements:

- Use numerically stable `torch.logsumexp`.
- Compute coord-token slot probabilities over coord-vocab ids only.
- Emit explicit logZ scope fields.
- Emit candidate length diagnostics, coord/non-coord score diagnostics, and
  valid-length-correlation counters with deterministic small-n behavior.
- Support PEM `disabled` and `replace_mp`.
- Use close-start and close-sequence helper outputs separately.
- Return a structured loss object with total loss, component losses, diagnostics, denominators, and metric payload.

- [ ] **Step 4: Run loss tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_loss.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation/losses.py tests/test_stage1_set_continuation_loss.py
git commit -m "feat(stage1): add set-continuation objective losses"
```

---

### Task 8: Branch-Local Auxiliary Adapters

**Files:**
- Create: `src/trainers/stage1_set_continuation/aux_adapters.py`
- Modify: `tests/test_stage1_set_continuation_loss.py`

- [ ] **Step 1: Write adapter tests**

Required tests:

- enabled coord aux without adapter fails fast before training;
- coord aux adapter calls canonical helper with branch-local coord masks;
- bbox geo adapter fails fast without decoded bbox state;
- bbox size aux depends on bbox geo decoded state;
- aux aggregation is uniform over scored valid candidates;
- skipped candidates are counted and logged;
- responsibility-weighted aux is rejected or unavailable in v1.

- [ ] **Step 2: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_loss.py -q
```

Expected:

- FAIL for missing adapter helpers.

- [ ] **Step 3: Implement adapters**

Implementation requirements:

- Do not import or compose ordinary one-sequence loss mixins.
- Reuse low-level helper functions from `src/trainers/losses/`.
- Accept branch logits, labels, coord positions, decoded bbox state, candidate masks, and config.
- Return mean-like aux atoms and counters.

- [ ] **Step 4: Run adapter tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_loss.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation/aux_adapters.py tests/test_stage1_set_continuation_loss.py
git commit -m "feat(stage1): add branch-local set-continuation aux adapters"
```

---

### Task 9: Trainer Integration and Smoke

**Files:**
- Create: `src/trainers/stage1_set_continuation/trainer.py`
- Create: `src/trainers/stage1_set_continuation/__init__.py`
- Create: `src/trainers/stage1_set_continuation/metrics.py`
- Modify: `src/sft.py`
- Create: `tests/test_stage1_set_continuation_trainer_smoke.py`
- Modify: `tests/test_stage1_metric_key_parity.py`

- [ ] **Step 1: Write tiny trainer smoke**

Smoke requirements:

- one image fixture or mocked image tensors;
- at least two objects;
- no packing;
- raw metadata survives collation;
- exact mode scores two branches independently from same prefix;
- `loss/mp` finite in MP mode;
- PEM replacement mode emits `loss/pem` and `loss/mp_diagnostic`;
- anti-close metrics emit when `R != empty`;
- weak schema-close metrics emit when `R = empty`;
- ordinary Stage-1 does not emit MP keys.

- [ ] **Step 2: Run smoke to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_trainer_smoke.py tests/test_stage1_metric_key_parity.py -q
```

Expected:

- New smoke fails because trainer integration does not exist.

- [ ] **Step 3: Implement trainer**

Implementation requirements:

- Pop/consume set-continuation metadata before model forward.
- Sample one prefix state per raw sample.
- Build candidate branches through branch encoder.
- Run repeated independent forwards.
- Aggregate candidate scores and losses.
- Normalize objective denominators correctly for full-prefix metric-only samples.
- Emit metrics with canonical names and aggregation-safe values.
- Preserve learning-rate/grad-accum metrics if the existing mixin is excluded.

- [ ] **Step 4: Run trainer smoke and parity tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_trainer_smoke.py tests/test_stage1_metric_key_parity.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation src/sft.py tests/test_stage1_set_continuation_trainer_smoke.py tests/test_stage1_metric_key_parity.py
git commit -m "feat(stage1): integrate set-continuation trainer"
```

---

### Task 10: Artifacts, Benchmark Matrix, and Config Profiles

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/bootstrap/experiment_manifest.py`
- Modify: `src/sft.py`
- Create: `configs/stage1/set_continuation/group_a_sft.yaml`
- Create: `configs/stage1/set_continuation/group_b_sft_weak_schema_close.yaml`
- Create: `configs/stage1/set_continuation/group_c_exact_mp.yaml`
- Create: `configs/stage1/set_continuation/group_d_mp_anti_close.yaml`
- Create: `configs/stage1/set_continuation/group_e_pem_replace.yaml`
- Create: `configs/stage1/set_continuation/group_f_leave_one_out.yaml`
- Create: `tests/test_stage1_set_continuation_benchmark_profiles.py`

- [ ] **Step 1: Write benchmark/profile tests**

Test requirements:

- A-F configs resolve;
- every config has stable `benchmark.group_id`;
- every config has `benchmark.control_group_id` where applicable;
- intended variable is declared;
- every canonical A-F config uses `training.packing: false`;
- every config resolves dataset JSONL, prompt variant, object field order, seed,
  resolution/preset, effective batch/sample budget, optimizer-step budget,
  checkpoint/base/adapter identity, inference decoding controls, and eval plan;
- every config pins coord-token settings, effective coord-slot scoring surface,
  `coord_soft_ce_w1`, `bbox_geo`, and `bbox_size_aux`;
- Group E uses `positive_evidence_margin.mode=replace_mp`;
- Group E records `threshold_space=full_entry_logZ` and threshold calibration provenance;
- benchmark matrix diff only includes approved fields;
- smoke artifact manifest includes group id, comparator, intended variable,
  comparability label, metric scope, eval view, branch semantics, realized
  branch/token budget, realized prefix-mode coverage, realized aux/coord
  settings, and set-continuation metric schema version.

- [ ] **Step 2: Run tests to verify failure**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_benchmark_profiles.py -q
```

Expected:

- FAIL because benchmark configs/artifact fields do not exist.

- [ ] **Step 3: Add configs and artifact fields**

Implementation requirements:

- Group A: ordinary SFT baseline.
- Group B: ordinary SFT with structural schema-close masked/downweighted.
- Group C: exact one-prefix MP.
- Group D: MP plus anti-close-start.
- Group E: PEM replacement mode with explicit `rho` or `log_rho`.
- Group F: leave-one-out emphasis.
- These A-F labels are local to this OpenSpec and supersede intermediate letter
  labels used in `progress/directions/full_idea_v5.md`.
- Canonical A-F profiles all set `training.packing: false`; any packed ordinary-SFT control uses a separate group id and is not part of the accuracy-comparable A-F matrix.
- Artifact fields include `benchmark_group_id`, `control_group_id`, `intended_variable`, `comparability_label`, `candidate_scoring_mode`, `logZ_estimator`, `prefix_attach_mode`, `branch_isolation`, `prefix_gradient`, `metric_schema_version`, `eval_scope`, `eval_view`, `realized_branch_token_budget`, `realized_prefix_mode_coverage`, `realized_aux_settings`, `effective_coord_slot_scoring`, and PEM threshold calibration provenance where applicable.

- [ ] **Step 4: Run benchmark tests**

```bash
rtk conda run -n ms python -m pytest tests/test_stage1_set_continuation_benchmark_profiles.py -q
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config/schema.py src/bootstrap/experiment_manifest.py src/sft.py configs/stage1/set_continuation tests/test_stage1_set_continuation_benchmark_profiles.py
git commit -m "feat(stage1): add set-continuation benchmark profiles"
```

---

### Task 11: Documentation and Final Verification

**Files:**
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/data/PACKING.md`
- Modify: `docs/training/README.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`

- [ ] **Step 1: Update canonical docs**

Required doc phrases:

- `custom.trainer_variant: stage1_set_continuation`
- `assistant_payload.objects`
- `loss/mp`
- `loss/pem`
- `loss/anti_close_start`
- `loss/weak_schema_close`
- `mp/logZ_scored_raw`
- `mp/logZ_remaining_exact`
- `mp/logZ_remaining_est`
- `stop/p_close_start_when_remaining_exists`
- `stop/p_continue_start_when_remaining_exists`
- `training.packing=true` fail-fast note
- coord-token-only support note
- repeated independent forward branch semantics
- benchmark Groups A-F

- [ ] **Step 2: Run docs grep checks**

```bash
rg -n "stage1_set_continuation|loss/mp|loss/pem|anti_close|weak_schema_close|assistant_payload.objects|stop/p_continue_start_when_remaining_exists" docs/training/STAGE1_OBJECTIVE.md docs/training/METRICS.md docs/data/PACKING.md docs/training/README.md docs/IMPLEMENTATION_MAP.md
```

Expected:

- All required docs contain the new route/contract phrases.

- [ ] **Step 3: Run full focused test set**

```bash
rtk conda run -n ms python -m pytest \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_train_batch_contract.py \
  tests/test_stage1_metric_key_parity.py \
  tests/test_stage1_set_continuation_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  tests/test_stage1_set_continuation_serialization.py \
  tests/test_stage1_set_continuation_sampler.py \
  tests/test_stage1_set_continuation_loss.py \
  tests/test_stage1_set_continuation_collator.py \
  tests/test_stage1_set_continuation_trainer_smoke.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  -q
```

Expected:

- PASS.

- [ ] **Step 4: Run static checks on touched Python paths**

Run the repo's fail-fast Python syntax check on the new implementation files:

```bash
rtk conda run -n ms python -m py_compile \
  src/trainers/stage1_set_continuation/*.py \
  src/data_collators/stage1_set_continuation_collator.py
```

Expected:

- PASS.

- [ ] **Step 5: Run OpenSpec validation when available**

```bash
openspec validate add-stage1-set-continuation-training --strict
```

Expected:

- PASS. If `openspec` is not on `PATH`, record that validation was not run and include the exact shell error.

- [ ] **Step 6: Commit docs and final verification updates**

```bash
git add docs/training/STAGE1_OBJECTIVE.md docs/training/METRICS.md docs/data/PACKING.md docs/training/README.md docs/IMPLEMENTATION_MAP.md
git commit -m "docs(stage1): document set-continuation training"
```

---

## Implementation Review Gates

- After Task 3, request review of serialization/span boundaries before trainer integration.
- After Task 6, request review of objective semantics before writing the trainer.
- After Task 8, request review of trainer integration before adding benchmark profiles.
- Before final merge, run the full focused test command and summarize any skipped checks.

## Known Non-Goals For V1

- No prefix-cache sharing.
- No branch attention masks.
- No raw-text integer coordinate mode.
- No packed set-continuation training.
- No RL, pseudo-labeling, external detector, or architecture change.
- No responsibility-weighted auxiliary losses.
- No chat/EOS stop loss under the set-continuation trainer.
