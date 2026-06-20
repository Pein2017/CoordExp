# Prefix-Denoising SFT V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build prefix-denoising SFT V1 as a Stage-1 compact detection extension with paired clean/noisy hard-CE supervision, optional sparse clean-to-noisy coord KL, and an explicit hybrid-packed runtime path.

**Architecture:** Add a typed `prefix_denoising` config surface on the current canonical `objective.id: teacher_forcing` route with `objective.profile: hard_sft` as the CE base, then route enabled configs into a dedicated hybrid dataset, model-ready collator path, trainer loss mixin, metrics, and hybrid packing contract. Keep bbox math in `src/datasets/geometry.py`, keep prefix-denoising types and builders under `src/detection/prefix_denoising/`, and preserve all existing non-V1 teacher-forcing and recursive packing guardrails.

**Tech Stack:** Python dataclasses, PyTorch, ms-swift trainer/collator integration, existing CoordExp `DetectionTrainingConfig`, compact-full templates, typed `MetricEvent`, pytest, YAML configs.

---

Date: 2026-06-14

Design spec: `docs/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`

Direction note: `progress/directions/prefix_denoising_sft_v1.md`

Status: historical implementation plan for the implemented V1 slice on
`codex/prefix-denoising-sft`. The plan passed into execution after review
approval. The original 2026-06-14 smoke is superseded; see
`progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
for the current post-repair tiny launch-health evidence.

## Review-Convergence Scope

| Field | Value |
|---|---|
| Mode | docs/spec/plan |
| Mutation level for this document | docs-only |
| Implementation mutation allowed by this plan | code/config/docs allowed after review approval |
| Source of truth | approved design spec, direction note, current code in this worktree |
| Stop condition | implementation landed, tiny launch-health evidence recorded, and final convergence review pending |
| Approval gates | historical: user review, review-convergence triage, and explicit implementation approval were completed before execution |

No review subagents were launched while writing this plan because the user said they will launch additional reviewers after the docs are complete.

## Hard Guardrails

| Guardrail | Requirement |
|---|---|
| CE objective | Hard clean-label CE only. No valid-set marginal CE, multi-positive trie CE, coordinate SoftCE, or previous recursive objective features. |
| Config base | Prefix-denoising config leaves use current canonical `objective.id: teacher_forcing` with `objective.profile: hard_sft`. Do not revive legacy `objective.id: sft` or `variant: sorted_sft`. |
| Object order | `data.object_ordering: sorted` is mandatory when `prefix_denoising.enabled: true`. |
| Branches | Exactly two multimodal segments per hybrid sample: `clean_full` and `noisy_full`. |
| KL | KL is optional by weight, asymmetric `stopgrad(clean_full) -> noisy_full`, coord-local, GT-windowed, and sparse over selected object sites. |
| Packing | Do not relax existing sidecar packing guards globally. Add a narrow prefix-denoising hybrid-packed path with explicit boundary metadata. |
| Geometry | Route bbox noising through `src/datasets/geometry.py`. Do not drop, reorder, clamp-repair, or silently resize geometry. |
| Cache | Reject or disable encoded-sample cache for V1. Static packing cache may store the pack plan only. |
| Model | Do not edit upstream HF/Qwen model files. |
| Decode | Keep free decode and logits constraints unchanged. Rollout/free-decode eval is out of V1 implementation scope. |

## Planned File Map

| Path | Role |
|---|---|
| `src/config/schema.py` | Add strict `PrefixDenoisingConfig` dataclasses, top-level schema acceptance, disabled semantics, and V1 contract validation. |
| `src/datasets/geometry.py` | Add constructive norm1000 integer bbox noiser and structured result types. |
| `src/detection/prefix_denoising/__init__.py` | Public exports for V1 prefix-denoising components. |
| `src/detection/prefix_denoising/types.py` | `PrefixDenoisingSegment`, `HybridPrefixDenoisingSample`, `PrefixDenoisingKLSite`, noising/skip metadata, packed boundary dataclasses. |
| `src/detection/prefix_denoising/builder.py` | Build sorted clean/noisy full segments and optional KL-site metadata from one base detection row. |
| `src/detection/prefix_denoising/dataset.py` | Dataset wrapper that emits one `HybridPrefixDenoisingSample` per base row and exposes epoch-invariant lengths. |
| `src/detection/prefix_denoising/loss.py` | Hard CE, local-window KL, token accuracy, and numerics helpers. |
| `src/detection/prefix_denoising/metrics.py` | Prefix-denoising `MetricEvent` constructors and required metric key list. |
| `src/detection/prefix_denoising/packing.py` | Atomic hybrid packing planner, `PackedHybridBoundaryMap`, and offset rewrite helpers. |
| `src/detection/dataset.py` | Register prefix-denoising sidecars in the canonical fail-fast model-input stripping boundary. |
| `src/detection/runtime.py` | Route enabled prefix-denoising configs to the hybrid dataset and runtime support policy. |
| `src/detection/packing.py` | Add prefix-denoising fingerprint fields and narrow eligibility metadata if needed by static packing helpers. |
| `src/data_collators/enrichers.py` | Attach prefix-denoising hybrid sidecars and reject incompatible packed sidecars outside the V1 path. |
| `src/data_collators/batch_extras_collator.py` | Register the prefix-denoising sidecar enricher and call the packed boundary-map producer immediately after the template collator flattens a packed batch. |
| `src/trainers/metrics/prefix_denoising.py` | Trainer mixin that owns model forward, hard CE, KL, metric buffering, and standard monitors for V1. |
| `src/trainers/metrics/mixins.py` | Re-export `PrefixDenoisingObjectiveMixin`. |
| `src/bootstrap/trainer_setup.py` | Compose `PrefixDenoisingObjectiveMixin` through the existing trainer composition owner. |
| `src/sft.py` | Parse runtime config, route datasets/collators/trainers, add runtime payload, add static packing fingerprint fields. |
| `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml` | Production-intent packed CE-only prefix-denoising ablation. |
| `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml` | Production-intent packed CE+KL prefix-denoising run. |
| `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml` | Tiny packed CE-only launch-health config. |
| `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml` | Tiny packed CE+KL launch-health config. |
| `configs/stage1/detection_teacher_forcing/README.md` | Route note for prefix-denoising config leaves and launch-health ladder. |
| `docs/training/METRICS.md` | Register prefix-denoising metric families after implementation. |
| `tests/test_prefix_denoising_config_contract.py` | Schema/runtime contract tests. |
| `tests/test_prefix_denoising_geometry.py` | Constructive noiser tests. |
| `tests/test_prefix_denoising_builder.py` | Hybrid builder and dataset tests. |
| `tests/test_prefix_denoising_collator.py` | Prefix-denoising sidecar enricher, packed boundary producer, and model-input boundary tests. |
| `tests/test_prefix_denoising_loss.py` | Hard CE and KL numerical tests. |
| `tests/test_prefix_denoising_metrics.py` | Metric key and denominator tests. |
| `tests/test_prefix_denoising_packing.py` | Hybrid pack plan and boundary-map tests. |
| `tests/test_prefix_denoising_runtime_integration.py` | Dataset-to-collator-to-dummy-loss contract test for model-ready hybrid batches. |
| `tests/test_stage1_static_packing_runtime_config.py` | Preserve existing non-V1 packing guard tests and add V1 fingerprint checks. |

## Task 0: Implementation Preflight

**Files:**
- Read: `docs/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
- Read: `progress/directions/prefix_denoising_sft_v1.md`
- Read: this plan
- Inspect: `src/config/schema.py`
- Inspect: `src/detection/runtime.py`
- Inspect: `src/detection/dataset.py`
- Inspect: `src/sft.py`
- Inspect: `src/data_collators/enrichers.py`
- Inspect: `src/trainers/metrics/teacher_forcing.py`

- [ ] **Step 1: Confirm branch and dirty state**

Run:

```bash
git -C /data/CoordExp/.worktrees/geometry-aware-denoising-sft branch --show-current
git -C /data/CoordExp/.worktrees/geometry-aware-denoising-sft status --short
```

Expected:

```text
codex/prefix-denoising-sft
```

Expected working state before implementation is clean on this branch. If new dirty files exist, identify whether they belong to prefix-denoising implementation before staging; do not revert or stage unrelated user work.

- [ ] **Step 2: Confirm the approved hard-CE decision is present**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path("docs/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md").read_text()
required = [
    "hard clean-label SFT CE only",
    "No valid-set marginal CE",
    "Do not activate marginal, multi-positive, trie, SoftCE",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"missing approved design statements: {missing}")
print("approved hard-CE design statements: ok")
PY
```

Expected:

```text
approved hard-CE design statements: ok
```

- [ ] **Step 3: Create an implementation branch checkpoint**

If no new implementation work has started, create a checkpoint commit only for current docs that are intended for this branch. Leave unrelated dirty files alone.

Run:

```bash
git -C /data/CoordExp/.worktrees/geometry-aware-denoising-sft status --short
```

Expected: the implementer can identify exactly which files belong to the prefix-denoising docs and which files are unrelated local work.

## Task 1: Config Schema And Runtime Contract

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/sft.py`
- Modify: `openspec/changes/<prefix-denoising-change>/` or record an explicit user-approved experiment-only decision before code
- Create: `tests/test_prefix_denoising_config_contract.py`
- Modify: `tests/test_detection_training_config_contract.py` only if shared helpers are needed

- [ ] **Step 0: Add the governance gate**

Before changing schema/runtime code, create an OpenSpec change for the compatibility-sensitive parts of V1 unless the user explicitly decides this remains branch-local experiment-only code. The OpenSpec change must cover:

- the `prefix_denoising` config schema and strict unknown-key behavior;
- hard-CE-only loss semantics and the optional sparse KL term;
- required metric keys including `llm_loss`, top-1/top-5, raw/weighted KL, and skip counters;
- encoded-sample cache ineligibility and static hybrid packing eligibility;
- the statement that rollout/free-decode evaluation remains outside V1 implementation.

Verification for this gate:

```bash
rg -n "prefix_denoising|prefix-denoising" openspec/specs openspec/changes docs/superpowers progress/directions
openspec validate <change-id> --strict
```

If the OpenSpec CLI is unavailable, record the command and blocker in the implementation summary. Do not proceed to implementation without either a validated OpenSpec change or an explicit user decision that V1 is experiment-only and not a stable contract.

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_prefix_denoising_config_contract.py`:

```python
from __future__ import annotations

import copy

import pytest

from src.config.schema import DetectionTrainingConfig


def _base_prefix_payload() -> dict[str, object]:
    return {
        "model": {
            "model": "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp",
            "model_type": "qwen3_vl",
        },
        "template": {
            "template": "qwen3_vl",
            "truncation_strategy": "raise",
            "max_length": 12000,
            "max_pixels": 1048576,
        },
        "training": {
            "run_name": "prefix-denoising-test",
            "num_train_epochs": 1,
            "packing": True,
            "eval_packing": False,
            "encoded_sample_cache": {"enabled": False},
        },
        "data": {
            "train_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl",
            "val_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl",
            "image_root": "public_data/coco/rescale_32_1024_bbox_max60",
            "object_ordering": "sorted",
        },
        "prompt": {
            "system_variant": "stage1_detection",
            "user_variant": "compact_detection",
            "include_template_summary": True,
            "prompt_variant_enabled": True,
        },
        "detection_template": {
            "id": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "strict_parse": True,
        },
        "token_rows": {
            "enabled": True,
            "tie_head": True,
            "groups": {
                "coord_geometry": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_999|>",
                    "expected_start": 151670,
                    "expected_end": 152669,
                },
                "compact_structure": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>", "<|box_start|>"],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|box_start|>": 151648,
                    },
                },
            },
            "embed_lr": 5.0e-5,
            "weight_decay": 0.0,
        },
        "objective": {
            "id": "teacher_forcing",
            "profile": "hard_sft",
            "modules": {
                "token_type_mass": {"enabled": False},
                "conditional_valid_set_likelihood": {"enabled": False},
                "within_valid_coverage": {"enabled": False, "coverage_strength": 0.0},
                "continuation_margin": {"enabled": False},
            },
        },
        "prefix_denoising": {
            "enabled": True,
            "noise": {
                "center_shift_frac": 0.08,
                "uniform_scale_range": [0.92, 1.08],
            },
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        },
        "packing": {
            "static_packing": True,
        },
        "evaluation": {
            "expected_template": "compact_full",
            "parser_mode": "strict_expected",
        },
        "validation": {
            "validate_span_alignment": True,
            "validate_template_capabilities": True,
            "fail_fast": True,
        },
    }


def _payload_with(**updates: object) -> dict[str, object]:
    payload = copy.deepcopy(_base_prefix_payload())
    for key, value in updates.items():
        payload[key] = value
    return payload


def test_prefix_denoising_schema_accepts_teacher_forcing_hard_sft() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_base_prefix_payload())

    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.data.object_ordering == "sorted"
    assert cfg.prefix_denoising.enabled is True
    assert cfg.prefix_denoising.noise.center_shift_frac == pytest.approx(0.08)
    assert cfg.prefix_denoising.current_object_kl.weight == pytest.approx(0.05)
    assert cfg.prefix_denoising.current_object_kl.window_radius == 8
    assert cfg.prefix_denoising.current_object_kl.num_objects_per_image == 1


def test_prefix_denoising_omitted_defaults_to_disabled() -> None:
    payload = _base_prefix_payload()
    payload.pop("prefix_denoising")

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.prefix_denoising.enabled is False


def test_prefix_denoising_weight_zero_is_ce_only() -> None:
    payload = _base_prefix_payload()
    prefix = dict(payload["prefix_denoising"])  # type: ignore[index]
    current_object_kl = dict(prefix["current_object_kl"])  # type: ignore[index]
    current_object_kl["weight"] = 0.0
    prefix["current_object_kl"] = current_object_kl
    payload["prefix_denoising"] = prefix

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.prefix_denoising.enabled is True
    assert cfg.prefix_denoising.current_object_kl.weight == pytest.approx(0.0)


def test_prefix_denoising_rejects_random_ordering() -> None:
    payload = _base_prefix_payload()
    data = dict(payload["data"])  # type: ignore[index]
    data["object_ordering"] = "random_permutation"
    payload["data"] = data

    with pytest.raises(ValueError, match=r"prefix_denoising.*data\.object_ordering.*sorted"):
        DetectionTrainingConfig.from_mapping(payload)


def test_prefix_denoising_rejects_valid_set_marginal_teacher_forcing() -> None:
    payload = _base_prefix_payload()
    payload["objective"] = {
        "id": "teacher_forcing",
        "profile": "pure_valid_set_marginal",
    }

    with pytest.raises(ValueError, match=r"prefix_denoising.*hard clean-label CE"):
        DetectionTrainingConfig.from_mapping(payload)


def test_prefix_denoising_rejects_explicit_target_ir_rollin_policy() -> None:
    payload = _base_prefix_payload()
    objective = dict(payload["objective"])  # type: ignore[index]
    objective["target_ir"] = {
        "rollin_policy": {"name": "random_permutation", "base_seed": 17},
        "exact_packing_mapping": {"enabled": False},
    }
    payload["objective"] = objective

    with pytest.raises(ValueError, match=r"prefix_denoising.*target_ir\.rollin_policy"):
        DetectionTrainingConfig.from_mapping(payload)


def test_prefix_denoising_rejects_coord_soft_ce() -> None:
    payload = _base_prefix_payload()
    objective = dict(payload["objective"])  # type: ignore[index]
    objective["coord_soft_ce"] = {"enabled": True}
    payload["objective"] = objective

    with pytest.raises(ValueError, match=r"prefix_denoising.*SoftCE"):
        DetectionTrainingConfig.from_mapping(payload)


def test_prefix_denoising_rejects_encoded_sample_cache() -> None:
    payload = _base_prefix_payload()
    training = dict(payload["training"])  # type: ignore[index]
    training["encoded_sample_cache"] = {"enabled": True}
    payload["training"] = training

    with pytest.raises(ValueError, match=r"prefix_denoising.*encoded_sample_cache"):
        DetectionTrainingConfig.from_mapping(payload)


def test_prefix_denoising_rejects_unknown_nested_key() -> None:
    payload = _base_prefix_payload()
    prefix = dict(payload["prefix_denoising"])  # type: ignore[index]
    prefix["sampler"] = {"kind": "cyclic"}
    payload["prefix_denoising"] = prefix

    with pytest.raises(ValueError, match=r"Unknown prefix_denoising keys"):
        DetectionTrainingConfig.from_mapping(payload)
```

- [ ] **Step 2: Run the tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_config_contract.py -q
```

Expected: FAIL because `DetectionTrainingConfig` does not yet accept the `prefix_denoising` top-level section.

- [ ] **Step 3: Add strict prefix-denoising dataclasses**

In `src/config/schema.py`, add dataclasses near `DetectionPackingConfig`:

```python
@dataclass(frozen=True)
class PrefixDenoisingNoiseConfig:
    center_shift_frac: float = 0.08
    uniform_scale_range: tuple[float, float] = (0.92, 1.08)

    def __post_init__(self) -> None:
        value = _detection_validate_nonnegative_finite_float(
            self.center_shift_frac,
            path="prefix_denoising.noise.center_shift_frac",
        )
        object.__setattr__(self, "center_shift_frac", value)
        raw_range = self.uniform_scale_range
        if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)):
            raise TypeError("prefix_denoising.noise.uniform_scale_range must be a two-number sequence")
        if len(raw_range) != 2:
            raise ValueError("prefix_denoising.noise.uniform_scale_range must contain exactly two numbers")
        low = _detection_validate_positive_finite_float(
            raw_range[0],
            path="prefix_denoising.noise.uniform_scale_range[0]",
        )
        high = _detection_validate_positive_finite_float(
            raw_range[1],
            path="prefix_denoising.noise.uniform_scale_range[1]",
        )
        if high < low:
            raise ValueError("prefix_denoising.noise.uniform_scale_range must be ordered [low, high]")
        object.__setattr__(self, "uniform_scale_range", (low, high))

    @classmethod
    def from_mapping(cls, payload: Any) -> "PrefixDenoisingNoiseConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("prefix_denoising.noise must be a mapping")
        return parse_dataclass_strict(cls, payload, path="prefix_denoising.noise")


@dataclass(frozen=True)
class PrefixDenoisingCurrentObjectKLConfig:
    weight: float = 0.05
    window_radius: int = 8
    num_objects_per_image: int = 1

    def __post_init__(self) -> None:
        weight = _detection_validate_nonnegative_finite_float(
            self.weight,
            path="prefix_denoising.current_object_kl.weight",
        )
        object.__setattr__(self, "weight", weight)
        for field_name in ("window_radius", "num_objects_per_image"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"prefix_denoising.current_object_kl.{field_name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"prefix_denoising.current_object_kl.{field_name} must be > 0")

    @classmethod
    def from_mapping(cls, payload: Any) -> "PrefixDenoisingCurrentObjectKLConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("prefix_denoising.current_object_kl must be a mapping")
        return parse_dataclass_strict(
            cls,
            payload,
            path="prefix_denoising.current_object_kl",
        )


@dataclass(frozen=True)
class PrefixDenoisingConfig:
    enabled: bool = False
    noise: PrefixDenoisingNoiseConfig = field(default_factory=PrefixDenoisingNoiseConfig)
    current_object_kl: PrefixDenoisingCurrentObjectKLConfig = field(
        default_factory=PrefixDenoisingCurrentObjectKLConfig
    )

    def __post_init__(self) -> None:
        _detection_validate_bool(self.enabled, path="prefix_denoising.enabled")

    @classmethod
    def from_mapping(cls, payload: Any) -> "PrefixDenoisingConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("prefix_denoising must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        noise = PrefixDenoisingNoiseConfig.from_mapping(data.pop("noise", {}))
        current_object_kl = PrefixDenoisingCurrentObjectKLConfig.from_mapping(
            data.pop("current_object_kl", {})
        )
        enabled = data.pop("enabled", False)
        if data:
            unknown = [
                f"prefix_denoising.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown prefix_denoising keys: {unknown}")
        return cls(
            enabled=enabled,
            noise=noise,
            current_object_kl=current_object_kl,
        )
```

Also add helper functions:

```python
def _detection_validate_nonnegative_finite_float(value: Any, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{path} must be finite and >= 0")
    return parsed


def _detection_validate_positive_finite_float(value: Any, *, path: str) -> float:
    parsed = _detection_validate_nonnegative_finite_float(value, path=path)
    if parsed <= 0.0:
        raise ValueError(f"{path} must be finite and > 0")
    return parsed
```

- [ ] **Step 4: Add top-level config acceptance and validation**

In `src/config/schema.py`:

- add `"prefix_denoising"` to `_DETECTION_OPTIONAL_SECTIONS`;
- add `prefix_denoising: PrefixDenoisingConfig = field(default_factory=PrefixDenoisingConfig)` to `DetectionTrainingConfig`;
- parse `prefix_denoising = PrefixDenoisingConfig.from_mapping(payload.get("prefix_denoising"))`;
- call `_detection_validate_prefix_denoising_contract` after `training` and `packing` are parsed;
- include `prefix_denoising=prefix_denoising` in the returned config.

Add this raw-payload guard before `objective` is parsed into
`TeacherForcingObjectiveConfig`. This guard exists because the parsed
teacher-forcing dataclass supplies a default `target_ir.rollin_policy`; prefix
denoising must reject only an explicit user-authored target-IR roll-in policy,
not the dataclass default:

```python
def _detection_validate_prefix_denoising_raw_contract(payload: Mapping[str, Any]) -> None:
    prefix_raw = payload.get("prefix_denoising")
    if not isinstance(prefix_raw, Mapping) or not bool(prefix_raw.get("enabled", False)):
        return
    objective_raw = payload.get("objective")
    if not isinstance(objective_raw, Mapping):
        return
    if "coord_soft_ce" in objective_raw:
        raise ValueError(
            "prefix_denoising uses hard clean-label CE only; objective.coord_soft_ce / SoftCE is not supported"
        )
    target_ir_raw = objective_raw.get("target_ir")
    if isinstance(target_ir_raw, Mapping) and "rollin_policy" in target_ir_raw:
        raise ValueError(
            "prefix_denoising owns sorted clean-GT ordering through its hybrid builder; "
            "remove objective.target_ir.rollin_policy"
        )
```

Call `_detection_validate_prefix_denoising_raw_contract(data)` in
`DetectionTrainingConfig.from_mapping` before `objective_raw` is converted by
`DetectionObjectiveConfig.from_mapping`.

Add this parsed validation helper:

```python
def _detection_validate_prefix_denoising_contract(
    *,
    prefix_denoising: PrefixDenoisingConfig,
    data: DetectionDataConfig,
    detection_template: DetectionTemplateConfig,
    objective: DetectionObjectiveConfig | TeacherForcingObjectiveConfig,
    packing: DetectionPackingConfig,
    training: Mapping[str, Any],
) -> None:
    if not prefix_denoising.enabled:
        return
    if detection_template.id != "compact_full":
        raise ValueError("prefix_denoising requires detection_template.id=compact_full")
    if detection_template.coordinate_surface != "coord_token":
        raise ValueError("prefix_denoising requires detection_template.coordinate_surface=coord_token")
    if detection_template.bbox_format != "xyxy":
        raise ValueError("prefix_denoising requires detection_template.bbox_format=xyxy")
    if data.object_ordering != "sorted":
        raise ValueError("prefix_denoising requires data.object_ordering=sorted")
    if getattr(objective, "id", None) != "teacher_forcing" or getattr(objective, "profile", None) != "hard_sft":
        raise ValueError(
            "prefix_denoising requires hard clean-label CE with "
            "objective.id=teacher_forcing and objective.profile=hard_sft"
        )
    modules = getattr(objective, "modules", None)
    if modules is not None:
        enabled_modules = []
        for name in ("token_type_mass", "conditional_valid_set_likelihood", "within_valid_coverage", "continuation_margin"):
            module = getattr(modules, name, None)
            if bool(getattr(module, "enabled", False)):
                enabled_modules.append(name)
        coverage = getattr(modules, "within_valid_coverage", None)
        if float(getattr(coverage, "coverage_strength", 0.0) or 0.0) != 0.0:
            enabled_modules.append("within_valid_coverage.coverage_strength")
        if enabled_modules:
            raise ValueError(f"prefix_denoising requires hard_sft with teacher-forcing modules disabled: {enabled_modules}")
    encoded_cache = training.get("encoded_sample_cache")
    if isinstance(encoded_cache, Mapping) and bool(encoded_cache.get("enabled", False)):
        raise ValueError("prefix_denoising requires training.encoded_sample_cache.enabled=false")
    if bool(training.get("use_logits_to_keep", False)):
        raise ValueError("prefix_denoising requires training.use_logits_to_keep=false")
    if packing.padding_free_packed:
        raise ValueError("prefix_denoising V1 uses training.packing plus static hybrid pack planning; explicit packing.padding_free_packed=true is deferred")
```

Do not revive legacy `objective.id=sft`. Prefix-denoising stays under `objective.id=teacher_forcing` with `objective.profile=hard_sft`, but it must bypass the existing target-IR random-permutation runtime by using its own sorted clean-GT hybrid builder. Update the teacher-forcing packing guard so `training.packing=true` and `packing.static_packing=true` are allowed only when `prefix_denoising.enabled=true` and the V1 hybrid materializer/boundary-map path is active. Non-V1 teacher-forcing configs must continue to reject packing exactly as before.

- [ ] **Step 5: Run schema tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_config_contract.py -q
python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py -q
```

Expected: all pass. Existing teacher-forcing random-permutation configs should still parse or reject exactly as before.

- [ ] **Step 6: Commit schema contract**

Run:

```bash
git add src/config/schema.py tests/test_prefix_denoising_config_contract.py
git commit -m "feat: add prefix denoising config contract"
```

## Task 2: Constructive Norm1000 Bbox Noiser

**Files:**
- Modify: `src/datasets/geometry.py`
- Create: `tests/test_prefix_denoising_geometry.py`

- [ ] **Step 1: Write failing noiser tests**

Create `tests/test_prefix_denoising_geometry.py`:

```python
from __future__ import annotations

import random

import pytest

from src.datasets.geometry import (
    BBoxNoiseConfig,
    construct_valid_norm1000_bbox_noise,
)


def _cfg() -> BBoxNoiseConfig:
    return BBoxNoiseConfig(center_shift_frac=0.08, uniform_scale_range=(0.92, 1.08))


def test_constructive_noise_returns_valid_4coord_changed_bbox() -> None:
    result = construct_valid_norm1000_bbox_noise(
        (100, 120, 260, 300),
        config=_cfg(),
        rng=random.Random(7),
        object_id="obj-1",
    )

    assert result.ok is True
    assert result.noisy_bbox is not None
    x1, y1, x2, y2 = result.noisy_bbox
    assert 0 <= x1 < x2 <= 999
    assert 0 <= y1 < y2 <= 999
    assert tuple(result.changed) == (True, True, True, True)
    assert result.clean_bins == (100, 120, 260, 300)
    assert result.noisy_bins != result.clean_bins


def test_constructive_noise_is_seed_deterministic() -> None:
    a = construct_valid_norm1000_bbox_noise((100, 120, 260, 300), config=_cfg(), rng=random.Random(11))
    b = construct_valid_norm1000_bbox_noise((100, 120, 260, 300), config=_cfg(), rng=random.Random(11))

    assert a.noisy_bbox == b.noisy_bbox
    assert a.provenance == b.provenance


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 1, 1),
        (0, 0, 999, 999),
        (998, 998, 999, 999),
        (10, 10, 11, 500),
        (10, 10, 500, 11),
    ],
)
def test_constructive_noise_reports_infeasible_without_repair(bbox: tuple[int, int, int, int]) -> None:
    result = construct_valid_norm1000_bbox_noise(
        bbox,
        config=BBoxNoiseConfig(center_shift_frac=0.0, uniform_scale_range=(1.0, 1.0)),
        rng=random.Random(3),
    )

    assert result.ok is False
    assert result.skip_reason in {"noise_infeasible_valid_bbox", "noise_infeasible_4coord_changed"}
    assert result.noisy_bbox is None


def test_invalid_clean_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise((5, 5, 5, 6), config=_cfg(), rng=random.Random(1))
```

- [ ] **Step 2: Run the tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_geometry.py -q
```

Expected: FAIL because `BBoxNoiseConfig` and `construct_valid_norm1000_bbox_noise` do not exist.

- [ ] **Step 3: Add noiser dataclasses and constructive helper**

In `src/datasets/geometry.py`, add:

```python
@dataclass(frozen=True)
class BBoxNoiseConfig:
    center_shift_frac: float = 0.08
    uniform_scale_range: tuple[float, float] = (0.92, 1.08)
    coord_min: int = 0
    coord_max: int = 999


@dataclass(frozen=True)
class BBoxNoiseResult:
    ok: bool
    clean_bbox: tuple[int, int, int, int]
    noisy_bbox: tuple[int, int, int, int] | None
    clean_bins: tuple[int, int, int, int]
    noisy_bins: tuple[int, int, int, int] | None
    changed: tuple[bool, bool, bool, bool]
    skip_reason: str | None
    provenance: dict[str, int | float | str]
```

Add helpers:

```python
def _coerce_norm1000_xyxy(bbox: Sequence[int | float], *, field_name: str) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"{field_name} must contain four coordinates")
    values = tuple(int(round(float(value))) for value in bbox)
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        raise ValueError(f"{field_name} must be valid norm1000 xyxy")
    return values


def construct_valid_norm1000_bbox_noise(
    clean_bbox: Sequence[int | float],
    *,
    config: BBoxNoiseConfig,
    rng: random.Random,
    object_id: str = "",
) -> BBoxNoiseResult:
    clean = _coerce_norm1000_xyxy(clean_bbox, field_name="clean bbox")
    x1, y1, x2, y2 = clean
    width = x2 - x1
    height = y2 - y1
    candidates: list[tuple[int, int, int, int]] = []
    max_dx = int(round(width * float(config.center_shift_frac)))
    max_dy = int(round(height * float(config.center_shift_frac)))
    scale_low, scale_high = config.uniform_scale_range
    scale_values = sorted({scale_low, 1.0, scale_high})
    cx2 = x1 + x2
    cy2 = y1 + y2
    for dx in range(-max_dx, max_dx + 1):
        for dy in range(-max_dy, max_dy + 1):
            for sx in scale_values:
                for sy in scale_values:
                    new_w = max(1, int(round(width * sx)))
                    new_h = max(1, int(round(height * sy)))
                    new_cx2 = cx2 + 2 * dx
                    new_cy2 = cy2 + 2 * dy
                    nx1 = int(round((new_cx2 - new_w) / 2.0))
                    ny1 = int(round((new_cy2 - new_h) / 2.0))
                    nx2 = nx1 + new_w
                    ny2 = ny1 + new_h
                    candidate = (nx1, ny1, nx2, ny2)
                    if not (0 <= nx1 < nx2 <= 999 and 0 <= ny1 < ny2 <= 999):
                        continue
                    if all(a != b for a, b in zip(clean, candidate, strict=True)):
                        candidates.append(candidate)
    unique_candidates = sorted(set(candidates))
    if not unique_candidates:
        return BBoxNoiseResult(
            ok=False,
            clean_bbox=clean,
            noisy_bbox=None,
            clean_bins=clean,
            noisy_bins=None,
            changed=(False, False, False, False),
            skip_reason="noise_infeasible_4coord_changed",
            provenance={"object_id": object_id, "candidate_count": 0},
        )
    noisy = unique_candidates[rng.randrange(len(unique_candidates))]
    return BBoxNoiseResult(
        ok=True,
        clean_bbox=clean,
        noisy_bbox=noisy,
        clean_bins=clean,
        noisy_bins=noisy,
        changed=tuple(a != b for a, b in zip(clean, noisy, strict=True)),
        skip_reason=None,
        provenance={
            "object_id": object_id,
            "candidate_count": len(unique_candidates),
            "selected_index": unique_candidates.index(noisy),
        },
    )
```

This candidate-grid helper is intentionally simple. It must respect the authored envelope exactly: a zero-strength config with `center_shift_frac=0.0` and `uniform_scale_range=(1.0, 1.0)` has no valid 4-coordinate-changing candidates and returns `ok=False`. Do not introduce an implicit one-bin minimum movement; default/fallback configs may still produce one-bin movement when it falls inside their explicit envelope. If later implementation needs a denser random profile, it must preserve the same public contract: direct valid construction, no clamp-repair, deterministic seed behavior, and explicit infeasible result.

Noising fallback policy:

- If launch-health shows high `noise_infeasible_4coord_changed` or small/thin-box
  skip rates, do not switch to milder noise as the first fix. A smaller envelope
  can shrink the feasible set under the strict 4/4-changed rule. First measure
  skip rate by clean bbox width/height bins, then either increase the authored
  envelope or implement direct feasible-set sampling with an explicit candidate
  cap.
- If launch-health shows finite but too-difficult noisy CE/KL behavior, such as a
  severe noisy-full CE gap, token-accuracy collapse, or poor teacher-vs-student
  local-window diagnostics with acceptable skip rates, the first milder fallback
  remains `center_shift_frac=0.04` and `uniform_scale_range=(0.96, 1.04)`.

- [ ] **Step 4: Run geometry tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_geometry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit geometry noiser**

Run:

```bash
git add src/datasets/geometry.py tests/test_prefix_denoising_geometry.py
git commit -m "feat: add constructive prefix denoising bbox noise"
```

## Task 3: Hybrid Types, Builder, And Dataset

**Files:**
- Create: `src/detection/prefix_denoising/__init__.py`
- Create: `src/detection/prefix_denoising/types.py`
- Create: `src/detection/prefix_denoising/builder.py`
- Create: `src/detection/prefix_denoising/dataset.py`
- Modify: `src/detection/runtime.py`
- Create: `tests/test_prefix_denoising_builder.py`

- [ ] **Step 1: Write failing hybrid builder tests**

Create `tests/test_prefix_denoising_builder.py` with a fake tokenizer/template fixture copied from existing compact-full tests where possible:

```python
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from src.config.schema import PrefixDenoisingConfig
from src.detection.prefix_denoising.builder import build_hybrid_prefix_denoising_sample


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        if text.startswith("<|coord_") and text.endswith("|>"):
            return [int(text[len("<|coord_") : -len("|>")])]
        return [ord(ch) + 1000 for ch in text]

    def convert_tokens_to_ids(self, token: str):
        return self.encode(token)[0]


class _Template:
    tokenizer = _Tokenizer()

    def encode(self, payload: dict):
        text = payload["messages"][-1]["content"]
        ids = []
        labels = []
        for token in text.split(" "):
            token_ids = self.tokenizer.encode(token)
            ids.extend(token_ids)
            labels.extend(token_ids)
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def _row(objects: list[dict] | None = None) -> dict:
    return {
        "image": "dummy.jpg",
        "objects": [
            {"desc": "red box", "bbox_2d": [100, 100, 200, 220]},
            {"desc": "blue box", "bbox_2d": [300, 320, 420, 470]},
        ] if objects is None else objects,
        "metadata": {"source": "unit"},
    }


def test_hybrid_builder_emits_two_segments_and_clean_labels(tmp_path) -> None:
    image = tmp_path / "dummy.jpg"
    image.write_bytes(b"fake")
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "noise": {"center_shift_frac": 0.08, "uniform_scale_range": [0.92, 1.08]},
            "current_object_kl": {"weight": 0.0, "window_radius": 8, "num_objects_per_image": 1},
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        row=_row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=_Template(),
        user_prompt="Locate objects.",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert sample.clean_full.branch_id == "clean_full"
    assert sample.noisy_full.branch_id == "noisy_full"
    assert sample.clean_full.labels == sample.noisy_full.labels
    assert len(sample.clean_full.input_ids) == len(sample.noisy_full.input_ids)
    assert sample.kl_sites == ()


def test_hybrid_builder_builds_kl_sites_when_weight_positive(tmp_path) -> None:
    image = tmp_path / "dummy.jpg"
    image.write_bytes(b"fake")
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "noise": {"center_shift_frac": 0.08, "uniform_scale_range": [0.92, 1.08]},
            "current_object_kl": {"weight": 0.05, "window_radius": 8, "num_objects_per_image": 1},
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        row=_row(),
        base_sample_id="unit-0",
        image_root=tmp_path,
        swift_template=_Template(),
        user_prompt="Locate objects.",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=2,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is True
    assert len(sample.kl_sites) == 4
    assert {site.coord_slot for site in sample.kl_sites} == {"x1", "y1", "x2", "y2"}
    assert all(site.clean_gt_bin in site.support_bins for site in sample.kl_sites)


def test_hybrid_builder_excludes_zero_object_rows_with_counter_reason(tmp_path) -> None:
    image = tmp_path / "dummy.jpg"
    image.write_bytes(b"fake")
    cfg = PrefixDenoisingConfig.from_mapping(
        {
            "enabled": True,
            "noise": {"center_shift_frac": 0.08, "uniform_scale_range": [0.92, 1.08]},
            "current_object_kl": {"weight": 0.05, "window_radius": 8, "num_objects_per_image": 1},
        }
    )

    sample = build_hybrid_prefix_denoising_sample(
        row=_row(objects=[]),
        base_sample_id="unit-empty",
        image_root=tmp_path,
        swift_template=_Template(),
        user_prompt="Locate objects.",
        system_prompt=None,
        prefix_denoising=cfg,
        epoch=0,
        rng=random.Random(5),
        max_length=12000,
    )

    assert sample.ok is False
    assert sample.skip_reason == "zero_object_hybrid_sample"
    assert sample.total_length == 0
```

- [ ] **Step 2: Run builder tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_builder.py -q
```

Expected: FAIL because the prefix-denoising package does not exist.

- [ ] **Step 3: Add focused dataclasses**

Create `src/detection/prefix_denoising/types.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal


PrefixDenoisingBranchId = Literal["clean_full", "noisy_full"]
CoordSlot = Literal["x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class PrefixDenoisingSegment:
    segment_id: str
    branch_id: PrefixDenoisingBranchId
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    supervised_positions: tuple[int, ...]
    ce_denominator: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixDenoisingKLSite:
    clean_segment_id: str
    noisy_segment_id: str
    object_index: int
    history_object_count: int
    coord_slot: CoordSlot
    clean_label_position: int
    noisy_label_position: int
    clean_gt_bin: int
    support_bins: tuple[int, ...]
    identical_prefix: bool = False


@dataclass(frozen=True)
class ResolvedPrefixDenoisingKLSite:
    clean_batch_index: int
    noisy_batch_index: int
    clean_label_position: int
    noisy_label_position: int
    clean_gt_bin: int
    support_bins: tuple[int, ...]
    coord_slot: CoordSlot
    object_index: int
    identical_prefix: bool = False


@dataclass(frozen=True)
class HybridPrefixDenoisingSample:
    ok: bool
    hybrid_sample_id: str
    base_sample_id: str
    clean_full: PrefixDenoisingSegment | None
    noisy_full: PrefixDenoisingSegment | None
    kl_sites: tuple[PrefixDenoisingKLSite, ...] = ()
    skip_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def total_length(self) -> int:
        if self.clean_full is None or self.noisy_full is None:
            return 0
        return len(self.clean_full.input_ids) + len(self.noisy_full.input_ids)
```

`support_bins` are coordinate bins in `[0, 999]`, not full-vocabulary token ids. Loss code must map bins through the tokenizer's coord-token id row before indexing logits. `PrefixDenoisingKLSite` is builder-local metadata; packed or collated loss code must consume `ResolvedPrefixDenoisingKLSite` with explicit physical batch indices and rewritten physical label positions. Do not default unresolved KL sites to batch row `0`.

Create `src/detection/prefix_denoising/__init__.py` with exports for these classes and the builder.

- [ ] **Step 4: Add the first builder implementation**

Create `src/detection/prefix_denoising/builder.py`.

The implementation should:

- parse raw detection rows using existing detection helpers where possible;
- exclude zero-object rows before rendering with `skip_reason="zero_object_hybrid_sample"` so duplicated no-op clean/noisy branches do not dilute noisy CE or KL diagnostics;
- sort objects by clean top-left order through the existing detection scene/order path;
- render compact-full entries with clean bbox tokens and noisy bbox tokens;
- use `construct_valid_norm1000_bbox_noise` for every valid object;
- call the same Swift template encoding path as `DetectionTrainingDataset._encode_messages`;
- create clean labels for both branches;
- compare clean/noisy lengths and label equality;
- preserve prompt/user/system labels as `-100`, never supervise physical position `0`, and supervise only assistant-response clean labels plus assistant stop tokens;
- locate coordinate label positions for every object slot;
- select `K_i` object indices by deterministic seeded shuffle;
- build no KL sites when weight is zero.

The initial implementation may use private helpers copied into the module from `DetectionTrainingDataset` only when they are small and stable. If a helper is already public, import it instead of duplicating it.

- [ ] **Step 5: Add dataset wrapper**

Create `src/detection/prefix_denoising/dataset.py`:

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset

from src.config.schema import PrefixDenoisingConfig
from src.detection.prefix_denoising.builder import build_hybrid_prefix_denoising_sample


class PrefixDenoisingTrainingDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        swift_template: Any,
        image_root: str | Path,
        user_prompt: str,
        system_prompt: str | None,
        prefix_denoising: PrefixDenoisingConfig,
        max_length: int,
        dataset_name: str,
        seed: int,
    ) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.swift_template = swift_template
        self.image_root = Path(image_root)
        self.user_prompt = str(user_prompt)
        self.system_prompt = system_prompt
        self.prefix_denoising = prefix_denoising
        self.max_length = int(max_length)
        self.dataset_name = str(dataset_name)
        self.seed = int(seed)
        self._epoch = 0
        self._eligible_indices, self.skip_counters = build_prefix_denoising_eligibility_index(
            rows=self.rows,
            image_root=self.image_root,
            swift_template=self.swift_template,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            prefix_denoising=self.prefix_denoising,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return len(self._eligible_indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _static_packing_length(self, index: int) -> int:
        base_idx = self._eligible_indices[int(index)]
        sample = self._build(base_idx, epoch=0)
        return sample.total_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self._eligible_indices[int(index)]
        sample = self._build(base_idx, epoch=self._epoch)
        if not sample.ok:
            raise RuntimeError(f"planned prefix denoising sample became ineligible: {sample.skip_reason}")
        return materialize_hybrid_model_ready_item(
            sample=sample,
            dataset_name=self.dataset_name,
        )

    def _build(self, index: int, *, epoch: int):
        base_idx = int(index)
        rng = random.Random((self.seed * 1000003) + (epoch * 9176) + base_idx)
        return build_hybrid_prefix_denoising_sample(
            row=self.rows[base_idx],
            base_sample_id=f"{self.dataset_name}-{base_idx}",
            image_root=self.image_root,
            swift_template=self.swift_template,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            prefix_denoising=self.prefix_denoising,
            epoch=epoch,
            rng=rng,
            max_length=self.max_length,
        )
```

Implementation note: `materialize_hybrid_model_ready_item` must return a real collator-ready sample, not only a sidecar. Its output contract is:

- `input_ids`, `labels`, and `attention_mask` for the two complete segments or for a pre-flattened hybrid item;
- multimodal fields required by the active Qwen-VL template, including duplicated logical ownership for `pixel_values` and `image_grid_thw`;
- `prefix_denoising_segment_meta` with `hybrid_sample_id`, `segment_id`, `branch_id`, local token start/end, and local supervised positions for both branches;
- `prefix_denoising_hybrid` sidecar;
- `length` equal to `len(clean_full) + len(noisy_full)`.

The private row-loading path must reuse the same JSONL loader and prompt/template resolution used by the current detection training dataset so schema diagnostics remain identical outside the new hybrid sample payload. Overlength or deterministic noising-infeasible rows are excluded by `_eligible_indices` before static packing; `__getitem__` must not be the normal skip mechanism.

- [ ] **Step 6: Route runtime dataset construction**

In `src/detection/runtime.py`, update `build_detection_dataset`:

- if `training_config.prefix_denoising.enabled` is true, return `PrefixDenoisingTrainingDataset`;
- use resolved prompts from `custom_config.user_prompt` and `system_prompt`;
- load rows via the same `load_jsonl_with_diagnostics` path as `DetectionTrainingDataset.from_jsonl`;
- pass `global_max_length` or template max length as `max_length`.

- [ ] **Step 7: Run builder tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_builder.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit hybrid builder**

Run:

```bash
git add src/detection/prefix_denoising tests/test_prefix_denoising_builder.py src/detection/runtime.py
git commit -m "feat: build prefix denoising hybrid samples"
```

## Task 4: Collator Sidecar And Batch Contract

**Files:**
- Modify: `src/detection/dataset.py`
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Create: `tests/test_prefix_denoising_collator.py`

- [ ] **Step 1: Write failing collator tests**

Create `tests/test_prefix_denoising_collator.py`:

```python
from __future__ import annotations

import pytest

from src.data_collators.enrichers import PrefixDenoisingHybridEnricher
from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.types import HybridPrefixDenoisingSample


def _sample(sample_id: str) -> HybridPrefixDenoisingSample:
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=sample_id,
        clean_full=None,
        noisy_full=None,
        kl_sites=(),
    )


def test_prefix_denoising_enricher_requires_all_rows_present_unpacked() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {}

    with pytest.raises(ValueError, match="prefix_denoising_hybrid sidecar must be present"):
        enricher(
            collated=collated,
            raw_batch=[{"prefix_denoising_hybrid": _sample("a")}, {}],
            packed=False,
        )


def test_prefix_denoising_enricher_attaches_tuple_unpacked() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {}

    enricher(
        collated=collated,
        raw_batch=[{"prefix_denoising_hybrid": _sample("a")}],
        packed=False,
    )

    assert "prefix_denoising_hybrid" in collated
    assert len(collated["prefix_denoising_hybrid"]) == 1  # type: ignore[arg-type]


def test_prefix_denoising_enricher_rejects_plain_packed_sidecar_without_boundary_map() -> None:
    enricher = PrefixDenoisingHybridEnricher()

    with pytest.raises(ValueError, match="PackedHybridBoundaryMap"):
        enricher(
            collated={},
            raw_batch=[[{"prefix_denoising_hybrid": _sample("a")}]],
            packed=True,
        )


def test_prefix_denoising_enricher_attaches_packed_hybrids_when_boundary_map_exists() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {"packed_hybrid_boundary_map": object()}

    enricher(
        collated=collated,
        raw_batch=[
            [
                {"prefix_denoising_hybrid": _sample("a")},
                {"prefix_denoising_hybrid": _sample("b")},
            ]
        ],
        packed=True,
    )

    assert "prefix_denoising_hybrid" in collated
    packed_groups = collated["prefix_denoising_hybrid"]
    assert len(packed_groups) == 1  # type: ignore[arg-type]
    assert len(packed_groups[0]) == 2  # type: ignore[index]


def test_prefix_denoising_sidecars_are_registered_at_model_boundary() -> None:
    batch = {
        "input_ids": object(),
        "labels": object(),
        "prefix_denoising_hybrid": object(),
        "prefix_denoising_segment_meta": object(),
        "packed_hybrid_boundary_map": object(),
        "prefix_denoising_resolved_kl_sites": object(),
        "sample_id": "unit-0",
    }

    stripped = strip_non_model_detection_sidecars(batch)

    assert sorted(stripped) == ["input_ids", "labels"]
```

- [ ] **Step 2: Run tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_collator.py -q
```

Expected: FAIL because `PrefixDenoisingHybridEnricher` does not exist.

- [ ] **Step 3: Add enricher**

In `src/data_collators/enrichers.py`, add:

```python
class PrefixDenoisingHybridEnricher:
    out_field = "prefix_denoising_hybrid"

    def __call__(
        self,
        *,
        collated: dict[str, Any],
        raw_batch: Sequence[Any],
        packed: bool,
    ) -> None:
        if packed:
            packed_groups: list[tuple[Any, ...]] = []
            has_sidecar = False
            missing_sidecar = False
            for pack in raw_batch:
                pack_seq = pack if isinstance(pack, (list, tuple)) else [pack]
                group: list[Any] = []
                for sample in pack_seq:
                    if isinstance(sample, Mapping) and self.out_field in sample:
                        has_sidecar = True
                        group.append(sample[self.out_field])
                    else:
                        missing_sidecar = True
                if group:
                    packed_groups.append(tuple(group))
            if not has_sidecar:
                return
            if missing_sidecar:
                raise ValueError(
                    "prefix_denoising_hybrid sidecar must be present for every sample in a packed prefix-denoising batch"
                )
            if "packed_hybrid_boundary_map" not in collated:
                raise ValueError(
                    "prefix_denoising_hybrid packed sidecars require PackedHybridBoundaryMap"
                )
            collated[self.out_field] = tuple(packed_groups)
            return

        present = [
            isinstance(row, Mapping) and self.out_field in row for row in raw_batch
        ]
        if not any(present):
            return
        if not all(present):
            raise ValueError(
                "prefix_denoising_hybrid sidecar must be present for every sample in an unpacked prefix-denoising batch"
            )
        collated[self.out_field] = tuple(
            row[self.out_field] for row in raw_batch if isinstance(row, Mapping)
        )
```

In `src/detection/dataset.py`, add the prefix-denoising sidecars to
`REGISTERED_DETECTION_SIDECAR_KEYS` so the canonical model-input boundary strips
them before `model(**inputs)`:

```python
REGISTERED_DETECTION_SIDECAR_KEYS: tuple[str, ...] = (
    "recursive_detection_targets",
    TEACHER_FORCING_TARGET_IR_KEY,
    "rendered_span_sources",
    "detection_supervision_view_metadata",
    "detection_metadata",
    "assistant_payload",
    "sample_id",
    "dataset",
    "base_idx",
    "prefix_denoising_hybrid",
    "prefix_denoising_segment_meta",
    "packed_hybrid_boundary_map",
    "prefix_denoising_resolved_kl_sites",
)
```

In `src/data_collators/batch_extras_collator.py`, instantiate and call
`PrefixDenoisingHybridEnricher` after `DatasetMetaEnricher` and before
token-type/proxy enrichers. This task does not add the packed boundary-map
producer; Task 7 owns that producer and wires it into this collator after the
base template collator has flattened a packed batch.

- [ ] **Step 4: Run collator tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_collator.py -q
```

Expected: PASS.

- [ ] **Step 5: Add a model-ready runtime integration test**

Create `tests/test_prefix_denoising_runtime_integration.py`. The test must build one tiny two-segment hybrid item, send it through the actual configured `BatchExtrasCollator`/template collator path, and assert the collated batch contains:

- `input_ids`, `labels`, `attention_mask`;
- Qwen-VL multimodal tensors or stand-in test fields for `pixel_values` and `image_grid_thw`;
- `prefix_denoising_segment_meta` for both `clean_full` and `noisy_full`;
- `prefix_denoising_hybrid`;
- no unregistered sidecars;
- local CE label positions and KL label positions that can be rewritten to packed/logit positions.

The test should then call a dummy `PrefixDenoisingObjectiveMixin.compute_loss` path with fixed logits and prove the trainer sees labels, segment metadata, and boundary metadata.

Run:

```bash
python -m pytest tests/test_prefix_denoising_collator.py tests/test_prefix_denoising_runtime_integration.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit sidecar contract**

Run:

```bash
git add src/detection/dataset.py src/data_collators/enrichers.py src/data_collators/batch_extras_collator.py tests/test_prefix_denoising_collator.py tests/test_prefix_denoising_runtime_integration.py
git commit -m "feat: carry prefix denoising hybrid sidecars"
```

## Task 5: CE-Only Loss Mixin And Metrics

**Files:**
- Create: `src/detection/prefix_denoising/loss.py`
- Create: `src/detection/prefix_denoising/metrics.py`
- Create: `src/trainers/metrics/prefix_denoising.py`
- Modify: `src/trainers/metrics/mixins.py`
- Modify: `src/bootstrap/trainer_setup.py`
- Modify: `src/sft.py`
- Create: `tests/test_prefix_denoising_loss.py`
- Create: `tests/test_prefix_denoising_metrics.py`
- Modify: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 1: Write failing CE loss tests**

Create `tests/test_prefix_denoising_loss.py`:

```python
from __future__ import annotations

import torch

from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    topk_accuracy_from_logits,
)


def test_branch_balanced_hard_ce_uses_shifted_positions_and_all_segments() -> None:
    logits = torch.full((1, 8, 4), -9.0, dtype=torch.float32)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    labels[0, 2] = 1
    labels[0, 3] = 2
    labels[0, 6] = 3
    logits[0, 1, 1] = 8.0
    logits[0, 2, 2] = 8.0
    logits[0, 5, 3] = 8.0
    logits[0, 2, 0] = 12.0  # same-position row would be wrong for label position 2
    spans = (
        PrefixDenoisingSegmentSpan(batch_index=0, token_start=0, token_end=4, branch_id="clean_full", segment_id="a:clean"),
        PrefixDenoisingSegmentSpan(batch_index=0, token_start=4, token_end=8, branch_id="noisy_full", segment_id="a:noisy"),
    )

    result = compute_branch_balanced_hard_ce(
        logits=logits,
        labels=labels,
        segment_spans=spans,
    )

    clean_ce = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2]]),
        torch.tensor([1, 2]),
    )
    noisy_ce = torch.nn.functional.cross_entropy(logits[0, 5].unsqueeze(0), torch.tensor([3]))
    torch.testing.assert_close(result.loss, 0.5 * clean_ce + 0.5 * noisy_ce)
    assert result.clean_denominator == 2
    assert result.noisy_denominator == 1


def test_topk_accuracy_uses_shifted_causal_positions() -> None:
    logits = torch.tensor([[[0.0, 5.0, 1.0], [5.0, 0.0, 1.0], [0.0, 1.0, 5.0]]], dtype=torch.float32)
    labels = torch.tensor([[-100, 1, -100]], dtype=torch.long)
    spans = (
        PrefixDenoisingSegmentSpan(batch_index=0, token_start=0, token_end=3, branch_id="clean_full", segment_id="a:clean"),
    )

    result = topk_accuracy_from_logits(logits=logits, labels=labels, segment_spans=spans, topk=(1, 2))

    assert result[1] == 1.0
    assert result[2] == 1.0
```

- [ ] **Step 2: Write failing metric tests**

Create `tests/test_prefix_denoising_metrics.py`:

```python
from __future__ import annotations

import pytest

from src.metrics.events import flatten_metric_events
from src.detection.prefix_denoising.metrics import (
    PREFIX_DENOISING_REQUIRED_METRIC_KEYS,
    prefix_denoising_ce_events,
)


def test_prefix_denoising_metric_keys_are_distinct_flat_keys() -> None:
    events = prefix_denoising_ce_events(
        ce_balanced=1.2,
        ce_clean=1.0,
        ce_noisy=1.4,
        ce_token_pooled=1.25,
        token_top1=0.5,
        token_top5=0.8,
        clean_denominator=10,
        noisy_denominator=12,
    )

    flat = flatten_metric_events(events)

    for key in PREFIX_DENOISING_REQUIRED_METRIC_KEYS:
        assert key in flat
    assert flat["prefix_denoising/global/loss/ce_balanced"] == pytest.approx(1.2)
    assert flat["prefix_denoising/clean_full/loss/ce"] == pytest.approx(1.0)
    assert flat["prefix_denoising/noisy_full/loss/ce"] == pytest.approx(1.4)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top1"] == pytest.approx(0.5)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top5"] == pytest.approx(0.8)
```

- [ ] **Step 3: Run tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_loss.py tests/test_prefix_denoising_metrics.py -q
python -m pytest tests/test_training_runtime_sft_integration.py tests/test_training_runtime_profile.py -q
```

Expected: FAIL because loss and metrics modules do not exist.

- [ ] **Step 4: Implement CE helpers**

Create `src/detection/prefix_denoising/loss.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrefixDenoisingSegmentSpan:
    batch_index: int
    token_start: int
    token_end: int
    branch_id: str
    segment_id: str


@dataclass(frozen=True)
class PrefixDenoisingCEResult:
    loss: torch.Tensor
    clean_ce: torch.Tensor
    noisy_ce: torch.Tensor
    token_pooled_ce: torch.Tensor
    clean_denominator: int
    noisy_denominator: int


def _segment_ce_sum(logits: torch.Tensor, labels: torch.Tensor, span: PrefixDenoisingSegmentSpan) -> tuple[torch.Tensor, int]:
    batch_index = int(span.batch_index)
    start = int(span.token_start)
    end = int(span.token_end)
    label_positions = torch.arange(start, end, device=labels.device)
    segment_labels = labels[batch_index, start:end]
    active = segment_labels.ne(-100)
    denominator = int(active.sum().detach().cpu().item())
    if denominator == 0:
        return logits[batch_index, start:end].float().sum() * 0.0, 0
    active_positions = label_positions[active]
    if torch.any(active_positions <= start):
        raise ValueError("prefix denoising labels at segment start cannot be supervised in causal LM")
    active_logits = logits[batch_index, active_positions - 1]
    active_labels = segment_labels[active]
    denominator = int(active_labels.numel())
    return F.cross_entropy(active_logits, active_labels, reduction="sum"), denominator


def compute_branch_balanced_hard_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_spans: tuple[PrefixDenoisingSegmentSpan, ...],
) -> PrefixDenoisingCEResult:
    if logits.ndim != 3:
        raise ValueError("prefix denoising CE requires logits [batch, time, vocab]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("prefix denoising labels must match logits batch/time shape")
    sums = {"clean_full": logits.float().sum() * 0.0, "noisy_full": logits.float().sum() * 0.0}
    denominators = {"clean_full": 0, "noisy_full": 0}
    for span in segment_spans:
        if span.branch_id not in sums:
            raise ValueError(f"unsupported prefix denoising branch_id: {span.branch_id!r}")
        ce_sum, denom = _segment_ce_sum(logits, labels, span)
        sums[span.branch_id] = sums[span.branch_id] + ce_sum
        denominators[span.branch_id] += denom
    clean_den = denominators["clean_full"]
    noisy_den = denominators["noisy_full"]
    if clean_den == 0 or noisy_den == 0:
        raise ValueError("prefix denoising CE requires supervised labels in both clean_full and noisy_full")
    clean_ce = sums["clean_full"] / float(clean_den)
    noisy_ce = sums["noisy_full"] / float(noisy_den)
    token_pooled = (sums["clean_full"] + sums["noisy_full"]) / float(clean_den + noisy_den)
    return PrefixDenoisingCEResult(
        loss=0.5 * clean_ce + 0.5 * noisy_ce,
        clean_ce=clean_ce,
        noisy_ce=noisy_ce,
        token_pooled_ce=token_pooled,
        clean_denominator=clean_den,
        noisy_denominator=noisy_den,
    )


def topk_accuracy_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_spans: tuple[PrefixDenoisingSegmentSpan, ...],
    topk: Iterable[int] = (1, 5),
) -> dict[int, float]:
    active_logits: list[torch.Tensor] = []
    active_labels: list[torch.Tensor] = []
    for span in segment_spans:
        batch_index = int(span.batch_index)
        start = int(span.token_start)
        end = int(span.token_end)
        label_positions = torch.arange(start, end, device=labels.device)
        segment_labels = labels[batch_index, start:end]
        active = segment_labels.ne(-100)
        if not bool(active.any()):
            continue
        positions = label_positions[active]
        if torch.any(positions <= start):
            raise ValueError("prefix denoising labels at segment start cannot be scored in causal LM")
        active_logits.append(logits[batch_index, positions - 1])
        active_labels.append(segment_labels[active])
    denominator = int(sum(int(item.numel()) for item in active_labels))
    if denominator == 0:
        return {int(k): 0.0 for k in topk}
    active_logits_tensor = torch.cat(active_logits, dim=0)
    active_labels_tensor = torch.cat(active_labels, dim=0)
    max_k = max(int(k) for k in topk)
    top = active_logits_tensor.topk(k=max_k, dim=-1).indices
    out: dict[int, float] = {}
    for k in topk:
        kk = int(k)
        correct = top[:, :kk].eq(active_labels_tensor.unsqueeze(-1)).any(dim=-1).float().sum()
        out[kk] = float((correct / float(denominator)).detach().cpu().item())
    return out
```

- [ ] **Step 5: Implement metric events**

Create `src/detection/prefix_denoising/metrics.py`:

```python
from __future__ import annotations

from src.metrics.events import weighted_mean_event


PREFIX_DENOISING_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "prefix_denoising/global/loss/ce_balanced",
    "prefix_denoising/global/loss/ce_token_pooled",
    "prefix_denoising/clean_full/loss/ce",
    "prefix_denoising/noisy_full/loss/ce",
    "prefix_denoising/global/token_acc/full_vocab/top1",
    "prefix_denoising/global/token_acc/full_vocab/top5",
)


def prefix_denoising_ce_events(
    *,
    ce_balanced: float,
    ce_clean: float,
    ce_noisy: float,
    ce_token_pooled: float,
    token_top1: float,
    token_top5: float,
    clean_denominator: int,
    noisy_denominator: int,
):
    total_denominator = float(int(clean_denominator) + int(noisy_denominator))
    return (
        weighted_mean_event(
            "prefix_denoising/global/loss/ce_balanced",
            ce_balanced,
            1.0,
            unit="token",
            metric_surface="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/loss/ce_token_pooled",
            ce_token_pooled,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
            diagnostic_only=True,
        ),
        weighted_mean_event(
            "prefix_denoising/clean_full/loss/ce",
            ce_clean,
            float(clean_denominator),
            unit="token",
            metric_surface="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/noisy_full/loss/ce",
            ce_noisy,
            float(noisy_denominator),
            unit="token",
            metric_surface="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/token_acc/full_vocab/top1",
            token_top1,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/token_acc/full_vocab/top5",
            token_top5,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
        ),
    )
```

Implementation note: `src.metrics.events.weighted_mean_event` takes `key`, `value`, `weight`, and keyword metadata; it stores `numerator=value * weight` and `denominator=weight`. Keep the calls above in that contract.

- [ ] **Step 6: Add trainer mixin**

Create `src/trainers/metrics/prefix_denoising.py`:

```python
from __future__ import annotations

from typing import MutableMapping

import torch

from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    topk_accuracy_from_logits,
)
from src.detection.prefix_denoising.metrics import prefix_denoising_ce_events
from src.metrics.events import flatten_metric_events
from src.metrics.reporter import SwiftMetricReporter
from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras
from src.trainers.teacher_forcing.forwards import prepare_forward_inputs


def _resolve_prefix_denoising_packing_enabled(trainer) -> bool:
    value = getattr(trainer, "prefix_denoising_packing_enabled", None)
    if value is None:
        raise ValueError(
            "prefix_denoising trainer requires explicit prefix_denoising_packing_enabled runtime state"
        )
    return bool(value)


class PrefixDenoisingObjectiveMixin:
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        if not isinstance(inputs, MutableMapping):
            raise TypeError("prefix_denoising objective requires dict-like inputs")
        maybe_pop_and_stash_batch_extras(self, inputs)
        hybrids = inputs.pop("prefix_denoising_hybrid", None)
        if hybrids is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("prefix_denoising requires labels tensor")
        segment_meta = tuple(inputs.pop("prefix_denoising_segment_meta", ()))
        if not segment_meta:
            raise ValueError("prefix_denoising requires prefix_denoising_segment_meta sidecar")
        boundary_map = inputs.pop("packed_hybrid_boundary_map", None)
        inputs.pop("prefix_denoising_resolved_kl_sites", None)
        packing_enabled = _resolve_prefix_denoising_packing_enabled(self)
        if packing_enabled and boundary_map is None:
            raise ValueError(
                "prefix_denoising packed training requires packed_hybrid_boundary_map sidecar"
            )
        segment_spans = tuple(
            PrefixDenoisingSegmentSpan(
                batch_index=int(item["batch_index"]),
                token_start=int(item["token_start"]),
                token_end=int(item["token_end"]),
                branch_id=str(item["branch_id"]),
                segment_id=str(item["segment_id"]),
            )
            for item in segment_meta
        )
        strip_non_model_detection_sidecars(inputs)
        ignored_keys = ["labels"]
        _, inputs_for_model, _ = prepare_forward_inputs(
            model=model,
            inputs=inputs,
            ignored_keys=ignored_keys,
            packing_enabled=packing_enabled,
            where="prefix_denoising",
        )
        outputs = model(**inputs_for_model)
        logits = outputs.logits
        ce = compute_branch_balanced_hard_ce(
            logits=logits,
            labels=labels,
            segment_spans=segment_spans,
        )
        acc = topk_accuracy_from_logits(logits=logits, labels=labels, segment_spans=segment_spans, topk=(1, 5))
        events = prefix_denoising_ce_events(
            ce_balanced=float(ce.loss.detach().cpu().item()),
            ce_clean=float(ce.clean_ce.detach().cpu().item()),
            ce_noisy=float(ce.noisy_ce.detach().cpu().item()),
            ce_token_pooled=float(ce.token_pooled_ce.detach().cpu().item()),
            token_top1=acc[1],
            token_top5=acc[5],
            clean_denominator=ce.clean_denominator,
            noisy_denominator=ce.noisy_denominator,
        )
        flat = flatten_metric_events(events)
        flat["llm_loss"] = float(ce.loss.detach().cpu().item())
        SwiftMetricReporter(self).update_many(flat)
        return (ce.loss, outputs) if return_outputs else ce.loss
```

Use `SwiftMetricReporter(self).update_many(flat)` for required monitors; do not introduce a private pending-list logger unless a real consumer is added and tested. Do not use `getattr(self, "_packing_enabled", lambda: False)()` in this mixin: `_packing_enabled` is a Stage-2 runtime helper, and falling back to `False` would skip the packed Qwen position-id assertion for V1.

- [ ] **Step 7: Compose trainer only when prefix-denoising is enabled**

In `src/trainers/metrics/mixins.py`, export `PrefixDenoisingObjectiveMixin`.

In `src/bootstrap/trainer_setup.py::compose_trainer_class`, add `prefix_denoising_cfg` and `prefix_denoising_runtime` as explicit inputs. If `prefix_denoising_cfg.enabled` is true, compose `PrefixDenoisingObjectiveMixin` instead of `TeacherForcingObjectiveMixin`; they are mutually exclusive owners of the token loss. The dynamically composed class must carry `prefix_denoising_packing_enabled = bool(prefix_denoising_runtime["packing_enabled"])`. Keep the existing `TeacherForcingObjectiveMixin` path unchanged for non-V1 configs. Add a test proving the enabled prefix path yields `issubclass(trainer_cls, PrefixDenoisingObjectiveMixin)`, does not also attach incompatible objective mixins, and raises if `prefix_denoising_packing_enabled` is absent while `prefix_denoising.enabled` is true.

- [ ] **Step 8: Run CE and metric tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_loss.py tests/test_prefix_denoising_metrics.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit CE-only loss and metrics**

Run:

```bash
git add src/detection/prefix_denoising/loss.py src/detection/prefix_denoising/metrics.py src/trainers/metrics/prefix_denoising.py src/trainers/metrics/mixins.py src/bootstrap/trainer_setup.py src/sft.py tests/test_prefix_denoising_loss.py tests/test_prefix_denoising_metrics.py tests/test_training_runtime_sft_integration.py
git commit -m "feat: add prefix denoising hard ce loss"
```

## Task 6: Sparse Local-Window KL

**Files:**
- Modify: `src/detection/prefix_denoising/loss.py`
- Modify: `src/detection/prefix_denoising/metrics.py`
- Modify: `src/trainers/metrics/prefix_denoising.py`
- Modify: `tests/test_prefix_denoising_loss.py`
- Modify: `tests/test_prefix_denoising_metrics.py`

- [ ] **Step 1: Add failing KL support and numerics tests**

Append to `tests/test_prefix_denoising_loss.py`:

```python
from src.detection.prefix_denoising.loss import (
    compute_local_coord_kl,
    coord_support_window,
)
from src.detection.prefix_denoising.types import PrefixDenoisingKLSite
from src.detection.prefix_denoising.types import ResolvedPrefixDenoisingKLSite


def test_coord_support_window_clips_at_edges() -> None:
    assert coord_support_window(clean_bin=2, radius=4, coord_min=0, coord_max=999) == (0, 1, 2, 3, 4, 5, 6)
    assert coord_support_window(clean_bin=998, radius=4, coord_min=0, coord_max=999) == (994, 995, 996, 997, 998, 999)


def test_local_coord_kl_maps_bins_to_coord_token_ids_and_detaches_teacher() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((2, 6, 2100), dtype=torch.float32, requires_grad=True)
    noisy_logits = torch.zeros((2, 6, 2100), dtype=torch.float32, requires_grad=True)
    clean_logits.data[1, 1, 1010] = 5.0
    noisy_logits.data[1, 1, 1010] = 3.0
    clean_logits.data[0, 1, 1010] = -20.0  # wrong if the site silently defaults to row 0
    clean_logits.data[1, 1, 10] = 30.0  # wrong if bin 10 is used as a vocab column
    site = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=1,
        noisy_batch_index=1,
        object_index=0,
        coord_slot="y1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=tuple(range(8, 13)),
        identical_prefix=False,
    )

    result = compute_local_coord_kl(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        sites=(site,),
        coord_token_ids=coord_token_ids,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert clean_logits.grad is None
    assert noisy_logits.grad is not None
    assert result.candidate_site_count == 1
    assert result.effective_site_count == 1


def test_k_two_builds_eight_candidate_sites_without_rescaling_raw_kl() -> None:
    # Builder fixture must use three objects and num_objects_per_image=2.
    sample = build_three_object_hybrid_fixture(num_objects_per_image=2, epoch=3, seed=17)
    assert len(sample.kl_sites) == 8
    assert len({site.object_index for site in sample.kl_sites}) == 2
    for object_index in {site.object_index for site in sample.kl_sites}:
        assert {site.coord_slot for site in sample.kl_sites if site.object_index == object_index} == {"x1", "y1", "x2", "y2"}

    # Loss fixture must prove raw KL is a mean over sites, not a sum.
    result_one_group = compute_uniform_fixture_kl(site_count=4)
    result_two_groups = compute_uniform_fixture_kl(site_count=8)
    torch.testing.assert_close(result_two_groups.raw_loss, result_one_group.raw_loss)
```

In that test file, implement `build_three_object_hybrid_fixture` as a local helper that exercises the real builder with three clean objects, `num_objects_per_image=2`, and deterministic seed/epoch; implement `compute_uniform_fixture_kl` as a local helper that creates repeated identical `ResolvedPrefixDenoisingKLSite` rows with explicit batch indices and fixed logits. These helpers are test scaffolds, not production APIs.

- [ ] **Step 2: Run KL tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_loss.py -q
```

Expected: FAIL because KL helpers do not exist.

- [ ] **Step 3: Implement KL helpers**

Add to `src/detection/prefix_denoising/loss.py`:

Import `ResolvedPrefixDenoisingKLSite` from `src.detection.prefix_denoising.types`; the loss helper must not accept unresolved builder-local `PrefixDenoisingKLSite` objects.

```python
@dataclass(frozen=True)
class PrefixDenoisingKLResult:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    candidate_site_count: int
    effective_site_count: int
    identical_prefix_site_count: int
    teacher_support_mass: float
    student_support_mass: float
    teacher_gt_prob_full_coord_vocab: float
    student_gt_prob_full_coord_vocab: float
    teacher_gt_prob_conditional: float
    student_gt_prob_conditional: float


def coord_support_window(
    *,
    clean_bin: int,
    radius: int,
    coord_min: int = 0,
    coord_max: int = 999,
) -> tuple[int, ...]:
    start = max(int(coord_min), int(clean_bin) - int(radius))
    end = min(int(coord_max), int(clean_bin) + int(radius))
    return tuple(range(start, end + 1))


def compute_local_coord_kl(
    *,
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    sites: tuple[ResolvedPrefixDenoisingKLSite, ...],
    coord_token_ids: torch.Tensor,
) -> PrefixDenoisingKLResult:
    if not sites:
        zero = noisy_logits.float().sum() * 0.0
        return PrefixDenoisingKLResult(
            loss=zero,
            raw_loss=zero,
            candidate_site_count=0,
            effective_site_count=0,
            identical_prefix_site_count=0,
            teacher_support_mass=0.0,
            student_support_mass=0.0,
            teacher_gt_prob_full_coord_vocab=0.0,
            student_gt_prob_full_coord_vocab=0.0,
            teacher_gt_prob_conditional=0.0,
            student_gt_prob_conditional=0.0,
        )
    losses: list[torch.Tensor] = []
    teacher_support_mass_values: list[float] = []
    student_support_mass_values: list[float] = []
    teacher_gt_full_values: list[float] = []
    student_gt_full_values: list[float] = []
    teacher_gt_values: list[float] = []
    student_gt_values: list[float] = []
    identical = 0
    coord_token_ids = coord_token_ids.to(device=noisy_logits.device, dtype=torch.long)
    for site in sites:
        if not isinstance(site.clean_batch_index, int) or not isinstance(site.noisy_batch_index, int):
            raise TypeError("resolved KL sites must carry explicit clean/noisy batch indices")
        clean_row = int(site.clean_label_position) - 1
        noisy_row = int(site.noisy_label_position) - 1
        support_bins = torch.tensor(tuple(int(v) for v in site.support_bins), device=noisy_logits.device, dtype=torch.long)
        support_token_ids = coord_token_ids.index_select(0, support_bins)
        gt_index = tuple(int(v) for v in site.support_bins).index(int(site.clean_gt_bin))
        gt_token_id = int(coord_token_ids[int(site.clean_gt_bin)].detach().cpu().item())
        clean_batch_index = int(site.clean_batch_index)
        noisy_batch_index = int(site.noisy_batch_index)
        teacher_full = torch.softmax(clean_logits[clean_batch_index, clean_row].detach().float(), dim=-1)
        student_full = torch.softmax(noisy_logits[noisy_batch_index, noisy_row].float(), dim=-1)
        teacher_support_mass_values.append(float(teacher_full.index_select(0, support_token_ids).sum().detach().cpu().item()))
        student_support_mass_values.append(float(student_full.index_select(0, support_token_ids).sum().detach().cpu().item()))
        teacher_gt_full_values.append(float(teacher_full[gt_token_id].detach().cpu().item()))
        student_gt_full_values.append(float(student_full[gt_token_id].detach().cpu().item()))
        teacher_local_logits = clean_logits[clean_batch_index, clean_row].detach().float().index_select(0, support_token_ids)
        student_local_logits = noisy_logits[noisy_batch_index, noisy_row].float().index_select(0, support_token_ids)
        teacher_prob = torch.softmax(teacher_local_logits, dim=-1)
        student_log_prob = torch.log_softmax(student_local_logits, dim=-1)
        student_prob = torch.softmax(student_local_logits, dim=-1)
        losses.append(torch.sum(teacher_prob * (torch.log(teacher_prob.clamp_min(1e-12)) - student_log_prob)))
        teacher_gt_values.append(float(teacher_prob[gt_index].detach().cpu().item()))
        student_gt_values.append(float(student_prob[gt_index].detach().cpu().item()))
        if bool(site.identical_prefix):
            identical += 1
    raw = torch.stack(losses).mean()
    return PrefixDenoisingKLResult(
        loss=raw,
        raw_loss=raw,
        candidate_site_count=len(sites),
        effective_site_count=len(sites),
        identical_prefix_site_count=identical,
        teacher_support_mass=sum(teacher_support_mass_values) / len(teacher_support_mass_values),
        student_support_mass=sum(student_support_mass_values) / len(student_support_mass_values),
        teacher_gt_prob_full_coord_vocab=sum(teacher_gt_full_values) / len(teacher_gt_full_values),
        student_gt_prob_full_coord_vocab=sum(student_gt_full_values) / len(student_gt_full_values),
        teacher_gt_prob_conditional=sum(teacher_gt_values) / len(teacher_gt_values),
        student_gt_prob_conditional=sum(student_gt_values) / len(student_gt_values),
    )
```

- [ ] **Step 4: Add KL metric events**

Extend `src/detection/prefix_denoising/metrics.py` with KL keys and an event helper for:

- `prefix_denoising/kl/local_window/raw`;
- `prefix_denoising/kl/local_window/weighted`;
- `prefix_denoising/kl/local_window/candidate_site_count`;
- `prefix_denoising/kl/local_window/site_count`;
- `prefix_denoising/kl/local_window/identical_prefix_site_count`;
- `prefix_denoising/kl/local_window/teacher_support_mass`;
- `prefix_denoising/kl/local_window/student_support_mass`;
- `prefix_denoising/kl/local_window/teacher_gt_prob_full_coord_vocab`;
- `prefix_denoising/kl/local_window/student_gt_prob_full_coord_vocab`;
- `prefix_denoising/kl/local_window/teacher_gt_prob_conditional`;
- `prefix_denoising/kl/local_window/student_gt_prob_conditional`;
- `prefix_denoising/kl/local_window/support_bin_count`;
- `prefix_denoising/kl/local_window/edge_truncation_rate`;
- `prefix_denoising/kl/local_window/teacher_top1_is_gt`;
- `prefix_denoising/kl/local_window/student_top1_is_gt`;
- slot-specific reduced keys for `x1`, `y1`, `x2`, and `y2` for support mass, full-vocab GT probability, local GT probability, and teacher-minus-student deltas.

Use distinct flat keys, not channel-only identity differences.

- [ ] **Step 5: Wire KL into trainer mixin**

In `src/trainers/metrics/prefix_denoising.py`:

- extract KL sites from the hybrid sidecar;
- skip KL construction entirely when weight is zero;
- source the coordinate-token id row from
  `src.tokens.coord.codec.get_coord_token_ids(tokenizer, validate=True)` using
  the trainer/template tokenizer, convert it once to a `torch.long` tensor on
  the logits device, and pass that tensor into `compute_local_coord_kl`;
- compute KL after model forward when weight is positive;
- total loss is `ce.loss + weight * kl.raw_loss`;
- log raw and weighted KL separately;
- publish `llm_loss` as the optimized scalar actually backpropagated: `ce.loss + weight * kl.raw_loss`;
- also log `prefix_denoising/global/loss/ce_balanced`, `prefix_denoising/kl/local_window/raw`, and `prefix_denoising/kl/local_window/weighted` separately so CE-only and KL-on runs remain comparable.

- [ ] **Step 6: Run KL tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_loss.py tests/test_prefix_denoising_metrics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit KL objective**

Run:

```bash
git add src/detection/prefix_denoising/loss.py src/detection/prefix_denoising/metrics.py src/trainers/metrics/prefix_denoising.py tests/test_prefix_denoising_loss.py tests/test_prefix_denoising_metrics.py
git commit -m "feat: add prefix denoising local coord kl"
```

## Task 7: Hybrid Packing And Boundary Map

**Files:**
- Create: `src/detection/prefix_denoising/packing.py`
- Modify: `src/detection/prefix_denoising/types.py`
- Modify: `src/detection/prefix_denoising/dataset.py`
- Modify: `src/sft.py`
- Modify: `src/detection/packing.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `tests/test_prefix_denoising_packing.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Write failing packing tests**

Create `tests/test_prefix_denoising_packing.py`:

```python
from __future__ import annotations

import torch

from src.detection.prefix_denoising.packing import (
    build_hybrid_pack_plan,
    materialize_packed_hybrid_boundary_map,
    maybe_attach_packed_hybrid_boundary_map,
)
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingSegment,
)


def _segment(segment_id: str, branch_id: str, length: int) -> PrefixDenoisingSegment:
    ids = tuple(range(length))
    return PrefixDenoisingSegment(
        segment_id=segment_id,
        branch_id=branch_id,
        input_ids=ids,
        labels=ids,
        attention_mask=tuple(1 for _ in ids),
        supervised_positions=tuple(range(1, length)),
        ce_denominator=length - 1,
        metadata={},
    )


def _hybrid(sample_id: str, clean_len: int, noisy_len: int) -> HybridPrefixDenoisingSample:
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=sample_id,
        clean_full=_segment(f"{sample_id}:clean", "clean_full", clean_len),
        noisy_full=_segment(f"{sample_id}:noisy", "noisy_full", noisy_len),
        kl_sites=(),
    )


def test_hybrid_pack_plan_keeps_hybrid_sample_atomic() -> None:
    plan = build_hybrid_pack_plan(
        samples=[_hybrid("a", 10, 10), _hybrid("b", 12, 12)],
        global_max_length=30,
    )

    assert len(plan.rows) == 2
    assert [item.hybrid_sample_id for item in plan.rows[0].items] == ["a"]
    assert [item.hybrid_sample_id for item in plan.rows[1].items] == ["b"]


def test_hybrid_pack_plan_packs_multiple_complete_hybrids_when_they_fit() -> None:
    plan = build_hybrid_pack_plan(
        samples=[_hybrid("a", 5, 5), _hybrid("b", 6, 6)],
        global_max_length=25,
    )

    assert len(plan.rows) == 1
    assert [item.hybrid_sample_id for item in plan.rows[0].items] == ["a", "b"]


def test_hybrid_pack_plan_records_overlength_exclusion() -> None:
    plan = build_hybrid_pack_plan(
        samples=[_hybrid("a", 20, 20)],
        global_max_length=30,
    )

    assert plan.rows == ()
    assert plan.exclusions["overlength_hybrid_sample"] == 1


def test_hybrid_pack_plan_allows_exact_global_max_length() -> None:
    plan = build_hybrid_pack_plan(
        samples=[_hybrid("a", 15, 15)],
        global_max_length=30,
    )

    assert len(plan.rows) == 1
    assert plan.rows[0].total_length == 30
    assert plan.exclusions == {}


def test_boundary_map_rewrites_two_hybrids_in_one_physical_row() -> None:
    plan = build_hybrid_pack_plan(
        samples=[_hybrid("a", 5, 5), _hybrid("b", 6, 6)],
        global_max_length=25,
    )
    boundary = materialize_packed_hybrid_boundary_map(plan.rows[0])

    assert {item.branch_id for item in boundary.boundaries} == {"clean_full", "noisy_full"}
    assert all(item.token_start < item.token_end for item in boundary.boundaries)
    assert boundary.ce_denominator_by_branch["clean_full"] > 0
    assert boundary.ce_denominator_by_branch["noisy_full"] > 0
    assert boundary.position_reset_offsets
    assert boundary.image_placeholder_owners
    assert boundary.visual_slice_owners


def test_boundary_map_producer_attaches_after_template_collate() -> None:
    collated = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3, 4, -100, 6, 7, 8]], dtype=torch.long),
        "position_ids": torch.arange(8, dtype=torch.long).view(1, 1, 8).repeat(3, 1, 1),
        "text_position_ids": torch.arange(8, dtype=torch.long).view(1, 8),
        "pixel_values": torch.zeros((2, 4), dtype=torch.float32),
        "image_grid_thw": torch.ones((2, 3), dtype=torch.long),
    }
    raw_batch = [[{"prefix_denoising_hybrid": _hybrid("a", 4, 4)}]]

    maybe_attach_packed_hybrid_boundary_map(collated=collated, raw_batch=raw_batch)

    assert "packed_hybrid_boundary_map" in collated
    boundary = collated["packed_hybrid_boundary_map"]
    assert boundary.segment_meta
    assert "prefix_denoising_segment_meta" in collated
    assert "prefix_denoising_resolved_kl_sites" in collated
```

- [ ] **Step 2: Run packing tests and verify the intended failure**

Run:

```bash
python -m pytest tests/test_prefix_denoising_packing.py -q
```

Expected: FAIL because `build_hybrid_pack_plan` does not exist.

- [ ] **Step 3: Add boundary and plan dataclasses**

Extend `src/detection/prefix_denoising/types.py`:

```python
@dataclass(frozen=True)
class PackedHybridBoundary:
    packed_row_index: int
    hybrid_sample_id: str
    segment_id: str
    branch_id: PrefixDenoisingBranchId
    token_start: int
    token_end: int
    supervised_start: int
    supervised_end: int
    label_position_offset: int
    logit_position_offset: int
    image_placeholder_start: int
    image_placeholder_end: int
    pixel_values_slice: tuple[int, int]
    image_grid_thw_slice: tuple[int, int]
    ce_denominator: int
    kl_site_count: int


@dataclass(frozen=True)
class PackedHybridBoundaryMap:
    packed_row_index: int
    boundaries: tuple[PackedHybridBoundary, ...]
    position_reset_offsets: tuple[int, ...]
    varlen_cu_seqlens: tuple[int, ...]
    ce_denominator_by_branch: Mapping[str, int]
    image_placeholder_owners: Mapping[tuple[int, int], str]
    visual_slice_owners: Mapping[tuple[int, int], str]
    segment_meta: tuple[Mapping[str, object], ...]
    resolved_kl_sites: tuple[ResolvedPrefixDenoisingKLSite, ...]
```

- [ ] **Step 4: Implement pack planner and post-collate boundary producer**

Create `src/detection/prefix_denoising/packing.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from src.detection.prefix_denoising.types import HybridPrefixDenoisingSample


@dataclass(frozen=True)
class HybridPackItem:
    hybrid_sample_id: str
    length: int


@dataclass(frozen=True)
class HybridPackRow:
    items: tuple[HybridPackItem, ...]
    total_length: int


@dataclass(frozen=True)
class HybridPackPlan:
    rows: tuple[HybridPackRow, ...]
    exclusions: dict[str, int]


def build_hybrid_pack_plan(
    *,
    samples: Sequence[HybridPrefixDenoisingSample],
    global_max_length: int,
) -> HybridPackPlan:
    rows: list[HybridPackRow] = []
    current: list[HybridPackItem] = []
    current_length = 0
    exclusions: Counter[str] = Counter()
    cap = int(global_max_length)
    for sample in samples:
        length = int(sample.total_length)
        if not sample.ok:
            exclusions[str(sample.skip_reason or "ineligible_hybrid_sample")] += 1
            continue
        if length > cap:
            exclusions["overlength_hybrid_sample"] += 1
            continue
        item = HybridPackItem(hybrid_sample_id=sample.hybrid_sample_id, length=length)
        if current and current_length + length > cap:
            rows.append(HybridPackRow(items=tuple(current), total_length=current_length))
            current = []
            current_length = 0
        current.append(item)
        current_length += length
    if current:
        rows.append(HybridPackRow(items=tuple(current), total_length=current_length))
    return HybridPackPlan(rows=tuple(rows), exclusions=dict(exclusions))
```

In the same module, add the post-collate producer that owns the live tensor
seam. The existing static packing wrapper groups raw dataset samples into a
`list[dict]` pack and sets template packing flags; the ms-swift/template collator
then flattens model tensors. Prefix-denoising must attach
`PackedHybridBoundaryMap` after `collate_fn(batch)` has produced `input_ids`,
`labels`, `position_ids` / `text_position_ids`, `pixel_values`, and
`image_grid_thw`, but before enrichers and the trainer loss consume prefix
sidecars:

```python
def maybe_attach_packed_hybrid_boundary_map(
    *,
    collated: dict[str, Any],
    raw_batch: Sequence[Any],
) -> None:
    if not _raw_batch_has_prefix_denoising_pack(raw_batch):
        return
    if "input_ids" not in collated or "labels" not in collated:
        raise ValueError("prefix_denoising packed boundary map requires collated input_ids and labels")
    if "position_ids" not in collated:
        raise ValueError("prefix_denoising packed boundary map requires collated position_ids")
    if "pixel_values" not in collated or "image_grid_thw" not in collated:
        raise ValueError("prefix_denoising packed boundary map requires pixel_values and image_grid_thw")
    boundary = build_packed_hybrid_boundary_map_from_collated(
        collated=collated,
        raw_batch=raw_batch,
    )
    collated["packed_hybrid_boundary_map"] = boundary
    collated["prefix_denoising_segment_meta"] = boundary.segment_meta
    collated["prefix_denoising_resolved_kl_sites"] = boundary.resolved_kl_sites
```

`build_packed_hybrid_boundary_map_from_collated` must inspect the flattened
physical batch emitted by the active template collator; it must not infer model
tensor offsets from the assignment plan alone. The boundary-map tests must cover
at least two hybrid samples in one physical packed row with nonempty KL sites and
must assert:

- CE label positions are mapped to causal logit rows by `label_position - 1` within the same segment boundary;
- KL clean/noisy label positions and support metadata are rewritten to physical batch/logit positions and emitted as `ResolvedPrefixDenoisingKLSite` objects with explicit clean/noisy batch indices;
- branch ids and segment ids survive flattening;
- position reset offsets and FlashAttention varlen metadata agree with segment boundaries when the active packed runtime emits varlen fields;
- image placeholders, `pixel_values`, and `image_grid_thw` slices have explicit segment owners;
- CE/KL denominators are preserved after packing.
- a dummy packed-isolation probe would fail if noisy-segment logits can attend
  to clean-segment coordinate-token edits across a segment boundary.

- [ ] **Step 5: Integrate with static packing runtime**

In `src/sft.py` and `src/detection/packing.py`:

- add prefix-denoising fingerprint fields: schema version, noise config, KL weight, KL window radius, `num_objects_per_image`, eligibility policy;
- allow `training.packing=true` and `packing.static_packing=true` only when `prefix_denoising.enabled=true`, the dataset is `PrefixDenoisingTrainingDataset`, and the collated batch carries `PackedHybridBoundaryMap`;
- continue rejecting non-V1 teacher-forcing/recursive sidecar packing;
- record `overlength_hybrid_sample` exclusions in runtime artifacts.
- inspect the actual packed batch for the repo's varlen/position-boundary fields and record them in launch-health artifacts. A plain 2D attention mask is not sufficient evidence for sidecar-active packed prefix denoising.

In `src/data_collators/batch_extras_collator.py`, import and call
`maybe_attach_packed_hybrid_boundary_map(collated=collated, raw_batch=batch)`
immediately after `collated = collate_fn(batch)` and before
`DatasetMetaEnricher`. This is the only V1 boundary-map insertion seam.

- [ ] **Step 6: Run packing tests**

Run:

```bash
python -m pytest tests/test_prefix_denoising_packing.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit hybrid packing**

Run:

```bash
git add src/detection/prefix_denoising/packing.py src/detection/prefix_denoising/types.py src/detection/prefix_denoising/dataset.py src/sft.py src/detection/packing.py src/data_collators/batch_extras_collator.py tests/test_prefix_denoising_packing.py tests/test_stage1_static_packing_runtime_config.py
git commit -m "feat: add prefix denoising hybrid packing contract"
```

## Task 8: Config Leaves And Docs

**Files:**
- Create: `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml`
- Create: `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml`
- Create: `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`
- Create: `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`
- Modify: `configs/stage1/detection_teacher_forcing/README.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/catalog.yaml`
- Modify: `progress/index.yaml`

- [ ] **Step 1: Add CE-only production config**

Create `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml` by extending `configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml`, then make these required changes:

```yaml
model:
  model: /data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
  model_type: qwen3_vl

training:
  artifact_subdir: compact_full_prefix_denoising_ce_only_bsz1x128_4epoch
  run_name: compact-full-prefix-denoising-ce-only-bsz1x128-4epoch
  per_device_train_batch_size: 1
  effective_batch_size: 128
  per_device_eval_batch_size: 1
  train_type: lora
  use_dora: true
  freeze_llm: false
  freeze_vit: true
  freeze_aligner: true
  target_modules: [all-linear]
  lora_rank: 16
  lora_alpha: 32
  packing: true
  packing_mode: static
  eval_packing: false
  encoded_sample_cache:
    enabled: false

data:
  train_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  val_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  image_root: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60
  object_ordering: sorted

objective:
  id: teacher_forcing
  profile: hard_sft
  target_ir: null
  modules:
    token_type_mass:
      enabled: false
    conditional_valid_set_likelihood:
      enabled: false
    within_valid_coverage:
      enabled: false
      coverage_strength: 0.0
    continuation_margin:
      enabled: false

prefix_denoising:
  enabled: true
  noise:
    center_shift_frac: 0.08
    uniform_scale_range: [0.92, 1.08]
  current_object_kl:
    weight: 0.0
    window_radius: 8
    num_objects_per_image: 1

packing:
  static_packing: true
  padding_free_packed: false
```

- [ ] **Step 2: Add KL production config**

Create `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml` by extending the CE-only production config and overriding only the run names and KL weight:

```yaml
training:
  artifact_subdir: compact_full_prefix_denoising_kl_w0p05_bsz1x128_4epoch
  run_name: compact-full-prefix-denoising-kl-w0p05-bsz1x128-4epoch

prefix_denoising:
  current_object_kl:
    weight: 0.05
```

Keep the rest identical to the CE-only config except for run/artifact names and KL weight.

- [ ] **Step 3: Add tiny smoke configs**

Create:

- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`
- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`

Each smoke leaf should extend the matching production config plus `smoke/common_prodlike.yaml`, then override:

```yaml
training:
  max_steps: 2
  per_device_train_batch_size: 1
  effective_batch_size: 1
  per_device_eval_batch_size: 1
  packing_min_fill_ratio: 0.1
  packing_drop_last: false
  packing_length_precompute_workers: 1
  static_packing_cache:
    root_dir: temp/static_packing_smoke/<leaf-name>

debug:
  enabled: true
  train_sample_limit: 4
  val_sample_limit: 1
```

The smoke configs are launch-health leaves, not clean baselines.

- [ ] **Step 4: Add docs routing and metric docs**

Update `configs/stage1/detection_teacher_forcing/README.md` with:

```markdown
## Prefix-Denoising SFT V1

Prefix-denoising V1 uses hard clean-label CE only with `objective.id:
teacher_forcing`, `objective.profile: hard_sft`, all teacher-forcing auxiliary
modules disabled, and `prefix_denoising.enabled: true`.
It requires `data.object_ordering: sorted`, coord-token compact-full `xyxy`,
encoded-sample cache disabled, and the V1 hybrid static packing path.

First launch-health order:

1. `smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`
2. `smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`
3. `prod/compact_full_prefix_denoising_ce_only.yaml` and
   `prod/compact_full_prefix_denoising_kl_w0p05.yaml` after smoke verification
```

Update `docs/training/METRICS.md` with the prefix-denoising metric key families from the design spec.

Update `docs/catalog.yaml` and `progress/index.yaml` only if the implementation creates new stable docs or changes router coverage.

- [ ] **Step 5: Parse configs**

Run:

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig

paths = [
    "configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml",
    "configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml",
    "configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml",
    "configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml",
]
for path in paths:
    cfg = ConfigLoader.load_materialized_training_config(path)
    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.prefix_denoising.enabled is True
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.data.object_ordering == "sorted"
    assert str(cfg.data.train_jsonl).startswith("/data/CoordExp/public_data/")
    assert str(cfg.data.val_jsonl).startswith("/data/CoordExp/public_data/")
    assert str(cfg.data.image_root).startswith("/data/CoordExp/public_data/")
    assert bool(cfg.training.get("packing", False)) is True
    assert "target_ir" not in cfg.to_mapping()["objective"]
    print(f"{path}: ok")
PY
```

Expected: all four configs parse and print `ok`.

- [ ] **Step 6: Commit configs and docs**

Run:

```bash
git add configs/stage1/detection_teacher_forcing docs/training/METRICS.md docs/catalog.yaml progress/index.yaml
git commit -m "docs: add prefix denoising launch configs"
```

## Task 9: Tiny Packed Launch-Health Verification

**Files:**
- No required code edits unless prior tests expose a bug.
- Artifact roots under `outputs/` are runtime artifacts and should not be committed.

- [ ] **Step 1: Run CE-only packed smoke**

Run:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml
```

Expected:

- training launches without schema/runtime guard failure;
- packed hybrid path is active in effective runtime;
- `prefix_denoising/global/loss/ce_balanced` is finite;
- `prefix_denoising/clean_full/loss/ce` is finite;
- `prefix_denoising/noisy_full/loss/ce` is finite;
- `prefix_denoising/kl/local_window/raw` is absent or zero because KL weight is zero;
- `llm_loss` is present;
- top-1 and top-5 token accuracy are present;
- skip counters are present, including `zero_object_hybrid_sample`,
  `noise_infeasible_4coord_changed`, and overlength hybrid exclusions;
- noising difficulty is summarized by clean bbox width/height bins so small/thin
  object filtering is visible before interpreting the smoke as full-data
  launch-health.

- [ ] **Step 2: Run KL-on packed smoke**

Run:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml
```

Expected:

- all CE-only smoke expectations still hold;
- `prefix_denoising/kl/local_window/raw` is finite;
- `prefix_denoising/kl/local_window/weighted` is finite;
- KL site counts are positive for rows with eligible objects;
- teacher/student support-mass diagnostics are present;
- no invalid bbox warnings dominate the tiny run.
- if skip rate is high, the report recommends larger explicit noise envelope or
  direct feasible-set sampling rather than the milder-noise fallback; if skip
  rate is acceptable but noisy CE/KL is too hard, the report recommends the
  milder-noise fallback, lower KL weight, wider KL window, or CE-only training
  continuation.

- [ ] **Step 3: Summarize launch-health artifacts**

Create a short progress note only after both smokes run:

```text
progress/diagnostics/YYYY-MM-DD_prefix_denoising_launch_health.md
```

Include:

- scope: `tiny`;
- config paths;
- checkpoint source;
- artifact roots;
- exact command lines;
- CE metrics;
- KL metrics;
- token top-1/top-5;
- noising skip counters;
- skip rate by clean bbox width/height bins and by skip reason;
- chosen fallback interpretation, explicitly one of `high_skip_rate`,
  `too_hard_noise`, `healthy_smoke`, or `blocked_runtime`;
- packing runtime payload;
- parse/drop counters if eval ran;
- statement that this is launch-health, not rollout/exposure-bias evidence.

- [ ] **Step 4: Commit launch-health note**

Run:

```bash
git add progress/diagnostics/YYYY-MM-DD_prefix_denoising_launch_health.md
git commit -m "docs: record prefix denoising launch health"
```

## Task 10: Final Verification Gate

**Files:**
- All files touched by implementation tasks.

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
python -m pytest \
  tests/test_prefix_denoising_config_contract.py \
  tests/test_prefix_denoising_geometry.py \
  tests/test_prefix_denoising_builder.py \
  tests/test_prefix_denoising_collator.py \
  tests/test_prefix_denoising_loss.py \
  tests/test_prefix_denoising_metrics.py \
  tests/test_prefix_denoising_packing.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_teacher_forcing_config_contract.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run config/docs parse checks**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml
for path in ["docs/catalog.yaml", "progress/index.yaml"]:
    if Path(path).exists():
        yaml.safe_load(Path(path).read_text())
        print(f"{path}: ok")
PY
```

Expected:

```text
docs/catalog.yaml: ok
progress/index.yaml: ok
```

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Run smoke configs**

Run the commands from Task 9. If hardware is unavailable, record the skip reason and do not claim launch-health success.

- [ ] **Step 5: Produce final implementation summary**

The final implementation report must include:

- changed files grouped by schema, geometry, builder, loss, packing, configs, docs;
- exact tests run;
- exact smoke commands run or skipped;
- artifact roots for smokes;
- remaining risk: V1 launch-health does not prove rollout exposure-bias improvement;
- next recommended review: `review-convergence-loop` with architecture, numerics, packing, and config/governance lanes.

## Plan Self-Review

Spec coverage:

- Config schema and gates are covered by Task 1.
- Constructive valid bbox noising is covered by Task 2.
- Hybrid sample structure and two-segment builder are covered by Task 3.
- Zero-object/no-op rows are excluded by Task 3.
- Prefix sidecar registration and model-input stripping are covered by Task 4.
- Hard CE only is covered by Tasks 1 and 5.
- Sparse asymmetric local-window KL is covered by Task 6.
- Metrics and standard monitors are covered by Task 5 and Task 6.
- Hybrid packing, post-collate boundary-map production, packed row metadata,
  visual ownership, and position-boundary semantics are covered by Task 7.
- Config leaves and docs routing are covered by Task 8.
- Launch-health order and noising skip-rate interpretation are covered by Task 9.
- Final verification and evidence scope are covered by Task 10.

Placeholder scan:

- This plan intentionally contains no placeholder markers.
- This plan does not authorize implementation before review convergence.

Type consistency:

- `PrefixDenoisingConfig`, `PrefixDenoisingNoiseConfig`, and `PrefixDenoisingCurrentObjectKLConfig` are the schema names used throughout.
- `HybridPrefixDenoisingSample`, `PrefixDenoisingSegment`, `PrefixDenoisingKLSite`, and `PackedHybridBoundaryMap` are the runtime names used throughout.
- `ResolvedPrefixDenoisingKLSite` is the only KL-site type consumed by packed loss code.
- `PrefixDenoisingObjectiveMixin` is the trainer integration name used throughout.

## Review-Convergence Round 1 Resolutions

Accepted P0 fixes:

- Causal-LM alignment: CE, token accuracy, and KL label sites must use
  `labels[position] -> logits[position - 1]`; physical position `0` and each
  packed segment's first token are excluded from supervision.
- Coordinate KL support: `support_bins` are coordinate bins and must be mapped
  through coord-token ids before indexing full-vocab logits.

Accepted P1 fixes:

- Config surface stays on canonical `objective.id: teacher_forcing` with
  `objective.profile: hard_sft`; legacy `objective.id: sft` is not revived.
- OpenSpec or explicit experiment-only governance gate is required before schema
  and metric contracts are implemented.
- Dataset/collator tasks now require model-ready fields, actual collator
  integration, `PackedHybridBoundaryMap`, and a dummy-loss integration test.
- Packing tasks now require exact-cap inclusion, pre-plan skip filtering,
  two-hybrid boundary-map rewriting, position/varlen boundary evidence, and
  visual ownership checks.
- Trainer composition is owned by `src/bootstrap/trainer_setup.py`.
- Metrics use `SwiftMetricReporter`; no private pending-log list is allowed
  without a tested consumer.
- KL and CE helpers must aggregate all clean/noisy segments in packed batches
  and include K>1 selected-object coverage tests.
- Packed KL sites must be resolved by the boundary map into
  `ResolvedPrefixDenoisingKLSite` objects with explicit physical batch indices;
  unresolved builder-local KL sites must not default to batch row `0`.

Accepted P2 fixes:

- `llm_loss` is the optimized total scalar actually backpropagated.
- Catalog and spec phase metadata were refreshed.
- Config verification uses `ConfigLoader.load_materialized_training_config`.
- Zero-strength bbox noising respects the authored envelope and returns
  infeasible instead of applying an implicit one-bin shift.

## Review-Convergence Round 2 Closure

Reviewer lanes:

- Packing/data/collator/runtime lane: converged with no remaining blocking
  findings.
- Config/runtime/docs-governance lane: converged with no remaining blocking
  findings.
- Loss numerics/KL lane: initially held on packed KL site row resolution and
  weak K>1 fixture coverage; the plan now requires
  `ResolvedPrefixDenoisingKLSite`, forbids row-0 fallback, requires
  boundary-map emission of resolved sites, and requires a real three-object K>1
  fixture with duplicate-site mean-preservation coverage. Focused re-review
  confirmed convergence.

## External Audit Revision

An external read-only audit at
`progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md` was reviewed
critically after Round 2. Accepted roadmap revisions:

- sidecar boundary: Task 4 now edits `src/detection/dataset.py`, registers
  prefix-denoising sidecars, and requires `PrefixDenoisingObjectiveMixin` to use
  `strip_non_model_detection_sidecars` instead of a private ignored-key list;
- packed trainer state: Task 5 now forbids `_packing_enabled` fallback and
  requires explicit `prefix_denoising_packing_enabled` runtime state;
- packed materialization seam: Task 7 now names the post-collate
  `PackedHybridBoundaryMap` producer and wires it immediately after the template
  collator flattens a packed batch;
- packed sidecar carry-through: Task 4 now requires packed
  `prefix_denoising_hybrid` sidecars to be attached when a boundary map exists,
  instead of returning before the trainer can see them;
- config ambiguity: Task 1 now rejects explicit
  `objective.target_ir.rollin_policy` and raw `objective.coord_soft_ce` when
  prefix denoising is enabled;
- noise policy: Task 2 and Task 9 now separate high-skip-rate remedies from
  too-hard-noise remedies;
- zero-object policy: Task 3 now excludes zero-object rows with
  `skip_reason="zero_object_hybrid_sample"`;
- coord-token ids: Task 6 now sources coord-token ids from
  `src.tokens.coord.codec.get_coord_token_ids(tokenizer, validate=True)`.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
as the historical task plan for the V1 implementation on
`codex/prefix-denoising-sft`.

Implementation has already started and the first launch-health evidence is
recorded in
`progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`.

For future continuation work, two execution options remain available:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with batch execution and checkpoints.
