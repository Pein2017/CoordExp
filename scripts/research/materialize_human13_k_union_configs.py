#!/usr/bin/env python3
"""Strict CPU-only Human-13 arm-plan materializer.

The default path validates immutable experiment-local YAML and emits JSON.  It
does not import a model implementation, load weights, allocate a GPU, construct
an optimizer, run a forward, or write a checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "human13_arm_config.v1"
PLAN_SCHEMA_VERSION = "human13_materialized_plans.v1"
UNIT_ID = "2026-08-12-human13-k-union-to-greedy-overfit-screen"
PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_union")
MILESTONES = (0, 1, 2, 4, 8, 16)
GLOBAL_MAX_LENGTH = 12_000
ZERO_MODEL_ACTIONS = {
    "model_imports": 0,
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
    "gpu_allocations": 0,
}


class MaterializationError(ValueError):
    """Raised when an arm or sealed applicability binding is not exact."""


@dataclass(frozen=True)
class FrozenSource:
    checkpoint_path: str
    base_model_path: str
    adapter_path: str
    special_embedding_path: str
    adapter_sha256: str
    special_embedding_sha256: str


@dataclass(frozen=True)
class TrainableSurface:
    language_tower_dora: bool
    vision_tower: bool
    multimodal_aligner: bool
    token_embeddings: bool
    base_language_weights: bool


@dataclass(frozen=True)
class AdamWContract:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerContract:
    name: str
    warmup_steps: int
    horizon_updates: int


@dataclass(frozen=True)
class ArmConfig:
    schema_version: str
    unit_id: str
    arm_id: str
    arm_name: str
    updates: bool
    source: FrozenSource
    trainable_surface: TrainableSurface | None
    optimizer: AdamWContract | None
    scheduler: SchedulerContract | None
    max_grad_norm: float | None
    global_max_length: int
    coefficients: tuple[float, float, float]
    renormalize_active_families: bool
    milestones: tuple[int, ...]
    applicability: str


@dataclass(frozen=True)
class Human13A8CensusBinding:
    schema_version: str
    census_schema_version: str
    manifest_identity: Mapping[str, str]
    frozen_targets_sha256: str
    artifact_sha256: str
    applicable: bool
    blocked: bool
    block_reason: str | None
    required_margin: float | None
    violating_site_count: int
    maximum_absolute_margin_drift: float
    target_bytes_unchanged: bool


FROZEN_SOURCE = FrozenSource(
    checkpoint_path=(
        "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
        "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
        "checkpoints/step-2444"
    ),
    base_model_path=(
        "/data/Qwen3-VL/model_cache/models/Qwen/"
        "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
    ),
    adapter_path=(
        "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
        "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
        "checkpoints/step-2444/adapter"
    ),
    special_embedding_path=(
        "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
        "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
        "checkpoints/step-2444/special_token_embeddings"
    ),
    adapter_sha256="49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da",
    special_embedding_sha256=(
        "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
    ),
)
LANGUAGE_DORA_ONLY = TrainableSurface(
    language_tower_dora=True,
    vision_tower=False,
    multimodal_aligner=False,
    token_embeddings=False,
    base_language_weights=False,
)
FROZEN_ADAMW = AdamWContract(
    name="adamw_torch",
    learning_rate=1.0e-5,
    betas=(0.9, 0.999),
    epsilon=1.0e-8,
    weight_decay=0.0,
)
FROZEN_SCHEDULER = SchedulerContract(
    name="cosine_with_warmup",
    warmup_steps=0,
    horizon_updates=16,
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "arm_id",
        "arm_name",
        "updates",
        "source",
        "trainable_surface",
        "optimizer",
        "scheduler",
        "max_grad_norm",
        "packing",
        "family_coefficients",
        "renormalize_active_families",
        "milestones",
        "applicability",
    }
)
_ARM_ORDER = {
    "frozen_source": 0,
    "full_gt_capacity": 1,
    "A0": 2,
    "A1": 3,
    "A3": 4,
    "A4": 5,
    "A7": 6,
    "A8-prime": 7,
    "A6": 8,
}
_ARM_NAMES = {
    "frozen_source": "Frozen Source",
    "full_gt_capacity": "Full-GT capacity",
    "A0": "A0 shared no-H background",
    "A1": "A1 full-H chain CE",
    "A3": "A3 uniform H1 hub",
    "A4": "A4 any-valid mass",
    "A6": "A6 natural donor H1",
    "A7": "A7 no-preservation H1",
    "A8-prime": "A8-prime full-H bottleneck",
}
_COEFFICIENTS = {
    "frozen_source": (0.0, 0.0, 0.0),
    "full_gt_capacity": (1.0, 0.0, 0.0),
    "A0": (0.0, 1.0, 1.0),
    "A1": (1.0, 1.0, 1.0),
    "A3": (1.0, 1.0, 1.0),
    "A4": (1.0, 1.0, 1.0),
    "A6": (1.0, 1.0, 1.0),
    "A7": (1.0, 0.0, 1.0),
    "A8-prime": (1.0, 1.0, 1.0),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise MaterializationError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise MaterializationError(f"{label} is missing fields: {sorted(missing)}")


def _source_from_dict(value: Any) -> FrozenSource:
    raw = _mapping(value, "source")
    fields = set(FrozenSource.__dataclass_fields__)
    _exact_fields(raw, fields, "source")
    source = FrozenSource(**{field: str(raw[field]) for field in fields})
    if source != FROZEN_SOURCE:
        raise MaterializationError("source differs from the frozen Source identity")
    return source


def _surface_from_dict(value: Any) -> TrainableSurface:
    raw = _mapping(value, "trainable_surface")
    fields = set(TrainableSurface.__dataclass_fields__)
    _exact_fields(raw, fields, "trainable_surface")
    if any(not isinstance(raw[field], bool) for field in fields):
        raise MaterializationError("trainable_surface values must be boolean")
    surface = TrainableSurface(**{field: raw[field] for field in fields})
    if surface != LANGUAGE_DORA_ONLY:
        raise MaterializationError("trainable_surface must be language-only DoRA")
    return surface


def _optimizer_from_dict(value: Any) -> AdamWContract:
    raw = _mapping(value, "optimizer")
    fields = set(AdamWContract.__dataclass_fields__)
    _exact_fields(raw, fields, "optimizer")
    betas = raw["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise MaterializationError("optimizer.betas must contain exactly two values")
    optimizer = AdamWContract(
        name=str(raw["name"]),
        learning_rate=float(raw["learning_rate"]),
        betas=(float(betas[0]), float(betas[1])),
        epsilon=float(raw["epsilon"]),
        weight_decay=float(raw["weight_decay"]),
    )
    if optimizer != FROZEN_ADAMW:
        raise MaterializationError(
            "optimizer learning_rate/betas/epsilon/weight_decay drifted"
        )
    return optimizer


def _scheduler_from_dict(value: Any) -> SchedulerContract:
    raw = _mapping(value, "scheduler")
    fields = set(SchedulerContract.__dataclass_fields__)
    _exact_fields(raw, fields, "scheduler")
    scheduler = SchedulerContract(
        name=str(raw["name"]),
        warmup_steps=int(raw["warmup_steps"]),
        horizon_updates=int(raw["horizon_updates"]),
    )
    if scheduler != FROZEN_SCHEDULER:
        raise MaterializationError(
            "scheduler differs from the frozen 16-update contract"
        )
    return scheduler


def load_arm_config(path: str | Path) -> ArmConfig:
    config_path = Path(path)
    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    unknown = set(raw) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(raw)
    if unknown:
        raise MaterializationError(f"config has unknown fields: {sorted(unknown)}")
    if missing:
        raise MaterializationError(f"config is missing fields: {sorted(missing)}")
    arm_id = str(raw["arm_id"])
    if arm_id not in _ARM_ORDER:
        raise MaterializationError(f"arm_id is not approved: {arm_id}")
    if raw["arm_name"] != _ARM_NAMES[arm_id]:
        raise MaterializationError(f"arm_name drifted for {arm_id}")
    if raw["schema_version"] != SCHEMA_VERSION or raw["unit_id"] != UNIT_ID:
        raise MaterializationError("config schema_version or unit_id mismatches")
    if not isinstance(raw["updates"], bool):
        raise MaterializationError("updates must be boolean")
    updates = raw["updates"]
    source = _source_from_dict(raw["source"])

    if updates:
        surface = _surface_from_dict(raw["trainable_surface"])
        optimizer = _optimizer_from_dict(raw["optimizer"])
        scheduler = _scheduler_from_dict(raw["scheduler"])
        max_grad_norm = float(raw["max_grad_norm"])
        if max_grad_norm != 1.0:
            raise MaterializationError("max_grad_norm must equal 1.0")
    else:
        if arm_id != "frozen_source" or any(
            raw[field] is not None
            for field in (
                "trainable_surface",
                "optimizer",
                "scheduler",
                "max_grad_norm",
            )
        ):
            raise MaterializationError("only Frozen Source may be a no-update plan")
        surface = optimizer = scheduler = max_grad_norm = None

    packing = _mapping(raw["packing"], "packing")
    _exact_fields(packing, {"global_max_length"}, "packing")
    global_max_length = int(packing["global_max_length"])
    if global_max_length != GLOBAL_MAX_LENGTH:
        raise MaterializationError("packing.global_max_length must equal 12000")
    weights = _mapping(raw["family_coefficients"], "family_coefficients")
    _exact_fields(weights, {"h", "source_replay", "duplicate"}, "family_coefficients")
    coefficients = tuple(
        float(weights[name]) for name in ("h", "source_replay", "duplicate")
    )
    if coefficients != _COEFFICIENTS[arm_id]:
        raise MaterializationError(f"family coefficients drifted for {arm_id}")
    if raw["renormalize_active_families"] is not False:
        raise MaterializationError("active objective families must not be renormalized")
    milestones = tuple(raw["milestones"])
    if milestones != MILESTONES:
        raise MaterializationError("milestones must be exactly 0,1,2,4,8,16")
    applicability = str(raw["applicability"])
    expected_applicability = {
        "A6": "sealed_eligible_h_mid_donor",
        "A8-prime": "sealed_aligned_census_margin",
    }.get(arm_id, "always")
    if applicability != expected_applicability:
        raise MaterializationError(f"applicability drifted for {arm_id}")

    return ArmConfig(
        schema_version=SCHEMA_VERSION,
        unit_id=UNIT_ID,
        arm_id=arm_id,
        arm_name=str(raw["arm_name"]),
        updates=updates,
        source=source,
        trainable_surface=surface,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=max_grad_norm,
        global_max_length=global_max_length,
        coefficients=coefficients,
        renormalize_active_families=False,
        milestones=milestones,
        applicability=applicability,
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def a6_binding_from_dict(value: Mapping[str, Any]) -> Any:
    # The runner owns this exact typed execution contract.  Import it only when
    # an A6 binding is actually supplied; ordinary dry-run materialization does
    # not import the packed/model-facing runner surface.
    from scripts.research.run_human13_k_union_overfit import (
        Human13A6DonorBinding,
        Human13A6DonorRecord,
        Human13ManifestIdentity,
        _a6_donor_artifact_sha256,
    )

    raw = _mapping(value, "A6 donor binding")
    identity_raw = _mapping(raw.get("manifest_identity"), "A6 manifest_identity")
    identity = Human13ManifestIdentity(**identity_raw)
    donors = tuple(
        Human13A6DonorRecord(
            **{
                **item,
                "donor_prefix_token_ids": tuple(item["donor_prefix_token_ids"]),
                "donor_prior_row_ids": tuple(item["donor_prior_row_ids"]),
            }
        )
        for item in raw.get("donors", ())
    )
    binding = Human13A6DonorBinding(
        schema_version=str(raw.get("schema_version")),
        manifest_identity=identity,
        frozen_targets_sha256=str(raw.get("frozen_targets_sha256")),
        artifact_sha256=str(raw.get("artifact_sha256")),
        applicable=raw.get("applicable"),
        donors=donors,
    )
    if binding.manifest_identity.unit_id != UNIT_ID:
        raise MaterializationError("A6 binding unit_id mismatches")
    if binding.manifest_identity.panel_sha256 != PANEL_SHA256:
        raise MaterializationError("A6 binding panel_sha256 mismatches")
    if binding.artifact_sha256 != _a6_donor_artifact_sha256(binding):
        raise MaterializationError("A6 binding artifact_sha256 mismatches")
    if not any(donor.h_mid_eligible for donor in binding.donors):
        raise MaterializationError("A6 binding has no eligible H_mid donor")
    return binding


def a6_binding_to_dict(binding: Any) -> dict[str, Any]:
    return asdict(binding)


def _load_a8_binding(path: Path) -> Human13A8CensusBinding:
    raw = dict(_mapping(json.loads(path.read_text(encoding="utf-8")), "census"))
    supplied_digest = raw.pop("artifact_sha256", None)
    if supplied_digest != _canonical_sha256(raw):
        raise MaterializationError("A8 census artifact_sha256 mismatches its payload")
    _exact_fields(
        raw,
        {
            "schema_version",
            "manifest_identity",
            "frozen_targets_sha256",
            "census",
        },
        "A8 census binding",
    )
    if raw.get("schema_version") != "human13_a8_census_binding.v1":
        raise MaterializationError("A8 census binding schema_version is not canonical")
    identity = _mapping(raw.get("manifest_identity"), "census manifest_identity")
    _exact_fields(
        identity,
        {"schema_version", "unit_id", "panel_sha256", "manifest_sha256"},
        "census manifest_identity",
    )
    if (
        identity.get("unit_id") != UNIT_ID
        or identity.get("panel_sha256") != PANEL_SHA256
    ):
        raise MaterializationError("A8 census manifest identity mismatches")
    census = _mapping(raw["census"], "canonical census")
    _exact_fields(
        census,
        {"schema_version", "frozen_targets", "aligned_surface", "a8_prime"},
        "canonical census",
    )
    if census["schema_version"] != "human13_k_union_no_update_census.v1":
        raise MaterializationError(
            "nested no-update census schema_version is not canonical"
        )
    frozen = _mapping(census["frozen_targets"], "census frozen_targets")
    _exact_fields(
        frozen,
        {"byte_identical", "sha256_before", "sha256_after"},
        "census frozen_targets",
    )
    frozen_sha = str(raw["frozen_targets_sha256"])
    if (
        frozen.get("byte_identical") is not True
        or frozen.get("sha256_before") != frozen_sha
        or frozen.get("sha256_after") != frozen_sha
    ):
        raise MaterializationError("A8 census did not preserve frozen target bytes")
    aligned = _mapping(census["aligned_surface"], "census aligned_surface")
    _exact_fields(
        aligned,
        {"all_finite", "maximum_absolute_margin_drift", "site_count"},
        "census aligned_surface",
    )
    if aligned["all_finite"] is not True or int(aligned["site_count"]) <= 0:
        raise MaterializationError("A8 census lacks complete finite aligned sites")
    maximum_drift = float(aligned["maximum_absolute_margin_drift"])
    if not math.isfinite(maximum_drift) or maximum_drift < 0.0:
        raise MaterializationError(
            "A8 census margin drift must be finite and nonnegative"
        )
    a8 = _mapping(census.get("a8_prime"), "census a8_prime")
    _exact_fields(
        a8,
        {
            "applicable",
            "blocked",
            "block_reason",
            "required_margin",
            "violating_site_count",
        },
        "census a8_prime",
    )
    if not isinstance(a8["applicable"], bool) or not isinstance(a8["blocked"], bool):
        raise MaterializationError("A8 applicable/blocked values must be boolean")
    required_margin = a8.get("required_margin")
    if required_margin is None:
        raise MaterializationError("A8 census lacks the drift-derived required margin")
    required_margin = float(required_margin)
    if not math.isclose(
        required_margin,
        maximum_drift + 1.0e-4,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise MaterializationError(
            "A8 required margin is not derived from aligned drift"
        )
    binding = Human13A8CensusBinding(
        schema_version=str(raw["schema_version"]),
        census_schema_version=str(census["schema_version"]),
        manifest_identity={str(k): str(v) for k, v in identity.items()},
        frozen_targets_sha256=frozen_sha,
        artifact_sha256=str(supplied_digest),
        applicable=a8.get("applicable"),
        blocked=a8.get("blocked"),
        block_reason=a8.get("block_reason"),
        required_margin=required_margin,
        violating_site_count=int(a8.get("violating_site_count", 0)),
        maximum_absolute_margin_drift=maximum_drift,
        target_bytes_unchanged=True,
    )
    if binding.applicable:
        if binding.blocked or binding.required_margin is None:
            raise MaterializationError(
                "applicable A8 census is blocked or lacks margin"
            )
        if not (0.0 < binding.required_margin <= 0.5):
            raise MaterializationError("A8 required_margin must be in (0, 0.5]")
    return binding


def _plan_for_config(
    config: ArmConfig,
    *,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    arm_slug = config.arm_id.lower().replace("-", "_")
    arm_root = (output_root / run_id / arm_slug).resolve()
    fresh_state_id = (
        None
        if not config.updates
        else hashlib.sha256(
            f"{run_id}\0{config.arm_id}\0fresh-adamw".encode()
        ).hexdigest()
    )
    return {
        "schema_version": "human13_resolved_arm_plan.v1",
        "unit_id": UNIT_ID,
        "arm_id": config.arm_id,
        "arm_name": config.arm_name,
        "updates": config.updates,
        "source": asdict(config.source),
        "trainable_surface": (
            None
            if config.trainable_surface is None
            else asdict(config.trainable_surface)
        ),
        "optimizer": None if config.optimizer is None else asdict(config.optimizer),
        "scheduler": None if config.scheduler is None else asdict(config.scheduler),
        "max_grad_norm": config.max_grad_norm,
        "global_max_length": config.global_max_length,
        "family_coefficients": dict(
            zip(("h", "source_replay", "duplicate"), config.coefficients, strict=True)
        ),
        "renormalize_active_families": False,
        "milestones": list(config.milestones),
        "output_root": str(arm_root),
        "optimizer_state_root": str(arm_root / "optimizer_state")
        if config.updates
        else None,
        "fresh_state_id": fresh_state_id,
        "resolved_plan_path": str(arm_root / "resolved_plan.json"),
    }


def materialize_plans(
    *,
    output_root: str | Path,
    run_id: str,
    config_root: str | Path = CONFIG_ROOT,
    census_path: str | Path | None = None,
    a6_binding_path: str | Path | None = None,
) -> dict[str, Any]:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise MaterializationError("run_id must be one nonempty path-safe component")
    configs = [
        load_arm_config(path) for path in sorted(Path(config_root).glob("*.yaml"))
    ]
    if tuple(config.arm_id for config in configs) != tuple(_ARM_ORDER):
        raise MaterializationError(
            "config directory must contain each exact approved arm once"
        )

    a6 = None
    if a6_binding_path is not None:
        a6 = a6_binding_from_dict(
            _mapping(
                json.loads(Path(a6_binding_path).read_text(encoding="utf-8")),
                "A6 donor binding",
            )
        )
    a8 = _load_a8_binding(Path(census_path)) if census_path is not None else None
    if a6 is not None and a8 is not None:
        if (
            a6.manifest_identity.manifest_sha256
            != a8.manifest_identity["manifest_sha256"]
            or a6.frozen_targets_sha256 != a8.frozen_targets_sha256
        ):
            raise MaterializationError(
                "A6 and A8 bindings do not share manifest/targets"
            )

    plans: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    for config in configs:
        if config.arm_id == "A6" and a6 is None:
            omitted.append(
                {
                    "arm_id": "A6",
                    "reason": "sealed_eligible_h_mid_donor_unavailable",
                }
            )
            continue
        if config.arm_id == "A8-prime" and (a8 is None or not a8.applicable):
            omitted.append(
                {
                    "arm_id": "A8-prime",
                    "reason": (
                        "sealed_a8_census_binding_unavailable"
                        if a8 is None
                        else str(a8.block_reason or "sealed_a8_census_blocked")
                    ),
                }
            )
            continue
        plan = _plan_for_config(config, output_root=Path(output_root), run_id=run_id)
        if config.arm_id == "A6":
            plan["a6_donor_binding"] = a6_binding_to_dict(a6)
        if config.arm_id == "A8-prime":
            plan["a8_census_binding"] = asdict(a8)
        plans.append(plan)

    roots = [plan["output_root"] for plan in plans]
    states = [plan["optimizer_state_root"] for plan in plans if plan["updates"]]
    if len(set(roots)) != len(roots) or len(set(states)) != len(states):
        raise MaterializationError("arm output or optimizer state roots are not unique")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mode": "dry_run",
        "actions": dict(ZERO_MODEL_ACTIONS),
        "source_independence": {
            "byte_identical_source": True,
            "fresh_independent_adamw": True,
            "cross_arm_parameter_sharing": False,
            "cross_arm_optimizer_state_sharing": False,
        },
        "plans": plans,
        "omitted_arms": sorted(
            omitted, key=lambda item: {"A6": 0, "A8-prime": 1}[item["arm_id"]]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--a6-binding", type=Path)
    parser.add_argument(
        "--write-receipt",
        type=Path,
        help="Explicitly write the plan receipt; default emits JSON to stdout only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize_plans(
        output_root=args.output_root,
        run_id=args.run_id,
        config_root=args.config_root,
        census_path=args.census,
        a6_binding_path=args.a6_binding,
    )
    encoded = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.write_receipt is not None:
        destination = args.write_receipt.resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite plan receipt: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
