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
import subprocess
import sys
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


@dataclass(frozen=True)
class Human13ManifestIdentity:
    schema_version: str
    unit_id: str
    panel_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class Human13A6DonorRecord:
    image_id: int
    owner_id: str
    target_row_id: str
    donor_trajectory_id: str
    donor_prefix_token_ids: tuple[int, ...]
    donor_prior_row_ids: tuple[str, ...]
    h_mid_eligible: bool


@dataclass(frozen=True)
class Human13A6DonorBinding:
    schema_version: str
    manifest_identity: Human13ManifestIdentity
    frozen_targets_sha256: str
    artifact_sha256: str
    applicable: bool
    donors: tuple[Human13A6DonorRecord, ...]


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
    expected_updates = arm_id != "frozen_source"
    if updates is not expected_updates:
        raise MaterializationError(
            "arm update matrix requires Frozen Source no-update and every other arm updating"
        )
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def a6_binding_to_dict(binding: Human13A6DonorBinding) -> dict[str, Any]:
    return asdict(binding)


def _load_manifest_document(
    path: Path,
) -> tuple[Mapping[str, Any], Human13ManifestIdentity, str]:
    # Admission belongs to the canonical manifest owner.  Run that validator in
    # a bounded CPU subprocess so this materializer remains free of model-stack
    # imports while still reusing the one authoritative semantic validator.
    admission = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from scripts.research.build_human13_k_union_manifest "
                "import load_manifest; "
                "load_manifest(sys.argv[1], require_full_panel=True)"
            ),
            str(path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if admission.returncode != 0:
        raise MaterializationError(
            "sealed Human-13 manifest failed canonical full-panel admission"
        )
    payload = path.read_bytes()
    raw = _mapping(json.loads(payload), "sealed Human-13 manifest")
    _exact_fields(
        raw,
        {
            "schema_version",
            "binding",
            "images",
            "arms",
            "denominators",
            "full_panel",
        },
        "sealed Human-13 manifest",
    )
    if raw["schema_version"] != "human13_k_union_manifest.v1":
        raise MaterializationError("manifest schema_version is not canonical")
    if raw["full_panel"] is not True:
        raise MaterializationError("A6/A8 requires a sealed full-panel manifest")
    canonical = (
        json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise MaterializationError("manifest is not canonically serialized")
    manifest_sha = hashlib.sha256(payload).hexdigest()
    digest_path = Path(f"{path}.sha256")
    if not digest_path.is_file() or digest_path.read_text(encoding="ascii") != (
        f"{manifest_sha}  {path.name}\n"
    ):
        raise MaterializationError("manifest digest receipt mismatches canonical bytes")
    manifest_binding = _mapping(raw["binding"], "manifest binding")
    panel = _mapping(manifest_binding.get("panel"), "manifest panel binding")
    if (
        manifest_binding.get("unit_id") != UNIT_ID
        or panel.get("panel_sha256") != PANEL_SHA256
    ):
        raise MaterializationError("manifest unit or panel identity mismatches")
    identity = Human13ManifestIdentity(
        schema_version=str(raw["schema_version"]),
        unit_id=UNIT_ID,
        panel_sha256=PANEL_SHA256,
        manifest_sha256=manifest_sha,
    )
    images = raw["images"]
    if not isinstance(images, list) or not images:
        raise MaterializationError("manifest images must be a nonempty list")
    projection = [
        {
            "image_id": image["image_id"],
            "selected_rows": [
                {
                    "owner_id": row["owner_id"],
                    "row_id": row["row_id"],
                    "token_ids": list(row["token_ids"]),
                    "target_token_mask": list(row["target_token_mask"]),
                }
                for row in image["selected_rows"]
            ],
        }
        for image in images
    ]
    return raw, identity, _canonical_sha256(projection)


def _a6_artifact_sha256(binding: Human13A6DonorBinding) -> str:
    payload = asdict(binding)
    payload.pop("artifact_sha256")
    return _canonical_sha256(payload)


def _derive_a6_binding(
    manifest: Mapping[str, Any],
    identity: Human13ManifestIdentity,
    frozen_targets_sha256: str,
) -> Human13A6DonorBinding | None:
    donors: list[Human13A6DonorRecord] = []
    for image_value in manifest["images"]:
        image = _mapping(image_value, "A6 manifest image")
        image_id = image.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise MaterializationError("A6 image provenance has invalid image_id")
        owners = [_mapping(item, "A6 owner") for item in image.get("owners", ())]
        owners_by_id = {str(item.get("owner_id")): item for item in owners}
        if len(owners_by_id) != len(owners):
            raise MaterializationError("A6 owner provenance is not unique")
        h_owner_ids = tuple(str(item) for item in image.get("h_owner_ids", ()))
        g_owner_ids = tuple(str(item) for item in image.get("g_owner_ids", ()))
        if not h_owner_ids or not g_owner_ids:
            continue
        try:
            max_g_index = max(
                int(owners_by_id[owner_id]["source_object_index"])
                for owner_id in g_owner_ids
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterializationError("A6 G-owner provenance is incomplete") from exc
        trajectories = [
            _mapping(item, "A6 trajectory") for item in image.get("trajectories", ())
        ]
        trajectories_by_id = {
            str(item.get("trajectory_id")): item for item in trajectories
        }
        if len(trajectories_by_id) != len(trajectories):
            raise MaterializationError("A6 trajectory provenance is not unique")
        for selected_value in image.get("selected_rows", ()):
            selected = _mapping(selected_value, "A6 selected row")
            owner_id = str(selected.get("owner_id"))
            row_id = str(selected.get("row_id"))
            trajectory_id = str(selected.get("trajectory_id"))
            if owner_id not in h_owner_ids or owner_id not in owners_by_id:
                raise MaterializationError("A6 selected owner provenance is invented")
            trajectory = trajectories_by_id.get(trajectory_id)
            if trajectory is None:
                raise MaterializationError(
                    "A6 selected trajectory provenance is invented"
                )
            rows = [
                _mapping(item, "A6 trajectory row")
                for item in trajectory.get("rows", ())
            ]
            rows_by_id = {str(item.get("row_id")): item for item in rows}
            if len(rows_by_id) != len(rows) or row_id not in rows_by_id:
                raise MaterializationError("A6 selected row provenance is invented")
            target = rows_by_id[row_id]
            target_start = target.get("token_start")
            raw_tokens = trajectory.get("raw_token_ids")
            if (
                isinstance(target_start, bool)
                or not isinstance(target_start, int)
                or not isinstance(raw_tokens, list)
                or not 0 < target_start <= len(raw_tokens)
            ):
                raise MaterializationError(
                    "A6 selected row token provenance is invalid"
                )
            duplicate_ids = {
                str(item) for item in trajectory.get("duplicate_row_ids", ())
            }
            retained_ids = {
                str(item) for item in trajectory.get("retained_row_ids", ())
            }
            if (
                duplicate_ids - set(rows_by_id)
                or retained_ids - set(rows_by_id)
                or duplicate_ids & retained_ids
            ):
                raise MaterializationError(
                    "A6 duplicate/retained row provenance is invalid"
                )
            removed: set[int] = set()
            for duplicate_id in duplicate_ids:
                duplicate = rows_by_id[duplicate_id]
                start, end = duplicate.get("token_start"), duplicate.get("token_end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end
                ):
                    raise MaterializationError("A6 duplicate row span is invalid")
                if end <= target_start:
                    removed.update(range(start, end))
            clean_prefix = tuple(
                int(token)
                for index, token in enumerate(raw_tokens[:target_start])
                if index not in removed
            )
            prior_rows = tuple(
                str(row["row_id"])
                for row in rows
                if str(row["row_id"]) in retained_ids
                and int(row["token_end"]) <= target_start
            )
            if not clean_prefix or not prior_rows or row_id in duplicate_ids:
                raise MaterializationError(
                    "A6 donor provenance has no clean native prior context"
                )
            owner_index = owners_by_id[owner_id].get("source_object_index")
            if isinstance(owner_index, bool) or not isinstance(owner_index, int):
                raise MaterializationError("A6 H-owner sort provenance is invalid")
            donors.append(
                Human13A6DonorRecord(
                    image_id=image_id,
                    owner_id=owner_id,
                    target_row_id=row_id,
                    donor_trajectory_id=trajectory_id,
                    donor_prefix_token_ids=clean_prefix,
                    donor_prior_row_ids=prior_rows,
                    h_mid_eligible=owner_index < max_g_index,
                )
            )
    if not donors or not any(item.h_mid_eligible for item in donors):
        return None
    unsealed = Human13A6DonorBinding(
        schema_version="human13_a6_donor_binding.v1",
        manifest_identity=identity,
        frozen_targets_sha256=frozen_targets_sha256,
        artifact_sha256="0" * 64,
        applicable=True,
        donors=tuple(donors),
    )
    return Human13A6DonorBinding(
        schema_version=unsealed.schema_version,
        manifest_identity=identity,
        frozen_targets_sha256=frozen_targets_sha256,
        artifact_sha256=_a6_artifact_sha256(unsealed),
        applicable=True,
        donors=tuple(donors),
    )


def _load_a8_binding(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    identity: Human13ManifestIdentity,
    frozen_targets_sha256: str,
) -> Human13A8CensusBinding:
    payload = path.read_bytes()
    census = _mapping(json.loads(payload), "canonical census")
    _exact_fields(
        census,
        {
            "schema_version",
            "trie",
            "coherent_chain",
            "frozen_targets",
            "aligned_surface",
            "a8_prime",
        },
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
    if (
        frozen.get("byte_identical") is not True
        or frozen.get("sha256_before") != frozen_targets_sha256
        or frozen.get("sha256_after") != frozen_targets_sha256
    ):
        raise MaterializationError("A8 census did not preserve frozen target bytes")
    trie = _mapping(census["trie"], "census trie")
    _exact_fields(
        trie,
        {"original_row_count", "unique_row_count", "exact_duplicate_count", "images"},
        "census trie",
    )
    selected_rows = [
        (str(image["image_id"]), _mapping(row, "A8 selected row"))
        for image in manifest["images"]
        for row in image["selected_rows"]
    ]
    unique_rows = {
        (image_id, tuple(int(token) for token in row["token_ids"]))
        for image_id, row in selected_rows
    }
    trie_images = trie["images"]
    if (
        int(trie["original_row_count"]) != len(selected_rows)
        or int(trie["unique_row_count"]) != len(unique_rows)
        or int(trie["exact_duplicate_count"]) != len(selected_rows) - len(unique_rows)
        or not isinstance(trie_images, list)
        or not trie_images
    ):
        raise MaterializationError("A8 census trie is incomplete")
    expected_children: dict[str, dict[tuple[int, ...], set[int]]] = {}
    for image_id, row in selected_rows:
        token_ids = tuple(int(token) for token in row["token_ids"])
        image_children = expected_children.setdefault(image_id, {})
        for offset, token in enumerate(token_ids):
            image_children.setdefault(token_ids[:offset], set()).add(token)
    seen_trie_images: set[str] = set()
    for image_value in trie_images:
        image = _mapping(image_value, "census trie image")
        _exact_fields(
            image,
            {
                "image_id",
                "nodes",
                "projected_token_ids",
                "projected_owner_ids",
                "reached_native_leaf",
            },
            "census trie image",
        )
        image_id = str(image["image_id"])
        if image_id in seen_trie_images or image_id not in expected_children:
            raise MaterializationError("A8 census trie image provenance mismatches")
        seen_trie_images.add(image_id)
        nodes = image["nodes"]
        if not isinstance(nodes, list):
            raise MaterializationError("A8 census trie nodes are incomplete")
        observed_children: dict[tuple[int, ...], tuple[int, ...]] = {}
        for node_value in nodes:
            node = _mapping(node_value, "census trie node")
            _exact_fields(
                node,
                {
                    "prefix_token_ids",
                    "viable_child_token_ids",
                    "actual_top1_token_id",
                    "actual_top1_is_viable_child",
                    "strongest_viable_child_token_id",
                    "strongest_viable_child_margin",
                    "top_tie_count",
                },
                "census trie node",
            )
            prefix = tuple(int(token) for token in node["prefix_token_ids"])
            children = tuple(int(token) for token in node["viable_child_token_ids"])
            if prefix in observed_children:
                raise MaterializationError("A8 census trie prefix is duplicated")
            observed_children[prefix] = children
        expected_image_children = {
            prefix: tuple(sorted(children))
            for prefix, children in expected_children[image_id].items()
        }
        if observed_children != expected_image_children:
            raise MaterializationError("A8 census trie is truncated or fabricated")
    if seen_trie_images != set(expected_children):
        raise MaterializationError("A8 census trie image census is incomplete")
    chain = _mapping(census["coherent_chain"], "census coherent_chain")
    _exact_fields(
        chain,
        {
            "site_count",
            "sites",
            "first_non_argmax_site",
            "minimum_strict_margin",
            "tie_site_count",
            "token_role_counts",
        },
        "census coherent_chain",
    )
    sites = chain["sites"]
    if (
        not isinstance(sites, list)
        or not sites
        or int(chain["site_count"]) != len(sites)
    ):
        raise MaterializationError("A8 coherent chain is incomplete")
    packed_margins: list[float] = []
    drifts: list[float] = []
    expected_sites = {
        (image_id, str(row["owner_id"]), offset): int(token)
        for image_id, row in selected_rows
        for offset, (token, included) in enumerate(
            zip(row["token_ids"], row["target_token_mask"], strict=True)
        )
        if included is True
    }
    observed_sites: set[tuple[str, str, int]] = set()
    for index, site_value in enumerate(sites):
        site = _mapping(site_value, f"census coherent_chain site {index}")
        site_key = (
            str(site.get("image_id")),
            str(site.get("owner_id")),
            int(site.get("token_offset", -1)),
        )
        if (
            site.get("site_index") != index
            or site.get("packed_finite") is not True
            or site.get("hf_finite") is not True
            or site.get("aligned_finite") is not True
            or site_key in observed_sites
            or expected_sites.get(site_key) != site.get("target_token_id")
        ):
            raise MaterializationError("A8 census lacks complete finite aligned sites")
        observed_sites.add(site_key)
        try:
            packed_margin = float(site["packed_target_margin"])
            hf_margin = float(site["hf_target_margin"])
            reported_drift = float(site["absolute_margin_drift"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterializationError("A8 aligned site margin is incomplete") from exc
        derived_drift = abs(packed_margin - hf_margin)
        if not all(
            math.isfinite(item) for item in (packed_margin, hf_margin, reported_drift)
        ):
            raise MaterializationError("A8 census aligned margins are not finite")
        if not math.isclose(
            reported_drift, derived_drift, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise MaterializationError("A8 aligned site drift is self-asserted")
        packed_margins.append(packed_margin)
        drifts.append(derived_drift)
    if observed_sites != set(expected_sites):
        raise MaterializationError("A8 coherent chain is truncated or fabricated")
    aligned = _mapping(census["aligned_surface"], "census aligned_surface")
    _exact_fields(
        aligned,
        {"all_finite", "maximum_absolute_margin_drift", "site_count"},
        "census aligned_surface",
    )
    if aligned["all_finite"] is not True or int(aligned["site_count"]) != len(sites):
        raise MaterializationError("A8 census lacks complete finite aligned sites")
    maximum_drift = max(drifts)
    if not math.isclose(
        float(aligned["maximum_absolute_margin_drift"]),
        maximum_drift,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise MaterializationError("A8 maximum drift is not derived from every site")
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
    required_margin = maximum_drift + 1.0e-4
    try:
        reported_required_margin = float(a8["required_margin"])
    except (TypeError, ValueError) as exc:
        raise MaterializationError(
            "A8 required margin is not derived from aligned drift"
        ) from exc
    if not math.isclose(
        reported_required_margin, required_margin, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise MaterializationError(
            "A8 required margin is not derived from aligned drift"
        )
    applicable = required_margin <= 0.5
    expected_reason = None if applicable else "required_margin_exceeds_0_5"
    expected_violations = (
        sum(margin < required_margin for margin in packed_margins) if applicable else 0
    )
    if (
        a8.get("applicable") is not applicable
        or a8.get("blocked") is applicable
        or a8.get("block_reason") != expected_reason
        or int(a8.get("violating_site_count", -1)) != expected_violations
    ):
        raise MaterializationError("A8 applicable/blocked state is not census-derived")
    binding = Human13A8CensusBinding(
        schema_version="human13_a8_census_binding.v1",
        census_schema_version=str(census["schema_version"]),
        manifest_identity=asdict(identity),
        frozen_targets_sha256=frozen_targets_sha256,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        applicable=applicable,
        blocked=not applicable,
        block_reason=a8.get("block_reason"),
        required_margin=required_margin,
        violating_site_count=int(a8.get("violating_site_count", 0)),
        maximum_absolute_margin_drift=maximum_drift,
        target_bytes_unchanged=True,
    )
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
    manifest_path: str | Path | None = None,
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

    manifest = identity = frozen_targets_sha256 = None
    if manifest_path is not None:
        manifest, identity, frozen_targets_sha256 = _load_manifest_document(
            Path(manifest_path)
        )
    if census_path is not None and manifest is None:
        raise MaterializationError("A8 census requires its sealed canonical manifest")
    a6 = (
        _derive_a6_binding(manifest, identity, frozen_targets_sha256)
        if manifest is not None
        and identity is not None
        and frozen_targets_sha256 is not None
        else None
    )
    a8 = (
        _load_a8_binding(
            Path(census_path),
            manifest=manifest,
            identity=identity,
            frozen_targets_sha256=frozen_targets_sha256,
        )
        if census_path is not None
        and identity is not None
        and frozen_targets_sha256 is not None
        else None
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
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Sealed canonical full-panel manifest used to derive A6 and bind A8.",
    )
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
        manifest_path=args.manifest,
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
