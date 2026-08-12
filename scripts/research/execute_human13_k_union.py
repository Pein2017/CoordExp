#!/usr/bin/env python3
"""CPU-only entry contract for the Human-13 arm plans.

This module is intentionally not a trainer.  It validates a resolved plan
against the canonical manifest, exposes the frozen exposure schedule, and
keeps execution unavailable until a separately verified runtime entry is
provided.  Importing it must not import a model stack or allocate a device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.research import materialize_human13_k_union_configs as materializer


UNIT_ID = materializer.UNIT_ID
PANEL_SHA256 = materializer.PANEL_SHA256
PLAN_SCHEMA_VERSION = "human13_resolved_arm_plan.v1"
MANIFEST_SCHEMA_VERSION = "human13_k_union_manifest.v1"
FROZEN_SOURCE = materializer.FROZEN_SOURCE
EXPOSURE_MILESTONES = (0, 1, 2, 4, 8, 16)
EXPECTED_IMAGE_SHA256 = {
    1584: "06b9d29a50b896f1bec14a267a57016723e54e205a1d1a40088237d95ce91206",
    2299: "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3",
    2685: "84514a8aed88aba07163aa5b1be5c6e0ee75da4351496cad062d1e355a261409",
    4134: "60dfd1369e0efa83dfa6c7d4035f4d9d66ca6ba9a0ec6b760349d0a0e30d7b34",
    5001: "faaecb19a8b681495f02e18493b8ae01c96d022767f25ded19f5cbec9d95cf31",
    6040: "585c27309e9130400849315b4bf49d99717f700507239f2bd4dc6fece3cbe894",
    7511: "2843c07959515a93d2791183c998462d76b34100185e57a258d8949f112296e7",
    10707: "eec1e22dc3ed6ff70d35dd771e05a20024264fdecaf5a187fdad616e6d20e5d6",
    13348: "14f222b4cbf6d90e60eb90eb72aebc0f83cebaf150b4c5999f107bb11294fc36",
    13923: "c5c32b9999259b6041e92815693b975f8ee2291b29a6577d983685b6d486796f",
    14038: "055f28bbd181590b7a7c4844bf488d7f1be752d39916e07034c279f5f387acd2",
    14439: "d09e1ef4ec3bcbfbe3e16a2f9b3ff92a743e4dc061eaca4add29f59787567f73",
    16228: "5aade4c6e9dbdbf3bd64072e813bae02975a342b84c312efc91983ac63300d49",
}
ZERO_ACTIONS = {
    "model_imports": 0,
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
    "gpu_allocations": 0,
}


class ExecutionContractError(ValueError):
    """Raised when a plan or runtime cannot prove the Human-13 contract."""


@dataclass(frozen=True)
class ResolvedPlan:
    arm_id: str
    updates: bool
    manifest_sha256: str
    output_root: Path
    optimizer_state_root: Path | None
    milestones: tuple[int, ...]
    source: Mapping[str, str]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class SourcePromptPrefix:
    image_id: int
    image_sha256: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Exposure:
    arm_id: str
    milestone: int
    updates: bool
    run_root: Path
    optimizer_state_root: Path | None


_REQUIRED_PLAN_FIELDS = {
    "schema_version",
    "unit_id",
    "arm_id",
    "updates",
    "source",
    "manifest_identity",
    "global_max_length",
    "milestones",
    "output_root",
    "optimizer_state_root",
    "fresh_state_id",
}
_OPTIONAL_PLAN_FIELDS = {
    "arm_name",
    "trainable_surface",
    "optimizer",
    "scheduler",
    "max_grad_norm",
    "family_coefficients",
    "renormalize_active_families",
    "a6_donor_binding",
    "a8_census_binding",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionContractError(f"{field} must be an object")
    return value


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _load_manifest_identity(path: Path) -> tuple[str, Mapping[str, Any]]:
    if not path.is_file():
        raise ExecutionContractError(f"manifest does not exist: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != (
        f"{digest}  {path.name}\n"
    ):
        raise ExecutionContractError("manifest digest receipt mismatches canonical bytes")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExecutionContractError("manifest is not valid JSON") from exc
    raw = _mapping(document, "manifest")
    canonical = (
        json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise ExecutionContractError("manifest is not canonically serialized")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExecutionContractError("manifest schema_version is not canonical")
    if raw.get("full_panel") is not True:
        raise ExecutionContractError("resolved plan requires a sealed full-panel manifest")
    binding = _mapping(raw.get("binding"), "manifest.binding")
    panel = _mapping(binding.get("panel"), "manifest.binding.panel")
    if binding.get("unit_id") != UNIT_ID or binding.get("purpose") != "overfit_only":
        raise ExecutionContractError("manifest unit or purpose identity mismatches")
    if panel.get("panel_sha256") != PANEL_SHA256:
        raise ExecutionContractError("manifest panel SHA-256 mismatches")
    return digest, raw


def _source_dict() -> dict[str, str]:
    return {
        key: str(value)
        for key, value in materializer.FROZEN_SOURCE.__dict__.items()
    }


def validate_resolved_plan(
    path: str | Path, manifest_path: str | Path
) -> ResolvedPlan:
    """Validate a plan's content binding without importing the model runtime."""

    plan_path = Path(path)
    try:
        document = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutionContractError(f"resolved plan does not exist: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExecutionContractError("resolved plan is not valid JSON") from exc
    raw = _mapping(document, "resolved plan")
    allowed = _REQUIRED_PLAN_FIELDS | _OPTIONAL_PLAN_FIELDS
    unknown = set(raw) - allowed
    missing = _REQUIRED_PLAN_FIELDS - set(raw)
    if unknown:
        raise ExecutionContractError(f"resolved plan has unknown fields: {sorted(unknown)}")
    if missing:
        raise ExecutionContractError(f"resolved plan is missing fields: {sorted(missing)}")
    manifest_digest, _manifest = _load_manifest_identity(Path(manifest_path))
    if raw["schema_version"] != PLAN_SCHEMA_VERSION or raw["unit_id"] != UNIT_ID:
        raise ExecutionContractError("resolved plan schema or unit identity mismatches")
    identity = _mapping(raw["manifest_identity"], "manifest_identity")
    if set(identity) != {"schema_version", "unit_id", "panel_sha256", "manifest_sha256"}:
        raise ExecutionContractError("manifest_identity fields are not exact")
    if (
        identity["schema_version"] != MANIFEST_SCHEMA_VERSION
        or identity["unit_id"] != UNIT_ID
        or identity["panel_sha256"] != PANEL_SHA256
        or _digest(identity["manifest_sha256"], "manifest_identity.manifest_sha256")
        != manifest_digest
    ):
        raise ExecutionContractError("resolved plan manifest SHA or unit identity mismatches")
    source = _mapping(raw["source"], "source")
    if dict(source) != _source_dict():
        raise ExecutionContractError("resolved plan source identity mismatches Frozen Source")
    arm_id = str(raw["arm_id"])
    if arm_id not in materializer._COEFFICIENTS:
        raise ExecutionContractError(f"arm_id is not approved: {arm_id}")
    updates = raw["updates"]
    if not isinstance(updates, bool) or updates != (arm_id != "frozen_source"):
        raise ExecutionContractError("Frozen Source is the only no-update arm")
    if raw["global_max_length"] != 12_000:
        raise ExecutionContractError("global_max_length must remain 12000")
    if tuple(raw["milestones"]) != EXPOSURE_MILESTONES:
        raise ExecutionContractError("milestones must be exactly 0,1,2,4,8,16")
    output_root = Path(str(raw["output_root"]))
    if not output_root.is_absolute():
        raise ExecutionContractError("output_root must be absolute")
    state_value = raw["optimizer_state_root"]
    optimizer_state_root = None if state_value is None else Path(str(state_value))
    if updates:
        if optimizer_state_root is None or not optimizer_state_root.is_absolute():
            raise ExecutionContractError(
                "updated arms require an absolute optimizer state root"
            )
        if not isinstance(raw["fresh_state_id"], str) or not raw["fresh_state_id"]:
            raise ExecutionContractError(
                "updated arms require a fresh optimizer state identity"
            )
    elif optimizer_state_root is not None or raw["fresh_state_id"] is not None:
        raise ExecutionContractError("Frozen Source cannot carry optimizer state")
    if "family_coefficients" in raw:
        coefficients = _mapping(raw["family_coefficients"], "family_coefficients")
        expected = dict(
            zip(
                ("h", "source_replay", "duplicate"),
                materializer._COEFFICIENTS[arm_id],
                strict=True,
            )
        )
        if dict(coefficients) != expected:
            raise ExecutionContractError("family coefficients drifted")
    if raw.get("renormalize_active_families", False) is not False:
        raise ExecutionContractError("active family coefficients must not be renormalized")
    return ResolvedPlan(
        arm_id=arm_id,
        updates=updates,
        manifest_sha256=manifest_digest,
        output_root=output_root,
        optimizer_state_root=optimizer_state_root,
        milestones=EXPOSURE_MILESTONES,
        source=dict(source),
        raw=raw,
    )


def _ids(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ExecutionContractError(
            f"{field} must be a sequence of non-negative token IDs"
        )
    return tuple(int(item) for item in value)


def verify_source_prompt_prefix_parity(
    records: Mapping[int | str, Mapping[str, Any]] | None = None,
    *,
    expected_image_ids: Sequence[int] | None = None,
    expected_prefix_token_ids: Mapping[int | str, Sequence[int]] | None = None,
    manifest: Mapping[str, Any] | Any | None = None,
    source_records: Mapping[int | str, Mapping[str, Any]] | None = None,
) -> dict[int, SourcePromptPrefix]:
    """Verify literal executed prompt IDs against the Source discovery prefix.

    The helper never decodes or re-tokenizes IDs.  ``input_prompt_token_ids``
    may be shorter because image placeholders expand; parity is checked on the
    executed prompt IDs that the backend actually consumed.
    """

    if records is None:
        records = source_records
    if records is None:
        raise ExecutionContractError("source records are required")
    if manifest is not None:
        images = (
            manifest.get("images")
            if isinstance(manifest, Mapping)
            else getattr(manifest, "images", None)
        )
        if not isinstance(images, (list, tuple)) or not images:
            raise ExecutionContractError("manifest has no source images")
        if expected_image_ids is None:
            expected_image_ids = tuple(
                int(image.get("image_id") if isinstance(image, Mapping) else image.image_id)
                for image in images
            )
        if expected_prefix_token_ids is None:
            prefixes: dict[int, tuple[int, ...]] = {}
            for image in images:
                image_id = int(
                    image.get("image_id") if isinstance(image, Mapping) else image.image_id
                )
                trajectories = (
                    image.get("trajectories")
                    if isinstance(image, Mapping)
                    else image.trajectories
                )
                if not trajectories:
                    raise ExecutionContractError(
                        f"manifest image {image_id} has no Source trajectory"
                    )
                source = trajectories[0]
                prefix = source.get("prefix") if isinstance(source, Mapping) else source.prefix
                token_ids = (
                    prefix.get("raw_token_ids")
                    if isinstance(prefix, Mapping)
                    else prefix.raw_token_ids
                )
                prefixes[image_id] = tuple(int(token) for token in token_ids)
            expected_prefix_token_ids = prefixes
    if expected_image_ids is None or expected_prefix_token_ids is None:
        raise ExecutionContractError("manifest-derived expected source prefixes are required")
    expected_ids = tuple(int(image_id) for image_id in expected_image_ids)
    observed_ids = {int(image_id) for image_id in records}
    if observed_ids != set(expected_ids):
        raise ExecutionContractError(
            "source records do not cover exactly the expected images"
        )
    result: dict[int, SourcePromptPrefix] = {}
    for image_id in expected_ids:
        record = _mapping(
            records.get(image_id, records.get(str(image_id))),
            f"source[{image_id}]",
        )
        if record.get("image_sha256") != EXPECTED_IMAGE_SHA256.get(image_id):
            raise ExecutionContractError(f"source image {image_id} identity mismatches")
        executed = _ids(
            record.get("executed_prompt_token_ids"),
            f"source[{image_id}].executed_prompt_token_ids",
        )
        expected = _ids(
            record.get("expected_executed_prompt_token_ids", executed),
            f"source[{image_id}].expected_executed_prompt_token_ids",
        )
        if executed != expected:
            raise ExecutionContractError(
                f"source image {image_id} prompt-prefix parity is not exact"
            )
        declared = _ids(
            expected_prefix_token_ids.get(
                image_id, expected_prefix_token_ids.get(str(image_id))
            ),
            f"expected_prefix_token_ids[{image_id}]",
        )
        if executed != declared:
            raise ExecutionContractError(
                f"source image {image_id} prompt-prefix parity mismatches manifest"
            )
        generated = _ids(
            record.get("generated_token_ids"),
            f"source[{image_id}].generated_token_ids",
        )
        if not generated:
            raise ExecutionContractError(f"source image {image_id} has no generated token IDs")
        result[image_id] = SourcePromptPrefix(
            image_id, str(record["image_sha256"]), executed, generated
        )
    return result


def plan_exposures(
    *,
    arm_id: str,
    output_root: str | Path,
    optimizer_state_root: str | Path | None = None,
) -> tuple[Exposure, ...]:
    """Build pure cumulative milestone contracts with isolated arm roots."""

    if arm_id not in materializer._COEFFICIENTS:
        raise ExecutionContractError(f"arm_id is not approved: {arm_id}")
    root = Path(output_root)
    if not root.is_absolute():
        raise ExecutionContractError("output_root must be absolute")
    updates = arm_id != "frozen_source"
    state = None if optimizer_state_root is None else Path(optimizer_state_root)
    if updates and (state is None or not state.is_absolute()):
        raise ExecutionContractError(
            "updated arms require an absolute optimizer state root"
        )
    if not updates and state is not None:
        raise ExecutionContractError("Frozen Source cannot have optimizer state")
    return tuple(
        Exposure(
            arm_id=arm_id,
            milestone=milestone,
            updates=updates and milestone > 0,
            run_root=root / f"milestone-{milestone}",
            optimizer_state_root=state,
        )
        for milestone in EXPOSURE_MILESTONES
    )


def dry_run_resolved_plan(path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    plan = validate_resolved_plan(path, manifest_path)
    schedule = plan_exposures(
        arm_id=plan.arm_id,
        output_root=plan.output_root,
        optimizer_state_root=plan.optimizer_state_root,
    )
    return {
        "schema_version": "human13_execution_receipt.v1",
        "mode": "dry_run",
        "execution_ready": False,
        "arm_id": plan.arm_id,
        "manifest_sha256": plan.manifest_sha256,
        "actions": dict(ZERO_ACTIONS),
        "schedule": [
            {
                "arm_id": item.arm_id,
                "milestone": item.milestone,
                "updates": item.updates,
                "run_root": str(item.run_root),
                "optimizer_state_root": (
                    None if item.optimizer_state_root is None else str(item.optimizer_state_root)
                ),
            }
            for item in schedule
        ],
    }


def execute_resolved_plan(
    path: str | Path,
    manifest_path: str | Path,
    *,
    execution_authorized: bool = False,
    runtime_factory: (
        Callable[[ResolvedPlan, tuple[Exposure, ...]], Mapping[str, Any]] | None
    ) = None,
) -> Mapping[str, Any]:
    """Fail closed until a separately verified production runtime is supplied."""

    plan = validate_resolved_plan(path, manifest_path)
    schedule = plan_exposures(
        arm_id=plan.arm_id,
        output_root=plan.output_root,
        optimizer_state_root=plan.optimizer_state_root,
    )
    if execution_authorized is not True:
        raise ExecutionContractError(
            "execute mode requires separate user model/GPU authority"
        )
    if runtime_factory is None:
        raise ExecutionContractError(
            "real runtime entry is unavailable; execution remains fail-closed"
        )
    if not callable(runtime_factory):
        raise ExecutionContractError("runtime factory is not callable")
    receipt = runtime_factory(plan, schedule)
    if not isinstance(receipt, Mapping):
        raise ExecutionContractError("runtime did not return a bound runtime receipt")
    if (
        receipt.get("execution_ready") is not True
        or receipt.get("runtime_entry_contract") != "human13_runtime.v1"
        or receipt.get("manifest_sha256") != plan.manifest_sha256
        or receipt.get("arm_id") != plan.arm_id
    ):
        raise ExecutionContractError("runtime did not return a bound runtime receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute:
        receipt = execute_resolved_plan(
            args.resolved_plan,
            args.manifest,
            execution_authorized=args.user_model_gpu_authority,
        )
    else:
        receipt = dry_run_resolved_plan(args.resolved_plan, args.manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPOSURE_MILESTONES",
    "EXPECTED_IMAGE_SHA256",
    "ExecutionContractError",
    "Exposure",
    "FROZEN_SOURCE",
    "MANIFEST_SCHEMA_VERSION",
    "PANEL_SHA256",
    "PLAN_SCHEMA_VERSION",
    "ResolvedPlan",
    "SourcePromptPrefix",
    "UNIT_ID",
    "ZERO_ACTIONS",
    "dry_run_resolved_plan",
    "execute_resolved_plan",
    "main",
    "plan_exposures",
    "validate_resolved_plan",
    "verify_source_prompt_prefix_parity",
]
