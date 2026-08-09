#!/usr/bin/env python3
"""Seal the one authorized post-execution finalization successor receipt.

The immutable pre-GPU parent receipt binds the finalizer and its test by hash,
and the three completed shards bind the parent receipt by semantic self hash.
A consumer repair discovered after execution therefore cannot be finalized by
re-sealing the parent.  This module writes exactly one canonical, write-once,
CPU-only successor receipt that authorizes two fixed binding slots and nothing
else, so the finalizer can consume the immutable execution set without the
parent, the plan, or any shard artifact changing.

The receipt binds the authority document, the parent receipt and plan by path,
raw and semantic self hash, all four artifacts of each of the three shards, the
complete execution root, the still-absent evidence root, the old and live
hashes of both authorized slots, every other parent binding as unchanged, the
finalization tool set, and the runner-recorded parent receipt pins.  It never
binds an evidence hash: evidence is produced after this receipt is sealed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import finalize_s_k10_h20_crossover as finalizer


UNIT_ID = finalizer.UNIT_ID
SCHEMA_VERSION = "s_k10_h20_crossover_finalization_receipt.v1"
STATUS = "sealed_post_execution"
RECEIPT_DIR_NAME = "finalization-receipt-v1"
RECEIPT_FILE_NAME = "finalization-receipt.json"
EVENT_IDS = finalizer.EVENT_IDS
EVENT_INDICES = finalizer.EVENT_INDICES
DEVICE_PLAN = dict(finalizer.DEVICE_PLAN)
SHARD_IDS = ("shard-000", "shard-001", "shard-002")
SHARD_ARTIFACTS = ("result.json", "runtime_identity.json", "terminal_summary.json", "aggregate.receipt.json")
# The authority names exactly these two slots and no generic override map.
AUTHORIZED_SLOTS: dict[str, str] = {
    "source_files.crossover_finalizer": (
        "validate declared file versus directory bindings with the parent sealer's "
        "deterministic inventory identity, and accept the runner-recorded "
        "pre_gpu_receipt_sha256 only when it equals the recomputed parent raw hash"
    ),
    "test_files.crossover_finalizer_test": (
        "cover both consumer repairs and the successor-receipt fail-closed contract"
    ),
}
FINALIZATION_TOOL_ROLES = ("finalizer", "finalizer_test", "successor_sealer", "successor_sealer_test")
PARENT_BINDING_GROUPS = ("input_bindings", "source_files", "test_files")

canonical_json_bytes = finalizer.canonical_json_bytes
sha256_bytes = finalizer.sha256_bytes
sha256_json = finalizer.sha256_json
sha256_file = finalizer.sha256_file
document_self_sha256 = finalizer.document_self_sha256


class FinalizationReceiptError(ValueError):
    """Raised when the successor boundary cannot be sealed exactly."""


def _guard(call: Any, *args: Any, **kwargs: Any) -> Any:
    """Reuse the finalizer's path/JSON contracts under this module's error."""

    try:
        return call(*args, **kwargs)
    except finalizer.EvidenceContractError as exc:
        raise FinalizationReceiptError(str(exc)) from exc


def _absolute(value: str | Path, label: str) -> Path:
    return _guard(finalizer._absolute, value, label)


def _regular_file(value: str | Path, label: str) -> Path:
    return _guard(finalizer._regular_file, value, label)


def _regular_dir(value: str | Path, label: str) -> Path:
    return _guard(finalizer._regular_dir, value, label)


def _read_json(value: str | Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _guard(finalizer._read_json, value, label)


def _sha(value: Any, label: str) -> str:
    return _guard(finalizer._sha, value, label)


def _self_hash(document: Mapping[str, Any], label: str) -> str:
    """Return a document's own semantic self hash under its declared field."""

    field = "self_sha256" if "self_sha256" in document else "result_sha256"
    if field not in document:
        raise FinalizationReceiptError(f"{label} carries no semantic self hash")
    declared = _sha(document[field], f"{label}.{field}")
    if declared != document_self_sha256(document, field):
        raise FinalizationReceiptError(f"{label}.{field} mismatch")
    return declared


def _file_ref(path: Path, label: str) -> dict[str, Any]:
    target = _regular_file(path, label)
    return {"path": str(target), "sha256": sha256_file(target), "size_bytes": target.stat().st_size}


def _document_ref(path: str | Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _regular_file(path, label)
    document, info = _read_json(target, label)
    return document, {
        "path": str(target),
        "raw_sha256": info["raw_sha256"],
        "self_sha256": _self_hash(document, label),
        "size_bytes": info["size_bytes"],
    }


def _parent_binding(document: Mapping[str, Any], slot: str) -> Mapping[str, Any]:
    group_name, _, role = slot.partition(".")
    group = document.get(group_name)
    if not isinstance(group, Mapping) or not isinstance(group.get(role), Mapping):
        raise FinalizationReceiptError(f"parent receipt lacks binding {slot}")
    return group[role]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_finalization_receipt(
    *,
    parent_receipt: str | Path,
    plan: str | Path,
    execution_root: str | Path,
    evidence_root: str | Path,
    authority: str | Path,
    finalizer_path: str | Path | None = None,
    finalizer_test_path: str | Path | None = None,
    sealer_path: str | Path | None = None,
    sealer_test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute every successor binding from live artifacts and return it."""

    parent_document, parent_ref = _document_ref(parent_receipt, "parent pre-GPU receipt")
    if parent_document.get("unit_id") != UNIT_ID or parent_document.get("status") != "sealed_pre_gpu":
        raise FinalizationReceiptError("parent receipt unit/status identity drifted")
    plan_document, plan_ref = _document_ref(plan, "plan")
    if plan_document.get("unit_id") != UNIT_ID:
        raise FinalizationReceiptError("plan unit identity drifted")
    if parent_document.get("plan_self_sha256") != plan_ref["self_sha256"]:
        raise FinalizationReceiptError("parent receipt binds a different plan")

    roots = parent_document.get("roots")
    if not isinstance(roots, Mapping):
        raise FinalizationReceiptError("parent receipt roots are missing")
    parent_execution = _absolute(roots["execution_root"]["path"], "parent execution root")
    parent_evidence = _absolute(roots["final_root"]["path"], "parent final root")

    execution = _regular_dir(execution_root, "execution root")
    if execution != parent_execution:
        raise FinalizationReceiptError("execution root differs from the parent-sealed execution root")
    if sorted(child.name for child in execution.iterdir()) != list(SHARD_IDS):
        raise FinalizationReceiptError("execution root is not exactly the three sealed shard roots")

    evidence = _absolute(evidence_root, "evidence root")
    if evidence != parent_evidence:
        raise FinalizationReceiptError("evidence root differs from the parent-sealed final root")
    if Path(evidence).is_symlink() or Path(evidence).exists():
        raise FinalizationReceiptError("evidence root must be absent and non-symlink at seal time")

    plan_events = plan_document.get("events")
    if not isinstance(plan_events, list) or len(plan_events) != 3:
        raise FinalizationReceiptError("plan does not bind exactly three events")

    shards: list[dict[str, Any]] = []
    for index, shard_id in enumerate(SHARD_IDS):
        root = _regular_dir(execution / shard_id, f"{shard_id} root")
        if sorted(child.name for child in root.iterdir()) != sorted(SHARD_ARTIFACTS):
            raise FinalizationReceiptError(f"{shard_id} is partial, duplicated, or carries a foreign file")
        artifacts: dict[str, Any] = {}
        for name in SHARD_ARTIFACTS:
            _document, ref = _document_ref(root / name, f"{shard_id}/{name}")
            artifacts[name] = ref
        result, _info = _read_json(root / "result.json", f"{shard_id}/result.json")
        runtime, _runtime_info = _read_json(root / "runtime_identity.json", f"{shard_id}/runtime_identity.json")
        event = plan_events[index]
        device = runtime.get("device")
        if not isinstance(device, Mapping) or device.get("physical_device") != DEVICE_PLAN[shard_id]:
            raise FinalizationReceiptError(f"{shard_id} physical device differs from the sealed device plan")
        if (
            result.get("shard_id") != shard_id
            or result.get("status") != "completed"
            or result.get("event_id") != EVENT_IDS[index]
            or result.get("event_id") != event.get("event_id")
            or result.get("event_index") != event.get("event_index")
            or result.get("image_id") != event.get("image_id")
            or result.get("plan_self_sha256") != plan_ref["self_sha256"]
            or result.get("pre_gpu_receipt_self_sha256") != parent_ref["self_sha256"]
        ):
            raise FinalizationReceiptError(f"{shard_id} result does not bind its planned event/parent identity")
        shards.append(
            {
                "shard_id": shard_id,
                "event_id": result["event_id"],
                "event_index": result["event_index"],
                "image_id": result["image_id"],
                "physical_device": DEVICE_PLAN[shard_id],
                "root": str(root),
                "artifacts": artifacts,
            }
        )

    tool_paths = {
        "finalizer": Path(finalizer_path) if finalizer_path is not None else Path(finalizer.__file__).resolve(),
        "finalizer_test": Path(finalizer_test_path)
        if finalizer_test_path is not None
        else _repo_root() / "tests" / "research" / "test_finalize_s_k10_h20_crossover.py",
        "successor_sealer": Path(sealer_path) if sealer_path is not None else Path(__file__).resolve(),
        "successor_sealer_test": Path(sealer_test_path)
        if sealer_test_path is not None
        else _repo_root() / "tests" / "research" / "test_seal_s_k10_h20_crossover_finalization_receipt.py",
    }
    tools = {role: _file_ref(tool_paths[role], f"finalization tool {role}") for role in FINALIZATION_TOOL_ROLES}

    slot_role_paths = {
        "source_files.crossover_finalizer": tools["finalizer"]["path"],
        "test_files.crossover_finalizer_test": tools["finalizer_test"]["path"],
    }
    authorized_drift: dict[str, Any] = {}
    for slot, cause in AUTHORIZED_SLOTS.items():
        group_name, _, role = slot.partition(".")
        binding = _parent_binding(parent_document, slot)
        old_sha = _sha(binding.get("sha256"), f"parent {slot}.sha256")
        live = _regular_file(binding.get("path"), f"{slot} live file")
        if str(live) != slot_role_paths[slot]:
            raise FinalizationReceiptError(f"{slot} live path differs from the finalization tool path")
        new_sha = sha256_file(live)
        if new_sha == old_sha:
            raise FinalizationReceiptError(f"{slot} has not drifted; the successor authority would be a no-op")
        authorized_drift[slot] = {
            "group": group_name,
            "role": role,
            "path": str(live),
            "old_sha256": old_sha,
            "new_sha256": new_sha,
            "size_bytes": live.stat().st_size,
            "cause": cause,
        }

    unchanged: dict[str, dict[str, Any]] = {}
    for group_name in PARENT_BINDING_GROUPS:
        group = parent_document.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise FinalizationReceiptError(f"parent receipt {group_name} are incomplete")
        entries: dict[str, Any] = {}
        for role, binding in sorted(group.items()):
            if f"{group_name}.{role}" in AUTHORIZED_SLOTS:
                continue
            checked = _guard(finalizer._ref, binding, f"parent {group_name}.{role}")
            entries[role] = {
                "path": checked["path"],
                "sha256": checked["sha256"],
                "kind": binding.get("kind", "file"),
            }
        unchanged[group_name] = entries

    authority_ref = _file_ref(Path(authority), "successor authority document")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "unit_id": UNIT_ID,
        "authorization": {
            "cpu_only": True,
            "gpu_used": False,
            "model_loaded": False,
            "no_training": True,
            "endpoint_semantics_unchanged": True,
            "authority": authority_ref,
            "authorized_slot_count": len(AUTHORIZED_SLOTS),
        },
        "parent": {
            "receipt": parent_ref,
            "plan": plan_ref,
            "document": dict(parent_document),
        },
        "execution": {
            "root": {"path": str(execution), "status": "complete_immutable", "shard_count": len(SHARD_IDS)},
            "shards": shards,
        },
        "evidence_root": {"path": str(evidence), "state": "absent_at_seal", "symlink": False},
        "authorized_drift": authorized_drift,
        "unchanged_parent_bindings": unchanged,
        "finalization_tools": tools,
        "runtime_input_pins": {
            "pre_gpu_receipt": parent_ref["raw_sha256"],
            "pre_gpu_receipt_sha256": parent_ref["raw_sha256"],
            "pre_gpu_receipt_self_sha256": parent_ref["self_sha256"],
        },
        "producer": _file_ref(Path(__file__).resolve(), "successor sealer producer"),
    }
    body["self_sha256"] = document_self_sha256(body)
    return body


def _write_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = _absolute(path, "finalization receipt output")
    if target.name != RECEIPT_FILE_NAME or target.parent.name != RECEIPT_DIR_NAME:
        raise FinalizationReceiptError(
            f"finalization receipt must be written as {RECEIPT_DIR_NAME}/{RECEIPT_FILE_NAME}"
        )
    payload = canonical_json_bytes(document) + b"\n"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable finalization receipt collision: {target}")
        return {"path": str(target), "raw_sha256": sha256_bytes(payload), "byte_identical": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return {"path": str(target), "raw_sha256": sha256_bytes(payload), "byte_identical": False}


def seal(
    *,
    parent_receipt: str | Path,
    plan: str | Path,
    execution_root: str | Path,
    evidence_root: str | Path,
    authority: str | Path,
    output: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the successor receipt and, when asked, write it exactly once."""

    document = build_finalization_receipt(
        parent_receipt=parent_receipt,
        plan=plan,
        execution_root=execution_root,
        evidence_root=evidence_root,
        authority=authority,
        **overrides,
    )
    result: dict[str, Any] = {"receipt": document, "self_sha256": document["self_sha256"]}
    if output is not None:
        result["write"] = _write_once(output, document)
    return result


def validate_finalization_receipt(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Re-validate a sealed receipt with the consumer's own contract."""

    return _guard(finalizer.validate_finalization_receipt, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help=f"{RECEIPT_DIR_NAME}/{RECEIPT_FILE_NAME}")
    parser.add_argument("--dry-run", action="store_true", help="build and print without writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output is None and not args.dry_run:
        print("blocked: --output is required unless --dry-run is given", file=sys.stderr)
        return 2
    try:
        result = seal(
            parent_receipt=args.parent_receipt,
            plan=args.plan,
            execution_root=args.execution_root,
            evidence_root=args.evidence_root,
            authority=args.authority,
            output=None if args.dry_run else args.output,
        )
    except (FinalizationReceiptError, FileExistsError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    summary = {
        "schema_version": result["receipt"]["schema_version"],
        "status": result["receipt"]["status"],
        "self_sha256": result["self_sha256"],
        "authorized_drift": sorted(result["receipt"]["authorized_drift"]),
        "write": result.get("write"),
    }
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
