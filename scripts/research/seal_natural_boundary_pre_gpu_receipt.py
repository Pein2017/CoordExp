#!/usr/bin/env python3
"""Seal and validate the natural-boundary pre-GPU launch receipt.

The pre-GPU receipt is intentionally a metadata-only artifact.  It binds the
already sealed research contract, the CPU census and support plan, the exact
S/H0 model identities, the code and test bytes that will run, and the installed
Qwen attention-mask probe.  It does not import a model, discover files with a
glob, start a process, or allocate a GPU.

``seal`` is write-once: an existing output is accepted only when its bytes are
identical.  ``validate`` re-hashes every strict binding, so a runtime cannot
silently pair a new runner or checkpoint with an old launch receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import sys
from typing import Any


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_pre_gpu_receipt.v4"
LEGACY_SCHEMA_VERSION = "natural_boundary_pre_gpu_receipt.v3"
RECEIPT_REVISION = "pre-gpu-receipt-v4"
CHECKPOINT = "S"
STEP = 2444
EVENT_ID = "S/gt:5001:15"
EVENT_OWNER_ID = "gt:5001:15"
EVENT_IMAGE_ID = 5001
EVENT_SOURCE_PANEL_OBJECT_INDEX = 15
RECEIPT_FILENAME = "pre-gpu-receipt.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_DIR = REPO_ROOT / (
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    f"{UNIT_ID}"
)
UNIT_PATH = UNIT_DIR / "unit.md"
SERIALIZATION_SUCCESSOR_AUTHORITY_PATH = UNIT_DIR / "serialization-successor-authority.md"
AUTHORITY_PATH = SERIALIZATION_SUCCESSOR_AUTHORITY_PATH
EXPECTED_AUTHORITY_SHA256 = "21af65617be1a47561474d3b27edff329677849d1443c23056da15ace065f724"
EXPECTED_AUTHORITY_STATUS = "active_user_authorized_exception"
EXPECTED_AUTHORITY_SOURCE_THREAD = "019fd043-5abb-7eb1-997c-d11a8316281a"
EXPECTED_AUTHORITY_SCOPE = "serialization_only_successor_exception"
AUTHORITY_SCOPE_MARKERS = (
    "Scope: `serialization_only_successor_exception`",
    "s-gt5001-live-gate-v3",
    "The exception does not authorize:",
)
AUDIT_JSON_PATH = UNIT_DIR / "prior-evidence-semantic-audit.json"
AUDIT_MD_PATH = UNIT_DIR / "prior-evidence-semantic-audit.md"
CONTRACT_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/contract-v1/contract.json"
)
CENSUS_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/cpu-census-v2"
)
CENSUS_PATH = CENSUS_ROOT / "admission-census.json"
CENSUS_RECORDS_PATH = CENSUS_ROOT / "admission-census.records.jsonl"
CENSUS_RECEIPT_PATH = CENSUS_ROOT / "admission-census.receipt.json"
# The census schema itself remains v1; the directory revision and exact
# immutable bytes distinguish the superseding geometry-bound census from the
# earlier v1 artifact.  Keep these hashes here so a default launch cannot
# silently fall back to the superseded census.
CENSUS_REVISION = "cpu-census-v2"
EXPECTED_CENSUS_REVISION = CENSUS_REVISION
EXPECTED_CENSUS_SHA256 = "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e"
EXPECTED_CENSUS_RECORDS_SHA256 = "41ebcb61c367fbf6e327f7d65e64df264cd6e92a0a82741ad3752b78fb088824"
EXPECTED_CENSUS_RECEIPT_SHA256 = "051afbc61e1b5356c366c9f95f7d68d0978b1b8457c2d02638c6868f111477fd"
EXPECTED_CENSUS_SELF_SHA256 = "11090ae4bc8c2f7f8a676f83e91194c2e0e7270a6d359c7dabf2cf771b38879d"
EXPECTED_EVENT_GEOMETRY_SHA256 = "dd92fd32e23a0ae67aca2f355e16885d5a008e9992ece54086eb6dd4893f2935"
SUPPORT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/support-completion-plan-v1"
)
SUPPORT_PLAN_PATH = SUPPORT_ROOT / "plan.json"
SUPPORT_PLAN_RECEIPT_PATH = SUPPORT_ROOT / "receipt.json"
SUPPORT_EXECUTION_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/support-execution-contract-v1"
)
SUPPORT_EXECUTION_RECEIPT_PATHS = tuple(
    SUPPORT_EXECUTION_ROOT / f"shard-{index}.json" for index in range(8)
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/{RECEIPT_REVISION}"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / RECEIPT_FILENAME
AUTHORIZED_GATE_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    f"{UNIT_ID}/s-gt5001-live-gate-v3"
)
GATE_OUTPUT_ROOT = AUTHORIZED_GATE_OUTPUT_ROOT
# Compatibility aliases keep command snippets from the launch handoff concise.
DEFAULT_CONTRACT = CONTRACT_PATH
DEFAULT_AUDIT_JSON = AUDIT_JSON_PATH
DEFAULT_AUDIT_MD = AUDIT_MD_PATH
DEFAULT_CENSUS = CENSUS_PATH
DEFAULT_CENSUS_RECORDS = CENSUS_RECORDS_PATH
DEFAULT_CENSUS_RECEIPT = CENSUS_RECEIPT_PATH
DEFAULT_SUPPORT_PLAN = SUPPORT_PLAN_PATH
DEFAULT_SUPPORT_PLAN_RECEIPT = SUPPORT_PLAN_RECEIPT_PATH

AUTHORED_CONFIG_PATH = REPO_ROOT / (
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
DEFAULT_H0_LEDGER_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
    "s-step2444-native-h0.json"
)
DEFAULT_H0_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/h0/"
    "qwen3-vl-2b-static-dynamic-owner-interface-s-step2444-h0-repair1"
)

DEFAULT_SOURCE_FILES: dict[str, Path] = {
    "natural_runner": REPO_ROOT / "scripts/research/run_natural_boundary_routing_history_probe.py",
    "s_gate_runner": REPO_ROOT / "scripts/research/run_s_primary_natural_boundary_gate.py",
    "support_runner": REPO_ROOT / "scripts/research/run_natural_boundary_support_completion.py",
    "support_planner": REPO_ROOT / "scripts/research/plan_natural_boundary_owner_support_completion.py",
    "finalizer": REPO_ROOT / "scripts/research/finalize_natural_boundary_routing_history_evidence.py",
    "attention_actuator": REPO_ROOT / "scripts/research/natural_boundary_attention_actuators.py",
    "residual_actuator": REPO_ROOT / "scripts/research/natural_boundary_residual_actuators.py",
    "pre_gpu_sealer": REPO_ROOT / "scripts/research/seal_natural_boundary_pre_gpu_receipt.py",
    "pre_gpu_materializer": REPO_ROOT / "scripts/research/materialize_natural_boundary_pre_gpu_evidence.py",
}
SOURCE_FILES = DEFAULT_SOURCE_FILES

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WILDCARD_CHARS = frozenset("*?[]")

# The names are deliberately boring.  They are a launch boundary, not an
# invitation to add an A2 or production route to this unit.
FORBIDDEN_SCOPE: dict[str, bool] = {
    "training": False,
    "A2": False,
    "wrapper_change": False,
    "token_change": False,
    "production_decode_change": False,
    "production_launch": False,
    "architecture_promotion": False,
}

REQUIRED_SOURCE_ROLES = (
    "natural_runner",
    "s_gate_runner",
    "support_runner",
    "finalizer",
    "attention_actuator",
    "residual_actuator",
    "pre_gpu_sealer",
    "pre_gpu_materializer",
)


class PreGpuReceiptError(ValueError):
    """Raised when a pre-GPU receipt cannot be established or replayed."""


# Friendly aliases used by launch wrappers and focused tests.
ReceiptError = PreGpuReceiptError
LaunchReceiptError = PreGpuReceiptError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreGpuReceiptError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def document_self_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("self_sha256", None)
    return sha256_json(payload)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PreGpuReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreGpuReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_positive_int(value: Any, label: str) -> int:
    """Read a JSON geometry field without coercing malformed values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise PreGpuReceiptError(f"{label} must be a positive integer")
    return int(value)


def _finite_mass_shape(value: Any, label: str) -> tuple[tuple[int, ...], list[float]]:
    """Return a rectangular JSON mass shape and flattened finite values."""

    flattened: list[float] = []

    def visit(node: Any, path: str) -> tuple[int, ...]:
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            if not node:
                raise PreGpuReceiptError(f"{label} contains an empty dimension at {path}")
            child_shapes = [visit(item, f"{path}[{index}]") for index, item in enumerate(node)]
            first = child_shapes[0]
            if any(shape != first for shape in child_shapes[1:]):
                raise PreGpuReceiptError(f"{label} is not rectangular")
            return (len(node), *first)
        if isinstance(node, bool) or not isinstance(node, Real):
            raise PreGpuReceiptError(f"{label} contains a non-numeric mass")
        numeric = float(node)
        if not math.isfinite(numeric):
            raise PreGpuReceiptError(f"{label} contains a non-finite mass")
        flattened.append(numeric)
        return ()

    shape = visit(value, label)
    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise PreGpuReceiptError(f"{label} must have shape [batch,q_heads,queries]")
    return shape, flattened


def _reject_wildcard(value: Any, label: str) -> None:
    if isinstance(value, str) and any(char in value for char in WILDCARD_CHARS):
        raise PreGpuReceiptError(f"{label} uses mutable wildcard discovery")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in {"glob", "wildcard", "pattern", "discover", "discovery"}:
                raise PreGpuReceiptError(f"{label} contains mutable {key_text} discovery")
            _reject_wildcard(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_wildcard(nested, f"{label}[{index}]")


def _regular_file(path: str | Path, label: str) -> Path:
    _reject_wildcard(str(path), label)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise PreGpuReceiptError(f"{label} must be an absolute path")
    if candidate.is_symlink():
        raise PreGpuReceiptError(f"{label} must be a regular non-symlink file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PreGpuReceiptError(f"{label} is missing: {candidate}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PreGpuReceiptError(f"{label} must be a regular non-symlink file: {candidate}")
    return resolved


def sha256_file(path: str | Path) -> str:
    resolved = _regular_file(path, "file")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot read {resolved}: {exc}") from exc
    return digest.hexdigest()


def _file_ref(path: str | Path, label: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != _sha(expected_sha256, f"{label}.sha256"):
        raise PreGpuReceiptError(f"{label} SHA-256 drifted")
    return {
        "label": label,
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def file_ref(path: str | Path, label: str = "file") -> dict[str, Any]:
    """Public exact-file identity helper used by runtime writers."""

    return _file_ref(path, label)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuReceiptError(f"cannot read {label} JSON: {resolved}") from exc
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} must be a JSON object")
    canonical_json_bytes(value)
    return dict(value)


def _authority_bindings(path: str | Path = AUTHORITY_PATH) -> dict[str, Any]:
    """Bind the exact user-owned serialization successor decision.

    The authority is intentionally a fixed regular file, rather than a
    discoverable document.  Its explicit status, source thread, and scope
    markers are checked in addition to the sealed byte hash so a copied or
    semantically altered authority cannot open the GPU boundary.
    """

    authority_ref = _file_ref(
        path,
        "serialization successor authority",
        expected_sha256=EXPECTED_AUTHORITY_SHA256,
    )
    expected_path = str(Path(AUTHORITY_PATH).resolve())
    if authority_ref["path"] != expected_path:
        raise PreGpuReceiptError(
            "serialization successor authority must be the exact regular authority file"
        )
    try:
        text = Path(authority_ref["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PreGpuReceiptError("cannot read serialization successor authority") from exc
    status_marker = f"Status: `{EXPECTED_AUTHORITY_STATUS}`"
    scope_marker = f"Scope: `{EXPECTED_AUTHORITY_SCOPE}`"
    source_marker = f"Source thread: `{EXPECTED_AUTHORITY_SOURCE_THREAD}`"
    if status_marker not in text:
        raise PreGpuReceiptError("serialization successor authority status marker is not active")
    if source_marker not in text:
        raise PreGpuReceiptError("serialization successor authority source thread marker drifted")
    if scope_marker not in text:
        raise PreGpuReceiptError("serialization successor authority scope marker drifted")
    for marker in AUTHORITY_SCOPE_MARKERS:
        if marker not in text:
            raise PreGpuReceiptError(
                f"serialization successor authority scope marker is missing: {marker}"
            )
    return {
        "file": authority_ref,
        "status": EXPECTED_AUTHORITY_STATUS,
        "source_thread": EXPECTED_AUTHORITY_SOURCE_THREAD,
        "scope": EXPECTED_AUTHORITY_SCOPE,
        "scope_markers": list(AUTHORITY_SCOPE_MARKERS),
    }


def _gate_output_root(
    path: str | Path | None = None,
    *,
    require_absent: bool = True,
) -> dict[str, Any]:
    """Validate one fresh, exact destination for the future S gate.

    No directory is created here.  The root must be absent at seal time and
    its direct parent must not be a symlink; this prevents accidental reuse of
    either old gate roots or a redirected output location.
    """

    value = GATE_OUTPUT_ROOT if path is None else path
    candidate = Path(value).expanduser()
    _reject_wildcard(str(candidate), "authorized gate output root")
    if not candidate.is_absolute():
        raise PreGpuReceiptError("authorized gate output root must be an absolute path")
    if candidate.name != "s-gt5001-live-gate-v3":
        raise PreGpuReceiptError(
            "authorized gate output root must end with s-gt5001-live-gate-v3"
        )
    parent = candidate.parent
    if parent.is_symlink():
        raise PreGpuReceiptError("authorized gate output root parent must not be a symlink")
    if parent.exists() and not parent.is_dir():
        raise PreGpuReceiptError("authorized gate output root parent must be a directory")
    # Reject a symlink in the existing ancestor chain without requiring any
    # parent directory to exist yet.
    ancestor = parent
    while True:
        if ancestor.is_symlink():
            raise PreGpuReceiptError(
                f"authorized gate output root ancestor must not be a symlink: {ancestor}"
            )
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    resolved = candidate.resolve(strict=False)
    if candidate.is_symlink():
        raise PreGpuReceiptError(
            f"authorized gate output root must be a regular non-symlink path: {resolved}"
        )
    if resolved.exists():
        if require_absent:
            raise PreGpuReceiptError(
                f"authorized gate output root must be absent and unused: {resolved}"
            )
        if not resolved.is_dir() or resolved.is_symlink():
            raise PreGpuReceiptError(
                f"authorized gate output root must be a regular non-symlink directory: {resolved}"
            )
    return {
        "path": str(resolved),
        "suffix": "s-gt5001-live-gate-v3",
        "status": "reserved_absent_pre_gpu",
        "parent": str(resolved.parent),
        "parent_is_symlink": False,
    }


def _write_json_once(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise PreGpuReceiptError("receipt output must be an absolute path")
    target = target.resolve()
    _reject_wildcard(str(target), "receipt output")
    payload = canonical_json_bytes(value) + b"\n"
    parent = target.parent
    if parent.exists() and parent.is_symlink():
        raise PreGpuReceiptError(f"output root must not be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise FileExistsError(f"output collision: {target} is not a regular file")
        existing = target.read_bytes()
        if existing != payload:
            raise FileExistsError(f"output collision: refusing to overwrite {target}")
        return {
            "status": "sealed",
            "path": str(target),
            "sha256": sha256_bytes(existing),
            "byte_identical": True,
        }
    # A caller may point at a pre-created run root.  Any unrelated entry means
    # that this is not a fresh immutable root and must not be reused.
    siblings = [item for item in parent.iterdir() if item.name != target.name]
    if siblings:
        raise FileExistsError(f"output root collision: {parent} is not empty")
    try:
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = target.read_bytes()
        if existing != payload:
            raise FileExistsError(f"output collision: refusing to overwrite {target}")
        return {
            "status": "sealed",
            "path": str(target),
            "sha256": sha256_bytes(existing),
            "byte_identical": True,
        }
    return {
        "status": "sealed",
        "path": str(target),
        "sha256": sha256_bytes(payload),
        "byte_identical": False,
    }


def _document_file_ref(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _nonempty_text(document.get("path"), f"{label}.path")
    digest = _sha(document.get("sha256"), f"{label}.sha256")
    ref = _file_ref(path, label, expected_sha256=digest)
    if "size_bytes" in document and document.get("size_bytes") != ref["size_bytes"]:
        raise PreGpuReceiptError(f"{label} size drifted")
    return ref


def _bind_input(path: str | Path | None, label: str) -> dict[str, Any]:
    if path is None:
        raise PreGpuReceiptError(f"{label} is missing")
    return _file_ref(path, label)


def _validate_contract(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = _file_ref(path, "sealed contract")
    document = _read_json(ref["path"], "sealed contract")
    if document.get("schema_version") != "natural_boundary_routing_history_contract.v1":
        raise PreGpuReceiptError("sealed contract schema is not the natural-boundary contract")
    if document.get("status") != "sealed" or document.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("sealed contract is not for this unit")
    if document.get("self_sha256") != document_self_sha256(document):
        raise PreGpuReceiptError("sealed contract self hash mismatch")
    boundary = document.get("boundary_contract")
    if not isinstance(boundary, Mapping):
        raise PreGpuReceiptError("sealed contract boundary_contract is missing")
    primary = boundary.get("primary")
    training = boundary.get("training")
    mutations = boundary.get("mutations")
    if not isinstance(primary, Mapping) or primary.get("checkpoint") != CHECKPOINT or primary.get("step") != 2444:
        raise PreGpuReceiptError("sealed contract does not bind S step-2444 as primary")
    if not isinstance(training, Mapping) or training.get("authorized") is not False:
        raise PreGpuReceiptError("sealed contract authorizes training")
    if not isinstance(mutations, Mapping) or any(value is True for value in mutations.values()):
        raise PreGpuReceiptError("sealed contract authorizes a forbidden mutation")
    return ref, document


def _audit_bindings(
    audit_path: str | Path | None,
    audit_json_path: str | Path | None,
    audit_md_path: str | Path | None,
    unit_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``audit_path`` is a convenience alias.  The JSON audit names the exact
    # Markdown bytes; no directory scan is ever used to locate it.
    if audit_path is not None:
        if audit_json_path is not None or audit_md_path is not None:
            raise PreGpuReceiptError("audit_path cannot be combined with audit_json_path/audit_md_path")
        candidate = _regular_file(audit_path, "audit")
        if candidate.suffix.lower() == ".json":
            audit_json_path = candidate
        else:
            audit_md_path = candidate
    if audit_json_path is None:
        audit_json_path = AUDIT_JSON_PATH
    if audit_md_path is None:
        audit_md_path = AUDIT_MD_PATH
    json_ref = _file_ref(audit_json_path, "prior-evidence semantic audit JSON")
    md_ref = _file_ref(audit_md_path, "prior-evidence semantic audit Markdown")
    audit = _read_json(json_ref["path"], "prior-evidence semantic audit")
    if audit.get("schema_version") != "prior_evidence_semantic_audit.v1":
        raise PreGpuReceiptError("audit schema is not the sealed prior-evidence audit")
    if audit.get("unit_id") != UNIT_ID or audit.get("status") != "complete_verified_bounded_narrowing":
        raise PreGpuReceiptError("audit is not complete for this unit")
    if audit.get("audit_document_path") not in {Path(md_ref["path"]).name, str(md_ref["path"]), str(Path(md_ref["path"]).resolve())}:
        raise PreGpuReceiptError("audit Markdown path is not the exact declared document")
    declared_md_hash = audit.get("audit_document_sha256")
    if declared_md_hash is not None and declared_md_hash != md_ref["sha256"]:
        raise PreGpuReceiptError("audit Markdown SHA-256 drifted")
    route = audit.get("route")
    if not isinstance(route, Mapping) or route.get("no_training_route_selected") is not True or route.get("s_role") != "primary_current_production_substrate":
        raise PreGpuReceiptError("audit route does not preserve the S/no-training boundary")
    hold = audit.get("checks", {}).get("s_hold") if isinstance(audit.get("checks"), Mapping) else None
    if not isinstance(hold, Mapping) or hold.get("event_id") != EVENT_ID:
        raise PreGpuReceiptError("audit does not bind S gt:5001:15")
    inputs = audit.get("inputs")
    if isinstance(inputs, Mapping) and inputs.get("new_unit_sha256") not in {None, unit_ref["sha256"]}:
        raise PreGpuReceiptError("audit new-unit hash differs from unit.md")
    return {
        "json": json_ref,
        "markdown": md_ref,
    }, audit


def _event_row(census: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = census.get("rows")
    if not isinstance(rows, list):
        raise PreGpuReceiptError("census rows are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("checkpoint") == CHECKPOINT
        and row.get("gt_owner_id") == EVENT_OWNER_ID
        and row.get("image_id") == EVENT_IMAGE_ID
        and row.get("source_panel_object_index") == EVENT_SOURCE_PANEL_OBJECT_INDEX
    ]
    if len(matches) != 1:
        raise PreGpuReceiptError("census must contain exactly one S gt:5001:15 row")
    return matches[0]


def _validate_census_revision(refs: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Reject a superseded ``cpu-census-v1`` path before reading semantics.

    Test fixtures may live in an arbitrary temporary directory, so the
    revision is inferred only from an explicit ``cpu-census-v*`` parent.  A
    real output tree must use one common v2 parent for the JSON, JSONL, and
    receipt; this prevents mixing a v1 envelope with v2 records (or vice
    versa) while preserving exact-file injection for focused tests.
    """

    parents = {Path(ref["path"]).parent.name for ref in refs.values()}
    versioned = {name for name in parents if name.startswith("cpu-census-v")}
    if versioned and versioned != {CENSUS_REVISION}:
        observed = ", ".join(sorted(versioned))
        raise PreGpuReceiptError(
            f"admission census must bind superseding {CENSUS_REVISION}; observed {observed}"
        )
    return CENSUS_REVISION if versioned else None


def _census_bindings(
    census_path: str | Path | None,
    records_path: str | Path | None,
    census_receipt_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    census_ref = _file_ref(census_path or CENSUS_PATH, "admission census")
    records_ref = _file_ref(records_path or CENSUS_RECORDS_PATH, "admission census records")
    receipt_ref = _file_ref(census_receipt_path or CENSUS_RECEIPT_PATH, "admission census receipt")
    revision = _validate_census_revision(
        {"census": census_ref, "records": records_ref, "receipt": receipt_ref}
    )
    # The live default is bound to the exact v2 bytes.  Custom fixture paths
    # remain valid for unit tests, but an explicit output-tree v1 path can
    # never reach the semantic checks below.
    is_coordexp_output = census_ref["path"].startswith("/data/CoordExp/outputs/")
    if is_coordexp_output and revision != CENSUS_REVISION:
        raise PreGpuReceiptError("admission census output path is not the accepted cpu-census-v2 revision")
    if is_coordexp_output:
        expected = (
            (census_ref, EXPECTED_CENSUS_SHA256, "admission census"),
            (records_ref, EXPECTED_CENSUS_RECORDS_SHA256, "admission census records"),
            (receipt_ref, EXPECTED_CENSUS_RECEIPT_SHA256, "admission census receipt"),
        )
        for ref, digest, label in expected:
            if ref["sha256"] != digest:
                raise PreGpuReceiptError(f"{label} is not the sealed {CENSUS_REVISION} artifact")
    census = _read_json(census_ref["path"], "admission census")
    if census.get("schema_version") != "natural_boundary_owner_admission_census.v1" or census.get("status") != "sealed" or census.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("admission census is not the sealed 784-row unit census")
    if census.get("self_sha256") != document_self_sha256(census):
        raise PreGpuReceiptError("admission census self hash mismatch")
    if is_coordexp_output and census.get("self_sha256") != EXPECTED_CENSUS_SELF_SHA256:
        raise PreGpuReceiptError("admission census self hash is not the sealed v2 identity")
    frozen = census.get("frozen_universe")
    if not isinstance(frozen, Mapping) or frozen.get("row_count") != 784 or frozen.get("physical_owner_count") != 392:
        raise PreGpuReceiptError("admission census universe is not 784 rows/392 owners")
    if is_coordexp_output and (not isinstance(census.get("rows"), list) or len(census["rows"]) != 784):
        raise PreGpuReceiptError("admission census output does not contain exactly 784 rows")
    records_bytes = Path(records_ref["path"]).read_bytes()
    records_hash = census.get("records_sha256")
    if records_hash != records_ref["sha256"]:
        raise PreGpuReceiptError("census records SHA-256 differs from census envelope")
    if sha256_bytes(records_bytes) != records_hash:
        raise PreGpuReceiptError("census records bytes drifted")
    record_lines = records_bytes.decode("utf-8").splitlines()
    if is_coordexp_output and len(record_lines) != 784:
        raise PreGpuReceiptError("admission census records output does not contain exactly 784 rows")
    for line_no, line in enumerate(record_lines, 1):
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PreGpuReceiptError(f"census records line {line_no} is not JSON") from exc
        if not isinstance(row, Mapping):
            raise PreGpuReceiptError(f"census records line {line_no} is not an object")
    census_receipt = _read_json(receipt_ref["path"], "admission census receipt")
    if census_receipt.get("schema_version") != "natural_boundary_owner_admission_census.v1.receipt" or census_receipt.get("status") != "sealed" or census_receipt.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("census receipt is not sealed for this unit")
    if census_receipt.get("self_sha256") != document_self_sha256(census_receipt):
        raise PreGpuReceiptError("census receipt self hash mismatch")
    if census_receipt.get("census_self_sha256") != census.get("self_sha256") or census_receipt.get("records_sha256") != records_hash:
        raise PreGpuReceiptError("census receipt does not bind census and records")
    if census_receipt.get("row_count") != 784 or census_receipt.get("physical_owner_count") != 392:
        raise PreGpuReceiptError("census receipt cardinality drifted")
    if census_receipt.get("support_completion_candidates_count") != 200:
        raise PreGpuReceiptError("census receipt does not bind 200 support candidates")
    row = _event_row(census)
    if row.get("disposition") != "eligible_verified_pair" or row.get("eligible_except_support") is not True:
        raise PreGpuReceiptError("S gt:5001:15 is not the sealed eligible census event")
    geometry = row.get("geometry")
    geometry_sha = geometry.get("geometry_sha256") if isinstance(geometry, Mapping) else None
    if revision == CENSUS_REVISION or is_coordexp_output:
        if geometry_sha != EXPECTED_EVENT_GEOMETRY_SHA256:
            raise PreGpuReceiptError("S gt:5001:15 geometry binding is not the sealed v2 identity")
    return {
        "census": census_ref,
        "records": records_ref,
        "receipt": receipt_ref,
        "revision": revision or (CENSUS_REVISION if census_ref["path"] == str(CENSUS_PATH.resolve()) else None),
    }, census, row


def _support_bindings(
    support_plan_path: str | Path | None,
    support_plan_receipt_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_ref = _file_ref(support_plan_path or SUPPORT_PLAN_PATH, "support-completion plan")
    receipt_ref = _file_ref(support_plan_receipt_path or SUPPORT_PLAN_RECEIPT_PATH, "support-completion plan receipt")
    plan = _read_json(plan_ref["path"], "support-completion plan")
    if plan.get("schema_version") != "natural_boundary_owner_support_completion_plan.v1" or plan.get("status") != "sealed_cpu_plan" or plan.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("support plan is not the sealed CPU plan")
    declared_content = _sha(plan.get("plan_content_sha256"), "support plan content hash")
    body = dict(plan)
    body.pop("plan_content_sha256", None)
    if sha256_json(body) != declared_content:
        raise PreGpuReceiptError("support plan content hash mismatch")
    if plan.get("checkpoint") != CHECKPOINT or plan.get("wrapper") != "object_box_closed":
        raise PreGpuReceiptError("support plan is not S/object_box_closed")
    execution = plan.get("execution_contract")
    if not isinstance(execution, Mapping) or execution.get("gpu_used") is not False or execution.get("model_loaded") is not False or execution.get("training") is not False:
        raise PreGpuReceiptError("support plan is not CPU-only")
    scope = plan.get("scope")
    work = plan.get("work")
    if not isinstance(scope, Mapping) or scope.get("support_completion_candidates") != 200 or scope.get("native_fn_denominator") != 220:
        raise PreGpuReceiptError("support plan scope is not the sealed 200/220 S denominator")
    if not isinstance(work, Mapping) or work.get("shard_count") != 8 or work.get("scalar_equivalent_forward_count") != 77428:
        raise PreGpuReceiptError("support plan work accounting drifted")
    plan_receipt = _read_json(receipt_ref["path"], "support-completion plan receipt")
    if plan_receipt.get("schema_version") != "natural_boundary_owner_support_completion_plan.v1.receipt.v1" or plan_receipt.get("status") != "sealed_cpu_plan" or plan_receipt.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("support plan receipt is not sealed CPU scope")
    if plan_receipt.get("plan_sha256") != plan_ref["sha256"] or plan_receipt.get("plan_content_sha256") != declared_content:
        raise PreGpuReceiptError("support plan receipt does not bind plan bytes")
    if plan_receipt.get("checkpoint") not in {None, CHECKPOINT}:
        raise PreGpuReceiptError("support plan receipt is not S checkpoint scope")
    if plan_receipt.get("context_count") != 200 or plan_receipt.get("support_completion_candidate_count") != 200 or plan_receipt.get("gpu_used") is not False or plan_receipt.get("model_loaded") is not False:
        raise PreGpuReceiptError("support plan receipt cardinality or CPU boundary drifted")
    if plan_receipt.get("execution_receipt_sha256") is not None:
        raise PreGpuReceiptError("support plan receipt already contains a realized GPU execution")
    return {
        "plan": plan_ref,
        "receipt": receipt_ref,
    }, plan, plan_receipt


def _support_execution_bindings(
    support_execution_receipts: Sequence[Any] | Mapping[Any, Any] | None,
    *,
    support_plan: Mapping[str, Any],
    census_refs: Mapping[str, Any],
    census: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind all eight CPU contract-summary receipts from the support runner.

    These are readiness receipts, not live model executions.  They are kept
    separate from the older sealed plan receipt because the plan predates the
    v2 geometry/hash repair.  Each summary must repeat the v2 census binding;
    the pre-GPU receipt also binds the census records and receipt directly via
    ``immutable_inputs``.
    """

    if support_execution_receipts is None:
        values: list[Any] = list(SUPPORT_EXECUTION_RECEIPT_PATHS)
    elif isinstance(support_execution_receipts, Mapping):
        _reject_wildcard(support_execution_receipts, "support execution receipts")
        values = []
        for index in range(8):
            if index in support_execution_receipts:
                values.append(support_execution_receipts[index])
            elif str(index) in support_execution_receipts:
                values.append(support_execution_receipts[str(index)])
            else:
                raise PreGpuReceiptError(f"support execution receipt for shard {index} is missing")
    elif isinstance(support_execution_receipts, Sequence) and not isinstance(
        support_execution_receipts, (str, bytes, bytearray)
    ):
        _reject_wildcard(support_execution_receipts, "support execution receipts")
        values = list(support_execution_receipts)
    else:
        raise PreGpuReceiptError("support_execution_receipts must be an exact eight-file sequence/object")
    if len(values) != 8:
        raise PreGpuReceiptError("support execution requires exactly eight shard contract receipts")

    rows = census.get("rows")
    if not isinstance(rows, list):
        raise PreGpuReceiptError("admission census rows are missing for support binding")
    owner_ids = sorted(
        str(row.get("gt_owner_id"))
        for row in rows
        if isinstance(row, Mapping)
        and row.get("checkpoint") == CHECKPOINT
        and row.get("native_fn") is True
        and row.get("disposition") == "support_unassessed"
    )
    if len(owner_ids) != 200 or len(set(owner_ids)) != 200:
        raise PreGpuReceiptError("admission census does not expose the sealed 200-owner support set")
    owner_ids_sha256 = sha256_json(owner_ids)
    expected_census = {
        "revision": CENSUS_REVISION,
        "path": census_refs["census"]["path"],
        "file_sha256": census_refs["census"]["sha256"],
        "self_sha256": census.get("self_sha256"),
        "s_owner_ids_sha256": owner_ids_sha256,
    }
    expected_plan_content = support_plan.get("plan_content_sha256")
    if not isinstance(expected_plan_content, str):
        raise PreGpuReceiptError("support plan content hash is missing for execution binding")
    plan_shards = support_plan.get("work", {}).get("per_shard") if isinstance(support_plan.get("work"), Mapping) else None
    expected_shards = {
        int(item.get("shard_index")): item
        for item in (plan_shards or ())
        if isinstance(item, Mapping)
    }
    if set(expected_shards) != set(range(8)):
        raise PreGpuReceiptError("support plan lacks exact per-shard accounting for execution binding")

    refs: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    seen_shards: set[int] = set()
    total_contexts = 0
    total_forwards = 0
    for position, value in enumerate(values):
        if isinstance(value, Mapping):
            path = value.get("path") or value.get("file")
            if path is None:
                raise PreGpuReceiptError(f"support execution contract shard {position} lacks an exact path")
            ref = _file_ref(
                path,
                f"support execution contract shard {position}",
                expected_sha256=value.get("sha256"),
            )
        else:
            ref = _file_ref(value, f"support execution contract shard {position}")
        document = _read_json(ref["path"], f"support execution contract shard {position}")
        if (
            document.get("schema_version")
            != "natural_boundary_owner_support_completion_execution.v1"
            or document.get("status") != "contract_ready"
            or document.get("unit_id") != UNIT_ID
        ):
            raise PreGpuReceiptError(f"support execution shard {position} is not a sealed CPU contract summary")
        if document.get("checkpoint") != CHECKPOINT or document.get("wrapper") != "object_box_closed" or document.get("parser") != "compact_object_box_closed_only":
            raise PreGpuReceiptError(f"support execution shard {position} is not S closed-wrapper scope")
        shard = document.get("shard_index")
        if isinstance(shard, bool) or not isinstance(shard, int) or not 0 <= shard < 8 or shard in seen_shards:
            raise PreGpuReceiptError(f"support execution shard {position} has a duplicate/invalid shard index")
        seen_shards.add(shard)
        if document.get("num_shards") != 8:
            raise PreGpuReceiptError(f"support execution shard {position} does not bind eight shards")
        if document.get("plan_content_sha256") != expected_plan_content:
            raise PreGpuReceiptError(f"support execution shard {position} plan content hash drifted")
        binding = document.get("census_binding")
        if not isinstance(binding, Mapping):
            raise PreGpuReceiptError(f"support execution shard {position} lacks census_binding")
        for key, expected in expected_census.items():
            if binding.get(key) != expected:
                raise PreGpuReceiptError(f"support execution shard {position} census binding {key} differs from v2")
        context_count = document.get("context_count")
        forward_count = document.get("scalar_equivalent_forward_count")
        if isinstance(context_count, bool) or not isinstance(context_count, int) or context_count <= 0:
            raise PreGpuReceiptError(f"support execution shard {position} context count is invalid")
        if isinstance(forward_count, bool) or not isinstance(forward_count, int) or forward_count <= 0:
            raise PreGpuReceiptError(f"support execution shard {position} forward count is invalid")
        expected_shard = expected_shards[shard]
        if context_count != expected_shard.get("context_count") or forward_count != expected_shard.get("scalar_equivalent_forward_count"):
            raise PreGpuReceiptError(f"support execution shard {position} context/forward totals differ from sealed plan")
        if document.get("batching_admitted") is not False or document.get("gpu_used") is not False or document.get("model_loaded") is not False:
            raise PreGpuReceiptError(f"support execution shard {position} crosses the CPU/batching boundary")
        estimates = document.get("batch_estimates")
        if not isinstance(estimates, Mapping) or estimates.get("batching_admitted") is not False:
            raise PreGpuReceiptError(f"support execution shard {position} batch estimates are not scalar-only")
        if document.get("legacy_frozen_candidate_registry_read") is not False or document.get("native_tp_calibration_scored") is not False:
            raise PreGpuReceiptError(f"support execution shard {position} changes the sealed control scope")
        refs[str(shard)] = ref
        documents[str(shard)] = document
        total_contexts += context_count
        total_forwards += forward_count

    if seen_shards != set(range(8)):
        raise PreGpuReceiptError("support execution contract receipts do not cover all shards")
    if total_contexts != 200 or total_forwards != 77428:
        raise PreGpuReceiptError("support execution contract receipts do not sum to 200 contexts/77428 forwards")
    return {
        "receipts": dict(sorted(refs.items(), key=lambda item: int(item[0]))),
        "shards": dict(sorted(documents.items(), key=lambda item: int(item[0]))),
        "shard_count": 8,
        "context_count": total_contexts,
        "scalar_equivalent_forward_count": total_forwards,
        "census_binding": expected_census,
    }, {
        "context_count": total_contexts,
        "scalar_equivalent_forward_count": total_forwards,
        "shard_count": 8,
        "census_binding": expected_census,
    }


def _source_entries(source_files: Mapping[str, Any] | Sequence[Any] | None) -> dict[str, dict[str, Any]]:
    raw: dict[str, Any]
    if source_files is None:
        raw = dict(DEFAULT_SOURCE_FILES)
    elif isinstance(source_files, Mapping):
        raw = dict(source_files)
    elif isinstance(source_files, Sequence) and not isinstance(source_files, (str, bytes, bytearray)):
        raw = {}
        for value in source_files:
            path = Path(str(value))
            name = path.name
            role = next((key for key, candidate in DEFAULT_SOURCE_FILES.items() if candidate.name == name), path.stem)
            raw[role] = value
    else:
        raise PreGpuReceiptError("source_files must be an object or array")
    _reject_wildcard(raw, "source_files")
    normalized: dict[str, dict[str, Any]] = {}
    aliases = {
        "runner": "natural_runner",
        "release_runner": "natural_runner",
        "gate_runner": "s_gate_runner",
        "support": "support_runner",
        "support_execution_runner": "support_runner",
        "attention": "attention_actuator",
        "residual": "residual_actuator",
        "aggregation_finalizer": "finalizer",
    }
    for key, value in raw.items():
        role = aliases.get(str(key), str(key))
        if isinstance(value, Mapping):
            path = value.get("path") or value.get("file")
            if path is None:
                raise PreGpuReceiptError(f"source_files.{role} lacks an exact path")
            expected = value.get("sha256")
            normalized[role] = _file_ref(path, f"source file {role}", expected_sha256=expected)
        else:
            normalized[role] = _file_ref(value, f"source file {role}")
        normalized[role]["role"] = role
    missing = [role for role in REQUIRED_SOURCE_ROLES if role not in normalized]
    if missing:
        raise PreGpuReceiptError(f"source_files missing conclusion-critical roles: {missing}")
    for role in ("pre_gpu_sealer", "pre_gpu_materializer"):
        expected_path = str(DEFAULT_SOURCE_FILES[role].resolve())
        if normalized[role]["path"] != expected_path:
            raise PreGpuReceiptError(
                f"source file {role} must bind the exact v4 implementation: {expected_path}"
            )
    if len(normalized) != len(set(normalized)):
        raise PreGpuReceiptError("source_files contain duplicate roles")
    return dict(sorted(normalized.items()))


def _focused_test_entries(
    focused_tests: Sequence[Any] | Mapping[str, Any] | None,
    test_receipt: str | Path | Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    if test_receipt is None:
        raise PreGpuReceiptError("focused test receipt is missing")
    receipt_ref: dict[str, Any] | None = None
    if isinstance(test_receipt, (str, Path)):
        receipt_ref = _file_ref(test_receipt, "focused test receipt")
        receipt = _read_json(receipt_ref["path"], "focused test receipt")
    elif isinstance(test_receipt, Mapping):
        _reject_wildcard(test_receipt, "focused test receipt")
        receipt = dict(test_receipt)
    else:
        raise PreGpuReceiptError("focused test receipt must be a JSON path or object")
    status = str(receipt.get("status", receipt.get("result", ""))).lower()
    passed = receipt.get("passed")
    exit_code = receipt.get("exit_code", receipt.get("returncode"))
    if status not in {"passed", "pass", "complete", "completed", "success"} and passed is not True:
        raise PreGpuReceiptError("focused test receipt is not clean/passed")
    if exit_code is not None and exit_code != 0:
        raise PreGpuReceiptError("focused test command returned non-zero")
    if receipt.get("fail_count", receipt.get("failed_count", 0)) not in {0, None}:
        raise PreGpuReceiptError("focused test receipt records failed tests")
    if receipt.get("errors", 0) not in {0, None} or receipt.get("failures", 0) not in {0, None}:
        raise PreGpuReceiptError("focused test receipt records errors/failures")
    command = receipt.get("command", receipt.get("passing_command"))
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
        command = " ".join(str(part) for part in command)
    if not isinstance(command, str) or not command.strip():
        raise PreGpuReceiptError("focused test receipt lacks the passing command")
    results = receipt.get("results")
    if isinstance(results, Mapping):
        for key in ("failed", "failures", "failed_count", "errors", "error_count"):
            if results.get(key) not in {None, 0, False}:
                raise PreGpuReceiptError("focused test receipt records failed results")
    entries_raw = receipt.get("focused_tests", receipt.get("tests", receipt.get("files")))
    if isinstance(entries_raw, Mapping):
        entries_raw = list(entries_raw.values())
    if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes, bytearray)) or not entries_raw:
        raise PreGpuReceiptError("focused test receipt lacks exact test file identities")
    recorded: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(entries_raw):
        if isinstance(item, Mapping):
            path = item.get("path") or item.get("file")
            digest = item.get("sha256")
            label = str(item.get("label", f"focused test {index}"))
        else:
            path, digest, label = item, None, f"focused test {index}"
        if path is None:
            raise PreGpuReceiptError(f"focused test {index} lacks an exact path")
        ref = _file_ref(path, label, expected_sha256=digest)
        recorded[ref["path"]] = ref
    if focused_tests is not None:
        supplied: list[Any]
        if isinstance(focused_tests, Mapping):
            supplied = list(focused_tests.values())
        elif isinstance(focused_tests, Sequence) and not isinstance(focused_tests, (str, bytes, bytearray)):
            supplied = list(focused_tests)
        else:
            raise PreGpuReceiptError("focused_tests must be an object or array")
        supplied_refs: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(supplied):
            if isinstance(item, Mapping):
                path = item.get("path") or item.get("file")
                digest = item.get("sha256")
            else:
                path, digest = item, None
            ref = _file_ref(path, f"focused test {index}", expected_sha256=digest)
            supplied_refs[ref["path"]] = ref
        if set(supplied_refs) != set(recorded) or any(supplied_refs[key]["sha256"] != recorded[key]["sha256"] for key in recorded):
            raise PreGpuReceiptError("focused test file identities disagree with passing receipt")
    tests = dict(sorted(recorded.items()))
    execution = {
        "status": "passed",
        "command": command.strip(),
        "exit_code": 0 if exit_code is None else exit_code,
        "test_count": len(tests),
        "tests": tests,
    }
    return receipt_ref, execution, receipt


def _mask_probe_binding(mask_probe: str | Path | Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if mask_probe is None:
        raise PreGpuReceiptError("installed-Qwen mask probe identity is missing")
    probe_ref: dict[str, Any] | None = None
    if isinstance(mask_probe, (str, Path)):
        probe_ref = _file_ref(mask_probe, "Qwen mask probe")
        probe = _read_json(probe_ref["path"], "Qwen mask probe")
    elif isinstance(mask_probe, Mapping):
        _reject_wildcard(mask_probe, "Qwen mask probe")
        probe = dict(mask_probe)
        path = probe.get("path")
        if path is not None:
            probe_ref = _file_ref(path, "Qwen mask probe")
            loaded = _read_json(probe_ref["path"], "Qwen mask probe")
            # The file is the authority; inline fields cannot silently shadow it.
            probe = loaded
    else:
        raise PreGpuReceiptError("Qwen mask probe must be a JSON path or object")
    status = str(probe.get("status", probe.get("disposition", ""))).lower()
    if status not in {"passed", "pass", "validated", "complete", "completed"} and probe.get("passed") is not True:
        raise PreGpuReceiptError("Qwen mask probe did not pass")
    installed = probe.get("installed") if isinstance(probe.get("installed"), Mapping) else probe
    torch_version = installed.get("torch_version", installed.get("torch"))
    transformers_version = installed.get("transformers_version", installed.get("transformers"))
    if isinstance(torch_version, Mapping):
        torch_version = torch_version.get("version")
    if isinstance(transformers_version, Mapping):
        transformers_version = transformers_version.get("version")
    if not isinstance(torch_version, str) or not torch_version.strip() or not isinstance(transformers_version, str) or not transformers_version.strip():
        raise PreGpuReceiptError("Qwen mask probe lacks installed torch/transformers versions")
    qwen = probe.get("qwen") if isinstance(probe.get("qwen"), Mapping) else probe
    qwen_class = qwen.get("model_class", qwen.get("class", qwen.get("qwen_model_class")))
    if not isinstance(qwen_class, str) or "qwen" not in qwen_class.lower():
        raise PreGpuReceiptError("Qwen mask probe lacks Qwen model identity")

    probe_values = probe.get("probes") if isinstance(probe.get("probes"), Mapping) else {}
    nested_probe = probe.get("mask_probe") if isinstance(probe.get("mask_probe"), Mapping) else {}

    def bool_alias(names: Sequence[str], label: str) -> None:
        for name in names:
            for container in (probe, probe_values, nested_probe):
                value = container.get(name) if isinstance(container, Mapping) else None
                if value is True or (isinstance(value, Mapping) and value.get("passed") is True):
                    return
        raise PreGpuReceiptError(f"Qwen mask probe lacks passing {label} attestation")

    bool_alias(("float_additive_4d_mask_passthrough", "float_additive_mask_passthrough", "float_4d_mask_passthrough"), "float-additive 4D mask pass-through")
    bool_alias(("all_layer_consumption", "all_layers_consumed", "all_layer_mask_consumption"), "all-layer mask consumption")
    if probe.get("silent_coercion") is True or probe.get("ignored_kwargs") or probe_values.get("silent_coercion") is True or probe_values.get("ignored_kwargs"):
        raise PreGpuReceiptError("Qwen mask probe reports silent coercion or ignored kwargs")

    # K14's CPU block-23 SDPA mass attestor is launch-critical.  It must
    # prove that the registry was restored and that the exact pre-existing
    # delegate was called once and left untouched; a generic ``passed`` flag
    # is insufficient because it can hide a fallback or a registry leak.
    attestation: Mapping[str, Any] | None = None
    attestation_flag: bool | None = None
    for container in (probe, probe_values, nested_probe):
        for key in (
            "block23_sdpa_mass_attestation",
            "k14_block23_sdpa_mass_attestation",
            "block23_sdpa_attestation",
        ):
            candidate = container.get(key) if isinstance(container, Mapping) else None
            if isinstance(candidate, Mapping):
                attestation = candidate
                break
            if candidate is not None:
                attestation_flag = candidate is True
        if attestation is not None:
            break
    if attestation is None:
        for container in (probe, probe_values, nested_probe):
            for key in ("block23_sdpa_mass_receipt", "k14_block23_sdpa_mass_receipt"):
                candidate = container.get(key) if isinstance(container, Mapping) else None
                if isinstance(candidate, Mapping):
                    attestation = candidate
                    break
            if attestation is not None:
                break
    if attestation is None:
        raise PreGpuReceiptError("Qwen mask probe lacks block23_sdpa_mass_attestation")
    # Some runners wrap the canonical attestor receipt in a phase/receipt
    # envelope.  Unwrap only named mappings; never search a directory or
    # discover an arbitrary object.
    attestation_envelope = attestation
    for key in ("receipt", "native_receipt", "attestor_receipt", "native"):
        nested = attestation.get(key)
        if isinstance(nested, Mapping) and not (
            "delegate_identity" in attestation and "registry_restored" in attestation
        ):
            attestation = nested
            break
    if isinstance(attestation_envelope.get("native"), Mapping) or isinstance(attestation_envelope.get("biased"), Mapping):
        if attestation_envelope.get("registry_restored") is not True or attestation_envelope.get("native_delegate_output_parity") is not True or attestation_envelope.get("biased_output_changed") is not True:
            raise PreGpuReceiptError("block23 SDPA envelope lacks native parity/K14 changed-output proof")
        comparison = attestation_envelope.get("comparison")
        if not isinstance(comparison, Mapping) or comparison.get("passed") is not True or comparison.get("mass_shift_observed") is not True:
            raise PreGpuReceiptError("block23 SDPA envelope lacks passed mass-shift comparison")
        native_delegate = attestation_envelope.get("native", {}).get("delegate_identity") if isinstance(attestation_envelope.get("native"), Mapping) else None
        biased_delegate = attestation_envelope.get("biased", {}).get("delegate_identity") if isinstance(attestation_envelope.get("biased"), Mapping) else None
        if not isinstance(native_delegate, Mapping) or not isinstance(biased_delegate, Mapping) or dict(native_delegate) != dict(biased_delegate):
            raise PreGpuReceiptError("block23 SDPA native/biased delegate identity differs")
    attestation_status = str(attestation.get("status", "")).lower()
    if attestation_flag is False or (attestation.get("passed") is not True and attestation_status not in {"passed", "pass", "validated", "complete", "completed"}):
        raise PreGpuReceiptError("block23_sdpa_mass_attestation did not pass")
    registry_restored = attestation.get("registry_restored")
    if registry_restored is not True:
        restoration = attestation.get("registry_restoration")
        if not isinstance(restoration, Mapping) or restoration.get("passed") is not True:
            raise PreGpuReceiptError("block23 SDPA attestation lacks registry restoration")
    if attestation.get("exactly_one_block23_call") is not True and attestation.get("block23_call_count") != 1:
        raise PreGpuReceiptError("block23 SDPA attestation lacks exactly one call")
    if attestation.get("delegate_untouched") is not True:
        raise PreGpuReceiptError("block23 SDPA attestation lacks untouched delegate proof")
    delegate = attestation.get("delegate_identity")
    if not isinstance(delegate, Mapping):
        raise PreGpuReceiptError("block23 SDPA attestation lacks exact delegate identity")
    if delegate.get("id") in {None, ""} or not isinstance(delegate.get("module"), str) or not delegate.get("module") or not isinstance(delegate.get("qualname"), str) or not delegate.get("qualname"):
        raise PreGpuReceiptError("block23 SDPA attestation delegate identity is incomplete")

    # The mass attestor runs on Qwen's grouped-query attention path.  A
    # receipt that only says "mass shifted" can accidentally describe a
    # 1:1 toy probe, so bind the producer's native geometry and every native /
    # biased per-head mass value.  Keep the check JSON-only: no model or torch
    # import belongs at this pre-GPU boundary.
    native_payload = attestation_envelope.get("native")
    biased_payload = attestation_envelope.get("biased")
    comparison = attestation_envelope.get("comparison")
    if not isinstance(native_payload, Mapping) or not isinstance(biased_payload, Mapping) or not isinstance(comparison, Mapping):
        raise PreGpuReceiptError("block23 SDPA envelope lacks native/biased mass receipts")

    def _single_record(payload: Mapping[str, Any], label: str) -> Mapping[str, Any]:
        records = payload.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)) or len(records) != 1 or not isinstance(records[0], Mapping):
            raise PreGpuReceiptError(f"{label} must contain exactly one mass record")
        return records[0]

    native_record = _single_record(native_payload, "native block23 SDPA receipt")
    biased_record = _single_record(biased_payload, "biased block23 SDPA receipt")
    for label, record in (("native", native_record), ("biased", biased_record)):
        for field in ("q_heads", "kv_heads", "num_key_value_groups", "groups"):
            if field not in record:
                raise PreGpuReceiptError(f"{label} block23 SDPA receipt lacks GQA field {field}")

    geometry_containers = (
        probe,
        probe_values,
        nested_probe,
        attestation_envelope,
        attestation,
        native_payload,
        biased_payload,
        native_record,
        biased_record,
        comparison,
    )

    def _geometry(names: Sequence[str], label: str) -> int:
        observed: list[int] = []
        for container in geometry_containers:
            if not isinstance(container, Mapping):
                continue
            for name in names:
                if name in container:
                    observed.append(_strict_positive_int(container[name], f"block23 {label}"))
        if not observed:
            raise PreGpuReceiptError(f"block23 SDPA receipts lack GQA field {label}")
        if any(value != observed[0] for value in observed[1:]):
            raise PreGpuReceiptError(f"block23 SDPA GQA {label} differs across receipts")
        return observed[0]

    q_heads = _geometry(("q_heads", "query_heads", "num_query_heads"), "q_heads")
    kv_heads = _geometry(("kv_heads", "key_value_heads", "num_key_value_heads"), "kv_heads")
    groups = _geometry(("groups", "num_key_value_groups", "gqa_groups"), "groups")
    if groups != 2 or q_heads != 16 or kv_heads != 8 or q_heads != kv_heads * groups:
        raise PreGpuReceiptError(
            "block23 SDPA receipt must prove production GQA q_heads=16, kv_heads=8, groups=2"
        )

    def _mass(container: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
        for name in names:
            if name in container:
                return container[name]
        raise PreGpuReceiptError(f"{label} is missing")

    native_record_shape, native_record_values = _finite_mass_shape(
        _mass(native_record, ("selected_mass", "native_mass"), "native selected_mass"),
        "native selected_mass",
    )
    biased_record_shape, biased_record_values = _finite_mass_shape(
        _mass(biased_record, ("selected_mass", "biased_mass"), "biased selected_mass"),
        "biased selected_mass",
    )
    native_shape, native_values = _finite_mass_shape(
        _mass(comparison, ("native_mass",), "comparison native_mass"),
        "comparison native_mass",
    )
    biased_shape, biased_values = _finite_mass_shape(
        _mass(comparison, ("biased_mass",), "comparison biased_mass"),
        "comparison biased_mass",
    )
    delta_shape, delta_values = _finite_mass_shape(
        _mass(comparison, ("delta_mass",), "comparison delta_mass"),
        "comparison delta_mass",
    )
    if not (native_shape == biased_shape == delta_shape == native_record_shape == biased_record_shape):
        raise PreGpuReceiptError("block23 native/biased per-head mass shapes differ")
    if native_shape[1] != q_heads:
        raise PreGpuReceiptError("block23 selected mass shape does not match q_heads")
    tolerance_value = comparison.get("tolerance", 0.0)
    if isinstance(tolerance_value, bool) or not isinstance(tolerance_value, Real) or not math.isfinite(float(tolerance_value)) or float(tolerance_value) < 0:
        raise PreGpuReceiptError("block23 mass comparison tolerance is invalid")
    tolerance = float(tolerance_value)
    compare_tolerance = max(tolerance, 1e-12)
    for index, (native_value, biased_value, delta_value) in enumerate(zip(native_values, biased_values, delta_values, strict=True)):
        if not math.isclose(delta_value, biased_value - native_value, rel_tol=1e-9, abs_tol=compare_tolerance):
            raise PreGpuReceiptError(f"block23 delta_mass disagrees with native/biased mass at index {index}")
    for label, observed, expected in (
        ("native", native_record_values, native_values),
        ("biased", biased_record_values, biased_values),
    ):
        if any(not math.isclose(left, right, rel_tol=1e-9, abs_tol=compare_tolerance) for left, right in zip(observed, expected, strict=True)):
            raise PreGpuReceiptError(f"block23 {label} record mass disagrees with comparison mass")
    all_query_heads_nonzero = bool(delta_values) and all(abs(value) > tolerance for value in delta_values)
    if not all_query_heads_nonzero:
        raise PreGpuReceiptError("block23 SDPA mass shift is not nonzero for every query head")
    if comparison.get("all_query_heads_nonzero_shift") is not True and comparison.get("all_head_nonzero_shift") is not True:
        raise PreGpuReceiptError("block23 mass comparison lacks all-query-head nonzero-shift proof")
    if comparison.get("mass_shift_observed") is not True or comparison.get("passed") is not True:
        raise PreGpuReceiptError("block23 SDPA envelope lacks passed mass-shift comparison")
    max_abs_delta = max(abs(value) for value in delta_values)
    min_abs_delta = min(abs(value) for value in delta_values)
    identity = {
        "status": "passed",
        "torch_version": torch_version.strip(),
        "transformers_version": transformers_version.strip(),
        "qwen_model_class": qwen_class,
        "float_additive_4d_mask_passthrough": True,
        "all_layer_consumption": True,
        "block23_sdpa_mass_attestation": {
            "schema_version": attestation.get("schema_version"),
            "status": "passed",
            "passed": True,
            "registry_restored": True,
            "exactly_one_block23_call": True,
            "block23_call_count": 1,
            "delegate_untouched": True,
            "delegate_identity": dict(delegate),
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "num_key_value_groups": groups,
            "groups": groups,
            "gqa": {
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "num_key_value_groups": groups,
                "groups": groups,
            },
            "selected_mass_shape": list(native_shape),
            "native_mass": native_values,
            "biased_mass": biased_values,
            "delta_mass": delta_values,
            "max_abs_delta_mass": max_abs_delta,
            "min_abs_delta_mass": min_abs_delta,
            "mass_shift_observed": True,
            "all_query_heads_nonzero_shift": True,
            "all_head_nonzero_shift": True,
            "same_delegate": True,
            "tolerance": tolerance,
        },
        "probe_sha256": None if probe_ref is None else probe_ref["sha256"],
    }
    return probe_ref, identity


def _path_from_nested(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    for key in (
        "adapter_path",
        "delta_path",
        "root_path",
        "path",
        "file",
        "source_path",
        "identity_path",
        "tensor_path",
        "metadata_path",
    ):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    for nested in value.values():
        if isinstance(nested, Mapping):
            found = _path_from_nested(nested)
            if found is not None:
                return found
        elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            for item in nested:
                if isinstance(item, Mapping):
                    found = _path_from_nested(item)
                    if found is not None:
                        return found
    return None


def _named_path_from_nested(value: Any, names: Sequence[str]) -> str | None:
    """Find one explicitly named file path without wildcard discovery."""

    if not isinstance(value, Mapping):
        return None
    wanted = {str(name) for name in names}
    for key, nested in value.items():
        if str(key) in wanted and isinstance(nested, str) and nested:
            return nested
    for nested in value.values():
        found = _named_path_from_nested(nested, names)
        if found is not None:
            return found
    return None


def _identity_source(value: Any, label: str) -> tuple[dict[str, Any], dict[str, Any] | None, Any]:
    _reject_wildcard(value, label)
    if isinstance(value, (str, Path)):
        path = _regular_file(value, label)
        return {"path": str(path), "sha256": sha256_file(path)}, _file_ref(path, label), None
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} must be an exact file path or object")
    source_path = value.get("path") or value.get("file") or value.get("source_path")
    source_ref = None
    if source_path is not None:
        source_ref = _file_ref(source_path, label, expected_sha256=value.get("sha256"))
    payload = dict(value)
    if source_ref is not None:
        payload["path"] = source_ref["path"]
        payload["sha256"] = source_ref["sha256"]
    else:
        canonical_json_bytes(payload)
    return payload, source_ref, None


def _default_h0_identity() -> dict[str, Any]:
    ledger_ref = _file_ref(DEFAULT_H0_LEDGER_PATH, "S H0 ledger")
    ledger = _read_json(ledger_ref["path"], "S H0 ledger")
    if ledger.get("checkpoint") != CHECKPOINT:
        raise PreGpuReceiptError("default H0 ledger is not S")
    infer = ledger.get("infer_config")
    resolved_path = infer.get("resolved", {}).get("path") if isinstance(infer, Mapping) else None
    if not isinstance(resolved_path, str):
        resolved_path = str(DEFAULT_H0_ROOT / "configs" / "resolved.json")
    resolved_ref = _file_ref(resolved_path, "S H0 resolved config")
    manifest_ref = _file_ref(DEFAULT_H0_ROOT / "run_manifest.json", "S H0 checkpoint manifest")
    summary_ref = _file_ref(DEFAULT_H0_ROOT / "summary.json", "S H0 checkpoint summary")
    manifest = _read_json(manifest_ref["path"], "S H0 checkpoint manifest")
    adapter = manifest.get("adapter_identity", {})
    embedding = manifest.get("embedding_delta_identity", {})
    return {
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "ledger": ledger_ref,
        "resolved_config": {"path": resolved_ref["path"], "sha256": resolved_ref["sha256"], "config_fingerprint": ledger.get("config_fingerprint")},
        "checkpoint_manifest": manifest_ref,
        "checkpoint_summary": summary_ref,
        "adapter": adapter,
        "embedding": embedding,
    }


def _model_identities(
    authored_config: str | Path | Mapping[str, Any] | None,
    h0_identity: str | Path | Mapping[str, Any] | None,
    identities: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provided = dict(identities or {})
    authored_value = authored_config if authored_config is not None else provided.get("authored_config", provided.get("authored_config_path", AUTHORED_CONFIG_PATH))
    h0_value = h0_identity if h0_identity is not None else provided.get("h0", provided.get("h0_identity", provided.get("resolved_h0")))
    authored_payload, authored_ref, _ = _identity_source(authored_value, "S authored config")
    if h0_value is None:
        h0_payload = _default_h0_identity()
        h0_ref = None
    else:
        h0_payload, h0_ref, _ = _identity_source(h0_value, "S H0 identity")
    # Accept the common nested runtime identity shape produced by the S gate.
    checkpoint_value = h0_payload.get("checkpoint")
    if checkpoint_value is None and isinstance(h0_payload.get("resolved"), Mapping):
        checkpoint_value = h0_payload["resolved"].get("checkpoint")
    if checkpoint_value != CHECKPOINT:
        raise PreGpuReceiptError(f"H0 identity checkpoint must be S, observed {checkpoint_value!r}")
    if h0_payload.get("step") not in {None, STEP}:
        raise PreGpuReceiptError("H0 identity is not step-2444")
    resolved_container = h0_payload.get("resolved_config", h0_payload.get("resolved", {}))
    runtime_h0 = (
        resolved_container.get("h0")
        if isinstance(resolved_container, Mapping) and isinstance(resolved_container.get("h0"), Mapping)
        else resolved_container
    )
    resolved = runtime_h0
    if not isinstance(resolved, Mapping):
        raise PreGpuReceiptError("H0 resolved config identity is missing")
    resolved_path = None
    if isinstance(resolved.get("root"), str):
        candidate = Path(str(resolved["root"])) / "configs" / "resolved.json"
        if candidate.is_file():
            resolved_path = str(candidate)
    if resolved_path is None:
        resolved_path = _path_from_nested(resolved)
    # ``run_s_primary_natural_boundary_gate.build_runtime_identity`` carries
    # the H0 root plus a resolved-config digest rather than repeating the
    # resolved file path.  Reconstruct that one exact path; this is not a
    # wildcard or directory discovery operation.
    if resolved_path is None and isinstance(resolved.get("root"), str):
        candidate = Path(str(resolved["root"])) / "configs" / "resolved.json"
        if candidate.is_file():
            resolved_path = str(candidate)
    if resolved_path is None:
        raise PreGpuReceiptError("H0 resolved config identity lacks an exact file")
    resolved_expected = resolved.get("sha256", resolved.get("resolved_config_sha256"))
    resolved_ref = _file_ref(resolved_path, "H0 resolved config", expected_sha256=resolved_expected)
    fingerprint = resolved.get("config_fingerprint", resolved.get("resolved_config_fingerprint", h0_payload.get("config_fingerprint")))
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise PreGpuReceiptError("H0 resolved config fingerprint is missing")
    # The exact model components may be nested in a run manifest or in the
    # compact adapter/embedding fields emitted by the gate adapter.
    adapter = h0_payload.get("adapter")
    embedding = h0_payload.get("embedding", h0_payload.get("embedding_delta"))
    runtime_h0 = resolved if isinstance(resolved, Mapping) else {}
    if adapter is None and isinstance(runtime_h0.get("adapter_path"), str):
        adapter_root = Path(runtime_h0["adapter_path"])
        adapter = {
            "path": str(adapter_root),
            "tensor_path": str(adapter_root / "adapter_model.safetensors"),
        }
    if embedding is None and isinstance(runtime_h0.get("embedding_delta_path"), str):
        embedding_root = Path(runtime_h0["embedding_delta_path"])
        embedding = {
            "path": str(embedding_root),
            "tensor_path": str(embedding_root / "special_token_embeddings.safetensors"),
        }
    if not isinstance(adapter, Mapping) or not isinstance(embedding, Mapping):
        model = h0_payload.get("model_identity") if isinstance(h0_payload.get("model_identity"), Mapping) else {}
        if not model and isinstance(h0_payload.get("resolved"), Mapping):
            nested_model = h0_payload["resolved"].get("model_identity")
            model = nested_model if isinstance(nested_model, Mapping) else {}
        adapter = adapter or model.get("adapter")
        embedding = embedding or model.get("embedding_delta")
    if not isinstance(adapter, Mapping) or not _path_from_nested(adapter):
        raise PreGpuReceiptError("S adapter identity is missing")
    if not isinstance(embedding, Mapping) or not _path_from_nested(embedding):
        raise PreGpuReceiptError("S embedding identity is missing")
    # Hash exact component files when supplied.  Directory roots are retained
    # as metadata, but a tensor/metadata file must still be present and bound.
    component_refs: dict[str, Any] = {}
    for name, value in (("adapter", adapter), ("embedding", embedding)):
        path = _path_from_nested(value)
        assert path is not None
        candidate = Path(path).expanduser()
        if candidate.is_symlink():
            raise PreGpuReceiptError(f"S {name} identity path is a symlink")
        if candidate.is_file():
            component_refs[name] = _file_ref(candidate, f"S {name} identity", expected_sha256=value.get("sha256"))
        elif candidate.is_dir():
            tensor_path = _named_path_from_nested(value, ("tensor_path",))
            if tensor_path is None:
                tensor_path = _named_path_from_nested(value, ("metadata_path",))
            if tensor_path is None:
                raise PreGpuReceiptError(f"S {name} identity directory lacks exact tensor/metadata file")
            component_refs[name] = {
                "root_path": str(candidate.resolve()),
                "file": _file_ref(tensor_path, f"S {name} tensor/metadata"),
            }
        else:
            raise PreGpuReceiptError(f"S {name} identity path is missing")
    result = {
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "authored_config": {"ref": authored_ref, "identity": authored_payload},
        "h0": {
            "ref": h0_ref,
            "identity": h0_payload,
            "resolved_config": {"ref": resolved_ref, "config_fingerprint": fingerprint},
            "adapter": component_refs["adapter"],
            "embedding": component_refs["embedding"],
        },
    }
    return result, {
        "authored_config": authored_ref,
        "h0_identity": h0_ref,
        "h0_resolved_config": resolved_ref,
        "adapter": component_refs["adapter"],
        "embedding": component_refs["embedding"],
    }


def _validate_forbidden_scope(scope: Mapping[str, Any] | None) -> dict[str, bool]:
    values = dict(FORBIDDEN_SCOPE)
    if scope is not None:
        if not isinstance(scope, Mapping):
            raise PreGpuReceiptError("forbidden_scope must be an object")
        aliases = {
            "wrapper": "wrapper_change",
            "token": "token_change",
            "production_decode": "production_decode_change",
            "a2": "A2",
        }
        for raw_key, value in scope.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key in values:
                if value is not False:
                    raise PreGpuReceiptError(f"forbidden scope {key} is not false")
                values[key] = False
            elif value is True:
                raise PreGpuReceiptError(f"unknown forbidden scope {key} is enabled")
    return values


def _validate_model_identity_document(model: Mapping[str, Any]) -> None:
    """Re-hash the model identity subset carried by a sealed receipt."""

    if model.get("checkpoint") != CHECKPOINT or model.get("step") != STEP:
        raise PreGpuReceiptError("model identity is not S step-2444")
    authored = model.get("authored_config")
    if not isinstance(authored, Mapping):
        raise PreGpuReceiptError("S authored config identity is missing")
    authored_ref = authored.get("ref")
    if authored_ref is not None:
        _validate_ref(authored_ref, "S authored config")
    h0 = model.get("h0")
    if not isinstance(h0, Mapping):
        raise PreGpuReceiptError("S H0 identity is missing")
    h0_identity = h0.get("identity")
    if isinstance(h0_identity, Mapping) and h0_identity.get("checkpoint") not in {None, CHECKPOINT}:
        raise PreGpuReceiptError("H0 identity contains a non-S checkpoint")
    h0_ref = h0.get("ref")
    if h0_ref is not None:
        _validate_ref(h0_ref, "S H0 identity")
    resolved = h0.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise PreGpuReceiptError("H0 resolved config identity is missing")
    resolved_ref = resolved.get("ref")
    if not isinstance(resolved_ref, Mapping):
        raise PreGpuReceiptError("H0 resolved config file identity is missing")
    _validate_ref(resolved_ref, "H0 resolved config")
    _nonempty_text(resolved.get("config_fingerprint"), "H0 resolved config fingerprint")
    for name in ("adapter", "embedding"):
        value = h0.get(name)
        if not isinstance(value, Mapping):
            raise PreGpuReceiptError(f"S {name} identity is missing")
        if isinstance(value.get("file"), Mapping):
            _validate_ref(value["file"], f"S {name} tensor/metadata")
        elif isinstance(value.get("path"), str):
            _validate_ref(value, f"S {name} identity")
        elif isinstance(value.get("root_path"), str):
            root = Path(value["root_path"]).expanduser()
            if root.is_symlink() or not root.is_dir():
                raise PreGpuReceiptError(f"S {name} identity root is not a regular directory")
            if isinstance(value.get("file"), Mapping):
                _validate_ref(value["file"], f"S {name} tensor/metadata")
            else:
                raise PreGpuReceiptError(f"S {name} identity root lacks an exact file")
        else:
            raise PreGpuReceiptError(f"S {name} identity lacks a file binding")


def build_receipt(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    contract_path: str | Path = CONTRACT_PATH,
    unit_path: str | Path = UNIT_PATH,
    authority_path: str | Path = AUTHORITY_PATH,
    gate_output_root: str | Path = GATE_OUTPUT_ROOT,
    audit_path: str | Path | None = None,
    audit_json_path: str | Path | None = None,
    audit_md_path: str | Path | None = None,
    census_path: str | Path | None = CENSUS_PATH,
    census_records_path: str | Path | None = CENSUS_RECORDS_PATH,
    census_receipt_path: str | Path | None = CENSUS_RECEIPT_PATH,
    support_plan_path: str | Path | None = SUPPORT_PLAN_PATH,
    support_plan_receipt_path: str | Path | None = SUPPORT_PLAN_RECEIPT_PATH,
    support_execution_receipts: Sequence[Any] | Mapping[Any, Any] | None = None,
    authored_config: str | Path | Mapping[str, Any] | None = None,
    h0_identity: str | Path | Mapping[str, Any] | None = None,
    identities: Mapping[str, Any] | None = None,
    source_files: Mapping[str, Any] | Sequence[Any] | None = None,
    focused_tests: Sequence[Any] | Mapping[str, Any] | None = None,
    test_receipt: str | Path | Mapping[str, Any] | None = None,
    mask_probe: str | Path | Mapping[str, Any] | None = None,
    forbidden_scope: Mapping[str, Any] | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Build a canonical pre-GPU receipt without writing it."""

    # Keep launch wrappers readable while accepting the spellings used by old
    # handoffs.  Unknown aliases fail rather than becoming hidden defaults.
    contract_path = aliases.pop("contract", contract_path)
    unit_path = aliases.pop("unit", unit_path)
    if test_receipt is None:
        test_receipt = aliases.pop("focused_test_receipt", aliases.pop("tests_receipt", None))
    if mask_probe is None:
        mask_probe = aliases.pop("qwen_mask_probe", aliases.pop("mask_probe_path", None))
    if h0_identity is None:
        h0_identity = aliases.pop("h0_identity_path", aliases.pop("runtime_identity", None))
    if support_execution_receipts is None:
        support_execution_receipts = aliases.pop(
            "support_execution_contract_receipts",
            aliases.pop("support_contract_receipts", aliases.pop("support_execution_paths", None)),
        )
    if aliases:
        raise PreGpuReceiptError(f"unknown build_receipt arguments: {sorted(aliases)}")
    _reject_wildcard(output, "receipt output")
    unit_ref = _file_ref(unit_path, "unit.md")
    if unit_ref["path"] != str(Path(UNIT_PATH).resolve()) and Path(unit_ref["path"]).name != "unit.md":
        raise PreGpuReceiptError("unit binding must be the natural-boundary unit.md")
    authority = _authority_bindings(authority_path)
    gate_root = _gate_output_root(gate_output_root, require_absent=True)
    contract_ref, contract = _validate_contract(contract_path)
    if contract.get("contract_files", {}).get("unit", {}).get("sha256") not in {None, unit_ref["sha256"]}:
        raise PreGpuReceiptError("sealed contract unit hash differs from supplied unit.md")
    audit_refs, audit = _audit_bindings(audit_path, audit_json_path, audit_md_path, unit_ref)
    census_refs, census, event_row = _census_bindings(census_path, census_records_path, census_receipt_path)
    support_refs, support_plan, support_plan_receipt = _support_bindings(support_plan_path, support_plan_receipt_path)
    support_execution_refs, support_execution_summary = _support_execution_bindings(
        support_execution_receipts,
        support_plan=support_plan,
        census_refs=census_refs,
        census=census,
    )
    model_identity, identity_refs = _model_identities(authored_config, h0_identity, identities)
    source_refs = _source_entries(source_files)
    test_ref, test_execution, test_document = _focused_test_entries(focused_tests, test_receipt)
    mask_ref, mask_identity = _mask_probe_binding(mask_probe)
    scope = _validate_forbidden_scope(forbidden_scope)
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        raise PreGpuReceiptError("receipt output must be an absolute path")
    output_path = output_path.resolve()
    if output_path.parent.name != RECEIPT_REVISION:
        raise PreGpuReceiptError(
            f"v4 receipt output must use an immutable {RECEIPT_REVISION} directory"
        )
    source_hashes = {role: value["sha256"] for role, value in source_refs.items()}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_revision": RECEIPT_REVISION,
        "status": "sealed_pre_gpu",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "event_binding": {
            "checkpoint": CHECKPOINT,
            "step": STEP,
            "event_id": EVENT_ID,
            "gt_owner_id": EVENT_OWNER_ID,
            "image_id": EVENT_IMAGE_ID,
            "source_panel_object_index": EVENT_SOURCE_PANEL_OBJECT_INDEX,
            "census_disposition": event_row.get("disposition"),
            "exact_prefix_sha256": event_row.get("exact_prefix_sha256"),
        },
        "contract_binding": {
            "file": contract_ref,
            "self_sha256": contract.get("self_sha256"),
        },
        "authority_binding": authority,
        "serialization_successor_authority": authority,
        "gate_output_root": gate_root["path"],
        "authorized_gate_output_root": gate_root["path"],
        "gate_output_binding": gate_root,
        "immutable_inputs": {
            "unit": unit_ref,
            "audit": audit_refs,
            "census": census_refs["census"],
            "census_records": census_refs["records"],
            "census_receipt": census_refs["receipt"],
            "census_revision": census_refs.get("revision"),
            "support_plan": support_refs["plan"],
            "support_plan_receipt": support_refs["receipt"],
            "support_execution_contracts": support_execution_refs["receipts"],
        },
        "support_execution_contract": support_execution_summary,
        "model_identity": model_identity,
        # Direct aliases make the receipt easy to consume from shell launch
        # wrappers while the structured sections above remain authoritative.
        "authored_config_identity": model_identity["authored_config"],
        "h0_identity": model_identity["h0"],
        "source_files": source_refs,
        "code_identity": {
            "required_roles": list(REQUIRED_SOURCE_ROLES),
            "sha256": source_hashes,
        },
        "focused_test_execution": test_execution,
        "focused_test_receipt": test_ref,
        "focused_test_file_hashes": test_execution["tests"],
        "mask_probe_identity": mask_identity,
        "mask_probe_receipt": mask_ref,
        "installed_qwen_mask_probe": mask_identity,
        "forbidden_scope": scope,
        "output_policy": {
            "receipt_path": str(output_path),
            "root": str(output_path.parent),
            "revision": RECEIPT_REVISION,
            "schema_version": SCHEMA_VERSION,
            "collision": "reject_nonempty_or_nonidentical",
            "overwrite": False,
            "repair": "new_immutable_root",
        },
        "runtime_identity_contract": {
            "required": True,
            "receipt_path_field": "pre_gpu_receipt_path",
            "receipt_sha256_field": "pre_gpu_receipt_sha256",
            "receipt_self_sha256_field": "pre_gpu_receipt_self_sha256",
            "code_hashes_field": "code_hashes",
            "receipt_path": str(output_path),
            "source_code_hashes": source_hashes,
            "repeat_exact_code_hashes": True,
        },
        "seal_inputs": {
            "authority_status": authority["status"],
            "authority_source_thread": authority["source_thread"],
            "authority_scope": authority["scope"],
            "gate_output_root": gate_root["path"],
            "audit_status": audit.get("status"),
            "census_self_sha256": census.get("self_sha256"),
            "census_records_sha256": census_refs["records"]["sha256"],
            "census_receipt_sha256": census_refs["receipt"]["sha256"],
            "support_plan_content_sha256": support_plan.get("plan_content_sha256"),
            "support_plan_receipt_status": support_plan_receipt.get("status"),
            "support_execution_contract_status": "validated_cpu_contract_summaries",
            "focused_test_status": test_document.get("status", test_document.get("result")),
            "mask_probe_status": mask_identity.get("status"),
        },
    }
    document["runtime_identity_requirements"] = document["runtime_identity_contract"]
    document["code_hashes"] = source_hashes
    document["self_sha256"] = document_self_sha256(document)
    return document


def _validate_ref(ref: Mapping[str, Any], label: str, *, strict: bool = True) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise PreGpuReceiptError(f"{label} is not a file identity")
    result = _document_file_ref(ref, label)
    if strict and result["sha256"] != ref.get("sha256"):
        raise PreGpuReceiptError(f"{label} hash drifted")
    return result


def validate_runtime_identity(
    identity: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    receipt: Mapping[str, Any] | None = None,
    expected_code_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail-closed validation for a runner's repeated receipt identity."""

    if not isinstance(identity, Mapping):
        raise PreGpuReceiptError("runtime identity must be an object")
    if identity.get("unit_id") != UNIT_ID or identity.get("checkpoint") != CHECKPOINT or identity.get("event_id") != EVENT_ID:
        raise PreGpuReceiptError("runtime identity unit/checkpoint/event differs from pre-GPU receipt")
    expected_path = str(_regular_file(receipt_path, "pre-GPU receipt"))
    path_value = identity.get("pre_gpu_receipt_path", identity.get("receipt_path"))
    self_value = identity.get("pre_gpu_receipt_self_sha256", identity.get("receipt_self_sha256"))
    digest_value = identity.get("pre_gpu_receipt_sha256", identity.get("receipt_sha256"))
    if path_value != expected_path:
        raise PreGpuReceiptError("runtime identity does not repeat the pre-GPU receipt path")
    expected_self = receipt.get("self_sha256") if isinstance(receipt, Mapping) else None
    if expected_self is not None and self_value != expected_self:
        raise PreGpuReceiptError("runtime identity pre-GPU receipt self hash differs")
    if digest_value is not None and digest_value != sha256_file(expected_path):
        raise PreGpuReceiptError("runtime identity pre-GPU receipt file hash differs")
    observed_codes = identity.get("code_hashes", identity.get("source_code_hashes"))
    if not isinstance(observed_codes, Mapping):
        raise PreGpuReceiptError("runtime identity does not repeat source code hashes")
    expected_codes = dict(expected_code_hashes or {})
    if not expected_codes and isinstance(receipt, Mapping):
        expected_codes = dict(receipt.get("code_identity", {}).get("sha256", {}))
    if dict(observed_codes) != expected_codes:
        raise PreGpuReceiptError("runtime identity source code hashes differ")
    return {
        "receipt_path": expected_path,
        "receipt_sha256": sha256_file(expected_path),
        "receipt_self_sha256": self_value,
        "code_hashes": dict(observed_codes),
    }


def validate_receipt(document: Mapping[str, Any], *, receipt_path: str | Path | None = None) -> None:
    """Re-hash and validate a previously materialized receipt."""

    if not isinstance(document, Mapping):
        raise PreGpuReceiptError("pre-GPU receipt must be a JSON object")
    required = {
        "schema_version",
        "receipt_revision",
        "status",
        "unit_id",
        "checkpoint",
        "step",
        "event_binding",
        "contract_binding",
        "authority_binding",
        "serialization_successor_authority",
        "gate_output_root",
        "authorized_gate_output_root",
        "gate_output_binding",
        "immutable_inputs",
        "support_execution_contract",
        "model_identity",
        "source_files",
        "code_identity",
        "focused_test_execution",
        "mask_probe_identity",
        "installed_qwen_mask_probe",
        "forbidden_scope",
        "output_policy",
        "runtime_identity_contract",
        "runtime_identity_requirements",
        "code_hashes",
        "self_sha256",
    }
    missing = sorted(required - set(document))
    if missing:
        raise PreGpuReceiptError(f"pre-GPU receipt is missing fields: {missing}")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("receipt_revision") != RECEIPT_REVISION
        or document.get("status") != "sealed_pre_gpu"
        or document.get("unit_id") != UNIT_ID
        or document.get("checkpoint") != CHECKPOINT
        or document.get("step") != STEP
    ):
        raise PreGpuReceiptError("pre-GPU receipt schema/status/unit mismatch")
    if document.get("self_sha256") != document_self_sha256(document):
        raise PreGpuReceiptError("pre-GPU receipt self hash mismatch")
    event = document.get("event_binding")
    if not isinstance(event, Mapping) or dict((key, event.get(key)) for key in ("checkpoint", "step", "event_id", "gt_owner_id", "image_id", "source_panel_object_index")) != {
        "checkpoint": CHECKPOINT,
        "step": STEP,
        "event_id": EVENT_ID,
        "gt_owner_id": EVENT_OWNER_ID,
        "image_id": EVENT_IMAGE_ID,
        "source_panel_object_index": EVENT_SOURCE_PANEL_OBJECT_INDEX,
    }:
        raise PreGpuReceiptError("pre-GPU receipt event binding is not S gt:5001:15")
    contract = document.get("contract_binding")
    if not isinstance(contract, Mapping):
        raise PreGpuReceiptError("contract binding is missing")
    contract_ref = contract.get("file")
    _validate_ref(contract_ref, "sealed contract")
    contract_doc = _read_json(contract_ref["path"], "sealed contract")
    _validate_contract(contract_ref["path"])
    if contract.get("self_sha256") != contract_doc.get("self_sha256") or contract_doc.get("self_sha256") != document_self_sha256(contract_doc):
        raise PreGpuReceiptError("contract self hash is not sealed")
    authority = document.get("authority_binding")
    if not isinstance(authority, Mapping):
        raise PreGpuReceiptError("serialization successor authority binding is missing")
    authority_file = authority.get("file")
    if not isinstance(authority_file, Mapping):
        raise PreGpuReceiptError("serialization successor authority file identity is missing")
    _validate_ref(authority_file, "serialization successor authority")
    rebound_authority = _authority_bindings(authority_file.get("path"))
    if dict(authority) != rebound_authority:
        raise PreGpuReceiptError("serialization successor authority binding drifted")
    if document.get("serialization_successor_authority") != authority:
        raise PreGpuReceiptError("serialization successor authority alias drifted")
    gate_binding = document.get("gate_output_binding")
    if not isinstance(gate_binding, Mapping):
        raise PreGpuReceiptError("authorized gate output root binding is missing")
    if document.get("authorized_gate_output_root") != gate_binding.get("path"):
        raise PreGpuReceiptError("authorized gate output root alias drifted")
    if document.get("gate_output_root") != gate_binding.get("path"):
        raise PreGpuReceiptError("gate output root alias drifted")
    rebound_gate = _gate_output_root(gate_binding.get("path"), require_absent=False)
    if dict(gate_binding) != rebound_gate:
        raise PreGpuReceiptError("authorized gate output root binding drifted")
    unit_ref = document["immutable_inputs"].get("unit")
    _validate_ref(unit_ref, "unit.md")
    if contract_doc.get("contract_files", {}).get("unit", {}).get("sha256") not in {None, unit_ref.get("sha256")}:
        raise PreGpuReceiptError("contract/unit identity mismatch")
    inputs = document.get("immutable_inputs")
    if not isinstance(inputs, Mapping):
        raise PreGpuReceiptError("immutable input bindings are missing")
    _validate_ref(inputs.get("unit"), "unit.md")
    audit = inputs.get("audit")
    if not isinstance(audit, Mapping) or set(audit) != {"json", "markdown"}:
        raise PreGpuReceiptError("audit JSON/Markdown bindings are incomplete")
    for label, ref in audit.items():
        _validate_ref(ref, f"audit {label}")
    _audit_bindings(
        None,
        audit_json_path=audit["json"]["path"],
        audit_md_path=audit["markdown"]["path"],
        unit_ref=unit_ref,
    )
    for key, label in (("census", "admission census"), ("census_records", "admission census records"), ("census_receipt", "admission census receipt"), ("support_plan", "support-completion plan"), ("support_plan_receipt", "support-completion plan receipt")):
        _validate_ref(inputs.get(key), label)
    if inputs.get("census_revision") != CENSUS_REVISION:
        raise PreGpuReceiptError("pre-GPU receipt is not bound to cpu-census-v2")
    execution_refs = inputs.get("support_execution_contracts")
    if not isinstance(execution_refs, Mapping):
        raise PreGpuReceiptError("support execution contract receipts are missing")
    # Re-run the semantic validators against the bound paths, catching any
    # change after seal rather than trusting copied summaries.
    census_refs, census_doc, _event = _census_bindings(inputs["census"]["path"], inputs["census_records"]["path"], inputs["census_receipt"]["path"])
    _support_plan_refs, support_plan_doc, _support_plan_receipt_doc = _support_bindings(inputs["support_plan"]["path"], inputs["support_plan_receipt"]["path"])
    _execution_refs, execution_summary = _support_execution_bindings(
        execution_refs,
        support_plan=support_plan_doc,
        census_refs=census_refs,
        census=census_doc,
    )
    if document.get("support_execution_contract") != execution_summary:
        raise PreGpuReceiptError("support execution contract summary alias drifted")
    seal_inputs = document.get("seal_inputs")
    if (
        not isinstance(seal_inputs, Mapping)
        or seal_inputs.get("authority_status") != EXPECTED_AUTHORITY_STATUS
        or seal_inputs.get("authority_source_thread") != EXPECTED_AUTHORITY_SOURCE_THREAD
        or seal_inputs.get("authority_scope") != EXPECTED_AUTHORITY_SCOPE
        or seal_inputs.get("gate_output_root") != gate_binding["path"]
        or seal_inputs.get("census_self_sha256") != census_doc.get("self_sha256")
        or seal_inputs.get("census_records_sha256") != inputs["census_records"].get("sha256")
        or seal_inputs.get("census_receipt_sha256") != inputs["census_receipt"].get("sha256")
    ):
        raise PreGpuReceiptError("direct v2 census dependency hashes drifted")
    _validate_forbidden_scope(document.get("forbidden_scope"))
    source_files = document.get("source_files")
    if not isinstance(source_files, Mapping):
        raise PreGpuReceiptError("source_files are missing")
    source_hashes: dict[str, str] = {}
    for role, ref in source_files.items():
        checked = _validate_ref(ref, f"source file {role}")
        source_hashes[str(role)] = checked["sha256"]
        if role in {"pre_gpu_sealer", "pre_gpu_materializer"}:
            expected_path = str(DEFAULT_SOURCE_FILES[role].resolve())
            if checked["path"] != expected_path:
                raise PreGpuReceiptError(
                    f"source file {role} must bind the exact v4 implementation"
                )
    code_identity = document.get("code_identity")
    if not isinstance(code_identity, Mapping) or dict(code_identity.get("sha256", {})) != source_hashes:
        raise PreGpuReceiptError("code identity hashes drifted")
    required_roles = code_identity.get("required_roles")
    if not isinstance(required_roles, list) or any(role not in source_files for role in REQUIRED_SOURCE_ROLES):
        raise PreGpuReceiptError("conclusion-critical source role is absent")
    tests = document.get("focused_test_execution")
    if not isinstance(tests, Mapping) or tests.get("status") != "passed" or tests.get("exit_code") != 0 or not isinstance(tests.get("command"), str) or not tests.get("command", "").strip():
        raise PreGpuReceiptError("focused test execution is not clean")
    test_map = tests.get("tests")
    if not isinstance(test_map, Mapping) or not test_map or tests.get("test_count") != len(test_map):
        raise PreGpuReceiptError("focused test file hashes are missing")
    for path, ref in test_map.items():
        _validate_ref(ref, f"focused test {path}")
    test_receipt_ref = document.get("focused_test_receipt")
    if test_receipt_ref is not None:
        _validate_ref(test_receipt_ref, "focused test receipt")
        _focused_test_entries(test_map, test_receipt_ref["path"])
    mask = document.get("mask_probe_identity")
    if not isinstance(mask, Mapping) or mask.get("status") != "passed" or mask.get("float_additive_4d_mask_passthrough") is not True or mask.get("all_layer_consumption") is not True:
        raise PreGpuReceiptError("installed-Qwen mask probe identity is not passed")
    block23_identity = mask.get("block23_sdpa_mass_attestation")
    if not isinstance(block23_identity, Mapping) or block23_identity.get("passed") is not True or block23_identity.get("registry_restored") is not True or block23_identity.get("exactly_one_block23_call") is not True or block23_identity.get("delegate_untouched") is not True or not isinstance(block23_identity.get("delegate_identity"), Mapping):
        raise PreGpuReceiptError("installed-Qwen mask probe lacks passed block23 SDPA attestation")
    if (
        block23_identity.get("groups") != 2
        or block23_identity.get("num_key_value_groups") != 2
        or not isinstance(block23_identity.get("q_heads"), Integral)
        or not isinstance(block23_identity.get("kv_heads"), Integral)
        or isinstance(block23_identity.get("q_heads"), bool)
        or isinstance(block23_identity.get("kv_heads"), bool)
        or block23_identity.get("q_heads") != 16
        or block23_identity.get("kv_heads") != 8
        or block23_identity.get("q_heads") != block23_identity.get("kv_heads") * 2
        or block23_identity.get("all_query_heads_nonzero_shift") is not True
        or block23_identity.get("mass_shift_observed") is not True
        or block23_identity.get("same_delegate") is not True
    ):
        raise PreGpuReceiptError("installed-Qwen mask probe lacks validated GQA mass-shift geometry")
    for field in ("native_mass", "biased_mass", "delta_mass"):
        if field not in block23_identity:
            raise PreGpuReceiptError(f"installed-Qwen mask probe lacks block23 {field}")
    mask_ref = document.get("mask_probe_receipt")
    if mask_ref is not None:
        _validate_ref(mask_ref, "Qwen mask probe")
        _, rebound_mask_identity = _mask_probe_binding(mask_ref["path"])
        if rebound_mask_identity != mask:
            raise PreGpuReceiptError("Qwen mask probe identity drifted after seal")
    output_policy = document.get("output_policy")
    if (
        not isinstance(output_policy, Mapping)
        or output_policy.get("revision") != RECEIPT_REVISION
        or output_policy.get("schema_version") != SCHEMA_VERSION
        or output_policy.get("collision") != "reject_nonempty_or_nonidentical"
        or output_policy.get("overwrite") is not False
        or not isinstance(output_policy.get("root"), str)
        or not Path(output_policy["root"]).is_absolute()
        or Path(output_policy["root"]).name != RECEIPT_REVISION
    ):
        raise PreGpuReceiptError("output collision policy is not fail-closed")
    runtime = document.get("runtime_identity_contract")
    if not isinstance(runtime, Mapping) or runtime.get("required") is not True or runtime.get("repeat_exact_code_hashes") is not True or runtime.get("source_code_hashes") != source_hashes:
        raise PreGpuReceiptError("runtime identity contract is incomplete")
    if receipt_path is not None:
        bound_path = str(_regular_file(receipt_path, "pre-GPU receipt"))
        if runtime.get("receipt_path") != bound_path:
            raise PreGpuReceiptError("runtime identity receipt path binding differs")
    model = document.get("model_identity")
    if not isinstance(model, Mapping):
        raise PreGpuReceiptError("model identity is missing")
    if model.get("checkpoint") != document.get("checkpoint") or model.get("step") != document.get("step"):
        raise PreGpuReceiptError("model identity checkpoint/step differs from receipt")
    _validate_model_identity_document(model)
    if document.get("authored_config_identity") != model.get("authored_config") or document.get("h0_identity") != model.get("h0"):
        raise PreGpuReceiptError("direct model identity aliases drifted")
    if document.get("installed_qwen_mask_probe") != document.get("mask_probe_identity"):
        raise PreGpuReceiptError("Qwen mask probe alias drifted")
    if document.get("runtime_identity_requirements") != runtime or document.get("code_hashes") != source_hashes:
        raise PreGpuReceiptError("runtime identity aliases drifted")
    if document.get("focused_test_file_hashes") != test_map:
        raise PreGpuReceiptError("focused test hash alias drifted")


def seal(output: str | Path = DEFAULT_OUTPUT, **kwargs: Any) -> dict[str, Any]:
    document = build_receipt(output=output, **kwargs)
    validate_receipt(document)
    # ``validate_receipt`` intentionally permits a completed gate root for
    # post-run audit.  Seal itself still owns the pre-GPU freshness boundary,
    # so re-check the destination immediately before writing the receipt.
    _gate_output_root(document["gate_output_root"], require_absent=True)
    result = _write_json_once(output, document)
    persisted = _read_json(result["path"], "sealed pre-GPU receipt")
    validate_receipt(persisted, receipt_path=result["path"])
    return {**result, "self_sha256": persisted["self_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    seal_parser = sub.add_parser("seal", help="build and write the immutable receipt")
    seal_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    seal_parser.add_argument("--contract", "--contract-path", dest="contract_path", type=Path, default=CONTRACT_PATH)
    seal_parser.add_argument("--unit", "--unit-path", dest="unit_path", type=Path, default=UNIT_PATH)
    seal_parser.add_argument("--authority", "--authority-path", dest="authority_path", type=Path, default=AUTHORITY_PATH)
    seal_parser.add_argument("--gate-output-root", type=Path, default=GATE_OUTPUT_ROOT)
    seal_parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON_PATH)
    seal_parser.add_argument("--audit-md", type=Path, default=AUDIT_MD_PATH)
    seal_parser.add_argument("--census", type=Path, default=CENSUS_PATH)
    seal_parser.add_argument("--census-records", type=Path, default=CENSUS_RECORDS_PATH)
    seal_parser.add_argument("--census-receipt", type=Path, default=CENSUS_RECEIPT_PATH)
    seal_parser.add_argument("--support-plan", type=Path, default=SUPPORT_PLAN_PATH)
    seal_parser.add_argument("--support-plan-receipt", type=Path, default=SUPPORT_PLAN_RECEIPT_PATH)
    seal_parser.add_argument(
        "--support-execution-receipt",
        action="append",
        type=Path,
        metavar="PATH",
        help="exact CPU contract-summary receipt; repeat once per shard (defaults to shard-0..7 paths)",
    )
    seal_parser.add_argument("--h0-identity", type=Path, default=None)
    seal_parser.add_argument("--authored-config", type=Path, default=None)
    seal_parser.add_argument("--test-receipt", type=Path, required=True)
    seal_parser.add_argument("--mask-probe", type=Path, required=True)
    seal_parser.add_argument("--source", action="append", metavar="ROLE=PATH", help="exact source role/path; repeat")
    validate_parser = sub.add_parser("validate", help="validate a sealed receipt")
    validate_parser.add_argument("receipt", type=Path)
    return parser


def _source_args(values: Sequence[str] | None) -> dict[str, str] | None:
    if values is None:
        return None
    result: dict[str, str] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path:
            raise PreGpuReceiptError("--source must be ROLE=PATH")
        if role in result:
            raise PreGpuReceiptError(f"--source repeats role {role}")
        result[role] = path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command is None:
            raise PreGpuReceiptError("command is required: use seal or validate")
        command = args.command
        if command == "validate":
            path = _regular_file(args.receipt, "pre-GPU receipt")
            document = _read_json(path, "pre-GPU receipt")
            validate_receipt(document, receipt_path=path)
            print(json.dumps({"status": "valid", "path": str(path), "self_sha256": document["self_sha256"]}, sort_keys=True))
            return 0
        if command != "seal":
            raise PreGpuReceiptError(f"unknown command {command!r}")
        result = seal(
            output=args.output,
            contract_path=args.contract_path,
            unit_path=args.unit_path,
            authority_path=args.authority_path,
            gate_output_root=args.gate_output_root,
            audit_json_path=args.audit_json,
            audit_md_path=args.audit_md,
            census_path=args.census,
            census_records_path=args.census_records,
            census_receipt_path=args.census_receipt,
            support_plan_path=args.support_plan,
            support_plan_receipt_path=args.support_plan_receipt,
            support_execution_receipts=args.support_execution_receipt,
            h0_identity=args.h0_identity,
            authored_config=args.authored_config,
            test_receipt=args.test_receipt,
            mask_probe=args.mask_probe,
            source_files=_source_args(args.source),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (PreGpuReceiptError, FileExistsError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


# Names used by launch wrappers in handoffs.
build_pre_gpu_receipt = build_receipt
validate_pre_gpu_receipt = validate_receipt
validate = validate_receipt


if __name__ == "__main__":
    raise SystemExit(main())
