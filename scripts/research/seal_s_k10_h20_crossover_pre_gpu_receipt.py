#!/usr/bin/env python3
"""Seal the S K10×H20 crossover at the CPU-only launch boundary.

The receipt is intentionally narrower than a general cohort authorization: it
binds one already-materialized plan, one exact three-shard device assignment,
and one no-training launch.  It refuses to seal while the successor runner or
finalizer (or any other conclusion-critical source/test) is absent.  No model,
CUDA context, or output execution root is created here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import materialize_s_k10_h20_crossover_plan as planner  # noqa: E402


UNIT_ID = planner.UNIT_ID
SCHEMA_VERSION = "s_k10_h20_crossover_pre_gpu_receipt.v1"
RUNTIME_IDENTITY_SCHEMA_VERSION = "s_k10_h20_crossover_runtime_identity_binding.v1"
PRE_GPU_PROBE_SCHEMA_VERSION = "s_k10_h20_crossover_event.v1.pre_gpu_probe.v1"
SOURCE_PREFLIGHT_SCHEMA_VERSION = (
    "s_k10_h20_crossover_event.v1.source_preflight.v1"
)
SOURCE_PREFLIGHT_ATTENTION_ARMS = (
    "H00",
    "H10",
    "H20",
    "K01",
    "K10",
    "K11",
    "K12",
    "K13",
    "K14B",
    "K14T",
)
STATUS = "sealed_pre_gpu"
PLAN_SCHEMA_VERSION = planner.SCHEMA_VERSION
UNIT_PATH = REPO_ROOT / (
    "docs/history/research-records/2026-09-15/investigations/"
    "qwen3-vl-dense-enumeration/experiments/2026-08-07-s-k10-h20-natural-crossover/unit.md"
)
DEVICE_PLAN = dict(planner.DEVICE_PLAN)
SHARD_COUNT = 3
H0_IDENTITY_FILE_ROLES = (
    "resolved_config",
    "run_manifest",
    "summary",
    "pred_token_trace",
    "image_plan",
    "adapter_config",
    "adapter_tensor",
    "embedding_metadata",
    "embedding_tensor",
)
BASE_MODEL_FILE_ROLES = (
    "config",
    "model_index",
    "weight_shard_1",
    "weight_shard_2",
    "tokenizer",
    "tokenizer_config",
    "added_tokens",
    "special_tokens_map",
    "chat_template_jinja",
    "chat_template_json",
    "preprocessor_config",
)

# These are deliberately exact paths, rather than a discovery glob.  The
# successor runner and finalizer are expected to be supplied by the adjacent
# implementation lane; until then sealing must fail cleanly.
CODE_ROLE_PATHS: dict[str, Path] = {
    "crossover_runner": REPO_ROOT / "scripts/research/run_s_k10_h20_crossover_shard.py",
    "attention_actuator": REPO_ROOT / "scripts/research/natural_boundary_attention_actuators.py",
    "residual_actuator": REPO_ROOT / "scripts/research/natural_boundary_residual_actuators.py",
    "base_gate": REPO_ROOT / "scripts/research/run_s_primary_natural_boundary_gate.py",
    "base_live_executor": REPO_ROOT / "scripts/research/s_natural_boundary_k_n_h_live_executor.py",
    "natural_runner": REPO_ROOT / "scripts/research/run_natural_boundary_routing_history_probe.py",
    "crossover_finalizer": REPO_ROOT / "scripts/research/finalize_s_k10_h20_crossover.py",
    "plan_materializer": REPO_ROOT / "scripts/research/materialize_s_k10_h20_crossover_plan.py",
    "pre_gpu_sealer": REPO_ROOT / "scripts/research/seal_s_k10_h20_crossover_pre_gpu_receipt.py",
}
CODE_ROLES = tuple(CODE_ROLE_PATHS)

TEST_PATHS: dict[str, Path] = {
    "crossover_runner_test": REPO_ROOT / "tests/research/test_run_s_k10_h20_crossover_shard.py",
    "crossover_finalizer_test": REPO_ROOT / "tests/research/test_finalize_s_k10_h20_crossover.py",
    "plan_materializer_test": REPO_ROOT / "tests/research/test_materialize_s_k10_h20_crossover_plan.py",
    "pre_gpu_sealer_test": REPO_ROOT / "tests/research/test_seal_s_k10_h20_crossover_pre_gpu_receipt.py",
    "attention_actuator_test": REPO_ROOT / "tests/research/test_natural_boundary_attention_actuators.py",
}
TEST_ROLES = tuple(TEST_PATHS)

REQUIRED_RUNTIME_KEYS = (
    "python_version",
    "torch_version",
    "transformers_version",
)


class PreGpuReceiptError(ValueError):
    """Raised for a missing, mutable, or semantically incompatible receipt input."""


ReceiptError = PreGpuReceiptError


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
    body = dict(document)
    body.pop("self_sha256", None)
    return sha256_json(body)


def _absolute(value: str | Path, label: str) -> Path:
    target = Path(value).expanduser()
    if not target.is_absolute():
        raise PreGpuReceiptError(f"{label} must be an absolute path")
    cursor = target
    while True:
        if cursor.is_symlink():
            raise PreGpuReceiptError(f"{label} must not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot resolve {label}: {target}") from exc
    if resolved != target:
        raise PreGpuReceiptError(f"{label} resolves through a symlink: {target}")
    return resolved


def _regular_file(value: str | Path, label: str) -> Path:
    target = _absolute(value, label)
    if target.is_symlink() or not target.is_file():
        raise PreGpuReceiptError(f"{label} must be an existing regular non-symlink file: {target}")
    return target


def _directory(value: str | Path, label: str, *, exists: bool = True) -> Path:
    target = _absolute(value, label)
    if target.is_symlink() or (exists and not target.is_dir()) or (not exists and target.exists()):
        state = "existing regular non-symlink directory" if exists else "absent non-symlink path"
        raise PreGpuReceiptError(f"{label} must be {state}: {target}")
    return target


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PreGpuReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def sha256_file(value: str | Path) -> str:
    path = _regular_file(value, "hash source")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _directory_inventory(value: str | Path, label: str) -> tuple[str, int]:
    root = _directory(value, label)
    entries: list[dict[str, Any]] = []
    try:
        for current_raw, dirs, files in os.walk(root, followlinks=False):
            current = Path(current_raw)
            dirs.sort()
            files.sort()
            for name in dirs:
                if (current / name).is_symlink():
                    raise PreGpuReceiptError(f"{label} contains a symlink directory: {current / name}")
            for name in files:
                path = current / name
                if path.is_symlink() or not path.is_file():
                    raise PreGpuReceiptError(f"{label} contains a non-regular file: {path}")
                entries.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot inventory {label}: {root}") from exc
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise PreGpuReceiptError(f"{label} inventory is empty")
    return sha256_json(entries), sum(item["size_bytes"] for item in entries)


def _directory_inventory_document(value: str | Path, label: str) -> dict[str, Any]:
    """Return the complete deterministic inventory used by the live loader."""

    root = _directory(value, label)
    entries: list[dict[str, Any]] = []
    try:
        for current_raw, dirs, files in os.walk(root, followlinks=False):
            current = Path(current_raw)
            dirs.sort()
            files.sort()
            for name in dirs:
                candidate = current / name
                if candidate.is_symlink():
                    raise PreGpuReceiptError(f"{label} contains a symlink directory: {candidate}")
            for name in files:
                candidate = current / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise PreGpuReceiptError(f"{label} contains a non-regular file: {candidate}")
                entries.append(
                    {
                        "relative_path": candidate.relative_to(root).as_posix(),
                        "sha256": sha256_file(candidate),
                        "size_bytes": candidate.stat().st_size,
                    }
                )
    except OSError as exc:
        raise PreGpuReceiptError(f"cannot inventory {label}: {root}") from exc
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise PreGpuReceiptError(f"{label} inventory is empty")
    return {
        "root": str(root),
        "file_count": len(entries),
        "files": entries,
        "inventory_sha256": sha256_json(entries),
    }


def _path_ref(value: Any, label: str, *, expected_path: str | Path | None = None, allow_directory: bool = True) -> dict[str, Any]:
    supplied_sha: str | None = None
    if isinstance(value, Mapping):
        raw_path = value.get("path") or value.get("file") or value.get("root")
        supplied_sha = value.get("sha256")
    else:
        raw_path = value
    if raw_path is None:
        raise PreGpuReceiptError(f"{label}.path is missing")
    target = _absolute(raw_path, label)
    if expected_path is not None and target != _absolute(expected_path, f"{label} expected path"):
        raise PreGpuReceiptError(f"{label} is not the exact expected path: {target}")
    if target.is_dir():
        if not allow_directory:
            raise PreGpuReceiptError(f"{label} must be a file")
        digest, size = _directory_inventory(target, label)
        kind = "directory"
    else:
        target = _regular_file(target, label)
        digest = sha256_file(target)
        size = target.stat().st_size
        kind = "file"
    if supplied_sha is not None:
        _sha(supplied_sha, f"{label}.sha256")
        if supplied_sha != digest:
            raise PreGpuReceiptError(f"{label} raw SHA-256 drifted")
    return {"path": str(target), "sha256": digest, "size_bytes": size, "kind": kind}


def _file_map(
    value: Mapping[str, Any] | None,
    roles: Sequence[str],
    label: str,
    root: Path,
    *,
    enforce_root: bool = True,
) -> dict[str, dict[str, Any]]:
    """Bind a complete named file set beneath one immutable directory."""

    if not isinstance(value, Mapping) or set(value) != set(roles):
        raise PreGpuReceiptError(f"{label} must contain exactly {tuple(roles)}")
    refs: dict[str, dict[str, Any]] = {}
    for role in roles:
        ref = _path_ref(value[role], f"{label}.{role}", allow_directory=False)
        path = Path(ref["path"])
        if enforce_root:
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise PreGpuReceiptError(f"{label}.{role} is outside bound root {root}") from exc
        refs[role] = ref
    return refs


def _file_map_sha(refs: Mapping[str, Mapping[str, Any]], roles: Sequence[str]) -> str:
    return sha256_json({role: refs[role]["sha256"] for role in roles})


def _json_ref(value: Any, label: str, *, canonical: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = _path_ref(value, label, allow_directory=False)
    path = Path(ref["path"])
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuReceiptError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise PreGpuReceiptError(f"{label} must contain a JSON object")
    document = dict(parsed)
    canonical_json_bytes(document)
    if canonical and raw != canonical_json_bytes(document) + b"\n":
        raise PreGpuReceiptError(f"{label} must be canonical JSON with one trailing newline")
    if "self_sha256" in document:
        _sha(document["self_sha256"], f"{label}.self_sha256")
        if document["self_sha256"] != document_self_sha256(document):
            raise PreGpuReceiptError(f"{label}.self_sha256 mismatch")
    semantic = (
        document.get("self_sha256")
        or document.get("result_sha256")
        or document.get("content_sha256")
        or document.get("plan_sha256")
        or sha256_json(document)
    )
    ref["semantic_sha256"] = semantic
    return ref, document


def _unit_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        # A bare path is accepted only for the exact owner unit file; metadata
        # such as unit_id and scope is not inferred from arbitrary text.
        ref = _path_ref(value, "unit authority", expected_path=UNIT_PATH, allow_directory=False)
        ref.update({"unit_id": UNIT_ID, "status": "active", "scope": "one fixed three-shard no-training launch"})
        return ref
    ref = _path_ref(value, "unit authority", expected_path=UNIT_PATH, allow_directory=False)
    if value.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("unit authority belongs to another unit")
    if value.get("status") not in {"active", "sealed", "active_user_authorized"}:
        raise PreGpuReceiptError("unit authority status is not active/sealed")
    if not isinstance(value.get("scope"), str) or not value["scope"].strip():
        raise PreGpuReceiptError("unit authority scope is missing")
    ref.update({"unit_id": UNIT_ID, "status": value["status"], "scope": value["scope"]})
    return ref


def _validate_plan(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = _path_ref(value, "crossover plan", allow_directory=False)
    try:
        document = json.loads(Path(ref["path"]).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuReceiptError(f"crossover plan is not readable JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise PreGpuReceiptError("crossover plan must contain an object")
    try:
        planner.validate_plan(ref["path"])
    except Exception as exc:
        raise PreGpuReceiptError(f"crossover plan fails deterministic validator: {exc}") from exc
    plan = dict(document)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("unit_id") != UNIT_ID or plan.get("status") != "planned":
        raise PreGpuReceiptError("crossover plan schema/status/unit identity drifted")
    if plan.get("no_training") is not True or plan.get("use_cache") is not False:
        raise PreGpuReceiptError("crossover plan crosses no-training/cache boundary")
    return ref, plan


def _normalize_shard_id(value: str | int) -> tuple[str, int]:
    if isinstance(value, bool):
        raise PreGpuReceiptError("shard_id must be shard-NNN or an integer 0..2")
    if isinstance(value, int):
        index = value
        shard = f"shard-{index:03d}"
    elif isinstance(value, str) and value.startswith("shard-"):
        try:
            index = int(value[6:])
        except ValueError as exc:
            raise PreGpuReceiptError("shard_id must be shard-NNN or an integer 0..2") from exc
        shard = value
    else:
        raise PreGpuReceiptError("shard_id must be shard-NNN or an integer 0..2")
    if not 0 <= index < SHARD_COUNT or shard != f"shard-{index:03d}":
        raise PreGpuReceiptError(f"shard_id is outside the frozen 0..{SHARD_COUNT - 1} range")
    return shard, index


def _validate_roots(
    execution_root: Any,
    final_root: Any,
    *,
    phase: str = "prelaunch",
    shard_id: str | int | None = None,
) -> dict[str, Any]:
    if phase not in {"prelaunch", "runtime"}:
        raise PreGpuReceiptError("root validation phase must be prelaunch or runtime")
    if phase == "prelaunch":
        execution = _directory(execution_root, "execution root", exists=False)
        final = _directory(final_root, "final root", exists=False)
        status = "reserved_absent_pre_gpu"
    else:
        execution = _absolute(execution_root, "execution root")
        final = _absolute(final_root, "final root")
        # A shard process may arrive after one or two siblings have completed.
        # The execution parent is therefore allowed to exist, but only with
        # exact shard directories; the current shard target and final root stay
        # absent until this process owns them.
        if final.exists():
            raise PreGpuReceiptError("final root must remain absent during shard runtime")
        if execution.exists():
            if execution.is_symlink() or not execution.is_dir():
                raise PreGpuReceiptError("execution root is not a regular directory")
            for child in sorted(execution.iterdir()):
                if child.is_symlink() or not child.is_dir():
                    raise PreGpuReceiptError(f"execution root contains an unexpected entry: {child}")
                try:
                    _normalize_shard_id(child.name)
                except PreGpuReceiptError as exc:
                    raise PreGpuReceiptError(f"execution root contains an unexpected entry: {child}") from exc
        if shard_id is not None:
            shard, _ = _normalize_shard_id(shard_id)
            target = execution / shard
            if target.exists() or target.is_symlink():
                raise PreGpuReceiptError(f"current shard output root must be absent: {target}")
        status = "authorized_runtime_root"
    if execution == final:
        raise PreGpuReceiptError("execution and final roots must be distinct")
    return {
        "execution_root": {"path": str(execution), "status": status},
        "final_root": {"path": str(final), "status": status},
    }


def _installed_runtime() -> dict[str, str]:
    try:
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": metadata.version("torch"),
            "transformers_version": metadata.version("transformers"),
        }
    except metadata.PackageNotFoundError as exc:
        raise PreGpuReceiptError(f"installed runtime package is unavailable: {exc.name}") from exc


def _runtime(value: Mapping[str, Any] | None) -> dict[str, Any]:
    observed = _installed_runtime()
    runtime = dict(value or {})
    for key in REQUIRED_RUNTIME_KEYS:
        if runtime.get(key) != observed[key]:
            raise PreGpuReceiptError(f"runtime.{key} differs from installed CPU-observed version {observed[key]}")
    defaults = {
        "backend": "hf",
        "dtype": "fp32",
        "attn_implementation": "sdpa",
        "no_training": True,
        "gpu_used": False,
        "model_loaded": False,
    }
    for key, default in defaults.items():
        runtime.setdefault(key, default)
    if runtime["backend"].lower() not in {"hf", "huggingface"} or runtime["dtype"].lower() not in {"fp32", "float32"} or runtime["attn_implementation"].lower() != "sdpa" or runtime["no_training"] is not True or runtime["gpu_used"] is not False or runtime["model_loaded"] is not False:
        raise PreGpuReceiptError("runtime must attest CPU-only HF fp32/SDPA no-training")
    canonical_json_bytes(runtime)
    return runtime


def _validate_focused(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _json_ref(value, "focused CPU evidence")
    if document.get("status") not in {"passed", "complete"}:
        raise PreGpuReceiptError("focused CPU evidence is not passing")
    if document.get("cpu_only") is not True or document.get("gpu_used") is not False or document.get("model_loaded") is not False:
        raise PreGpuReceiptError("focused CPU evidence crosses GPU/model boundary")
    tests = document.get("focused_tests") or document.get("tests")
    if tests is not None and (not isinstance(tests, Sequence) or isinstance(tests, (str, bytes, bytearray))):
        raise PreGpuReceiptError("focused CPU evidence test identities are malformed")
    return ref, document


def _validate_source_preflight_envelope(value: Any) -> dict[str, Any]:
    """Reject an absent or self-asserted preseal result before deep binding."""

    if not isinstance(value, Mapping):
        raise PreGpuReceiptError("source preflight evidence is missing")
    document = dict(value)
    if (
        document.get("schema_version") != SOURCE_PREFLIGHT_SCHEMA_VERSION
        or document.get("status") != "passed"
        or document.get("phase") != "preseal_model_free_production"
        or document.get("unit_id") != UNIT_ID
        or document.get("receipt_independent") is not True
    ):
        raise PreGpuReceiptError(
            "source preflight must be passing, preseal, receipt-independent evidence"
        )
    if document.get("self_sha256") != document_self_sha256(document):
        raise PreGpuReceiptError("source preflight self_sha256 mismatch")
    for key in (
        "gpu_used",
        "model_loaded",
        "backend_session_opened",
        "output_root_created",
    ):
        if document.get(key) is not False:
            raise PreGpuReceiptError(f"source preflight {key} must be false")
    return document


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise PreGpuReceiptError(f"{label} fields are incomplete or unexpected")


def _require_false_flags(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    for key in keys:
        if value.get(key) is not False:
            raise PreGpuReceiptError(f"{label}.{key} must be false")


def _validate_unattested_consumption(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} is missing")
    receipt = dict(value)
    if (
        receipt.get("required") is not True
        or receipt.get("status") != "unattested"
        or receipt.get("exact_same_tensor_all_layers_required") is not True
        or receipt.get("passed") is True
    ):
        raise PreGpuReceiptError(
            f"{label} must remain required and unattested before a model forward"
        )
    declared_layers = receipt.get("declared_layer_count")
    if declared_layers is not None and declared_layers != 28:
        raise PreGpuReceiptError(f"{label} does not declare all 28 layers")
    canonical_json_bytes(receipt)
    return receipt


def _validate_full_runtime_cohort(
    value: Any,
    *,
    input_refs: Mapping[str, Mapping[str, Any]],
    manifest_doc: Mapping[str, Any],
) -> dict[str, Any]:
    label = "source preflight full-runtime cohort"
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} is missing")
    receipt = dict(value)
    required = {
        "status",
        "event_count",
        "event_identities_sha256",
        "authoritative_bindings_sha256",
        "processor_context_bindings",
        "processor_context_bindings_sha256",
        "cohort_path",
        "cohort_sha256",
        "cohort_manifest_path",
        "cohort_manifest_sha256",
        "receipt_sha256",
    }
    _require_exact_fields(receipt, required, label)
    bindings = receipt.get("processor_context_bindings")
    if (
        receipt.get("status") != "passed"
        or receipt.get("event_count") != 11
        or not isinstance(bindings, list)
        or len(bindings) != 11
        or any(
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("event_id"), str)
            or not binding.get("event_id")
            for binding in bindings
        )
        or len({binding["event_id"] for binding in bindings}) != 11
    ):
        raise PreGpuReceiptError(f"{label} is not the exact eleven-event receipt")
    for key in (
        "event_identities_sha256",
        "authoritative_bindings_sha256",
        "processor_context_bindings_sha256",
        "cohort_sha256",
        "cohort_manifest_sha256",
        "receipt_sha256",
    ):
        _sha(receipt.get(key), f"{label}.{key}")
    if receipt["processor_context_bindings_sha256"] != sha256_json(bindings):
        raise PreGpuReceiptError(f"{label} processor-context hash mismatch")
    manifest_events = manifest_doc.get("events")
    if (
        manifest_doc.get("event_count") != 11
        or not isinstance(manifest_events, list)
        or len(manifest_events) != 11
    ):
        raise PreGpuReceiptError(
            "source preflight bound manifest is not the exact eleven-event authority"
        )
    manifest_event_ids: list[str] = []
    admitted_identities: list[tuple[Any, ...]] = []
    for event in manifest_events:
        if not isinstance(event, Mapping):
            raise PreGpuReceiptError(
                "source preflight bound manifest event is malformed"
            )
        owner_refs = event.get("owner_refs")
        event_id = event.get("event_id")
        if (
            event.get("checkpoint") != "S"
            or not isinstance(event_id, str)
            or not event_id
            or not isinstance(owner_refs, Mapping)
            or not isinstance(owner_refs.get("gt_owner_id"), str)
            or isinstance(event.get("image_id"), bool)
            or not isinstance(event.get("image_id"), int)
            or any(
                isinstance(owner_refs.get(key), bool)
                or not isinstance(owner_refs.get(key), int)
                for key in (
                    "source_panel_object_index",
                    "derived_panel_object_index",
                )
            )
        ):
            raise PreGpuReceiptError(
                "source preflight bound manifest event identity is malformed"
            )
        manifest_event_ids.append(event_id)
        admitted_identities.append(
            (
                "S",
                owner_refs["gt_owner_id"],
                event["image_id"],
                owner_refs["source_panel_object_index"],
                owner_refs["derived_panel_object_index"],
            )
        )
    observed_event_ids = [binding["event_id"] for binding in bindings]
    if observed_event_ids != manifest_event_ids:
        raise PreGpuReceiptError(
            f"{label} event order/membership differs from the bound manifest"
        )
    if receipt.get("event_identities_sha256") != sha256_json(admitted_identities):
        raise PreGpuReceiptError(
            f"{label} event identity hash differs from the bound manifest"
        )
    expected_source_bindings = {
        "cohort_path": input_refs["cohort"]["path"],
        "cohort_sha256": input_refs["cohort"]["sha256"],
        "cohort_manifest_path": input_refs["cohort_manifest"]["path"],
        "cohort_manifest_sha256": input_refs["cohort_manifest"]["sha256"],
    }
    if any(
        receipt.get(key) != expected
        for key, expected in expected_source_bindings.items()
    ):
        raise PreGpuReceiptError(
            f"{label} cohort/manifest bindings differ from current refs"
        )
    body = dict(receipt)
    observed_receipt_sha = body.pop("receipt_sha256")
    if observed_receipt_sha != sha256_json(body):
        raise PreGpuReceiptError(f"{label} receipt_sha256 mismatch")
    return receipt


def _validate_factory_metadata(
    value: Any,
    *,
    input_refs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    label = "source preflight factory model metadata"
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} is missing")
    receipt = dict(value)
    required = {
        "status",
        "source",
        "base_model_dir",
        "base_model_inventory_sha256",
        "config_path",
        "config_raw_sha256",
        "layer_count",
        "head_count",
        "device",
        "executable_model_present",
        "model_loader_called",
        "receipt_sha256",
    }
    _require_exact_fields(receipt, required, label)
    if (
        receipt.get("status") != "passed"
        or receipt.get("source") != "sealed_base_model_config_metadata_only"
        or receipt.get("base_model_dir") != input_refs["base_model_dir"]["path"]
        or receipt.get("base_model_inventory_sha256")
        != input_refs["base_model_dir"]["sha256"]
        or receipt.get("device") != "cpu"
        or receipt.get("executable_model_present") is not False
        or receipt.get("model_loader_called") is not False
    ):
        raise PreGpuReceiptError(f"{label} crossed the model-free boundary")
    for key in ("layer_count", "head_count"):
        count = receipt.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PreGpuReceiptError(f"{label}.{key} must be a positive integer")
    config_path = _regular_file(receipt.get("config_path"), f"{label}.config_path")
    try:
        config_path.relative_to(Path(input_refs["base_model_dir"]["path"]))
    except ValueError as exc:
        raise PreGpuReceiptError(f"{label}.config_path is outside the base model") from exc
    if receipt.get("config_raw_sha256") != sha256_file(config_path):
        raise PreGpuReceiptError(f"{label}.config_raw_sha256 mismatch")
    body = dict(receipt)
    observed_receipt_sha = body.pop("receipt_sha256")
    if observed_receipt_sha != sha256_json(body):
        raise PreGpuReceiptError(f"{label}.receipt_sha256 mismatch")
    return receipt


def _validate_source_preflight_context(
    value: Any,
    *,
    expected_event: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"source preflight event {expected_event['event_id']}"
    if not isinstance(value, Mapping):
        raise PreGpuReceiptError(f"{label} context is missing")
    context = dict(value)
    required = {
        "event_id",
        "event_index",
        "image_id",
        "exact_history_sha256",
        "seeded_prefix_sha256",
        "natural_context_sha256",
        "natural_identity_sha256",
        "attention_factory_arms",
        "attention_factory_contracts",
        "built_factory_receipts",
        "k14_reference_positions",
        "c11_receipt",
    }
    _require_exact_fields(context, required, label)
    for key in ("event_id", "event_index", "image_id"):
        if context.get(key) != expected_event.get(key):
            raise PreGpuReceiptError(f"{label}.{key} differs from the exact plan event")
    for key in (
        "exact_history_sha256",
        "seeded_prefix_sha256",
        "natural_context_sha256",
        "natural_identity_sha256",
    ):
        _sha(context.get(key), f"{label}.{key}")
    if context.get("exact_history_sha256") != expected_event.get("prefix_sha256"):
        raise PreGpuReceiptError(
            f"{label}.exact_history_sha256 differs from the exact plan prefix"
        )

    arms = context.get("attention_factory_arms")
    contracts = context.get("attention_factory_contracts")
    if (
        arms != list(SOURCE_PREFLIGHT_ATTENTION_ARMS)
        or not isinstance(contracts, Mapping)
        or set(contracts) != set(SOURCE_PREFLIGHT_ATTENTION_ARMS)
    ):
        raise PreGpuReceiptError(f"{label} attention factory set/order drifted")
    factory_fields = {
        "protocol",
        "arm_id",
        "image_key_positions_sha256",
        "b_exclusive_positions_sha256",
        "latest_row_key_positions_sha256",
        "layer_count",
        "head_count",
        "device",
    }
    for arm in SOURCE_PREFLIGHT_ATTENTION_ARMS:
        contract = contracts[arm]
        if not isinstance(contract, Mapping):
            raise PreGpuReceiptError(f"{label} {arm} factory contract is missing")
        _require_exact_fields(contract, factory_fields, f"{label} {arm} factory")
        if contract.get("arm_id") != arm or contract.get("device") != "cpu":
            raise PreGpuReceiptError(f"{label} {arm} factory identity/device drifted")
        for key in (
            "image_key_positions_sha256",
            "b_exclusive_positions_sha256",
            "latest_row_key_positions_sha256",
        ):
            _sha(contract.get(key), f"{label} {arm}.{key}")
        for key in ("layer_count", "head_count"):
            count = contract.get(key)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise PreGpuReceiptError(f"{label} {arm}.{key} is invalid")

    built = context.get("built_factory_receipts")
    if not isinstance(built, Mapping) or set(built) != {"K01", "K10", "H20"}:
        raise PreGpuReceiptError(f"{label} built K01/K10/H20 receipts are incomplete")
    built_fields = {
        "status",
        "mask_sha256",
        "selected_positions",
        "layer_consumption_attestation",
        "all_layer_consumption_attestation",
        "no_op_parity",
        "receipt_sha256",
    }
    for arm in ("K01", "K10", "H20"):
        receipt = built[arm]
        if not isinstance(receipt, Mapping):
            raise PreGpuReceiptError(f"{label} built {arm} receipt is missing")
        _require_exact_fields(receipt, built_fields, f"{label} built {arm}")
        if receipt.get("status") != "ready":
            raise PreGpuReceiptError(f"{label} built {arm} is not ready")
        _sha(receipt.get("mask_sha256"), f"{label} built {arm}.mask_sha256")
        _sha(receipt.get("receipt_sha256"), f"{label} built {arm}.receipt_sha256")
        positions = receipt.get("selected_positions")
        if not isinstance(positions, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in positions
        ):
            raise PreGpuReceiptError(f"{label} built {arm} positions are malformed")
        _validate_unattested_consumption(
            receipt.get("layer_consumption_attestation"),
            f"{label} built {arm}.layer_consumption_attestation",
        )
        _validate_unattested_consumption(
            receipt.get("all_layer_consumption_attestation"),
            f"{label} built {arm}.all_layer_consumption_attestation",
        )
    k01_parity = built["K01"].get("no_op_parity")
    if (
        not isinstance(k01_parity, Mapping)
        or k01_parity.get("required") is not True
        or k01_parity.get("status") != "unassessed"
    ):
        raise PreGpuReceiptError(f"{label} K01 no-op parity was not left unassessed")

    k14 = context.get("k14_reference_positions")
    if not isinstance(k14, Mapping) or set(k14) != {"K14T", "K14B"}:
        raise PreGpuReceiptError(f"{label} K14T/K14B geometry is incomplete")
    for arm in ("K14T", "K14B"):
        positions = k14[arm]
        if (
            not isinstance(positions, list)
            or not positions
            or any(isinstance(item, bool) or not isinstance(item, int) for item in positions)
            or positions != sorted(set(positions))
        ):
            raise PreGpuReceiptError(f"{label} {arm} positions are empty or malformed")
        if contracts[arm].get("arm_id") != arm:
            raise PreGpuReceiptError(f"{label} {arm} positions do not match its factory")
    if contracts["K14T"].get("b_exclusive_positions_sha256") != sha256_json(
        k14["K14T"]
    ):
        raise PreGpuReceiptError(f"{label} K14T positions differ from its factory")

    c11 = context.get("c11_receipt")
    if not isinstance(c11, Mapping):
        raise PreGpuReceiptError(f"{label} C11 receipt is missing")
    if (
        c11.get("unit_id") != UNIT_ID
        or c11.get("arm_id") != "C11"
        or c11.get("cell_id") != "C11"
        or c11.get("status") != "ready"
        or c11.get("component_order") != ["K10", "H20"]
        or c11.get("component_changed_cell_union") is not True
        or c11.get("exact_scope") is not True
    ):
        raise PreGpuReceiptError(f"{label} C11 is not the exact K10+H20 composition")
    children = c11.get("children")
    if (
        not isinstance(children, list)
        or [child.get("arm_id") for child in children if isinstance(child, Mapping)]
        != ["K10", "H20"]
    ):
        raise PreGpuReceiptError(f"{label} C11 child order/identity drifted")
    _validate_unattested_consumption(
        c11.get("layer_consumption_attestation"),
        f"{label} C11.layer_consumption_attestation",
    )
    _validate_unattested_consumption(
        c11.get("all_layer_consumption_attestation"),
        f"{label} C11.all_layer_consumption_attestation",
    )
    canonical_json_bytes(context)
    return context


def _reject_pre_gpu_receipt_cycle(value: Any, label: str = "source preflight") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).startswith("pre_gpu_receipt"):
                raise PreGpuReceiptError(f"{label} contains a pre-GPU receipt cycle")
            _reject_pre_gpu_receipt_cycle(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_pre_gpu_receipt_cycle(nested, f"{label}[{index}]")


def _validate_source_preflight(
    value: Any,
    *,
    plan_ref: Mapping[str, Any],
    plan_doc: Mapping[str, Any],
    manifest_doc: Mapping[str, Any],
    input_refs: Mapping[str, Mapping[str, Any]],
    code_refs: Mapping[str, Mapping[str, Any]],
    test_refs: Mapping[str, Mapping[str, Any]],
    roots: Mapping[str, Any],
) -> dict[str, Any]:
    document = _validate_source_preflight_envelope(value)
    required = {
        "schema_version",
        "status",
        "phase",
        "unit_id",
        "plan_binding",
        "source_bindings",
        "code_bindings",
        "test_bindings",
        "reserved_roots",
        "model_free_production_path",
        "gpu_used",
        "model_loaded",
        "backend_session_opened",
        "output_root_created",
        "receipt_independent",
        "self_sha256",
    }
    _require_exact_fields(document, required, "source preflight")
    _reject_pre_gpu_receipt_cycle(document)
    expected_plan = {
        "path": plan_ref["path"],
        "raw_sha256": plan_ref["sha256"],
        "self_sha256": plan_doc["self_sha256"],
    }
    if document.get("plan_binding") != expected_plan:
        raise PreGpuReceiptError("source preflight plan binding drifted")
    expected_sources = {
        role: {
            "path": ref["path"],
            "sha256": ref["sha256"],
            "kind": ref["kind"],
        }
        for role, ref in input_refs.items()
    }
    if document.get("source_bindings") != expected_sources:
        raise PreGpuReceiptError("source preflight input bindings differ from current refs")
    expected_code = {
        role: {"path": ref["path"], "sha256": ref["sha256"]}
        for role, ref in code_refs.items()
    }
    expected_tests = {
        role: {"path": ref["path"], "sha256": ref["sha256"]}
        for role, ref in test_refs.items()
    }
    if document.get("code_bindings") != expected_code:
        raise PreGpuReceiptError("source preflight code bindings differ from current refs")
    if document.get("test_bindings") != expected_tests:
        raise PreGpuReceiptError("source preflight test bindings differ from current refs")
    if document.get("reserved_roots") != roots:
        raise PreGpuReceiptError("source preflight reserved roots differ from the seal")

    production = document.get("model_free_production_path")
    if not isinstance(production, Mapping):
        raise PreGpuReceiptError("source preflight model-free production receipt is missing")
    production = dict(production)
    production_fields = {
        "status",
        "cpu_contract",
        "full_runtime_cohort",
        "processor_only",
        "factory_model_metadata",
        "selected_factory_contexts",
        "selected_event_count",
        "gpu_used",
        "model_loaded",
        "model_loader_called",
        "output_root_created",
        "receipt_sha256",
    }
    _require_exact_fields(production, production_fields, "source preflight production")
    body = dict(production)
    observed_receipt_sha = body.pop("receipt_sha256")
    if observed_receipt_sha != sha256_json(body):
        raise PreGpuReceiptError("source preflight production receipt_sha256 mismatch")
    if production.get("status") != "passed":
        raise PreGpuReceiptError("source preflight production path is not passing")
    _require_false_flags(
        production,
        ("gpu_used", "model_loaded", "model_loader_called", "output_root_created"),
        "source preflight production",
    )
    cpu_contract = production.get("cpu_contract")
    if (
        not isinstance(cpu_contract, Mapping)
        or set(cpu_contract) != {"status", "event_count", "identity_sha256"}
        or cpu_contract.get("status") != "passed"
        or cpu_contract.get("event_count") != 11
    ):
        raise PreGpuReceiptError("source preflight CPU contract is not eleven-event")
    _sha(cpu_contract.get("identity_sha256"), "source preflight CPU identity")
    full_runtime = _validate_full_runtime_cohort(
        production.get("full_runtime_cohort"),
        input_refs=input_refs,
        manifest_doc=manifest_doc,
    )
    processor = production.get("processor_only")
    if not isinstance(processor, Mapping):
        raise PreGpuReceiptError("source preflight processor-only receipt is missing")
    _require_exact_fields(
        processor,
        {"status", "load_model", "model_present", "backend_session_opened", "identity_sha256"},
        "source preflight processor-only",
    )
    if processor.get("status") != "passed":
        raise PreGpuReceiptError("source preflight processor-only receipt is not passing")
    _require_false_flags(
        processor,
        ("load_model", "model_present", "backend_session_opened"),
        "source preflight processor-only",
    )
    _sha(processor.get("identity_sha256"), "source preflight processor identity")
    _validate_factory_metadata(
        production.get("factory_model_metadata"), input_refs=input_refs
    )
    contexts = production.get("selected_factory_contexts")
    if (
        production.get("selected_event_count") != 3
        or not isinstance(contexts, list)
        or len(contexts) != 3
        or [context.get("event_id") for context in contexts if isinstance(context, Mapping)]
        != list(planner.EVENT_IDS)
    ):
        raise PreGpuReceiptError("source preflight selected event order/count drifted")
    full_event_ids = {
        binding["event_id"] for binding in full_runtime["processor_context_bindings"]
    }
    if not set(planner.EVENT_IDS).issubset(full_event_ids):
        raise PreGpuReceiptError("source preflight selected events are outside the full cohort")
    for context, expected_event in zip(contexts, plan_doc["events"], strict=True):
        _validate_source_preflight_context(context, expected_event=expected_event)
    return document


def _validate_probe(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, document = _json_ref(value, "installed-Qwen probe")
    if document.get("schema_version") != PRE_GPU_PROBE_SCHEMA_VERSION:
        raise PreGpuReceiptError("installed-Qwen probe schema is not the crossover runner pre-GPU schema")
    if document.get("unit_id") != UNIT_ID or document.get("status") != "passed":
        raise PreGpuReceiptError("installed-Qwen probe is not passing for this crossover unit")
    if document.get("no_gpu_launch") is not True:
        raise PreGpuReceiptError("installed-Qwen probe must attest no GPU launch")
    composition = document.get("composition")
    consumption = document.get("installed_qwen_consumption")
    if not isinstance(composition, Mapping) or not isinstance(consumption, Mapping):
        raise PreGpuReceiptError("installed-Qwen probe lacks exact composition/installed_qwen_consumption evidence")
    if composition.get("status") != "passed":
        raise PreGpuReceiptError("installed-Qwen composition evidence is not passing")
    if consumption.get("status") != "passed" or consumption.get("all_layer_consumption") is not True:
        raise PreGpuReceiptError("installed-Qwen consumption is not an all-layer passing receipt")
    layer_consumption = consumption.get("layer_consumption")
    if not isinstance(layer_consumption, Mapping) or layer_consumption.get("passed") is not True or layer_consumption.get("all_layers_identical") is not True:
        raise PreGpuReceiptError("installed-Qwen layer consumption attestation is incomplete")
    block23 = consumption.get("block23_sdpa_mass_receipt")
    if not isinstance(block23, Mapping) or block23.get("status") != "passed":
        raise PreGpuReceiptError("installed-Qwen block-23 SDPA mass attestation is not passing")
    _validate_source_preflight_envelope(document.get("source_preflight"))
    return ref, document


def _validate_code(source_files: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    values = dict(source_files or {role: path for role, path in CODE_ROLE_PATHS.items()})
    if set(values) != set(CODE_ROLE_PATHS):
        raise PreGpuReceiptError(f"source_files must contain exactly {tuple(CODE_ROLE_PATHS)}")
    refs: dict[str, dict[str, Any]] = {}
    for role, expected in CODE_ROLE_PATHS.items():
        refs[role] = _path_ref(values[role], f"source file {role}", expected_path=expected, allow_directory=False)
        refs[role]["role"] = role
    return refs


def _validate_tests(test_files: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    values = dict(test_files or {role: path for role, path in TEST_PATHS.items()})
    if set(values) != set(TEST_PATHS):
        raise PreGpuReceiptError(f"test_files must contain exactly {tuple(TEST_PATHS)}")
    refs: dict[str, dict[str, Any]] = {}
    for role, expected in TEST_PATHS.items():
        refs[role] = _path_ref(values[role], f"test file {role}", expected_path=expected, allow_directory=False)
        refs[role]["role"] = role
    return refs


def _device_plan(plan: Mapping[str, Any], supplied: Mapping[str, Any] | None) -> dict[str, str]:
    observed = dict(supplied or plan.get("device_plan") or {})
    if observed != DEVICE_PLAN or dict(plan.get("device_plan") or {}) != DEVICE_PLAN:
        raise PreGpuReceiptError("device plan must assign shard-000..002 to physical GPUs 0..2 exactly")
    return observed


def _build_receipt(
    *,
    plan: Any,
    unit: Any,
    focused_evidence: Any,
    installed_probe: Any,
    execution_root: Any,
    final_root: Any,
    output: Any,
    config: Any,
    panel: Any,
    cohort: Any,
    h0: Any,
    manifest: Any,
    census: Any,
    execution_plan: Any,
    cohort_manifest: Any,
    h0_root: Any,
    h0_dir: Any,
    base_model_dir: Any,
    h0_identity_files: Mapping[str, Any] | None,
    base_model_files: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
    source_files: Mapping[str, Any] | None,
    test_files: Mapping[str, Any] | None,
    device_assignments: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan_ref, plan_doc = _validate_plan(plan)
    output_path = _absolute(output, "pre-GPU receipt output")
    unit_ref = _unit_ref(unit)
    focused_ref, focused_doc = _validate_focused(focused_evidence)
    probe_ref, probe_doc = _validate_probe(installed_probe)
    roots = _validate_roots(execution_root, final_root)
    # The frozen live executor consumes these exact names.  Do not infer a
    # directory, manifest companion, or model payload from a broad root.
    input_values = {
        "manifest": manifest,
        "census": census,
        "execution_plan": execution_plan,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model_dir,
    }
    if any(value is None for value in input_values.values()):
        raise PreGpuReceiptError(
            "manifest, census, execution_plan, config, panel, cohort, cohort_manifest, "
            "h0_root, h0_dir, and base_model_dir are all required"
        )
    input_refs = {
        key: _path_ref(value, f"input {key}", allow_directory=key in {"h0_root", "h0_dir", "base_model_dir"})
        for key, value in input_values.items()
    }
    manifest_source = plan_doc.get("source_bindings", {}).get("manifest")
    census_source = plan_doc.get("source_bindings", {}).get("census")
    if not isinstance(manifest_source, Mapping) or input_refs["manifest"]["path"] != manifest_source.get("path"):
        raise PreGpuReceiptError("current manifest path differs from deterministic plan source")
    if not isinstance(census_source, Mapping) or input_refs["census"]["path"] != census_source.get("path"):
        raise PreGpuReceiptError("current census path differs from deterministic plan source")
    if input_refs["execution_plan"]["path"] != plan_ref["path"] or input_refs["execution_plan"]["sha256"] != plan_ref["sha256"]:
        raise PreGpuReceiptError("current execution_plan must be the exact materialized plan")
    if Path(input_refs["h0_dir"]["path"]).parent != Path(input_refs["h0_root"]["path"]):
        raise PreGpuReceiptError("h0_dir must be one exact child of h0_root")
    if not isinstance(h0_identity_files, Mapping) and isinstance(h0, Mapping):
        h0_identity_files = h0.get("identity_files")
    if not isinstance(base_model_files, Mapping) and isinstance(h0, Mapping):
        base_model_files = h0.get("base_model_files")
    if not isinstance(h0_identity_files, Mapping) or not isinstance(base_model_files, Mapping):
        raise PreGpuReceiptError("complete h0_identity_files and base_model_files maps are required")
    h0_files = _file_map(
        h0_identity_files,
        H0_IDENTITY_FILE_ROLES,
        "h0_identity_files",
        Path(input_refs["h0_root"]["path"]),
        enforce_root=False,
    )
    # Adapter and embedding payloads may be sibling roots selected by the H0
    # resolved config, so H0 identity files are intentionally not constrained
    # to h0_dir.  Their exact paths and bytes are still sealed above.
    base_files = _file_map(base_model_files, BASE_MODEL_FILE_ROLES, "base_model_files", Path(input_refs["base_model_dir"]["path"]))
    base_inventory = _directory_inventory_document(input_refs["base_model_dir"]["path"], "base model directory")
    inventory_by_path = {
        str(Path(base_inventory["root"]) / item["relative_path"]): item
        for item in base_inventory["files"]
    }
    for role, ref in base_files.items():
        item = inventory_by_path.get(ref["path"])
        if item is None or item["sha256"] != ref["sha256"]:
            raise PreGpuReceiptError(f"base_model_files.{role} is absent from complete base-model inventory")
    manifest_ref, manifest_doc = _json_ref(input_refs["manifest"], "input manifest")
    census_ref, census_doc = _json_ref(input_refs["census"], "input census")
    cohort_manifest_ref, cohort_manifest_doc = _json_ref(input_refs["cohort_manifest"], "input cohort_manifest")
    if manifest_ref["sha256"] != input_refs["manifest"]["sha256"] or census_ref["sha256"] != input_refs["census"]["sha256"]:
        raise PreGpuReceiptError("input manifest/census raw identity changed while parsing")
    runtime_input_hashes = {
        "manifest_raw_sha256": input_refs["manifest"]["sha256"],
        "manifest_self_sha256": manifest_doc.get("self_sha256"),
        "census_v3_raw_sha256": input_refs["census"]["sha256"],
        "census_v3_self_sha256": census_doc.get("self_sha256"),
        "execution_plan_raw_sha256": input_refs["execution_plan"]["sha256"],
        "execution_plan_sha256": plan_doc["self_sha256"],
        "config_sha256": input_refs["config"]["sha256"],
        "panel_sha256": input_refs["panel"]["sha256"],
        "cohort_sha256": input_refs["cohort"]["sha256"],
        "cohort_manifest_raw_sha256": input_refs["cohort_manifest"]["sha256"],
        "cohort_manifest_semantic_sha256": cohort_manifest_ref.get("semantic_sha256"),
        "h0_root_sha256": input_refs["h0_root"]["sha256"],
        "h0_dir_sha256": input_refs["h0_dir"]["sha256"],
        "h0_identity_files_sha256": _file_map_sha(h0_files, H0_IDENTITY_FILE_ROLES),
        "base_model_files_sha256": _file_map_sha(base_files, BASE_MODEL_FILE_ROLES),
        "base_model_inventory_sha256": base_inventory["inventory_sha256"],
    }
    runtime_doc = _runtime(runtime)
    code_refs = _validate_code(source_files)
    test_refs = _validate_tests(test_files)
    device_doc = _device_plan(plan_doc, device_assignments)
    source_preflight_doc = _validate_source_preflight(
        probe_doc.get("source_preflight"),
        plan_ref=plan_ref,
        plan_doc=plan_doc,
        manifest_doc=manifest_doc,
        input_refs=input_refs,
        code_refs=code_refs,
        test_refs=test_refs,
        roots=roots,
    )
    source_event_bindings = [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
            "prefix_sha256": event["prefix_sha256"],
            "geometry_sha256": event["geometry_sha256"],
        }
        for event in plan_doc["events"]
    ]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "unit_id": UNIT_ID,
        "pre_gpu_receipt_path": str(output_path),
        "no_training": True,
        "launch_scope": "one_fixed_three_shard_no_training_launch",
        "plan": plan_ref,
        "plan_self_sha256": plan_doc["self_sha256"],
        "unit_authority": unit_ref,
        "input_bindings": input_refs,
        "input_paths": {key: ref["path"] for key, ref in input_refs.items()},
        "input_hashes": runtime_input_hashes,
        "h0_identity_files": h0_files,
        "h0_identity_files_sha256": runtime_input_hashes["h0_identity_files_sha256"],
        "base_model_files": base_files,
        "base_model_files_sha256": runtime_input_hashes["base_model_files_sha256"],
        "base_model_inventory": base_inventory,
        "base_model_inventory_sha256": base_inventory["inventory_sha256"],
        "cohort_manifest_document": cohort_manifest_doc,
        "focused_cpu_evidence": focused_ref,
        "focused_cpu_evidence_document": focused_doc,
        "installed_qwen_probe": probe_ref,
        "installed_qwen_probe_document": probe_doc,
        "source_preflight_document": source_preflight_doc,
        "source_preflight_self_sha256": source_preflight_doc["self_sha256"],
        "runtime": runtime_doc,
        "forced_math": {"enabled": True, "backend": "MATH"},
        "device_plan": device_doc,
        "event_bindings": source_event_bindings,
        "source_selection": {
            "source_unit_id": planner.SOURCE_UNIT_ID,
            "event_ids": list(planner.EVENT_IDS),
            "event_count": 3,
            "image_count": 3,
            "cell_order": list(planner.CELL_ORDER),
            "admission_mode": planner.OPENER_MODE,
            "opener_injected": False,
            "use_cache": False,
            "max_rows": planner.MAX_ROWS,
            "max_row_tokens": planner.MAX_ROW_TOKENS,
        },
        "roots": roots,
        "source_files": code_refs,
        "test_files": test_refs,
        "execution_policy": {
            "no_event_reorder": True,
            "no_reselection": True,
            "no_sweep": True,
            "no_a3": True,
            "no_p4": True,
            "no_training": True,
            "at_most_once": True,
            "fresh_execution_root_required": True,
            "fresh_final_root_required": True,
        },
        "runtime_identity_contract": {
            "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
            "required": True,
            "receipt_path_field": "pre_gpu_receipt_path",
            "receipt_raw_sha_field": "pre_gpu_receipt_sha256",
            "receipt_self_sha_field": "pre_gpu_receipt_self_sha256",
            "input_paths_field": "input_paths",
            "input_hashes_field": "input_hashes",
            "device_assignment_field": "device_assignment",
            "code_hashes_field": "code_hashes",
        },
    }
    document["self_sha256"] = document_self_sha256(document)
    return document


def _write_once(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = _absolute(path, "pre-GPU receipt output")
    payload = canonical_json_bytes(document) + b"\n"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"immutable pre-GPU receipt collision: {target}")
        return {"path": str(target), "sha256": sha256_bytes(payload), "byte_identical": True}
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
    return {"path": str(target), "sha256": sha256_bytes(payload), "byte_identical": False}


def seal_pre_gpu_receipt(
    plan: str | Path | Mapping[str, Any],
    unit: str | Path | Mapping[str, Any],
    focused_evidence: str | Path | Mapping[str, Any],
    installed_probe: str | Path | Mapping[str, Any],
    execution_root: str | Path,
    final_root: str | Path,
    output: str | Path,
    *,
    config: Any = None,
    panel: Any = None,
    cohort: Any = None,
    h0: Any = None,
    manifest: Any = None,
    census: Any = None,
    execution_plan: Any = None,
    cohort_manifest: Any = None,
    h0_root: Any = None,
    h0_dir: Any = None,
    base_model_dir: Any = None,
    h0_identity_files: Mapping[str, Any] | None = None,
    base_model_files: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    source_files: Mapping[str, Any] | None = None,
    test_files: Mapping[str, Any] | None = None,
    device_assignments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write a write-once pre-GPU receipt."""

    document = _build_receipt(
        plan=plan,
        unit=unit,
        focused_evidence=focused_evidence,
        installed_probe=installed_probe,
        execution_root=execution_root,
        final_root=final_root,
        output=output,
        config=config,
        panel=panel,
        cohort=cohort,
        h0=h0,
        manifest=manifest,
        census=census,
        execution_plan=execution_plan,
        cohort_manifest=cohort_manifest,
        h0_root=h0_root,
        h0_dir=h0_dir,
        base_model_dir=base_model_dir,
        h0_identity_files=h0_identity_files,
        base_model_files=base_model_files,
        runtime=runtime,
        source_files=source_files,
        test_files=test_files,
        device_assignments=device_assignments,
    )
    _write_once(output, document)
    return document


build_receipt = seal_pre_gpu_receipt


def _load_receipt_document(
    value: str | Path | Mapping[str, Any],
    explicit_path: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        document = dict(value)
        receipt_path = _regular_file(explicit_path, "pre-GPU receipt") if explicit_path is not None else None
        if receipt_path is not None:
            raw = receipt_path.read_bytes()
            try:
                parsed = json.loads(raw)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PreGpuReceiptError(f"pre-GPU receipt is not readable JSON: {exc}") from exc
            if not isinstance(parsed, Mapping) or raw != canonical_json_bytes(parsed) + b"\n" or dict(parsed) != document:
                raise PreGpuReceiptError("supplied receipt object differs from canonical receipt path")
    else:
        receipt_path = _regular_file(value, "pre-GPU receipt")
        if explicit_path is not None and receipt_path != _regular_file(explicit_path, "pre-GPU receipt expected path"):
            raise PreGpuReceiptError("explicit receipt path differs from supplied receipt")
        raw = receipt_path.read_bytes()
        try:
            parsed = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreGpuReceiptError(f"pre-GPU receipt is not readable JSON: {exc}") from exc
        if not isinstance(parsed, Mapping) or raw != canonical_json_bytes(parsed) + b"\n":
            raise PreGpuReceiptError("pre-GPU receipt must be canonical JSON with one trailing newline")
        document = dict(parsed)
    return document, receipt_path


def validate_pre_gpu_receipt(
    value: str | Path | Mapping[str, Any],
    *,
    phase: str = "prelaunch",
    receipt_path: str | Path | None = None,
    shard_id: str | int | None = None,
) -> dict[str, Any]:
    """Revalidate every receipt binding before a successor runtime may load.

    ``phase='prelaunch'`` requires both output roots to remain absent.  A
    runtime shard may instead validate with ``phase='runtime'`` and its exact
    shard id; sibling shard directories are then accepted while the current
    target and final root remain absent.
    """

    document, loaded_path = _load_receipt_document(value, receipt_path)
    receipt_path_obj = loaded_path
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != STATUS or document.get("unit_id") != UNIT_ID:
        raise PreGpuReceiptError("pre-GPU receipt schema/status/unit identity drifted")
    declared_path = document.get("pre_gpu_receipt_path")
    if not isinstance(declared_path, str) or _absolute(declared_path, "receipt pre_gpu_receipt_path") != (receipt_path_obj if receipt_path_obj is not None else _absolute(declared_path, "receipt pre_gpu_receipt_path")):
        raise PreGpuReceiptError("pre-GPU receipt path binding is missing or drifted")
    if document.get("self_sha256") != document_self_sha256(document):
        raise PreGpuReceiptError("pre-GPU receipt self_sha256 mismatch")
    plan_ref, plan_doc = _validate_plan(document.get("plan"))
    if plan_doc.get("self_sha256") != document.get("plan_self_sha256"):
        raise PreGpuReceiptError("receipt plan semantic identity drifted")
    if document.get("input_bindings", {}).get("execution_plan", {}).get("path") != plan_ref["path"]:
        raise PreGpuReceiptError("receipt execution_plan is not the exact materialized plan")
    _unit_ref(document.get("unit_authority"))
    roots = document.get("roots")
    if not isinstance(roots, Mapping):
        raise PreGpuReceiptError("receipt roots are missing")
    for key in ("execution_root", "final_root"):
        if not isinstance(roots.get(key), Mapping):
            raise PreGpuReceiptError(f"receipt {key} binding is malformed")
        if phase == "prelaunch" and roots[key].get("status") != "reserved_absent_pre_gpu":
            raise PreGpuReceiptError(f"receipt {key} is not reserved absent before GPU")
    _validate_roots(
        roots.get("execution_root", {}).get("path"),
        roots.get("final_root", {}).get("path"),
        phase=phase,
        shard_id=shard_id,
    )

    required_inputs = (
        "manifest",
        "census",
        "execution_plan",
        "config",
        "panel",
        "cohort",
        "cohort_manifest",
        "h0_root",
        "h0_dir",
        "base_model_dir",
    )
    bindings = document.get("input_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(required_inputs):
        raise PreGpuReceiptError("receipt input bindings are incomplete")
    input_paths = document.get("input_paths")
    if not isinstance(input_paths, Mapping) or set(input_paths) != set(required_inputs):
        raise PreGpuReceiptError("receipt input paths are incomplete")
    input_refs: dict[str, dict[str, Any]] = {}
    for key in required_inputs:
        ref = _path_ref(bindings[key], f"receipt input {key}", allow_directory=key in {"h0_root", "h0_dir", "base_model_dir"})
        if ref != bindings[key] or input_paths.get(key) != ref["path"]:
            raise PreGpuReceiptError(f"receipt input {key} path/hash drifted")
        input_refs[key] = ref
    if input_refs["execution_plan"]["path"] != plan_ref["path"] or input_refs["execution_plan"]["sha256"] != plan_ref["sha256"]:
        raise PreGpuReceiptError("receipt execution_plan raw identity drifted")
    if Path(input_refs["h0_dir"]["path"]).parent != Path(input_refs["h0_root"]["path"]):
        raise PreGpuReceiptError("receipt h0_dir parent differs from h0_root")

    manifest_ref, manifest_doc = _json_ref(input_refs["manifest"], "receipt input manifest")
    census_ref, census_doc = _json_ref(input_refs["census"], "receipt input census")
    cohort_manifest_ref, cohort_manifest_doc = _json_ref(input_refs["cohort_manifest"], "receipt input cohort_manifest")
    expected_hashes = {
        "manifest_raw_sha256": input_refs["manifest"]["sha256"],
        "manifest_self_sha256": manifest_doc.get("self_sha256"),
        "census_v3_raw_sha256": input_refs["census"]["sha256"],
        "census_v3_self_sha256": census_doc.get("self_sha256"),
        "execution_plan_raw_sha256": input_refs["execution_plan"]["sha256"],
        "execution_plan_sha256": plan_doc["self_sha256"],
        "config_sha256": input_refs["config"]["sha256"],
        "panel_sha256": input_refs["panel"]["sha256"],
        "cohort_sha256": input_refs["cohort"]["sha256"],
        "cohort_manifest_raw_sha256": input_refs["cohort_manifest"]["sha256"],
        "cohort_manifest_semantic_sha256": cohort_manifest_ref.get("semantic_sha256"),
        "h0_root_sha256": input_refs["h0_root"]["sha256"],
        "h0_dir_sha256": input_refs["h0_dir"]["sha256"],
    }
    stored_hashes = document.get("input_hashes")
    if not isinstance(stored_hashes, Mapping) or dict(stored_hashes) != expected_hashes | {
        "h0_identity_files_sha256": stored_hashes.get("h0_identity_files_sha256") if isinstance(stored_hashes, Mapping) else None,
        "base_model_files_sha256": stored_hashes.get("base_model_files_sha256") if isinstance(stored_hashes, Mapping) else None,
        "base_model_inventory_sha256": stored_hashes.get("base_model_inventory_sha256") if isinstance(stored_hashes, Mapping) else None,
    }:
        # The three inventory hashes are checked immediately below; report all
        # other input drift without allowing a caller to omit a hash key.
        if not isinstance(stored_hashes, Mapping) or any(stored_hashes.get(key) != value for key, value in expected_hashes.items()):
            raise PreGpuReceiptError("receipt input hashes drifted")
    if document.get("cohort_manifest_document") != cohort_manifest_doc:
        raise PreGpuReceiptError("receipt cohort_manifest document drifted")

    h0_files = _file_map(
        document.get("h0_identity_files"),
        H0_IDENTITY_FILE_ROLES,
        "receipt h0_identity_files",
        Path(input_refs["h0_root"]["path"]),
        enforce_root=False,
    )
    base_files = _file_map(
        document.get("base_model_files"),
        BASE_MODEL_FILE_ROLES,
        "receipt base_model_files",
        Path(input_refs["base_model_dir"]["path"]),
    )
    base_inventory = _directory_inventory_document(input_refs["base_model_dir"]["path"], "receipt base model directory")
    if document.get("base_model_inventory") != base_inventory or document.get("base_model_inventory_sha256") != base_inventory["inventory_sha256"]:
        raise PreGpuReceiptError("receipt base-model inventory drifted")
    inventory_by_path = {
        str(Path(base_inventory["root"]) / item["relative_path"]): item
        for item in base_inventory["files"]
    }
    for role, ref in base_files.items():
        item = inventory_by_path.get(ref["path"])
        if item is None or item["sha256"] != ref["sha256"]:
            raise PreGpuReceiptError(f"receipt base_model_files.{role} drifted")
    h0_hash = _file_map_sha(h0_files, H0_IDENTITY_FILE_ROLES)
    base_hash = _file_map_sha(base_files, BASE_MODEL_FILE_ROLES)
    if document.get("h0_identity_files_sha256") != h0_hash or document.get("base_model_files_sha256") != base_hash:
        raise PreGpuReceiptError("receipt H0/base-model file hashes drifted")
    if stored_hashes.get("h0_identity_files_sha256") != h0_hash or stored_hashes.get("base_model_files_sha256") != base_hash or stored_hashes.get("base_model_inventory_sha256") != base_inventory["inventory_sha256"]:
        raise PreGpuReceiptError("receipt inventory input hashes drifted")

    focused_ref, focused_doc = _validate_focused(document.get("focused_cpu_evidence"))
    if focused_ref != document.get("focused_cpu_evidence") or focused_doc != document.get("focused_cpu_evidence_document"):
        raise PreGpuReceiptError("receipt focused CPU evidence drifted")
    probe_ref, probe_doc = _validate_probe(document.get("installed_qwen_probe"))
    if probe_ref != document.get("installed_qwen_probe") or probe_doc != document.get("installed_qwen_probe_document"):
        raise PreGpuReceiptError("receipt installed-Qwen probe drifted")
    runtime_doc = _runtime(document.get("runtime"))
    if runtime_doc != document.get("runtime"):
        raise PreGpuReceiptError("receipt runtime identity drifted")
    if document.get("forced_math") != {"enabled": True, "backend": "MATH"}:
        raise PreGpuReceiptError("receipt is not bound to forced MATH attention")
    if _device_plan(plan_doc, document.get("device_plan")) != document.get("device_plan"):
        raise PreGpuReceiptError("receipt device plan drifted")
    code_refs = _validate_code(document.get("source_files"))
    tests = _validate_tests(document.get("test_files"))
    if code_refs != document.get("source_files") or tests != document.get("test_files"):
        raise PreGpuReceiptError("receipt source/test identities drifted")
    source_preflight_doc = _validate_source_preflight(
        probe_doc.get("source_preflight"),
        plan_ref=plan_ref,
        plan_doc=plan_doc,
        manifest_doc=manifest_doc,
        input_refs=input_refs,
        code_refs=code_refs,
        test_refs=tests,
        roots=roots,
    )
    if (
        document.get("source_preflight_document") != source_preflight_doc
        or document.get("source_preflight_self_sha256")
        != source_preflight_doc["self_sha256"]
    ):
        raise PreGpuReceiptError("receipt source preflight binding drifted")
    expected_events = [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
            "prefix_sha256": event["prefix_sha256"],
            "geometry_sha256": event["geometry_sha256"],
        }
        for event in plan_doc["events"]
    ]
    if document.get("event_bindings") != expected_events:
        raise PreGpuReceiptError("receipt event/prefix/geometry bindings drifted")
    expected_selection = {
        "source_unit_id": planner.SOURCE_UNIT_ID,
        "event_ids": list(planner.EVENT_IDS),
        "event_count": 3,
        "image_count": 3,
        "cell_order": list(planner.CELL_ORDER),
        "admission_mode": planner.OPENER_MODE,
        "opener_injected": False,
        "use_cache": False,
        "max_rows": planner.MAX_ROWS,
        "max_row_tokens": planner.MAX_ROW_TOKENS,
    }
    if document.get("source_selection") != expected_selection:
        raise PreGpuReceiptError("receipt source selection drifted")
    if document.get("no_training") is not True or document.get("launch_scope") != "one_fixed_three_shard_no_training_launch":
        raise PreGpuReceiptError("receipt launch scope crosses no-training boundary")
    return {
        "receipt": document,
        "self_sha256": document["self_sha256"],
        "receipt_sha256": sha256_json(document),
        "receipt_raw_sha256": sha256_file(receipt_path_obj) if receipt_path_obj is not None else None,
        "path": str(receipt_path_obj) if receipt_path_obj else None,
    }


validate_receipt = validate_pre_gpu_receipt


def _runtime_code_hashes(source_files: Mapping[str, Any]) -> dict[str, str]:
    refs = _validate_code(source_files)
    hashes = {role: refs[role]["sha256"] for role in CODE_ROLES}
    # The frozen live executor names this seam directly.  Keep the original
    # source role too so a receipt remains auditable without alias expansion.
    hashes["live_executor"] = hashes["base_live_executor"]
    hashes["runner"] = hashes["crossover_runner"]
    hashes["finalizer"] = hashes["crossover_finalizer"]
    hashes["materializer"] = hashes["plan_materializer"]
    hashes["sealer"] = hashes["pre_gpu_sealer"]
    return hashes


def _shard_event(plan_doc: Mapping[str, Any], shard_id: str, shard_index: int) -> dict[str, Any]:
    shards = plan_doc.get("shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise PreGpuReceiptError("crossover plan shard list is not the exact three-shard plan")
    matches = [item for item in shards if isinstance(item, Mapping) and item.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise PreGpuReceiptError(f"crossover plan lacks exact {shard_id} entry")
    shard = matches[0]
    if shard.get("shard_index") != shard_index or shard.get("physical_device") != DEVICE_PLAN[shard_id]:
        raise PreGpuReceiptError(f"crossover plan {shard_id} assignment drifted")
    events = shard.get("events")
    if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], Mapping):
        raise PreGpuReceiptError(f"crossover plan {shard_id} must contain one event")
    return dict(events[0])


def runtime_identity_binding(
    value: str | Path | Mapping[str, Any],
    *,
    receipt_path: str | Path | None = None,
    shard_id: str | int,
    observed_cuda_visible_devices: str,
) -> dict[str, Any]:
    """Derive the exact per-shard identity consumed by the frozen executor."""

    shard, shard_index = _normalize_shard_id(shard_id)
    checked = validate_pre_gpu_receipt(
        value,
        phase="runtime",
        receipt_path=receipt_path,
        shard_id=shard,
    )
    document = checked["receipt"]
    path_value = checked["path"] or document.get("pre_gpu_receipt_path")
    receipt_file = _regular_file(path_value, "runtime identity receipt")
    if str(receipt_file) != document.get("pre_gpu_receipt_path"):
        raise PreGpuReceiptError("runtime identity receipt path differs from receipt binding")
    physical = DEVICE_PLAN[shard]
    if observed_cuda_visible_devices != physical:
        raise PreGpuReceiptError("observed CUDA_VISIBLE_DEVICES differs from frozen shard device")
    plan_ref, plan_doc = _validate_plan(document["plan"])
    del plan_ref
    event = _shard_event(plan_doc, shard, shard_index)
    assignment_body = {
        "shard_id": shard,
        "shard_index": shard_index,
        "physical_device": physical,
        "observed_cuda_visible_devices": observed_cuda_visible_devices,
        "logical_device": "cuda:0",
        "device_count": 1,
        "event_index": event.get("event_index"),
        "event_id": event.get("event_id"),
        "event_sha256": event.get("event_sha256"),
    }
    if not all(isinstance(assignment_body[key], (str, int)) for key in ("event_id", "event_sha256", "event_index")):
        raise PreGpuReceiptError("runtime identity shard event binding is malformed")
    code_hashes = _runtime_code_hashes(document["source_files"])
    body: dict[str, Any] = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": "S",
        "step": 2444,
        "substrate": planner.PRIMARY["substrate"],
        "pre_gpu_receipt_path": str(receipt_file),
        "pre_gpu_receipt_sha256": sha256_file(receipt_file),
        "pre_gpu_receipt_self_sha256": document["self_sha256"],
        "plan_self_sha256": document["plan_self_sha256"],
        "input_paths": dict(document["input_paths"]),
        "input_hashes": dict(document["input_hashes"]),
        "h0_identity_files_sha256": document["h0_identity_files_sha256"],
        "base_model_files_sha256": document["base_model_files_sha256"],
        "base_model_inventory_sha256": document["base_model_inventory_sha256"],
        "code_hashes": code_hashes,
        "runtime": dict(document["runtime"]),
        "forced_math": dict(document["forced_math"]),
        "device_policy": {
            "shard_count": SHARD_COUNT,
            "device_plan": dict(DEVICE_PLAN),
            "logical_device": "cuda:0",
            "per_shard_device_count": 1,
        },
        "device_assignment": assignment_body | {"authorization_sha256": sha256_json(assignment_body)},
        "event_binding": event,
        "claim_scope": document["launch_scope"],
        "no_training": True,
    }
    body["binding_sha256"] = sha256_json(body)
    return body


def validate_runtime_identity(
    value: Mapping[str, Any],
    receipt: str | Path | Mapping[str, Any],
    *,
    receipt_path: str | Path,
    shard_id: str | int,
    observed_cuda_visible_devices: str,
    expected_code_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rehash and validate one runtime identity immediately before execution."""

    if not isinstance(value, Mapping):
        raise PreGpuReceiptError("runtime identity must be a JSON object")
    identity = dict(value)
    shard, shard_index = _normalize_shard_id(shard_id)
    checked = validate_pre_gpu_receipt(
        receipt,
        phase="runtime",
        receipt_path=receipt_path,
        shard_id=shard,
    )
    expected = runtime_identity_binding(
        checked["receipt"],
        receipt_path=receipt_path,
        shard_id=shard,
        observed_cuda_visible_devices=observed_cuda_visible_devices,
    )
    if identity != expected:
        # Return a useful field-level failure without accepting a partial map.
        if identity.get("binding_sha256") != document_self_sha256(identity, "binding_sha256"):
            raise PreGpuReceiptError("runtime identity binding_sha256 mismatch")
        raise PreGpuReceiptError("runtime identity differs from rehashed receipt/shard binding")
    if expected_code_hashes is not None and dict(expected_code_hashes) != identity["code_hashes"]:
        raise PreGpuReceiptError("runtime identity code hashes differ from expected current code")
    if identity["device_assignment"]["shard_id"] != shard or identity["device_assignment"]["shard_index"] != shard_index:
        raise PreGpuReceiptError("runtime identity shard assignment drifted")
    if identity["device_assignment"]["observed_cuda_visible_devices"] != observed_cuda_visible_devices:
        raise PreGpuReceiptError("runtime identity observed CUDA device drifted")
    return {
        "identity": identity,
        "binding_sha256": identity["binding_sha256"],
        "receipt": checked,
    }


validate_runtime_binding = validate_runtime_identity


def _runtime_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    _, document = _json_ref(path, "runtime JSON")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--unit", type=Path, default=UNIT_PATH)
    parser.add_argument("--focused-evidence", type=Path, required=True)
    parser.add_argument("--installed-probe", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--h0-root", type=Path, required=True)
    parser.add_argument("--h0-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--h0-identity-files-json", type=Path, required=True)
    parser.add_argument("--base-model-files-json", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = seal_pre_gpu_receipt(
            args.plan,
            args.unit,
            args.focused_evidence,
            args.installed_probe,
            args.execution_root,
            args.final_root,
            args.output,
            config=args.config,
            panel=args.panel,
            cohort=args.cohort,
            h0=args.h0,
            manifest=args.manifest,
            census=args.census,
            execution_plan=args.execution_plan,
            cohort_manifest=args.cohort_manifest,
            h0_root=args.h0_root,
            h0_dir=args.h0_dir,
            base_model_dir=args.base_model_dir,
            h0_identity_files=_json_ref(args.h0_identity_files_json, "H0 identity files JSON")[1],
            base_model_files=_json_ref(args.base_model_files_json, "base model files JSON")[1],
            runtime=_runtime_json(args.runtime_json),
        )
    except (PreGpuReceiptError, planner.PlanError, FileExistsError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(document).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
