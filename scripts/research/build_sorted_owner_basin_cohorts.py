#!/usr/bin/env python3
"""Build the CPU-only Task-2 owner cohorts and manual-adjudication queue.

The primary cohort is defined only by the sealed RP=1.0 natural greedy and
K=16 ledgers.  The production RP=1.10 greedy run and any registered sampling
panel are retained as separate positive-evidence strata; neither can change the
primary cohort denominator.  This module never runs a model and never assigns
the downstream high-confidence absence label.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import build_sorted_owner_basin_census as census


COHORT_ASSIGNMENT_SCHEMA_VERSION = "sorted-owner-basin-cohort-assignment.v2"
SAMPLING_SUPPORT_SCHEMA_VERSION = "sorted-owner-basin-sampling-support.v2"
MANUAL_REVIEW_SCHEMA_VERSION = "sorted-owner-basin-manual-review-queue.v2"
RECEIPT_SCHEMA_VERSION = "sorted-owner-basin-cohort-receipt.v2"
MANIFEST_SCHEMA_VERSION = "sorted-owner-basin-cohort-artifact-manifest.v2"
REGISTERED_SUPPORT_SCHEMA_VERSION = "sorted-owner-basin-registered-sampling-support.v2"
REGISTERED_POLICY_RECEIPT_SCHEMA_VERSION = (
    "sorted-owner-basin-registered-sampling-policy-receipt.v1"
)
REGISTERED_PREDICTION_SCHEMA_VERSION = (
    "sorted-owner-basin-registered-sampling-prediction-row.v1"
)

EXPECTED_OWNER_SCHEMA = "sorted-owner-basin-owner-ledger.v2"
EXPECTED_PREDICTION_SCHEMA = "sorted-owner-basin-prediction-row-ledger.v2"
EXPECTED_MATRIX_SCHEMA = "sorted-owner-basin-owner-trajectory-matrix.v2"
EXPECTED_CENSUS_MANIFEST_SCHEMA = "sorted-owner-basin-census-artifact-manifest.v2"
EXPECTED_TASK0_RECEIPT_SCHEMA = "sorted-owner-basin-task0-execution-receipt.v2"
EXPECTED_AMBIGUITY_RECEIPT_SCHEMA = "sorted-owner-basin-ambiguity-receipt.v2"
EXPECTED_SENTINEL_SCHEMA = "sorted-owner-basin-sentinel-registry.v2"
EXPECTED_SENTINEL_SELECTION_RECEIPT_SCHEMA = (
    "sorted-owner-basin-sentinel-selection-receipt.v1"
)
EXPECTED_SENTINEL_CONFIRMATION_RECEIPT_SCHEMA = (
    "sorted-owner-basin-sentinel-selection-confirmation-receipt.v2"
)
EXPECTED_CONTROL_SCHEMA = "sorted-owner-basin-control-registry.v2"
EXPECTED_MATCHER_SCHEMA = "sorted-owner-basin-matcher.v2"

EXPECTED_GREEDY_SEED = 0
EXPECTED_SAMPLED_SEEDS = tuple(range(21001, 21017))
EXPECTED_HORIZON = 3084
PRIMARY_POLICY_STRATUM = "primary_rp_1.00"
RP110_POLICY_STRATUM = "production_rp_1.10_greedy"

DEFAULT_MEANINGFUL_LOOSE_IOU = 0.10
DEFAULT_LOOSE_IOU_LOWER = 0.05
DEFAULT_LOOSE_IOU_UPPER = 0.15

COHORT_STATUSES = (
    "greedy_strict_present",
    "strict_rescued",
    "loose_only_b1",
    "no_free_spatial_support",
    "positive_overlap_neutral",
    "strict_ambiguity_neutral",
)
SUPPORT_STATUSES = (
    "strict_positive",
    "loose_positive_overlap",
    "null_no_positive_evidence",
    "not_supplied",
    "neutral_excluded_global_strict_ambiguity",
)

DEFAULT_CENSUS_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final"
)
DEFAULT_RP110_RUN_DIR = Path(
    "/data/CoordExp/outputs/coordexp_swift/infer/val200/"
    "qwen3-vl-2b-desc-first-geo-sorted-step4887-human-refined12-hf-fp32"
)
DEFAULT_REGISTRY_DIR = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-08-01-sorted-owner-basin-landscape-and-repair"
)

_PRED_ID_RE = re.compile(
    r"^pred:sorted:(greedy|sampled):(\d+):([^:]+):(\d+)$"
)
_TRAJECTORY_ID_RE = re.compile(
    r"^trajectory:sorted:rp1\.00:(greedy|sampled):(\d+):([^:]+)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNIT_RELATIVE_ROOT = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-08-01-sorted-owner-basin-landscape-and-repair"
)


class CohortContractError(ValueError):
    """Raised before incompatible evidence can enter Task 2."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    source = path.resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CohortContractError(f"invalid JSON: {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CohortContractError(f"expected a JSON object: {source}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = path.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CohortContractError(f"invalid JSONL: {source}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise CohortContractError(f"JSONL row is not an object: {source}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortContractError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CohortContractError(f"{label} must be a list")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CohortContractError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise CohortContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _task0_receipt_digest(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in receipt.items()
                if key != "execution_receipt_content_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _find_forced_continuation_paths(value: Any, path: str = "$") -> list[str]:
    """Find continuation markers without conflating ambiguity forced-edge proofs."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if "forced_continuation" in normalized:
                found.append(child_path)
            found.extend(_find_forced_continuation_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forced_continuation_paths(child, f"{path}[{index}]"))
    return found


def _run_git(repository_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise CohortContractError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _is_relevant_untracked(relative_path: str) -> bool:
    path = Path(relative_path)
    if path == _UNIT_RELATIVE_ROOT or _UNIT_RELATIVE_ROOT in path.parents:
        return True
    return (
        len(path.parts) >= 3
        and path.parts[:2] in {("scripts", "research"), ("tests", "research")}
        and "sorted_owner_basin" in path.name
    )


def _repository_state(repository_root: Path | None = None) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[2]
    root = (
        Path(
            _run_git(source_root, "rev-parse", "--show-toplevel")
            .decode("utf-8")
            .strip()
        )
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    tracked_diff = _run_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    tracked_paths = sorted(
        item.decode("utf-8")
        for item in _run_git(root, "diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
        if item
    )
    untracked_paths = sorted(
        item.decode("utf-8")
        for item in _run_git(root, "ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if item and _is_relevant_untracked(item.decode("utf-8"))
    )
    relevant_untracked = []
    for relative in untracked_paths:
        source = (root / relative).resolve(strict=True)
        if not source.is_file():
            raise CohortContractError(f"relevant untracked source is not a file: {source}")
        relevant_untracked.append(
            {
                "path": str(source),
                "relative_path": relative,
                "bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )
    return {
        "root": str(root),
        "head": _run_git(root, "rev-parse", "HEAD").decode("utf-8").strip(),
        "tracked_dirty_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_dirty_diff_bytes": len(tracked_diff),
        "tracked_dirty_paths": tracked_paths,
        "relevant_untracked_files": relevant_untracked,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cpu_runtime_identity() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "execution_device": "cpu",
        "gpu_or_model_execution": "none",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model,
        },
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), float(right), abs_tol=tolerance
    )


def _box_metrics(
    predicted: Sequence[float], target: Sequence[float]
) -> dict[str, float]:
    px1, py1, px2, py2 = (float(item) for item in predicted)
    tx1, ty1, tx2, ty2 = (float(item) for item in target)
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    prediction_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = prediction_area + target_area - intersection
    return {
        "intersection_area": intersection,
        "intersection_over_union": intersection / union if union > 0 else 0.0,
        "intersection_over_prediction": intersection / prediction_area if prediction_area > 0 else 0.0,
        "intersection_over_ground_truth": intersection / target_area if target_area > 0 else 0.0,
    }


def _alias_adjacency(matcher: Mapping[str, Any]) -> dict[str, set[str]]:
    if matcher.get("schema_version") != EXPECTED_MATCHER_SCHEMA:
        raise CohortContractError("matcher-contract.json has an unsupported schema")
    if matcher.get("matcher_id") != EXPECTED_MATCHER_SCHEMA:
        raise CohortContractError("matcher contract matcher_id changed")
    if not _close(matcher.get("iou_threshold"), 0.5):
        raise CohortContractError("matcher IoU threshold must remain 0.5")
    expected_objective = ["maximum_cardinality", "maximum_total_iou"]
    if matcher.get("assignment_objective") != expected_objective:
        raise CohortContractError("matcher assignment objective changed")
    if (
        matcher.get("deterministic_final_tie_break")
        != "gt_original_annotation_index_then_pred_original_row_index"
    ):
        raise CohortContractError("matcher deterministic tie-break changed")
    if not isinstance(matcher.get("global_optimum_ambiguity"), str) or not isinstance(
        matcher.get("ambiguous_global_optimum_accounting"), str
    ):
        raise CohortContractError("matcher lacks explicit global-optimum neutral ambiguity")
    category_namespace = _require_mapping(matcher.get("category_namespace"), "matcher.category_namespace")
    if category_namespace.get("local_evaluator_category_id_join") != "forbidden":
        raise CohortContractError("matcher permits an evaluator-local category join")
    alias_table = _require_mapping(matcher.get("alias_table"), "matcher.alias_table")
    if alias_table.get("schema_version") != "sorted-owner-basin-aliases.v1":
        raise CohortContractError("matcher alias table schema changed")
    adjacency: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(_require_list(alias_table.get("aliases"), "matcher.alias_table.aliases")):
        item = _require_mapping(raw, f"matcher.alias_table.aliases[{index}]")
        left = census._normalize_description(item.get("left"))
        right = census._normalize_description(item.get("right"))
        if not left or not right or left == right:
            raise CohortContractError("matcher alias table contains an invalid pair")
        pair = (left, right) if left < right else (right, left)
        if pair in seen:
            raise CohortContractError("matcher alias table repeats a pair")
        seen.add(pair)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return dict(adjacency)


def _compatible(
    prediction: Mapping[str, Any], owner: Mapping[str, Any], aliases: Mapping[str, set[str]]
) -> bool:
    pred = prediction.get("normalized_description")
    target = owner.get("normalized_description")
    if not isinstance(pred, str) or not isinstance(target, str):
        return False
    return pred == target or target in aliases.get(pred, set())


def _verify_ref(root: Path, manifest: Mapping[str, Any], key: str, expected_name: str) -> Path:
    artifacts = _require_mapping(manifest.get("artifacts"), "census manifest artifacts")
    ref = _require_mapping(artifacts.get(key), f"census manifest artifacts.{key}")
    if ref.get("path") != expected_name:
        raise CohortContractError(f"census manifest {key} path changed from {expected_name}")
    path = (root / expected_name).resolve(strict=True)
    actual = _sha256_file(path)
    if ref.get("sha256") != actual:
        raise CohortContractError(f"stale digest for sealed census artifact {expected_name}")
    return path


def _validate_task0_execution_receipt(
    receipt_path: Path,
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> tuple[Mapping[str, Any], str]:
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != EXPECTED_TASK0_RECEIPT_SCHEMA:
        raise CohortContractError("unsupported Task0 execution receipt schema")
    if receipt.get("execution_status") != "completed":
        raise CohortContractError("Task0 execution receipt is not completed")
    logical_digest = _require_sha256(
        receipt.get("execution_receipt_content_sha256"),
        "Task0 execution receipt content digest",
    )
    digest_contract = _require_mapping(
        receipt.get("content_digest_contract"), "Task0 receipt content_digest_contract"
    )
    if dict(digest_contract) != {
        "algorithm": "sha256",
        "canonicalization": (
            "UTF-8 JSON with ensure_ascii=false, sorted keys, and compact separators"
        ),
        "excluded_top_level_fields": ["execution_receipt_content_sha256"],
        "field": "execution_receipt_content_sha256",
    }:
        raise CohortContractError("Task0 execution receipt digest contract changed")
    if _task0_receipt_digest(receipt) != logical_digest:
        raise CohortContractError("Task0 execution receipt logical digest is stale")
    if manifest.get("execution_receipt_content_sha256") != logical_digest:
        raise CohortContractError("Task0 manifest execution-receipt binding changed")

    command = _require_mapping(receipt.get("command"), "Task0 receipt command")
    argv = _require_list(command.get("argv"), "Task0 receipt command.argv")
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise CohortContractError("Task0 execution receipt lacks exact argv")
    if command.get("shell_escaped") != shlex.join(argv):
        raise CohortContractError("Task0 execution receipt command rendering changed")

    repository = _require_mapping(receipt.get("repository"), "Task0 receipt repository")
    _require_sha256(repository.get("tracked_dirty_diff_sha256"), "Task0 dirty diff digest")
    head = _require_string(repository.get("head"), "Task0 repository head")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise CohortContractError("Task0 repository head is not a commit digest")
    repository_root = Path(
        _require_string(repository.get("root"), "Task0 repository root")
    ).resolve(strict=True)
    tracked_dirty_diff_bytes = repository.get("tracked_dirty_diff_bytes")
    if not isinstance(tracked_dirty_diff_bytes, int) or tracked_dirty_diff_bytes < 0:
        raise CohortContractError("Task0 repository dirty-diff byte count is invalid")
    tracked_paths = _require_list(
        repository.get("tracked_dirty_paths"), "Task0 repository tracked_dirty_paths"
    )
    if not all(isinstance(path, str) and path for path in tracked_paths):
        raise CohortContractError("Task0 repository tracked paths are invalid")
    untracked_files = _require_list(
        repository.get("relevant_untracked_files"),
        "Task0 repository relevant_untracked_files",
    )
    implementation_entries: list[Mapping[str, Any]] = []
    for index, raw in enumerate(untracked_files):
        item = _require_mapping(raw, f"Task0 relevant_untracked_files[{index}]")
        path = Path(
            _require_string(item.get("path"), f"Task0 untracked file {index}.path")
        )
        relative_path = _require_string(
            item.get("relative_path"), f"Task0 untracked file {index}.relative_path"
        )
        if path != repository_root / relative_path:
            raise CohortContractError("Task0 untracked file path is not repository-relative")
        _require_sha256(item.get("sha256"), f"Task0 untracked file {relative_path}.sha256")
        untracked_bytes = item.get("bytes")
        if not isinstance(untracked_bytes, int) or untracked_bytes < 0:
            raise CohortContractError("Task0 untracked file byte count is invalid")
        if relative_path == "scripts/research/build_sorted_owner_basin_census.py":
            implementation_entries.append(item)
    if len(implementation_entries) != 1:
        raise CohortContractError("Task0 receipt lacks a unique implementation-source binding")
    task0_implementation = repository_root / str(implementation_entries[0]["relative_path"])
    if (
        not task0_implementation.is_file()
        or task0_implementation.stat().st_size != implementation_entries[0]["bytes"]
        or _sha256_file(task0_implementation) != implementation_entries[0]["sha256"]
    ):
        raise CohortContractError("Task0 implementation-source binding is stale")

    runtime = _require_mapping(receipt.get("runtime"), "Task0 receipt runtime")
    if runtime.get("execution_device") != "cpu" or runtime.get("cuda_inventory_role") != (
        "observed_environment_metadata_not_used_for_census_execution"
    ):
        raise CohortContractError("Task0 receipt does not attest CPU-only execution")
    runtime_python = _require_mapping(runtime.get("python"), "Task0 runtime.python")
    if runtime_python.get("version") != platform.python_version():
        raise CohortContractError("Task0 receipt Python runtime differs from the consuming runtime")

    inputs = _require_mapping(receipt.get("inputs"), "Task0 receipt inputs")
    bound_files = _require_list(inputs.get("bound_files"), "Task0 receipt inputs.bound_files")
    required_roles = {
        "frozen_panel",
        "matched_rp_1_0_greedy",
        "matched_rp_1_0_sampled_shard_0",
        "matched_rp_1_0_sampled_shard_1",
    }
    bound_by_role: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(bound_files):
        item = _require_mapping(raw, f"Task0 bound_files[{index}]")
        role = _require_string(item.get("role"), f"Task0 bound_files[{index}].role")
        if role in required_roles and role in bound_by_role:
            raise CohortContractError(f"Task0 execution receipt repeats bound role {role}")
        path = Path(_require_string(item.get("path"), f"Task0 bound file {role}.path")).resolve(
            strict=True
        )
        digest = _require_sha256(item.get("sha256"), f"Task0 bound file {role}.sha256")
        if path.stat().st_size != item.get("bytes") or _sha256_file(path) != digest:
            raise CohortContractError(f"Task0 execution receipt bound source is stale: {role}")
        if role in required_roles:
            bound_by_role[role] = item
    if not required_roles.issubset(bound_by_role):
        raise CohortContractError("Task0 execution receipt omits required source bindings")

    matcher_path = artifact_paths["matcher"]
    matcher = _read_json(matcher_path)
    if matcher.get("execution_receipt_content_sha256") != logical_digest:
        raise CohortContractError("Task0 matcher is not bound to its execution receipt")
    return receipt, logical_digest


def _load_census(census_dir: Path) -> dict[str, Any]:
    root = census_dir.resolve(strict=True)
    manifest_path = root / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EXPECTED_CENSUS_MANIFEST_SCHEMA:
        raise CohortContractError("unsupported Task0 artifact-manifest schema")
    forced = _require_mapping(
        manifest.get("forced_continuation_absence"), "manifest.forced_continuation_absence"
    )
    if forced.get("status") != "proved_by_structural_source_scan" or forced.get(
        "forbidden_marker_paths"
    ) != []:
        raise CohortContractError("Task0 did not prove forced-continuation absence")
    paths = {
        "execution_receipt": _verify_ref(
            root, manifest, "execution_receipt", "execution-receipt.json"
        ),
        "matcher": _verify_ref(root, manifest, "matcher_contract", "matcher-contract.json"),
        "owners": _verify_ref(root, manifest, "owner_ledger", "owner-ledger.jsonl"),
        "predictions": _verify_ref(
            root, manifest, "prediction_ledger", "prediction-row-ledger.jsonl"
        ),
        "matrix": _verify_ref(
            root, manifest, "owner_trajectory_matrix", "owner-trajectory-matrix.jsonl"
        ),
        "ambiguity_receipts": _verify_ref(
            root, manifest, "ambiguity_receipts", "ambiguity-receipts.jsonl"
        ),
    }
    execution_receipt, execution_receipt_sha256 = _validate_task0_execution_receipt(
        paths["execution_receipt"], manifest, paths
    )
    matcher = _read_json(paths["matcher"])
    aliases = _alias_adjacency(matcher)
    owners = _read_jsonl(paths["owners"])
    predictions = _read_jsonl(paths["predictions"])
    matrix = _read_jsonl(paths["matrix"])
    ambiguity_receipts = _read_jsonl(paths["ambiguity_receipts"])
    if _find_forced_continuation_paths([owners, predictions, matrix, ambiguity_receipts]):
        raise CohortContractError("sealed ledgers contain forced-continuation markers")
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "paths": paths,
        "execution_receipt": execution_receipt,
        "execution_receipt_sha256": execution_receipt_sha256,
        "matcher": matcher,
        "aliases": aliases,
        "owners": owners,
        "predictions": predictions,
        "matrix": matrix,
        "ambiguity_receipts": ambiguity_receipts,
    }


def _validate_ledgers(bundle: Mapping[str, Any], *, require_full_panel: bool) -> dict[str, Any]:
    owners = bundle["owners"]
    predictions = bundle["predictions"]
    matrix = bundle["matrix"]
    aliases = bundle["aliases"]
    manifest = bundle["manifest"]
    execution_receipt_sha256 = bundle["execution_receipt_sha256"]

    owner_by_id: dict[str, dict[str, Any]] = {}
    owners_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in owners:
        if row.get("schema_version") != EXPECTED_OWNER_SCHEMA:
            raise CohortContractError("owner ledger schema changed")
        if row.get("execution_receipt_content_sha256") != execution_receipt_sha256:
            raise CohortContractError("owner ledger execution-receipt binding changed")
        gt_owner_id = _require_string(row.get("gt_owner_id"), "owner.gt_owner_id")
        image_id = _require_string(row.get("image_id"), f"owner {gt_owner_id}.image_id")
        original_index = row.get("original_annotation_index")
        if not isinstance(original_index, int) or isinstance(original_index, bool) or original_index < 0:
            raise CohortContractError(f"owner {gt_owner_id} has invalid original_annotation_index")
        if gt_owner_id != f"gt:{image_id}:{original_index}":
            raise CohortContractError(f"owner identity was renumbered: {gt_owner_id}")
        if gt_owner_id in owner_by_id:
            raise CohortContractError(f"duplicate gt_owner_id: {gt_owner_id}")
        census._pixel_box(row.get("bbox_xyxy"), context=f"owner {gt_owner_id}.bbox_xyxy")
        if row.get("mapped_gt_owner_id") != gt_owner_id:
            raise CohortContractError(f"owner {gt_owner_id} mapping changed")
        owner_by_id[gt_owner_id] = row
        owners_by_image[image_id].append(row)
    if require_full_panel and (len(owners_by_image) != 12 or len(owner_by_id) != 346):
        raise CohortContractError("Task2 primary panel must contain 12 images and 346 owners")

    sources = _require_mapping(manifest.get("sources"), "census manifest sources")
    greedy_ref = _require_mapping(sources.get("matched_rp_1_0_greedy"), "primary greedy source")
    sampled_refs = _require_list(
        sources.get("matched_rp_1_0_sampled_shards"), "primary sampled sources"
    )
    allowed_rollout_digests = {_require_string(greedy_ref.get("sha256"), "greedy sha256")}
    allowed_rollout_digests.update(
        _require_string(_require_mapping(item, "sampled source").get("sha256"), "sampled sha256")
        for item in sampled_refs
    )

    prediction_by_id: dict[str, dict[str, Any]] = {}
    predictions_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row.get("schema_version") != EXPECTED_PREDICTION_SCHEMA:
            raise CohortContractError("prediction ledger schema changed")
        if row.get("execution_receipt_content_sha256") != execution_receipt_sha256:
            raise CohortContractError("prediction ledger execution-receipt binding changed")
        pred_row_id = _require_string(row.get("pred_row_id"), "prediction.pred_row_id")
        match = _PRED_ID_RE.fullmatch(pred_row_id)
        if match is None:
            raise CohortContractError(f"prediction identity has unsupported form: {pred_row_id}")
        mode, seed_text, id_image, index_text = match.groups()
        seed = row.get("seed")
        row_index = row.get("original_row_index")
        image_id = row.get("image_id")
        if (
            row.get("decode_mode") != mode
            or seed != int(seed_text)
            or image_id != id_image
            or row_index != int(index_text)
        ):
            raise CohortContractError(f"prediction identity was renumbered: {pred_row_id}")
        expected_seed = EXPECTED_GREEDY_SEED if mode == "greedy" else None
        if (expected_seed is not None and seed != expected_seed) or (
            mode == "sampled" and seed not in EXPECTED_SAMPLED_SEEDS
        ):
            raise CohortContractError(f"prediction {pred_row_id} has an unregistered seed")
        if row.get("policy_stratum") != PRIMARY_POLICY_STRATUM:
            raise CohortContractError(f"prediction {pred_row_id} mixes a non-primary policy")
        trajectory_id = _require_string(row.get("trajectory_id"), f"prediction {pred_row_id}.trajectory_id")
        expected_trajectory = f"trajectory:sorted:rp1.00:{mode}:{seed}:{image_id}"
        if trajectory_id != expected_trajectory:
            raise CohortContractError(f"prediction {pred_row_id} has a renumbered trajectory")
        if pred_row_id in prediction_by_id:
            raise CohortContractError(f"duplicate pred_row_id: {pred_row_id}")
        if row.get("source_artifact_sha256") not in allowed_rollout_digests:
            raise CohortContractError(f"prediction {pred_row_id} has a stale or foreign source digest")
        if row.get("row_kind") == "complete_prediction":
            census._pixel_box(row.get("bbox_xyxy"), context=f"prediction {pred_row_id}.bbox_xyxy")
            if not isinstance(row.get("normalized_description"), str):
                raise CohortContractError(f"prediction {pred_row_id} lacks normalized description")
        prediction_by_id[pred_row_id] = row
        predictions_by_trajectory[trajectory_id].append(row)

    matrix_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    trajectory_meta: dict[str, tuple[str, int, str]] = {}
    for row in matrix:
        if row.get("schema_version") != EXPECTED_MATRIX_SCHEMA:
            raise CohortContractError("owner-trajectory matrix schema changed")
        if row.get("execution_receipt_content_sha256") != execution_receipt_sha256:
            raise CohortContractError("matrix execution-receipt binding changed")
        gt_owner_id = _require_string(row.get("gt_owner_id"), "matrix.gt_owner_id")
        owner = owner_by_id.get(gt_owner_id)
        if owner is None or row.get("image_id") != owner["image_id"]:
            raise CohortContractError(f"matrix row has a foreign owner: {gt_owner_id}")
        trajectory_id = _require_string(row.get("trajectory_id"), "matrix.trajectory_id")
        match = _TRAJECTORY_ID_RE.fullmatch(trajectory_id)
        if match is None:
            raise CohortContractError(f"matrix trajectory identity changed: {trajectory_id}")
        mode, seed_text, image_id = match.groups()
        seed = int(seed_text)
        if (
            row.get("policy_stratum") != PRIMARY_POLICY_STRATUM
            or row.get("decode_mode") != mode
            or row.get("seed") != seed
            or row.get("image_id") != image_id
        ):
            raise CohortContractError(f"matrix row mixes policy or trajectory identity: {trajectory_id}")
        if (mode == "greedy" and seed != EXPECTED_GREEDY_SEED) or (
            mode == "sampled" and seed not in EXPECTED_SAMPLED_SEEDS
        ):
            raise CohortContractError(f"matrix contains an unregistered seed: {trajectory_id}")
        key = (gt_owner_id, trajectory_id)
        if key in matrix_by_key:
            raise CohortContractError(f"duplicate owner-trajectory matrix row: {key}")
        matrix_by_key[key] = row
        trajectory_meta[trajectory_id] = (mode, seed, image_id)

    expected_trajectories = {
        f"trajectory:sorted:rp1.00:greedy:0:{image_id}"
        for image_id in owners_by_image
    }
    expected_trajectories.update(
        f"trajectory:sorted:rp1.00:sampled:{seed}:{image_id}"
        for image_id in owners_by_image
        for seed in EXPECTED_SAMPLED_SEEDS
    )
    if set(trajectory_meta) != expected_trajectories:
        missing = sorted(expected_trajectories - set(trajectory_meta))
        extra = sorted(set(trajectory_meta) - expected_trajectories)
        raise CohortContractError(
            f"missing K16 seeds or unexpected trajectories; missing={missing[:3]} extra={extra[:3]}"
        )
    expected_matrix_keys = {
        (owner_id, trajectory_id)
        for owner_id, owner in owner_by_id.items()
        for trajectory_id, (_, _, image_id) in trajectory_meta.items()
        if image_id == owner["image_id"]
    }
    if set(matrix_by_key) != expected_matrix_keys:
        raise CohortContractError("owner-trajectory matrix is not the complete owner-by-K16 panel")

    ambiguity_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguity_by_id: dict[str, dict[str, Any]] = {}
    for row in bundle["ambiguity_receipts"]:
        if row.get("schema_version") != EXPECTED_AMBIGUITY_RECEIPT_SCHEMA:
            raise CohortContractError("ambiguity-receipt schema changed")
        if row.get("execution_receipt_content_sha256") != execution_receipt_sha256:
            raise CohortContractError("ambiguity receipt execution binding changed")
        receipt_id = _require_string(
            row.get("ambiguity_receipt_id"), "ambiguity_receipt_id"
        )
        trajectory_id = _require_string(row.get("trajectory_id"), "ambiguity trajectory_id")
        if receipt_id in ambiguity_by_id or trajectory_id not in expected_trajectories:
            raise CohortContractError(f"duplicate or foreign ambiguity receipt {receipt_id}")
        pred_ids = _require_list(row.get("pred_row_ids"), f"ambiguity {receipt_id}.pred_row_ids")
        owner_ids = _require_list(row.get("gt_owner_ids"), f"ambiguity {receipt_id}.gt_owner_ids")
        if not pred_ids or not owner_ids:
            raise CohortContractError(f"ambiguity receipt {receipt_id} has an empty ambiguity class")
        if any(pred_id not in prediction_by_id for pred_id in pred_ids) or any(
            owner_id not in owner_by_id for owner_id in owner_ids
        ):
            raise CohortContractError(f"ambiguity receipt {receipt_id} has a foreign key")
        foreign_keys = _require_mapping(row.get("foreign_keys"), f"ambiguity {receipt_id}.foreign_keys")
        if (
            foreign_keys.get("trajectory_id") != trajectory_id
            or foreign_keys.get("pred_row_ids") != pred_ids
            or foreign_keys.get("gt_owner_ids") != owner_ids
        ):
            raise CohortContractError(f"ambiguity receipt {receipt_id} foreign keys drifted")
        ambiguity_by_id[receipt_id] = row
        ambiguity_by_trajectory[trajectory_id].append(row)

    neutral_owner_ids: set[str] = set()
    neutral_pred_row_ids: set[str] = set()
    greedy_strict_owner_ids: set[str] = set()
    sampled_strict_owner_ids: set[str] = set()

    # Re-run the sealed matcher contract and compare both ledgers.  No aliases
    # beyond matcher-contract.json are introduced here.
    for trajectory_id, (mode, seed, image_id) in sorted(trajectory_meta.items()):
        complete_predictions = [
            row
            for row in predictions_by_trajectory.get(trajectory_id, [])
            if row.get("row_kind") == "complete_prediction"
        ]
        trajectory_receipt = census._trajectory_receipt(
            {
                "trajectory_id": trajectory_id,
                "image_id": image_id,
                "decode_mode": mode,
                "seed": seed,
                "source_artifact_sha256": str(
                    _require_mapping(
                        matrix_by_key[
                            (
                                str(owners_by_image[image_id][0]["gt_owner_id"]),
                                trajectory_id,
                            )
                        ].get("source_digests"),
                        f"matrix {trajectory_id}.source_digests",
                    )["rollout_artifact"]
                ),
                "predictions": complete_predictions,
            },
            owners_by_image[image_id],
            aliases,
        )
        matches = trajectory_receipt["matches"]
        trajectory_pred_receipts = {
            str(item["pred_row_id"]): item
            for item in trajectory_receipt["prediction_receipts"]
        }
        trajectory_neutral_owners = set(trajectory_receipt["neutral_owner_ids"])
        trajectory_neutral_predictions = set(trajectory_receipt["neutral_pred_row_ids"])
        neutral_owner_ids.update(trajectory_neutral_owners)
        neutral_pred_row_ids.update(trajectory_neutral_predictions)
        match_by_pred = {item["pred_row_id"]: item for item in matches}
        optimal_by_pred: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        optimal_by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for edge in trajectory_receipt["globally_optimal_edge_receipts"]:
            optimal_by_pred[str(edge["pred_row_id"])].append(edge)
            optimal_by_owner[str(edge["gt_owner_id"])].append(edge)
        expected_ambiguity_receipts = sorted(
            trajectory_receipt["ambiguity_receipts"],
            key=lambda item: str(item["ambiguity_receipt_id"]),
        )
        observed_ambiguity_receipts = sorted(
            ambiguity_by_trajectory.get(trajectory_id, []),
            key=lambda item: str(item["ambiguity_receipt_id"]),
        )
        for item in expected_ambiguity_receipts:
            item["execution_receipt_content_sha256"] = execution_receipt_sha256
            item["source_digests"] = {
                **item["source_digests"],
                "execution_receipt_content": execution_receipt_sha256,
            }
        if observed_ambiguity_receipts != expected_ambiguity_receipts:
            raise CohortContractError(f"ambiguity receipt ledger drifted for {trajectory_id}")
        matches_by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in matches:
            matches_by_owner[str(item["gt_owner_id"])].append(item)
        for prediction in complete_predictions:
            pred_row_id = str(prediction["pred_row_id"])
            expected = match_by_pred.get(pred_row_id)
            expected_owner = expected["gt_owner_id"] if expected else None
            expected_status = (
                "ambiguous_neutral"
                if pred_row_id in trajectory_neutral_predictions
                else (expected["strict_match_status"] if expected else "unmatched")
            )
            if (
                prediction.get("strict_match_gt_owner_id") != expected_owner
                or prediction.get("strict_match_status") != expected_status
            ):
                raise CohortContractError(f"strict matcher receipt drifted for {pred_row_id}")
            if expected and not _close(
                prediction.get("strict_match_iou"), expected["intersection_over_union"]
            ):
                raise CohortContractError(f"strict matcher IoU drifted for {pred_row_id}")
            if prediction.get("globally_optimal_edge_receipts") != optimal_by_pred.get(
                pred_row_id, []
            ):
                raise CohortContractError(
                    f"global ambiguity edge receipt drifted for {pred_row_id}"
                )
            if prediction.get("ambiguity_receipt_ids") != trajectory_pred_receipts[pred_row_id].get(
                "ambiguity_receipt_ids"
            ):
                raise CohortContractError(
                    f"prediction ambiguity foreign keys drifted for {pred_row_id}"
                )
        for owner in owners_by_image[image_id]:
            gt_owner_id = str(owner["gt_owner_id"])
            row = matrix_by_key[(gt_owner_id, trajectory_id)]
            owner_matches = matches_by_owner.get(gt_owner_id, [])
            expected_pred_ids = sorted(str(item["pred_row_id"]) for item in owner_matches)
            expected_presence = any(
                item["strict_match_status"] == "matched" for item in owner_matches
            )
            expected_neutral = gt_owner_id in trajectory_neutral_owners
            if sorted(row.get("matched_pred_row_ids", [])) != expected_pred_ids or row.get(
                "strict_match_presence"
            ) is not expected_presence:
                raise CohortContractError(
                    f"matrix strict receipt drifted for {gt_owner_id} / {trajectory_id}"
                )
            if row.get("global_ambiguity_presence") is not expected_neutral or row.get(
                "globally_optimal_edge_receipts"
            ) != optimal_by_owner.get(gt_owner_id, []):
                raise CohortContractError(
                    f"matrix neutral ambiguity receipt drifted for {gt_owner_id} / {trajectory_id}"
                )
            expected_ambiguity_ids = sorted(
                str(item["ambiguity_receipt_id"])
                for item in expected_ambiguity_receipts
                if gt_owner_id in item["gt_owner_ids"]
            )
            if row.get("ambiguity_receipt_ids") != expected_ambiguity_ids:
                raise CohortContractError(
                    f"matrix ambiguity foreign keys drifted for {gt_owner_id} / {trajectory_id}"
                )
            if expected_presence:
                if mode == "greedy":
                    greedy_strict_owner_ids.add(gt_owner_id)
                else:
                    sampled_strict_owner_ids.add(gt_owner_id)

    raw_trajectory_neutral_owner_ids = set(neutral_owner_ids)
    primary_metrics = _require_mapping(manifest.get("primary_metrics"), "Task0 primary_metrics")
    greedy_metrics = _require_mapping(
        primary_metrics.get("matched_rp_1_0_greedy"), "Task0 greedy metrics"
    )
    sampled_metrics = _require_mapping(
        primary_metrics.get("matched_rp_1_0_k16_any_hit"), "Task0 K16 metrics"
    )
    paired_metrics = _require_mapping(
        sampled_metrics.get("ambiguity_neutral_paired_set"),
        "Task0 ambiguity_neutral_paired_set",
    )
    neutral_owner_ids = set(
        _require_list(
            paired_metrics.get("excluded_neutral_gt_owner_ids"),
            "Task0 paired excluded_neutral_gt_owner_ids",
        )
    )
    if not neutral_owner_ids.issubset(raw_trajectory_neutral_owner_ids):
        raise CohortContractError("Task0 paired neutral set contains an owner without an ambiguity receipt")
    eligible_owner_ids = set(owner_by_id) - neutral_owner_ids
    greedy_strict_owner_ids &= eligible_owner_ids
    sampled_strict_owner_ids &= eligible_owner_ids
    strict_rescued_owner_ids = sampled_strict_owner_ids - greedy_strict_owner_ids

    for owner_id, owner in owner_by_id.items():
        expected_ambiguity_ids = sorted(
            receipt_id
            for receipt_id, receipt in ambiguity_by_id.items()
            if owner_id in receipt["gt_owner_ids"]
        )
        if owner.get("ambiguity_receipt_ids") != expected_ambiguity_ids:
            raise CohortContractError(f"owner ambiguity foreign keys drifted for {owner_id}")
        decision = _require_mapping(
            owner.get("decision_eligibility"), f"owner {owner_id}.decision_eligibility"
        )
        paired = _require_mapping(
            decision.get("greedy_k16_paired"),
            f"owner {owner_id}.decision_eligibility.greedy_k16_paired",
        )
        if paired.get("eligible") is not (owner_id in eligible_owner_ids):
            raise CohortContractError(f"owner paired eligibility drifted for {owner_id}")

    if set(greedy_metrics.get("neutral_gt_owner_ids", [])) - neutral_owner_ids:
        raise CohortContractError("Task0 greedy neutral-owner set is inconsistent")
    if set(sampled_metrics.get("neutral_gt_owner_ids", [])) - neutral_owner_ids:
        raise CohortContractError("Task0 K16 neutral-owner set is inconsistent")
    if sampled_metrics.get("strict_rescued_gt_owner_ids") != sorted(strict_rescued_owner_ids):
        raise CohortContractError("Task0 strict-rescued set is not neutral-excluded set algebra")
    if paired_metrics.get("excluded_neutral_gt_owner_ids") != sorted(neutral_owner_ids):
        raise CohortContractError("Task0 paired neutral-owner set is stale")
    if paired_metrics.get("owner_denominator") != len(eligible_owner_ids):
        raise CohortContractError("Task0 paired effective denominator is stale")
    if paired_metrics.get("strict_rescued_gt_owner_ids") != sorted(strict_rescued_owner_ids):
        raise CohortContractError("Task0 paired strict-rescued set is stale")

    return {
        "owner_by_id": owner_by_id,
        "owners_by_image": dict(owners_by_image),
        "prediction_by_id": prediction_by_id,
        "predictions_by_trajectory": dict(predictions_by_trajectory),
        "matrix_by_key": matrix_by_key,
        "trajectory_meta": trajectory_meta,
        "neutral_owner_ids": neutral_owner_ids,
        "neutral_pred_row_ids": neutral_pred_row_ids,
        "eligible_owner_ids": eligible_owner_ids,
        "greedy_strict_owner_ids": greedy_strict_owner_ids,
        "sampled_strict_owner_ids": sampled_strict_owner_ids,
        "strict_rescued_owner_ids": strict_rescued_owner_ids,
    }


def _validate_control_registry(
    path: Path, bundle: Mapping[str, Any], validated: Mapping[str, Any], *, require_full_panel: bool
) -> tuple[Mapping[str, Any], dict[str, list[str]]]:
    source = path.resolve(strict=True)
    registry = _read_json(source)
    if registry.get("schema_version") != EXPECTED_CONTROL_SCHEMA:
        raise CohortContractError("unsupported control registry schema")
    if registry.get("status") != "lead_frozen_before_scoring_and_resealed_to_final_task0_v2":
        raise CohortContractError("control registry is not lead-frozen")
    controls = _require_list(registry.get("controls"), "control registry controls")
    if require_full_panel and len(controls) != 8:
        raise CohortContractError("current full-panel control registry must contain eight entries")
    digests = _require_mapping(registry.get("source_digests"), "control registry source_digests")
    expected = {
        "task0_census_artifact_manifest_sha256": _sha256_file(bundle["manifest_path"]),
        "task0_execution_receipt_content_sha256": bundle["execution_receipt_sha256"],
        "task0_execution_receipt_file_sha256": _sha256_file(
            bundle["paths"]["execution_receipt"]
        ),
        "owner_ledger_sha256": _sha256_file(bundle["paths"]["owners"]),
        "owner_trajectory_matrix_sha256": _sha256_file(bundle["paths"]["matrix"]),
    }
    for key, value in expected.items():
        if digests.get(key) != value:
            raise CohortContractError(f"control registry has a stale {key}")
    if Path(_require_string(digests.get("task0_v2_root"), "control task0_v2_root")).resolve(
        strict=True
    ) != bundle["root"]:
        raise CohortContractError("control registry points to a different Task0-v2 root")
    confirmation_path = path.parent / "sentinel-selection-confirmation-receipt.json"
    if _sha256_file(confirmation_path.resolve(strict=True)) != digests.get(
        "sentinel_selection_confirmation_receipt_sha256"
    ):
        raise CohortContractError("control registry has a stale sentinel confirmation receipt")
    roles_by_owner: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for raw in controls:
        item = _require_mapping(raw, "control registry entry")
        control_id = _require_string(item.get("control_id"), "control.control_id")
        if control_id in seen:
            raise CohortContractError(f"duplicate control_id: {control_id}")
        seen.add(control_id)
        owner_fields = ["gt_owner_id", "covering_gt_owner_id", "target_gt_owner_id"]
        referenced = [str(item[field]) for field in owner_fields if item.get(field) is not None]
        if not referenced:
            raise CohortContractError(f"control {control_id} has no owner foreign key")
        for owner_id in referenced:
            owner = validated["owner_by_id"].get(owner_id)
            if owner is None:
                raise CohortContractError(f"control {control_id} references unknown owner {owner_id}")
            roles_by_owner[owner_id].append(f"{control_id}:{item.get('role')}")
        for field in ("matched_pred_row_id", "covering_pred_row_id"):
            pred_id = item.get(field)
            if pred_id is not None and pred_id not in validated["prediction_by_id"]:
                raise CohortContractError(f"control {control_id} references unknown row {pred_id}")
    return registry, {key: sorted(value) for key, value in roles_by_owner.items()}


def _validate_sentinel_registry(
    path: Path, bundle: Mapping[str, Any], validated: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    source = path.resolve(strict=True)
    registry = _read_json(source)
    if registry.get("schema_version") != EXPECTED_SENTINEL_SCHEMA:
        raise CohortContractError("unsupported sentinel registry schema")
    if registry.get("selection_status") != (
        "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2"
    ):
        raise CohortContractError("sentinel registry is not lead-frozen")
    if registry.get("claim_scope") != "outcome_selected_case_studies_only_no_prevalence":
        raise CohortContractError("sentinel registry may not support prevalence")
    digests = _require_mapping(registry.get("source_digests"), "sentinel source_digests")
    sources = _require_mapping(bundle["manifest"].get("sources"), "census sources")
    sampled = _require_list(sources.get("matched_rp_1_0_sampled_shards"), "sampled sources")
    expected = {
        "task0_census_artifact_manifest_sha256": _sha256_file(bundle["manifest_path"]),
        "task0_execution_receipt_content_sha256": bundle["execution_receipt_sha256"],
        "task0_execution_receipt_file_sha256": _sha256_file(
            bundle["paths"]["execution_receipt"]
        ),
        "task0_owner_ledger_sha256": _sha256_file(bundle["paths"]["owners"]),
        "task0_owner_trajectory_matrix_sha256": _sha256_file(bundle["paths"]["matrix"]),
        "human_refined_panel_sha256": _require_mapping(sources.get("panel"), "panel source").get("sha256"),
        "matched_rp_1_0_greedy_sha256": _require_mapping(
            sources.get("matched_rp_1_0_greedy"), "greedy source"
        ).get("sha256"),
        "matched_rp_1_0_sampled_shard_0_sha256": _require_mapping(sampled[0], "sampled[0]").get("sha256"),
        "matched_rp_1_0_sampled_shard_1_sha256": _require_mapping(sampled[1], "sampled[1]").get("sha256"),
    }
    for key, value in expected.items():
        if digests.get(key) != value:
            raise CohortContractError(f"sentinel registry has a stale {key}")
    if Path(_require_string(digests.get("task0_v2_root"), "sentinel task0_v2_root")).resolve(
        strict=True
    ) != bundle["root"]:
        raise CohortContractError("sentinel registry points to a different Task0-v2 root")
    selection_ref = _require_mapping(registry.get("selection_receipt"), "sentinel selection_receipt")
    receipt_path = (source.parent / _require_string(selection_ref.get("path"), "selection receipt path")).resolve(strict=True)
    if _sha256_file(receipt_path) != selection_ref.get("sha256"):
        raise CohortContractError("sentinel selection receipt digest is stale")
    selection_receipt = _read_json(receipt_path)
    if selection_receipt.get("schema_version") != EXPECTED_SENTINEL_SELECTION_RECEIPT_SCHEMA:
        raise CohortContractError("unsupported sentinel selection receipt schema")
    if (
        selection_receipt.get("selection_status")
        != "lead_reviewed_and_frozen_before_landscape_scoring"
    ):
        raise CohortContractError("sentinel selection receipt is not lead-reviewed and frozen")
    if selection_receipt.get("claim_scope") != registry.get("claim_scope"):
        raise CohortContractError("sentinel receipt and registry claim scopes disagree")
    anti_leakage = _require_mapping(
        selection_receipt.get("anti_leakage_contract"), "sentinel receipt anti_leakage_contract"
    )
    if anti_leakage.get("landscape_scores_used_for_selection") is not False:
        raise CohortContractError("sentinel selection used outcome-leaking landscape scores")
    receipt_sources = _require_mapping(
        selection_receipt.get("source_artifacts"), "sentinel receipt source_artifacts"
    )
    original_expected = {
        key: expected[key]
        for key in (
            "human_refined_panel_sha256",
            "matched_rp_1_0_greedy_sha256",
            "matched_rp_1_0_sampled_shard_0_sha256",
            "matched_rp_1_0_sampled_shard_1_sha256",
        )
    }
    for key, value in original_expected.items():
        if receipt_sources.get(key) != value:
            raise CohortContractError(f"original sentinel receipt has a stale {key}")

    visual_review = _require_mapping(
        selection_receipt.get("visual_review"), "sentinel receipt visual_review"
    )
    visual_path = Path(
        _require_string(visual_review.get("image_path"), "sentinel visual image_path")
    ).resolve(strict=True)
    if _sha256_file(visual_path) != visual_review.get("image_sha256"):
        raise CohortContractError("sentinel visual-review image digest is stale")
    far_selection = _require_mapping(
        selection_receipt.get("deterministic_far_person_selection"),
        "sentinel receipt deterministic_far_person_selection",
    )
    far_path = Path(
        _require_string(far_selection.get("image_path"), "sentinel far-person image_path")
    ).resolve(strict=True)
    if _sha256_file(far_path) != far_selection.get("image_sha256"):
        raise CohortContractError("sentinel far-person image digest is stale")

    by_owner: dict[str, Mapping[str, Any]] = {}
    for raw in _require_list(registry.get("sentinels"), "sentinel registry sentinels"):
        item = _require_mapping(raw, "sentinel")
        owner_id = _require_string(item.get("gt_owner_id"), "sentinel.gt_owner_id")
        owner = validated["owner_by_id"].get(owner_id)
        if owner is None:
            raise CohortContractError(f"sentinel references unknown owner {owner_id}")
        if (
            item.get("image_id") != owner["image_id"]
            or item.get("original_annotation_index") != owner["original_annotation_index"]
            or list(item.get("bbox_xyxy", [])) != list(owner["bbox_xyxy"])
        ):
            raise CohortContractError(f"sentinel owner provenance changed for {owner_id}")
        if item.get("prior_non_recovery_status") != "verified_primary_natural_zero_spatial_support":
            raise CohortContractError(f"sentinel {owner_id} lacks exact prior non-recovery provenance")
        if owner_id in by_owner:
            raise CohortContractError(f"duplicate sentinel owner {owner_id}")
        by_owner[owner_id] = item
    receipt_owner_ids = {
        str(item.get("gt_owner_id"))
        for section in (visual_review, far_selection)
        for item in _require_list(section.get("owners"), "sentinel receipt owners")
        if isinstance(item, Mapping)
    }
    if receipt_owner_ids != set(by_owner):
        raise CohortContractError("sentinel receipt and registry owner sets disagree")

    confirmation_ref = _require_mapping(
        registry.get("selection_confirmation_receipt"),
        "sentinel selection_confirmation_receipt",
    )
    confirmation_path = (
        source.parent
        / _require_string(confirmation_ref.get("path"), "confirmation receipt path")
    ).resolve(strict=True)
    confirmation_digest = _sha256_file(confirmation_path)
    if confirmation_digest != confirmation_ref.get("sha256"):
        raise CohortContractError("sentinel confirmation receipt digest is stale")
    confirmation = _read_json(confirmation_path)
    if confirmation.get("schema_version") != EXPECTED_SENTINEL_CONFIRMATION_RECEIPT_SCHEMA:
        raise CohortContractError("unsupported sentinel confirmation receipt schema")
    if confirmation.get("confirmation_status") != (
        "lead_reconfirmed_against_final_task0_v2_before_landscape_scoring"
    ):
        raise CohortContractError("sentinel confirmation receipt is not final")
    if confirmation.get("claim_scope") != registry.get("claim_scope"):
        raise CohortContractError("sentinel confirmation and registry claim scopes disagree")
    confirmation_anti_leakage = _require_mapping(
        confirmation.get("anti_leakage_contract"),
        "sentinel confirmation anti_leakage_contract",
    )
    if (
        confirmation_anti_leakage.get("landscape_scores_used_for_original_selection")
        is not False
        or confirmation_anti_leakage.get("landscape_scores_used_for_v2_confirmation")
        is not False
        or confirmation_anti_leakage.get("selection_membership_changed") is not False
    ):
        raise CohortContractError("sentinel confirmation violates the anti-leakage contract")
    original_ref = _require_mapping(
        confirmation.get("original_selection_receipt"),
        "sentinel confirmation original_selection_receipt",
    )
    if dict(original_ref) != {
        "path": selection_ref.get("path"),
        "sha256": selection_ref.get("sha256"),
    }:
        raise CohortContractError("sentinel confirmation does not bind the original receipt")
    final_task0 = _require_mapping(
        confirmation.get("final_task0_v2"), "sentinel confirmation final_task0_v2"
    )
    final_expected = {
        "root": str(bundle["root"]),
        "artifact_manifest_sha256": expected["task0_census_artifact_manifest_sha256"],
        "execution_receipt_content_sha256": expected[
            "task0_execution_receipt_content_sha256"
        ],
        "execution_receipt_file_sha256": expected["task0_execution_receipt_file_sha256"],
        "owner_ledger_sha256": expected["task0_owner_ledger_sha256"],
        "owner_trajectory_matrix_sha256": expected[
            "task0_owner_trajectory_matrix_sha256"
        ],
    }
    for key, value in final_expected.items():
        if final_task0.get(key) != value:
            raise CohortContractError(f"sentinel confirmation has a stale final Task0 {key}")
    confirmed = _require_list(
        confirmation.get("confirmed_sentinels"), "sentinel confirmation confirmed_sentinels"
    )
    confirmed_by_owner: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(confirmed):
        item = _require_mapping(raw, f"confirmed_sentinels[{index}]")
        owner_id = _require_string(item.get("gt_owner_id"), "confirmed sentinel owner")
        if owner_id in confirmed_by_owner:
            raise CohortContractError(f"duplicate confirmed sentinel owner {owner_id}")
        confirmed_by_owner[owner_id] = item
    if set(confirmed_by_owner) != set(by_owner):
        raise CohortContractError("sentinel confirmation and registry owner sets disagree")
    for owner_id, item in confirmed_by_owner.items():
        rows = [
            row
            for (candidate_owner_id, _), row in validated["matrix_by_key"].items()
            if candidate_owner_id == owner_id
        ]
        if (
            item.get("decision_eligible") is not True
            or owner_id not in validated["eligible_owner_ids"]
            or item.get("trajectory_count") != len(rows)
            or item.get("trajectory_count") != 17
            or item.get("strict_match_count") != sum(
                int(bool(row.get("strict_match_presence"))) for row in rows
            )
            or float(item.get("max_semantic_compatible_iou", math.nan))
            != max((float(row["max_semantic_compatible_iou"]) for row in rows), default=0.0)
        ):
            raise CohortContractError(f"sentinel confirmation does not replay for {owner_id}")
    return registry, by_owner


def _natural_support(
    owner: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, set[str]],
    *,
    thresholds: tuple[float, float, float],
) -> dict[str, Any]:
    compatible_rows: list[tuple[Mapping[str, Any], dict[str, float]]] = []
    for prediction in predictions:
        if (
            prediction.get("row_kind") != "complete_prediction"
            or prediction.get("image_id") != owner["image_id"]
            or not _compatible(prediction, owner, aliases)
        ):
            continue
        metrics = _box_metrics(prediction["bbox_xyxy"], owner["bbox_xyxy"])
        compatible_rows.append((prediction, metrics))
    maxima = {
        key: max((metrics[key] for _, metrics in compatible_rows), default=0.0)
        for key in (
            "intersection_area",
            "intersection_over_union",
            "intersection_over_prediction",
            "intersection_over_ground_truth",
        )
    }
    positive = [
        str(prediction["pred_row_id"])
        for prediction, metrics in compatible_rows
        if metrics["intersection_area"] > 0.0
    ]
    lower, base, upper = thresholds
    decisions = {
        "lower": maxima["intersection_over_union"] >= lower,
        "predeclared": maxima["intersection_over_union"] >= base,
        "upper": maxima["intersection_over_union"] >= upper,
    }
    return {
        "semantic_compatible_prediction_count": len(compatible_rows),
        "any_positive_overlap": bool(positive),
        "blocks_no_free": bool(positive),
        "no_free_spatial_support": not positive,
        "max_semantic_compatible_iou": maxima["intersection_over_union"],
        "max_semantic_compatible_gt_coverage": maxima["intersection_over_ground_truth"],
        "max_semantic_compatible_prediction_coverage": maxima[
            "intersection_over_prediction"
        ],
        "max_semantic_compatible_intersection_area": maxima["intersection_area"],
        "supporting_pred_row_ids": sorted(positive),
        "meaningful_loose_iou_diagnostic": {
            "metric": "intersection_over_union",
            "lower_threshold": lower,
            "predeclared_threshold": base,
            "upper_threshold": upper,
            "positive_at_threshold": decisions,
            "stable_across_threshold_band": len(set(decisions.values())) == 1,
            "stable_meaningful_positive_across_threshold_band": all(decisions.values()),
            "cohort_rule_independent_of_diagnostic_thresholds": True,
        },
        "no_free_support_stable_across_threshold_band": True,
    }


def _support_status(strict: bool, natural: Mapping[str, Any]) -> str:
    if strict:
        return "strict_positive"
    if natural["any_positive_overlap"]:
        return "loose_positive_overlap"
    return "null_no_positive_evidence"


def _load_rp110_support(
    run_dir: Path | None,
    bundle: Mapping[str, Any],
    validated: Mapping[str, Any],
    aliases: Mapping[str, set[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None]:
    owner_by_id = validated["owner_by_id"]
    if run_dir is None:
        return {
            owner_id: {
                "status": "not_supplied",
                "supporting_pred_row_ids": [],
                "max_semantic_compatible_iou": None,
            }
            for owner_id in owner_by_id
        }, None
    root = run_dir.resolve(strict=True)
    manifest_path = root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    recorded = _require_mapping(
        _require_mapping(bundle["manifest"].get("sources"), "census sources").get(
            "optional_production_rp_1_10_manifest"
        ),
        "Task0 RP1.10 manifest registration",
    )
    manifest_digest = _sha256_file(manifest_path)
    if recorded.get("sha256") != manifest_digest:
        raise CohortContractError("RP1.10 run manifest differs from the Task0-registered manifest")
    if manifest.get("terminal_status") != "completed":
        raise CohortContractError("RP1.10 run is not terminally completed")
    policy = _require_mapping(manifest.get("generation_policy"), "RP1.10 generation_policy")
    expected_policy = {
        "do_sample": False,
        "max_new_tokens": EXPECTED_HORIZON,
        "repetition_penalty": 1.1,
        "temperature": 0,
        "top_p": 1,
    }
    for key, value in expected_policy.items():
        if policy.get(key) != value:
            raise CohortContractError(f"RP1.10 policy {key} must be {value!r}")
    components = census._model_components(
        _require_mapping(manifest.get("model_identity"), "RP1.10 model_identity"),
        str(manifest_path),
    )
    if components != recorded.get("model_components"):
        raise CohortContractError("RP1.10 checkpoint identity differs from Task0")
    artifacts = _require_mapping(manifest.get("artifacts"), "RP1.10 artifacts")
    scored_name = artifacts.get("gt_vs_pred_scored")
    if scored_name != "gt_vs_pred_scored.jsonl":
        raise CohortContractError("RP1.10 manifest does not own gt_vs_pred_scored.jsonl")
    scored_path = (root / scored_name).resolve(strict=True)
    rows = _read_jsonl(scored_path)
    by_image: dict[str, Mapping[str, Any]] = {}
    predictions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = census._canonical_image_id(row.get("example_id"))
        if image_id not in validated["owners_by_image"] or image_id in by_image:
            raise CohortContractError(f"RP1.10 has duplicate or foreign image {image_id}")
        by_image[image_id] = row
        raw_predictions = _require_list(row.get("pred"), f"RP1.10 image {image_id}.pred")
        seen_indices: set[int] = set()
        for list_index, raw in enumerate(raw_predictions):
            pred = _require_mapping(raw, f"RP1.10 image {image_id} pred[{list_index}]")
            row_index = pred.get("generated_order")
            if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index in seen_indices:
                raise CohortContractError(f"RP1.10 image {image_id} has renumbered predictions")
            seen_indices.add(row_index)
            description = census._normalize_description(pred.get("description"))
            if not description:
                raise CohortContractError(f"RP1.10 image {image_id} row {row_index} lacks description")
            box = census._pixel_box(
                pred.get("bbox"), context=f"RP1.10 image {image_id} row {row_index}.bbox"
            )
            predictions_by_image[image_id].append(
                {
                    "pred_row_id": f"pred:sorted:greedy-rp1.10:0:{image_id}:{row_index}",
                    "image_id": image_id,
                    "original_row_index": row_index,
                    "normalized_description": description,
                    "bbox_xyxy": box,
                }
            )
    if set(by_image) != set(validated["owners_by_image"]):
        raise CohortContractError("RP1.10 scored rows do not exactly cover the Task0 panel")

    result: dict[str, dict[str, Any]] = {}
    for image_id, owners in validated["owners_by_image"].items():
        predictions = predictions_by_image[image_id]
        matches = census._min_cost_max_cardinality_assignment(predictions, owners, aliases)
        strict_by_owner: dict[str, list[str]] = defaultdict(list)
        for match in matches:
            if match["strict_match_status"] == "matched":
                strict_by_owner[str(match["gt_owner_id"])].append(str(match["pred_row_id"]))
        for owner in owners:
            owner_id = str(owner["gt_owner_id"])
            spatial = _natural_support(
                owner,
                predictions,
                aliases,
                thresholds=(
                    DEFAULT_LOOSE_IOU_LOWER,
                    DEFAULT_MEANINGFUL_LOOSE_IOU,
                    DEFAULT_LOOSE_IOU_UPPER,
                ),
            )
            strict_ids = sorted(strict_by_owner.get(owner_id, []))
            result[owner_id] = {
                "status": _support_status(bool(strict_ids), spatial),
                "supporting_pred_row_ids": strict_ids or spatial["supporting_pred_row_ids"],
                "max_semantic_compatible_iou": spatial["max_semantic_compatible_iou"],
            }
    return result, {
        "run_manifest": manifest_digest,
        "gt_vs_pred_scored": _sha256_file(scored_path),
    }


def _load_registered_support(
    paths: Sequence[Path],
    validated: Mapping[str, Any],
    aliases: Mapping[str, set[str]],
    *,
    thresholds: tuple[float, float, float],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_refs: list[dict[str, str]] = []
    owner_ids = set(validated["owner_by_id"])
    registration_ids: set[str] = set()
    for path in paths:
        source = path.resolve(strict=True)
        rows = _read_jsonl(source)
        if not rows:
            raise CohortContractError(f"registered sampling support is empty: {source}")
        seen_owners: set[str] = set()
        registration_id: str | None = None
        policy_receipt: Mapping[str, Any] | None = None
        policy_receipt_path: Path | None = None
        prediction_source_path: Path | None = None
        prediction_source_digest: str | None = None
        for row in rows:
            if row.get("schema_version") != REGISTERED_SUPPORT_SCHEMA_VERSION:
                raise CohortContractError(f"registered support schema changed: {source}")
            current_registration = _require_string(
                row.get("registration_id"), "registered support registration_id"
            )
            if registration_id is None:
                registration_id = current_registration
            elif registration_id != current_registration:
                raise CohortContractError(f"registered support file mixes registrations: {source}")
            receipt_ref = _require_mapping(
                row.get("policy_receipt"), "registered support policy_receipt"
            )
            current_receipt_path = Path(
                _require_string(receipt_ref.get("path"), "registered policy receipt path")
            ).resolve(strict=True)
            if _sha256_file(current_receipt_path) != receipt_ref.get("sha256"):
                raise CohortContractError("registered sampling policy receipt digest is stale")
            if policy_receipt_path is None:
                policy_receipt_path = current_receipt_path
                policy_receipt = _read_json(current_receipt_path)
            elif policy_receipt_path != current_receipt_path:
                raise CohortContractError("registered support file mixes policy receipts")
            owner_id = _require_string(row.get("gt_owner_id"), "registered support gt_owner_id")
            if owner_id not in owner_ids or owner_id in seen_owners:
                raise CohortContractError(f"registered support has duplicate/foreign owner {owner_id}")
            seen_owners.add(owner_id)
            status = row.get("support_status")
            if status not in SUPPORT_STATUSES[:3]:
                raise CohortContractError(f"registered support has invalid status {status!r}")
            artifact = _require_mapping(row.get("source_artifact"), "registered source_artifact")
            artifact_path = Path(_require_string(artifact.get("path"), "registered artifact path")).resolve(
                strict=True
            )
            artifact_digest = _require_sha256(
                artifact.get("sha256"), "registered source artifact sha256"
            )
            if _sha256_file(artifact_path) != artifact_digest:
                raise CohortContractError(f"registered support has stale artifact digest: {artifact_path}")
            if prediction_source_path is None:
                prediction_source_path = artifact_path
                prediction_source_digest = artifact_digest
            elif prediction_source_path != artifact_path or prediction_source_digest != artifact_digest:
                raise CohortContractError("registered support file mixes source artifacts")
            normalized = dict(row)
            by_owner[owner_id].append(normalized)
        if seen_owners != owner_ids:
            raise CohortContractError(f"registered support must explicitly cover every owner: {source}")
        assert registration_id is not None and policy_receipt is not None
        assert policy_receipt_path is not None and prediction_source_path is not None
        assert prediction_source_digest is not None
        if registration_id in registration_ids:
            raise CohortContractError(f"duplicate registration_id: {registration_id}")
        registration_ids.add(registration_id)
        if policy_receipt.get("schema_version") != REGISTERED_POLICY_RECEIPT_SCHEMA_VERSION:
            raise CohortContractError("unsupported registered sampling policy receipt schema")
        if policy_receipt.get("status") != "frozen_before_generation_and_scoring":
            raise CohortContractError("registered sampling policy is not frozen")
        if policy_receipt.get("registration_id") != registration_id:
            raise CohortContractError("registered sampling policy registration_id disagrees")
        policy_digest = _require_sha256(
            policy_receipt.get("receipt_sha256"), "registered policy receipt_sha256"
        )
        if _receipt_digest(policy_receipt) != policy_digest:
            raise CohortContractError("registered sampling policy receipt logical digest is stale")
        policy = _require_mapping(policy_receipt.get("policy"), "registered policy")
        if policy.get("decode_mode") != "sampled":
            raise CohortContractError("registered sampling policy decode_mode must be sampled")
        repetition_penalty = policy.get("repetition_penalty")
        if not (_close(repetition_penalty, 1.0) or _close(repetition_penalty, 1.1)):
            raise CohortContractError("registered sampling policy has an unregistered repetition penalty")
        if policy.get("max_new_tokens") != EXPECTED_HORIZON:
            raise CohortContractError("registered sampling policy horizon changed")
        seeds = _require_list(policy.get("seeds"), "registered policy seeds")
        if (
            not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise CohortContractError("registered sampling policy seeds must be unique integers")
        temperature = policy.get("temperature")
        top_p = policy.get("top_p")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or float(temperature) <= 0
            or not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not 0 < float(top_p) <= 1
        ):
            raise CohortContractError("registered sampling policy temperature/top_p are invalid")
        _require_string(policy.get("donor_selection_rule"), "registered donor_selection_rule")
        receipt_source = _require_mapping(
            policy_receipt.get("source_artifact"), "registered policy source_artifact"
        )
        if (
            Path(str(receipt_source.get("path"))).resolve(strict=True) != prediction_source_path
            or receipt_source.get("sha256") != prediction_source_digest
        ):
            raise CohortContractError("registered policy and support source-artifact bindings disagree")

        source_predictions = _read_jsonl(prediction_source_path)
        pred_by_id: dict[str, dict[str, Any]] = {}
        predictions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in source_predictions:
            if prediction.get("schema_version") != REGISTERED_PREDICTION_SCHEMA_VERSION:
                raise CohortContractError("registered prediction source schema changed")
            if prediction.get("registration_id") != registration_id:
                raise CohortContractError("registered prediction has a foreign registration_id")
            pred_id = _require_string(
                prediction.get("pred_row_id"), "registered prediction pred_row_id"
            )
            image_id = _require_string(
                prediction.get("image_id"), f"registered prediction {pred_id}.image_id"
            )
            if image_id not in validated["owners_by_image"] or pred_id in pred_by_id:
                raise CohortContractError(f"registered prediction is duplicate/foreign: {pred_id}")
            if prediction.get("decode_mode") != "sampled" or prediction.get("seed") not in seeds:
                raise CohortContractError(f"registered prediction policy drifted: {pred_id}")
            row_index = prediction.get("original_row_index")
            if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
                raise CohortContractError(f"registered prediction has invalid row index: {pred_id}")
            normalized_description = census._normalize_description(
                prediction.get("normalized_description")
            )
            if normalized_description != prediction.get("normalized_description"):
                raise CohortContractError(f"registered prediction description is not normalized: {pred_id}")
            box = census._pixel_box(
                prediction.get("bbox_xyxy"),
                context=f"registered prediction {pred_id}.bbox_xyxy",
            )
            normalized_prediction = {
                **prediction,
                "row_kind": "complete_prediction",
                "bbox_xyxy": box,
                "normalized_description": normalized_description,
            }
            pred_by_id[pred_id] = normalized_prediction
            predictions_by_image[image_id].append(normalized_prediction)

        strict_owner_by_pred: dict[str, str] = {}
        neutral_registered_pred_ids: set[str] = set()
        for image_id, owners in validated["owners_by_image"].items():
            analysis = census._global_assignment_analysis(
                predictions_by_image.get(image_id, []), owners, aliases
            )
            neutral_registered_pred_ids.update(analysis["ambiguous_pred_row_ids"])
            for match in analysis["matches"]:
                if match["strict_match_status"] == "matched":
                    strict_owner_by_pred[str(match["pred_row_id"])] = str(match["gt_owner_id"])

        rows_by_owner = {
            str(row["gt_owner_id"]): row
            for row in rows
        }
        for owner_id, support_row in rows_by_owner.items():
            owner = validated["owner_by_id"][owner_id]
            support_status = str(support_row["support_status"])
            supporting_ids = _require_list(
                support_row.get("supporting_pred_row_ids"),
                f"registered support {owner_id}.supporting_pred_row_ids",
            )
            if any(not isinstance(item, str) or not item for item in supporting_ids) or len(
                set(supporting_ids)
            ) != len(supporting_ids):
                raise CohortContractError(f"registered support {owner_id} has invalid row IDs")
            if support_status in {"strict_positive", "loose_positive_overlap"} and not supporting_ids:
                raise CohortContractError("registered positive support requires nonempty evidence")
            if support_status == "null_no_positive_evidence" and supporting_ids:
                raise CohortContractError("registered null support must not carry positive evidence")
            for pred_id in supporting_ids:
                prediction = pred_by_id.get(pred_id)
                if prediction is None:
                    raise CohortContractError(f"registered support references unknown row {pred_id}")
                if prediction["image_id"] != owner["image_id"] or not _compatible(
                    prediction, owner, aliases
                ):
                    raise CohortContractError(
                        f"registered support row {pred_id} has the wrong owner/semantic relation"
                    )
                metrics = _box_metrics(prediction["bbox_xyxy"], owner["bbox_xyxy"])
                if metrics["intersection_area"] <= 0:
                    raise CohortContractError(
                        f"registered support row {pred_id} has no positive spatial relation"
                    )
                if support_status == "strict_positive" and (
                    strict_owner_by_pred.get(pred_id) != owner_id
                    or pred_id in neutral_registered_pred_ids
                ):
                    raise CohortContractError(
                        f"registered strict support row {pred_id} is not a committed strict owner match"
                    )
            spatial = _natural_support(
                owner,
                predictions_by_image.get(str(owner["image_id"]), []),
                aliases,
                thresholds=thresholds,
            )
            declared_max = support_row.get("max_semantic_compatible_iou")
            if not _close(declared_max, spatial["max_semantic_compatible_iou"]):
                raise CohortContractError(
                    f"registered support max IoU drifted for {owner_id}"
                )
            stable_meaningful = spatial["meaningful_loose_iou_diagnostic"][
                "stable_meaningful_positive_across_threshold_band"
            ]
            reviewed = support_row.get("review_status") == "lead_reviewed"
            actual_strict = any(
                strict_owner_by_pred.get(pred_id) == owner_id
                for pred_id in pred_by_id
            )
            actual_loose = spatial["any_positive_overlap"] and (stable_meaningful or reviewed)
            if support_status == "strict_positive" and not actual_strict:
                raise CohortContractError(f"registered strict-positive claim is false for {owner_id}")
            if support_status == "loose_positive_overlap" and (actual_strict or not actual_loose):
                raise CohortContractError(f"registered loose-positive claim is false for {owner_id}")
            if support_status == "null_no_positive_evidence" and (actual_strict or actual_loose):
                raise CohortContractError(f"registered null claim hides positive evidence for {owner_id}")

        source_refs.append(
            {
                "path": str(source),
                "sha256": _sha256_file(source),
                "policy_receipt_path": str(policy_receipt_path),
                "policy_receipt_sha256": _sha256_file(policy_receipt_path),
                "source_artifact_path": str(prediction_source_path),
                "source_artifact_sha256": prediction_source_digest,
            }
        )
    return dict(by_owner), source_refs


def _manual_review_queue(
    validated: Mapping[str, Any], aliases: Mapping[str, set[str]], source_digests: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in sorted(
        validated["prediction_by_id"].values(), key=lambda item: str(item["pred_row_id"])
    ):
        if prediction.get("strict_match_status") == "matched":
            continue
        ambiguity_neutral = str(prediction["pred_row_id"]) in validated[
            "neutral_pred_row_ids"
        ]
        compatible: list[dict[str, Any]] = []
        max_any_iou: float | None = None
        if prediction.get("row_kind") == "complete_prediction":
            max_any_iou = 0.0
            for owner in validated["owners_by_image"][prediction["image_id"]]:
                metrics = _box_metrics(prediction["bbox_xyxy"], owner["bbox_xyxy"])
                max_any_iou = max(max_any_iou, metrics["intersection_over_union"])
                if _compatible(prediction, owner, aliases):
                    compatible.append(
                        {
                            "gt_owner_id": owner["gt_owner_id"],
                            "decision_eligible": owner["gt_owner_id"]
                            in validated["eligible_owner_ids"],
                            "intersection_over_union": metrics["intersection_over_union"],
                            "intersection_over_ground_truth": metrics[
                                "intersection_over_ground_truth"
                            ],
                        }
                    )
        compatible.sort(
            key=lambda item: (-float(item["intersection_over_union"]), str(item["gt_owner_id"]))
        )
        rows.append(
            {
                "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
                "pred_row_id": prediction["pred_row_id"],
                "image_id": prediction["image_id"],
                "trajectory_id": prediction["trajectory_id"],
                "decode_mode": prediction["decode_mode"],
                "seed": prediction["seed"],
                "original_row_index": prediction["original_row_index"],
                "row_kind": prediction["row_kind"],
                "description": prediction.get("description"),
                "normalized_description": prediction.get("normalized_description"),
                "bbox_xyxy": prediction.get("bbox_xyxy"),
                "strict_match_status": prediction["strict_match_status"],
                "ambiguity_receipt_ids": sorted(
                    str(item) for item in prediction.get("ambiguity_receipt_ids", [])
                ),
                "decision_eligibility": {
                    "status": (
                        "excluded_global_strict_ambiguity"
                        if ambiguity_neutral
                        else "eligible_for_manual_adjudication"
                    ),
                    "included_in_b2_repair_or_calibration": not ambiguity_neutral,
                    "raw_diagnostics_preserved": True,
                },
                "max_any_owner_iou": max_any_iou,
                "compatible_owners": compatible,
                "physical_axis_status": "unresolved",
                "physical_relation": None,
                "semantic_axis_status": "unresolved",
                "semantic_relation": None,
                "adjudication_policy": "manual_only_no_automatic_adjudication",
                "foreign_keys": {
                    "pred_row_id": prediction["pred_row_id"],
                    "trajectory_id": prediction["trajectory_id"],
                    "compatible_gt_owner_ids": [item["gt_owner_id"] for item in compatible],
                },
                "source_digests": dict(source_digests),
                "null_status": {
                    "physical_relation": "unresolved_pending_manual_review",
                    "semantic_relation": "unresolved_pending_manual_review",
                    "max_any_owner_iou": (
                        "present" if max_any_iou is not None else "not_evaluable_non_complete_row"
                    ),
                },
            }
        )
    return rows


def build_cohorts(
    *,
    census_dir: Path,
    sentinel_registry_path: Path,
    control_registry_path: Path,
    rp110_run_dir: Path | None = None,
    registered_support_paths: Sequence[Path] = (),
    meaningful_loose_iou: float = DEFAULT_MEANINGFUL_LOOSE_IOU,
    loose_iou_lower: float = DEFAULT_LOOSE_IOU_LOWER,
    loose_iou_upper: float = DEFAULT_LOOSE_IOU_UPPER,
    require_full_panel: bool = True,
) -> dict[str, Any]:
    """Build serializable Task-2 rows without writing files."""

    thresholds = (loose_iou_lower, meaningful_loose_iou, loose_iou_upper)
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in thresholds):
        raise CohortContractError("loose-support thresholds must be numeric")
    if not 0.0 <= loose_iou_lower <= meaningful_loose_iou <= loose_iou_upper <= 1.0:
        raise CohortContractError("loose-support thresholds must satisfy 0 <= lower <= base <= upper <= 1")

    bundle = _load_census(census_dir)
    validated = _validate_ledgers(bundle, require_full_panel=require_full_panel)
    control_registry, control_roles = _validate_control_registry(
        control_registry_path, bundle, validated, require_full_panel=require_full_panel
    )
    sentinel_registry, sentinels = _validate_sentinel_registry(
        sentinel_registry_path, bundle, validated
    )
    rp110, rp110_digests = _load_rp110_support(
        rp110_run_dir, bundle, validated, bundle["aliases"]
    )
    registered, registered_refs = _load_registered_support(
        registered_support_paths,
        validated,
        bundle["aliases"],
        thresholds=thresholds,
    )

    input_digests = {
        "task0_artifact_manifest": _sha256_file(bundle["manifest_path"]),
        "task0_execution_receipt_file": _sha256_file(bundle["paths"]["execution_receipt"]),
        "task0_execution_receipt_logical": bundle["execution_receipt_sha256"],
        "matcher_contract": _sha256_file(bundle["paths"]["matcher"]),
        "owner_ledger": _sha256_file(bundle["paths"]["owners"]),
        "prediction_row_ledger": _sha256_file(bundle["paths"]["predictions"]),
        "owner_trajectory_matrix": _sha256_file(bundle["paths"]["matrix"]),
        "sentinel_registry": _sha256_file(sentinel_registry_path.resolve(strict=True)),
        "sentinel_selection_receipt": _require_sha256(
            _require_mapping(
                sentinel_registry.get("selection_receipt"),
                "sentinel registry selection_receipt",
            ).get("sha256"),
            "sentinel selection receipt sha256",
        ),
        "sentinel_selection_confirmation_receipt": _require_sha256(
            _require_mapping(
                sentinel_registry.get("selection_confirmation_receipt"),
                "sentinel registry selection_confirmation_receipt",
            ).get("sha256"),
            "sentinel selection confirmation receipt sha256",
        ),
        "control_registry": _sha256_file(control_registry_path.resolve(strict=True)),
    }
    all_predictions = list(validated["prediction_by_id"].values())
    admitted_predictions = [
        prediction
        for prediction in all_predictions
        if prediction["pred_row_id"] not in validated["neutral_pred_row_ids"]
    ]
    cohort_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for owner_id, owner in sorted(validated["owner_by_id"].items()):
        image_id = str(owner["image_id"])
        primary_eligible = owner_id in validated["eligible_owner_ids"]
        greedy_strict = owner_id in validated["greedy_strict_owner_ids"]
        sampled_matrix = [
            validated["matrix_by_key"][
                (owner_id, f"trajectory:sorted:rp1.00:sampled:{seed}:{image_id}")
            ]
            for seed in EXPECTED_SAMPLED_SEEDS
        ]
        sampled_strict = owner_id in validated["sampled_strict_owner_ids"]
        raw_natural = _natural_support(
            owner, all_predictions, bundle["aliases"], thresholds=thresholds
        )
        admitted_natural = _natural_support(
            owner, admitted_predictions, bundle["aliases"], thresholds=thresholds
        )
        reviewed_b1 = any(role.endswith(":b1_loose_only") for role in control_roles.get(owner_id, []))
        stable_meaningful_loose = admitted_natural["meaningful_loose_iou_diagnostic"][
            "stable_meaningful_positive_across_threshold_band"
        ]
        neutral_reason = None
        if not primary_eligible:
            cohort = "strict_ambiguity_neutral"
            neutral_reason = "global_strict_assignment_ambiguity"
        elif greedy_strict:
            cohort = "greedy_strict_present"
        elif sampled_strict:
            cohort = "strict_rescued"
        elif admitted_natural["any_positive_overlap"] and (
            reviewed_b1 or stable_meaningful_loose
        ):
            cohort = "loose_only_b1"
        elif raw_natural["any_positive_overlap"]:
            cohort = "positive_overlap_neutral"
            neutral_reason = (
                "only_neutral_prediction_overlap"
                if not admitted_natural["any_positive_overlap"]
                else "subthreshold_or_threshold_band_unstable_positive_overlap"
            )
        else:
            cohort = "no_free_spatial_support"

        positive_overlap_neutral = cohort == "positive_overlap_neutral"
        b1_no_free_eligible = primary_eligible and not positive_overlap_neutral

        sentinel = sentinels.get(owner_id)
        if sentinel is not None and cohort != "no_free_spatial_support":
            raise CohortContractError(
                f"sentinel registry contradicts exact natural support for {owner_id}"
            )
        registered_rows = registered.get(owner_id, [])
        registered_positive = [
            row
            for row in registered_rows
            if row["support_status"] in {"strict_positive", "loose_positive_overlap"}
        ]
        if registered_rows:
            registered_status = (
                "strict_positive"
                if any(row["support_status"] == "strict_positive" for row in registered_rows)
                else (
                    "loose_positive_overlap"
                    if registered_positive
                    else "null_no_positive_evidence"
                )
            )
        else:
            registered_status = "not_supplied"
        rp110_status = rp110[owner_id]["status"]
        blocks_later = rp110_status in {
            "strict_positive",
            "loose_positive_overlap",
        } or registered_status in {"strict_positive", "loose_positive_overlap"}

        cohort_rows.append(
            {
                "schema_version": COHORT_ASSIGNMENT_SCHEMA_VERSION,
                "gt_owner_id": owner_id,
                "diagnostic_owner_id": owner["diagnostic_owner_id"],
                "image_id": image_id,
                "cohort": cohort,
                "primary_eligibility": {
                    "status": (
                        "excluded_global_strict_ambiguity"
                        if not primary_eligible
                        else (
                            "excluded_unreviewed_or_subthreshold_positive_overlap"
                            if positive_overlap_neutral
                            else "eligible"
                        )
                    ),
                    "included_in_greedy_tp_fn": primary_eligible,
                    "included_in_strict_rescued": primary_eligible,
                    "included_in_b1_or_no_free": b1_no_free_eligible,
                    "included_in_repair_or_calibration": b1_no_free_eligible,
                    "neutral_reason": neutral_reason,
                },
                "primary_strict_support": {
                    "rp1_00_greedy": greedy_strict,
                    "rp1_00_k16_any": sampled_strict,
                    "strict_rescued": not greedy_strict and sampled_strict,
                    "global_ambiguity_present": not primary_eligible,
                },
                "natural_spatial_support": {
                    **raw_natural,
                    "admitted_non_neutral_support": admitted_natural,
                    "reviewed_b1_control": reviewed_b1,
                    "admits_loose_only_b1": cohort == "loose_only_b1",
                    "positive_overlap_neutral": cohort == "positive_overlap_neutral",
                },
                "auxiliary_positive_evidence": {
                    "rp1_10_greedy_status": rp110_status,
                    "registered_sampling_any_status": registered_status,
                    "blocks_later_absence_claim": blocks_later,
                    "null_semantics": "null_is_no_positive_evidence_not_absence_evidence",
                    "excluded_from_primary_k16_denominator": True,
                },
                "sentinel": {
                    "is_sentinel": sentinel is not None,
                    "sentinel_id": sentinel.get("sentinel_id") if sentinel else None,
                    "sentinel_kind": sentinel.get("sentinel_kind") if sentinel else None,
                    "outcome_selected_no_prevalence": sentinel is not None,
                },
                "control_roles": control_roles.get(owner_id, []),
                "foreign_keys": {
                    "gt_owner_id": owner_id,
                    "diagnostic_owner_id": owner["diagnostic_owner_id"],
                    "supporting_pred_row_ids": raw_natural["supporting_pred_row_ids"],
                    "sentinel_id": sentinel.get("sentinel_id") if sentinel else None,
                },
                "source_digests": dict(input_digests),
                "null_status": {
                    "sentinel_id": "present" if sentinel else "not_a_sentinel",
                    "rp1_10_greedy": (
                        "evaluated" if rp110_status != "not_supplied" else "not_supplied"
                    ),
                    "registered_sampling": (
                        "evaluated" if registered_rows else "not_supplied"
                    ),
                    "downstream_absence_adjudication": "not_assigned_by_task2",
                },
            }
        )

        sampled_strict_ids = sorted(
            {
                pred_id
                for row in sampled_matrix
                if row["strict_match_presence"]
                for pred_id in row["matched_pred_row_ids"]
            }
        )
        sampled_natural_predictions = [
            prediction
            for prediction in all_predictions
            if prediction.get("image_id") == image_id
            and prediction.get("decode_mode") == "sampled"
        ]
        sampled_spatial = _natural_support(
            owner, sampled_natural_predictions, bundle["aliases"], thresholds=thresholds
        )
        primary_sampling_status = (
            _support_status(bool(sampled_strict_ids), sampled_spatial)
            if primary_eligible
            else "neutral_excluded_global_strict_ambiguity"
        )
        sampling_rows.append(
            _sampling_row(
                owner=owner,
                support_panel="primary_rp1_00_k16",
                policy_stratum=PRIMARY_POLICY_STRATUM,
                decode_mode="sampled",
                registration_id="fixed-seeds-21001-through-21016",
                support_status=primary_sampling_status,
                supporting_pred_row_ids=sampled_strict_ids
                or sampled_spatial["supporting_pred_row_ids"],
                max_iou=sampled_spatial["max_semantic_compatible_iou"],
                input_digests=input_digests,
            )
        )
        sampling_rows.append(
            _sampling_row(
                owner=owner,
                support_panel="auxiliary_rp1_10_greedy",
                policy_stratum=RP110_POLICY_STRATUM,
                decode_mode="greedy",
                registration_id="task0-registered-production-greedy",
                support_status=rp110_status,
                supporting_pred_row_ids=rp110[owner_id]["supporting_pred_row_ids"],
                max_iou=rp110[owner_id]["max_semantic_compatible_iou"],
                input_digests=input_digests,
            )
        )
        if registered_rows:
            for row in registered_rows:
                sampling_rows.append(
                    _sampling_row(
                        owner=owner,
                        support_panel="registered_sampling",
                        policy_stratum=str(row.get("policy_stratum")),
                        decode_mode=str(row.get("decode_mode", "sampled")),
                        registration_id=str(row["registration_id"]),
                        support_status=str(row["support_status"]),
                        supporting_pred_row_ids=list(row.get("supporting_pred_row_ids", [])),
                        max_iou=row.get("max_semantic_compatible_iou"),
                        input_digests=input_digests,
                    )
                )
        else:
            sampling_rows.append(
                _sampling_row(
                    owner=owner,
                    support_panel="registered_sampling",
                    policy_stratum="not_supplied",
                    decode_mode="sampled",
                    registration_id="not_supplied",
                    support_status="not_supplied",
                    supporting_pred_row_ids=[],
                    max_iou=None,
                    input_digests=input_digests,
                )
            )

    manual_rows = _manual_review_queue(validated, bundle["aliases"], input_digests)
    return {
        "cohort_assignments": cohort_rows,
        "sampling_support": sampling_rows,
        "manual_review_queue": manual_rows,
        "input_digests": input_digests,
        "rp110_digests": rp110_digests,
        "registered_support_sources": registered_refs,
        "control_registry": control_registry,
        "sentinel_registry": sentinel_registry,
        "thresholds": {
            "metric": "intersection_over_union",
            "lower": loose_iou_lower,
            "predeclared": meaningful_loose_iou,
            "upper": loose_iou_upper,
            "role": "diagnostic_only",
            "exact_no_free_predicate": "no semantic-compatible primary natural bbox has intersection_area > 0",
        },
    }


def _sampling_row(
    *,
    owner: Mapping[str, Any],
    support_panel: str,
    policy_stratum: str,
    decode_mode: str,
    registration_id: str,
    support_status: str,
    supporting_pred_row_ids: list[str],
    max_iou: Any,
    input_digests: Mapping[str, str],
) -> dict[str, Any]:
    if support_status not in SUPPORT_STATUSES:
        raise CohortContractError(f"invalid sampling support status {support_status!r}")
    positive = support_status in {"strict_positive", "loose_positive_overlap"}
    evidence_semantics = (
        "positive_blocks_later_absence_claim"
        if positive
        else (
            "neutral_ambiguity_is_excluded_from_decision_denominators"
            if support_status == "neutral_excluded_global_strict_ambiguity"
            else "null_is_no_positive_evidence_not_absence_evidence"
        )
    )
    return {
        "schema_version": SAMPLING_SUPPORT_SCHEMA_VERSION,
        "support_panel": support_panel,
        "gt_owner_id": owner["gt_owner_id"],
        "image_id": owner["image_id"],
        "policy_stratum": policy_stratum,
        "decode_mode": decode_mode,
        "registration_id": registration_id,
        "support_status": support_status,
        "supporting_pred_row_ids": sorted(supporting_pred_row_ids),
        "max_semantic_compatible_iou": max_iou,
        "evidence_semantics": evidence_semantics,
        "excluded_from_primary_k16_denominator": support_panel != "primary_rp1_00_k16",
        "foreign_keys": {
            "gt_owner_id": owner["gt_owner_id"],
            "supporting_pred_row_ids": sorted(supporting_pred_row_ids),
        },
        "source_digests": dict(input_digests),
        "null_status": {
            "supporting_pred_row_ids": "present" if supporting_pred_row_ids else "empty",
            "max_semantic_compatible_iou": "present" if max_iou is not None else "not_supplied",
        },
    }


def write_cohorts(
    output_dir: Path, payload: Mapping[str, Any], *, command_line: Sequence[str]
) -> dict[str, Path]:
    parent = output_dir.expanduser().parent.resolve(strict=True)
    destination = parent / output_dir.name
    try:
        destination.mkdir(mode=0o755, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise CohortContractError(
            f"refusing to use existing output directory: {destination}"
        ) from exc
    paths = {
        "cohort_assignments": destination / "cohort-assignments.jsonl",
        "sampling_support": destination / "sampling-support.jsonl",
        "manual_review_queue": destination / "manual-review-queue.jsonl",
        "cohort_receipt": destination / "cohort-receipt.json",
        "artifact_manifest": destination / "artifact-manifest.json",
    }
    counts = defaultdict(int)
    for row in payload["cohort_assignments"]:
        counts[row["cohort"]] += 1
    argv = [str(item) for item in command_line]
    if not argv:
        raise CohortContractError("execution receipt requires a nonempty command line")
    implementation_path = Path(__file__).resolve()
    test_path = implementation_path.parents[2] / "tests/research/test_build_sorted_owner_basin_cohorts.py"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "execution_status": "completed_cpu_only",
        "command": {
            "argv": argv,
            "shell_escaped": shlex.join(argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "repository": _repository_state(),
        "runtime": _cpu_runtime_identity(),
        "implementation": {
            "path": str(implementation_path),
            "sha256": _sha256_file(implementation_path),
            "test_path": str(test_path),
            "test_sha256": _sha256_file(test_path),
        },
        "input_digests": payload["input_digests"],
        "rp1_10_auxiliary_digests": payload["rp110_digests"],
        "registered_support_sources": payload["registered_support_sources"],
        "thresholds": payload["thresholds"],
        "row_counts": {
            "cohort_assignments": len(payload["cohort_assignments"]),
            "sampling_support": len(payload["sampling_support"]),
            "manual_review_queue": len(payload["manual_review_queue"]),
            "cohorts": dict(sorted(counts.items())),
        },
        "schemas": {
            "cohort_assignments": COHORT_ASSIGNMENT_SCHEMA_VERSION,
            "sampling_support": SAMPLING_SUPPORT_SCHEMA_VERSION,
            "manual_review_queue": MANUAL_REVIEW_SCHEMA_VERSION,
        },
        "status_enums": {
            "cohort": list(COHORT_STATUSES),
            "support_status": list(SUPPORT_STATUSES),
            "manual_physical_axis_status": ["unresolved"],
            "manual_semantic_axis_status": ["unresolved"],
        },
        "foreign_key_contracts": {
            "cohort.gt_owner_id": "owner-ledger.jsonl.gt_owner_id; one-to-one total",
            "cohort.supporting_pred_row_ids": "prediction-row-ledger.jsonl.pred_row_id",
            "manual.pred_row_id": "prediction-row-ledger.jsonl.pred_row_id; every non-matched natural row exactly once",
            "sentinel.gt_owner_id": "sentinel-registry.json.sentinels[].gt_owner_id",
        },
        "unresolved_neutrality": "manual-review rows remain unresolved and cannot decide collision, downstream absence, or repair gain",
        "claim_boundary": "Task2 assigns no downstream high-confidence absence label and no prevalence estimate",
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    receipt_digest = receipt["receipt_sha256"]
    for family in ("cohort_assignments", "sampling_support", "manual_review_queue"):
        for row in payload[family]:
            row["execution_receipt_sha256"] = receipt_digest
            row["source_digests"] = {
                **row["source_digests"],
                "task2_execution_receipt": receipt_digest,
            }

    _write_jsonl(paths["cohort_assignments"], payload["cohort_assignments"])
    _write_jsonl(paths["sampling_support"], payload["sampling_support"])
    _write_jsonl(paths["manual_review_queue"], payload["manual_review_queue"])
    _write_json(paths["cohort_receipt"], receipt)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "execution_receipt_sha256": receipt_digest,
        "artifacts": {
            name: {"path": path.name, "sha256": _sha256_file(path)}
            for name, path in paths.items()
            if name != "artifact_manifest"
        },
        "sources": {
            **payload["input_digests"],
            "rp1_10_auxiliary": payload["rp110_digests"],
            "registered_sampling": payload["registered_support_sources"],
        },
        "primary_denominator_policy": "RP1.0 natural greedy plus fixed K16 seeds 21001..21016 only",
        "auxiliary_positive_evidence_policy": "RP1.10 and registered sampling remain separate; positive blocks later absence, null is non-evidence",
        "forced_continuation_absence": "inherited_and_reverified_from_sealed_task0_ledgers",
    }
    _write_json(paths["artifact_manifest"], manifest)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, default=DEFAULT_CENSUS_DIR)
    parser.add_argument(
        "--sentinel-registry",
        type=Path,
        default=DEFAULT_REGISTRY_DIR / "sentinel-registry.json",
    )
    parser.add_argument(
        "--control-registry",
        type=Path,
        default=DEFAULT_REGISTRY_DIR / "control-registry.json",
    )
    parser.add_argument(
        "--rp110-run-dir",
        type=Path,
        default=DEFAULT_RP110_RUN_DIR,
        help="exact Task0-registered human-refined12 RP=1.10 production run root",
    )
    parser.add_argument(
        "--omit-rp110",
        action="store_true",
        help="emit RP=1.10 as not_supplied (never as negative evidence)",
    )
    parser.add_argument(
        "--registered-sampling-support",
        type=Path,
        action="append",
        default=[],
        help="complete per-owner registered sampling support JSONL; repeat per registration",
    )
    parser.add_argument("--meaningful-loose-iou", type=float, default=DEFAULT_MEANINGFUL_LOOSE_IOU)
    parser.add_argument("--loose-iou-lower", type=float, default=DEFAULT_LOOSE_IOU_LOWER)
    parser.add_argument("--loose-iou-upper", type=float, default=DEFAULT_LOOSE_IOU_UPPER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-fixture-panel",
        action="store_true",
        help="test-only: permit a panel other than the frozen 12-image/346-owner scope",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_cohorts(
        census_dir=args.census_dir,
        sentinel_registry_path=args.sentinel_registry,
        control_registry_path=args.control_registry,
        rp110_run_dir=None if args.omit_rp110 else args.rp110_run_dir,
        registered_support_paths=args.registered_sampling_support,
        meaningful_loose_iou=args.meaningful_loose_iou,
        loose_iou_lower=args.loose_iou_lower,
        loose_iou_upper=args.loose_iou_upper,
        require_full_panel=not args.allow_fixture_panel,
    )
    paths = write_cohorts(args.output_dir, payload, command_line=sys.argv)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
