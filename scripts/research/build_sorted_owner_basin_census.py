#!/usr/bin/env python3
"""Freeze the CPU-only natural-trajectory owner census for the Sorted study.

This is deliberately an experiment-local reader.  It does not run a model,
infer aliases, adjudicate unmatched rows, or make a landscape/cohort claim.
It validates the frozen RP=1.0 natural artifacts, writes permanent IDs, and
records the exact matching contract used for the per-trajectory any-hit owner
union.  The latter is *not* the historical cross-trajectory medoid-detection
union used by ``compute_sampled_union_f1_metrics.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
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

from src.common.errors import DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy, parse_source_bbox_tokens


SCHEMA_VERSION = "sorted-owner-basin-census.v2"
MATCHER_SCHEMA_VERSION = "sorted-owner-basin-matcher.v2"
OWNER_LEDGER_SCHEMA_VERSION = "sorted-owner-basin-owner-ledger.v2"
PREDICTION_LEDGER_SCHEMA_VERSION = "sorted-owner-basin-prediction-row-ledger.v2"
MATRIX_SCHEMA_VERSION = "sorted-owner-basin-owner-trajectory-matrix.v2"
NATIVE_REPLAY_SCHEMA_VERSION = "sorted-owner-basin-native-replay.v2"
AMBIGUITY_RECEIPT_SCHEMA_VERSION = "sorted-owner-basin-ambiguity-receipt.v2"
EXECUTION_RECEIPT_SCHEMA_VERSION = "sorted-owner-basin-task0-execution-receipt.v2"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"

EXPECTED_HORIZON = 3084
EXPECTED_GREEDY_SEED = 0
EXPECTED_SAMPLED_SEEDS = tuple(range(21001, 21017))
EXPECTED_PANEL_IMAGE_COUNT = 12
EXPECTED_PANEL_OWNER_COUNT = 346
IOU_THRESHOLD = 0.50
_EPSILON = 1e-12
_IMAGE_SUFFIX_RE = re.compile(r"(?:^|_)(\d+)$")
_FORCED_KEY_RE = re.compile(r"(?:^|[_-])(force(?:d)?|intervention)(?:[_-]|$)", re.IGNORECASE)
_UNIT_RELATIVE_ROOT = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-08-01-sorted-owner-basin-landscape-and-repair"
)

DEFAULT_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/"
    "evaluation-inputs/human-refined-12.coord.jsonl"
)
DEFAULT_GREEDY = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/greedy/greedy.json"
)
DEFAULT_SAMPLED_SHARDS = (
    Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-0.json"
    ),
    Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-1.json"
    ),
)

_SCHEMA_COMPATIBILITY = {
    SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-census.v1",
        "migration": "use v2 ambiguity-neutral and paired-owner metrics; v1 counts are diagnostic only",
    },
    MATCHER_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-matcher.v1",
        "migration": "enumerate the complete global optimum before the deterministic representative",
    },
    OWNER_LEDGER_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-owner-ledger.v1",
        "migration": "honor decision_eligibility and ambiguity_receipt_ids before owner-set comparisons",
    },
    PREDICTION_LEDGER_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-prediction-row-ledger.v1",
        "migration": "treat ambiguous_neutral rows as excluded rather than TP, FP, or FN evidence",
    },
    MATRIX_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-owner-trajectory-matrix.v1",
        "migration": "join ambiguity_receipt_ids and exclude neutral cells from decision denominators",
    },
    NATIVE_REPLAY_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-native-replay.v1",
        "migration": "validate the execution receipt binding before consuming replay status",
    },
    AMBIGUITY_RECEIPT_SCHEMA_VERSION: {
        "replaces_schema_version": None,
        "migration": "new v2 ledger; join by ambiguity_receipt_id",
    },
    EXECUTION_RECEIPT_SCHEMA_VERSION: {
        "replaces_schema_version": "sorted-owner-basin-task0-execution-receipt.v1",
        "migration": (
            "distinguish execution_receipt_content_sha256 from the serialized receipt artifact SHA"
        ),
    },
}


class CensusContractError(ValueError):
    """Raised before an incompatible source can enter the census."""


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _schema_compatibility(schema_version: str) -> dict[str, Any]:
    return dict(_SCHEMA_COMPATIBILITY[schema_version])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repository_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise CensusContractError(
            f"git {' '.join(args)} failed in {repository_root}: "
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
    root = (
        Path(
            _run_git(Path(__file__).resolve().parents[2], "rev-parse", "--show-toplevel")
            .decode("utf-8")
            .strip()
        )
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    head = _run_git(root, "rev-parse", "HEAD").decode("utf-8").strip()
    tracked_diff = _run_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    tracked_paths = [
        item.decode("utf-8")
        for item in _run_git(root, "diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
        if item
    ]
    untracked_paths = [
        item.decode("utf-8")
        for item in _run_git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if item and _is_relevant_untracked(item.decode("utf-8"))
    ]
    relevant_untracked = []
    for relative in sorted(untracked_paths):
        source = (root / relative).resolve(strict=True)
        if not source.is_file():
            raise CensusContractError(f"relevant untracked source is not a regular file: {source}")
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
        "head": head,
        "tracked_dirty_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_dirty_diff_bytes": len(tracked_diff),
        "tracked_dirty_paths": sorted(tracked_paths),
        "relevant_untracked_files": relevant_untracked,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_identity() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        torch_device_count = int(torch.cuda.device_count()) if torch_cuda_available else 0
        torch_devices = []
        for index in range(torch_device_count):
            properties = torch.cuda.get_device_properties(index)
            torch_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
        torch_identity: dict[str, Any] = {
            "version": str(torch.__version__),
            "compiled_cuda_version": torch.version.cuda,
            "cuda_available": torch_cuda_available,
            "device_count": torch_device_count,
            "devices": torch_devices,
        }
    except Exception as exc:  # pragma: no cover - depends on host runtime
        torch_identity = {
            "version": _package_version("torch"),
            "runtime_probe_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "execution_device": "cpu",
        "cuda_inventory_role": "observed_environment_metadata_not_used_for_census_execution",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "torch_runtime": torch_identity,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model,
        },
    }


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return _json_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "execution_receipt_content_sha256"
        }
    )


def _bound_file(role: str, source: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(source["path"])).resolve(strict=True)
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": str(source["sha256"]),
    }


def _build_execution_receipt(
    *,
    execution_argv: Sequence[str] | None,
    panel_digest: Mapping[str, Any],
    greedy: Mapping[str, Any],
    sampled: Sequence[Mapping[str, Any]],
    alias_source: Mapping[str, Any] | None,
    production: Mapping[str, Any] | None,
    model_file_hashes: Sequence[Mapping[str, Any]],
    matcher_semantics: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    process_argv = [str(Path(sys.executable).resolve()), *sys.argv]
    argv = [str(item) for item in execution_argv] if execution_argv is not None else process_argv
    bound_files = [
        _bound_file("frozen_panel", panel_digest),
        _bound_file("matched_rp_1_0_greedy", greedy),
        *[_bound_file(f"matched_rp_1_0_sampled_shard_{index}", item) for index, item in enumerate(sampled)],
    ]
    if alias_source is not None:
        bound_files.append(_bound_file("immutable_alias_table", alias_source))
    if production is not None:
        bound_files.append(_bound_file("production_rp_1_10_manifest", production))
    for item in model_file_hashes:
        bound_files.append(
            {
                "role": f"model_component:{item['component']}",
                "path": str(item["path"]),
                "bytes": int(item["bytes"]),
                "sha256": str(item["sha256"]),
            }
        )

    prompt_identities = [
        {
            "trajectory_id": str(trajectory["trajectory_id"]),
            "image_id": str(trajectory["image_id"]),
            "decode_mode": str(trajectory["decode_mode"]),
            "seed": int(trajectory["seed"]),
            "prompt_token_ids_sha256": str(trajectory["prompt_token_ids_sha256"]),
            "generated_token_ids_sha256": str(trajectory["generated_token_ids_sha256"]),
            "chat_text_sha256": str(trajectory["chat_text_sha256"]),
            "source_image_sha256": str(trajectory["source_image_sha256"]),
            "executed_media_sha256": str(trajectory["executed_media_sha256"]),
            "observed_image_grid_thw": trajectory["observed_image_grid_thw"],
        }
        for trajectory in trajectories
    ]
    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "schema_compatibility": _schema_compatibility(EXECUTION_RECEIPT_SCHEMA_VERSION),
        "execution_status": "completed",
        "execution_surface": "deterministic_cpu_census_no_model_inference",
        "content_digest_contract": {
            "field": "execution_receipt_content_sha256",
            "algorithm": "sha256",
            "canonicalization": (
                "UTF-8 JSON with ensure_ascii=false, sorted keys, and compact separators"
            ),
            "excluded_top_level_fields": ["execution_receipt_content_sha256"],
        },
        "command": {
            "capture_mode": "explicit_census_argv" if execution_argv is not None else "calling_process_argv",
            "argv": argv,
            "shell_escaped": shlex.join(argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "repository": _repository_state(),
        "runtime": _runtime_identity(),
        "coordinate_contract": {
            "coordinate_bin_min": 0,
            "coordinate_bin_max": 999,
            "source_token_grammar": "<|coord_N|> with integer N in [0, 999]",
            "pixel_conversion": "round(value * extent / 1000)",
            "axis_extents": {"x": "image_width", "y": "image_height"},
            "rollout_prediction_geometry": "stored pixel-space xyxy; no norm1000 conversion",
        },
        "matcher_contract": dict(matcher_semantics),
        "inputs": {
            "panel": dict(panel_digest),
            "greedy_artifact": {"path": greedy["path"], "sha256": greedy["sha256"]},
            "sampled_artifacts": [{"path": item["path"], "sha256": item["sha256"]} for item in sampled],
            "alias_table": dict(alias_source) if alias_source is not None else None,
            "production_policy_manifest": dict(production) if production is not None else None,
            "bound_files": bound_files,
        },
        "model_tokenizer_processor_identity": {
            "model_components": dict(greedy["model_components"]),
            "model_identity_sha256": str(greedy["model_identity_sha256"]),
            "exact_identity": greedy["exact_model_identity"],
            "tokenizer_identity": greedy["tokenizer_identity"],
            "processor_identity": greedy["processor_identity"],
            "generation_config_fingerprint": greedy["generation_config_fingerprint"],
            "execution_model_identity": greedy["execution_model_identity"],
            "likelihood_semantics": greedy["likelihood_semantics"],
            "model_component_files": list(model_file_hashes),
        },
        "policies": {
            "greedy_config": dict(greedy["config"]),
            "sampled_configs": [dict(item["config"]) for item in sampled],
            "production_generation_policy": production["generation_policy"] if production is not None else None,
        },
        "prompt_media_identities": prompt_identities,
        "output_schema_contracts": {
            "execution_receipt": {
                "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(EXECUTION_RECEIPT_SCHEMA_VERSION),
            },
            "census": {"schema_version": SCHEMA_VERSION, "compatibility": _schema_compatibility(SCHEMA_VERSION)},
            "matcher": {
                "schema_version": MATCHER_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(MATCHER_SCHEMA_VERSION),
            },
            "owner_ledger": {
                "schema_version": OWNER_LEDGER_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(OWNER_LEDGER_SCHEMA_VERSION),
            },
            "prediction_ledger": {
                "schema_version": PREDICTION_LEDGER_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(PREDICTION_LEDGER_SCHEMA_VERSION),
            },
            "owner_trajectory_matrix": {
                "schema_version": MATRIX_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(MATRIX_SCHEMA_VERSION),
            },
            "native_replay": {
                "schema_version": NATIVE_REPLAY_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(NATIVE_REPLAY_SCHEMA_VERSION),
            },
            "ambiguity_receipt": {
                "schema_version": AMBIGUITY_RECEIPT_SCHEMA_VERSION,
                "compatibility": _schema_compatibility(AMBIGUITY_RECEIPT_SCHEMA_VERSION),
            },
        },
    }
    receipt["execution_receipt_content_sha256"] = _receipt_digest(receipt)
    return receipt


def _validate_execution_receipt_binding(census: Mapping[str, Any]) -> None:
    receipt = _require_mapping(census.get("execution_receipt"), "census.execution_receipt")
    expected = receipt.get("execution_receipt_content_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or _receipt_digest(receipt) != expected:
        raise CensusContractError("execution receipt binding mismatch: receipt digest is invalid")
    matcher = _require_mapping(census.get("matcher_contract"), "census.matcher_contract")
    if matcher.get("execution_receipt_content_sha256") != expected:
        raise CensusContractError("execution receipt binding mismatch: matcher contract")
    for family in (
        "owner_ledger",
        "prediction_ledger",
        "owner_trajectory_matrix",
        "native_replay",
        "ambiguity_receipts",
    ):
        rows = _require_list(census.get(family), f"census.{family}")
        if any(
            not isinstance(row, Mapping)
            or row.get("execution_receipt_content_sha256") != expected
            for row in rows
        ):
            raise CensusContractError(f"execution receipt binding mismatch: {family}")

    repository = _require_mapping(receipt.get("repository"), "execution receipt repository")
    current_repository = _repository_state(Path(str(repository.get("root"))))
    if current_repository != dict(repository):
        raise CensusContractError("execution receipt binding mismatch: repository state changed")
    runtime = _require_mapping(receipt.get("runtime"), "execution receipt runtime")
    if _runtime_identity() != dict(runtime):
        raise CensusContractError("execution receipt binding mismatch: runtime identity changed")

    inputs = _require_mapping(receipt.get("inputs"), "execution receipt inputs")
    bound_files = _require_list(inputs.get("bound_files"), "execution receipt inputs.bound_files")
    for index, raw_file in enumerate(bound_files):
        item = _require_mapping(raw_file, f"execution receipt bound_files[{index}]")
        path = Path(str(item.get("path"))).resolve(strict=True)
        if path.stat().st_size != item.get("bytes") or _sha256_file(path) != item.get("sha256"):
            raise CensusContractError(f"execution receipt binding mismatch: bound file changed: {path}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CensusContractError(f"invalid JSON: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_description(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _canonical_image_id(value: Any) -> str:
    text = str(value)
    if text.isdigit():
        return str(int(text))
    match = _IMAGE_SUFFIX_RE.search(text)
    return str(int(match.group(1))) if match is not None else text


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CensusContractError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CensusContractError(f"{context} must be a list")
    return value


def _pixel_box(value: Any, *, context: str) -> tuple[float, float, float, float]:
    """Read a rollout pixel-space box without normalizing or rounding it."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise CensusContractError(f"{context} must contain four xyxy coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
        raise CensusContractError(f"{context} must contain finite numeric pixel coordinates")
    coords = tuple(float(item) for item in value)
    x1, y1, x2, y2 = (float(item) for item in coords)
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise CensusContractError(f"{context} is not a valid xyxy box: {value!r}")
    return (x1, y1, x2, y2)


def _source_bins_to_pixel_box(value: Any, *, width: Any, height: Any, context: str) -> tuple[int, int, int, int]:
    """Apply the canonical source-token parser and norm1000 pixel conversion."""

    try:
        bins = parse_source_bbox_tokens(value, field=context)
        return coord_bins_to_pixel_xyxy(bins, image_width=width, image_height=height, field=context)
    except DataContractError as exc:
        raise CensusContractError(f"{context} violates canonical source coordinate contract: {exc}") from exc


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _model_components(identity: Mapping[str, Any], context: str) -> dict[str, str]:
    """Extract only stable execution components, never a local evaluator ID."""

    nested = identity.get("model_identity")
    model = nested if isinstance(nested, Mapping) else identity
    base = _require_mapping(model.get("base"), f"{context}.model_identity.base")
    adapter = _require_mapping(model.get("adapter"), f"{context}.model_identity.adapter")
    delta = _require_mapping(model.get("embedding_delta"), f"{context}.model_identity.embedding_delta")
    delta_identity = _require_mapping(delta.get("identity"), f"{context}.embedding_delta.identity")
    values = {
        "base_model_path": base.get("path"),
        "adapter_path": adapter.get("adapter_path"),
        "embedding_delta_path": delta_identity.get("delta_path"),
    }
    missing = [name for name, value in values.items() if not isinstance(value, str) or not value]
    if missing:
        raise CensusContractError(f"{context} model identity is missing {', '.join(missing)}")
    return {name: str(value) for name, value in values.items()}


def _find_forced_paths(value: Any, *, path: str = "$") -> list[str]:
    """Reject structural forced-continuation markers without scanning model text."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _FORCED_KEY_RE.search(str(key)):
                found.append(child)
            if str(key) in {"experiment_mode", "mode", "generation_mode"} and isinstance(item, str):
                if "forced" in item.lower() or "intervention" in item.lower():
                    found.append(child)
            found.extend(_find_forced_paths(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forced_paths(item, path=f"{path}[{index}]"))
    return found


def _load_alias_table(path: Path | None) -> tuple[dict[str, set[str]], dict[str, Any], dict[str, str] | None]:
    if path is None:
        return {}, {"schema_version": "sorted-owner-basin-aliases.v1", "aliases": []}, None
    source = path.resolve(strict=True)
    payload = _require_mapping(_read_json(source), f"alias table {source}")
    if payload.get("schema_version") != "sorted-owner-basin-aliases.v1":
        raise CensusContractError("alias table schema_version must be 'sorted-owner-basin-aliases.v1'")
    raw_aliases = _require_list(payload.get("aliases"), "alias table aliases")
    adjacency: dict[str, set[str]] = defaultdict(set)
    canonical_pairs: list[dict[str, str]] = []
    for index, raw in enumerate(raw_aliases):
        item = _require_mapping(raw, f"alias table aliases[{index}]")
        left = _normalize_description(item.get("left"))
        right = _normalize_description(item.get("right"))
        if not left or not right or left == right:
            raise CensusContractError(f"alias table aliases[{index}] must join two distinct nonempty descriptions")
        adjacency[left].add(right)
        adjacency[right].add(left)
        canonical_pairs.append({"left": min(left, right), "right": max(left, right)})
    unique_pairs = sorted({(item["left"], item["right"]) for item in canonical_pairs})
    normalized = {
        "schema_version": "sorted-owner-basin-aliases.v1",
        "aliases": [{"left": left, "right": right} for left, right in unique_pairs],
    }
    return dict(adjacency), normalized, {"path": str(source), "sha256": _sha256_file(source)}


def _compatible_description(prediction: Mapping[str, Any], owner: Mapping[str, Any], aliases: Mapping[str, set[str]]) -> bool:
    pred_description = str(prediction["normalized_description"])
    owner_description = str(owner["normalized_description"])
    return pred_description == owner_description or owner_description in aliases.get(pred_description, set())


def _load_panel(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    source = path.resolve(strict=True)
    records: list[dict[str, Any]] = []
    owners_by_image: dict[str, list[dict[str, Any]]] = {}
    seen_images: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = _require_mapping(json.loads(line), f"panel line {line_number}")
        image_id = _canonical_image_id(raw.get("image_id"))
        if not image_id or image_id in seen_images:
            raise CensusContractError(f"panel has duplicate or empty image_id at line {line_number}")
        seen_images.add(image_id)
        width = raw.get("width")
        height = raw.get("height")
        if isinstance(width, bool) or not isinstance(width, int) or isinstance(height, bool) or not isinstance(height, int):
            raise CensusContractError(f"panel image {image_id} width/height must be positive integer source dimensions")
        if width <= 0 or height <= 0:
            raise CensusContractError(f"panel image {image_id} width/height must be positive")
        raw_objects = _require_list(raw.get("objects"), f"panel image {image_id}.objects")
        owners: list[dict[str, Any]] = []
        for original_index, raw_object in enumerate(raw_objects):
            obj = _require_mapping(raw_object, f"panel image {image_id}.objects[{original_index}]")
            raw_description = obj.get("category_name", obj.get("desc"))
            description = _normalize_description(raw_description)
            if not description:
                raise CensusContractError(f"panel owner {image_id}:{original_index} lacks a description")
            if "bbox_2d" in obj:
                box = _source_bins_to_pixel_box(obj["bbox_2d"], width=width, height=height, context=f"panel owner {image_id}:{original_index}.bbox_2d")
            else:
                box = _pixel_box(obj.get("bbox", obj.get("bbox_xyxy")), context=f"panel owner {image_id}:{original_index}.bbox")
            category_id = obj.get("category_id")
            if category_id is not None and not isinstance(category_id, int):
                raise CensusContractError(f"panel owner {image_id}:{original_index} has non-integer official category_id")
            gt_owner_id = f"gt:{image_id}:{original_index}"
            owners.append(
                {
                    "gt_owner_id": gt_owner_id,
                    "diagnostic_owner_id": f"diagnostic:{gt_owner_id}",
                    "image_id": image_id,
                    "original_annotation_index": original_index,
                    "description": str(raw_description),
                    "normalized_description": description,
                    "official_coco_category_id": category_id,
                    "bbox_xyxy": box,
                    "source_annotation": dict(obj),
                }
            )
        records.append({"image_id": image_id, "width": width, "height": height, "owner_count": len(owners)})
        owners_by_image[image_id] = owners
    if not records:
        raise CensusContractError("panel contains no image records")
    return records, owners_by_image, {"path": str(source), "sha256": _sha256_file(source)}


def _validate_config(config: Mapping[str, Any], *, mode: str, source: Path) -> None:
    if config.get("decode_mode") != mode:
        raise CensusContractError(f"{source} config.decode_mode must be {mode!r}")
    if config.get("max_new_tokens") != EXPECTED_HORIZON:
        raise CensusContractError(f"{source} config.max_new_tokens must be {EXPECTED_HORIZON}, not {config.get('max_new_tokens')!r}")
    if config.get("model_dtype") != "fp32":
        raise CensusContractError(f"{source} config.model_dtype must be 'fp32'")
    if float(config.get("repetition_penalty", math.nan)) != 1.0:
        raise CensusContractError(f"{source} config.repetition_penalty must be 1.0")
    if mode == "greedy":
        if float(config.get("temperature", math.nan)) != 0.0 or float(config.get("top_p", math.nan)) != 1.0:
            raise CensusContractError(f"{source} greedy policy must use temperature=0 and top_p=1")
        if config.get("seeds") != [EXPECTED_GREEDY_SEED]:
            raise CensusContractError(f"{source} greedy config.seeds must be [{EXPECTED_GREEDY_SEED}]")
    else:
        if float(config.get("temperature", math.nan)) != 0.4 or float(config.get("top_p", math.nan)) != 0.95:
            raise CensusContractError(f"{source} sampled policy must use temperature=0.4 and top_p=0.95")
        if tuple(config.get("seeds", ())) != EXPECTED_SAMPLED_SEEDS:
            raise CensusContractError(f"{source} sampled config.seeds must be 21001 through 21016")


def _validate_token_hash(row: Mapping[str, Any], field: str, source: Path, rollout_index: int) -> str:
    value = row.get(field)
    declared = row.get(f"{field}_sha256")
    if not isinstance(value, list) or not value or not isinstance(declared, str) or not declared:
        raise CensusContractError(f"{source} rollout {rollout_index} lacks {field} and its digest")
    actual = _json_digest(value)
    if actual != declared:
        raise CensusContractError(f"{source} rollout {rollout_index} {field}_sha256 does not match exact IDs")
    return actual


def _load_rollout_artifact(path: Path, *, mode: str, aliases: Mapping[str, set[str]]) -> dict[str, Any]:
    source = path.resolve(strict=True)
    source_sha256 = _sha256_file(source)
    payload = _require_mapping(_read_json(source), f"rollout artifact {source}")
    if payload.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise CensusContractError(f"{source} has unsupported rollout schema")
    config = _require_mapping(payload.get("config"), f"{source}.config")
    _validate_config(config, mode=mode, source=source)
    identity = _require_mapping(payload.get("model_identity"), f"{source}.model_identity")
    components = _model_components(identity, str(source))
    metadata = _require_mapping(payload.get("prompt_metadata"), f"{source}.prompt_metadata")
    raw_rollouts = _require_list(payload.get("rollouts"), f"{source}.rollouts")
    if payload.get("rollout_count") != len(raw_rollouts):
        raise CensusContractError(f"{source} rollout_count does not match rollouts")
    forced_paths = _find_forced_paths(payload)
    if forced_paths:
        raise CensusContractError(f"{source} contains forced-continuation markers: {', '.join(forced_paths[:3])}")

    config_images = {_canonical_image_id(item) for item in _require_list(config.get("image_ids"), f"{source}.config.image_ids")}
    if not config_images:
        raise CensusContractError(f"{source} config.image_ids is empty")
    rows: list[dict[str, Any]] = []
    seen_trajectory: set[tuple[str, int]] = set()
    for rollout_index, raw_rollout in enumerate(raw_rollouts):
        rollout = _require_mapping(raw_rollout, f"{source}.rollouts[{rollout_index}]")
        image_id = _canonical_image_id(rollout.get("image_id"))
        example_id = str(rollout.get("example_id", ""))
        if not image_id or image_id not in config_images or _canonical_image_id(example_id) != image_id:
            raise CensusContractError(f"{source} rollout {rollout_index} image/example identity disagrees with config")
        seed = rollout.get("seed")
        if not isinstance(seed, int):
            raise CensusContractError(f"{source} rollout {rollout_index} has non-integer seed")
        expected_seed_set = {EXPECTED_GREEDY_SEED} if mode == "greedy" else set(EXPECTED_SAMPLED_SEEDS)
        if seed not in expected_seed_set:
            raise CensusContractError(f"{source} rollout {rollout_index} has invalid {mode} seed {seed}")
        if rollout.get("decode_mode") != mode:
            raise CensusContractError(f"{source} rollout {rollout_index} decode_mode disagrees with config")
        trajectory_key = (image_id, seed)
        if trajectory_key in seen_trajectory:
            raise CensusContractError(f"{source} repeats trajectory {image_id}:{seed}")
        seen_trajectory.add(trajectory_key)
        stop_reason = rollout.get("stop_reason")
        if stop_reason not in {"im_end", "length"}:
            raise CensusContractError(f"{source} rollout {rollout_index} lacks natural termination metadata")
        prompt_hash = _validate_token_hash(rollout, "prompt_token_ids", source, rollout_index)
        generated_hash = _validate_token_hash(rollout, "generated_token_ids", source, rollout_index)
        media_hash = rollout.get("executed_media_sha256")
        if not isinstance(media_hash, str) or not media_hash:
            raise CensusContractError(f"{source} rollout {rollout_index} lacks executed_media_sha256")
        prompt_record = _require_mapping(metadata.get(example_id), f"{source}.prompt_metadata[{example_id!r}]")
        prompt_ids = _require_list(prompt_record.get("prompt_token_ids"), f"{source}.prompt_metadata[{example_id!r}].prompt_token_ids")
        if _json_digest(prompt_ids) != prompt_hash or prompt_ids != rollout.get("prompt_token_ids"):
            raise CensusContractError(f"{source} rollout {rollout_index} prompt IDs disagree with prompt metadata")
        if not isinstance(prompt_record.get("chat_text_sha256"), str) or not prompt_record.get("chat_text_sha256"):
            raise CensusContractError(f"{source} prompt metadata {example_id} lacks chat_text_sha256")
        if not isinstance(prompt_record.get("image_sha256"), str) or not prompt_record.get("image_sha256"):
            raise CensusContractError(f"{source} prompt metadata {example_id} lacks image_sha256")

        predictions_container = _require_mapping(rollout.get("predictions"), f"{source} rollout {rollout_index}.predictions")
        if predictions_container.get("parse_status") not in {"accepted", "accepted_with_drops"} or predictions_container.get("metric_bearing") is not True:
            raise CensusContractError(f"{source} rollout {rollout_index} is not an accepted metric-bearing parse")
        predictions = _require_list(predictions_container.get("predictions"), f"{source} rollout {rollout_index}.predictions.predictions")
        if predictions_container.get("valid_prediction_count") != len(predictions):
            raise CensusContractError(f"{source} rollout {rollout_index} valid_prediction_count disagrees with rows")
        dropped_predictions = _require_list(predictions_container.get("dropped_predictions"), f"{source} rollout {rollout_index}.predictions.dropped_predictions")
        if predictions_container.get("dropped_prediction_count") != len(dropped_predictions):
            raise CensusContractError(f"{source} rollout {rollout_index} dropped_prediction_count disagrees with rows")
        normalized_predictions: list[dict[str, Any]] = []
        normalized_invalid_predictions: list[dict[str, Any]] = []
        used_original_indices: set[int] = set()
        for list_index, raw_prediction in enumerate(predictions):
            prediction = _require_mapping(raw_prediction, f"{source} rollout {rollout_index} prediction {list_index}")
            row_index = prediction.get("generated_order")
            if not isinstance(row_index, int) or row_index < 0 or row_index in used_original_indices:
                raise CensusContractError(f"{source} rollout {rollout_index} has invalid or repeated generated_order")
            used_original_indices.add(row_index)
            description = _normalize_description(prediction.get("description"))
            try:
                if not description:
                    raise CensusContractError("missing normalized description")
                box = _pixel_box(prediction.get("bbox"), context=f"{source} rollout {rollout_index} prediction {row_index}.bbox")
            except CensusContractError as exc:
                # The existing parser can mark a grammar span accepted while
                # its decoded geometry is degenerate.  Retain its immutable
                # source row, but do not silently admit it to strict matching.
                normalized_invalid_predictions.append(
                    {
                        "pred_row_id": f"pred:sorted:{mode}:{seed}:{image_id}:{row_index}",
                        "image_id": image_id,
                        "example_id": example_id,
                        "policy_stratum": "primary_rp_1.00",
                        "decode_mode": mode,
                        "seed": seed,
                        "original_row_index": row_index,
                        "description": prediction.get("description"),
                        "normalized_description": description or None,
                        "bbox_xyxy": None,
                        "object_span_id": prediction.get("object_span_id"),
                        "raw_span_sha256": prediction.get("raw_span_sha256"),
                        "row_kind": "parser_invalid_geometry_or_description",
                        "parser_invalid_reason": str(exc),
                        "source_artifact_path": str(source),
                        "source_artifact_sha256": source_sha256,
                    }
                )
                continue
            pred_row_id = f"pred:sorted:{mode}:{seed}:{image_id}:{row_index}"
            normalized_predictions.append(
                {
                    "pred_row_id": pred_row_id,
                    "image_id": image_id,
                    "example_id": example_id,
                    "policy_stratum": "primary_rp_1.00",
                    "decode_mode": mode,
                    "seed": seed,
                    "original_row_index": row_index,
                    "description": str(prediction.get("description")),
                    "normalized_description": description,
                    "bbox_xyxy": box,
                    "object_span_id": prediction.get("object_span_id"),
                    "raw_span_sha256": prediction.get("raw_span_sha256"),
                    "row_kind": "complete_prediction",
                    "source_artifact_path": str(source),
                    "source_artifact_sha256": source_sha256,
                }
            )
        normalized_dropped_predictions: list[dict[str, Any]] = []
        for list_index, raw_prediction in enumerate(dropped_predictions):
            prediction = _require_mapping(raw_prediction, f"{source} rollout {rollout_index} dropped prediction {list_index}")
            row_index = prediction.get("generated_order")
            if not isinstance(row_index, int) or row_index < 0 or row_index in used_original_indices:
                raise CensusContractError(f"{source} rollout {rollout_index} has invalid or repeated dropped generated_order")
            used_original_indices.add(row_index)
            normalized_dropped_predictions.append(
                {
                    "pred_row_id": f"pred:sorted:{mode}:{seed}:{image_id}:{row_index}",
                    "image_id": image_id,
                    "example_id": example_id,
                    "policy_stratum": "primary_rp_1.00",
                    "decode_mode": mode,
                    "seed": seed,
                    "original_row_index": row_index,
                    "description": None,
                    "normalized_description": None,
                    "bbox_xyxy": None,
                    "object_span_id": prediction.get("object_span_id"),
                    "raw_span_sha256": prediction.get("raw_span_sha256"),
                    "row_kind": "parser_dropped",
                    "parser_drop_reason": prediction.get("reason"),
                    "source_artifact_path": str(source),
                    "source_artifact_sha256": source_sha256,
                }
            )
        rows.append(
            {
                "trajectory_id": f"trajectory:sorted:rp1.00:{mode}:{seed}:{image_id}",
                "image_id": image_id,
                "example_id": example_id,
                "policy_stratum": "primary_rp_1.00",
                "decode_mode": mode,
                "seed": seed,
                "stop_reason": str(stop_reason),
                "prompt_token_ids_sha256": prompt_hash,
                "generated_token_ids_sha256": generated_hash,
                "executed_media_sha256": str(media_hash),
                "chat_text_sha256": str(prompt_record["chat_text_sha256"]),
                "source_image_sha256": str(prompt_record["image_sha256"]),
                "observed_image_grid_thw": rollout.get("observed_image_grid_thw"),
                "predictions": normalized_predictions,
                "invalid_predictions": normalized_invalid_predictions,
                "dropped_predictions": normalized_dropped_predictions,
                "source_artifact_path": str(source),
                "source_artifact_sha256": source_sha256,
            }
        )
    return {
        "path": str(source),
        "sha256": source_sha256,
        "config": dict(config),
        "model_components": components,
        "model_identity_sha256": _json_digest(identity),
        "exact_model_identity": dict(identity),
        "tokenizer_identity": identity.get("tokenizer_identity"),
        "processor_identity": identity.get("processor_identity"),
        "generation_config_fingerprint": identity.get("generation_config_fingerprint"),
        "execution_model_identity": identity.get("execution_model_identity"),
        "likelihood_semantics": identity.get("likelihood_semantics"),
        "rows": rows,
        "forced_continuation_absent": True,
        "forced_marker_paths": [],
    }


def _validate_primary_panel(
    panel_records: Sequence[Mapping[str, Any]],
    owners_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    greedy: Mapping[str, Any],
    sampled: Sequence[Mapping[str, Any]],
    *,
    require_full_panel: bool,
) -> None:
    panel_images = {str(record["image_id"]) for record in panel_records}
    if require_full_panel and (len(panel_images) != EXPECTED_PANEL_IMAGE_COUNT or sum(len(items) for items in owners_by_image.values()) != EXPECTED_PANEL_OWNER_COUNT):
        raise CensusContractError(
            f"frozen panel must contain {EXPECTED_PANEL_IMAGE_COUNT} images and {EXPECTED_PANEL_OWNER_COUNT} owners"
        )
    if len(sampled) != 2:
        raise CensusContractError("primary census requires exactly two sampled shards")
    greedy_images = {str(item["image_id"]) for item in greedy["rows"]}
    if greedy_images != panel_images or len(greedy["rows"]) != len(panel_images):
        raise CensusContractError("greedy artifacts do not contain exactly one trajectory per frozen-panel image")
    for row in greedy["rows"]:
        if row["seed"] != EXPECTED_GREEDY_SEED:
            raise CensusContractError("greedy trajectory seed must be 0")
    sampled_rows = [row for artifact in sampled for row in artifact["rows"]]
    seen = {(str(row["image_id"]), int(row["seed"])) for row in sampled_rows}
    expected = {(image_id, seed) for image_id in panel_images for seed in EXPECTED_SAMPLED_SEEDS}
    if seen != expected or len(sampled_rows) != len(expected):
        raise CensusContractError("sampled shards must provide every frozen image once for every seed 21001..21016")
    for artifact in sampled:
        config_images = {_canonical_image_id(item) for item in artifact["config"]["image_ids"]}
        row_images = {str(row["image_id"]) for row in artifact["rows"]}
        if config_images != row_images:
            raise CensusContractError(f"{artifact['path']} config image_ids disagree with its rollout images")
    if {_canonical_image_id(item) for item in greedy["config"]["image_ids"]} != panel_images:
        raise CensusContractError("greedy config image_ids disagree with frozen panel")

    baseline_components = greedy["model_components"]
    baseline_identity = greedy["model_identity_sha256"]
    for artifact in sampled:
        if artifact["model_components"] != baseline_components or artifact["model_identity_sha256"] != baseline_identity:
            raise CensusContractError("natural artifacts disagree on checkpoint/model/tokenizer identity")
        for key, value in artifact["config"].items():
            if key in {"decode_mode", "temperature", "top_p", "seeds", "image_ids", "device"}:
                continue
            if greedy["config"].get(key) != value:
                raise CensusContractError(f"policy-independent config mismatch for {key!r}")
    all_rows = list(greedy["rows"]) + sampled_rows
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_image[str(row["image_id"])].append(row)
    for image_id, rows in by_image.items():
        for field in ("prompt_token_ids_sha256", "executed_media_sha256", "chat_text_sha256", "source_image_sha256", "observed_image_grid_thw"):
            values = {json.dumps(row[field], sort_keys=True) if isinstance(row[field], (list, dict)) else str(row[field]) for row in rows}
            if len(values) != 1:
                raise CensusContractError(f"natural artifacts disagree on {field} for image {image_id}")


def _assignment_inputs(
    predictions: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, set[str]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[tuple[int, int], float]]:
    ordered_predictions = sorted(predictions, key=lambda item: int(item["original_row_index"]))
    ordered_owners = sorted(owners, key=lambda item: int(item["original_annotation_index"]))
    edges: dict[tuple[int, int], float] = {}
    for pred_index, prediction in enumerate(ordered_predictions):
        for owner_index, owner in enumerate(ordered_owners):
            if not _compatible_description(prediction, owner, aliases):
                continue
            overlap = _iou(prediction["bbox_xyxy"], owner["bbox_xyxy"])
            if overlap >= IOU_THRESHOLD:
                edges[(pred_index, owner_index)] = overlap
    return ordered_predictions, ordered_owners, edges


def _solve_assignment_indices(
    prediction_count: int,
    owner_count: int,
    edges: Mapping[tuple[int, int], float],
    *,
    forced_edges: frozenset[tuple[int, int]] = frozenset(),
    forbidden_predictions: frozenset[int] = frozenset(),
    forbidden_owners: frozenset[int] = frozenset(),
) -> list[tuple[int, int]] | None:
    """Solve the frozen cardinality/total-IoU objective under constraints."""

    forced_predictions = [edge[0] for edge in forced_edges]
    forced_owners = [edge[1] for edge in forced_edges]
    if (
        any(edge not in edges for edge in forced_edges)
        or len(set(forced_predictions)) != len(forced_predictions)
        or len(set(forced_owners)) != len(forced_owners)
        or any(index in forbidden_predictions for index in forced_predictions)
        or any(index in forbidden_owners for index in forced_owners)
    ):
        return None

    used_predictions = set(forced_predictions) | set(forbidden_predictions)
    used_owners = set(forced_owners) | set(forbidden_owners)
    source = 0
    prediction_base = 1
    owner_base = prediction_base + prediction_count
    sink = owner_base + owner_count
    graph: list[list[list[Any]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: float, metadata: tuple[int, int] | None = None) -> None:
        graph[start].append([end, len(graph[end]), capacity, cost, metadata])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost, None])

    for pred_index in range(prediction_count):
        if pred_index not in used_predictions:
            add_edge(source, prediction_base + pred_index, 1, 0.0)
    for owner_index in range(owner_count):
        if owner_index not in used_owners:
            add_edge(owner_base + owner_index, sink, 1, 0.0)
    for (pred_index, owner_index), overlap in sorted(edges.items(), key=lambda item: (item[0][1], item[0][0])):
        if pred_index in used_predictions or owner_index in used_owners:
            continue
        add_edge(prediction_base + pred_index, owner_base + owner_index, 1, -overlap, (pred_index, owner_index))

    while True:
        distance = [math.inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0.0
        queue: deque[int] = deque([source])
        in_queue = {source}
        while queue:
            node = queue.popleft()
            in_queue.discard(node)
            for edge_index, edge in enumerate(graph[node]):
                target, _, capacity, cost, _ = edge
                candidate = distance[node] + float(cost)
                if int(capacity) <= 0 or distance[target] <= candidate + _EPSILON:
                    continue
                distance[target] = candidate
                previous[target] = (node, edge_index)
                if target not in in_queue:
                    queue.append(target)
                    in_queue.add(target)
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            previous_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[previous_node][edge_index]
            edge[2] = int(edge[2]) - 1
            graph[node][int(edge[1])][2] = int(graph[node][int(edge[1])][2]) + 1
            node = previous_node

    selected = set(forced_edges)
    for pred_index in range(prediction_count):
        for edge in graph[prediction_base + pred_index]:
            target, _, capacity, _, metadata = edge
            if not (owner_base <= int(target) < sink) or metadata is None or int(capacity) != 0:
                continue
            selected.add((pred_index, int(target) - owner_base))
    return sorted(selected)


def _assignment_objective(
    solution: Sequence[tuple[int, int]] | None,
    edges: Mapping[tuple[int, int], float],
) -> tuple[int, float] | None:
    if solution is None:
        return None
    return len(solution), math.fsum(edges[edge] for edge in solution)


def _same_assignment_objective(
    observed: tuple[int, float] | None,
    expected: tuple[int, float],
) -> bool:
    return observed is not None and observed[0] == expected[0] and math.isclose(
        observed[1], expected[1], rel_tol=0.0, abs_tol=_EPSILON
    )


def _global_assignment_analysis(
    predictions: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Enumerate the optimal-assignment face before the deterministic tie-break."""

    ordered_predictions, ordered_owners, edges = _assignment_inputs(predictions, owners, aliases)
    if not ordered_predictions or not ordered_owners or not edges:
        return {
            "matches": [],
            "globally_optimal_edges": [],
            "ambiguity_classes": [],
            "ambiguous_pred_row_ids": [],
            "ambiguous_gt_owner_ids": [],
            "optimum_cardinality": 0,
            "optimum_total_iou": 0.0,
        }
    base_solution = _solve_assignment_indices(len(ordered_predictions), len(ordered_owners), edges)
    base_objective = _assignment_objective(base_solution, edges)
    if base_objective is None:
        raise CensusContractError("internal matcher failure: unconstrained assignment is infeasible")

    participating_edges: set[tuple[int, int]] = set()
    forced_edge_objectives: dict[tuple[int, int], tuple[int, float]] = {}
    for edge in sorted(edges, key=lambda item: (item[1], item[0])):
        constrained = _solve_assignment_indices(
            len(ordered_predictions), len(ordered_owners), edges, forced_edges=frozenset({edge})
        )
        constrained_objective = _assignment_objective(constrained, edges)
        if _same_assignment_objective(constrained_objective, base_objective):
            participating_edges.add(edge)
            forced_edge_objectives[edge] = constrained_objective  # type: ignore[assignment]

    participating_by_prediction: dict[int, set[int]] = defaultdict(set)
    participating_by_owner: dict[int, set[int]] = defaultdict(set)
    for pred_index, owner_index in participating_edges:
        participating_by_prediction[pred_index].add(owner_index)
        participating_by_owner[owner_index].add(pred_index)

    ambiguous_predictions: set[int] = set()
    for pred_index, owner_indices in participating_by_prediction.items():
        without_prediction = _solve_assignment_indices(
            len(ordered_predictions),
            len(ordered_owners),
            edges,
            forbidden_predictions=frozenset({pred_index}),
        )
        if len(owner_indices) > 1 or _same_assignment_objective(
            _assignment_objective(without_prediction, edges), base_objective
        ):
            ambiguous_predictions.add(pred_index)
    ambiguous_owners: set[int] = set()
    for owner_index, pred_indices in participating_by_owner.items():
        without_owner = _solve_assignment_indices(
            len(ordered_predictions),
            len(ordered_owners),
            edges,
            forbidden_owners=frozenset({owner_index}),
        )
        if len(pred_indices) > 1 or _same_assignment_objective(
            _assignment_objective(without_owner, edges), base_objective
        ):
            ambiguous_owners.add(owner_index)

    ambiguity_components: list[dict[str, Any]] = []
    pending_predictions = set(ambiguous_predictions)
    pending_owners = set(ambiguous_owners)
    while pending_predictions or pending_owners:
        component_predictions: set[int] = set()
        component_owners: set[int] = set()
        queue: deque[tuple[str, int]] = deque()
        if pending_predictions:
            queue.append(("prediction", min(pending_predictions)))
        else:
            queue.append(("owner", min(pending_owners)))
        while queue:
            kind, index = queue.popleft()
            if kind == "prediction":
                if index in component_predictions:
                    continue
                component_predictions.add(index)
                pending_predictions.discard(index)
                for owner_index in participating_by_prediction.get(index, set()):
                    if owner_index in ambiguous_owners and owner_index not in component_owners:
                        queue.append(("owner", owner_index))
            else:
                if index in component_owners:
                    continue
                component_owners.add(index)
                pending_owners.discard(index)
                for pred_index in participating_by_owner.get(index, set()):
                    if pred_index in ambiguous_predictions and pred_index not in component_predictions:
                        queue.append(("prediction", pred_index))
        pred_row_ids = sorted(str(ordered_predictions[index]["pred_row_id"]) for index in component_predictions)
        owner_ids = sorted(str(ordered_owners[index]["gt_owner_id"]) for index in component_owners)
        semantic_signatures = {
            (
                str(ordered_predictions[index]["normalized_description"]),
                tuple(float(value) for value in ordered_predictions[index]["bbox_xyxy"]),
            )
            for index in component_predictions
        }
        exact_duplicate_exchange = len(component_predictions) > 1 and len(semantic_signatures) == 1
        reason = (
            "globally_indistinguishable_exact_duplicate_prediction_rows"
            if exact_duplicate_exchange
            else "multiple_globally_optimal_assignments"
        )
        component_edges = sorted(
            (
                edge
                for edge in participating_edges
                if edge[0] in component_predictions and edge[1] in component_owners
            ),
            key=lambda item: (item[1], item[0]),
        )
        class_key = {"pred_row_ids": pred_row_ids, "gt_owner_ids": owner_ids, "reason": reason}
        ambiguity_components.append(
            {
                "ambiguity_class_local_id": f"ambiguity-class:{_json_digest(class_key)[:16]}",
                "reason": reason,
                "pred_row_ids": pred_row_ids,
                "gt_owner_ids": owner_ids,
                "prediction_semantic_signatures": [
                    {"normalized_description": signature[0], "bbox_xyxy": list(signature[1])}
                    for signature in sorted(semantic_signatures)
                ],
                "objective_equality": {
                    "maximum_cardinality": base_objective[0],
                    "maximum_total_iou": base_objective[1],
                    "numeric_equality_tolerance": _EPSILON,
                    "all_forced_edges_preserve_global_objective": True,
                },
                "globally_optimal_edge_receipts": [
                    {
                        "pred_row_id": str(ordered_predictions[pred_index]["pred_row_id"]),
                        "gt_owner_id": str(ordered_owners[owner_index]["gt_owner_id"]),
                        "intersection_over_union": edges[(pred_index, owner_index)],
                        "forced_solution_cardinality": forced_edge_objectives[(pred_index, owner_index)][0],
                        "forced_solution_total_iou": forced_edge_objectives[(pred_index, owner_index)][1],
                    }
                    for pred_index, owner_index in component_edges
                ],
            }
        )

    ambiguity_class_ids_by_prediction: dict[int, list[str]] = defaultdict(list)
    ambiguity_class_ids_by_owner: dict[int, list[str]] = defaultdict(list)
    prediction_index_by_id = {
        str(prediction["pred_row_id"]): index for index, prediction in enumerate(ordered_predictions)
    }
    owner_index_by_id = {str(owner["gt_owner_id"]): index for index, owner in enumerate(ordered_owners)}
    for component in ambiguity_components:
        class_id = str(component["ambiguity_class_local_id"])
        for pred_row_id in component["pred_row_ids"]:
            ambiguity_class_ids_by_prediction[prediction_index_by_id[str(pred_row_id)]].append(class_id)
        for owner_id in component["gt_owner_ids"]:
            ambiguity_class_ids_by_owner[owner_index_by_id[str(owner_id)]].append(class_id)

    # Freeze the documented GT-index then prediction-row-index final tie only
    # after the complete optimal face has been identified.
    forced: set[tuple[int, int]] = set()
    for edge in sorted(participating_edges, key=lambda item: (item[1], item[0])):
        if any(edge[0] == selected[0] or edge[1] == selected[1] for selected in forced):
            continue
        candidate_forced = frozenset({*forced, edge})
        candidate = _solve_assignment_indices(
            len(ordered_predictions), len(ordered_owners), edges, forced_edges=candidate_forced
        )
        if _same_assignment_objective(_assignment_objective(candidate, edges), base_objective):
            forced.add(edge)
        if len(forced) == base_objective[0]:
            break
    deterministic = _solve_assignment_indices(
        len(ordered_predictions), len(ordered_owners), edges, forced_edges=frozenset(forced)
    )
    if not _same_assignment_objective(_assignment_objective(deterministic, edges), base_objective):
        raise CensusContractError("internal matcher failure: deterministic tie-break changed the frozen objective")

    matches: list[dict[str, Any]] = []
    for pred_index, owner_index in deterministic or []:
        prediction = ordered_predictions[pred_index]
        owner = ordered_owners[owner_index]
        ambiguity = pred_index in ambiguous_predictions or owner_index in ambiguous_owners
        ambiguity_class_ids = sorted(
            {
                *ambiguity_class_ids_by_prediction.get(pred_index, []),
                *ambiguity_class_ids_by_owner.get(owner_index, []),
            }
        )
        globally_optimal_owner_ids = [
            str(ordered_owners[index]["gt_owner_id"])
            for index in sorted(participating_by_prediction.get(pred_index, set()))
        ]
        globally_optimal_pred_ids = [
            str(ordered_predictions[index]["pred_row_id"])
            for index in sorted(participating_by_owner.get(owner_index, set()))
        ]
        matches.append(
            {
                "pred_row_id": str(prediction["pred_row_id"]),
                "gt_owner_id": str(owner["gt_owner_id"]),
                "intersection_over_union": edges[(pred_index, owner_index)],
                "semantic_relation": "exact"
                if prediction["normalized_description"] == owner["normalized_description"]
                else "compatible_alias",
                "strict_match_status": "ambiguous_neutral" if ambiguity else "matched",
                "ambiguity_reason": (
                    next(
                        component["reason"]
                        for component in ambiguity_components
                        if component["ambiguity_class_local_id"] in ambiguity_class_ids
                    )
                    if ambiguity
                    else None
                ),
                "ambiguity_class_ids": ambiguity_class_ids,
                "ambiguous_same_description_gt_owner_ids": globally_optimal_owner_ids if ambiguity else [],
                "globally_optimal_gt_owner_ids": globally_optimal_owner_ids,
                "globally_optimal_pred_row_ids": globally_optimal_pred_ids,
            }
        )
    optimal_edge_receipts = [
        {
            "pred_row_id": str(ordered_predictions[pred_index]["pred_row_id"]),
            "gt_owner_id": str(ordered_owners[owner_index]["gt_owner_id"]),
            "intersection_over_union": edges[(pred_index, owner_index)],
        }
        for pred_index, owner_index in sorted(participating_edges, key=lambda item: (item[1], item[0]))
    ]
    return {
        "matches": sorted(matches, key=lambda item: item["pred_row_id"]),
        "globally_optimal_edges": optimal_edge_receipts,
        "ambiguity_classes": ambiguity_components,
        "ambiguous_pred_row_ids": sorted(str(ordered_predictions[index]["pred_row_id"]) for index in ambiguous_predictions),
        "ambiguous_gt_owner_ids": sorted(str(ordered_owners[index]["gt_owner_id"]) for index in ambiguous_owners),
        "optimum_cardinality": base_objective[0],
        "optimum_total_iou": base_objective[1],
    }


def _min_cost_max_cardinality_assignment(
    predictions: Sequence[Mapping[str, Any]], owners: Sequence[Mapping[str, Any]], aliases: Mapping[str, set[str]]
) -> list[dict[str, Any]]:
    """Return the deterministic representative after global ambiguity audit."""

    return _global_assignment_analysis(predictions, owners, aliases)["matches"]


def _trajectory_receipt(
    trajectory: Mapping[str, Any], owners: Sequence[Mapping[str, Any]], aliases: Mapping[str, set[str]]
) -> dict[str, Any]:
    predictions = trajectory["predictions"]
    analysis = _global_assignment_analysis(predictions, owners, aliases)
    ambiguity_receipts: list[dict[str, Any]] = []
    ambiguity_id_by_local: dict[str, str] = {}
    for item in analysis["ambiguity_classes"]:
        local_id = str(item["ambiguity_class_local_id"])
        receipt_id = f"ambiguity:{trajectory['trajectory_id']}:{local_id.rsplit(':', 1)[-1]}"
        ambiguity_id_by_local[local_id] = receipt_id
        ambiguity_receipts.append(
            {
                "schema_version": AMBIGUITY_RECEIPT_SCHEMA_VERSION,
                "schema_compatibility": _schema_compatibility(AMBIGUITY_RECEIPT_SCHEMA_VERSION),
                "ambiguity_receipt_id": receipt_id,
                "trajectory_id": trajectory["trajectory_id"],
                "image_id": trajectory["image_id"],
                "decode_mode": trajectory["decode_mode"],
                "seed": trajectory["seed"],
                **{key: value for key, value in item.items() if key != "ambiguity_class_local_id"},
                "foreign_keys": {
                    "trajectory_id": trajectory["trajectory_id"],
                    "pred_row_ids": item["pred_row_ids"],
                    "gt_owner_ids": item["gt_owner_ids"],
                },
                "source_digests": {"rollout_artifact": trajectory["source_artifact_sha256"]},
            }
        )
    matches = [
        {
            **item,
            "ambiguity_receipt_ids": [
                ambiguity_id_by_local[class_id] for class_id in item["ambiguity_class_ids"]
            ],
        }
        for item in analysis["matches"]
    ]
    ambiguity_ids_by_prediction: dict[str, list[str]] = defaultdict(list)
    for ambiguity_receipt in ambiguity_receipts:
        for pred_row_id in ambiguity_receipt["pred_row_ids"]:
            ambiguity_ids_by_prediction[str(pred_row_id)].append(
                str(ambiguity_receipt["ambiguity_receipt_id"])
            )
    matched_by_pred = {str(item["pred_row_id"]): item for item in matches}
    globally_optimal_by_pred: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in analysis["globally_optimal_edges"]:
        globally_optimal_by_pred[str(edge["pred_row_id"])].append(edge)
    ambiguous_predictions = set(analysis["ambiguous_pred_row_ids"])
    receipts: list[dict[str, Any]] = []
    for prediction in predictions:
        candidates = []
        max_any_iou = 0.0
        for owner in owners:
            overlap = _iou(prediction["bbox_xyxy"], owner["bbox_xyxy"])
            max_any_iou = max(max_any_iou, overlap)
            if _compatible_description(prediction, owner, aliases):
                candidates.append({"gt_owner_id": owner["gt_owner_id"], "intersection_over_union": overlap})
        candidates.sort(key=lambda item: (-float(item["intersection_over_union"]), str(item["gt_owner_id"])))
        pred_row_id = str(prediction["pred_row_id"])
        assignment = matched_by_pred.get(pred_row_id)
        is_ambiguous = pred_row_id in ambiguous_predictions
        globally_optimal_edges = globally_optimal_by_pred.get(pred_row_id, [])
        receipts.append(
            {
                "pred_row_id": pred_row_id,
                "strict_match_gt_owner_id": assignment["gt_owner_id"] if assignment else None,
                "strict_match_status": "ambiguous_neutral"
                if is_ambiguous
                else (assignment["strict_match_status"] if assignment else "unmatched"),
                "intersection_over_union": assignment["intersection_over_union"] if assignment else None,
                "semantic_relation": assignment["semantic_relation"] if assignment else None,
                "eligible_owner_iou_receipts": candidates,
                "max_any_owner_iou": max_any_iou,
                "ambiguous_same_description_gt_owner_ids": sorted(
                    str(item["gt_owner_id"]) for item in globally_optimal_edges
                )
                if is_ambiguous
                else [],
                "globally_optimal_edge_receipts": globally_optimal_edges,
                "ambiguity_receipt_ids": sorted(ambiguity_ids_by_prediction.get(pred_row_id, [])),
            }
        )
    committed = [item for item in matches if item["strict_match_status"] == "matched"]
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "matches": matches,
        "committed_matches": committed,
        "prediction_receipts": receipts,
        "assigned_owner_ids": sorted({str(item["gt_owner_id"]) for item in matches}),
        "committed_owner_ids": sorted({str(item["gt_owner_id"]) for item in committed}),
        "neutral_pred_row_ids": analysis["ambiguous_pred_row_ids"],
        "neutral_owner_ids": analysis["ambiguous_gt_owner_ids"],
        "globally_optimal_edge_receipts": analysis["globally_optimal_edges"],
        "ambiguity_receipts": ambiguity_receipts,
        "optimum_cardinality": analysis["optimum_cardinality"],
        "optimum_total_iou": analysis["optimum_total_iou"],
    }


def _metrics(owner_count: int, prediction_count: int, receipt: Mapping[str, Any]) -> dict[str, Any]:
    assigned = len(receipt["assigned_owner_ids"])
    committed = len(receipt["committed_owner_ids"])
    neutral_predictions = len(receipt["neutral_pred_row_ids"])
    neutral_owners = len(receipt["neutral_owner_ids"])
    false_positives = prediction_count - committed - neutral_predictions
    false_negatives = owner_count - committed - neutral_owners
    return {
        "automatic_deterministic_assignment": {
            "tp": assigned,
            "fp": prediction_count - assigned,
            "fn": owner_count - assigned,
            "decision_bearing": False,
        },
        "neutral_ambiguity_excluded_owner_presence": {
            "tp": committed,
            "fp": false_positives,
            "fn": false_negatives,
            "neutral_assignment_count": assigned - committed,
            "neutral_prediction_count": neutral_predictions,
            "neutral_owner_count": neutral_owners,
            "precision_denominator": committed + false_positives,
            "recall_denominator": committed + false_negatives,
        },
    }


def _hash_model_component_files(components: Mapping[str, str]) -> list[dict[str, Any]]:
    """Hash all payload files below the three frozen component roots.

    This is intentionally content-addressed, including base weight shards.  It
    is CPU/disk-only and may take time; it avoids treating an identity path as
    a digest.
    """

    files: list[dict[str, Any]] = []
    for component_name, root_text in sorted(components.items()):
        root = Path(root_text).resolve(strict=True)
        if not root.is_dir():
            raise CensusContractError(f"model component root is not a directory: {root}")
        for file in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item)):
            files.append(
                {
                    "component": component_name,
                    "path": str(file),
                    "relative_path": str(file.relative_to(root)),
                    "bytes": file.stat().st_size,
                    "sha256": _sha256_file(file),
                }
            )
    return files


def _validate_optional_production_manifest(path: Path, primary_components: Mapping[str, str], primary_identity_sha: str) -> dict[str, Any]:
    source = path.resolve(strict=True)
    payload = _require_mapping(_read_json(source), f"production manifest {source}")
    if payload.get("artifact_schema_version") != 1:
        raise CensusContractError(f"{source} is not an artifact_schema_version=1 production manifest")
    policy = _require_mapping(payload.get("generation_policy"), f"{source}.generation_policy")
    expected = {"do_sample": False, "max_new_tokens": EXPECTED_HORIZON, "repetition_penalty": 1.1, "temperature": 0, "top_p": 1}
    for key, value in expected.items():
        if policy.get(key) != value:
            raise CensusContractError(f"{source} production policy {key!r} must be {value!r}")
    if payload.get("terminal_status") != "completed":
        raise CensusContractError(f"{source} production manifest is not completed")
    identity = _require_mapping(payload.get("model_identity"), f"{source}.model_identity")
    components = _model_components(identity, str(source))
    if components != dict(primary_components):
        raise CensusContractError("RP=1.10 production manifest checkpoint identity differs from primary RP=1.0 artifact")
    # Different artifact families expose different incidental identity fields;
    # component equality is the required cross-family identity comparison.
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "policy_stratum": "production_rp_1.10_greedy",
        "generation_policy": dict(policy),
        "model_components": components,
        "exact_model_identity": dict(identity),
        "model_identity_sha256": _json_digest(identity),
        "primary_identity_sha256": primary_identity_sha,
        "separate_from_primary_any_hit_union": True,
    }


def build_census(
    *,
    panel_path: Path,
    greedy_path: Path,
    sampled_paths: Sequence[Path],
    production_manifest_path: Path | None = None,
    alias_table_path: Path | None = None,
    require_full_panel: bool = True,
    hash_model_components: bool = True,
    execution_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build all serializable Task-0/1 census payloads without writing files."""

    aliases, alias_document, alias_source = _load_alias_table(alias_table_path)
    panel_records, owners_by_image, panel_digest = _load_panel(panel_path)
    greedy = _load_rollout_artifact(greedy_path, mode="greedy", aliases=aliases)
    sampled = [_load_rollout_artifact(path, mode="sampled", aliases=aliases) for path in sampled_paths]
    _validate_primary_panel(panel_records, owners_by_image, greedy, sampled, require_full_panel=require_full_panel)
    all_trajectories = sorted(
        list(greedy["rows"]) + [row for artifact in sampled for row in artifact["rows"]],
        key=lambda item: (str(item["image_id"]), str(item["decode_mode"]), int(item["seed"])),
    )
    receipts_by_trajectory: dict[str, dict[str, Any]] = {}
    for trajectory in all_trajectories:
        image_id = str(trajectory["image_id"])
        receipts_by_trajectory[str(trajectory["trajectory_id"])] = _trajectory_receipt(trajectory, owners_by_image[image_id], aliases)

    prediction_receipts = {
        item["pred_row_id"]: item
        for receipt in receipts_by_trajectory.values()
        for item in receipt["prediction_receipts"]
    }
    ambiguity_ledger = [
        item
        for trajectory_id in sorted(receipts_by_trajectory)
        for item in receipts_by_trajectory[trajectory_id]["ambiguity_receipts"]
    ]
    all_owners = [owner for image_id in sorted(owners_by_image) for owner in owners_by_image[image_id]]
    model_file_hashes = _hash_model_component_files(greedy["model_components"]) if hash_model_components else []
    production = None
    if production_manifest_path is not None:
        production = _validate_optional_production_manifest(
            production_manifest_path,
            greedy["model_components"],
            greedy["model_identity_sha256"],
        )
    matcher_semantics = {
        "schema_version": MATCHER_SCHEMA_VERSION,
        "schema_compatibility": _schema_compatibility(MATCHER_SCHEMA_VERSION),
        "matcher_id": MATCHER_SCHEMA_VERSION,
        "category_namespace": {
            "owner_category_id": "official_gapped_coco_category_id",
            "strict_join": "normalized_description_or_predeclared_symmetric_alias_only",
            "local_evaluator_category_id_join": "forbidden",
        },
        "normalized_description_rule": "lowercase; underscores-to-spaces; collapse whitespace",
        "alias_policy": "no inferred aliases; only immutable symmetric aliases from alias table",
        "alias_table": alias_document,
        "alias_table_source": alias_source,
        "iou_threshold": IOU_THRESHOLD,
        "numeric_equality_tolerance": _EPSILON,
        "assignment_objective": ["maximum_cardinality", "maximum_total_iou"],
        "global_optimum_ambiguity": (
            "enumerate every eligible edge and matched-vs-unmatched node state across the complete "
            "maximum-cardinality, maximum-total-IoU optimal face before deterministic tie-breaking"
        ),
        "deterministic_final_tie_break": "gt_original_annotation_index_then_pred_original_row_index",
        "ambiguous_global_optimum_accounting": (
            "affected predictions and owners are neutral and excluded from committed TP/FP/FN denominators"
        ),
        "source_digests": {
            "panel": panel_digest["sha256"],
            "greedy": greedy["sha256"],
            "sampled": [item["sha256"] for item in sampled],
        },
    }
    execution_receipt = _build_execution_receipt(
        execution_argv=execution_argv,
        panel_digest=panel_digest,
        greedy=greedy,
        sampled=sampled,
        alias_source=alias_source,
        production=production,
        model_file_hashes=model_file_hashes,
        matcher_semantics=matcher_semantics,
        trajectories=all_trajectories,
    )
    execution_receipt_content_sha256 = str(
        execution_receipt["execution_receipt_content_sha256"]
    )
    matcher_contract = {
        **matcher_semantics,
        "execution_receipt_content_sha256": execution_receipt_content_sha256,
    }
    owner_source_digests = {"panel": panel_digest["sha256"]}
    owner_ledger = [
        {
            "schema_version": OWNER_LEDGER_SCHEMA_VERSION,
            "gt_owner_id": owner["gt_owner_id"],
            "diagnostic_owner_id": owner["diagnostic_owner_id"],
            "diagnostic_owner_kind": "frozen_gt",
            "mapped_gt_owner_id": owner["gt_owner_id"],
            "image_id": owner["image_id"],
            "original_annotation_index": owner["original_annotation_index"],
            "description": owner["description"],
            "normalized_description": owner["normalized_description"],
            "official_coco_category_id": owner["official_coco_category_id"],
            "bbox_xyxy": list(owner["bbox_xyxy"]),
            "foreign_keys": {"panel_image_id": owner["image_id"], "mapped_gt_owner_id": owner["gt_owner_id"]},
            "null_status": {"aux_owner_mapping": "not_materialized_no_review_input", "unresolved_mapping": "not_materialized_no_review_input"},
            "source_digests": owner_source_digests,
        }
        for owner in all_owners
    ]
    prediction_ledger: list[dict[str, Any]] = []
    for trajectory in all_trajectories:
        for prediction in [*trajectory["predictions"], *trajectory["invalid_predictions"], *trajectory["dropped_predictions"]]:
            if prediction["row_kind"] != "complete_prediction":
                status = "not_evaluable_parser_dropped" if prediction["row_kind"] == "parser_dropped" else "not_evaluable_invalid_geometry_or_description"
                prediction_ledger.append(
                    {
                        "schema_version": PREDICTION_LEDGER_SCHEMA_VERSION,
                        **prediction,
                        "trajectory_id": trajectory["trajectory_id"],
                        "strict_match_gt_owner_id": None,
                        "strict_match_status": status,
                        "strict_match_iou": None,
                        "semantic_relation": None,
                        "eligible_owner_iou_receipts": [],
                        "max_any_owner_iou": None,
                        "ambiguous_same_description_gt_owner_ids": [],
                        "globally_optimal_edge_receipts": [],
                        "ambiguity_receipt_ids": [],
                        "foreign_keys": {"trajectory_id": trajectory["trajectory_id"], "gt_owner_id": None},
                        "null_status": {
                            "strict_match_gt_owner_id": status,
                            "diagnostic_owner_id": "not_assigned_task0_1",
                        },
                        "source_digests": {"rollout_artifact": prediction["source_artifact_sha256"]},
                    }
                )
                continue
            receipt = prediction_receipts[prediction["pred_row_id"]]
            strict_owner = receipt["strict_match_gt_owner_id"]
            prediction_ledger.append(
                {
                    "schema_version": PREDICTION_LEDGER_SCHEMA_VERSION,
                    **prediction,
                    "trajectory_id": trajectory["trajectory_id"],
                    "strict_match_gt_owner_id": strict_owner,
                    "strict_match_status": receipt["strict_match_status"],
                    "strict_match_iou": receipt["intersection_over_union"],
                    "semantic_relation": receipt["semantic_relation"],
                    "eligible_owner_iou_receipts": receipt["eligible_owner_iou_receipts"],
                    "max_any_owner_iou": receipt["max_any_owner_iou"],
                    "ambiguous_same_description_gt_owner_ids": receipt["ambiguous_same_description_gt_owner_ids"],
                    "globally_optimal_edge_receipts": receipt["globally_optimal_edge_receipts"],
                    "ambiguity_receipt_ids": receipt["ambiguity_receipt_ids"],
                    "foreign_keys": {"trajectory_id": trajectory["trajectory_id"], "gt_owner_id": strict_owner},
                    "null_status": {
                        "strict_match_gt_owner_id": "present" if strict_owner else "no_eligible_strict_assignment",
                        "diagnostic_owner_id": "not_assigned_task0_1",
                    },
                    "source_digests": {"rollout_artifact": prediction["source_artifact_sha256"]},
                }
            )

    matrix: list[dict[str, Any]] = []
    for trajectory in all_trajectories:
        receipt = receipts_by_trajectory[trajectory["trajectory_id"]]
        neutral_owner_ids = set(receipt["neutral_owner_ids"])
        optimal_edges_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ambiguity_ids_by_owner: dict[str, list[str]] = defaultdict(list)
        for edge in receipt["globally_optimal_edge_receipts"]:
            optimal_edges_by_owner[str(edge["gt_owner_id"])].append(edge)
        for ambiguity_receipt in receipt["ambiguity_receipts"]:
            for owner_id in ambiguity_receipt["gt_owner_ids"]:
                ambiguity_ids_by_owner[str(owner_id)].append(
                    str(ambiguity_receipt["ambiguity_receipt_id"])
                )
        by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in receipt["prediction_receipts"]:
            if item["strict_match_gt_owner_id"]:
                by_owner[str(item["strict_match_gt_owner_id"])].append(item)
        for owner in owners_by_image[str(trajectory["image_id"])]:
            compatible_rows = [
                prediction
                for prediction in trajectory["predictions"]
                if _compatible_description(prediction, owner, aliases)
            ]
            max_compatible_iou = max((_iou(item["bbox_xyxy"], owner["bbox_xyxy"]) for item in compatible_rows), default=0.0)
            matched = by_owner.get(str(owner["gt_owner_id"]), [])
            owner_id = str(owner["gt_owner_id"])
            owner_is_neutral = owner_id in neutral_owner_ids
            statuses = ["ambiguous_neutral"] if owner_is_neutral else sorted(
                {str(item["strict_match_status"]) for item in matched}
            )
            matrix.append(
                {
                    "schema_version": MATRIX_SCHEMA_VERSION,
                    "gt_owner_id": owner["gt_owner_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "image_id": owner["image_id"],
                    "policy_stratum": trajectory["policy_stratum"],
                    "decode_mode": trajectory["decode_mode"],
                    "seed": trajectory["seed"],
                    "strict_match_presence": any(item["strict_match_status"] == "matched" for item in matched),
                    "automatic_assignment_presence": bool(matched),
                    "matched_pred_row_ids": [item["pred_row_id"] for item in matched],
                    "global_ambiguity_presence": owner_is_neutral,
                    "globally_optimal_edge_receipts": optimal_edges_by_owner.get(owner_id, []),
                    "ambiguity_receipt_ids": sorted(ambiguity_ids_by_owner.get(owner_id, [])),
                    "strict_match_statuses": statuses,
                    "max_semantic_compatible_iou": max_compatible_iou,
                    "semantic_compatible_prediction_count": len(compatible_rows),
                    "semantic_relation": "exact_or_declared_alias" if compatible_rows else "none",
                    "foreign_keys": {"gt_owner_id": owner["gt_owner_id"], "trajectory_id": trajectory["trajectory_id"]},
                    "null_status": {"matched_pred_row_ids": "present" if matched else "empty", "semantic_support": "present" if compatible_rows else "absent"},
                    "source_digests": {"panel": panel_digest["sha256"], "rollout_artifact": trajectory["source_artifact_sha256"]},
                }
            )

    greedy_metrics: dict[str, Any] = {
        "automatic_deterministic_assignment": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "decision_bearing": False,
        },
        "neutral_ambiguity_excluded_owner_presence": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "neutral_assignment_count": 0,
            "neutral_prediction_count": 0,
            "neutral_owner_count": 0,
            "precision_denominator": 0,
            "recall_denominator": 0,
        },
    }
    greedy_owner_presence: set[str] = set()
    greedy_automatic_owner_presence: set[str] = set()
    greedy_neutral_owners: set[str] = set()
    for trajectory in greedy["rows"]:
        receipt = receipts_by_trajectory[trajectory["trajectory_id"]]
        row_metrics = _metrics(len(owners_by_image[trajectory["image_id"]]), len(trajectory["predictions"]), receipt)
        for field in greedy_metrics:
            for key, value in row_metrics[field].items():
                if isinstance(value, bool):
                    greedy_metrics[field][key] = value
                else:
                    greedy_metrics[field][key] += value
        greedy_owner_presence.update(receipt["committed_owner_ids"])
        greedy_automatic_owner_presence.update(receipt["assigned_owner_ids"])
        greedy_neutral_owners.update(receipt["neutral_owner_ids"])
    sampled_owner_presence: set[str] = set()
    sampled_automatic_owner_presence: set[str] = set()
    sampled_neutral_owners: set[str] = set()
    for artifact in sampled:
        for trajectory in artifact["rows"]:
            receipt = receipts_by_trajectory[trajectory["trajectory_id"]]
            sampled_owner_presence.update(receipt["committed_owner_ids"])
            sampled_automatic_owner_presence.update(receipt["assigned_owner_ids"])
            sampled_neutral_owners.update(receipt["neutral_owner_ids"])
    all_owner_ids = {str(owner["gt_owner_id"]) for owner in all_owners}
    greedy_neutral_owners -= greedy_owner_presence
    sampled_neutral_owners -= sampled_owner_presence
    paired_neutral_owners = greedy_neutral_owners | sampled_neutral_owners
    paired_eligible_owner_ids = all_owner_ids - paired_neutral_owners
    paired_greedy_owner_presence = greedy_owner_presence & paired_eligible_owner_ids
    paired_sampled_owner_presence = sampled_owner_presence & paired_eligible_owner_ids
    raw_deterministic_rescued = sampled_automatic_owner_presence - greedy_automatic_owner_presence
    paired_rescued = paired_sampled_owner_presence - paired_greedy_owner_presence

    ambiguity_ids_by_owner: dict[str, list[str]] = defaultdict(list)
    for ambiguity_receipt in ambiguity_ledger:
        for owner_id in ambiguity_receipt["gt_owner_ids"]:
            ambiguity_ids_by_owner[str(owner_id)].append(str(ambiguity_receipt["ambiguity_receipt_id"]))
    for row in owner_ledger:
        owner_id = str(row["gt_owner_id"])
        row["ambiguity_receipt_ids"] = sorted(ambiguity_ids_by_owner.get(owner_id, []))
        row["decision_eligibility"] = {
            "greedy_natural": {
                "eligible": owner_id not in greedy_neutral_owners,
                "status": "globally_ambiguous_neutral"
                if owner_id in greedy_neutral_owners
                else "eligible",
            },
            "k16_any_hit": {
                "eligible": owner_id not in sampled_neutral_owners,
                "status": "globally_ambiguous_neutral"
                if owner_id in sampled_neutral_owners
                else "eligible",
            },
            "greedy_k16_paired": {
                "eligible": owner_id in paired_eligible_owner_ids,
                "status": "excluded_by_either_policy_global_ambiguity"
                if owner_id in paired_neutral_owners
                else "eligible",
            },
        }
    primary_metrics = {
        "estimand": "per_trajectory_any_hit_physical_owner_union",
        "not_estimand": "cross_trajectory_medoid_detection_union",
        "owner_denominator": len(all_owner_ids),
        "matched_rp_1_0_greedy": {
            "owner_presence_count": len(greedy_owner_presence),
            "neutral_owner_count": len(greedy_neutral_owners),
            "neutral_gt_owner_ids": sorted(greedy_neutral_owners),
            "effective_owner_denominator": len(all_owner_ids - greedy_neutral_owners),
            "missed_owner_count": len(all_owner_ids - greedy_owner_presence - greedy_neutral_owners),
            "aggregate_detection_counts": greedy_metrics,
            "automatic_deterministic_assignment": {
                "owner_presence_count": len(greedy_automatic_owner_presence),
                "owner_denominator": len(all_owner_ids),
                "decision_bearing": False,
                "reason": "deterministic representative of globally ambiguous assignments",
            },
        },
        "matched_rp_1_0_k16_any_hit": {
            "owner_presence_count": len(sampled_owner_presence),
            "neutral_owner_count": len(sampled_neutral_owners),
            "neutral_gt_owner_ids": sorted(sampled_neutral_owners),
            "effective_owner_denominator": len(all_owner_ids - sampled_neutral_owners),
            "missed_owner_count": len(all_owner_ids - sampled_owner_presence - sampled_neutral_owners),
            "automatic_assignment_owner_presence_count": len(sampled_automatic_owner_presence),
            "strict_rescued_definition": "ambiguity_neutral_paired_set",
            "strict_rescued_owner_count": len(paired_rescued),
            "strict_rescued_gt_owner_ids": sorted(paired_rescued),
            "raw_absolute_owner_set": {
                "owner_denominator": len(all_owner_ids),
                "greedy_automatic_deterministic_owner_presence_count": len(
                    greedy_automatic_owner_presence
                ),
                "k16_automatic_deterministic_owner_presence_count": len(
                    sampled_automatic_owner_presence
                ),
                "strict_rescued_owner_count": len(raw_deterministic_rescued),
                "strict_rescued_gt_owner_ids": sorted(raw_deterministic_rescued),
                "decision_bearing": False,
                "reason": "includes deterministic representatives of globally ambiguous assignments",
            },
            "ambiguity_neutral_paired_set": {
                "owner_denominator": len(paired_eligible_owner_ids),
                "excluded_neutral_owner_count": len(paired_neutral_owners),
                "excluded_neutral_gt_owner_ids": sorted(paired_neutral_owners),
                "greedy_owner_presence_count": len(paired_greedy_owner_presence),
                "k16_owner_presence_count": len(paired_sampled_owner_presence),
                "k16_missed_owner_count": len(paired_eligible_owner_ids - paired_sampled_owner_presence),
                "strict_rescued_owner_count": len(paired_rescued),
                "strict_rescued_gt_owner_ids": sorted(paired_rescued),
                "decision_bearing": True,
            },
        },
    }
    native_replay = [
        {
            "schema_version": NATIVE_REPLAY_SCHEMA_VERSION,
            "trajectory_id": trajectory["trajectory_id"],
            "image_id": trajectory["image_id"],
            "policy_stratum": trajectory["policy_stratum"],
            "decode_mode": trajectory["decode_mode"],
            "seed": trajectory["seed"],
            "replay_status": "not_attempted_gpu_required",
            "scope": "CPU-only Task-0/1 census; no FP32 model generation was executed",
            "reason": "Native replay requires the declared checkpoint and current FP32 GPU runtime; stored token and termination evidence is retained for later admission.",
            "foreign_keys": {"trajectory_id": trajectory["trajectory_id"]},
            "null_status": {"native_replay_result": "not_attempted_gpu_required"},
            "source_digests": {"rollout_artifact": trajectory["source_artifact_sha256"], "prompt_token_ids": trajectory["prompt_token_ids_sha256"], "generated_token_ids": trajectory["generated_token_ids_sha256"]},
        }
        for trajectory in all_trajectories
    ]
    for ledger in (owner_ledger, prediction_ledger, matrix, native_replay, ambiguity_ledger):
        for row in ledger:
            row["schema_compatibility"] = _schema_compatibility(str(row["schema_version"]))
            row["execution_receipt_content_sha256"] = execution_receipt_content_sha256
            row["source_digests"] = {
                **row["source_digests"],
                "execution_receipt_content": execution_receipt_content_sha256,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_compatibility": _schema_compatibility(SCHEMA_VERSION),
        "panel_digest": panel_digest,
        "greedy": greedy,
        "sampled": sampled,
        "model_file_hashes": model_file_hashes,
        "execution_receipt": execution_receipt,
        "matcher_contract": matcher_contract,
        "owner_ledger": owner_ledger,
        "prediction_ledger": prediction_ledger,
        "owner_trajectory_matrix": matrix,
        "native_replay": native_replay,
        "ambiguity_receipts": ambiguity_ledger,
        "primary_metrics": primary_metrics,
        "production_policy_panel": production,
        "limitations": [
            "No aliases are inferred. The default alias table is empty.",
            "No reviewed auxiliary or unresolved owner mapping was supplied; Task 0/1 records their absence rather than inventing a diagnostic owner.",
            "Native replay is explicitly not attempted in this CPU-only census and cannot admit an intervention.",
            "Sentinel registry sealing requires concrete human-reviewed sentinel provenance and is intentionally outside this source-only ledger builder.",
        ],
    }


def write_census(output_dir: Path, census: Mapping[str, Any]) -> dict[str, Path]:
    _validate_execution_receipt_binding(census)
    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CensusContractError(f"refusing to overwrite nonempty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    matcher_path = destination / "matcher-contract.json"
    _write_json(matcher_path, census["matcher_contract"])
    paths = {
        "execution_receipt": destination / "execution-receipt.json",
        "matcher_contract": matcher_path,
        "owner_ledger": destination / "owner-ledger.jsonl",
        "prediction_ledger": destination / "prediction-row-ledger.jsonl",
        "owner_trajectory_matrix": destination / "owner-trajectory-matrix.jsonl",
        "native_replay": destination / "native-replay.jsonl",
        "ambiguity_receipts": destination / "ambiguity-receipts.jsonl",
        "artifact_manifest": destination / "artifact-manifest.json",
    }
    _write_json(paths["execution_receipt"], census["execution_receipt"])
    _write_jsonl(paths["owner_ledger"], census["owner_ledger"])
    _write_jsonl(paths["prediction_ledger"], census["prediction_ledger"])
    _write_jsonl(paths["owner_trajectory_matrix"], census["owner_trajectory_matrix"])
    _write_jsonl(paths["native_replay"], census["native_replay"])
    _write_jsonl(paths["ambiguity_receipts"], census["ambiguity_receipts"])
    manifest = {
        "schema_version": "sorted-owner-basin-census-artifact-manifest.v2",
        "schema_compatibility": {
            "replaces_schema_version": "sorted-owner-basin-census-artifact-manifest.v1",
            "migration": (
                "consume v2 decision_eligibility, ambiguity receipts, and paired-owner denominators; "
                "v1 deterministic counts are non-decision-bearing"
            ),
        },
        "census_schema_version": census["schema_version"],
        "execution_receipt_content_sha256": census["execution_receipt"][
            "execution_receipt_content_sha256"
        ],
        "artifacts": {name: {"path": path.name, "sha256": _sha256_file(path)} for name, path in paths.items() if name != "artifact_manifest"},
        "sources": {
            "panel": census["panel_digest"],
            "matched_rp_1_0_greedy": {"path": census["greedy"]["path"], "sha256": census["greedy"]["sha256"]},
            "matched_rp_1_0_sampled_shards": [{"path": item["path"], "sha256": item["sha256"]} for item in census["sampled"]],
            "model_component_files": census["model_file_hashes"],
            "optional_production_rp_1_10_manifest": census["production_policy_panel"],
        },
        "primary_metrics": census["primary_metrics"],
        "forced_continuation_absence": {
            "status": "proved_by_structural_source_scan",
            "artifacts": [census["greedy"]["path"], *[item["path"] for item in census["sampled"]]],
            "forbidden_marker_paths": [],
        },
        "native_replay_scope": "not_attempted_gpu_required for all primary natural trajectories",
        "limitations": census["limitations"],
    }
    _write_json(paths["artifact_manifest"], manifest)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="frozen human-refined-12 JSONL panel")
    parser.add_argument("--greedy-artifact", type=Path, default=DEFAULT_GREEDY, help="matched RP=1.0 greedy rollout artifact")
    parser.add_argument("--sampled-shard", type=Path, action="append", default=None, help="one of the two matched RP=1.0 sampled rollout shards; repeat twice")
    parser.add_argument("--production-manifest", type=Path, default=None, help="optional RP=1.10 production run_manifest.json; recorded separately")
    parser.add_argument("--alias-table", type=Path, default=None, help="optional immutable sorted-owner-basin-aliases.v1 table")
    parser.add_argument("--output-dir", required=True, type=Path, help="new empty output directory")
    parser.add_argument("--allow-fixture-panel", action="store_true", help="test-only: permit a non-12-image synthetic panel while retaining 3084 and fixed seed checks")
    parser.add_argument("--skip-model-content-hash", action="store_true", help="test-only: omit recursive component file digests")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    execution_argv = (
        [str(Path(sys.executable).resolve()), *sys.argv]
        if argv is None
        else [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *raw_argv]
    )
    sampled_paths = tuple(args.sampled_shard) if args.sampled_shard is not None else DEFAULT_SAMPLED_SHARDS
    census = build_census(
        panel_path=args.panel,
        greedy_path=args.greedy_artifact,
        sampled_paths=sampled_paths,
        production_manifest_path=args.production_manifest,
        alias_table_path=args.alias_table,
        require_full_panel=not args.allow_fixture_panel,
        hash_model_components=not args.skip_model_content_hash,
        execution_argv=execution_argv,
    )
    paths = write_census(args.output_dir, census)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
