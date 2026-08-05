#!/usr/bin/env python3
"""Build exact-token Sorted owner-basin contexts and natural chronology on CPU.

The builder never decodes or tokenizes text.  Row boundaries are recovered
only from a sealed structural-token registry and the stored
``generated_token_ids``.  Owner assignments come only from the permanent
census ledgers or explicit reviewed labels already present in those ledgers.
Geometry is never used to infer a B2 collision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any


ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
OWNER_SCHEMA_VERSION = "sorted-owner-basin-owner-ledger.v2"
PREDICTION_SCHEMA_VERSION = "sorted-owner-basin-prediction-row-ledger.v2"
MATRIX_SCHEMA_VERSION = "sorted-owner-basin-owner-trajectory-matrix.v2"
AMBIGUITY_SCHEMA_VERSION = "sorted-owner-basin-ambiguity-receipt.v2"
TASK0_EXECUTION_SCHEMA_VERSION = "sorted-owner-basin-task0-execution-receipt.v2"
TASK0_MANIFEST_SCHEMA_VERSION = "sorted-owner-basin-census-artifact-manifest.v2"
TASK0_CENSUS_SCHEMA_VERSION = "sorted-owner-basin-census.v2"
MATCHER_SCHEMA_VERSION = "sorted-owner-basin-matcher.v2"
NATIVE_REPLAY_SCHEMA_VERSION = "sorted-owner-basin-native-replay.v2"
STRUCTURAL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-structural-token-registry.v1"
SENTINEL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-sentinel-registry.v2"
CONTROL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-control-registry.v2"
SENTINEL_SELECTION_SCHEMA_VERSION = "sorted-owner-basin-sentinel-selection-receipt.v1"
SENTINEL_CONFIRMATION_SCHEMA_VERSION = (
    "sorted-owner-basin-sentinel-selection-confirmation-receipt.v2"
)
CONTEXT_SCHEMA_VERSION = "sorted-owner-basin-first-skip-context.v2"
CHRONOLOGY_SCHEMA_VERSION = "sorted-owner-basin-natural-duplication-chronology.v2"
RECEIPT_SCHEMA_VERSION = "sorted-owner-basin-context-builder-receipt.v2"

UNRESOLVED_SCAN_STATUS = "grounding_or_order_conditioned_accessibility_unresolved"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORCED_KEY_RE = re.compile(
    r"(?:^|[_-])(force(?:d)?|intervention)(?:[_-]|$)", re.IGNORECASE
)

# Frozen experiment-local Qwen coordinate grammar.  These IDs are verified
# against every stored generated sequence; no tokenizer is loaded.
EXPERIMENT_STRUCTURAL_RULES = {
    "object_ref_start": 151646,
    "object_ref_end": 151647,
    "box_start": 151648,
    "box_end": 151649,
    "coordinate_min": 151670,
    "coordinate_max": 152669,
    "terminal_storage": "excluded",
    "terminal_token_ids": [],
    "stop_reason": "im_end",
    "short_gap_max": 2,
    "cyclic_min_occurrences": 3,
}


class ContextContractError(ValueError):
    """Raised before stale or semantically incompatible inputs are admitted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextContractError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContextContractError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContextContractError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContextContractError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: Any, label: str) -> str:
    result = _string(value, label)
    if not _SHA256_RE.fullmatch(result):
        raise ContextContractError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise ContextContractError(f"{label} is not valid JSON: {path}") from exc


def _read_jsonl(
    path: Path, label: str, *, allow_empty: bool = False
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ContextContractError(f"{label} line {line_number} is blank")
        try:
            rows.append(_mapping(json.loads(line), f"{label} line {line_number}"))
        except json.JSONDecodeError as exc:
            raise ContextContractError(
                f"{label} line {line_number} is not valid JSON"
            ) from exc
    if not rows and not allow_empty:
        raise ContextContractError(f"{label} must contain at least one row")
    return rows


def _canonical_image_id(value: Any) -> str:
    text = str(value)
    if text.isdigit():
        return str(int(text))
    match = re.search(r"(?:^|_)(\d+)$", text)
    return str(int(match.group(1))) if match else text


def _find_forced_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _FORCED_KEY_RE.search(str(key)):
                found.append(child_path)
            if str(key) in {"experiment_mode", "mode", "generation_mode"}:
                if isinstance(child, str) and (
                    "forced" in child.lower() or "intervention" in child.lower()
                ):
                    found.append(child_path)
            found.extend(_find_forced_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forced_paths(child, f"{path}[{index}]"))
    return found


def _verify_expected_digest(path: Path, expected: str, label: str) -> str:
    expected_digest = _digest(expected, f"expected {label} digest")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ContextContractError(f"{label} digest does not match the expected digest")
    return actual


def _validate_source_digests(
    registry: Mapping[str, Any], actual: Mapping[str, str], label: str
) -> None:
    declared = _mapping(registry.get("source_digests"), f"{label}.source_digests")
    if set(declared) != set(actual):
        raise ContextContractError(
            f"{label}.source_digests must contain exactly {sorted(actual)}"
        )
    for name, digest in actual.items():
        if _digest(declared.get(name), f"{label}.source_digests.{name}") != digest:
            raise ContextContractError(f"{label} has a stale {name} digest")


def _task0_receipt_content_digest(receipt: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "execution_receipt_content_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_task0_v2_bundle(
    input_paths: Mapping[str, Path], actual: Mapping[str, str]
) -> dict[str, Any]:
    ledger_names = ("owner_ledger", "prediction_row_ledger", "owner_trajectory_matrix")
    parents = {input_paths[name].parent for name in ledger_names}
    if len(parents) != 1:
        raise ContextContractError(
            "the three census ledgers must share one Task-0 root"
        )
    root = next(iter(parents))
    implicit_paths = {
        "task0_census_artifact_manifest": root / "artifact-manifest.json",
        "task0_execution_receipt": root / "execution-receipt.json",
        "task0_matcher_contract": root / "matcher-contract.json",
        "task0_native_replay": root / "native-replay.jsonl",
        "task0_ambiguity_receipts": root / "ambiguity-receipts.jsonl",
    }
    missing = [path.name for path in implicit_paths.values() if not path.is_file()]
    if missing:
        raise ContextContractError(
            "Task-0-v2 bundle is incomplete; missing: " + ", ".join(sorted(missing))
        )
    manifest_path = implicit_paths["task0_census_artifact_manifest"]
    manifest = _read_json(manifest_path, "Task-0 census artifact manifest")
    if manifest.get("schema_version") != TASK0_MANIFEST_SCHEMA_VERSION:
        raise ContextContractError(
            "Task-0 census manifest must be v2; v1 fallback is forbidden"
        )
    if manifest.get("census_schema_version") != TASK0_CENSUS_SCHEMA_VERSION:
        raise ContextContractError("Task-0 manifest does not bind the v2 census schema")
    forced_absence = _mapping(
        manifest.get("forced_continuation_absence"),
        "Task-0 manifest forced-continuation absence",
    )
    if (
        forced_absence.get("status") != "proved_by_structural_source_scan"
        or forced_absence.get("forbidden_marker_paths") != []
    ):
        raise ContextContractError(
            "Task-0 manifest does not prove natural-only sources"
        )
    implicit_digests = {
        name: sha256_file(path) for name, path in implicit_paths.items()
    }
    artifacts = _mapping(manifest.get("artifacts"), "Task-0 manifest.artifacts")
    manifest_names = {
        "owner_ledger": "owner_ledger",
        "prediction_row_ledger": "prediction_ledger",
        "owner_trajectory_matrix": "owner_trajectory_matrix",
        "task0_execution_receipt": "execution_receipt",
        "task0_matcher_contract": "matcher_contract",
        "task0_native_replay": "native_replay",
        "task0_ambiguity_receipts": "ambiguity_receipts",
    }
    for input_name, artifact_name in manifest_names.items():
        artifact = _mapping(
            artifacts.get(artifact_name), f"Task-0 manifest.{artifact_name}"
        )
        path = input_paths.get(input_name, implicit_paths.get(input_name))
        if path is None or artifact.get("path") != path.name:
            raise ContextContractError(
                f"Task-0 manifest has a foreign {input_name} path"
            )
        digest = actual.get(input_name, implicit_digests.get(input_name))
        if artifact.get("sha256") != digest:
            raise ContextContractError(
                f"Task-0 manifest has a stale {input_name} digest"
            )
    receipt = _read_json(
        implicit_paths["task0_execution_receipt"], "Task-0 execution receipt"
    )
    if receipt.get("schema_version") != TASK0_EXECUTION_SCHEMA_VERSION:
        raise ContextContractError("Task-0 execution receipt must be v2")
    if receipt.get("execution_status") != "completed":
        raise ContextContractError("Task-0 execution receipt is not completed")
    if (
        receipt.get("execution_surface")
        != "deterministic_cpu_census_no_model_inference"
    ):
        raise ContextContractError(
            "Task-0 execution receipt has a foreign execution surface"
        )
    content_digest = _digest(
        receipt.get("execution_receipt_content_sha256"),
        "Task-0 execution receipt content digest",
    )
    if _task0_receipt_content_digest(receipt) != content_digest:
        raise ContextContractError("Task-0 execution receipt content digest is invalid")
    if manifest.get("execution_receipt_content_sha256") != content_digest:
        raise ContextContractError(
            "Task-0 manifest disagrees with execution receipt content"
        )
    matcher = _read_json(
        implicit_paths["task0_matcher_contract"], "Task-0 matcher contract"
    )
    if matcher.get("schema_version") != MATCHER_SCHEMA_VERSION:
        raise ContextContractError("Task-0 matcher contract must be v2")
    if matcher.get("execution_receipt_content_sha256") != content_digest:
        raise ContextContractError(
            "Task-0 matcher is not bound to the execution receipt"
        )
    ambiguity_rows = _read_jsonl(
        implicit_paths["task0_ambiguity_receipts"],
        "Task-0 ambiguity receipts",
        allow_empty=True,
    )
    ambiguity_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(ambiguity_rows):
        if row.get("schema_version") != AMBIGUITY_SCHEMA_VERSION:
            raise ContextContractError(f"ambiguity receipt row {index} must be v2")
        receipt_id = _string(
            row.get("ambiguity_receipt_id"),
            f"ambiguity receipt row {index}.ambiguity_receipt_id",
        )
        if receipt_id in ambiguity_by_id:
            raise ContextContractError(f"duplicate ambiguity receipt {receipt_id}")
        if row.get("execution_receipt_content_sha256") != content_digest:
            raise ContextContractError(
                f"ambiguity receipt {receipt_id} has a stale receipt binding"
            )
        ambiguity_by_id[receipt_id] = row
    native_rows = _read_jsonl(
        implicit_paths["task0_native_replay"], "Task-0 native replay"
    )
    for index, row in enumerate(native_rows):
        if row.get("schema_version") != NATIVE_REPLAY_SCHEMA_VERSION:
            raise ContextContractError(f"native replay row {index} must be v2")
        if row.get("execution_receipt_content_sha256") != content_digest:
            raise ContextContractError(
                f"native replay row {index} has a stale receipt binding"
            )
    sources = _mapping(manifest.get("sources"), "Task-0 manifest.sources")
    source_greedy = _mapping(
        sources.get("matched_rp_1_0_greedy"), "Task-0 manifest matched greedy"
    )
    if source_greedy.get("sha256") != actual["greedy"]:
        raise ContextContractError("Task-0 manifest has a stale greedy digest")
    receipt_inputs = _mapping(receipt.get("inputs"), "Task-0 receipt.inputs")
    receipt_greedy = _mapping(
        receipt_inputs.get("greedy_artifact"), "Task-0 receipt greedy artifact"
    )
    if receipt_greedy.get("sha256") != actual["greedy"]:
        raise ContextContractError("Task-0 execution receipt has a stale greedy digest")
    panel = _mapping(sources.get("panel"), "Task-0 manifest panel")
    sampled = _list(
        sources.get("matched_rp_1_0_sampled_shards"), "Task-0 manifest sampled shards"
    )
    if len(sampled) != 2:
        raise ContextContractError(
            "Task-0 manifest must bind exactly two sampled shards"
        )
    registry_sources = {
        "human_refined_panel_sha256": _digest(
            panel.get("sha256"), "Task-0 panel digest"
        ),
        "matched_rp_1_0_greedy_sha256": actual["greedy"],
        "matched_rp_1_0_sampled_shard_0_sha256": _digest(
            _mapping(sampled[0], "Task-0 sampled shard 0").get("sha256"),
            "Task-0 sampled shard 0 digest",
        ),
        "matched_rp_1_0_sampled_shard_1_sha256": _digest(
            _mapping(sampled[1], "Task-0 sampled shard 1").get("sha256"),
            "Task-0 sampled shard 1 digest",
        ),
    }
    return {
        "root": str(root.resolve()),
        "manifest": manifest,
        "manifest_sha256": implicit_digests["task0_census_artifact_manifest"],
        "execution_receipt": receipt,
        "execution_receipt_content_sha256": content_digest,
        "ambiguity_by_id": ambiguity_by_id,
        "native_replay_rows": native_rows,
        "registry_sources": registry_sources,
        "implicit_digests": implicit_digests,
    }


def _validate_frozen_registries(
    sentinel: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    sentinel_path: Path,
    actual: Mapping[str, str],
    task0: Mapping[str, Any],
    owners: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    manifest_digest = str(task0["manifest_sha256"])
    manifest_sources = _mapping(task0["registry_sources"], "Task0 registry sources")
    receipt_content = str(task0["execution_receipt_content_sha256"])
    implicit = _mapping(task0["implicit_digests"], "Task0 implicit digests")
    task0_root = str(task0["root"])
    sentinel_sources = _mapping(
        sentinel.get("source_digests"), "sentinel registry.source_digests"
    )
    expected_sentinel = {
        **dict(manifest_sources),
        "task0_v2_root": task0_root,
        "task0_census_artifact_manifest_sha256": manifest_digest,
        "task0_execution_receipt_content_sha256": receipt_content,
        "task0_execution_receipt_file_sha256": implicit["task0_execution_receipt"],
        "task0_owner_ledger_sha256": actual["owner_ledger"],
        "task0_owner_trajectory_matrix_sha256": actual["owner_trajectory_matrix"],
    }
    if set(sentinel_sources) != set(expected_sentinel):
        raise ContextContractError(
            "sentinel registry has an unexpected source-digest contract"
        )
    for name, digest in expected_sentinel.items():
        if sentinel_sources.get(name) != digest:
            raise ContextContractError(f"sentinel registry has a stale {name}")

    control_sources = _mapping(
        control.get("source_digests"), "control registry.source_digests"
    )
    expected_control = {
        "task0_v2_root": task0_root,
        "task0_census_artifact_manifest_sha256": manifest_digest,
        "task0_execution_receipt_content_sha256": receipt_content,
        "task0_execution_receipt_file_sha256": implicit["task0_execution_receipt"],
        "owner_ledger_sha256": actual["owner_ledger"],
        "owner_trajectory_matrix_sha256": actual["owner_trajectory_matrix"],
    }
    confirmation_binding = _mapping(
        sentinel.get("selection_confirmation_receipt"),
        "sentinel registry.selection_confirmation_receipt",
    )
    confirmation_path = sentinel_path.parent / _string(
        confirmation_binding.get("path"), "sentinel confirmation receipt path"
    )
    confirmation_path = confirmation_path.resolve(strict=True)
    confirmation_sha256 = sha256_file(confirmation_path)
    if confirmation_binding.get("sha256") != confirmation_sha256:
        raise ContextContractError("sentinel confirmation receipt digest is stale")
    expected_control["sentinel_selection_confirmation_receipt_sha256"] = (
        confirmation_sha256
    )
    optional_control_source_digests = {"candidate_bank_foil_review_sha256"}
    control_source_names = set(control_sources)
    if not set(expected_control).issubset(control_source_names) or (
        control_source_names - set(expected_control) - optional_control_source_digests
    ):
        raise ContextContractError(
            "control registry has an unexpected source-digest contract"
        )
    for name, digest in expected_control.items():
        if control_sources.get(name) != digest:
            raise ContextContractError(f"control registry has a stale {name}")
    if "candidate_bank_foil_review_sha256" in control_sources:
        _digest(
            control_sources["candidate_bank_foil_review_sha256"],
            "control registry.source_digests.candidate_bank_foil_review_sha256",
        )

    selection_binding = _mapping(
        sentinel.get("selection_receipt"), "sentinel registry.selection_receipt"
    )
    selection_path = sentinel_path.parent / _string(
        selection_binding.get("path"), "sentinel selection receipt path"
    )
    selection_path = selection_path.resolve(strict=True)
    selection_sha256 = sha256_file(selection_path)
    if selection_binding.get("sha256") != selection_sha256:
        raise ContextContractError("sentinel selection receipt digest is stale")
    selection = _read_json(selection_path, "sentinel selection receipt")
    if selection.get("schema_version") != SENTINEL_SELECTION_SCHEMA_VERSION:
        raise ContextContractError("sentinel selection receipt schema is unsupported")
    if (
        selection.get("selection_status")
        != "lead_reviewed_and_frozen_before_landscape_scoring"
    ):
        raise ContextContractError("sentinel selection receipt status is not frozen")
    selection_sources = _mapping(
        selection.get("source_artifacts"), "sentinel selection receipt sources"
    )
    for name, digest in manifest_sources.items():
        if selection_sources.get(name) != digest:
            raise ContextContractError(
                f"sentinel selection receipt has stale source {name}"
            )

    confirmation = _read_json(confirmation_path, "sentinel confirmation receipt")
    if confirmation.get("schema_version") != SENTINEL_CONFIRMATION_SCHEMA_VERSION:
        raise ContextContractError("sentinel confirmation receipt must be v2")
    if (
        confirmation.get("confirmation_status")
        != "lead_reconfirmed_against_final_task0_v2_before_landscape_scoring"
    ):
        raise ContextContractError("sentinel confirmation receipt status is not frozen")
    original = _mapping(
        confirmation.get("original_selection_receipt"),
        "sentinel confirmation original selection receipt",
    )
    if (
        original.get("path") != selection_path.name
        or original.get("sha256") != selection_sha256
    ):
        raise ContextContractError(
            "sentinel confirmation does not bind the original receipt"
        )
    final_task0 = _mapping(
        confirmation.get("final_task0_v2"), "sentinel confirmation final Task0-v2"
    )
    expected_final = {
        "root": task0_root,
        "artifact_manifest_sha256": manifest_digest,
        "execution_receipt_content_sha256": receipt_content,
        "execution_receipt_file_sha256": implicit["task0_execution_receipt"],
        "owner_ledger_sha256": actual["owner_ledger"],
        "owner_trajectory_matrix_sha256": actual["owner_trajectory_matrix"],
    }
    for name, digest in expected_final.items():
        if final_task0.get(name) != digest:
            raise ContextContractError(
                f"sentinel confirmation has stale final Task0 field {name}"
            )
    registry_owner_ids = {
        str(item.get("gt_owner_id"))
        for item in _list(sentinel.get("sentinels"), "sentinel registry.sentinels")
    }
    confirmed = _list(
        confirmation.get("confirmed_sentinels"), "sentinel confirmation sentinels"
    )
    confirmed_ids: set[str] = set()
    for raw in confirmed:
        item = _mapping(raw, "confirmed sentinel")
        owner_id = _string(item.get("gt_owner_id"), "confirmed sentinel owner")
        confirmed_ids.add(owner_id)
        if owner_id not in owners or not owners[owner_id]["paired_decision_eligible"]:
            raise ContextContractError(
                f"confirmed sentinel {owner_id} is not decision eligible"
            )
        if (
            item.get("decision_eligible") is not True
            or item.get("trajectory_count") != 17
            or item.get("strict_match_count") != 0
            or float(item.get("max_semantic_compatible_iou", -1.0)) != 0.0
        ):
            raise ContextContractError(
                f"confirmed sentinel {owner_id} violates confirmation"
            )
        if owners[owner_id].get("ambiguity_receipt_ids"):
            raise ContextContractError(
                f"confirmed sentinel {owner_id} remains ambiguous"
            )
    if confirmed_ids != registry_owner_ids:
        raise ContextContractError(
            "sentinel registry and confirmation membership disagree"
        )
    return {
        "sentinel_selection_receipt": selection_sha256,
        "sentinel_selection_confirmation_receipt": confirmation_sha256,
    }


def _validate_ids(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    ids = _list(value, label)
    if not allow_empty and not ids:
        raise ContextContractError(f"{label} must not be empty")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids
    ):
        raise ContextContractError(
            f"{label} must contain non-negative integer token IDs"
        )
    return [int(item) for item in ids]


def _validate_declared_token_hash(
    row: Mapping[str, Any], field: str, label: str
) -> list[int]:
    ids = _validate_ids(row.get(field), f"{label}.{field}")
    declared = _digest(row.get(f"{field}_sha256"), f"{label}.{field}_sha256")
    if declared != sha256_json(ids):
        raise ContextContractError(f"{label}.{field}_sha256 does not bind exact IDs")
    return ids


def _parse_structural_registry(
    registry: Mapping[str, Any], greedy_digest: str
) -> dict[str, Any]:
    if registry.get("schema_version") != STRUCTURAL_REGISTRY_SCHEMA_VERSION:
        raise ContextContractError("unsupported structural-token registry schema")
    if registry.get("status") != "sealed":
        raise ContextContractError("structural-token registry must be sealed")
    _validate_source_digests(registry, {"greedy": greedy_digest}, "structural registry")
    tokens = _mapping(registry.get("tokens"), "structural registry.tokens")
    parsed = {
        name: _integer(tokens.get(name), f"structural registry.tokens.{name}")
        for name in ("object_ref_start", "object_ref_end", "box_start", "box_end")
    }
    if len(set(parsed.values())) != len(parsed):
        raise ContextContractError("structural marker token IDs must be distinct")
    coordinate_min = _integer(
        tokens.get("coordinate_token_id_min"),
        "structural registry.tokens.coordinate_token_id_min",
    )
    coordinate_max = _integer(
        tokens.get("coordinate_token_id_max"),
        "structural registry.tokens.coordinate_token_id_max",
    )
    if coordinate_max < coordinate_min:
        raise ContextContractError("coordinate token range is inverted")
    if set(parsed.values()) & set(range(coordinate_min, coordinate_max + 1)):
        raise ContextContractError(
            "structural markers overlap the coordinate token range"
        )
    terminal = _mapping(registry.get("terminal"), "structural registry.terminal")
    storage = terminal.get("storage")
    if storage not in {"included", "excluded"}:
        raise ContextContractError("terminal.storage must be 'included' or 'excluded'")
    token_ids = _validate_ids(
        terminal.get("token_ids"),
        "structural registry.terminal.token_ids",
        allow_empty=storage == "excluded",
    )
    if storage == "included" and not token_ids:
        raise ContextContractError("included terminal requires token_ids")
    stop_reason = _string(
        terminal.get("stop_reason"), "structural registry.terminal.stop_reason"
    )
    chronology = _mapping(registry.get("chronology"), "structural registry.chronology")
    return {
        **parsed,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "terminal_storage": storage,
        "terminal_token_ids": token_ids,
        "stop_reason": stop_reason,
        "short_gap_max": _integer(
            chronology.get("short_gap_max_intervening_rows"),
            "structural registry.chronology.short_gap_max_intervening_rows",
            minimum=1,
        ),
        "cyclic_min_occurrences": _integer(
            chronology.get("cyclic_min_occurrences"),
            "structural registry.chronology.cyclic_min_occurrences",
            minimum=3,
        ),
    }


def _split_exact_rows(
    generated_ids: Sequence[int], rules: Mapping[str, Any], label: str
) -> tuple[list[tuple[int, int]], int]:
    terminal_start = len(generated_ids)
    row_ids = list(generated_ids)
    if rules["terminal_storage"] == "included":
        terminal = list(rules["terminal_token_ids"])
        if row_ids[-len(terminal) :] != terminal:
            raise ContextContractError(
                f"{label} lacks the declared included terminal IDs"
            )
        terminal_start -= len(terminal)
        row_ids = row_ids[:terminal_start]
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(row_ids):
        start = position
        if row_ids[position] != rules["object_ref_start"]:
            raise ContextContractError(
                f"{label} has non-row token {row_ids[position]} at generated index {position}"
            )
        position += 1
        while position < len(row_ids) and row_ids[position] not in {
            rules["object_ref_start"],
            rules["object_ref_end"],
        }:
            position += 1
        if position >= len(row_ids) or row_ids[position] != rules["object_ref_end"]:
            raise ContextContractError(f"{label} row {len(spans)} lacks object_ref_end")
        if position == start + 1:
            raise ContextContractError(
                f"{label} row {len(spans)} has an empty description"
            )
        position += 1
        if position >= len(row_ids) or row_ids[position] != rules["box_start"]:
            raise ContextContractError(f"{label} row {len(spans)} lacks box_start")
        position += 1
        coordinates = row_ids[position : position + 4]
        if len(coordinates) != 4 or any(
            token_id < rules["coordinate_min"] or token_id > rules["coordinate_max"]
            for token_id in coordinates
        ):
            raise ContextContractError(
                f"{label} row {len(spans)} lacks four declared coordinate tokens"
            )
        position += 4
        if position >= len(row_ids) or row_ids[position] != rules["box_end"]:
            raise ContextContractError(f"{label} row {len(spans)} lacks box_end")
        position += 1
        spans.append((start, position))
    return spans, terminal_start


def _load_greedy(
    path: Path, digest: str, structural: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, "greedy artifact")
    forced = _find_forced_paths(payload)
    if forced:
        raise ContextContractError(
            "greedy artifact contains forced-continuation structure: "
            + ", ".join(forced[:3])
        )
    if payload.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise ContextContractError("unsupported greedy artifact schema")
    config = _mapping(payload.get("config"), "greedy artifact.config")
    if config.get("decode_mode") != "greedy" or config.get("seeds") != [0]:
        raise ContextContractError("source must be the natural seed-0 greedy artifact")
    if float(config.get("repetition_penalty", -1)) != 1.0:
        raise ContextContractError(
            "source greedy artifact must use repetition penalty 1.0"
        )
    rollouts = _list(payload.get("rollouts"), "greedy artifact.rollouts")
    if payload.get("rollout_count") != len(rollouts):
        raise ContextContractError("greedy rollout_count disagrees with rollouts")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rollouts):
        row = _mapping(value, f"greedy rollout {index}")
        image_id = _canonical_image_id(row.get("image_id"))
        if image_id in result:
            raise ContextContractError(f"duplicate greedy image {image_id}")
        if row.get("decode_mode") != "greedy" or row.get("seed") != 0:
            raise ContextContractError(
                f"greedy rollout {index} has foreign policy metadata"
            )
        if row.get("stop_reason") != structural["stop_reason"]:
            raise ContextContractError(
                f"greedy rollout {index} has an unexpected stop reason"
            )
        prompt_ids = _validate_declared_token_hash(
            row, "prompt_token_ids", f"rollout {index}"
        )
        generated_ids = _validate_declared_token_hash(
            row, "generated_token_ids", f"rollout {index}"
        )
        spans, terminal_start = _split_exact_rows(
            generated_ids, structural, f"greedy rollout {index}"
        )
        parsed = _mapping(row.get("predictions"), f"greedy rollout {index}.predictions")
        source_indices: list[int] = []
        for key in ("predictions", "dropped_predictions"):
            for item in _list(
                parsed.get(key), f"greedy rollout {index}.predictions.{key}"
            ):
                generated_order = _mapping(item, "parsed prediction").get(
                    "generated_order"
                )
                if isinstance(generated_order, int):
                    source_indices.append(generated_order)
        if sorted(source_indices) != list(range(len(spans))):
            raise ContextContractError(
                f"greedy rollout {index} parsed row chronology disagrees with structural rows"
            )
        result[image_id] = {
            "image_id": image_id,
            "trajectory_id": f"trajectory:sorted:rp1.00:greedy:0:{image_id}",
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "row_spans": spans,
            "terminal_start": terminal_start,
            "prompt_sha256": sha256_json(prompt_ids),
            "generated_sha256": sha256_json(generated_ids),
            "source_artifact_sha256": digest,
            "stop_reason": str(row["stop_reason"]),
        }
    if not result:
        raise ContextContractError("greedy artifact contains no rollouts")
    return result


def _load_owners(
    rows: Sequence[Mapping[str, Any]],
    receipt_content_sha256: str,
    ambiguity_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    owners: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != OWNER_SCHEMA_VERSION:
            raise ContextContractError(f"owner ledger row {index} must be v2")
        owner_id = _string(
            row.get("gt_owner_id"), f"owner ledger row {index}.gt_owner_id"
        )
        if owner_id in owners:
            raise ContextContractError(f"duplicate gt_owner_id {owner_id}")
        bbox = _list(row.get("bbox_xyxy"), f"owner ledger row {index}.bbox_xyxy")
        if len(bbox) != 4 or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in bbox
        ):
            raise ContextContractError(f"owner {owner_id} has an invalid bbox_xyxy")
        image_id = _canonical_image_id(row.get("image_id"))
        if row.get("execution_receipt_content_sha256") != receipt_content_sha256:
            raise ContextContractError(f"owner {owner_id} has a stale receipt binding")
        source_digests = _mapping(
            row.get("source_digests"), f"owner {owner_id}.source_digests"
        )
        if source_digests.get("execution_receipt_content") != receipt_content_sha256:
            raise ContextContractError(
                f"owner {owner_id} lacks the v2 receipt source digest"
            )
        decision = _mapping(
            row.get("decision_eligibility"), f"owner {owner_id}.decision_eligibility"
        )
        greedy_decision = _mapping(
            decision.get("greedy_natural"),
            f"owner {owner_id}.decision_eligibility.greedy_natural",
        )
        paired_decision = _mapping(
            decision.get("greedy_k16_paired"),
            f"owner {owner_id}.decision_eligibility.greedy_k16_paired",
        )
        if not isinstance(greedy_decision.get("eligible"), bool) or not isinstance(
            paired_decision.get("eligible"), bool
        ):
            raise ContextContractError(
                f"owner {owner_id} has malformed decision eligibility"
            )
        ambiguity_ids = _list(
            row.get("ambiguity_receipt_ids"), f"owner {owner_id}.ambiguity_receipt_ids"
        )
        for receipt_id in ambiguity_ids:
            if receipt_id not in ambiguity_by_id:
                raise ContextContractError(
                    f"owner {owner_id} references unknown ambiguity receipt"
                )
            if owner_id not in ambiguity_by_id[receipt_id].get("gt_owner_ids", []):
                raise ContextContractError(
                    f"owner {owner_id} ambiguity foreign key disagrees"
                )
        owners[owner_id] = {
            **dict(row),
            "image_id": image_id,
            "original_annotation_index": _integer(
                row.get("original_annotation_index"),
                f"owner {owner_id}.original_annotation_index",
            ),
            "bbox_xyxy": [float(item) for item in bbox],
            "greedy_decision_eligible": bool(greedy_decision["eligible"]),
            "paired_decision_eligible": bool(paired_decision["eligible"]),
        }
    return owners


def _load_predictions(
    rows: Sequence[Mapping[str, Any]],
    greedy: Mapping[str, Mapping[str, Any]],
    greedy_digest: str,
    receipt_content_sha256: str,
    ambiguity_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("schema_version") != PREDICTION_SCHEMA_VERSION:
            raise ContextContractError(f"prediction ledger row {index} must be v2")
        if row.get("decode_mode") != "greedy" or row.get("seed") != 0:
            continue
        pred_id = _string(row.get("pred_row_id"), f"prediction row {index}.pred_row_id")
        if pred_id in by_id:
            raise ContextContractError(f"duplicate pred_row_id {pred_id}")
        image_id = _canonical_image_id(row.get("image_id"))
        if image_id not in greedy:
            raise ContextContractError(
                f"prediction row {pred_id} references an unknown greedy image"
            )
        source_digests = _mapping(
            row.get("source_digests"), f"prediction row {pred_id}.source_digests"
        )
        if source_digests.get("rollout_artifact") != greedy_digest:
            raise ContextContractError(
                f"prediction row {pred_id} has a stale rollout digest"
            )
        if (
            row.get("execution_receipt_content_sha256") != receipt_content_sha256
            or source_digests.get("execution_receipt_content") != receipt_content_sha256
        ):
            raise ContextContractError(
                f"prediction row {pred_id} has a stale v2 receipt binding"
            )
        ambiguity_ids = _list(
            row.get("ambiguity_receipt_ids"),
            f"prediction row {pred_id}.ambiguity_receipt_ids",
        )
        for receipt_id in ambiguity_ids:
            if receipt_id not in ambiguity_by_id:
                raise ContextContractError(
                    f"prediction row {pred_id} references unknown ambiguity"
                )
            if pred_id not in ambiguity_by_id[receipt_id].get("pred_row_ids", []):
                raise ContextContractError(
                    f"prediction row {pred_id} ambiguity foreign key disagrees"
                )
        if row.get("strict_match_status") == "ambiguous_neutral" and not ambiguity_ids:
            raise ContextContractError(
                f"ambiguous prediction row {pred_id} lacks an ambiguity receipt"
            )
        trajectory_id = _string(
            row.get("trajectory_id"), f"prediction row {pred_id}.trajectory_id"
        )
        if trajectory_id != greedy[image_id]["trajectory_id"]:
            raise ContextContractError(
                f"prediction row {pred_id} has a foreign trajectory"
            )
        normalized = {
            **dict(row),
            "image_id": image_id,
            "original_row_index": _integer(
                row.get("original_row_index"),
                f"prediction row {pred_id}.original_row_index",
            ),
        }
        by_id[pred_id] = normalized
        by_image[image_id].append(normalized)
    for image_id, image_rows in by_image.items():
        image_rows.sort(key=lambda item: int(item["original_row_index"]))
        expected = list(range(len(greedy[image_id]["row_spans"])))
        observed = [int(item["original_row_index"]) for item in image_rows]
        if observed != expected:
            raise ContextContractError(
                f"prediction ledger chronology for image {image_id} is not immutable and complete"
            )
    if set(by_image) != set(greedy):
        raise ContextContractError(
            "prediction ledger does not cover every greedy image"
        )
    return by_id, dict(by_image)


def _validate_matrix(
    rows: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    greedy: Mapping[str, Mapping[str, Any]],
    greedy_digest: str,
    receipt_content_sha256: str,
    ambiguity_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    seen: set[tuple[str, str]] = set()
    expected = {
        (owner_id, greedy[str(owner["image_id"])]["trajectory_id"])
        for owner_id, owner in owners.items()
    }
    for index, row in enumerate(rows):
        if row.get("schema_version") != MATRIX_SCHEMA_VERSION:
            raise ContextContractError(f"matrix row {index} must be v2")
        if row.get("decode_mode") != "greedy" or row.get("seed") != 0:
            continue
        owner_id = _string(row.get("gt_owner_id"), f"matrix row {index}.gt_owner_id")
        trajectory_id = _string(
            row.get("trajectory_id"), f"matrix row {index}.trajectory_id"
        )
        key = (owner_id, trajectory_id)
        if owner_id not in owners or key not in expected or key in seen:
            raise ContextContractError(f"matrix row {index} has a stale foreign key")
        seen.add(key)
        source_digests = _mapping(
            row.get("source_digests"), f"matrix row {index}.source_digests"
        )
        if source_digests.get("rollout_artifact") != greedy_digest:
            raise ContextContractError(f"matrix row {index} has a stale rollout digest")
        if (
            row.get("execution_receipt_content_sha256") != receipt_content_sha256
            or source_digests.get("execution_receipt_content") != receipt_content_sha256
        ):
            raise ContextContractError(
                f"matrix row {index} has a stale v2 receipt binding"
            )
        ambiguity_ids = _list(
            row.get("ambiguity_receipt_ids"),
            f"matrix row {index}.ambiguity_receipt_ids",
        )
        for receipt_id in ambiguity_ids:
            if receipt_id not in ambiguity_by_id:
                raise ContextContractError(
                    f"matrix row {index} references unknown ambiguity"
                )
            if owner_id not in ambiguity_by_id[receipt_id].get("gt_owner_ids", []):
                raise ContextContractError(
                    f"matrix row {index} ambiguity foreign key disagrees"
                )
        if bool(row.get("global_ambiguity_presence")) != (
            not owners[owner_id]["greedy_decision_eligible"]
        ):
            raise ContextContractError(
                f"matrix row {index} disagrees with owner decision eligibility"
            )
        matched_ids = _list(
            row.get("matched_pred_row_ids"), f"matrix row {index}.matched_pred_row_ids"
        )
        for pred_id in matched_ids:
            if pred_id not in predictions:
                raise ContextContractError(
                    f"matrix row {index} references unknown prediction {pred_id}"
                )
            if predictions[pred_id].get("strict_match_gt_owner_id") != owner_id:
                raise ContextContractError(
                    f"matrix row {index} disagrees with prediction assignment"
                )
    if seen != expected:
        raise ContextContractError(
            "owner trajectory matrix lacks complete greedy foreign-key coverage"
        )


def _owner_identity_maps(
    owners: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    identities: dict[str, str] = {}
    by_image: dict[str, list[str]] = defaultdict(list)
    for owner_id, owner in owners.items():
        identities[owner_id] = owner_id
        diagnostic = owner.get("diagnostic_owner_id")
        if isinstance(diagnostic, str) and diagnostic:
            if diagnostic in identities and identities[diagnostic] != owner_id:
                raise ContextContractError(
                    f"diagnostic owner identity {diagnostic} is not unique"
                )
            identities[diagnostic] = owner_id
        by_image[str(owner["image_id"])].append(owner_id)
    for image_id, owner_ids in by_image.items():
        owner_ids.sort(
            key=lambda owner_id: (
                float(owners[owner_id]["bbox_xyxy"][1]),
                float(owners[owner_id]["bbox_xyxy"][0]),
                int(owners[owner_id]["original_annotation_index"]),
                owner_id,
            )
        )
    return identities, dict(by_image)


def _row_assignment(
    row: Mapping[str, Any],
    identities: Mapping[str, str],
    owners: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, str, str]:
    status = row.get("strict_match_status")
    strict_owner = row.get("strict_match_gt_owner_id")
    if (
        status == "matched"
        and isinstance(strict_owner, str)
        and strict_owner in identities
    ):
        owner_id = identities[strict_owner]
        if not owners[owner_id]["greedy_decision_eligible"]:
            return None, "neutral", "globally_ambiguous_owner_or_row"
        return owner_id, "new_owner", "strict_match"
    physical_relation = row.get("physical_relation")
    assignment_status = row.get("assignment_status", row.get("adjudication_status"))
    diagnostic = row.get("diagnostic_owner_id")
    if (
        physical_relation in {"new_owner", "covered_owner_duplicate"}
        and assignment_status in {"resolved", "reviewed_resolved"}
        and isinstance(diagnostic, str)
        and diagnostic in identities
    ):
        owner_id = identities[diagnostic]
        if not owners[owner_id]["greedy_decision_eligible"]:
            return None, "neutral", "globally_ambiguous_owner_or_row"
        return owner_id, str(physical_relation), "reviewed_assignment"
    if status == "ambiguous_neutral" or physical_relation == "unresolved":
        return None, "neutral", "globally_ambiguous_owner_or_row"
    row_kind = row.get("row_kind")
    if row_kind in {"parser_dropped", "parser_invalid_geometry_or_description"}:
        return None, "contaminating", "malformed_or_parser_dropped_row"
    return None, "contaminating", "unmatched_or_unreviewed_row"


def _prefix_binding(
    trajectory: Mapping[str, Any], generated_end: int
) -> dict[str, Any]:
    prompt = list(trajectory["prompt_ids"])
    generated = list(trajectory["generated_ids"][:generated_end])
    model_input = [*prompt, *generated]
    return {
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": trajectory["prompt_sha256"],
        "self_prefix_generated_token_ids": generated,
        "self_prefix_generated_token_ids_sha256": sha256_json(generated),
        "model_input_token_ids": model_input,
        "model_input_token_ids_sha256": sha256_json(model_input),
    }


def _source_binding(
    trajectory: Mapping[str, Any], upstream: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "prompt_token_ids_sha256": trajectory["prompt_sha256"],
        "generated_token_ids_sha256": trajectory["generated_sha256"],
        "greedy_artifact_sha256": trajectory["source_artifact_sha256"],
        "upstream_artifact_digests": dict(sorted(upstream.items())),
    }


def _context_row(
    *,
    context_id: str,
    context_kind: str,
    owner_id: str,
    image_id: str,
    trajectory: Mapping[str, Any],
    generated_end: int | None,
    upstream: Mapping[str, str],
    status: str,
    eligibility: str,
    source_pred_row_id: str | None,
    registry_id: str | None,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "record_type": "exact_context",
        "context_id": context_id,
        "context_kind": context_kind,
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "policy_stratum": "primary_rp_1.00",
        "decode_mode": "greedy",
        "seed": 0,
        "stop_reason": trajectory["stop_reason"],
        "status": status,
        "eligibility": eligibility,
        "source_pred_row_id": source_pred_row_id,
        "registry_id": registry_id,
        "foreign_keys": {
            "gt_owner_id": owner_id,
            "trajectory_id": trajectory["trajectory_id"],
            "pred_row_id": source_pred_row_id,
        },
        "source_binding": _source_binding(trajectory, upstream),
        "details": dict(details),
    }
    row["exact_prefix"] = (
        _prefix_binding(trajectory, generated_end)
        if generated_end is not None
        else None
    )
    return row


def _contamination_receipt(
    row: Mapping[str, Any], reason: str, owner_id: str | None
) -> dict[str, Any]:
    return {
        "reason": reason,
        "pred_row_id": row["pred_row_id"],
        "original_row_index": row["original_row_index"],
        "row_kind": row.get("row_kind"),
        "strict_match_status": row.get("strict_match_status"),
        "deterministic_gt_owner_id": owner_id,
        "ambiguity_receipt_ids": list(row.get("ambiguity_receipt_ids", [])),
    }


def _candidate_receipt_row(
    *,
    candidate_id: str,
    image_id: str,
    trajectory: Mapping[str, Any],
    upstream: Mapping[str, str],
    status: str,
    reason: str,
    skipped_owner_id: str | None,
    successor_owner_id: str | None,
    successor_row: Mapping[str, Any],
    first_contamination: Mapping[str, Any] | None,
    repair_context_ids: Sequence[str],
    owners: Mapping[str, Mapping[str, Any]],
    contributes_to_candidate_denominator: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "record_type": "first_skip_candidate_receipt",
        "context_id": f"receipt:{candidate_id}",
        "context_kind": "first_skip_candidate_receipt",
        "gt_owner_id": skipped_owner_id,
        "image_id": image_id,
        "policy_stratum": "primary_rp_1.00",
        "decode_mode": "greedy",
        "seed": 0,
        "stop_reason": trajectory["stop_reason"],
        "status": status,
        "eligibility": (
            "repair_candidate_pending_native_replay"
            if status == "admitted_candidate"
            else "neutral_not_repair_eligible"
        ),
        "source_pred_row_id": successor_row["pred_row_id"],
        "registry_id": None,
        "foreign_keys": {
            "gt_owner_id": skipped_owner_id,
            "trajectory_id": trajectory["trajectory_id"],
            "pred_row_id": successor_row["pred_row_id"],
        },
        "source_binding": _source_binding(trajectory, upstream),
        "exact_prefix": None,
        "details": {
            "candidate_id": candidate_id,
            "candidate_reason": reason,
            "first_skipped_gt_owner_id": skipped_owner_id,
            "first_skipped_original_annotation_index": (
                owners[skipped_owner_id]["original_annotation_index"]
                if skipped_owner_id is not None
                else None
            ),
            "native_successor_gt_owner_id": successor_owner_id,
            "native_successor_original_annotation_index": (
                owners[successor_owner_id]["original_annotation_index"]
                if successor_owner_id is not None
                else None
            ),
            "native_successor_pred_row_id": successor_row["pred_row_id"],
            "native_successor_original_row_index": successor_row["original_row_index"],
            "first_contamination": dict(first_contamination)
            if first_contamination is not None
            else None,
            "repair_context_ids": list(repair_context_ids),
            "contributes_to_reconstructible_candidate_denominator": (
                contributes_to_candidate_denominator
            ),
        },
    }


def _build_first_skip_contexts(
    owners: Mapping[str, Mapping[str, Any]],
    owners_by_image: Mapping[str, list[str]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, str],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id, rows in sorted(predictions_by_image.items()):
        order = owners_by_image[image_id]
        rank = {owner_id: index for index, owner_id in enumerate(order)}
        assigned_prefix: list[str] = []
        seen_owners: set[str] = set()
        frozen_first_skips: set[str] = set()
        first_contamination: dict[str, Any] | None = None
        trajectory = greedy[image_id]
        for row in rows:
            row_index = int(row["original_row_index"])
            owner_id, relation, assignment_reason = _row_assignment(
                row, identities, owners
            )
            contamination_reason: str | None = None
            if owner_id is None:
                contamination_reason = assignment_reason
            elif relation == "covered_owner_duplicate" or owner_id in seen_owners:
                contamination_reason = "duplicate_or_repeat_owner"
            elif relation != "new_owner":
                contamination_reason = "non_new_owner_assignment"
            if contamination_reason is not None:
                deterministic_owner_id: str | None = owner_id
                strict_owner = row.get("strict_match_gt_owner_id")
                if (
                    deterministic_owner_id is None
                    and isinstance(strict_owner, str)
                    and strict_owner in identities
                ):
                    deterministic_owner_id = identities[strict_owner]
                contaminated_skipped: list[str] = []
                if (
                    deterministic_owner_id is not None
                    and deterministic_owner_id in rank
                    and contamination_reason != "duplicate_or_repeat_owner"
                ):
                    contaminated_skipped = [
                        item
                        for item in order[: rank[deterministic_owner_id]]
                        if item not in assigned_prefix
                    ]
                contamination = _contamination_receipt(
                    row, contamination_reason, deterministic_owner_id
                )
                if first_contamination is None:
                    first_contamination = contamination
                candidate_id = f"first-skip-rejection:{image_id}:{row['pred_row_id']}"
                output.append(
                    _candidate_receipt_row(
                        candidate_id=candidate_id,
                        image_id=image_id,
                        trajectory=trajectory,
                        upstream=upstream,
                        status="rejected_neutral_history",
                        reason=contamination_reason,
                        skipped_owner_id=(
                            contaminated_skipped[0] if contaminated_skipped else None
                        ),
                        successor_owner_id=deterministic_owner_id,
                        successor_row=row,
                        first_contamination=first_contamination,
                        repair_context_ids=[],
                        owners=owners,
                        contributes_to_candidate_denominator=bool(contaminated_skipped),
                    )
                )
                continue

            if owner_id is not None:
                owner_rank = rank[owner_id]
                expected_prior = order[:owner_rank]
                skipped = [
                    item for item in expected_prior if item not in assigned_prefix
                ]
                exact_prior = [item for item in expected_prior if item not in skipped]
                if skipped:
                    skipped_owner = skipped[0]
                    case_id = (
                        f"first-skip:{image_id}:{row['pred_row_id']}:{skipped_owner}"
                    )
                    pre_id = f"ctx:{case_id}:P_pre"
                    post_id = f"ctx:{case_id}:P_post"
                    rejection_reason: str | None = None
                    if first_contamination is not None:
                        rejection_reason = "earlier_history_contamination"
                    elif len(skipped) != 1:
                        rejection_reason = "multiple_earlier_skipped_owners"
                    elif assigned_prefix != exact_prior:
                        rejection_reason = "noncanonical_prior_owner_order"
                    elif skipped_owner in frozen_first_skips:
                        rejection_reason = "first_skipped_owner_already_frozen"
                    if rejection_reason is not None:
                        rejection = first_contamination or _contamination_receipt(
                            row, rejection_reason, owner_id
                        )
                        output.append(
                            _candidate_receipt_row(
                                candidate_id=case_id,
                                image_id=image_id,
                                trajectory=trajectory,
                                upstream=upstream,
                                status="rejected_neutral_candidate",
                                reason=rejection_reason,
                                skipped_owner_id=skipped_owner,
                                successor_owner_id=owner_id,
                                successor_row=row,
                                first_contamination=rejection,
                                repair_context_ids=[],
                                owners=owners,
                                contributes_to_candidate_denominator=True,
                            )
                        )
                        assigned_prefix.append(owner_id)
                        seen_owners.add(owner_id)
                        continue
                    frozen_first_skips.add(skipped_owner)
                    start, end = trajectory["row_spans"][row_index]
                    common = {
                        "case_id": case_id,
                        "skipped_gt_owner_id": skipped_owner,
                        "skipped_original_annotation_index": owners[skipped_owner][
                            "original_annotation_index"
                        ],
                        "native_successor_gt_owner_id": owner_id,
                        "native_successor_original_annotation_index": owners[owner_id][
                            "original_annotation_index"
                        ],
                        "native_successor_original_row_index": row_index,
                        "prior_assigned_gt_owner_ids": list(assigned_prefix),
                        "reference_order": "(y1,x1,original_annotation_index)",
                    }
                    output.append(
                        _context_row(
                            context_id=pre_id,
                            context_kind="P_pre",
                            owner_id=skipped_owner,
                            image_id=image_id,
                            trajectory=trajectory,
                            generated_end=start,
                            upstream=upstream,
                            status="admitted",
                            eligibility="clean_first_skip_repair_candidate_pending_native_replay",
                            source_pred_row_id=str(row["pred_row_id"]),
                            registry_id=None,
                            details=common,
                        )
                    )
                    output.append(
                        _context_row(
                            context_id=post_id,
                            context_kind="P_post",
                            owner_id=skipped_owner,
                            image_id=image_id,
                            trajectory=trajectory,
                            generated_end=end,
                            upstream=upstream,
                            status="admitted",
                            eligibility="clean_first_skip_repair_candidate_pending_native_replay",
                            source_pred_row_id=str(row["pred_row_id"]),
                            registry_id=None,
                            details=common,
                        )
                    )
                    output.append(
                        _candidate_receipt_row(
                            candidate_id=case_id,
                            image_id=image_id,
                            trajectory=trajectory,
                            upstream=upstream,
                            status="admitted_candidate",
                            reason="clean_unique_first_skip",
                            skipped_owner_id=skipped_owner,
                            successor_owner_id=owner_id,
                            successor_row=row,
                            first_contamination=None,
                            repair_context_ids=[pre_id, post_id],
                            owners=owners,
                            contributes_to_candidate_denominator=True,
                        )
                    )
                assigned_prefix.append(owner_id)
                seen_owners.add(owner_id)
    return output


def _parse_registry_cases(
    registry: Mapping[str, Any], schema: str, field: str, label: str
) -> list[Mapping[str, Any]]:
    if registry.get("schema_version") != schema:
        raise ContextContractError(f"unsupported {label} schema")
    sentinel_is_frozen = (
        schema == SENTINEL_REGISTRY_SCHEMA_VERSION
        and registry.get("selection_status")
        == "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2"
    )
    if not sentinel_is_frozen:
        raise ContextContractError(f"{label} must be frozen against final Task0-v2")
    return [
        _mapping(item, f"{label}.{field}")
        for item in _list(registry.get(field), f"{label}.{field}")
    ]


def _parse_b2_cases(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if registry.get("schema_version") != CONTROL_REGISTRY_SCHEMA_VERSION:
        raise ContextContractError("control registry must be v2; fallback is forbidden")
    if (
        registry.get("status")
        != "lead_frozen_before_scoring_and_resealed_to_final_task0_v2"
    ):
        raise ContextContractError("control registry is not sealed to final Task0-v2")
    cases: list[Mapping[str, Any]] = []
    for item in _list(registry.get("controls"), "control registry.controls"):
        control = _mapping(item, "control registry control")
        if control.get("role") != "b2_distinct_same_description_pair":
            continue
        if (
            control.get("physical_identity_review")
            != "distinct_people_confirmed_by_lead_visual_inspection"
        ):
            raise ContextContractError(
                "B2 control lacks the frozen distinct-owner review"
            )
        cases.append(
            {
                "b2_case_id": _string(
                    control.get("control_id"), "B2 control.control_id"
                ),
                "target_gt_owner_id": control.get("target_gt_owner_id"),
                "covering_pred_row_id": control.get("covering_pred_row_id"),
                "review_status": "reviewed_candidate",
                "review_provenance": control.get("physical_identity_review"),
                "control_ids": [control.get("control_id")],
                "covering_gt_owner_id": control.get("covering_gt_owner_id"),
            }
        )
    if not cases:
        raise ContextContractError("control registry contains no reviewed B2 pair")
    return cases


def _parse_control_owner_cases(
    registry: Mapping[str, Any], owners: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if registry.get("schema_version") != CONTROL_REGISTRY_SCHEMA_VERSION:
        raise ContextContractError("control registry must be v2; fallback is forbidden")
    if (
        registry.get("status")
        != "lead_frozen_before_scoring_and_resealed_to_final_task0_v2"
    ):
        raise ContextContractError("control registry is not sealed to final Task0-v2")
    by_owner: dict[str, dict[str, set[str]]] = {}
    owner_fields = ("gt_owner_id", "covering_gt_owner_id", "target_gt_owner_id")
    for item in _list(registry.get("controls"), "control registry.controls"):
        control = _mapping(item, "control registry control")
        control_id = _string(control.get("control_id"), "control.control_id")
        role = _string(control.get("role"), f"control {control_id}.role")
        referenced_owner_ids = {
            str(control[field])
            for field in owner_fields
            if isinstance(control.get(field), str)
        }
        if not referenced_owner_ids:
            raise ContextContractError(f"control {control_id} has no owner foreign key")
        for owner_id in referenced_owner_ids:
            if owner_id not in owners:
                raise ContextContractError(
                    f"control {control_id} references unknown owner {owner_id}"
                )
            if str(control.get("image_id")) != str(owners[owner_id]["image_id"]):
                raise ContextContractError(
                    f"control {control_id} crosses image identities"
                )
            if not owners[owner_id]["greedy_decision_eligible"]:
                continue
            aggregate = by_owner.setdefault(
                owner_id,
                {"control_ids": set(), "control_roles": set(), "strata": set()},
            )
            aggregate["control_ids"].add(control_id)
            aggregate["control_roles"].add(role)
            aggregate["strata"].update(
                str(stratum)
                for stratum in _list(
                    control.get("strata", []), f"control {control_id}.strata"
                )
            )
    return [
        {
            "gt_owner_id": owner_id,
            "control_ids": sorted(values["control_ids"]),
            "control_roles": sorted(values["control_roles"]),
            "strata": sorted(values["strata"]),
        }
        for owner_id, values in sorted(by_owner.items())
    ]


def _clear_reference_prefix(
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, str],
    owners: Mapping[str, Mapping[str, Any]],
    sentinel_owner_id: str,
) -> tuple[int | None, str | None]:
    target = owners[sentinel_owner_id]
    target_position = _reference_position(target)
    for row in rows:
        owner_id, relation, reason = _row_assignment(row, identities, owners)
        if owner_id is None or relation != "new_owner":
            return None, reason
        owner = owners[owner_id]
        position = _reference_position(owner)
        if position > target_position:
            return int(row["original_row_index"]), None
    return None, "no_native_row_follows_sentinel_reference_position"


def _reference_position(owner: Mapping[str, Any]) -> tuple[float, float, int]:
    """Return the exact canonical ``(y1, x1, original_annotation_index)`` tuple."""

    return (
        float(owner["bbox_xyxy"][1]),
        float(owner["bbox_xyxy"][0]),
        int(owner["original_annotation_index"]),
    )


def _build_owner_position_contexts(
    *,
    context_namespace: str,
    owner_id: str,
    registry_id: str,
    common: Mapping[str, Any],
    eligibility: str,
    reference_eligibility: str | None,
    unresolved_eligibility: str,
    owners: Mapping[str, Mapping[str, Any]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, str],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    image_id = str(owners[owner_id]["image_id"])
    trajectory = greedy[image_id]
    contexts = [
        _context_row(
            context_id=f"ctx:{context_namespace}:root",
            context_kind="root",
            owner_id=owner_id,
            image_id=image_id,
            trajectory=trajectory,
            generated_end=0,
            upstream=upstream,
            status="admitted",
            eligibility=eligibility,
            source_pred_row_id=None,
            registry_id=registry_id,
            details=common,
        ),
        _context_row(
            context_id=f"ctx:{context_namespace}:natural_stop",
            context_kind="natural_stop",
            owner_id=owner_id,
            image_id=image_id,
            trajectory=trajectory,
            generated_end=int(trajectory["terminal_start"]),
            upstream=upstream,
            status="admitted",
            eligibility=eligibility,
            source_pred_row_id=None,
            registry_id=registry_id,
            details={
                **dict(common),
                "terminal_storage": "bound_by_structural_token_contract",
            },
        ),
    ]
    reference_index, reason = _clear_reference_prefix(
        predictions_by_image[image_id], identities, owners, owner_id
    )
    if reference_index is None:
        contexts.append(
            _context_row(
                context_id=f"ctx:{context_namespace}:reference_scan_position",
                context_kind="reference_scan_position",
                owner_id=owner_id,
                image_id=image_id,
                trajectory=trajectory,
                generated_end=None,
                upstream=upstream,
                status=UNRESOLVED_SCAN_STATUS,
                eligibility=unresolved_eligibility,
                source_pred_row_id=None,
                registry_id=registry_id,
                details={**dict(common), "unresolved_reason": reason},
            )
        )
        return contexts
    source_row = predictions_by_image[image_id][reference_index]
    start, _ = trajectory["row_spans"][reference_index]
    contexts.append(
        _context_row(
            context_id=f"ctx:{context_namespace}:reference_scan_position",
            context_kind="reference_scan_position",
            owner_id=owner_id,
            image_id=image_id,
            trajectory=trajectory,
            generated_end=start,
            upstream=upstream,
            status="admitted",
            eligibility=reference_eligibility or eligibility,
            source_pred_row_id=str(source_row["pred_row_id"]),
            registry_id=registry_id,
            details={**dict(common), "reference_original_row_index": reference_index},
        )
    )
    return contexts


def _build_sentinel_contexts(
    cases: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, str],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        sentinel_id = _string(case.get("sentinel_id"), f"sentinel {index}.sentinel_id")
        if sentinel_id in seen:
            raise ContextContractError(f"duplicate sentinel_id {sentinel_id}")
        seen.add(sentinel_id)
        owner_id = _string(
            case.get("gt_owner_id"), f"sentinel {sentinel_id}.gt_owner_id"
        )
        if owner_id not in owners:
            raise ContextContractError(
                f"sentinel {sentinel_id} references unknown owner {owner_id}"
            )
        reviewed_eligible = case.get("review_status") == "eligible"
        frozen_non_recovery = (
            case.get("prior_non_recovery_status")
            == "verified_primary_natural_zero_spatial_support"
        )
        if not (reviewed_eligible or frozen_non_recovery):
            raise ContextContractError(f"sentinel {sentinel_id} is not frozen eligible")
        common = {
            "sentinel_id": sentinel_id,
            "sentinel_original_annotation_index": owners[owner_id][
                "original_annotation_index"
            ],
            "repair_eligible": False,
        }
        contexts.extend(
            _build_owner_position_contexts(
                context_namespace=f"sentinel:{sentinel_id}",
                owner_id=owner_id,
                registry_id=sentinel_id,
                common=common,
                eligibility="sentinel_diagnostic_only",
                reference_eligibility="sentinel_diagnostic_only_not_repair_eligible",
                unresolved_eligibility="not_c_eligible_not_repair_eligible",
                owners=owners,
                predictions_by_image=predictions_by_image,
                identities=identities,
                greedy=greedy,
                upstream=upstream,
            )
        )
    return contexts


def _build_control_contexts(
    cases: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, str],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for case in cases:
        owner_id = str(case["gt_owner_id"])
        control_ids = [str(item) for item in case["control_ids"]]
        common = {
            "control_owner_id": owner_id,
            "control_ids": control_ids,
            "control_roles": [str(item) for item in case["control_roles"]],
            "strata": [str(item) for item in case["strata"]],
            "control_original_annotation_index": owners[owner_id][
                "original_annotation_index"
            ],
            "repair_eligible": False,
        }
        contexts.extend(
            _build_owner_position_contexts(
                context_namespace=f"control-owner:{owner_id}",
                owner_id=owner_id,
                registry_id=control_ids[0],
                common=common,
                eligibility="control_diagnostic_only",
                reference_eligibility=None,
                unresolved_eligibility="not_control_diagnostic_eligible",
                owners=owners,
                predictions_by_image=predictions_by_image,
                identities=identities,
                greedy=greedy,
                upstream=upstream,
            )
        )
    return contexts


def _build_b2_contexts(
    cases: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = _string(case.get("b2_case_id"), f"B2 case {index}.b2_case_id")
        if case_id in seen:
            raise ContextContractError(f"duplicate b2_case_id {case_id}")
        seen.add(case_id)
        if case.get("review_status") != "reviewed_candidate":
            raise ContextContractError(f"B2 case {case_id} is not a reviewed candidate")
        owner_id = _string(
            case.get("target_gt_owner_id"), f"B2 case {case_id}.target_gt_owner_id"
        )
        pred_id = _string(
            case.get("covering_pred_row_id"), f"B2 case {case_id}.covering_pred_row_id"
        )
        if owner_id not in owners or pred_id not in predictions:
            raise ContextContractError(f"B2 case {case_id} has an unknown foreign key")
        row = predictions[pred_id]
        covering_owner_id = case.get("covering_gt_owner_id")
        if not isinstance(covering_owner_id, str) or covering_owner_id not in owners:
            raise ContextContractError(
                f"B2 case {case_id} lacks its covering owner foreign key"
            )
        image_id = str(owners[owner_id]["image_id"])
        if row["image_id"] != image_id:
            raise ContextContractError(f"B2 case {case_id} crosses image identities")
        if row.get("row_kind") != "complete_prediction":
            raise ContextContractError(
                f"B2 case {case_id} covering row is not structurally complete"
            )
        target_eligible = bool(owners[owner_id]["greedy_decision_eligible"])
        covering_eligible = bool(owners[covering_owner_id]["greedy_decision_eligible"])
        row_eligible = (
            row.get("strict_match_status") == "matched"
            and row.get("strict_match_gt_owner_id") == covering_owner_id
            and not row.get("ambiguity_receipt_ids")
        )
        row_index = int(row["original_row_index"])
        start, end = greedy[image_id]["row_spans"][row_index]
        common = {
            "b2_case_id": case_id,
            "target_original_annotation_index": owners[owner_id][
                "original_annotation_index"
            ],
            "covering_original_row_index": row_index,
            "causal_collision_inferred": False,
            "review_provenance": case.get("review_provenance"),
            "control_ids": case.get("control_ids", []),
            "covering_gt_owner_id": covering_owner_id,
        }
        if not (target_eligible and covering_eligible and row_eligible):
            rejected = _context_row(
                context_id=f"receipt:B2:{case_id}",
                context_kind="B2_candidate_receipt",
                owner_id=owner_id,
                image_id=image_id,
                trajectory=greedy[image_id],
                generated_end=None,
                upstream=upstream,
                status="rejected_neutral_global_ambiguity",
                eligibility="raw_diagnostic_neutral_not_b2_eligible",
                source_pred_row_id=pred_id,
                registry_id=case_id,
                details={
                    **common,
                    "target_decision_eligible": target_eligible,
                    "covering_decision_eligible": covering_eligible,
                    "covering_row_decision_eligible": row_eligible,
                },
            )
            rejected["record_type"] = "b2_candidate_receipt"
            contexts.append(rejected)
            continue
        for kind, boundary in (("B2_before", start), ("B2_after", end)):
            contexts.append(
                _context_row(
                    context_id=f"ctx:B2:{case_id}:{kind}",
                    context_kind=kind,
                    owner_id=owner_id,
                    image_id=image_id,
                    trajectory=greedy[image_id],
                    generated_end=boundary,
                    upstream=upstream,
                    status="admitted_reviewed_candidate_only",
                    eligibility="b2_diagnostic_pending_controls_and_scoring",
                    source_pred_row_id=pred_id,
                    registry_id=case_id,
                    details=common,
                )
            )
    return contexts


def _bbox_drift(
    current: Mapping[str, Any], previous: Mapping[str, Any]
) -> dict[str, Any]:
    current_box = current.get("bbox_xyxy")
    previous_box = previous.get("bbox_xyxy")
    if not isinstance(current_box, list) or not isinstance(previous_box, list):
        return {"status": "unavailable", "extent_drift": None, "bbox_delta_xyxy": None}
    delta = [
        float(right) - float(left)
        for left, right in zip(previous_box, current_box, strict=True)
    ]
    return {
        "status": "measured",
        "extent_drift": any(value != 0.0 for value in delta),
        "bbox_delta_xyxy": delta,
    }


def _bbox_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _neutral_semantic_geometry_recurrences(
    image_id: str,
    rows: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    upstream: Mapping[str, str],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    prior_rows: list[Mapping[str, Any]] = []
    for current in rows:
        current_box = current.get("bbox_xyxy")
        if not isinstance(current_box, list):
            prior_rows.append(current)
            continue
        current_description = current.get("normalized_description")
        reviewed_cluster = current.get("reviewed_description_cluster_id")
        selected: tuple[Mapping[str, Any], list[Any], str, str, float] | None = None
        for previous in reversed(prior_rows):
            previous_box = previous.get("bbox_xyxy")
            if not isinstance(previous_box, list):
                continue
            identical = [float(value) for value in previous_box] == [
                float(value) for value in current_box
            ]
            overlap = _bbox_iou(previous_box, current_box)
            exact_description = (
                isinstance(current_description, str)
                and current_description
                and current_description == previous.get("normalized_description")
            )
            similar_reviewed = (
                isinstance(reviewed_cluster, str)
                and reviewed_cluster
                and reviewed_cluster == previous.get("reviewed_description_cluster_id")
            )
            if not identical and not (
                (exact_description or similar_reviewed) and overlap >= 0.9
            ):
                continue
            description_relation = (
                "exact"
                if exact_description
                else "reviewed_similar"
                if similar_reviewed
                else "unrestricted_identical_bbox"
            )
            geometry_relation = "byte_identical_bbox" if identical else "high_overlap"
            selected = (
                previous,
                previous_box,
                description_relation,
                geometry_relation,
                overlap,
            )
            break
        if selected is not None:
            (
                previous,
                previous_box,
                description_relation,
                geometry_relation,
                overlap,
            ) = selected
            diagnostics.append(
                {
                    "schema_version": CHRONOLOGY_SCHEMA_VERSION,
                    "record_type": "neutral_semantic_geometry_recurrence",
                    "image_id": image_id,
                    "gt_owner_id": None,
                    "trajectory_id": trajectory["trajectory_id"],
                    "from_pred_row_id": previous["pred_row_id"],
                    "to_pred_row_id": current["pred_row_id"],
                    "from_original_row_index": previous["original_row_index"],
                    "to_original_row_index": current["original_row_index"],
                    "description_relation": description_relation,
                    "geometry_relation": geometry_relation,
                    "intersection_over_union": overlap,
                    "from_bbox_xyxy": previous_box,
                    "to_bbox_xyxy": current_box,
                    "physical_owner_assignment": "neutral_not_assigned",
                    "causal_duplication_inferred": False,
                    "decision_eligibility": "raw_diagnostic_only",
                    "source_binding": _source_binding(trajectory, upstream),
                }
            )
        prior_rows.append(current)
    return diagnostics


def _build_chronology(
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, str],
    owners: Mapping[str, Mapping[str, Any]],
    greedy: Mapping[str, Mapping[str, Any]],
    upstream: Mapping[str, str],
    short_gap_max: int,
    cyclic_min_occurrences: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id, rows in sorted(predictions_by_image.items()):
        output.extend(
            _neutral_semantic_geometry_recurrences(
                image_id, rows, greedy[image_id], upstream
            )
        )
        occurrences: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        neutral: list[dict[str, Any]] = []
        for row in rows:
            owner_id, relation, reason = _row_assignment(row, identities, owners)
            if owner_id is None:
                neutral.append(
                    {
                        "pred_row_id": row["pred_row_id"],
                        "original_row_index": row["original_row_index"],
                        "neutral_reason": reason,
                    }
                )
                continue
            occurrences[owner_id].append({**dict(row), "assignment_relation": relation})
        for owner_id, owner_rows in sorted(occurrences.items()):
            events: list[dict[str, Any]] = []
            gaps: list[int] = []
            for occurrence_index in range(1, len(owner_rows)):
                previous = owner_rows[occurrence_index - 1]
                current = owner_rows[occurrence_index]
                gap = int(current["original_row_index"]) - int(
                    previous["original_row_index"]
                )
                intervening = gap - 1
                gaps.append(gap)
                base_class = (
                    "consecutive"
                    if intervening == 0
                    else "short_gap"
                    if intervening <= short_gap_max
                    else "long_gap"
                )
                cyclic = (
                    occurrence_index + 1 >= cyclic_min_occurrences
                    and len(gaps) >= 2
                    and gaps[-1] == gaps[-2]
                )
                events.append(
                    {
                        "from_pred_row_id": previous["pred_row_id"],
                        "to_pred_row_id": current["pred_row_id"],
                        "from_original_row_index": previous["original_row_index"],
                        "to_original_row_index": current["original_row_index"],
                        "intervening_row_count": intervening,
                        "recurrence_class": "cyclic" if cyclic else base_class,
                        "base_gap_class": base_class,
                        "assignment_relation": current["assignment_relation"],
                        "extent": _bbox_drift(current, previous),
                    }
                )
            output.append(
                {
                    "schema_version": CHRONOLOGY_SCHEMA_VERSION,
                    "record_type": "owner_summary",
                    "image_id": image_id,
                    "gt_owner_id": owner_id,
                    "trajectory_id": greedy[image_id]["trajectory_id"],
                    "occurrence_pred_row_ids": [
                        row["pred_row_id"] for row in owner_rows
                    ],
                    "occurrence_original_row_indices": [
                        row["original_row_index"] for row in owner_rows
                    ],
                    "recurrence_events": events,
                    "recurrence_counts": {
                        name: sum(event["recurrence_class"] == name for event in events)
                        for name in ("consecutive", "short_gap", "long_gap", "cyclic")
                    },
                    "ambiguous_assignments_included": False,
                    "source_binding": _source_binding(greedy[image_id], upstream),
                }
            )
        for item in neutral:
            output.append(
                {
                    "schema_version": CHRONOLOGY_SCHEMA_VERSION,
                    "record_type": "neutral_assignment",
                    "image_id": image_id,
                    "gt_owner_id": None,
                    "trajectory_id": greedy[image_id]["trajectory_id"],
                    **item,
                    "contributes_to_duplicate_timing": False,
                    "source_binding": _source_binding(greedy[image_id], upstream),
                }
            )
    return output


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _git_output(args: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    return completed.stdout


def _repository_state(relevant_paths: Sequence[Path]) -> dict[str, Any]:
    root = Path(
        _git_output(("rev-parse", "--show-toplevel"), Path.cwd())
        .decode("utf-8")
        .strip()
    ).resolve()
    relative_paths: list[str] = []
    for path in relevant_paths:
        resolved = path.resolve()
        try:
            relative_paths.append(str(resolved.relative_to(root)))
        except ValueError:
            continue
    relative_paths = sorted(set(relative_paths))
    tracked_diff = _git_output(
        ("diff", "--binary", "HEAD", "--", *relative_paths), root
    )
    status_lines = (
        _git_output(
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *relative_paths,
            ),
            root,
        )
        .decode("utf-8", errors="replace")
        .splitlines()
    )
    files = []
    for relative_path in relative_paths:
        path = root / relative_path
        if path.is_file():
            files.append(
                {
                    "relative_path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "root": str(root),
        "head": _git_output(("rev-parse", "HEAD"), root).decode("utf-8").strip(),
        "relevant_paths": relative_paths,
        "relevant_status_lines": status_lines,
        "relevant_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "relevant_tracked_diff_bytes": len(tracked_diff),
        "relevant_file_hashes": files,
    }


def _runtime_receipt() -> dict[str, Any]:
    return {
        "execution_device": "cpu",
        "gpu_model_or_inference_used": False,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_sorted_owner_basin_contexts(
    *,
    greedy_artifact: str | Path,
    owner_ledger: str | Path,
    prediction_row_ledger: str | Path,
    owner_trajectory_matrix: str | Path,
    structural_token_registry: str | Path | None,
    sentinel_registry: str | Path,
    reviewed_b2_registry: str | Path,
    output_dir: str | Path,
    expected_digests: Mapping[str, str],
    execution_argv: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Validate sealed inputs and materialize the three Task-6 CPU artifacts."""

    input_paths: dict[str, Path] = {
        "greedy": Path(greedy_artifact).expanduser().resolve(strict=True),
        "owner_ledger": Path(owner_ledger).expanduser().resolve(strict=True),
        "prediction_row_ledger": Path(prediction_row_ledger)
        .expanduser()
        .resolve(strict=True),
        "owner_trajectory_matrix": Path(owner_trajectory_matrix)
        .expanduser()
        .resolve(strict=True),
        "sentinel_registry": Path(sentinel_registry).expanduser().resolve(strict=True),
        "reviewed_b2_registry": Path(reviewed_b2_registry)
        .expanduser()
        .resolve(strict=True),
    }
    if structural_token_registry is not None:
        input_paths["structural_token_registry"] = (
            Path(structural_token_registry).expanduser().resolve(strict=True)
        )
    if set(expected_digests) != set(input_paths):
        raise ContextContractError(
            f"expected_digests must contain exactly {sorted(input_paths)}"
        )
    actual = {
        name: _verify_expected_digest(path, expected_digests[name], name)
        for name, path in input_paths.items()
    }
    if "structural_token_registry" in input_paths:
        structural_document = _read_json(
            input_paths["structural_token_registry"], "structural-token registry"
        )
        structural = _parse_structural_registry(structural_document, actual["greedy"])
    else:
        structural = dict(EXPERIMENT_STRUCTURAL_RULES)
        actual["structural_token_contract"] = sha256_json(structural)
    greedy = _load_greedy(input_paths["greedy"], actual["greedy"], structural)
    task0 = _load_task0_v2_bundle(input_paths, actual)
    actual.update(task0["implicit_digests"])
    prompt_receipts = {
        str(item.get("trajectory_id")): _mapping(item, "Task0 prompt/media identity")
        for item in _list(
            _mapping(task0["execution_receipt"], "Task0 execution receipt").get(
                "prompt_media_identities"
            ),
            "Task0 prompt/media identities",
        )
    }
    for trajectory in greedy.values():
        identity = prompt_receipts.get(str(trajectory["trajectory_id"]))
        if identity is None:
            raise ContextContractError(
                "Task0 receipt lacks a greedy prompt/token identity"
            )
        if (
            identity.get("prompt_token_ids_sha256") != trajectory["prompt_sha256"]
            or identity.get("generated_token_ids_sha256")
            != trajectory["generated_sha256"]
        ):
            raise ContextContractError(
                "Task0 receipt token identity disagrees with greedy source"
            )
    owner_rows = _read_jsonl(input_paths["owner_ledger"], "owner ledger")
    prediction_rows = _read_jsonl(
        input_paths["prediction_row_ledger"], "prediction-row ledger"
    )
    matrix_rows = _read_jsonl(
        input_paths["owner_trajectory_matrix"], "owner-trajectory matrix"
    )
    owners = _load_owners(
        owner_rows,
        str(task0["execution_receipt_content_sha256"]),
        task0["ambiguity_by_id"],
    )
    predictions, predictions_by_image = _load_predictions(
        prediction_rows,
        greedy,
        actual["greedy"],
        str(task0["execution_receipt_content_sha256"]),
        task0["ambiguity_by_id"],
    )
    _validate_matrix(
        matrix_rows,
        owners,
        predictions,
        greedy,
        actual["greedy"],
        str(task0["execution_receipt_content_sha256"]),
        task0["ambiguity_by_id"],
    )
    identities, owners_by_image = _owner_identity_maps(owners)
    if set(owners_by_image) != set(greedy):
        raise ContextContractError("owner ledger and greedy artifact image sets differ")

    sentinel_document = _read_json(
        input_paths["sentinel_registry"], "sentinel registry"
    )
    b2_document = _read_json(
        input_paths["reviewed_b2_registry"], "reviewed B2 registry"
    )
    registry_receipts = _validate_frozen_registries(
        sentinel_document,
        b2_document,
        sentinel_path=input_paths["sentinel_registry"],
        actual=actual,
        task0=task0,
        owners=owners,
    )
    actual.update(registry_receipts)
    sentinel_cases = _parse_registry_cases(
        sentinel_document,
        SENTINEL_REGISTRY_SCHEMA_VERSION,
        "sentinels",
        "sentinel registry",
    )
    b2_cases = _parse_b2_cases(b2_document)
    control_owner_cases = _parse_control_owner_cases(b2_document, owners)

    contexts = _build_first_skip_contexts(
        owners,
        owners_by_image,
        predictions_by_image,
        identities,
        greedy,
        actual,
    )
    contexts.extend(
        _build_sentinel_contexts(
            sentinel_cases,
            owners,
            predictions_by_image,
            identities,
            greedy,
            actual,
        )
    )
    contexts.extend(
        _build_control_contexts(
            control_owner_cases,
            owners,
            predictions_by_image,
            identities,
            greedy,
            actual,
        )
    )
    contexts.extend(_build_b2_contexts(b2_cases, owners, predictions, greedy, actual))
    contexts.sort(key=lambda row: str(row["context_id"]))
    chronology = _build_chronology(
        predictions_by_image,
        identities,
        owners,
        greedy,
        actual,
        _integer(structural.get("short_gap_max"), "short_gap_max"),
        _integer(structural.get("cyclic_min_occurrences"), "cyclic_min_occurrences"),
    )
    chronology.sort(
        key=lambda row: (
            str(row["image_id"]),
            str(row.get("gt_owner_id") or "~"),
            str(row.get("pred_row_id") or ""),
        )
    )

    destination = Path(output_dir).expanduser().resolve()
    context_path = destination / "first-skip-contexts.jsonl"
    chronology_path = destination / "natural-duplication-chronology.jsonl"
    receipt_path = destination / "context-builder-receipt.json"
    context_bytes = _jsonl_bytes(contexts)
    chronology_bytes = _jsonl_bytes(chronology)
    context_digest = hashlib.sha256(context_bytes).hexdigest()
    chronology_digest = hashlib.sha256(chronology_bytes).hexdigest()
    argv = (
        [str(item) for item in execution_argv]
        if execution_argv is not None
        else [str(Path(sys.executable).resolve()), *sys.argv]
    )
    repository_paths = [
        Path(__file__),
        Path(__file__).resolve().parents[2]
        / "tests/research/test_build_sorted_owner_basin_contexts.py",
        *input_paths.values(),
        input_paths["sentinel_registry"].parent / "sentinel-selection-receipt.json",
        input_paths["sentinel_registry"].parent
        / "sentinel-selection-confirmation-receipt.json",
    ]
    candidate_receipts = [
        row for row in contexts if row["record_type"] == "first_skip_candidate_receipt"
    ]
    denominator_receipts = [
        row
        for row in candidate_receipts
        if row["details"]["contributes_to_reconstructible_candidate_denominator"]
    ]
    ambiguous_denominator_receipts = [
        row
        for row in denominator_receipts
        if row["details"]["candidate_reason"] == "globally_ambiguous_owner_or_row"
        or (
            isinstance(row["details"].get("first_contamination"), Mapping)
            and row["details"]["first_contamination"].get("reason")
            == "globally_ambiguous_owner_or_row"
        )
    ]
    stop_rule_6_denominator = len(denominator_receipts)
    stop_rule_6_numerator = len(ambiguous_denominator_receipts)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "execution_status": "completed",
        "execution_surface": "deterministic_cpu_only_no_model_no_tokenizer",
        "content_digest_contract": {
            "field": "receipt_content_sha256",
            "algorithm": "sha256",
            "canonicalization": "UTF-8 JSON with ensure_ascii=true, sorted keys, compact separators",
            "excluded_top_level_fields": ["receipt_content_sha256"],
        },
        "forced_continuation_artifacts_included": False,
        "scientific_conclusions_emitted": False,
        "input_digests": dict(sorted(actual.items())),
        "task0_v2_binding": {
            "root": task0["root"],
            "execution_receipt_content_sha256": task0[
                "execution_receipt_content_sha256"
            ],
            "v1_fallback_allowed": False,
        },
        "command": {
            "argv": argv,
            "shell_escaped": shlex.join(argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "repository": _repository_state(repository_paths),
        "outputs": {
            "first-skip-contexts.jsonl": {
                "sha256": context_digest,
                "row_count": len(contexts),
            },
            "natural-duplication-chronology.jsonl": {
                "sha256": chronology_digest,
                "row_count": len(chronology),
            },
        },
        "counts": {
            "clean_first_skip_context_rows": sum(
                row["context_kind"] in {"P_pre", "P_post"} for row in contexts
            ),
            "first_skip_candidate_receipt_rows": len(candidate_receipts),
            "first_skip_candidate_denominator": len(denominator_receipts),
            "admitted_first_skip_candidates": sum(
                row["status"] == "admitted_candidate" for row in denominator_receipts
            ),
            "rejected_neutral_first_skip_candidates": sum(
                row["status"] != "admitted_candidate" for row in denominator_receipts
            ),
            "non_candidate_rejected_history_receipts": len(candidate_receipts)
            - len(denominator_receipts),
            "sentinel_context_rows": sum(
                row["context_kind"]
                in {"root", "natural_stop", "reference_scan_position"}
                and "sentinel_id" in row["details"]
                for row in contexts
            ),
            "control_diagnostic_context_rows": sum(
                row["context_kind"]
                in {"root", "natural_stop", "reference_scan_position"}
                and "control_owner_id" in row["details"]
                for row in contexts
            ),
            "resolved_control_diagnostic_context_rows": sum(
                row["context_kind"]
                in {"root", "natural_stop", "reference_scan_position"}
                and "control_owner_id" in row["details"]
                and row["status"] != UNRESOLVED_SCAN_STATUS
                for row in contexts
            ),
            "unresolved_control_reference_contexts": sum(
                row["context_kind"] == "reference_scan_position"
                and "control_owner_id" in row["details"]
                and row["status"] == UNRESOLVED_SCAN_STATUS
                for row in contexts
            ),
            "unresolved_reference_scan_contexts": sum(
                row["status"] == UNRESOLVED_SCAN_STATUS for row in contexts
            ),
            "reviewed_b2_context_rows": sum(
                row["context_kind"] in {"B2_before", "B2_after"} for row in contexts
            ),
            "neutral_chronology_rows": sum(
                row["record_type"] == "neutral_assignment" for row in chronology
            ),
            "neutral_semantic_geometry_recurrence_rows": sum(
                row["record_type"] == "neutral_semantic_geometry_recurrence"
                for row in chronology
            ),
        },
        "first_skip_candidate_denominator": {
            "definition": "every natural row that exposes at least one earlier unassigned reference owner, including admitted, repeated, ambiguous, unmatched, or otherwise contaminated transitions; rejected rows that expose no skipped owner remain receipts but do not enter this denominator",
            "candidate_ids": [
                row["details"]["candidate_id"] for row in denominator_receipts
            ],
            "status_counts": {
                status: sum(row["status"] == status for row in denominator_receipts)
                for status in sorted(
                    {str(row["status"]) for row in denominator_receipts}
                )
            },
            "reason_counts": {
                reason: sum(
                    row["details"]["candidate_reason"] == reason
                    for row in denominator_receipts
                )
                for reason in sorted(
                    {
                        str(row["details"]["candidate_reason"])
                        for row in denominator_receipts
                    }
                )
            },
            "non_candidate_rejected_history_receipt_count": len(candidate_receipts)
            - len(denominator_receipts),
            "stop_rule_6": {
                "rule": "stop if globally ambiguous first-skip candidates exceed half of the reconstructible first-skip candidate denominator",
                "globally_ambiguous_candidate_count": stop_rule_6_numerator,
                "candidate_denominator": stop_rule_6_denominator,
                "globally_ambiguous_fraction": (
                    stop_rule_6_numerator / stop_rule_6_denominator
                    if stop_rule_6_denominator
                    else None
                ),
                "triggered": (
                    stop_rule_6_numerator * 2 > stop_rule_6_denominator
                    if stop_rule_6_denominator
                    else None
                ),
            },
        },
        "chronology_rules": {
            "short_gap_max_intervening_rows": structural["short_gap_max"],
            "cyclic_min_occurrences": structural["cyclic_min_occurrences"],
        },
        "runtime": _runtime_receipt(),
    }
    receipt["receipt_content_sha256"] = sha256_json(receipt)
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite or reuse output directory: {destination}"
        ) from exc
    _atomic_write(context_path, context_bytes)
    _atomic_write(chronology_path, chronology_bytes)
    _atomic_write(receipt_path, receipt_bytes)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-artifact", required=True, type=Path)
    parser.add_argument("--owner-ledger", required=True, type=Path)
    parser.add_argument("--prediction-row-ledger", required=True, type=Path)
    parser.add_argument("--owner-trajectory-matrix", required=True, type=Path)
    parser.add_argument(
        "--structural-token-registry",
        type=Path,
        help="optional sealed override; otherwise use the frozen experiment-local Qwen IDs",
    )
    parser.add_argument("--sentinel-registry", required=True, type=Path)
    parser.add_argument("--reviewed-b2-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    for name in (
        "greedy",
        "owner-ledger",
        "prediction-row-ledger",
        "owner-trajectory-matrix",
        "sentinel-registry",
        "reviewed-b2-registry",
    ):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-structural-token-registry-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    expected = {
        "greedy": args.expected_greedy_sha256,
        "owner_ledger": args.expected_owner_ledger_sha256,
        "prediction_row_ledger": args.expected_prediction_row_ledger_sha256,
        "owner_trajectory_matrix": args.expected_owner_trajectory_matrix_sha256,
        "sentinel_registry": args.expected_sentinel_registry_sha256,
        "reviewed_b2_registry": args.expected_reviewed_b2_registry_sha256,
    }
    if (args.structural_token_registry is None) != (
        args.expected_structural_token_registry_sha256 is None
    ):
        raise ContextContractError(
            "--structural-token-registry and its expected digest must be supplied together"
        )
    if args.structural_token_registry is not None:
        expected["structural_token_registry"] = (
            args.expected_structural_token_registry_sha256
        )
    receipt = build_sorted_owner_basin_contexts(
        greedy_artifact=args.greedy_artifact,
        owner_ledger=args.owner_ledger,
        prediction_row_ledger=args.prediction_row_ledger,
        owner_trajectory_matrix=args.owner_trajectory_matrix,
        structural_token_registry=args.structural_token_registry,
        sentinel_registry=args.sentinel_registry,
        reviewed_b2_registry=args.reviewed_b2_registry,
        output_dir=args.output_dir,
        expected_digests=expected,
        execution_argv=[
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            *raw_argv,
        ],
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
