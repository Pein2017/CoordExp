#!/usr/bin/env python3
"""Run exact-prefix successor behavior for the sorted FN mechanism unit.

The CPU contract layer is intentionally import-safe.  Torch, transformers,
and the HF backend are imported only by :func:`run_live`.  Logical role ids
and prompt/self-prefix token splits come from the frozen successor registry
and owner-context ledger; this module never derives a prefix from text or GT
ordering.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "sorted-fn-successor-behavior.v1"
CONTRACT_RECEIPT_SCHEMA_VERSION = "sorted-fn-successor-behavior-contract.v1"
SAMPLING_ADMISSION_SCHEMA_VERSION = "sorted-fn-successor-sampling-admission.v2"
LANDSCAPE_ADMISSION_SCHEMA_VERSION = "sorted-fn-successor-behavior-landscape-admission.v1"
DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION = "sorted-fn-description-equivalence.v1"
REGISTRY_SCHEMA_VERSION = "sorted-fn-mechanism-registry.v2"
LEDGER_SCHEMA_VERSION = "sorted_owner_basin_owner_context_ledger.v2"
LANDSCAPE_RECEIPT_SCHEMA_VERSION = "sorted_fn_successor_score_shard_merge.v2"
SUCCESSOR_SCORE_ROW_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores.v1"
SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION = "sorted_fn_fixed_budget_scores_receipt.v1"
LANDSCAPE_DECISION_CHANNEL = "raw_model_logprob.complete_box_logprob_sum"
MECHANISM_DECISION_RULES_SCHEMA_VERSION = "sorted-fn-mechanism-decision-rules.v1"
PLANNER_RECEIPT_SCHEMA_VERSION = "sorted-fn-successor-input-plan-receipt.v2"
RUNTIME_IDENTITY_SCHEMA_VERSION = "sorted-owner-basin-runtime-identity.v1"
UNIT_ID = "2026-08-02-sorted-false-negative-mechanism-decomposition"

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
GREEDY_TOP_P = 1.0
PRIMARY_REPETITION_PENALTY = 1.0
POLICY_REPETITION_PENALTY = 1.1
ADMITTED_LANDSCAPE_CONDITIONS = frozenset(
    {"usable_target_strict_region_peak", "multiple_separated_owner_localized_peaks"}
)
SEMANTIC_DRIFT_IOU_THRESHOLD = 0.5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BehaviorContractError(ValueError):
    """Raised before model loading when a behavior contract is inadmissible."""


_RUNTIME_IDENTITY_CACHE: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BehaviorContractError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BehaviorContractError(f"{label} must be an array")
    return value


def _ids(value: Any, label: str, *, allow_empty: bool = True) -> list[int]:
    values = _sequence(value, label)
    if not allow_empty and not values:
        raise BehaviorContractError(f"{label} must not be empty")
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BehaviorContractError(f"{label} must contain non-negative integers")
        result.append(int(item))
    return result


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise BehaviorContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BehaviorContractError(f"{label} is not valid JSON") from exc
    return dict(_mapping(value, label))


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(dict(_mapping(json.loads(line), f"{label}:{line_number}")))
        except json.JSONDecodeError as exc:
            raise BehaviorContractError(f"{label}:{line_number} is not valid JSON") from exc
    if not rows:
        raise BehaviorContractError(f"{label} is empty")
    return rows


def _resolved_file(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise BehaviorContractError(f"{label} does not exist") from exc
    if not path.is_file():
        raise BehaviorContractError(f"{label} must be a regular file")
    return path


def _validate_file_ref(value: Any, label: str) -> dict[str, Any]:
    ref = _mapping(value, label)
    path = _resolved_file(str(ref.get("path", "")), f"{label}.path")
    expected = _digest(ref.get("sha256"), f"{label}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise BehaviorContractError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )
    return {"path": str(path), "sha256": observed}


def _document_digest(document: Mapping[str, Any], digest_key: str, label: str) -> str:
    declared = _digest(document.get(digest_key), f"{label}.{digest_key}")
    content = {key: value for key, value in document.items() if key != digest_key}
    observed = sha256_json(content)
    if observed != declared:
        raise BehaviorContractError(
            f"{label} digest mismatch: observed {observed}, expected {declared}"
        )
    return observed


def _iter_registry_roles(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    smoke = _mapping(registry.get("smoke"), "registry.smoke")
    roles = [
        dict(_mapping(item, "registry.smoke.roles[]"))
        for item in _sequence(smoke.get("roles"), "registry.smoke.roles")
    ]
    envelope = _mapping(
        smoke.get("null_pair_envelope"), "registry.smoke.null_pair_envelope"
    )
    for pair in _sequence(envelope.get("pairs", []), "null_pair_envelope.pairs"):
        for item in _sequence(_mapping(pair, "null pair").get("roles"), "null pair.roles"):
            roles.append(dict(_mapping(item, "null pair role")))
    role_ids = [str(item.get("role_id", "")) for item in roles]
    if any(not value for value in role_ids) or len(role_ids) != len(set(role_ids)):
        raise BehaviorContractError("registry role ids are missing or duplicated")
    return roles


def validate_registry(
    registry_path: Path, include_role_ids: Sequence[str] = ()
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate the frozen registry and select exact logical roles."""

    registry = _read_json(registry_path, "FN mechanism registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise BehaviorContractError("FN mechanism registry schema_version is not recognized")
    if registry.get("unit_id") != UNIT_ID:
        raise BehaviorContractError("FN mechanism registry unit_id mismatch")
    registry_digest = _document_digest(registry, "registry_digest", "FN mechanism registry")
    sources = _mapping(registry.get("sources"), "registry.sources")
    source_receipt: dict[str, Any] = {}
    for key in ("owner_ledger", "prediction_row_ledger", "cohort_assignments"):
        source_receipt[key] = _validate_file_ref(sources.get(key), f"registry.sources.{key}")
    rollout_refs = []
    for index, value in enumerate(
        _sequence(sources.get("rollout_artifacts"), "registry.sources.rollout_artifacts")
    ):
        rollout_refs.append(_validate_file_ref(value, f"registry rollout[{index}]"))
    source_receipt["rollout_artifacts"] = rollout_refs

    owner_index = {
        str(row.get("gt_owner_id")): row
        for row in _read_jsonl(Path(source_receipt["owner_ledger"]["path"]), "owner ledger")
    }
    prediction_index = {
        str(row.get("pred_row_id")): row
        for row in _read_jsonl(
            Path(source_receipt["prediction_row_ledger"]["path"]),
            "prediction row ledger",
        )
    }
    try:
        from scripts.research.build_sorted_fn_mechanism_registry import (
            _prefix_before_row_index,
            load_rollout_trajectories,
        )

        trajectories = load_rollout_trajectories(
            [Path(item["path"]) for item in rollout_refs]
        )
    except (OSError, ValueError) as exc:
        raise BehaviorContractError(
            f"registry rollout lineage is invalid: {exc}"
        ) from exc

    roles = _iter_registry_roles(registry)
    by_id = {str(role["role_id"]): role for role in roles}
    requested = list(dict.fromkeys(str(value) for value in include_role_ids))
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise BehaviorContractError(f"unknown registered role ids: {unknown}")
    selected = [by_id[value] for value in requested] if requested else roles
    if not selected:
        raise BehaviorContractError("no registered roles selected")

    for role in roles:
        role_id = str(role["role_id"])
        owner_id = str(role.get("gt_owner_id", ""))
        if not owner_id.startswith("gt:"):
            raise BehaviorContractError(f"role {role_id!r} lacks a permanent gt_owner_id")
        prefix = _mapping(role.get("prefix"), f"role {role_id}.prefix")
        ids = _ids(prefix.get("token_ids"), f"role {role_id}.prefix.token_ids")
        observed = sha256_json(ids)
        if observed != prefix.get("token_ids_sha256"):
            raise BehaviorContractError(f"role {role_id!r} exact prefix digest mismatch")
        trajectory = _mapping(role.get("trajectory"), f"role {role_id}.trajectory")
        if str(trajectory.get("image_id", "")) != owner_id.split(":", 2)[1]:
            raise BehaviorContractError(f"role {role_id!r} owner/trajectory image mismatch")
        trajectory_key = (
            str(trajectory.get("image_id")),
            str(trajectory.get("decode_mode")),
            int(trajectory.get("seed")),
        )
        frozen_trajectory = trajectories.get(trajectory_key)
        if frozen_trajectory is None:
            raise BehaviorContractError(
                f"role {role_id!r} references an unknown immutable trajectory"
            )
        provenance = _mapping(role.get("provenance"), f"role {role_id}.provenance")
        if (
            provenance.get("source_artifact_path")
            != frozen_trajectory["source_artifact_path"]
            or provenance.get("source_artifact_sha256")
            != frozen_trajectory["source_artifact_sha256"]
        ):
            raise BehaviorContractError(f"role {role_id!r} source lineage mismatch")
        for pred_row_id in prefix.get("prefix_pred_row_ids", []):
            row = prediction_index.get(str(pred_row_id))
            if row is None:
                raise BehaviorContractError(
                    f"role {role_id!r} prefix references unknown pred row {pred_row_id!r}"
                )
            if (
                str(row.get("image_id")) != trajectory_key[0]
                or str(row.get("decode_mode")) != trajectory_key[1]
                or int(row.get("seed")) != trajectory_key[2]
                or row.get("source_artifact_path")
                != frozen_trajectory["source_artifact_path"]
                or row.get("source_artifact_sha256")
                != frozen_trajectory["source_artifact_sha256"]
            ):
                raise BehaviorContractError(
                    f"role {role_id!r} prefix pred-row lineage mismatch"
                )

        if role.get("role_kind") == "due_turn_context":
            successor_pred_row_id = str(role.get("reference_pred_row_id", ""))
            successor_row = prediction_index.get(successor_pred_row_id)
            if successor_row is None:
                raise BehaviorContractError(
                    f"due-turn role {role_id!r} references an unknown successor pred row"
                )
            if successor_row.get("strict_match_gt_owner_id") != owner_id:
                raise BehaviorContractError(
                    f"due-turn role {role_id!r} successor owner mismatch"
                )
            try:
                cut = frozen_trajectory["pred_row_ids"].index(successor_pred_row_id)
            except ValueError as exc:
                raise BehaviorContractError(
                    f"due-turn role {role_id!r} successor is outside its trajectory"
                ) from exc
            expected_prefix = _prefix_before_row_index(frozen_trajectory, cut)
            if (
                expected_prefix["token_ids"] != ids
                or expected_prefix["prefix_pred_row_ids"]
                != list(prefix.get("prefix_pred_row_ids", []))
            ):
                raise BehaviorContractError(
                    f"due-turn role {role_id!r} is not the exact prefix before its registered successor"
                )

    # The clean first-skip contrast is meaningful only as an asymmetric pair.
    pre = by_id.get("first_skip:P_pre")
    post = by_id.get("first_skip:P_post")
    first_skip_successor_binding: dict[str, Any] | None = None
    if (pre is None) != (post is None):
        raise BehaviorContractError("registry must contain both P_pre and P_post")
    if pre is not None and post is not None:
        pre_ids = _ids(_mapping(pre["prefix"], "P_pre.prefix")["token_ids"], "P_pre tokens")
        post_ids = _ids(_mapping(post["prefix"], "P_post.prefix")["token_ids"], "P_post tokens")
        if len(post_ids) <= len(pre_ids) or post_ids[: len(pre_ids)] != pre_ids:
            raise BehaviorContractError("P_post must append the exact native successor to P_pre")
        if pre.get("gt_owner_id") != post.get("gt_owner_id"):
            raise BehaviorContractError("P_pre/P_post target owner mismatch")
        if not str(post.get("successor_gt_owner_id", "")).startswith("gt:"):
            raise BehaviorContractError("P_post lacks a permanent successor owner id")
        successor_owner_id = str(post["successor_gt_owner_id"])
        successor_pred_row_id = str(post.get("successor_pred_row_id", ""))
        successor_owner = owner_index.get(successor_owner_id)
        successor_row = prediction_index.get(successor_pred_row_id)
        if successor_owner is None or str(successor_owner.get("image_id")) != str(
            _mapping(post["trajectory"], "P_post.trajectory").get("image_id")
        ):
            raise BehaviorContractError("P_pre/P_post successor owner is absent or on another image")
        if successor_row is None or successor_row.get("strict_match_gt_owner_id") != successor_owner_id:
            raise BehaviorContractError("P_pre/P_post successor pred row owner mismatch")
        pre_pred_ids = list(_mapping(pre["prefix"], "P_pre.prefix").get("prefix_pred_row_ids", []))
        post_pred_ids = list(_mapping(post["prefix"], "P_post.prefix").get("prefix_pred_row_ids", []))
        if post_pred_ids != [*pre_pred_ids, successor_pred_row_id]:
            raise BehaviorContractError(
                "P_post pred-row lineage must append exactly the registered successor to P_pre"
            )
        trajectory = _mapping(post["trajectory"], "P_post.trajectory")
        trajectory_key = (
            str(trajectory.get("image_id")),
            str(trajectory.get("decode_mode")),
            int(trajectory.get("seed")),
        )
        frozen_trajectory = trajectories.get(trajectory_key)
        if frozen_trajectory is None:
            raise BehaviorContractError("P_pre/P_post immutable trajectory is missing")
        try:
            successor_index = frozen_trajectory["pred_row_ids"].index(successor_pred_row_id)
        except ValueError as exc:
            raise BehaviorContractError(
                "P_pre/P_post successor pred row is outside the immutable trajectory"
            ) from exc
        expected_pre = _prefix_before_row_index(frozen_trajectory, successor_index)
        expected_post = _prefix_before_row_index(frozen_trajectory, successor_index + 1)
        if expected_pre["token_ids"] != pre_ids or expected_post["token_ids"] != post_ids:
            raise BehaviorContractError(
                "P_pre/P_post token extension is not the exact registered successor row"
            )
        first_skip_successor_binding = {
            "gt_owner_id": successor_owner_id,
            "pred_row_id": successor_pred_row_id,
            "pre_role_id": "first_skip:P_pre",
            "post_role_id": "first_skip:P_post",
            "pre_prefix_token_ids_sha256": sha256_json(pre_ids),
            "post_prefix_token_ids_sha256": sha256_json(post_ids),
            "source_artifact_path": frozen_trajectory["source_artifact_path"],
            "source_artifact_sha256": frozen_trajectory["source_artifact_sha256"],
        }

    return registry, selected, {
        "path": str(registry_path),
        "file_sha256": sha256_file(registry_path),
        "registry_digest": registry_digest,
        "verified_sources": source_receipt,
        "first_skip_successor_binding": first_skip_successor_binding,
    }


def validate_runtime_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolved_file(path, "runtime identity")
    file_sha256 = sha256_file(path)
    cache_key = (str(path), file_sha256)
    cached = _RUNTIME_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached[0]), deepcopy(cached[1])
    document = _read_json(path, "runtime identity")
    if document.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise BehaviorContractError("runtime identity schema_version is not recognized")
    if document.get("status") != "frozen":
        raise BehaviorContractError("runtime identity is not frozen")
    digest = _document_digest(document, "receipt_digest", "runtime identity")
    sources = _mapping(document.get("sources"), "runtime identity.sources")
    task0_ref = _mapping(
        sources.get("task0_execution_receipt"),
        "runtime identity.sources.task0_execution_receipt",
    )
    manifest_ref = _mapping(
        sources.get("production_run_manifest"),
        "runtime identity.sources.production_run_manifest",
    )
    task0_path = _resolved_file(
        str(task0_ref.get("path", "")), "Task-0 execution receipt"
    )
    manifest_path = _resolved_file(
        str(manifest_ref.get("path", "")), "production run manifest"
    )
    if (
        sha256_file(task0_path) != task0_ref.get("file_sha256")
        or sha256_file(manifest_path) != manifest_ref.get("file_sha256")
    ):
        raise BehaviorContractError(
            "runtime identity source receipt/manifest file digest mismatch"
        )
    try:
        from scripts.research.build_sorted_owner_basin_runtime_identity import (
            build_sorted_owner_basin_runtime_identity,
        )

        with tempfile.TemporaryDirectory(prefix="sorted-fn-runtime-identity-") as temp_dir:
            rebuilt = build_sorted_owner_basin_runtime_identity(
                execution_receipt=task0_path,
                production_manifest=manifest_path,
                output=Path(temp_dir) / "runtime-identity.json",
            )
    except (OSError, ValueError) as exc:
        raise BehaviorContractError(
            f"frozen runtime identity could not be recomputed from source files: {exc}"
        ) from exc
    if rebuilt != document:
        raise BehaviorContractError(
            "frozen runtime identity differs from exact source recomputation"
        )
    task0 = _read_json(task0_path, "Task-0 execution receipt")
    panel_ref = _mapping(
        _mapping(task0.get("inputs"), "Task-0 inputs").get("panel"),
        "Task-0 inputs.panel",
    )
    panel_path = _resolved_file(str(panel_ref.get("path", "")), "Task-0 frozen panel")
    panel_sha256 = sha256_file(panel_path)
    if panel_sha256 != panel_ref.get("sha256"):
        raise BehaviorContractError("Task-0 frozen panel SHA-256 mismatch")
    receipt = {
        "path": str(path),
        "sha256": file_sha256,
        "receipt_digest": digest,
        "recomputation_status": "exact_source_rebuild_passed",
        "task0_execution_receipt": {
            "path": str(task0_path),
            "sha256": sha256_file(task0_path),
        },
        "production_run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "frozen_source_panel": {
            "path": str(panel_path),
            "sha256": panel_sha256,
        },
    }
    _RUNTIME_IDENTITY_CACHE[cache_key] = (deepcopy(document), deepcopy(receipt))
    return document, receipt


def validate_static_live_config_binding(
    *,
    infer_config_path: Path,
    source_jsonl_path: Path,
    runtime_identity: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse the complete infer config and bind it to frozen live identity."""

    try:
        from src.config.fingerprint import sha256_json as config_sha256_json
        from src.config.inference import load_infer_config

        resolved = load_infer_config(infer_config_path)
    except Exception as exc:
        raise BehaviorContractError(
            f"infer config is incomplete or invalid: {type(exc).__name__}: {exc}"
        ) from exc
    config = resolved.config
    if config.backend.type != "hf" or str(config.model.dtype) != "fp32":
        raise BehaviorContractError("live behavior config must be HF FP32")
    configured_source = Path(config.data.input_jsonl).expanduser().resolve(strict=True)
    if configured_source != source_jsonl_path:
        raise BehaviorContractError(
            "infer config source JSONL differs from the explicitly bound source"
        )
    frozen_panel = _mapping(
        runtime_receipt.get("frozen_source_panel"), "frozen source panel binding"
    )
    if (
        source_jsonl_path != Path(str(frozen_panel.get("path", ""))).resolve()
        or sha256_file(source_jsonl_path) != frozen_panel.get("sha256")
    ):
        raise BehaviorContractError(
            "source JSONL differs from the panel frozen by runtime identity"
        )
    runtime_source = _mapping(
        _mapping(runtime_identity.get("runtime"), "runtime identity.runtime").get(
            "identity_source"
        ),
        "runtime identity.runtime.identity_source",
    )
    generation_fingerprint = config_sha256_json(
        config.generation.model_dump(mode="json")
    )
    if resolved.fingerprint != _mapping(
        runtime_source.get("resolved_config_fingerprints"),
        "runtime resolved config fingerprints",
    ).get("infer_config"):
        raise BehaviorContractError(
            "resolved infer config fingerprint differs from frozen runtime identity"
        )
    if generation_fingerprint != runtime_source.get(
        "generation_config_fingerprint"
    ):
        raise BehaviorContractError(
            "generation config fingerprint differs from frozen runtime identity"
        )
    model_source = _mapping(
        _mapping(runtime_identity.get("model"), "runtime identity.model").get(
            "identity_source"
        ),
        "runtime identity.model.identity_source",
    )
    model_identity = _mapping(
        model_source.get("model_identity"), "frozen model identity"
    )
    adapter_identity = _mapping(
        model_identity.get("adapter"), "frozen adapter identity"
    )
    embedding_identity = _mapping(
        _mapping(model_identity.get("embedding_delta"), "frozen embedding delta").get(
            "identity"
        ),
        "frozen embedding delta identity",
    )
    configured_components = {
        "base_model_path": str(Path(config.model.base_model).expanduser().resolve()),
        "adapter_path": (
            None
            if config.adapter is None
            else str(Path(config.adapter.path).expanduser().resolve())
        ),
        "embedding_delta_path": (
            None
            if config.embedding_delta is None
            else str(Path(config.embedding_delta.path).expanduser().resolve())
        ),
    }
    frozen_components = {
        "base_model_path": _mapping(
            model_identity.get("base"), "frozen base identity"
        ).get("path"),
        "adapter_path": adapter_identity.get("adapter_path"),
        "embedding_delta_path": embedding_identity.get("delta_path"),
    }
    if configured_components != frozen_components:
        raise BehaviorContractError(
            "infer config model components differ from frozen runtime identity"
        )
    return {
        "status": "passed_cpu_before_live_load",
        "resolved_config_fingerprint": resolved.fingerprint,
        "generation_config_fingerprint": generation_fingerprint,
        "configured_components": configured_components,
        "source_jsonl": {
            "path": str(source_jsonl_path),
            "sha256": sha256_file(source_jsonl_path),
        },
    }


def validate_live_runtime_identity(
    *,
    observed_receipt: Mapping[str, Any],
    frozen_identity: Mapping[str, Any],
    resolved_config_fingerprint: str,
    generation_config_fingerprint: str,
) -> dict[str, Any]:
    """Compare an opened HF session to every frozen execution identity.

    This is called immediately after opening the session and before prompt
    materialization or any generation.  Every comparison is exact except the
    known non-semantic ``performance`` timing block, which the identity
    builder itself excludes.
    """

    model_block = _mapping(frozen_identity.get("model"), "frozen model identity")
    tokenizer_block = _mapping(
        frozen_identity.get("tokenizer"), "frozen tokenizer identity"
    )
    runtime_block = _mapping(
        frozen_identity.get("runtime"), "frozen runtime identity"
    )
    model_source = _mapping(model_block.get("identity_source"), "frozen model source")
    tokenizer_source = _mapping(
        tokenizer_block.get("identity_source"), "frozen tokenizer source"
    )
    runtime_source = _mapping(
        runtime_block.get("identity_source"), "frozen runtime source"
    )
    runtime_keys = (
        "backend",
        "backend_mode",
        "backend_version",
        "generation_config_fingerprint",
        "likelihood_semantics",
        "processor_identity",
        "response_family",
    )
    expected_runtime = {
        key: deepcopy(runtime_source[key])
        for key in runtime_keys
    }
    observed_runtime = {
        key: deepcopy(observed_receipt.get(key)) for key in expected_runtime
    }
    expected_settings = dict(
        _mapping(
            runtime_source.get("effective_settings"),
            "frozen runtime effective_settings",
        )
    )
    observed_settings = dict(
        _mapping(
            observed_receipt.get("effective_settings"),
            "observed runtime effective_settings",
        )
    )
    expected_settings.pop("performance", None)
    observed_settings.pop("performance", None)
    postload_keys = frozenset(
        {"observed_attn_implementation", "observed_model_dtype"}
    )
    if set(expected_settings) & postload_keys:
        raise BehaviorContractError(
            "frozen runtime effective_settings contain post-load observation fields"
        )
    unknown_observed_settings = (
        set(observed_settings) - set(expected_settings) - postload_keys
    )
    observed_configured_settings = {
        key: deepcopy(observed_settings[key])
        for key in expected_settings
        if key in observed_settings
    }
    expected_runtime["effective_settings"] = expected_settings
    observed_runtime["effective_settings"] = observed_configured_settings

    expected_attn = _mapping(
        _mapping(
            expected_settings.get("backend_options"),
            "frozen effective_settings.backend_options",
        ).get("hf"),
        "frozen effective_settings.backend_options.hf",
    ).get("attn_implementation")
    observed_attn = observed_settings.get("observed_attn_implementation")
    observed_dtype = observed_settings.get("observed_model_dtype")
    observed_dtype_valid = False
    if isinstance(observed_dtype, Mapping) and set(observed_dtype) == {
        "parameter_dtype_counts",
        "parameter_dtype_names",
    }:
        dtype_names = observed_dtype.get("parameter_dtype_names")
        dtype_counts = observed_dtype.get("parameter_dtype_counts")
        observed_dtype_valid = (
            isinstance(dtype_names, list)
            and dtype_names == ["torch.float32"]
            and isinstance(dtype_counts, Mapping)
            and set(dtype_counts) == {"torch.float32"}
            and isinstance(dtype_counts.get("torch.float32"), int)
            and not isinstance(dtype_counts.get("torch.float32"), bool)
            and dtype_counts["torch.float32"] > 0
        )
    checks = {
        "model_identity": observed_receipt.get("model_identity")
        == model_source.get("model_identity"),
        "model_identity_fingerprint": sha256_json(observed_receipt.get("model_identity"))
        == model_source.get("model_identity_fingerprint"),
        "tokenizer_identity": observed_receipt.get("tokenizer_identity")
        == tokenizer_source.get("tokenizer_identity"),
        "runtime_identity": observed_runtime == expected_runtime,
        "effective_settings_extra_fields": not unknown_observed_settings,
        "observed_attn_implementation": isinstance(expected_attn, str)
        and bool(expected_attn)
        and observed_attn == expected_attn,
        "observed_model_dtype": observed_dtype_valid,
        "processor_identity_fingerprint": sha256_json(
            observed_receipt.get("processor_identity")
        )
        == runtime_source.get("processor_identity_fingerprint"),
        "resolved_config_fingerprint": resolved_config_fingerprint
        == _mapping(
            runtime_source.get("resolved_config_fingerprints"),
            "frozen resolved config fingerprints",
        ).get("infer_config"),
        "generation_config_fingerprint": generation_config_fingerprint
        == runtime_source.get("generation_config_fingerprint")
        == observed_receipt.get("generation_config_fingerprint"),
        "precision": runtime_source.get("precision") == "float32",
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise BehaviorContractError(
            f"opened HF runtime identity mismatch before generation: {failed}"
        )
    return {
        "status": "passed_before_generation",
        "checks": checks,
        "observed_projection_sha256": sha256_json(
            {
                "model_identity": observed_receipt.get("model_identity"),
                "tokenizer_identity": observed_receipt.get("tokenizer_identity"),
                "runtime": observed_runtime,
                "resolved_config_fingerprint": resolved_config_fingerprint,
            }
        ),
        "frozen_projection_sha256": sha256_json(
            {
                "model_identity": model_source.get("model_identity"),
                "tokenizer_identity": tokenizer_source.get("tokenizer_identity"),
                "runtime": expected_runtime,
                "resolved_config_fingerprint": _mapping(
                    runtime_source.get("resolved_config_fingerprints"),
                    "frozen resolved config fingerprints",
                ).get("infer_config"),
            }
        ),
    }


def validate_owner_context_ledger(
    ledger_path: Path,
    *,
    roles: Sequence[Mapping[str, Any]],
    registry_digest: str,
    runtime_identity: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(ledger_path, "owner-context ledger")
    by_context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise BehaviorContractError("owner-context ledger schema_version is not recognized")
        context_id = str(row.get("context_id", ""))
        if not context_id or context_id in by_context:
            raise BehaviorContractError("owner-context ledger context ids are missing or duplicated")
        by_context[context_id] = row

    selected: dict[str, dict[str, Any]] = {}
    expected_identity_digest = runtime_identity.get("receipt_digest")
    for role in roles:
        role_id = str(role["role_id"])
        context_id = f"ctx:fn:{role_id}"
        row = by_context.get(context_id)
        if row is None:
            raise BehaviorContractError(f"owner-context ledger lacks role {role_id!r}")
        if row.get("gt_owner_id") != role.get("gt_owner_id"):
            raise BehaviorContractError(f"role {role_id!r} owner-context owner mismatch")
        if str(row.get("image_id")) != str(
            _mapping(role.get("trajectory"), f"role {role_id}.trajectory").get("image_id")
        ):
            raise BehaviorContractError(f"role {role_id!r} owner-context image mismatch")
        source_binding = _mapping(
            _mapping(row.get("context_provenance"), "context_provenance").get("source_binding"),
            "context_provenance.source_binding",
        )
        if source_binding.get("registry_digest") != registry_digest:
            raise BehaviorContractError(f"role {role_id!r} registry provenance mismatch")
        upstream = _mapping(row.get("upstream_digests"), "ledger upstream_digests")
        if upstream.get("fn_mechanism_registry_sha256") != registry_digest:
            raise BehaviorContractError(f"role {role_id!r} upstream registry digest mismatch")

        context = _mapping(row.get("context_tokens"), f"role {role_id}.context_tokens")
        role_tokens = _ids(
            _mapping(role.get("prefix"), f"role {role_id}.prefix").get("token_ids"),
            f"role {role_id}.registry tokens",
        )
        ledger_tokens = _ids(context.get("token_ids"), f"role {role_id}.ledger tokens")
        if ledger_tokens != role_tokens or context.get("token_ids_sha256") != sha256_json(role_tokens):
            raise BehaviorContractError(f"role {role_id!r} ledger did not copy exact registry tokens")
        prompt_count = context.get("prompt_prefix_token_count")
        if isinstance(prompt_count, bool) or not isinstance(prompt_count, int) or not 0 < prompt_count <= len(role_tokens):
            raise BehaviorContractError(f"role {role_id!r} prompt/self-prefix split is invalid")
        prompt = role_tokens[:prompt_count]
        self_prefix = role_tokens[prompt_count:]
        if context.get("prompt_token_ids_sha256") != sha256_json(prompt):
            raise BehaviorContractError(f"role {role_id!r} prompt digest mismatch")
        if context.get("self_prefix_generated_token_ids_sha256") != sha256_json(self_prefix):
            raise BehaviorContractError(f"role {role_id!r} self-prefix digest mismatch")

        canonical = _mapping(row.get("canonical_description"), "canonical_description")
        forced = _ids(
            canonical.get("forced_row_prefix_through_box_start_token_ids"),
            f"role {role_id}.forced description tokens",
            allow_empty=False,
        )
        if forced[0] != OBJECT_REF_START or forced[-2:] != [OBJECT_REF_END, BOX_START]:
            raise BehaviorContractError(f"role {role_id!r} canonical description grammar mismatch")
        if canonical.get("forced_row_prefix_through_box_start_sha256") != sha256_json(forced):
            raise BehaviorContractError(f"role {role_id!r} forced description digest mismatch")
        vocabulary = _mapping(row.get("runtime_vocabulary_receipt"), "runtime vocabulary receipt")
        if expected_identity_digest is not None and vocabulary.get("identity_receipt_digest") != expected_identity_digest:
            raise BehaviorContractError(f"role {role_id!r} runtime identity receipt mismatch")
        expected_vocabulary = {
            "model_vocab_size": runtime_identity.get("model_vocab_size"),
            "model_identity_sha256": _mapping(
                runtime_identity.get("model"), "runtime model identity"
            ).get("identity_sha256"),
            "tokenizer_identity_sha256": _mapping(
                runtime_identity.get("tokenizer"), "runtime tokenizer identity"
            ).get("identity_sha256"),
            "runtime_identity_sha256": _mapping(
                runtime_identity.get("runtime"), "runtime execution identity"
            ).get("identity_sha256"),
        }
        if any(
            vocabulary.get(key) != expected
            for key, expected in expected_vocabulary.items()
        ):
            raise BehaviorContractError(
                f"role {role_id!r} runtime vocabulary/model/tokenizer identity mismatch"
            )
        token_registry = _mapping(row.get("token_registry"), "ledger token_registry")
        token_registry_digest = token_registry.get("registry_sha256")
        token_registry_content = {
            key: value for key, value in token_registry.items() if key != "registry_sha256"
        }
        if (
            token_registry_digest != sha256_json(token_registry_content)
            or vocabulary.get("token_registry_sha256") != token_registry_digest
            or token_registry.get("model_vocab_size")
            != runtime_identity.get("model_vocab_size")
            or token_registry.get("schema_tokens") != runtime_identity.get("schema_tokens")
        ):
            raise BehaviorContractError(
                f"role {role_id!r} sealed token-registry identity mismatch"
            )
        selected[role_id] = {
            **deepcopy(row),
            "exact_full_prefix_token_ids": role_tokens,
            "prompt_token_ids": prompt,
            "self_prefix_token_ids": self_prefix,
            "forced_description_token_ids": forced,
        }
    return selected, {"path": str(ledger_path), "sha256": sha256_file(ledger_path), "row_count": len(rows)}


def validate_landscape_receipt(
    path: Path,
    *,
    registry_path: Path,
    registry_file_sha256: str,
    registry_digest: str,
    ledger_path: Path,
    ledger_sha256: str,
    role_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(path, "landscape merge receipt")
    if receipt.get("schema_version") != LANDSCAPE_RECEIPT_SCHEMA_VERSION:
        raise BehaviorContractError("landscape receipt schema_version is not recognized")
    if receipt.get("unit_id") != UNIT_ID:
        raise BehaviorContractError("landscape receipt unit_id mismatch")
    if receipt.get("generic_arbitrary_role_merger") is not True:
        raise BehaviorContractError(
            "landscape receipt is not the current generic arbitrary-role merger"
        )
    decision_channel = _mapping(
        receipt.get("decision_channel"), "landscape decision_channel"
    )
    if (
        decision_channel.get("name") != LANDSCAPE_DECISION_CHANNEL
        or float(decision_channel.get("primary_repetition_penalty_stratum", -1))
        != PRIMARY_REPETITION_PENALTY
        or decision_channel.get("auxiliary_policy_is_not_a_model_likelihood")
        is not True
    ):
        raise BehaviorContractError(
            "landscape receipt does not preserve the v2 raw-rp1 decision channel"
        )
    scorer_provenance = _mapping(
        receipt.get("successor_scorer_provenance"),
        "landscape successor_scorer_provenance",
    )
    if (
        scorer_provenance.get("unit_id") != UNIT_ID
        or scorer_provenance.get("row_schema_version")
        != SUCCESSOR_SCORE_ROW_SCHEMA_VERSION
        or scorer_provenance.get("receipt_schema_version")
        != SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION
    ):
        raise BehaviorContractError(
            "landscape receipt does not bind the current successor scorer schemas"
        )
    sources = _mapping(receipt.get("source_digests"), "landscape source_digests")
    registry_ref = _mapping(sources.get("fn_mechanism_registry"), "landscape registry ref")
    ledger_ref = _mapping(sources.get("owner_context_ledger"), "landscape ledger ref")
    if Path(str(registry_ref.get("path", ""))).expanduser().resolve() != registry_path or registry_ref.get("sha256") != registry_file_sha256:
        raise BehaviorContractError("landscape receipt registry provenance mismatch")
    if Path(str(ledger_ref.get("path", ""))).expanduser().resolve() != ledger_path or ledger_ref.get("sha256") != ledger_sha256:
        raise BehaviorContractError("landscape receipt owner-context provenance mismatch")
    decision_rules_ref = _mapping(
        sources.get("decision_rules"), "landscape decision-rules ref"
    )
    decision_rules_path = _resolved_file(
        str(decision_rules_ref.get("path", "")), "landscape decision rules"
    )
    if decision_rules_ref.get("sha256") != sha256_file(decision_rules_path):
        raise BehaviorContractError("landscape decision-rules digest mismatch")
    mechanism_ref = _mapping(
        sources.get("mechanism_decision_rules"),
        "landscape mechanism-decision-rules ref",
    )
    mechanism_path = _resolved_file(
        str(mechanism_ref.get("path", "")), "landscape mechanism decision rules"
    )
    if mechanism_ref.get("sha256") != sha256_file(mechanism_path):
        raise BehaviorContractError("landscape mechanism-rules digest mismatch")
    mechanism_document = _read_json(mechanism_path, "mechanism decision rules")
    if (
        mechanism_document.get("schema_version")
        != MECHANISM_DECISION_RULES_SCHEMA_VERSION
        or mechanism_document.get("unit_id") != UNIT_ID
    ):
        raise BehaviorContractError("mechanism decision rules schema/unit mismatch")
    mechanism_self_digest = _document_digest(
        mechanism_document, "self_digest", "mechanism decision rules"
    )
    mechanism_upstream = _mapping(
        mechanism_document.get("upstream_digests"),
        "mechanism decision rules upstream_digests",
    )
    if (
        mechanism_upstream.get("execution_landscape_decision_rules_sha256")
        != decision_rules_ref.get("sha256")
        or mechanism_upstream.get("fn_mechanism_registry_sha256")
        != registry_digest
        or mechanism_ref.get("self_digest") != mechanism_self_digest
        or mechanism_ref.get("parent_execution_rules_sha256")
        != decision_rules_ref.get("sha256")
    ):
        raise BehaviorContractError(
            "landscape mechanism-rules parent/self binding mismatch"
        )
    fixed_budget_ref = _mapping(
        sources.get("fixed_budget_candidates"),
        "landscape fixed-budget candidates ref",
    )
    fixed_budget_path = _resolved_file(
        str(fixed_budget_ref.get("path", "")), "landscape fixed-budget candidates"
    )
    if fixed_budget_ref.get("sha256") != sha256_file(fixed_budget_path):
        raise BehaviorContractError("landscape fixed-budget digest mismatch")
    planner_ref = _mapping(receipt.get("planner_receipt"), "landscape planner receipt")
    planner_path = _resolved_file(
        str(planner_ref.get("path", "")), "landscape planner receipt"
    )
    if planner_ref.get("sha256") != sha256_file(planner_path):
        raise BehaviorContractError("landscape planner receipt digest mismatch")
    planner = _read_json(planner_path, "landscape planner receipt")
    if (
        planner.get("schema_version") != PLANNER_RECEIPT_SCHEMA_VERSION
        or planner.get("unit_id") != UNIT_ID
        or planner.get("receipt_digest") != planner_ref.get("receipt_digest")
    ):
        raise BehaviorContractError("landscape planner receipt schema/digest mismatch")
    _document_digest(planner, "receipt_digest", "landscape planner receipt")
    planner_sources = _mapping(planner.get("sources"), "landscape planner sources")
    planner_registry_ref = _mapping(
        planner_sources.get("registry"), "landscape planner registry source"
    )
    if (
        Path(str(planner_registry_ref.get("path", ""))).expanduser().resolve()
        != registry_path
        or planner_registry_ref.get("sha256") != registry_file_sha256
    ):
        raise BehaviorContractError("landscape planner registry binding mismatch")
    rules_template_ref = _mapping(
        planner_sources.get("rules_template"),
        "landscape planner rules-template source",
    )
    rules_template_path = _resolved_file(
        str(rules_template_ref.get("path", "")),
        "landscape planner rules template",
    )
    if rules_template_ref.get("sha256") != sha256_file(rules_template_path):
        raise BehaviorContractError("landscape planner rules-template digest mismatch")
    if mechanism_upstream.get("rules_template_sha256") != rules_template_ref.get(
        "sha256"
    ):
        raise BehaviorContractError(
            "mechanism decision rules do not bind the planner's rules template"
        )
    planner_outputs = _mapping(planner.get("outputs"), "landscape planner outputs")
    expected_planner_outputs = {
        "owner_context_ledger": (ledger_path, ledger_ref.get("sha256")),
        "landscape_decision_rules": (
            decision_rules_path,
            decision_rules_ref.get("sha256"),
        ),
        "mechanism_decision_rules": (
            mechanism_path,
            mechanism_ref.get("sha256"),
        ),
        "fixed_budget_candidates": (
            fixed_budget_path,
            fixed_budget_ref.get("sha256"),
        ),
    }
    for output_key, (expected_path, expected_sha256) in expected_planner_outputs.items():
        output_ref = _mapping(
            planner_outputs.get(output_key),
            f"landscape planner output {output_key}",
        )
        if (
            Path(str(output_ref.get("path", ""))).expanduser().resolve()
            != expected_path
            or output_ref.get("sha256") != expected_sha256
        ):
            raise BehaviorContractError(
                f"landscape planner output {output_key} binding mismatch"
            )
    neighborhood_assertion = _mapping(
        planner.get("neighborhood_consistency"),
        "landscape planner neighborhood consistency",
    )
    if neighborhood_assertion.get("status") != "consistent":
        raise BehaviorContractError(
            "landscape planner lacks a passed candidate-neighborhood assertion"
        )
    selected = set(
        str(value)
        for value in _sequence(
            _mapping(receipt.get("context_selection"), "landscape context_selection").get("selected_context_ids"),
            "landscape selected_context_ids",
        )
    )
    missing = sorted({f"ctx:fn:{role_id}" for role_id in role_ids} - selected)
    if missing:
        raise BehaviorContractError(f"selected behavior roles lack landscape scores: {missing}")
    selected_rungs = [
        str(value)
        for value in _sequence(
            receipt.get("selected_rungs"), "landscape selected_rungs"
        )
    ]
    if not selected_rungs or len(selected_rungs) != len(set(selected_rungs)):
        raise BehaviorContractError(
            "landscape selected_rungs are empty or duplicated"
        )
    return receipt, {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": receipt["schema_version"],
        "decision_bearing_channel": LANDSCAPE_DECISION_CHANNEL,
        "native_repetition_penalty_stratum": PRIMARY_REPETITION_PENALTY,
        "selected_rungs": selected_rungs,
        "successor_scorer_schema": {
            "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
            "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
        },
        "mechanism_decision_rules": {
            "path": str(mechanism_path),
            "sha256": sha256_file(mechanism_path),
            "self_digest": mechanism_self_digest,
            "parent_execution_rules_sha256": decision_rules_ref.get("sha256"),
            "planner_receipt_digest": planner_ref.get("receipt_digest"),
            "registry_digest": registry_digest,
            "fixed_budget_candidates_sha256": fixed_budget_ref.get("sha256"),
        },
    }


def validate_landscape_admission(
    path: Path,
    *,
    registry_digest: str,
    landscape_receipt_sha256: str,
    mechanism_decision_rules_binding: Mapping[str, Any],
    selected_roles: Sequence[Mapping[str, Any]],
    selected_rungs: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a passed likelihood-landscape decision for every behavior role."""

    document = _read_json(path, "behavior landscape admission")
    if (
        document.get("schema_version") != LANDSCAPE_ADMISSION_SCHEMA_VERSION
        or document.get("status") != "passed"
    ):
        raise BehaviorContractError(
            "behavior landscape admission schema/status is not passed"
        )
    if document.get("registry_digest") != registry_digest:
        raise BehaviorContractError("behavior landscape admission registry mismatch")
    if document.get("landscape_receipt_sha256") != landscape_receipt_sha256:
        raise BehaviorContractError("behavior landscape admission receipt mismatch")
    expected_mechanism_binding = {
        key: value
        for key, value in mechanism_decision_rules_binding.items()
        if key != "path"
    }
    if document.get("mechanism_decision_rules") != expected_mechanism_binding:
        raise BehaviorContractError(
            "behavior landscape admission mechanism-rules binding mismatch"
        )
    entries = _sequence(document.get("admitted_roles"), "landscape admitted_roles")
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in entries:
        entry = _mapping(raw, "landscape admitted role")
        role_id = str(entry.get("role_id", ""))
        if not role_id or role_id in by_role:
            raise BehaviorContractError(
                "landscape admitted role ids are missing or duplicated"
            )
        by_role[role_id] = entry
    selected_ids = {str(role["role_id"]) for role in selected_roles}
    missing = sorted(selected_ids - set(by_role))
    if missing:
        raise BehaviorContractError(
            f"selected behavior roles lack passed landscape admission: {missing}"
        )
    for role in selected_roles:
        role_id = str(role["role_id"])
        entry = by_role[role_id]
        condition = _mapping(
            entry.get("landscape_condition"),
            f"landscape admission {role_id}.landscape_condition",
        )
        if (
            entry.get("status") != "passed"
            or entry.get("gt_owner_id") != role.get("gt_owner_id")
            or entry.get("context_id") != f"ctx:fn:{role_id}"
            or condition.get("status") != "passed"
            or condition.get("name") not in ADMITTED_LANDSCAPE_CONDITIONS
            or condition.get("rung") not in {"L1", "L2"}
            or condition.get("rung") not in set(selected_rungs)
        ):
            raise BehaviorContractError(
                f"selected role {role_id!r} lacks an owner-matched passed landscape condition"
            )
    receipt = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": LANDSCAPE_ADMISSION_SCHEMA_VERSION,
        "admitted_role_ids": sorted(selected_ids),
    }
    return document, receipt


def validate_description_equivalence(
    path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load an optional frozen description/category equivalence map.

    Without this artifact semantic relation is deliberately unknown; lexical
    inequality is never promoted to semantic drift.
    """

    if path is None:
        return None, {
            "status": "not_supplied_semantic_relation_neutral",
            "semantic_drift_decision_eligible": False,
        }
    resolved = _resolved_file(path, "description equivalence")
    document = _read_json(resolved, "description equivalence")
    if (
        document.get("schema_version") != DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION
        or document.get("status") != "frozen"
    ):
        raise BehaviorContractError(
            "description equivalence schema/status is not frozen"
        )
    digest = _document_digest(
        document, "equivalence_digest", "description equivalence"
    )
    categories = _mapping(document.get("categories"), "description equivalence.categories")
    alias_owner: dict[str, str] = {}
    for category, raw_aliases in categories.items():
        category_name = _normalise_description(category)
        aliases = {
            _normalise_description(value)
            for value in _sequence(raw_aliases, f"description aliases {category}")
        }
        aliases.add(category_name)
        if not category_name or "" in aliases:
            raise BehaviorContractError("description equivalence contains an empty alias")
        for alias in aliases:
            existing = alias_owner.get(alias)
            if existing is not None and existing != category_name:
                raise BehaviorContractError(
                    f"description alias {alias!r} belongs to multiple categories"
                )
            alias_owner[alias] = category_name
    if not alias_owner:
        raise BehaviorContractError("description equivalence contains no aliases")
    return {**document, "alias_to_category": alias_owner}, {
        "status": "frozen",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "equivalence_digest": digest,
        "semantic_drift_decision_eligible": True,
    }


def classify_semantic_relation(
    *,
    observed_description: Any,
    canonical_description: Any,
    iou_to_target: float,
    equivalence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observed = _normalise_description(observed_description)
    canonical = _normalise_description(canonical_description)
    if equivalence is None:
        relation = "unknown_neutral_no_frozen_equivalence"
    else:
        aliases = _mapping(
            equivalence.get("alias_to_category"), "description alias_to_category"
        )
        observed_category = aliases.get(observed)
        canonical_category = aliases.get(canonical)
        if observed_category is None or canonical_category is None:
            relation = "unknown_neutral_unmapped_description"
        elif observed_category == canonical_category:
            relation = "equivalent"
        else:
            relation = "semantic_mismatch"
    high_overlap = float(iou_to_target) >= SEMANTIC_DRIFT_IOU_THRESHOLD
    return {
        "observed_normalized_description": observed,
        "canonical_normalized_description": canonical,
        "relation": relation,
        "iou_to_target": float(iou_to_target),
        "high_overlap": high_overlap,
        "semantic_drift_supported": bool(
            high_overlap and relation == "semantic_mismatch"
        ),
    }


def validate_repetition_penalties(values: Sequence[float]) -> tuple[float, ...]:
    requested = tuple(dict.fromkeys(float(value) for value in (values or [1.0])))
    if any(value not in {PRIMARY_REPETITION_PENALTY, POLICY_REPETITION_PENALTY} for value in requested):
        raise BehaviorContractError("repetition penalty arms are limited to 1.00 and 1.10")
    return requested


def _prior_greedy_canonical_recovery_failed(role: Mapping[str, Any]) -> bool:
    """Recompute the prior forced-greedy autonomous-suffix recovery result."""

    arms = _mapping(role.get("arms"), "prior behavior role arms")
    forced = _mapping(
        arms.get("forced_description_greedy"),
        "prior behavior forced-description greedy arm",
    )
    rows = list(_sequence(forced.get("rows"), "prior greedy rows"))
    released = list(
        _sequence(forced.get("released_suffix_rows"), "prior greedy released rows")
    )
    autonomous_ids = [
        str(value)
        for value in _sequence(
            forced.get("autonomous_evidence_row_ids"),
            "prior greedy autonomous row ids",
        )
    ]
    intervention = _mapping(
        forced.get("intervention"), "prior greedy intervention"
    )
    if (
        forced.get("arm_kind") != "forced_description"
        or not rows
        or released != rows[1:]
        or intervention.get("row") != rows[0]
        or intervention.get("accounting_status")
        != "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set"
        or autonomous_ids
        != [
            str(_mapping(row, "prior greedy released row").get("pred_row_id"))
            for row in released
        ]
    ):
        raise BehaviorContractError(
            "bound prior behavior does not expose the exact forced-greedy autonomous suffix partition"
        )
    target = str(role.get("gt_owner_id"))
    strict = {
        str(owner)
        for raw_row in released
        for owner in _sequence(
            _mapping(raw_row, "prior greedy released row").get(
                "strict_matched_owner_ids", []
            ),
            "prior greedy strict owners",
        )
    }
    loose = {
        str(owner)
        for raw_row in released
        for owner in _sequence(
            _mapping(raw_row, "prior greedy released row").get(
                "loose_matched_owner_ids", []
            ),
            "prior greedy loose owners",
        )
    }
    declared_strict = forced.get("autonomous_suffix_target_recovered_strict")
    declared_loose = forced.get("autonomous_suffix_target_recovered_loose")
    if (
        not isinstance(declared_strict, bool)
        or not isinstance(declared_loose, bool)
        or declared_strict != (target in strict)
        or declared_loose != (target in loose)
    ):
        raise BehaviorContractError(
            "bound prior behavior greedy recovery booleans disagree with its autonomous suffix rows"
        )
    return not declared_strict and not declared_loose


def validate_sampling_admission(
    admission: Mapping[str, Any] | None,
    *,
    sampling_flags: Mapping[str, Any],
    registry_digest: str,
    landscape_receipt_sha256: str,
    mechanism_decision_rules_binding: Mapping[str, Any],
    selected_roles: Sequence[Mapping[str, Any]],
    selected_rungs: Sequence[str],
) -> dict[str, Any] | None:
    """Validate conditional sampling and its frozen decode tuple."""

    supplied_flags = {key: value for key, value in sampling_flags.items() if value is not None and value != []}
    if admission is None:
        if supplied_flags:
            raise BehaviorContractError("sampling flags require an explicit passed admission JSON")
        return None
    if (
        admission.get("schema_version") != SAMPLING_ADMISSION_SCHEMA_VERSION
        or admission.get("unit_id") != UNIT_ID
        or admission.get("status") != "passed"
    ):
        raise BehaviorContractError("sampling admission schema/status is not passed")
    if admission.get("registry_digest") != registry_digest:
        raise BehaviorContractError("sampling admission registry digest mismatch")
    if admission.get("landscape_receipt_sha256") != landscape_receipt_sha256:
        raise BehaviorContractError("sampling admission landscape receipt mismatch")
    mechanism_path = _resolved_file(
        str(mechanism_decision_rules_binding.get("path", "")),
        "sampling mechanism decision rules",
    )
    mechanism_document = _read_json(
        mechanism_path, "sampling mechanism decision rules"
    )
    if (
        mechanism_document.get("schema_version")
        != MECHANISM_DECISION_RULES_SCHEMA_VERSION
        or mechanism_document.get("unit_id") != UNIT_ID
    ):
        raise BehaviorContractError(
            "sampling mechanism decision rules schema/unit mismatch"
        )
    mechanism_self_digest = _document_digest(
        mechanism_document,
        "self_digest",
        "sampling mechanism decision rules",
    )
    mechanism_conditional = _mapping(
        mechanism_document.get("conditional_sampling"),
        "mechanism rules conditional_sampling",
    )
    if (
        any(
            isinstance(mechanism_conditional.get(key), bool)
            or not isinstance(mechanism_conditional.get(key), (int, float))
            for key in ("temperature", "top_p", "repetition_penalty")
        )
        or not isinstance(mechanism_conditional.get("null_semantics"), str)
        or not mechanism_conditional.get("null_semantics")
    ):
        raise BehaviorContractError(
            "sampling mechanism decision rules do not freeze a complete decode/null tuple"
        )
    expected_mechanism_binding = {
        "path": str(mechanism_path),
        "sha256": sha256_file(mechanism_path),
        "self_digest": mechanism_self_digest,
        "conditional_sampling": {
            "temperature": mechanism_conditional.get("temperature"),
            "top_p": mechanism_conditional.get("top_p"),
            "repetition_penalty": mechanism_conditional.get("repetition_penalty"),
            "null_semantics": mechanism_conditional.get("null_semantics"),
        },
    }
    if (
        mechanism_decision_rules_binding.get("sha256")
        != expected_mechanism_binding["sha256"]
        or mechanism_decision_rules_binding.get("self_digest")
        != mechanism_self_digest
        or admission.get("mechanism_decision_rules")
        != expected_mechanism_binding
    ):
        raise BehaviorContractError(
            "sampling admission mechanism-rules binding mismatch"
        )
    condition = _mapping(
        admission.get("landscape_condition"), "sampling landscape_condition"
    )
    if (
        condition.get("status") != "passed"
        or condition.get("name")
        != "per_role_localized_landscape_with_greedy_failure"
        or condition.get("greedy_canonical_description_recovery_failed") is not True
    ):
        raise BehaviorContractError(
            "sampling admission lacks the per-role landscape plus prior-greedy-failure gate"
        )
    admitted_roles = [
        str(value) for value in _sequence(admission.get("admitted_role_ids"), "admitted_role_ids")
    ]
    if len(admitted_roles) != len(set(admitted_roles)) or not admitted_roles:
        raise BehaviorContractError("sampling admitted role ids are empty or duplicated")
    selected_by_id = {
        str(role.get("role_id")): role for role in selected_roles
    }
    if len(selected_by_id) != len(selected_roles):
        raise BehaviorContractError("selected sampling roles are duplicated")
    unknown = sorted(set(admitted_roles) - set(selected_by_id))
    if unknown:
        raise BehaviorContractError(f"sampling admission names unselected roles: {unknown}")
    behavior_ref = _mapping(
        admission.get("behavior_output"), "sampling behavior output binding"
    )
    behavior_path = _resolved_file(
        str(behavior_ref.get("path", "")), "sampling prior behavior output"
    )
    if behavior_ref.get("sha256") != sha256_file(behavior_path):
        raise BehaviorContractError("sampling prior behavior output digest mismatch")
    behavior_document = _read_json(behavior_path, "sampling prior behavior output")
    if (
        behavior_document.get("schema_version") != SCHEMA_VERSION
        or behavior_document.get("unit_id") != UNIT_ID
    ):
        raise BehaviorContractError("sampling prior behavior output schema/unit mismatch")
    behavior_content_sha256 = _document_digest(
        behavior_document,
        "output_content_sha256",
        "sampling prior behavior output",
    )
    if behavior_ref.get("output_content_sha256") != behavior_content_sha256:
        raise BehaviorContractError(
            "sampling prior behavior content identity mismatch"
        )
    behavior_contract = _mapping(
        behavior_document.get("contract"), "sampling prior behavior contract"
    )
    behavior_registry = _mapping(
        behavior_contract.get("registry"), "sampling prior behavior registry"
    )
    behavior_landscape = _mapping(
        behavior_contract.get("landscape"), "sampling prior behavior landscape"
    )
    if (
        behavior_registry.get("registry_digest") != registry_digest
        or behavior_landscape.get("sha256") != landscape_receipt_sha256
        or behavior_contract.get("sampling_admission") is not None
    ):
        raise BehaviorContractError(
            "sampling prior behavior is not the exact greedy-only registry/landscape run"
        )
    primary_views = [
        _mapping(view, "sampling prior behavior policy view")
        for view in _sequence(
            behavior_document.get("policy_views"),
            "sampling prior behavior policy views",
        )
        if float(_mapping(view, "sampling prior behavior policy view").get(
            "repetition_penalty", -1
        ))
        == 1.0
    ]
    if len(primary_views) != 1:
        raise BehaviorContractError(
            "sampling prior behavior must contain exactly one rp=1.0 policy view"
        )
    behavior_roles = [
        _mapping(role, "sampling prior behavior role")
        for role in _sequence(
            primary_views[0].get("roles"), "sampling prior behavior roles"
        )
    ]
    behavior_by_id = {str(role.get("role_id")): role for role in behavior_roles}
    if len(behavior_by_id) != len(behavior_roles):
        raise BehaviorContractError("sampling prior behavior role ids are duplicated")
    condition_by_role = _mapping(
        condition.get("condition_by_role"), "sampling condition_by_role"
    )
    if set(condition_by_role) != set(admitted_roles):
        raise BehaviorContractError(
            "sampling per-role conditions do not exactly match admitted roles"
        )
    for role_id in admitted_roles:
        selected_role = selected_by_id[role_id]
        behavior_role = behavior_by_id.get(role_id)
        if behavior_role is None:
            raise BehaviorContractError(
                f"sampling prior behavior omits admitted role {role_id!r}"
            )
        role_condition = _mapping(
            condition_by_role[role_id], f"sampling condition for {role_id}"
        )
        landscape_condition = _mapping(
            role_condition.get("landscape_condition"),
            f"sampling landscape condition for {role_id}",
        )
        greedy_condition = _mapping(
            role_condition.get("greedy_canonical_description_recovery"),
            f"sampling greedy condition for {role_id}",
        )
        if (
            role_condition.get("role_id") != role_id
            or role_condition.get("context_id") != f"ctx:fn:{role_id}"
            or role_condition.get("gt_owner_id")
            != selected_role.get("gt_owner_id")
            or behavior_role.get("gt_owner_id") != selected_role.get("gt_owner_id")
            or landscape_condition.get("status") != "passed"
            or landscape_condition.get("name")
            not in ADMITTED_LANDSCAPE_CONDITIONS
            or landscape_condition.get("rung") not in {"L1", "L2"}
            or landscape_condition.get("rung") not in set(selected_rungs)
            or greedy_condition.get("status") != "failed"
            or greedy_condition.get("arm") != "forced_description_greedy"
            or greedy_condition.get("target_recovered") is not False
            or greedy_condition.get("behavior_role_content_sha256")
            != sha256_json(behavior_role)
            or not _prior_greedy_canonical_recovery_failed(behavior_role)
        ):
            raise BehaviorContractError(
                f"sampling role {role_id!r} lacks exact localized-landscape and prior greedy-failure evidence"
            )
    decode = dict(_mapping(admission.get("decode_parameters"), "sampling decode_parameters"))
    seeds = _ids(decode.get("seeds"), "sampling seeds", allow_empty=False)
    k = decode.get("k")
    horizon = decode.get("horizon_rows")
    if (
        float(decode.get("temperature", -1))
        != float(mechanism_conditional.get("temperature", -2))
        or float(decode.get("top_p", -1))
        != float(mechanism_conditional.get("top_p", -2))
        or float(decode.get("repetition_penalty", -1))
        != float(mechanism_conditional.get("repetition_penalty", -2))
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k != len(seeds)
        or len(set(seeds)) != len(seeds)
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 8
        or admission.get("finite_sampling_failure_semantics")
        != mechanism_conditional.get("null_semantics")
    ):
        raise BehaviorContractError(
            "sampling decode parameters/null semantics disagree with mechanism rules or K/seeds/horizon constraints"
        )
    assertions = {
        "temperature": sampling_flags.get("temperature"),
        "top_p": sampling_flags.get("top_p"),
        "k": sampling_flags.get("k"),
        "horizon_rows": sampling_flags.get("horizon_rows"),
        "seeds": sampling_flags.get("seeds"),
    }
    for key, asserted in assertions.items():
        if asserted is None or asserted == []:
            continue
        expected = seeds if key == "seeds" else decode[key]
        observed = [int(v) for v in asserted] if key == "seeds" else asserted
        if observed != expected:
            raise BehaviorContractError(f"sampling flag {key} disagrees with admission")
    return {**deepcopy(dict(admission)), "decode_parameters": {**decode, "seeds": seeds}}


def _prediction_index(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(contract["registry"]["verified_sources"]["prediction_row_ledger"]["path"])
    return {str(row["pred_row_id"]): row for row in _read_jsonl(path, "prediction row ledger")}


def _prefix_owner_ids(role: Mapping[str, Any], prediction_index: Mapping[str, Mapping[str, Any]]) -> list[str]:
    prefix = _mapping(role.get("prefix"), "role.prefix")
    owners = []
    for pred_row_id in _sequence(prefix.get("prefix_pred_row_ids", []), "prefix_pred_row_ids"):
        row = prediction_index.get(str(pred_row_id))
        if row is None:
            raise BehaviorContractError(f"prefix references unknown pred_row_id {pred_row_id!r}")
        owner = row.get("strict_match_gt_owner_id")
        if owner is not None:
            owners.append(str(owner))
    return sorted(set(owners))


def build_contract(
    *,
    registry_path: Path,
    owner_context_ledger_path: Path,
    landscape_receipt_path: Path,
    landscape_admission_path: Path,
    runtime_identity_path: Path,
    infer_config_path: Path,
    source_jsonl_path: Path,
    include_role_ids: Sequence[str] = (),
    repetition_penalties: Sequence[float] = (),
    suffix_horizon_rows: int = 4,
    max_new_tokens: int = 128,
    malformed_limit: int = 2,
    sampling_admission_path: Path | None = None,
    sampling_flags: Mapping[str, Any] | None = None,
    description_equivalence_path: Path | None = None,
) -> dict[str, Any]:
    """Build the complete CPU-only live-execution contract."""

    if not 1 <= int(suffix_horizon_rows) <= 8:
        raise BehaviorContractError("suffix horizon rows must be in [1,8]")
    if int(max_new_tokens) <= 0 or int(malformed_limit) <= 0:
        raise BehaviorContractError("generation token/malformed limits must be positive")
    registry_path = _resolved_file(registry_path, "FN mechanism registry")
    ledger_path = _resolved_file(owner_context_ledger_path, "owner-context ledger")
    landscape_path = _resolved_file(landscape_receipt_path, "landscape receipt")
    runtime_path = _resolved_file(runtime_identity_path, "runtime identity")
    infer_path = _resolved_file(infer_config_path, "infer config")
    source_path = _resolved_file(source_jsonl_path, "source JSONL")
    registry, roles, registry_receipt = validate_registry(registry_path, include_role_ids)
    runtime_identity, runtime_receipt = validate_runtime_identity(runtime_path)
    static_live_config_binding = validate_static_live_config_binding(
        infer_config_path=infer_path,
        source_jsonl_path=source_path,
        runtime_identity=runtime_identity,
        runtime_receipt=runtime_receipt,
    )
    ledger, ledger_receipt = validate_owner_context_ledger(
        ledger_path,
        roles=roles,
        registry_digest=registry_receipt["registry_digest"],
        runtime_identity=runtime_identity,
    )
    landscape, landscape_receipt = validate_landscape_receipt(
        landscape_path,
        registry_path=registry_path,
        registry_file_sha256=registry_receipt["file_sha256"],
        registry_digest=registry_receipt["registry_digest"],
        ledger_path=ledger_path,
        ledger_sha256=ledger_receipt["sha256"],
        role_ids=[str(role["role_id"]) for role in roles],
    )
    landscape_admission_path = _resolved_file(
        landscape_admission_path, "behavior landscape admission"
    )
    _landscape_admission, landscape_admission_receipt = validate_landscape_admission(
        landscape_admission_path,
        registry_digest=registry_receipt["registry_digest"],
        landscape_receipt_sha256=landscape_receipt["sha256"],
        mechanism_decision_rules_binding=landscape_receipt[
            "mechanism_decision_rules"
        ],
        selected_roles=roles,
        selected_rungs=landscape_receipt["selected_rungs"],
    )
    description_equivalence, description_equivalence_receipt = (
        validate_description_equivalence(description_equivalence_path)
    )
    admission_document = (
        _read_json(_resolved_file(sampling_admission_path, "sampling admission"), "sampling admission")
        if sampling_admission_path is not None
        else None
    )
    sampling = validate_sampling_admission(
        admission_document,
        sampling_flags=sampling_flags or {},
        registry_digest=registry_receipt["registry_digest"],
        landscape_receipt_sha256=landscape_receipt["sha256"],
        mechanism_decision_rules_binding=landscape_receipt[
            "mechanism_decision_rules"
        ],
        selected_roles=roles,
        selected_rungs=landscape_receipt["selected_rungs"],
    )
    policies = validate_repetition_penalties(repetition_penalties)
    prediction_index = {
        str(row["pred_row_id"]): row
        for row in _read_jsonl(
            Path(registry_receipt["verified_sources"]["prediction_row_ledger"]["path"]),
            "prediction row ledger",
        )
    }
    successor_binding = registry_receipt.get("first_skip_successor_binding")
    successor_id = (
        None
        if successor_binding is None
        else str(_mapping(successor_binding, "first-skip successor binding")["gt_owner_id"])
    )
    normalized_roles = []
    for role in roles:
        role_id = str(role["role_id"])
        successor = role.get("successor_gt_owner_id")
        if role_id == "first_skip:P_pre":
            successor = successor_id
        normalized_roles.append(
            {
                **deepcopy(dict(role)),
                "successor_gt_owner_id": successor,
                "prefix_owner_ids": _prefix_owner_ids(role, prediction_index),
                "ledger": ledger[role_id],
            }
        )
    return {
        "schema_version": CONTRACT_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "registry": registry_receipt,
        "owner_context_ledger": ledger_receipt,
        "landscape": landscape_receipt,
        "landscape_admission": landscape_admission_receipt,
        "runtime_identity": runtime_receipt,
        "static_live_config_binding": static_live_config_binding,
        "frozen_runtime_identity": {
            "model": deepcopy(runtime_identity["model"]),
            "tokenizer": deepcopy(runtime_identity["tokenizer"]),
            "runtime": deepcopy(runtime_identity["runtime"]),
        },
        "infer_config": {"path": str(infer_path), "sha256": sha256_file(infer_path)},
        "source_jsonl": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "selected_role_ids": [str(role["role_id"]) for role in normalized_roles],
        "roles": normalized_roles,
        "policy_repetition_penalties": list(policies),
        "generation_limits": {
            "suffix_horizon_rows": int(suffix_horizon_rows),
            "max_new_tokens": int(max_new_tokens),
            "malformed_limit": int(malformed_limit),
        },
        "sampling_admission": sampling,
        "description_equivalence": description_equivalence,
        "description_equivalence_receipt": description_equivalence_receipt,
        "likelihood_semantics": {
            "source": "bound_external_landscape_receipt",
            "raw_likelihood_relabelled": False,
            "rp_1_00_and_rp_1_10_are_separate_behavior_policy_views": True,
        },
    }


def first_divergence(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if int(left_value) != int(right_value):
            return {"index": index, "left_token_id": int(left_value), "right_token_id": int(right_value)}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left_token_id": int(left[len(right)]) if len(left) > len(right) else None,
            "right_token_id": int(right[len(left)]) if len(right) > len(left) else None,
        }
    return {"index": None, "left_token_id": None, "right_token_id": None}


def _complete(row: Mapping[str, Any]) -> bool:
    stop = row.get("row_stop")
    return bool(
        row.get("status") == "success"
        and isinstance(stop, Mapping)
        and stop.get("stop_reason") == "complete_row"
        and row.get("raw_generated_token_ids")
    )


def _owner_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {
        str(owner)
        for row in rows
        for owner in row.get(key, [])
    }


def _row_id(
    *, role_id: str, arm_id: str, image_id: str, policy_id: str, seed: int | None, row_index: int, input_prefix_digest: str
) -> str:
    seed_text = "none" if seed is None else str(seed)
    suffix = sha256_json(
        [UNIT_ID, role_id, arm_id, image_id, policy_id, seed_text, row_index, input_prefix_digest]
    )[:16]
    return f"pred:sorted:successor:{policy_id}:{image_id}:{role_id}:{arm_id}:{seed_text}:{row_index}:{suffix}"


def _freeze_row(
    row: Mapping[str, Any],
    *,
    role: Mapping[str, Any],
    arm_id: str,
    policy_id: str,
    seed: int | None,
    row_index: int,
    input_prefix: Sequence[int],
) -> dict[str, Any]:
    result = deepcopy(dict(row))
    ids = _ids(result.get("raw_generated_token_ids", []), "generated row tokens")
    digest = sha256_json(ids)
    declared = result.get("raw_generated_token_ids_sha256")
    if declared is not None and declared != digest:
        raise BehaviorContractError("generated row token digest mismatch")
    prefix = [int(value) for value in input_prefix]
    prefix_digest = sha256_json(prefix)
    pred_row_id = _row_id(
        role_id=str(role["role_id"]),
        arm_id=arm_id,
        image_id=str(_mapping(role["trajectory"], "role trajectory")["image_id"]),
        policy_id=policy_id,
        seed=seed,
        row_index=row_index,
        input_prefix_digest=prefix_digest,
    )
    predictions = []
    for prediction_index, prediction in enumerate(result.get("parsed_predictions", [])):
        item = dict(prediction)
        item["pred_row_id"] = pred_row_id
        item["prediction_index"] = prediction_index
        predictions.append(item)
    result.update(
        {
            "pred_row_id": pred_row_id,
            "row_index": row_index,
            "input_prefix_token_ids": prefix,
            "input_prefix_token_ids_sha256": prefix_digest,
            "raw_generated_token_ids": ids,
            "raw_generated_token_ids_sha256": digest,
            "parsed_predictions": predictions,
            "description_box_evidence": [
                {
                    "prediction_index": item["prediction_index"],
                    "description": item.get("description"),
                    "bbox": item.get("bbox", item.get("bbox_xyxy")),
                }
                for item in predictions
            ],
        }
    )
    return result


def _append(prefix: Sequence[int], row: Mapping[str, Any]) -> list[int]:
    return [*map(int, prefix), *map(int, row.get("raw_generated_token_ids", []))] if _complete(row) else list(map(int, prefix))


Generator = Callable[..., Mapping[str, Any]]


def _run_arm(
    *,
    generator: Generator,
    role: Mapping[str, Any],
    arm_id: str,
    policy_id: str,
    repetition_penalty: float,
    prefix: Sequence[int],
    forced: Sequence[int] | None,
    mode: str,
    seed: int | None,
    temperature: float,
    top_p: float,
    horizon_rows: int,
    require_target_recovery_before_suffix: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    current = list(map(int, prefix))
    first = generator(
        role=role,
        arm_id=arm_id,
        prefix_token_ids=current,
        forced_row_prefix_token_ids=forced,
        mode=mode,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        row_index=0,
    )
    first_row = _freeze_row(
        first, role=role, arm_id=arm_id, policy_id=policy_id, seed=seed, row_index=0, input_prefix=current
    )
    rows.append(first_row)
    current = _append(current, first_row)
    target = str(role["gt_owner_id"])
    is_intervention = forced is not None
    intervention_target_strict = bool(
        is_intervention
        and target in set(first_row.get("strict_matched_owner_ids", []))
    )
    intervention_target_loose = bool(
        is_intervention
        and target in set(first_row.get("loose_matched_owner_ids", []))
    )
    released = _complete(first_row) and (
        not require_target_recovery_before_suffix
        or intervention_target_strict
        or intervention_target_loose
    )
    if released:
        for row_index in range(1, int(horizon_rows)):
            generated = generator(
                role=role,
                arm_id=arm_id,
                prefix_token_ids=current,
                forced_row_prefix_token_ids=None,
                mode="greedy",
                seed=None,
                temperature=0.0,
                top_p=GREEDY_TOP_P,
                repetition_penalty=repetition_penalty,
                row_index=row_index,
            )
            frozen = _freeze_row(
                generated, role=role, arm_id=arm_id, policy_id=policy_id, seed=seed, row_index=row_index, input_prefix=current
            )
            rows.append(frozen)
            next_prefix = _append(current, frozen)
            if next_prefix == current:
                break
            current = next_prefix
    # Keep three non-interchangeable layers: the forced intervention row,
    # its autonomous released suffix, and the final intervention-conditioned
    # trajectory.  Row zero is excluded only from suffix diagnostics; it is
    # included in the final conditioned owner set and arm comparison.
    autonomous_rows = rows if not is_intervention else rows[1:]
    autonomous_strict = _owner_set(autonomous_rows, "strict_matched_owner_ids")
    autonomous_loose = _owner_set(autonomous_rows, "loose_matched_owner_ids")
    conditioned_strict = _owner_set(rows, "strict_matched_owner_ids")
    conditioned_loose = _owner_set(rows, "loose_matched_owner_ids")
    prefix_owners = set(str(value) for value in role.get("prefix_owner_ids", []))
    autonomous_occurrences = Counter(
        str(owner)
        for row in autonomous_rows
        for owner in row.get("strict_matched_owner_ids", [])
    )
    conditioned_occurrences = Counter(
        str(owner)
        for row in rows
        for owner in row.get("strict_matched_owner_ids", [])
    )
    intervention = None
    if forced is not None:
        intervention = {
            "row": first_row,
            "target_realized_strict": intervention_target_strict,
            "target_realized_loose": intervention_target_loose,
            "owner_ids_strict": sorted(
                str(value) for value in first_row.get("strict_matched_owner_ids", [])
            ),
            "owner_ids_loose": sorted(
                str(value) for value in first_row.get("loose_matched_owner_ids", [])
            ),
            "semantic_relation_evidence": list(
                first_row.get("semantic_relation_evidence", [])
            ),
            "accounting_status": "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set",
        }
    return {
        "arm_id": arm_id,
        "arm_kind": "free_next_row" if forced is None and mode == "greedy" and seed is None else "forced_description",
        "producer": {
            "mode": mode,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "suffix_release_mode": "bounded_native_greedy",
            "horizon_rows": int(horizon_rows),
        },
        "initial_self_prefix_token_ids": list(map(int, prefix)),
        "initial_self_prefix_token_ids_sha256": sha256_json(list(map(int, prefix))),
        "forced_description_token_ids": None if forced is None else list(map(int, forced)),
        "forced_description_token_ids_sha256": None if forced is None else sha256_json(list(map(int, forced))),
        "rows": rows,
        "intervention": intervention,
        "released_suffix_rows": rows if forced is None else rows[1:],
        "autonomous_evidence_row_ids": [
            str(row["pred_row_id"]) for row in autonomous_rows
        ],
        "intervention_target_recovered_strict": (
            intervention_target_strict if is_intervention else None
        ),
        "intervention_target_recovered_loose": (
            intervention_target_loose if is_intervention else None
        ),
        "autonomous_suffix_target_recovered_strict": target in autonomous_strict,
        "autonomous_suffix_target_recovered_loose": target in autonomous_loose,
        "target_recovered_strict": (
            intervention_target_strict if is_intervention else target in conditioned_strict
        ),
        "target_recovered_loose": (
            intervention_target_loose if is_intervention else target in conditioned_loose
        ),
        "suffix_release_admitted": released,
        "autonomous_suffix_unique_strict_owner_ids": sorted(autonomous_strict),
        "autonomous_suffix_unique_loose_owner_ids": sorted(autonomous_loose),
        "final_intervention_conditioned_unique_strict_owner_ids": sorted(
            prefix_owners | conditioned_strict
        ),
        "final_intervention_conditioned_unique_loose_owner_ids": sorted(
            prefix_owners | conditioned_loose
        ),
        "final_unique_strict_owner_ids": sorted(prefix_owners | conditioned_strict),
        "final_unique_loose_owner_ids": sorted(prefix_owners | conditioned_loose),
        "duplicate_evidence": {
            "final_conditioned_prefix_owner_repeats": sorted(
                conditioned_strict & prefix_owners
            ),
            "final_conditioned_within_arm_repeats": sorted(
                owner for owner, count in conditioned_occurrences.items() if count > 1
            ),
            "autonomous_suffix_prefix_owner_repeats": sorted(
                autonomous_strict & prefix_owners
            ),
            "autonomous_suffix_within_arm_repeats": sorted(
                owner for owner, count in autonomous_occurrences.items() if count > 1
            ),
        },
        "semantic_drift_evidence": [
            evidence
            for row in autonomous_rows
            for evidence in row.get("semantic_drift_evidence", [])
        ],
        "semantic_relation_neutral_evidence": [
            evidence
            for row in autonomous_rows
            for evidence in row.get("semantic_relation_neutral_evidence", [])
        ],
        "invalid_row_ids": [
            str(row["pred_row_id"]) for row in autonomous_rows if not _complete(row)
        ],
        "natural_stop_row_ids": [
            str(row["pred_row_id"])
            for row in autonomous_rows
            if isinstance(row.get("row_stop"), Mapping) and row["row_stop"].get("stop_reason") == "terminal"
        ],
    }


def compare_arms(
    free: Mapping[str, Any], forced: Mapping[str, Any], *, target_owner_id: str, successor_owner_id: str | None
) -> dict[str, Any]:
    free_set = set(str(value) for value in free.get("final_unique_strict_owner_ids", []))
    forced_set = set(str(value) for value in forced.get("final_unique_strict_owner_ids", []))
    gained = forced_set - free_set
    retained = forced_set & free_set
    lost = free_set - forced_set
    successor_state = None
    if successor_owner_id:
        successor_state = {
            "owner_id": successor_owner_id,
            "free_present": successor_owner_id in free_set,
            "forced_present": successor_owner_id in forced_set,
            "retained": successor_owner_id in retained,
            "lost": successor_owner_id in lost,
            "gained": successor_owner_id in gained,
        }
    free_ids = free.get("rows", [{}])[0].get("raw_generated_token_ids", [])
    forced_ids = forced.get("rows", [{}])[0].get("raw_generated_token_ids", [])
    intervention_target_recovered = bool(
        forced.get("intervention_target_recovered_strict")
        or forced.get("intervention_target_recovered_loose")
    )
    return {
        "free_vs_forced_first_divergence": first_divergence(free_ids, forced_ids),
        "intervention_target_recovered_strict": bool(
            forced.get("intervention_target_recovered_strict")
        ),
        "intervention_target_recovered_loose": bool(
            forced.get("intervention_target_recovered_loose")
        ),
        "autonomous_suffix_target_recovered_strict": bool(
            forced.get("autonomous_suffix_target_recovered_strict")
        ),
        "autonomous_suffix_target_recovered_loose": bool(
            forced.get("autonomous_suffix_target_recovered_loose")
        ),
        "target_recovered_strict": bool(
            forced.get("intervention_target_recovered_strict")
        ),
        "target_recovered_loose": bool(
            forced.get("intervention_target_recovered_loose")
        ),
        "downstream_owner_accounting": {
            "accounting_layer": "final_intervention_conditioned_row0_plus_autonomous_suffix",
            "gained_owner_ids": sorted(gained),
            "retained_owner_ids": sorted(retained),
            "lost_owner_ids": sorted(lost),
            "target_recovery_exchange": bool(
                intervention_target_recovered and (lost - {target_owner_id})
            ),
            "exchange_lost_owner_ids": (
                sorted(lost - {target_owner_id})
                if intervention_target_recovered
                else []
            ),
        },
        "successor": successor_state,
    }


def execute_behavior_contract(contract: Mapping[str, Any], generator: Generator) -> dict[str, Any]:
    """Execute a validated contract through an injected row generator."""

    policy_views = []
    limits = _mapping(contract.get("generation_limits"), "generation_limits")
    greedy_horizon = int(limits["suffix_horizon_rows"])
    admission = contract.get("sampling_admission")
    for rp in contract.get("policy_repetition_penalties", [1.0]):
        rp = float(rp)
        policy_id = "rp_1_00" if rp == 1.0 else "rp_1_10"
        role_results = []
        for role in contract["roles"]:
            ledger = _mapping(role.get("ledger"), "role.ledger")
            prefix = _ids(ledger.get("self_prefix_token_ids"), "self prefix")
            forced = _ids(ledger.get("forced_description_token_ids"), "forced description", allow_empty=False)
            free = _run_arm(
                generator=generator,
                role=role,
                arm_id="free_next_row",
                policy_id=policy_id,
                repetition_penalty=rp,
                prefix=prefix,
                forced=None,
                mode="greedy",
                seed=None,
                temperature=0.0,
                top_p=1.0,
                horizon_rows=greedy_horizon,
                require_target_recovery_before_suffix=False,
            )
            forced_arm = _run_arm(
                generator=generator,
                role=role,
                arm_id="forced_description_greedy",
                policy_id=policy_id,
                repetition_penalty=rp,
                prefix=prefix,
                forced=forced,
                mode="greedy",
                seed=None,
                temperature=0.0,
                top_p=1.0,
                horizon_rows=greedy_horizon,
                require_target_recovery_before_suffix=True,
            )
            samples = []
            if admission is not None and rp == 1.0 and role["role_id"] in admission["admitted_role_ids"]:
                decode = admission["decode_parameters"]
                for seed in decode["seeds"]:
                    samples.append(
                        _run_arm(
                            generator=generator,
                            role=role,
                            arm_id=f"forced_description_sample_seed_{seed}",
                            policy_id=policy_id,
                            repetition_penalty=1.0,
                            prefix=prefix,
                            forced=forced,
                            mode="sample",
                            seed=int(seed),
                            temperature=float(decode["temperature"]),
                            top_p=float(decode["top_p"]),
                            horizon_rows=int(decode["horizon_rows"]),
                            require_target_recovery_before_suffix=True,
                        )
                    )
            role_results.append(
                {
                    "role_id": role["role_id"],
                    "role_kind": role["role_kind"],
                    "gt_owner_id": role["gt_owner_id"],
                    "diagnostic_owner_id": ledger["diagnostic_owner_id"],
                    "successor_gt_owner_id": role.get("successor_gt_owner_id"),
                    "exact_full_prefix_token_ids_sha256": sha256_json(ledger["exact_full_prefix_token_ids"]),
                    "exact_self_prefix_token_ids_sha256": sha256_json(prefix),
                    "arms": {
                        "free_next_row": free,
                        "forced_description_greedy": forced_arm,
                        "forced_description_low_temperature_samples": samples,
                    },
                    "comparison": compare_arms(
                        free,
                        forced_arm,
                        target_owner_id=str(role["gt_owner_id"]),
                        successor_owner_id=(
                            None if role.get("successor_gt_owner_id") is None else str(role["successor_gt_owner_id"])
                        ),
                    ),
                }
            )
        policy_views.append(
            {
                "policy_view_id": policy_id,
                "repetition_penalty": rp,
                "likelihood_semantics": "external raw likelihood remains unchanged and is not relabelled as this behavior arm",
                "roles": role_results,
            }
        )
    content = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "contract": {key: deepcopy(value) for key, value in contract.items() if key != "roles"},
        "selected_role_prefixes": [
            {
                "role_id": role["role_id"],
                "full_prefix_token_ids_sha256": sha256_json(role["ledger"]["exact_full_prefix_token_ids"]),
                "self_prefix_token_ids_sha256": sha256_json(role["ledger"]["self_prefix_token_ids"]),
            }
            for role in contract["roles"]
        ],
        "policy_views": policy_views,
    }
    return {**content, "output_content_sha256": sha256_json(content)}


def _write_create_or_identical(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    encoded = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
        status = "created"
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise BehaviorContractError("output already exists with different content; refusing to overwrite") from None
        status = "identical_existing"
    return {"path": str(path), "sha256": sha256_file(path), "status": status}


def _git_code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root, text=True
        ).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"], cwd=root
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "runner_sha256": sha256_file(Path(__file__))}
    dirty_digest = hashlib.sha256()
    dirty_digest.update(b"tracked-diff\0")
    dirty_digest.update(tracked_diff)
    untracked_paths = sorted(
        value.decode("utf-8", errors="surrogateescape")
        for value in untracked_raw.split(b"\0")
        if value
    )
    for relative in untracked_paths:
        path = root / relative
        if not path.is_file():
            continue
        dirty_digest.update(b"untracked-file\0")
        dirty_digest.update(relative.encode("utf-8", errors="surrogateescape"))
        dirty_digest.update(b"\0")
        dirty_digest.update(path.read_bytes())
    helpers = [
        root / "scripts/research/run_local_branch_causal_value.py",
        root / "scripts/research/run_greedy_prefix_forced_owner_path.py",
        root / "scripts/research/run_exact_prefix_owner_compositionality.py",
        root / "scripts/research/run_exact_prefix_sampled_rescue.py",
        root / "scripts/research/run_same_covered_set_prefix_order_probe.py",
        root / "src/inference/backend.py",
        root / "src/inference/hf_backend.py",
        root / "src/inference/image_plan.py",
        root / "src/inference/parsing.py",
        root / "src/inference/prompt.py",
        root / "src/inference/runtime.py",
    ]
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "dirty_diff_content_sha256": dirty_digest.hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked_file_count": len(untracked_paths),
        "runner_sha256": sha256_file(Path(__file__)),
        "helper_sha256": {str(path.relative_to(root)): sha256_file(path) for path in helpers},
    }


def _normalise_description(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())).strip()


def _owner_pixel_bbox_to_entity_norm1000(
    bbox_xyxy_pixel: Any, *, decoded_width: int, decoded_height: int, label: str
) -> list[float]:
    """Convert an owner-ledger pixel-space bbox into the entity norm1000 domain.

    The owner ledger stores ``bbox_xyxy`` in decoded-image pixel space with no
    embedded width/height, so the caller must supply the exact per-image
    decoded dimensions.  A pixel box outside those bounds is refused instead
    of silently normalized, since that is the signature of a pixel/norm1000
    domain mix-up rather than a valid box.
    """

    if (
        not isinstance(bbox_xyxy_pixel, Sequence)
        or isinstance(bbox_xyxy_pixel, (str, bytes))
        or len(bbox_xyxy_pixel) != 4
    ):
        raise BehaviorContractError(f"{label} bbox_xyxy must contain four values")
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy_pixel)
    except (TypeError, ValueError) as exc:
        raise BehaviorContractError(f"{label} bbox_xyxy values must be numeric") from exc
    if decoded_width <= 0 or decoded_height <= 0:
        raise BehaviorContractError(f"{label} decoded image dimensions must be positive")
    if not (0 <= x1 < x2 <= decoded_width and 0 <= y1 < y2 <= decoded_height):
        raise BehaviorContractError(
            f"{label} bbox_xyxy [{x1}, {y1}, {x2}, {y2}] is outside the decoded "
            f"image pixel domain ({decoded_width}x{decoded_height}); refusing to "
            "normalize a pixel/norm1000 domain mismatch"
        )
    return [
        x1 / decoded_width * 1000,
        y1 / decoded_height * 1000,
        x2 / decoded_width * 1000,
        y2 / decoded_height * 1000,
    ]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return 0.0 if union <= 0 else inter / union


def _annotate_runtime_row(
    row: dict[str, Any], *, entity_ledger: Sequence[Mapping[str, Any]], target: Mapping[str, Any], width: int, height: int, covered: Sequence[str], description_equivalence: Mapping[str, Any] | None
) -> dict[str, Any]:
    from scripts.research.run_local_branch_causal_value import _annotate_owner_matches
    from scripts.research.run_same_covered_set_prefix_order_probe import match_predictions_to_entities

    _annotate_owner_matches(
        row,
        entity_ledger=entity_ledger,
        image_width=width,
        image_height=height,
        covered_entity_ids=covered,
    )
    loose = match_predictions_to_entities(
        row.get("parsed_predictions", []),
        entity_ledger,
        image_width=width,
        image_height=height,
        iou_threshold=1e-12,
        ambiguity_margin=0.05,
        restrict_to_person=False,
    )
    row["loose_entity_matches"] = loose
    row["loose_matched_owner_ids"] = sorted(
        {str(item["matched_entity_id"]) for item in loose if item.get("status") == "matched"}
    )
    target_box = [float(value) for value in target["ground_truth"]["box"]]
    canonical = _normalise_description(target["canonical_description"]["text"])
    semantic_relations = []
    for index, prediction in enumerate(row.get("parsed_predictions", [])):
        box = prediction.get("bbox", prediction.get("bbox_xyxy"))
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        normalized = [
            float(box[0]) / width * 1000,
            float(box[1]) / height * 1000,
            float(box[2]) / width * 1000,
            float(box[3]) / height * 1000,
        ]
        overlap = _iou(normalized, target_box)
        relation = classify_semantic_relation(
            observed_description=prediction.get("description", ""),
            canonical_description=canonical,
            iou_to_target=overlap,
            equivalence=description_equivalence,
        )
        semantic_relations.append(
            {
                "prediction_index": index,
                "target_gt_owner_id": target["gt_owner_id"],
                "predicted_bbox_norm1000": normalized,
                **relation,
            }
        )
    row["semantic_relation_evidence"] = semantic_relations
    row["semantic_drift_evidence"] = [
        item for item in semantic_relations if item["semantic_drift_supported"]
    ]
    row["semantic_relation_neutral_evidence"] = [
        item
        for item in semantic_relations
        if item["relation"].startswith("unknown_neutral")
    ]
    return row


def _generate_forced_partial_sample(
    *, session: Any, native_inputs: Mapping[str, Any], parent_prefix: Sequence[int], forced: Sequence[int], tokenizer: Any, width: int, height: int, repetition_penalty: float, temperature: float, top_p: float, seed: int, max_new_tokens: int, malformed_limit: int, row_index: int
) -> dict[str, Any]:
    """Sampling twin of the production greedy forced-partial-row helper."""

    import torch
    from transformers import StoppingCriteriaList
    from scripts.research.run_local_branch_causal_value import _append_exact_prefix, hash_prefix_token_ids
    from scripts.research.run_same_covered_set_prefix_order_probe import (
        _RowStoppingCriteria,
        extract_row_stop,
        validate_generated_row_boundary,
    )
    from src.inference.parsing import parse_compact_object_box_closed

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    parent = [int(value) for value in parent_prefix]
    forced_ids = [int(value) for value in forced]
    native_width = int(native_inputs["input_ids"].shape[1])
    model_inputs, model_width = _append_exact_prefix(native_inputs, parent + forced_ids)
    row_start = native_width + len(parent)
    kwargs = {
        **model_inputs,
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "eos_token_id": session._im_end_token_id(),
        "pad_token_id": session._pad_token_id(),
        "return_dict_in_generate": True,
        "output_scores": False,
        "stopping_criteria": StoppingCriteriaList(
            [_RowStoppingCriteria(prompt_width=row_start, tokenizer=tokenizer, malformed_limit=malformed_limit)]
        ),
    }
    with torch.inference_mode():
        output = session._model.generate(**kwargs)
    sequences = output.sequences
    if int(sequences.shape[0]) != 1:
        raise RuntimeError("forced sampled partial-row generation did not return one sequence")
    full = [int(value) for value in sequences[0, row_start:].tolist()]
    released = [int(value) for value in sequences[0, model_width:].tolist()]
    raw_text = tokenizer.decode(full, skip_special_tokens=False)
    stop = validate_generated_row_boundary(raw_text, extract_row_stop(raw_text, malformed_limit=malformed_limit))
    if stop.get("stop_reason") == "contaminated_complete_row":
        status, evidence, predictions = "failed", {"parse_status": "not_run", "predictions": [], "dropped_predictions": []}, []
    else:
        parsed = parse_compact_object_box_closed(
            stop.get("row_text") or raw_text,
            row_id=f"sorted-fn-successor:forced-sample:{seed}:{row_index}",
            row_index=row_index,
            image_width=width,
            image_height=height,
        )
        status, evidence, predictions = "success", parsed.to_artifact_dict(), parsed.predictions
    return {
        "mode": "forced_prefix_then_sample",
        "seed": int(seed),
        "status": status,
        "parent_prefix_token_ids": parent,
        "parent_prefix_token_ids_sha256": hash_prefix_token_ids(parent),
        "forced_row_prefix_token_ids": forced_ids,
        "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(forced_ids),
        "released_tail_token_ids": released,
        "raw_generated_token_ids": full,
        "raw_generated_token_ids_sha256": hash_prefix_token_ids(full),
        "raw_generated_text": raw_text,
        "row_stop": stop,
        "parse_evidence": evidence,
        "parsed_predictions": predictions,
    }


def run_live(contract: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    """Load one explicit CUDA HF runtime and execute the validated contract."""

    if re.fullmatch(r"cuda:\d+", device) is None:
        raise BehaviorContractError("live execution requires an explicit CUDA device such as cuda:0")
    import torch
    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend
    from scripts.research.run_local_branch_causal_value import (
        _generate_after_forced_partial_row,
        _generate_row,
        _single_native_inputs,
    )

    if not torch.cuda.is_available():
        raise BehaviorContractError("explicit CUDA live execution requested but CUDA is unavailable")
    torch.cuda.set_device(torch.device(device))
    resolved = load_infer_config(Path(contract["infer_config"]["path"]))
    config = resolved.config
    if config.backend.type != "hf":
        raise BehaviorContractError("successor behavior requires backend.type: hf")
    if Path(config.data.input_jsonl).expanduser().resolve(strict=True) != Path(contract["source_jsonl"]["path"]):
        raise BehaviorContractError("infer config source JSONL disagrees with frozen source")
    if str(config.model.dtype) != "fp32":
        raise BehaviorContractError("decision-bearing successor behavior requires FP32")
    frozen_runtime_source = _mapping(
        _mapping(
            _mapping(contract["frozen_runtime_identity"], "frozen runtime identity").get(
                "runtime"
            ),
            "frozen runtime block",
        ).get("identity_source"),
        "frozen runtime identity source",
    )
    generation_fingerprint = config_sha256_json(
        config.generation.model_dump(mode="json")
    )
    if resolved.fingerprint != _mapping(
        frozen_runtime_source.get("resolved_config_fingerprints"),
        "frozen resolved config fingerprints",
    ).get("infer_config"):
        raise BehaviorContractError(
            "infer config fingerprint disagrees with frozen runtime identity"
        )
    if generation_fingerprint != frozen_runtime_source.get(
        "generation_config_fingerprint"
    ):
        raise BehaviorContractError(
            "generation config fingerprint disagrees with frozen runtime identity"
        )
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=generation_fingerprint,
    )
    raw_rows = list(load_raw_examples(Path(contract["source_jsonl"]["path"])))
    raw_by_image = {}
    for row in raw_rows:
        source = row.metadata.get("source")
        if isinstance(source, Mapping) and source.get("image_id") is not None:
            raw_by_image[str(source["image_id"])] = row
    owner_rows = _read_jsonl(
        Path(contract["registry"]["verified_sources"]["owner_ledger"]["path"]), "owner ledger"
    )
    owners_by_image: dict[str, list[dict[str, Any]]] = {}
    for owner in owner_rows:
        owners_by_image.setdefault(str(owner["image_id"]), []).append(
            {
                "entity_id": str(owner["gt_owner_id"]),
                "description": str(owner["normalized_description"]),
                "bbox_xyxy_pixel": [float(value) for value in owner["bbox_xyxy"]],
                "verification": "verified",
                "positive_only": True,
                "source": "frozen_fn_owner_ledger",
            }
        )
    template = _template_config(config)
    runtime_cache: dict[str, dict[str, Any]] = {}
    observed_runtime: dict[str, Any] = {}
    live_identity_receipt: dict[str, Any] = {}
    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        tokenizer = opened._tokenizer
        if tokenizer is None:
            raise RuntimeError("HF session lacks tokenizer")
        observed_runtime.update(opened.receipt.to_artifact_dict())
        live_identity_receipt.update(
            validate_live_runtime_identity(
                observed_receipt=observed_runtime,
                frozen_identity=contract["frozen_runtime_identity"],
                resolved_config_fingerprint=resolved.fingerprint,
                generation_config_fingerprint=generation_fingerprint,
            )
        )
        for role in contract["roles"]:
            image_id = str(role["trajectory"]["image_id"])
            if image_id in runtime_cache:
                continue
            raw = raw_by_image.get(image_id)
            if raw is None:
                raise BehaviorContractError(f"source JSONL lacks image {image_id}")
            plan = plan_image_batch(
                [raw], components=frontend.qwen, processor_config=_processor_config(config), row_indices=[0]
            ).rows[0]
            prompt = build_prompt_record(
                raw,
                template,
                processor=frontend.qwen.processor,
                row_index=0,
                merged_visual_tokens=plan.merged_visual_tokens,
            )
            grid = tuple(int(value) for value in plan.expected_image_grid_thw)
            request = DecodeRequest(
                request_id=f"sorted-fn-successor:{image_id}",
                chat_text=prompt.chat_text,
                input_prompt_token_ids=tuple(prompt.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
                image_path=plan.image_path,
                declared_image_width=plan.declared_width,
                declared_image_height=plan.declared_height,
                decoded_image_width=plan.decoded_width,
                decoded_image_height=plan.decoded_height,
                image_sha256=plan.image_content_sha256,
                expected_image_grid_thw=(grid[0], grid[1], grid[2]),
                logical_transform_id=plan.logical_transform_id,
                generation_policy=GenerationPolicy(
                    max_new_tokens=1,
                    repetition_penalty=1.0,
                    temperature=0.0,
                    top_p=1.0,
                ),
            )
            native_inputs, executed, _, _ = opened._materialize_native_inputs((request,))
            one_native = _single_native_inputs(native_inputs)
            decoded_width = int(plan.decoded_width)
            decoded_height = int(plan.decoded_height)
            entity_ledger = [
                {
                    **{
                        key: value
                        for key, value in owner.items()
                        if key != "bbox_xyxy_pixel"
                    },
                    "bbox_norm1000": _owner_pixel_bbox_to_entity_norm1000(
                        owner["bbox_xyxy_pixel"],
                        decoded_width=decoded_width,
                        decoded_height=decoded_height,
                        label=f"owner {owner['entity_id']!r} image {image_id!r}",
                    ),
                }
                for owner in owners_by_image[image_id]
            ]
            runtime_cache[image_id] = {
                "native_inputs": one_native,
                "executed_prompt_token_ids": [int(value) for value in executed[0]],
                "width": decoded_width,
                "height": decoded_height,
                "entity_ledger": entity_ledger,
            }

        limits = contract["generation_limits"]

        def generate(**kwargs: Any) -> Mapping[str, Any]:
            role = kwargs["role"]
            image_id = str(role["trajectory"]["image_id"])
            state = runtime_cache[image_id]
            expected_prompt = role["ledger"]["prompt_token_ids"]
            if state["executed_prompt_token_ids"] != expected_prompt:
                raise BehaviorContractError(f"role {role['role_id']!r} executed prompt mismatch")
            forced = kwargs["forced_row_prefix_token_ids"]
            if forced is None:
                row = _generate_row(
                    session=opened,
                    native_inputs=state["native_inputs"],
                    prefix_token_ids=kwargs["prefix_token_ids"],
                    tokenizer=tokenizer,
                    image_width=state["width"],
                    image_height=state["height"],
                    mode=kwargs["mode"],
                    seed=kwargs["seed"],
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                    repetition_penalty=kwargs["repetition_penalty"],
                    max_new_tokens=int(limits["max_new_tokens"]),
                    malformed_limit=int(limits["malformed_limit"]),
                    row_index=kwargs["row_index"],
                )
            elif kwargs["mode"] == "greedy":
                row = _generate_after_forced_partial_row(
                    session=opened,
                    native_inputs=state["native_inputs"],
                    parent_prefix_token_ids=kwargs["prefix_token_ids"],
                    forced_row_prefix_token_ids=forced,
                    tokenizer=tokenizer,
                    image_width=state["width"],
                    image_height=state["height"],
                    repetition_penalty=kwargs["repetition_penalty"],
                    max_new_tokens=int(limits["max_new_tokens"]),
                    malformed_limit=int(limits["malformed_limit"]),
                    row_index=kwargs["row_index"],
                )
            else:
                row = _generate_forced_partial_sample(
                    session=opened,
                    native_inputs=state["native_inputs"],
                    parent_prefix=kwargs["prefix_token_ids"],
                    forced=forced,
                    tokenizer=tokenizer,
                    width=state["width"],
                    height=state["height"],
                    repetition_penalty=kwargs["repetition_penalty"],
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                    seed=kwargs["seed"],
                    max_new_tokens=int(limits["max_new_tokens"]),
                    malformed_limit=int(limits["malformed_limit"]),
                    row_index=kwargs["row_index"],
                )
            return _annotate_runtime_row(
                dict(row),
                entity_ledger=state["entity_ledger"],
                target=role["ledger"],
                width=state["width"],
                height=state["height"],
                covered=role.get("prefix_owner_ids", []),
                description_equivalence=contract.get("description_equivalence"),
            )

        document = execute_behavior_contract(contract, generate)
    content = {key: value for key, value in document.items() if key != "output_content_sha256"}
    content["runtime_observed"] = observed_runtime
    content["live_identity_admission"] = live_identity_receipt
    code_provenance = _git_code_provenance()
    content["identity_binding"] = {
        "code_provenance": code_provenance,
        "checkpoint_identity": contract["runtime_identity"],
        "tokenizer_identity": contract["runtime_identity"],
        "prompt_identity": [
            {"role_id": role["role_id"], "prompt_token_ids_sha256": sha256_json(role["ledger"]["prompt_token_ids"])}
            for role in contract["roles"]
        ],
        "matcher_identity": {
            "strict_iou_threshold": 0.5,
            "loose_iou_threshold": "positive_overlap",
            "ambiguity_margin": 0.05,
            "same_normalized_description_required": True,
            "implementation_sha256": code_provenance.get("helper_sha256", {}).get(
                "scripts/research/run_same_covered_set_prefix_order_probe.py"
            ),
        },
    }
    return {**content, "output_content_sha256": sha256_json(content)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--owner-context-ledger", type=Path, required=True)
    parser.add_argument("--landscape-receipt", type=Path, required=True)
    parser.add_argument("--landscape-admission", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--include-role-id", action="append", default=[])
    parser.add_argument("--repetition-penalty", action="append", type=float, default=[])
    parser.add_argument("--suffix-horizon-rows", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--sampling-admission", type=Path)
    parser.add_argument("--description-equivalence", type=Path)
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--sampling-top-p", type=float)
    parser.add_argument("--sampling-k", type=int)
    parser.add_argument("--sampling-seed", action="append", type=int, default=[])
    parser.add_argument("--sampling-horizon-rows", type=int)
    parser.add_argument("--validate-contract-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        contract = build_contract(
            registry_path=args.registry,
            owner_context_ledger_path=args.owner_context_ledger,
            landscape_receipt_path=args.landscape_receipt,
            landscape_admission_path=args.landscape_admission,
            runtime_identity_path=args.runtime_identity,
            infer_config_path=args.infer_config,
            source_jsonl_path=args.source_jsonl,
            include_role_ids=args.include_role_id,
            repetition_penalties=args.repetition_penalty,
            suffix_horizon_rows=args.suffix_horizon_rows,
            max_new_tokens=args.max_new_tokens,
            malformed_limit=args.malformed_limit,
            sampling_admission_path=args.sampling_admission,
            description_equivalence_path=args.description_equivalence,
            sampling_flags={
                "temperature": args.sampling_temperature,
                "top_p": args.sampling_top_p,
                "k": args.sampling_k,
                "seeds": args.sampling_seed,
                "horizon_rows": args.sampling_horizon_rows,
            },
        )
        if args.validate_contract_only:
            print(
                json.dumps(
                    {
                        "status": "contract_valid",
                        "schema_version": contract["schema_version"],
                        "selected_role_ids": contract["selected_role_ids"],
                        "sampling_admitted": contract["sampling_admission"] is not None,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None or args.device is None:
            raise BehaviorContractError("live execution requires --output and an explicit --device cuda:N")
        document = run_live(contract, device=args.device)
        receipt = _write_create_or_identical(args.output.expanduser().resolve(), document)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output": receipt,
                    "selected_role_count": len(contract["selected_role_ids"]),
                    "policy_views": contract["policy_repetition_penalties"],
                    "sampling_admitted": contract["sampling_admission"] is not None,
                },
                sort_keys=True,
            )
        )
        return 0
    except (BehaviorContractError, OSError) as exc:
        raise SystemExit(f"successor behavior contract failed before/at execution: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
