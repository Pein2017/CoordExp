#!/usr/bin/env python3
"""Execute the sealed S native-FN support-completion plan.

The CPU planner intentionally emits the complete 200-context denominator and
all candidate-row identities.  The older support runner cannot execute this
plan: it reconstructs its work from the prior frozen-candidate registry
and includes native-TP calibration contexts.  This driver has one authority:
the materialized plan.  It never reads that old registry.

Execution is scalar-only because the canonical exact-history scorer has no
admitted candidate-batch API.  A shard receipt is resumable and immutable; a
resume reads a prior receipt and writes a new attempt receipt, never mutating
the prior artifact.  GPU/model work is available only through ``live`` mode;
``contract`` mode validates the complete CPU plan and emits no model work.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import plan_natural_boundary_owner_support_completion as planner  # noqa: E402
from scripts.research import run_static_dynamic_owner_support_probe as support_probe  # noqa: E402


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_owner_support_completion_execution.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt.v1"
FAILURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.failure.v1"
CHECKPOINT = "S"
WRAPPER = "object_box_closed"
PARSER = "compact_object_box_closed_only"
NUM_SHARDS = 8
EXPECTED_CONTEXTS = 200
EXPECTED_SCALAR_FORWARDS = 77428
EXPECTED_NATIVE_TP = 172
EXPECTED_NATIVE_FN = 220
EXPECTED_MEASURED_FN = 20
EXPECTED_UNSCORED_TP = 160
EXPECTED_H0_SHA256 = "5a73390ce66a0c318964d684a0c75c8a3756d8cde4708a9015eb2628aafca3d9"
EXPECTED_SOURCE_PANEL_SHA256 = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
EXPECTED_DERIVED_PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
EXPECTED_DERIVED_RECEIPT_SHA256 = "cd1273f627f7bdfcb16e9ca6e4a50e9d51f0163081ea7458d6d3deabe9bb82c0"
EXPECTED_CONFIG_FINGERPRINT = "a5ba21ab3e495fa22770ae5d156372d7894fec61f4ac905553db52540b8ead12"
DEFAULT_CENSUS_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "cpu-census-v2/admission-census.json"
)
EXPECTED_CENSUS_REVISION = "cpu-census-v2"
EXPECTED_CENSUS_FILE_SHA256 = "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e"
EXPECTED_CENSUS_SELF_SHA256 = "11090ae4bc8c2f7f8a676f83e91194c2e0e7270a6d359c7dabf2cf771b38879d"
EXPECTED_CENSUS_OWNER_IDS_SHA256 = "a666e116db950491027915cd65f4c1a949199d27be7c0c42f72e8583b863f551"
EXPECTED_CENSUS_CANDIDATE_IDS_SHA256 = "286c296e5b60c4f1bd3190f7e61aba1432309a9cd824f123a6d59d288a2f3072"
PLAN_BATCH_SIZE = 16
POOLED_BATCH_ESTIMATE = 4840
SHARD_LOCAL_BATCH_ESTIMATE = 4843


class SupportCompletionExecutionError(ValueError):
    """Raised when a plan, shard, or capture identity is incompatible."""


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
        raise SupportCompletionExecutionError("value is not canonical JSON serializable") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_source(source: str | Path | Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SupportCompletionExecutionError(f"invalid JSON input: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw), "byte_count": len(raw)}
    if isinstance(source, Mapping):
        value = dict(source)
        return value, {"inline": True, "sha256": sha256_json(value), "byte_count": None}
    raise SupportCompletionExecutionError("plan must be a JSON path or mapping")


def _document_self_sha256(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("self_sha256", None)
    return sha256_json(body)


def validate_admission_census(
    source: str | Path | Mapping[str, Any] = DEFAULT_CENSUS_PATH,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the accepted 784-row census and its exact S completion set."""

    try:
        raw, source_info = _read_json_source(source)
    except OSError as exc:
        raise SupportCompletionExecutionError("admission census input is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise SupportCompletionExecutionError("admission census must be a JSON object")
    if source_info.get("sha256") != EXPECTED_CENSUS_FILE_SHA256:
        raise SupportCompletionExecutionError("admission census file hash is not the accepted sealed artifact")
    if isinstance(source_info.get("path"), str) and Path(source_info["path"]).parent.name != EXPECTED_CENSUS_REVISION:
        raise SupportCompletionExecutionError("admission census path is not the accepted cpu-census-v2 revision")
    if raw.get("schema_version") != "natural_boundary_owner_admission_census.v1" or raw.get("status") != "sealed" or raw.get("unit_id") != UNIT_ID:
        raise SupportCompletionExecutionError("admission census schema/status/unit identity drifted")
    if raw.get("self_sha256") != EXPECTED_CENSUS_SELF_SHA256 or raw.get("self_sha256") != _document_self_sha256(raw):
        raise SupportCompletionExecutionError("admission census self hash mismatch")
    frozen = raw.get("frozen_universe")
    if not isinstance(frozen, Mapping) or frozen.get("row_count") != 784 or frozen.get("physical_owner_count") != 392:
        raise SupportCompletionExecutionError("admission census is not the sealed 784-row/392-owner universe")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 784:
        raise SupportCompletionExecutionError("admission census rows are not the sealed 784-row set")
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("checkpoint") == CHECKPOINT
        and row.get("native_fn") is True
        and row.get("disposition") == "support_unassessed"
    ]
    owner_ids = [str(row["gt_owner_id"]) for row in selected]
    if len(selected) != EXPECTED_CONTEXTS or len(set(owner_ids)) != EXPECTED_CONTEXTS:
        raise SupportCompletionExecutionError("admission census S support-unassessed owner set is not exactly 200")
    sorted_owner_ids = sorted(owner_ids)
    owner_ids_sha256 = sha256_json(sorted_owner_ids)
    if owner_ids_sha256 != EXPECTED_CENSUS_OWNER_IDS_SHA256:
        raise SupportCompletionExecutionError("admission census S owner-ID hash is not the accepted 200-owner set")
    candidates = raw.get("support_completion_candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CONTEXTS:
        raise SupportCompletionExecutionError("admission census support_completion_candidates is not exactly 200 rows")
    candidate_owner_ids = [str(row.get("gt_owner_id")) for row in candidates if isinstance(row, Mapping)]
    if len(candidate_owner_ids) != EXPECTED_CONTEXTS or sorted(candidate_owner_ids) != sorted_owner_ids:
        raise SupportCompletionExecutionError("admission census candidate owner IDs disagree with S rows")
    if raw.get("support_completion_candidates_sha256") != EXPECTED_CENSUS_CANDIDATE_IDS_SHA256 or raw.get("support_completion_candidates_sha256") != sha256_json(candidates):
        raise SupportCompletionExecutionError("admission census support-candidate hash mismatch")
    summary = raw.get("summary")
    summary_s = summary.get(CHECKPOINT) if isinstance(summary, Mapping) else None
    if not isinstance(summary_s, Mapping) or summary_s.get("native_fn") != EXPECTED_NATIVE_FN or summary_s.get("native_tp") != EXPECTED_NATIVE_TP or summary_s.get("support_unassessed") != EXPECTED_CONTEXTS:
        raise SupportCompletionExecutionError("admission census S summary denominator drifted")
    if plan is not None:
        contexts = plan.get("contexts")
        scope = plan.get("scope")
        if not isinstance(contexts, list) or not isinstance(scope, Mapping):
            raise SupportCompletionExecutionError("plan lacks context/scope owner identity for census binding")
        plan_owner_ids = [str(row.get("gt_owner_id")) for row in contexts if isinstance(row, Mapping)]
        if len(plan_owner_ids) != EXPECTED_CONTEXTS or sorted(plan_owner_ids) != sorted_owner_ids:
            raise SupportCompletionExecutionError("admission census S owner IDs do not equal plan context owners")
        if scope.get("completion_owner_ids_sha256") != owner_ids_sha256:
            raise SupportCompletionExecutionError("plan completion owner-ID hash does not bind admission census")
    return {
        "revision": EXPECTED_CENSUS_REVISION,
        "path": source_info.get("path"),
        "file_sha256": source_info["sha256"],
        "self_sha256": raw["self_sha256"],
        "s_owner_ids_sha256": owner_ids_sha256,
        "s_owner_count": len(sorted_owner_ids),
        "support_completion_candidates_sha256": raw["support_completion_candidates_sha256"],
    }


def _strict_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SupportCompletionExecutionError(f"{label} must be a 64-character SHA-256")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise SupportCompletionExecutionError(f"{label} must be a lowercase SHA-256")
    return lowered


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SupportCompletionExecutionError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SupportCompletionExecutionError(f"{label} is not finite")
    return result


def _candidate_identity_hash(candidate: Mapping[str, Any]) -> str:
    """Recompute the planner's physical candidate identity from a row."""

    try:
        image = int(candidate["image_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupportCompletionExecutionError("candidate image_id is malformed") from exc
    candidate_id = str(candidate.get("candidate_id", ""))
    category = str(candidate.get("normalized_description", "")).lower()
    tokens = candidate.get("coord_token_ids")
    bins = candidate.get("coord_bins")
    if not candidate_id or not category or not isinstance(tokens, list) or not isinstance(bins, list):
        raise SupportCompletionExecutionError("candidate identity fields are incomplete")
    if len(bins) != 4 or not tokens or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tokens
    ):
        raise SupportCompletionExecutionError("candidate coordinate identity is malformed")
    core = {
        "candidate_id": candidate_id,
        "image_id": image,
        "normalized_description": category,
        "coord_bins": [int(item) for item in bins],
        "coord_token_ids": [int(item) for item in tokens],
        "decoded_bbox_pixel_xyxy": list(candidate.get("decoded_bbox_pixel_xyxy") or []),
        "identity_rule": "digest_image_category_coord_tokens",
    }
    core["coord_token_ids_sha256"] = sha256_json(core["coord_token_ids"])
    return sha256_json(core)


def _target_row_hash(candidate: Mapping[str, Any]) -> str:
    nested = candidate.get("target_row")
    if not isinstance(nested, Mapping):
        raise SupportCompletionExecutionError("candidate target-row identity is missing")
    core = {
        "candidate_id": nested.get("candidate_id"),
        "category_name": nested.get("category_name"),
        "coord_token_ids": nested.get("coord_token_ids"),
        "coord_token_ids_sha256": nested.get("coord_token_ids_sha256"),
        "row_encoding": nested.get("row_encoding"),
        "tokenizer_binding": nested.get("tokenizer_binding"),
    }
    return sha256_json(core)


def _context_identity_hash(context: Mapping[str, Any]) -> str:
    owner = context.get("target_owner_row")
    if not isinstance(owner, Mapping):
        raise SupportCompletionExecutionError("context target owner row is missing")
    core = {
        "stable_key": context.get("stable_key"),
        "owner_identity": owner,
        "exact_prefix_sha256": context.get("exact_prefix_sha256"),
        "target_rows_sha256": context.get("target_rows_sha256"),
    }
    return sha256_json(core)


def _validate_context(context: Mapping[str, Any], *, position: int) -> dict[str, Any]:
    required = (
        "context_id",
        "stable_key",
        "context_kind",
        "checkpoint",
        "image_id",
        "gt_owner_id",
        "category_name",
        "exact_prefix_token_ids",
        "exact_prefix_sha256",
        "candidate_ids",
        "target_rows",
        "candidate_identity_hashes",
        "target_row_identity_hashes",
    )
    if any(key not in context for key in required):
        raise SupportCompletionExecutionError(f"context {position} is missing an identity field")
    row = dict(context)
    stable = row["stable_key"]
    if not isinstance(stable, str) or not stable:
        raise SupportCompletionExecutionError(f"context {position} stable_key is missing")
    expected_stable = "|".join(
        (
            "candidate",
            CHECKPOINT,
            str(row.get("image_id")),
            str(row.get("gt_owner_id")),
            str(row.get("exact_prefix_sha256")),
        )
    )
    if row.get("stable_key_rule") != "candidate|checkpoint|image_id|gt_owner_id|exact_prefix_sha256" or stable != expected_stable:
        raise SupportCompletionExecutionError(f"context {position} stable shard identity drifted")
    expected_id = planner.context_id_for_stable_key(stable)
    if row.get("context_id") != expected_id:
        raise SupportCompletionExecutionError(f"context {position} context_id does not bind stable_key")
    if row.get("context_kind") != "support_completion_candidate":
        raise SupportCompletionExecutionError(f"context {position} is not a support-completion context")
    if row.get("checkpoint") != CHECKPOINT:
        raise SupportCompletionExecutionError(f"context {position} is not S checkpoint scope")
    if row.get("native_fn") is not True or row.get("native_tp") is not False:
        raise SupportCompletionExecutionError(f"context {position} is not an unassessed native-FN context")
    if row.get("strict_complete_row") is not False or row.get("natural_boundary_valid") is not True:
        raise SupportCompletionExecutionError(f"context {position} has invalid native boundary scope")
    if row.get("status") != "planned":
        raise SupportCompletionExecutionError(f"context {position} is not a sealed planned context")
    if row.get("batching_admitted") is not False or row.get("no_future_or_intervention_leakage") is not True:
        raise SupportCompletionExecutionError(f"context {position} execution contract drifted")
    if row.get("measured_wall_time_seconds") is not None or row.get("realized_scalar_forward_count") is not None:
        raise SupportCompletionExecutionError(f"context {position} contains realized execution fields")
    try:
        image_id = int(row["image_id"])
    except (TypeError, ValueError) as exc:
        raise SupportCompletionExecutionError(f"context {position} image_id is malformed") from exc
    row["image_id"] = image_id
    prefix = row["exact_prefix_token_ids"]
    if not isinstance(prefix, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prefix
    ):
        raise SupportCompletionExecutionError(f"context {position} exact prefix token IDs are malformed")
    _strict_hash(row["exact_prefix_sha256"], f"context {position} exact_prefix_sha256")
    if row["exact_prefix_sha256"] != sha256_json(prefix) or row.get("exact_prefix_identity_sha256") != sha256_json(prefix):
        raise SupportCompletionExecutionError(f"context {position} exact prefix hash mismatch")
    ids = row["candidate_ids"]
    target_rows = row["target_rows"]
    candidate_hashes = row["candidate_identity_hashes"]
    target_hashes = row["target_row_identity_hashes"]
    if not isinstance(ids, list) or not ids or len(ids) != len(set(map(str, ids))):
        raise SupportCompletionExecutionError(f"context {position} candidate IDs are missing or duplicated")
    if not isinstance(target_rows, list) or len(target_rows) != len(ids):
        raise SupportCompletionExecutionError(f"context {position} target-row denominator mismatch")
    if not isinstance(candidate_hashes, list) or len(candidate_hashes) != len(ids):
        raise SupportCompletionExecutionError(f"context {position} candidate identity hash count mismatch")
    if not isinstance(target_hashes, list) or len(target_hashes) != len(ids):
        raise SupportCompletionExecutionError(f"context {position} target-row identity hash count mismatch")
    if ids != [item.get("candidate_id") if isinstance(item, Mapping) else None for item in target_rows]:
        raise SupportCompletionExecutionError(f"context {position} candidate/target-row identity mismatch")
    if row.get("candidate_ids_sha256") != sha256_json(ids):
        raise SupportCompletionExecutionError(f"context {position} candidate IDs hash mismatch")
    if row.get("candidate_identity_hashes_sha256") != sha256_json(candidate_hashes):
        raise SupportCompletionExecutionError(f"context {position} candidate identity hash-list mismatch")
    if row.get("target_rows_sha256") != sha256_json(target_rows):
        raise SupportCompletionExecutionError(f"context {position} target rows hash mismatch")
    if row.get("target_row_identity_hashes_sha256") != sha256_json(target_hashes):
        raise SupportCompletionExecutionError(f"context {position} target-row hash-list mismatch")
    if row.get("candidate_count") != len(ids) or row.get("scalar_equivalent_forward_count") != len(ids):
        raise SupportCompletionExecutionError(f"context {position} scalar denominator mismatch")
    for index, candidate in enumerate(target_rows):
        if not isinstance(candidate, Mapping):
            raise SupportCompletionExecutionError(f"context {position} candidate {index} is not an object")
        if int(candidate.get("image_id", -1)) != image_id:
            raise SupportCompletionExecutionError(f"context {position} candidate {index} image identity drifted")
        if str(candidate.get("normalized_description", "")).lower() != str(row["category_name"]).lower():
            raise SupportCompletionExecutionError(f"context {position} candidate {index} category identity drifted")
        actual_candidate_hash = _candidate_identity_hash(candidate)
        actual_target_hash = _target_row_hash(candidate)
        if actual_candidate_hash != candidate.get("candidate_identity_sha256") or actual_candidate_hash != candidate_hashes[index]:
            raise SupportCompletionExecutionError(f"context {position} candidate {index} identity hash drifted")
        if actual_target_hash != target_hashes[index]:
            raise SupportCompletionExecutionError(f"context {position} target-row {index} identity hash drifted")
        nested = candidate.get("target_row")
        if not isinstance(nested, Mapping) or nested.get("candidate_id") != candidate.get("candidate_id"):
            raise SupportCompletionExecutionError(f"context {position} target-row {index} candidate binding drifted")
    if row.get("context_identity_sha256") != _context_identity_hash(row):
        raise SupportCompletionExecutionError(f"context {position} context identity hash drifted")
    row["context_plan_position"] = int(row.get("context_plan_position", position))
    return row


def validate_execution_plan(
    source: str | Path | Mapping[str, Any],
    *,
    expected_plan_sha256: str | None = None,
    census_path: str | Path | Mapping[str, Any] = DEFAULT_CENSUS_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fail-closed validate the sealed 200-context execution plan."""

    raw, source_info = _read_json_source(source)
    if not isinstance(raw, Mapping):
        raise SupportCompletionExecutionError("support-completion plan must be a JSON object")
    try:
        plan = planner.validate_plan(raw, strict_contract=True)
    except planner.SupportCompletionPlanError as exc:
        raise SupportCompletionExecutionError(str(exc)) from exc
    # A materialized plan is paired with the planner receipt.  When that
    # sibling exists, bind the execution to its byte-level plan hash as well
    # as the in-document content hash; a copied or silently rewritten plan is
    # then rejected before any model/runtime setup.
    source_path = source_info.get("path")
    if isinstance(source_path, str):
        materialized_receipt_path = Path(source_path).with_name("receipt.json")
        if materialized_receipt_path.is_file():
            try:
                materialized_receipt = json.loads(materialized_receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SupportCompletionExecutionError("materialized plan receipt is unreadable") from exc
            if not isinstance(materialized_receipt, Mapping):
                raise SupportCompletionExecutionError("materialized plan receipt is not an object")
            if materialized_receipt.get("plan_sha256") != source_info["sha256"] or materialized_receipt.get("plan_content_sha256") != plan.get("plan_content_sha256"):
                raise SupportCompletionExecutionError("materialized plan receipt does not bind this plan")
            source_info = {**source_info, "receipt_path": str(materialized_receipt_path.resolve()), "receipt_sha256": sha256_file(materialized_receipt_path)}
    if expected_plan_sha256 is not None:
        expected_hash = _strict_hash(expected_plan_sha256, "expected plan SHA-256")
        if source_info["sha256"] != expected_hash and plan.get("plan_content_sha256") != expected_hash:
            raise SupportCompletionExecutionError("materialized plan/file content hash does not match expected plan SHA-256")
    census_binding = validate_admission_census(census_path, plan=plan)
    source_info = {**source_info, "census": census_binding}
    if plan.get("checkpoint") != CHECKPOINT or plan.get("wrapper") != WRAPPER or plan.get("parser") != PARSER:
        raise SupportCompletionExecutionError("plan is not the S step-2444 native closed-wrapper substrate")
    if plan.get("unit_id") != UNIT_ID or plan.get("status") != "sealed_cpu_plan":
        raise SupportCompletionExecutionError("plan unit/status identity is not sealed")
    execution = plan.get("execution_contract")
    if not isinstance(execution, Mapping) or execution.get("batching_admitted") is not False:
        raise SupportCompletionExecutionError("plan execution contract admits unsupported batching")
    if execution.get("model_loaded") is not False or execution.get("gpu_used") is not False or execution.get("training") is not False:
        raise SupportCompletionExecutionError("plan is not the CPU-only sealed planner output")
    panel = plan.get("panel")
    h0 = plan.get("h0_lineage")
    scope = plan.get("scope")
    reuse = plan.get("calibration_reuse")
    if not isinstance(panel, Mapping) or panel.get("source_sha256") != EXPECTED_SOURCE_PANEL_SHA256 or panel.get("derived_sha256") != EXPECTED_DERIVED_PANEL_SHA256 or panel.get("derived_receipt_sha256") != EXPECTED_DERIVED_RECEIPT_SHA256:
        raise SupportCompletionExecutionError("plan derived panel identity is not the frozen 13-image panel")
    if panel.get("image_count") != 13 or panel.get("owner_count") != 392:
        raise SupportCompletionExecutionError("plan panel denominator is not the frozen 13-image/392-owner panel")
    if not isinstance(h0, Mapping):
        raise SupportCompletionExecutionError("plan H0 lineage is missing")
    if h0.get("source_sha256") != EXPECTED_H0_SHA256 or h0.get("source", {}).get("sha256") != EXPECTED_H0_SHA256:
        raise SupportCompletionExecutionError("plan H0 lineage is not S step-2444 native H0")
    h0_path = str(h0.get("source", {}).get("path", ""))
    if Path(h0_path).name != "s-step2444-native-h0.json":
        raise SupportCompletionExecutionError("plan H0 source path is not s-step2444-native-h0.json")
    if h0.get("checkpoint") != CHECKPOINT or h0.get("config_fingerprint") != EXPECTED_CONFIG_FINGERPRINT:
        raise SupportCompletionExecutionError("plan H0 checkpoint/config identity drifted")
    if h0.get("record_count") != EXPECTED_NATIVE_TP + EXPECTED_NATIVE_FN or h0.get("native_tp_count") != EXPECTED_NATIVE_TP or h0.get("native_fn_count") != EXPECTED_NATIVE_FN:
        raise SupportCompletionExecutionError("plan H0 denominator is not 172 TP + 220 FN")
    if not isinstance(scope, Mapping) or (
        scope.get("native_fn_denominator"),
        scope.get("support_measured_fn_retained"),
        scope.get("support_unassessed_fn"),
        scope.get("support_completion_candidates"),
        scope.get("native_tp_calibration_complete"),
        scope.get("native_tp_other_not_scored"),
    ) != (EXPECTED_NATIVE_FN, EXPECTED_MEASURED_FN, 200, EXPECTED_CONTEXTS, EXPECTED_NATIVE_TP, EXPECTED_UNSCORED_TP):
        raise SupportCompletionExecutionError("plan support denominator/calibration scope drifted")
    if not isinstance(reuse, Mapping) or reuse.get("reused") is not True or reuse.get("recomputed") is not False or reuse.get("observation_count") != EXPECTED_NATIVE_TP or reuse.get("excluded") != 0:
        raise SupportCompletionExecutionError("plan calibration reuse would score or recompute TP controls")
    if reuse.get("calibration_sha256") != plan.get("support_lineage", {}).get("calibration_sha256"):
        raise SupportCompletionExecutionError("plan calibration identity drifted")
    contexts_raw = plan.get("contexts")
    if not isinstance(contexts_raw, list) or len(contexts_raw) != EXPECTED_CONTEXTS:
        raise SupportCompletionExecutionError("plan must contain exactly 200 support-completion contexts")
    contexts = [_validate_context(row, position=index) for index, row in enumerate(contexts_raw)]
    context_ids = [str(row["context_id"]) for row in contexts]
    owner_ids = [str(row["gt_owner_id"]) for row in contexts]
    if len(set(context_ids)) != EXPECTED_CONTEXTS or len(set(owner_ids)) != EXPECTED_CONTEXTS:
        raise SupportCompletionExecutionError("plan contexts do not cover 200 unique owners")
    if plan.get("context_ids_sha256") != sha256_json(context_ids) or plan.get("context_owner_ids_sha256") != sha256_json(owner_ids):
        raise SupportCompletionExecutionError("plan context denominator hashes drifted")
    work = plan.get("work")
    if not isinstance(work, Mapping) or work.get("shard_count") != NUM_SHARDS or work.get("batching_admitted") is not False:
        raise SupportCompletionExecutionError("plan work accounting is not the eight-shard scalar contract")
    total = 0
    for shard in range(NUM_SHARDS):
        assigned = [row for row in contexts if int(row["shard_index"]) == shard]
        declared = next((item for item in work.get("per_shard", ()) if item.get("shard_index") == shard), None)
        if not isinstance(declared, Mapping):
            raise SupportCompletionExecutionError(f"plan is missing shard {shard} accounting")
        scalar = sum(int(row["scalar_equivalent_forward_count"]) for row in assigned)
        total += scalar
        if declared.get("context_count") != len(assigned) or declared.get("scalar_equivalent_forward_count") != scalar:
            raise SupportCompletionExecutionError(f"plan shard {shard} assignment/denominator mismatch")
    if total != EXPECTED_SCALAR_FORWARDS or work.get("scalar_equivalent_forward_count") != EXPECTED_SCALAR_FORWARDS:
        raise SupportCompletionExecutionError("plan scalar denominator is not 77,428")
    estimates = batch_estimates(plan)
    if estimates["candidate_batch_size"] != PLAN_BATCH_SIZE or estimates["pooled_ceiling"] != POOLED_BATCH_ESTIMATE or estimates["sum_shard_local_ceilings"] != SHARD_LOCAL_BATCH_ESTIMATE:
        raise SupportCompletionExecutionError("plan pooled/shard-local batch estimates drifted")
    candidate_index: dict[str, str] = {}
    for context in contexts:
        for candidate, identity_hash in zip(context["candidate_ids"], context["candidate_identity_hashes"], strict=True):
            old = candidate_index.get(str(candidate))
            if old is not None and old != str(identity_hash):
                raise SupportCompletionExecutionError(f"candidate identity index conflict: {candidate}")
            candidate_index[str(candidate)] = str(identity_hash)
    if plan.get("candidate_identity_index_count") != len(candidate_index) or plan.get("candidate_identity_index_sha256") != sha256_json(candidate_index):
        raise SupportCompletionExecutionError("plan candidate identity index drifted")
    normalized = dict(plan)
    normalized["contexts"] = contexts
    return normalized, source_info


def shard_contexts(
    plan: Mapping[str, Any],
    *,
    shard_index: int,
    num_shards: int = NUM_SHARDS,
) -> list[dict[str, Any]]:
    """Return exactly the content-stable contexts assigned to one shard."""

    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < num_shards:
        raise SupportCompletionExecutionError("shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards != NUM_SHARDS:
        raise SupportCompletionExecutionError("support completion requires exactly eight shards")
    contexts = [dict(row) for row in plan["contexts"] if int(row["shard_index"]) == shard_index]
    contexts.sort(key=lambda row: int(row.get("context_plan_position", 0)))
    declared = next(row for row in plan["work"]["per_shard"] if int(row["shard_index"]) == shard_index)
    if len(contexts) != int(declared["context_count"]):
        raise SupportCompletionExecutionError(f"shard {shard_index} context assignment is incomplete")
    if sum(int(row["scalar_equivalent_forward_count"]) for row in contexts) != int(declared["scalar_equivalent_forward_count"]):
        raise SupportCompletionExecutionError(f"shard {shard_index} scalar assignment is incomplete")
    return contexts


def batch_estimates(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Report pooled versus shard-local ceilings without implying batching."""

    work = plan.get("work")
    if not isinstance(work, Mapping):
        raise SupportCompletionExecutionError("plan work accounting is missing")
    batch_size = work.get("candidate_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise SupportCompletionExecutionError("plan candidate batch size is invalid")
    scalar_total = int(work.get("scalar_equivalent_forward_count", -1))
    per_shard = work.get("per_shard")
    if not isinstance(per_shard, list):
        raise SupportCompletionExecutionError("plan per-shard work accounting is missing")
    shard_local = sum(
        math.ceil(int(row.get("scalar_equivalent_forward_count", -1)) / batch_size)
        for row in per_shard
        if isinstance(row, Mapping)
    )
    pooled = math.ceil(scalar_total / batch_size)
    return {
        "candidate_batch_size": batch_size,
        "pooled_ceiling": pooled,
        "sum_shard_local_ceilings": shard_local,
        "batching_admitted": False,
        "status": "estimate_only_exact_history_api_scalar_only",
    }


def validate_device_mapping(
    *,
    shard_index: int,
    num_shards: int,
    device: str | None = None,
    shard_device_map: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Validate a future explicit logical-shard/device assignment."""

    if num_shards != NUM_SHARDS or isinstance(shard_index, bool) or not 0 <= shard_index < num_shards:
        raise SupportCompletionExecutionError("device mapping must target one of the eight sealed shards")
    selected = device
    normalized_map: dict[str, str] = {}
    if shard_device_map is not None:
        if not isinstance(shard_device_map, Mapping):
            raise SupportCompletionExecutionError("shard_device_map must be an object")
        for key, value in shard_device_map.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise SupportCompletionExecutionError("shard_device_map has a non-integer shard key") from exc
            if not 0 <= index < num_shards:
                raise SupportCompletionExecutionError("shard_device_map shard index is outside the sealed plan")
            if not isinstance(value, str):
                raise SupportCompletionExecutionError("shard_device_map device must be a CUDA string")
            normalized_map[str(index)] = _normalize_cuda_device(value)
        mapped = normalized_map.get(str(shard_index))
        if mapped is not None:
            if selected is not None and _normalize_cuda_device(selected) != mapped:
                raise SupportCompletionExecutionError("explicit device disagrees with shard_device_map")
            selected = mapped
    normalized = _normalize_cuda_device(selected) if selected is not None else None
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "logical_device": normalized,
        "shard_device_map": normalized_map,
        "mapping_status": "validated" if normalized is not None else "unbound",
    }


def _normalize_cuda_device(value: str) -> str:
    if not isinstance(value, str):
        raise SupportCompletionExecutionError("device must be a CUDA device string")
    text = value.strip().lower()
    if text == "cuda":
        return "cuda:0"
    match = re.fullmatch(r"cuda:([0-9]+)", text)
    if match is None:
        raise SupportCompletionExecutionError(f"unsupported CUDA device: {value!r}")
    return f"cuda:{int(match.group(1))}"


def attest_runtime_device(
    runtime_identity: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the opened runtime is on the explicitly assigned device."""

    expected = mapping.get("logical_device")
    if not isinstance(expected, str):
        raise SupportCompletionExecutionError("live execution requires an explicit logical CUDA device assignment")
    expected = _normalize_cuda_device(expected)
    if not isinstance(runtime_identity, Mapping):
        raise SupportCompletionExecutionError("opened scorer has no runtime identity for device attestation")
    observed_values: dict[str, str] = {}
    for key in ("device", "effective_device", "normalized_device", "torch_current_device"):
        value = runtime_identity.get(key)
        if value is None:
            continue
        observed_values[key] = _normalize_cuda_device(str(value))
    if not observed_values:
        raise SupportCompletionExecutionError("opened scorer runtime identity has no device")
    if any(value != expected for value in observed_values.values()):
        raise SupportCompletionExecutionError(
            f"opened scorer device does not match assigned logical device: expected {expected}, observed {observed_values}"
        )
    return {
        "status": "validated",
        "assigned_logical_device": expected,
        "observed_devices": observed_values,
        "physical_device_id": runtime_identity.get("physical_device_id"),
    }


def attest_scorer_device(
    scorer: Any,
    mapping: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest the native HF runtime after opening, with a CPU-test seam."""

    session = getattr(scorer, "_session", None)
    if session is not None:
        runtime_identity = support_probe.validate_live_runtime_identity(
            session,
            expected_checkpoint=CHECKPOINT,
            expected_config_fingerprint=plan["h0_lineage"]["config_fingerprint"],
        )
    else:
        runtime_identity = getattr(scorer, "runtime_identity", None)
    return attest_runtime_device(runtime_identity, mapping)


def validate_batching_request(
    *,
    candidate_batch_size: int = 1,
    batching_admitted: bool = False,
    batching_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject candidate batching unless a separately admitted API exists."""

    if isinstance(candidate_batch_size, bool) or not isinstance(candidate_batch_size, int) or candidate_batch_size <= 0:
        raise SupportCompletionExecutionError("candidate_batch_size must be a positive integer")
    if batching_admitted:
        if not isinstance(batching_admission, Mapping) or batching_admission.get("status") != "admitted" or batching_admission.get("parity_status") != "passed":
            raise SupportCompletionExecutionError("candidate batching requires a separate parity-admitted receipt")
        raise SupportCompletionExecutionError("batching is not implemented by the exact-history scalar driver")
    if candidate_batch_size != 1:
        raise SupportCompletionExecutionError("candidate batching is not admitted; use scalar candidate_batch_size=1")
    if batching_admission is not None:
        raise SupportCompletionExecutionError("batching admission cannot be supplied when batching_admitted=false")
    return {
        "candidate_batch_size": 1,
        "batching_admitted": False,
        "status": "not_admitted_exact_history_api_scalar_only",
    }


@dataclass(frozen=True)
class CandidateBank:
    """Physical bank plus exact native H0 records used by the scorer."""

    panel: Any
    h0: Any
    physical: tuple[Mapping[str, Any], ...]
    groups: Mapping[tuple[int, str], tuple[Mapping[str, Any], ...]]
    by_id: Mapping[str, Mapping[str, Any]]
    h0_by_owner: Mapping[str, Mapping[str, Any]]
    plan_content_sha256: str | None = None


def build_candidate_bank(
    plan: Mapping[str, Any],
    *,
    source_panel: str | Path | Mapping[str, Any] | None = None,
    derived_panel: str | Path | Mapping[str, Any] | None = None,
    derived_receipt: str | Path | Mapping[str, Any] | None = None,
    h0_ledger: str | Path | Mapping[str, Any] | None = None,
) -> CandidateBank:
    """Rebuild only the physical bank needed to recover support semantics.

    This function never consults the historical frozen candidate registry.  It
    checks that every plan target row is present in the authoritative physical
    bank and that no extra candidate is silently substituted.
    """

    panel_meta = plan["panel"]
    h0_meta = plan["h0_lineage"]
    source = source_panel or panel_meta["source"]["path"]
    derived = derived_panel or panel_meta["derived"]["path"]
    receipt = derived_receipt or panel_meta["derived_receipt"]["path"]
    h0_source = h0_ledger or h0_meta["source"]["path"]
    panel = support_probe.load_panel_inputs(source, derived, receipt)
    if panel.source_info.get("sha256") != panel_meta.get("source_sha256") or panel.derived_info.get("sha256") != panel_meta.get("derived_sha256") or panel.receipt_info.get("sha256") != panel_meta.get("derived_receipt_sha256"):
        raise SupportCompletionExecutionError("runtime panel input hash does not match sealed plan")
    h0 = support_probe.load_h0_inputs(h0_source, panel=panel, checkpoint=CHECKPOINT)
    if h0.source_info.get("sha256") != h0_meta.get("source_sha256"):
        raise SupportCompletionExecutionError("runtime H0 ledger hash does not match sealed plan")
    if sha256_json(list(h0.records)) != h0_meta.get("records_sha256"):
        raise SupportCompletionExecutionError("runtime H0 records do not match sealed plan")
    physical, _accounting = support_probe.build_physical_owner_bank(panel)
    groups_mut: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in physical:
        cid = str(candidate.get("candidate_id"))
        if cid in by_id and _candidate_identity_hash(by_id[cid]) != _candidate_identity_hash(candidate):
            raise SupportCompletionExecutionError(f"physical bank candidate identity collision: {cid}")
        by_id[cid] = candidate
        key = (int(candidate["image_id"]), str(candidate["normalized_description"]).lower())
        groups_mut.setdefault(key, []).append(candidate)
    groups = {key: tuple(value) for key, value in groups_mut.items()}
    h0_by_owner = {str(row["gt_owner_id"]): row for row in h0.records}
    if len(h0_by_owner) != EXPECTED_NATIVE_TP + EXPECTED_NATIVE_FN:
        raise SupportCompletionExecutionError("runtime H0 owner denominator drifted")
    for context in plan["contexts"]:
        key = (int(context["image_id"]), str(context["category_name"]).lower())
        actual_group = groups.get(key, ())
        actual_ids = {str(item["candidate_id"]) for item in actual_group}
        planned_ids = {str(item) for item in context["candidate_ids"]}
        if actual_ids != planned_ids:
            raise SupportCompletionExecutionError(f"physical candidate bank mismatch for {context['context_id']}")
        for candidate in context["target_rows"]:
            actual = by_id.get(str(candidate["candidate_id"]))
            if actual is None or _candidate_identity_hash(actual) != candidate.get("candidate_identity_sha256"):
                raise SupportCompletionExecutionError(f"physical candidate identity mismatch for {candidate.get('candidate_id')}")
        raw = h0_by_owner.get(str(context["gt_owner_id"]))
        if raw is None or int(raw["image_id"]) != int(context["image_id"]):
            raise SupportCompletionExecutionError(f"H0 owner is absent for {context['context_id']}")
        if raw.get("native_fn") is not True or raw.get("native_tp") is not False or raw.get("strict_complete_row") is not False:
            raise SupportCompletionExecutionError(f"refusing non-FN/complete H0 owner for {context['context_id']}")
        if raw.get("support_status") != "not_measured":
            raise SupportCompletionExecutionError(f"refusing pre-measured H0 owner for {context['context_id']}")
        if raw.get("exact_prefix_token_ids") != context["exact_prefix_token_ids"] or raw.get("exact_prefix_sha256") != context["exact_prefix_sha256"]:
            raise SupportCompletionExecutionError(f"H0 exact prefix mismatch for {context['context_id']}")
    return CandidateBank(panel, h0, tuple(physical), groups, by_id, h0_by_owner, str(plan["plan_content_sha256"]))


def make_hf_scorer(plan: Mapping[str, Any], bank: CandidateBank, infer_config: str | Path) -> Any:
    """Construct the established native HF scorer without opening a model."""

    contract = support_probe.validate_checkpoint_config(
        infer_config,
        checkpoint=CHECKPOINT,
        panel=bank.panel,
        h0=bank.h0,
    )
    if contract.wrapper != WRAPPER or contract.config_fingerprint != plan["h0_lineage"]["config_fingerprint"]:
        raise SupportCompletionExecutionError("infer config is not the sealed S step-2444 recipe")
    return support_probe.HFNativeRowScorer(contract, bank.panel, bank.h0)


def _normalize_resume_observation(row: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("context_id") != context["context_id"]:
        raise SupportCompletionExecutionError("resume observation context identity mismatch")
    status = row.get("status")
    if status not in {"measured", "partial", "failed"}:
        raise SupportCompletionExecutionError(f"resume observation has unknown status: {status!r}")
    scores = row.get("candidate_scores")
    if scores is None:
        scores = {}
    if not isinstance(scores, Mapping):
        raise SupportCompletionExecutionError("resume candidate_scores must be an object")
    planned = {str(item) for item in context["candidate_ids"]}
    normalized_scores: dict[str, float] = {}
    for key, value in scores.items():
        if str(key) not in planned:
            raise SupportCompletionExecutionError("resume candidate score is outside the sealed plan")
        normalized_scores[str(key)] = _finite(value, f"resume candidate score {key}")
    if row.get("candidate_ids") != list(context["candidate_ids"]):
        raise SupportCompletionExecutionError("resume observation candidate denominator drifted")
    expected_score_hash = sha256_json(dict(sorted(normalized_scores.items()))) if normalized_scores else None
    if row.get("candidate_scores_sha256") != expected_score_hash:
        raise SupportCompletionExecutionError("resume candidate score hash mismatch")
    return {**dict(row), "candidate_scores": normalized_scores}


def _load_resume_receipt(
    source: str | Path | Mapping[str, Any],
    plan: Mapping[str, Any],
    assigned: Sequence[Mapping[str, Any]],
    shard_index: int,
    *,
    census_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw, _info = _read_json_source(source)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise SupportCompletionExecutionError("resume receipt schema is incompatible")
    declared_receipt_hash = raw.get("receipt_content_sha256")
    if not isinstance(declared_receipt_hash, str):
        raise SupportCompletionExecutionError("resume receipt content hash is missing")
    receipt_without_hash = dict(raw)
    receipt_without_hash.pop("receipt_content_sha256", None)
    if sha256_json(receipt_without_hash) != declared_receipt_hash:
        raise SupportCompletionExecutionError("resume receipt content hash mismatch")
    if raw.get("plan_content_sha256") != plan.get("plan_content_sha256") or raw.get("shard_index") != shard_index or raw.get("num_shards") != NUM_SHARDS:
        raise SupportCompletionExecutionError("resume receipt is foreign to this plan/shard")
    census_aliases = {
        "census_file_sha256": "file_sha256",
        "census_self_sha256": "self_sha256",
        "census_s_owner_ids_sha256": "s_owner_ids_sha256",
    }
    for receipt_key, binding_key in census_aliases.items():
        if raw.get(receipt_key) != census_binding.get(binding_key):
            raise SupportCompletionExecutionError(f"resume receipt {receipt_key} drifted")
    if raw.get("census_binding") != dict(census_binding):
        raise SupportCompletionExecutionError("resume receipt census binding drifted")
    if raw.get("batch_estimates") != batch_estimates(plan):
        raise SupportCompletionExecutionError("resume receipt batch estimates drifted")
    assigned_ids = [str(row["context_id"]) for row in assigned]
    if raw.get("assigned_context_ids_sha256") != sha256_json(assigned_ids):
        raise SupportCompletionExecutionError("resume receipt assigned-context identity drifted")
    observations_raw = raw.get("observations")
    if not isinstance(observations_raw, list):
        raise SupportCompletionExecutionError("resume receipt has no observations")
    failure_log = raw.get("failure_log")
    if not isinstance(failure_log, list):
        raise SupportCompletionExecutionError("resume receipt has no failure log")
    expected_failure_hash = sha256_bytes(
        b"".join(canonical_json_bytes(item) + b"\n" for item in failure_log)
    )
    if raw.get("failure_log_content_sha256") != expected_failure_hash:
        raise SupportCompletionExecutionError("resume receipt failure-log hash mismatch")
    expected_scalar = sum(int(row["scalar_equivalent_forward_count"]) for row in assigned)
    if raw.get("expected_scalar_forward_count") != expected_scalar:
        raise SupportCompletionExecutionError("resume receipt scalar denominator drifted")
    if raw.get("support_calibration_sha256") != plan["calibration_reuse"]["calibration_sha256"]:
        raise SupportCompletionExecutionError("resume receipt calibration identity drifted")
    by_id = {str(row["context_id"]): row for row in assigned}
    observations: list[dict[str, Any]] = []
    seen_observation_ids: set[str] = set()
    for item in observations_raw:
        if not isinstance(item, Mapping) or str(item.get("context_id")) not in by_id:
            raise SupportCompletionExecutionError("resume receipt contains a foreign context")
        context_id = str(item["context_id"])
        if context_id in seen_observation_ids:
            raise SupportCompletionExecutionError("resume receipt duplicates a context observation")
        seen_observation_ids.add(context_id)
        observations.append(_normalize_resume_observation(item, by_id[context_id]))
    return dict(raw), observations


def execute_shard(
    plan: Mapping[str, Any] | str | Path,
    *,
    shard_index: int,
    scorer: Any,
    bank: CandidateBank,
    num_shards: int = NUM_SHARDS,
    census_path: str | Path | Mapping[str, Any] = DEFAULT_CENSUS_PATH,
    device: str | None = None,
    shard_device_map: Mapping[Any, Any] | None = None,
    candidate_batch_size: int = 1,
    batching_admitted: bool = False,
    batching_admission: Mapping[str, Any] | None = None,
    resume_receipt: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one exact scalar shard and return an immutable-ready result."""

    if isinstance(plan, (str, Path)):
        plan, source_info = validate_execution_plan(plan, census_path=census_path)
    else:
        plan, source_info = validate_execution_plan(plan, census_path=census_path)
    census_binding = source_info["census"]
    if not isinstance(bank, CandidateBank):
        raise SupportCompletionExecutionError("execute_shard requires a validated CandidateBank")
    if bank.plan_content_sha256 != plan.get("plan_content_sha256"):
        raise SupportCompletionExecutionError("candidate bank provenance does not match the sealed plan")
    batch = validate_batching_request(
        candidate_batch_size=candidate_batch_size,
        batching_admitted=batching_admitted,
        batching_admission=batching_admission,
    )
    mapping = validate_device_mapping(
        shard_index=shard_index,
        num_shards=num_shards,
        device=device,
        shard_device_map=shard_device_map,
    )
    if mapping.get("logical_device") is None:
        raise SupportCompletionExecutionError(
            "live shard execution requires --device or an explicit shard_device_map entry"
        )
    # The runner does not rewrite process-wide device visibility.  The caller
    # must bind one logical CUDA device, and the opened HF runtime is attested
    # against that binding below.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [token.strip() for token in visible_devices.split(",")] if visible_devices else []
    if len(visible_tokens) != 1 or not visible_tokens[0] or visible_tokens[0] == "-1":
        raise SupportCompletionExecutionError(
            "live shard execution requires caller-bound single-device CUDA_VISIBLE_DEVICES"
        )
    assigned = shard_contexts(plan, shard_index=shard_index, num_shards=num_shards)
    prior_receipt: dict[str, Any] | None = None
    prior_observations: list[dict[str, Any]] = []
    if resume_receipt is not None:
        prior_receipt, prior_observations = _load_resume_receipt(
            resume_receipt,
            plan,
            assigned,
            shard_index,
            census_binding=census_binding,
        )
    observations_by_id = {str(row["context_id"]): row for row in prior_observations}
    failures: list[dict[str, Any]] = []
    if prior_receipt is not None and isinstance(prior_receipt.get("failure_log"), list):
        failures.extend(dict(item) for item in prior_receipt["failure_log"] if isinstance(item, Mapping))
    attempt_failures: list[dict[str, Any]] = []
    attempted = 0
    reused = 0
    scorer_open = getattr(scorer, "open", None)
    scorer_close = getattr(scorer, "close", None)
    opened = False
    runtime_device_attestation: dict[str, Any] | None = None
    try:
        if callable(scorer_open):
            scorer_open()
            opened = True
        runtime_device_attestation = attest_scorer_device(scorer, mapping, plan=plan)
        if not callable(getattr(scorer, "score", None)):
            raise SupportCompletionExecutionError("scorer does not expose exact scalar score(record,candidate)")
        for context in assigned:
            context_id = str(context["context_id"])
            prior = observations_by_id.get(context_id)
            score_map: dict[str, float] = {}
            if prior is not None:
                score_map.update({str(key): _finite(value, f"resume score {context_id}.{key}") for key, value in prior.get("candidate_scores", {}).items()})
                reused += len(score_map)
            context_failures: list[dict[str, Any]] = []
            group = bank.groups[(int(context["image_id"]), str(context["category_name"]).lower())]
            for target in context["target_rows"]:
                candidate_id = str(target["candidate_id"])
                if candidate_id in score_map:
                    continue
                candidate = bank.by_id[candidate_id]
                try:
                    value = scorer.score(bank.h0_by_owner[str(context["gt_owner_id"])], candidate)
                    score_map[candidate_id] = _finite(value, f"candidate score {context_id}.{candidate_id}")
                    attempted += 1
                except Exception as exc:  # noqa: BLE001 - failure is persisted and no fallback is allowed
                    failure = {
                        "schema_version": FAILURE_SCHEMA_VERSION,
                        "plan_content_sha256": plan["plan_content_sha256"],
                        "shard_index": shard_index,
                        "context_id": context_id,
                        "gt_owner_id": str(context["gt_owner_id"]),
                        "image_id": int(context["image_id"]),
                        "candidate_id": candidate_id,
                        "phase": "teacher_forced_candidate_score",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    attempt_failures.append(failure)
                    context_failures.append(failure)
            if context_failures or len(score_map) != len(context["candidate_ids"]):
                observations_by_id[context_id] = {
                    "context_id": context_id,
                    "stable_key": context["stable_key"],
                    "gt_owner_id": context["gt_owner_id"],
                    "image_id": int(context["image_id"]),
                    "status": "failed" if context_failures else "partial",
                    "candidate_ids": list(context["candidate_ids"]),
                    "candidate_scores": dict(sorted(score_map.items())),
                    "candidate_score_count": len(score_map),
                    "candidate_scores_sha256": sha256_json(dict(sorted(score_map.items()))) if score_map else None,
                    "support_features": None,
                    "failure_count": len(context_failures),
                }
                continue
            try:
                features = support_probe.support_features(score_map, group, owner_id=str(context["gt_owner_id"]))
            except Exception as exc:  # feature recomputation is also a persisted failure, never a fallback
                failure = {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "plan_content_sha256": plan["plan_content_sha256"],
                    "shard_index": shard_index,
                    "context_id": context_id,
                    "gt_owner_id": str(context["gt_owner_id"]),
                    "image_id": int(context["image_id"]),
                    "candidate_id": None,
                    "phase": "support_feature_recompute",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                attempt_failures.append(failure)
                observations_by_id[context_id] = {
                    "context_id": context_id,
                    "stable_key": context["stable_key"],
                    "gt_owner_id": context["gt_owner_id"],
                    "image_id": int(context["image_id"]),
                    "status": "failed",
                    "candidate_ids": list(context["candidate_ids"]),
                    "candidate_scores": dict(sorted(score_map.items())),
                    "candidate_score_count": len(score_map),
                    "candidate_scores_sha256": sha256_json(dict(sorted(score_map.items()))),
                    "support_features": None,
                    "failure_count": 1,
                }
                continue
            observations_by_id[context_id] = {
                "context_id": context_id,
                "stable_key": context["stable_key"],
                "gt_owner_id": context["gt_owner_id"],
                "image_id": int(context["image_id"]),
                "status": "measured",
                "candidate_ids": list(context["candidate_ids"]),
                "candidate_scores": dict(sorted(score_map.items())),
                "candidate_score_count": len(score_map),
                "candidate_scores_sha256": sha256_json(dict(sorted(score_map.items()))),
                "support_features": dict(features),
                "failure_count": 0,
            }
    finally:
        if opened and callable(scorer_close):
            scorer_close()
    observations = [observations_by_id[str(context["context_id"])] for context in assigned]
    expected_scalar = sum(int(context["scalar_equivalent_forward_count"]) for context in assigned)
    observed_scalar = sum(int(row.get("candidate_score_count", 0)) for row in observations)
    # Historical failures remain in the immutable receipt for provenance, but
    # a later attempt may repair them.  Completion is therefore based on this
    # attempt's failures plus the full current observation set.
    complete = observed_scalar == expected_scalar and all(row.get("status") == "measured" for row in observations) and not attempt_failures
    receipt_core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "completed" if complete else "incomplete",
        "plan_content_sha256": plan["plan_content_sha256"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "device_mapping": mapping,
        "batching": batch,
        "batch_estimates": batch_estimates(plan),
        "census_binding": dict(census_binding),
        "census_file_sha256": census_binding["file_sha256"],
        "census_self_sha256": census_binding["self_sha256"],
        "census_s_owner_ids_sha256": census_binding["s_owner_ids_sha256"],
        "support_calibration_sha256": plan["calibration_reuse"]["calibration_sha256"],
        "support_rule": plan["support_lineage"]["support_rule"],
        "runtime_device_attestation": runtime_device_attestation,
        "assigned_context_count": len(assigned),
        "assigned_context_ids_sha256": sha256_json([str(row["context_id"]) for row in assigned]),
        "expected_scalar_forward_count": expected_scalar,
        "realized_scalar_forward_count": observed_scalar,
        "attempted_scalar_forward_count": attempted,
        "resumed_reused_scalar_forward_count": reused,
        "complete_assigned_observations": complete,
        "observations": observations,
        "failure_log": failures,
        "failure_count": len(failures),
        "native_tp_calibration_scored": False,
        "native_tp_other_not_scored": EXPECTED_UNSCORED_TP,
        "support_completion_denominator_only": True,
        "legacy_frozen_candidate_registry_read": False,
        "no_future_or_intervention_leakage": True,
    }
    failure_payload = b"".join(canonical_json_bytes(item) + b"\n" for item in failures)
    receipt_core["failure_log_content_sha256"] = sha256_bytes(failure_payload)
    receipt_core["receipt_content_sha256"] = sha256_json(receipt_core)
    return {
        "receipt": receipt_core,
        "failures": failures,
        "plan": plan,
        "contexts": assigned,
        "expected_scalar_forward_count": expected_scalar,
        "realized_scalar_forward_count": observed_scalar,
        "complete": complete,
    }


def _write_immutable(path: str | Path, payload: bytes) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != payload:
            raise SupportCompletionExecutionError(f"existing output is not identical: {destination}")
    else:
        destination.write_bytes(payload)
    return sha256_bytes(payload)


def write_shard_artifacts(
    result: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    failure_log_path: str | Path,
) -> dict[str, Any]:
    """Persist a shard receipt and JSONL failure log without overwrite."""

    receipt = result.get("receipt")
    failures = result.get("failures")
    if not isinstance(receipt, Mapping) or not isinstance(failures, list):
        raise SupportCompletionExecutionError("execution result has no receipt/failure payload")
    receipt_payload = canonical_json_bytes(receipt) + b"\n"
    failure_payload = b"".join(canonical_json_bytes(item) + b"\n" for item in failures)
    if receipt.get("failure_log_content_sha256") != sha256_bytes(failure_payload):
        raise SupportCompletionExecutionError("receipt/failure-log content hash mismatch")
    receipt_sha = _write_immutable(receipt_path, receipt_payload)
    failure_sha = _write_immutable(failure_log_path, failure_payload)
    return {
        "receipt_path": str(Path(receipt_path).expanduser().resolve()),
        "failure_log_path": str(Path(failure_log_path).expanduser().resolve()),
        "receipt_sha256": receipt_sha,
        "failure_log_sha256": failure_sha,
        "status": receipt.get("status"),
    }


def contract_summary(
    plan: Mapping[str, Any],
    *,
    shard_index: int,
    device: str | None = None,
    shard_device_map: Mapping[Any, Any] | None = None,
    census_path: str | Path | Mapping[str, Any] = DEFAULT_CENSUS_PATH,
) -> dict[str, Any]:
    """CPU-only readiness summary; does not load a model or score a row."""

    mapping = validate_device_mapping(
        shard_index=shard_index,
        num_shards=NUM_SHARDS,
        device=device,
        shard_device_map=shard_device_map,
    )
    contexts = shard_contexts(plan, shard_index=shard_index, num_shards=NUM_SHARDS)
    census_binding = validate_admission_census(census_path, plan=plan)
    estimates = batch_estimates(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "contract_ready",
        "plan_content_sha256": plan["plan_content_sha256"],
        "census_binding": dict(census_binding),
        "census_file_sha256": census_binding["file_sha256"],
        "census_self_sha256": census_binding["self_sha256"],
        "census_s_owner_ids_sha256": census_binding["s_owner_ids_sha256"],
        "checkpoint": CHECKPOINT,
        "wrapper": WRAPPER,
        "parser": PARSER,
        "shard_index": shard_index,
        "num_shards": NUM_SHARDS,
        "context_count": len(contexts),
        "scalar_equivalent_forward_count": sum(int(row["scalar_equivalent_forward_count"]) for row in contexts),
        "candidate_rows_authority": "sealed_plan_context_target_rows",
        "legacy_frozen_candidate_registry_read": False,
        "native_tp_calibration_scored": False,
        "native_tp_other_not_scored": EXPECTED_UNSCORED_TP,
        "batching_admitted": False,
        "batch_estimates": estimates,
        "device_mapping": mapping,
        "gpu_used": False,
        "model_loaded": False,
    }


def _parse_device_map(raw: str | None) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupportCompletionExecutionError("--device-map must be JSON object") from exc
    if not isinstance(value, Mapping):
        raise SupportCompletionExecutionError("--device-map must be JSON object")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS_PATH)
    parser.add_argument("--mode", choices=("contract", "live"), default="contract")
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--device")
    parser.add_argument("--device-map", help="JSON object mapping shard indices to CUDA devices")
    parser.add_argument("--infer-config", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--batching-admitted", action="store_true")
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    plan, _source_info = validate_execution_plan(
        args.plan,
        expected_plan_sha256=args.plan_sha256,
        census_path=args.census,
    )
    device_map = _parse_device_map(args.device_map)
    if args.mode == "contract":
        summary = contract_summary(
            plan,
            shard_index=args.shard_index,
            device=args.device,
            shard_device_map=device_map,
            census_path=args.census,
        )
        if args.receipt is not None:
            _write_immutable(args.receipt, canonical_json_bytes(summary) + b"\n")
        return summary
    if args.infer_config is None:
        raise SupportCompletionExecutionError("live mode requires --infer-config")
    if args.receipt is None or args.failure_log is None:
        raise SupportCompletionExecutionError("live mode requires --receipt and --failure-log")
    bank = build_candidate_bank(plan)
    scorer = make_hf_scorer(plan, bank, args.infer_config)
    result = execute_shard(
        plan,
        shard_index=args.shard_index,
        scorer=scorer,
        bank=bank,
        num_shards=args.num_shards,
        census_path=args.census,
        device=args.device,
        shard_device_map=device_map,
        candidate_batch_size=args.candidate_batch_size,
        batching_admitted=args.batching_admitted,
        resume_receipt=args.resume_receipt,
    )
    artifact = write_shard_artifacts(result, receipt_path=args.receipt, failure_log_path=args.failure_log)
    return {"receipt": result["receipt"], "artifacts": artifact}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_cli(args)
    except SupportCompletionExecutionError as exc:
        print(f"natural-boundary-support-completion: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
