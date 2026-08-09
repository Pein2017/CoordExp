#!/usr/bin/env python3
"""Merge the sealed natural-boundary support-completion shard receipts.

This is a CPU-only, write-once reducer for the 2026-08-06 S support plan.  It
does not read the prior frozen cohort registry and does not load a model.  The
sealed plan is the only shard partition authority; every receipt must account
for one of its eight content-stable partitions exactly once.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_natural_boundary_support_completion as execution


SCHEMA_VERSION = "natural_boundary_owner_support_completion_ledger.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt.v1"
PLAN_SCHEMA_VERSION = "natural_boundary_owner_support_completion_plan.v1"
PLAN_STATUS = "sealed_cpu_plan"
UNIT_ID = execution.UNIT_ID
CHECKPOINT = execution.CHECKPOINT
NUM_SHARDS = execution.NUM_SHARDS
EXPECTED_CONTEXTS = execution.EXPECTED_CONTEXTS
EXPECTED_SCALAR_FORWARDS = execution.EXPECTED_SCALAR_FORWARDS
EXPECTED_NATIVE_FN = execution.EXPECTED_NATIVE_FN
EXPECTED_MEASURED_FN = execution.EXPECTED_MEASURED_FN
EXPECTED_PANEL_SHA256 = execution.EXPECTED_SOURCE_PANEL_SHA256
EXPECTED_DERIVED_PANEL_SHA256 = execution.EXPECTED_DERIVED_PANEL_SHA256
EXPECTED_PLAN_SHA256 = "1b7e97af291b58aec50849ae09aa883148d55610c281a9edb543fce46ec21d4c"
EXPECTED_EXECUTION_CONTRACT = {
    "batching_admitted": False,
    "cpu_only_planner": True,
    "gpu_used": False,
    "model_loaded": False,
    "native_h0_prefix_only": True,
    "source_frozen_candidate_registry_mutated": False,
    "support_capture_held_until_s_gate": True,
    "teacher_forced_diagnostic_only": True,
    "training": False,
}
EXPECTED_CENSUS_BINDING = {
    "revision": "cpu-census-v2",
    "path": "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/cpu-census-v2/admission-census.json",
    "file_sha256": "dd1c61abb9acff7f4fc42380365439ee931db604525749f297bbc3e191a26a2e",
    "self_sha256": "11090ae4bc8c2f7f8a676f83e91194c2e0e7270a6d359c7dabf2cf771b38879d",
    "s_owner_ids_sha256": "a666e116db950491027915cd65f4c1a949199d27be7c0c42f72e8583b863f551",
    "s_owner_count": 200,
    "support_completion_candidates_sha256": "286c296e5b60c4f1bd3190f7e61aba1432309a9cd824f123a6d59d288a2f3072",
}


class MergeContractError(ValueError):
    """Raised when a plan, shard, or support lineage is not decision-ready."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MergeContractError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    try:
        return sha256_bytes(Path(path).expanduser().resolve(strict=True).read_bytes())
    except OSError as exc:
        raise MergeContractError(f"cannot read input {path}: {exc}") from exc


def _read_json(source: str | Path | Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        raw_path = Path(source).expanduser()
        if raw_path.is_symlink() or not raw_path.is_file():
            raise MergeContractError(f"input is not a regular non-symlink file: {raw_path}")
        path = raw_path.resolve(strict=True)
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MergeContractError(f"invalid JSON: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw)}
    if isinstance(source, Mapping):
        value = dict(source)
        return value, {"inline": True, "sha256": sha256_json(value)}
    raise MergeContractError("input must be a JSON path or object")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MergeContractError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise MergeContractError(f"{label} is not finite")
    return result


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise MergeContractError(f"{label} is not a lowercase SHA-256")
    return value.lower()


def _write_once(path: str | Path, payload: bytes) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise MergeContractError(f"refusing to overwrite immutable artifact: {destination}")
    else:
        destination.write_bytes(payload)
    return sha256_bytes(payload)


def _validate_plan_generic(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("status") != PLAN_STATUS:
        raise MergeContractError("plan schema/status is not the sealed CPU plan v1")
    if plan.get("unit_id") != UNIT_ID or plan.get("checkpoint") != CHECKPOINT:
        raise MergeContractError("plan unit/checkpoint identity drifted")
    if plan.get("execution_contract") != EXPECTED_EXECUTION_CONTRACT:
        raise MergeContractError("plan execution contract drifted")
    plan_hash = plan.get("plan_content_sha256")
    if not isinstance(plan_hash, str):
        raise MergeContractError("plan content hash is missing")
    body = dict(plan)
    body.pop("plan_content_sha256", None)
    if sha256_json(body) != plan_hash:
        raise MergeContractError("plan content hash mismatch")
    contexts = plan.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != EXPECTED_CONTEXTS:
        raise MergeContractError("plan must contain exactly 200 contexts")
    by_id: dict[str, dict[str, Any]] = {}
    for context in contexts:
        if not isinstance(context, Mapping):
            raise MergeContractError("plan context is not an object")
        context_id = str(context.get("context_id"))
        if context_id in by_id:
            raise MergeContractError(f"plan repeats context {context_id}")
        if context.get("native_fn") is not True or context.get("native_tp") is not False:
            raise MergeContractError(f"plan context {context_id} is not an unassessed native FN")
        by_id[context_id] = dict(context)
    work = plan.get("work")
    if not isinstance(work, Mapping) or work.get("shard_count") != NUM_SHARDS:
        raise MergeContractError("plan is not the sealed eight-shard contract")
    if int(work.get("scalar_equivalent_forward_count", -1)) != EXPECTED_SCALAR_FORWARDS:
        raise MergeContractError("plan scalar denominator is not 77,428")
    return dict(plan), by_id


def _validate_shard(
    source: str | Path | Mapping[str, Any],
    *,
    expected_index: int,
    plan: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]],
    census_binding: Mapping[str, Any] | None = None,
    strict_live: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, info = _read_json(source)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != execution.RECEIPT_SCHEMA_VERSION:
        raise MergeContractError(f"shard {expected_index} has incompatible receipt schema")
    if raw.get("status") != "completed":
        raise MergeContractError(f"shard {expected_index} is not completed")
    if raw.get("unit_id") != UNIT_ID or raw.get("plan_content_sha256") != plan.get("plan_content_sha256"):
        raise MergeContractError(f"shard {expected_index} is foreign to the plan")
    if raw.get("shard_index") != expected_index or raw.get("num_shards") != NUM_SHARDS:
        raise MergeContractError(f"shard {expected_index} has the wrong shard identity")
    if strict_live:
        if raw.get("receipt_content_sha256") != sha256_json({key: value for key, value in raw.items() if key != "receipt_content_sha256"}):
            raise MergeContractError(f"shard {expected_index} receipt content hash mismatch")
        if raw.get("census_binding") != dict(census_binding or EXPECTED_CENSUS_BINDING):
            raise MergeContractError(f"shard {expected_index} census binding mismatch")
        if raw.get("census_file_sha256") != EXPECTED_CENSUS_BINDING["file_sha256"] or raw.get("census_self_sha256") != EXPECTED_CENSUS_BINDING["self_sha256"] or raw.get("census_s_owner_ids_sha256") != EXPECTED_CENSUS_BINDING["s_owner_ids_sha256"]:
            raise MergeContractError(f"shard {expected_index} census identity mismatch")
        if raw.get("support_calibration_sha256") != plan.get("calibration_reuse", {}).get("calibration_sha256") or raw.get("support_rule") != plan.get("support_lineage", {}).get("support_rule") or raw.get("no_future_or_intervention_leakage") is not True:
            raise MergeContractError(f"shard {expected_index} support lineage mismatch")
    assigned = execution.shard_contexts(plan, shard_index=expected_index)
    assigned_ids = [str(row["context_id"]) for row in assigned]
    if raw.get("assigned_context_ids_sha256") != sha256_json(assigned_ids):
        raise MergeContractError(f"shard {expected_index} assigned-context hash mismatch")
    expected_scalar = sum(int(row["scalar_equivalent_forward_count"]) for row in assigned)
    if raw.get("assigned_context_count") != len(assigned) or raw.get("expected_scalar_forward_count") != expected_scalar:
        raise MergeContractError(f"shard {expected_index} denominator mismatch")
    if raw.get("realized_scalar_forward_count") != expected_scalar or raw.get("complete_assigned_observations") is not True:
        raise MergeContractError(f"shard {expected_index} is incomplete")
    failures = raw.get("failure_log")
    if raw.get("failure_count") != 0 or failures != []:
        raise MergeContractError(f"shard {expected_index} contains failures")
    if raw.get("failure_log_content_sha256") != sha256_bytes(b""):
        raise MergeContractError(f"shard {expected_index} failure-log hash is non-empty")
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != len(assigned):
        raise MergeContractError(f"shard {expected_index} observation count mismatch")
    expected_set = set(assigned_ids)
    observed_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise MergeContractError(f"shard {expected_index} observation is not an object")
        context_id = str(observation.get("context_id"))
        if context_id not in expected_set or context_id in observed_ids:
            raise MergeContractError(f"shard {expected_index} has foreign/duplicate context {context_id}")
        observed_ids.add(context_id)
        if observation.get("status") != "measured" or not isinstance(observation.get("support_features"), Mapping):
            raise MergeContractError(f"shard {expected_index} context {context_id} is not measured")
        context = contexts[context_id]
        expected_count = int(context["scalar_equivalent_forward_count"])
        if observation.get("candidate_score_count") != expected_count:
            raise MergeContractError(f"shard {expected_index} context {context_id} score count mismatch")
        scores = observation.get("candidate_scores")
        if not isinstance(scores, Mapping) or set(map(str, scores)) != set(map(str, context["candidate_ids"])):
            raise MergeContractError(f"shard {expected_index} context {context_id} candidate IDs mismatch")
        normalized = {str(key): _finite(value, f"{context_id}.{key}") for key, value in scores.items()}
        if observation.get("candidate_scores_sha256") != sha256_json(dict(sorted(normalized.items()))):
            raise MergeContractError(f"shard {expected_index} context {context_id} score hash mismatch")
    if observed_ids != expected_set:
        raise MergeContractError(f"shard {expected_index} observations do not cover its assigned contexts")
    return dict(raw), info


def _support_value(features: Mapping[str, Any], calibration: Mapping[str, Any]) -> bool:
    if features.get("assessed") is not True:
        return False
    peak = features.get("peak_lift")
    concentration = features.get("local_concentration")
    if peak is None or concentration is None:
        return False
    return (
        _finite(peak, "peak_lift") >= _finite(calibration["theta_peak_lift"], "theta_peak_lift") + _finite(calibration["epsilon"], "epsilon")
        and _finite(concentration, "local_concentration") >= _finite(calibration["theta_local_concentration"], "theta_local_concentration") + _finite(calibration["epsilon"], "epsilon")
    )


def _load_h0(plan: Mapping[str, Any], source: str | Path | Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    h0_meta = plan.get("h0_lineage")
    if not isinstance(h0_meta, Mapping):
        raise MergeContractError("plan lacks H0 lineage")
    selected = source or h0_meta.get("source", {}).get("path")
    if not selected:
        raise MergeContractError("H0 source path is missing")
    value, info = _read_json(selected)
    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
        raise MergeContractError("H0 source is not an owner ledger")
    expected = str(h0_meta.get("source_sha256"))
    if info.get("sha256") != expected:
        raise MergeContractError("H0 source hash differs from sealed plan")
    if sha256_json(value["records"]) != h0_meta.get("records_sha256"):
        raise MergeContractError("H0 records hash differs from sealed plan")
    if value.get("checkpoint") != CHECKPOINT or value.get("config_fingerprint") != h0_meta.get("config_fingerprint"):
        raise MergeContractError("H0 checkpoint/config identity drifted")
    by_owner = {str(row.get("gt_owner_id")): dict(row) for row in value["records"] if isinstance(row, Mapping)}
    if len(by_owner) != execution.EXPECTED_NATIVE_TP + EXPECTED_NATIVE_FN:
        raise MergeContractError("H0 owner denominator drifted")
    return dict(value), info


def merge_support_receipts(
    plan: Mapping[str, Any],
    shard_sources: Mapping[int, str | Path | Mapping[str, Any]],
    *,
    prior_support: str | Path | Mapping[str, Any] | None = None,
    h0_source: str | Path | Mapping[str, Any] | None = None,
    input_plan_sha256: str | None = None,
    input_census_binding: Mapping[str, Any] | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    """Validate eight complete receipts and produce the 220-owner S ledger."""

    plan, contexts = _validate_plan_generic(plan)
    if not test_only and (input_plan_sha256 is None or not isinstance(input_census_binding, Mapping) or not input_census_binding):
        raise MergeContractError("live merge requires sealed plan byte hash and non-empty census binding")
    if not test_only and dict(input_census_binding) != EXPECTED_CENSUS_BINDING:
        raise MergeContractError("live merge census binding is not the sealed census-v2 identity")
    if input_plan_sha256 is not None and input_plan_sha256 != EXPECTED_PLAN_SHA256:
        raise MergeContractError("plan byte hash is not the sealed v1 plan")
    if set(shard_sources) != set(range(NUM_SHARDS)):
        raise MergeContractError("merge requires exactly one receipt for each shard 0..7")
    shards: list[dict[str, Any]] = []
    shard_refs: list[dict[str, Any]] = []
    for index in range(NUM_SHARDS):
        receipt, info = _validate_shard(shard_sources[index], expected_index=index, plan=plan, contexts=contexts, census_binding=input_census_binding, strict_live=not test_only)
        shards.append(receipt)
        shard_refs.append({"shard_index": index, "path": info.get("path"), "sha256": info["sha256"]})
    h0, h0_info = _load_h0(plan, h0_source)
    h0_by_owner = {str(row["gt_owner_id"]): row for row in h0["records"]}
    calibration = plan.get("calibration_reuse", {}).get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("calibration_sha256") != plan.get("calibration_reuse", {}).get("calibration_sha256"):
        raise MergeContractError("frozen calibration is missing or detached")
    lineage = plan.get("support_lineage")
    if not isinstance(lineage, Mapping):
        raise MergeContractError("sealed plan lacks prior support lineage")
    support_rule = lineage.get("support_rule")
    if not isinstance(support_rule, Mapping):
        raise MergeContractError("sealed plan lacks a support rule")
    expected_prior_sha = lineage.get("file_sha256")
    expected_prior_count = lineage.get("record_count")
    expected_prior_path = lineage.get("path")
    if not prior_support and not test_only:
        raise MergeContractError("live merge requires prior support lineage input")
    if prior_support is None:
        raise MergeContractError("prior support lineage input is missing")
    prior_value, prior_info = _read_json(prior_support)
    if not isinstance(prior_value, Mapping) or prior_info.get("sha256") != expected_prior_sha:
        raise MergeContractError("prior S support lineage differs from sealed plan")
    if expected_prior_path and prior_info.get("path") != str(Path(expected_prior_path).expanduser().resolve()):
        raise MergeContractError("prior S support path differs from sealed plan")
    if expected_prior_count is not None and len(prior_value.get("records", [])) != int(expected_prior_count):
        raise MergeContractError("prior S support record count differs from sealed plan")
    if prior_value.get("checkpoint") != CHECKPOINT or prior_value.get("calibration", {}).get("calibration_sha256") != calibration.get("calibration_sha256"):
        raise MergeContractError("prior support calibration/checkpoint identity drifted")
    prior_records = [dict(row) for row in prior_value.get("records", []) if isinstance(row, Mapping) and row.get("native_fn") is True]
    expected_measured_prior = int(plan.get("scope", {}).get("support_measured_fn_retained", EXPECTED_MEASURED_FN))
    if len(prior_records) != expected_measured_prior:
        raise MergeContractError("prior support must retain exactly 20 measured native-FN records")
    by_owner: dict[str, dict[str, Any]] = {}
    for row in prior_records:
        owner = str(row.get("gt_owner_id"))
        if owner in by_owner or owner not in h0_by_owner:
            raise MergeContractError(f"prior support owner is duplicated/absent from H0: {owner}")
        if "support_rule" not in row:
            row["support_rule"] = dict(support_rule)
        elif row["support_rule"] != support_rule:
            raise MergeContractError(f"prior support owner {owner} support rule differs from sealed plan")
        by_owner[owner] = row
    for shard in shards:
        for observation in shard["observations"]:
            context = contexts[str(observation["context_id"])]
            owner = str(context["gt_owner_id"])
            if owner in by_owner:
                raise MergeContractError(f"new support repeats prior owner {owner}")
            h0_row = h0_by_owner.get(owner)
            if h0_row is None or h0_row.get("native_fn") is not True or h0_row.get("strict_complete_row") is not False:
                raise MergeContractError(f"support owner {owner} is not an S native FN")
            if h0_row.get("exact_prefix_sha256") != context.get("exact_prefix_sha256") or h0_row.get("exact_prefix_token_ids") != context.get("exact_prefix_token_ids"):
                raise MergeContractError(f"support owner {owner} exact prefix differs from plan")
            features = dict(observation["support_features"])
            row = dict(h0_row)
            row.update(
                {
                    "support_status": "measured",
                    "verified_support": _support_value(features, calibration),
                    "verified_support_claim": True,
                    "support_features": features,
                    "support_calibration_sha256": calibration["calibration_sha256"],
                    "support_rule": dict(plan["support_lineage"]["support_rule"]),
                    "support_semantics": "physical_owner_specific_complete_row_geometry_local_peak",
                    "teacher_forced_is_behavioral_transfer": False,
                    "no_future_or_intervention_leakage": True,
                    "candidate_score_count": int(observation["candidate_score_count"]),
                    "candidate_scores_sha256": observation["candidate_scores_sha256"],
                    "support_completion_context_id": str(observation["context_id"]),
                }
            )
            by_owner[owner] = row
    if len(by_owner) != EXPECTED_NATIVE_FN:
        raise MergeContractError(f"merged S support denominator is {len(by_owner)}, expected {EXPECTED_NATIVE_FN}")
    records = [by_owner[owner] for owner in sorted(by_owner)]
    image_ids = sorted({int(row["image_id"]) for row in records})
    if len(image_ids) != 13:
        raise MergeContractError(f"merged S support must cover 13 images, observed {len(image_ids)}")
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "config_fingerprint": plan["h0_lineage"]["config_fingerprint"],
        "source_panel_sha256": EXPECTED_PANEL_SHA256,
        "derived_panel_sha256": EXPECTED_DERIVED_PANEL_SHA256,
        "run_kind": "native_h0",
        "arm": "native",
        "history_complete": True,
        "native_outcome_only": False,
        "verified_support_claim": True,
        "support_rule": dict(plan["support_lineage"]["support_rule"]),
        "calibration": dict(calibration),
        "plan_binding": {
            "plan_content_sha256": plan["plan_content_sha256"],
            "plan_sha256": input_plan_sha256 or EXPECTED_PLAN_SHA256,
            "census_binding": dict(input_census_binding or {}),
        },
        "prior_support_binding": {"path": prior_info.get("path"), "sha256": prior_info["sha256"], "record_count": len(prior_records)},
        "shard_bindings": shard_refs,
        "record_count": len(records),
        "image_count": len(image_ids),
        "image_ids": image_ids,
        "records": records,
    }
    ledger["records_sha256"] = sha256_json(records)
    ledger["content_sha256"] = sha256_json(ledger)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "unit_id": UNIT_ID,
        "checkpoint": CHECKPOINT,
        "record_count": len(records),
        "image_count": len(image_ids),
        "scalar_equivalent_forward_count": EXPECTED_SCALAR_FORWARDS,
        "verified_support_count": sum(row["verified_support"] is True for row in records),
        "records_sha256": ledger["records_sha256"],
        "ledger_content_sha256": ledger["content_sha256"],
        "plan_content_sha256": plan["plan_content_sha256"],
        "plan_sha256": input_plan_sha256 or EXPECTED_PLAN_SHA256,
        "h0_source": {"path": h0_info.get("path"), "sha256": h0_info["sha256"]},
        "prior_support": {"path": prior_info.get("path"), "sha256": prior_info["sha256"]},
        "shards": shard_refs,
    }
    receipt["self_sha256"] = sha256_json(receipt)
    return {"ledger": ledger, "receipt": receipt}


def merge_from_paths(
    plan_path: str | Path,
    census_path: str | Path,
    shard_sources: Mapping[int, str | Path],
    *,
    prior_support: str | Path | None = None,
    h0_source: str | Path | None = None,
    output: str | Path | None = None,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    plan, plan_info = _read_json(plan_path)
    if not isinstance(plan, Mapping) or plan_info.get("sha256") != EXPECTED_PLAN_SHA256:
        raise MergeContractError("plan path is not the sealed support-completion plan v1")
    census_binding = execution.validate_admission_census(census_path, plan=plan)
    # Full runner validation is deliberately done before reduction; it rejects
    # a copied/resealed plan and binds the CPU census revision.
    execution.validate_execution_plan(plan_path, expected_plan_sha256=EXPECTED_PLAN_SHA256, census_path=census_path)
    result = merge_support_receipts(
        plan,
        shard_sources,
        prior_support=prior_support,
        h0_source=h0_source,
        input_plan_sha256=plan_info["sha256"],
        input_census_binding=census_binding,
    )
    if output is not None:
        _write_once(output, canonical_json_bytes(result["ledger"]) + b"\n")
    if receipt_output is not None:
        _write_once(receipt_output, canonical_json_bytes(result["receipt"]) + b"\n")
    return result


def _parse_shards(values: Sequence[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        left, sep, right = value.partition("=")
        if not sep or not left.isdigit() or not right:
            raise MergeContractError("--shard must be INDEX=RECEIPT_PATH")
        index = int(left)
        if index in parsed:
            raise MergeContractError(f"duplicate shard {index}")
        parsed[index] = Path(right)
    if set(parsed) != set(range(NUM_SHARDS)):
        raise MergeContractError("exactly one shard receipt for each index 0..7 is required")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--shard", action="append", required=True, metavar="INDEX=RECEIPT_PATH")
    parser.add_argument("--prior-support", type=Path, required=True)
    parser.add_argument("--h0", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", dest="receipt_output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = merge_from_paths(
            args.plan,
            args.census,
            _parse_shards(args.shard),
            prior_support=args.prior_support,
            h0_source=args.h0,
            output=args.output,
            receipt_output=args.receipt_output,
        )
    except (MergeContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
