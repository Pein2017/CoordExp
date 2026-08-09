from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import scripts.research.analyze_s_natural_boundary_k_n_h_evidence as analyzer_module
from scripts.research.analyze_s_natural_boundary_k_n_h_evidence import (
    ARM_ORDER,
    EvidenceAnalysisError,
    K13_LAYER_RECEIPT_SCHEMA_VERSION,
    K13_MASK_RECEIPT_SCHEMA_VERSION,
    K13_NOT_APPLICABLE_REASON,
    PRIMARY,
    RECEIPT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    UNIT_ID,
    _delta,
    _dynamic_disposition,
    _k13_applicability,
    _k13_attestation,
    _producer_ref,
    _summarize_arm,
    _validate_gate_binding,
    _validate_natural_receipts,
    _validate_result_bindings,
    analyze_cohort,
    document_self_sha256,
    sha256_json,
)
from scripts.research.merge_s_natural_boundary_k_n_h_shards import _merge_shards_test_only
from scripts.research.plan_s_natural_boundary_k_n_h_execution import build_plan, validate_plan
from scripts.research.run_s_natural_boundary_k_n_h_cohort import ELIGIBILITY_PREDICATE_KEYS
from scripts.research.run_s_natural_boundary_k_n_h_shard import _run_shard_test_only


OPENER = 1
NATIVE_STOP = 2
INVALID = 3

_COHORT_FIXTURE_PATH = Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py")
_COHORT_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_analyzer_cohort_fixture", _COHORT_FIXTURE_PATH
)
assert _COHORT_FIXTURE_SPEC is not None and _COHORT_FIXTURE_SPEC.loader is not None
_COHORT_FIXTURE = importlib.util.module_from_spec(_COHORT_FIXTURE_SPEC)
_COHORT_FIXTURE_SPEC.loader.exec_module(_COHORT_FIXTURE)


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def _write_noncanonical_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(document, ensure_ascii=True, indent=2).encode() + b"\n")


def _identity_file(tmp_path: Path, name: str) -> dict[str, str]:
    path = tmp_path / name
    path.write_text(f"{name}\n", encoding="utf-8")
    return {"id": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _event_identity(index: int, image_count: int) -> tuple[int, int]:
    return 5001 + (index % image_count), 15 + (index // image_count)


def _census_rows(event_count: int, image_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(event_count):
        image_id, target_index = _event_identity(index, image_count)
        for owner_index in (2, target_index, 19):
            if (image_id, owner_index) in seen:
                continue
            seen.add((image_id, owner_index))
            rows.append({
                "checkpoint": "S",
                "gt_owner_id": f"gt:{image_id}:{owner_index}",
                "image_id": image_id,
                "source_panel_object_index": owner_index,
                "derived_panel_object_index": owner_index,
            })
    filler = 0
    while len(rows) < 392:
        rows.append({
            "checkpoint": "S",
            "gt_owner_id": f"gt:{9000 + filler // 100}:{100 + filler}",
            "image_id": 9000 + filler // 100,
            "source_panel_object_index": 100 + filler,
            "derived_panel_object_index": 100 + filler,
        })
        filler += 1
    rows.extend({
        "checkpoint": "A",
        "gt_owner_id": f"gt:{12000 + index // 100}:{index}",
        "image_id": 12000 + index // 100,
        "source_panel_object_index": index,
        "derived_panel_object_index": index,
    } for index in range(392))
    return rows


def _build_census(
    tmp_path: Path,
    event_count: int,
    image_count: int,
) -> tuple[dict[str, object], Path]:
    document: dict[str, object] = {
        "schema_version": "natural_boundary_owner_admission_census.v3",
        "status": "sealed",
        "unit_id": UNIT_ID,
        "census_revision": "census-v3",
        "rows": _census_rows(event_count, image_count),
    }
    document["self_sha256"] = sha256_json(document)
    path = tmp_path / "census-v3.json"
    _write_json(path, document)
    return document, path


def _build_manifest(
    tmp_path: Path,
    census_path: Path,
    event_count: int,
    image_count: int,
    k13_statuses: tuple[str, ...],
) -> tuple[dict[str, object], Path]:
    thresholds = {"minimum_event_count": 3, "minimum_image_count": 2, "epsilon": 0.002}
    rule_id = "synthetic_static_eligible_v3"
    opener_contract = {
        "status": "runner_resolved",
        "resolver": "serialization_successor_runner",
        "token_name": "<|object_ref_start|>",
        "resolution_semantics": "pre_opener_natural_prefix_ends_before_object_ref_start",
        "contract_sha256": sha256_json({"token_name": "<|object_ref_start|>", "resolver": "serialization_successor_runner"}),
    }
    events: list[dict[str, object]] = []
    for index in range(event_count):
        image_id, target_index = _event_identity(index, image_count)
        event_id = f"gt:{image_id}:{target_index}"
        k13_applicable = k13_statuses[index] == "applicable"
        competitor_cells = [1] if k13_applicable else []
        competitor_owner = f"gt:{image_id}:20" if k13_applicable else None
        competitor_weights = [{"cell_index": 1, "overlap_fraction": 1.0}] if k13_applicable else []
        competitor_region_receipt = {
            "status": "available" if k13_applicable else "not_applicable",
            "available": k13_applicable,
            "not_measured_reason": None if k13_applicable else "no_verified_same_class_competitor",
            "cell_indices": competitor_cells,
            "visual_indices": competitor_cells,
            "fractional_weights": competitor_weights,
            "weight_sum": 1.0 if k13_applicable else 0.0,
            "cell_count": len(competitor_cells),
            "cell_indices_sha256": sha256_json(competitor_cells),
            "weights_sha256": sha256_json(competitor_weights),
        }
        event: dict[str, object] = {
            "event_index": index,
            "event_id": event_id,
            "image_id": image_id,
            "owner_refs": {
                "gt_owner_id": event_id,
                "covered_owner_ids": [f"gt:{image_id}:2"],
                "covered_A_owner_id": f"gt:{image_id}:2",
                "source_panel_object_index": target_index,
                "derived_panel_object_index": target_index,
            },
            **PRIMARY,
            "admission": "admitted",
            "eligibility": {
                "admitted": True,
                "rule_id": rule_id,
                "thresholds": thresholds,
                "predicates": {key: True for key in ELIGIBILITY_PREDICATE_KEYS},
            },
            "natural_boundary": {
                "pre_opener_natural": True,
                "opener_seeded": False,
                "opener_injected": False,
                "synthetic_opener_injections": 0,
                "opener_token_id": None,
                "opener_token_contract": opener_contract,
                "prefix_token_ids": [9],
                "prefix_sha256": sha256_json([9]),
                "history_token_ids": [9],
                "history_sha256": sha256_json([9]),
            },
            "same_class_competitor_owner_id": competitor_owner,
            "image_cell_regions": {"same_class_competitor": competitor_cells},
            "image_cell_region_receipts": {"same_class_competitor": competitor_region_receipt},
        }
        event["event_sha256"] = sha256_json(event)
        events.append(event)
    cohort = {"id": "synthetic-cohort", "frozen_arms": list(ARM_ORDER)}
    operator = {"id": "synthetic-operator", "object_ref_start_token_id": OPENER}
    backend = {"id": "synthetic-backend"}
    document: dict[str, object] = {
        "schema_version": "s_natural_boundary_admitted_event_manifest.v3",
        "status": "sealed",
        "unit_id": UNIT_ID,
        "primary": dict(PRIMARY),
        "source_census": {
            "revision": "census-v3",
            "path": str(census_path),
            "sha256": hashlib.sha256(census_path.read_bytes()).hexdigest(),
            "hash_semantics": "canonical_json_document_with_trailing_newline",
        },
        "panel": _identity_file(tmp_path, "panel.jsonl"),
        "cohort": {**cohort, "sha256": sha256_json(cohort)},
        "operator": {**operator, "sha256": sha256_json(operator)},
        "backend": {**backend, "sha256": sha256_json(backend)},
        "eligibility": {"rule_id": rule_id, "thresholds": thresholds},
        "admission_gate": {
            "minimum_event_count": 3,
            "minimum_image_count": 2,
            "status": "deferred_to_successor_runner",
        },
        "arm_order": list(ARM_ORDER),
        "events": events,
        "event_count": len(events),
    }
    document["self_sha256"] = sha256_json(document)
    path = tmp_path / "manifest.json"
    _write_json(path, document)
    return document, path


def _build_gate(tmp_path: Path, *, noncanonical: bool = False) -> tuple[dict[str, object], Path]:
    write_json = _write_noncanonical_json if noncanonical else _write_json
    runtime: dict[str, object] = {
        "schema_version": "s_primary_natural_boundary_gate.v1.runtime_identity.v1",
        "unit_id": UNIT_ID,
        "checkpoint": "S",
        "event_id": "gt:5001:15",
        "gpu_launch_authorized": False,
        "no_training": True,
        "event": {"event_id": "gt:5001:15"},
    }
    runtime["identity_sha256"] = sha256_json(runtime)
    write_json(tmp_path / "runtime_identity.json", runtime)
    counts = {arm: (27 if index < 11 else 1) for index, arm in enumerate(ARM_ORDER)}
    gate: dict[str, object] = {
        "schema_version": "s_primary_natural_boundary_gate.v1",
        "unit_id": UNIT_ID,
        "checkpoint": "S",
        "event_id": "gt:5001:15",
        "gpu_launch_authorized": False,
        "no_training": True,
        "arm_order": list(ARM_ORDER),
        "arms": {
            arm: {"scalar_forward_count": count, "runtime_scalar_forward_count": count}
            for arm, count in counts.items()
        },
        "runtime_identity_sha256": runtime["identity_sha256"],
    }
    gate["result_sha256"] = sha256_json(gate)
    path = tmp_path / "gate.json"
    write_json(path, gate)
    return gate, path


def _row(
    *,
    index: int,
    status: str,
    first: int,
    owner_match: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "row_index": index,
        "status": status,
        "stop_reason": status,
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "opener_token_id": OPENER,
        "initial_prefix_last_token_id": 9,
        "opener_generated_by_model": first == OPENER,
        "row_started": first == OPENER,
        "first_generated_token_id": first,
        "token_ids": [first],
        "token_ids_sha256": sha256_json([first]),
        **({"owner_match": owner_match, "owner_match_status": owner_match["status"]} if owner_match is not None else {}),
    }


def _k13_runtime_receipt(index: int, status: str) -> dict[str, object]:
    image_positions = [1, 2]
    competitor_positions = [1] if status == "applicable" else []
    if status == "applicable":
        layer = {
            "schema_version": K13_LAYER_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "passed": True,
            "all_layers_identical": True,
            "missing_layers": [],
            "repeated_layers": [],
            "errors": [],
        }
        attention = {
            "schema_version": K13_MASK_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "arm_id": "K13",
            "status": "ready",
            "reason": None,
            "image_key_positions": image_positions,
            "query_positions": [9],
            "selected_positions": competitor_positions,
            "query_positions_sha256": sha256_json([9]),
            "selected_positions_sha256": sha256_json(competitor_positions),
            "selected_key_count": len(competitor_positions),
            "fixed_absolute_image_key_positions": image_positions,
            "active_image_key_positions": image_positions,
            "fixed_absolute_same_class_competitor_positions": competitor_positions,
            "active_same_class_competitor_positions": competitor_positions,
            "fixed_competitor_positions_sha256": sha256_json(competitor_positions),
            "factory_arm_id": "K13",
            "layer_consumption_attestation": layer,
            "all_layer_consumption_attestation": layer,
        }
    elif status == "not_applicable":
        nested = {
            "schema_version": K13_LAYER_RECEIPT_SCHEMA_VERSION,
            "required": True,
            "status": "unattested",
            "exact_same_tensor_all_layers_required": True,
        }
        attention = {
            "schema_version": K13_MASK_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "arm_id": "K13",
            "status": "not_applicable",
            "reason": K13_NOT_APPLICABLE_REASON,
            "image_key_positions": image_positions,
            "query_positions": [9],
            "selected_positions": competitor_positions,
            "query_positions_sha256": sha256_json([9]),
            "selected_positions_sha256": sha256_json(competitor_positions),
            "selected_key_count": 0,
            "fixed_absolute_image_key_positions": image_positions,
            "active_image_key_positions": image_positions,
            "fixed_absolute_same_class_competitor_positions": competitor_positions,
            "active_same_class_competitor_positions": competitor_positions,
            "fixed_competitor_positions_sha256": sha256_json(competitor_positions),
            "factory_arm_id": "K13",
            "layer_consumption_attestation": nested,
            "all_layer_consumption_attestation": nested,
        }
        layer = {"status": "not_applicable", "passed": True}
    else:
        raise AssertionError(status)
    return {
        "step": index,
        "use_cache": False,
        "attention_actuation_receipt": attention,
        "layer_consumption_attestation": layer,
    }


def _arm_result(
    arm: str,
    event: dict[str, object],
    *,
    kind: str,
    k13_status: str = "applicable",
) -> dict[str, object]:
    image_id = event["image_id"]
    target = event["event_id"]
    covered = event["owner_refs"]["covered_owner_ids"][0]
    other = f"gt:{image_id}:19"
    if kind == "target":
        specs = [("closure", OPENER, target)]
    elif kind == "other":
        specs = [("closure", OPENER, other)]
    elif kind == "covered":
        specs = [("closure", OPENER, covered)]
    elif kind == "unmatched":
        specs = [("closure", OPENER, None)]
    elif kind == "duplicate":
        specs = [("closure", OPENER, other), ("closure", OPENER, other)]
    elif kind == "native_stop":
        specs = [("native_stop", NATIVE_STOP, None)]
    elif kind == "invalid":
        specs = [("invalid", INVALID, None)]
    elif kind == "malformed":
        specs = [("over_continuation", INVALID, None)]
    else:
        raise AssertionError(kind)
    rows: list[dict[str, object]] = []
    seen_endpoint: set[str] = set()
    seen_before = {covered}
    parsed: list[dict[str, object]] = []
    for index, (status, first, owner_id) in enumerate(specs):
        owner_match = None
        strict = status == "closure" and owner_id is not None
        if status == "closure":
            owner_match = (
                {"status": "unique", "owner_id": owner_id, "physical_match": True, "source_specific": True}
                if strict
                else {"status": "unmatched", "owner_id": None, "physical_match": False, "source_specific": True}
            )
        duplicate = bool(strict and owner_id in seen_endpoint)
        covered_repeat = bool(strict and owner_id == covered)
        row = _row(index=index, status=status, first=first, owner_match=owner_match)
        parsed.append({
            "row": row,
            "owner_id": owner_id,
            "strict": strict,
            "duplicate": duplicate,
            "covered_repeat": covered_repeat,
        })
        if strict:
            seen_endpoint.add(owner_id)
        rows.append(row)
    parse = {
        "valid_rows": sum(item["row"]["status"] == "closure" for item in parsed),
        "duplicate_rows": sum(item["duplicate"] for item in parsed),
        "unmatched_rows": sum(item["row"]["status"] == "closure" and not item["strict"] for item in parsed),
        "ambiguous_rows": 0,
        "malformed_rows": sum(item["row"]["status"] == "over_continuation" for item in parsed),
        "invalid_rows": sum(item["row"]["status"] in {"invalid", "over_continuation", "max_budget"} for item in parsed),
    }
    terminal = specs[-1][0]
    stop = {"stopped": terminal != "closure", "stop_reason": terminal}
    row_entry = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": specs[0][1],
        "opener_generated_by_model": specs[0][1] == OPENER,
        "opener_injected": False,
        "row_started": specs[0][1] == OPENER,
    }
    for item in parsed:
        row = item["row"]
        owner_id = item["owner_id"]
        row["owner_bookkeeping"] = {
            "covered_owner_ids_before": [covered],
            "seen_owner_ids_before": sorted(seen_before),
            "matched_owner_id": owner_id,
            "strict_physical_owner_match": item["strict"],
            "covered_repeat": item["covered_repeat"],
            "duplicate": item["duplicate"],
            "new_target_owner": bool(item["strict"] and owner_id != covered),
            "horizon_status": terminal,
            "row_entry": row_entry,
            "parse": parse,
            "stop": stop,
        }
        if item["strict"]:
            seen_before.add(owner_id)
    strict_rows = [item["owner_id"] for item in parsed if item["strict"]]
    endpoint_ids = sorted(set(strict_rows))
    generated = [token for row in rows for token in row["token_ids"]]
    count = len(generated)
    runtime_receipts: list[dict[str, object]] = [
        {"step": index, "use_cache": False} for index in range(count)
    ]
    if arm == "K13":
        runtime_receipts = [_k13_runtime_receipt(index, k13_status) for index in range(count)]
    return {
        "arm_id": arm,
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "opener_token_id": OPENER,
        "initial_prefix_last_token_id": 9,
        "opener_generated_by_model": specs[0][1] == OPENER,
        "first_generated_token_id": specs[0][1],
        "native_stop_token_ids": [NATIVE_STOP],
        "no_cache_scalar_recompute": True,
        "rows": rows,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": sha256_json(generated),
        "scalar_forward_count": count,
        "scalar_receipts": [{"step": index, "use_cache": False} for index in range(count)],
        "runtime_scalar_forward_count": count,
        "runtime_scalar_receipts": runtime_receipts,
        "full_logit_parity": {
            "status": "reference_captured",
            "reference_arm": arm,
            "reference_step_count": count,
            "candidate_step_count": 0,
            "per_forward_max_abs_delta": None,
            "tolerance": 1e-4,
            "passed": True,
        },
        "terminal_reason": terminal,
        "owner_bookkeeping": {
            "raw_endpoint_owner_ids": endpoint_ids,
            "covered_repeat_owner_ids": sorted(set(endpoint_ids) & {covered}),
            "new_target_owner_ids": sorted(set(endpoint_ids) - {covered}),
            "row_entry": row_entry,
            "parse": parse,
            "stop": stop,
            "horizon_status": terminal,
            "row_count": len(rows),
            "strict_physical_owner_match_count": len(strict_rows),
            "duplicate_count": parse["duplicate_rows"],
            "unmatched_count": parse["unmatched_rows"],
        },
    }


def _executor_identity(shard_index: int, claim_scope: dict[str, object]) -> dict[str, object]:
    identity = _COHORT_FIXTURE._executor_identity(shard_index)
    pre_gpu = identity["pre_gpu"]
    assert isinstance(pre_gpu, dict)
    pre_gpu["claim_scope"] = copy.deepcopy(claim_scope)
    pre_gpu.pop("binding_sha256", None)
    pre_gpu["binding_sha256"] = sha256_json(pre_gpu)
    identity.pop("identity_sha256", None)
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _executor(
    *,
    shard_index: int,
    claim_scope: dict[str, object],
    k10_invalid: bool = False,
    k14b_target: bool = False,
    k13_statuses: tuple[str, ...] | None = None,
):
    def execute(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        kinds = {arm: "other" for arm in arm_order}
        kinds.update({
            "K10": "invalid" if k10_invalid else "target",
            "K11": "malformed",
            "K12": "unmatched",
            "K13": "duplicate",
            "K14T": "target",
            "K14B": "target" if k14b_target else "other",
            "N10": "invalid",
            "N20": "native_stop",
        })
        k13_status = "applicable" if k13_statuses is None else k13_statuses[event["event_index"]]
        arms = {
            arm: _arm_result(
                arm,
                event,
                kind=kinds[arm],
                k13_status=k13_status,
            )
            for arm in arm_order
        }
        identity = _executor_identity(shard_index, claim_scope)
        for result in arms.values():
            result["executor_identity"] = copy.deepcopy(identity)
        return {"arms": arms}
    return execute


def _build_run(
    tmp_path: Path,
    *,
    event_count: int = 4,
    image_count: int | None = None,
    noncanonical_gate: bool = False,
    k10_invalid: bool = False,
    k14b_target: bool = False,
    k13_statuses: tuple[str, ...] | None = None,
) -> dict[str, object]:
    resolved_image_count = event_count if image_count is None else image_count
    if resolved_image_count < 1 or resolved_image_count > event_count:
        raise ValueError("image_count must be between one and event_count")
    if k13_statuses is not None and (
        len(k13_statuses) != event_count
        or any(status not in {"applicable", "not_applicable"} for status in k13_statuses)
    ):
        raise ValueError("k13_statuses must provide one valid status per event")
    resolved_k13_statuses = k13_statuses or ("applicable",) * event_count
    census, census_path = _build_census(tmp_path, event_count, resolved_image_count)
    manifest, manifest_path = _build_manifest(
        tmp_path,
        census_path,
        event_count,
        resolved_image_count,
        resolved_k13_statuses,
    )
    gate, gate_path = _build_gate(tmp_path, noncanonical=noncanonical_gate)
    plan = build_plan(manifest_path, gate_path)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    shards_root = tmp_path / "shards"
    for index in range(8):
        executor = _executor(
            shard_index=index,
            claim_scope=plan["claim_scope"],
            k10_invalid=k10_invalid,
            k14b_target=k14b_target,
            k13_statuses=resolved_k13_statuses,
        )
        _run_shard_test_only(
            manifest_path,
            plan_path,
            shards_root / f"shard-{index:03d}",
            shard_id=index,
            executor=executor,
        )
    merged_root = tmp_path / "merged"
    _merge_shards_test_only(manifest_path, plan_path, shards_root, merged_root)
    return {
        "census": census,
        "census_path": census_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "gate": gate,
        "gate_path": gate_path,
        "plan": plan,
        "plan_path": plan_path,
        "shards_root": shards_root,
        "aggregate_path": merged_root / "aggregate.json",
    }


def _analyze(run: dict[str, object], **kwargs: object) -> dict[str, object]:
    return analyze_cohort(
        run["aggregate_path"],
        run["manifest_path"],
        run["census_path"],
        plan=run["plan_path"],
        gate_receipt=run["gate_path"],
        shards_root=run["shards_root"],
        **kwargs,
    )


def test_gate_binding_accepts_canonical_and_immutable_noncanonical_inputs_but_rejects_tamper(
    valid_run: dict[str, object],
    tmp_path: Path,
) -> None:
    canonical_binding = _validate_gate_binding(
        valid_run["gate_path"],
        validate_plan(valid_run["plan_path"]),
    )
    assert canonical_binding["raw_file_sha256"]

    noncanonical_run = _build_run(tmp_path / "noncanonical", noncanonical_gate=True)
    gate_path = noncanonical_run["gate_path"]
    runtime_path = gate_path.with_name("runtime_identity.json")
    gate_raw = gate_path.read_bytes()
    runtime_raw = runtime_path.read_bytes()
    plan_info = validate_plan(noncanonical_run["plan_path"])

    binding = _validate_gate_binding(gate_path, plan_info)
    assert binding["raw_file_sha256"] == hashlib.sha256(gate_raw).hexdigest()
    assert binding["runtime_identity"]["raw_file_sha256"] == hashlib.sha256(runtime_raw).hexdigest()

    gate_path.write_bytes(gate_raw + b" \n")
    with pytest.raises(EvidenceAnalysisError, match="raw/self binding"):
        _validate_gate_binding(gate_path, plan_info)
    gate_path.write_bytes(gate_raw)

    gate_document = json.loads(gate_raw)
    gate_document["semantic_tamper"] = True
    _write_noncanonical_json(gate_path, gate_document)
    with pytest.raises(EvidenceAnalysisError, match="result_sha256 mismatch"):
        _validate_gate_binding(gate_path, plan_info)
    gate_path.write_bytes(gate_raw)

    runtime_document = json.loads(runtime_raw)
    runtime_document["semantic_tamper"] = True
    _write_noncanonical_json(runtime_path, runtime_document)
    with pytest.raises(EvidenceAnalysisError, match="runtime identity self hash mismatch"):
        _validate_gate_binding(gate_path, plan_info)


@pytest.fixture(scope="module")
def valid_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _build_run(tmp_path_factory.mktemp("valid-cohort"))


def _delta_summary(
    *,
    arm_id: str,
    strict_owner_ids: list[str] | None = None,
    covered_repeat_owner_ids: list[str] | None = None,
    uncovered_owner_gain_ids: list[str] | None = None,
    first_token_class: str = "object_ref_start",
    terminal_reason: str = "closure",
    unmatched_rows: int = 0,
    ambiguous_rows: int = 0,
    duplicate_rows: int = 0,
    malformed_rows: int = 0,
    invalid_rows: int = 0,
    target_strict_release: bool = False,
) -> dict[str, object]:
    strict = list(strict_owner_ids or [])
    unmatched = int(unmatched_rows)
    return {
        "arm_id": arm_id,
        "first_token_class": first_token_class,
        "terminal_reason": terminal_reason,
        "complete_natural_rows": len(strict) + unmatched,
        "strict_physical_owner_match_count": len(strict),
        "strict_physical_owner_ids": strict,
        "unmatched_rows": unmatched,
        "duplicate_rows": int(duplicate_rows),
        "ambiguous_rows": int(ambiguous_rows),
        "malformed_rows": int(malformed_rows),
        "invalid_rows": int(invalid_rows),
        "target_strict_release": bool(target_strict_release),
        "covered_repeat_owner_ids": list(covered_repeat_owner_ids or []),
        "uncovered_owner_gain_ids": list(uncovered_owner_gain_ids or []),
    }


def _disposition(
    intervention: dict[str, object],
    baseline: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    delta = _delta(intervention, baseline)
    return delta, _dynamic_disposition(intervention, baseline, delta)


def test_dynamic_equal_count_strict_owner_substitution_qualifies() -> None:
    baseline = _delta_summary(
        arm_id="N01",
        strict_owner_ids=["owner-a"],
        uncovered_owner_gain_ids=["owner-a"],
    )
    intervention = _delta_summary(
        arm_id="N10",
        strict_owner_ids=["owner-b"],
        uncovered_owner_gain_ids=["owner-b"],
    )
    delta, disposition = _disposition(intervention, baseline)
    assert delta["strict_physical_owner_match_count_delta"] == 0
    assert delta["strict_owner_ids_added"] == ["owner-b"]
    assert delta["strict_owner_ids_removed"] == ["owner-a"]
    assert disposition["endpoint_change_reasons"] == [
        "strict_owner_ids_added",
        "strict_owner_ids_removed",
    ]
    assert disposition["endpoint_changed"] is True
    assert disposition["factor_qualified"] is True


def test_dynamic_equal_count_target_flip_qualifies() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10", target_strict_release=True)
    delta, disposition = _disposition(intervention, baseline)
    assert delta["strict_physical_owner_match_count_delta"] == 0
    assert delta["target_strict_release_delta"] == 1
    assert disposition["endpoint_change_reasons"] == ["target_strict_release_delta"]
    assert disposition["factor_qualified"] is True


def test_dynamic_identical_arms_have_no_endpoint_change() -> None:
    baseline = _delta_summary(arm_id="N01", strict_owner_ids=["owner-a"])
    intervention = _delta_summary(arm_id="N10", strict_owner_ids=["owner-a"])
    delta, disposition = _disposition(intervention, baseline)
    assert delta["strict_owner_ids_added"] == []
    assert delta["strict_owner_ids_removed"] == []
    assert disposition["endpoint_change_reasons"] == []
    assert disposition["endpoint_changed"] is False
    assert disposition["factor_qualified"] is False


def test_dynamic_degenerate_owner_loss_never_qualifies() -> None:
    baseline = _delta_summary(
        arm_id="N01",
        strict_owner_ids=["target"],
        uncovered_owner_gain_ids=["target"],
    )
    intervention = _delta_summary(
        arm_id="N10",
        first_token_class="other_invalid",
        terminal_reason="invalid",
    )
    _, disposition = _disposition(intervention, baseline)
    assert disposition["endpoint_changed"] is True
    assert disposition["degenerate_grammar_disruption"] is True
    assert disposition["factor_qualified"] is False
    assert disposition["crossover_candidate"] is False


def test_dynamic_ambiguous_only_delta_qualifies() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10", ambiguous_rows=1)
    delta, disposition = _disposition(intervention, baseline)
    assert delta["ambiguous_rows_delta"] == 1
    assert disposition["endpoint_change_reasons"] == ["ambiguous_rows_delta"]
    assert disposition["factor_qualified"] is True


def test_dynamic_unclassified_delta_fails_closed() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10")
    delta = _delta(intervention, baseline)
    delta["future_endpoint_metric_delta"] = 1
    with pytest.raises(EvidenceAnalysisError, match="not exactly partitioned"):
        _dynamic_disposition(intervention, baseline, delta)


def test_dynamic_complete_row_invariant_fails_closed() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10", strict_owner_ids=["owner-a"])
    delta = _delta(intervention, baseline)
    delta["complete_natural_rows_delta"] = 0
    with pytest.raises(EvidenceAnalysisError, match="complete_natural_rows_delta"):
        _dynamic_disposition(intervention, baseline, delta)


def test_dynamic_covered_repeat_outside_strict_delta_fails_closed() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10")
    delta = _delta(intervention, baseline)
    delta["covered_repeat_owner_ids_added"] = ["owner-not-strict"]
    with pytest.raises(EvidenceAnalysisError, match="covered_repeat_owner_ids_added"):
        _dynamic_disposition(intervention, baseline, delta)


def test_dynamic_uncovered_gain_outside_strict_delta_fails_closed() -> None:
    baseline = _delta_summary(arm_id="N01")
    intervention = _delta_summary(arm_id="N10")
    delta = _delta(intervention, baseline)
    delta["uncovered_owner_gain_ids_removed"] = ["owner-not-strict"]
    with pytest.raises(EvidenceAnalysisError, match="uncovered_owner_gain_ids_removed"):
        _dynamic_disposition(intervention, baseline, delta)


def test_formal_eight_shard_chain_and_scientific_outcomes(valid_run: dict[str, object], tmp_path: Path) -> None:
    result = _analyze(
        valid_run,
        output=tmp_path / "evidence.json",
        receipt_output=tmp_path / "evidence.receipt.json",
    )
    evidence = result["evidence"]
    first = evidence["events"][0]
    assert evidence["denominators"]["shards"] == 8
    assert evidence["denominators"]["census_checkpoint_owner_rows"] == {"S": 392, "A": 392}
    assert first["arms"]["K12"]["unmatched_rows"] == 1
    assert first["arms"]["K12"]["mechanically_valid"] is True
    assert first["arms"]["K13"]["duplicate_rows"] == 1
    assert first["arms"]["K11"]["malformed_rows"] == 1
    assert first["arms"]["N20"]["native_stop"] is True
    assert first["intervention_minus_own_baseline"]["K10"]["baseline_arm"] == "K01"
    assert first["intervention_minus_own_baseline"]["N10"]["baseline_arm"] == "N01"
    assert first["intervention_minus_own_baseline"]["H10"]["baseline_arm"] == "H00"
    assert evidence["input_bindings"]["merged"]["aggregate"]["raw_file_sha256"]
    assert evidence["input_bindings"]["events"][0]["result"]["semantic_self_sha256"]
    assert evidence["self_sha256"] == document_self_sha256(evidence)
    assert result["receipt"]["self_sha256"] == document_self_sha256(result["receipt"])
    assert all(result["receipt"]["next_step_flags"][key] is False for key in ("authorize_crossover", "authorize_a3", "authorize_p4", "authorize_training"))


def test_v3_schema_and_receipt_bind_dynamic_history_effects(valid_run: dict[str, object]) -> None:
    result = _analyze(valid_run)
    evidence = result["evidence"]
    receipt = result["receipt"]
    assert evidence["schema_version"] == SCHEMA_VERSION == "s_natural_boundary_k_n_h_evidence.v3"
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION == f"{SCHEMA_VERSION}.receipt"
    assert receipt["dynamic_history_effects"] == evidence["dynamic_history_effects"]
    assert evidence["self_sha256"] == document_self_sha256(evidence)
    assert receipt["self_sha256"] == document_self_sha256(receipt)


def test_evidence_and_receipt_bind_current_analyzer_producer_and_self_hashes(
    valid_run: dict[str, object],
) -> None:
    result = _analyze(valid_run)
    evidence = result["evidence"]
    receipt = result["receipt"]
    expected = _producer_ref()
    assert evidence["producer"] == expected
    assert receipt["producer"] == expected
    assert evidence["producer"] == receipt["producer"]
    producer_path = Path(expected["path"])
    assert producer_path.is_absolute()
    assert producer_path.is_file()
    assert not producer_path.is_symlink()
    assert expected["sha256"] == hashlib.sha256(producer_path.read_bytes()).hexdigest()
    assert expected["size_bytes"] == producer_path.stat().st_size
    assert evidence["self_sha256"] == document_self_sha256(evidence)
    assert receipt["self_sha256"] == document_self_sha256(receipt)

    tampered_evidence = copy.deepcopy(evidence)
    tampered_evidence["producer"]["size_bytes"] += 1
    assert tampered_evidence["self_sha256"] != document_self_sha256(tampered_evidence)
    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["producer"]["sha256"] = "f" * 64
    assert tampered_receipt["self_sha256"] != document_self_sha256(tampered_receipt)


def test_producer_ref_rejects_symlinked_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "analyzer.py"
    target.write_text("# target\n", encoding="utf-8")
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.symlink_to(target)
    monkeypatch.setattr(analyzer_module, "__file__", str(entrypoint))
    with pytest.raises(EvidenceAnalysisError, match="producer is not a regular non-symlink file"):
        _producer_ref()


def test_k13_applicable_retains_delta_and_measured_denominator(valid_run: dict[str, object]) -> None:
    evidence = _analyze(valid_run)["evidence"]
    first = evidence["events"][0]
    assert first["k13_applicability"] == {
        "status": "applicable",
        "reason": None,
        "measured": True,
        "excluded_from_effect_metrics": False,
        "executed_forward_count": 2,
    }
    assert first["intervention_minus_own_baseline"]["K13"]["status"] == "measured"
    assert evidence["denominators"]["k13_applicability"] == {
        "observed_event_count": 4,
        "measured_event_count": 4,
        "not_applicable_event_count": 0,
        "not_applicable_reasons": {},
    }
    assert evidence["denominators"]["arm_event_denominators"]["K13"]["event_count"] == 4


def test_k13_all_not_applicable_is_reported_and_excluded_from_effects(tmp_path: Path) -> None:
    run = _build_run(
        tmp_path,
        k13_statuses=("not_applicable",) * 4,
    )
    evidence = _analyze(run)["evidence"]
    assert {
        event["k13_applicability"]["status"] for event in evidence["events"]
    } == {"not_applicable"}
    assert all("K13" not in event["intervention_minus_own_baseline"] for event in evidence["events"])
    k13 = evidence["denominators"]["arm_event_denominators"]["K13"]
    assert k13["event_count"] == 0
    assert k13["measured_event_count"] == 0
    assert k13["observed_event_count"] == 4
    assert k13["not_applicable_event_count"] == 4
    assert k13["not_applicable_reasons"] == {
        K13_NOT_APPLICABLE_REASON: 4,
    }
    assert evidence["denominators"]["k13_applicability"]["not_applicable_event_count"] == 4
    # The trajectory remains present and classified; N/A is not relabeled as
    # a scientific unmatched, invalid, duplicate, or STOP outcome.
    assert all(event["arms"]["K13"]["duplicate_rows"] == 1 for event in evidence["events"])


def test_k13_mixed_applicability_measures_only_applicable_events(tmp_path: Path) -> None:
    run = _build_run(
        tmp_path,
        k13_statuses=("applicable", "not_applicable", "applicable", "not_applicable"),
    )
    evidence = _analyze(run)["evidence"]
    assert [event["k13_applicability"]["status"] for event in evidence["events"]] == [
        "applicable",
        "not_applicable",
        "applicable",
        "not_applicable",
    ]
    assert ["K13" in event["intervention_minus_own_baseline"] for event in evidence["events"]] == [
        True,
        False,
        True,
        False,
    ]
    k13 = evidence["denominators"]["arm_event_denominators"]["K13"]
    assert k13["event_count"] == 2
    assert k13["measured_event_count"] == 2
    assert k13["not_applicable_event_count"] == 2


def test_k13_conflicting_executed_receipts_fail_closed(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    arm = _arm_result("K13", event, kind="duplicate")
    arm["runtime_scalar_receipts"][1]["layer_consumption_attestation"] = {
        "status": "not_applicable",
        "passed": True,
    }
    with pytest.raises(EvidenceAnalysisError, match="applicable attestation is malformed|schema is malformed|conflicting"):
        _k13_applicability(arm, event)


def test_k13_not_applicable_construction_stub_is_exactly_four_keys(valid_run: dict[str, object]) -> None:
    event_id = valid_run["manifest"]["events"][0]["event_id"]
    valid = {
        "schema_version": K13_LAYER_RECEIPT_SCHEMA_VERSION,
        "required": True,
        "status": "unattested",
        "exact_same_tensor_all_layers_required": True,
    }
    observed = _k13_attestation(
        valid,
        event_id=event_id,
        label="construction",
        not_applicable=True,
    )
    assert observed == valid
    assert "unit_id" not in observed

    # The construction stub has no unit binding.  Adding even a null or
    # otherwise foreign/correct unit is a producer-shape mutation.
    for unit_id in (UNIT_ID, "foreign-unit", None):
        mutated = copy.deepcopy(valid)
        mutated["unit_id"] = unit_id
        with pytest.raises(EvidenceAnalysisError, match="not_applicable layer attestation is malformed"):
            _k13_attestation(
                mutated,
                event_id=event_id,
                label="construction",
                not_applicable=True,
            )

    for field, replacement in (
        ("schema_version", "foreign-schema"),
        ("required", False),
        ("status", "ready"),
        ("exact_same_tensor_all_layers_required", False),
    ):
        mutated = copy.deepcopy(valid)
        mutated[field] = replacement
        with pytest.raises(EvidenceAnalysisError):
            _k13_attestation(
                mutated,
                event_id=event_id,
                label="construction",
                not_applicable=True,
            )
    mutated = copy.deepcopy(valid)
    mutated["extra"] = True
    with pytest.raises(EvidenceAnalysisError, match="not_applicable layer attestation is malformed"):
        _k13_attestation(
            mutated,
            event_id=event_id,
            label="construction",
            not_applicable=True,
        )


@pytest.mark.parametrize("location", ("direct", "nested"))
def test_k13_applicable_layer_attestation_requires_unit_id(
    valid_run: dict[str, object],
    location: str,
) -> None:
    event = copy.deepcopy(valid_run["manifest"]["events"][0])
    arm = _arm_result("K13", event, kind="duplicate", k13_status="applicable")
    receipt = arm["runtime_scalar_receipts"][0]
    if location == "direct":
        receipt["layer_consumption_attestation"].pop("unit_id")
    else:
        receipt["attention_actuation_receipt"]["layer_consumption_attestation"].pop("unit_id")
    with pytest.raises(EvidenceAnalysisError, match="layer attestation unit is foreign"):
        _k13_applicability(arm, event)


def _set_k13_event_geometry(event: dict[str, object], status: str) -> None:
    applicable = status == "applicable"
    cells = [1] if applicable else []
    weights = [{"cell_index": 1, "overlap_fraction": 1.0}] if applicable else []
    event["same_class_competitor_owner_id"] = "gt:5001:20" if applicable else None
    event["image_cell_regions"] = {"same_class_competitor": cells}
    event["image_cell_region_receipts"] = {
        "same_class_competitor": {
            "status": "available" if applicable else "not_applicable",
            "available": applicable,
            "not_measured_reason": None if applicable else "no_verified_same_class_competitor",
            "cell_indices": cells,
            "visual_indices": cells,
            "fractional_weights": weights,
            "weight_sum": 1.0 if applicable else 0.0,
            "cell_count": len(cells),
            "cell_indices_sha256": sha256_json(cells),
            "weights_sha256": sha256_json(weights),
        }
    }


def test_k13_receipts_cannot_cross_event_or_arm_geometry(valid_run: dict[str, object]) -> None:
    app_event = copy.deepcopy(valid_run["manifest"]["events"][0])
    app_arm = _arm_result("K13", app_event, kind="duplicate", k13_status="applicable")
    app_receipt = app_arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"]
    app_receipt["arm_id"] = "K12"
    with pytest.raises(EvidenceAnalysisError, match="identity is foreign"):
        _k13_applicability(app_arm, app_event)

    app_arm = _arm_result("K13", app_event, kind="duplicate", k13_status="applicable")
    app_arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"]["factory_arm_id"] = "K12"
    with pytest.raises(EvidenceAnalysisError, match="factory identity is foreign"):
        _k13_applicability(app_arm, app_event)

    na_event = copy.deepcopy(app_event)
    _set_k13_event_geometry(na_event, "not_applicable")
    na_arm = _arm_result("K13", na_event, kind="duplicate", k13_status="not_applicable")
    na_receipt = na_arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"]
    na_receipt["fixed_absolute_same_class_competitor_positions"] = [1]
    na_receipt["fixed_competitor_positions_sha256"] = sha256_json([1])
    with pytest.raises(EvidenceAnalysisError, match="not_applicable competitor positions are nonempty"):
        _k13_applicability(na_arm, na_event)

    na_arm = _arm_result("K13", na_event, kind="duplicate", k13_status="not_applicable")
    na_arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"]["reason"] = "image keys unavailable"
    with pytest.raises(EvidenceAnalysisError, match="reason is non-canonical"):
        _k13_applicability(na_arm, na_event)

    with pytest.raises(EvidenceAnalysisError, match="status disagrees"):
        _k13_applicability(
            _arm_result("K13", app_event, kind="duplicate", k13_status="not_applicable"),
            app_event,
        )
    with pytest.raises(EvidenceAnalysisError, match="status disagrees"):
        _k13_applicability(
            _arm_result("K13", na_event, kind="duplicate", k13_status="applicable"),
            na_event,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("unit_id", "foreign-unit", "identity is foreign"),
        ("schema_version", "foreign-schema", "schema is foreign"),
        ("fixed_competitor_positions_sha256", "f" * 64, "positions or hashes are malformed"),
    ),
)
def test_k13_missing_or_foreign_live_receipt_binding_fails_closed(
    valid_run: dict[str, object],
    field: str,
    replacement: str,
    message: str,
) -> None:
    event = copy.deepcopy(valid_run["manifest"]["events"][0])
    arm = _arm_result("K13", event, kind="duplicate", k13_status="applicable")
    arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"][field] = replacement
    with pytest.raises(EvidenceAnalysisError, match=message):
        _k13_applicability(arm, event)


@pytest.mark.parametrize("field", ("unit_id", "schema_version", "fixed_competitor_positions_sha256"))
def test_k13_missing_live_receipt_binding_fails_closed(
    valid_run: dict[str, object],
    field: str,
) -> None:
    event = copy.deepcopy(valid_run["manifest"]["events"][0])
    arm = _arm_result("K13", event, kind="duplicate", k13_status="applicable")
    arm["runtime_scalar_receipts"][0]["attention_actuation_receipt"].pop(field)
    with pytest.raises(EvidenceAnalysisError):
        _k13_applicability(arm, event)


def test_k10_case_replication_and_checkpoint_floors_are_separate(valid_run: dict[str, object]) -> None:
    evidence = _analyze(valid_run)["evidence"]
    k10 = evidence["qualification"]["K10_hard_oracle"]
    assert k10["case_level_event_count"] == 4
    assert k10["hard_replication_floor"] == {"minimum_events": 2, "minimum_images": 2}
    assert k10["hard_replication_status"] == "qualified"
    assert k10["checkpoint_floor"] == {"minimum_events": 3, "minimum_images": 2}
    assert k10["checkpoint_status"] == "qualified"


@pytest.mark.parametrize(
    ("event_count", "image_count", "hard_replication_status"),
    ((2, 2, "qualified"), (3, 1, "hold"), (2, 1, "hold")),
)
def test_case_study_scope_analyzes_all_events_but_forces_claims_hold(
    tmp_path: Path,
    event_count: int,
    image_count: int,
    hard_replication_status: str,
) -> None:
    result = _analyze(
        _build_run(tmp_path, event_count=event_count, image_count=image_count)
    )
    evidence = result["evidence"]
    scope = evidence["claim_scope"]
    assert scope == {
        "execution_scope": "case_study",
        "event_count": event_count,
        "image_count": image_count,
        "minimum_checkpoint_event_count": 3,
        "minimum_checkpoint_image_count": 2,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": True,
    }
    assert len(evidence["events"]) == event_count
    assert evidence["claim_decision"] == {
        "execution_scope": "case_study",
        "analyzed_event_count": event_count,
        "all_case_events_analyzed": True,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "checkpoint_claim_status": "hold",
        "static_direction_claim_status": "hold",
        "training_claim_status": "hold",
    }
    k10 = evidence["qualification"]["K10_hard_oracle"]
    assert k10["case_level_event_count"] == event_count
    assert k10["case_level_status"] == "qualified"
    assert k10["hard_replication_status"] == hard_replication_status
    assert k10["checkpoint_status"] == "hold"
    assert evidence["qualification"]["checkpoint_level_status"] == "hold"
    assert evidence["qualification"]["static_direction_status"] == "hold"
    assert evidence["qualification"]["training_status"] == "hold"
    assert result["receipt"]["claim_scope"] == scope
    assert result["receipt"]["claim_decision"] == evidence["claim_decision"]
    assert evidence["dynamic_history_effects"]["N10"]["factor_qualified_event_count"] == 0


def test_four_invalid_k10_events_never_qualify(tmp_path: Path) -> None:
    evidence = _analyze(_build_run(tmp_path, k10_invalid=True))["evidence"]
    k10 = evidence["qualification"]["K10_hard_oracle"]
    assert k10["case_level_event_count"] == 0
    assert k10["hard_replication_status"] == "hold"
    assert k10["checkpoint_status"] == "hold"


def test_k14_target_background_control_cannot_qualify(tmp_path: Path) -> None:
    evidence = _analyze(_build_run(tmp_path, k14b_target=True))["evidence"]
    k14 = evidence["qualification"]["K14_finite_salience"]
    assert k14["qualifying_event_count"] == 0
    assert {row["K14B_control_status"] for row in k14["events"]} == {"target_B"}
    assert k14["checkpoint_status"] == "hold"


def test_invalid_first_token_history_effect_is_degenerate_not_factor(valid_run: dict[str, object]) -> None:
    evidence = _analyze(valid_run)["evidence"]
    first = evidence["events"][0]["dynamic_factor_disposition"]["N10"]
    assert first["endpoint_changed"] is True
    assert first["degenerate_grammar_disruption"] is True
    assert first["factor_qualified"] is False
    assert first["crossover_candidate"] is False
    assert evidence["dynamic_history_effects"]["N10"]["factor_qualified_event_count"] == 0


def test_wrong_opener_provenance_is_rejected_before_classification(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    arm = _arm_result("K10", event, kind="target")
    arm["opener_generated_by_model"] = False
    with pytest.raises(EvidenceAnalysisError, match="opener_generated_by_model disagrees"):
        _validate_natural_receipts(arm, "K10")
    arm = _arm_result("K10", event, kind="target")
    arm["rows"][0]["initial_prefix_last_token_id"] = 8
    with pytest.raises(EvidenceAnalysisError, match="arm/first-row natural-boundary receipts disagree"):
        _validate_natural_receipts(arm, "K10")


def test_missing_row_synthetic_injection_count_defaults_to_zero_but_nonzero_is_rejected(
    valid_run: dict[str, object],
) -> None:
    event = valid_run["manifest"]["events"][0]
    arm = _arm_result("K10", event, kind="target")
    arm["rows"][0].pop("synthetic_opener_injections")
    assert _validate_natural_receipts(arm, "K10")["first_token_class"] == "object_ref_start"

    for value in (1, -1, None, "0", False, 0.0):
        arm = _arm_result("K10", event, kind="target")
        arm["rows"][0]["synthetic_opener_injections"] = value
        with pytest.raises(EvidenceAnalysisError, match="not exact natural pre-opener release"):
            _validate_natural_receipts(arm, "K10")


def test_malformed_parse_count_is_not_coerced_or_maxed(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    arm = _arm_result("K12", event, kind="unmatched")
    arm["owner_bookkeeping"]["parse"]["unmatched_rows"] = 0
    with pytest.raises(EvidenceAnalysisError, match="formal arm validation failed|parse counts differ"):
        _summarize_arm(
            arm,
            arm_id="K12",
            event=event,
            census_info={
                "by_key": {
                    ("S", row["gt_owner_id"]): row
                    for row in valid_run["census"]["rows"]
                    if row["checkpoint"] == "S"
                }
            },
            claim_scope=valid_run["plan"]["claim_scope"],
        )


def test_closure_owner_match_status_must_match_receipt(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    arm = _arm_result("K10", event, kind="target")
    arm["executor_identity"] = _executor_identity(0, valid_run["plan"]["claim_scope"])
    arm["rows"][0]["owner_match_status"] = "unmatched"
    with pytest.raises(EvidenceAnalysisError, match="owner_match_status differs"):
        _summarize_arm(
            arm,
            arm_id="K10",
            event=event,
            census_info={
                "by_key": {
                    ("S", row["gt_owner_id"]): row
                    for row in valid_run["census"]["rows"]
                    if row["checkpoint"] == "S"
                }
            },
            claim_scope=valid_run["plan"]["claim_scope"],
        )


def test_arm_pre_gpu_claim_scope_must_match_manifest_scope(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    expected_scope = valid_run["plan"]["claim_scope"]
    foreign_scope = copy.deepcopy(expected_scope)
    foreign_scope["event_count"] = 3
    arm = _arm_result("K10", event, kind="target")
    arm["executor_identity"] = _executor_identity(0, foreign_scope)
    with pytest.raises(EvidenceAnalysisError, match="pre-GPU claim_scope differs"):
        _summarize_arm(
            arm,
            arm_id="K10",
            event=event,
            census_info={
                "by_key": {
                    ("S", row["gt_owner_id"]): row
                    for row in valid_run["census"]["rows"]
                    if row["checkpoint"] == "S"
                }
            },
            claim_scope=expected_scope,
        )


def test_merged_claim_scope_must_match_manifest_scope(
    valid_run: dict[str, object],
    tmp_path: Path,
) -> None:
    source_root = valid_run["aggregate_path"].parent
    aggregate = json.loads((source_root / "aggregate.json").read_text(encoding="utf-8"))
    receipt = json.loads((source_root / "aggregate.receipt.json").read_text(encoding="utf-8"))
    foreign_scope = copy.deepcopy(aggregate["claim_scope"])
    foreign_scope["event_count"] = 3
    aggregate["claim_scope"] = foreign_scope
    aggregate["execution_qualification"] = copy.deepcopy(foreign_scope)
    aggregate["aggregate_sha256"] = document_self_sha256(aggregate, "aggregate_sha256")
    receipt["claim_scope"] = copy.deepcopy(foreign_scope)
    receipt["execution_qualification"] = copy.deepcopy(foreign_scope)
    receipt["aggregate_sha256"] = aggregate["aggregate_sha256"]
    receipt["receipt_sha256"] = document_self_sha256(receipt, "receipt_sha256")
    foreign_root = tmp_path / "foreign-merged"
    _write_json(foreign_root / "aggregate.json", aggregate)
    _write_json(foreign_root / "aggregate.receipt.json", receipt)
    run = dict(valid_run)
    run["aggregate_path"] = foreign_root / "aggregate.json"
    with pytest.raises(EvidenceAnalysisError, match="claim_scope"):
        _analyze(run)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("manifest_sha256", None),
        ("manifest_self_sha256", "f" * 64),
        ("source_census_sha256", None),
        ("source_census_sha256", "e" * 64),
    ),
)
def test_result_missing_or_foreign_manifest_census_binding_fails(
    valid_run: dict[str, object],
    field: str,
    replacement: object,
) -> None:
    event = valid_run["manifest"]["events"][0]
    result_path = next(valid_run["shards_root"].glob("shard-*/event-000000-*/result.json"))
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document[field] = replacement
    document["result_sha256"] = document_self_sha256(document, "result_sha256")
    with pytest.raises(EvidenceAnalysisError, match="manifest/census binding is missing or foreign"):
        _validate_result_bindings(
            document,
            ref={"result_sha256": document["result_sha256"]},
            event=event,
            manifest_info={
                "manifest_sha256": hashlib.sha256(valid_run["manifest_path"].read_bytes()).hexdigest(),
                "manifest_self_sha256": valid_run["manifest"]["self_sha256"],
            },
            census_info={"raw_file_sha256": hashlib.sha256(valid_run["census_path"].read_bytes()).hexdigest()},
        )


def test_missing_result_ref_hash_and_self_comparison_are_rejected(valid_run: dict[str, object]) -> None:
    event = valid_run["manifest"]["events"][0]
    result_path = next(valid_run["shards_root"].glob("shard-*/event-000000-*/result.json"))
    document = json.loads(result_path.read_text(encoding="utf-8"))
    kwargs = {
        "document": document,
        "event": event,
        "manifest_info": {
            "manifest_sha256": document["manifest_sha256"],
            "manifest_self_sha256": document["manifest_self_sha256"],
        },
        "census_info": {"raw_file_sha256": document["source_census_sha256"]},
    }
    with pytest.raises(EvidenceAnalysisError, match="reference hash is missing"):
        _validate_result_bindings(ref={}, **kwargs)
    foreign = copy.deepcopy(document)
    foreign["result_sha256"] = "a" * 64
    with pytest.raises(EvidenceAnalysisError, match="reference/self hash mismatch"):
        _validate_result_bindings(ref={"result_sha256": "a" * 64}, document=foreign, event=event, manifest_info=kwargs["manifest_info"], census_info=kwargs["census_info"])


def test_output_rejects_symlink_and_dangling_symlink(valid_run: dict[str, object], tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("foreign\n", encoding="utf-8")
    symlink = tmp_path / "evidence.json"
    symlink.symlink_to(target)
    with pytest.raises(EvidenceAnalysisError, match="collision or symlink"):
        _analyze(valid_run, output=symlink)
    symlink.unlink()
    symlink.symlink_to(tmp_path / "missing.json")
    with pytest.raises(EvidenceAnalysisError, match="collision or symlink"):
        _analyze(valid_run, output=symlink)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EvidenceAnalysisError, match="resolves through a symlink"):
        _analyze(valid_run, output=symlink_parent / "evidence.json")
