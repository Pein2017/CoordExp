from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.research.aggregate_static_dynamic_owner_interface_shards import (
    AggregationError,
    EligibleHoldSpec,
    ELIGIBLE_HOLD_SCHEMA_VERSION,
    _build_evidence_readiness,
    _collect_h0_baselines,
    _derive_bundle_event,
    _event_pair_contract,
    _load_census_sources,
    _load_cohort,
    _load_support_sources,
    _p1_bundle_rows,
    _read_payload,
    _validate_runtime_attestation,
    _validate_bundle_partition,
    aggregate_shards,
    _crossover_observation,
    _horizon_qualification,
    _p4_qualification,
    build_evidence_bundle,
    _p1_observations,
    parse_cohort_selector,
    parse_eligible_hold_selector,
    parse_shard_selector,
    sha256_json,
)


UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
RUNTIME_SCHEMA = "static_dynamic_owner_interface_experiment.v1"
GRADIENT_SCHEMA = "static_dynamic_gradient_path_audit.v2"
RUNTIME_ATTESTATION_SCHEMA = f"{RUNTIME_SCHEMA}.runtime_attestation.v1"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _runtime_attestation() -> dict[str, object]:
    visible = {"raw": "5", "tokens": ["5"], "selected_physical_device": "5"}
    support_runtime = {
        "status": "validated",
        "passed": True,
        "device": "cuda:0",
        "effective_device": "cuda:0",
        "normalized_device": "cuda:0",
        "torch_current_device": "cuda:0",
        "cuda_visible_devices": visible,
        "physical_device_id": "5",
    }
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA,
        "status": "validated",
        "passed": True,
        "device": "cuda:0",
        "effective_device": "cuda:0",
        "normalized_device": "cuda:0",
        "logical_selected_device": "cuda:0",
        "model_device": "cuda:0",
        "torch_current_device": "cuda:0",
        "first_parameter_device": "cuda:0",
        "cuda_visible_devices": visible,
        "physical_device_id": "5",
        "physical_device_index": 5,
        "physical_device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "physical_device_uuid_normalized": "01234567-89ab-cdef-0123-456789abcdef",
        "torch_device_uuid_raw": "01234567-89ab-cdef-0123-456789abcdef",
        "pid": 1234,
        "timestamp_utc": "2026-08-05T00:00:00Z",
        "support_runtime_identity": support_runtime,
    }


def test_runtime_attestation_requires_exact_physical_and_logical_identity() -> None:
    attestation = _runtime_attestation()
    _validate_runtime_attestation(attestation, context="runtime")
    missing = dict(attestation)
    del missing["physical_device_uuid"]
    with pytest.raises(AggregationError, match="missing field"):
        _validate_runtime_attestation(missing, context="runtime")
    multi = dict(attestation)
    multi["cuda_visible_devices"] = {"raw": "5,6", "tokens": ["5", "6"], "selected_physical_device": "5"}
    with pytest.raises(AggregationError, match="exactly one token"):
        _validate_runtime_attestation(multi, context="runtime")
    logical = dict(attestation)
    logical["model_device"] = "cuda:1"
    with pytest.raises(AggregationError, match="disagrees with normalized"):
        _validate_runtime_attestation(logical, context="runtime")
    physical = dict(attestation)
    physical["physical_device_id"] = "4"
    with pytest.raises(AggregationError, match="physical_device_id"):
        _validate_runtime_attestation(physical, context="runtime")
    malformed_uuid = dict(attestation)
    malformed_uuid["physical_device_uuid"] = "uuid"
    with pytest.raises(AggregationError, match="physical_device_uuid"):
        _validate_runtime_attestation(malformed_uuid, context="runtime")
    malformed_timestamp = dict(attestation)
    malformed_timestamp["timestamp_utc"] = "not-a-timestamp"
    with pytest.raises(AggregationError, match="ISO-8601"):
        _validate_runtime_attestation(malformed_timestamp, context="runtime")
    non_utc_timestamp = dict(attestation)
    non_utc_timestamp["timestamp_utc"] = "2026-08-05T00:00:00+01:00"
    with pytest.raises(AggregationError, match="timezone UTC"):
        _validate_runtime_attestation(non_utc_timestamp, context="runtime")


def _owner_result(owner_id: str, *, status: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "native_parse": {"valid": True, "parse_status": "accepted"},
        "owner_match": {"status": "unique", "owner_id": owner_id},
        "generation_status": "complete",
        "complete_row": True,
        "stop_reason": "box_end",
    }
    if status is not None:
        result["status"] = status
    return result


def _bookkeeping(owner_id: str, net: int) -> dict[str, object]:
    gained = [f"gt:gained:{owner_id}"] if net > 0 else []
    lost = [f"gt:lost:{owner_id}"] if net < 0 else []
    return {
        "status": "completed",
        "owner_bookkeeping": {
            "G": gained,
            "K": [owner_id],
            "L": lost,
            "net": len(gained) - len(lost),
            "repeat_hazard": {"t+1": 0},
            "parse": {"valid_rows": 1, "duplicate_rows": 0, "unmatched_rows": 0, "ambiguous_rows": 0, "malformed_rows": 0, "invalid_rows": 0},
        },
    }


def _endpoint_row(
    owner_id: str,
    *,
    checkpoint: str = "S",
    image_id: int = 1,
    target_owner: str = "gt:1:2",
    duplicate: bool = False,
    at_stop: bool = False,
    remaining_owner_ids: list[str] | None = None,
    source_specific: bool = True,
    physical_match: bool = True,
) -> dict[str, object]:
    remaining = ["gt:1:3"] if remaining_owner_ids is None else remaining_owner_ids
    h0_hash = sha256_json([])
    natural_hash = sha256_json([151646])
    return {
        "generation_status": "complete",
        "complete_row": True,
        "stop_reason": "box_end",
        "native_parse": {"valid": True, "parse_status": "accepted"},
        "owner_match": {
            "status": "unique",
            "owner_id": owner_id,
            "source_specific": source_specific,
            "physical_match": physical_match,
        },
        "endpoint_evidence": {
            "status": "measured",
            "natural": True,
            "teacher_forced": False,
            "strict_native_endpoint": {
                "status": "accepted",
                "native_parse_status": "accepted",
                "source_specific_physical_owner_match": source_specific and physical_match,
                "owner_match_status": "unique",
                "owner_id": owner_id,
                "target_owner_id": target_owner,
            },
            "outcome": {
                "valid_row": True,
                "duplicate": duplicate,
                "unmatched": False,
                "ambiguous": False,
                "malformed": False,
                "generated_token_count": 8,
            },
            "remaining_independently_verified_support_at_stop": {
                "status": "measured",
                "owner_ids": remaining,
                "count": len(remaining),
                "at_stop": at_stop,
            },
            "identity_binding": {
                "event_id": target_owner,
                "image_id": image_id,
                "checkpoint": checkpoint,
                "natural_prefix_sha256": natural_hash,
                "h0_exact_prefix_sha256": h0_hash,
            },
        },
    }


def _strict_horizon(
    owners: list[str],
    *,
    requested: int,
    covered: tuple[str, ...] = (),
    checkpoint: str = "S",
    image_id: int = 1,
    target_owner: str = "gt:1:2",
    stopped: bool = False,
) -> dict[str, object]:
    seen = set(covered)
    rows: list[dict[str, object]] = []
    bookkeeping_rows: list[dict[str, object]] = []
    duplicates: list[bool] = []
    for index, owner in enumerate(owners):
        duplicate = owner in seen
        duplicates.append(duplicate)
        seen.add(owner)
        row = _endpoint_row(
            owner,
            checkpoint=checkpoint,
            image_id=image_id,
            target_owner=target_owner,
            duplicate=duplicate,
            at_stop=stopped and index == len(owners) - 1,
            remaining_owner_ids=[] if stopped and index == len(owners) - 1 else ["gt:1:3"],
        )
        rows.append(row)
        bookkeeping_rows.append(
            {
                "complete": True,
                "parse_status": "accepted",
                "owner_id": owner,
                "duplicate": duplicate,
                "unmatched": False,
                "ambiguous": False,
                "malformed": False,
                "invalid": False,
            }
        )
    arm = set(owners)
    base = set(covered)
    gained, retained, lost = sorted(arm - base), sorted(arm & base), sorted(base - arm)
    repeat = {
        f"t+{offset}": int(offset <= len(owners) and owners[offset - 1] in base)
        for offset in range(1, 4)
    }
    complete = len(owners) >= requested
    return {
        "status": "completed" if complete else "stopped_early",
        "rows": rows,
        "row_count": len(rows),
        "owner_bookkeeping": {
            "rows": bookkeeping_rows,
            "horizon_requested": requested,
            "horizon_rows_generated": len(rows),
            "horizon_complete": complete,
            "G": gained,
            "K": retained,
            "L": lost,
            "net": len(gained) - len(lost),
            "repeat_hazard": repeat,
            "parse": {
                "valid_rows": len(rows),
                "duplicate_rows": sum(duplicates),
                "unmatched_rows": 0,
                "ambiguous_rows": 0,
                "malformed_rows": 0,
                "invalid_rows": 0,
            },
            "stop": {
                "stopped": stopped,
                "stop_reason": "terminal" if stopped else "horizon_exhausted",
            },
        },
    }


def _gradient_receipt() -> dict[str, object]:
    def objective(value: float) -> dict[str, object]:
        return {
            "value": value,
            "finite": True,
            "target_control_ratio": 2.0,
            "gradients": {
                "image_residual": {"present": True, "finite": True, "norm": 1.0, "max_abs": 0.5},
                "matched_background": {"present": True, "finite": True, "norm": 0.5, "max_abs": 0.25},
                "latest_terminal_carrier": {"present": True, "finite": True, "norm": 1.0, "max_abs": 0.5},
                "latest_row_span": {"present": True, "finite": True, "norm": 0.5, "max_abs": 0.25},
            },
        }

    objectives = {
        "target_b_complete_row_nll": objective(3.0),
        "uncovered_b_vs_covered_a_margin_loss": objective(1.0),
        "fixed_sum_coupled": objective(4.0),
    }
    return {
        "schema_version": GRADIENT_SCHEMA,
        "status": "valid",
        "objectives": objectives,
        "lm_head": {
            name: {"present": True, "finite": True, "norm": 1.0}
            for name in objectives
        },
        "non_target_owner_effects": {
            name: {
                "status": "measured",
                "owner_count": 1,
                "owners": {
                    "gt:non-target": {
                        "present": True,
                        "finite": True,
                        "norm": 0.25,
                        "max_abs": 0.1,
                    }
                },
            }
            for name in objectives
        },
        "path_checks": {
            "optimizer_used": False,
            "lm_head_only_path": False,
            "model_parameter_mutated": False,
            "parameter_grad_mutated": False,
            "audit_input_mutated": False,
            "visual_state_detached": [],
        },
    }


def _event(
    event_id: str,
    image_id: int,
    *,
    checkpoint: str = "S",
    ledger_path: Path = Path("/tmp/static-dynamic-test-ledger.json"),
    ledger_record_index: int = 0,
    net_shift: int = 0,
) -> dict[str, object]:
    mapping_rows = [
        {
            "owner_id": f"gt:{image_id}:0",
            "source_index": 0,
            "derived_index": 0,
            "coco_ann_id": 1,
            "mapping_method": "coco_ann_id",
        }
    ]
    baseline = f"gt:base:{event_id}"
    p1_arms = {probe: _owner_result(baseline) for probe in ("K00", "K01", "K10", "K11", "K12", "K13")}
    p1_arms["K10"] = _owner_result(f"gt:target:{event_id}")
    for block in (13, 23):
        for arm in ("R00", "R10", "R11", "R12"):
            p1_arms[f"{arm}_block{block}"] = _owner_result(baseline)
    for arm in ("R00", "R10"):
        p1_arms[f"{arm}_block27"] = _owner_result(baseline)
    p2 = {arm: {"horizon_1": _bookkeeping(baseline, net_shift), "horizon_3": _bookkeeping(baseline, net_shift)} for arm in ("D00", "D01", "D10", "D11", "D12", "D20")}
    p2["D21"] = {"status": "not_applicable", "reason": "same-parent donor is unavailable"}
    p3 = {cell: {"horizon_1": _bookkeeping(baseline, 0), "horizon_3": _bookkeeping(baseline, 0)} for cell in ("Y00", "Y10", "Y01", "Y11")}
    p3["Y10"]["horizon_1"] = _bookkeeping(baseline, 1)
    p3["Y10"]["horizon_3"] = _bookkeeping(baseline, 1)
    p3["Y01"]["horizon_1"] = _bookkeeping(baseline, -1)
    p3["Y01"]["horizon_3"] = _bookkeeping(baseline, -1)
    p3["Y11"]["horizon_1"] = _bookkeeping(baseline, 2)
    p3["Y11"]["horizon_3"] = _bookkeeping(baseline, 2)
    receipt = _gradient_receipt()
    return {
        "event_id": event_id,
        "image_id": image_id,
        "checkpoint": "CHECKPOINT_SET_BY_FIXTURE",
        "runtime_attestation": _runtime_attestation(),
        "prefix": {
            "target_owner_id": event_id,
            "natural_boundary": 0,
            "covered_owner_ids": [],
            "h0": {
                "exact_generated_history_prefix_sha256": sha256_json([]),
                "exact_generated_history_token_count": 0,
                "ledger_source_path": str(ledger_path.resolve()),
                "ledger_record_index": ledger_record_index,
                "exact_prefix_token_ids": [],
            },
            "model_input": {
                "prefix_sha256": sha256_json([151646]),
                "prefix_token_count": 1,
                "row_opener_token_id": 151646,
                "wrapper_mode": "commit" if checkpoint == "A" else "closed",
            },
            "owner_mapping": {
                "schema_version": "owner_interface.source_derived_mapping.v1",
                "image_id": image_id,
                "source_owner_count": 1,
                "derived_owner_count": 1,
                "source_to_derived": mapping_rows,
                "mapping_sha256": sha256_json(mapping_rows),
                "mapping_method_census": {"coco_ann_id": 1},
            },
        },
        "p1": {"status": "attempted", "arms": p1_arms},
        "p2": {"status": "attempted", "arms": p2},
        "p3": {"status": "attempted", "cells": p3},
        "p4": receipt,
    }


def _make_fixture(
    tmp_path: Path,
    *,
    owners_per_image: int = 3,
    with_pair_statuses: bool = False,
) -> tuple[dict[str, Path], dict[tuple[str, int], Path]]:
    derived = tmp_path / "derived.jsonl"
    source = tmp_path / "source.jsonl"
    derived.write_text('{"image_id": 1}\n', encoding="utf-8")
    source.write_text('{"image_id": 1}\n', encoding="utf-8")
    image_ids = (2299, 4134, 5001, 6040, 7511, 10707, 14038, 16228)
    base_events = [
        {"gt_owner_id": f"gt:{image_id}:{owner_index}", "image_id": image_id}
        for image_id in image_ids
        for owner_index in range(owners_per_image)
    ]
    events_by_checkpoint = {
        "S": base_events,
        "A": [{**event, "gt_owner_id": "gt:2299:99"} if event["gt_owner_id"] == "gt:2299:0" else event for event in base_events],
    }
    cohorts: dict[str, Path] = {}
    cohort_manifests: dict[str, Path] = {}
    ledgers: dict[str, Path] = {}
    for checkpoint, events in events_by_checkpoint.items():
        cohort = tmp_path / f"cohort-{checkpoint}.json"
        ledger = tmp_path / f"ledger-{checkpoint}.json"
        ledger_records = [
            {
                "unit_id": UNIT_ID,
                "checkpoint": checkpoint,
                "gt_owner_id": event["gt_owner_id"],
                "image_id": event["image_id"],
                "natural_boundary": 0,
                "exact_prefix_sha256": sha256_json([]),
                "exact_prefix_token_ids": [],
                "covered_owner_ids": [],
            }
            for event in events
        ]
        _write_json(
            ledger,
            {
                "schema_version": "static_dynamic_native_h0_owner_ledger.v1",
                "unit_id": UNIT_ID,
                "checkpoint": checkpoint,
                "config_fingerprint": f"fingerprint-{checkpoint}",
                "source_panel_sha256": _hash_file(source),
                "derived_panel_sha256": _hash_file(derived),
                "run_kind": "native_h0",
                "history_complete": True,
                "records": ledger_records,
            },
        )
        source_hash = _hash_file(source)
        derived_hash = _hash_file(derived)
        ledger_hash = _hash_file(ledger)
        eligible_ordinal = 11 if checkpoint == "S" else 4
        cohort_events = []
        for ordinal, event in enumerate(events, 1):
            cohort_event = {
                **event,
                "disposition": "established",
                "checkpoint_status": {checkpoint: {"natural_boundary": 0}},
            }
            if with_pair_statuses:
                pair_status = "verified_pair" if ordinal == eligible_ordinal else "no_verified_B"
                cohort_event["A_B"] = {
                    checkpoint: {
                        "pair_status": pair_status,
                        "A_latest_covered": None,
                        "B_verified_uncovered": (
                            {
                                "gt_owner_id": event["gt_owner_id"],
                                "verified_support": True,
                                "strict_complete_row": False,
                                "natural_boundary": 0,
                                "exact_prefix_sha256": sha256_json([]),
                            }
                            if pair_status == "verified_pair"
                            else None
                        ),
                    }
                }
            cohort_events.append(cohort_event)
        _write_json(
            cohort,
            {
                "schema_version": "static_dynamic_owner_interface_cohort.v1",
                "unit_id": UNIT_ID,
                "execution_contract": {"cpu_only": True, "h0_execution": False, "gpu_launch": False, "val200_index_fallback": False},
                "frozen_pool": {"required_images": list(image_ids)},
                "retention": {"min_events": 24, "max_events": 32, "retained_events": len(cohort_events), "status": "within_bounds"},
                "events": cohort_events,
                "subsets": {
                    "legacy12": {"event_count": sum(event["image_id"] != 2299 for event in cohort_events)},
                    "image2299": {"event_count": sum(event["image_id"] == 2299 for event in cohort_events)},
                },
                "sources": {
                    "derived_panel": {"path": str(derived), "sha256": derived_hash},
                    "source_panel": {"path": str(source), "sha256": source_hash},
                    "h0_ledgers": [{"path": str(ledger), "sha256": ledger_hash}],
                },
                "ledger_contract": {"support_requires_same_boundary_and_exact_prefix": True},
                "indeterminate_policy": {"missing_or_ambiguous_identity": "retain_indeterminate"},
            },
        )
        cohort_manifest = cohort.with_name(cohort.name.replace(".json", ".manifest.json"))
        cohort_value = json.loads(cohort.read_text(encoding="utf-8"))
        _write_json(
            cohort_manifest,
            {
                "schema_version": "static_dynamic_owner_interface_cohort.v1.manifest",
                "unit_id": UNIT_ID,
                "cohort_sha256": _hash_file(cohort),
                "cohort_content_sha256": sha256_json(cohort_value),
                "source_hashes": {"derived_panel": derived_hash, "source_panel": source_hash, "h0_ledgers": [ledger_hash]},
                "event_count": len(cohort_events),
                "legacy12_event_count": sum(event["image_id"] != 2299 for event in cohort_events),
                "image2299_event_count": sum(event["image_id"] == 2299 for event in cohort_events),
            },
        )
        cohorts[checkpoint] = cohort
        cohort_manifests[checkpoint] = cohort_manifest
        ledgers[checkpoint] = ledger
    shard_paths: dict[tuple[str, int], Path] = {}
    for checkpoint in ("S", "A"):
        events = events_by_checkpoint[checkpoint]
        cohort = cohorts[checkpoint]
        cohort_manifest = cohort_manifests[checkpoint]
        ledger = ledgers[checkpoint]
        for index in range(4):
            path = tmp_path / f"{checkpoint}-{index}"
            path.mkdir()
            pairs = [
                (ordinal, str(event["gt_owner_id"]), str(event["image_id"]))
                for ordinal, event in enumerate(events)
                if ordinal % 4 == index
            ]
            rows = [
                _event(
                    event_id,
                    int(image_id),
                    checkpoint=checkpoint,
                    ledger_path=ledger,
                    ledger_record_index=ordinal,
                    net_shift=1 if checkpoint == "A" else 0,
                )
                for ordinal, event_id, image_id in pairs
            ]
            for row in rows:
                row["checkpoint"] = checkpoint
            config_hash = hashlib.sha256(f"config-{checkpoint}".encode()).hexdigest()
            config_path = tmp_path / f"config-{checkpoint}.yaml"
            config_path.write_text(f"config-{checkpoint}", encoding="utf-8")
            cohort_hash = _hash_file(cohort)
            panel_hash = _hash_file(derived)
            h0_root = tmp_path / f"h0-{checkpoint}"
            (h0_root / "configs").mkdir(parents=True, exist_ok=True)
            model_path = tmp_path / f"model-{checkpoint}"
            adapter_path = tmp_path / f"adapter-{checkpoint}"
            embedding_path = tmp_path / f"embedding-{checkpoint}"
            for model_identity_path in (model_path, adapter_path, embedding_path):
                model_identity_path.mkdir(exist_ok=True)
            _write_json(h0_root / "summary.json", {"terminal_status": "completed", "checkpoint": checkpoint})
            _write_json(
                h0_root / "run_manifest.json",
                {
                    "terminal_status": "completed",
                    "checkpoint": checkpoint,
                    "resolved_config_fingerprints": {"infer_config": f"fingerprint-{checkpoint}"},
                    "model_identity": {"base": {"path": str(model_path)}},
                    "adapter_identity": {"adapter_path": str(adapter_path)},
                    "embedding_delta_identity": {"identity": {"delta_path": str(embedding_path)}},
                },
            )
            _write_json(h0_root / "configs" / "resolved.json", {"resolution": {"fingerprint": f"fingerprint-{checkpoint}"}})
            identity = {
                "schema_version": RUNTIME_SCHEMA,
                "unit_id": UNIT_ID,
                "checkpoint": checkpoint,
                "stage": "all",
                "execution_mode": "live_intervention" if with_pair_statuses else None,
                "config_path": str(config_path),
                "config_sha256": config_hash,
                "resolved_config_fingerprint": f"fingerprint-{checkpoint}",
                "panel_path": str(derived),
                "panel_sha256": panel_hash,
                "panel_identity": {
                    "derived_panel_sha256": panel_hash,
                    "source_panel_sha256": _hash_file(source),
                    "cohort_manifest_path": str(cohort_manifest),
                    "cohort_manifest_sha256": _hash_file(cohort_manifest),
                    "h0_ledger_sha256": {str(ledger.resolve()): _hash_file(ledger)},
                },
                "cohort_path": str(cohort),
                "cohort_sha256": cohort_hash,
                "event_count": len(rows),
                "h0": {
                    "root": str(h0_root),
                    "summary_sha256": _hash_file(h0_root / "summary.json"),
                    "run_manifest_sha256": _hash_file(h0_root / "run_manifest.json"),
                    "resolved_config_sha256": _hash_file(h0_root / "configs" / "resolved.json"),
                    "resolved_config_fingerprint": f"fingerprint-{checkpoint}",
                    "manifest_config_fingerprint": f"fingerprint-{checkpoint}",
                    "model_base_path": str(model_path),
                    "adapter_path": str(adapter_path),
                    "embedding_delta_path": str(embedding_path),
                },
                "runtime_attestation": _runtime_attestation(),
            }
            _write_json(path / "runtime_identity.json", identity)
            _write_json(path / "exact_prefix_manifest.json", {"schema_version": RUNTIME_SCHEMA, "identity": identity, "events": [{"event_id": row["event_id"], "image_id": row["image_id"], "checkpoint": checkpoint, "prefix": row["prefix"]} for row in rows]})
            _write_json(path / "intervention_manifest.json", {"schema_version": RUNTIME_SCHEMA, "events": [{"event_id": row["event_id"], "image_id": row["image_id"], "stages": ["p1", "p2", "p3", "p4"]} for row in rows]})
            (path / "per_event_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
            _write_json(
                path / "gradient_receipt.json",
                {
                    "schema_version": GRADIENT_SCHEMA,
                    "runtime_attestation": _runtime_attestation(),
                    "receipts": [{"event_id": row["event_id"], "receipt": row["p4"]} for row in rows],
                },
            )
            _write_json(path / "terminal_summary.json", {**identity, "status": "completed", "events_attempted": len(rows)})
            shard_paths[(checkpoint, index)] = path
    return cohorts, shard_paths


def _not_applicable_attestation() -> dict[str, object]:
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA,
        "status": "not_applicable",
        "passed": False,
        "reason": "ineligible_contract_materialization_cpu_only",
    }


def _rewrite_root(
    path: Path,
    *,
    identity: dict[str, object],
    rows: list[dict[str, object]],
    stages: list[str],
) -> None:
    _write_json(path / "runtime_identity.json", identity)
    _write_json(
        path / "exact_prefix_manifest.json",
        {
            "schema_version": RUNTIME_SCHEMA,
            "identity": identity,
            "events": [
                {
                    "event_id": row["event_id"],
                    "image_id": row["image_id"],
                    "checkpoint": row["checkpoint"],
                    "prefix": row["prefix"],
                }
                for row in rows
            ],
        },
    )
    _write_json(
        path / "intervention_manifest.json",
        {
            "schema_version": RUNTIME_SCHEMA,
            "events": [
                {"event_id": row["event_id"], "image_id": row["image_id"], "stages": stages}
                for row in rows
            ],
        },
    )
    (path / "per_event_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipts = [
        {"event_id": row["event_id"], "receipt": row["p4"]}
        for row in rows
        if row.get("p4", {}).get("status") in {"valid", "technical_invalid"}
    ]
    _write_json(
        path / "gradient_receipt.json",
        {
            "schema_version": GRADIENT_SCHEMA,
            "runtime_attestation": identity["runtime_attestation"],
            "receipts": receipts,
        },
    )
    _write_json(
        path / "terminal_summary.json",
        {
            **identity,
            "status": "completed",
            "events_attempted": len(rows),
            "valid_event_count": 0 if stages == ["p4"] else sum(row.get("p1", {}).get("status") == "attempted" for row in rows),
        },
    )


def _make_hybrid_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], list, list[EligibleHoldSpec], Path]:
    cohorts, roots = _make_fixture(
        tmp_path,
        owners_per_image=4,
        with_pair_statuses=True,
    )
    original_rows = {
        key: [json.loads(line) for line in (path / "per_event_results.jsonl").read_text(encoding="utf-8").splitlines()]
        for key, path in roots.items()
    }
    s_pair = ("gt:5001:2", "5001")
    a_pair = ("gt:2299:3", "2299")
    a_row = next(row for row in original_rows[("A", 3)] if (row["event_id"], str(row["image_id"])) == a_pair)

    a_base = tmp_path / "A-3-eligible-base"
    a_overlay = tmp_path / "A-3-p4-repair"
    shutil.copytree(roots[("A", 3)], a_base)
    shutil.copytree(roots[("A", 3)], a_overlay)

    for key, path in roots.items():
        checkpoint, _index = key
        rows = [
            row
            for row in original_rows[key]
            if (checkpoint, row["event_id"], str(row["image_id"]))
            not in {("S", *s_pair), ("A", *a_pair)}
        ]
        identity = json.loads((path / "runtime_identity.json").read_text(encoding="utf-8"))
        identity["execution_mode"] = "ineligible_contract_materialization"
        identity["runtime_attestation"] = _not_applicable_attestation()
        identity["event_count"] = len(rows)
        for row in rows:
            pair_status = "no_verified_B"
            reason = f"active cohort pair is not actuator-eligible: {pair_status}"
            row["runtime_attestation"] = _not_applicable_attestation()
            row["eligibility"] = {
                "status": "invalid/uninterpretable",
                "pair_status": pair_status,
                "reason": reason,
                "actuators_called": False,
            }
            row["p1"] = {
                "status": "invalid/uninterpretable",
                "reason": reason,
                "arms": {
                    probe: {"status": "invalid/uninterpretable", "reason": reason}
                    for probe in (
                        "K00", "K01", "K10", "K11", "K12", "K13",
                        "R00_block13", "R10_block13", "R11_block13", "R12_block13",
                        "R00_block23", "R10_block23", "R11_block23", "R12_block23",
                        "R00_block27", "R10_block27",
                    )
                },
            }
            row["p2"] = {
                "status": "invalid/uninterpretable",
                "reason": reason,
                "arms": {probe: {"status": "invalid/uninterpretable", "reason": reason} for probe in ("D00", "D01", "D10", "D11", "D12", "D20", "D21")},
            }
            row["p3"] = {
                "status": "invalid/uninterpretable",
                "reason": reason,
                "cells": {probe: {"status": "invalid/uninterpretable", "reason": reason} for probe in ("Y00", "Y10", "Y01", "Y11")},
            }
            row["p4"] = {"status": "invalid/uninterpretable", "reason": reason}
        _rewrite_root(path, identity=identity, rows=rows, stages=["p1", "p2", "p3", "p4"])

    base_identity = json.loads((a_base / "runtime_identity.json").read_text(encoding="utf-8"))
    base_identity["execution_mode"] = "live_intervention"
    base_identity["event_count"] = 1
    a_row["eligibility"] = {
        "status": "eligible",
        "pair_status": "verified_pair",
        "reason": None,
        "actuators_called": True,
    }
    a_row["p4"] = {
        "schema_version": GRADIENT_SCHEMA,
        "status": "technical_invalid",
        "invalid_reasons": ["forward_capture_contract_invalid: fragmented carrier identity"],
    }
    _rewrite_root(a_base, identity=base_identity, rows=[a_row], stages=["p1", "p2", "p3", "p4"])

    overlay_identity = json.loads(json.dumps(base_identity))
    overlay_identity["stage"] = "p4"
    overlay_row = {
        key: json.loads(json.dumps(a_row[key]))
        for key in ("event_id", "image_id", "checkpoint", "runtime_attestation", "eligibility", "prefix")
    }
    overlay_row["p4"] = _gradient_receipt()
    _rewrite_root(a_overlay, identity=overlay_identity, rows=[overlay_row], stages=["p4"])

    attempt_roots: list[Path] = []
    for attempt in (1, 2):
        attempt_root = tmp_path / f"S-eligible-attempt-{attempt}"
        attempt_root.mkdir()
        attempt_identity = {
            "schema_version": RUNTIME_SCHEMA,
            "unit_id": UNIT_ID,
            "checkpoint": "S",
            "attempt": attempt,
        }
        _write_json(attempt_root / "runtime_identity.json", attempt_identity)
        attempt_roots.append(attempt_root)
    hold_path = tmp_path / "S-ordinal11-HOLD.json"
    hold_receipt_path = tmp_path / "S-ordinal11-HOLD.receipt.json"
    unit_path = tmp_path / "frozen-unit.md"
    tasks_path = tmp_path / "frozen-tasks.md"
    unit_path.write_text("frozen unit\n", encoding="utf-8")
    tasks_path.write_text("frozen tasks\n", encoding="utf-8")
    frozen = {
        "unit": {"path": str(unit_path), "sha256": _hash_file(unit_path), "size_bytes": unit_path.stat().st_size},
        "tasks": {"path": str(tasks_path), "sha256": _hash_file(tasks_path), "size_bytes": tasks_path.stat().st_size},
    }
    invalid_cell = {
        "execution_status": "not_sealed",
        "metrics": None,
        "model_output": None,
        "reason_code": "pre_actuator_technical_failure_repair_exhausted",
        "scientific_observation": None,
        "status": "invalid/uninterpretable",
    }
    matrix_cells = {
        "p1": (
            "K00", "K01", "K10", "K11", "K12", "K13",
            "R00_block13", "R10_block13", "R11_block13", "R12_block13",
            "R00_block23", "R10_block23", "R11_block23", "R12_block23",
            "R00_block27", "R10_block27",
        ),
        "p2": tuple(f"{arm}.horizon_{horizon}" for arm in ("D00", "D01", "D10", "D11", "D12", "D20", "D21") for horizon in (1, 3)),
        "p3": tuple(f"{cell}.horizon_{horizon}" for cell in ("Y00", "Y10", "Y01", "Y11") for horizon in (1, 3)),
        "p4": ("target_b_complete_row_nll", "uncovered_b_vs_covered_a_margin_loss", "fixed_sum_coupled"),
    }
    attempt_lineage = []
    for role, root in zip(("initial", "repair1"), attempt_roots, strict=True):
        identity_path = root / "runtime_identity.json"
        digest = _hash_file(identity_path)
        census = [{"relative_path": "runtime_identity.json", "sha256": digest, "size_bytes": identity_path.stat().st_size}]
        attempt_lineage.append(
            {
                "cause": {
                    "code": f"synthetic_{role}_pre_actuator_failure",
                    "evidence_level": "lead_observed_unattested",
                    "summary": f"synthetic {role} production-shaped failure",
                    "verbatim_stderr": None,
                },
                "directory": str(root),
                "file_census": census,
                "file_census_sha256": sha256_json(census),
                "persisted_failure_log": {"path": None, "sha256": None, "status": "unavailable"},
                "role": role,
                "runtime_attestation": {
                    "checkpoint": "S",
                    "passed": True,
                    "physical_device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                    "pid": 1 if role == "initial" else 2,
                    "status": "validated",
                    "timestamp_utc": "2026-08-05T00:00:00Z" if role == "initial" else "2026-08-05T00:01:00Z",
                },
                "runtime_identity_path": str(identity_path),
                "runtime_identity_sha256": digest,
                "runtime_identity_size_bytes": identity_path.stat().st_size,
            }
        )
    s_ledger = Path(json.loads(cohorts["S"].read_text(encoding="utf-8"))["sources"]["h0_ledgers"][0]["path"])
    hold = {
        "schema_version": ELIGIBLE_HOLD_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "sealed",
        "kind": "eligible_pre_actuator_technical_hold",
        "receipt_path": str(hold_receipt_path),
        "frozen_contract": frozen,
        "event_binding": {
            "checkpoint": "S",
            "cohort_eligibility": "eligible_verified_pair",
            "cohort_sha256": _hash_file(cohorts["S"]),
            "event_id": s_pair[0],
            "expected_h0_exact_prefix_sha256": sha256_json([]),
            "geometry_disposition": "eligible_verified_pair_regions",
            "geometry_status": "available",
            "image_id": int(s_pair[1]),
            "natural_boundary": 0,
            "ordinal": 11,
            "source_panel_object_index": 2,
        },
        "h0_binding": {
            "covered_a_owner_id": "gt:5001:0",
            "native_h0_ledger_path": str(s_ledger),
            "native_h0_ledger_sha256": _hash_file(s_ledger),
            "native_h0_record_sha256": sha256_json({"synthetic": "h0"}),
            "support_ledger_path": str(s_ledger),
            "support_ledger_sha256": _hash_file(s_ledger),
            "support_record_sha256": sha256_json({"synthetic": "support"}),
            "target_b_owner_id": s_pair[0],
            "verified_support": True,
        },
        "classification": {
            "actuators_called": None,
            "complete_non_scored": False,
            "matrix_status": "eligible_pre_actuator_hold",
            "scored": False,
        },
        "repair_policy": {
            "attempt_count": 2,
            "attempt_roles": ["initial", "repair1"],
            "exhausted": True,
            "repair_count": 1,
            "rule": "one_exact_repair_then_invalid_uninterpretable",
            "unit_sha256": frozen["unit"]["sha256"],
        },
        "execution_evidence": {
            "actual_actuator_invocation_count": None,
            "actual_model_forward_count": None,
            "lead_observation_evidence_level": "unattested",
            "lead_observed_pre_actuator": True,
            "persisted_artifacts_absent": ["exact_prefix_manifest.json", "intervention_manifest.json", "per_event_results.jsonl", "gradient_receipt.json", "terminal_summary.json"],
            "persisted_event_result_count": 0,
            "persisted_failure_log": {"path": None, "sha256": None, "status": "unavailable"},
            "persisted_gradient_receipt_count": 0,
            "persisted_intervention_receipt_count": 0,
            "persisted_terminal_count": 0,
            "receipt_bearing_actuator_cell_count": 0,
            "receipt_bearing_scientific_cell_count": 0,
        },
        "attempt_lineage": attempt_lineage,
        "matrix_dispositions": {
            stage: {
                "cells": {cell: json.loads(json.dumps(invalid_cell)) for cell in cells},
                "origin": "administrative_disposition_not_model_output",
                "status": "invalid/uninterpretable",
            }
            for stage, cells in matrix_cells.items()
        },
    }
    hold["self_sha256"] = sha256_json(hold)
    _write_json(hold_path, hold)
    input_refs = [
        {"role": role, **ref}
        for role, ref in (("frozen_unit", frozen["unit"]), ("frozen_tasks", frozen["tasks"]))
    ]
    input_refs.extend(
        {
            "role": f"attempt:{attempt['role']}:runtime_identity",
            "path": attempt["runtime_identity_path"],
            "sha256": attempt["runtime_identity_sha256"],
            "size_bytes": attempt["runtime_identity_size_bytes"],
        }
        for attempt in attempt_lineage
    )
    receipt = {
        "schema_version": f"{ELIGIBLE_HOLD_SCHEMA_VERSION}.receipt.v1",
        "unit_id": UNIT_ID,
        "kind": "eligible_pre_actuator_technical_hold",
        "leaf_path": str(hold_path),
        "leaf_sha256": _hash_file(hold_path),
        "leaf_self_sha256": hold["self_sha256"],
        "input_set_sha256": sha256_json(input_refs),
        "inputs": input_refs,
    }
    receipt["self_sha256"] = sha256_json(receipt)
    _write_json(hold_receipt_path, receipt)
    specs = [
        parse_shard_selector(f"{checkpoint}:{index}/4={roots[(checkpoint, index)]}")
        for checkpoint in ("S", "A")
        for index in range(4)
    ]
    specs.extend(
        [
            parse_shard_selector(f"A:3/4={a_base}"),
            parse_shard_selector(f"A:3/4={a_overlay}"),
        ]
    )
    return cohorts, specs, [parse_eligible_hold_selector(f"S:11={hold_path}")], a_overlay


def test_parse_selector_and_aggregate_four_way_with_evidence_views(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    assert parse_cohort_selector(f"S={cohorts['S']}")[0] == "S"
    assert parse_cohort_selector(f"A={cohorts['A']}")[0] == "A"
    specs = [parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}") for checkpoint in ("S", "A") for index in range(4)]
    output = tmp_path / "summary.json"
    receipt = tmp_path / "receipt.json"
    summary = aggregate_shards(cohort_paths=cohorts, shard_specs=specs, output_path=output, receipt_path=receipt)
    assert summary["status"] == "completed"
    assert summary["cohorts"]["S"]["event_count"] == 24
    assert summary["cohorts"]["A"]["event_count"] == 24
    assert summary["cohorts"]["S"]["event_set_sha256"] != summary["cohorts"]["A"]["event_set_sha256"]
    s_events = {row["event_id"] for row in summary["event_results"] if row["checkpoint"] == "S"}
    a_events = {row["event_id"] for row in summary["event_results"] if row["checkpoint"] == "A"}
    assert s_events != a_events
    assert summary["aggregates"]["S"]["p3"]["crossover.horizon_1"]["all"]["metrics"]["tau"]["mean"] == 1.0
    assert summary["aggregates"]["S"]["p4"]["fixed_sum_coupled"]["all"]["metrics"]["gradient_norm.image_residual"]["count"] == 24
    assert summary["aggregates"]["S"]["p2"]["D21.horizon_1"]["all"]["indeterminate_count"] == 24
    assert summary["evidence_views"]["A"]["crossover"]["crossover.horizon_3"]["all"]["metrics"]["tau"]["count"] == 24
    written = json.loads(output.read_text(encoding="utf-8"))
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    assert written["self_sha256"] == sha256_json({key: value for key, value in written.items() if key != "self_sha256"})
    assert receipt_value["summary_sha256"] == _hash_file(output)
    assert receipt_value["input_sha256"] == written["input_sha256"]
    assert receipt_value["self_sha256"] == sha256_json({key: value for key, value in receipt_value.items() if key != "self_sha256"})


def test_aggregate_exact_hybrid_topology_with_cpu_sparse_hold_and_p4_overlay(tmp_path: Path) -> None:
    cohorts, specs, holds, _overlay = _make_hybrid_fixture(tmp_path)
    summary = aggregate_shards(
        cohort_paths=cohorts,
        shard_specs=specs,
        eligible_hold_specs=holds,
    )
    assert len(summary["event_results"]) == 64
    assert summary["hypotheses"] == {f"H{index}": None for index in range(1, 6)}
    assert summary["interpretation"] is None
    assert summary["recommendation"] is None
    hold = next(
        row for row in summary["event_results"]
        if row["checkpoint"] == "S" and row["event_id"] == "gt:5001:2"
    )
    assert hold["source_kind"] == "eligible_hold"
    assert {value["status"] for value in hold["validity"].values()} == {"technical_invalid"}
    assert all(
        observation["metrics"] == {}
        for stage in hold["observations"].values()
        for observation in stage.values()
    )
    repaired = next(
        row for row in summary["event_results"]
        if row["checkpoint"] == "A" and row["event_id"] == "gt:2299:3"
    )
    assert repaired["source_kind"] == "live_all_with_p4_repair"
    assert repaired["p4_repair_lineage"]["superseded_stages"] == ["p4"]
    assert repaired["p4_repair_lineage"]["base"]["p4"]["status"] == "technical_invalid"
    assert repaired["p4_repair_lineage"]["repair"]["p4"]["status"] == "valid"
    assert repaired["validity"]["p4"]["status"] == "valid"


def test_hybrid_rejects_second_p4_overlay_and_tampered_hold(tmp_path: Path) -> None:
    cohorts, specs, holds, overlay = _make_hybrid_fixture(tmp_path)
    second_overlay = tmp_path / "A-3-p4-repair-second"
    shutil.copytree(overlay, second_overlay)
    with pytest.raises(AggregationError, match="more than one P4 repair"):
        aggregate_shards(
            cohort_paths=cohorts,
            shard_specs=[*specs, parse_shard_selector(f"A:3/4={second_overlay}")],
            eligible_hold_specs=holds,
        )
    hold_path = holds[0].path
    hold = json.loads(hold_path.read_text(encoding="utf-8"))
    hold["matrix_dispositions"]["p4"]["cells"]["fixed_sum_coupled"]["metrics"] = 0.0
    _write_json(hold_path, hold)
    with pytest.raises(AggregationError, match="self hash mismatch"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs, eligible_hold_specs=holds)


def test_hybrid_rejects_cpu_materialization_of_verified_eligible_event(tmp_path: Path) -> None:
    cohorts, specs, holds, _overlay = _make_hybrid_fixture(tmp_path)
    cpu_template = next(spec.path for spec in specs if spec.checkpoint == "S" and spec.index == 2)
    eligible_cpu = tmp_path / "S-2-eligible-cpu"
    shutil.copytree(cpu_template, eligible_cpu)
    identity = json.loads((eligible_cpu / "runtime_identity.json").read_text(encoding="utf-8"))
    hold = json.loads(holds[0].path.read_text(encoding="utf-8"))
    row = json.loads((eligible_cpu / "per_event_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["event_id"] = hold["event_binding"]["event_id"]
    row["image_id"] = hold["event_binding"]["image_id"]
    ledger_path = Path(json.loads(cohorts["S"].read_text(encoding="utf-8"))["sources"]["h0_ledgers"][0]["path"])
    row["prefix"] = _event(
        row["event_id"],
        int(row["image_id"]),
        checkpoint="S",
        ledger_path=ledger_path,
        ledger_record_index=10,
    )["prefix"]
    row["eligibility"]["pair_status"] = "verified_pair"
    identity["event_count"] = 1
    _rewrite_root(eligible_cpu, identity=identity, rows=[row], stages=["p1", "p2", "p3", "p4"])
    with pytest.raises(AggregationError, match="not cohort-proven ineligible"):
        aggregate_shards(
            cohort_paths=cohorts,
            shard_specs=[*specs, parse_shard_selector(f"S:2/4={eligible_cpu}")],
            eligible_hold_specs=[],
        )


def test_hybrid_rejects_non_not_applicable_cpu_attestation_and_overlay_replacement(tmp_path: Path) -> None:
    cohorts, specs, holds, overlay = _make_hybrid_fixture(tmp_path)
    cpu = next(spec.path for spec in specs if spec.checkpoint == "S" and spec.index == 0)
    identity = json.loads((cpu / "runtime_identity.json").read_text(encoding="utf-8"))
    identity["runtime_attestation"] = _runtime_attestation()
    _write_json(cpu / "runtime_identity.json", identity)
    with pytest.raises(AggregationError, match="exact ineligible_contract_materialization_cpu_only"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs, eligible_hold_specs=holds)

    replacement_root = tmp_path / "silent-replacement"
    replacement_root.mkdir()
    cohorts, specs, holds, overlay = _make_hybrid_fixture(replacement_root)
    rows = [json.loads(line) for line in (overlay / "per_event_results.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["p1"] = {"status": "attempted", "arms": {}}
    (overlay / "per_event_results.jsonl").write_text(
        json.dumps(rows[0], sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AggregationError, match="silent replacement"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs, eligible_hold_specs=holds)


def test_real_hybrid_topology_reaches_bundle_with_administrative_hold(tmp_path: Path) -> None:
    root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover"
    )
    cohorts = {
        "S": root / "cohort/s-step2444-final-support.json",
        "A": root / "cohort/a3-step2445-final-support.json",
    }
    specs = [
        parse_shard_selector(
            f"{checkpoint}:{index}/4={root}/p1-p4/non-scored/{stem}-shard-{index}"
        )
        for checkpoint, stem in (("S", "s-step2444"), ("A", "a3-step2445"))
        for index in range(4)
    ]
    specs.extend(
        [
            parse_shard_selector(
                f"A:3/4={root}/p1-p4-live-smoke/a3-step2445-ordinal4-all-gridshape-repair2"
            ),
            parse_shard_selector(
                f"A:3/4={root}/p1-p4-live-smoke/a3-step2445-ordinal4-p4-fragmented-capture-repair1"
            ),
        ]
    )
    holds = [
        parse_eligible_hold_selector(
            f"S:11={root}/p1-p4/eligible-hold/s-step2444-ordinal11/eligible_hold_leaf.json"
        )
    ]
    summary = aggregate_shards(
        cohort_paths=cohorts,
        shard_specs=specs,
        eligible_hold_specs=holds,
    )
    result = build_evidence_bundle(
        summary,
        output_dir=tmp_path,
        census_paths={
            "S": root / "observational-census/merged/s-step2444/p1-census.jsonl",
            "A": root / "observational-census/merged/a3-step2445/p1-census.jsonl",
        },
    )
    bundle = result["bundle"]
    assert bundle["status"] == "completed"
    assert bundle["qualification_status"] == "hold"
    assert bundle["hypotheses"] == {f"H{index}": None for index in range(1, 6)}
    assert bundle["recommendation"] is None
    assert bundle["artifacts"]["raw_event_evidence"]["record_count"] == 64
    assert {item["row_count"] for rows in bundle["census_sources"].values() for item in rows} == {992}
    raw_rows = [
        json.loads(line)
        for line in Path(result["paths"]["raw_event_evidence"]).read_text(encoding="utf-8").splitlines()
    ]
    hold = next(row for row in raw_rows if row["checkpoint"] == "S" and row["event_id"] == "gt:5001:15")
    assert hold["matrix_status"] == "eligible_pre_actuator_hold"
    assert hold["runtime_identity"] is None
    assert hold["actuators_called"] is None
    assert hold["p4"]["readiness"]["readiness"] == "hold"
    assert all(
        observation["metrics"] == {}
        for stage in hold["observations"].values()
        for observation in stage.values()
    )
    repaired = next(row for row in raw_rows if row["checkpoint"] == "A" and row["event_id"] == "gt:2299:2")
    overlay_ref_keys = {
        key for key in repaired["raw_refs"] if key.startswith("p4_repair_overlay.")
    }
    assert overlay_ref_keys == {
        "p4_repair_overlay.exact_prefix_manifest",
        "p4_repair_overlay.gradient_receipt",
        "p4_repair_overlay.intervention_manifest",
        "p4_repair_overlay.per_event_results",
        "p4_repair_overlay.runtime_identity",
        "p4_repair_overlay.terminal_summary",
    }
    for key in overlay_ref_keys:
        ref = repaired["raw_refs"][key]
        assert "a3-step2445-ordinal4-p4-fragmented-capture-repair1" in ref["path"]
        assert _hash_file(Path(ref["path"])) == ref["sha256"]


def test_fail_closed_on_missing_shard_or_cross_shard_event(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    specs = [parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}") for checkpoint in ("S", "A") for index in range(4)]
    with pytest.raises(AggregationError, match="shard selectors"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs[:-1])
    # Replace one shard's event with a duplicate owner from another modulo-4
    # bucket; the explicit cohort partition must reject it.
    path = paths[("S", 0)] / "per_event_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["event_id"] = "gt:5001:2"
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(AggregationError, match="event ownership|duplicate"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


def test_fail_closed_on_terminal_runtime_attestation_drift(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    specs = [
        parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}")
        for checkpoint in ("S", "A")
        for index in range(4)
    ]
    terminal_path = paths[("S", 0)] / "terminal_summary.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["runtime_attestation"]["model_device"] = "cuda:1"
    _write_json(terminal_path, terminal)
    with pytest.raises(AggregationError, match="terminal runtime attestation"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


def test_fail_closed_on_exact_prefix_runtime_attestation_drift(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    specs = [
        parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}")
        for checkpoint in ("S", "A")
        for index in range(4)
    ]
    prefix_path = paths[("S", 0)] / "exact_prefix_manifest.json"
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    prefix["identity"]["runtime_attestation"]["model_device"] = "cuda:1"
    _write_json(prefix_path, prefix)
    with pytest.raises(AggregationError, match="exact prefix identity runtime attestation"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


def test_fail_closed_on_stage_gap_and_nonfinite_gradient(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    path = paths[("A", 2)] / "per_event_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["p3"]
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    specs = [parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}") for checkpoint in ("S", "A") for index in range(4)]
    with pytest.raises(AggregationError, match="missing stage p3"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


@pytest.mark.parametrize(
    ("stage", "container", "key"),
    (("p1", "arms", "K10"), ("p2", "arms", "D10"), ("p3", "cells", "Y11")),
)
def test_fail_closed_on_declared_probe_gap(
    tmp_path: Path,
    stage: str,
    container: str,
    key: str,
) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    path = paths[("S", 0)] / "per_event_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0][stage][container][key]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    specs = [
        parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}")
        for checkpoint in ("S", "A")
        for index in range(4)
    ]
    with pytest.raises(AggregationError, match="missing probe"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


def test_fail_closed_on_nonfinite_json_number(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    path = paths[("S", 1)] / "per_event_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["p4"]["objectives"]["fixed_sum_coupled"]["value"] = float("nan")
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    specs = [parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}") for checkpoint in ("S", "A") for index in range(4)]
    with pytest.raises(AggregationError, match="non-finite"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs)


def test_fail_closed_on_missing_checkpoint_cohort_and_output_collisions(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    specs = [parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}") for checkpoint in ("S", "A") for index in range(4)]
    with pytest.raises(AggregationError, match="exactly one cohort"):
        aggregate_shards(cohort_paths={"S": cohorts["S"]}, shard_specs=specs)
    with pytest.raises(AggregationError, match="distinct checkpoint-specific"):
        aggregate_shards(cohort_paths={"S": cohorts["S"], "A": cohorts["S"]}, shard_specs=specs)
    with pytest.raises(AggregationError, match="cohort path"):
        aggregate_shards(cohort_paths={"S": cohorts["A"], "A": cohorts["S"]}, shard_specs=specs)
    output = tmp_path / "summary.json"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output collision"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs, output_path=output, receipt_path=tmp_path / "new-receipt.json")
    output.unlink()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="receipt collision"):
        aggregate_shards(cohort_paths=cohorts, shard_specs=specs, output_path=tmp_path / "new-summary.json", receipt_path=receipt)


def test_p1_delta_is_indeterminate_when_k00_baseline_is_invalid() -> None:
    row = _event("gt:2299:0", 2299)
    row["p1"]["arms"]["K00"] = {"status": "invalid/uninterpretable", "reason": "native prefix mismatch"}
    observations = _p1_observations(row)
    assert observations["K00"]["validity"] == "invalid"
    assert observations["K10"]["validity"] == "indeterminate"
    assert observations["K10"]["metrics"] == {}


def test_p1_parse_owner_burden_is_valid_with_empty_owner_match() -> None:
    row = _event("gt:2299:0", 2299)
    row["p1"]["arms"]["K10"]["owner_match"] = {"status": "unmatched", "owner_id": None}
    observations = _p1_observations(row)
    assert observations["K10"]["validity"] == "valid"
    assert observations["K10"]["owner_ids"] == []
    assert observations["K10"]["metrics"]["parse.unmatched"] == 1.0
    assert sum(
        observations["K10"]["metrics"].get(name, 0.0)
        for name in ("parse.valid", "parse.malformed", "parse.unmatched", "parse.ambiguous")
    ) == 1.0


def test_p1_missing_generation_receipt_is_invalid_not_an_aggregator_crash() -> None:
    row = _event("gt:2299:0", 2299)
    del row["p1"]["arms"]["K10"]["generation_status"]
    observation = _p1_observations(row)["K10"]
    assert observation["validity"] == "invalid"
    assert "generation_status" in str(observation["reason"])
    assert observation["metrics"] == {}


def test_horizon_qualification_preserves_gkl_and_repeat_t1_t3() -> None:
    raw = _strict_horizon(["gt:new", "gt:base", "gt:new"], requested=3, covered=("gt:base",))
    qualified = _horizon_qualification(
        raw,
        context="test.horizon",
        checkpoint="S",
        event_id="gt:1:2",
        image_id="1",
        target_owner="gt:1:2",
        covered_owner_ids=("gt:base",),
        expected_horizon=3,
        prefix_hashes={sha256_json([]), sha256_json([151646])},
        support_rows={},
    )
    assert qualified["qualification"] == "qualified"
    assert qualified["observation"]["delta"]["G"] == ["gt:new"]
    assert qualified["observation"]["metrics"]["repeat_hazard.t+2"] == 1.0


def test_p3_arithmetic_is_recomputed_from_four_cells() -> None:
    row = _event("gt:2299:0", 2299)
    crossover = _crossover_observation(row, 1)
    assert crossover["validity"] == "valid"
    assert crossover["metrics"]["Delta_static"] == 1.0
    assert crossover["metrics"]["Delta_dynamic"] == -1.0
    assert crossover["metrics"]["tau"] == 1.0


def test_p4_pass_through_requires_full_mass_and_non_target_receipt() -> None:
    p4 = _gradient_receipt()
    readiness, missing = _p4_qualification(p4, context="test.p4")
    assert readiness["qualification"] == "indeterminate"
    assert any("grammar_stop_invalid_mass" in item for item in missing)
    p4["grammar_stop_invalid_mass"] = {
        name: {
            "status": "reported",
            "finite": True,
            "token_count": 1,
            "mean_probability": value,
            "max_probability": value,
            "min_probability": value,
        }
        for name, value in (("grammar", 1.0), ("stop", 0.0), ("invalid", 0.0))
    }
    p4["objectives"]["uncovered_b_vs_covered_a_margin_loss"].update(
        {
            "reported_margin_higher_is_better": 1.0,
            "reported_uncovered_b_mean_logprob": -1.0,
            "reported_covered_a_mean_logprob": -2.0,
        }
    )
    readiness, missing = _p4_qualification(p4, context="test.p4")
    assert readiness["qualification"] == "qualified"
    assert missing == []


def test_build_bundle_fails_closed_before_scoring_incomplete_h0_denominator(tmp_path: Path) -> None:
    cohorts, paths = _make_fixture(tmp_path)
    specs = [
        parse_shard_selector(f"{checkpoint}:{index}/4={paths[(checkpoint, index)]}")
        for checkpoint in ("S", "A")
        for index in range(4)
    ]
    summary = aggregate_shards(cohort_paths=cohorts, shard_specs=specs)
    with pytest.raises(AggregationError, match="derived_receipt|13 H0 image baselines"):
        build_evidence_bundle(summary, output_dir=tmp_path / "bundle")


def _qualified_p4() -> dict[str, object]:
    p4 = _gradient_receipt()
    p4["grammar_stop_invalid_mass"] = {
        name: {
            "status": "reported",
            "finite": True,
            "token_count": 1,
            "mean_probability": value,
            "max_probability": value,
            "min_probability": value,
        }
        for name, value in (("grammar", 0.8), ("stop", 0.1), ("invalid", 0.1))
    }
    p4["objectives"]["uncovered_b_vs_covered_a_margin_loss"].update(
        {
            "reported_margin_higher_is_better": 1.0,
            "reported_uncovered_b_mean_logprob": -1.0,
            "reported_covered_a_mean_logprob": -2.0,
        }
    )
    return p4


def test_path_descriptors_load_and_bind_support_and_census_hashes(tmp_path: Path) -> None:
    source_hash, derived_hash, cohort_hash, h0_hash = (char * 64 for char in "1234")
    prefix_ids = [1, 2]
    support_path = tmp_path / "support.json"
    _write_json(
        support_path,
        {
            "unit_id": UNIT_ID,
            "checkpoint": "S",
            "source_panel_sha256": source_hash,
            "derived_panel_sha256": derived_hash,
            "h0_source_sha256": h0_hash,
            "records": [
                {
                    "unit_id": UNIT_ID,
                    "checkpoint": "S",
                    "source_panel_sha256": source_hash,
                    "derived_panel_sha256": derived_hash,
                    "image_id": 1,
                    "gt_owner_id": "gt:1:2",
                    "natural_boundary": 2,
                    "exact_prefix_sha256": sha256_json(prefix_ids),
                    "exact_prefix_token_ids": prefix_ids,
                    "verified_support": True,
                }
            ],
        },
    )
    payload, source_ref = _read_payload(support_path, "support")
    assert payload["checkpoint"] == "S"
    assert source_ref == {"path": str(support_path.resolve()), "sha256": _hash_file(support_path)}
    cohort_sources = {
        checkpoint: {"source_panel": source_hash, "derived_panel": derived_hash, "support_ledgers": {}}
        for checkpoint in ("S", "A")
    }
    ledgers = {
        checkpoint: {"ledger_refs": [{"sha256": h0_hash}]}
        for checkpoint in ("S", "A")
    }
    _rows, indexed, missing = _load_support_sources(
        cohort_sources=cohort_sources,
        support_sources={"S": {"path": support_path, "sha256": _hash_file(support_path)}},
        ledgers_by_checkpoint=ledgers,
    )
    assert ("S", "1", "gt:1:2", 2, sha256_json(prefix_ids)) in indexed
    assert missing == []

    census_path = tmp_path / "census.json"
    _write_json(
        census_path,
        {
            "unit_id": UNIT_ID,
            "checkpoint": "S",
            "identity": {"cohort": {"sha256": cohort_hash}, "panel": {"sha256": derived_hash}},
            "rows": [{"checkpoint": "S", "image_id": 1}],
        },
    )
    cohorts = {
        checkpoint: {"sha256": cohort_hash, "payload": {"events": [{"gt_owner_id": f"gt:{checkpoint}", "image_id": 1}]}}
        for checkpoint in ("S", "A")
    }
    census, census_missing = _load_census_sources(
        {"S": {"path": census_path, "sha256": _hash_file(census_path)}},
        cohort_payloads=cohorts,
        source_hashes=cohort_sources,
    )
    assert census["S"][0]["row_count"] == 1
    assert census_missing == []
    with pytest.raises(AggregationError, match="content hash mismatch"):
        _load_census_sources(
            {"S": {"path": census_path, "sha256": "0" * 64}},
            cohort_payloads=cohorts,
            source_hashes=cohort_sources,
        )


def _install_derived_receipt(cohort_path: Path) -> Path:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    source = Path(cohort["sources"]["source_panel"]["path"])
    derived = Path(cohort["sources"]["derived_panel"]["path"])
    receipt_path = cohort_path.with_name(f"{cohort_path.stem}.derived.receipt.json")
    images = [{"image_id": image_id} for image_id in range(13)]
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "unit_id": UNIT_ID,
            "receipt_path": str(receipt_path),
            "source_path": str(source),
            "source_sha256": _hash_file(source),
            "derived_path": str(derived),
            "derived_sha256": _hash_file(derived),
            "row_count": 13,
            "images_manifest": images,
            "images_manifest_sha256": sha256_json(images),
        },
    )
    cohort["sources"]["derived_receipt"] = {"path": str(receipt_path), "sha256": _hash_file(receipt_path)}
    _write_json(cohort_path, cohort)
    manifest_path = cohort_path.with_name(cohort_path.name.replace(".json", ".manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cohort_sha256"] = _hash_file(cohort_path)
    manifest["cohort_content_sha256"] = sha256_json(cohort)
    manifest["source_hashes"]["derived_receipt"] = _hash_file(receipt_path)
    _write_json(manifest_path, manifest)
    return receipt_path


@pytest.mark.parametrize("deleted_field", ["source_sha256", "derived_path", "unit_id"])
def test_derived_receipt_required_bindings_reject_rehashed_deletion(tmp_path: Path, deleted_field: str) -> None:
    cohorts, _paths = _make_fixture(tmp_path)
    cohort_path = cohorts["S"]
    receipt_path = _install_derived_receipt(cohort_path)
    _load_cohort(cohort_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    del receipt[deleted_field]
    _write_json(receipt_path, receipt)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["sources"]["derived_receipt"]["sha256"] = _hash_file(receipt_path)
    _write_json(cohort_path, cohort)
    manifest_path = cohort_path.with_name(cohort_path.name.replace(".json", ".manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cohort_sha256"] = _hash_file(cohort_path)
    manifest["cohort_content_sha256"] = sha256_json(cohort)
    manifest["source_hashes"]["derived_receipt"] = _hash_file(receipt_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(AggregationError, match="derived_receipt"):
        _load_cohort(cohort_path)


def _h0_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    image_ids = tuple(str(value) for value in (1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 14038, 16228, 19642, 22969, 23754))
    source_hash, derived_hash = "1" * 64, "2" * 64
    payloads: dict[str, object] = {}
    sources: dict[str, object] = {}
    paths: dict[str, Path] = {}
    for checkpoint in ("S", "A"):
        path = tmp_path / f"strict-h0-{checkpoint}.json"
        records = [
            {
                "unit_id": UNIT_ID,
                "run_kind": "native_h0",
                "history_complete": True,
                "checkpoint": checkpoint,
                "config_fingerprint": f"config-{checkpoint}",
                "source_panel_sha256": source_hash,
                "derived_panel_sha256": derived_hash,
                "gt_owner_id": f"gt:{image_id}:0",
                "image_id": int(image_id),
                "native_tp": True,
                "native_fn": False,
                "strict_complete_row": True,
                "natural_boundary_valid": True,
                "natural_boundary": 0,
                "due_boundary_index": 0,
                "parse_status": "accepted",
                "decode_stop_reason": "im_end",
                "excludes_stop": True,
                "exact_prefix_sha256": sha256_json([]),
                "exact_prefix_token_ids": [],
                "covered_owner_ids": [],
            }
            for image_id in image_ids
        ]
        _write_json(
            path,
            {
                "schema_version": "static_dynamic_native_h0_owner_ledger.v1",
                "unit_id": UNIT_ID,
                "checkpoint": checkpoint,
                "config_fingerprint": f"config-{checkpoint}",
                "run_kind": "native_h0",
                "history_complete": True,
                "records": records,
            },
        )
        payloads[checkpoint] = {"payload": {"events": [{"image_id": 2299}]}}
        sources[checkpoint] = {
            "h0_ledgers": {str(path): _hash_file(path)},
            "source_panel": source_hash,
            "derived_panel": derived_hash,
            "source_panel_image_ids": sorted(image_ids, key=int),
        }
        paths[checkpoint] = path
    return payloads, sources, paths


def test_h0_exact_source_panel_set_rejects_same_cross_checkpoint_mutation(tmp_path: Path) -> None:
    payloads, sources, paths = _h0_fixture(tmp_path)
    assert len(_collect_h0_baselines(cohort_payloads=payloads, source_hashes=sources)[0]) == 26
    for checkpoint in ("S", "A"):
        ledger = json.loads(paths[checkpoint].read_text(encoding="utf-8"))
        ledger["records"][0]["image_id"] = 9999
        ledger["records"][0]["gt_owner_id"] = "gt:9999:0"
        _write_json(paths[checkpoint], ledger)
        sources[checkpoint]["h0_ledgers"] = {str(paths[checkpoint]): _hash_file(paths[checkpoint])}
    with pytest.raises(AggregationError, match="exact admitted source-panel"):
        _collect_h0_baselines(cohort_payloads=payloads, source_hashes=sources)


@pytest.mark.parametrize("deleted_field", ["exact_prefix_token_ids", "natural_boundary"])
def test_h0_missing_prefix_or_boundary_proof_rejects(tmp_path: Path, deleted_field: str) -> None:
    payloads, sources, paths = _h0_fixture(tmp_path)
    ledger = json.loads(paths["S"].read_text(encoding="utf-8"))
    del ledger["records"][0][deleted_field]
    _write_json(paths["S"], ledger)
    sources["S"]["h0_ledgers"] = {str(paths["S"]): _hash_file(paths["S"])}
    with pytest.raises(AggregationError, match="exact_prefix_token_ids|natural_boundary"):
        _collect_h0_baselines(cohort_payloads=payloads, source_hashes=sources)


def test_raw_partition_revalidates_exact_cohort_modulo_ownership() -> None:
    payloads = {
        checkpoint: {
            "payload": {
                "events": [
                    {"gt_owner_id": f"gt:{index + 1}:{index}", "image_id": index + 1}
                    for index in range(8)
                ]
            }
        }
        for checkpoint in ("S", "A")
    }
    rows = [
        {
            "checkpoint": checkpoint,
            "event_id": f"gt:{index + 1}:{index}",
            "image_id": str(index + 1),
            "shard": {"index": index % 4},
        }
        for checkpoint in ("S", "A")
        for index in range(8)
    ]
    _validate_bundle_partition(rows, cohort_payloads=payloads)
    for mutation in ("remove", "duplicate", "foreign"):
        changed = json.loads(json.dumps(rows))
        if mutation == "remove":
            changed.pop(0)
        elif mutation == "duplicate":
            changed[0] = dict(changed[4])
        else:
            changed[0]["event_id"] = "gt:foreign"
        with pytest.raises(AggregationError, match="ownership|repeats"):
            _validate_bundle_partition(changed, cohort_payloads=payloads)


def _verified_pair_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    h0_hash = sha256_json([])
    mapping_rows = [
        {"owner_id": "gt:1:1", "source_index": 1},
        {"owner_id": "gt:1:2", "source_index": 2},
    ]
    event = {
        "gt_owner_id": "gt:1:2",
        "A_B": {
            "S": {
                "pair_status": "verified_pair",
                "A_latest_covered": {
                    "gt_owner_id": "gt:1:1",
                    "source_panel_object_index": 1,
                    "natural_boundary": 1,
                    "strict_complete_row": True,
                },
                "B_verified_uncovered": {
                    "gt_owner_id": "gt:1:2",
                    "verified_support": True,
                    "strict_complete_row": False,
                    "natural_boundary": 2,
                    "exact_prefix_sha256": h0_hash,
                },
            }
        },
    }
    prefix = {
        "target_owner_id": "gt:1:2",
        "natural_boundary": 2,
        "covered_owner_ids": ["gt:1:1"],
        "event_eligibility": {"status": "eligible", "pair_status": "verified_pair"},
        "h0": {"exact_generated_history_prefix_sha256": h0_hash},
        "owner_mapping": {"source_to_derived": mapping_rows},
    }
    eligibility = {"status": "eligible", "pair_status": "verified_pair", "actuators_called": True}
    return event, prefix, eligibility


@pytest.mark.parametrize("mutation", ["missing_a", "missing_b_hash", "covered_b", "invalid_eligibility", "false_actuator"])
def test_verified_pair_requires_full_a_b_prefix_and_actuator_contract(mutation: str) -> None:
    event, prefix, eligibility = _verified_pair_inputs()
    _event_pair_contract(event, checkpoint="S", prefix=prefix, event_eligibility=eligibility)
    if mutation == "missing_a":
        del event["A_B"]["S"]["A_latest_covered"]
    elif mutation == "missing_b_hash":
        del event["A_B"]["S"]["B_verified_uncovered"]["exact_prefix_sha256"]
    elif mutation == "covered_b":
        prefix["covered_owner_ids"].append("gt:1:2")
    elif mutation == "invalid_eligibility":
        prefix["event_eligibility"]["status"] = "invalid/uninterpretable"
    else:
        eligibility["actuators_called"] = False
    with pytest.raises(AggregationError):
        _event_pair_contract(event, checkpoint="S", prefix=prefix, event_eligibility=eligibility)


def test_runner_shaped_verified_event_derives_and_malformed_bindings_hold() -> None:
    cohort_event, pair_prefix, eligibility = _verified_pair_inputs()
    target_owner = "gt:1:2"
    prefix = {
        **pair_prefix,
        "model_input": {"prefix_sha256": sha256_json([151646])},
    }
    p1_arms = {
        probe: _endpoint_row(target_owner, target_owner=target_owner)
        for probe in (
            "K00", "K01", "K10", "K11", "K12", "K13",
            *(f"{arm}_block{block}" for block in (13, 23, 27) for arm in ("R00", "R10", "R11", "R12")),
        )
    }
    p2 = {
        arm: {
            "horizon_1": _strict_horizon([target_owner], requested=1),
            "horizon_3": _strict_horizon([target_owner] * 3, requested=3),
        }
        for arm in ("D00", "D01", "D10", "D11", "D12", "D20")
    }
    p2["D21"] = {"status": "not_applicable", "reason": "same-parent donor is unavailable"}
    p3 = {
        cell: {
            "horizon_1": _strict_horizon([target_owner], requested=1),
            "horizon_3": _strict_horizon([target_owner] * 3, requested=3),
        }
        for cell in ("Y00", "Y10", "Y01", "Y11")
    }
    raw = {
        "checkpoint": "S",
        "event_id": target_owner,
        "image_id": 1,
        "eligibility": eligibility,
        "prefix": prefix,
        "p1": {"status": "attempted", "arms": p1_arms},
        "p2": {"status": "attempted", "arms": p2},
        "p3": {"status": "attempted", "cells": p3},
        "p4": _gradient_receipt(),
    }
    loaded = {
        "checkpoint": "S",
        "event_id": target_owner,
        "image_id": "1",
        "row": raw,
        "identity": {
            "runtime": {},
            "exact_prefix": {"identity": {}, "events": [{"event_id": target_owner, "prefix": prefix}]},
            "terminal": {"status": "completed"},
            "refs": {},
        },
        "shard": {"index": 0, "count": 1},
    }
    derived, _missing = _derive_bundle_event(
        loaded,
        cohort_event=cohort_event,
        support_rows={},
    )
    assert derived["matrix_status"] == "scored_candidate"
    assert derived["actuators_called"] is True
    assert derived["cohort"]["accepted_source_specific_match"] is True

    false_actuator = json.loads(json.dumps(loaded))
    false_actuator["row"]["eligibility"]["actuators_called"] = False
    with pytest.raises(AggregationError, match="actuators_called"):
        _derive_bundle_event(false_actuator, cohort_event=cohort_event, support_rows={})

    missing_mapping = json.loads(json.dumps(loaded))
    missing_mapping["row"]["prefix"]["owner_mapping"]["source_to_derived"] = []
    with pytest.raises(AggregationError, match="panel-bound"):
        _derive_bundle_event(missing_mapping, cohort_event=cohort_event, support_rows={})

    wrong_checkpoint = json.loads(json.dumps(loaded))
    wrong_checkpoint["row"]["p1"]["arms"]["K00"]["endpoint_evidence"]["identity_binding"]["checkpoint"] = None
    held, missing = _derive_bundle_event(
        wrong_checkpoint,
        cohort_event=cohort_event,
        support_rows={},
    )
    assert held["stage_readiness"]["p1"]["status"] == "hold"
    assert any("identity checkpoint/image mismatch" in reason for reason in missing)


def test_nested_runner_stop_is_required_and_external_support_cannot_substitute() -> None:
    stopped = _strict_horizon(["gt:1:2"], requested=3, stopped=True)
    qualified = _horizon_qualification(
        stopped,
        context="nested.stop",
        checkpoint="S",
        event_id="gt:1:2",
        image_id="1",
        target_owner="gt:1:2",
        covered_owner_ids=(),
        expected_horizon=3,
        prefix_hashes={sha256_json([]), sha256_json([151646])},
        support_rows={("A", "1", "gt:1:2", 0, "0" * 64): [{"verified_support": False}]},
    )
    assert qualified["qualification"] == "qualified"
    assert qualified["stop"]["reason"] == "runner_endpoint_remaining_support_empty"
    stopped["rows"][0]["endpoint_evidence"]["remaining_independently_verified_support_at_stop"] = {
        "status": "not_measured",
        "owner_ids": [],
        "count": 0,
        "at_stop": True,
    }
    held = _horizon_qualification(
        stopped,
        context="nested.stop",
        checkpoint="S",
        event_id="gt:1:2",
        image_id="1",
        target_owner="gt:1:2",
        covered_owner_ids=(),
        expected_horizon=3,
        prefix_hashes={sha256_json([]), sha256_json([151646])},
        support_rows={("S", "1", "gt:1:2", 0, sha256_json([])): [{"verified_support": False}]},
    )
    assert held["qualification"] == "technical_invalid"


@pytest.mark.parametrize("mutation", ["missing_rows", "empty_bookkeeping", "negative_parse", "fabricated_g"])
def test_nested_horizon_missing_or_fabricated_receipts_hold(mutation: str) -> None:
    raw = _strict_horizon(["gt:1:2"], requested=1)
    if mutation == "missing_rows":
        del raw["rows"]
    elif mutation == "empty_bookkeeping":
        raw["owner_bookkeeping"] = {}
    elif mutation == "negative_parse":
        raw["owner_bookkeeping"]["parse"]["valid_rows"] = -1
    else:
        raw["owner_bookkeeping"]["G"] = ["gt:fabricated"]
    held = _horizon_qualification(
        raw,
        context="nested.invalid",
        checkpoint="S",
        event_id="gt:1:2",
        image_id="1",
        target_owner="gt:1:2",
        covered_owner_ids=(),
        expected_horizon=1,
        prefix_hashes={sha256_json([]), sha256_json([151646])},
        support_rows={},
    )
    assert held["qualification"] == "technical_invalid"
    assert held["missing_evidence"]


def test_p1_target_hit_requires_exact_source_specific_physical_owner() -> None:
    event, _cohort_prefix, _eligibility = _verified_pair_inputs()
    prefix = {
        "h0": {"exact_generated_history_prefix_sha256": sha256_json([])},
        "model_input": {"prefix_sha256": sha256_json([151646])},
        "covered_owner_ids": ["gt:1:1"],
    }
    arms = {
        probe: _endpoint_row("gt:1:1", target_owner="gt:1:2", duplicate=True)
        for probe in ("K00", "K01", "K10", "K11", "K12", "K13")
    }
    arms["K10"] = _endpoint_row("gt:1:2", target_owner="gt:1:2")
    for block in (13, 23, 27):
        for arm in ("R00", "R10", "R11", "R12"):
            arms[f"{arm}_block{block}"] = _endpoint_row("gt:1:1", target_owner="gt:1:2", duplicate=True)
    raw_event = {"checkpoint": "S", "event_id": "gt:1:2", "image_id": 1, "prefix": prefix, "p1": {"status": "attempted", "arms": arms}}
    pair = {"accepted_source_specific_match": True, "target_owner_id": "gt:1:2"}
    rows, missing = _p1_bundle_rows(raw_event, pair=pair, context="p1.strict")
    assert rows["K10"]["target_hit"] is True
    assert missing == []
    arms["K10"]["owner_match"]["source_specific"] = False
    rows, missing = _p1_bundle_rows(raw_event, pair=pair, context="p1.strict")
    assert rows["K10"]["target_hit"] is None
    assert rows["K10"]["qualification"] == "technical_invalid"
    assert missing


@pytest.mark.parametrize("mutation", ["mass_finite_false", "non_target_not_measured", "non_target_missing_gradient", "missing_lm_head", "gradient_nonfinite"])
def test_p4_requires_measured_finite_mass_non_target_lm_head_and_gradients(mutation: str) -> None:
    p4 = _qualified_p4()
    assert _p4_qualification(p4, context="p4.strict")[0]["qualification"] == "qualified"
    if mutation == "mass_finite_false":
        p4["grammar_stop_invalid_mass"]["grammar"]["finite"] = False
    elif mutation == "non_target_not_measured":
        p4["non_target_owner_effects"]["fixed_sum_coupled"] = {"status": "not_measured", "owners": {"x": {"finite": True}}}
    elif mutation == "non_target_missing_gradient":
        p4["non_target_owner_effects"]["fixed_sum_coupled"]["owners"]["gt:non-target"]["present"] = False
    elif mutation == "missing_lm_head":
        del p4["lm_head"]["fixed_sum_coupled"]
    else:
        p4["objectives"]["fixed_sum_coupled"]["gradients"]["image_residual"]["finite"] = False
    readiness, missing = _p4_qualification(p4, context="p4.strict")
    assert readiness["readiness"] == "hold"
    assert missing


def test_any_eligible_unqualified_stage_and_empty_census_force_explicit_hold() -> None:
    ready_stages = {stage: {"status": "ready", "reasons": []} for stage in ("p1", "p2", "p3", "p4")}
    by_checkpoint = {
        checkpoint: [
            {
                "event_id": f"gt:{checkpoint}:1",
                "cohort": {"accepted_source_specific_match": True},
                "stage_readiness": json.loads(json.dumps(ready_stages)),
            }
        ]
        for checkpoint in ("S", "A")
    }
    by_checkpoint["S"][0]["stage_readiness"]["p2"] = {"status": "hold", "reasons": ["technical_invalid horizon"]}
    census = {"S": [{"row_count": 1}], "A": [{"row_count": 0}]}
    readiness, missing = _build_evidence_readiness(
        by_checkpoint=by_checkpoint,
        census_by_checkpoint=census,
        missing_evidence=[],
    )
    assert readiness["overall"] == "hold"
    assert readiness["p2"]["S"]["status"] == "hold"
    assert readiness["p1"]["A"]["status"] == "hold"
    assert any("technical_invalid horizon" in item for item in missing)
    assert any("census has no rows" in item for item in missing)


def test_bundle_finalizer_has_no_model_runtime_import_or_call() -> None:
    source = Path("scripts/research/aggregate_static_dynamic_owner_interface_shards.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from transformers" not in source
    assert "model.generate" not in source
