from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.research import finalize_s_k10_h20_crossover as finalizer
from scripts.research import seal_s_k10_h20_crossover_finalization_receipt as successor_sealer


REAL_GATE_RESULT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "s-k-n-h-execution-native-fn-supersession-v2/shard-002/"
    "event-000002-gt-2299-29/result.json"
)

REAL_S_V2_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "s-k-n-h-execution-native-fn-supersession-v2"
)
REAL_S_V2_EVENTS = (
    (2, "gt:2299:29", "shard-002", "event-000002-gt-2299-29"),
    (5, "gt:13348:14", "shard-005", "event-000005-gt-13348-14"),
    (8, "gt:16228:15", "shard-000", "event-000008-gt-16228-15"),
)


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(finalizer.canonical_json_bytes(value) + b"\n")
    return path


def _self(value: dict[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    result = dict(value)
    body = dict(result)
    body.pop(field, None)
    result[field] = finalizer.sha256_json(body)
    return result


def _layer_attestation() -> dict[str, Any]:
    return {
        "passed": True,
        "layer_count": 28,
        "layer_indices": list(range(28)),
        "missing_layers": [],
        "repeated_layers": [],
        "errors": [],
    }


def _scalar_unattested_placeholder() -> dict[str, Any]:
    """The production scalar receipt emitted before a model forward."""

    return {
        "exact_same_tensor_all_layers_required": True,
        "required": True,
        "schema_version": "natural_boundary_attention_actuators.v1.layer_consumption.v1",
        "status": "unattested",
    }


def _owner_row(
    row_index: int,
    owner: str | None,
    *,
    covered: list[str],
    seen_before: list[str],
    opener: int,
    token: int,
) -> dict[str, Any]:
    strict = owner is not None
    row_tokens = [opener, token]
    match = (
        {
            "status": "unique",
            "owner_id": owner,
            "source_specific": True,
            "physical_match": True,
        }
        if strict
        else {
            "status": "unmatched",
            "owner_id": None,
            "source_specific": False,
            "physical_match": False,
        }
    )
    row_book = {
        "covered_owner_ids_before": list(covered),
        "seen_owner_ids_before": list(seen_before),
        "matched_owner_id": owner,
        "strict_physical_owner_match": strict,
        "covered_repeat": bool(strict and owner in covered),
        "duplicate": False,
        "new_target_owner": bool(strict and owner not in covered),
    }
    return {
        "row_index": row_index,
        "status": "closure",
        "stop_reason": "closure",
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "opener_generated_by_model": True,
        "row_started": True,
        "initial_prefix_last_token_id": 13,
        "first_generated_token_id": opener,
        "opener_token_id": opener,
        "token_ids": row_tokens,
        "token_ids_sha256": finalizer.sha256_json(row_tokens),
        "owner_match": match,
        "owner_match_status": match["status"],
        "owner_bookkeeping": row_book,
    }


def _raw(
    event: Mapping[str, Any],
    arm: str,
    *,
    owners: list[str | None],
    technical: bool = False,
    c11: bool = False,
) -> dict[str, Any]:
    opener = 90
    prefix_tokens = [10, 11, 12, 13]
    exact_history = [1, 2, 3]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    covered = [f"covered:{event['event_id']}"]
    for row_index, owner in enumerate(owners):
        row = _owner_row(
            row_index,
            owner,
            covered=covered,
            seen_before=sorted(set(covered) | seen),
            opener=opener,
            token=100 + row_index,
        )
        rows.append(row)
        if owner is not None:
            seen.add(owner)
    generated = [token for row in rows for token in row["token_ids"]]
    parse = {
        "valid_rows": 3,
        "duplicate_rows": 0,
        "unmatched_rows": sum(owner is None for owner in owners),
        "ambiguous_rows": 0,
        "malformed_rows": 0,
        "invalid_rows": 0,
    }
    stop = {"stopped": False, "stop_reason": "closure"}
    for row in rows:
        row["owner_bookkeeping"]["parse"] = dict(parse)
        row["owner_bookkeeping"]["stop"] = dict(stop)
        row["owner_bookkeeping"]["horizon_status"] = "closure"
    row_entry = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": opener,
        "opener_generated_by_model": True,
        "opener_injected": False,
        "row_started": True,
    }
    for row in rows:
        row["owner_bookkeeping"]["row_entry"] = dict(row_entry)
    runtime_receipts: list[dict[str, Any]] = []
    scalar_receipts: list[dict[str, Any]] = []
    mrope_seed = f"mrope:{event['event_id']}"
    for step in range(len(generated)):
        ids = prefix_tokens + generated[:step]
        receipt = {
            "step": step,
            "sequence_length": len(ids),
            "input_ids_sha256": finalizer.sha256_json(ids),
            "input_ids": list(ids),
            "mrope_hash": finalizer.sha256_json([mrope_seed, step]),
            "use_cache": False,
        }
        if not technical:
            receipt["layer_consumption_attestation"] = _layer_attestation()
        runtime_receipts.append(dict(receipt))
        scalar_receipts.append(
            {
                "step": step,
                "input_ids": list(ids),
                "input_ids_sha256": finalizer.sha256_json(ids),
                "use_cache": False,
            }
        )
    raw: dict[str, Any] = {
        "arm_id": arm,
        "event_id": event["event_id"],
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "opener_seeded": False,
        "opener_generated_by_model": True,
        "opener_token_id": opener,
        "first_generated_token_id": opener,
        "initial_prefix_last_token_id": prefix_tokens[-1],
        "native_stop_token_ids": [999],
        "prefix": {
            "event_id": event["event_id"],
            "exact_history_token_ids": exact_history,
            "prefix_token_ids": prefix_tokens,
            "prefix_token_ids_sha256": finalizer.sha256_json(prefix_tokens),
        },
        "prefix_identity": {
            "natural_prefix_token_ids": prefix_tokens,
            "natural_prefix_sha256": finalizer.sha256_json(prefix_tokens),
        },
        "no_cache_scalar_recompute": True,
        "rows": rows,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": finalizer.sha256_json(generated),
        "terminal_reason": "closure",
        "owner_bookkeeping": {
            "covered_owner_ids": list(covered),
            "raw_endpoint_owner_ids": sorted({owner for owner in owners if owner is not None}),
            "covered_repeat_owner_ids": [],
            "new_target_owner_ids": sorted({owner for owner in owners if owner is not None} - set(covered)),
            "parse": parse,
            "stop": stop,
            "horizon_status": "closure",
            "row_entry": row_entry,
            "row_count": 3,
            "strict_physical_owner_match_count": sum(owner is not None for owner in owners),
            "unmatched_count": parse["unmatched_rows"],
        },
        "covered_owner_ids": list(covered),
        "target_owner_id": event["target_owner_id"],
        "scalar_forward_count": len(scalar_receipts),
        "runtime_scalar_forward_count": len(runtime_receipts),
        "scalar_receipts": scalar_receipts,
        "runtime_scalar_receipts": runtime_receipts,
    }
    if not technical:
        raw["full_logit_parity"] = {
            "passed": True,
            "per_forward_max_abs_delta": 0.0,
            "tolerance": 1e-4,
        }
    if c11:
        raw["composition_receipt"] = {
            "arm_id": "C11",
            "cell_id": "C11",
            "status": "ready",
            "component_order": ["K10", "H20"],
            "children": [{"arm_id": "K10"}, {"arm_id": "H20"}],
        }
        raw["transport_receipt"] = {
            "arm_id": "K10",
            "transport_arm_id": "K10",
            "status": "ready",
            "use_cache": False,
            "opener_injected": False,
        }
    return raw


def _first_token_outcome(
    event: Mapping[str, Any],
    *,
    token: int,
    terminal: str,
) -> dict[str, Any]:
    """Build one exact natural-boundary row for STOP or invalid-first outcomes."""

    raw = _raw(event, "K10", owners=[None, None, None])
    prefix_last = raw["initial_prefix_last_token_id"]
    opener = raw["opener_token_id"]
    row_entry = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": token,
        "opener_generated_by_model": False,
        "opener_injected": False,
        "row_started": False,
    }
    row_book = {
        "covered_owner_ids_before": list(raw["covered_owner_ids"]),
        "seen_owner_ids_before": list(raw["covered_owner_ids"]),
        "matched_owner_id": None,
        "strict_physical_owner_match": False,
        "covered_repeat": False,
        "duplicate": False,
        "new_target_owner": False,
        "horizon_status": terminal,
        "row_entry": dict(row_entry),
        "parse": {
            "valid_rows": 0,
            "duplicate_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "malformed_rows": 0,
            "invalid_rows": int(terminal == "invalid"),
        },
        "stop": {"stopped": True, "stop_reason": terminal},
    }
    row = {
        "row_index": 0,
        "status": terminal,
        "stop_reason": terminal,
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "opener_generated_by_model": False,
        "row_started": False,
        "initial_prefix_last_token_id": prefix_last,
        "first_generated_token_id": token,
        "opener_token_id": opener,
        "token_ids": [token],
        "token_ids_sha256": finalizer.sha256_json([token]),
        "owner_bookkeeping": row_book,
    }
    parse = dict(row_book["parse"])
    stop = dict(row_book["stop"])
    raw.update(
        {
            "opener_generated_by_model": False,
            "first_generated_token_id": token,
            "generated_token_ids": [token],
            "generated_token_ids_sha256": finalizer.sha256_json([token]),
            "terminal_reason": terminal,
            "stop_reason": terminal,
            "rows": [row],
            "scalar_forward_count": 1,
            "runtime_scalar_forward_count": 1,
            "scalar_receipts": raw["scalar_receipts"][:1],
            "runtime_scalar_receipts": raw["runtime_scalar_receipts"][:1],
            "owner_bookkeeping": {
                "covered_owner_ids": list(raw["covered_owner_ids"]),
                "raw_endpoint_owner_ids": [],
                "covered_repeat_owner_ids": [],
                "new_target_owner_ids": [],
                "row_entry": dict(row_entry),
                "parse": parse,
                "stop": stop,
                "horizon_status": terminal,
                "row_count": 1,
                "strict_physical_owner_match_count": 0,
                "unmatched_count": 0,
            },
        }
    )
    return raw


def _refresh_generated_trajectory(raw: dict[str, Any], generated: list[int]) -> None:
    prefix = list(raw["prefix"]["prefix_token_ids"])
    for key, runtime in (("scalar_receipts", False), ("runtime_scalar_receipts", True)):
        templates = list(raw[key])
        while len(templates) < len(generated):
            template = copy.deepcopy(templates[-1])
            if "mrope_hash" in template:
                template["mrope_hash"] = finalizer.sha256_json(
                    ["terminal-location-fixture", len(templates)]
                )
            templates.append(template)
        receipts: list[dict[str, Any]] = []
        for step in range(len(generated)):
            receipt = copy.deepcopy(templates[step])
            ids = prefix + generated[:step]
            receipt["step"] = step
            receipt["input_ids"] = ids
            receipt["input_ids_sha256"] = finalizer.sha256_json(ids)
            if runtime:
                receipt["sequence_length"] = len(ids)
            receipts.append(receipt)
        raw[key] = receipts
    raw["scalar_forward_count"] = len(generated)
    raw["runtime_scalar_forward_count"] = len(generated)
    raw["generated_token_ids"] = list(generated)
    raw["generated_token_ids_sha256"] = finalizer.sha256_json(generated)


def _refresh_terminal_bookkeeping(
    raw: dict[str, Any],
    *,
    terminal: str,
    parse: Mapping[str, int],
) -> None:
    covered = list(raw["covered_owner_ids"])
    covered_set = set(covered)
    seen: set[str] = set()
    strict_sequence: list[str] = []
    row_entry = {
        "admission_mode": "pre_opener_natural",
        "first_generated_token_id": raw["first_generated_token_id"],
        "opener_generated_by_model": raw["opener_generated_by_model"],
        "opener_injected": False,
        "row_started": raw["opener_generated_by_model"],
    }
    stop = {"stopped": terminal != "closure", "stop_reason": terminal}
    for row in raw["rows"]:
        match = row.get("owner_match")
        owner = match.get("owner_id") if isinstance(match, Mapping) else None
        strict = bool(
            row.get("status") == "closure"
            and isinstance(match, Mapping)
            and match.get("status") in {"unique", "matched"}
            and match.get("source_specific") is True
            and match.get("physical_match") is True
            and owner is not None
        )
        duplicate = bool(strict and owner in seen)
        row["owner_bookkeeping"] = {
            "covered_owner_ids_before": list(covered),
            "seen_owner_ids_before": sorted(covered_set | seen),
            "matched_owner_id": owner if strict else None,
            "strict_physical_owner_match": strict,
            "covered_repeat": bool(strict and owner in covered_set),
            "duplicate": duplicate,
            "new_target_owner": bool(strict and owner not in covered_set),
            "horizon_status": terminal,
            "row_entry": dict(row_entry),
            "parse": dict(parse),
            "stop": dict(stop),
        }
        if strict:
            strict_sequence.append(str(owner))
            seen.add(str(owner))
    strict_set = set(strict_sequence)
    raw["owner_bookkeeping"] = {
        "covered_owner_ids": list(covered),
        "raw_endpoint_owner_ids": sorted(strict_set),
        "covered_repeat_owner_ids": sorted(strict_set & covered_set),
        "new_target_owner_ids": sorted(strict_set - covered_set),
        "row_entry": row_entry,
        "parse": dict(parse),
        "stop": stop,
        "horizon_status": terminal,
        "row_count": len(raw["rows"]),
        "strict_physical_owner_match_count": len(strict_sequence),
        "duplicate_count": int(parse["duplicate_rows"]),
        "unmatched_count": int(parse["unmatched_rows"]),
    }


def _terminal_location_outcome(event: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "within_row_stop":
        raw = _raw(event, "K10", owners=[None, None, None])
        raw["rows"] = raw["rows"][:1]
        row = raw["rows"][0]
        row["token_ids"] = [raw["opener_token_id"], 100, 999]
        row["token_ids_sha256"] = finalizer.sha256_json(row["token_ids"])
        row["status"] = "native_stop"
        row["stop_reason"] = "native_stop"
        row.pop("owner_match", None)
        row.pop("owner_match_status", None)
        terminal = "native_stop"
        parse = {
            "valid_rows": 0,
            "duplicate_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "malformed_rows": 0,
            "invalid_rows": 0,
        }
        generated = list(row["token_ids"])
    elif kind == "lookahead_stop":
        raw = _raw(event, "K10", owners=["owner:0", "owner:1", None])
        raw["rows"] = raw["rows"][:2]
        terminal = "native_stop"
        parse = {
            "valid_rows": 2,
            "duplicate_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "malformed_rows": 0,
            "invalid_rows": 0,
        }
        generated = [token for row in raw["rows"] for token in row["token_ids"]] + [999]
    elif kind == "over_continuation":
        raw = _raw(event, "K10", owners=["owner:0", None, None])
        raw["rows"] = raw["rows"][:1]
        row = raw["rows"][0]
        row["status"] = "over_continuation"
        row["reason"] = "token_after_complete_row"
        row["stop_reason"] = "over_continuation"
        row["over_continuation"] = {"selected_token_id": 777}
        row.pop("owner_match", None)
        row.pop("owner_match_status", None)
        terminal = "over_continuation"
        parse = {
            "valid_rows": 0,
            "duplicate_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "malformed_rows": 1,
            "invalid_rows": 1,
        }
        generated = list(row["token_ids"]) + [777]
    else:  # pragma: no cover - test helper call sites are frozen below
        raise AssertionError(f"unknown terminal-location fixture {kind}")

    raw["terminal_reason"] = terminal
    raw["stop_reason"] = terminal
    _refresh_generated_trajectory(raw, generated)
    _refresh_terminal_bookkeeping(raw, terminal=terminal, parse=parse)
    return raw


def _fixture(tmp_path: Path, *, unmatched: bool = False) -> tuple[Path, Path, list[Path]]:
    source = _write(tmp_path / "source.json", {"source": "fixture"})
    source_ref = {"path": str(source), "sha256": finalizer.sha256_file(source)}
    events: list[dict[str, Any]] = []
    for local, event_index in enumerate(finalizer.EVENT_INDICES):
        event_id = finalizer.EVENT_IDS[local]
        events.append(
            {
                "event_index": event_index,
                "event_id": event_id,
                "image_id": 100 + local,
                "event_sha256": f"{local + 1:064x}",
                "target_owner_id": f"target:{local}",
                "covered_owner_ids": [f"covered:{event_id}"],
                "prefix_sha256": finalizer.sha256_json([1, 2, 3]),
            }
        )
    plan = _self(
        {
            "schema_version": finalizer.PLAN_SCHEMA_VERSION,
            "status": "planned",
            "unit_id": finalizer.UNIT_ID,
            "primary": dict(finalizer.PRIMARY),
            "event_count": 3,
            "image_count": 3,
            "events": events,
            "cell_order": list(finalizer.CELL_ORDER),
            "technical_control": "C00",
            "cells": {
                "C00": {"source_arm": "K01"},
                "C10": {"source_arm": "K10"},
                "C01": {"source_arm": "H20"},
                "C11": {"source_arms": ["K10", "H20"]},
            },
            "source_bindings": {"source": source_ref},
            "device_plan": dict(finalizer.DEVICE_PLAN),
            "shards": [
                {
                    "shard_index": local,
                    "shard_id": f"shard-{local:03d}",
                    "physical_device": finalizer.DEVICE_PLAN[f"shard-{local:03d}"],
                    "logical_device": "cuda:0",
                    "events": [events[local]],
                }
                for local in range(3)
            ],
            "no_training": True,
            "use_cache": False,
        }
    )
    plan_path = _write(tmp_path / "plan.json", plan)
    code = _write(tmp_path / "runner.py", "runner\n")
    code_ref = {"path": str(code), "sha256": finalizer.sha256_file(code)}
    input_hash = finalizer.sha256_file(source)
    execution_root = tmp_path / "execution"
    final_root = tmp_path / "final"
    pre_gpu = _self(
        {
            "schema_version": finalizer.PRE_GPU_SCHEMA_VERSION,
            "status": "sealed_pre_gpu",
            "unit_id": finalizer.UNIT_ID,
            "no_training": True,
            "plan": {"path": str(plan_path), "sha256": finalizer.sha256_file(plan_path)},
            "plan_self_sha256": plan["self_sha256"],
            "roots": {
                "execution_root": {"path": str(execution_root), "status": "authorized_runtime_root"},
                "final_root": {"path": str(final_root), "status": "authorized_runtime_root"},
            },
            "source_selection": {
                "event_ids": list(finalizer.EVENT_IDS),
                "event_count": 3,
                "image_count": 3,
                "cell_order": list(finalizer.CELL_ORDER),
                "opener_injected": False,
                "use_cache": False,
            },
            "device_plan": dict(finalizer.DEVICE_PLAN),
            "event_bindings": [
                {key: event[key] for key in ("event_index", "event_id", "image_id", "event_sha256", "prefix_sha256")}
                for event in events
            ],
            "execution_policy": {key: True for key in ("no_event_reorder", "no_reselection", "no_sweep", "no_a3", "no_p4", "no_training", "at_most_once")},
            "runtime": {"no_training": True, "gpu_used": False, "model_loaded": False},
            "input_bindings": {"source": source_ref},
            "input_hashes": {"source": input_hash},
            "source_files": {"runner": code_ref},
        }
    )
    pre_path = _write(tmp_path / "pre_gpu.json", pre_gpu)
    roots: list[Path] = []
    for local, event in enumerate(events):
        shard_id = f"shard-{local:03d}"
        root = execution_root / shard_id
        root.mkdir(parents=True)
        roots.append(root)
        owner_rows = [f"base:{local}", f"middle:{local}", event["target_owner_id"]]
        raw_cells: dict[str, dict[str, Any]] = {}
        for cell, arm in finalizer.TRANSPORT_ARMS.items():
            cell_owners = list(owner_rows)
            if unmatched and cell == "C10" and local == 0:
                cell_owners[1] = None
            raw_cells[cell] = {
                "cell_id": cell,
                "transport_arm_id": arm,
                "result": _raw(event, arm, owners=cell_owners, c11=cell == "C11"),
            }
            if cell == "C11":
                raw_cells[cell]["composition_arm_id"] = "C11"
        technical_raw = _raw(event, "K00", owners=owner_rows, technical=True)
        result_body: dict[str, Any] = {
            "schema_version": finalizer.EVENT_SCHEMA_VERSION,
            "status": "completed",
            "unit_id": finalizer.UNIT_ID,
            "shard_id": shard_id,
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_index": event["event_index"],
            "plan_sha256": plan["self_sha256"],
            "plan_self_sha256": plan["self_sha256"],
            "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
            "source_event_sha256": event["event_sha256"],
            "runtime_identity_sha256": None,
            "technical_control": {"K00": {"transport_arm_id": "K00", "result": technical_raw}},
            "cells": raw_cells,
        }
        assignment = finalizer._expected_device_assignment(shard_id, event)
        runtime = _self(
            {
                "schema_version": finalizer.RUNTIME_SCHEMA_VERSION,
                "status": "completed",
                "unit_id": finalizer.UNIT_ID,
                "shard_id": shard_id,
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "device": {"logical_device": "cuda:0", "physical_device": finalizer.DEVICE_PLAN[shard_id]},
                "code_hashes": {"runner": code_ref["sha256"]},
                "input_hashes": {"source": input_hash},
                "plan_sha256": plan["self_sha256"],
                "plan_self_sha256": plan["self_sha256"],
                "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
                "device_assignment": assignment,
            }
        )
        result_body["runtime_identity_sha256"] = runtime["self_sha256"]
        result = _self(result_body, field="result_sha256")
        terminal_cells = {}
        for cell, envelope in raw_cells.items():
            raw = envelope["result"]
            terminal_cells[cell] = {
                "cell_id": cell,
                "transport_arm_id": envelope["transport_arm_id"],
                "composition_arm_id": "C11" if cell == "C11" else None,
                "admission_mode": raw["admission_mode"],
                "opener_injected": False,
                "terminal_reason": raw["terminal_reason"],
                "stop_reason": raw.get("stop_reason"),
                "raw_result_sha256": finalizer.sha256_json(raw),
            }
        terminal = _self(
            {
                "schema_version": finalizer.TERMINAL_SCHEMA_VERSION,
                "status": "completed",
                "unit_id": finalizer.UNIT_ID,
                "shard_id": shard_id,
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "result_sha256": result["result_sha256"],
                "cells": terminal_cells,
            }
        )
        result_path = _write(root / "result.json", result)
        runtime_path = _write(root / "runtime_identity.json", runtime)
        terminal_path = _write(root / "terminal_summary.json", terminal)
        aggregate = _self(
            {
                "schema_version": finalizer.SHARD_RECEIPT_SCHEMA_VERSION,
                "status": "completed",
                "unit_id": finalizer.UNIT_ID,
                "shard_id": shard_id,
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "plan_sha256": plan["self_sha256"],
                "plan_self_sha256": plan["self_sha256"],
                "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
                "result_sha256": result["result_sha256"],
                "runtime_identity_sha256": runtime["self_sha256"],
                "terminal_summary_self_sha256": terminal["self_sha256"],
                "result_raw_sha256": finalizer.sha256_file(result_path),
                "runtime_identity_raw_sha256": finalizer.sha256_file(runtime_path),
                "terminal_summary_raw_sha256": finalizer.sha256_file(terminal_path),
            }
        )
        _write(root / "aggregate.receipt.json", aggregate)
    return plan_path, pre_path, roots


def _real_s_v2_result_paths() -> list[Path]:
    return [
        REAL_S_V2_ROOT / shard / event_dir / "result.json"
        for _, _, shard, event_dir in REAL_S_V2_EVENTS
    ]


def _real_artifact_fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    """Wrap frozen S-v2 arms in temporary current-finalizer envelopes."""

    plan_path, pre_path, roots = _fixture(tmp_path)
    source_paths = _real_s_v2_result_paths()
    source_documents = [json.loads(path.read_text()) for path in source_paths]
    events: list[dict[str, Any]] = []
    for (event_index, event_id, _shard, _event_dir), document in zip(REAL_S_V2_EVENTS, source_documents):
        source_event = document["event"]
        natural_boundary = source_event["natural_boundary"]
        owner_refs = source_event["owner_refs"]
        events.append(
            {
                "event_index": event_index,
                "event_id": event_id,
                "image_id": source_event["image_id"],
                "event_sha256": document["event_sha256"],
                "target_owner_id": owner_refs["gt_owner_id"],
                "covered_owner_ids": list(owner_refs["covered_owner_ids"]),
                "prefix_sha256": natural_boundary["history_sha256"],
            }
        )

    plan = json.loads(plan_path.read_text())
    plan["events"] = events
    plan["source_bindings"] = {
        **plan["source_bindings"],
        **{
            f"frozen_result_{index}": {
                "path": str(path),
                "sha256": finalizer.sha256_file(path),
            }
            for index, path in enumerate(source_paths)
        },
    }
    for index, shard in enumerate(plan["shards"]):
        shard["events"] = [events[index]]
    plan = _self(plan)
    _write(plan_path, plan)

    pre_gpu = json.loads(pre_path.read_text())
    pre_gpu["plan"] = {
        "path": str(plan_path),
        "sha256": finalizer.sha256_file(plan_path),
    }
    pre_gpu["plan_self_sha256"] = plan["self_sha256"]
    pre_gpu["event_bindings"] = [
        {
            key: event[key]
            for key in ("event_index", "event_id", "image_id", "event_sha256", "prefix_sha256")
        }
        for event in events
    ]
    pre_gpu = _self(pre_gpu)
    _write(pre_path, pre_gpu)

    for index, (event, document, root) in enumerate(zip(events, source_documents, roots)):
        result_path = root / "result.json"
        runtime_path = root / "runtime_identity.json"
        terminal_path = root / "terminal_summary.json"
        aggregate_path = root / "aggregate.receipt.json"

        arms = document["arms"]
        k10 = copy.deepcopy(arms["K10"])
        c11 = copy.deepcopy(k10)
        c11["composition_receipt"] = {
            "arm_id": "C11",
            "cell_id": "C11",
            "status": "ready",
            "component_order": ["K10", "H20"],
            "children": [{"arm_id": "K10"}, {"arm_id": "H20"}],
        }
        c11["transport_receipt"] = {
            "arm_id": "K10",
            "transport_arm_id": "K10",
            "status": "ready",
            "use_cache": False,
            "opener_injected": False,
        }
        cells = {
            "C00": {
                "cell_id": "C00",
                "transport_arm_id": "K01",
                "result": copy.deepcopy(arms["K01"]),
            },
            "C10": {
                "cell_id": "C10",
                "transport_arm_id": "K10",
                "result": copy.deepcopy(arms["K10"]),
            },
            "C01": {
                "cell_id": "C01",
                "transport_arm_id": "H20",
                "result": copy.deepcopy(arms["H20"]),
            },
            "C11": {
                "cell_id": "C11",
                "transport_arm_id": "K10",
                "composition_arm_id": "C11",
                "result": c11,
            },
        }

        result = json.loads(result_path.read_text())
        result.update(
            {
                "shard_id": f"shard-{index:03d}",
                "event_id": event["event_id"],
                "image_id": event["image_id"],
                "event_index": event["event_index"],
                "plan_sha256": plan["self_sha256"],
                "plan_self_sha256": plan["self_sha256"],
                "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
                "source_event_sha256": event["event_sha256"],
                "technical_control": {
                    "K00": {
                        "transport_arm_id": "K00",
                        "result": copy.deepcopy(arms["K00"]),
                    }
                },
                "cells": cells,
            }
        )

        runtime = json.loads(runtime_path.read_text())
        runtime.update(
            {
                "shard_id": f"shard-{index:03d}",
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "plan_sha256": plan["self_sha256"],
                "plan_self_sha256": plan["self_sha256"],
                "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
                "device_assignment": finalizer._expected_device_assignment(
                    f"shard-{index:03d}", event
                ),
            }
        )
        runtime = _self(runtime)
        result["runtime_identity_sha256"] = runtime["self_sha256"]
        result = _self(result, field="result_sha256")

        terminal = json.loads(terminal_path.read_text())
        terminal.update(
            {
                "shard_id": f"shard-{index:03d}",
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "result_sha256": result["result_sha256"],
                "cells": {
                    cell: {
                        "cell_id": cell,
                        "transport_arm_id": envelope["transport_arm_id"],
                        "composition_arm_id": "C11" if cell == "C11" else None,
                        "admission_mode": envelope["result"]["admission_mode"],
                        "opener_injected": envelope["result"]["opener_injected"],
                        "terminal_reason": envelope["result"]["terminal_reason"],
                        "stop_reason": envelope["result"].get("stop_reason"),
                        "raw_result_sha256": finalizer.sha256_json(envelope["result"]),
                    }
                    for cell, envelope in cells.items()
                },
            }
        )
        terminal = _self(terminal)

        result_path = _write(result_path, result)
        runtime_path = _write(runtime_path, runtime)
        terminal_path = _write(terminal_path, terminal)
        aggregate = json.loads(aggregate_path.read_text())
        aggregate.update(
            {
                "shard_id": f"shard-{index:03d}",
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "plan_sha256": plan["self_sha256"],
                "plan_self_sha256": plan["self_sha256"],
                "pre_gpu_receipt_self_sha256": pre_gpu["self_sha256"],
                "result_sha256": result["result_sha256"],
                "runtime_identity_sha256": runtime["self_sha256"],
                "terminal_summary_self_sha256": terminal["self_sha256"],
                "result_raw_sha256": finalizer.sha256_file(result_path),
                "runtime_identity_raw_sha256": finalizer.sha256_file(runtime_path),
                "terminal_summary_raw_sha256": finalizer.sha256_file(terminal_path),
            }
        )
        _write(aggregate_path, _self(aggregate))
    return plan_path, pre_path, roots


def test_realistic_source_indices_devices_and_horizon_derivation(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    result = finalizer.finalize(plan, pre_gpu, roots)
    evidence = result["evidence"]
    assert [event["event_index"] for event in evidence["events"]] == [2, 5, 8]
    assert [event["image_id"] for event in evidence["events"]] == [100, 101, 102]
    assert evidence["source_specific_crossover_status"] == "qualified"
    assert evidence["component_contrasts"]["C10-C00_static"]["metrics"]["target_release"] == [0, 0, 0]
    utility = evidence["events"][0]["cells"]["C10"]["owner_utility"]["horizon_1"]
    assert utility["G"] == [] and utility["K"] == ["base:0"] and utility["L"] == []
    assert utility["net_charged"] == 0
    assert evidence["utilities"]["horizon_1"]["net_charged"]["tau"] == 0


def test_strict_then_unmatched_retains_seen_owner_and_keeps_descriptive_sets(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path, unmatched=True)
    result = finalizer.finalize(plan, pre_gpu, roots)
    endpoint = result["evidence"]["events"][0]["cells"]["C10"]
    assert endpoint["scientific_status"] == "unmatched"
    assert endpoint["mechanically_valid"] is True
    # The third row is still allowed to classify against the first strict owner;
    # an unmatched row is neutral to ``seen_owner_ids_before`` rather than
    # erasing that history.
    raw_rows = json.loads((roots[0] / "result.json").read_text())["cells"]["C10"]["result"]["rows"]
    assert raw_rows[1]["owner_bookkeeping"]["seen_owner_ids_before"] == ["base:0", "covered:gt:2299:29"]
    assert endpoint["owner_utility"]["horizon_3"]["treatment_unmatched_rows"] == 1
    assert result["evidence"]["component_tau"]["strict_count"]["mean"] is None
    assert result["evidence"]["utilities"]["horizon_3"]["net_charged"]["tau"] is None


def test_k00_without_mask_attestation_but_cells_require_exact_all_28(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    finalizer.finalize(plan, pre_gpu, roots)
    value = json.loads((roots[0] / "result.json").read_text())
    value["cells"]["C10"]["result"]["runtime_scalar_receipts"][0].pop("layer_consumption_attestation")
    _write(roots[0] / "result.json", _self(value, field="result_sha256"))
    with pytest.raises(finalizer.EvidenceContractError):
        finalizer.finalize(plan, pre_gpu, roots)


def test_scalar_unattested_placeholder_requires_matching_runtime_all_28() -> None:
    event = {
        "event_id": "gt:scalar-placeholder:0",
        "target_owner_id": "target:scalar-placeholder",
    }
    raw = _raw(event, "K10", owners=["owner:0", "owner:1", None])
    raw["scalar_receipts"][0]["attention_mask"] = {
        "receipt": {
            "all_layer_consumption_attestation": _scalar_unattested_placeholder(),
        }
    }

    parsed = finalizer._validate_natural_raw(raw, "K10", "scalar-placeholder.K10", event=event)
    assert parsed["scalar"]["scalar_forward_count"] == len(raw["scalar_receipts"])
    runtime_attestation = raw["runtime_scalar_receipts"][0]["layer_consumption_attestation"]
    assert runtime_attestation["passed"] is True
    assert runtime_attestation["layer_count"] == 28
    assert runtime_attestation["layer_indices"] == list(range(28))

    invalid = copy.deepcopy(raw)
    invalid["runtime_scalar_receipts"][0]["layer_consumption_attestation"]["passed"] = False
    with pytest.raises(
        finalizer.EvidenceContractError,
        match=r"runtime_scalar_receipts\[0\] all-layer consumption did not pass",
    ):
        finalizer._validate_natural_raw(invalid, "K10", "scalar-placeholder.K10", event=event)

    for field, value in (
        ("layer_count", 27),
        ("layer_indices", list(range(27))),
    ):
        invalid = copy.deepcopy(raw)
        invalid["runtime_scalar_receipts"][0]["layer_consumption_attestation"][field] = value
        with pytest.raises(
            finalizer.EvidenceContractError,
            match=r"runtime_scalar_receipts\[0\] does not attest exactly layers 0\.\.27",
        ):
            finalizer._validate_natural_raw(invalid, "K10", "scalar-placeholder.K10", event=event)


def _real_construction_placeholders() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the real base and composed C11 pre-forward placeholders.

    The finalizer is standard-library only, so it restates these producer
    shapes.  Building them here from the real actuator module is what keeps the
    restatement pinned to ``compose_k10_h20``.
    """

    import torch

    from scripts.research import natural_boundary_attention_actuators as attention

    common = {
        "image_key_positions": (1, 2, 3),
        "b_exclusive_positions": (2,),
        "latest_row_key_positions": (5, 6),
        "layer_count": 28,
        "head_count": 16,
        "device": "cpu",
        "dtype": torch.bool,
    }
    k10 = attention.build_scalar_step_factory("K10", **common).build(
        8, query_position=7, device="cpu"
    )
    h20 = attention.build_scalar_step_factory("H20", **common).build(
        8, query_position=7, device="cpu"
    )
    composed = attention.compose_k10_h20(k10, h20, cell_id="C11")
    base = dict(k10.receipt()["layer_consumption_attestation"])
    c11 = dict(composed.receipt()["layer_consumption_attestation"])
    assert len(base) == 4 and len(c11) == 7
    assert attention.construction_consumption_placeholder_kind(base) == "base"
    assert attention.construction_consumption_placeholder_kind(c11) == "composed"
    return base, c11


def test_scalar_placeholders_match_the_real_producer_shapes() -> None:
    base, c11 = _real_construction_placeholders()

    assert finalizer._construction_placeholder_kind(base) == "base"
    assert finalizer._construction_placeholder_kind(c11) == "composed"
    assert c11["passed"] is False
    assert c11["declared_layer_count"] == 28
    assert c11["declared_sequence_length"] == 8

    for mutated in (
        {**c11, "passed": True},
        {**c11, "status": "attested"},
        {**c11, "declared_layer_count": 27},
        {**c11, "declared_sequence_length": 0},
        {**c11, "observed_layers": list(range(28))},
        {key: value for key, value in c11.items() if key != "declared_sequence_length"},
        {**base, "declared_layer_count": 28},
        {**base, "schema_version": "natural_boundary_attention_actuators.v1"},
    ):
        assert finalizer._construction_placeholder_kind(mutated) is None


def test_scalar_receipt_accepts_the_real_c11_placeholder_but_runtime_requires_a_pass() -> None:
    event = {"event_id": "gt:c11-placeholder:0", "target_owner_id": "target:c11-placeholder"}
    _base, c11 = _real_construction_placeholders()
    raw = _raw(event, "K10", owners=["owner:0", "owner:1", None])
    raw["scalar_receipts"][0]["attention_actuation_receipt"] = {
        "cell_id": "C11",
        "layer_consumption_attestation": dict(c11),
        "all_layer_consumption_attestation": dict(c11),
    }

    parsed = finalizer._validate_natural_raw(raw, "K10", "c11-placeholder.K10", event=event)
    assert parsed["scalar"]["scalar_forward_count"] == len(raw["scalar_receipts"])

    # The same placeholder is never consumption evidence on a runtime receipt.
    runtime_only = copy.deepcopy(raw)
    runtime_only["runtime_scalar_receipts"][0]["layer_consumption_attestation"] = dict(c11)
    with pytest.raises(
        finalizer.EvidenceContractError,
        match=r"runtime_scalar_receipts\[0\] all-layer consumption did not pass",
    ):
        finalizer._validate_natural_raw(runtime_only, "K10", "c11-placeholder.K10", event=event)

    drifted = copy.deepcopy(raw)
    drifted["scalar_receipts"][0]["attention_actuation_receipt"][
        "layer_consumption_attestation"
    ] = {**c11, "passed": True}
    with pytest.raises(
        finalizer.EvidenceContractError,
        match=r"scalar_receipts\[0\] does not attest exactly layers 0\.\.27",
    ):
        finalizer._validate_natural_raw(drifted, "K10", "c11-placeholder.K10", event=event)


@pytest.mark.parametrize(
    ("terminal", "token", "invalid_rows", "scientific_status"),
    [
        ("native_stop", 999, 0, "native_STOP"),
        ("invalid", 777, 1, "invalid_token_grammar"),
    ],
)
def test_initial_native_stop_and_invalid_first_token_are_mechanical_scientific_outcomes(
    terminal: str,
    token: int,
    invalid_rows: int,
    scientific_status: str,
) -> None:
    event = {
        "event_id": f"gt:first-token:{terminal}",
        "target_owner_id": f"target:first-token:{terminal}",
        "image_id": 1,
        "event_index": 0,
    }
    raw = _first_token_outcome(event, token=token, terminal=terminal)

    parsed = finalizer._validate_natural_raw(raw, "K10", f"first-token.{terminal}", event=event)
    endpoint = finalizer._endpoint_document("C10", event, parsed)
    assert parsed["mechanically_valid"] is True
    assert parsed["opener_generated_by_model"] is False
    assert parsed["first_generated_token_id"] == token
    assert parsed["row_admission"] == 0
    assert parsed["rows"][0]["row_started"] is False
    assert parsed["parse"]["invalid_rows"] == invalid_rows
    assert parsed["native_stop"] is (terminal == "native_stop")
    assert endpoint["scientific_status"] == scientific_status


def test_started_row_ending_in_native_stop_has_one_admission() -> None:
    event = {
        "event_id": "gt:within-row-native-stop",
        "target_owner_id": "target:within-row-native-stop",
        "image_id": 2,
        "event_index": 1,
    }
    raw = _terminal_location_outcome(event, "within_row_stop")

    parsed = finalizer._validate_natural_raw(raw, "K10", "within-row-native-stop.K10", event=event)
    endpoint = finalizer._endpoint_document("C10", event, parsed)
    assert parsed["mechanically_valid"] is True
    assert parsed["opener_generated_by_model"] is True
    assert parsed["row_admission"] == 1
    assert parsed["native_stop"] is True
    assert parsed["rows"][0]["token_ids"][-1] == 999
    assert raw["generated_token_ids"] == raw["rows"][0]["token_ids"]
    assert endpoint["scientific_status"] == "native_STOP"


def test_two_closure_rows_then_lookahead_native_stop_has_two_admissions() -> None:
    event = {
        "event_id": "gt:lookahead-native-stop",
        "target_owner_id": "target:lookahead-native-stop",
        "image_id": 3,
        "event_index": 2,
    }
    raw = _terminal_location_outcome(event, "lookahead_stop")

    parsed = finalizer._validate_natural_raw(raw, "K10", "lookahead-native-stop.K10", event=event)
    endpoint = finalizer._endpoint_document("C10", event, parsed)
    flattened = [token for row in raw["rows"] for token in row["token_ids"]]
    assert [row["status"] for row in parsed["rows"]] == ["closure", "closure"]
    assert parsed["row_admission"] == 2
    assert parsed["parse"]["valid_rows"] == 2
    assert parsed["native_stop"] is True
    assert raw["generated_token_ids"] == flattened + [999]
    assert endpoint["scientific_status"] == "native_STOP"


def test_over_continuation_suffix_matches_selected_token_receipt() -> None:
    event = {
        "event_id": "gt:over-continuation-suffix",
        "target_owner_id": "target:over-continuation-suffix",
        "image_id": 4,
        "event_index": 3,
    }
    raw = _terminal_location_outcome(event, "over_continuation")

    parsed = finalizer._validate_natural_raw(raw, "K10", "over-continuation-suffix.K10", event=event)
    endpoint = finalizer._endpoint_document("C10", event, parsed)
    row = raw["rows"][0]
    assert parsed["mechanically_valid"] is True
    assert parsed["row_admission"] == 1
    assert parsed["parse"]["malformed_rows"] == 1
    assert parsed["parse"]["invalid_rows"] == 1
    assert raw["generated_token_ids"] == row["token_ids"] + [row["over_continuation"]["selected_token_id"]]
    assert endpoint["scientific_status"] == "malformed"


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_native_stop_token_set_is_required_and_unique(mutation: str) -> None:
    event = {"event_id": f"gt:native-stop-set:{mutation}", "target_owner_id": "target:native-stop-set"}
    raw = _first_token_outcome(event, token=999, terminal="native_stop")
    if mutation == "missing":
        raw.pop("native_stop_token_ids")
        message = r"native_stop_token_ids is malformed"
    else:
        raw["native_stop_token_ids"] = [999, 999]
        message = r"native_stop_token_ids contains duplicates"
    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._validate_natural_raw(raw, "K10", f"native-stop-set.{mutation}", event=event)


@pytest.mark.parametrize("mutation", ["no_terminal_stop_location", "terminal_mismatch"])
def test_native_stop_token_and_terminal_reason_must_agree(mutation: str) -> None:
    event = {"event_id": f"gt:native-stop-terminal:{mutation}", "target_owner_id": "target:native-stop-terminal"}
    if mutation == "no_terminal_stop_location":
        raw = _terminal_location_outcome(event, "lookahead_stop")
        flattened = [token for row in raw["rows"] for token in row["token_ids"]]
        _refresh_generated_trajectory(raw, flattened)
        message = r"native STOP terminal lacks an exact terminal token"
    else:
        raw = _first_token_outcome(event, token=999, terminal="native_stop")
        raw["terminal_reason"] = "invalid"
        message = r"native terminal token disagrees with terminal_reason"
    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._validate_natural_raw(raw, "K10", f"native-stop-terminal.{mutation}", event=event)


@pytest.mark.parametrize("mutation", ["non_stop_suffix", "extra_suffix_token"])
def test_native_stop_lookahead_suffix_must_be_one_exact_stop_token(mutation: str) -> None:
    event = {"event_id": f"gt:native-stop-suffix:{mutation}", "target_owner_id": "target:native-stop-suffix"}
    raw = _terminal_location_outcome(event, "lookahead_stop")
    flattened = [token for row in raw["rows"] for token in row["token_ids"]]
    suffix = [777] if mutation == "non_stop_suffix" else [999, 777]
    _refresh_generated_trajectory(raw, flattened + suffix)

    with pytest.raises(
        finalizer.EvidenceContractError,
        match=r"generated_token_ids differs from row token trajectory",
    ):
        finalizer._validate_natural_raw(raw, "K10", f"native-stop-suffix.{mutation}", event=event)


@pytest.mark.parametrize("mutation", ["wrong_terminal", "wrong_row_status"])
def test_native_stop_lookahead_suffix_requires_native_terminal_after_closure(mutation: str) -> None:
    event = {"event_id": f"gt:native-stop-suffix-context:{mutation}", "target_owner_id": "target:native-stop-context"}
    raw = _terminal_location_outcome(event, "lookahead_stop")
    parse = dict(raw["owner_bookkeeping"]["parse"])
    if mutation == "wrong_terminal":
        raw["terminal_reason"] = "closure"
        raw["stop_reason"] = "closure"
        _refresh_terminal_bookkeeping(raw, terminal="closure", parse=parse)
    else:
        row = raw["rows"][-1]
        row["status"] = "invalid"
        row["stop_reason"] = "invalid"
        row.pop("owner_match", None)
        row.pop("owner_match_status", None)
        parse.update({"valid_rows": 1, "invalid_rows": 1})
        _refresh_terminal_bookkeeping(raw, terminal="native_stop", parse=parse)

    with pytest.raises(
        finalizer.EvidenceContractError,
        match=r"generated_token_ids differs from row token trajectory",
    ):
        finalizer._validate_natural_raw(raw, "K10", f"native-stop-suffix-context.{mutation}", event=event)


def test_within_row_native_stop_must_be_the_last_row_token() -> None:
    event = {"event_id": "gt:misplaced-within-row-stop", "target_owner_id": "target:misplaced-stop"}
    raw = _terminal_location_outcome(event, "within_row_stop")
    row = raw["rows"][0]
    row["token_ids"] = [raw["opener_token_id"], 999, 100]
    row["token_ids_sha256"] = finalizer.sha256_json(row["token_ids"])
    _refresh_generated_trajectory(raw, list(row["token_ids"]))

    with pytest.raises(finalizer.EvidenceContractError, match=r"native STOP token is misplaced"):
        finalizer._validate_natural_raw(raw, "K10", "misplaced-within-row-stop.K10", event=event)


@pytest.mark.parametrize("mutation", ["mismatch", "missing"])
def test_over_continuation_requires_one_matching_suffix(mutation: str) -> None:
    event = {"event_id": f"gt:over-continuation:{mutation}", "target_owner_id": "target:over-continuation"}
    raw = _terminal_location_outcome(event, "over_continuation")
    row_tokens = list(raw["rows"][0]["token_ids"])
    generated = row_tokens + [778] if mutation == "mismatch" else row_tokens
    _refresh_generated_trajectory(raw, generated)
    message = (
        r"over-continuation token disagrees with trajectory"
        if mutation == "mismatch"
        else r"over-continuation terminal lacks its exact lookahead token"
    )

    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._validate_natural_raw(raw, "K10", f"over-continuation.{mutation}", event=event)


@pytest.mark.parametrize("foreign_token", [90, 999])
def test_over_continuation_suffix_cannot_be_an_opener_or_native_stop(foreign_token: int) -> None:
    event = {"event_id": f"gt:over-continuation:foreign-{foreign_token}", "target_owner_id": "target:over-continuation"}
    raw = _terminal_location_outcome(event, "over_continuation")
    raw["rows"][0]["over_continuation"]["selected_token_id"] = foreign_token
    generated = list(raw["rows"][0]["token_ids"]) + [foreign_token]
    _refresh_generated_trajectory(raw, generated)

    with pytest.raises(finalizer.EvidenceContractError, match=r"over-continuation lookahead token is foreign"):
        finalizer._validate_natural_raw(raw, "K10", f"over-continuation.foreign-{foreign_token}", event=event)


@pytest.mark.parametrize("mutation", ["missing", "aggregate_drift", "per_row_drift"])
def test_row_entry_bookkeeping_is_required_and_identical(mutation: str) -> None:
    event = {"event_id": f"gt:row-entry:{mutation}", "target_owner_id": "target:row-entry"}
    raw = _first_token_outcome(event, token=777, terminal="invalid")
    if mutation == "missing":
        raw["owner_bookkeeping"].pop("row_entry")
        message = r"owner_bookkeeping.row_entry is missing"
    elif mutation == "aggregate_drift":
        raw["owner_bookkeeping"]["row_entry"]["row_started"] = True
        message = r"owner_bookkeeping.row_entry differs for row_started"
    else:
        raw["rows"][0]["owner_bookkeeping"]["row_entry"]["first_generated_token_id"] = 778
        message = r"rows\[0\] row_entry bookkeeping differs"
    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._validate_natural_raw(raw, "K10", f"row-entry.{mutation}", event=event)


def test_row_started_must_agree_with_opener_provenance() -> None:
    event = {"event_id": "gt:row-started-drift", "target_owner_id": "target:row-started"}
    raw = _first_token_outcome(event, token=777, terminal="invalid")
    raw["rows"][0]["row_started"] = True
    with pytest.raises(finalizer.EvidenceContractError, match=r"row_started disagrees with opener provenance"):
        finalizer._validate_natural_raw(raw, "K10", "row-started-drift.K10", event=event)


def test_arm_and_first_row_first_token_receipts_must_agree() -> None:
    event = {"event_id": "gt:first-row-disagreement", "target_owner_id": "target:first-row"}
    raw = _first_token_outcome(event, token=777, terminal="invalid")
    raw["rows"][0]["first_generated_token_id"] = 778
    raw["rows"][0]["token_ids"] = [778]
    raw["rows"][0]["token_ids_sha256"] = finalizer.sha256_json([778])
    raw["generated_token_ids"] = [778]
    raw["generated_token_ids_sha256"] = finalizer.sha256_json([778])
    with pytest.raises(finalizer.EvidenceContractError, match=r"arm/first-row natural-boundary receipts disagree"):
        finalizer._validate_natural_raw(raw, "K10", "first-row-disagreement.K10", event=event)


def _sealed_input_bindings(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    """A receipt-shaped input_bindings group with one file and one directory.

    The directory identity is restated here from the sealer's contract -- entries
    of ``relative_path``/``sha256``/``size_bytes`` sorted by relative path, hashed
    as canonical JSON -- so the finalizer cannot drift away from the sealer by
    changing its own helper.
    """

    census = tmp_path / "admission-census.json"
    _write(census, {"census": "fixture"})

    root = tmp_path / "model_dir"
    (root / "nested").mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"model": "fixture"}')
    (root / "nested" / "weights.bin").write_bytes(b"weights-fixture-bytes")
    entries = [
        {
            "relative_path": relative,
            "sha256": finalizer.sha256_bytes((root / relative).read_bytes()),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in ("config.json", "nested/weights.bin")
    ]
    group = {
        "census": {
            "path": str(census),
            "sha256": finalizer.sha256_file(census),
            "size_bytes": census.stat().st_size,
            "kind": "file",
        },
        "base_model_dir": {
            "path": str(root),
            "sha256": finalizer.sha256_json(entries),
            "size_bytes": sum(item["size_bytes"] for item in entries),
            "kind": "directory",
        },
    }
    return group, census, root


def test_sealed_file_and_directory_bindings_are_accepted(tmp_path: Path) -> None:
    group, census, root = _sealed_input_bindings(tmp_path)

    for key, value in group.items():
        checked = finalizer._hash_ref(value, f"pre-GPU input_bindings.{key}")
        assert checked["sha256"] == value["sha256"]
        assert checked["size_bytes"] == value["size_bytes"]
        assert checked["kind"] == value["kind"]

    # A binding without ``kind`` keeps the pre-existing file contract.
    legacy = {"path": str(census), "raw_sha256": finalizer.sha256_file(census)}
    assert finalizer._hash_ref(legacy, "plan.source_bindings.census")["raw_sha256"] == legacy["raw_sha256"]
    assert finalizer._directory_inventory(root, "base model directory") == (
        group["base_model_dir"]["sha256"],
        group["base_model_dir"]["size_bytes"],
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("directory_sha", r"raw SHA-256 drifted"),
        ("directory_size", r"size_bytes drifted"),
        ("directory_kind_on_file", r"must be an existing regular non-symlink directory"),
        ("file_kind_on_directory", r"must be an existing regular non-symlink file"),
        ("unknown_kind", r"kind must be one of"),
        ("absent_directory", r"must be an existing regular non-symlink directory"),
        ("symlink_root", r"must not traverse a symlink"),
        ("symlink_child_directory", r"contains a symlink directory"),
        ("symlink_child_file", r"contains a non-regular file"),
        ("added_file", r"raw SHA-256 drifted"),
        ("empty_directory", r"inventory is empty"),
        ("file_sha", r"raw SHA-256 drifted"),
        ("file_size", r"size_bytes drifted"),
    ],
)
def test_sealed_binding_drift_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    group, census, root = _sealed_input_bindings(tmp_path)
    directory = dict(group["base_model_dir"])
    file_ref = dict(group["census"])
    target, label = directory, "pre-GPU input_bindings.base_model_dir"

    if mutation == "directory_sha":
        directory["sha256"] = "0" * 64
    elif mutation == "directory_size":
        directory["size_bytes"] = directory["size_bytes"] + 1
    elif mutation == "directory_kind_on_file":
        directory["path"] = str(census)
    elif mutation == "file_kind_on_directory":
        target, label = file_ref, "pre-GPU input_bindings.census"
        file_ref["path"] = str(root)
    elif mutation == "unknown_kind":
        directory["kind"] = "dir"
    elif mutation == "absent_directory":
        directory["path"] = str(tmp_path / "missing_model_dir")
    elif mutation == "symlink_root":
        link = tmp_path / "linked_model_dir"
        link.symlink_to(root, target_is_directory=True)
        directory["path"] = str(link)
    elif mutation == "symlink_child_directory":
        (root / "linked_nested").symlink_to(root / "nested", target_is_directory=True)
    elif mutation == "symlink_child_file":
        (root / "linked_config.json").symlink_to(root / "config.json")
    elif mutation == "added_file":
        (root / "extra.json").write_bytes(b"{}")
    elif mutation == "empty_directory":
        empty = tmp_path / "empty_model_dir"
        empty.mkdir()
        directory["path"] = str(empty)
    elif mutation == "file_sha":
        target, label = file_ref, "pre-GPU input_bindings.census"
        file_ref["sha256"] = "0" * 64
    else:
        target, label = file_ref, "pre-GPU input_bindings.census"
        file_ref["size_bytes"] = file_ref["size_bytes"] + 1

    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._hash_ref(target, label)


def test_wrong_contrast_sign_is_rejected_by_nonzero_fixture(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    result = finalizer.finalize(plan, pre_gpu, roots)
    deltas = result["evidence"]["component_contrasts"]["C10-C00_static"]["metrics"]["strict_count"]
    assert deltas == [0, 0, 0]
    # Directly exercise the nonzero sign-bearing arithmetic.
    events = [{"event_id": "x", "image_id": 1, "source_specific_event_status": "qualified", "source_specific_unqualified_reason": None, "cells": {cell: {"metrics": {metric: 0 for metric in finalizer.METRICS}} for cell in finalizer.CELL_ORDER}}]
    events[0]["cells"]["C10"]["metrics"]["strict_count"] = 2
    contrasts, _ = finalizer._numeric_contrasts(events)
    assert contrasts["C10-C00_static"]["metrics"]["strict_count"] == [2]


@pytest.mark.parametrize("mutation", ["event", "device", "runtime", "roots"])
def test_fail_closed_for_provenance_and_runtime_drift(tmp_path: Path, mutation: str) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    if mutation == "event":
        value = json.loads((roots[0] / "result.json").read_text())
        value["event_index"] = 0
        _write(roots[0] / "result.json", _self(value, field="result_sha256"))
    elif mutation == "device":
        value = json.loads((roots[0] / "runtime_identity.json").read_text())
        value["device"]["physical_device"] = "2"
        _write(roots[0] / "runtime_identity.json", _self(value))
    elif mutation == "runtime":
        value = json.loads((roots[0] / "runtime_identity.json").read_text())
        value["code_hashes"]["runner"] = "f" * 64
        _write(roots[0] / "runtime_identity.json", _self(value))
    else:
        value = json.loads(Path(pre_gpu).read_text())
        value["roots"]["execution_root"]["path"] = str(tmp_path / "other")
        _write(Path(pre_gpu), _self(value))
    with pytest.raises(finalizer.EvidenceContractError):
        finalizer.finalize(plan, pre_gpu, roots)


def test_write_once_and_final_root_binding(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    output = tmp_path / "final" / "evidence.json"
    receipt = tmp_path / "final" / "evidence.receipt.json"
    first = finalizer.finalize(plan, pre_gpu, roots, output=output, receipt_output=receipt)
    second = finalizer.finalize(plan, pre_gpu, roots, output=output, receipt_output=receipt)
    assert first["evidence"]["self_sha256"] == second["evidence"]["self_sha256"]
    finalizer.validate_evidence(json.loads(output.read_text()))
    assert json.loads(receipt.read_text())["evidence_sha256"] == first["evidence"]["self_sha256"]


@pytest.mark.skipif(not REAL_GATE_RESULT.is_file(), reason="artifact-valid S gate result is unavailable")
def test_artifact_valid_s_gate_shape_is_read_only_compatibility_probe() -> None:
    """Check the production prefix/trajectory shape without copying or mutating it."""

    document = json.loads(REAL_GATE_RESULT.read_text())
    arm = document["arms"]["K00"]
    prefix = arm["prefix"]
    assert len(prefix["exact_history_token_ids"]) > 0
    assert len(prefix["prefix_token_ids"]) > len(prefix["exact_history_token_ids"])
    assert prefix["prefix_token_ids_sha256"] == finalizer.sha256_json(prefix["prefix_token_ids"])
    runtime = arm["runtime_scalar_receipts"]
    assert runtime[0]["step"] == 0
    assert runtime[0]["sequence_length"] == len(prefix["prefix_token_ids"])
    assert runtime[1]["input_ids_sha256"] != runtime[0]["input_ids_sha256"]
    assert runtime[1]["mrope_hash"] != runtime[0]["mrope_hash"]


@pytest.mark.skipif(
    not all(path.is_file() for path in _real_s_v2_result_paths()),
    reason="frozen S-v2 selected-event result artifacts are unavailable",
)
def test_frozen_s_v2_real_arms_finalize_through_tmp_shard_envelopes(tmp_path: Path) -> None:
    source_paths = _real_s_v2_result_paths()
    source_bytes = [path.read_bytes() for path in source_paths]
    source_documents = [json.loads(raw) for raw in source_bytes]

    plan, pre_gpu, roots = _real_artifact_fixture(tmp_path)
    result = finalizer.finalize(plan, pre_gpu, roots)
    evidence = result["evidence"]

    assert [event["event_index"] for event in evidence["events"]] == [2, 5, 8]
    assert [event["event_id"] for event in evidence["events"]] == list(finalizer.EVENT_IDS)
    assert [event["image_id"] for event in evidence["events"]] == [2299, 13348, 16228]
    assert evidence["source_specific_crossover_status"] == "unqualified"
    assert not (tmp_path / "final").exists()

    for index, (event, source) in enumerate(zip(evidence["events"], source_documents)):
        source_event = source["event"]
        expected_history_sha = source_event["natural_boundary"]["history_sha256"]
        assert event["source_specific_event_status"] == "unqualified"
        assert event["cells"]["C11"]["mechanically_valid"] is True
        for cell, arm in (("C00", "K01"), ("C10", "K10"), ("C01", "H20"), ("C11", "K10")):
            raw = source["arms"][arm]
            endpoint = event["cells"][cell]
            assert endpoint["mechanically_valid"] is True
            assert endpoint["prefix_sha256"] == expected_history_sha
            assert endpoint["mrope_hash"] == raw["runtime_scalar_receipts"][0]["mrope_hash"]
            expected_strict = sum(
                row.get("owner_match", {}).get("status") in {"unique", "matched"}
                and row.get("owner_match", {}).get("source_specific") is True
                and row.get("owner_match", {}).get("physical_match") is True
                for row in raw["rows"]
            )
            assert endpoint["strict_count"] == expected_strict
            assert endpoint["metrics"]["unmatched"] == raw["owner_bookkeeping"]["parse"]["unmatched_rows"]
            assert endpoint["metrics"]["duplicates"] == raw["owner_bookkeeping"]["parse"]["duplicate_rows"]
            for horizon in ("horizon_1", "horizon_3"):
                utility = endpoint["owner_utility"][horizon]
                assert utility["reference_cell"] == "C00"
                assert utility["estimand"] == "treatment-minus-C00 owner-set delta over first N rows"
            prefix = raw["prefix"]
            assert prefix["prefix_token_ids_sha256"] == finalizer.sha256_json(prefix["prefix_token_ids"])
            assert raw["scalar_receipts"][0]["input_ids"] == prefix["prefix_token_ids"]
            assert raw["runtime_scalar_receipts"][0]["sequence_length"] == len(prefix["prefix_token_ids"])
            assert raw["runtime_scalar_receipts"][1]["sequence_length"] == len(prefix["prefix_token_ids"]) + 1

            if index == 0 and cell == "C10":
                assert endpoint["complete_rows"] == 3
                assert endpoint["strict_count"] == 1
                assert endpoint["metrics"]["unmatched"] == 2
                assert endpoint["metrics"]["row_admission"] == 3

        assert event["cells"]["C10"]["parse"]["unmatched_rows"] > 0 or event["cells"]["C01"]["parse"]["unmatched_rows"] > 0
        assert event["cells"]["C10"]["owner_utility"]["source_specific_status"] == "unqualified"
        assert event["cells"]["C11"]["owner_utility"]["source_specific_status"] == "unqualified"
        for horizon in ("horizon_1", "horizon_3"):
            assert evidence["utilities"][horizon]["net_charged"]["tau"] is None
            assert evidence["utilities"][horizon]["status"] == "unqualified"
        assert event["technical_control"]["cell_id"] == "K00"
        assert event["technical_control"]["result_sha256"]
        assert evidence["events"][index]["event_id"] == source_event["event_id"]

    assert [path.read_bytes() for path in source_paths] == source_bytes


# --- post-execution finalization successor -----------------------------------


REAL_FINALIZER = Path(finalizer.__file__).resolve()
REAL_FINALIZER_TEST = REAL_FINALIZER.parents[2] / "tests" / "research" / "test_finalize_s_k10_h20_crossover.py"
REAL_SEALER = Path(successor_sealer.__file__).resolve()
REAL_SEALER_TEST = REAL_FINALIZER.parents[2] / "tests" / "research" / "test_seal_s_k10_h20_crossover_finalization_receipt.py"


def _selfed(document: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != field}
    result = dict(body)
    result[field] = finalizer.sha256_json(body)
    return result


def _successor_fixture(tmp_path: Path) -> dict[str, Any]:
    """A synthetic parent/execution set whose two slots point at the real tools."""

    unit = tmp_path / "unit"
    execution_root = unit / "execution-v4"
    evidence_root = unit / "evidence-v4"
    authority = tmp_path / "authority.md"
    authority.write_text("# successor authority fixture\n")
    census = tmp_path / "census.json"
    _write(census, {"census": "fixture"})

    plan_events = [
        {"event_id": event_id, "event_index": index, "image_id": 100 + position}
        for position, (event_id, index) in enumerate(zip(finalizer.EVENT_IDS, finalizer.EVENT_INDICES))
    ]
    plan_document = _selfed(
        {
            "schema_version": finalizer.PLAN_SCHEMA_VERSION,
            "unit_id": finalizer.UNIT_ID,
            "status": "planned",
            "events": plan_events,
        }
    )
    plan_path = unit / "plan-v1" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(finalizer.canonical_json_bytes(plan_document) + b"\n")
    plan_info = {
        "document": plan_document,
        "path": str(plan_path),
        "raw_sha256": finalizer.sha256_file(plan_path),
        "self_sha256": plan_document["self_sha256"],
        "events": plan_events,
    }

    parent_document = _selfed(
        {
            "schema_version": finalizer.PRE_GPU_SCHEMA_VERSION,
            "unit_id": finalizer.UNIT_ID,
            "status": "sealed_pre_gpu",
            "plan_self_sha256": plan_document["self_sha256"],
            "roots": {
                "execution_root": {"path": str(execution_root)},
                "final_root": {"path": str(evidence_root)},
            },
            "input_bindings": {
                "census": {
                    "path": str(census),
                    "sha256": finalizer.sha256_file(census),
                    "size_bytes": census.stat().st_size,
                    "kind": "file",
                }
            },
            "input_hashes": {"census_sha256": finalizer.sha256_file(census)},
            "source_files": {
                "crossover_finalizer": {"path": str(REAL_FINALIZER), "sha256": "a" * 64, "kind": "file"}
            },
            "test_files": {
                "crossover_finalizer_test": {"path": str(REAL_FINALIZER_TEST), "sha256": "b" * 64, "kind": "file"}
            },
        }
    )
    parent_path = unit / "pre-gpu-receipt-v5" / "pre-gpu-receipt.json"
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path.write_bytes(finalizer.canonical_json_bytes(parent_document) + b"\n")

    for index, shard_id in enumerate(successor_sealer.SHARD_IDS):
        root = execution_root / shard_id
        root.mkdir(parents=True, exist_ok=True)
        result = _selfed(
            {
                "schema_version": finalizer.EVENT_SCHEMA_VERSION,
                "unit_id": finalizer.UNIT_ID,
                "status": "completed",
                "shard_id": shard_id,
                "event_id": finalizer.EVENT_IDS[index],
                "event_index": finalizer.EVENT_INDICES[index],
                "image_id": 100 + index,
                "plan_self_sha256": plan_document["self_sha256"],
                "pre_gpu_receipt_self_sha256": parent_document["self_sha256"],
            },
            "result_sha256",
        )
        (root / "result.json").write_bytes(finalizer.canonical_json_bytes(result) + b"\n")
        for name, body in (
            ("runtime_identity.json", {"shard_id": shard_id, "device": {"physical_device": finalizer.DEVICE_PLAN[shard_id], "logical_device": "cuda:0"}}),
            ("terminal_summary.json", {"shard_id": shard_id, "status": "completed"}),
            ("aggregate.receipt.json", {"shard_id": shard_id, "event_id": finalizer.EVENT_IDS[index]}),
        ):
            (root / name).write_bytes(finalizer.canonical_json_bytes(_selfed(body)) + b"\n")

    receipt = successor_sealer.build_finalization_receipt(
        parent_receipt=parent_path,
        plan=plan_path,
        execution_root=execution_root,
        evidence_root=evidence_root,
        authority=authority,
    )
    return {
        "receipt": receipt,
        "plan_info": plan_info,
        "plan_path": plan_path,
        "parent_path": parent_path,
        "parent_document": parent_document,
        "execution_root": execution_root,
        "evidence_root": evidence_root,
        "authority": authority,
        "census": census,
    }


def test_finalization_successor_authorizes_exactly_two_slots(tmp_path: Path) -> None:
    fixture = _successor_fixture(tmp_path)
    checked = finalizer._validate_finalization_receipt(fixture["receipt"], fixture["plan_info"])

    assert set(checked["allowance"]) == set(finalizer.AUTHORIZED_FINALIZATION_SLOTS)
    assert checked["allowance"]["source_files.crossover_finalizer"] == finalizer.sha256_file(REAL_FINALIZER)
    assert checked["allowance"]["test_files.crossover_finalizer_test"] == finalizer.sha256_file(REAL_FINALIZER_TEST)
    assert checked["parent_raw_sha256"] == finalizer.sha256_file(fixture["parent_path"])
    assert checked["parent_self_sha256"] == fixture["parent_document"]["self_sha256"]
    assert checked["evidence_root"] == str(fixture["evidence_root"])
    assert checked["self_sha256"] == fixture["receipt"]["self_sha256"]


def test_parent_ref_validation_consumes_only_the_two_authorized_slots(tmp_path: Path) -> None:
    fixture = _successor_fixture(tmp_path)
    parent = fixture["parent_document"]
    info = {"path": str(fixture["parent_path"]), "raw_sha256": finalizer.sha256_file(fixture["parent_path"]), "size_bytes": 0}
    checked = finalizer._validate_finalization_receipt(fixture["receipt"], fixture["plan_info"])

    # Without the successor the parent's own binding of this file is fatal.
    with pytest.raises(finalizer.EvidenceContractError, match=r"source_files.crossover_finalizer raw SHA-256 drifted"):
        finalizer._hash_ref(parent["source_files"]["crossover_finalizer"], "pre-GPU source_files.crossover_finalizer")

    # With it, exactly that slot validates against the live file.
    finalizer._hash_ref(
        parent["source_files"]["crossover_finalizer"],
        "pre-GPU source_files.crossover_finalizer",
        expected_sha256=checked["allowance"]["source_files.crossover_finalizer"],
    )
    # An unrelated binding is never relaxed by the successor.
    fixture["census"].write_bytes(finalizer.canonical_json_bytes({"census": "drifted"}) + b"\n")
    with pytest.raises(finalizer.EvidenceContractError, match=r"input_bindings.census raw SHA-256 drifted"):
        finalizer._hash_ref(parent["input_bindings"]["census"], "pre-GPU input_bindings.census")
    assert info["raw_sha256"] == checked["parent_raw_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", r"schema/status/unit identity drifted"),
        ("status", r"schema/status/unit identity drifted"),
        ("unit", r"schema/status/unit identity drifted"),
        ("self_tamper", r"finalization receipt.self_sha256 mismatch"),
        ("cpu_only", r"authorization.cpu_only must be true"),
        ("gpu_used", r"authorization.gpu_used must be false"),
        ("model_loaded", r"authorization.model_loaded must be false"),
        ("no_training", r"authorization.no_training must be true"),
        ("endpoint_semantics", r"authorization.endpoint_semantics_unchanged must be true"),
        ("authority_hash", r"finalization authority document differs from the live file"),
        ("parent_raw", r"parent raw SHA-256 differs from the live parent"),
        ("parent_self", r"parent self SHA-256 differs from the live parent"),
        ("parent_document", r"parent document differs from the live parent"),
        ("plan_raw", r"plan raw SHA-256 differs from the supplied plan"),
        ("plan_self", r"plan self SHA-256 differs from the supplied plan"),
        ("old_hash", r"old hash differs from the parent binding"),
        ("new_hash", r"new hash differs from the live file"),
        ("third_slot", r"must authorize exactly"),
        ("one_slot", r"must authorize exactly"),
        ("foreign_finalizer_path", r"path differs from the parent binding"),
        ("missing_cause", r"cause"),
        ("shard_order", r"identity/order drifted"),
        ("shard_event", r"event/device binding differs from the plan"),
        ("shard_device", r"event/device binding differs from the plan"),
        ("shard_artifact_raw", r"raw SHA-256 differs from the immutable artifact"),
        ("shard_artifact_self", r"self SHA-256 differs from the immutable artifact"),
        ("shard_artifact_path", r"path is not under its shard root"),
        ("shard_artifact_set", r"artifact set drifted"),
        ("execution_status", r"not declared complete_immutable"),
        ("evidence_hash", r"must not bind an evidence hash"),
        ("evidence_exists", r"must still be absent and non-symlink"),
        ("pins_raw", r"runtime input pins differ from the recomputed parent"),
        ("pins_self", r"runtime input pins differ from the recomputed parent"),
        ("unchanged_set", r"unchanged source_files set differs from the parent"),
        ("unchanged_hash", r"unchanged input_bindings.census differs from the parent"),
        ("tool_set", r"tools must be exactly"),
        ("sealer_hash", r"finalization tool successor_sealer differs from the live file"),
        ("sealer_test_hash", r"finalization tool successor_sealer_test differs from the live file"),
    ],
)
def test_finalization_successor_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    fixture = _successor_fixture(tmp_path)
    receipt = copy.deepcopy(fixture["receipt"])
    finalizer_slot = "source_files.crossover_finalizer"
    reself = True

    if mutation == "schema":
        receipt["schema_version"] = "other.v1"
    elif mutation == "status":
        receipt["status"] = "draft"
    elif mutation == "unit":
        receipt["unit_id"] = "other-unit"
    elif mutation == "self_tamper":
        receipt["self_sha256"] = "c" * 64
        reself = False
    elif mutation in {"cpu_only", "no_training", "endpoint_semantics"}:
        key = "endpoint_semantics_unchanged" if mutation == "endpoint_semantics" else mutation
        receipt["authorization"][key] = False
    elif mutation in {"gpu_used", "model_loaded"}:
        receipt["authorization"][mutation] = True
    elif mutation == "authority_hash":
        receipt["authorization"]["authority"]["sha256"] = "d" * 64
    elif mutation == "parent_raw":
        receipt["parent"]["receipt"]["raw_sha256"] = "e" * 64
    elif mutation == "parent_self":
        receipt["parent"]["receipt"]["self_sha256"] = "f" * 64
    elif mutation == "parent_document":
        receipt["parent"]["document"]["no_training"] = True
    elif mutation == "plan_raw":
        receipt["parent"]["plan"]["raw_sha256"] = "0" * 64
    elif mutation == "plan_self":
        receipt["parent"]["plan"]["self_sha256"] = "1" * 64
    elif mutation == "old_hash":
        receipt["authorized_drift"][finalizer_slot]["old_sha256"] = "2" * 64
    elif mutation == "new_hash":
        receipt["authorized_drift"][finalizer_slot]["new_sha256"] = "3" * 64
    elif mutation == "third_slot":
        receipt["authorized_drift"]["source_files.crossover_runner"] = dict(receipt["authorized_drift"][finalizer_slot])
    elif mutation == "one_slot":
        receipt["authorized_drift"].pop("test_files.crossover_finalizer_test")
    elif mutation == "foreign_finalizer_path":
        foreign = tmp_path / "foreign_finalizer.py"
        foreign.write_text("# foreign\n")
        receipt["authorized_drift"][finalizer_slot]["path"] = str(foreign)
    elif mutation == "missing_cause":
        receipt["authorized_drift"][finalizer_slot]["cause"] = ""
    elif mutation == "shard_order":
        receipt["execution"]["shards"][0], receipt["execution"]["shards"][1] = (
            receipt["execution"]["shards"][1],
            receipt["execution"]["shards"][0],
        )
    elif mutation == "shard_event":
        receipt["execution"]["shards"][0]["image_id"] = 999
    elif mutation == "shard_device":
        receipt["execution"]["shards"][2]["physical_device"] = "3"
    elif mutation == "shard_artifact_raw":
        receipt["execution"]["shards"][1]["artifacts"]["result.json"]["raw_sha256"] = "4" * 64
    elif mutation == "shard_artifact_self":
        receipt["execution"]["shards"][1]["artifacts"]["terminal_summary.json"]["self_sha256"] = "5" * 64
    elif mutation == "shard_artifact_path":
        other = fixture["execution_root"] / "shard-000" / "result.json"
        receipt["execution"]["shards"][2]["artifacts"]["result.json"]["path"] = str(other)
    elif mutation == "shard_artifact_set":
        receipt["execution"]["shards"][0]["artifacts"].pop("aggregate.receipt.json")
    elif mutation == "execution_status":
        receipt["execution"]["root"]["status"] = "partial"
    elif mutation == "evidence_hash":
        receipt["evidence_root"]["sha256"] = "6" * 64
    elif mutation == "evidence_exists":
        fixture["evidence_root"].mkdir(parents=True)
    elif mutation == "pins_raw":
        receipt["runtime_input_pins"]["pre_gpu_receipt_sha256"] = "7" * 64
    elif mutation == "pins_self":
        receipt["runtime_input_pins"]["pre_gpu_receipt_self_sha256"] = "8" * 64
    elif mutation == "unchanged_set":
        receipt["unchanged_parent_bindings"]["source_files"]["crossover_runner"] = {"path": "/x", "sha256": "9" * 64}
    elif mutation == "unchanged_hash":
        receipt["unchanged_parent_bindings"]["input_bindings"]["census"]["sha256"] = "a" * 64
    elif mutation == "tool_set":
        receipt["finalization_tools"].pop("successor_sealer_test")
    elif mutation == "sealer_hash":
        receipt["finalization_tools"]["successor_sealer"]["sha256"] = "b" * 64
    else:
        receipt["finalization_tools"]["successor_sealer_test"]["sha256"] = "c" * 64

    candidate = _selfed(receipt) if reself else receipt
    with pytest.raises(finalizer.EvidenceContractError, match=message):
        finalizer._validate_finalization_receipt(candidate, fixture["plan_info"])


def test_finalization_successor_rejects_a_foreign_running_finalizer(tmp_path: Path) -> None:
    fixture = _successor_fixture(tmp_path)
    receipt = copy.deepcopy(fixture["receipt"])
    foreign = tmp_path / "foreign_tool.py"
    foreign.write_text("# foreign finalization tool\n")
    receipt["finalization_tools"]["finalizer"] = {
        "path": str(foreign),
        "sha256": finalizer.sha256_file(foreign),
        "size_bytes": foreign.stat().st_size,
    }
    with pytest.raises(finalizer.EvidenceContractError, match=r"path differs from the running finalization tool"):
        finalizer._validate_finalization_receipt(_selfed(receipt), fixture["plan_info"])


def test_finalization_successor_rejects_a_different_parent_or_plan_path(tmp_path: Path) -> None:
    fixture = _successor_fixture(tmp_path)
    other_parent = tmp_path / "other-parent.json"
    other_parent.write_bytes(finalizer.canonical_json_bytes(_selfed({"unit_id": finalizer.UNIT_ID})) + b"\n")
    with pytest.raises(finalizer.EvidenceContractError, match=r"binds a different parent receipt path"):
        finalizer._validate_finalization_receipt(fixture["receipt"], fixture["plan_info"], pre_gpu_path=other_parent)

    other_plan = dict(fixture["plan_info"])
    other_plan["path"] = str(tmp_path / "other-plan.json")
    Path(other_plan["path"]).write_bytes(b"{}\n")
    with pytest.raises(finalizer.EvidenceContractError, match=r"binds a different plan path"):
        finalizer._validate_finalization_receipt(fixture["receipt"], other_plan)


def _runtime_document(pre_gpu: Mapping[str, Any], extras: Mapping[str, str]) -> dict[str, Any]:
    sealed_inputs = finalizer._sealed_input_hashes(pre_gpu)
    sealed_code = finalizer._sealed_code_hashes(pre_gpu)
    return {"code_hashes": dict(sealed_code), "input_hashes": {**sealed_inputs, **extras}}


def test_runtime_receipt_key_is_admitted_only_under_a_successor(tmp_path: Path) -> None:
    fixture = _successor_fixture(tmp_path)
    checked = finalizer._validate_finalization_receipt(fixture["receipt"], fixture["plan_info"])
    pre_gpu = {"document": fixture["parent_document"]}
    pins = {
        "pre_gpu_receipt": checked["parent_raw_sha256"],
        "pre_gpu_receipt_sha256": checked["parent_raw_sha256"],
        "pre_gpu_receipt_self_sha256": checked["parent_self_sha256"],
    }
    document = _runtime_document(pre_gpu, pins)

    with pytest.raises(finalizer.EvidenceContractError, match=r"unexpected runner-local key pre_gpu_receipt_sha256"):
        finalizer._validate_runtime_hash_maps(document, pre_gpu, "runtime identity")
    finalizer._validate_runtime_hash_maps(document, pre_gpu, "runtime identity", successor=checked)

    for key in ("pre_gpu_receipt", "pre_gpu_receipt_sha256", "pre_gpu_receipt_self_sha256"):
        drifted = _runtime_document(pre_gpu, {**pins, key: "d" * 64})
        with pytest.raises(finalizer.EvidenceContractError, match=rf"input_hashes.{key} does not pin the recomputed parent"):
            finalizer._validate_runtime_hash_maps(drifted, pre_gpu, "runtime identity", successor=checked)

    missing = _runtime_document(pre_gpu, {key: value for key, value in pins.items() if key != "pre_gpu_receipt_sha256"})
    with pytest.raises(finalizer.EvidenceContractError, match=r"pre_gpu_receipt_sha256 does not pin the recomputed parent"):
        finalizer._validate_runtime_hash_maps(missing, pre_gpu, "runtime identity", successor=checked)

    unknown = _runtime_document(pre_gpu, {**pins, "runner_local_extra": "e" * 64})
    with pytest.raises(finalizer.EvidenceContractError, match=r"unexpected runner-local key runner_local_extra"):
        finalizer._validate_runtime_hash_maps(unknown, pre_gpu, "runtime identity", successor=checked)


def test_finalize_without_a_successor_keeps_the_strict_parent_contract(tmp_path: Path) -> None:
    plan, pre_gpu, roots = _fixture(tmp_path)
    baseline = finalizer.finalize(plan, pre_gpu, roots)
    assert "finalization_successor" not in baseline["evidence"]["source_bindings"]
    assert finalizer._parser().parse_args(
        ["--plan", "p", "--pre-gpu-receipt", "r", "--shard", "s", "--output", "o", "--receipt", "c"]
    ).finalization_receipt is None
