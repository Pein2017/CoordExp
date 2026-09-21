#!/usr/bin/env python3
"""World8 owner-valid natural-branch admission at the frozen Image2299 tail."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_manifold_match_anchor_overfit as manifold
from scripts.research import run_image2299_ota_sqp_lite as ota
from src.inference.backend import token_ids_sha256


base = ota.base
parallel = ota.parallel
projected = ota.projected

SCHEMA_VERSION = "image2299.owner_valid_natural_branch.v1"
UNIT_ID = "2026-08-30-image2299-owner-valid-natural-branch"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
START_RECEIPT = (
    ota.OUTPUT_ROOT / "20260830T-image2299-ota-sqp-lite-r32-step-v1" / "receipt.json"
)
START_RECEIPT_SHA256 = "a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c"
START_CHECKPOINT = START_RECEIPT.parent / "checkpoint-selected-r32-step"
START_SURFACE_SHA256 = "2ac0e8a963d91ac7a4ba42322272889efd74bd46578c1ac44e16e531d1246676"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
START_ROUTE_SHA256 = "5df0ac25aa871ddc0550298e70cd6012ce4dc3b6c6068b97dfa0cddca2f02a67"
BRANCH_PREFIX_LENGTH = 301
BRANCH_PREFIX_SHA256 = "80c999c1e5e0cdfffeab00535892a17499db0910abae133c9b625bbac560f76d"
REACH_POSITIONS = (297, 298, 299, 300)
REACH_TOKENS = (151646, 8987, 151647, 151648)
CURRENT_TOKEN = 152590
PRIOR_TOKEN = 152576
ALIAS_COUNT = 104
CATALOG_SHA256 = "94f8e2f5695f13b57d5bec48472d5bf8298c5ac517e4bb140c536534c2ed60ac"
X1_TOKENS = (
    151820, 151821, 151822, 151866, 151867, 151868, 152018, 152019, 152020,
    152030, 152031, 152032, 152041, 152042, 152043, 152100, 152101, 152102,
    152131, 152132, 152133, 152188, 152189, 152190, 152228, 152229, 152230,
    152241, 152242, 152243, 152244, 152245, 152246, 152304, 152305, 152306,
    152398, 152399, 152400,
)
X1_TOKENS_SHA256 = "7590ab9336208bcb058f21544a569a67834c647971d011916e200d964c44ee40"
WORLD_SIZE = 8
RADII = ota.RADII
REACH_FLOOR = 1.0e-4
SAFETY_ETA = ota.SAFETY_ETA
TRUST_DELTA = 1.0 / 2048.0
INCUMBENT_CUT_COUNT = 7
MAX_FIRST_PANEL_ROWS = 14
MAX_CLOSURE_PANEL_ROWS = 15
RAW_COUNTER_KEYS = tuple(sorted(projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS))
RESOURCE_BOUND = {
    "gpu_count": WORLD_SIZE,
    "forced_completions_total": len(X1_TOKENS),
    "linearizations_max": 2,
    "gradient_backwards_total_max": MAX_FIRST_PANEL_ROWS + MAX_CLOSURE_PANEL_ROWS,
    "natural_panel_rollouts_max": 2 * WORLD_SIZE,
    "natural_decodes_total_max": len(X1_TOKENS) + 2 * WORLD_SIZE + 1,
    "generated_tokens_per_decode_max": base.NATURAL_MAX_TOKENS,
    "teacher_forwards_total_max": 4,
    "model_forward_calls_total_max": (
        (len(X1_TOKENS) + 2 * WORLD_SIZE + 1) * base.NATURAL_MAX_TOKENS + 4
    ),
    "checkpoint_saves_max": 1,
    "cold_reloads_total": 1,
    "rank0_gradient_payload_bytes_max": MAX_CLOSURE_PANEL_ROWS * ota.R32_STEP_ELEMENT_COUNT * 4,
    "max_peak_cuda_reserved_bytes_per_rank": 64 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
    "wall_time_seconds_max": 1_200,
}
TERMINAL_STATUSES = {
    "cold_match_level_promotion",
    "branch_compiled_suffix_failed",
    "preservation_conflict",
    "no_safe_branch_crossing",
    "empty_conditional_positive_set",
    "vertical_no_update_discovery_and_sensitivity_complete",
}


class NaturalBranchHold(RuntimeError):
    """A frozen identity, execution, solver, or admission contract failed closed."""


def _catalog_fingerprint(admitted: Sequence[Mapping[str, Any]]) -> str:
    compact = [
        {
            "owner": str(item.get("owner", "")),
            "row_sha256": str(item.get("row_sha256", "")),
            "token_ids": list(map(int, item.get("token_ids", ()))),
            "witness_sha256": str(item.get("witness_sha256", "")),
        }
        for item in admitted
    ]
    return base._hash(compact)


def _validate_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    cold = dict(receipt.get("cold", {}))
    packet = dict(cold.get("packet", {}))
    evaluation = dict(packet.get("evaluation", {}))
    joint = dict(evaluation.get("joint_gate", {}))
    route = list(map(int, evaluation.get("generated_token_ids", ())))
    parent_owner_ids = list(map(str, evaluation.get("matched_target_owner_ids", ())))
    aliases = list(dict(receipt.get("alias_catalog", {})).get("admitted", ()))
    rejected = list(dict(receipt.get("alias_catalog", {})).get("rejected", ()))
    saved = dict(receipt.get("saved_checkpoint", {}))
    readback = dict(saved.get("readback", {}))
    start_state = dict(receipt.get("start_state", {}))
    source_snapshot = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source_snapshot.get("path", "")))
    parent_checkpoint = Path(str(readback.get("parent_checkpoint", "")))

    compact_ok = all(
        bool(item.get("admitted"))
        and len(list(item.get("token_ids", ()))) == base.ROW_TOKENS
        and token_ids_sha256(item["token_ids"]) == item.get("row_sha256")
        and token_ids_sha256([*route[:297], *map(int, item["token_ids"]), base.EOS])
        == item.get("witness_sha256")
        for item in aliases
    )
    x1_tokens = tuple(sorted({int(item["token_ids"][4]) for item in aliases}))
    readback_payloads = {
        str(item.get("payload_id", "")): dict(item)
        for item in readback.get("payloads", ())
    }
    if (
        receipt.get("schema_version") != ota.SCHEMA_VERSION
        or receipt.get("unit_id") != ota.UNIT_ID
        or receipt.get("status") != "r32_step_selected_working_cold_reproduced"
        or receipt.get("run_id") != START_RECEIPT.parent.name
        or saved.get("checkpoint") != str(START_CHECKPOINT)
        or readback.get("child_checkpoint") != str(START_CHECKPOINT)
        or not parent_checkpoint.is_dir()
        or set(readback_payloads) != {"adapter", "special_token_embeddings"}
        or int(readback_payloads["adapter"].get("tensor_key_count", -1))
        != ota.R32_STEP_TENSOR_COUNT
        or int(readback_payloads["special_token_embeddings"].get("tensor_key_count", -1)) != 1
        or not source_path.is_file()
        or base._sha256(source_path) != source_snapshot.get("sha256")
        or token_ids_sha256(route) != START_ROUTE_SHA256
        or len(route) != 34 * base.ROW_TOKENS + 1
        or route[-1:] != [base.EOS]
        or base.EOS in route[:-1]
        or token_ids_sha256(route[:BRANCH_PREFIX_LENGTH]) != BRANCH_PREFIX_SHA256
        or tuple(route[REACH_POSITIONS[0]:BRANCH_PREFIX_LENGTH]) != REACH_TOKENS
        or route[BRANCH_PREFIX_LENGTH] != CURRENT_TOKEN
        or list(map(int, start_state.get("natural_token_ids", ())))[BRANCH_PREFIX_LENGTH]
        != PRIOR_TOKEN
        or len(parent_owner_ids) != 33
        or len(set(parent_owner_ids)) != 33
        or not bool(evaluation.get("all_parent_owners_retained"))
        or not bool(joint.get("natural_row_aligned_eos"))
        or dict(receipt.get("terminal_surface", {})).get("aggregate_sha256")
        != START_SURFACE_SHA256
        or dict(cold.get("surface", {})).get("aggregate_sha256") != START_SURFACE_SHA256
        or int(dict(cold.get("surface", {})).get("tensor_count", -1))
        != ota.R32_STEP_TENSOR_COUNT
        or int(dict(cold.get("surface", {})).get("element_count", -1))
        != ota.R32_STEP_ELEMENT_COUNT
        or start_state.get("frozen_surface_sha256") != FROZEN_SURFACE_SHA256
        or cold.get("frozen_surface") != FROZEN_SURFACE_SHA256
        or len(aliases) != ALIAS_COUNT
        or rejected
        or not compact_ok
        or _catalog_fingerprint(aliases) != CATALOG_SHA256
        or x1_tokens != X1_TOKENS
        or token_ids_sha256(x1_tokens) != X1_TOKENS_SHA256
        or {str(item.get("owner", "")) for item in aliases} != set(base.MISSING_OWNERS)
    ):
        raise NaturalBranchHold("HOLD: final r32 receipt/route/surface/catalog binding drifted")
    return {
        "route_tokens": route,
        "branch_prefix_tokens": route[:BRANCH_PREFIX_LENGTH],
        "parent_owner_ids": parent_owner_ids,
        "aliases": aliases,
        "x1_tokens": list(x1_tokens),
        "reference_packet_identity": ota._ota_state_identity(packet),
        "reference_raw_hard_counters": dict(joint.get("hard_raw_counters", {})),
        "parent_checkpoint": parent_checkpoint,
        "checkpoint_readback": readback,
    }


def _start_contract(*, verify_checkpoint_payload: bool = True) -> dict[str, Any]:
    if (
        not START_RECEIPT.is_file()
        or base._sha256(START_RECEIPT) != START_RECEIPT_SHA256
        or not START_CHECKPOINT.is_dir()
    ):
        raise NaturalBranchHold("HOLD: immutable owner-valid start artifact drifted")
    try:
        receipt = json.loads(START_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise NaturalBranchHold(f"HOLD: immutable owner-valid receipt is unreadable: {error}") from error
    if not isinstance(receipt, dict):
        raise NaturalBranchHold("HOLD: immutable owner-valid receipt is not an object")
    contract = _validate_receipt_payload(receipt)
    if verify_checkpoint_payload:
        live_readback = base.full_root.checkpoint_pair_receipt(
            contract["parent_checkpoint"], START_CHECKPOINT,
        )
        if live_readback != contract["checkpoint_readback"]:
            raise NaturalBranchHold("HOLD: final r32 checkpoint payload readback drifted")
    return {"receipt": receipt, **contract}


def _binding_receipt() -> dict[str, Any]:
    contract = _start_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_cuda",
        "start_receipt": str(START_RECEIPT),
        "start_receipt_sha256": START_RECEIPT_SHA256,
        "start_checkpoint": str(START_CHECKPOINT),
        "start_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "start_route_sha256": START_ROUTE_SHA256,
        "branch_prefix_length": BRANCH_PREFIX_LENGTH,
        "branch_prefix_sha256": BRANCH_PREFIX_SHA256,
        "reach_positions": list(REACH_POSITIONS),
        "reach_tokens": list(REACH_TOKENS),
        "current_token": CURRENT_TOKEN,
        "prior_token": PRIOR_TOKEN,
        "parent_owner_ids": contract["parent_owner_ids"],
        "alias_count": len(contract["aliases"]),
        "catalog_sha256": CATALOG_SHA256,
        "x1_token_count": len(contract["x1_tokens"]),
        "x1_tokens": contract["x1_tokens"],
        "x1_tokens_sha256": X1_TOKENS_SHA256,
        "world_size": WORLD_SIZE,
        "resource_bound": RESOURCE_BOUND,
    }


def _forced_completion(
    prefix_tokens: Sequence[int], x1_token: int, suffix_tokens: Sequence[int],
) -> list[int]:
    prefix = list(map(int, prefix_tokens))
    suffix = list(map(int, suffix_tokens))
    if (
        len(prefix) != BRANCH_PREFIX_LENGTH
        or token_ids_sha256(prefix) != BRANCH_PREFIX_SHA256
        or int(x1_token) not in X1_TOKENS
        or not suffix
        or base.EOS in suffix[:-1]
    ):
        raise NaturalBranchHold("HOLD: forced prefix+x1 natural release boundary drifted")
    full = [*prefix, int(x1_token), *suffix]
    if full[BRANCH_PREFIX_LENGTH] != int(x1_token):
        raise NaturalBranchHold("HOLD: forced x1 was not consumed at position 301")
    return full


def _owner_valid_gate(
    evaluation: Mapping[str, Any], *, parent_owner_ids: Sequence[str],
) -> dict[str, Any]:
    parent = set(map(str, parent_owner_ids))
    owners = set(map(str, evaluation.get("matched_target_owner_ids", ())))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    matcher = dict(joint.get("matcher", {}))
    ledger_matcher = dict(dict(evaluation.get("ledger", {})).get("matcher", {}))
    receipts = list(ledger_matcher.get("prediction_receipts", ()))
    statuses = [str(item.get("strict_match_status", "")) for item in receipts]
    status_counts = dict(matcher.get("strict_status_counts", {}))
    committed = set(map(str, matcher.get("committed_owner_ids", ())))
    raw = dict(joint.get("hard_raw_counters", {}))
    raw_schema_ok = set(raw) == set(RAW_COUNTER_KEYS) and all(
        isinstance(value, (bool, int)) and int(value) >= 0 for value in raw.values()
    )
    debt: dict[str, bool] = {
        "parent_identity": len(parent_owner_ids) != 33 or len(parent) != 33,
        "not_proper_superset": not owners > parent,
        "parent_loss": not parent.issubset(owners),
        "parser_drop": int(parser.get("dropped_prediction_count", -1)) != 0,
        "parser_cardinality": int(parser.get("valid_prediction_count", -1)) != len(receipts),
        "not_every_row_strict": statuses != ["matched"] * len(receipts)
        or status_counts != {"matched": len(receipts)},
        "matcher_cardinality": int(matcher.get("optimum_cardinality", -1)) != len(owners)
        or int(matcher.get("committed_owner_count", -1)) != len(owners)
        or committed != owners
        or int(ledger_matcher.get("optimum_cardinality", -1))
        != int(matcher.get("optimum_cardinality", -2))
        or set(map(str, ledger_matcher.get("committed_owner_ids", ()))) != committed,
        "matcher_ambiguity": bool(ledger_matcher.get("neutral_pred_row_ids"))
        or bool(ledger_matcher.get("ambiguity_receipts")),
        "raw_counter_schema": not raw_schema_ok,
        "row_aligned_eos": not bool(joint.get("natural_row_aligned_eos"))
        or not tokens
        or tokens[-1] != base.EOS
        or base.EOS in tokens[:-1],
        "token_budget": len(tokens) >= base.NATURAL_MAX_TOKENS,
    }
    for key in RAW_COUNTER_KEYS:
        debt[f"raw:{key}"] = not raw_schema_ok or int(raw.get(key, -1)) != 0
    active = {key: failed for key, failed in debt.items() if failed}
    return {
        "passed": not active,
        "debt": active,
        "matched_owner_ids": sorted(owners),
        "gained_owner_ids": sorted(owners - parent),
        "parent_owner_ids": sorted(parent),
        "raw_hard_counters": raw,
    }


def _conditional_record(
    *, catalog_index: int, x1_token: int, prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int], evaluation: Mapping[str, Any],
    parent_owner_ids: Sequence[str], static_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full = _forced_completion(prefix_tokens, x1_token, suffix_tokens)
    if list(map(int, evaluation.get("generated_token_ids", ()))) != full:
        raise NaturalBranchHold("HOLD: conditional evaluator route differs from forced completion")
    gate = _owner_valid_gate(evaluation, parent_owner_ids=parent_owner_ids)
    return {
        "catalog_index": int(catalog_index),
        "x1_token_id": int(x1_token),
        "forced_token_position": BRANCH_PREFIX_LENGTH,
        "forced_prefix_token_count": BRANCH_PREFIX_LENGTH + 1,
        "forced_prefix_sha256": token_ids_sha256([*map(int, prefix_tokens), int(x1_token)]),
        "suffix_token_ids": list(map(int, suffix_tokens)),
        "suffix_token_ids_sha256": token_ids_sha256(suffix_tokens),
        "generated_token_ids": full,
        "generated_token_ids_sha256": token_ids_sha256(full),
        "static_alias_count": len(static_aliases),
        "static_alias_row_sha256": sorted(str(item["row_sha256"]) for item in static_aliases),
        "positive": bool(gate["passed"]),
        "gate": gate,
        "evaluation": dict(evaluation),
    }


def _verify_discovery(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [dict(item) for item in records if item.get("evaluation_error") is not None]
    indices = [int(item.get("catalog_index", -1)) for item in records]
    tokens = [int(item.get("x1_token_id", -1)) for item in records]
    if errors:
        raise NaturalBranchHold(f"HOLD: forced conditional discovery failed: {errors}")
    if (
        len(records) != len(X1_TOKENS)
        or sorted(indices) != list(range(len(X1_TOKENS)))
        or len(indices) != len(set(indices))
        or sorted(tokens) != list(X1_TOKENS)
        or len(tokens) != len(set(tokens))
        or any(int(item.get("x1_token_id", -1)) != X1_TOKENS[int(item["catalog_index"])] for item in records)
        or any(int(item.get("forced_token_position", -1)) != BRANCH_PREFIX_LENGTH for item in records)
        or any(int(item.get("forced_prefix_token_count", -1)) != BRANCH_PREFIX_LENGTH + 1 for item in records)
        or any(
            token_ids_sha256(item.get("generated_token_ids", ()))
            != item.get("generated_token_ids_sha256")
            for item in records
        )
    ):
        raise NaturalBranchHold("HOLD: forced discovery exactly-once coverage/route binding drifted")
    ordered = sorted((dict(item) for item in records), key=lambda item: int(item["catalog_index"]))
    positives = [int(item["x1_token_id"]) for item in ordered if bool(item.get("positive"))]
    return {
        "records": ordered,
        "coverage": {
            "world_size": WORLD_SIZE,
            "expected_count": len(X1_TOKENS),
            "observed_count": len(ordered),
            "exactly_once": True,
            "x1_tokens_sha256": token_ids_sha256(X1_TOKENS),
        },
        "positive_tokens": positives,
        "positive_tokens_sha256": token_ids_sha256(positives),
    }


def _competitor_bundle(
    top_outside_positive: int, *, closure_token: int | None = None,
) -> list[dict[str, Any]]:
    ordered = [
        (CURRENT_TOKEN, "current_natural"),
        (PRIOR_TOKEN, "previous_natural"),
        (int(top_outside_positive), "highest_outside_P"),
    ]
    if closure_token is not None:
        ordered.append((int(closure_token), "closure_switch"))
    deduplicated: list[dict[str, Any]] = []
    for token, source in ordered:
        existing = next((item for item in deduplicated if item["token_id"] == token), None)
        if existing is None:
            deduplicated.append({"token_id": token, "sources": [source]})
        else:
            existing["sources"].append(source)
    return deduplicated


def _hard_disjunctive_selection(
    logits: torch.Tensor, positive_tokens: Sequence[int], *, closure_token: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    positives = sorted(set(map(int, positive_tokens)))
    if (
        logits.ndim != 1
        or not positives
        or min(positives) < 0
        or max(positives) >= logits.numel()
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise NaturalBranchHold("HOLD: malformed branch logits/conditional-positive set")
    positive_scores = logits[torch.tensor(positives, dtype=torch.long, device=logits.device)]
    positive_index = int(torch.argmax(positive_scores).item())
    target = positives[positive_index]
    outside = logits.detach().clone()
    outside[torch.tensor(positives, dtype=torch.long, device=logits.device)] = -torch.inf
    top_outside = int(torch.argmax(outside).item())
    competitors = _competitor_bundle(top_outside, closure_token=closure_token)
    if closure_token is not None and int(closure_token) in positives:
        raise NaturalBranchHold("HOLD: closure competitor is conditionally positive")
    margins = torch.stack([logits[target] - logits[item["token_id"]] for item in competitors])
    if not bool(torch.isfinite(margins).all().item()):
        raise NaturalBranchHold("HOLD: branch hard margins are non-finite")
    return margins, {
        "positive_tokens": positives,
        "positive_logits_sha256": base._hash([float(value) for value in positive_scores.detach().cpu()]),
        "p_star": target,
        "p_star_logit": float(logits[target].detach()),
        "highest_outside_P": top_outside,
        "highest_outside_P_logit": float(logits[top_outside].detach()),
        "competitors": [
            {
                **item,
                "logit": float(logits[item["token_id"]].detach()),
                "hard_margin": float(margin.detach()),
            }
            for item, margin in zip(competitors, margins, strict=True)
        ],
        "objective": "hard_p_star_vs_each_deduplicated_negative",
        "positive_aggregation": None,
    }


def _reach_terms(logits: torch.Tensor, prefix: Sequence[int]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if logits.ndim != 2 or logits.shape[0] <= max(REACH_POSITIONS):
        raise NaturalBranchHold("HOLD: branch teacher logits do not cover reach positions")
    terms, records = [], []
    for position, expected in zip(REACH_POSITIONS, REACH_TOKENS, strict=True):
        if int(prefix[position]) != expected:
            raise NaturalBranchHold("HOLD: frozen reach token drifted")
        row = logits[position]
        competitors = row.detach().clone()
        competitors[expected] = -torch.inf
        competitor = int(torch.argmax(competitors).item())
        term = row[expected] - row[competitor]
        terms.append(term)
        records.append({
            "cut_id": f"R{position}",
            "position": position,
            "target_token_id": expected,
            "competitor_token_id": competitor,
            "raw_margin": float(term.detach()),
            "floor": REACH_FLOOR,
            "solver_margin_offset": SAFETY_ETA - REACH_FLOOR,
            "prefix_sha256": token_ids_sha256(prefix[:position]),
        })
    return torch.stack(terms), records


def _active_row_order(branch: Mapping[str, Any], *, closure: bool) -> list[str]:
    order = [f"N{index}" for index, _item in enumerate(branch["competitors"])]
    order.extend(f"R{position}" for position in REACH_POSITIONS)
    order.extend(f"G{index}" for index in range(INCUMBENT_CUT_COUNT))
    maximum = MAX_CLOSURE_PANEL_ROWS if closure else MAX_FIRST_PANEL_ROWS
    if len(order) > maximum or len(order) != len(branch["competitors"]) + 11:
        raise NaturalBranchHold("HOLD: active gradient row count/order exceeded the frozen panel")
    return order


def _linearize(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    names: Sequence[str], parameters: Sequence[torch.nn.Parameter],
    route_tokens: Sequence[int], prefix_tokens: Sequence[int],
    positive_tokens: Sequence[int], closure_token: int | None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    teacher = [*map(int, prefix_tokens), CURRENT_TOKEN]
    branch_logits = base.full_root._teacher_forced_route_logits(
        model=model, native_inputs=native_inputs, route_tokens=teacher, pad_token_id=pad,
    )
    branch_terms, branch = _hard_disjunctive_selection(
        branch_logits[BRANCH_PREFIX_LENGTH], positive_tokens, closure_token=closure_token,
    )
    reach_terms, reach = _reach_terms(branch_logits, prefix_tokens)
    first_terms = torch.cat((branch_terms, reach_terms))
    first_gradients = ota._margin_gradients(first_terms, parameters)
    del branch_logits, first_terms, branch_terms, reach_terms

    guard_bindings = ota._seed_active_cut_bindings(route_tokens)
    if (
        len(guard_bindings) != INCUMBENT_CUT_COUNT
        or [item["cut_id"] for item in guard_bindings] != [f"G{i}" for i in range(7)]
    ):
        raise NaturalBranchHold("HOLD: frozen G0-G6 binding drifted")
    guard_logits = base.full_root._teacher_forced_route_logits(
        model=model, native_inputs=native_inputs,
        route_tokens=guard_bindings[0]["teacher_tokens"], pad_token_id=pad,
    )
    guard_terms = torch.stack([projected._margin(guard_logits, item) for item in guard_bindings])
    guard_margins = [float(value) for value in guard_terms.detach().cpu()]
    guard_gradients = ota._margin_gradients(guard_terms, parameters)
    del guard_logits, guard_terms

    gradients = [*first_gradients, *guard_gradients]
    closure = closure_token is not None
    order = _active_row_order(branch, closure=closure)
    if len(gradients) != len(order):
        raise NaturalBranchHold("HOLD: materialized gradient rows differ from sealed order")
    gram = ota._gradient_gram(gradients)
    target_margins = [float(item["hard_margin"]) for item in branch["competitors"]]
    raw_reach = [float(item["raw_margin"]) for item in reach]
    adjusted_reach = [value + SAFETY_ETA - REACH_FLOOR for value in raw_reach]
    solver = ota._solve_final_margin_span(
        target_margins, [*adjusted_reach, *guard_margins], gram,
        eta=SAFETY_ETA, delta=TRUST_DELTA,
    )
    direction = ota._materialize_span_direction(gradients, solver["coefficients"])
    materialized_norm = math.sqrt(max(0.0, projected._fp64_dot(direction, direction)))
    if not math.isclose(
        materialized_norm, float(solver["ordinary_l2_norm"]),
        rel_tol=1.0e-5, abs_tol=ota.SOLVER_TOLERANCE,
    ):
        raise NaturalBranchHold("HOLD: materialized natural-branch direction norm drifted")
    predicted_cuts = list(map(float, solver["predicted_cut_margins"]))
    predicted_raw_reach = [
        value - (SAFETY_ETA - REACH_FLOOR)
        for value in predicted_cuts[:len(REACH_POSITIONS)]
    ]
    if min(predicted_raw_reach) < REACH_FLOOR - ota.SOLVER_TOLERANCE:
        raise NaturalBranchHold("HOLD: solved reach cut fell below its strict floor")
    direction_snapshot = projected._direction_snapshot(names, direction)
    receipt = {
        "closure": closure,
        "closure_competitor": closure_token,
        "branch": branch,
        "reach_cuts": reach,
        "incumbent_cuts": [
            {key: value for key, value in item.items() if key != "teacher_tokens"}
            for item in guard_bindings
        ],
        "incumbent_cut_margins": guard_margins,
        "active_row_order": order,
        "active_gradient_count": len(order),
        "gradient_payload": {
            "shape": [len(order), ota.R32_STEP_ELEMENT_COUNT],
            "dtype": "torch.float32",
            "bytes": len(order) * ota.R32_STEP_ELEMENT_COUNT * 4,
            "stored_in_receipt": False,
        },
        "gradient_rows": [
            {
                "gradient_id": gradient_id,
                "ordinary_l2_norm": math.sqrt(max(0.0, float(gram[index, index]))),
            }
            for index, gradient_id in enumerate(order)
        ],
        "gram": {
            "shape": list(gram.shape),
            "matrix": gram.tolist(),
            "sha256": base._hash(gram.tolist()),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
        },
        "solver": solver | {
            "reach_floor": REACH_FLOOR,
            "predicted_raw_reach_margins": predicted_raw_reach,
        },
        "direction": direction_snapshot,
        "materialized_direction_l2_norm": materialized_norm,
    }
    del gradients, first_gradients, guard_gradients
    return direction, receipt


def _debt_increase(
    raw: Mapping[str, Any], baseline_raw: Mapping[str, Any],
) -> dict[str, int]:
    keys = set(raw) | set(baseline_raw)
    return {
        key: int(raw.get(key, 0)) - int(baseline_raw.get(key, 0))
        for key in sorted(keys)
        if int(raw.get(key, 0)) > int(baseline_raw.get(key, 0))
    }


def _candidate_record(
    *, packet: Mapping[str, Any], rank: int, radius: float,
    prefix_tokens: Sequence[int], positive_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], baseline_raw: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = dict(packet.get("evaluation", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    gate = _owner_valid_gate(evaluation, parent_owner_ids=parent_owner_ids)
    owners = set(map(str, gate["matched_owner_ids"]))
    parent = set(map(str, parent_owner_ids))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    raw = dict(joint.get("hard_raw_counters", {}))
    prefix_intact = tokens[:BRANCH_PREFIX_LENGTH] == list(map(int, prefix_tokens))
    token301 = tokens[BRANCH_PREFIX_LENGTH] if len(tokens) > BRANCH_PREFIX_LENGTH else None
    crossing = token301 in set(map(int, positive_tokens))
    increase = _debt_increase(raw, baseline_raw)
    conflict_reasons = {
        "earlier_prefix_divergence": not prefix_intact,
        "parent_loss": not parent.issubset(owners),
        "additional_raw_debt": bool(increase),
        "parser_drop": int(parser.get("dropped_prediction_count", -1)) != 0,
        "eos_or_budget_debt": bool(gate["debt"].get("row_aligned_eos"))
        or bool(gate["debt"].get("token_budget")),
        "unresolved_debt_with_owner_gain": owners > parent and not gate["passed"],
    }
    active_conflicts = {key: value for key, value in conflict_reasons.items() if value}
    return {
        "rank": int(rank),
        "radius": float(radius),
        "generated_token_ids": tokens,
        "generated_token_ids_sha256": token_ids_sha256(tokens),
        "packet_identity": ota._ota_state_identity(packet),
        "surface_sha256": str(surface.get("aggregate_sha256", "")),
        "branch_prefix_intact": prefix_intact,
        "reach_positions_intact": all(
            len(tokens) > position and tokens[position] == expected
            for position, expected in zip(REACH_POSITIONS, REACH_TOKENS, strict=True)
        ),
        "token301": token301,
        "token301_in_P": crossing,
        "proper_owner_superset": owners > parent,
        "owner_valid_gate": gate,
        "additional_raw_debt": increase,
        "preservation_conflict": bool(crossing and active_conflicts),
        "preservation_conflict_reasons": active_conflicts if crossing else {},
        "warm_promotion_screen": bool(crossing and prefix_intact and gate["passed"]),
    }


def _panel_decision(
    candidates: Sequence[Mapping[str, Any]], *, positive_tokens: Sequence[int],
    active_negative_tokens: Sequence[int], closure_used: bool,
) -> dict[str, Any]:
    errors = [dict(item.get("evaluation_error", {})) for item in candidates if item.get("evaluation_error")]
    by_radius = {float(item.get("radius", math.nan)): item for item in candidates}
    if errors:
        raise NaturalBranchHold(f"HOLD: natural radius evaluation failed: {errors}")
    if len(candidates) != WORLD_SIZE or set(by_radius) != set(RADII):
        raise NaturalBranchHold("HOLD: natural candidate panel is not exactly eight radii")
    promotions = [dict(item) for item in candidates if bool(item.get("warm_promotion_screen"))]
    if promotions:
        selected = max(
            promotions,
            key=lambda item: (
                len(item["owner_valid_gate"]["matched_owner_ids"]), float(item["radius"]),
            ),
        )
        return {"decision": "warm_promotion_screen", "selected": selected}
    crossings = [dict(item) for item in candidates if bool(item.get("token301_in_P"))]
    suffix_failures = [
        item for item in crossings
        if not item.get("preservation_conflict") and not item.get("proper_owner_superset")
    ]
    if suffix_failures:
        return {
            "decision": "branch_compiled_suffix_failed",
            "selected": max(suffix_failures, key=lambda item: float(item["radius"])),
        }
    conflicts = [item for item in crossings if bool(item.get("preservation_conflict"))]
    if conflicts:
        return {
            "decision": "preservation_conflict",
            "selected": max(conflicts, key=lambda item: float(item["radius"])),
        }
    positives = set(map(int, positive_tokens))
    active = set(map(int, active_negative_tokens))
    switches = [
        dict(item) for item in sorted(candidates, key=lambda item: -float(item["radius"]))
        if bool(item.get("branch_prefix_intact"))
        and item.get("token301") is not None
        and int(item["token301"]) not in positives
        and int(item["token301"]) not in active
    ]
    if switches and not closure_used:
        return {
            "decision": "need_closure",
            "closure_competitor": int(switches[0]["token301"]),
            "switch_candidates": [
                {"rank": item["rank"], "radius": item["radius"], "token301": item["token301"]}
                for item in switches
            ],
        }
    return {"decision": "no_safe_branch_crossing", "selected": None}


def _checkpoint_allowed(status: str) -> bool:
    return status == "warm_promotion_screen"


def _validate_saved_promotion(saved: Mapping[str, Any], checkpoint: Path) -> None:
    readback = dict(saved.get("readback", {}))
    payloads = {
        str(item.get("payload_id", "")): dict(item)
        for item in readback.get("payloads", ())
    }
    if (
        saved.get("checkpoint") != str(checkpoint)
        or readback.get("parent_checkpoint") != str(START_CHECKPOINT)
        or readback.get("child_checkpoint") != str(checkpoint)
        or set(payloads) != {"adapter", "special_token_embeddings"}
        or int(payloads["adapter"].get("tensor_key_count", -1)) != ota.R32_STEP_TENSOR_COUNT
        or int(payloads["adapter"].get("changed_tensor_count", 0)) <= 0
        or int(payloads["special_token_embeddings"].get("tensor_key_count", -1)) != 1
        or int(payloads["special_token_embeddings"].get("changed_tensor_count", -1)) != 0
    ):
        raise NaturalBranchHold("HOLD: promotion-only checkpoint readback drifted")


def _finalize_promotion(
    *, warm: Mapping[str, Any], cold_packet: Mapping[str, Any],
    cold_surface_sha256: str, frozen_surface_sha256: str,
    parent_owner_ids: Sequence[str],
) -> dict[str, Any]:
    cold_gate = _owner_valid_gate(cold_packet["evaluation"], parent_owner_ids=parent_owner_ids)
    identity_exact = ota._ota_state_identity(cold_packet) == warm.get("packet_identity")
    surface_exact = cold_surface_sha256 == warm.get("surface_sha256")
    frozen_exact = frozen_surface_sha256 == FROZEN_SURFACE_SHA256
    if not (cold_gate["passed"] and identity_exact and surface_exact and frozen_exact):
        raise NaturalBranchHold("HOLD: saved warm promotion failed authoritative cold parity/gate")
    return {
        "status": "cold_match_level_promotion",
        "cold_gate": cold_gate,
        "warm_cold_packet_identity_exact": identity_exact,
        "warm_cold_surface_exact": surface_exact,
        "frozen_surface_exact": frozen_exact,
    }


def _enforce_budget(counts: Mapping[str, int], *, mode: str, positives_empty: bool) -> None:
    expected_keys = {
        "forced_completions", "natural_decodes", "teacher_forwards", "linearizations",
        "gradient_backwards", "panel_rollouts", "checkpoint_saves", "cold_reloads",
    }
    if (
        set(counts) != expected_keys
        or int(counts.get("forced_completions", -1)) != len(X1_TOKENS)
        or int(counts.get("linearizations", -1)) > 2
        or int(counts.get("panel_rollouts", -1)) > 2 * WORLD_SIZE
        or int(counts.get("checkpoint_saves", -1)) > 1
        or int(counts.get("cold_reloads", -1)) != 1
        or int(counts.get("teacher_forwards", -1)) > RESOURCE_BOUND["teacher_forwards_total_max"]
        or int(counts.get("gradient_backwards", -1))
        > RESOURCE_BOUND["gradient_backwards_total_max"]
        or int(counts.get("natural_decodes", -1)) > RESOURCE_BOUND["natural_decodes_total_max"]
        or (mode == "vertical" and any(int(counts.get(key, 0)) for key in (
            "linearizations", "gradient_backwards", "panel_rollouts", "checkpoint_saves",
        )))
        or (positives_empty and any(int(counts.get(key, 0)) for key in (
            "linearizations", "gradient_backwards", "panel_rollouts", "checkpoint_saves",
        )))
    ):
        raise NaturalBranchHold("HOLD: owner-valid natural-branch execution budget drifted")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise NaturalBranchHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise NaturalBranchHold("HOLD: natural-branch runner snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _evaluate_forced(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    prefix: Sequence[int], x1_token: int, target: Mapping[str, Any], raw_example: Any,
    parent_owner_ids: Sequence[str], aliases: Sequence[Mapping[str, Any]], catalog_index: int,
) -> dict[str, Any]:
    forced = [*map(int, prefix), int(x1_token)]
    suffix = base.full_root._greedy_release(
        model=model, native_inputs=native_inputs, prefix=forced,
        eos_token_id=base.EOS, pad_token_id=pad,
    )
    full = _forced_completion(prefix, x1_token, suffix)
    evaluation = manifold._match_evaluation(
        tokenizer=tokenizer, tokens=full, target=target,
        witness_tokens=full, raw_example=raw_example,
        parent_owners=parent_owner_ids,
        label=f"owner-valid-forced-x1-{x1_token}",
    )
    static_aliases = [item for item in aliases if int(item["token_ids"][4]) == int(x1_token)]
    return _conditional_record(
        catalog_index=catalog_index, x1_token=x1_token, prefix_tokens=prefix,
        suffix_tokens=suffix, evaluation=evaluation, parent_owner_ids=parent_owner_ids,
        static_aliases=static_aliases,
    )


def _live_packet(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    target: Mapping[str, Any], raw_example: Any, parent_owner_ids: Sequence[str],
    alias_tokens: Sequence[int], label: str,
) -> dict[str, Any]:
    return projected._state_packet(
        model=model, tokenizer=tokenizer, native_inputs=native_inputs, pad=pad,
        target=target, raw_example=raw_example, parent_owners=parent_owner_ids,
        alias_tokens=alias_tokens, label=label,
    )


def _cold_state(
    checkpoint: Path, *, target: Mapping[str, Any], parent_owner_ids: Sequence[str],
    alias_tokens: Sequence[int], label: str,
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(checkpoint)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise NaturalBranchHold("HOLD: cold natural-branch gate requires concrete FP32 HF")
        model, tokenizer = opened._model, opened._tokenizer
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
            raise NaturalBranchHold("HOLD: cold natural-branch prompt identity drifted")
        names, parameters = ota._step_trainable_surface(model, r32_step=True)
        surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
        frozen = base._frozen_surface(model, names)
        packet = _live_packet(
            model=model, tokenizer=tokenizer, native_inputs=native_inputs,
            pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
            parent_owner_ids=parent_owner_ids, alias_tokens=alias_tokens, label=label,
        )
        return {
            "packet": packet,
            "surface": surface,
            "frozen_surface_sha256": frozen,
            "runtime": opened.receipt.to_artifact_dict(),
        }


def _assert_anchor_cold(cold: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    packet_exact = ota._ota_state_identity(cold["packet"]) == contract["reference_packet_identity"]
    surface_exact = dict(cold["surface"]).get("aggregate_sha256") == START_SURFACE_SHA256
    frozen_exact = cold.get("frozen_surface_sha256") == FROZEN_SURFACE_SHA256
    if not (packet_exact and surface_exact and frozen_exact):
        raise NaturalBranchHold("HOLD: nonpromotion terminal anchor cold identity drifted")
    return {
        "checkpoint": str(START_CHECKPOINT),
        "packet_identity_exact": packet_exact,
        "surface_exact": surface_exact,
        "frozen_surface_exact": frozen_exact,
        "surface": cold["surface"],
        "runtime": cold["runtime"],
    }


def _gather_totals(local: Mapping[str, int]) -> tuple[list[dict[str, int]], dict[str, int]]:
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, {"rank": dist.get_rank(), **dict(local)})
    totals = {
        key: sum(int(item[key]) for item in gathered)
        for key in local
    }
    return list(gathered), totals


def run(*, run_id: str, mode: str) -> Path:
    if mode not in {"vertical", "experiment"}:
        raise NaturalBranchHold("HOLD: mode must be vertical or experiment")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise NaturalBranchHold("HOLD: owner-valid natural branch requires torchrun --nproc_per_node=8")
    contract = _start_contract()  # Every rank fails frozen CPU bindings before CUDA/NCCL.
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    stage = "distributed_initialization"
    owns_output = False
    source_snapshot: Mapping[str, Any] | None = None
    saved: Mapping[str, Any] | None = None
    discovery: Mapping[str, Any] | None = None
    branch_sensitivity: Mapping[str, Any] | None = None
    panels: list[dict[str, Any]] = []
    closure: Mapping[str, Any] | None = None
    terminal_decision: Mapping[str, Any] | None = None
    local_counts = {
        "forced_completions": 0,
        "natural_decodes": 0,
        "teacher_forwards": 0,
        "linearizations": 0,
        "gradient_backwards": 0,
        "panel_rollouts": 0,
        "checkpoint_saves": 0,
        "cold_reloads": 0,
    }
    try:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise NaturalBranchHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        stage = "live_anchor_and_discovery"
        _parent_admission, _parent_setup, target = base._bindings()
        setup = base._setup_for_checkpoint(START_CHECKPOINT)
        route = contract["route_tokens"]
        prefix = contract["branch_prefix_tokens"]
        parent_owner_ids = contract["parent_owner_ids"]
        aliases = contract["aliases"]
        alias_tokens = list(map(int, aliases[0]["token_ids"]))
        initial_surface: Mapping[str, Any] | None = None
        runtime: Mapping[str, Any] | None = None
        selected_warm: Mapping[str, Any] | None = None
        selected_surface: Mapping[str, Any] | None = None

        with base.open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise NaturalBranchHold("HOLD: owner-valid run requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if (
                token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
            ):
                raise NaturalBranchHold("HOLD: live prompt/image identity drifted")
            names, parameters = ota._step_trainable_surface(model, r32_step=True)
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            clones = parallel._clone_parameters(parameters)
            initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_before = base._frozen_surface(model, names)
            if (
                initial_surface.get("aggregate_sha256") != START_SURFACE_SHA256
                or frozen_before != FROZEN_SURFACE_SHA256
            ):
                raise NaturalBranchHold("HOLD: loaded final r32 anchor surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)

            local_discovery: list[dict[str, Any]] = []
            local_discovery_error: dict[str, Any] | None = None
            try:
                for index in range(rank, len(X1_TOKENS), WORLD_SIZE):
                    local_discovery.append(_evaluate_forced(
                        model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                        pad=int(tokenizer.pad_token_id), prefix=prefix,
                        x1_token=X1_TOKENS[index], target=target,
                        raw_example=setup["raw_example"], parent_owner_ids=parent_owner_ids,
                        aliases=aliases, catalog_index=index,
                    ))
                    local_counts["forced_completions"] += 1
                    local_counts["natural_decodes"] += 1
            except BaseException as error:
                local_discovery_error = {"type": type(error).__name__, "error": str(error)}
            gathered_discovery: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_discovery, {
                "rank": rank,
                "records": local_discovery,
                "error": local_discovery_error,
            })

            def seal_discovery() -> dict[str, Any]:
                errors = [item for item in gathered_discovery if item["error"] is not None]
                if errors:
                    raise NaturalBranchHold(f"HOLD: distributed forced discovery failed: {errors}")
                return _verify_discovery([
                    record for item in gathered_discovery for record in item["records"]
                ])

            discovery = base._rank0_call(seal_discovery)
            positive_tokens = list(map(int, discovery["positive_tokens"]))
            sealed_discovery: list[Any] = [
                {
                    "positive_tokens": positive_tokens,
                    "positive_tokens_sha256": discovery["positive_tokens_sha256"],
                }
                if rank == 0 else None
            ]
            dist.broadcast_object_list(sealed_discovery, src=0)
            ota._agree_hash(sealed_discovery[0], label="conditional-positive set")
            base.full_root._assert_full_root_sentinels(model, sentinels)
            after_discovery = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if after_discovery != initial_surface or base._frozen_surface(model, names) != frozen_before:
                raise NaturalBranchHold("HOLD: no-update forced discovery mutated the anchor")

            if not positive_tokens:
                terminal_decision = {"decision": "empty_conditional_positive_set"}
            elif mode == "vertical":
                sensitivity_box: list[Any] = [None]
                if rank == 0:
                    with torch.inference_mode():
                        logits = base.full_root._teacher_forced_route_logits(
                            model=model, native_inputs=native_inputs,
                            route_tokens=[*prefix, CURRENT_TOKEN],
                            pad_token_id=int(tokenizer.pad_token_id),
                        )
                        _margins, sensitivity = _hard_disjunctive_selection(
                            logits[BRANCH_PREFIX_LENGTH], positive_tokens,
                        )
                    sensitivity_box[0] = sensitivity
                    local_counts["teacher_forwards"] += 1
                dist.broadcast_object_list(sensitivity_box, src=0)
                branch_sensitivity = sensitivity_box[0]
                ota._agree_hash(branch_sensitivity, label="vertical branch sensitivity")
                terminal_decision = {
                    "decision": "vertical_no_update_discovery_and_sensitivity_complete"
                }
            else:
                closure_token: int | None = None
                for panel_index in range(2):
                    stage = f"linearization_{panel_index + 1}"
                    parallel._restore_parameters(parameters, clones)
                    if base.full_root._full_root_surface_snapshot(names, parameters)[1] != initial_surface:
                        raise NaturalBranchHold("HOLD: panel did not start from immutable anchor")
                    gradient_box: list[Any] = [None]
                    direction: tuple[torch.Tensor, ...] | None = None
                    if rank == 0:
                        try:
                            direction, linearization = _linearize(
                                model=model, native_inputs=native_inputs,
                                pad=int(tokenizer.pad_token_id), names=names,
                                parameters=parameters, route_tokens=route,
                                prefix_tokens=prefix, positive_tokens=positive_tokens,
                                closure_token=closure_token,
                            )
                            gradient_box[0] = {"ok": True, "linearization": linearization}
                            local_counts["teacher_forwards"] += 2
                            local_counts["linearizations"] += 1
                            local_counts["gradient_backwards"] += int(linearization["active_gradient_count"])
                        except BaseException as error:
                            gradient_box[0] = {
                                "ok": False, "type": type(error).__name__, "error": str(error),
                            }
                    dist.broadcast_object_list(gradient_box, src=0)
                    if not gradient_box[0].get("ok"):
                        raise NaturalBranchHold(
                            "HOLD: rank-zero natural-branch linearization failed: "
                            f"{gradient_box[0].get('type')}: {gradient_box[0].get('error')}"
                        )
                    linearization = gradient_box[0]["linearization"]
                    ota._agree_hash(linearization, label=f"panel-{panel_index + 1} linearization/solver")
                    if rank != 0:
                        direction = tuple(torch.zeros_like(parameter) for parameter in parameters)
                    assert direction is not None
                    for value in direction:
                        dist.broadcast(value, src=0)
                    local_direction = projected._direction_snapshot(names, direction)
                    if local_direction != linearization["direction"]:
                        raise NaturalBranchHold("HOLD: broadcast natural-branch direction drifted")
                    ota._agree_hash(local_direction, label=f"panel-{panel_index + 1} direction")
                    active_negative_tokens = [
                        int(item["token_id"]) for item in linearization["branch"]["competitors"]
                    ]
                    branch_sensitivity = linearization["branch"]

                    stage = f"natural_panel_{panel_index + 1}"
                    parallel._restore_parameters(parameters, clones)
                    radius = parallel._radius_for_rank(rank)
                    ota._apply_trust_displacement(
                        parameters, clones, direction, radius=radius,
                    )
                    base.full_root._assert_full_root_sentinels(model, sentinels)
                    local_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                    try:
                        packet = _live_packet(
                            model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                            pad=int(tokenizer.pad_token_id), target=target,
                            raw_example=setup["raw_example"], parent_owner_ids=parent_owner_ids,
                            alias_tokens=alias_tokens,
                            label=f"owner-valid-natural-panel-{panel_index + 1}-rank-{rank}",
                        )
                        local_candidate = _candidate_record(
                            packet=packet, rank=rank, radius=radius, prefix_tokens=prefix,
                            positive_tokens=positive_tokens, parent_owner_ids=parent_owner_ids,
                            baseline_raw=contract["reference_raw_hard_counters"],
                            surface=local_surface,
                        )
                        local_counts["panel_rollouts"] += 1
                        local_counts["natural_decodes"] += 1
                    except BaseException as error:
                        local_candidate = {
                            "rank": rank, "radius": radius,
                            "evaluation_error": {"type": type(error).__name__, "error": str(error)},
                        }
                    gathered_candidates: list[Any] = [None] * WORLD_SIZE
                    dist.all_gather_object(gathered_candidates, local_candidate)
                    decision_box: list[Any] = [None]
                    if rank == 0:
                        decision_box[0] = _panel_decision(
                            gathered_candidates, positive_tokens=positive_tokens,
                            active_negative_tokens=active_negative_tokens,
                            closure_used=closure_token is not None,
                        )
                    dist.broadcast_object_list(decision_box, src=0)
                    decision = decision_box[0]
                    panels.append({
                        "panel_index": panel_index + 1,
                        "closure_competitor": closure_token,
                        "linearization": linearization,
                        "candidates": gathered_candidates,
                        "decision": decision,
                    })
                    if decision["decision"] == "need_closure":
                        parallel._restore_parameters(parameters, clones)
                        restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                        if restored != initial_surface:
                            raise NaturalBranchHold("HOLD: first panel leaked state before closure")
                        closure_token = int(decision["closure_competitor"])
                        closure = {
                            "run": True,
                            "source_panel": panel_index + 1,
                            "competitor_token_id": closure_token,
                            "switch_candidates": decision["switch_candidates"],
                        }
                        continue
                    terminal_decision = decision
                    if _checkpoint_allowed(decision["decision"]):
                        selected_warm = dict(decision["selected"])
                        selected_rank = int(selected_warm["rank"])
                        for parameter in parameters:
                            dist.broadcast(parameter.data, src=selected_rank)
                        selected_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                        if selected_surface.get("aggregate_sha256") != selected_warm.get("surface_sha256"):
                            raise NaturalBranchHold("HOLD: selected warm rank surface broadcast drifted")
                        checkpoint = output / "checkpoint-owner-valid-promotion"
                        saved = base._rank0_call(lambda model=model: {
                            "checkpoint": str(checkpoint),
                            "readback": base.full_root._save_weights_only_checkpoint(
                                model=model, source_checkpoint=START_CHECKPOINT,
                                destination=checkpoint,
                            ),
                        })
                        _validate_saved_promotion(saved, checkpoint)
                        if rank == 0:
                            local_counts["checkpoint_saves"] += 1
                    else:
                        parallel._restore_parameters(parameters, clones)
                        restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                        if restored != initial_surface:
                            raise NaturalBranchHold("HOLD: nonpromotion panel leaked a displacement")
                    break
                if terminal_decision is None:
                    raise NaturalBranchHold("HOLD: two-panel outcome accounting drifted")
            if saved is None:
                parallel._restore_parameters(parameters, clones)
                restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                if restored != initial_surface:
                    raise NaturalBranchHold("HOLD: terminal nonpromotion did not restore the anchor")
            base.full_root._assert_full_root_sentinels(model, sentinels)
            runtime = opened.receipt.to_artifact_dict()
            ota._agree_hash(runtime, label="runtime identity")

        del model, tokenizer, native_inputs, prompts, parameters, names, sentinels, clones
        torch.cuda.empty_cache()
        dist.barrier()
        stage = "authoritative_cold_gate"
        if saved is not None:
            cold = base._rank0_call(lambda: _cold_state(
                Path(str(saved["checkpoint"])), target=target,
                parent_owner_ids=parent_owner_ids, alias_tokens=alias_tokens,
                label="owner-valid-promotion-cold",
            ))
            assert selected_warm is not None and selected_surface is not None
            promotion = _finalize_promotion(
                warm=selected_warm, cold_packet=cold["packet"],
                cold_surface_sha256=cold["surface"]["aggregate_sha256"],
                frozen_surface_sha256=cold["frozen_surface_sha256"],
                parent_owner_ids=parent_owner_ids,
            )
            status = promotion["status"]
            cold_receipt = {
                **promotion,
                "checkpoint": str(saved["checkpoint"]),
                "packet_identity": ota._ota_state_identity(cold["packet"]),
                "surface": cold["surface"],
                "runtime": cold["runtime"],
            }
        else:
            cold = base._rank0_call(lambda: _cold_state(
                START_CHECKPOINT, target=target, parent_owner_ids=parent_owner_ids,
                alias_tokens=alias_tokens, label="owner-valid-nonpromotion-anchor-cold",
            ))
            cold_receipt = _assert_anchor_cold(cold, contract)
            status = str(terminal_decision["decision"])
        if rank == 0:
            local_counts["cold_reloads"] += 1
            local_counts["natural_decodes"] += 1

        local_resources = {
            "rank": rank,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(local_rank)),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(local_rank).total_memory),
        }
        gathered_resources: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_resources, local_resources)
        per_rank_counts, total_counts = _gather_totals(local_counts)
        _enforce_budget(total_counts, mode=mode, positives_empty=not discovery["positive_tokens"])
        elapsed = time.monotonic() - started
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or max(int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources)
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes_per_rank"]
        ):
            raise NaturalBranchHold("HOLD: owner-valid natural-branch runtime resource bound exceeded")
        if status not in TERMINAL_STATUSES:
            raise NaturalBranchHold("HOLD: terminal owner-valid status drifted")
        if rank == 0:
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": status,
                "run_id": run_id,
                "mode": mode,
                "runner_source_snapshot": source_snapshot,
                "bindings": _binding_receipt(),
                "protocol": {
                    "world_size": WORLD_SIZE,
                    "forced_completion": "exact prefix301+x1 through HF prefix injection then greedy EOS",
                    "forced_routes_are_promotion_eligible": False,
                    "positive_definition": "proper Parent33 superset, every row strict, zero hard debt",
                    "branch_objective": "hard p_star versus each deduplicated active negative",
                    "radii_by_rank": {str(index): radius for index, radius in enumerate(RADII)},
                    "reach_floor": REACH_FLOOR,
                    "incumbent_floor": SAFETY_ETA,
                    "trust_delta": TRUST_DELTA,
                    "surface": "588 FP32 r32/alpha64 language-DoRA tensors / 35438592 elements",
                    "optimizer": None,
                    "ce": None,
                    "fisher": None,
                    "aligner_update": False,
                    "tied_embedding_update": False,
                    "warm_is_screen_only": True,
                    "cold_saved_gate_is_authoritative": True,
                    "vertical_foreground_displacement": False if mode == "vertical" else None,
                },
                "discovery": discovery,
                "conditional_positive_tokens": discovery["positive_tokens"],
                "conditional_positive_tokens_sha256": discovery["positive_tokens_sha256"],
                "branch_sensitivity": branch_sensitivity,
                "panels": panels,
                "closure": closure or {"run": False},
                "terminal_decision": terminal_decision,
                "saved_checkpoint": saved,
                "authoritative_cold_gate": cold_receipt,
                "initial_surface": initial_surface,
                "counts": {"per_rank": per_rank_counts, "total": total_counts},
                "resources": {
                    "per_rank": gathered_resources,
                    "max_peak_cuda_allocated_bytes": max(
                        int(item["peak_cuda_allocated_bytes"]) for item in gathered_resources
                    ),
                    "max_peak_cuda_reserved_bytes": max(
                        int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources
                    ),
                    "predeclared_bound": RESOURCE_BOUND,
                },
                "runtime": runtime,
                "wall_time_seconds": elapsed,
                "claim_boundary": (
                    "The vertical is no-update discovery and branch sensitivity only; it never applies "
                    "foreground displacement and cannot promote."
                    if mode == "vertical" else
                    "One cold single-image owner promotion, or a bounded negative limited to the sealed "
                    "39-token catalog, exact prefix, r32 trust region, active competitors, and two panels."
                ),
            }
            base._atomic_json(output / "receipt.json", receipt)
            if (output / "receipt.json").stat().st_size > RESOURCE_BOUND["output_artifact_bytes_max"]:
                raise NaturalBranchHold("HOLD: natural-branch receipt exceeded artifact bound")
        dist.barrier()
        return output
    except BaseException as error:
        if dist.is_initialized() and dist.get_rank() == 0 and owns_output:
            base._atomic_json(output / "receipt.json", {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": "HOLD",
                "run_id": run_id,
                "mode": mode,
                "hold_stage": stage,
                "stop_reason": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "runner_source_snapshot": source_snapshot,
                "start_receipt": str(START_RECEIPT),
                "start_receipt_sha256": START_RECEIPT_SHA256,
                "saved_checkpoint": saved,
                "counts_rank0_partial": local_counts,
                "wall_time_seconds": time.monotonic() - started,
            })
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true", help="CPU-only exact frozen check")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mode", choices=("vertical", "experiment"),
        help="vertical is no-update; experiment is the atomic authorized foreground run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id or args.mode:
            raise SystemExit("--check-bindings is CPU-only and cannot be combined with a GPU run")
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if not args.run_id or args.mode is None:
        raise SystemExit("GPU execution requires --run-id and exactly one --mode")
    print(run(run_id=args.run_id, mode=args.mode))


if __name__ == "__main__":
    main()
