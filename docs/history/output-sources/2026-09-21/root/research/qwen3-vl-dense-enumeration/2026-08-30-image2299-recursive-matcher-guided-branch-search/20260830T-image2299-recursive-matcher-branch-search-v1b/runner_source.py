#!/usr/bin/env python3
"""World8 recursive matcher-guided person branch search for frozen Image2299 r32."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_manifold_match_anchor_overfit as manifold
from scripts.research import run_image2299_ota_sqp_lite as ota
from scripts.research import run_image2299_owner_valid_natural_branch as owner_valid
from src.inference.backend import token_ids_sha256


base = ota.base

SCHEMA_VERSION = "image2299.recursive_matcher_branch_search.v1"
UNIT_ID = "2026-08-30-image2299-recursive-matcher-guided-branch-search"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
MODEL_RECEIPT = owner_valid.START_RECEIPT
MODEL_RECEIPT_SHA256 = "a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c"
START_CHECKPOINT = owner_valid.START_CHECKPOINT
DISCOVERY_RECEIPT = (
    owner_valid.OUTPUT_ROOT
    / "20260830T-image2299-owner-valid-natural-branch-vertical-v1"
    / "receipt.json"
)
DISCOVERY_RECEIPT_SHA256 = "434a9488bfef3633d76ad1b0204b8ee0a62ef1615bf118a68ea5532572c80ca8"
START_SURFACE_SHA256 = "2ac0e8a963d91ac7a4ba42322272889efd74bd46578c1ac44e16e531d1246676"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
AUTHORITY_DESCRIPTION_MAP_SHA256 = "288baf5b9fb9cdb598e64c7bcdbcf8a2f3814fce392106be2dfd252a85b4e354"
CONTROL_PARENT_ROUTE_SHA256 = "9e5969757e8ddc419b6a3e2affc90917c27e830718b713fd7c458f9577f82751"
WORLD_SIZE = 8
PERSON_OWNER_COUNT = 38
TIE_OWNER_COUNT = 8
DEPTH_ZERO_PERSON_COUNT = 33
MAX_INTERVENTIONS = 6
MAX_WARM_COMPLETIONS = 100
MAX_REPLAY_COMPLETIONS = 6
MAX_DYNAMIC_ACTIONS = 15
ROW_OPEN = (151646, 8987, 151647, 151648)
RAW_COUNTER_KEYS = tuple(sorted(ota.projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS))
DEPTH_ZERO_TOKENS = (152030, 152031, 152032, 152041, 152042, 152043)
DEPTH_ZERO_ROUTE_SHA256 = {
    152030: "0c472a5c44d34ecac580e3a51f68fba3444c7553e10038b7144d91d1d1d9a6f5",
    152031: "9b2696cbefcad85c07f9c73f1cdb1869cdfb8bf46076eeb8dd67f25726ae069e",
    152032: "7a39d314548b8f8a7cbae23727a110e836b0c449aa459a0fe3e31fe93d4147d0",
    152041: "c359aacde19248641d91f39df779bf5617aa6a0cc942e62460325a3a951af3f9",
    152042: "7ea17cbab5a83a572dc7b18651b2fe5c872bef1395d31ee1ff8d0ab838e9345e",
    152043: "98ff8f94ddefa8e6b71f5fd3e7e3940788aae86f1a434138011134af6095e485",
}
DEPTH_ZERO_LEDGER_SHA256 = {
    152030: "7885f4300d6b55394ae6a2408f30d47e449e36e9631726638b0ff43f6a254003",
    152031: "b1db836c21329be0f9cbf96a3dcfa7c9b20cc910fe95acc10457e498d53ca828",
    152032: "6601ab223c20d2eec45f67251dc2cbcc4d41ce5247cc660f1daf7c9fa8e1df8d",
    152041: "01dc688abfbaac2b061afc07992a3efeb216cb041f153574dd90997a58566510",
    152042: "059287b9bfbb18e0457e4692ce0964975df8e8c6e65e32722413ec0f28460020",
    152043: "6f22309c110419d2b13d21ebb3aa242a4fef50b4f1d439b8aa113f241d980312",
}
RESOURCE_BOUND = {
    "gpu_count": WORLD_SIZE,
    "cumulative_interventions_max": MAX_INTERVENTIONS,
    "warm_controlled_completions_max": MAX_WARM_COMPLETIONS,
    "cold_replay_completions_max": MAX_REPLAY_COMPLETIONS,
    "branch_logit_forwards_max": MAX_INTERVENTIONS,
    "generated_tokens_per_decode_max": base.NATURAL_MAX_TOKENS,
    "model_forward_calls_total_max": (
        (MAX_WARM_COMPLETIONS + MAX_REPLAY_COMPLETIONS) * base.NATURAL_MAX_TOKENS
        + MAX_INTERVENTIONS
    ),
    "max_peak_cuda_reserved_bytes_per_rank": 64 * 2**30,
    "output_artifact_bytes_max": 200_000_000,
    "wall_time_seconds_max": 1_200,
}
TERMINAL_STATUSES = {
    "controlled_38_person_success",
    "controlled_search_exhausted",
    "controlled_depth_exhausted",
}


class RecursiveBranchHold(RuntimeError):
    """An immutable identity, search, collective, replay, or resource contract failed."""


def _load_receipt(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    if not path.is_file() or base._sha256(path) != digest:
        raise RecursiveBranchHold(f"HOLD: immutable {label} receipt drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RecursiveBranchHold(f"HOLD: immutable {label} receipt unreadable: {error}") from error
    if not isinstance(value, dict):
        raise RecursiveBranchHold(f"HOLD: immutable {label} receipt is not an object")
    return value


def _description_map(
    authority_owners: Sequence[Mapping[str, Any]], target: Mapping[str, Any],
) -> dict[str, str]:
    authority = {
        str(item.get("gt_owner_id", "")): str(item.get("description", ""))
        for item in authority_owners
    }
    rows = list(target.get("rows", ()))
    target_map = {
        str(item.get("owner", "")): str(item.get("description", "")) for item in rows
    }
    if (
        len(authority) != PERSON_OWNER_COUNT + TIE_OWNER_COUNT
        or len(target_map) != len(authority)
        or authority != target_map
        or base._hash(authority) != AUTHORITY_DESCRIPTION_MAP_SHA256
        or sum(value == "person" for value in authority.values()) != PERSON_OWNER_COUNT
        or sum(value == "tie" for value in authority.values()) != TIE_OWNER_COUNT
        or set(authority.values()) != {"person", "tie"}
        or dict(target.get("global_match", {})).get("person_prediction_count")
        != PERSON_OWNER_COUNT
        or dict(target.get("global_match", {})).get("tie_prediction_count") != TIE_OWNER_COUNT
    ):
        raise RecursiveBranchHold("HOLD: authority/target description map drifted")
    return authority


def _node_gate(
    evaluation: Mapping[str, Any], *, parent_owner_ids: Sequence[str],
) -> dict[str, Any]:
    parent_list = list(map(str, parent_owner_ids))
    parent = set(parent_list)
    owner_list = list(map(str, evaluation.get("matched_target_owner_ids", ())))
    owners = set(owner_list)
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    compact = dict(joint.get("matcher", {}))
    ledger = dict(evaluation.get("ledger", {}))
    ledger_parser = dict(ledger.get("parse", {}))
    ledger_matcher = dict(ledger.get("matcher", {}))
    receipts = list(ledger_matcher.get("prediction_receipts", ()))
    statuses = [str(item.get("strict_match_status", "")) for item in receipts]
    compact_committed = set(map(str, compact.get("committed_owner_ids", ())))
    ledger_committed = set(map(str, ledger_matcher.get("committed_owner_ids", ())))
    raw = dict(joint.get("hard_raw_counters", {}))
    raw_schema_ok = set(raw) == set(RAW_COUNTER_KEYS) and all(
        isinstance(value, (bool, int)) and int(value) >= 0 for value in raw.values()
    )
    debt: dict[str, bool] = {
        "parent_identity": not parent or len(parent_list) != len(parent),
        "owner_identity": not owners or len(owner_list) != len(owners),
        "not_proper_superset": not owners > parent,
        "parent_loss": not parent.issubset(owners),
        "parser_drop": int(parser.get("dropped_prediction_count", -1)) != 0
        or int(ledger_parser.get("dropped_prediction_count", -1)) != 0,
        "parser_cardinality": int(parser.get("valid_prediction_count", -1)) != len(receipts)
        or len(list(ledger_parser.get("predictions", ()))) != len(receipts),
        "not_every_prediction_strict": statuses != ["matched"] * len(receipts)
        or dict(compact.get("strict_status_counts", {})) != {"matched": len(receipts)},
        "compact_matcher": int(compact.get("optimum_cardinality", -1)) != len(owners)
        or int(compact.get("committed_owner_count", -1)) != len(owners)
        or compact_committed != owners,
        "ledger_matcher": int(ledger_matcher.get("optimum_cardinality", -1))
        != int(compact.get("optimum_cardinality", -2))
        or ledger_committed != compact_committed,
        "matcher_ambiguity": bool(ledger_matcher.get("neutral_pred_row_ids"))
        or bool(ledger_matcher.get("ambiguity_receipts")),
        "raw_counter_schema": not raw_schema_ok,
        "evaluator_hard_counter": int(evaluation.get("hard_counter_count", -1)) != 0,
        "row_aligned_eos": not bool(joint.get("natural_row_aligned_eos"))
        or not tokens
        or tokens[-1] != base.EOS
        or base.EOS in tokens[:-1]
        or (len(tokens) - 1) % base.ROW_TOKENS != 0,
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


def _validate_depth_zero(
    receipt: Mapping[str, Any], *, parent_owner_ids: Sequence[str],
    descriptions: Mapping[str, str],
) -> list[dict[str, Any]]:
    source = dict(receipt.get("runner_source_snapshot", {}))
    source_path = Path(str(source.get("path", "")))
    records = [
        dict(item) for item in dict(receipt.get("discovery", {})).get("records", ())
        if int(item.get("x1_token_id", -1)) in DEPTH_ZERO_TOKENS
    ]
    by_token = {int(item.get("x1_token_id", -1)): item for item in records}
    coverage = dict(dict(receipt.get("discovery", {})).get("coverage", {}))
    if (
        receipt.get("schema_version") != owner_valid.SCHEMA_VERSION
        or receipt.get("unit_id") != owner_valid.UNIT_ID
        or receipt.get("status") != "vertical_no_update_discovery_and_sensitivity_complete"
        or receipt.get("mode") != "vertical"
        or not source_path.is_file()
        or base._sha256(source_path) != source.get("sha256")
        or tuple(sorted(by_token)) != DEPTH_ZERO_TOKENS
        or len(records) != len(DEPTH_ZERO_TOKENS)
        or coverage.get("world_size") != WORLD_SIZE
        or coverage.get("expected_count") != len(owner_valid.X1_TOKENS)
        or coverage.get("observed_count") != len(owner_valid.X1_TOKENS)
        or coverage.get("exactly_once") is not True
    ):
        raise RecursiveBranchHold("HOLD: owner-valid discovery identity/six-route set drifted")
    parent = set(map(str, parent_owner_ids))
    expected_gain = {"gt:2299:0", "gt:2299:1", "gt:2299:20"}
    sealed: list[dict[str, Any]] = []
    for token in DEPTH_ZERO_TOKENS:
        record = by_token[token]
        evaluation = dict(record.get("evaluation", {}))
        route = list(map(int, record.get("generated_token_ids", ())))
        gate = _node_gate(evaluation, parent_owner_ids=parent_owner_ids)
        owners = set(map(str, gate["matched_owner_ids"]))
        if (
            not bool(record.get("positive"))
            or not gate["passed"]
            or route != list(map(int, evaluation.get("generated_token_ids", ())))
            or len(route) != 325
            or route[owner_valid.BRANCH_PREFIX_LENGTH] != token
            or token_ids_sha256(route) != DEPTH_ZERO_ROUTE_SHA256[token]
            or record.get("generated_token_ids_sha256") != DEPTH_ZERO_ROUTE_SHA256[token]
            or base._hash(dict(evaluation.get("ledger", {})))
            != DEPTH_ZERO_LEDGER_SHA256[token]
            or owners - parent != expected_gain
            or sum(descriptions.get(item) == "person" for item in owners)
            != DEPTH_ZERO_PERSON_COUNT
            or len(owners) != 36
        ):
            raise RecursiveBranchHold(f"HOLD: depth-zero route/ledger drifted for x1={token}")
        sealed.append(record)
    return sealed


def _start_contract(*, verify_checkpoint_payload: bool = True) -> dict[str, Any]:
    model_receipt = _load_receipt(MODEL_RECEIPT, MODEL_RECEIPT_SHA256, label="model")
    try:
        model = owner_valid._validate_receipt_payload(model_receipt)
    except BaseException as error:
        raise RecursiveBranchHold(f"HOLD: model receipt binding failed: {error}") from error
    if verify_checkpoint_payload:
        live_readback = base.full_root.checkpoint_pair_receipt(
            model["parent_checkpoint"], START_CHECKPOINT,
        )
        if live_readback != model["checkpoint_readback"]:
            raise RecursiveBranchHold("HOLD: selected-r32-step checkpoint readback drifted")
    discovery_receipt = _load_receipt(
        DISCOVERY_RECEIPT, DISCOVERY_RECEIPT_SHA256, label="discovery",
    )
    _admission, _parent_setup, target = base._bindings()
    setup = base._setup_for_checkpoint(START_CHECKPOINT)
    _line, authority_owners, authority_source = base._load_authority()
    descriptions = _description_map(authority_owners, target)
    parent_route = [*map(int, model["route_tokens"][:297]), base.EOS]
    parent_owners = list(map(str, model["parent_owner_ids"]))
    if (
        model_receipt.get("runner_source_snapshot") is None
        or dict(discovery_receipt.get("bindings", {})).get("start_receipt_sha256")
        != MODEL_RECEIPT_SHA256
        or dict(discovery_receipt.get("bindings", {})).get("start_surface_sha256")
        != START_SURFACE_SHA256
        or dict(discovery_receipt.get("bindings", {})).get("frozen_surface_sha256")
        != FROZEN_SURFACE_SHA256
        or model_receipt.get("terminal_surface", {}).get("aggregate_sha256")
        != START_SURFACE_SHA256
        or token_ids_sha256(parent_route) != CONTROL_PARENT_ROUTE_SHA256
        or len(parent_route) != 298
        or parent_route[-1] != base.EOS
        or len(parent_owners) != 33
        or len(set(parent_owners)) != 33
        or not set(parent_owners).issubset(descriptions)
        or setup["plan"].get("prompt_token_ids_sha256") != base.PROMPT_TOKEN_SHA256
        or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
        or target.get("token_ids_sha256") != setup["target"].get("token_ids_sha256")
    ):
        raise RecursiveBranchHold("HOLD: source/checkpoint/prompt/image/Parent33 binding drifted")
    aliases = list(model["aliases"])
    depth_zero = _validate_depth_zero(
        discovery_receipt, parent_owner_ids=parent_owners, descriptions=descriptions,
    )
    return {
        "model_receipt": model_receipt,
        "discovery_receipt": discovery_receipt,
        "checkpoint_readback": model["checkpoint_readback"],
        "parent_route_tokens": parent_route,
        "parent_owner_ids": parent_owners,
        "aliases": aliases,
        "description_map": descriptions,
        "target": target,
        "setup": setup,
        "authority_source": authority_source,
        "depth_zero_records": depth_zero,
    }


def _binding_receipt() -> dict[str, Any]:
    contract = _start_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "verified",
        "execution_surface": "cpu_only_no_cuda",
        "model_receipt": str(MODEL_RECEIPT),
        "model_receipt_sha256": MODEL_RECEIPT_SHA256,
        "discovery_receipt": str(DISCOVERY_RECEIPT),
        "discovery_receipt_sha256": DISCOVERY_RECEIPT_SHA256,
        "checkpoint": str(START_CHECKPOINT),
        "checkpoint_readback": contract["checkpoint_readback"],
        "start_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "prompt_token_ids_sha256": base.PROMPT_TOKEN_SHA256,
        "image_sha256": base.IMAGE_SHA256,
        "target_sha256": base.TARGET_SHA256,
        "authority": contract["authority_source"],
        "authority_description_map_sha256": AUTHORITY_DESCRIPTION_MAP_SHA256,
        "person_owner_count": PERSON_OWNER_COUNT,
        "tie_owner_count": TIE_OWNER_COUNT,
        "parent_owner_ids": contract["parent_owner_ids"],
        "control_parent_route_sha256": CONTROL_PARENT_ROUTE_SHA256,
        "alias_count": len(contract["aliases"]),
        "alias_catalog_sha256": owner_valid.CATALOG_SHA256,
        "depth_zero_x1_tokens": list(DEPTH_ZERO_TOKENS),
        "depth_zero_route_sha256": DEPTH_ZERO_ROUTE_SHA256,
        "depth_zero_ledger_sha256": DEPTH_ZERO_LEDGER_SHA256,
        "world_size": WORLD_SIZE,
        "resource_bound": RESOURCE_BOUND,
    }


def _missing_person_actions(
    parent_owner_ids: Sequence[str], *, descriptions: Mapping[str, str],
    aliases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parent = set(map(str, parent_owner_ids))
    if not parent.issubset(descriptions):
        raise RecursiveBranchHold("HOLD: node owners are outside authority")
    missing_people = {
        owner_id for owner_id, description in descriptions.items()
        if description == "person" and owner_id not in parent
    }
    actions: dict[int, dict[str, Any]] = {}
    for alias in aliases:
        owner_id = str(alias.get("owner", ""))
        tokens = list(map(int, alias.get("token_ids", ())))
        if (
            len(tokens) != base.ROW_TOKENS
            or token_ids_sha256(tokens) != alias.get("row_sha256")
        ):
            raise RecursiveBranchHold("HOLD: sealed OTA alias row drifted")
        if owner_id not in missing_people:
            continue
        if tuple(tokens[:4]) != ROW_OPEN:
            raise RecursiveBranchHold("HOLD: missing-person alias grammar drifted")
        x1 = tokens[4]
        item = actions.setdefault(x1, {"x1_token_id": x1, "owner_ids": [], "aliases": []})
        if owner_id not in item["owner_ids"]:
            item["owner_ids"].append(owner_id)
        item["aliases"].append({
            "owner": owner_id,
            "row_sha256": str(alias.get("row_sha256", "")),
            "witness_sha256": str(alias.get("witness_sha256", "")),
        })
    result = [actions[token] for token in sorted(actions)]
    for item in result:
        item["owner_ids"].sort()
        item["aliases"].sort(key=lambda value: (value["owner"], value["row_sha256"]))
    if len(result) > MAX_DYNAMIC_ACTIONS:
        raise RecursiveBranchHold("HOLD: dynamic missing-person action bound exceeded")
    return result


def _splice_branch_prefix(node_tokens: Sequence[int], x1_token: int) -> dict[str, Any]:
    node = list(map(int, node_tokens))
    if (
        not node
        or node[-1] != base.EOS
        or base.EOS in node[:-1]
        or (len(node) - 1) % base.ROW_TOKENS != 0
        or len(node) >= base.NATURAL_MAX_TOKENS
    ):
        raise RecursiveBranchHold("HOLD: branch parent is not a row-aligned terminal route")
    position = len(node) - 1 + len(ROW_OPEN)
    prefix = [*node[:-1], *ROW_OPEN, int(x1_token)]
    if prefix[position] != int(x1_token):
        raise RecursiveBranchHold("HOLD: x1 forced coordinate position drifted")
    return {
        "tokens": prefix,
        "token_ids_sha256": token_ids_sha256(prefix),
        "x1_position": position,
    }


def _forced_completion(
    node_tokens: Sequence[int], x1_token: int, suffix_tokens: Sequence[int],
) -> dict[str, Any]:
    binding = _splice_branch_prefix(node_tokens, x1_token)
    suffix = list(map(int, suffix_tokens))
    full = [*binding["tokens"], *suffix]
    if (
        not suffix
        or suffix[-1] != base.EOS
        or base.EOS in suffix[:-1]
        or full[binding["x1_position"]] != int(x1_token)
        or (len(full) - 1) % base.ROW_TOKENS != 0
    ):
        raise RecursiveBranchHold("HOLD: forced prefix did not release to row-aligned EOS")
    return {
        "forced_prefix_tokens": binding["tokens"],
        "forced_prefix_sha256": binding["token_ids_sha256"],
        "x1_position": binding["x1_position"],
        "suffix_token_ids": suffix,
        "suffix_token_ids_sha256": token_ids_sha256(suffix),
        "generated_token_ids": full,
        "generated_token_ids_sha256": token_ids_sha256(full),
    }


def _logit_packet(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    node_tokens: Sequence[int], actions: Sequence[Mapping[str, Any]], depth: int,
) -> dict[str, Any]:
    tokens = [int(item["x1_token_id"]) for item in actions]
    if not tokens or len(tokens) != len(set(tokens)):
        raise RecursiveBranchHold("HOLD: branch-logit action catalog is empty or duplicated")
    binding = _splice_branch_prefix(node_tokens, tokens[0])
    branch_without_x1 = binding["tokens"][:-1]
    with torch.inference_mode():
        logits = base.full_root._teacher_forced_route_logits(
            model=model,
            native_inputs=native_inputs,
            route_tokens=[*branch_without_x1, tokens[0]],
            pad_token_id=pad,
        )
        vector = logits[binding["x1_position"]]
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all().item()):
        raise RecursiveBranchHold("HOLD: branch x1 logit vector is malformed")
    values = {str(token): float(vector[token].item()) for token in tokens}
    return {
        "depth": int(depth),
        "x1_position": binding["x1_position"],
        "branch_prefix_sha256": token_ids_sha256(branch_without_x1),
        "x1_tokens": tokens,
        "values": values,
        "values_sha256": base._hash(values),
        "full_logit_vector_sha256": base.full_root._tensor_sha256(vector),
    }


def _person_count(owner_ids: Sequence[str], descriptions: Mapping[str, str]) -> int:
    owners = set(map(str, owner_ids))
    if not owners.issubset(descriptions):
        raise RecursiveBranchHold("HOLD: matched owner outside authority")
    return sum(descriptions[item] == "person" for item in owners)


def _candidate_value(candidate: Mapping[str, Any]) -> tuple[int, int, int, float, int]:
    return (
        int(candidate["matched_person_count"]),
        int(candidate["matched_owner_count"]),
        -int(candidate["cumulative_intervention_count"]),
        float(candidate["x1_logit"]),
        -int(candidate["x1_token_id"]),
    )


def _select_best(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    admitted = [dict(item) for item in candidates if bool(item.get("admitted"))]
    if not admitted:
        raise RecursiveBranchHold("HOLD: cannot select from an empty admitted set")
    return max(admitted, key=_candidate_value)


def _terminal_status(
    *, matched_person_count: int, cumulative_interventions: int, admitted_child_count: int,
) -> str | None:
    if matched_person_count == PERSON_OWNER_COUNT:
        return "controlled_38_person_success"
    if not 0 <= matched_person_count < PERSON_OWNER_COUNT:
        raise RecursiveBranchHold("HOLD: matched-person denominator drifted")
    if admitted_child_count == 0:
        return "controlled_search_exhausted"
    if cumulative_interventions >= MAX_INTERVENTIONS:
        return "controlled_depth_exhausted"
    return None


def _candidate_record(
    *, action_index: int, action: Mapping[str, Any], parent_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], suffix_tokens: Sequence[int],
    evaluation: Mapping[str, Any], descriptions: Mapping[str, str],
    logit_packet: Mapping[str, Any], cumulative_intervention_count: int,
) -> dict[str, Any]:
    x1 = int(action["x1_token_id"])
    completion = _forced_completion(parent_tokens, x1, suffix_tokens)
    if list(map(int, evaluation.get("generated_token_ids", ()))) != completion["generated_token_ids"]:
        raise RecursiveBranchHold("HOLD: evaluator route differs from forced completion")
    if int(logit_packet.get("x1_position", -1)) != completion["x1_position"]:
        raise RecursiveBranchHold("HOLD: x1 logit/consumption coordinate drifted")
    gate = _node_gate(evaluation, parent_owner_ids=parent_owner_ids)
    owners = list(gate["matched_owner_ids"])
    return {
        "action_index": int(action_index),
        "x1_token_id": x1,
        "action_owner_ids": list(map(str, action.get("owner_ids", ()))),
        "action_aliases": list(action.get("aliases", ())),
        **completion,
        "x1_logit": float(dict(logit_packet["values"])[str(x1)]),
        "matched_person_count": _person_count(owners, descriptions),
        "matched_owner_count": len(owners),
        "matched_owner_ids": owners,
        "cumulative_intervention_count": int(cumulative_intervention_count),
        "admitted": bool(gate["passed"]),
        "gate": gate,
        "evaluation": dict(evaluation),
    }


def _verify_exactly_once(
    gathered: Sequence[Mapping[str, Any]], *, actions: Sequence[Mapping[str, Any]], depth: int,
) -> list[dict[str, Any]]:
    ranks = [int(item.get("rank", -1)) for item in gathered]
    errors = [dict(item) for item in gathered if item.get("error") is not None]
    records = [dict(record) for item in gathered for record in item.get("records", ())]
    indices = [int(item.get("action_index", -1)) for item in records]
    expected_tokens = [int(item["x1_token_id"]) for item in actions]
    observed_tokens = [int(item.get("x1_token_id", -1)) for item in records]
    if errors or any(item.get("evaluation_error") is not None for item in records):
        raise RecursiveBranchHold(f"HOLD: depth {depth} candidate evaluation failed")
    if (
        sorted(ranks) != list(range(WORLD_SIZE))
        or len(ranks) != len(set(ranks))
        or sorted(indices) != list(range(len(actions)))
        or len(indices) != len(set(indices))
        or observed_tokens != [expected_tokens[index] for index in indices]
        or any(index % WORLD_SIZE != int(record.get("rank", -1)) for index, record in zip(indices, records))
    ):
        raise RecursiveBranchHold(f"HOLD: depth {depth} world8 exactly-once coverage drifted")
    return sorted(records, key=lambda item: int(item["action_index"]))


def _depth_zero_candidates(
    records: Sequence[Mapping[str, Any]], *, descriptions: Mapping[str, str],
    logits: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        x1 = int(record["x1_token_id"])
        evaluation = dict(record["evaluation"])
        owners = list(map(str, evaluation["matched_target_owner_ids"]))
        result.append({
            "action_index": DEPTH_ZERO_TOKENS.index(x1),
            "x1_token_id": x1,
            "generated_token_ids": list(map(int, evaluation["generated_token_ids"])),
            "generated_token_ids_sha256": evaluation["generated_token_ids_sha256"],
            "matched_owner_ids": sorted(owners),
            "matched_owner_count": len(owners),
            "matched_person_count": _person_count(owners, descriptions),
            "cumulative_intervention_count": 1,
            "x1_logit": float(dict(logits["values"])[str(x1)]),
            "admitted": True,
            "gate": dict(record["gate"]),
            "evaluation": evaluation,
        })
    return result


def _transcript_entry(
    selected: Mapping[str, Any], *, parent_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], depth: int,
) -> dict[str, Any]:
    x1 = int(selected["x1_token_id"])
    binding = _splice_branch_prefix(parent_tokens, x1)
    full = list(map(int, selected["generated_token_ids"]))
    if full[:len(binding["tokens"])] != binding["tokens"]:
        raise RecursiveBranchHold("HOLD: selected child does not consume its recorded forced prefix")
    suffix = full[len(binding["tokens"]):]
    completion = _forced_completion(parent_tokens, x1, suffix)
    if completion["generated_token_ids_sha256"] != selected["generated_token_ids_sha256"]:
        raise RecursiveBranchHold("HOLD: selected child route hash drifted")
    return {
        "depth": int(depth),
        "x1_token_id": x1,
        "x1_logit": float(selected["x1_logit"]),
        "parent_owner_ids": list(map(str, parent_owner_ids)),
        "parent_route_sha256": token_ids_sha256(parent_tokens),
        **completion,
        "matched_owner_ids": list(map(str, selected["matched_owner_ids"])),
        "matched_person_count": int(selected["matched_person_count"]),
        "matched_owner_count": int(selected["matched_owner_count"]),
        "gate_debt": dict(dict(selected["gate"]).get("debt", {})),
    }


def _verify_replay_record(
    recorded: Mapping[str, Any], replayed: Mapping[str, Any], *, previous_tokens: Sequence[int],
) -> None:
    x1 = int(recorded.get("x1_token_id", -1))
    binding = _splice_branch_prefix(previous_tokens, x1)
    exact_keys = (
        "forced_prefix_tokens", "forced_prefix_sha256", "x1_position",
        "suffix_token_ids", "suffix_token_ids_sha256", "generated_token_ids",
        "generated_token_ids_sha256", "matched_owner_ids", "matched_person_count",
        "matched_owner_count", "gate_debt",
    )
    if (
        list(map(int, recorded.get("forced_prefix_tokens", ()))) != binding["tokens"]
        or any(recorded.get(key) != replayed.get(key) for key in exact_keys)
    ):
        raise RecursiveBranchHold("HOLD: fresh replay differs from committed intervention")


def _evaluate_action(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    action_index: int, action: Mapping[str, Any], parent_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], target: Mapping[str, Any], raw_example: Any,
    descriptions: Mapping[str, str], logit_packet: Mapping[str, Any], depth: int,
) -> dict[str, Any]:
    x1 = int(action["x1_token_id"])
    prefix = _splice_branch_prefix(parent_tokens, x1)["tokens"]
    suffix = base.full_root._greedy_release(
        model=model, native_inputs=native_inputs, prefix=prefix,
        eos_token_id=base.EOS, pad_token_id=pad,
    )
    completion = _forced_completion(parent_tokens, x1, suffix)
    evaluation = manifold._match_evaluation(
        tokenizer=tokenizer, tokens=completion["generated_token_ids"], target=target,
        witness_tokens=completion["generated_token_ids"], raw_example=raw_example,
        parent_owners=parent_owner_ids, label=f"recursive-depth-{depth}-x1-{x1}",
    )
    return _candidate_record(
        action_index=action_index, action=action, parent_tokens=parent_tokens,
        parent_owner_ids=parent_owner_ids, suffix_tokens=suffix, evaluation=evaluation,
        descriptions=descriptions, logit_packet=logit_packet,
        cumulative_intervention_count=depth + 1,
    )


def _cold_replay(
    *, transcript: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(START_CHECKPOINT)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise RecursiveBranchHold("HOLD: fresh replay requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        model.eval()
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
            setup["requests"][:1]
        )
        names, parameters = ota._step_trainable_surface(model, r32_step=True)
        surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
        if (
            token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
            or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
            or surface.get("aggregate_sha256") != START_SURFACE_SHA256
            or base._frozen_surface(model, names) != FROZEN_SURFACE_SHA256
        ):
            raise RecursiveBranchHold("HOLD: fresh replay source/surface identity drifted")
        previous_tokens = list(map(int, contract["parent_route_tokens"]))
        previous_owners = list(map(str, contract["parent_owner_ids"]))
        replay_records: list[dict[str, Any]] = []
        for recorded in transcript:
            x1 = int(recorded["x1_token_id"])
            prefix = _splice_branch_prefix(previous_tokens, x1)["tokens"]
            suffix = base.full_root._greedy_release(
                model=model, native_inputs=native_inputs, prefix=prefix,
                eos_token_id=base.EOS, pad_token_id=int(tokenizer.pad_token_id),
            )
            completion = _forced_completion(previous_tokens, x1, suffix)
            evaluation = manifold._match_evaluation(
                tokenizer=tokenizer, tokens=completion["generated_token_ids"],
                target=contract["target"], witness_tokens=completion["generated_token_ids"],
                raw_example=setup["raw_example"], parent_owners=previous_owners,
                label=f"recursive-cold-replay-depth-{recorded['depth']}",
            )
            gate = _node_gate(evaluation, parent_owner_ids=previous_owners)
            owners = list(gate["matched_owner_ids"])
            replayed = {
                **completion,
                "matched_owner_ids": owners,
                "matched_person_count": _person_count(owners, contract["description_map"]),
                "matched_owner_count": len(owners),
                "gate_debt": dict(gate["debt"]),
            }
            _verify_replay_record(recorded, replayed, previous_tokens=previous_tokens)
            replay_records.append(replayed)
            previous_tokens = completion["generated_token_ids"]
            previous_owners = owners
        if transcript and token_ids_sha256(previous_tokens) != transcript[-1]["generated_token_ids_sha256"]:
            raise RecursiveBranchHold("HOLD: fresh replay final route identity drifted")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RecursiveBranchHold("HOLD: inference-only replay materialized gradients")
        return {
            "records": replay_records,
            "completion_count": len(replay_records),
            "final_generated_token_ids": previous_tokens,
            "final_generated_token_ids_sha256": token_ids_sha256(previous_tokens),
            "surface": surface,
            "runtime": opened.receipt.to_artifact_dict(),
        }


def _enforce_budget(counts: Mapping[str, int], *, cumulative_interventions: int) -> None:
    expected = {"warm_controlled_completions", "branch_logit_forwards", "replay_completions"}
    if (
        set(counts) != expected
        or not 1 <= cumulative_interventions <= MAX_INTERVENTIONS
        or int(counts.get("warm_controlled_completions", -1)) > MAX_WARM_COMPLETIONS
        or int(counts.get("branch_logit_forwards", -1)) > MAX_INTERVENTIONS
        or int(counts.get("replay_completions", -1)) != cumulative_interventions
        or int(counts.get("replay_completions", -1)) > MAX_REPLAY_COMPLETIONS
    ):
        raise RecursiveBranchHold("HOLD: recursive branch-search execution budget drifted")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RecursiveBranchHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise RecursiveBranchHold("HOLD: runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _gather_totals(local: Mapping[str, int]) -> tuple[list[dict[str, int]], dict[str, int]]:
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, {"rank": dist.get_rank(), **dict(local)})
    return list(gathered), {
        key: sum(int(item[key]) for item in gathered) for key in local
    }


def _require_world8(value: int) -> None:
    if value != WORLD_SIZE:
        raise RecursiveBranchHold("HOLD: recursive branch search requires torchrun --nproc_per_node=8")


def run(*, run_id: str) -> Path:
    _require_world8(int(os.environ.get("WORLD_SIZE", "0")))
    contract = _start_contract()  # Every rank fails immutable CPU bindings before CUDA/NCCL.
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    stage = "distributed_initialization"
    owns_output = False
    source_snapshot: Mapping[str, Any] | None = None
    transcript: list[dict[str, Any]] = []
    depth_records: list[dict[str, Any]] = []
    final_evaluation: Mapping[str, Any] | None = None
    status: str | None = None
    local_counts = {
        "warm_controlled_completions": 0,
        "branch_logit_forwards": 0,
        "replay_completions": 0,
    }
    try:
        dist.init_process_group(backend="nccl")
        _require_world8(dist.get_world_size())
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise RecursiveBranchHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        stage = "warm_recursive_search"
        setup = contract["setup"]
        target = contract["target"]
        descriptions = contract["description_map"]
        parent_tokens = list(map(int, contract["parent_route_tokens"]))
        parent_owners = list(map(str, contract["parent_owner_ids"]))
        initial_surface: Mapping[str, Any] | None = None
        warm_runtime: Mapping[str, Any] | None = None

        with base.open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise RecursiveBranchHold("HOLD: warm search requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1]
            )
            if (
                token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path)) != base.IMAGE_SHA256
            ):
                raise RecursiveBranchHold("HOLD: warm prompt/image identity drifted")
            names, parameters = ota._step_trainable_surface(model, r32_step=True)
            initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_before = base._frozen_surface(model, names)
            if (
                initial_surface.get("aggregate_sha256") != START_SURFACE_SHA256
                or frozen_before != FROZEN_SURFACE_SHA256
            ):
                raise RecursiveBranchHold("HOLD: warm selected-r32-step surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)

            depth_zero_actions = [{"x1_token_id": token} for token in DEPTH_ZERO_TOKENS]
            logits = base._rank0_call(lambda model=model, native_inputs=native_inputs,
                tokenizer=tokenizer, parent_tokens=parent_tokens: _logit_packet(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                node_tokens=parent_tokens, actions=depth_zero_actions, depth=0,
            ))
            if rank == 0:
                local_counts["branch_logit_forwards"] += 1
            ota._agree_hash(logits, label="recursive-depth-0-x1-logits")
            alternatives = _depth_zero_candidates(
                contract["depth_zero_records"], descriptions=descriptions, logits=logits,
            )
            selected = _select_best(alternatives)
            transcript.append(_transcript_entry(
                selected, parent_tokens=parent_tokens, parent_owner_ids=parent_owners, depth=0,
            ))
            parent_tokens = list(map(int, selected["generated_token_ids"]))
            parent_owners = list(map(str, selected["matched_owner_ids"]))
            final_evaluation = dict(selected["evaluation"])
            depth_records.append({
                "depth": 0,
                "missing_person_owner_ids": sorted(
                    item for item, value in descriptions.items()
                    if value == "person" and item not in set(contract["parent_owner_ids"])
                ),
                "actions": depth_zero_actions,
                "logits": logits,
                "alternatives": alternatives,
                "sealed_discovery_coverage": dict(
                    contract["discovery_receipt"]["discovery"]["coverage"]
                ),
                "selected_x1_token_id": int(selected["x1_token_id"]),
            })
            if _person_count(parent_owners, descriptions) != DEPTH_ZERO_PERSON_COUNT:
                raise RecursiveBranchHold("HOLD: depth-zero controlled node is not 33 persons")
            status = _terminal_status(
                matched_person_count=DEPTH_ZERO_PERSON_COUNT,
                cumulative_interventions=1,
                admitted_child_count=1,
            )

            for depth in range(1, MAX_INTERVENTIONS):
                if status is not None:
                    break
                actions = _missing_person_actions(
                    parent_owners, descriptions=descriptions, aliases=contract["aliases"],
                )
                missing = sorted(
                    item for item, value in descriptions.items()
                    if value == "person" and item not in set(parent_owners)
                )
                if not actions:
                    depth_records.append({
                        "depth": depth, "missing_person_owner_ids": missing,
                        "actions": [], "logits": None, "alternatives": [],
                        "selected_x1_token_id": None,
                    })
                    status = "controlled_search_exhausted"
                    break
                logits = base._rank0_call(lambda model=model, native_inputs=native_inputs,
                    tokenizer=tokenizer, parent_tokens=parent_tokens,
                    actions=actions, depth=depth: _logit_packet(
                    model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    node_tokens=parent_tokens, actions=actions, depth=depth,
                ))
                if rank == 0:
                    local_counts["branch_logit_forwards"] += 1
                ota._agree_hash(logits, label=f"recursive-depth-{depth}-x1-logits")
                local_records: list[dict[str, Any]] = []
                for action_index in range(rank, len(actions), WORLD_SIZE):
                    action = actions[action_index]
                    try:
                        local_records.append(_evaluate_action(
                            model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                            pad=int(tokenizer.pad_token_id), action_index=action_index,
                            action=action, parent_tokens=parent_tokens,
                            parent_owner_ids=parent_owners, target=target,
                            raw_example=setup["raw_example"], descriptions=descriptions,
                            logit_packet=logits, depth=depth,
                        ))
                    except BaseException as error:
                        local_records.append({
                            "rank": rank,
                            "action_index": action_index,
                            "x1_token_id": int(action["x1_token_id"]),
                            "evaluation_error": {
                                "type": type(error).__name__, "error": str(error),
                            },
                        })
                    local_records[-1]["rank"] = rank
                    local_counts["warm_controlled_completions"] += 1
                gathered: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered, {
                    "rank": rank, "records": local_records, "error": None,
                })
                alternatives = base._rank0_call(lambda: _verify_exactly_once(
                    gathered, actions=actions, depth=depth,
                ))
                ota._agree_hash(alternatives, label=f"recursive-depth-{depth}-alternatives")
                admitted = [item for item in alternatives if bool(item.get("admitted"))]
                selected = _select_best(admitted) if admitted else None
                depth_records.append({
                    "depth": depth,
                    "missing_person_owner_ids": missing,
                    "actions": actions,
                    "logits": logits,
                    "alternatives": alternatives,
                    "selected_x1_token_id": (
                        None if selected is None else int(selected["x1_token_id"])
                    ),
                })
                if selected is None:
                    status = "controlled_search_exhausted"
                    break
                transcript.append(_transcript_entry(
                    selected, parent_tokens=parent_tokens,
                    parent_owner_ids=parent_owners, depth=depth,
                ))
                parent_tokens = list(map(int, selected["generated_token_ids"]))
                parent_owners = list(map(str, selected["matched_owner_ids"]))
                final_evaluation = dict(selected["evaluation"])
                status = _terminal_status(
                    matched_person_count=int(selected["matched_person_count"]),
                    cumulative_interventions=len(transcript),
                    admitted_child_count=len(admitted),
                )

            if status is None:
                raise RecursiveBranchHold("HOLD: recursive search exited without terminal status")
            after_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if (
                after_surface != initial_surface
                or base._frozen_surface(model, names) != frozen_before
                or any(parameter.grad is not None for parameter in model.parameters())
            ):
                raise RecursiveBranchHold("HOLD: inference-only warm search mutated model state")
            warm_runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, parameters, names
        torch.cuda.empty_cache()
        dist.barrier()
        stage = "fresh_rank0_replay"
        replay = base._rank0_call(lambda: _cold_replay(transcript=transcript, contract=contract))
        if rank == 0:
            local_counts["replay_completions"] = len(transcript)
        ota._agree_hash(replay, label="recursive-cold-replay")

        local_resources = {
            "rank": rank,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(local_rank)),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(local_rank).total_memory),
        }
        gathered_resources: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_resources, local_resources)
        per_rank_counts, total_counts = _gather_totals(local_counts)
        _enforce_budget(total_counts, cumulative_interventions=len(transcript))
        elapsed = time.monotonic() - started
        if (
            elapsed > RESOURCE_BOUND["wall_time_seconds_max"]
            or max(int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources)
            > RESOURCE_BOUND["max_peak_cuda_reserved_bytes_per_rank"]
        ):
            raise RecursiveBranchHold("HOLD: recursive search resource bound exceeded")
        if status not in TERMINAL_STATUSES or final_evaluation is None:
            raise RecursiveBranchHold("HOLD: recursive terminal receipt/status drifted")
        final_gate = _node_gate(
            final_evaluation,
            parent_owner_ids=(
                contract["parent_owner_ids"] if len(transcript) == 1
                else transcript[-2]["matched_owner_ids"]
            ),
        )
        if not final_gate["passed"]:
            raise RecursiveBranchHold("HOLD: terminal route no longer passes its committed gate")
        if rank == 0:
            ledger = dict(final_evaluation["ledger"])
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": status,
                "run_id": run_id,
                "runner_source_snapshot": source_snapshot,
                "bindings": _binding_receipt(),
                "protocol": {
                    "world_size": WORLD_SIZE,
                    "search": "one committed child per depth; rejected owner sets are never unioned",
                    "action_source": "sealed OTA aliases for authoritative missing persons only",
                    "forced_row_open_tokens": list(ROW_OPEN),
                    "value": [
                        "matched_person_count", "matched_owner_count",
                        "fewer_cumulative_interventions", "current_policy_x1_logit",
                        "lower_x1_token_id",
                    ],
                    "gradient": None,
                    "optimizer": None,
                    "weight_update": False,
                    "checkpoint_save": False,
                },
                "depths": depth_records,
                "intervention_transcript": transcript,
                "final": {
                    "generated_token_ids": list(map(int, final_evaluation["generated_token_ids"])),
                    "generated_token_ids_sha256": final_evaluation["generated_token_ids_sha256"],
                    "matched_owner_ids": list(map(str, final_evaluation["matched_target_owner_ids"])),
                    "matched_owner_count": len(final_evaluation["matched_target_owner_ids"]),
                    "matched_person_count": _person_count(
                        final_evaluation["matched_target_owner_ids"], descriptions,
                    ),
                    "matched_tie_count": sum(
                        descriptions[item] == "tie"
                        for item in final_evaluation["matched_target_owner_ids"]
                    ),
                    "gate": final_gate,
                    "parsed_rows": list(dict(ledger.get("parse", {})).get("predictions", ())),
                    "ledger": ledger,
                },
                "fresh_replay": replay,
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
                "runtime": {"warm": warm_runtime, "fresh_rank0": replay["runtime"]},
                "wall_time_seconds": elapsed,
                "claim_boundary": (
                    "One frozen-r32 controlled-decoding trajectory on Image2299 only; not ordinary "
                    "greedy learning, transfer, completeness, or recovery of all eight ties."
                ),
            }
            base._atomic_json(output / "receipt.json", receipt)
            if (output / "receipt.json").stat().st_size > RESOURCE_BOUND["output_artifact_bytes_max"]:
                raise RecursiveBranchHold("HOLD: recursive receipt exceeded artifact bound")
        dist.barrier()
        return output
    except BaseException as error:
        if dist.is_initialized() and dist.get_rank() == 0 and owns_output:
            base._atomic_json(output / "receipt.json", {
                "schema_version": SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "status": "HOLD",
                "run_id": run_id,
                "hold_stage": stage,
                "stop_reason": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "runner_source_snapshot": source_snapshot,
                "model_receipt": str(MODEL_RECEIPT),
                "model_receipt_sha256": MODEL_RECEIPT_SHA256,
                "discovery_receipt": str(DISCOVERY_RECEIPT),
                "discovery_receipt_sha256": DISCOVERY_RECEIPT_SHA256,
                "intervention_transcript": transcript,
                "counts_rank0_partial": local_counts,
                "wall_time_seconds": time.monotonic() - started,
            })
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true", help="CPU-only frozen check")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id:
            raise SystemExit("--check-bindings is CPU-only and cannot be combined with --run-id")
        print(json.dumps(_binding_receipt(), indent=2, sort_keys=True))
        return
    if not args.run_id:
        raise SystemExit("GPU execution requires explicit --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
