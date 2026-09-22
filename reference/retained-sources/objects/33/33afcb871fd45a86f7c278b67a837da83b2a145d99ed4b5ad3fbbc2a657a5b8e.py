#!/usr/bin/env python3
"""Eight-way DoRA corridor search with pure coordinate target descent."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_certified_46_owner_path_overfit as base
from scripts.research import run_image2299_parallel_match_corridor as parallel
from src.inference.backend import open_backend_session, token_ids_sha256
from src.inference.hf_backend import HFBackendSession


SCHEMA_VERSION = "image2299.projected_owner_corridor.v17"
OUTPUT_ROOT = parallel.OUTPUT_ROOT / "projected-owner-corridor"
TRIGGER_RECEIPT = (
    OUTPUT_ROOT / "20260830T-tail-person-stabilizer-v16" / "receipt.json"
)
TRIGGER_SHA256 = "5d75628a378af217f01b4619595073d03570440a8057e768be02c3041a4f040a"
TRIGGER_SOURCE = TRIGGER_RECEIPT.parent / "runner_source.py"
TRIGGER_SOURCE_SHA256 = "06436436215b418472556655e02f86f431c669dd3ff8117b75e36dd20ce5a0bb"
TRIGGER_STATUS = "bounded_negative_no_match_level_promotion"
TRIGGER_STOP = "no_feasible_target_improving_observed_bad_token_guard_radius"
START_CHECKPOINT = (
    TRIGGER_RECEIPT.parent / "checkpoint-terminal-step-05"
)
TRIGGER_START_CHECKPOINT = (
    OUTPUT_ROOT / "20260830T-unsupported-person-corridor-v15" / "checkpoint-terminal-step-20"
)
NATURAL_TOKEN_SHA256 = "d7df6b7b9788d9bbbfe778a904962f44d74d562bc12f2e802353627091223732"
TRIGGER_INITIAL_NATURAL_SHA256 = "58d4a7732177fd8a1f2695cf985e8338a9cd6b543609f790702d40ad77722e12"
CLEAN_PARENT_TOKEN_SHA256 = base.PARENT_ROUTE_SHA256
TRIGGER_INITIAL_SURFACE_SHA256 = "160b041b6813fe815eaaa6c928f46af63f148db09a9fa318e5c19273b084bfb6"
START_SURFACE_SHA256 = "f951bece4f43990f98f513d9f9f324faa4919ced56402cc8480433947ade14a9"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
LEXICAL_TARGET_POSITION = 298
STABILIZER_POSITION = 298
STABILIZER_PERSON_TOKEN = 8987
STABILIZER_COMPETITOR_TOKENS = (48731, 34196)
TARGET_POSITION, TARGET_GOOD, TARGET_BAD = 301, 152399, 152590
TARGET_MARGIN = -8.25871467590332
TRIGGER_STABILIZER_MARGIN = -0.6558265686035156
TARGET_TEACHER_SHA256 = "2621f7d6654e303fa4add9cbb63429e5f0d9aff4172a9bf50829abef0b682228"
TRIGGER_GUARD_MARGINS = {
    "G0": 0.38808441162109375,
    "G1": 0.2008800506591797,
    "G2": 1.0918045043945312,
    "G3": 1.741231918334961,
    "G4": 0.369140625,
    "G5": 0.12581253051757812,
}
GUARD_SPECS = (
    {
        "guard_id": "G0", "row_index": 6, "owner": "gt:2299:44",
        "observed_owner": "gt:2299:18", "cascade_gained_owner_ids": (),
        "position": 55, "bad_token_id": 8987,
        "failed_row": (151646, 8987, 151647, 151648, 151867, 151935, 151966, 152190, 151649),
    },
    {
        "guard_id": "G1", "row_index": 4, "owner": "gt:2299:7",
        "observed_owner": "gt:2299:38", "cascade_gained_owner_ids": ("gt:2299:35",),
        "position": 40, "bad_token_id": 151804,
        "failed_row": (151646, 8987, 151647, 151648, 151804, 152305, 151944, 152669, 151649),
    },
    {
        "guard_id": "G2", "row_index": 21, "owner": "gt:2299:19",
        "observed_owner": "gt:2299:34", "cascade_gained_owner_ids": ("gt:2299:34",),
        "position": 194, "bad_token_id": 152041,
        "failed_row": (151646, 8987, 151647, 151648, 152305, 152041, 152400, 152392, 151649),
    },
    {
        "guard_id": "G3", "row_index": 4, "owner": "gt:2299:7",
        "observed_owner": "gt:2299:22", "cascade_gained_owner_ids": ("gt:2299:18",),
        "position": 41, "bad_token_id": 151867,
        "failed_row": (151646, 8987, 151647, 151648, 151801, 151867, 152669, 152669, 151649),
    },
    {
        "guard_id": "G4", "row_index": 23, "owner": "gt:2299:10",
        "observed_owner": "gt:2299:4", "cascade_gained_owner_ids": ("gt:2299:32",),
        "position": 208, "bad_token_id": 8987,
        "failed_row": (151646, 8987, 151647, 151648, 152379, 151762, 152459, 152059, 151649),
    },
    {
        "guard_id": "G5", "row_index": 15, "owner": "gt:2299:26",
        "observed_owner": "unmatched", "cascade_gained_owner_ids": (),
        "position": 141, "bad_token_id": 152178,
        "failed_row": (151646, 8987, 151647, 151648, 152103, 152078, 152178, 152275, 151649),
    },
)
ACTIVE_GUARD_IDS = ("G5",)
FEASIBILITY_GUARD_IDS = ("G0", "G1", "G2", "G3", "G4")
WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS = {
    "duplicate_person_count_iou95": 0, "malformed_count": 0,
    "matcher_ambiguity_neutral_count": 0, "matcher_unmatched_count": 1,
    "other_unknown_neutral_count": 0, "unknown_neutral_tie_count": 0,
    "unmatched_person_count": 1, "unsupported_person_count": 1,
}
TRIGGER_INITIAL_TAIL = (151646, 48731, 151647, 151648, 152596, 152097, 152617, 152138, 151649)
DUPLICATE_TAIL = (151646, 8987, 151647, 151648, 152590, 152041, 152657, 152576, 151649)
PERSON_NEAR_MISS_TAIL = (151646, 8987, 151647, 151648, 152596, 152041, 152657, 152576, 151649)
WORLD_SIZE = 8
MAX_ACCEPTED_STEPS = 30
PRIOR_ACCEPTED_STEPS = 153
COMBINED_STEP_CEILING = PRIOR_ACCEPTED_STEPS + MAX_ACCEPTED_STEPS
RADII = parallel.RADII
PROJECTION_TOLERANCE = 1e-7
DUPLICATE_ROUTE_SHA256 = NATURAL_TOKEN_SHA256
PERSON_NEAR_MISS_ROUTE_SHA256 = "3f5971c5b6293cd1aef5171bb5daf419fdd4c16c7096c609a4cd21bfe062c819"

ProjectedCorridorHold = base.Certified46Hold
_promotion_gate = parallel._promotion_gate
_strict_corridor_gate = parallel._corridor_gate
_clone_parameters = parallel._clone_parameters
_restore_parameters = parallel._restore_parameters
_legacy_state_identity = parallel._state_identity


def _trigger() -> dict[str, Any]:
    if (
        not TRIGGER_RECEIPT.is_file()
        or base._sha256(TRIGGER_RECEIPT) != TRIGGER_SHA256
        or not TRIGGER_SOURCE.is_file()
        or base._sha256(TRIGGER_SOURCE) != TRIGGER_SOURCE_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: projected-corridor trigger receipt identity drifted")
    receipt = json.loads(TRIGGER_RECEIPT.read_text(encoding="utf-8"))
    warm_record = dict(receipt.get("warm_final", {}))
    cold_record = dict(receipt.get("cold", {}))
    warm = dict(warm_record.get("packet", {}))
    cold = dict(cold_record.get("packet", {}))
    evaluation = dict(warm.get("evaluation", {}))
    causal = dict(warm.get("causal_ledger", {}))
    joint = dict(evaluation.get("joint_gate", {}))
    matcher = dict(joint.get("matcher", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    owners = list(map(str, receipt.get("parent_owner_ids", ())))
    attempts = list(receipt.get("attempts", ()))
    bindings = dict(receipt.get("bindings", {}))
    saved = dict(receipt.get("saved_checkpoint", {}))
    source = dict(receipt.get("runner_source_snapshot", {}))
    initial_surface = dict(receipt.get("initial_surface", {}))
    terminal_surface = dict(receipt.get("terminal_surface", {}))
    current_pair = dict(bindings.get("terminal_target_pair", {}))
    current_guards = list(bindings.get("terminal_guard_pairs", ()))
    expected_guards = _guard_bindings(tokens)
    expected_guard_gate = _observed_bad_token_guard_gate(tokens, tokens)
    tail = dict(causal.get("rows", ())[-1]) if len(causal.get("rows", ())) == 34 else {}
    guard_keys = (
        "guard_id", "row_index", "position", "good_token_id", "bad_token_id",
        "current_row", "real_prefix_sha256", "teacher_tokens_sha256",
    )

    def guard_identities(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value.get(key) for key in guard_keys} for value in values]

    if (
        receipt.get("schema_version") != "image2299.projected_owner_corridor.v16"
        or receipt.get("status") != TRIGGER_STATUS
        or receipt.get("stop_reason") != TRIGGER_STOP
        or int(receipt.get("accepted_step_count", -1)) != 5
        or source != {"path": str(TRIGGER_SOURCE), "sha256": TRIGGER_SOURCE_SHA256}
        or bindings.get("start_checkpoint") != str(TRIGGER_START_CHECKPOINT)
        or saved.get("checkpoint") != str(START_CHECKPOINT)
        or dict(saved.get("readback", {})).get("child_checkpoint") != str(START_CHECKPOINT)
        or dict(saved.get("readback", {})).get("parent_checkpoint") != str(TRIGGER_START_CHECKPOINT)
        or not START_CHECKPOINT.is_dir()
        or len(owners) != 33 or len(set(owners)) != 33
        or bindings.get("prompt_sha256") != base.PROMPT_TOKEN_SHA256
        or bindings.get("target_sha256") != base.TARGET_SHA256
        or bindings.get("authority_sha256") != base.AUTHORITY_SHA256
        or bindings.get("natural_token_ids_sha256") != TRIGGER_INITIAL_NATURAL_SHA256
        or current_pair != _target_binding(tokens)
        or guard_identities(current_guards) != guard_identities(expected_guards)
        or (
            current_pair.get("pair_position"), current_pair.get("good_token_id"),
            current_pair.get("bad_token_id"), current_pair.get("teacher_tokens_sha256"),
            current_pair.get("natural_token_ids_sha256"),
        ) != (TARGET_POSITION, TARGET_GOOD, TARGET_BAD, TARGET_TEACHER_SHA256, NATURAL_TOKEN_SHA256)
        or evaluation.get("generated_token_ids_sha256") != NATURAL_TOKEN_SHA256
        or token_ids_sha256(tokens) != NATURAL_TOKEN_SHA256
        or len(tokens) != 307 or list(tokens[-10:-1]) != list(DUPLICATE_TAIL)
        or tokens[STABILIZER_POSITION] != STABILIZER_PERSON_TOKEN
        or tokens[-1:] != [base.EOS]
        or evaluation.get("matched_target_owner_count") != 33
        or set(map(str, evaluation.get("matched_target_owner_ids", ()))) != set(owners)
        or not bool(evaluation.get("all_parent_owners_retained"))
        or matcher.get("strict_status_counts") != {"matched": 33, "unmatched": 1}
        or set(map(str, matcher.get("committed_owner_ids", ()))) != set(owners)
        or int(matcher.get("committed_owner_count", -1)) != 33
        or evaluation.get("hard_counter_count") != 3
        or joint.get("hard_raw_counters") != WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS
        or causal.get("first_event", {}).get("kind") != "duplicate"
        or list(map(int, causal.get("first_event", {}).get("bad_span_tokens", ()))) != list(DUPLICATE_TAIL)
        or tail.get("parse_state") != "valid" or tail.get("decision") != "duplicate"
        or tail.get("normalized_description") != "person"
        or tail.get("raw_person_duplicate") is not False
        or bool(warm.get("promotion_gate", {}).get("passed"))
        or not bool(warm.get("corridor_gate", {}).get("passed"))
        or _state_identity(warm) != _state_identity(cold)
        or receipt.get("warm_cold_identity") != _state_identity(cold)
        or warm_record.get("target_margin") != TARGET_MARGIN
        or cold_record.get("target_margin") != TARGET_MARGIN
        or warm_record.get("guard_margins") != TRIGGER_GUARD_MARGINS
        or cold_record.get("guard_margins") != TRIGGER_GUARD_MARGINS
        or dict(warm_record.get("stabilizer", {})).get("margin") != TRIGGER_STABILIZER_MARGIN
        or warm_record.get("stabilizer") != cold_record.get("stabilizer")
        or warm_record.get("observed_bad_token_guard_gate") != expected_guard_gate
        or cold_record.get("observed_bad_token_guard_gate") != expected_guard_gate
        or not expected_guard_gate["passed"]
        or cold_record.get("target_pair") != current_pair
        or guard_identities(cold_record.get("guard_pairs", ())) != guard_identities(current_guards)
        or initial_surface.get("aggregate_sha256") != TRIGGER_INITIAL_SURFACE_SHA256
        or terminal_surface.get("aggregate_sha256") != START_SURFACE_SHA256
        or cold_record.get("surface") != terminal_surface
        or any(
            surface.get("tensor_count") != 588 or surface.get("element_count") != 18_006_016
            for surface in (initial_surface, terminal_surface)
        )
        or receipt.get("frozen_surface_before") != FROZEN_SURFACE_SHA256
        or receipt.get("frozen_surface_after") != FROZEN_SURFACE_SHA256
        or cold_record.get("frozen_surface") != FROZEN_SURFACE_SHA256
        or len(attempts) != 6
    ):
        raise ProjectedCorridorHold("HOLD: v16 trigger state/checkpoint/surface/cold identity drifted")

    accepted_receipts = []
    previous_route = dict(attempts[0].get("frozen_state", {})).get("generated_token_ids_sha256")
    if previous_route != TRIGGER_INITIAL_NATURAL_SHA256:
        raise ProjectedCorridorHold("HOLD: v16 accepted progression start route drifted")
    for index, attempt in enumerate(attempts[:5], start=1):
        panel = list(attempt.get("candidate_panel", ()))
        by_rank = {int(item.get("rank", -1)): item for item in panel}
        selected = dict(attempt.get("selected_broadcast", {}))
        selected_rank = int(selected.get("source_rank", -1))
        selected_state = dict(selected.get("state", {}))
        if (
            int(attempt.get("attempt", -1)) != index
            or int(attempt.get("accepted_steps_before", -1)) != index - 1
            or attempt.get("decision") != "accept_largest_radius_in_priority_class"
            or dict(attempt.get("frozen_state", {})).get("generated_token_ids_sha256") != previous_route
            or len(panel) != WORLD_SIZE or set(by_rank) != set(range(WORLD_SIZE))
            or tuple(float(by_rank[rank].get("radius", math.nan)) for rank in range(WORLD_SIZE)) != RADII
            or selected_rank not in by_rank
            or selected.get("class") != "corridor"
            or float(selected.get("radius", math.nan)) != float(by_rank[selected_rank]["radius"])
            or selected_state != by_rank[selected_rank].get("state")
            or bool(selected_state.get("promotion_gate", {}).get("passed"))
            or not bool(selected_state.get("corridor_gate", {}).get("passed"))
        ):
            raise ProjectedCorridorHold("HOLD: v16 accepted progression drifted")
        previous_route = selected_state.get("generated_token_ids_sha256")
        accepted_receipts.append({
            "attempt": index, "source_rank": selected_rank, "radius": selected["radius"],
        })
    rejected = attempts[-1]
    rejected_panel = list(rejected.get("candidate_panel", ()))
    rejected_by_rank = {int(item.get("rank", -1)): item for item in rejected_panel}
    if (
        previous_route != NATURAL_TOKEN_SHA256
        or dict(attempts[4].get("selected_broadcast", {})).get("state", {}).get("generated_token_ids") != tokens
        or int(rejected.get("attempt", -1)) != 6
        or int(rejected.get("accepted_steps_before", -1)) != 5
        or rejected.get("decision") != "no_eligible_radius_restore_current"
        or rejected.get("selected_broadcast") is not None
        or dict(rejected.get("frozen_state", {})).get("generated_token_ids_sha256") != NATURAL_TOKEN_SHA256
        or len(rejected_panel) != WORLD_SIZE or set(rejected_by_rank) != set(range(WORLD_SIZE))
        or tuple(float(rejected_by_rank[rank].get("radius", math.nan)) for rank in range(WORLD_SIZE)) != RADII
        or dict(rejected.get("target_pair", {})).get("pair_position") != TARGET_POSITION
        or dict(rejected.get("target_pair", {})).get("good_token_id") != TARGET_GOOD
        or dict(rejected.get("target_pair", {})).get("bad_token_id") != TARGET_BAD
    ):
        raise ProjectedCorridorHold("HOLD: v16 terminal route/pair progression drifted")
    return {
        "path": str(TRIGGER_RECEIPT), "sha256": TRIGGER_SHA256,
        "status": TRIGGER_STATUS, "stop_reason": TRIGGER_STOP,
        "accepted_step_count": 5, "attempt_count": 6, "start_checkpoint": str(START_CHECKPOINT),
        "natural_token_ids": tokens, "natural_token_ids_sha256": NATURAL_TOKEN_SHA256,
        "parent_owner_ids": owners, "terminal_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "predecessor_bindings": bindings, "accepted_attempts": accepted_receipts,
        "terminal_route_sha256": NATURAL_TOKEN_SHA256,
        "terminal_target_pair": {key: value for key, value in current_pair.items() if key != "teacher_tokens"},
    }


def _working_corridor_gate(
    evaluation: Mapping[str, Any], causal_ledger: Mapping[str, Any],
    parent_owners: Sequence[str],
) -> dict[str, Any]:
    strict = _strict_corridor_gate(evaluation, causal_ledger, parent_owners)
    if strict["passed"]:
        return {**strict, "admission": "strict_parallel_corridor"}

    parent = set(map(str, parent_owners))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    matcher = dict(joint.get("matcher", {}))
    rows = list(causal_ledger.get("rows", ()))
    first = dict(causal_ledger.get("first_event") or {})
    backbone = rows[:33]
    backbone_owners = [str(row.get("owner", "")) for row in backbone]
    tail = dict(rows[-1]) if len(rows) == 34 else {}
    first_kind = str(first.get("kind", "missing"))
    tail_kind = str(tail.get("decision", "missing"))
    subtype = first_kind if first_kind == tail_kind else "inconsistent"
    debt = {
        "parent_identity": len(parent_owners) != 33 or len(parent) != 33,
        "token_shape": (
            len(tokens) != 34 * base.ROW_TOKENS + 1
            or tokens[-1:] != [base.EOS] or base.EOS in tokens[:-1]
            or len(tokens) >= base.NATURAL_MAX_TOKENS
        ),
        "parser_shape": (
            int(parser.get("valid_prediction_count", -1)) != 34
            or int(parser.get("dropped_prediction_count", -1)) != 0
            or int(parser.get("person_prediction_count", -1)) != 30
            or int(parser.get("tie_prediction_count", -1)) != 3
        ),
        "strict_global_match": (
            matcher.get("strict_status_counts") != {"matched": 33, "unmatched": 1}
            or int(matcher.get("optimum_cardinality", -1)) != 33
        ),
        "global_owner_set": (
            set(map(str, matcher.get("committed_owner_ids", ()))) != parent
            or int(matcher.get("committed_owner_count", -1)) != 33
            or set(map(str, evaluation.get("matched_target_owner_ids", ()))) != parent
            or int(evaluation.get("matched_target_owner_count", -1)) != 33
            or not bool(evaluation.get("all_parent_owners_retained"))
        ),
        "chronological_backbone": (
            len(backbone) != 33 or set(backbone_owners) != parent
            or len(backbone_owners) != len(set(backbone_owners))
            or any(
                row.get("parse_state") != "valid"
                or row.get("decision") != "accepted"
                or int(row.get("row_index", -1)) != index
                or int(row.get("token_start", -1)) != index * base.ROW_TOKENS
                for index, row in enumerate(backbone)
            )
            or list(map(str, causal_ledger.get("locked_owner_ids", ()))) != backbone_owners
        ),
        "unsupported_person_tail": (
            len(rows) != 34
            or tail.get("parse_state") != "valid"
            or tail_kind not in {"near_miss", "duplicate"}
            or tail.get("normalized_description") != "person"
            or int(tail.get("row_index", -1)) != 33
            or int(tail.get("token_start", -1)) != 297
            or first_kind != tail_kind
            or int(first.get("event_start", -1)) != 297
            or int(first.get("row_index", -1)) != 33
            or list(map(int, first.get("bad_span_tokens", ()))) != tokens[297:306]
            or tail.get("raw_person_duplicate") is not False
        ),
        "segmentation": not bool(causal_ledger.get("segmentation_trusted")),
        "causal_debt": int(causal_ledger.get("causal_hard_counter_count", -1)) != 1,
        "row_aligned_eos": not bool(joint.get("natural_row_aligned_eos")),
        "counter_debt": (
            int(evaluation.get("hard_counter_count", -1)) != 3
            or joint.get("hard_raw_counters") != WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS
        ),
    }
    active = {key: value for key, value in debt.items() if value}
    return {
        "passed": not active,
        "kind": f"transient_person_{subtype}",
        "admission": f"working_corridor_person_{subtype}_optimization_state_only",
        "promotion_eligible": False,
        "scientific_success": False,
        "debt": active,
    }


def _state_packet(**kwargs: Any) -> dict[str, Any]:
    packet = parallel._state_packet(**kwargs)
    strict = packet["corridor_gate"]
    packet["strict_corridor_gate"] = strict
    packet["corridor_gate"] = _working_corridor_gate(
        packet["evaluation"], packet["causal_ledger"], kwargs["parent_owners"],
    )
    return packet


def _state_identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    identity = parallel._state_identity(packet)
    identity["strict_corridor_gate"] = packet["strict_corridor_gate"]
    return identity

def _observed_bad_token_guard_gate(
    tokens: Sequence[int], incumbent: Sequence[int],
) -> dict[str, Any]:
    candidate = list(map(int, tokens))
    incumbent_tokens = list(map(int, incumbent))
    current_bindings = _guard_bindings(incumbent_tokens)
    bindings = []
    for spec, current in zip(GUARD_SPECS, current_bindings, strict=True):
        row = int(spec["row_index"])
        position = int(spec["position"])
        candidate_row = candidate[row * base.ROW_TOKENS:(row + 1) * base.ROW_TOKENS]
        candidate_token = candidate[position] if position < len(candidate) else None
        bindings.append({
            "guard_id": spec["guard_id"], "row_index": row, "position": position,
            "bad_token_id": int(spec["bad_token_id"]),
            "current_good_token_id": int(current["good_token_id"]),
            "current_row_sha256": current["current_row_sha256"],
            "current_prefix_sha256": current["real_prefix_sha256"],
            "candidate_token_id": candidate_token,
            "candidate_row_sha256": (
                token_ids_sha256(candidate_row) if len(candidate_row) == base.ROW_TOKENS else None
            ),
            "candidate_prefix_sha256": (
                token_ids_sha256(candidate[:position]) if position <= len(candidate) else None
            ),
            "candidate_avoids_bad_token": (
                candidate_token is not None and candidate_token != int(spec["bad_token_id"])
            ),
        })
    return {
        "guard_ids": [str(spec["guard_id"]) for spec in GUARD_SPECS],
        "bindings": bindings,
        "lcp_with_incumbent": base._exact_prefix(incumbent_tokens, candidate),
        "passed": all(binding["candidate_avoids_bad_token"] for binding in bindings),
    }


def _guard_bindings(tokens: Sequence[int]) -> list[dict[str, Any]]:
    natural = list(map(int, tokens))
    teacher_sha256 = token_ids_sha256(natural)
    guards = []
    for spec in GUARD_SPECS:
        start = int(spec["row_index"]) * base.ROW_TOKENS
        position = int(spec["position"])
        current_row = natural[start:start + base.ROW_TOKENS]
        bad_token = int(spec["bad_token_id"])
        if (
            len(current_row) != base.ROW_TOKENS
            or base.EOS in current_row
            or not start <= position < start + base.ROW_TOKENS
            or list(spec["failed_row"])[position - start] != bad_token
            or natural[position] == bad_token
        ):
            raise ProjectedCorridorHold(
                f"HOLD: {spec['guard_id']} current on-policy token equals observed bad token"
            )
        guards.append({
            "guard_id": spec["guard_id"], "row_index": spec["row_index"],
            "owner": spec["owner"], "incumbent_owner": spec["owner"],
            "observed_owner": spec["observed_owner"],
            "cascade_gained_owner_ids": list(spec["cascade_gained_owner_ids"]),
            "position": position, "good_token_id": natural[position],
            "bad_token_id": bad_token,
            "current_row": current_row, "current_row_sha256": token_ids_sha256(current_row),
            "failed_row": list(spec["failed_row"]),
            "earliest_divergence_position": position,
            "real_prefix_sha256": token_ids_sha256(natural[:position]),
            "teacher_tokens": natural, "teacher_tokens_sha256": teacher_sha256,
        })
    return guards


def _target_binding(tokens: Sequence[int]) -> dict[str, Any]:
    natural = list(map(int, tokens))
    token_sha256 = token_ids_sha256(natural)
    boundary = 33 * base.ROW_TOKENS
    clean_length, tail_length = boundary + 1, boundary + base.ROW_TOKENS + 1
    if (
        len(natural) not in {clean_length, tail_length}
        or natural[-1:] != [base.EOS]
        or base.EOS in natural[:-1]
    ):
        raise ProjectedCorridorHold("HOLD: target teacher violates clean298/tail307 row shape")
    pair = parallel._frozen_pair(natural, parallel.ALIAS_TOKENS)
    route_kind = "clean_parent33" if len(natural) == clean_length else "single_unassigned_tail"
    if (
        int(pair.get("boundary", -1)) != boundary
        or not (boundary <= int(pair.get("pair_position", -1)) < tail_length - 1)
        or pair["teacher_tokens"][:int(pair["pair_position"])]
        != natural[:int(pair["pair_position"])]
    ):
        raise ProjectedCorridorHold("HOLD: dynamic target teacher is not on the real corridor prefix")
    if route_kind == "clean_parent33" and (
        pair["pair_position"], pair["good_token_id"], pair["bad_token_id"]
    ) != (boundary, parallel.ALIAS_TOKENS[0], base.EOS):
        raise ProjectedCorridorHold("HOLD: clean corridor target pair identity drifted")
    pair["route_kind"] = route_kind
    pair["natural_token_ids_sha256"] = token_sha256
    return pair


def _stabilizer_binding(tokens: Sequence[int]) -> dict[str, Any]:
    natural = list(map(int, tokens))
    if (
        len(natural) != 34 * base.ROW_TOKENS + 1
        or natural[-1:] != [base.EOS]
        or natural[STABILIZER_POSITION] != STABILIZER_PERSON_TOKEN
    ):
        raise ProjectedCorridorHold("HOLD: tail-person stabilizer requires natural person at position 298")
    return {
        "position": STABILIZER_POSITION,
        "person_token_id": STABILIZER_PERSON_TOKEN,
        "competitor_token_ids": list(STABILIZER_COMPETITOR_TOKENS),
        "teacher_tokens": natural,
        "teacher_tokens_sha256": token_ids_sha256(natural),
    }


def _margin(logits: torch.Tensor, pair: Mapping[str, Any]) -> torch.Tensor:
    position = int(pair["position"] if "position" in pair else pair["pair_position"])
    good, bad = int(pair["good_token_id"]), int(pair["bad_token_id"])
    if logits.ndim != 2 or not (0 <= position < logits.shape[0]) or min(good, bad) < 0 or max(good, bad) >= logits.shape[1]:
        raise ProjectedCorridorHold("HOLD: malformed projected-corridor margin input")
    value = logits[position, good] - logits[position, bad]
    if not bool(torch.isfinite(value).item()):
        raise ProjectedCorridorHold("HOLD: non-finite projected-corridor margin")
    return value


def _stabilizer_margin(logits: torch.Tensor, binding: Mapping[str, Any]) -> torch.Tensor:
    position = int(binding.get("position", -1))
    person = int(binding.get("person_token_id", -1))
    competitors = tuple(map(int, binding.get("competitor_token_ids", ())))
    if (
        logits.ndim != 2 or not (0 <= position < logits.shape[0])
        or competitors != STABILIZER_COMPETITOR_TOKENS
        or min((person, *competitors)) < 0 or max((person, *competitors)) >= logits.shape[1]
    ):
        raise ProjectedCorridorHold("HOLD: malformed tail-person stabilizer margin input")
    value = logits[position, person] - torch.logsumexp(logits[position, list(competitors)], dim=0)
    if not bool(torch.isfinite(value).item()):
        raise ProjectedCorridorHold("HOLD: non-finite tail-person stabilizer margin")
    return value


def _fp64_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    if len(left) != len(right) or not left:
        raise ProjectedCorridorHold("HOLD: projection surface length drifted")
    total = 0.0
    for lhs, rhs in zip(left, right, strict=True):
        if lhs.shape != rhs.shape:
            raise ProjectedCorridorHold("HOLD: projection surface shape drifted")
        total += float(torch.sum(lhs.detach().double() * rhs.detach().double()).item())
    return total


def _project_direction(
    target_gradient: Sequence[torch.Tensor],
    g5_gradient: Sequence[torch.Tensor] | None,
    stabilizer_gradient: Sequence[torch.Tensor] | None = None,
    *, pair_position: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    target_norm_squared = _fp64_dot(target_gradient, target_gradient)
    if not math.isfinite(target_norm_squared) or target_norm_squared <= 0.0:
        raise ProjectedCorridorHold("HOLD: zero/non-finite target gradient")
    target_norm = math.sqrt(target_norm_squared)
    if pair_position < LEXICAL_TARGET_POSITION:
        raise ProjectedCorridorHold("HOLD: unsupported dynamic target position")
    if pair_position > LEXICAL_TARGET_POSITION:
        if g5_gradient is not None or stabilizer_gradient is not None:
            raise ProjectedCorridorHold("HOLD: coordinate target must use pure target-loss descent")
        direction = [(-target.detach() / target_norm).detach() for target in target_gradient]
        direction_norm = math.sqrt(max(0.0, _fp64_dot(direction, direction)))
        target_derivative = _fp64_dot(target_gradient, direction)
        if (
            not math.isfinite(direction_norm)
            or not math.isfinite(target_derivative)
            or abs(direction_norm - 1.0) > PROJECTION_TOLERANCE
            or target_derivative >= 0.0
        ):
            raise ProjectedCorridorHold("HOLD: coordinate direction lacks strict target descent")
        return direction, {
            "mode": "normalized_unit_target_loss_descent_only",
            "pair_position": pair_position,
            "fp64_norm_squared": {"target_loss": target_norm_squared},
            "fp64_pairwise_dots": {"target_loss": {"target_loss": target_norm_squared}},
            "fp64_cosine_matrix": {"target_loss": {"target_loss": 1.0}},
            "fp64_directional_derivatives": {"target_loss": target_derivative},
            "unit_component_norms": {"target_descent": 1.0},
            "pre_normalization_norm": 1.0, "direction_norm": direction_norm,
            "projection_tolerance": PROJECTION_TOLERANCE,
        }
    if stabilizer_gradient is not None:
        raise ProjectedCorridorHold("HOLD: lexical target must not use tail-person stabilizer")
    if g5_gradient is None:
        raise ProjectedCorridorHold("HOLD: lexical target requires G5 gradient")
    auxiliary_gradient = g5_gradient
    auxiliary_id = "G5_margin"
    auxiliary_norm_squared = _fp64_dot(auxiliary_gradient, auxiliary_gradient)
    target_auxiliary_dot = _fp64_dot(target_gradient, auxiliary_gradient)
    if (
        not math.isfinite(auxiliary_norm_squared) or auxiliary_norm_squared <= 0.0
        or not math.isfinite(target_auxiliary_dot)
    ):
        raise ProjectedCorridorHold(f"HOLD: zero/non-finite {auxiliary_id} gradient")
    auxiliary_norm = math.sqrt(auxiliary_norm_squared)
    raw_direction = [
        (-target.detach() / target_norm + auxiliary.detach() / auxiliary_norm).detach()
        for target, auxiliary in zip(target_gradient, auxiliary_gradient, strict=True)
    ]
    raw_norm_squared = _fp64_dot(raw_direction, raw_direction)
    if not math.isfinite(raw_norm_squared) or raw_norm_squared <= 0.0:
        raise ProjectedCorridorHold(f"HOLD: target-descent and {auxiliary_id}-ascent directions cancel")
    raw_norm = math.sqrt(raw_norm_squared)
    direction = [(value / raw_norm).detach() for value in raw_direction]
    direction_norm = math.sqrt(max(0.0, _fp64_dot(direction, direction)))
    target_derivative = _fp64_dot(target_gradient, direction)
    auxiliary_derivative = _fp64_dot(auxiliary_gradient, direction)
    cosine = target_auxiliary_dot / (target_norm * auxiliary_norm)
    if (
        not math.isfinite(direction_norm)
        or not math.isfinite(target_derivative)
        or not math.isfinite(auxiliary_derivative)
        or abs(direction_norm - 1.0) > PROJECTION_TOLERANCE
        or target_derivative >= 0.0
        or auxiliary_derivative <= 0.0
    ):
        raise ProjectedCorridorHold(
            f"HOLD: direction lacks strict target descent/{auxiliary_id} ascent"
        )
    return direction, {
        "mode": "normalized_unit_target_descent_plus_unit_G5_margin_ascent",
        "pair_position": pair_position,
        "fp64_norm_squared": {
            "target_loss": target_norm_squared, auxiliary_id: auxiliary_norm_squared,
        },
        "fp64_pairwise_dots": {
            "target_loss": {"target_loss": target_norm_squared, auxiliary_id: target_auxiliary_dot},
            auxiliary_id: {"target_loss": target_auxiliary_dot, auxiliary_id: auxiliary_norm_squared},
        },
        "fp64_cosine_matrix": {
            "target_loss": {"target_loss": 1.0, auxiliary_id: cosine},
            auxiliary_id: {"target_loss": cosine, auxiliary_id: 1.0},
        },
        "fp64_directional_derivatives": {
            "target_loss": target_derivative, auxiliary_id: auxiliary_derivative,
        },
        "unit_component_norms": {"target_descent": 1.0, f"{auxiliary_id}_ascent": 1.0},
        "pre_normalization_norm": raw_norm, "direction_norm": direction_norm,
        "projection_tolerance": PROJECTION_TOLERANCE,
    }


def _direction_snapshot(names: Sequence[str], direction: Sequence[torch.Tensor]) -> dict[str, Any]:
    tensors = [
        {"name": name, "sha256": base.full_root._tensor_sha256(value), "numel": value.numel()}
        for name, value in zip(names, direction, strict=True)
    ]
    return {
        "tensor_count": len(tensors), "element_count": sum(int(item["numel"]) for item in tensors),
        "aggregate_sha256": base._hash({"tensors": tensors}),
    }


def _apply_radius(
    parameters: Sequence[torch.nn.Parameter], clones: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor], *, radius: float,
) -> None:
    if radius not in RADII or len(parameters) != len(clones) or len(parameters) != len(direction):
        raise ProjectedCorridorHold("HOLD: malformed projected radius application")
    with torch.no_grad():
        for parameter, clone, delta in zip(parameters, clones, direction, strict=True):
            if parameter.shape != clone.shape or parameter.shape != delta.shape:
                raise ProjectedCorridorHold("HOLD: projected radius surface drifted")
            parameter.copy_(clone + base.LEARNING_RATE * radius * delta)


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]], *, base_target_margin: float,
    base_guard_margins: Mapping[str, float], base_target_position: int,
    base_stabilizer_margin: float | None,
) -> Mapping[str, Any] | None:
    by_rank = {int(item["rank"]): item for item in candidates}
    expected_guards = {str(spec["guard_id"]) for spec in GUARD_SPECS}
    if (
        len(candidates) != WORLD_SIZE
        or set(by_rank) != set(range(WORLD_SIZE))
        or tuple(float(by_rank[index]["radius"]) for index in range(WORLD_SIZE)) != RADII
        or set(base_guard_margins) != expected_guards
        or any(
            not math.isfinite(float(base_guard_margins[guard_id]))
            or float(base_guard_margins[guard_id]) <= 0.0
            for guard_id in expected_guards
        )
    ):
        raise ProjectedCorridorHold("HOLD: projected candidate radius panel is incomplete")
    promotions = [item for item in candidates if bool(item.get("promotion"))]
    if promotions:
        return max(promotions, key=lambda item: float(item["radius"]))
    eligible = [
        item for item in candidates
        if bool(item.get("corridor"))
        and bool(dict(item.get("observed_bad_token_guard_gate", {})).get("passed"))
        and math.isfinite(float(item.get("target_margin", math.nan)))
        and float(item["target_margin"]) > base_target_margin
        and set(item.get("frozen_guard_margins", {})) == expected_guards
        and (
            base_target_position == LEXICAL_TARGET_POSITION
            or (
                base_target_position > LEXICAL_TARGET_POSITION
                and len(item.get("state", {}).get("generated_token_ids", ())) > STABILIZER_POSITION
                and int(item["state"]["generated_token_ids"][STABILIZER_POSITION])
                == STABILIZER_PERSON_TOKEN
            )
        )
        and (
            base_target_position != LEXICAL_TARGET_POSITION
            or (
                math.isfinite(float(item["frozen_guard_margins"].get("G5", math.nan)))
                and float(item["frozen_guard_margins"]["G5"])
                > float(base_guard_margins["G5"])
            )
        )
        and set(item.get("guard_margins", {})) == expected_guards
        and all(
            math.isfinite(float(item["guard_margins"][guard_id]))
            and float(item["guard_margins"][guard_id]) > 0.0
            for guard_id in expected_guards
        )
    ]
    return max(eligible, key=lambda item: float(item["radius"])) if eligible else None


def _require_world_size(value: int) -> None:
    if value != WORLD_SIZE:
        raise ProjectedCorridorHold("HOLD: requires torchrun --nproc_per_node=8")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ProjectedCorridorHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise ProjectedCorridorHold("HOLD: runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _candidate_state(packet: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(packet["evaluation"])
    causal = dict(packet["causal_ledger"])
    return {
        "generated_token_ids": evaluation["generated_token_ids"],
        "generated_token_ids_sha256": evaluation["generated_token_ids_sha256"],
        "evaluator": {
            key: evaluation[key] for key in (
                "exact_target_prefix", "matched_target_owner_count", "matched_target_owner_ids",
                "all_parent_owners_retained", "hard_counter_count", "joint_gate",
            )
        },
        "causal": {
            key: causal.get(key) for key in (
                "segmentation_trusted", "locked_owner_ids", "final_missing_owner_ids",
                "first_event", "causal_hard_counter_count", "token_ids_sha256",
            )
        } | {
            "raw_person_duplicate": (
                causal.get("rows", ())[-1].get("raw_person_duplicate")
                if causal.get("rows") else None
            ),
        },
        "promotion_gate": packet["promotion_gate"],
        "strict_corridor_gate": packet["strict_corridor_gate"],
        "corridor_gate": packet["corridor_gate"],
        "optimization_state": (
            packet["corridor_gate"].get("kind")
            if packet["corridor_gate"].get("passed")
            and str(packet["corridor_gate"].get("kind", "")).startswith("transient_person_")
            else None
        ),
    }


def _teacher_margins(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    target_pair: Mapping[str, Any], guard_pairs: Sequence[Mapping[str, Any]],
    stabilizer_binding: Mapping[str, Any] | None,
) -> tuple[float, dict[str, float], float | None]:
    expected_guard_ids = {str(spec["guard_id"]) for spec in GUARD_SPECS}
    if (
        len(guard_pairs) != len(GUARD_SPECS)
        or {str(pair.get("guard_id")) for pair in guard_pairs} != expected_guard_ids
        or len({pair.get("teacher_tokens_sha256") for pair in guard_pairs}) != 1
    ):
        raise ProjectedCorridorHold("HOLD: exact six-guard teacher set drifted")
    with torch.inference_mode():
        target_logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs, route_tokens=target_pair["teacher_tokens"],
            pad_token_id=pad,
        )
        target_margin = float(_margin(target_logits, target_pair).item())
        guard_logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs, route_tokens=guard_pairs[0]["teacher_tokens"],
            pad_token_id=pad,
        )
        guard_margins = {
            str(pair["guard_id"]): float(_margin(guard_logits, pair).item()) for pair in guard_pairs
        }
        stabilizer_margin = (
            float(_stabilizer_margin(guard_logits, stabilizer_binding).item())
            if stabilizer_binding is not None else None
        )
    return target_margin, guard_margins, stabilizer_margin


def _terminal_stabilizer_evidence(
    *, stabilizer_binding: Mapping[str, Any], margin: float,
) -> dict[str, Any]:
    if not math.isfinite(margin):
        raise ProjectedCorridorHold("HOLD: non-finite terminal stabilizer diagnostic")
    return {
        "teacher": {key: value for key, value in stabilizer_binding.items() if key != "teacher_tokens"},
        "margin": margin, "role": "diagnostic_only_not_optimized_or_selected",
    }


def _cold_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(checkpoint)
    with open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise ProjectedCorridorHold("HOLD: cold evaluation requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
            raise ProjectedCorridorHold("HOLD: cold prompt identity drifted")
        names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
        surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
        frozen = base._frozen_surface(model, names)
        packet = _state_packet(
            model=model, tokenizer=tokenizer, native_inputs=native_inputs,
            pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
            parent_owners=parent_owners, alias_tokens=parallel.ALIAS_TOKENS, label=label,
        )
        cold_tokens = packet["evaluation"]["generated_token_ids"]
        if packet["promotion_gate"]["passed"]:
            target_pair, guard_pairs, guard_gate = None, [], None
            target_margin, guard_margins, stabilizer = None, {}, None
        else:
            target_pair = _target_binding(cold_tokens)
            guard_pairs = _guard_bindings(cold_tokens)
            stabilizer_binding = _stabilizer_binding(cold_tokens)
            guard_gate = _observed_bad_token_guard_gate(cold_tokens, cold_tokens)
            target_margin, guard_margins, stabilizer_margin = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=target_pair, guard_pairs=guard_pairs,
                stabilizer_binding=stabilizer_binding,
            )
            stabilizer = _terminal_stabilizer_evidence(
                stabilizer_binding=stabilizer_binding, margin=stabilizer_margin,
            )
            if stabilizer["margin"] != stabilizer_margin:
                raise ProjectedCorridorHold("HOLD: cold stabilizer margin evidence drifted")
        return {
            "packet": packet, "target_margin": target_margin, "guard_margins": guard_margins,
            "stabilizer": stabilizer,
            "target_pair": target_pair, "guard_pairs": guard_pairs,
            "observed_bad_token_guard_gate": guard_gate,
            "surface": surface, "frozen_surface": frozen,
            "runtime": opened.receipt.to_artifact_dict(),
        }


def run(*, run_id: str) -> Path:
    _require_world_size(int(os.environ.get("WORLD_SIZE", "0")))
    run_id = base._safe_run_id(run_id)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    output = OUTPUT_ROOT / run_id
    owns_output = False
    try:
        if local_rank < 0:
            raise ProjectedCorridorHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        trigger = base._rank0_call(_trigger)
        _admission, _parent_setup, target = base._bindings()
        setup = base._setup_for_checkpoint(START_CHECKPOINT)

        def admit_before_model_load() -> dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(
                setup["frontend"].launch.model_path, trust_remote_code=True,
            )
            try:
                return parallel._admit_alias(tokenizer=tokenizer, raw_example=setup["raw_example"])
            finally:
                del tokenizer

        alias = base._rank0_call(admit_before_model_load)
        initial_target_pair = _target_binding(trigger["natural_token_ids"])
        initial_guard_pairs = _guard_bindings(trigger["natural_token_ids"])
        attempts: list[dict[str, Any]] = []
        accepted_steps = 0
        final_warm: Mapping[str, Any] | None = None
        saved: Mapping[str, Any] | None = None
        stop_reason = "max_30_accepted_updates_without_warm_promotion"
        runtime: Mapping[str, Any] | None = None
        initial_surface: Mapping[str, Any] | None = None
        terminal_surface: Mapping[str, Any] | None = None
        frozen_before: str | None = None
        frozen_after: str | None = None
        warm_stabilizer: Mapping[str, Any] | None = None
        parent_owners: tuple[str, ...] = ()

        with open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise ProjectedCorridorHold("HOLD: requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
                raise ProjectedCorridorHold("HOLD: live prompt identity drifted")
            parent_owners = tuple(map(str, trigger["parent_owner_ids"]))
            base._assert_missing_membership(parent_owners)
            names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
            if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                raise ProjectedCorridorHold("HOLD: exact 588/18006016 FP32 DoRA surface drifted")
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if initial_surface["aggregate_sha256"] != START_SURFACE_SHA256:
                raise ProjectedCorridorHold("HOLD: cold-loaded predecessor DoRA surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)
            frozen_before = base._frozen_surface(model, names)
            if frozen_before != FROZEN_SURFACE_SHA256:
                raise ProjectedCorridorHold("HOLD: cold-loaded predecessor frozen surface drifted")

            baseline = base._rank0_call(lambda model=model, tokenizer=tokenizer, native_inputs=native_inputs: _state_packet(
                model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                parent_owners=parent_owners, alias_tokens=alias["token_ids"], label="projected-baseline",
            ))
            if (
                baseline["evaluation"]["generated_token_ids_sha256"] != NATURAL_TOKEN_SHA256
                or baseline["evaluation"]["generated_token_ids"] != trigger["natural_token_ids"]
                or baseline["promotion_gate"]["passed"]
                or not baseline["corridor_gate"]["passed"]
            ):
                raise ProjectedCorridorHold("HOLD: cold baseline does not reproduce the Parent33 tail")
            initial_stabilizer_binding = _stabilizer_binding(trigger["natural_token_ids"])
            baseline_target_margin, baseline_guard_margins, baseline_stabilizer_margin = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=initial_target_pair, guard_pairs=initial_guard_pairs,
                stabilizer_binding=initial_stabilizer_binding,
            )
            if (
                baseline_target_margin != TARGET_MARGIN
                or baseline_guard_margins != TRIGGER_GUARD_MARGINS
                or set(baseline_guard_margins) != {str(spec["guard_id"]) for spec in GUARD_SPECS}
                or not all(math.isfinite(value) for value in baseline_guard_margins.values())
                or not all(value > 0.0 for value in baseline_guard_margins.values())
                or baseline_stabilizer_margin is None
                or not math.isfinite(baseline_stabilizer_margin)
            ):
                raise ProjectedCorridorHold("HOLD: cold baseline teacher margins drifted")

            for attempt in range(1, MAX_ACCEPTED_STEPS + 1):
                current = baseline if attempt == 1 else base._rank0_call(lambda model=model, tokenizer=tokenizer, native_inputs=native_inputs: _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=alias["token_ids"],
                    label=f"projected-current-{attempt:02d}",
                ))
                if current["promotion_gate"]["passed"]:
                    final_warm = current
                    stop_reason = "current_match_level_promotion"
                    break
                if not current["corridor_gate"]["passed"]:
                    raise ProjectedCorridorHold("HOLD: current accepted policy left the match corridor")
                current_tokens = current["evaluation"]["generated_token_ids"]
                target_pair = _target_binding(current_tokens)
                guard_pairs = _guard_bindings(current_tokens)
                stabilizer_binding = (
                    None if int(target_pair["pair_position"]) == LEXICAL_TARGET_POSITION
                    else _stabilizer_binding(current_tokens)
                )
                current_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]

                target_logits = base.full_root._teacher_forced_route_logits(
                    model=model, native_inputs=native_inputs, route_tokens=target_pair["teacher_tokens"],
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                target_margin_tensor = _margin(target_logits, target_pair)
                target_loss = torch.nn.functional.softplus(-target_margin_tensor)
                target_gradients = torch.autograd.grad(target_loss, parameters, allow_unused=True)
                del target_logits, target_loss
                guard_logits = base.full_root._teacher_forced_route_logits(
                    model=model, native_inputs=native_inputs, route_tokens=guard_pairs[0]["teacher_tokens"],
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                guard_margin_tensors = {
                    str(pair["guard_id"]): _margin(guard_logits, pair) for pair in guard_pairs
                }
                stabilizer_margin_tensor = (
                    _stabilizer_margin(guard_logits, stabilizer_binding)
                    if stabilizer_binding is not None else None
                )
                del guard_logits
                g5_gradients = (
                    torch.autograd.grad(
                        guard_margin_tensors[ACTIVE_GUARD_IDS[0]], parameters, allow_unused=True,
                    )
                    if int(target_pair["pair_position"]) == LEXICAL_TARGET_POSITION else None
                )
                reduced_gradients: dict[str, list[torch.Tensor]] = {}
                gradient_sets = [("target_loss", target_gradients)]
                if g5_gradients is not None:
                    gradient_sets.append(("G5_margin", g5_gradients))
                for gradient_id, gradients in gradient_sets:
                    reduced: list[torch.Tensor] = []
                    for value in gradients:
                        if value is None or not bool(torch.isfinite(value).all().item()):
                            raise ProjectedCorridorHold(
                                f"HOLD: disconnected/non-finite {gradient_id} gradient"
                            )
                        value = value.detach()
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        reduced.append(value / WORLD_SIZE)
                    reduced_gradients[gradient_id] = reduced
                direction, projection = _project_direction(
                    reduced_gradients["target_loss"], reduced_gradients.get("G5_margin"),
                    pair_position=int(target_pair["pair_position"]),
                )
                direction_snapshot = _direction_snapshot(names, direction)
                projection["direction"] = direction_snapshot
                projection["distributed_gradient_reduction"] = "all_reduce_mean_on_all_eight_ranks"
                projection_identities: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(projection_identities, base._hash(projection))
                if len(set(projection_identities)) != 1:
                    raise ProjectedCorridorHold("HOLD: rank projection disagreement")
                base_target_margin = float(target_margin_tensor.detach().item())
                base_guard_margins = {
                    guard_id: float(value.detach().item())
                    for guard_id, value in guard_margin_tensors.items()
                }
                base_stabilizer_margin = (
                    float(stabilizer_margin_tensor.detach().item())
                    if stabilizer_margin_tensor is not None else None
                )
                clones = _clone_parameters(parameters)

                radius = RADII[rank]
                _apply_radius(parameters, clones, direction, radius=radius)
                base.full_root._assert_full_root_sentinels(model, sentinels)
                candidate_target_margin, candidate_frozen_guard_margins, candidate_frozen_stabilizer_margin = _teacher_margins(
                    model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target_pair=target_pair, guard_pairs=guard_pairs,
                    stabilizer_binding=stabilizer_binding,
                )
                candidate_packet = _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=alias["token_ids"],
                    label=f"projected-candidate-{attempt:02d}-rank-{rank}",
                )
                candidate_tokens = candidate_packet["evaluation"]["generated_token_ids"]
                candidate_guard_gate = _observed_bad_token_guard_gate(
                    candidate_tokens, current_tokens,
                )
                candidate_promoted = bool(candidate_packet["promotion_gate"]["passed"])
                candidate_corridor = bool(candidate_packet["corridor_gate"]["passed"])
                candidate_on_policy_target_margin: float | None = None
                candidate_guard_margins: dict[str, float] = {}
                candidate_on_policy_stabilizer_margin: float | None = None
                candidate_target_pair_receipt: dict[str, Any] | None = None
                candidate_guard_pairs_receipt: list[dict[str, Any]] = []
                if not candidate_promoted and candidate_corridor and candidate_guard_gate["passed"]:
                    candidate_target_pair = _target_binding(candidate_tokens)
                    candidate_guard_pairs = _guard_bindings(candidate_tokens)
                    candidate_stabilizer_binding = (
                        _stabilizer_binding(candidate_tokens)
                        if len(candidate_tokens) > STABILIZER_POSITION
                        and int(candidate_tokens[STABILIZER_POSITION]) == STABILIZER_PERSON_TOKEN
                        else None
                    )
                    (
                        candidate_on_policy_target_margin,
                        candidate_guard_margins,
                        candidate_on_policy_stabilizer_margin,
                    ) = _teacher_margins(
                        model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                        target_pair=candidate_target_pair, guard_pairs=candidate_guard_pairs,
                        stabilizer_binding=candidate_stabilizer_binding,
                    )
                    candidate_target_pair_receipt = {
                        key: value for key, value in candidate_target_pair.items()
                        if key != "teacher_tokens"
                    }
                    candidate_guard_pairs_receipt = [
                        {key: value for key, value in pair.items() if key != "teacher_tokens"}
                        for pair in candidate_guard_pairs
                    ]
                candidate_summary = {
                    "rank": rank, "radius": radius,
                    "base_target_margin": base_target_margin, "target_margin": candidate_target_margin,
                    "frozen_guard_margins": candidate_frozen_guard_margins,
                    "base_guard_margins": base_guard_margins, "guard_margins": candidate_guard_margins,
                    "base_stabilizer_margin": base_stabilizer_margin,
                    "frozen_stabilizer_margin": candidate_frozen_stabilizer_margin,
                    "on_policy_stabilizer_margin": candidate_on_policy_stabilizer_margin,
                    "on_policy_target_margin": candidate_on_policy_target_margin,
                    "on_policy_target_pair": candidate_target_pair_receipt,
                    "on_policy_guard_pairs": candidate_guard_pairs_receipt,
                    "target_margin_improved": candidate_target_margin > base_target_margin,
                    "active_guard_improved": (
                        math.isfinite(float(candidate_frozen_guard_margins.get("G5", math.nan)))
                        and float(candidate_frozen_guard_margins["G5"])
                        > float(base_guard_margins["G5"])
                    ),
                    "guards_improved": {
                        guard_id: (
                            math.isfinite(float(candidate_guard_margins.get(guard_id, math.nan)))
                            and float(candidate_guard_margins[guard_id]) > base_guard_margins[guard_id]
                        )
                        for guard_id in base_guard_margins
                    },
                    "candidate_lcp": candidate_guard_gate["lcp_with_incumbent"],
                    "observed_bad_token_guard_gate": candidate_guard_gate,
                    "promotion": candidate_promoted,
                    "corridor": candidate_corridor,
                    "state": _candidate_state(candidate_packet),
                }
                candidates: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(candidates, candidate_summary)
                selected = _select_candidate(
                    candidates, base_target_margin=base_target_margin,
                    base_guard_margins=base_guard_margins,
                    base_target_position=int(target_pair["pair_position"]),
                    base_stabilizer_margin=base_stabilizer_margin,
                )
                record: dict[str, Any] = {
                    "attempt": attempt, "accepted_steps_before": accepted_steps,
                    "frozen_state": _candidate_state(current), "frozen_surface": current_surface,
                    "target_pair": {key: value for key, value in target_pair.items() if key != "teacher_tokens"},
                    "guard_pairs": [
                        {key: value for key, value in pair.items() if key != "teacher_tokens"}
                        for pair in guard_pairs
                    ],
                    "current_teacher_token_ids_sha256": target_pair["natural_token_ids_sha256"],
                    "target_margin": base_target_margin, "guard_margins": base_guard_margins,
                    "stabilizer": (
                        None if stabilizer_binding is None else {
                            "teacher": {
                                key: value for key, value in stabilizer_binding.items()
                                if key != "teacher_tokens"
                            },
                            "margin": base_stabilizer_margin,
                            "role": "diagnostic_only_not_optimized_or_selected",
                        }
                    ),
                    "projection": projection, "candidate_panel": candidates, "selected_broadcast": None,
                }
                if selected is None:
                    _restore_parameters(parameters, clones)
                    restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                    if restored != current_surface:
                        raise ProjectedCorridorHold("HOLD: exact clone restore failed")
                    base._surface_agreement(restored, dist.group.WORLD)
                    record["decision"] = "no_eligible_radius_restore_current"
                    attempts.append(record)
                    final_warm = current
                    stop_reason = "no_feasible_target_improving_observed_bad_token_guard_radius"
                    break

                selected_rank = int(selected["rank"])
                for parameter in parameters:
                    dist.broadcast(parameter.data, src=selected_rank)
                selected_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                base._surface_agreement(selected_surface, dist.group.WORLD)
                base.full_root._assert_full_root_sentinels(model, sentinels)
                selected_packet_box: list[Any] = [candidate_packet if rank == selected_rank else None]
                dist.broadcast_object_list(selected_packet_box, src=selected_rank)
                selected_packet = selected_packet_box[0]
                record["selected_broadcast"] = {
                    "source_rank": selected_rank, "radius": float(selected["radius"]),
                    "class": "promotion" if selected["promotion"] else "corridor",
                    "target_margin": float(selected["target_margin"]),
                    "guard_margins": dict(selected["guard_margins"]),
                    "stabilizer_margin": selected.get("frozen_stabilizer_margin"),
                    "surface": selected_surface, "state": _candidate_state(selected_packet),
                }
                record["decision"] = "accept_largest_radius_in_priority_class"
                attempts.append(record)
                accepted_steps += 1
                if selected_packet["promotion_gate"]["passed"]:
                    final_warm = selected_packet
                    stop_reason = "first_warm_match_level_promotion"
                    break

            if final_warm is None:
                final_warm = base._rank0_call(lambda model=model, tokenizer=tokenizer, native_inputs=native_inputs: _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=alias["token_ids"],
                    label="projected-final-step-30",
                ))
            promoted = bool(final_warm["promotion_gate"]["passed"])
            if not promoted and not final_warm["corridor_gate"]["passed"]:
                raise ProjectedCorridorHold("HOLD: terminal policy is neither promotion nor corridor-feasible")
            if promoted:
                terminal_target_pair, terminal_guard_pairs, terminal_guard_gate = None, [], None
                warm_target_margin, warm_guard_margins = None, {}
                warm_stabilizer = None
            else:
                terminal_guard_gate = _observed_bad_token_guard_gate(
                    final_warm["evaluation"]["generated_token_ids"],
                    final_warm["evaluation"]["generated_token_ids"],
                )
                terminal_target_pair = _target_binding(final_warm["evaluation"]["generated_token_ids"])
                terminal_guard_pairs = _guard_bindings(final_warm["evaluation"]["generated_token_ids"])
                terminal_stabilizer_binding = _stabilizer_binding(
                    final_warm["evaluation"]["generated_token_ids"]
                )
                warm_target_margin, warm_guard_margins, warm_stabilizer_margin = _teacher_margins(
                    model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target_pair=terminal_target_pair, guard_pairs=terminal_guard_pairs,
                    stabilizer_binding=terminal_stabilizer_binding,
                )
                warm_stabilizer = _terminal_stabilizer_evidence(
                    stabilizer_binding=terminal_stabilizer_binding, margin=warm_stabilizer_margin,
                )
                if (
                    not terminal_guard_gate["passed"]
                    or not all(math.isfinite(value) and value > 0.0 for value in warm_guard_margins.values())
                    or warm_stabilizer["margin"] != warm_stabilizer_margin
                ):
                    raise ProjectedCorridorHold("HOLD: terminal corridor emitted an observed bad token or left feasibility")
            if accepted_steps == 0:
                saved = {"checkpoint": str(START_CHECKPOINT), "readback": {"reused_predecessor": True}}
            else:
                checkpoint = output / (
                    f"checkpoint-promotion-step-{accepted_steps:02d}"
                    if promoted else f"checkpoint-terminal-step-{accepted_steps:02d}"
                )
                saved = base._rank0_call(lambda model=model: {
                    "checkpoint": str(checkpoint),
                    "readback": base.full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=START_CHECKPOINT, destination=checkpoint,
                    ),
                })
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = base._frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise ProjectedCorridorHold("HOLD: frozen non-DoRA surface mutated")
            runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, opened
        torch.cuda.empty_cache()
        dist.barrier()
        assert saved is not None and final_warm is not None and terminal_surface is not None
        cold = base._rank0_call(lambda: _cold_evaluate(
            Path(saved["checkpoint"]), target=target, parent_owners=parent_owners,
            label="projected-corridor-cold",
        ))
        if (
            _state_identity(final_warm) != _state_identity(cold["packet"])
            or terminal_target_pair != cold["target_pair"]
            or terminal_guard_pairs != cold["guard_pairs"]
            or warm_target_margin != cold["target_margin"]
            or warm_guard_margins != cold["guard_margins"]
            or warm_stabilizer != cold["stabilizer"]
            or terminal_guard_gate != cold["observed_bad_token_guard_gate"]
            or terminal_surface != cold["surface"]
            or cold["frozen_surface"] != FROZEN_SURFACE_SHA256
        ):
            raise ProjectedCorridorHold("HOLD: warm/cold terminal identity mismatch")
        promoted = bool(final_warm["promotion_gate"]["passed"])
        owner_count = int(final_warm["evaluation"]["matched_target_owner_count"])
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "cold_match_level_promotion" if promoted else "bounded_negative_no_match_level_promotion",
            "run_id": run_id, "runner_source_snapshot": source_snapshot, "trigger": trigger,
            "bindings": {
                "start_checkpoint": str(START_CHECKPOINT), "predecessor_receipt": str(TRIGGER_RECEIPT),
                "predecessor_receipt_sha256": TRIGGER_SHA256,
                "prompt_sha256": base.PROMPT_TOKEN_SHA256,
                "target": str(base.TARGET_PATH), "target_sha256": base.TARGET_SHA256,
                "authority": str(base.AUTHORITY_PATH), "authority_sha256": base.AUTHORITY_SHA256,
                "natural_token_ids_sha256": NATURAL_TOKEN_SHA256,
                "initial_target_pair": initial_target_pair,
                "initial_guard_pairs": initial_guard_pairs,
                "terminal_target_pair": terminal_target_pair,
                "terminal_guard_pairs": terminal_guard_pairs,
            },
            "alias": alias,
            "guard_roles": {
                "active_guard_ids": list(ACTIVE_GUARD_IDS),
                "feasibility_guard_ids": list(FEASIBILITY_GUARD_IDS),
            },
            "protocol": {
                "world_size": WORLD_SIZE, "accepted_step_budget": MAX_ACCEPTED_STEPS,
                "prior_accepted_steps": PRIOR_ACCEPTED_STEPS,
                "combined_step_ceiling": COMBINED_STEP_CEILING,
                "radii_by_rank": {str(index): RADII[index] for index in range(WORLD_SIZE)},
                "base_learning_rate": base.LEARNING_RATE,
                "surface": "588 FP32 DoRA tensors / 18006016 elements only",
                "frozen": "all non-DoRA including tied delta, embeddings, aligner, vision, wrappers",
                "update": "theta + LR*radius*dynamic normalized direction; no optimizer state",
                "projection": "at lexical position 298, equal unit target-loss descent plus G5 ascent; at coordinate positions >298, normalized pure target-loss descent only",
                "selection": "proper Parent superset first; otherwise largest working-corridor observed-bad-token-safe radius strictly improving the frozen-current target while keeping natural token 8987 at position 298; G5 must strictly improve only at lexical position 298; all six candidate-on-policy guards stay positive; stabilizer margin is diagnostic only",
                "observed_bad_token_guards": "each guard fixes only lesion row_index, position, and bad token; good token, row SHA, and real-prefix SHA rebind to the current natural route",
                "acceptance": "match-level proper Parent superset only; never canonical-token identity",
                "dynamic_teachers": True, "fixed_guard_count": 6,
                "active_guard_ids": list(ACTIVE_GUARD_IDS),
                "feasibility_guard_ids": list(FEASIBILITY_GUARD_IDS),
                "gradient_objectives": {
                    "lexical_position_298": ["target_loss", "G5_margin"],
                    "post_description": ["target_loss"],
                },
                "tail_person_stabilizer": {
                    "position": STABILIZER_POSITION,
                    "person_token_id": STABILIZER_PERSON_TOKEN,
                    "competitor_token_ids": list(STABILIZER_COMPETITOR_TOKENS),
                    "margin": "z_person-logsumexp(z_tie,z_man)",
                    "role": "diagnostic_only_not_optimized_or_selected",
                },
                "unsupported_person_semantics": "near_miss and duplicate are subtype-specific transient optimization states only; never promotion or scientific success",
            },
            "parent_owner_ids": list(parent_owners), "accepted_step_count": accepted_steps,
            "optimization_state_contract": {
                "transient_person_near_miss": "working corridor only; never promotion or scientific success",
                "transient_person_duplicate": "working corridor only when raw duplicate evidence is false; never promotion or scientific success",
            },
            "attempts": attempts, "initial_surface": initial_surface, "terminal_surface": terminal_surface,
            "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
            "saved_checkpoint": saved,
            "warm_final": {
                "packet": final_warm, "target_margin": warm_target_margin,
                "guard_margins": warm_guard_margins,
                "stabilizer": warm_stabilizer,
                "observed_bad_token_guard_gate": terminal_guard_gate,
            },
            "cold": cold, "warm_cold_identity": _state_identity(final_warm),
            "promotion_scope": (
                "literal_46_owner_match" if promoted and owner_count == 46
                else "proper_parent_superset" if promoted else "none"
            ),
            "stop_reason": stop_reason, "runtime": runtime,
        }
        base._rank0_call(lambda: base._atomic_json(output / "receipt.json", receipt))
        dist.barrier()
        return output
    except BaseException as error:
        if rank == 0 and owns_output:
            base._atomic_json(output / "failure.json", {
                "schema_version": SCHEMA_VERSION + ".failure", "status": "mechanical_failure",
                "run_id": run_id, "runner_sha256": base._sha256(Path(__file__)),
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
            })
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
