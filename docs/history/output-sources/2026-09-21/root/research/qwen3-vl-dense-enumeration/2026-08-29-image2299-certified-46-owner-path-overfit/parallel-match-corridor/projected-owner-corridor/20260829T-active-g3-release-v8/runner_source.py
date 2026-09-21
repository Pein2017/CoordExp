#!/usr/bin/env python3
"""Eight-way DoRA causal-prefix corridor search with one active owner guard."""

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


SCHEMA_VERSION = "image2299.projected_owner_corridor.v8"
OUTPUT_ROOT = parallel.OUTPUT_ROOT / "projected-owner-corridor"
TRIGGER_RECEIPT = (
    OUTPUT_ROOT / "20260829T-causal-prefix-manifold-v7" / "receipt.json"
)
TRIGGER_SHA256 = "bf411c0ef8debf280d4371f6eb9900acb7a7b0c1a9648cb3ee34cef70811b89b"
TRIGGER_SOURCE = TRIGGER_RECEIPT.parent / "runner_source.py"
TRIGGER_SOURCE_SHA256 = "790b4b992889965d020b8d9a32b245470bbf31f3387ca0f0141cf587ac045b2a"
TRIGGER_STATUS = "bounded_negative_no_match_level_promotion"
TRIGGER_STOP = "no_feasible_margin_improving_guard_preserving_radius"
START_CHECKPOINT = (
    TRIGGER_RECEIPT.parent / "checkpoint-terminal-step-13"
)
TRIGGER_START_CHECKPOINT = (
    OUTPUT_ROOT / "20260829T-multi-guard-balanced-v6" / "checkpoint-terminal-step-06"
)
NATURAL_TOKEN_SHA256 = "c17b74f9cd4145845674a2faa41aa9bebd100acdd9cd9c45b807117b3de05bb8"
TRIGGER_INITIAL_NATURAL_SHA256 = "9fa8770d30a12749dd5e12a69bb46c0862d0d67cdc7954be8f4c77f4b5418007"
CLEAN_PARENT_TOKEN_SHA256 = base.PARENT_ROUTE_SHA256
TRIGGER_INITIAL_SURFACE_SHA256 = "a0b9add91f6e7b07e81175b1c4d98ccef97de8bcd112658effa16e9156c1c3ef"
START_SURFACE_SHA256 = "eaa50bc87bd4445083c53d2d545c5557a45a810f48b3f92f9e9bc3e7dae535c0"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
TARGET_POSITION, TARGET_GOOD, TARGET_BAD = 298, 8987, 48731
TARGET_MARGIN = -1.5727157592773438
TARGET_TEACHER_SHA256 = "8e9b05b5dcd8458ee287f5cece8d3d4456101242fae9d732a7b0c72f34d19ab0"
TRIGGER_GUARD_MARGINS = {
    "G0": 0.4136085510253906,
    "G1": 0.2293682098388672,
    "G2": 0.802220344543457,
}
GUARD_SPECS = (
    {
        "guard_id": "G0", "row_index": 6, "owner": "gt:2299:44",
        "observed_owner": "gt:2299:18", "cascade_gained_owner_ids": (), "position": 55,
        "good_token_id": 48731, "bad_token_id": 8987,
        "current_row": (151646, 48731, 151647, 151648, 151875, 152420, 151894, 152477, 151649),
        "failed_row": (151646, 8987, 151647, 151648, 151867, 151935, 151966, 152190, 151649),
        "real_prefix_sha256": "4ef946bafdee4ea75fb31bcf1bf78c8996906463a6e10a05143fecff836bb9af",
    },
    {
        "guard_id": "G1", "row_index": 4, "owner": "gt:2299:7",
        "observed_owner": "gt:2299:38", "cascade_gained_owner_ids": ("gt:2299:35",), "position": 40,
        "good_token_id": 151801, "bad_token_id": 151804,
        "current_row": (151646, 8987, 151647, 151648, 151801, 151670, 151896, 152120, 151649),
        "failed_row": (151646, 8987, 151647, 151648, 151804, 152305, 151944, 152669, 151649),
        "real_prefix_sha256": "27cefc965f0f09137227fb6ca486ab3d01ec89da5aa890b9d455421f08540f5c",
    },
    {
        "guard_id": "G2", "row_index": 21, "owner": "gt:2299:19",
        "observed_owner": "gt:2299:34", "cascade_gained_owner_ids": ("gt:2299:34",), "position": 194,
        "good_token_id": 151987, "bad_token_id": 152041,
        "current_row": (151646, 8987, 151647, 151648, 152305, 151987, 152400, 152169, 151649),
        "failed_row": (151646, 8987, 151647, 151648, 152305, 152041, 152400, 152392, 151649),
        "real_prefix_sha256": "28269b15686c408d02b37d1e5c494e225267dfb177d2d764a8a2a6be210f24ed",
    },
    {
        "guard_id": "G3", "row_index": 4, "owner": "gt:2299:7",
        "observed_owner": "gt:2299:22", "cascade_gained_owner_ids": ("gt:2299:18",),
        "position": 41, "good_token_id": 151670, "bad_token_id": 151867,
        "current_row": (151646, 8987, 151647, 151648, 151801, 151670, 151896, 152120, 151649),
        "failed_row": (151646, 8987, 151647, 151648, 151801, 151867, 152669, 152669, 151649),
        "real_prefix_sha256": "368df9f0df58d53534f1db8831123cdd85fcb65d59eed9bb8025ac882d988f48",
    },
)
ACTIVE_GUARD_IDS = ("G3",)
FEASIBILITY_GUARD_IDS = ("G0", "G1", "G2")
MAX_GUARD_POSITION = max(int(spec["position"]) for spec in GUARD_SPECS)
CAUSAL_PREFIX_LENGTH = MAX_GUARD_POSITION + 1
IMMUTABLE_PREFIX_SHA256 = "db90b4812866e6d295c51fc802ed806d71e6a1652b4df034b59112dff28eeb25"
REJECTED_CANDIDATE_SHA256 = "f7b26f63fe17251ec3f2437a93e3e0dd9f5a38d313b8f35dac35b414cd9a5a7b"
REJECTED_LCP = 41
REJECTED_ROW = (151646, 8987, 151647, 151648, 151801, 151867, 152669, 152669, 151649)
REJECTED_LOST_OWNER_IDS = (
    "gt:2299:7", "gt:2299:10", "gt:2299:12", "gt:2299:19", "gt:2299:21", "gt:2299:44",
)
REJECTED_GAINED_OWNER_IDS = ("gt:2299:18",)
REJECTED_HARD_COUNTERS = {
    "duplicate_person_count_iou95": 0, "malformed_count": 0,
    "matcher_ambiguity_neutral_count": 0, "matcher_unmatched_count": 1,
    "other_unknown_neutral_count": 0, "unknown_neutral_tie_count": 0,
    "unmatched_person_count": 1, "unsupported_person_count": 1,
}
OBSERVED_TAIL = (151646, 48731, 151647, 151648, 152596, 152097, 152617, 152138, 151649)
WORLD_SIZE = 8
MAX_ACCEPTED_STEPS = 14
PRIOR_ACCEPTED_STEPS = 66
COMBINED_STEP_CEILING = PRIOR_ACCEPTED_STEPS + MAX_ACCEPTED_STEPS
RADII = parallel.RADII
PROJECTION_TOLERANCE = 1e-7

ProjectedCorridorHold = base.Certified46Hold
_promotion_gate = parallel._promotion_gate
_corridor_gate = parallel._corridor_gate
_clone_parameters = parallel._clone_parameters
_restore_parameters = parallel._restore_parameters
_state_packet = parallel._state_packet
_state_identity = parallel._state_identity


def _trigger() -> dict[str, Any]:
    if (
        not TRIGGER_RECEIPT.is_file()
        or base._sha256(TRIGGER_RECEIPT) != TRIGGER_SHA256
        or not TRIGGER_SOURCE.is_file()
        or base._sha256(TRIGGER_SOURCE) != TRIGGER_SOURCE_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: projected-corridor trigger receipt identity drifted")
    receipt = json.loads(TRIGGER_RECEIPT.read_text(encoding="utf-8"))
    warm_final = dict(receipt.get("warm_final", {}))
    warm = dict(warm_final.get("packet", {}))
    cold_record = dict(receipt.get("cold", {}))
    cold = dict(cold_record.get("packet", {}))
    evaluation = dict(warm.get("evaluation", {}))
    causal = dict(warm.get("causal_ledger", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    clean_parent_tokens = base._parent_tokens()
    owners = list(map(str, receipt.get("parent_owner_ids", ())))
    saved = dict(receipt.get("saved_checkpoint", {}))
    attempts = list(receipt.get("attempts", ()))
    last = dict(attempts[-1]) if len(attempts) == 14 else {}
    panel = list(last.get("candidate_panel", ()))
    terminal_surface = dict(receipt.get("terminal_surface", {}))
    initial_surface = dict(receipt.get("initial_surface", {}))
    bindings = dict(receipt.get("bindings", {}))
    current_pair = dict(bindings.get("terminal_target_pair", {}))
    current_guards = list(bindings.get("terminal_guard_pairs", ()))
    source = dict(receipt.get("runner_source_snapshot", {}))
    if (
        receipt.get("schema_version") != "image2299.projected_owner_corridor.v7"
        or receipt.get("status") != TRIGGER_STATUS
        or receipt.get("stop_reason") != TRIGGER_STOP
        or int(receipt.get("accepted_step_count", -1)) != 13
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
        or (
            current_pair.get("pair_position"), current_pair.get("good_token_id"),
            current_pair.get("bad_token_id"), current_pair.get("teacher_tokens_sha256"),
        ) != (TARGET_POSITION, TARGET_GOOD, TARGET_BAD, TARGET_TEACHER_SHA256)
        or [guard.get("guard_id") for guard in current_guards] != ["G0", "G1", "G2"]
        or any(guard.get("teacher_tokens_sha256") != NATURAL_TOKEN_SHA256 for guard in current_guards)
        or evaluation.get("generated_token_ids_sha256") != NATURAL_TOKEN_SHA256
        or token_ids_sha256(tokens) != NATURAL_TOKEN_SHA256
        or token_ids_sha256(clean_parent_tokens) != CLEAN_PARENT_TOKEN_SHA256
        or _state_identity(warm) != _state_identity(cold)
        or receipt.get("warm_cold_identity") != _state_identity(cold)
        or warm_final.get("target_margin") != TARGET_MARGIN
        or cold_record.get("target_margin") != TARGET_MARGIN
        or warm_final.get("guard_margins") != TRIGGER_GUARD_MARGINS
        or cold_record.get("guard_margins") != TRIGGER_GUARD_MARGINS
        or cold_record.get("target_pair") != current_pair
        or cold_record.get("guard_pairs") != current_guards
        or terminal_surface.get("tensor_count") != 588
        or terminal_surface.get("element_count") != 18_006_016
        or terminal_surface.get("aggregate_sha256") != START_SURFACE_SHA256
        or initial_surface.get("aggregate_sha256") != TRIGGER_INITIAL_SURFACE_SHA256
        or initial_surface.get("tensor_count") != 588
        or initial_surface.get("element_count") != 18_006_016
        or cold_record.get("surface") != terminal_surface
        or receipt.get("frozen_surface_before") != FROZEN_SURFACE_SHA256
        or receipt.get("frozen_surface_after") != FROZEN_SURFACE_SHA256
        or cold_record.get("frozen_surface") != FROZEN_SURFACE_SHA256
        or list(tokens[-10:-1]) != list(OBSERVED_TAIL) or tokens[-1:] != [base.EOS]
        or causal.get("first_event", {}).get("kind") != "invalid_unmatched"
        or int(causal.get("first_event", {}).get("event_start", -1)) != 297
        or list(map(int, causal.get("first_event", {}).get("bad_span_tokens", ()))) != list(OBSERVED_TAIL)
        or list(map(str, causal.get("locked_owner_ids", ()))) != owners
        or bool(warm.get("promotion_gate", {}).get("passed"))
        or not bool(warm.get("corridor_gate", {}).get("passed"))
        or [int(item.get("attempt", -1)) for item in attempts] != list(range(1, 15))
        or [int(item.get("accepted_steps_before", -1)) for item in attempts] != list(range(14))
        or any(item.get("decision") != "accept_largest_radius_in_priority_class" for item in attempts[:-1])
        or int(last.get("attempt", -1)) != 14
        or int(last.get("accepted_steps_before", -1)) != 13
        or last.get("decision") != "no_eligible_radius_restore_current"
        or last.get("selected_broadcast") is not None
        or last.get("target_margin") != TARGET_MARGIN
        or last.get("guard_margins") != TRIGGER_GUARD_MARGINS
        or len(panel) != WORLD_SIZE
        or {int(item.get("rank", -1)) for item in panel} != set(range(WORLD_SIZE))
    ):
        raise ProjectedCorridorHold("HOLD: v7 trigger state/checkpoint/surface identity drifted")
    rejected_tokens: list[int] | None = None
    expected_matched = (set(owners) - set(REJECTED_LOST_OWNER_IDS)) | set(REJECTED_GAINED_OWNER_IDS)
    for item in panel:
        state = dict(item.get("state", {}))
        failed_evaluation = dict(state.get("evaluator", {}))
        failed_joint = dict(failed_evaluation.get("joint_gate", {}))
        failed_causal = dict(state.get("causal", {}))
        failed_promotion = dict(state.get("promotion_gate", {}))
        failed_tokens = list(map(int, state.get("generated_token_ids", ())))
        failed_matched = list(map(str, failed_evaluation.get("matched_target_owner_ids", ())))
        guard_improvements = dict(item.get("guards_improved", {}))
        if (
            float(item.get("radius", math.nan)) != RADII[int(item.get("rank", -1))]
            or not bool(item.get("target_margin_improved"))
            or guard_improvements != {"G0": True, "G1": True, "G2": True}
            or item.get("base_target_margin") != TARGET_MARGIN
            or item.get("base_guard_margins") != TRIGGER_GUARD_MARGINS
            or bool(item.get("promotion")) or bool(item.get("corridor"))
            or bool(dict(item.get("causal_prefix_gate", {})).get("passed"))
            or failed_evaluation.get("exact_target_prefix") != REJECTED_LCP
            or failed_evaluation.get("hard_counter_count") != 3
            or failed_evaluation.get("matched_target_owner_count") != 28
            or bool(failed_evaluation.get("all_parent_owners_retained"))
            or state.get("generated_token_ids_sha256") != REJECTED_CANDIDATE_SHA256
            or failed_causal.get("token_ids_sha256") != REJECTED_CANDIDATE_SHA256
            or token_ids_sha256(failed_tokens) != REJECTED_CANDIDATE_SHA256
            or base._exact_prefix(tokens, failed_tokens) != REJECTED_LCP
            or failed_tokens[4 * base.ROW_TOKENS:5 * base.ROW_TOKENS] != list(REJECTED_ROW)
            or set(failed_matched) != expected_matched
            or (set(owners) - set(failed_matched)) != set(REJECTED_LOST_OWNER_IDS)
            or (set(failed_matched) - set(owners)) != set(REJECTED_GAINED_OWNER_IDS)
            or failed_joint.get("hard_raw_counters") != REJECTED_HARD_COUNTERS
            or failed_joint.get("raw_counters") != {**REJECTED_HARD_COUNTERS, "valid_prediction_count": 29}
            or failed_causal.get("causal_hard_counter_count") != 1
            or set(map(str, failed_causal.get("locked_owner_ids", ()))) != expected_matched
            or failed_causal.get("first_event") != {
                "bad_span_tokens": list(REJECTED_ROW), "divergence": 0,
                "event_start": 36, "kind": "near_miss",
                "near_miss_owner": "gt:2299:22", "row_index": 4,
            }
            or bool(dict(state.get("corridor_gate", {})).get("passed"))
            or failed_promotion.get("added_owner_ids") != list(REJECTED_GAINED_OWNER_IDS)
            or failed_promotion.get("matched_owner_ids") != failed_matched
        ):
            raise ProjectedCorridorHold("HOLD: v7 rejected radius panel identity drifted")
        rejected_tokens = failed_tokens
    if rejected_tokens is None:
        raise ProjectedCorridorHold("HOLD: v7 rejected radius panel absent")
    prefix_gate = _causal_prefix_gate(rejected_tokens, tokens)
    if prefix_gate["passed"] or prefix_gate["lcp_with_incumbent"] != REJECTED_LCP:
        raise ProjectedCorridorHold("HOLD: v7 rejected route did not leave at the G3 lesion")
    _guard_bindings(tokens)
    return {
        "path": str(TRIGGER_RECEIPT), "sha256": TRIGGER_SHA256,
        "status": TRIGGER_STATUS, "stop_reason": TRIGGER_STOP,
        "accepted_step_count": 13, "start_checkpoint": str(START_CHECKPOINT),
        "natural_token_ids": tokens, "natural_token_ids_sha256": NATURAL_TOKEN_SHA256,
        "parent_owner_ids": owners,
        "terminal_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "predecessor_bindings": bindings,
        "rejected_candidate": {
            "attempt": 14, "ranks": list(range(WORLD_SIZE)), "radii": list(RADII),
            "lcp": REJECTED_LCP, "token_ids_sha256": REJECTED_CANDIDATE_SHA256,
            "token_ids": rejected_tokens, "failed_row": list(REJECTED_ROW),
            "lost_owner_ids": list(REJECTED_LOST_OWNER_IDS),
            "gained_owner_ids": list(REJECTED_GAINED_OWNER_IDS),
            "matched_owner_ids": sorted(expected_matched), "causal_prefix_gate": prefix_gate,
        },
    }


def _causal_prefix_gate(
    tokens: Sequence[int], immutable_incumbent: Sequence[int],
) -> dict[str, Any]:
    candidate = list(map(int, tokens))
    incumbent = list(map(int, immutable_incumbent))
    if (
        token_ids_sha256(incumbent) != NATURAL_TOKEN_SHA256
        or token_ids_sha256(incumbent[:CAUSAL_PREFIX_LENGTH]) != IMMUTABLE_PREFIX_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: immutable causal guard prefix identity drifted")
    candidate_prefix_sha256 = token_ids_sha256(candidate[:CAUSAL_PREFIX_LENGTH])
    return {
        "required_through_position": MAX_GUARD_POSITION,
        "required_prefix_length": CAUSAL_PREFIX_LENGTH,
        "immutable_prefix_sha256": IMMUTABLE_PREFIX_SHA256,
        "candidate_prefix_sha256": candidate_prefix_sha256,
        "lcp_with_incumbent": base._exact_prefix(incumbent, candidate),
        "passed": (
            len(candidate) >= CAUSAL_PREFIX_LENGTH
            and candidate_prefix_sha256 == IMMUTABLE_PREFIX_SHA256
        ),
    }


def _guard_bindings(tokens: Sequence[int]) -> list[dict[str, Any]]:
    natural = list(map(int, tokens))
    teacher_sha256 = token_ids_sha256(natural)
    if (
        len(natural) < CAUSAL_PREFIX_LENGTH
        or token_ids_sha256(natural[:CAUSAL_PREFIX_LENGTH]) != IMMUTABLE_PREFIX_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: causal guard teacher prefix drifted")
    guards = []
    for spec in GUARD_SPECS:
        start = int(spec["row_index"]) * base.ROW_TOKENS
        position = int(spec["position"])
        current_row = natural[start:start + base.ROW_TOKENS]
        if (
            len(current_row) != base.ROW_TOKENS
            or base.EOS in current_row
            or position != start + base._exact_prefix(spec["current_row"], spec["failed_row"])
            or natural[position] != int(spec["good_token_id"])
            or token_ids_sha256(natural[:position]) != spec["real_prefix_sha256"]
        ):
            raise ProjectedCorridorHold(f"HOLD: {spec['guard_id']} current on-policy guard drifted")
        guards.append({
            "guard_id": spec["guard_id"], "row_index": spec["row_index"],
            "owner": spec["owner"], "incumbent_owner": spec["owner"],
            "observed_owner": spec["observed_owner"],
            "cascade_gained_owner_ids": list(spec["cascade_gained_owner_ids"]),
            "position": position, "good_token_id": spec["good_token_id"],
            "bad_token_id": spec["bad_token_id"],
            "current_row": current_row, "reference_row": list(spec["current_row"]),
            "failed_row": list(spec["failed_row"]),
            "earliest_divergence_position": position,
            "real_prefix_sha256": spec["real_prefix_sha256"],
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
        or token_ids_sha256(natural[:CAUSAL_PREFIX_LENGTH]) != IMMUTABLE_PREFIX_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: target teacher violates row shape or causal guard prefix identity")
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


def _margin(logits: torch.Tensor, pair: Mapping[str, Any]) -> torch.Tensor:
    position = int(pair["position"] if "position" in pair else pair["pair_position"])
    good, bad = int(pair["good_token_id"]), int(pair["bad_token_id"])
    if logits.ndim != 2 or not (0 <= position < logits.shape[0]) or min(good, bad) < 0 or max(good, bad) >= logits.shape[1]:
        raise ProjectedCorridorHold("HOLD: malformed projected-corridor margin input")
    value = logits[position, good] - logits[position, bad]
    if not bool(torch.isfinite(value).item()):
        raise ProjectedCorridorHold("HOLD: non-finite projected-corridor margin")
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
    g3_gradient: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if len(target_gradient) != len(g3_gradient):
        raise ProjectedCorridorHold("HOLD: active-guard projection surface length drifted")
    names = ("target_loss", "G3_margin")
    gradients = [target_gradient, g3_gradient]
    norm_squared = {name: _fp64_dot(gradient, gradient) for name, gradient in zip(names, gradients, strict=True)}
    if any(not math.isfinite(value) or value <= 0.0 for value in norm_squared.values()):
        raise ProjectedCorridorHold("HOLD: zero/non-finite target or guard gradient")
    norms = {name: math.sqrt(norm_squared[name]) for name in names}
    units = [
        [value.detach() / norms[name] for value in gradient]
        for name, gradient in zip(names, gradients, strict=True)
    ]
    raw = [
        (-units[0][index] + units[1][index]).detach()
        for index in range(len(target_gradient))
    ]
    raw_norm = math.sqrt(max(0.0, _fp64_dot(raw, raw)))
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        raise ProjectedCorridorHold("HOLD: normalized multi-guard directions cancel")
    direction = [(value / raw_norm).detach() for value in raw]
    direction_norm = math.sqrt(max(0.0, _fp64_dot(direction, direction)))
    derivatives = {
        name: _fp64_dot(gradient, direction)
        for name, gradient in zip(names, gradients, strict=True)
    }
    pairwise_dots = {
        left_name: {
            right_name: _fp64_dot(left, right)
            for right_name, right in zip(names, gradients, strict=True)
        }
        for left_name, left in zip(names, gradients, strict=True)
    }
    cosine_matrix = {
        left_name: {
            right_name: pairwise_dots[left_name][right_name] / (norms[left_name] * norms[right_name])
            for right_name in names
        }
        for left_name in names
    }
    if (
        not math.isfinite(direction_norm)
        or not all(math.isfinite(value) for row in pairwise_dots.values() for value in row.values())
        or not all(math.isfinite(value) for value in derivatives.values())
        or abs(direction_norm - 1.0) > PROJECTION_TOLERANCE
        or derivatives["target_loss"] >= 0.0
        or derivatives["G3_margin"] <= 0.0
    ):
        raise ProjectedCorridorHold("HOLD: active-guard direction does not improve target and G3")
    return direction, {
        "mode": "active_release_target_plus_g3",
        "fp64_norm_squared": norm_squared,
        "fp64_pairwise_dots": pairwise_dots,
        "fp64_cosine_matrix": cosine_matrix,
        "fp64_directional_derivatives": derivatives,
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
    base_guard_margins: Mapping[str, float],
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
        and bool(dict(item.get("causal_prefix_gate", {})).get("passed"))
        and math.isfinite(float(item.get("target_margin", math.nan)))
        and float(item["target_margin"]) > base_target_margin
        and set(item.get("guard_margins", {})) == expected_guards
        and all(
            math.isfinite(float(item["guard_margins"][guard_id]))
            and float(item["guard_margins"][guard_id]) > 0.0
            for guard_id in expected_guards
        )
        and float(item["guard_margins"]["G3"]) > float(base_guard_margins["G3"])
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
        },
        "promotion_gate": packet["promotion_gate"], "corridor_gate": packet["corridor_gate"],
    }


def _teacher_margins(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    target_pair: Mapping[str, Any], guard_pairs: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    if (
        len(guard_pairs) != 4
        or {str(pair.get("guard_id")) for pair in guard_pairs} != {"G0", "G1", "G2", "G3"}
        or len({pair.get("teacher_tokens_sha256") for pair in guard_pairs}) != 1
    ):
        raise ProjectedCorridorHold("HOLD: exact four-guard teacher set drifted")
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
    return target_margin, guard_margins


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
        target_pair = _target_binding(cold_tokens)
        guard_pairs = _guard_bindings(cold_tokens)
        target_margin, guard_margins = _teacher_margins(
            model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
            target_pair=target_pair, guard_pairs=guard_pairs,
        )
        return {
            "packet": packet, "target_margin": target_margin, "guard_margins": guard_margins,
            "target_pair": target_pair, "guard_pairs": guard_pairs,
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
        stop_reason = "max_14_accepted_updates_without_warm_promotion"
        runtime: Mapping[str, Any] | None = None
        initial_surface: Mapping[str, Any] | None = None
        terminal_surface: Mapping[str, Any] | None = None
        frozen_before: str | None = None
        frozen_after: str | None = None
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
            baseline_target_margin, baseline_guard_margins = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=initial_target_pair, guard_pairs=initial_guard_pairs,
            )
            if (
                baseline_target_margin != TARGET_MARGIN
                or {
                    guard_id: baseline_guard_margins.get(guard_id)
                    for guard_id in FEASIBILITY_GUARD_IDS
                } != TRIGGER_GUARD_MARGINS
                or not all(math.isfinite(value) for value in baseline_guard_margins.values())
                or not all(value > 0.0 for value in baseline_guard_margins.values())
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
                raw_g3_gradients = torch.autograd.grad(
                    guard_margin_tensors["G3"], parameters, allow_unused=True,
                )
                del guard_logits
                target_gradient: list[torch.Tensor] = []
                g3_gradient: list[torch.Tensor] = []
                for parameter_index, target_value in enumerate(target_gradients):
                    values = [target_value, raw_g3_gradients[parameter_index]]
                    if any(
                        value is None or not bool(torch.isfinite(value).all().item())
                        for value in values
                    ):
                        raise ProjectedCorridorHold("HOLD: disconnected/non-finite target or G3 gradient")
                    reduced = []
                    for value in values:
                        value = value.detach()
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        reduced.append(value / WORLD_SIZE)
                    target_gradient.append(reduced[0])
                    g3_gradient.append(reduced[1])
                direction, projection = _project_direction(target_gradient, g3_gradient)
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
                clones = _clone_parameters(parameters)

                radius = RADII[rank]
                _apply_radius(parameters, clones, direction, radius=radius)
                base.full_root._assert_full_root_sentinels(model, sentinels)
                candidate_target_margin, candidate_guard_margins = _teacher_margins(
                    model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target_pair=target_pair, guard_pairs=guard_pairs,
                )
                candidate_packet = _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=alias["token_ids"],
                    label=f"projected-candidate-{attempt:02d}-rank-{rank}",
                )
                candidate_tokens = candidate_packet["evaluation"]["generated_token_ids"]
                candidate_prefix_gate = _causal_prefix_gate(
                    candidate_tokens, trigger["natural_token_ids"],
                )
                candidate_summary = {
                    "rank": rank, "radius": radius,
                    "base_target_margin": base_target_margin, "target_margin": candidate_target_margin,
                    "base_guard_margins": base_guard_margins, "guard_margins": candidate_guard_margins,
                    "target_margin_improved": candidate_target_margin > base_target_margin,
                    "guards_improved": {
                        guard_id: candidate_guard_margins[guard_id] > base_guard_margins[guard_id]
                        for guard_id in base_guard_margins
                    },
                    "candidate_lcp": candidate_prefix_gate["lcp_with_incumbent"],
                    "causal_prefix_gate": candidate_prefix_gate,
                    "promotion": bool(candidate_packet["promotion_gate"]["passed"]),
                    "corridor": bool(candidate_packet["corridor_gate"]["passed"]),
                    "state": _candidate_state(candidate_packet),
                }
                candidates: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(candidates, candidate_summary)
                selected = _select_candidate(
                    candidates, base_target_margin=base_target_margin,
                    base_guard_margins=base_guard_margins,
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
                    stop_reason = "no_feasible_target_g3_improving_guard_positive_radius"
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
                    label="projected-final-step-14",
                ))
            promoted = bool(final_warm["promotion_gate"]["passed"])
            if not promoted and not final_warm["corridor_gate"]["passed"]:
                raise ProjectedCorridorHold("HOLD: terminal policy is neither promotion nor corridor-feasible")
            terminal_target_pair = _target_binding(final_warm["evaluation"]["generated_token_ids"])
            terminal_guard_pairs = _guard_bindings(final_warm["evaluation"]["generated_token_ids"])
            warm_target_margin, warm_guard_margins = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=terminal_target_pair, guard_pairs=terminal_guard_pairs,
            )
            terminal_prefix_gate = _causal_prefix_gate(
                final_warm["evaluation"]["generated_token_ids"], trigger["natural_token_ids"],
            )
            if not promoted and (
                not terminal_prefix_gate["passed"]
                or not all(math.isfinite(value) and value > 0.0 for value in warm_guard_margins.values())
            ):
                raise ProjectedCorridorHold("HOLD: terminal corridor left causal prefix or guard feasibility")
            if accepted_steps == 0:
                saved = {"checkpoint": str(START_CHECKPOINT), "readback": {"reused_immutable_predecessor": True}}
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
                "update": "theta + LR*radius*globally-clipped projected direction; no optimizer state",
                "projection": "unit target-loss descent plus unit G3-margin ascent, summed then renormalized once; G0-G2 have no gradient contribution",
                "selection": "proper Parent superset first; otherwise largest corridor-feasible causal-prefix radius strictly improving target and G3 with all four margins strictly positive",
                "causal_prefix_manifold": f"tokens 0..{MAX_GUARD_POSITION} equal the immutable v7 incumbent; later tokens may vary",
                "acceptance": "match-level proper Parent superset only; never canonical-token identity",
                "dynamic_teachers": True, "fixed_guard_count": 4,
                "active_guard_ids": list(ACTIVE_GUARD_IDS),
                "feasibility_guard_ids": list(FEASIBILITY_GUARD_IDS),
                "gradient_objectives": ["target_loss", "G3_margin"],
            },
            "parent_owner_ids": list(parent_owners), "accepted_step_count": accepted_steps,
            "attempts": attempts, "initial_surface": initial_surface, "terminal_surface": terminal_surface,
            "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
            "saved_checkpoint": saved,
            "warm_final": {
                "packet": final_warm, "target_margin": warm_target_margin,
                "guard_margins": warm_guard_margins,
                "causal_prefix_gate": terminal_prefix_gate,
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
