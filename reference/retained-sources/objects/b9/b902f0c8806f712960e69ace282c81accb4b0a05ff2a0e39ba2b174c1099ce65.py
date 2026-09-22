#!/usr/bin/env python3
"""Eight-way DoRA corridor search projected against the first owner exchange."""

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


SCHEMA_VERSION = "image2299.projected_owner_corridor.v3"
OUTPUT_ROOT = parallel.OUTPUT_ROOT / "projected-owner-corridor"
TRIGGER_RECEIPT = (
    OUTPUT_ROOT / "20260829T-projected-owner-corridor-v1" / "receipt.json"
)
TRIGGER_SHA256 = "6eeb786ba8e689277fc5733e9b2ea93ad8897d634d4cf4d8c27ef945d4133ab7"
TRIGGER_SOURCE = TRIGGER_RECEIPT.parent / "runner_source.py"
TRIGGER_SOURCE_SHA256 = "bab6ae3eaa272420e9572f4b67305076f570e3c717bacf9c5c4d42eb90670b6f"
TRIGGER_STATUS = "bounded_negative_no_match_level_promotion"
TRIGGER_STOP = "no_feasible_margin_improving_guard_preserving_radius"
START_CHECKPOINT = (
    parallel.OUTPUT_ROOT
    / "20260829T-parallel-match-corridor-v1c"
    / "checkpoint-terminal-step-36"
)
NATURAL_TOKEN_SHA256 = "9fa8770d30a12749dd5e12a69bb46c0862d0d67cdc7954be8f4c77f4b5418007"
START_SURFACE_SHA256 = "6c4e8a11fe27e87ef7e361da1c63c15a1784cab1cb2fb5cdfe61048964428c00"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
TARGET_POSITION, TARGET_GOOD, TARGET_BAD = 298, 8987, 48731
TARGET_MARGIN = -2.0594635009765625
TARGET_TEACHER_SHA256 = "672a4c3ec9216bf237a8a6269d8f9276dae57dad40bf529c36582d3ffd72c84a"
GUARD_POSITION, GUARD_GOOD, GUARD_BAD = 55, 48731, 8987
GUARD_ROW_INDEX, GUARD_OWNER = 6, "gt:2299:44"
FAILED_GUARD_OWNER = "gt:2299:18"
GUARD_ROW = (151646, 48731, 151647, 151648, 151875, 152420, 151894, 152477, 151649)
FAILED_GUARD_ROW = (151646, 8987, 151647, 151648, 151867, 151935, 151966, 152190, 151649)
GUARD_REAL_PREFIX_SHA256 = "4ef946bafdee4ea75fb31bcf1bf78c8996906463a6e10a05143fecff836bb9af"
FAILED_CANDIDATE_SHA256 = "3795729374ea06509e09f31b77136a932eeb01adb48613a54aaf55d91e8551ff"
FAILED_LOST_OWNER_IDS = (
    "gt:2299:10", "gt:2299:12", "gt:2299:19", "gt:2299:26", "gt:2299:44",
)
FAILED_GAINED_OWNER_IDS = (FAILED_GUARD_OWNER,)
OBSERVED_TAIL = (151646, 48731, 151647, 151648, 152596, 152097, 152617, 152138, 151649)
WORLD_SIZE = 8
MAX_ACCEPTED_STEPS = 14
RADII = parallel.RADII
GUARD_MARGIN_FLOOR = 0.0
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
    cold = dict(dict(receipt.get("cold", {})).get("packet", {}))
    evaluation = dict(warm.get("evaluation", {}))
    causal = dict(warm.get("causal_ledger", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    owners = list(map(str, receipt.get("parent_owner_ids", ())))
    saved = dict(receipt.get("saved_checkpoint", {}))
    attempts = list(receipt.get("attempts", ()))
    last = dict(attempts[0]) if len(attempts) == 1 else {}
    panel = list(last.get("candidate_panel", ()))
    rank7 = next((dict(item) for item in panel if int(item.get("rank", -1)) == 7), {})
    failed_state = dict(rank7.get("state", {}))
    failed_evaluation = dict(failed_state.get("evaluator", {}))
    failed_promotion = dict(failed_state.get("promotion_gate", {}))
    failed_tokens = list(map(int, failed_state.get("generated_token_ids", ())))
    failed_matched = list(map(str, failed_evaluation.get("matched_target_owner_ids", ())))
    failed_lost = {owner for owner in owners if owner not in failed_matched}
    failed_gained = {owner for owner in failed_matched if owner not in owners}
    terminal_surface = dict(receipt.get("terminal_surface", {}))
    initial_surface = dict(receipt.get("initial_surface", {}))
    cold_record = dict(receipt.get("cold", {}))
    bindings = dict(receipt.get("bindings", {}))
    source = dict(receipt.get("runner_source_snapshot", {}))
    predecessor = dict(receipt.get("trigger", {}))
    if (
        receipt.get("schema_version") != "image2299.projected_owner_corridor.v1"
        or receipt.get("status") != TRIGGER_STATUS
        or receipt.get("stop_reason") != TRIGGER_STOP
        or int(receipt.get("accepted_step_count", -1)) != 0
        or source != {"path": str(TRIGGER_SOURCE), "sha256": TRIGGER_SOURCE_SHA256}
        or bindings.get("start_checkpoint") != str(START_CHECKPOINT)
        or saved.get("checkpoint") != str(START_CHECKPOINT)
        or saved.get("readback") != {"reused_immutable_predecessor": True}
        or not START_CHECKPOINT.is_dir()
        or len(owners) != 33 or len(set(owners)) != 33
        or bindings.get("prompt_sha256") != base.PROMPT_TOKEN_SHA256
        or bindings.get("target_sha256") != base.TARGET_SHA256
        or bindings.get("authority_sha256") != base.AUTHORITY_SHA256
        or evaluation.get("generated_token_ids_sha256") != NATURAL_TOKEN_SHA256
        or token_ids_sha256(tokens) != NATURAL_TOKEN_SHA256
        or predecessor.get("natural_token_ids") != tokens
        or predecessor.get("natural_token_ids_sha256") != NATURAL_TOKEN_SHA256
        or _state_identity(warm) != _state_identity(cold)
        or receipt.get("warm_cold_identity") != _state_identity(cold)
        or warm_final.get("target_margin") != cold_record.get("target_margin")
        or warm_final.get("guard_margin") != cold_record.get("guard_margin")
        or terminal_surface.get("tensor_count") != 588
        or terminal_surface.get("element_count") != 18_006_016
        or terminal_surface.get("aggregate_sha256") != START_SURFACE_SHA256
        or initial_surface != terminal_surface
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
        or int(last.get("attempt", -1)) != 1
        or int(last.get("accepted_steps_before", -1)) != 0
        or last.get("decision") != "no_eligible_radius_restore_current"
        or last.get("selected_broadcast") is not None
        or len(panel) != WORLD_SIZE
        or {int(item.get("rank", -1)) for item in panel} != set(range(WORLD_SIZE))
        or float(rank7.get("radius", math.nan)) != RADII[7]
        or not bool(rank7.get("target_margin_improved"))
        or bool(rank7.get("promotion")) or bool(rank7.get("corridor"))
        or bool(rank7.get("specimen_identity"))
        or failed_evaluation.get("exact_target_prefix") != GUARD_POSITION
        or failed_state.get("generated_token_ids_sha256") != FAILED_CANDIDATE_SHA256
        or token_ids_sha256(failed_tokens) != FAILED_CANDIDATE_SHA256
        or base._exact_prefix(tokens, failed_tokens) != GUARD_POSITION
        or failed_tokens[GUARD_ROW_INDEX * base.ROW_TOKENS:(GUARD_ROW_INDEX + 1) * base.ROW_TOKENS]
        != list(FAILED_GUARD_ROW)
        or failed_lost != set(FAILED_LOST_OWNER_IDS)
        or failed_gained != set(FAILED_GAINED_OWNER_IDS)
        or failed_promotion.get("added_owner_ids") != list(FAILED_GAINED_OWNER_IDS)
        or failed_promotion.get("matched_owner_ids") != failed_matched
    ):
        raise ProjectedCorridorHold("HOLD: trigger does not bind the first owner-exchange specimen")
    row_start = GUARD_ROW_INDEX * base.ROW_TOKENS
    if (
        tokens[row_start:row_start + base.ROW_TOKENS] != list(GUARD_ROW)
        or base._exact_prefix(GUARD_ROW, FAILED_GUARD_ROW) != GUARD_POSITION - row_start
        or token_ids_sha256(tokens[:GUARD_POSITION]) != GUARD_REAL_PREFIX_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: guard rows do not first diverge at absolute position 55")
    return {
        "path": str(TRIGGER_RECEIPT), "sha256": TRIGGER_SHA256,
        "status": TRIGGER_STATUS, "stop_reason": TRIGGER_STOP,
        "accepted_step_count": 0, "start_checkpoint": str(START_CHECKPOINT),
        "natural_token_ids": tokens, "natural_token_ids_sha256": NATURAL_TOKEN_SHA256,
        "parent_owner_ids": owners,
        "terminal_surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "predecessor_bindings": bindings,
        "predecessor_trigger": receipt.get("trigger"),
        "failed_candidate": {
            "attempt": 1, "rank": 7, "radius": RADII[7], "lcp": GUARD_POSITION,
            "token_ids_sha256": FAILED_CANDIDATE_SHA256,
            "lost_owner_ids": list(FAILED_LOST_OWNER_IDS),
            "gained_owner_ids": list(FAILED_GAINED_OWNER_IDS),
        },
    }


def _guard_binding(tokens: Sequence[int]) -> dict[str, Any]:
    natural = list(map(int, tokens))
    start = GUARD_ROW_INDEX * base.ROW_TOKENS
    if (
        token_ids_sha256(natural) != NATURAL_TOKEN_SHA256
        or natural[start:start + base.ROW_TOKENS] != list(GUARD_ROW)
        or GUARD_POSITION != start + base._exact_prefix(GUARD_ROW, FAILED_GUARD_ROW)
        or natural[GUARD_POSITION] != GUARD_GOOD
        or token_ids_sha256(natural[:GUARD_POSITION]) != GUARD_REAL_PREFIX_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: natural guard prefix identity drifted")
    return {
        "row_index": GUARD_ROW_INDEX, "owner": GUARD_OWNER, "position": GUARD_POSITION,
        "failed_replacement_owner": FAILED_GUARD_OWNER,
        "good_token_id": GUARD_GOOD, "bad_token_id": GUARD_BAD,
        "accepted_row": list(GUARD_ROW), "failed_row": list(FAILED_GUARD_ROW),
        "earliest_divergence_position": GUARD_POSITION,
        "real_prefix_sha256": token_ids_sha256(natural[:GUARD_POSITION]),
        "teacher_tokens": natural, "teacher_tokens_sha256": token_ids_sha256(natural),
    }


def _target_binding(tokens: Sequence[int]) -> dict[str, Any]:
    pair = parallel._frozen_pair(tokens, parallel.ALIAS_TOKENS)
    if (
        pair["pair_position"] != TARGET_POSITION
        or pair["good_token_id"] != TARGET_GOOD
        or pair["bad_token_id"] != TARGET_BAD
        or pair["teacher_tokens_sha256"] != TARGET_TEACHER_SHA256
    ):
        raise ProjectedCorridorHold("HOLD: target teacher pair identity drifted")
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
    target_gradient: Sequence[torch.Tensor], guard_gradient: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    delta0 = [-gradient.detach() for gradient in target_gradient]
    h_dot_delta0 = _fp64_dot(guard_gradient, delta0)
    h_norm_sq = _fp64_dot(guard_gradient, guard_gradient)
    if not math.isfinite(h_dot_delta0) or not math.isfinite(h_norm_sq) or h_norm_sq <= 0.0:
        raise ProjectedCorridorHold("HOLD: zero/non-finite guard gradient")
    alpha = max(0.0, -h_dot_delta0 / h_norm_sq)
    direction = [
        (delta + alpha * guard).detach()
        for delta, guard in zip(delta0, guard_gradient, strict=True)
    ]
    projected_norm = math.sqrt(max(0.0, _fp64_dot(direction, direction)))
    preclip_h_dot = _fp64_dot(guard_gradient, direction)
    preclip_g_dot = _fp64_dot(target_gradient, direction)
    if not math.isfinite(projected_norm) or projected_norm <= 0.0:
        raise ProjectedCorridorHold("HOLD: zero/non-finite projected direction")
    scale = min(1.0, 1.0 / projected_norm)
    direction = [(value * scale).detach() for value in direction]
    clipped_norm = math.sqrt(max(0.0, _fp64_dot(direction, direction)))
    postclip_h_dot = _fp64_dot(guard_gradient, direction)
    postclip_g_dot = _fp64_dot(target_gradient, direction)
    if (
        not all(math.isfinite(value) for value in (alpha, preclip_h_dot, preclip_g_dot, clipped_norm, postclip_h_dot, postclip_g_dot))
        or clipped_norm <= 0.0 or clipped_norm > 1.0 + PROJECTION_TOLERANCE
        or preclip_h_dot < -PROJECTION_TOLERANCE
        or postclip_h_dot < -PROJECTION_TOLERANCE
        or preclip_g_dot >= 0.0 or postclip_g_dot >= 0.0
    ):
        raise ProjectedCorridorHold("HOLD: projected direction violates target or guard derivative")
    return direction, {
        "fp64_h_dot_delta0": h_dot_delta0, "fp64_h_norm_squared": h_norm_sq,
        "alpha": alpha, "constraint_active": alpha > 0.0,
        "projected_norm_before_global_clip": projected_norm,
        "global_clip_scale": scale, "direction_norm": clipped_norm,
        "fp64_h_dot_direction_preclip": preclip_h_dot,
        "fp64_target_g_dot_direction_preclip": preclip_g_dot,
        "fp64_h_dot_direction_postclip": postclip_h_dot,
        "fp64_target_g_dot_direction_postclip": postclip_g_dot,
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
    candidates: Sequence[Mapping[str, Any]], *, base_target_margin: float, base_guard_margin: float,
) -> Mapping[str, Any] | None:
    by_rank = {int(item["rank"]): item for item in candidates}
    if (
        set(by_rank) != set(range(WORLD_SIZE))
        or tuple(float(by_rank[index]["radius"]) for index in range(WORLD_SIZE)) != RADII
    ):
        raise ProjectedCorridorHold("HOLD: projected candidate radius panel is incomplete")
    promotions = [item for item in candidates if bool(item.get("promotion"))]
    if promotions:
        return max(promotions, key=lambda item: float(item["radius"]))
    eligible = [
        item for item in candidates
        if bool(item.get("corridor"))
        and bool(item.get("specimen_identity"))
        and math.isfinite(float(item.get("target_margin", math.nan)))
        and float(item["target_margin"]) > base_target_margin
        and math.isfinite(float(item.get("guard_margin", math.nan)))
        and float(item["guard_margin"]) > GUARD_MARGIN_FLOOR
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
    target_pair: Mapping[str, Any], guard_pair: Mapping[str, Any],
) -> tuple[float, float]:
    with torch.inference_mode():
        target_logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs, route_tokens=target_pair["teacher_tokens"],
            pad_token_id=pad,
        )
        target_margin = float(_margin(target_logits, target_pair).item())
        guard_logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs, route_tokens=guard_pair["teacher_tokens"],
            pad_token_id=pad,
        )
        guard_margin = float(_margin(guard_logits, guard_pair).item())
    return target_margin, guard_margin


def _cold_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], parent_owners: Sequence[str],
    target_pair: Mapping[str, Any], guard_pair: Mapping[str, Any], label: str,
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
        target_margin, guard_margin = _teacher_margins(
            model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
            target_pair=target_pair, guard_pair=guard_pair,
        )
        return {
            "packet": packet, "target_margin": target_margin, "guard_margin": guard_margin,
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
        target_pair = _target_binding(trigger["natural_token_ids"])
        guard_pair = _guard_binding(trigger["natural_token_ids"])
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
            baseline_target_margin, baseline_guard_margin = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=target_pair, guard_pair=guard_pair,
            )
            if baseline_target_margin != TARGET_MARGIN or not math.isfinite(baseline_guard_margin):
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
                    model=model, native_inputs=native_inputs, route_tokens=guard_pair["teacher_tokens"],
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                guard_margin_tensor = _margin(guard_logits, guard_pair)
                guard_gradients = torch.autograd.grad(guard_margin_tensor, parameters, allow_unused=True)
                del guard_logits
                target_gradient: list[torch.Tensor] = []
                guard_gradient: list[torch.Tensor] = []
                for target_value, guard_value in zip(target_gradients, guard_gradients, strict=True):
                    if target_value is None or guard_value is None or not bool(torch.isfinite(target_value).all().item()) or not bool(torch.isfinite(guard_value).all().item()):
                        raise ProjectedCorridorHold("HOLD: disconnected/non-finite target or guard gradient")
                    target_value = target_value.detach()
                    guard_value = guard_value.detach()
                    dist.all_reduce(target_value, op=dist.ReduceOp.SUM)
                    dist.all_reduce(guard_value, op=dist.ReduceOp.SUM)
                    target_gradient.append(target_value / WORLD_SIZE)
                    guard_gradient.append(guard_value / WORLD_SIZE)
                direction, projection = _project_direction(target_gradient, guard_gradient)
                direction_snapshot = _direction_snapshot(names, direction)
                projection["direction"] = direction_snapshot
                projection_identities: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(projection_identities, base._hash(projection))
                if len(set(projection_identities)) != 1:
                    raise ProjectedCorridorHold("HOLD: rank projection disagreement")
                base_target_margin = float(target_margin_tensor.detach().item())
                base_guard_margin = float(guard_margin_tensor.detach().item())
                clones = _clone_parameters(parameters)

                radius = RADII[rank]
                _apply_radius(parameters, clones, direction, radius=radius)
                base.full_root._assert_full_root_sentinels(model, sentinels)
                candidate_target_margin, candidate_guard_margin = _teacher_margins(
                    model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                    target_pair=target_pair, guard_pair=guard_pair,
                )
                candidate_packet = _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=alias["token_ids"],
                    label=f"projected-candidate-{attempt:02d}-rank-{rank}",
                )
                candidate_summary = {
                    "rank": rank, "radius": radius,
                    "base_target_margin": base_target_margin, "target_margin": candidate_target_margin,
                    "base_guard_margin": base_guard_margin, "guard_margin": candidate_guard_margin,
                    "target_margin_improved": candidate_target_margin > base_target_margin,
                    "guard_preserved": candidate_guard_margin > GUARD_MARGIN_FLOOR,
                    "specimen_identity": (
                        candidate_packet["evaluation"]["generated_token_ids"]
                        == trigger["natural_token_ids"]
                    ),
                    "promotion": bool(candidate_packet["promotion_gate"]["passed"]),
                    "corridor": bool(candidate_packet["corridor_gate"]["passed"]),
                    "state": _candidate_state(candidate_packet),
                }
                candidates: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(candidates, candidate_summary)
                selected = _select_candidate(
                    candidates, base_target_margin=base_target_margin,
                    base_guard_margin=base_guard_margin,
                )
                record: dict[str, Any] = {
                    "attempt": attempt, "accepted_steps_before": accepted_steps,
                    "frozen_state": _candidate_state(current), "frozen_surface": current_surface,
                    "target_margin": base_target_margin, "guard_margin": base_guard_margin,
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
                    stop_reason = "no_feasible_margin_improving_guard_preserving_radius"
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
                    "guard_margin": float(selected["guard_margin"]),
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
            warm_target_margin, warm_guard_margin = _teacher_margins(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                target_pair=target_pair, guard_pair=guard_pair,
            )
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
            target_pair=target_pair, guard_pair=guard_pair, label="projected-corridor-cold",
        ))
        if (
            _state_identity(final_warm) != _state_identity(cold["packet"])
            or warm_target_margin != cold["target_margin"]
            or warm_guard_margin != cold["guard_margin"]
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
                "target_pair": {key: value for key, value in target_pair.items() if key != "teacher_tokens"},
                "guard_pair": {key: value for key, value in guard_pair.items() if key != "teacher_tokens"},
            },
            "alias": alias,
            "protocol": {
                "world_size": WORLD_SIZE, "accepted_step_budget": MAX_ACCEPTED_STEPS,
                "prior_accepted_steps": 36, "combined_step_ceiling": 50,
                "radii_by_rank": {str(index): RADII[index] for index in range(WORLD_SIZE)},
                "base_learning_rate": base.LEARNING_RATE,
                "surface": "588 FP32 DoRA tensors / 18006016 elements only",
                "frozen": "all non-DoRA including tied delta, embeddings, aligner, vision, wrappers",
                "update": "theta + LR*radius*globally-clipped projected direction; no optimizer state",
                "projection": "delta0=-grad softplus(z_bad-z_good); one fixed behavioral guard half-space",
                "guard_margin_floor": GUARD_MARGIN_FLOOR,
                "selection": "proper Parent superset with zero debt before auxiliaries; otherwise largest target-improving guard-preserving corridor radius",
                "fixed_specimen": "non-promotion steps retain the exact natural route so both protected teachers stay on-policy",
                "acceptance": "match-level proper Parent superset only; never canonical-token identity",
                "dynamic_guards": False,
            },
            "parent_owner_ids": list(parent_owners), "accepted_step_count": accepted_steps,
            "attempts": attempts, "initial_surface": initial_surface, "terminal_surface": terminal_surface,
            "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
            "saved_checkpoint": saved,
            "warm_final": {
                "packet": final_warm, "target_margin": warm_target_margin,
                "guard_margin": warm_guard_margin,
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
