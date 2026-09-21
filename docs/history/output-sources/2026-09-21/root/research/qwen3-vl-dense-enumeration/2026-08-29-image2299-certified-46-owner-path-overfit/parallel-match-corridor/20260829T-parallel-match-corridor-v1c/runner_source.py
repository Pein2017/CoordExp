#!/usr/bin/env python3
"""Eight-way DoRA backtracking inside the Parent33 match corridor."""

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
import torch.nn.functional as F
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_certified_46_owner_path_overfit as base
from scripts.research import run_image2299_manifold_match_anchor_overfit as manifold
from src.inference.backend import open_backend_session, token_ids_sha256
from src.inference.hf_backend import HFBackendSession


SCHEMA_VERSION = "image2299.parallel_match_corridor.v1"
OUTPUT_ROOT = base.OUTPUT_ROOT / "parallel-match-corridor"
TRIGGER_RECEIPT = (
    base.OUTPUT_ROOT / "manifold-match-anchor"
    / "20260829T-manifold-match-anchor-v1" / "receipt.json"
)
TRIGGER_SHA256 = "993308e320ae9ca9608abda133cd6fe96f8f820a7722fa2d2330cbd89d3d51f5"
TRIGGER_STATUS = "bounded_negative_no_match_level_promotion"
TRIGGER_STOP = "max_50_updates_without_warm_promotion"
ALIAS_TOKENS = (151646, 8987, 151647, 151648, 152399, 152078, 152472, 152446, 151649)
ALIAS_SHA256 = "b814b97253384b7bcfed5883b5b89f204b07259a6018aeb704fbceb592364427"
WORLD_SIZE = 8
MAX_ACCEPTED_STEPS = 50
RADII = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
TAIL_DECISIONS = frozenset({"near_miss", "invalid_unmatched"})
TAIL_DERIVED_COUNTERS = frozenset({
    "matcher_unmatched_count", "unmatched_person_count",
    "unknown_neutral_tie_count", "other_unknown_neutral_count",
})
FORBIDDEN_COUNTERS = frozenset({
    "malformed_count", "matcher_ambiguity_neutral_count", "duplicate_person_count_iou95",
})

ParallelCorridorHold = base.Certified46Hold
_promotion_gate = manifold._promotion_gate


def _trigger() -> dict[str, Any]:
    if not TRIGGER_RECEIPT.is_file() or base._sha256(TRIGGER_RECEIPT) != TRIGGER_SHA256:
        raise ParallelCorridorHold("HOLD: parallel-corridor trigger receipt identity drifted")
    receipt = json.loads(TRIGGER_RECEIPT.read_text(encoding="utf-8"))
    selected = dict(receipt.get("selected_alias", {}))
    bindings = dict(receipt.get("bindings", {}))
    parent = list(map(str, receipt.get("parent_owner_ids", ())))
    if (
        receipt.get("status") != TRIGGER_STATUS
        or receipt.get("stop_reason") != TRIGGER_STOP
        or bindings.get("parent_checkpoint") != str(base.PARENT_CHECKPOINT)
        or bindings.get("parent_route_sha256") != base.PARENT_ROUTE_SHA256
        or bindings.get("target_sha256") != base.TARGET_SHA256
        or bindings.get("authority_sha256") != base.AUTHORITY_SHA256
        or selected.get("name") != "A"
        or list(map(int, selected.get("token_ids", ()))) != list(ALIAS_TOKENS)
        or selected.get("row_sha256") != ALIAS_SHA256
        or len(parent) != 33
        or len(set(parent)) != 33
    ):
        raise ParallelCorridorHold("HOLD: trigger does not bind Parent33, evaluator, and alias A")
    return {
        "path": str(TRIGGER_RECEIPT), "sha256": TRIGGER_SHA256,
        "status": TRIGGER_STATUS, "stop_reason": TRIGGER_STOP,
        "start_checkpoint": str(base.PARENT_CHECKPOINT),
        "parent_owner_ids": parent,
        "selected_alias": {
            "name": "A", "token_ids": list(ALIAS_TOKENS), "row_sha256": ALIAS_SHA256,
        },
        "evaluator": {
            "target_sha256": base.TARGET_SHA256,
            "authority_sha256": base.AUTHORITY_SHA256,
            "match_level_acceptance": True,
        },
    }


def _admit_alias(*, tokenizer: Any, raw_example: Any) -> dict[str, Any]:
    candidate = next(item for item in manifold.ALIAS_CANDIDATES if item["name"] == "A")
    admitted = manifold._admit_alias_candidate(
        tokenizer=tokenizer, raw_example=raw_example, candidate=candidate,
    )
    if admitted["token_ids"] != list(ALIAS_TOKENS) or admitted["row_sha256"] != ALIAS_SHA256:
        raise ParallelCorridorHold("HOLD: locally admitted alias A identity drifted")
    return admitted


def _counter_debt(counters: Mapping[str, Any], *, tail: bool) -> bool:
    try:
        values = {str(key): int(value) for key, value in counters.items()}
    except (TypeError, ValueError):
        return True
    if any(value < 0 for value in values.values()):
        return True
    if tail:
        if values.get("matcher_unmatched_count", -1) != 1:
            return True
        if any(values.get(key, 0) != 0 for key in FORBIDDEN_COUNTERS):
            return True
        if any(value != 0 for key, value in values.items() if key not in TAIL_DERIVED_COUNTERS | FORBIDDEN_COUNTERS):
            return True
        return any(values.get(key, 0) > 1 for key in TAIL_DERIVED_COUNTERS)
    return any(values.values())


def _corridor_gate(
    evaluation: Mapping[str, Any], causal_ledger: Mapping[str, Any],
    parent_owners: Sequence[str],
) -> dict[str, Any]:
    """Accept only Parent33+EOS or Parent33+one unassigned final row+EOS."""
    parent = set(map(str, parent_owners))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    matcher = dict(joint.get("matcher", {}))
    statuses = dict(matcher.get("strict_status_counts", {}))
    committed = set(map(str, matcher.get("committed_owner_ids", ())))
    rows = list(causal_ledger.get("rows", ()))
    tail = len(tokens) == 34 * base.ROW_TOKENS + 1
    expected_rows = 34 if tail else 33
    expected_statuses = {"matched": 33, **({"unmatched": 1} if tail else {})}
    first = dict(causal_ledger.get("first_event") or {})
    backbone = rows[:33]
    backbone_owners = [str(row.get("owner", "")) for row in backbone]
    event_ok = (
        first.get("kind") in TAIL_DECISIONS
        and int(first.get("event_start", -1)) == 297
        and int(first.get("row_index", -1)) == 33
    ) if tail else (
        first.get("kind") == "premature_eos"
        and int(first.get("event_start", -1)) == 297
        and int(first.get("row_index", -1)) == 33
    )
    tail_ok = (
        len(rows) == 34
        and rows[-1].get("parse_state") == "valid"
        and rows[-1].get("decision") in TAIL_DECISIONS
        and rows[-1].get("decision") == first.get("kind")
        and int(rows[-1].get("row_index", -1)) == 33
        and int(rows[-1].get("token_start", -1)) == 297
    ) if tail else len(rows) == 33
    debt = {
        "parent_identity": len(parent) != 33 or len(parent_owners) != 33,
        "token_shape": (
            len(tokens) != expected_rows * base.ROW_TOKENS + 1
            or not tokens or tokens[-1] != base.EOS or base.EOS in tokens[:-1]
            or len(tokens) >= base.NATURAL_MAX_TOKENS
        ),
        "parser_shape": (
            int(parser.get("valid_prediction_count", -1)) != expected_rows
            or int(parser.get("dropped_prediction_count", -1)) != 0
        ),
        "strict_global_match": statuses != expected_statuses,
        "global_owner_set": (
            committed != parent
            or int(matcher.get("committed_owner_count", -1)) != 33
            or set(map(str, evaluation.get("matched_target_owner_ids", ()))) != parent
        ),
        "parent_exchange": set(backbone_owners) != parent or len(backbone_owners) != len(set(backbone_owners)),
        "chronological_backbone": (
            len(backbone) != 33
            or any(
                row.get("parse_state") != "valid"
                or row.get("decision") != "accepted"
                or int(row.get("row_index", -1)) != index
                or int(row.get("token_start", -1)) != index * base.ROW_TOKENS
                for index, row in enumerate(backbone)
            )
            or list(map(str, causal_ledger.get("locked_owner_ids", ()))) != backbone_owners
        ),
        "tail_structure": not tail_ok,
        "event_structure": not event_ok,
        "segmentation": not bool(causal_ledger.get("segmentation_trusted")),
        "causal_debt": int(causal_ledger.get("causal_hard_counter_count", -1)) != int(tail),
        "row_aligned_eos": not bool(joint.get("natural_row_aligned_eos")),
        "forbidden_global_debt": _counter_debt(dict(joint.get("hard_raw_counters", {})), tail=tail),
    }
    active = {key: value for key, value in debt.items() if value}
    return {
        "passed": not active,
        "kind": "single_unassigned_tail" if tail else "clean_parent33",
        "debt": active,
    }


def _frozen_pair(tokens: Sequence[int], alias_tokens: Sequence[int]) -> dict[str, Any]:
    natural = list(map(int, tokens))
    alias = list(map(int, alias_tokens))
    boundary = 33 * base.ROW_TOKENS
    target_suffix = [*alias, base.EOS]
    if len(alias) != base.ROW_TOKENS or base.EOS in alias or len(natural) <= boundary:
        raise ParallelCorridorHold("HOLD: malformed dynamic suffix pair input")
    actual_suffix = natural[boundary:]
    lcp = base._exact_prefix(actual_suffix, target_suffix)
    if lcp >= min(len(actual_suffix), len(target_suffix)):
        raise ParallelCorridorHold("HOLD: corridor state has no divergent alias pair")
    teacher = [*natural[:boundary], *target_suffix]
    position = boundary + lcp
    good, bad = target_suffix[lcp], actual_suffix[lcp]
    if teacher[:position] != natural[:position] or good == bad:
        raise ParallelCorridorHold("HOLD: frozen pair is not on the real natural prefix")
    return {
        "boundary": boundary, "lcp": lcp, "pair_position": position,
        "good_token_id": good, "bad_token_id": bad,
        "actual_suffix": actual_suffix, "actual_suffix_sha256": token_ids_sha256(actual_suffix),
        "target_suffix": target_suffix, "target_suffix_sha256": token_ids_sha256(target_suffix),
        "teacher_tokens": teacher, "teacher_tokens_sha256": token_ids_sha256(teacher),
        "real_prefix_sha256": token_ids_sha256(natural[:position]),
    }


def _pairwise_loss(logits: torch.Tensor, pair: Mapping[str, Any]) -> tuple[torch.Tensor, float]:
    position = int(pair["pair_position"])
    good, bad = int(pair["good_token_id"]), int(pair["bad_token_id"])
    if logits.ndim != 2 or not (0 <= position < logits.shape[0]) or good == bad:
        raise ParallelCorridorHold("HOLD: malformed frozen-pair logits")
    selected = logits[position]
    if max(good, bad) >= selected.shape[0] or min(good, bad) < 0 or not bool(torch.isfinite(selected[[good, bad]]).all().item()):
        raise ParallelCorridorHold("HOLD: non-finite/out-of-vocabulary frozen pair")
    margin = selected[good] - selected[bad]
    loss = F.softplus(-margin)
    return loss, float(margin.detach().item())


def _radius_for_rank(rank: int) -> float:
    if rank not in range(WORLD_SIZE):
        raise ParallelCorridorHold("HOLD: radius rank is outside world8")
    return RADII[rank]


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    by_rank = {int(item["rank"]): item for item in candidates}
    if set(by_rank) != set(range(WORLD_SIZE)) or tuple(float(by_rank[index]["radius"]) for index in range(WORLD_SIZE)) != RADII:
        raise ParallelCorridorHold("HOLD: candidate radius panel is incomplete")
    promotions = [item for item in candidates if bool(item.get("promotion"))]
    if promotions:
        return max(promotions, key=lambda item: float(item["radius"]))
    eligible = [
        item for item in candidates
        if bool(item.get("margin_improved"))
        and bool(item.get("corridor"))
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (bool(item.get("promotion")), float(item["radius"])))


def _clone_parameters(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _restore_parameters(parameters: Sequence[torch.nn.Parameter], clones: Sequence[torch.Tensor]) -> None:
    if len(parameters) != len(clones):
        raise ParallelCorridorHold("HOLD: clone restore surface length drifted")
    with torch.no_grad():
        for parameter, clone in zip(parameters, clones, strict=True):
            if parameter.shape != clone.shape:
                raise ParallelCorridorHold("HOLD: clone restore surface shape drifted")
            parameter.copy_(clone)


def _apply_radius(
    parameters: Sequence[torch.nn.Parameter], clones: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor], *, radius: float,
) -> None:
    if radius not in RADII or len(parameters) != len(clones) or len(parameters) != len(direction):
        raise ParallelCorridorHold("HOLD: malformed parallel radius application")
    with torch.no_grad():
        for parameter, clone, gradient in zip(parameters, clones, direction, strict=True):
            if parameter.shape != clone.shape or parameter.shape != gradient.shape:
                raise ParallelCorridorHold("HOLD: radius direction surface drifted")
            parameter.copy_(clone - (base.LEARNING_RATE * radius) * gradient)


def _require_world_size(value: int) -> None:
    if value != WORLD_SIZE:
        raise ParallelCorridorHold("HOLD: requires torchrun --nproc_per_node=8")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ParallelCorridorHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise ParallelCorridorHold("HOLD: runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _state_packet(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    target: Mapping[str, Any], raw_example: Any, parent_owners: Sequence[str],
    alias_tokens: Sequence[int], label: str,
) -> dict[str, Any]:
    tokens = base._natural_tokens(model=model, native_inputs=native_inputs, pad=pad)
    witness = [*base._parent_tokens()[:-1], *map(int, alias_tokens), base.EOS]
    evaluation = manifold._match_evaluation(
        tokenizer=tokenizer, tokens=tokens, target=target, witness_tokens=witness,
        raw_example=raw_example, parent_owners=parent_owners, label=label,
    )
    causal = base._causal_locked_ledger(tokenizer=tokenizer, tokens=tokens, raw_example=raw_example)
    return {
        "evaluation": evaluation,
        "causal_ledger": causal,
        "promotion_gate": _promotion_gate(evaluation, parent_owners),
        "corridor_gate": _corridor_gate(evaluation, causal, parent_owners),
    }


def _state_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(packet["evaluation"])
    causal = dict(packet["causal_ledger"])
    return {
        "generated_token_count": len(evaluation["generated_token_ids"]),
        "generated_token_ids_sha256": evaluation["generated_token_ids_sha256"],
        "matched_target_owner_ids": evaluation["matched_target_owner_ids"],
        "promotion_gate": packet["promotion_gate"],
        "corridor_gate": packet["corridor_gate"],
        "causal": {
            "locked_owner_ids": causal.get("locked_owner_ids"),
            "first_event": causal.get("first_event"),
            "causal_hard_counter_count": causal.get("causal_hard_counter_count"),
        },
    }


def _state_identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation": base._evaluation_identity(packet["evaluation"]),
        "causal": base._causal_identity(packet["causal_ledger"]),
        "promotion_gate": packet["promotion_gate"],
        "corridor_gate": packet["corridor_gate"],
    }


def _cold_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], parent_owners: Sequence[str],
    alias_tokens: Sequence[int], label: str,
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(checkpoint)
    with open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise ParallelCorridorHold("HOLD: cold evaluation requires concrete FP32 HF backend")
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
            raise ParallelCorridorHold("HOLD: cold prompt identity drifted")
        packet = _state_packet(
            model=opened._model, tokenizer=opened._tokenizer, native_inputs=native_inputs,
            pad=int(opened._tokenizer.pad_token_id), target=target,
            raw_example=setup["raw_example"], parent_owners=parent_owners,
            alias_tokens=alias_tokens, label=label,
        )
        return {"packet": packet, "runtime": opened.receipt.to_artifact_dict()}


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
            raise ParallelCorridorHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        trigger = base._rank0_call(_trigger)
        _admission, setup, target = base._bindings()
        parent_tokens = base._parent_tokens()

        def admit_before_model_load() -> dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(
                setup["frontend"].launch.model_path, trust_remote_code=True,
            )
            try:
                return _admit_alias(tokenizer=tokenizer, raw_example=setup["raw_example"])
            finally:
                del tokenizer

        alias = base._rank0_call(admit_before_model_load)
        attempts: list[dict[str, Any]] = []
        accepted_steps = 0
        final_warm: Mapping[str, Any] | None = None
        saved: Mapping[str, Any] | None = None
        stop_reason = "max_50_accepted_steps_without_warm_promotion"
        runtime: Mapping[str, Any] | None = None
        initial_surface: Mapping[str, Any] | None = None
        terminal_surface: Mapping[str, Any] | None = None
        frozen_before: str | None = None
        frozen_after: str | None = None
        parent_owners: tuple[str, ...] = ()

        with open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise ParallelCorridorHold("HOLD: requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
                raise ParallelCorridorHold("HOLD: live prompt identity drifted")
            parent_ledger = base._parse_and_match(
                tokenizer=tokenizer, generated_token_ids=parent_tokens,
                label="parallel-corridor-parent33", raw_example=setup["raw_example"],
            )
            parent_owners = base._strict_owner_order(parent_ledger)
            base._assert_missing_membership(parent_owners)
            if set(parent_owners) != set(trigger["parent_owner_ids"]):
                raise ParallelCorridorHold("HOLD: trigger/live Parent owner identity drifted")

            names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
            if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                raise ParallelCorridorHold("HOLD: exact 588/18006016 FP32 DoRA surface drifted")
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            _unused, initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)
            base._surface_agreement(initial_surface, dist.group.WORLD)
            frozen_before = base._frozen_surface(model, names)
            frozen_identities: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(frozen_identities, frozen_before)
            if len(set(frozen_identities)) != 1:
                raise ParallelCorridorHold("HOLD: initial frozen surface rank disagreement")

            for attempt in range(1, MAX_ACCEPTED_STEPS + 1):
                current = base._rank0_call(lambda: _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target,
                    raw_example=setup["raw_example"], parent_owners=parent_owners,
                    alias_tokens=alias["token_ids"], label=f"parallel-current-{attempt:02d}",
                ))
                if current["promotion_gate"]["passed"]:
                    final_warm = current
                    stop_reason = "current_match_level_promotion"
                    break
                if not current["corridor_gate"]["passed"]:
                    raise ParallelCorridorHold("HOLD: current accepted policy left the match corridor")
                pair = _frozen_pair(current["evaluation"]["generated_token_ids"], alias["token_ids"])
                pair_identities: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(pair_identities, {
                    "teacher_tokens_sha256": pair["teacher_tokens_sha256"],
                    "real_prefix_sha256": pair["real_prefix_sha256"],
                    "pair_position": pair["pair_position"],
                    "good_token_id": pair["good_token_id"],
                    "bad_token_id": pair["bad_token_id"],
                })
                if len({base._hash(identity) for identity in pair_identities}) != 1:
                    raise ParallelCorridorHold("HOLD: rank frozen-pair identity disagreement")
                current_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]

                for parameter in parameters:
                    parameter.grad = None
                logits = base.full_root._teacher_forced_route_logits(
                    model=model, native_inputs=native_inputs, route_tokens=pair["teacher_tokens"],
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                loss, base_margin = _pairwise_loss(logits, pair)
                base_margins: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(base_margins, base_margin)
                if len(set(base_margins)) != 1:
                    raise ParallelCorridorHold("HOLD: rank base frozen-pair margin disagreement")
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                        raise ParallelCorridorHold("HOLD: disconnected/non-finite pairwise gradient")
                    parameter.grad = gradient.detach()
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(WORLD_SIZE)
                norm = torch.nn.utils.clip_grad_norm_(parameters, base.GRAD_CLIP)
                if not bool(torch.isfinite(norm).item()) or float(norm) <= 0.0:
                    raise ParallelCorridorHold("HOLD: zero/non-finite pairwise gradient")
                direction = [parameter.grad.detach().clone() for parameter in parameters]
                clones = _clone_parameters(parameters)
                del logits

                radius = _radius_for_rank(rank)
                _apply_radius(parameters, clones, direction, radius=radius)
                base.full_root._assert_full_root_sentinels(model, sentinels)
                with torch.inference_mode():
                    candidate_logits = base.full_root._teacher_forced_route_logits(
                        model=model, native_inputs=native_inputs, route_tokens=pair["teacher_tokens"],
                        pad_token_id=int(tokenizer.pad_token_id),
                    )
                    _candidate_loss, candidate_margin = _pairwise_loss(candidate_logits, pair)
                candidate_packet = _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target,
                    raw_example=setup["raw_example"], parent_owners=parent_owners,
                    alias_tokens=alias["token_ids"], label=f"parallel-candidate-{attempt:02d}-rank-{rank}",
                )
                candidate_summary = {
                    "rank": rank, "radius": radius,
                    "base_margin": base_margin, "candidate_margin": candidate_margin,
                    "margin_improved": math.isfinite(candidate_margin) and candidate_margin > base_margin,
                    "promotion": bool(candidate_packet["promotion_gate"]["passed"]),
                    "corridor": bool(candidate_packet["corridor_gate"]["passed"]),
                    "state": _state_summary(candidate_packet),
                }
                candidates: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(candidates, candidate_summary)
                selected = _select_candidate(candidates)
                record: dict[str, Any] = {
                    "attempt": attempt,
                    "accepted_steps_before": accepted_steps,
                    "frozen_state": _state_summary(current),
                    "frozen_surface": current_surface,
                    "frozen_pair": {key: value for key, value in pair.items() if key != "teacher_tokens"},
                    "base_pairwise_loss": float(loss.detach().item()),
                    "base_frozen_pair_margin": base_margin,
                    "gradient_norm_pre_clip": float(norm),
                    "gradient_reduction": "same loss on every rank; SUM all-reduce then divide 8; global clip 1",
                    "candidate_panel": candidates,
                    "selected": None,
                }
                if selected is None:
                    _restore_parameters(parameters, clones)
                    restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                    if restored != current_surface:
                        raise ParallelCorridorHold("HOLD: exact clone restore failed")
                    base._surface_agreement(restored, dist.group.WORLD)
                    record["decision"] = "no_eligible_radius_restore_current"
                    attempts.append(record)
                    final_warm = current
                    stop_reason = "no_feasible_margin_improving_radius"
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
                record["selected"] = {
                    "rank": selected_rank, "radius": float(selected["radius"]),
                    "class": "promotion" if selected["promotion"] else "corridor",
                    "candidate_margin": float(selected["candidate_margin"]),
                    "surface": selected_surface,
                    "state": _state_summary(selected_packet),
                }
                record["decision"] = "accept_largest_radius_in_priority_class"
                attempts.append(record)
                accepted_steps += 1
                if selected_packet["promotion_gate"]["passed"]:
                    final_warm = selected_packet
                    stop_reason = "first_warm_match_level_promotion"
                    break

            if final_warm is None:
                final_warm = base._rank0_call(lambda: _state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target,
                    raw_example=setup["raw_example"], parent_owners=parent_owners,
                    alias_tokens=alias["token_ids"], label="parallel-final-step-50",
                ))
            promoted = bool(final_warm["promotion_gate"]["passed"])
            if not promoted and not final_warm["corridor_gate"]["passed"]:
                raise ParallelCorridorHold("HOLD: terminal policy is neither promotion nor corridor-feasible")
            if accepted_steps == 0:
                saved = {
                    "checkpoint": str(base.PARENT_CHECKPOINT),
                    "readback": {"reused_immutable_parent": True},
                }
            else:
                checkpoint = output / (
                    f"checkpoint-promotion-step-{accepted_steps:02d}"
                    if promoted else f"checkpoint-terminal-step-{accepted_steps:02d}"
                )
                saved = base._rank0_call(lambda: {
                    "checkpoint": str(checkpoint),
                    "readback": base.full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=base.PARENT_CHECKPOINT,
                        destination=checkpoint,
                    ),
                })
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = base._frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise ParallelCorridorHold("HOLD: frozen non-DoRA surface mutated")
            runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, opened
        torch.cuda.empty_cache()
        dist.barrier()
        assert saved is not None and final_warm is not None
        cold = base._rank0_call(lambda: _cold_evaluate(
            Path(saved["checkpoint"]), target=target, parent_owners=parent_owners,
            alias_tokens=alias["token_ids"], label="parallel-corridor-cold",
        ))
        if _state_identity(final_warm) != _state_identity(cold["packet"]):
            raise ParallelCorridorHold("HOLD: warm/cold natural policy identity mismatch")
        promoted = bool(final_warm["promotion_gate"]["passed"])
        status = "cold_match_level_promotion" if promoted else "bounded_negative_no_match_level_promotion"
        receipt = {
            "schema_version": SCHEMA_VERSION, "status": status, "run_id": run_id,
            "runner_source_snapshot": source_snapshot, "trigger": trigger,
            "bindings": {
                "parent_checkpoint": str(base.PARENT_CHECKPOINT),
                "parent_route_sha256": base.PARENT_ROUTE_SHA256,
                "prompt_sha256": base.PROMPT_TOKEN_SHA256,
                "target": str(base.TARGET_PATH), "target_sha256": base.TARGET_SHA256,
                "authority": str(base.AUTHORITY_PATH), "authority_sha256": base.AUTHORITY_SHA256,
            },
            "alias": alias,
            "protocol": {
                "world_size": WORLD_SIZE, "accepted_step_budget": MAX_ACCEPTED_STEPS,
                "radii_by_rank": {str(rank): RADII[rank] for rank in range(WORLD_SIZE)},
                "base_learning_rate": base.LEARNING_RATE, "gradient_clip": base.GRAD_CLIP,
                "surface": "588 FP32 DoRA tensors / 18006016 elements only",
                "frozen": "all non-DoRA including tied delta, embeddings, aligner, vision, wrappers",
                "update": "one manual -LR*radius*globally-clipped-gradient step; no optimizer state",
                "selection": "promotion before corridor; largest eligible radius within class",
                "acceptance": "match-level owner superset only; alias token identity is never acceptance",
            },
            "parent_owner_ids": list(parent_owners),
            "accepted_step_count": accepted_steps, "attempts": attempts,
            "initial_surface": initial_surface, "terminal_surface": terminal_surface,
            "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
            "saved_checkpoint": saved,
            "warm_final": final_warm, "cold": cold,
            "warm_cold_identity": _state_identity(final_warm),
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
