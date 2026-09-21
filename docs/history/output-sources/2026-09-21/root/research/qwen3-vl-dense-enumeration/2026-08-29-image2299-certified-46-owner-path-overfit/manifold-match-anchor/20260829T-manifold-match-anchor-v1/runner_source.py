#!/usr/bin/env python3
"""Add one evaluator-matched owner to immutable Parent33 with a KL anchor."""

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
from src.inference.backend import open_backend_session, token_ids_sha256
from src.inference.hf_backend import HFBackendSession


SCHEMA_VERSION = "image2299.manifold_match_anchor_overfit.v1"
OUTPUT_ROOT = base.OUTPUT_ROOT / "manifold-match-anchor"
TRIGGER_RECEIPT = (
    base.OUTPUT_ROOT / "tokenwise-dora50"
    / "20260829T-certified46-tokenwise-dora50-v2" / "receipt.json"
)
TRIGGER_SHA256 = "6dedd7f932285b7657858ff8f9e41b2e0c85fe33f4815b42b41c906bd8bdf51a"
TRIGGER_STATUS = "tokenwise_dora50_no_promotion_tied_row_gate_open"
TARGET_OWNER = "gt:2299:32"
WORLD_SIZE = 8
MAX_UPDATES = 50
KL_POSITIONS = tuple(range(297))
ALIAS_POSITIONS = tuple(range(297, 306))
SUFFIX_POSITIONS = tuple(range(297, 307))
CE_RANK = 0
LOSS_MASS = 0.5

ALIAS_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "A", "role": "optimization_candidate", "optimization_eligible": True,
        "token_ids": [151646, 8987, 151647, 151648, 152399, 152078, 152472, 152446, 151649],
        "row_sha256": "b814b97253384b7bcfed5883b5b89f204b07259a6018aeb704fbceb592364427",
        "expected_iou": 0.6621046038208581,
        "provenance": "exact predecessor v2 warm+cold",
    },
    {
        "name": "B", "role": "optimization_candidate", "optimization_eligible": True,
        "token_ids": [151646, 8987, 151647, 151648, 152399, 152041, 152472, 152441, 151649],
        "row_sha256": "020d1b8e94cf6eda58a2d4b733997380001b8d0d29b87aeb4cf570aa931b4ba7",
        "expected_iou": 0.6068919273490568, "observed_frequency": 21,
        "provenance": "broad rollout rank1",
    },
    {
        "name": "canonical", "role": "diagnostic_negative_control", "optimization_eligible": False,
        "token_ids": [151646, 8987, 151647, 151648, 152399, 152094, 152468, 152460, 151649],
        "row_sha256": "de0f919d33a6289a012bae26e06a1639bdc7152304264b57e0c0572465c76ec2",
        "expected_iou": 0.7321818609496287,
        "provenance": "canonical gt32 row; never an optimization target",
    },
)

ManifoldAnchorHold = base.Certified46Hold


def _trigger() -> dict[str, Any]:
    if not TRIGGER_RECEIPT.is_file() or base._sha256(TRIGGER_RECEIPT) != TRIGGER_SHA256:
        raise ManifoldAnchorHold("HOLD: manifold trigger receipt identity drifted")
    receipt = json.loads(TRIGGER_RECEIPT.read_text(encoding="utf-8"))
    parent_owners = tuple(map(str, receipt.get("final_champion_owner_ids", ())))
    if (
        receipt.get("status") != TRIGGER_STATUS
        or receipt.get("stop_reason") != TRIGGER_STATUS
        or receipt.get("final_champion_checkpoint") != str(base.PARENT_CHECKPOINT)
        or len(parent_owners) != 33
        or len(set(parent_owners)) != 33
    ):
        raise ManifoldAnchorHold("HOLD: trigger does not bind immutable Parent33")
    return {
        "path": str(TRIGGER_RECEIPT), "sha256": TRIGGER_SHA256,
        "status": TRIGGER_STATUS, "champion_checkpoint": str(base.PARENT_CHECKPOINT),
        "champion_owner_ids": sorted(parent_owners),
    }


def _admit_alias_candidate(
    *, tokenizer: Any, raw_example: Any, candidate: Mapping[str, Any],
) -> dict[str, Any]:
    tokens = list(map(int, candidate.get("token_ids", ())))
    row_hash = token_ids_sha256(tokens)
    if len(tokens) != base.ROW_TOKENS or row_hash != candidate.get("row_sha256"):
        raise ManifoldAnchorHold("HOLD: alias row identity drifted")
    prediction, parse_state = base._parse_compact_row(
        tokenizer=tokenizer, row_tokens=tokens, raw_example=raw_example, row_index=33,
    )
    if prediction is None or parse_state != "valid":
        raise ManifoldAnchorHold(f"HOLD: alias {candidate.get('name')} is not parser-clean")
    _line, owners, _binding = base._load_authority()
    normalized = base.canonical_matcher._normalize_description(prediction.get("description"))
    box = tuple(float(value) for value in prediction["bbox"])
    strict_edges = [
        {"owner": str(owner["gt_owner_id"]), "iou": base.canonical_matcher._iou(box, owner["bbox_xyxy"])}
        for owner in owners
        if base.canonical_matcher._compatible_description(
            {"normalized_description": normalized}, owner, {},
        )
        and base.canonical_matcher._iou(box, owner["bbox_xyxy"]) >= base.canonical_matcher.IOU_THRESHOLD
    ]
    if len(strict_edges) != 1 or strict_edges[0]["owner"] != TARGET_OWNER:
        raise ManifoldAnchorHold(f"HOLD: alias {candidate.get('name')} is not strict-unique gt32")
    iou = float(strict_edges[0]["iou"])
    if not math.isclose(iou, float(candidate["expected_iou"]), abs_tol=1e-12, rel_tol=0.0):
        raise ManifoldAnchorHold(f"HOLD: alias {candidate.get('name')} IoU drifted")
    return {
        **dict(candidate), "token_ids": tokens, "row_sha256": row_hash,
        "parse_state": parse_state, "prediction": prediction,
        "strict_edges": strict_edges, "admitted": True,
    }


def _admit_alias_shortlist(*, tokenizer: Any, raw_example: Any) -> list[dict[str, Any]]:
    admitted = [
        _admit_alias_candidate(tokenizer=tokenizer, raw_example=raw_example, candidate=candidate)
        for candidate in ALIAS_CANDIDATES
    ]
    if sum(bool(item["optimization_eligible"]) for item in admitted) != 2:
        raise ManifoldAnchorHold("HOLD: alias optimization shortlist is not exactly A/B")
    return admitted


def _compile_witness(parent_tokens: Sequence[int], alias_tokens: Sequence[int]) -> list[int]:
    parent = list(map(int, parent_tokens))
    alias = list(map(int, alias_tokens))
    if len(parent) != 298 or parent[-1:] != [base.EOS] or len(alias) != base.ROW_TOKENS or base.EOS in alias:
        raise ManifoldAnchorHold("HOLD: malformed Parent33/alias witness input")
    witness = [*parent[:-1], *alias, base.EOS]
    if witness[:297] != parent[:297] or len(witness) != 307 or witness[-1] != base.EOS:
        raise ManifoldAnchorHold("HOLD: suffix witness changed a Parent prefix")
    return witness


def _target_margins(logits: torch.Tensor, tokens: Sequence[int], positions: Sequence[int]) -> list[float]:
    if logits.ndim != 2 or logits.shape[0] != len(tokens) or not bool(torch.isfinite(logits).all().item()):
        raise ManifoldAnchorHold("HOLD: teacher logits are malformed")
    records: list[float] = []
    for position in positions:
        target = int(tokens[position])
        values = logits[position]
        competitor = values.clone()
        competitor[target] = -torch.inf
        records.append(float((values[target] - competitor.max()).item()))
    return records


def _candidate_score(logits: torch.Tensor, witness_tokens: Sequence[int], row_hash: str) -> dict[str, Any]:
    margins = _target_margins(logits, witness_tokens, ALIAS_POSITIONS)
    positions = torch.tensor(ALIAS_POSITIONS, dtype=torch.long, device=logits.device)
    targets = torch.tensor([witness_tokens[position] for position in ALIAS_POSITIONS], dtype=torch.long, device=logits.device)
    mean_ce = F.cross_entropy(logits.index_select(0, positions), targets, reduction="mean")
    return {
        "row_sha256": row_hash, "teacher_mean_ce": float(mean_ce.item()),
        "minimum_margin": min(margins), "per_alias_margin": margins,
    }


def _select_alias(scores: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [item for item in scores if bool(item.get("optimization_eligible")) and bool(item.get("admitted"))]
    if {str(item.get("name")) for item in eligible} != {"A", "B"}:
        raise ManifoldAnchorHold("HOLD: admitted alias selection requires exactly A and B")
    return min(
        eligible,
        key=lambda item: (
            float(item["teacher_mean_ce"]), -float(item["minimum_margin"]), str(item["row_sha256"]),
        ),
    )


def _kl_positions(*, rank: int, world_size: int = WORLD_SIZE) -> tuple[int, ...]:
    if world_size != WORLD_SIZE or rank not in range(world_size):
        raise ManifoldAnchorHold("HOLD: KL partition requires exactly eight ranks")
    return tuple(position for position in KL_POSITIONS if position % world_size == rank)


def _suffix_frontier(logits: torch.Tensor, witness_tokens: Sequence[int]) -> dict[str, Any]:
    margins = _target_margins(logits, witness_tokens, SUFFIX_POSITIONS)
    relative = next((index for index, margin in enumerate(margins) if margin <= 0.0), None)
    rule = "first_non_positive_margin"
    if relative is None:
        relative = min(range(len(margins)), key=lambda index: (margins[index], index))
        rule = "weakest_positive_suffix_margin"
    return {
        "position": SUFFIX_POSITIONS[relative], "relative_suffix_position": relative,
        "margin": margins[relative], "per_suffix_margin": margins, "rule": rule,
        "target_token_id": int(witness_tokens[SUFFIX_POSITIONS[relative]]),
    }


def _cache_reference_log_probs(logits: torch.Tensor, *, rank: int) -> torch.Tensor:
    positions = _kl_positions(rank=rank)
    index = torch.tensor(positions, dtype=torch.long, device=logits.device)
    return F.log_softmax(logits.index_select(0, index).float(), dim=-1).detach()


def _equal_mass_loss(
    logits: torch.Tensor, witness_tokens: Sequence[int], reference_log_probs: torch.Tensor,
    *, frontier_position: int, rank: int, world_size: int = WORLD_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    positions = _kl_positions(rank=rank, world_size=world_size)
    if frontier_position not in SUFFIX_POSITIONS or logits.shape[0] != len(witness_tokens):
        raise ManifoldAnchorHold("HOLD: equal-mass loss inputs are malformed")
    index = torch.tensor(positions, dtype=torch.long, device=logits.device)
    current_log_probs = F.log_softmax(logits.index_select(0, index).float(), dim=-1)
    if reference_log_probs.shape != current_log_probs.shape or not bool(torch.isfinite(reference_log_probs).all().item()):
        raise ManifoldAnchorHold("HOLD: cached Parent KL distribution drifted")
    local_kl_sum = (
        reference_log_probs.exp() * (reference_log_probs - current_log_probs)
    ).sum()
    kl_term = LOSS_MASS * local_kl_sum / len(KL_POSITIONS)
    if rank == CE_RANK:
        target = torch.tensor([int(witness_tokens[frontier_position])], dtype=torch.long, device=logits.device)
        ce_unweighted = F.cross_entropy(logits[frontier_position:frontier_position + 1], target)
        ce_term = LOSS_MASS * ce_unweighted
        owns_ce = True
    else:
        ce_unweighted = logits.sum() * 0.0
        ce_term = ce_unweighted
        owns_ce = False
    loss = kl_term + ce_term
    if not bool(torch.isfinite(loss).item()):
        raise ManifoldAnchorHold("HOLD: non-finite equal-mass loss")
    return loss, {
        "rank": rank, "kl_positions": list(positions), "kl_position_count": len(positions),
        "kl_contribution": float(kl_term.detach()), "owns_target_ce": owns_ce,
        "target_ce_contribution": float(ce_term.detach()), "frontier_position": frontier_position,
        "normalization": {
            "kl": "0.5 * local KL sum / 297, then SUM gradient all-reduce",
            "target_ce": "0.5 * one-token CE on rank0 only, then SUM gradient all-reduce",
        },
    }


def _promotion_gate(evaluation: Mapping[str, Any], parent_owners: Sequence[str]) -> dict[str, Any]:
    parent = set(map(str, parent_owners))
    owners = set(map(str, evaluation.get("matched_target_owner_ids", ())))
    joint = dict(evaluation.get("joint_gate", {}))
    parser = dict(joint.get("parser", {}))
    matcher = dict(joint.get("matcher", {}))
    statuses = dict(matcher.get("strict_status_counts", {}))
    hard = dict(joint.get("hard_raw_counters", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    debts = {
        "not_a_proper_owner_superset": not owners > parent,
        "parent_owner_or_tie_loss": not parent.issubset(owners),
        "malformed": int(parser.get("dropped_prediction_count", -1)) != 0,
        "strict_match_debt": statuses != {"matched": len(owners)},
        "owner_cardinality_drift": int(matcher.get("committed_owner_count", -1)) != len(owners),
        "evaluator_hard_counter": int(evaluation.get("hard_counter_count", -1)) != 0 or any(hard.values()),
        "non_row_aligned_eos": not bool(joint.get("natural_row_aligned_eos")),
        "token_budget_debt": len(tokens) >= base.NATURAL_MAX_TOKENS,
    }
    return {
        "passed": not any(debts.values()), "debt": debts,
        "matched_owner_ids": sorted(owners), "added_owner_ids": sorted(owners - parent),
        "gt32_appeared": TARGET_OWNER in owners,
    }


def _match_evaluation(
    *, tokenizer: Any, tokens: Sequence[int], target: Mapping[str, Any],
    witness_tokens: Sequence[int], raw_example: Any, parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    """Global evaluator receipt without requiring the canonical 46-row token path."""
    ledger = base._parse_and_match(
        tokenizer=tokenizer, generated_token_ids=tokens, label=label, raw_example=raw_example,
    )
    receipts = list(ledger["matcher"].get("prediction_receipts", ()))
    owners = {
        str(row.get("strict_match_gt_owner_id"))
        for row in receipts
        if row.get("strict_match_status") == "matched" and row.get("strict_match_gt_owner_id")
    }
    gate = base.full_root._full_root_joint_owner_equivalence_from_ledger(
        tokens=tokens, ledger=ledger, target=target,
    )
    hard = sum(int(value) for value in gate["hard_raw_counters"].values())
    hard += int(not gate["natural_row_aligned_eos"])
    return {
        "generated_token_ids": list(map(int, tokens)),
        "generated_token_ids_sha256": token_ids_sha256(tokens),
        "exact_target_prefix": base._exact_prefix(tokens, witness_tokens),
        "matched_target_owner_count": len(owners),
        "matched_target_owner_ids": sorted(owners),
        "all_parent_owners_retained": set(map(str, parent_owners)).issubset(owners),
        "hard_counter_count": hard, "joint_gate": gate, "ledger": ledger,
    }


def _require_world_size(value: int) -> None:
    if value != WORLD_SIZE:
        raise ManifoldAnchorHold("HOLD: requires torchrun --nproc_per_node=8")


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ManifoldAnchorHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise ManifoldAnchorHold("HOLD: runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _score_aliases(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    parent_tokens: Sequence[int], admissions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    with torch.inference_mode():
        for admission in admissions:
            if not admission["optimization_eligible"]:
                scores.append(dict(admission))
                continue
            witness = _compile_witness(parent_tokens, admission["token_ids"])
            logits = base.full_root._teacher_forced_route_logits(
                model=model, native_inputs=native_inputs, route_tokens=witness, pad_token_id=pad,
            )
            scores.append({**dict(admission), **_candidate_score(logits, witness, str(admission["row_sha256"]))})
    return scores


def _warm_evaluate(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    target: Mapping[str, Any], witness: Sequence[int], raw_example: Any,
    parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    tokens = base._natural_tokens(model=model, native_inputs=native_inputs, pad=pad)
    evaluation = _match_evaluation(
        tokenizer=tokenizer, tokens=tokens, target=target, witness_tokens=witness,
        raw_example=raw_example, parent_owners=parent_owners, label=label,
    )
    return {"evaluation": evaluation, "promotion_gate": _promotion_gate(evaluation, parent_owners)}


def _cold_evaluate(
    checkpoint: Path, *, target: Mapping[str, Any], witness: Sequence[int],
    parent_owners: Sequence[str], label: str,
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(checkpoint)
    with open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise ManifoldAnchorHold("HOLD: cold evaluation requires concrete FP32 HF backend")
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
            raise ManifoldAnchorHold("HOLD: cold prompt identity drifted")
        tokens = base._natural_tokens(
            model=opened._model, native_inputs=native_inputs, pad=int(opened._tokenizer.pad_token_id),
        )
        evaluation = _match_evaluation(
            tokenizer=opened._tokenizer, tokens=tokens, target=target,
            witness_tokens=witness, raw_example=setup["raw_example"],
            parent_owners=parent_owners, label=label,
        )
        return {**evaluation, "runtime": opened.receipt.to_artifact_dict()}


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
            raise ManifoldAnchorHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        trigger = base._rank0_call(_trigger)
        _admission, setup, target = base._bindings()
        parent_tokens = base._parent_tokens()

        def admit_before_model_load() -> list[dict[str, Any]]:
            tokenizer = AutoTokenizer.from_pretrained(
                setup["frontend"].launch.model_path, trust_remote_code=True,
            )
            try:
                return _admit_alias_shortlist(tokenizer=tokenizer, raw_example=setup["raw_example"])
            finally:
                del tokenizer

        alias_admissions = base._rank0_call(admit_before_model_load)
        update_records: list[dict[str, Any]] = []
        saved: dict[str, Any] | None = None
        final_warm: dict[str, Any] | None = None
        selected: Mapping[str, Any] | None = None
        parent_owners: tuple[str, ...] = ()
        initial_surface: Mapping[str, Any] | None = None
        terminal_surface: Mapping[str, Any] | None = None
        frozen_before: str | None = None
        frozen_after: str | None = None
        reference_receipts: list[Any] = [None] * WORLD_SIZE
        runtime: Mapping[str, Any] | None = None
        stop_reason = "max_50_updates_without_warm_promotion"

        with open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise ManifoldAnchorHold("HOLD: requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if token_ids_sha256(prompts[0]) != base.PROMPT_TOKEN_SHA256:
                raise ManifoldAnchorHold("HOLD: live prompt identity drifted")
            parent_ledger = base._parse_and_match(
                tokenizer=tokenizer, generated_token_ids=parent_tokens,
                label="manifold-parent33", raw_example=setup["raw_example"],
            )
            parent_owners = base._strict_owner_order(parent_ledger)
            base._assert_missing_membership(parent_owners)
            if set(parent_owners) != set(trigger["champion_owner_ids"]):
                raise ManifoldAnchorHold("HOLD: trigger/live Parent owner identity drifted")

            alias_scores = base._rank0_call(lambda: _score_aliases(
                model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                parent_tokens=parent_tokens, admissions=alias_admissions,
            ))
            selected = dict(_select_alias(alias_scores))
            selected_packet = [selected]
            dist.broadcast_object_list(selected_packet, src=0)
            selected = dict(selected_packet[0])
            witness = _compile_witness(parent_tokens, selected["token_ids"])

            with torch.inference_mode():
                reference_logits = base.full_root._teacher_forced_route_logits(
                    model=model, native_inputs=native_inputs, route_tokens=witness,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                reference_log_probs = _cache_reference_log_probs(reference_logits, rank=rank)
            reference_local = {
                "rank": rank, "positions": list(_kl_positions(rank=rank)),
                "log_probs_sha256": base.full_root._tensor_sha256(reference_log_probs),
            }
            dist.all_gather_object(reference_receipts, reference_local)
            del reference_logits

            names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
            if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                raise ManifoldAnchorHold("HOLD: exact 588/18006016 FP32 DoRA surface drifted")
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            _originals, initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)
            frozen_before = base._frozen_surface(model, names)
            optimizer = torch.optim.SGD(
                parameters, lr=base.LEARNING_RATE, momentum=0.0, weight_decay=0.0,
            )

            for step in range(1, MAX_UPDATES + 1):
                optimizer.zero_grad(set_to_none=True)
                logits = base.full_root._teacher_forced_route_logits(
                    model=model, native_inputs=native_inputs, route_tokens=witness,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                frontier = _suffix_frontier(logits.detach(), witness)
                frontier_positions: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(frontier_positions, int(frontier["position"]))
                if len(set(frontier_positions)) != 1:
                    raise ManifoldAnchorHold("HOLD: rank frontier position disagreement")
                loss, allocation = _equal_mass_loss(
                    logits, witness, reference_log_probs,
                    frontier_position=int(frontier["position"]), rank=rank,
                )
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                        raise ManifoldAnchorHold("HOLD: disconnected/non-finite manifold gradient")
                    parameter.grad = gradient.detach()
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                norm = torch.nn.utils.clip_grad_norm_(parameters, base.GRAD_CLIP)
                if not bool(torch.isfinite(norm).item()) or float(norm) <= 0.0:
                    raise ManifoldAnchorHold("HOLD: zero/non-finite manifold gradient")
                optimizer.step()
                surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                base._surface_agreement(surface, dist.group.WORLD)
                base.full_root._assert_full_root_sentinels(model, sentinels)

                allocations: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(allocations, allocation)
                final_warm = base._rank0_call(lambda: _warm_evaluate(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, witness=witness,
                    raw_example=setup["raw_example"], parent_owners=parent_owners,
                    label=f"manifold-warm-step-{step:02d}",
                ))
                update_records.append({
                    "step": step, "frontier": frontier, "loss_by_rank": allocations,
                    "global_equal_mass_loss": sum(
                        float(item["kl_contribution"]) + float(item["target_ce_contribution"])
                        for item in allocations
                    ),
                    "gradient_norm_pre_clip": float(norm),
                    "warm": final_warm,
                })
                stop_packet = [bool(final_warm["promotion_gate"]["passed"])]
                dist.broadcast_object_list(stop_packet, src=0)
                if stop_packet[0]:
                    stop_reason = "first_warm_evaluator_level_promotion_candidate"
                    saved = base._rank0_call(lambda: {
                        "checkpoint": str(output / f"checkpoint-candidate-step-{step:02d}"),
                        "readback": base.full_root._save_weights_only_checkpoint(
                            model=model, source_checkpoint=base.PARENT_CHECKPOINT,
                            destination=output / f"checkpoint-candidate-step-{step:02d}",
                        ),
                    })
                    break

            if saved is None:
                saved = base._rank0_call(lambda: {
                    "checkpoint": str(output / "checkpoint-terminal-step-50"),
                    "readback": base.full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=base.PARENT_CHECKPOINT,
                        destination=output / "checkpoint-terminal-step-50",
                    ),
                })
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = base._frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise ManifoldAnchorHold("HOLD: frozen non-DoRA surface mutated")
            if terminal_surface == initial_surface:
                raise ManifoldAnchorHold("HOLD: permitted DoRA surface did not change")
            runtime = opened.receipt.to_artifact_dict()

        del model, tokenizer, native_inputs, prompts, opened
        torch.cuda.empty_cache()
        dist.barrier()
        assert saved is not None and final_warm is not None and selected is not None
        witness = _compile_witness(parent_tokens, selected["token_ids"])
        cold = base._rank0_call(lambda: _cold_evaluate(
            Path(saved["checkpoint"]), target=target, witness=witness,
            parent_owners=parent_owners, label="manifold-cold",
        ))
        cold_gate = _promotion_gate(cold, parent_owners)
        if (
            base._evaluation_identity(final_warm["evaluation"]) != base._evaluation_identity(cold)
            or final_warm["promotion_gate"] != cold_gate
        ):
            raise ManifoldAnchorHold("HOLD: warm/cold evaluator identity mismatch")
        warm_promoted = bool(final_warm["promotion_gate"]["passed"])
        status = "cold_match_level_promotion" if warm_promoted else "bounded_negative_no_match_level_promotion"
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
            "alias_candidates": alias_scores,
            "selected_alias": {
                "name": selected["name"], "row_sha256": selected["row_sha256"],
                "token_ids": selected["token_ids"], "teacher_mean_ce": selected["teacher_mean_ce"],
                "minimum_margin": selected["minimum_margin"],
                "selection_rule": "lowest mean CE, then highest minimum margin, then stable row hash",
            },
            "witness": {
                "token_count": len(witness), "token_ids_sha256": token_ids_sha256(witness),
                "parent_prefix_sha256": token_ids_sha256(witness[:297]),
                "shape": "33 real Parent rows + selected natural gt32 alias + EOS",
            },
            "protocol": {
                "world_size": WORLD_SIZE, "max_updates": MAX_UPDATES,
                "surface": "588 FP32 DoRA tensors / 18006016 elements only",
                "frozen": "shared_embed_delta, embeddings, aligner, vision, wrappers, all non-DoRA",
                "optimizer": {"name": "SGD", "learning_rate": base.LEARNING_RATE, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": base.GRAD_CLIP},
                "loss": "equal mass: frontier one-token CE + Parent-row-token forward KL",
                "kl_mask": list(KL_POSITIONS), "kl_excluded_intervention_eos_position": 297,
                "kl_partition": {str(rank): list(_kl_positions(rank=rank)) for rank in range(WORLD_SIZE)},
                "gradient_reduction": "SUM all-reduce; target CE owned by rank0 only",
                "acceptance": "strict matched-owner proper superset with Parent retention and zero evaluator debt; token hash irrelevant",
            },
            "parent_owner_ids": list(parent_owners), "reference_distribution_by_rank": reference_receipts,
            "initial_surface": initial_surface, "terminal_surface": terminal_surface,
            "frozen_surface_before": frozen_before, "frozen_surface_after": frozen_after,
            "updates": update_records, "saved_checkpoint": saved,
            "warm_final": final_warm, "cold": cold, "cold_promotion_gate": cold_gate,
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
