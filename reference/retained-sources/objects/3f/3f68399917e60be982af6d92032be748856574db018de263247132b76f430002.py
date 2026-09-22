#!/usr/bin/env python3
"""One-relinearization, world8 Image2299 OTA-SQP-lite GPU smoke."""

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
from scipy.optimize import minimize

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_image2299_certified_46_owner_path_overfit as base
from scripts.research import run_image2299_manifold_match_anchor_overfit as manifold
from scripts.research import run_image2299_parallel_match_corridor as parallel
from scripts.research import run_image2299_projected_owner_corridor as projected
from src.inference.backend import token_ids_sha256


SCHEMA_VERSION = "image2299.ota_sqp_lite.v1"
UNIT_ID = "2026-08-30-image2299-ota-sqp-lite"
OUTPUT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration") / UNIT_ID
START_CHECKPOINT = (
    projected.OUTPUT_ROOT / "20260830T-gt17-gt32-coordinate-guard-v18"
    / "checkpoint-terminal-step-16"
)
START_RECEIPT = START_CHECKPOINT.parent / "receipt.json"
START_RECEIPT_SHA256 = "81827fe7d4e82600b52824207cac9b7b2845f345a2b6b2b5ef3c6509e3c18fce"
START_NATURAL_SHA256 = "dd483c20f1c21bc01678f551205143f163810ad7e2c4adb6f4e7b5cbc351afad"
START_PREFIX_SHA256 = "9543406a07249d21057bcc11e39f7a8069ce72efbf605d93bb20545b65284411"
START_PREFIX_EOS_SHA256 = "9e5969757e8ddc419b6a3e2affc90917c27e830718b713fd7c458f9577f82751"
START_SURFACE_SHA256 = "1a21f07c9b415a9e543db183785752ca327f59764f8be92d12cd405c77b4f822"
FROZEN_SURFACE_SHA256 = "c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5"
PARENT_ROUTE_SHA256 = base.PARENT_ROUTE_SHA256
PROMPT_TOKEN_SHA256 = base.PROMPT_TOKEN_SHA256
IMAGE_SHA256 = base.IMAGE_SHA256
MISSING_OWNERS = base.MISSING_OWNERS
RADII = projected.RADII
MAX_ACCEPTED_UPDATES = 16
MAX_SAME_DIVERGENCE_RELINEARIZATIONS = 3
WORLD_SIZE = 8
OPTIMIZATION_PREFIX_TOKENS = 297
ACTION_GRADIENTS = base.ROW_TOKENS + 1
CUT_GRADIENTS = 7
ACTIVE_GRADIENTS = ACTION_GRADIENTS + CUT_GRADIENTS
PERSON_TOKEN, TIE_TOKEN, CHAIR_TOKEN = 8987, 48731, 34196
R16, ALPHA16, R32, ALPHA32 = 16, 32, 32, 64
SOLVER_TOLERANCE = 1e-7
SAFETY_ETA = 0.01
TRUST_DELTA = base.LEARNING_RATE


class OtaSqpHold(RuntimeError):
    """A frozen identity, numerical, or admission contract failed closed."""


def _binding_receipt() -> dict[str, Any]:
    if (
        not START_CHECKPOINT.is_dir()
        or not START_RECEIPT.is_file()
        or base._sha256(START_RECEIPT) != START_RECEIPT_SHA256
        or base._sha256(base.TARGET_PATH) != base.TARGET_SHA256
        or token_ids_sha256(base._parent_tokens()) != PARENT_ROUTE_SHA256
    ):
        raise OtaSqpHold("HOLD: frozen OTA-SQP-lite specimen identity drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "execution_surface": "gpu_smoke_available_check_bindings_is_cpu_only",
        "start_checkpoint": str(START_CHECKPOINT),
        "start_receipt_sha256": START_RECEIPT_SHA256,
        "parent_route_sha256": PARENT_ROUTE_SHA256,
        "prompt_token_sha256": PROMPT_TOKEN_SHA256,
        "image_sha256": IMAGE_SHA256,
        "missing_owner_ids": sorted(MISSING_OWNERS),
        "world_size": WORLD_SIZE,
        "optimization_prefix_tokens": OPTIMIZATION_PREFIX_TOKENS,
        "active_gradient_count": ACTIVE_GRADIENTS,
    }


def _start_state() -> dict[str, Any]:
    _binding_receipt()
    receipt = json.loads(START_RECEIPT.read_text(encoding="utf-8"))
    warm = dict(dict(receipt.get("warm_final", {})).get("packet", {}))
    cold = dict(dict(receipt.get("cold", {})).get("packet", {}))
    warm_evaluation, cold_evaluation = map(dict, (warm.get("evaluation", {}), cold.get("evaluation", {})))
    warm_causal, cold_causal = map(dict, (warm.get("causal_ledger", {}), cold.get("causal_ledger", {})))
    tokens = list(map(int, warm_evaluation.get("generated_token_ids", ())))
    owners = list(map(str, receipt.get("parent_owner_ids", ())))
    raw = dict(dict(warm_evaluation.get("joint_gate", {})).get("hard_raw_counters", {}))
    if (
        receipt.get("schema_version") != "image2299.projected_owner_corridor.v18"
        or receipt.get("status") != "bounded_negative_no_match_level_promotion"
        or dict(receipt.get("saved_checkpoint", {})).get("checkpoint") != str(START_CHECKPOINT)
        or token_ids_sha256(tokens) != START_NATURAL_SHA256
        or token_ids_sha256(tokens[:OPTIMIZATION_PREFIX_TOKENS]) != START_PREFIX_SHA256
        or token_ids_sha256([*tokens[:OPTIMIZATION_PREFIX_TOKENS], base.EOS]) != START_PREFIX_EOS_SHA256
        or cold_evaluation.get("generated_token_ids") != tokens
        or len(tokens) != 34 * base.ROW_TOKENS + 1
        or tokens[-1:] != [base.EOS]
        or base.EOS in tokens[:-1]
        or len(owners) != 33 or len(set(owners)) != 33
        or set(map(str, warm_evaluation.get("matched_target_owner_ids", ()))) != set(owners)
        or int(warm_evaluation.get("hard_counter_count", -1)) != 3
        or raw != projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS
        or int(warm_causal.get("causal_hard_counter_count", -1)) != 1
        or dict(warm_causal.get("first_event") or {}).get("event_start") != OPTIMIZATION_PREFIX_TOKENS
        or dict(warm_causal.get("first_event") or {}).get("kind") not in {"duplicate", "near_miss"}
        or projected._state_identity(warm) != projected._state_identity(cold)
        or warm_causal.get("first_event") != cold_causal.get("first_event")
        or dict(receipt.get("terminal_surface", {})).get("aggregate_sha256") != START_SURFACE_SHA256
        or dict(dict(receipt.get("cold", {})).get("surface", {})).get("aggregate_sha256") != START_SURFACE_SHA256
        or receipt.get("frozen_surface_after") != FROZEN_SURFACE_SHA256
        or dict(receipt.get("cold", {})).get("frozen_surface") != FROZEN_SURFACE_SHA256
    ):
        raise OtaSqpHold("HOLD: v18 cold working-state identity drifted")
    return {
        "receipt": str(START_RECEIPT), "receipt_sha256": START_RECEIPT_SHA256,
        "natural_token_ids": tokens, "natural_token_ids_sha256": START_NATURAL_SHA256,
        "optimization_prefix_token_ids": tokens[:OPTIMIZATION_PREFIX_TOKENS],
        "optimization_prefix_sha256": token_ids_sha256(tokens[:OPTIMIZATION_PREFIX_TOKENS]),
        "designated_tail_token_ids": tokens[OPTIMIZATION_PREFIX_TOKENS:-1],
        "parent_owner_ids": owners, "surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "raw_hard_counters": raw, "causal_event": dict(warm_causal["first_event"]),
    }


def _compile_alias_witness(parent_tokens: Sequence[int], alias_tokens: Sequence[int]) -> list[int]:
    parent, alias = list(map(int, parent_tokens)), list(map(int, alias_tokens))
    if (
        len(parent) != 33 * base.ROW_TOKENS + 1
        or parent[-1:] != [base.EOS]
        or base.EOS in parent[:-1]
        or len(alias) != base.ROW_TOKENS
        or base.EOS in alias
    ):
        raise OtaSqpHold("HOLD: alias witness is not clean Parent33 + one row + EOS")
    return [*parent[:-1], *alias, base.EOS]


def _default_alias_candidates() -> list[dict[str, Any]]:
    if base._sha256(base.TARGET_PATH) != base.TARGET_SHA256:
        raise OtaSqpHold("HOLD: frozen target library identity drifted")
    target = json.loads(base.TARGET_PATH.read_text(encoding="utf-8"))
    _rows, by_owner = base._target_rows(target)
    candidates = [
        {
            "owner": owner,
            "name": "frozen-target",
            "source": "frozen_target_library",
            "token_ids": list(map(int, by_owner[owner]["token_ids"])),
        }
        for owner in sorted(MISSING_OWNERS)
    ]
    candidates.extend(
        {
            "owner": manifold.TARGET_OWNER,
            "name": str(item["name"]),
            "source": str(item["provenance"]),
            "token_ids": list(map(int, item["token_ids"])),
        }
        for item in manifold.ALIAS_CANDIDATES
    )
    for owner in sorted(MISSING_OWNERS):
        row = by_owner[owner]
        tokens = list(map(int, row["token_ids"]))
        coordinates = list(map(int, row.get("coord_bins", ())))
        if len(coordinates) != 4 or tokens[4:8] != [151670 + value for value in coordinates]:
            raise OtaSqpHold("HOLD: frozen target coordinate-token grammar drifted")
        for coordinate_index, offset in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)):
            neighbor = coordinates.copy()
            neighbor[coordinate_index] += offset
            if not (0 <= min(neighbor) and max(neighbor) <= 999 and neighbor[0] < neighbor[2] and neighbor[1] < neighbor[3]):
                continue
            neighbor_tokens = tokens.copy()
            neighbor_tokens[4:8] = [151670 + value for value in neighbor]
            candidates.append({
                "owner": owner,
                "name": f"neighbor-c{coordinate_index}{offset:+d}",
                "source": "bounded_coordinate_neighbor",
                "token_ids": neighbor_tokens,
            })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        row_sha = token_ids_sha256(candidate["token_ids"])
        unique.setdefault((str(candidate["owner"]), row_sha), {**candidate, "row_sha256": row_sha})
    bounded: list[dict[str, Any]] = []
    grouped = {owner: 0 for owner in MISSING_OWNERS}
    for candidate in unique.values():
        owner = str(candidate["owner"])
        if grouped[owner] < 8:
            bounded.append(candidate)
            grouped[owner] += 1
    if set(grouped) != set(MISSING_OWNERS) or any(count not in range(1, 9) for count in grouped.values()):
        raise OtaSqpHold("HOLD: alias catalog does not cover every missing owner within budget")
    return bounded


def _global_alias_gate(
    ledger: Mapping[str, Any], *, parent_owner_ids: Sequence[str], target_owner: str,
) -> dict[str, Any]:
    parent = set(map(str, parent_owner_ids))
    parse, matcher = dict(ledger.get("parse", {})), dict(ledger.get("matcher", {}))
    receipts = list(matcher.get("prediction_receipts", ()))
    statuses = [str(item.get("strict_match_status", "")) for item in receipts]
    owners = [str(item.get("strict_match_gt_owner_id", "")) for item in receipts]
    expected = parent | {str(target_owner)}
    debt = {
        "parent_identity": len(parent) != 33 or len(parent_owner_ids) != 33,
        "target_already_occupied": str(target_owner) in parent,
        "parser": (
            int(parse.get("valid_prediction_count", -1)) != 34
            or int(parse.get("dropped_prediction_count", -1)) != 0
        ),
        "global_match_status": len(receipts) != 34 or statuses != ["matched"] * 34,
        "global_owner_set": set(owners) != expected or len(owners) != len(set(owners)),
        "global_cardinality": (
            int(matcher.get("optimum_cardinality", -1)) != 34
            or set(map(str, matcher.get("committed_owner_ids", ()))) != expected
        ),
        "ambiguity": bool(matcher.get("neutral_pred_row_ids")) or bool(matcher.get("ambiguity_receipts")),
    }
    active = {name: failed for name, failed in debt.items() if failed}
    return {"passed": not active, "debt": active, "committed_owner_ids": sorted(set(owners))}


def _admit_alias(
    *, tokenizer: Any, raw_example: Any, parent_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    owner = str(candidate.get("owner", ""))
    tokens = list(map(int, candidate.get("token_ids", ())))
    witness = _compile_alias_witness(parent_tokens, tokens)
    ledger = base._parse_and_match(
        tokenizer=tokenizer,
        generated_token_ids=witness,
        label=f"ota-alias-{owner.rsplit(':', 1)[-1]}-{token_ids_sha256(tokens)[:12]}",
        raw_example=raw_example,
    )
    gate = _global_alias_gate(ledger, parent_owner_ids=parent_owner_ids, target_owner=owner)
    if not gate["passed"]:
        raise OtaSqpHold(f"HOLD: alias failed production global matcher: {gate['debt']}")
    return {
        **dict(candidate),
        "owner": owner,
        "token_ids": tokens,
        "row_sha256": token_ids_sha256(tokens),
        "witness_sha256": token_ids_sha256(witness),
        "global_match": gate,
        "admitted": True,
    }


def _admit_alias_catalog(
    *, tokenizer: Any, raw_example: Any, parent_tokens: Sequence[int],
    parent_owner_ids: Sequence[str], candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    admitted, rejected = [], []
    for candidate in candidates or _default_alias_candidates():
        try:
            admitted.append(_admit_alias(
                tokenizer=tokenizer, raw_example=raw_example, parent_tokens=parent_tokens,
                parent_owner_ids=parent_owner_ids, candidate=candidate,
            ))
        except OtaSqpHold as exc:
            rejected.append({**dict(candidate), "admitted": False, "reason": str(exc)})
    if {str(item["owner"]) for item in admitted} != set(MISSING_OWNERS):
        raise OtaSqpHold("HOLD: production matcher left an owner without an admitted alias")
    return {"admitted": admitted, "rejected": rejected}


def _shortlist_aliases(
    admissions: Sequence[Mapping[str, Any]], scores: Mapping[str, float],
) -> dict[str, list[dict[str, Any]]]:
    by_owner: dict[str, list[dict[str, Any]]] = {owner: [] for owner in MISSING_OWNERS}
    for admission in admissions:
        owner, row_sha = str(admission.get("owner", "")), str(admission.get("row_sha256", ""))
        score = float(scores.get(row_sha, math.nan))
        if owner not in by_owner or not admission.get("admitted") or not math.isfinite(score):
            raise OtaSqpHold("HOLD: alias forward score/admission binding is incomplete")
        by_owner[owner].append({**dict(admission), "complete_action_min": score})
    if any(not values or len(values) > 8 for values in by_owner.values()):
        raise OtaSqpHold("HOLD: admitted catalog coverage/budget drifted")
    return {
        owner: sorted(values, key=lambda item: (-float(item["complete_action_min"]), str(item["row_sha256"])))[:2]
        for owner, values in by_owner.items()
    }


def _complete_action_binding(real_prefix: Sequence[int], action_tokens: Sequence[int]) -> dict[str, Any]:
    prefix, action = list(map(int, real_prefix)), list(map(int, action_tokens))
    if len(action) != base.ROW_TOKENS + 1 or action[-1] != base.EOS or base.EOS in action[:-1]:
        raise OtaSqpHold("HOLD: complete action must be one parser row followed by EOS")
    teacher = [*prefix, *action]
    return {
        "real_prefix_sha256": token_ids_sha256(prefix),
        "action_sha256": token_ids_sha256(action),
        "teacher_tokens_sha256": token_ids_sha256(teacher),
        "action_start": len(prefix),
        "action_length": len(action),
    }


def _complete_action_margin_terms(
    logits: torch.Tensor, *, teacher_tokens: Sequence[int], real_prefix_tokens: Sequence[int],
    natural_tokens: Sequence[int], action_tokens: Sequence[int], binding: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    teacher = list(map(int, teacher_tokens))
    prefix, natural, action = map(lambda values: list(map(int, values)), (real_prefix_tokens, natural_tokens, action_tokens))
    expected = _complete_action_binding(prefix, action)
    if (
        dict(binding) != expected
        or teacher != [*prefix, *action]
        or natural[:len(prefix)] != prefix
        or logits.ndim != 2
        or logits.shape[0] != len(teacher)
        or logits.shape[1] <= max(PERSON_TOKEN, TIE_TOKEN, CHAIR_TOKEN, max(action))
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise OtaSqpHold("HOLD: complete-action teacher/prefix/logit identity drifted")
    terms, records = [], []
    for relative, target in enumerate(action):
        position = len(prefix) + relative
        row = logits[position]
        competitors = row.clone()
        competitors[target] = -torch.inf
        competitor = int(torch.argmax(competitors).item())
        margin = row[target] - row[competitor]
        terms.append(margin)
        record = {
            "position": position,
            "relative_action_position": relative,
            "target_token_id": target,
            "competitor_token_id": competitor,
            "prefix_sha256": token_ids_sha256(teacher[:position]),
            "margin": float(margin.detach()),
        }
        if relative == 1 and target == PERSON_TOKEN:
            record["required_person_competitors"] = {
                "tie": float((row[target] - row[TIE_TOKEN]).detach()),
                "chair": float((row[target] - row[CHAIR_TOKEN]).detach()),
            }
        records.append(record)
    stacked = torch.stack(terms)
    return stacked, {
        "binding": expected,
        "per_token": records,
        "minimum_margin": float(stacked.detach().min()),
        "weakest_relative_position": int(torch.argmin(stacked.detach()).item()),
    }


def _margin_gradients(
    margin_terms: torch.Tensor, parameters: Sequence[torch.nn.Parameter],
) -> list[tuple[torch.Tensor, ...]]:
    if margin_terms.ndim != 1 or not parameters or not margin_terms.requires_grad:
        raise OtaSqpHold("HOLD: margin gradient inputs are malformed")
    gradients: list[tuple[torch.Tensor, ...]] = []
    for index, term in enumerate(margin_terms):
        values = torch.autograd.grad(term, parameters, retain_graph=index + 1 < len(margin_terms), allow_unused=True)
        if any(value is None for value in values):
            raise OtaSqpHold("HOLD: active margin is disconnected from the declared surface")
        detached = tuple(value.detach().float() for value in values if value is not None)
        if any(not bool(torch.isfinite(value).all().item()) for value in detached):
            raise OtaSqpHold("HOLD: active margin gradient is non-finite")
        gradients.append(detached)
    return gradients


def _gradient_gram(gradients: Sequence[Sequence[torch.Tensor]]) -> np.ndarray:
    if not gradients or not gradients[0]:
        raise OtaSqpHold("HOLD: empty active gradient span")
    shape = tuple(tuple(value.shape for value in row) for row in gradients)
    if any(row != shape[0] for row in shape):
        raise OtaSqpHold("HOLD: active gradients do not share one parameter surface")
    device = gradients[0][0].device
    gram = torch.zeros((len(gradients), len(gradients)), dtype=torch.float64, device=device)
    for parameter_index in range(len(gradients[0])):
        block = torch.stack([row[parameter_index].flatten() for row in gradients]).double()
        gram.addmm_(block, block.T)
    return gram.cpu().numpy()


def _solve_final_margin_span(
    target_margins: Sequence[float], cut_margins: Sequence[float], gram: np.ndarray,
    *, eta: float, delta: float,
) -> dict[str, Any]:
    target = np.asarray(target_margins, dtype=np.float64)
    cuts = np.asarray(cut_margins, dtype=np.float64)
    matrix = np.asarray(gram, dtype=np.float64)
    count, active = len(target), len(target) + len(cuts)
    if (
        count < 1 or matrix.shape != (active, active) or not np.isfinite(matrix).all()
        or not np.isfinite(target).all() or not np.isfinite(cuts).all()
        or not math.isfinite(eta) or eta <= 0.0 or not math.isfinite(delta) or delta <= 0.0
        or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0)
        or float(np.linalg.eigvalsh(matrix).min(initial=0.0)) < -SOLVER_TOLERANCE
    ):
        raise OtaSqpHold("HOLD: malformed/non-PSD SQP-lite span problem")

    def objective(x: np.ndarray) -> float:
        return -float(x[-1])

    def constraints(x: np.ndarray) -> np.ndarray:
        coefficients, rho = x[:-1], x[-1]
        projected = matrix @ coefficients
        return np.concatenate((target + projected[:count] - rho, cuts + projected[count:] - eta, [delta**2 - coefficients @ matrix @ coefficients]))

    initial = np.zeros(active + 1, dtype=np.float64)
    initial[-1] = float(target.min())
    result = minimize(
        objective, initial, method="SLSQP", constraints={"type": "ineq", "fun": constraints},
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success or result.x.shape != initial.shape or not np.isfinite(result.x).all():
        raise OtaSqpHold(f"HOLD: SQP-lite solver failed: {result.message}")
    coefficients, rho = result.x[:-1], float(result.x[-1])
    projected = matrix @ coefficients
    predicted_target = target + projected[:count]
    predicted_cuts = cuts + projected[count:]
    norm = math.sqrt(max(0.0, float(coefficients @ matrix @ coefficients)))
    residual = constraints(result.x)
    if (
        float(residual.min()) < -SOLVER_TOLERANCE
        or norm > delta + SOLVER_TOLERANCE
        or (len(cuts) and float(predicted_cuts.min()) < eta - SOLVER_TOLERANCE)
        or not math.isclose(rho, float(predicted_target.min()), abs_tol=SOLVER_TOLERANCE, rel_tol=0.0)
        or float(predicted_target.min()) <= float(target.min()) + SOLVER_TOLERANCE
    ):
        raise OtaSqpHold("HOLD: SQP-lite primal/sign/residual certificate failed")
    return {
        "coefficients": coefficients.tolist(),
        "rho": rho,
        "ordinary_l2_norm": norm,
        "predicted_target_margins": predicted_target.tolist(),
        "target_margin_changes": projected[:count].tolist(),
        "predicted_cut_margins": predicted_cuts.tolist(),
        "eta": eta,
        "delta": delta,
        "minimum_primal_residual": float(residual.min()),
        "solver": {"method": "scipy.optimize.SLSQP", "iterations": int(result.nit)},
    }


def _materialize_span_direction(
    gradients: Sequence[Sequence[torch.Tensor]], coefficients: Sequence[float],
) -> tuple[torch.Tensor, ...]:
    if len(gradients) != len(coefficients) or not gradients:
        raise OtaSqpHold("HOLD: malformed span materialization")
    return tuple(
        sum((float(coefficient) * row[index] for coefficient, row in zip(coefficients, gradients, strict=True)), torch.zeros_like(gradients[0][index]))
        for index in range(len(gradients[0]))
    )


def _seed_active_cut_bindings(
    route_tokens: Sequence[int], low_slack: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    cuts = [{**item, "cut_id": str(item["guard_id"]), "source": "projected_G0_G6"} for item in projected._guard_bindings(route_tokens)]
    seen = {str(item["cut_id"]) for item in cuts}
    for item in low_slack:
        cut_id, margin = str(item.get("cut_id", "")), float(item.get("margin", math.nan))
        if not cut_id or cut_id in seen or not math.isfinite(margin) or margin <= 0.0:
            raise OtaSqpHold("HOLD: malformed/non-strict low-slack incumbent cut")
        cuts.append(dict(item))
        seen.add(cut_id)
    return cuts


def _candidate_gate(
    candidate: Mapping[str, Any], *, parent_owner_ids: Sequence[str],
    incumbent_action_min: float, incumbent_divergence: int, eta: float,
) -> dict[str, Any]:
    parent = set(map(str, parent_owner_ids))
    owners = set(map(str, candidate.get("matched_owner_ids", ())))
    raw = dict(candidate.get("raw_hard_counters", {}))
    raw_keys = set(projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS)
    statuses = dict(candidate.get("matcher_status_counts", {}))
    committed = set(map(str, candidate.get("matcher_committed_owner_ids", ())))
    cuts = list(map(float, candidate.get("active_cut_margins", ())))
    action_min = float(candidate.get("complete_action_min", math.nan))
    divergence = int(candidate.get("first_natural_divergence", -1))
    structural_debt = {
        "parent_identity": len(parent) != 33 or len(parent_owner_ids) != 33,
        "parent_exchange": not parent.issubset(owners),
        "not_cold_reloaded": not bool(candidate.get("cold_reloaded")),
        "non_row_aligned_eos": not bool(candidate.get("row_aligned_eos")),
        "token_budget": bool(candidate.get("token_budget_exhausted")),
        "raw_counter_schema": (
            set(raw) != raw_keys
            or any(not isinstance(value, (int, bool)) or int(value) < 0 for value in raw.values())
        ),
        "parser_matcher_binding": (
            int(candidate.get("parser_valid_prediction_count", -1)) != sum(int(value) for value in statuses.values())
            or int(candidate.get("parser_dropped_prediction_count", -1)) != 0
            or int(candidate.get("matcher_optimum_cardinality", -1)) != len(owners)
            or committed != owners
        ),
    }
    promotion_debt = {
        **structural_debt,
        "not_proper_superset": not owners > parent,
        "hard_debt": any(int(value) != 0 for value in raw.values()),
        "causal_debt": int(candidate.get("causal_hard_counter_count", 0)) != 0,
        "strict_match_status": statuses != {"matched": len(owners)},
    }
    promotion_active = {name: failed for name, failed in promotion_debt.items() if failed}
    progress = action_min > float(incumbent_action_min) or divergence > int(incumbent_divergence)
    exact_designated_tail = (
        raw == projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS
        and int(candidate.get("causal_hard_counter_count", -1)) == 1
        and int(candidate.get("designated_tail_debt_count", -1)) == 1
        and bool(candidate.get("working_corridor_passed"))
    )
    working_debt = {
        **structural_debt,
        "owner_cardinality": owners != parent,
        "designated_tail_debt": not exact_designated_tail,
        "strict_match_status": statuses != {"matched": len(parent), "unmatched": 1},
        "non_finite_action_margin": not math.isfinite(action_min),
        "incumbent_cut": eta <= 0.0 or not cuts or any(not math.isfinite(value) or value < eta for value in cuts),
        "no_exact_progress": not progress,
    }
    working_active = {name: failed for name, failed in working_debt.items() if failed}
    return {
        "promotion": {"passed": not promotion_active, "debt": promotion_active},
        "working": {"passed": not working_active, "debt": working_active},
        "progress": {"action_min": action_min, "first_natural_divergence": divergence},
    }


def _warm_candidate_screen(
    candidate: Mapping[str, Any], *, parent_owner_ids: Sequence[str],
    incumbent_action_min: float, incumbent_divergence: int, eta: float,
) -> dict[str, Any]:
    probe = {**dict(candidate), "cold_reloaded": True}
    gate = _candidate_gate(
        probe, parent_owner_ids=parent_owner_ids,
        incumbent_action_min=incumbent_action_min,
        incumbent_divergence=incumbent_divergence, eta=eta,
    )
    candidate_gate = _candidate_gate(
        candidate, parent_owner_ids=parent_owner_ids,
        incumbent_action_min=incumbent_action_min,
        incumbent_divergence=incumbent_divergence, eta=eta,
    )
    if candidate_gate["promotion"]["passed"] or candidate_gate["working"]["passed"]:
        raise OtaSqpHold("HOLD: warm candidate acquired authoritative cold admission")
    return {
        "authoritative": False,
        "promotion_screen_passed": bool(gate["promotion"]["passed"]),
        "working_screen_passed": bool(gate["working"]["passed"]),
        "authoritative_cold_gate": candidate_gate,
    }


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]], *, parent_owner_ids: Sequence[str],
    incumbent_action_min: float, incumbent_divergence: int, eta: float,
) -> dict[str, Any] | None:
    by_radius = {float(item.get("radius", math.nan)): item for item in candidates}
    if set(by_radius) != set(RADII) or len(candidates) != len(RADII):
        raise OtaSqpHold("HOLD: candidate panel is not the frozen eight-radius panel")
    gated = [
        {**dict(candidate), "gate": _candidate_gate(
            candidate, parent_owner_ids=parent_owner_ids,
            incumbent_action_min=incumbent_action_min,
            incumbent_divergence=incumbent_divergence, eta=eta,
        )}
        for candidate in candidates
    ]
    promotions = [item for item in gated if item["gate"]["promotion"]["passed"]]
    if promotions:
        return max(promotions, key=lambda item: (len(set(item["matched_owner_ids"])), float(item["radius"])))
    working = [item for item in gated if item["gate"]["working"]["passed"]]
    return max(working, key=lambda item: float(item["radius"])) if working else None


def _select_warm_candidate(
    candidates: Sequence[Mapping[str, Any]], *, parent_owner_ids: Sequence[str],
    incumbent_action_min: float, incumbent_divergence: int, eta: float,
) -> dict[str, Any] | None:
    failures = [
        dict(item["evaluation_error"])
        for item in candidates if item.get("evaluation_error") is not None
    ]
    if failures:
        raise OtaSqpHold(f"HOLD: warm radius evaluation failed: {failures}")
    by_radius = {float(item.get("radius", math.nan)): item for item in candidates}
    if set(by_radius) != set(RADII) or len(candidates) != len(RADII):
        raise OtaSqpHold("HOLD: warm candidate panel is not the frozen eight-radius panel")
    screened = [
        {**dict(item), "warm_screen": _warm_candidate_screen(
            item, parent_owner_ids=parent_owner_ids,
            incumbent_action_min=incumbent_action_min,
            incumbent_divergence=incumbent_divergence, eta=eta,
        )}
        for item in candidates
    ]
    promotions = [item for item in screened if item["warm_screen"]["promotion_screen_passed"]]
    if promotions:
        return max(promotions, key=lambda item: (len(set(item["matched_owner_ids"])), float(item["radius"])))
    working = [item for item in screened if item["warm_screen"]["working_screen_passed"]]
    return max(working, key=lambda item: float(item["radius"])) if working else None


def _stop_decision(
    *, accepted_updates: int, same_divergence_relinearizations: int,
    cold_promotion: bool = False, hold_reason: str | None = None,
) -> dict[str, str]:
    if accepted_updates < 0 or same_divergence_relinearizations < 0:
        raise OtaSqpHold("HOLD: negative OTA-SQP-lite accounting")
    if hold_reason:
        return {"decision": "HOLD", "reason": hold_reason}
    if cold_promotion:
        return {"decision": "STOP_SUCCESS", "reason": "clean_cold_34_owner_promotion"}
    if accepted_updates >= MAX_ACCEPTED_UPDATES:
        return {"decision": "STOP_BOUNDED_NEGATIVE", "reason": "r16_update_budget_exhausted"}
    if same_divergence_relinearizations >= MAX_SAME_DIVERGENCE_RELINEARIZATIONS:
        return {"decision": "STOP_BRANCH", "reason": "same_divergence_owner_set_three_relinearizations"}
    return {"decision": "CONTINUE", "reason": "within_frozen_budget"}


def _expand_r16_dora_factors(
    lora_a: torch.Tensor, lora_b: torch.Tensor, magnitude: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != R16
        or lora_b.shape[1] != R16 or lora_b.shape[0] != magnitude.numel()
        or not all(bool(torch.isfinite(value).all().item()) for value in (lora_a, lora_b, magnitude))
    ):
        raise OtaSqpHold("HOLD: malformed r16 DoRA factors")
    expanded_a = lora_a.new_empty((R32, lora_a.shape[1]))
    expanded_b = lora_b.new_zeros((lora_b.shape[0], R32))
    expanded_a[:R16].copy_(lora_a)
    expanded_b[:, :R16].copy_(lora_b)
    values = torch.arange(
        1, (R32 - R16) * lora_a.shape[1] + 1, device=lora_a.device, dtype=torch.float64,
    ).reshape(R32 - R16, lora_a.shape[1])
    expanded_a[R16:].copy_((torch.sin(values) / math.sqrt(lora_a.shape[1])).to(lora_a.dtype))
    if (
        not bool((expanded_a[R16:] != 0).all().item())
        or not torch.equal(expanded_b[:, R16:], torch.zeros_like(expanded_b[:, R16:]))
        or ALPHA16 / R16 != ALPHA32 / R32
        or not torch.allclose(
            (ALPHA16 / R16) * (lora_b @ lora_a),
            (ALPHA32 / R32) * (expanded_b @ expanded_a),
            atol=0.0, rtol=0.0,
        )
    ):
        raise OtaSqpHold("HOLD: r32 expansion is not function-preserving/live")
    return {"lora_A": expanded_a, "lora_B": expanded_b, "magnitude": magnitude.clone()}


def _prepare_output(output: Path) -> dict[str, Any]:
    if output.exists():
        raise OtaSqpHold(f"HOLD: refusing overwrite: {output}")
    output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = output / "runner_source.py"
    snapshot.write_bytes(source)
    if snapshot.read_bytes() != source:
        raise OtaSqpHold("HOLD: runner source snapshot readback drifted")
    return {"path": str(snapshot), "sha256": base._sha256(snapshot)}


def _apply_trust_displacement(
    parameters: Sequence[torch.nn.Parameter], clones: Sequence[torch.Tensor],
    displacement: Sequence[torch.Tensor], *, radius: float,
) -> None:
    if radius not in RADII or not (len(parameters) == len(clones) == len(displacement)):
        raise OtaSqpHold("HOLD: malformed OTA trust-radius application")
    with torch.no_grad():
        for parameter, clone, delta in zip(parameters, clones, displacement, strict=True):
            if parameter.shape != clone.shape or parameter.shape != delta.shape:
                raise OtaSqpHold("HOLD: OTA trust-radius surface drifted")
            parameter.copy_(clone + radius * delta)


def _agree_hash(value: Any, *, label: str) -> str:
    digest = base._hash(value)
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise OtaSqpHold(f"HOLD: rank disagreement for {label}")
    return digest


def _ota_state_identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    identity = projected._state_identity(packet)
    identity["evaluation"] = dict(identity["evaluation"])
    identity["evaluation"].pop("exact_target_prefix", None)
    return identity


def _margin_panel(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int,
    natural_tokens: Sequence[int], action_tokens: Sequence[int],
) -> dict[str, Any]:
    prefix = list(map(int, natural_tokens[:OPTIMIZATION_PREFIX_TOKENS]))
    action = list(map(int, action_tokens))
    binding = _complete_action_binding(prefix, action)
    teacher = [*prefix, *action]
    with torch.inference_mode():
        logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs, route_tokens=teacher, pad_token_id=pad,
        )
        _terms, panel = _complete_action_margin_terms(
            logits, teacher_tokens=teacher, real_prefix_tokens=prefix,
            natural_tokens=natural_tokens, action_tokens=action, binding=binding,
        )
    return panel


def _cut_margin_panel(
    *, model: Any, native_inputs: Mapping[str, Any], pad: int, route_tokens: Sequence[int],
) -> tuple[list[dict[str, Any]], list[float]]:
    bindings = _seed_active_cut_bindings(route_tokens)
    if len(bindings) != CUT_GRADIENTS or [item["cut_id"] for item in bindings] != [f"G{i}" for i in range(7)]:
        raise OtaSqpHold("HOLD: active G0-G6 cut set drifted")
    with torch.inference_mode():
        logits = base.full_root._teacher_forced_route_logits(
            model=model, native_inputs=native_inputs,
            route_tokens=bindings[0]["teacher_tokens"], pad_token_id=pad,
        )
        margins = [float(projected._margin(logits, item).item()) for item in bindings]
    return bindings, margins


def _candidate_record(
    *, packet: Mapping[str, Any], radius: float, rank: int, action_tokens: Sequence[int],
    incumbent_tokens: Sequence[int], action_panel: Mapping[str, Any] | None,
    cut_bindings: Sequence[Mapping[str, Any]], cut_margins: Sequence[float], cold_reloaded: bool,
) -> dict[str, Any]:
    evaluation = dict(packet["evaluation"])
    causal = dict(packet["causal_ledger"])
    joint = dict(evaluation.get("joint_gate", {}))
    parser, matcher = map(dict, (joint.get("parser", {}), joint.get("matcher", {})))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    raw = dict(joint.get("hard_raw_counters", {}))
    corridor = bool(dict(packet.get("corridor_gate", {})).get("passed"))
    return {
        "rank": rank, "radius": radius, "cold_reloaded": cold_reloaded,
        "matched_owner_ids": list(map(str, evaluation.get("matched_target_owner_ids", ()))),
        "matcher_committed_owner_ids": list(map(str, matcher.get("committed_owner_ids", ()))),
        "matcher_status_counts": dict(matcher.get("strict_status_counts", {})),
        "matcher_optimum_cardinality": int(matcher.get("optimum_cardinality", -1)),
        "parser_valid_prediction_count": int(parser.get("valid_prediction_count", -1)),
        "parser_dropped_prediction_count": int(parser.get("dropped_prediction_count", -1)),
        "raw_hard_counters": raw,
        "hard_debt_counts": {},
        "causal_hard_counter_count": int(causal.get("causal_hard_counter_count", -1)),
        "designated_tail_debt_count": int(corridor and not bool(dict(packet.get("promotion_gate", {})).get("passed"))),
        "working_corridor_passed": corridor,
        "row_aligned_eos": bool(joint.get("natural_row_aligned_eos")),
        "token_budget_exhausted": len(tokens) >= base.NATURAL_MAX_TOKENS,
        "complete_action_min": (
            -1.0e30 if action_panel is None else float(action_panel["minimum_margin"])
        ),
        "first_natural_divergence": (
            base._exact_prefix(tokens[OPTIMIZATION_PREFIX_TOKENS:], action_tokens)
            if len(tokens) >= OPTIMIZATION_PREFIX_TOKENS else -1
        ),
        "active_cut_margins": list(map(float, cut_margins)),
        "active_cut_bindings": [
            {key: value for key, value in item.items() if key != "teacher_tokens"}
            for item in cut_bindings
        ],
        "action_panel": action_panel,
        "packet_identity": _ota_state_identity(packet),
        "state": projected._candidate_state(packet),
        "generated_token_ids_sha256": evaluation.get("generated_token_ids_sha256"),
        "incumbent_lcp": base._exact_prefix(incumbent_tokens, tokens),
    }


def _evaluate_state(
    *, model: Any, tokenizer: Any, native_inputs: Mapping[str, Any], pad: int,
    target: Mapping[str, Any], raw_example: Any, parent_owners: Sequence[str],
    action_tokens: Sequence[int], incumbent_tokens: Sequence[int], label: str,
    radius: float, rank: int, cold_reloaded: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    packet = projected._state_packet(
        model=model, tokenizer=tokenizer, native_inputs=native_inputs, pad=pad,
        target=target, raw_example=raw_example, parent_owners=parent_owners,
        alias_tokens=action_tokens[:-1], label=label,
    )
    action_panel: Mapping[str, Any] | None = None
    cut_bindings: Sequence[Mapping[str, Any]] = ()
    cut_margins: Sequence[float] = ()
    forward_count = 0
    if not bool(packet["promotion_gate"]["passed"]) and bool(packet["corridor_gate"]["passed"]):
        action_panel = _margin_panel(
            model=model, native_inputs=native_inputs, pad=pad,
            natural_tokens=packet["evaluation"]["generated_token_ids"], action_tokens=action_tokens,
        )
        cut_bindings, cut_margins = _cut_margin_panel(
            model=model, native_inputs=native_inputs, pad=pad,
            route_tokens=packet["evaluation"]["generated_token_ids"],
        )
        forward_count = 2
    return packet, _candidate_record(
        packet=packet, radius=radius, rank=rank, action_tokens=action_tokens,
        incumbent_tokens=incumbent_tokens, action_panel=action_panel,
        cut_bindings=cut_bindings, cut_margins=cut_margins, cold_reloaded=cold_reloaded,
    ), forward_count


def _cold_state(
    checkpoint: Path, *, target: Mapping[str, Any], parent_owners: Sequence[str],
    action_tokens: Sequence[int], incumbent_tokens: Sequence[int], radius: float,
    measure_candidate: bool,
) -> dict[str, Any]:
    setup = base._setup_for_checkpoint(checkpoint)
    with base.open_backend_session(setup["frontend"].launch) as opened:
        if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
            raise OtaSqpHold("HOLD: cold OTA evaluation requires concrete FP32 HF backend")
        model, tokenizer = opened._model, opened._tokenizer
        native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
        if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
            raise OtaSqpHold("HOLD: cold OTA prompt identity drifted")
        names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
        surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
        frozen = base._frozen_surface(model, names)
        if measure_candidate:
            packet, candidate, forwards = _evaluate_state(
                model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                parent_owners=parent_owners, action_tokens=action_tokens,
                incumbent_tokens=incumbent_tokens, label="ota-selected-cold",
                radius=radius, rank=0, cold_reloaded=True,
            )
        else:
            packet = projected._state_packet(
                model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                parent_owners=parent_owners, alias_tokens=action_tokens[:-1],
                label="ota-incumbent-terminal-cold",
            )
            candidate, forwards = None, 0
        return {
            "packet": packet, "candidate": candidate, "teacher_forward_count": forwards,
            "surface": surface, "frozen_surface": frozen,
            "runtime": opened.receipt.to_artifact_dict(),
        }


def run(*, run_id: str) -> Path:
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise OtaSqpHold("HOLD: OTA-SQP-lite smoke requires torchrun --nproc_per_node=8")
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    rank = int(os.environ.get("RANK", "0"))
    owns_output = False
    counters = {
        "teacher_forwards": 0, "gradient_backwards": 0, "natural_decodes": 0,
        "checkpoint_saves": 0, "cold_reloads": 0,
    }
    saved: Mapping[str, Any] | None = None
    source_snapshot: Mapping[str, Any] | None = None
    try:
        dist.init_process_group(backend="nccl")
        rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise OtaSqpHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        bindings, start = _binding_receipt(), _start_state()
        _parent_admission, _parent_setup, target = base._bindings()
        setup = base._setup_for_checkpoint(START_CHECKPOINT)
        current_tokens = list(map(int, start["natural_token_ids"]))
        prefix = list(map(int, start["optimization_prefix_token_ids"]))
        parent_owners = list(map(str, start["parent_owner_ids"]))
        parent_tokens = [*prefix, base.EOS]
        alias_catalog: Mapping[str, Any] | None = None
        ranking: Mapping[str, Any] | None = None
        baseline_packet: Mapping[str, Any] | None = None
        candidate_panel: list[dict[str, Any]] = []
        selected: Mapping[str, Any] | None = None
        linearization: Mapping[str, Any] | None = None
        solver: Mapping[str, Any] | None = None
        direction_receipt: Mapping[str, Any] | None = None
        initial_surface: Mapping[str, Any] | None = None
        terminal_surface: Mapping[str, Any] | None = None
        runtime: Mapping[str, Any] | None = None

        with base.open_backend_session(setup["frontend"].launch) as opened:
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise OtaSqpHold("HOLD: OTA smoke requires concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(setup["requests"][:1])
            if token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256:
                raise OtaSqpHold("HOLD: live OTA prompt identity drifted")
            if base._sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256:
                raise OtaSqpHold("HOLD: live OTA image identity drifted")
            names, parameters = base.full_root._trainable_surface(model, dora_all_only=True)
            if len(parameters) != 588 or sum(parameter.numel() for parameter in parameters) != 18_006_016:
                raise OtaSqpHold("HOLD: exact 588/18006016 FP32 language-DoRA surface drifted")
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            initial_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if initial_surface["aggregate_sha256"] != START_SURFACE_SHA256:
                raise OtaSqpHold("HOLD: v18 loaded DoRA surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)
            frozen_before = base._frozen_surface(model, names)
            if frozen_before != FROZEN_SURFACE_SHA256:
                raise OtaSqpHold("HOLD: v18 frozen surface drifted")

            def baseline_and_aliases(
                *, model: Any = model, tokenizer: Any = tokenizer,
                native_inputs: Mapping[str, Any] = native_inputs,
            ) -> dict[str, Any]:
                nonlocal counters
                parent_ledger = base._parse_and_match(
                    tokenizer=tokenizer, generated_token_ids=parent_tokens,
                    label="ota-v18-real-prefix-parent33", raw_example=setup["raw_example"],
                )
                prefix_owners = list(base._strict_owner_order(parent_ledger))
                if set(prefix_owners) != set(parent_owners):
                    raise OtaSqpHold("HOLD: real v18 pre-tail prefix lost the Parent33 owner set")
                admissions = _admit_alias_catalog(
                    tokenizer=tokenizer, raw_example=setup["raw_example"],
                    parent_tokens=parent_tokens, parent_owner_ids=parent_owners,
                )
                first = admissions["admitted"][0]
                counters["natural_decodes"] += 1
                packet = projected._state_packet(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, alias_tokens=first["token_ids"],
                    label="ota-v18-loaded-baseline",
                )
                if (
                    packet["evaluation"]["generated_token_ids"] != current_tokens
                    or not packet["corridor_gate"]["passed"]
                    or packet["promotion_gate"]["passed"]
                ):
                    raise OtaSqpHold("HOLD: loaded v18 baseline did not reproduce its cold working state")
                return {
                    "prefix_owner_ids": prefix_owners,
                    "prefix_route_sha256": token_ids_sha256(parent_tokens),
                    "admissions": admissions, "packet": packet,
                }

            baseline_bundle = base._rank0_call(baseline_and_aliases)
            baseline_packet = baseline_bundle["packet"]
            alias_catalog = baseline_bundle["admissions"]
            admissions = list(alias_catalog["admitted"])
            local_scores: list[dict[str, Any]] = []
            local_score_error: dict[str, str] | None = None
            try:
                for index in range(rank, len(admissions), WORLD_SIZE):
                    admission = admissions[index]
                    action = [*map(int, admission["token_ids"]), base.EOS]
                    panel = _margin_panel(
                        model=model, native_inputs=native_inputs, pad=int(tokenizer.pad_token_id),
                        natural_tokens=current_tokens, action_tokens=action,
                    )
                    counters["teacher_forwards"] += 1
                    local_scores.append({
                        "catalog_index": index, "owner": admission["owner"],
                        "row_sha256": admission["row_sha256"], "panel": panel,
                    })
            except BaseException as error:
                local_score_error = {"type": type(error).__name__, "error": str(error)}
            gathered_scores: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_scores, {
                "rank": rank, "records": local_scores, "error": local_score_error,
            })

            def choose_action() -> dict[str, Any]:
                errors = [item for item in gathered_scores if item["error"] is not None]
                if errors:
                    raise OtaSqpHold(f"HOLD: distributed alias ranking failed: {errors}")
                records = [record for item in gathered_scores for record in item["records"]]
                if len(records) != len(admissions) or {int(item["catalog_index"]) for item in records} != set(range(len(admissions))):
                    raise OtaSqpHold("HOLD: distributed alias ranking coverage drifted")
                scores = {str(item["row_sha256"]): float(item["panel"]["minimum_margin"]) for item in records}
                shortlist = _shortlist_aliases(admissions, scores)
                finalists = [item for values in shortlist.values() for item in values]
                chosen = sorted(
                    finalists,
                    key=lambda item: (-float(item["complete_action_min"]), str(item["owner"]), str(item["row_sha256"])),
                )[0]
                return {
                    "distributed_records": sorted(records, key=lambda item: int(item["catalog_index"])),
                    "shortlist": shortlist, "selected": chosen,
                }

            ranking = base._rank0_call(choose_action)
            selected_action = dict(ranking["selected"])
            action_tokens = [*map(int, selected_action["token_ids"]), base.EOS]
            incumbent_action_min = float(selected_action["complete_action_min"])
            incumbent_divergence = base._exact_prefix(
                current_tokens[OPTIMIZATION_PREFIX_TOKENS:], action_tokens,
            )
            cut_bindings = _seed_active_cut_bindings(current_tokens)
            linearization_manifest = {
                "prefix_token_count": len(prefix), "prefix_sha256": token_ids_sha256(prefix),
                "prefix_eos_sha256": token_ids_sha256(parent_tokens),
                "action_owner": selected_action["owner"],
                "action_row_sha256": selected_action["row_sha256"],
                "action_tokens": action_tokens, "action_sha256": token_ids_sha256(action_tokens),
                "target_gradient_count": ACTION_GRADIENTS,
                "cut_gradient_count": CUT_GRADIENTS,
                "cuts": [
                    {key: value for key, value in item.items() if key != "teacher_tokens"}
                    for item in cut_bindings
                ],
            }
            sealed_manifest: list[Any] = [linearization_manifest if rank == 0 else None]
            dist.broadcast_object_list(sealed_manifest, src=0)
            linearization_manifest = sealed_manifest[0]
            _agree_hash(linearization_manifest, label="selected action/prefix/cuts")

            gradient_packet: list[Any] = [None]
            direction: tuple[torch.Tensor, ...] | None = None
            if rank == 0:
                try:
                    teacher = [*prefix, *action_tokens]
                    binding = _complete_action_binding(prefix, action_tokens)
                    target_logits = base.full_root._teacher_forced_route_logits(
                        model=model, native_inputs=native_inputs,
                        route_tokens=teacher, pad_token_id=int(tokenizer.pad_token_id),
                    )
                    target_terms, target_panel = _complete_action_margin_terms(
                        target_logits, teacher_tokens=teacher, real_prefix_tokens=prefix,
                        natural_tokens=current_tokens, action_tokens=action_tokens, binding=binding,
                    )
                    target_gradients = _margin_gradients(target_terms, parameters)
                    counters["teacher_forwards"] += 1
                    counters["gradient_backwards"] += len(target_gradients)
                    del target_logits, target_terms
                    cut_logits = base.full_root._teacher_forced_route_logits(
                        model=model, native_inputs=native_inputs,
                        route_tokens=cut_bindings[0]["teacher_tokens"],
                        pad_token_id=int(tokenizer.pad_token_id),
                    )
                    cut_terms = torch.stack([projected._margin(cut_logits, item) for item in cut_bindings])
                    cut_margins = [float(value) for value in cut_terms.detach().cpu().tolist()]
                    cut_gradients = _margin_gradients(cut_terms, parameters)
                    counters["teacher_forwards"] += 1
                    counters["gradient_backwards"] += len(cut_gradients)
                    del cut_logits, cut_terms
                    gradients = [*target_gradients, *cut_gradients]
                    if len(gradients) != ACTIVE_GRADIENTS:
                        raise OtaSqpHold("HOLD: OTA linearization is not exactly 10 action + 7 cut gradients")
                    gram = _gradient_gram(gradients)
                    solution = _solve_final_margin_span(
                        [float(item["margin"]) for item in target_panel["per_token"]],
                        cut_margins, gram, eta=SAFETY_ETA, delta=TRUST_DELTA,
                    )
                    direction = _materialize_span_direction(gradients, solution["coefficients"])
                    direction_snapshot = projected._direction_snapshot(names, direction)
                    element_count = sum(parameter.numel() for parameter in parameters)
                    gradient_rows = [
                        {
                            "gradient_id": (
                                f"action-{index:02d}"
                                if index < ACTION_GRADIENTS else f"G{index - ACTION_GRADIENTS}"
                            ),
                            "ordinary_l2_norm": math.sqrt(max(0.0, float(gram[index, index]))),
                        }
                        for index in range(ACTIVE_GRADIENTS)
                    ]
                    linearization = {
                        **linearization_manifest,
                        "target_panel": target_panel, "cut_margins": cut_margins,
                        "gradient_payload": {
                            "shape": [ACTIVE_GRADIENTS, element_count],
                            "dtype": "torch.float32", "bytes": ACTIVE_GRADIENTS * element_count * 4,
                            "stored_in_receipt": False, "materialized_on_rank": 0,
                        },
                        "gradient_rows": gradient_rows,
                        "gram": {
                            "shape": list(gram.shape), "dtype": "float64", "bytes": int(gram.nbytes),
                            "sha256": base._hash(gram.tolist()), "matrix": gram.tolist(),
                            "minimum_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
                        },
                    }
                    gradient_packet[0] = {
                        "ok": True, "linearization": linearization,
                        "solver": solution, "direction": direction_snapshot,
                    }
                    del gradients, target_gradients, cut_gradients
                except BaseException as error:
                    gradient_packet[0] = {
                        "ok": False, "type": type(error).__name__, "error": str(error),
                    }
            dist.broadcast_object_list(gradient_packet, src=0)
            if not gradient_packet[0]["ok"]:
                raise OtaSqpHold(
                    f"HOLD: rank-zero OTA linearization failed: {gradient_packet[0]['type']}: {gradient_packet[0]['error']}"
                )
            linearization = gradient_packet[0]["linearization"]
            solver = gradient_packet[0]["solver"]
            direction_receipt = gradient_packet[0]["direction"]
            _agree_hash(linearization, label="Gram/linearization")
            _agree_hash(solver, label="SLSQP solution")
            if rank != 0:
                direction = tuple(torch.zeros_like(parameter) for parameter in parameters)
            assert direction is not None
            for value in direction:
                dist.broadcast(value, src=0)
            local_direction = projected._direction_snapshot(names, direction)
            if local_direction != direction_receipt:
                raise OtaSqpHold("HOLD: broadcast direction identity drifted")
            _agree_hash(local_direction, label="materialized direction")
            torch.cuda.empty_cache()

            clones = parallel._clone_parameters(parameters)
            parallel._restore_parameters(parameters, clones)
            radius = parallel._radius_for_rank(rank)
            _apply_trust_displacement(parameters, clones, direction, radius=radius)
            base.full_root._assert_full_root_sentinels(model, sentinels)
            try:
                counters["natural_decodes"] += 1
                candidate_packet, local_candidate, candidate_forwards = _evaluate_state(
                    model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                    pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                    parent_owners=parent_owners, action_tokens=action_tokens,
                    incumbent_tokens=current_tokens, label=f"ota-warm-radius-rank-{rank}",
                    radius=radius, rank=rank, cold_reloaded=False,
                )
                counters["teacher_forwards"] += candidate_forwards
            except BaseException as error:
                local_candidate = {
                    "rank": rank, "radius": radius, "cold_reloaded": False,
                    "matched_owner_ids": [], "matcher_committed_owner_ids": [],
                    "matcher_status_counts": {}, "matcher_optimum_cardinality": -1,
                    "parser_valid_prediction_count": -1, "parser_dropped_prediction_count": -1,
                    "raw_hard_counters": {}, "hard_debt_counts": {"evaluation_failure": 1},
                    "causal_hard_counter_count": -1, "designated_tail_debt_count": -1,
                    "working_corridor_passed": False, "row_aligned_eos": False,
                    "token_budget_exhausted": False, "complete_action_min": -1.0e30,
                    "first_natural_divergence": -1, "active_cut_margins": [],
                    "evaluation_error": {"type": type(error).__name__, "error": str(error)},
                }
            gathered_candidates: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_candidates, local_candidate)
            candidate_panel = list(gathered_candidates)
            selected_box: list[Any] = [None]
            if rank == 0:
                selected_box[0] = _select_warm_candidate(
                    candidate_panel, parent_owner_ids=parent_owners,
                    incumbent_action_min=incumbent_action_min,
                    incumbent_divergence=incumbent_divergence, eta=SAFETY_ETA,
                )
            dist.broadcast_object_list(selected_box, src=0)
            selected = selected_box[0]
            if selected is None:
                parallel._restore_parameters(parameters, clones)
                restored = base.full_root._full_root_surface_snapshot(names, parameters)[1]
                if restored != initial_surface:
                    raise OtaSqpHold("HOLD: rejected radius panel leaked a parameter mutation")
                base._surface_agreement(restored, dist.group.WORLD)
            else:
                selected_rank = int(selected["rank"])
                for parameter in parameters:
                    dist.broadcast(parameter.data, src=selected_rank)
                base._surface_agreement(
                    base.full_root._full_root_surface_snapshot(names, parameters)[1], dist.group.WORLD,
                )
                base.full_root._assert_full_root_sentinels(model, sentinels)
                checkpoint = output / "checkpoint-selected-smoke"
                saved = base._rank0_call(lambda model=model: {
                    "checkpoint": str(checkpoint),
                    "readback": base.full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=START_CHECKPOINT, destination=checkpoint,
                    ),
                })
                if rank == 0:
                    counters["checkpoint_saves"] += 1
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = base._frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise OtaSqpHold("HOLD: OTA smoke mutated the frozen non-DoRA surface")
            runtime = opened.receipt.to_artifact_dict()
            _agree_hash(runtime, label="runtime identity")

        del model, tokenizer, native_inputs, prompts, opened, parameters, names, sentinels, clones, direction, parameter, value
        torch.cuda.empty_cache()
        dist.barrier()
        cold_checkpoint = START_CHECKPOINT if saved is None else Path(str(saved["checkpoint"]))
        cold = base._rank0_call(lambda: _cold_state(
            cold_checkpoint, target=target, parent_owners=parent_owners,
            action_tokens=action_tokens, incumbent_tokens=current_tokens,
            radius=0.0 if selected is None else float(selected["radius"]),
            measure_candidate=selected is not None,
        ))
        if rank == 0:
            counters["cold_reloads"] += 1
            counters["natural_decodes"] += 1
            counters["teacher_forwards"] += int(cold["teacher_forward_count"])
        cold_gate: Mapping[str, Any] | None = None
        if selected is None:
            if (
                _ota_state_identity(cold["packet"]) != _ota_state_identity(baseline_packet)
                or cold["surface"] != initial_surface
                or cold["frozen_surface"] != FROZEN_SURFACE_SHA256
            ):
                raise OtaSqpHold("HOLD: no-selection terminal incumbent cold identity drifted")
            status = "bounded_negative_no_admissible_radius"
            stop_reason = "one_relinearization_eight_radius_panel_had_no_admissible_candidate"
        else:
            cold_candidate = dict(cold["candidate"])
            cold_gate = _candidate_gate(
                cold_candidate, parent_owner_ids=parent_owners,
                incumbent_action_min=incumbent_action_min,
                incumbent_divergence=incumbent_divergence, eta=SAFETY_ETA,
            )
            if (
                _ota_state_identity(cold["packet"]) != selected["packet_identity"]
                or cold["surface"] != terminal_surface
                or cold["frozen_surface"] != FROZEN_SURFACE_SHA256
                or cold_candidate.get("action_panel") != selected.get("action_panel")
                or cold_candidate.get("active_cut_margins") != selected.get("active_cut_margins")
            ):
                raise OtaSqpHold("HOLD: selected warm/cold save-reload identity drifted")
            if cold_gate["promotion"]["passed"]:
                status, stop_reason = "cold_match_level_promotion", "clean_cold_34_owner_promotion"
            elif cold_gate["working"]["passed"]:
                status, stop_reason = "smoke_selected_working_cold_reproduced", "one_relinearization_smoke_complete"
            else:
                raise OtaSqpHold(f"HOLD: selected cold checkpoint failed admission: {cold_gate}")

        local_resources = {
            "rank": rank,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(local_rank)),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(local_rank).total_memory),
        }
        gathered_resources: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_resources, local_resources)
        gathered_counters: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_counters, {"rank": rank, **counters})
        if rank == 0:
            totals = {
                key: sum(int(item[key]) for item in gathered_counters)
                for key in counters
            }
            receipt = {
                "schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID,
                "status": status, "stop_reason": stop_reason, "run_id": run_id,
                "runner_source_snapshot": source_snapshot,
                "bindings": bindings | {
                    "start_natural_sha256": START_NATURAL_SHA256,
                    "start_surface_sha256": START_SURFACE_SHA256,
                    "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
                    "optimization_prefix_sha256": START_PREFIX_SHA256,
                    "optimization_prefix_eos_sha256": START_PREFIX_EOS_SHA256,
                },
                "start_state": start,
                "protocol": {
                    "tier": "production_shaped_smoke_only", "world_size": WORLD_SIZE,
                    "relinearizations": 1, "foreground_updates_run": 0,
                    "radii_by_rank": {str(index): RADII[index] for index in range(WORLD_SIZE)},
                    "base_step_scale": base.LEARNING_RATE,
                    "safety_eta": SAFETY_ETA, "trust_delta": TRUST_DELTA,
                    "surface": "588 FP32 language DoRA tensors / 18006016 elements",
                    "optimizer": None, "ce": None, "fisher": None,
                    "fixed_gt32_G6_control": {"run": False, "reason": "main-arm integration smoke only"},
                    "r32": {"gate": "not_opened_by_smoke", "run": False},
                    "aligner": {"gate": "not_opened_by_smoke", "run": False},
                    "warm_is_screen_only": True,
                    "cold_selected_gate_is_authoritative": True,
                },
                "alias_catalog": alias_catalog, "ranking": ranking,
                "baseline_identity": _ota_state_identity(baseline_packet),
                "linearization": linearization, "solver": solver,
                "direction": direction_receipt, "candidate_panel": candidate_panel,
                "warm_selected_screen": selected, "authoritative_cold_gate": cold_gate,
                "initial_surface": initial_surface, "terminal_surface": terminal_surface,
                "saved_checkpoint": saved, "cold": cold,
                "counts": {"per_rank": gathered_counters, "total": totals},
                "resources": {
                    "per_rank": gathered_resources,
                    "max_peak_cuda_allocated_bytes": max(
                        int(item["peak_cuda_allocated_bytes"]) for item in gathered_resources
                    ),
                    "max_peak_cuda_reserved_bytes": max(
                        int(item["peak_cuda_reserved_bytes"]) for item in gathered_resources
                    ),
                },
                "wall_time_seconds": time.monotonic() - started,
                "runtime": runtime,
            }
            base._atomic_json(output / "receipt.json", receipt)
        dist.barrier()
        return output
    except BaseException as error:
        if dist.is_initialized() and dist.get_rank() == 0 and owns_output:
            base._atomic_json(output / "receipt.json", {
                "schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID,
                "status": "HOLD", "stop_reason": str(error), "run_id": run_id,
                "runner_sha256": base._sha256(Path(__file__)),
                "runner_source_snapshot": source_snapshot,
                "start_checkpoint": str(START_CHECKPOINT),
                "start_receipt_sha256": START_RECEIPT_SHA256,
                "saved_checkpoint": saved, "counts_rank0_partial": counters,
                "wall_time_seconds": time.monotonic() - started,
                "error_type": type(error).__name__, "traceback": traceback.format_exc(),
                "r32": {"run": False}, "aligner": {"run": False},
            })
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bindings", action="store_true", help="verify CPU-only frozen identities")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="run exactly one world8 r16 relinearization")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check_bindings:
        if args.run_id or args.smoke:
            raise SystemExit("--check-bindings is CPU-only and cannot be combined with a smoke run")
        print(json.dumps(_binding_receipt() | {"start_state": _start_state()}, indent=2, sort_keys=True))
        return
    if not args.smoke or not args.run_id:
        raise SystemExit("GPU execution requires both --smoke and --run-id")
    print(run(run_id=args.run_id))


if __name__ == "__main__":
    main()
