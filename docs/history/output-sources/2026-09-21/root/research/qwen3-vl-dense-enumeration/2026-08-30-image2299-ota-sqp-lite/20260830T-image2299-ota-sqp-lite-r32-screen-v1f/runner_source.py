#!/usr/bin/env python3
"""One-relinearization world8 Image2299 OTA-SQP-lite smoke or continuation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import minimize
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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
LEAD_ACCEPTED_SMOKE_V2_RECEIPT = (
    OUTPUT_ROOT / "20260830T-image2299-ota-sqp-lite-smoke-v2" / "receipt.json"
)
LEAD_ACCEPTED_SMOKE_V2_RECEIPT_SHA256 = (
    "ad51f46f76274fc2e68218f17ced0443982261e03204a4c9a34993c1dcde4ba0"
)
LEAD_ACCEPTED_COMPARATOR_V3B_RECEIPT = (
    OUTPUT_ROOT / "20260830T-image2299-ota-sqp-lite-continuation-v3b" / "receipt.json"
)
LEAD_ACCEPTED_COMPARATOR_V3B_RECEIPT_SHA256 = (
    "32f821b4f004625cd84e7599cf85fa202d57bbebb7b2b70920ec58c18f1934d8"
)
V2_CHECKPOINT = LEAD_ACCEPTED_SMOKE_V2_RECEIPT.parent / "checkpoint-selected-smoke"
V2_SURFACE_SHA256 = "b85952abf74e982980696753e006a1bc0b17386546480c53a1373eef5b8b1406"
V3B_ACTION_SHA256 = "d5372788c27310b6f969b0861a0498f165a28554204eda35bdda462670b81781"
V3B_ACTION_ROW_SHA256 = "eb439d76a1550961c15e005e39c5ff6c682ae21f0bc6382762033fd0bc22ba87"
V3B_TARGET_PANEL_SHA256 = "caf6604826dded3f228c747414e46f1dfa104fd6be149204a0f35a275b6e7e88"
V3B_CUT_MARGINS_SHA256 = "08a3ba7fd5aa262a258ddbd9a6781c6d16e72e6b3f26b5417420d51a96117f72"
V3B_ACTIVE_CUTS_SHA256 = "3e562b4990bb36fa50fa27ebcf5b3b00a82eff092a0c00b020ba2b02aef02c0e"
V3B_GRAM_SHA256 = "ffebe0172ca5289493874ad3166c5183eee93c8d4ac7494a3ea27d519120be61"
V3B_SOLVER_SHA256 = "1020cd3f30fdbd4eec2b2d9fe12691166706cd06aedd75517959d3e98caa9de1"
V3B_COMPARATOR_SHA256 = "ccf6d516eb004ad4f52b67ea814b5768676773b98ee4dd2f9ee152f537d92665"
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
# Numerical rank-width calibration; natural behavior remains bit-exact.
R32_MARGIN_DRIFT_TOLERANCE = SAFETY_ETA / 100.0
R32_OLD_GRAM_RTOL = 5e-5
R32_OLD_RHO_TOLERANCE = 5e-6
R32_RHO_ADVANTAGE_TOLERANCE = SOLVER_TOLERANCE
ADAPTER_TENSOR = Path("adapter/adapter_model.safetensors")
ADAPTER_CONFIG = Path("adapter/adapter_config.json")
SPECIAL_TENSOR = Path("special_token_embeddings/special_token_embeddings.safetensors")
SPECIAL_CONFIG = Path("special_token_embeddings/special_token_embeddings.json")
COUNTER_KEYS = (
    "teacher_forwards", "gradient_backwards", "natural_decodes",
    "checkpoint_saves", "cold_reloads",
)
CONTINUABLE_STATUSES = {
    "smoke_selected_working_cold_reproduced": "one_relinearization_smoke_complete",
    "continuation_selected_working_cold_reproduced": "one_step_continuation_complete",
}


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
        "checkpoint": str(START_CHECKPOINT), "accepted_update_ordinal": 0,
        "natural_token_ids": tokens, "natural_token_ids_sha256": START_NATURAL_SHA256,
        "optimization_prefix_token_ids": tokens[:OPTIMIZATION_PREFIX_TOKENS],
        "optimization_prefix_sha256": token_ids_sha256(tokens[:OPTIMIZATION_PREFIX_TOKENS]),
        "designated_tail_token_ids": tokens[OPTIMIZATION_PREFIX_TOKENS:-1],
        "parent_owner_ids": owners, "surface_sha256": START_SURFACE_SHA256,
        "frozen_surface_sha256": FROZEN_SURFACE_SHA256,
        "raw_hard_counters": raw, "causal_event": dict(warm_causal["first_event"]),
        "previous_action": None, "previous_cuts": [], "relinearization_signature": None,
    }


def _relinearization_signature(
    owner_ids: Sequence[str], first_natural_divergence: int,
) -> dict[str, Any]:
    owners = sorted(map(str, owner_ids))
    divergence = int(first_natural_divergence)
    if len(owners) < 33 or len(set(owners)) != len(owners) or divergence < 0:
        raise OtaSqpHold("HOLD: malformed owner-set/first-divergence signature")
    payload = {"owner_ids": owners, "first_natural_divergence": divergence}
    return {**payload, "sha256": base._hash(payload)}


def _parse_continuation_start_state(receipt_path: Path, receipt_sha256: str) -> dict[str, Any]:
    receipt_path = Path(receipt_path)
    supplied_sha = str(receipt_sha256).lower()
    if (
        len(supplied_sha) != 64
        or any(character not in "0123456789abcdef" for character in supplied_sha)
        or not receipt_path.is_file()
        or base._sha256(receipt_path) != supplied_sha
    ):
        raise OtaSqpHold("HOLD: supplied predecessor receipt/hash identity drifted")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise OtaSqpHold(f"HOLD: supplied predecessor receipt is unreadable: {error}") from error
    if not isinstance(receipt, dict):
        raise OtaSqpHold("HOLD: supplied predecessor receipt is not an object")

    status = str(receipt.get("status", ""))
    bindings = dict(receipt.get("bindings", {}))
    protocol = dict(receipt.get("protocol", {}))
    snapshot = dict(receipt.get("runner_source_snapshot", {}))
    snapshot_path = Path(str(snapshot.get("path", "")))
    snapshot_sha = str(snapshot.get("sha256", ""))
    saved = dict(receipt.get("saved_checkpoint", {}))
    readback = dict(saved.get("readback", {}))
    checkpoint = Path(str(saved.get("checkpoint", "")))
    source_checkpoint = Path(str(readback.get("parent_checkpoint", "")))
    cold = dict(receipt.get("cold", {}))
    cold_packet = dict(cold.get("packet", {}))
    evaluation = dict(cold_packet.get("evaluation", {}))
    causal = dict(cold_packet.get("causal_ledger", {}))
    joint = dict(evaluation.get("joint_gate", {}))
    cold_candidate = dict(cold.get("candidate", {}))
    warm = dict(receipt.get("warm_selected_screen", {}))
    gate = dict(receipt.get("authoritative_cold_gate", {}))
    terminal_surface = dict(receipt.get("terminal_surface", {}))
    cold_surface = dict(cold.get("surface", {}))
    initial_surface = dict(receipt.get("initial_surface", {}))
    start_state = dict(receipt.get("start_state", {}))
    linearization = dict(receipt.get("linearization", {}))
    ranking = dict(receipt.get("ranking", {}))
    selected_action = dict(ranking.get("selected", {}))
    counts = dict(receipt.get("counts", {}))
    per_rank = list(counts.get("per_rank", ()))
    totals = dict(counts.get("total", {}))

    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("unit_id") != UNIT_ID
        or status not in CONTINUABLE_STATUSES
        or receipt.get("stop_reason") != CONTINUABLE_STATUSES.get(status)
        or not isinstance(receipt.get("run_id"), str)
        or not receipt.get("run_id")
        or bindings.get("schema_version") != SCHEMA_VERSION
        or bindings.get("unit_id") != UNIT_ID
        or int(bindings.get("world_size", -1)) != WORLD_SIZE
        or bindings.get("prompt_token_sha256") != PROMPT_TOKEN_SHA256
        or bindings.get("image_sha256") != IMAGE_SHA256
        or set(map(str, bindings.get("missing_owner_ids", ()))) != set(MISSING_OWNERS)
        or int(bindings.get("active_gradient_count", -1)) != ACTIVE_GRADIENTS
        or int(bindings.get("optimization_prefix_tokens", -1)) != OPTIMIZATION_PREFIX_TOKENS
        or protocol.get("optimizer") is not None
        or protocol.get("ce") is not None
        or protocol.get("fisher") is not None
        or protocol.get("world_size") != WORLD_SIZE
        or protocol.get("relinearizations") != 1
        or protocol.get("trust_delta") != TRUST_DELTA
        or protocol.get("base_step_scale") != base.LEARNING_RATE
        or protocol.get("radii_by_rank") != {str(index): RADII[index] for index in range(WORLD_SIZE)}
        or not bool(protocol.get("warm_is_screen_only"))
        or not bool(protocol.get("cold_selected_gate_is_authoritative"))
        or dict(protocol.get("r32", {})).get("run") is not False
        or dict(protocol.get("aligner", {})).get("run") is not False
    ):
        raise OtaSqpHold("HOLD: predecessor OTA schema/status/protocol drifted")

    if (
        not snapshot_path.is_file()
        or len(snapshot_sha) != 64
        or base._sha256(snapshot_path) != snapshot_sha
        or not checkpoint.is_dir()
        or not source_checkpoint.is_dir()
        or readback.get("child_checkpoint") != str(checkpoint)
        or saved.get("checkpoint") != str(checkpoint)
        or bindings.get("start_checkpoint") != str(source_checkpoint)
        or readback.get("parent_checkpoint") != str(source_checkpoint)
    ):
        raise OtaSqpHold("HOLD: predecessor source snapshot/checkpoint binding drifted")
    payloads = {str(item.get("payload_id", "")): dict(item) for item in readback.get("payloads", ())}
    if (
        set(payloads) != {"adapter", "special_token_embeddings"}
        or int(payloads["adapter"].get("tensor_key_count", -1)) != 588
        or int(payloads["adapter"].get("changed_tensor_count", 0)) <= 0
        or int(payloads["special_token_embeddings"].get("tensor_key_count", -1)) != 1
        or int(payloads["special_token_embeddings"].get("changed_tensor_count", -1)) != 0
    ):
        raise OtaSqpHold("HOLD: predecessor checkpoint payload readback drifted")
    try:
        replayed_readback = base.full_root.checkpoint_pair_receipt(source_checkpoint, checkpoint)
    except BaseException as error:
        raise OtaSqpHold(f"HOLD: predecessor checkpoint CPU readback failed: {error}") from error
    if replayed_readback != readback:
        raise OtaSqpHold("HOLD: predecessor checkpoint CPU readback identity drifted")

    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    token_sha = token_ids_sha256(tokens)
    prefix = tokens[:OPTIMIZATION_PREFIX_TOKENS]
    prefix_sha = token_ids_sha256(prefix)
    owners = list(map(str, causal.get("locked_owner_ids", ())))
    causal_event = dict(causal.get("first_event") or {})
    raw = dict(joint.get("hard_raw_counters", {}))
    raw_keys = set(projected.WORKING_UNSUPPORTED_PERSON_HARD_COUNTERS)
    expected_owners = set(map(str, joint.get("expected_owner_ids", ())))
    if (
        len(tokens) != 34 * base.ROW_TOKENS + 1
        or tokens[-1:] != [base.EOS]
        or base.EOS in tokens[:-1]
        or len(prefix) != OPTIMIZATION_PREFIX_TOKENS
        or token_sha != evaluation.get("generated_token_ids_sha256")
        or token_sha != cold_candidate.get("generated_token_ids_sha256")
        or not bool(joint.get("natural_row_aligned_eos"))
        or not bool(cold_candidate.get("row_aligned_eos"))
        or bool(cold_candidate.get("token_budget_exhausted"))
        or len(owners) != 33
        or len(set(owners)) != 33
        or set(owners) != set(map(str, evaluation.get("matched_target_owner_ids", ())))
        or set(owners) != set(map(str, cold_candidate.get("matched_owner_ids", ())))
        or set(owners) != set(map(str, start_state.get("parent_owner_ids", ())))
        or set(owners) | set(MISSING_OWNERS) != expected_owners
        or set(owners) & set(MISSING_OWNERS)
        or set(raw) != raw_keys
        or any(not isinstance(value, (int, bool)) or int(value) < 0 for value in raw.values())
        or raw != cold_candidate.get("raw_hard_counters")
        or int(evaluation.get("hard_counter_count", -1)) != sum(map(int, raw.values()))
        or int(causal.get("causal_hard_counter_count", -1)) != 1
        or int(cold_candidate.get("causal_hard_counter_count", -1)) != 1
        or causal.get("token_ids_sha256") != token_sha
        or int(causal_event.get("event_start", -1)) != OPTIMIZATION_PREFIX_TOKENS
        or int(causal_event.get("row_index", -1)) != 33
        or causal_event.get("kind") not in {"duplicate", "near_miss"}
        or list(map(int, causal_event.get("bad_span_tokens", ())))
        != tokens[OPTIMIZATION_PREFIX_TOKENS:-1]
    ):
        raise OtaSqpHold("HOLD: predecessor cold Parent33/route/counter identity drifted")

    if (
        _ota_state_identity(cold_packet) != warm.get("packet_identity")
        or cold_candidate.get("packet_identity") != warm.get("packet_identity")
        or any(
            cold_candidate.get(key) != warm.get(key)
            for key in (
                "action_panel", "active_cut_bindings", "active_cut_margins",
                "matched_owner_ids", "matcher_committed_owner_ids", "matcher_status_counts",
                "matcher_optimum_cardinality", "parser_valid_prediction_count",
                "parser_dropped_prediction_count", "raw_hard_counters",
                "causal_hard_counter_count", "designated_tail_debt_count",
                "working_corridor_passed", "row_aligned_eos", "token_budget_exhausted",
                "complete_action_min", "first_natural_divergence",
                "generated_token_ids_sha256",
            )
        )
        or not bool(dict(gate.get("working", {})).get("passed"))
        or bool(dict(gate.get("promotion", {})).get("passed"))
        or dict(gate.get("working", {})).get("debt") != {}
    ):
        raise OtaSqpHold("HOLD: predecessor warm/cold authoritative working gate drifted")

    action_tokens = list(map(int, linearization.get("action_tokens", ())))
    action_panel = dict(cold_candidate.get("action_panel", {}))
    action_binding = dict(action_panel.get("binding", {}))
    old_tokens = list(map(int, start_state.get("natural_token_ids", ())))
    previous_cuts = list(cold_candidate.get("active_cut_bindings", ()))
    previous_cut_margins = list(map(float, cold_candidate.get("active_cut_margins", ())))
    rebound_cuts = [
        {key: value for key, value in item.items() if key != "teacher_tokens"}
        for item in _seed_active_cut_bindings(tokens)
    ]
    if (
        len(action_tokens) != ACTION_GRADIENTS
        or action_tokens[-1:] != [base.EOS]
        or token_ids_sha256(action_tokens) != linearization.get("action_sha256")
        or token_ids_sha256(action_tokens[:-1]) != linearization.get("action_row_sha256")
        or action_tokens[:-1] != list(map(int, selected_action.get("token_ids", ())))
        or linearization.get("action_owner") != selected_action.get("owner")
        or linearization.get("action_row_sha256") != selected_action.get("row_sha256")
        or int(linearization.get("prefix_token_count", -1)) != OPTIMIZATION_PREFIX_TOKENS
        or linearization.get("prefix_sha256") != token_ids_sha256(old_tokens[:OPTIMIZATION_PREFIX_TOKENS])
        or linearization.get("prefix_eos_sha256") != token_ids_sha256([
            *old_tokens[:OPTIMIZATION_PREFIX_TOKENS], base.EOS,
        ])
        or action_binding != _complete_action_binding(prefix, action_tokens)
        or len(list(action_panel.get("per_token", ()))) != ACTION_GRADIENTS
        or float(action_panel.get("minimum_margin", math.nan))
        != float(cold_candidate.get("complete_action_min", math.nan))
        or previous_cuts != rebound_cuts
        or [item.get("cut_id") for item in previous_cuts] != [f"G{index}" for index in range(CUT_GRADIENTS)]
        or len(previous_cut_margins) != CUT_GRADIENTS
        or any(not math.isfinite(value) or value < SAFETY_ETA for value in previous_cut_margins)
    ):
        raise OtaSqpHold("HOLD: predecessor action/cut/current-prefix evidence drifted")

    incumbent_divergence = base._exact_prefix(
        old_tokens[OPTIMIZATION_PREFIX_TOKENS:], action_tokens,
    )
    computed_gate = _candidate_gate(
        cold_candidate, parent_owner_ids=owners,
        incumbent_action_min=float(selected_action.get("complete_action_min", math.nan)),
        incumbent_divergence=incumbent_divergence, eta=SAFETY_ETA,
    )
    if computed_gate != gate:
        raise OtaSqpHold("HOLD: predecessor authoritative cold gate does not recompute")

    if (
        terminal_surface != cold_surface
        or terminal_surface.get("aggregate_sha256") != cold_surface.get("aggregate_sha256")
        or int(terminal_surface.get("tensor_count", -1)) != 588
        or int(terminal_surface.get("element_count", -1)) != 18_006_016
        or initial_surface.get("aggregate_sha256") != bindings.get("start_surface_sha256")
        or cold.get("frozen_surface") != bindings.get("frozen_surface_sha256")
        or start_state.get("frozen_surface_sha256") != bindings.get("frozen_surface_sha256")
    ):
        raise OtaSqpHold("HOLD: predecessor terminal/cold/frozen surface identity drifted")

    valid_count = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if (
        len(per_rank) != WORLD_SIZE
        or {int(item.get("rank", -1)) for item in per_rank} != set(range(WORLD_SIZE))
        or any(set(item) != {"rank", *COUNTER_KEYS} for item in per_rank)
        or any(not all(valid_count(item[key]) for key in COUNTER_KEYS) for item in per_rank)
        or set(totals) != set(COUNTER_KEYS)
        or any(not valid_count(totals[key]) for key in COUNTER_KEYS)
        or any(totals[key] != sum(int(item[key]) for item in per_rank) for key in COUNTER_KEYS)
        or totals != {
            "teacher_forwards": 124, "gradient_backwards": ACTIVE_GRADIENTS,
            "natural_decodes": 10, "checkpoint_saves": 1, "cold_reloads": 1,
        }
    ):
        raise OtaSqpHold("HOLD: predecessor full raw execution counts drifted")

    ordinal = receipt.get("accepted_update_ordinal")
    if ordinal is None:
        if (
            supplied_sha != LEAD_ACCEPTED_SMOKE_V2_RECEIPT_SHA256
            or status != "smoke_selected_working_cold_reproduced"
            or receipt.get("run_id") != LEAD_ACCEPTED_SMOKE_V2_RECEIPT.parent.name
        ):
            raise OtaSqpHold("HOLD: predecessor accepted update ordinal is absent")
        ordinal = 1
    if (
        not isinstance(ordinal, int) or isinstance(ordinal, bool)
        or ordinal < 1 or ordinal >= MAX_ACCEPTED_UPDATES
    ):
        raise OtaSqpHold("HOLD: predecessor accepted update ordinal drifted")

    signature = _relinearization_signature(
        owners, int(cold_candidate.get("first_natural_divergence", -1)),
    )
    previous_action = {
        "owner": str(linearization["action_owner"]),
        "row_sha256": str(linearization["action_row_sha256"]),
        "action_sha256": str(linearization["action_sha256"]),
        "action_tokens": action_tokens,
        "minimum_margin": float(cold_candidate["complete_action_min"]),
        "first_natural_divergence": int(cold_candidate["first_natural_divergence"]),
    }
    predecessor = {
        "receipt": str(receipt_path), "receipt_sha256": supplied_sha,
        "run_id": str(receipt["run_id"]), "status": status,
        "accepted_update_ordinal": ordinal, "checkpoint": str(checkpoint),
        "checkpoint_readback_sha256": base._hash(readback),
        "runner_source_snapshot": snapshot,
        "natural_token_ids_sha256": token_sha,
        "surface_sha256": str(terminal_surface["aggregate_sha256"]),
        "frozen_surface_sha256": str(cold["frozen_surface"]),
        "previous_action_sha256": base._hash(previous_action),
        "previous_cuts_sha256": base._hash({
            "bindings": previous_cuts, "margins": previous_cut_margins,
        }),
        "relinearization_signature": signature,
    }
    return {
        "receipt": str(receipt_path), "receipt_sha256": supplied_sha,
        "checkpoint": str(checkpoint), "accepted_update_ordinal": ordinal,
        "predecessor_status": status, "predecessor": predecessor,
        "natural_token_ids": tokens, "natural_token_ids_sha256": token_sha,
        "optimization_prefix_token_ids": prefix, "optimization_prefix_sha256": prefix_sha,
        "designated_tail_token_ids": tokens[OPTIMIZATION_PREFIX_TOKENS:-1],
        "parent_owner_ids": owners,
        "surface_sha256": str(terminal_surface["aggregate_sha256"]),
        "frozen_surface_sha256": str(cold["frozen_surface"]),
        "raw_hard_counters": raw, "causal_event": causal_event,
        "previous_action": previous_action,
        "previous_cuts": previous_cuts,
        "previous_cut_margins": previous_cut_margins,
        "relinearization_signature": signature,
    }


def _continuation_start_state(receipt_path: Path, receipt_sha256: str) -> dict[str, Any]:
    try:
        return _parse_continuation_start_state(receipt_path, receipt_sha256)
    except OtaSqpHold:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise OtaSqpHold(f"HOLD: malformed predecessor OTA receipt structure: {error}") from error


def _continuation_receipt_fields(
    *, start: Mapping[str, Any], status: str,
    terminal_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    previous_ordinal = int(start["accepted_update_ordinal"])
    attempted = previous_ordinal + 1
    accepted = status in {
        "continuation_selected_working_cold_reproduced",
        "continuation_cold_match_level_promotion",
    }
    terminal_signature = (
        _relinearization_signature(
            terminal_candidate.get("matched_owner_ids", ()),
            int(terminal_candidate.get("first_natural_divergence", -1)),
        )
        if terminal_candidate is not None else start["relinearization_signature"]
    )
    predecessor_signature = start["relinearization_signature"]
    return {
        "attempted_update_ordinal": attempted,
        "accepted_update_ordinal": attempted if accepted else previous_ordinal,
        "predecessor": start["predecessor"],
        "predecessor_relinearization_signature": predecessor_signature,
        "relinearization_signature": terminal_signature,
        "same_signature_as_predecessor": (
            terminal_signature["sha256"] == predecessor_signature["sha256"]
        ),
    }


def _resolve_start_context(
    start_receipt: Path | None, start_receipt_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if start_receipt is None and start_receipt_sha256 is None:
        return _binding_receipt(), _start_state()
    if start_receipt is None or start_receipt_sha256 is None:
        raise OtaSqpHold("HOLD: continuation requires both predecessor receipt path and SHA-256")
    start = _continuation_start_state(start_receipt, start_receipt_sha256)
    prefix = list(map(int, start["optimization_prefix_token_ids"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "execution_surface": "gpu_bounded_one_step_continuation_check_start_receipt_is_cpu_only",
        "start_checkpoint": start["checkpoint"],
        "start_receipt": start["receipt"],
        "start_receipt_sha256": start["receipt_sha256"],
        "start_natural_sha256": start["natural_token_ids_sha256"],
        "start_surface_sha256": start["surface_sha256"],
        "frozen_surface_sha256": start["frozen_surface_sha256"],
        "parent_route_sha256": PARENT_ROUTE_SHA256,
        "prompt_token_sha256": PROMPT_TOKEN_SHA256,
        "image_sha256": IMAGE_SHA256,
        "missing_owner_ids": sorted(MISSING_OWNERS),
        "world_size": WORLD_SIZE,
        "optimization_prefix_tokens": OPTIMIZATION_PREFIX_TOKENS,
        "optimization_prefix_sha256": token_ids_sha256(prefix),
        "optimization_prefix_eos_sha256": token_ids_sha256([*prefix, base.EOS]),
        "active_gradient_count": ACTIVE_GRADIENTS,
        "predecessor_accepted_update_ordinal": start["accepted_update_ordinal"],
    }, start


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
    ):
        raise OtaSqpHold("HOLD: malformed/non-PSD SQP-lite span problem")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    eigen_scale = max(1.0, float(np.abs(eigenvalues).max(initial=0.0)))
    if float(eigenvalues.min(initial=0.0)) < -SOLVER_TOLERANCE * eigen_scale:
        raise OtaSqpHold("HOLD: malformed/non-PSD SQP-lite span problem")
    eigen_threshold = max(np.finfo(np.float64).eps * active * eigen_scale, 1.0e-18)
    keep = eigenvalues > eigen_threshold
    if not bool(keep.any()):
        raise OtaSqpHold("HOLD: zero numerical rank in SQP-lite gradient span")
    positive_values = eigenvalues[keep]
    positive_vectors = eigenvectors[:, keep]
    projected_basis = positive_vectors * np.sqrt(positive_values)

    def objective(x: np.ndarray) -> float:
        return -float(x[-1])

    def constraints(x: np.ndarray) -> np.ndarray:
        unit, rho = x[:-1], x[-1]
        projected = delta * (projected_basis @ unit)
        return np.concatenate((
            target + projected[:count] - rho,
            cuts + projected[count:] - eta,
            [1.0 - unit @ unit],
        ))

    def constraint_jacobian(x: np.ndarray) -> np.ndarray:
        unit = x[:-1]
        rows = np.zeros((active + 1, len(unit) + 1), dtype=np.float64)
        rows[:count, :-1] = delta * projected_basis[:count]
        rows[:count, -1] = -1.0
        rows[count:active, :-1] = delta * projected_basis[count:]
        rows[-1, :-1] = -2.0 * unit
        return rows

    initial = np.zeros(int(keep.sum()) + 1, dtype=np.float64)
    initial[-1] = float(target.min())
    result = minimize(
        objective, initial, method="SLSQP",
        jac=lambda x: np.r_[np.zeros(len(x) - 1, dtype=np.float64), -1.0],
        constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success or result.x.shape != initial.shape or not np.isfinite(result.x).all():
        raise OtaSqpHold(f"HOLD: SQP-lite solver failed: {result.message}")
    unit, rho = result.x[:-1], float(result.x[-1])
    coefficients = positive_vectors @ (delta * unit / np.sqrt(positive_values))
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
        "solver": {
            "method": "scipy.optimize.SLSQP",
            "coordinates": "positive_Gram_eigenspace_unit_ball",
            "numerical_span_rank": int(keep.sum()),
            "eigenvalue_threshold": eigen_threshold,
            "iterations": int(result.nit),
        },
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
    ):
        raise OtaSqpHold("HOLD: r32 expansion is not function-preserving/live")
    return {"lora_A": expanded_a, "lora_B": expanded_b, "magnitude": magnitude.clone()}


def _r32_triplet_receipt(
    source_a: torch.Tensor, source_b: torch.Tensor, source_magnitude: torch.Tensor,
    child_a: torch.Tensor, child_b: torch.Tensor, child_magnitude: torch.Tensor,
) -> dict[str, int]:
    if (
        source_a.ndim != 2 or source_b.ndim != 2 or source_a.shape[0] != R16
        or source_b.shape[1] != R16 or source_b.shape[0] != source_magnitude.numel()
        or child_a.shape != (R32, source_a.shape[1])
        or child_b.shape != (source_b.shape[0], R32)
        or child_magnitude.shape != source_magnitude.shape
        or not all(
            bool(torch.isfinite(value).all().item())
            for value in (source_a, source_b, source_magnitude, child_a, child_b, child_magnitude)
        )
        or not torch.equal(child_a[:R16], source_a)
        or not torch.equal(child_b[:, :R16], source_b)
        or not torch.equal(child_magnitude, source_magnitude)
        or not bool((child_a[R16:] != 0).all().item())
        or torch.count_nonzero(child_b[:, R16:]).item() != 0
        or ALPHA16 / R16 != ALPHA32 / R32
    ):
        raise OtaSqpHold("HOLD: r32 adapter triplet is corrupt or dead")
    return {
        "source_element_count": source_a.numel() + source_b.numel() + source_magnitude.numel(),
        "child_element_count": child_a.numel() + child_b.numel() + child_magnitude.numel(),
        "old_coordinate_element_count": source_a.numel() + source_b.numel() + source_magnitude.numel(),
        "added_a_element_count": child_a[R16:].numel(),
        "added_b_element_count": child_b[:, R16:].numel(),
    }


def _r32_tensor_receipt(
    source: Mapping[str, torch.Tensor], child: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(source) != set(child) or len(source) != 588:
        raise OtaSqpHold("HOLD: r32 adapter tensor keys/count drifted")
    suffixes = (".lora_A.weight", ".lora_B.weight", ".lora_magnitude_vector")
    prefixes = sorted(key.removesuffix(suffixes[0]) for key in source if key.endswith(suffixes[0]))
    if len(prefixes) != 196 or any(
        {f"{prefix}{suffix}" for suffix in suffixes} - set(source) for prefix in prefixes
    ):
        raise OtaSqpHold("HOLD: r32 adapter is not exactly 196 DoRA triplets")
    totals = {
        "source_element_count": 0,
        "child_element_count": 0,
        "old_coordinate_element_count": 0,
        "added_a_element_count": 0,
        "added_b_element_count": 0,
    }
    for prefix in prefixes:
        values = _r32_triplet_receipt(
            source[f"{prefix}{suffixes[0]}"], source[f"{prefix}{suffixes[1]}"],
            source[f"{prefix}{suffixes[2]}"], child[f"{prefix}{suffixes[0]}"],
            child[f"{prefix}{suffixes[1]}"], child[f"{prefix}{suffixes[2]}"],
        )
        for key, value in values.items():
            totals[key] += int(value)
    if (
        totals["source_element_count"] != 18_006_016
        or totals["child_element_count"]
        != totals["old_coordinate_element_count"]
        + totals["added_a_element_count"] + totals["added_b_element_count"]
    ):
        raise OtaSqpHold("HOLD: r32 adapter element accounting drifted")
    return {
        "tensor_count": len(child), "module_triplet_count": len(prefixes), **totals,
    }


def _r32_checkpoint_receipt(source_checkpoint: Path, child_checkpoint: Path) -> dict[str, Any]:
    expected_files = {ADAPTER_TENSOR, ADAPTER_CONFIG, SPECIAL_TENSOR, SPECIAL_CONFIG}
    if (
        not source_checkpoint.is_dir() or not child_checkpoint.is_dir()
        or {path.relative_to(child_checkpoint) for path in child_checkpoint.rglob("*") if path.is_file()}
        != expected_files
    ):
        raise OtaSqpHold("HOLD: r32 checkpoint payload set drifted")
    try:
        source_config = json.loads((source_checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
        child_config = json.loads((child_checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise OtaSqpHold(f"HOLD: r32 adapter config is unreadable: {error}") from error
    expected_config = dict(source_config)
    expected_config.update({"r": R32, "lora_alpha": ALPHA32})
    if (
        not isinstance(source_config, dict) or not isinstance(child_config, dict)
        or source_config.get("r") != R16 or source_config.get("lora_alpha") != ALPHA16
        or source_config.get("use_dora") is not True
        or source_config.get("use_rslora") is not False
        or source_config.get("lora_dropout") != 0
        or child_config != expected_config
        or child_config.get("lora_alpha") / child_config.get("r") != ALPHA16 / R16
    ):
        raise OtaSqpHold("HOLD: r32 adapter config is not the frozen mechanical expansion")
    source_tensors = load_file(str(source_checkpoint / ADAPTER_TENSOR), device="cpu")
    child_tensors = load_file(str(child_checkpoint / ADAPTER_TENSOR), device="cpu")
    tensor_receipt = _r32_tensor_receipt(source_tensors, child_tensors)
    with safe_open(source_checkpoint / ADAPTER_TENSOR, framework="pt", device="cpu") as opened:
        source_metadata = opened.metadata() or {}
    with safe_open(child_checkpoint / ADAPTER_TENSOR, framework="pt", device="cpu") as opened:
        child_metadata = opened.metadata() or {}
    if source_metadata != child_metadata:
        raise OtaSqpHold("HOLD: r32 safetensors metadata drifted")
    payloads = {
        "source_adapter_tensor_sha256": base._sha256(source_checkpoint / ADAPTER_TENSOR),
        "child_adapter_tensor_sha256": base._sha256(child_checkpoint / ADAPTER_TENSOR),
        "source_adapter_config_sha256": base._sha256(source_checkpoint / ADAPTER_CONFIG),
        "child_adapter_config_sha256": base._sha256(child_checkpoint / ADAPTER_CONFIG),
        "source_special_tensor_sha256": base._sha256(source_checkpoint / SPECIAL_TENSOR),
        "child_special_tensor_sha256": base._sha256(child_checkpoint / SPECIAL_TENSOR),
        "source_special_config_sha256": base._sha256(source_checkpoint / SPECIAL_CONFIG),
        "child_special_config_sha256": base._sha256(child_checkpoint / SPECIAL_CONFIG),
    }
    if (
        payloads["source_special_tensor_sha256"] != payloads["child_special_tensor_sha256"]
        or payloads["source_special_config_sha256"] != payloads["child_special_config_sha256"]
        or (source_checkpoint / SPECIAL_TENSOR).read_bytes()
        != (child_checkpoint / SPECIAL_TENSOR).read_bytes()
        or (source_checkpoint / SPECIAL_CONFIG).read_bytes()
        != (child_checkpoint / SPECIAL_CONFIG).read_bytes()
    ):
        raise OtaSqpHold("HOLD: r32 expansion changed the special-token embedding payload")
    return {
        "source_checkpoint": str(source_checkpoint),
        "child_checkpoint": str(child_checkpoint),
        "adapter": tensor_receipt | {
            "source_config": {key: source_config.get(key) for key in (
                "r", "lora_alpha", "use_dora", "use_rslora", "lora_dropout",
                "base_model_name_or_path", "target_modules",
            )},
            "child_config": {key: child_config.get(key) for key in (
                "r", "lora_alpha", "use_dora", "use_rslora", "lora_dropout",
                "base_model_name_or_path", "target_modules",
            )},
            "safetensors_metadata": child_metadata,
        },
        "payload_sha256": payloads,
    }


def _materialize_r32_checkpoint(source_checkpoint: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or not source_checkpoint.is_dir():
        raise OtaSqpHold(f"HOLD: invalid/existing r32 checkpoint destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        source_tensors = load_file(str(source_checkpoint / ADAPTER_TENSOR), device="cpu")
        child_tensors: dict[str, torch.Tensor] = {}
        for key, value in source_tensors.items():
            if key.endswith(".lora_A.weight"):
                prefix = key.removesuffix(".lora_A.weight")
                expanded = _expand_r16_dora_factors(
                    value,
                    source_tensors[f"{prefix}.lora_B.weight"],
                    source_tensors[f"{prefix}.lora_magnitude_vector"],
                )
                child_tensors[key] = expanded["lora_A"].contiguous()
                child_tensors[f"{prefix}.lora_B.weight"] = expanded["lora_B"].contiguous()
                child_tensors[f"{prefix}.lora_magnitude_vector"] = expanded["magnitude"].contiguous()
        if set(child_tensors) != set(source_tensors):
            raise OtaSqpHold("HOLD: r32 expansion did not cover every adapter tensor")
        (temporary / "adapter").mkdir()
        (temporary / "special_token_embeddings").mkdir()
        with safe_open(source_checkpoint / ADAPTER_TENSOR, framework="pt", device="cpu") as opened:
            metadata = opened.metadata()
        save_file(child_tensors, str(temporary / ADAPTER_TENSOR), metadata=metadata)
        config = json.loads((source_checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
        config.update({"r": R32, "lora_alpha": ALPHA32})
        (temporary / ADAPTER_CONFIG).write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        shutil.copyfile(source_checkpoint / SPECIAL_TENSOR, temporary / SPECIAL_TENSOR)
        shutil.copyfile(source_checkpoint / SPECIAL_CONFIG, temporary / SPECIAL_CONFIG)
        receipt = _r32_checkpoint_receipt(source_checkpoint, temporary)
        os.replace(temporary, destination)
        receipt["child_checkpoint"] = str(destination)
        for relative, key in (
            (ADAPTER_TENSOR, "child_adapter_tensor_sha256"),
            (ADAPTER_CONFIG, "child_adapter_config_sha256"),
            (SPECIAL_TENSOR, "child_special_tensor_sha256"),
            (SPECIAL_CONFIG, "child_special_config_sha256"),
        ):
            if base._sha256(destination / relative) != receipt["payload_sha256"][key]:
                raise OtaSqpHold("HOLD: atomic r32 checkpoint rename/readback drifted")
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_hash_bound_receipt(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    supplied = str(digest).lower()
    if (
        len(supplied) != 64 or any(character not in "0123456789abcdef" for character in supplied)
        or not path.is_file() or base._sha256(path) != supplied
    ):
        raise OtaSqpHold(f"HOLD: {label} receipt/hash identity drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise OtaSqpHold(f"HOLD: {label} receipt is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise OtaSqpHold(f"HOLD: {label} receipt is not an object")
    return value


def _r32_comparator_receipt(
    receipt_path: Path, receipt_sha256: str, *, r16: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        Path(receipt_path).resolve() != LEAD_ACCEPTED_COMPARATOR_V3B_RECEIPT.resolve()
        or str(receipt_sha256).lower() != LEAD_ACCEPTED_COMPARATOR_V3B_RECEIPT_SHA256
    ):
        raise OtaSqpHold("HOLD: r32 screen requires the sealed v3b comparator receipt")
    receipt = _load_hash_bound_receipt(
        Path(receipt_path), receipt_sha256, label="sealed r16 v3b comparator",
    )
    bindings = dict(receipt.get("bindings", {}))
    protocol = dict(receipt.get("protocol", {}))
    linearization = dict(receipt.get("linearization", {}))
    target_panel = dict(linearization.get("target_panel", {}))
    cut_margins = list(map(float, linearization.get("cut_margins", ())))
    gram_record = dict(linearization.get("gram", {}))
    solver = dict(receipt.get("solver", {}))
    ranking = dict(receipt.get("ranking", {}))
    selected = dict(ranking.get("selected", {}))
    action_tokens = list(map(int, linearization.get("action_tokens", ())))
    active_cuts = [
        {key: value for key, value in item.items() if key != "teacher_tokens"}
        for item in _seed_active_cut_bindings(r16["natural_token_ids"])
    ]
    comparator_bundle = {
        "action_owner": linearization.get("action_owner"),
        "action_tokens": action_tokens,
        "prefix_sha256": linearization.get("prefix_sha256"),
        "prefix_eos_sha256": linearization.get("prefix_eos_sha256"),
        "target_panel": target_panel,
        "cut_margins": cut_margins,
        "active_cut_bindings": active_cuts,
        "gram_matrix": gram_record.get("matrix"),
        "solver": solver,
    }
    try:
        gram = np.asarray(gram_record.get("matrix"), dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(gram)
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        raise OtaSqpHold(f"HOLD: sealed r16 comparator Gram is malformed: {error}") from error
    cold = dict(receipt.get("cold", {}))
    cold_packet = dict(cold.get("packet", {}))
    cold_evaluation = dict(cold_packet.get("evaluation", {}))
    counts = dict(receipt.get("counts", {}))
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("unit_id") != UNIT_ID
        or receipt.get("status") != "continuation_bounded_negative_no_admissible_radius"
        or receipt.get("stop_reason")
        != "one_relinearization_eight_radius_panel_had_no_admissible_candidate"
        or bindings.get("start_receipt") != r16["receipt"]
        or bindings.get("start_receipt_sha256") != r16["receipt_sha256"]
        or bindings.get("start_checkpoint") != r16["checkpoint"]
        or bindings.get("start_surface_sha256") != V2_SURFACE_SHA256
        or bindings.get("start_natural_sha256") != START_NATURAL_SHA256
        or bindings.get("prompt_token_sha256") != PROMPT_TOKEN_SHA256
        or bindings.get("image_sha256") != IMAGE_SHA256
        or int(bindings.get("active_gradient_count", -1)) != ACTIVE_GRADIENTS
        or int(bindings.get("optimization_prefix_tokens", -1)) != OPTIMIZATION_PREFIX_TOKENS
        or protocol.get("optimizer") is not None or protocol.get("ce") is not None
        or protocol.get("fisher") is not None or protocol.get("trust_delta") != TRUST_DELTA
        or protocol.get("safety_eta") != SAFETY_ETA
        or dict(protocol.get("r32", {})).get("run") is not False
        or receipt.get("saved_checkpoint") is not None
        or receipt.get("attempted_update_ordinal") != 2
        or receipt.get("accepted_update_ordinal") != 1
        or linearization.get("action_owner") != "gt:2299:32"
        or action_tokens[-1:] != [base.EOS] or len(action_tokens) != ACTION_GRADIENTS
        or token_ids_sha256(action_tokens) != V3B_ACTION_SHA256
        or token_ids_sha256(action_tokens[:-1]) != V3B_ACTION_ROW_SHA256
        or linearization.get("action_sha256") != V3B_ACTION_SHA256
        or linearization.get("action_row_sha256") != V3B_ACTION_ROW_SHA256
        or selected.get("owner") != "gt:2299:32"
        or list(map(int, selected.get("token_ids", ()))) != action_tokens[:-1]
        or linearization.get("prefix_sha256") != r16["optimization_prefix_sha256"]
        or linearization.get("prefix_eos_sha256") != token_ids_sha256([
            *r16["optimization_prefix_token_ids"], base.EOS,
        ])
        or base._hash(target_panel) != V3B_TARGET_PANEL_SHA256
        or base._hash(cut_margins) != V3B_CUT_MARGINS_SHA256
        or base._hash(active_cuts) != V3B_ACTIVE_CUTS_SHA256
        or gram.shape != (ACTIVE_GRADIENTS, ACTIVE_GRADIENTS)
        or not np.isfinite(gram).all() or not np.allclose(gram, gram.T, atol=1e-10, rtol=0.0)
        or base._hash(gram.tolist()) != V3B_GRAM_SHA256
        or gram_record.get("sha256") != V3B_GRAM_SHA256
        or not math.isclose(
            float(gram_record.get("minimum_eigenvalue", math.nan)), float(eigenvalues.min()),
            rel_tol=0.0, abs_tol=SOLVER_TOLERANCE,
        )
        or base._hash(solver) != V3B_SOLVER_SHA256
        or solver.get("eta") != SAFETY_ETA or solver.get("delta") != TRUST_DELTA
        or base._hash(comparator_bundle) != V3B_COMPARATOR_SHA256
        or dict(receipt.get("initial_surface", {})).get("aggregate_sha256") != V2_SURFACE_SHA256
        or dict(receipt.get("terminal_surface", {})).get("aggregate_sha256") != V2_SURFACE_SHA256
        or dict(cold.get("surface", {})).get("aggregate_sha256") != V2_SURFACE_SHA256
        or cold.get("frozen_surface") != FROZEN_SURFACE_SHA256
        or cold_evaluation.get("generated_token_ids_sha256") != START_NATURAL_SHA256
        or counts.get("total") != {
            "teacher_forwards": 122, "gradient_backwards": ACTIVE_GRADIENTS,
            "natural_decodes": 10, "checkpoint_saves": 0, "cold_reloads": 1,
        }
    ):
        raise OtaSqpHold("HOLD: sealed r16 v3b comparator contract drifted")
    target_margins = [float(item["margin"]) for item in target_panel.get("per_token", ())]
    if len(target_margins) != ACTION_GRADIENTS or len(cut_margins) != CUT_GRADIENTS:
        raise OtaSqpHold("HOLD: sealed r16 comparator panel cardinality drifted")
    return {
        "receipt": str(receipt_path), "receipt_sha256": str(receipt_sha256).lower(),
        "comparator_bundle_sha256": V3B_COMPARATOR_SHA256,
        "action_owner": "gt:2299:32", "action_tokens": action_tokens,
        "action_sha256": V3B_ACTION_SHA256, "action_row_sha256": V3B_ACTION_ROW_SHA256,
        "prefix_sha256": linearization["prefix_sha256"],
        "prefix_eos_sha256": linearization["prefix_eos_sha256"],
        "target_panel": target_panel, "target_panel_sha256": V3B_TARGET_PANEL_SHA256,
        "target_margins": target_margins,
        "cut_bindings": active_cuts, "active_cuts_sha256": V3B_ACTIVE_CUTS_SHA256,
        "cut_margins": cut_margins, "cut_margins_sha256": V3B_CUT_MARGINS_SHA256,
        "gram": gram, "gram_sha256": V3B_GRAM_SHA256,
        "gram_minimum_eigenvalue": float(eigenvalues.min()),
        "solver": solver, "solver_sha256": V3B_SOLVER_SHA256,
    }


def _r32_inputs(
    r16_receipt: Path, r16_receipt_sha256: str,
    comparator_receipt: Path, comparator_receipt_sha256: str,
) -> dict[str, Any]:
    if (
        Path(r16_receipt).resolve() != LEAD_ACCEPTED_SMOKE_V2_RECEIPT.resolve()
        or str(r16_receipt_sha256).lower() != LEAD_ACCEPTED_SMOKE_V2_RECEIPT_SHA256
    ):
        raise OtaSqpHold("HOLD: r32 screen requires the lead-accepted v2 r16 receipt")
    r16 = _continuation_start_state(Path(r16_receipt), r16_receipt_sha256)
    if (
        r16["checkpoint"] != str(V2_CHECKPOINT)
        or r16["surface_sha256"] != V2_SURFACE_SHA256
        or r16["natural_token_ids_sha256"] != START_NATURAL_SHA256
        or r16["frozen_surface_sha256"] != FROZEN_SURFACE_SHA256
    ):
        raise OtaSqpHold("HOLD: r32 v2 source checkpoint/route/surface drifted")
    source_receipt = _load_hash_bound_receipt(
        Path(r16_receipt), r16_receipt_sha256, label="lead-accepted v2 r16",
    )
    r16["reference_packet_identity"] = _ota_state_identity(source_receipt["cold"]["packet"])
    comparator = _r32_comparator_receipt(
        comparator_receipt, comparator_receipt_sha256, r16=r16,
    )
    return {"r16": r16, "comparator": comparator}


def _r32_trainable_surface(
    model: Any, *, expected_tensor_count: int, expected_element_count: int,
) -> tuple[list[str], list[torch.nn.Parameter]]:
    names: list[str] = []
    parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = (
            ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name
        )
        parameter.requires_grad_(enabled)
        if enabled:
            if parameter.dtype != torch.float32:
                raise OtaSqpHold(f"HOLD: r32 FP32 DoRA surface drifted at {name}")
            names.append(name)
            parameters.append(parameter)
    if (
        len(parameters) != expected_tensor_count
        or sum(parameter.numel() for parameter in parameters) != expected_element_count
        or sum(".lora_A." in name for name in names) != 196
        or sum(".lora_B." in name for name in names) != 196
        or sum(".lora_magnitude_vector" in name for name in names) != 196
    ):
        raise OtaSqpHold("HOLD: loaded r32 DoRA tensor/element surface drifted")
    return names, parameters


def _r32_coordinate_grams(
    names: Sequence[str], gradients: Sequence[Sequence[torch.Tensor]],
) -> dict[str, Any]:
    if len(gradients) != ACTIVE_GRADIENTS or not names or len(names) != len(gradients[0]):
        raise OtaSqpHold("HOLD: r32 coordinate split received a malformed gradient surface")
    device = gradients[0][0].device
    old = torch.zeros((ACTIVE_GRADIENTS, ACTIVE_GRADIENTS), dtype=torch.float64, device=device)
    added = torch.zeros_like(old)
    counts = {"a": 0, "b": 0, "magnitude": 0, "old": 0, "added_a": 0, "added_b": 0}
    new_a_nonzero = 0
    new_b_live_rows = [False] * ACTIVE_GRADIENTS
    live_new_b_parameters = 0
    for parameter_index, name in enumerate(names):
        blocks = [row[parameter_index] for row in gradients]
        if any(value.shape != blocks[0].shape for value in blocks):
            raise OtaSqpHold("HOLD: r32 gradient row shapes drifted")
        if ".lora_A." in name:
            if blocks[0].ndim != 2 or blocks[0].shape[0] != R32:
                raise OtaSqpHold("HOLD: r32 A-coordinate shape drifted")
            old_blocks = torch.stack([value[:R16].flatten() for value in blocks]).double()
            added_blocks = torch.stack([value[R16:].flatten() for value in blocks]).double()
            counts["a"] += 1
            counts["old"] += blocks[0][:R16].numel()
            counts["added_a"] += blocks[0][R16:].numel()
            new_a_nonzero += int(torch.count_nonzero(added_blocks).item())
        elif ".lora_B." in name:
            if blocks[0].ndim != 2 or blocks[0].shape[1] != R32:
                raise OtaSqpHold("HOLD: r32 B-coordinate shape drifted")
            old_blocks = torch.stack([value[:, :R16].flatten() for value in blocks]).double()
            added_blocks = torch.stack([value[:, R16:].flatten() for value in blocks]).double()
            counts["b"] += 1
            counts["old"] += blocks[0][:, :R16].numel()
            counts["added_b"] += blocks[0][:, R16:].numel()
            live = added_blocks.square().sum(dim=1) > 0
            live_new_b_parameters += int(bool(live.any().item()))
            new_b_live_rows = [previous or bool(value) for previous, value in zip(
                new_b_live_rows, live.tolist(), strict=True,
            )]
        elif "lora_magnitude_vector" in name:
            old_blocks = torch.stack([value.flatten() for value in blocks]).double()
            added_blocks = torch.zeros((ACTIVE_GRADIENTS, 0), dtype=torch.float64, device=device)
            counts["magnitude"] += 1
            counts["old"] += blocks[0].numel()
        else:
            raise OtaSqpHold(f"HOLD: non-DoRA parameter entered r32 gradient surface: {name}")
        old.addmm_(old_blocks, old_blocks.T)
        if added_blocks.shape[1]:
            added.addmm_(added_blocks, added_blocks.T)
    if (
        counts["a"] != counts["b"] or counts["a"] != counts["magnitude"]
        or counts["a"] != 196 or new_a_nonzero != 0
        or not all(new_b_live_rows) or live_new_b_parameters == 0
    ):
        raise OtaSqpHold("HOLD: r32 added coordinates are dead or new-A gradients are nonzero")
    old_array, added_array = old.cpu().numpy(), added.cpu().numpy()
    full_array = old_array + added_array
    return {
        "old_gram": old_array, "added_gram": added_array, "full_gram": full_array,
        "coordinate_counts": counts,
        "new_a_nonzero_gradient_elements": new_a_nonzero,
        "new_b_live_gradient_rows": sum(new_b_live_rows),
        "live_new_b_parameter_count": live_new_b_parameters,
    }


def _r32_tangent_gate(
    *, comparator_gram: np.ndarray, old_gram: np.ndarray, added_gram: np.ndarray,
    old_solver: Mapping[str, Any], full_solver: Mapping[str, Any], comparator_rho: float,
) -> dict[str, Any]:
    old_drift = old_gram - comparator_gram
    comparator_spectral = max(1.0, float(np.linalg.norm(comparator_gram, ord=2)))
    spectral_relative_drift = float(np.linalg.norm(old_drift, ord=2)) / comparator_spectral
    max_absolute_scaled_drift = float(np.max(np.abs(old_drift))) / comparator_spectral
    if (
        spectral_relative_drift > R32_OLD_GRAM_RTOL
        or max_absolute_scaled_drift > R32_OLD_GRAM_RTOL
    ):
        raise OtaSqpHold("HOLD: r32 old-coordinate Gram drifted from the sealed r16 comparator")
    old_rho = float(old_solver.get("rho", math.nan))
    full_rho = float(full_solver.get("rho", math.nan))
    if not math.isclose(old_rho, comparator_rho, rel_tol=0.0, abs_tol=R32_OLD_RHO_TOLERANCE):
        raise OtaSqpHold("HOLD: r32 old-coordinate SQP drifted from the sealed r16 comparator")
    eigenvalues = np.linalg.eigvalsh((added_gram + added_gram.T) / 2.0)
    scale = max(1.0, float(np.abs(eigenvalues).max(initial=0.0)))
    psd = float(eigenvalues.min(initial=0.0)) >= -SOLVER_TOLERANCE * scale
    nonzero = float(eigenvalues.max(initial=0.0)) > np.finfo(np.float64).eps * len(eigenvalues) * scale
    advantage = full_rho > comparator_rho + R32_RHO_ADVANTAGE_TOLERANCE
    if not psd:
        raise OtaSqpHold("HOLD: r32 added-coordinate Gram is not numerically PSD")
    return {
        "status": "r32_tangent_gate_open" if nonzero and advantage else "r32_tangent_no_feasible_advantage",
        "added_component_established": bool(nonzero and advantage),
        "added_gram_numerically_psd": bool(psd),
        "added_gram_nonzero": bool(nonzero),
        "added_gram_minimum_eigenvalue": float(eigenvalues.min()),
        "added_gram_maximum_eigenvalue": float(eigenvalues.max()),
        "sealed_r16_rho": comparator_rho,
        "r32_old_coordinate_rho": old_rho,
        "r32_full_rho": full_rho,
        "r32_minus_r16_rho": full_rho - comparator_rho,
        "strict_advantage_tolerance": R32_RHO_ADVANTAGE_TOLERANCE,
        "strict_feasible_advantage": bool(advantage),
        "old_gram_max_absolute_drift": float(np.max(np.abs(old_gram - comparator_gram))),
        "old_gram_spectral_relative_drift": spectral_relative_drift,
        "old_gram_max_absolute_scaled_drift": max_absolute_scaled_drift,
        "old_gram_max_scaled_relative_drift": float(np.max(
            np.abs(old_gram - comparator_gram) / np.maximum(1.0, np.abs(comparator_gram))
        )),
    }


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


def _r32_step_zero_identity(packet: Mapping[str, Any], r16: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(packet.get("evaluation", {}))
    causal = dict(packet.get("causal_ledger", {}))
    joint = dict(evaluation.get("joint_gate", {}))
    tokens = list(map(int, evaluation.get("generated_token_ids", ())))
    owners = list(map(str, evaluation.get("matched_target_owner_ids", ())))
    raw = dict(joint.get("hard_raw_counters", {}))
    event = dict(causal.get("first_event") or {})
    if (
        _ota_state_identity(packet) != r16["reference_packet_identity"]
        or tokens != list(map(int, r16["natural_token_ids"]))
        or token_ids_sha256(tokens) != START_NATURAL_SHA256
        or set(owners) != set(map(str, r16["parent_owner_ids"]))
        or raw != r16["raw_hard_counters"]
        or event != r16["causal_event"]
        or not bool(joint.get("natural_row_aligned_eos"))
        or tokens[-1:] != [base.EOS] or base.EOS in tokens[:-1]
    ):
        raise OtaSqpHold("HOLD: r32 cold step-zero natural behavior changed from v2")
    return {
        "exact_packet_identity": True,
        "natural_token_ids_sha256": START_NATURAL_SHA256,
        "natural_tokens_exact": True,
        "owner_set_exact": True,
        "owner_ids": sorted(owners),
        "raw_hard_counters_exact": True,
        "raw_hard_counters": raw,
        "causal_event_exact": True,
        "causal_event": event,
        "natural_row_aligned_eos": True,
    }


def _r32_margin_drift(
    *, live_target_panel: Mapping[str, Any], live_cut_margins: Sequence[float],
    comparator: Mapping[str, Any],
) -> dict[str, Any]:
    reference_panel = dict(comparator["target_panel"])
    live_records = list(live_target_panel.get("per_token", ()))
    reference_records = list(reference_panel.get("per_token", ()))

    def structure(record: Mapping[str, Any]) -> dict[str, Any]:
        value = {key: item for key, item in record.items() if key != "margin"}
        if "required_person_competitors" in value:
            value["required_person_competitors"] = sorted(value["required_person_competitors"])
        return value

    if (
        dict(live_target_panel.get("binding", {})) != dict(reference_panel.get("binding", {}))
        or live_target_panel.get("weakest_relative_position")
        != reference_panel.get("weakest_relative_position")
        or len(live_records) != ACTION_GRADIENTS or len(reference_records) != ACTION_GRADIENTS
        or [structure(item) for item in live_records] != [structure(item) for item in reference_records]
        or len(live_cut_margins) != CUT_GRADIENTS
    ):
        raise OtaSqpHold("HOLD: r32 action/competitor/cut panel identity drifted")
    target_drifts = [
        float(live["margin"]) - float(reference["margin"])
        for live, reference in zip(live_records, reference_records, strict=True)
    ]
    person_drifts = {
        key: float(live_records[1]["required_person_competitors"][key])
        - float(reference_records[1]["required_person_competitors"][key])
        for key in ("tie", "chair")
    }
    cut_drifts = [
        float(live) - float(reference)
        for live, reference in zip(live_cut_margins, comparator["cut_margins"], strict=True)
    ]
    all_drifts = [*target_drifts, *person_drifts.values(), *cut_drifts]
    maximum = max(map(abs, all_drifts), default=0.0)
    return {
        "tolerance": R32_MARGIN_DRIFT_TOLERANCE,
        "target_margin_drifts": target_drifts,
        "required_person_competitor_margin_drifts": person_drifts,
        "cut_margin_drifts": cut_drifts,
        "maximum_absolute_drift": maximum,
        "within_tolerance": maximum <= R32_MARGIN_DRIFT_TOLERANCE,
    }


def screen_r32(
    *, run_id: str, r16_receipt: Path, r16_receipt_sha256: str,
    comparator_receipt: Path, comparator_receipt_sha256: str,
) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise OtaSqpHold("HOLD: r32 tangent screen is exactly one process/GPU")
    run_id = base._safe_run_id(run_id)
    output = OUTPUT_ROOT / run_id
    started = time.monotonic()
    source_snapshot = _prepare_output(output)
    stage = "input_contract"
    expansion: Mapping[str, Any] | None = None
    inputs: Mapping[str, Any] | None = None
    surface: Mapping[str, Any] | None = None
    frozen_surface: str | None = None
    linearization: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    counters = {
        "teacher_forwards": 0, "gradient_backwards": 0, "natural_decodes": 0,
        "cold_reloads": 0, "mechanical_checkpoint_materializations": 0,
        "trained_checkpoint_saves": 0, "parameter_updates": 0,
    }
    try:
        inputs = _r32_inputs(
            r16_receipt, r16_receipt_sha256, comparator_receipt, comparator_receipt_sha256,
        )
        r16, comparator = inputs["r16"], inputs["comparator"]
        stage = "mechanical_expansion"
        checkpoint = output / "checkpoint-r32-step-zero"
        expansion = _materialize_r32_checkpoint(Path(r16["checkpoint"]), checkpoint)
        counters["mechanical_checkpoint_materializations"] = 1
        expected_elements = int(expansion["adapter"]["child_element_count"])
        stage = "cold_hf_step_zero"
        if not torch.cuda.is_available():
            raise OtaSqpHold("HOLD: r32 tangent screen requires one CUDA device")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        _parent_admission, _parent_setup, target = base._bindings()
        setup = base._setup_for_checkpoint(checkpoint)
        with base.open_backend_session(setup["frontend"].launch) as opened:
            counters["cold_reloads"] = 1
            if type(opened) is not base.HFBackendSession or opened._model is None or opened._tokenizer is None:
                raise OtaSqpHold("HOLD: r32 screen requires the production concrete FP32 HF backend")
            model, tokenizer = opened._model, opened._tokenizer
            model.eval()
            native_inputs, prompts, _grids, _media = opened._materialize_native_inputs(
                setup["requests"][:1],
            )
            if (
                token_ids_sha256(prompts[0]) != PROMPT_TOKEN_SHA256
                or base._sha256(Path(setup["raw_example"].image.path)) != IMAGE_SHA256
            ):
                raise OtaSqpHold("HOLD: r32 production prompt/image identity drifted")
            names, parameters = _r32_trainable_surface(
                model,
                expected_tensor_count=int(expansion["adapter"]["tensor_count"]),
                expected_element_count=expected_elements,
            )
            if (
                len(parameters) != int(expansion["adapter"]["tensor_count"])
                or sum(parameter.numel() for parameter in parameters) != expected_elements
                or any(parameter.dtype != torch.float32 for parameter in parameters)
            ):
                raise OtaSqpHold("HOLD: loaded r32 DoRA tensor count/dtype/element surface drifted")
            sentinels = base.full_root._full_root_nontrainable_sentinels(model)
            surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if (
                int(surface.get("tensor_count", -1)) != int(expansion["adapter"]["tensor_count"])
                or int(surface.get("element_count", -1)) != expected_elements
            ):
                raise OtaSqpHold("HOLD: loaded r32 surface snapshot accounting drifted")
            frozen_surface = base._frozen_surface(model, names)
            if frozen_surface != r16["frozen_surface_sha256"]:
                raise OtaSqpHold("HOLD: r32 expansion changed the non-DoRA frozen surface")
            counters["natural_decodes"] = 1
            packet = projected._state_packet(
                model=model, tokenizer=tokenizer, native_inputs=native_inputs,
                pad=int(tokenizer.pad_token_id), target=target, raw_example=setup["raw_example"],
                parent_owners=r16["parent_owner_ids"], alias_tokens=comparator["action_tokens"][:-1],
                label="ota-r32-cold-step-zero",
            )
            step_zero_identity = _r32_step_zero_identity(packet, r16)
            prefix = list(map(int, r16["optimization_prefix_token_ids"]))
            action_tokens = list(map(int, comparator["action_tokens"]))
            binding = _complete_action_binding(prefix, action_tokens)
            if (
                binding != comparator["target_panel"]["binding"]
                or token_ids_sha256(prefix) != comparator["prefix_sha256"]
            ):
                raise OtaSqpHold("HOLD: r32 real-prefix/action binding drifted")
            stage = "matched_r32_gradients"
            teacher = [*prefix, *action_tokens]
            target_logits = base.full_root._teacher_forced_route_logits(
                model=model, native_inputs=native_inputs, route_tokens=teacher,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            target_terms, target_panel = _complete_action_margin_terms(
                target_logits, teacher_tokens=teacher, real_prefix_tokens=prefix,
                natural_tokens=r16["natural_token_ids"], action_tokens=action_tokens,
                binding=binding,
            )
            counters["teacher_forwards"] += 1
            del target_logits
            cut_bindings = _seed_active_cut_bindings(r16["natural_token_ids"])
            stripped_cuts = [
                {key: value for key, value in item.items() if key != "teacher_tokens"}
                for item in cut_bindings
            ]
            if base._hash(stripped_cuts) != comparator["active_cuts_sha256"]:
                raise OtaSqpHold("HOLD: r32 G0-G6 binding hash drifted")
            cut_logits = base.full_root._teacher_forced_route_logits(
                model=model, native_inputs=native_inputs,
                route_tokens=cut_bindings[0]["teacher_tokens"],
                pad_token_id=int(tokenizer.pad_token_id),
            )
            cut_terms = torch.stack([projected._margin(cut_logits, item) for item in cut_bindings])
            live_cut_margins = [float(value) for value in cut_terms.detach().cpu().tolist()]
            counters["teacher_forwards"] += 1
            del cut_logits
            margin_drift = _r32_margin_drift(
                live_target_panel=target_panel, live_cut_margins=live_cut_margins,
                comparator=comparator,
            )
            linearization = {
                "action_owner": comparator["action_owner"],
                "action_sha256": comparator["action_sha256"],
                "prefix_sha256": comparator["prefix_sha256"],
                "target_panel": target_panel,
                "cut_margins": live_cut_margins,
                "step_zero_margin_drift": margin_drift,
            }
            if not margin_drift["within_tolerance"]:
                raise OtaSqpHold("HOLD: r32 step-zero teacher target/cut margin drift exceeded tolerance")
            target_gradients = _margin_gradients(target_terms, parameters)
            cut_gradients = _margin_gradients(cut_terms, parameters)
            counters["gradient_backwards"] += len(target_gradients) + len(cut_gradients)
            del target_terms, cut_terms
            gradients = [*target_gradients, *cut_gradients]
            if len(gradients) != ACTIVE_GRADIENTS:
                raise OtaSqpHold("HOLD: r32 screen did not compute exactly 10 action + G0-G6 gradients")
            coordinate = _r32_coordinate_grams(names, gradients)
            if (
                coordinate["coordinate_counts"]["old"]
                != int(expansion["adapter"]["old_coordinate_element_count"])
                or coordinate["coordinate_counts"]["added_a"]
                != int(expansion["adapter"]["added_a_element_count"])
                or coordinate["coordinate_counts"]["added_b"]
                != int(expansion["adapter"]["added_b_element_count"])
            ):
                raise OtaSqpHold("HOLD: r32 loaded/expanded coordinate accounting disagreed")
            stage = "matched_sqp_comparison"
            old_solver = _solve_final_margin_span(
                comparator["target_margins"], comparator["cut_margins"],
                coordinate["old_gram"], eta=SAFETY_ETA, delta=TRUST_DELTA,
            )
            full_solver = _solve_final_margin_span(
                comparator["target_margins"], comparator["cut_margins"],
                coordinate["full_gram"], eta=SAFETY_ETA, delta=TRUST_DELTA,
            )
            old_gram_drift = coordinate["old_gram"] - comparator["gram"]
            linearization = {
                **dict(linearization or {}),
                "coordinate_split": {
                    **coordinate["coordinate_counts"],
                    "new_a_nonzero_gradient_elements": coordinate["new_a_nonzero_gradient_elements"],
                    "new_b_live_gradient_rows": coordinate["new_b_live_gradient_rows"],
                    "live_new_b_parameter_count": coordinate["live_new_b_parameter_count"],
                },
                "gram": {
                    label: {
                        "shape": list(matrix.shape), "matrix": matrix.tolist(),
                        "sha256": base._hash(matrix.tolist()),
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
                        "maximum_eigenvalue": float(np.linalg.eigvalsh(matrix).max()),
                    }
                    for label, matrix in (
                        ("sealed_r16", comparator["gram"]),
                        ("old_r16_coordinates", coordinate["old_gram"]),
                        ("added_coordinates", coordinate["added_gram"]),
                        ("full_r32", coordinate["full_gram"]),
                    )
                },
                "old_coordinate_solver": old_solver,
                "full_r32_solver": full_solver,
                "old_comparator_drift": {
                    "gram_max_absolute": float(np.max(np.abs(old_gram_drift))),
                    "gram_max_scaled_relative": float(np.max(
                        np.abs(old_gram_drift) / np.maximum(1.0, np.abs(comparator["gram"]))
                    )),
                    "gram_spectral_relative": float(np.linalg.norm(old_gram_drift, ord=2))
                    / max(1.0, float(np.linalg.norm(comparator["gram"], ord=2))),
                    "gram_max_absolute_scaled": float(np.max(np.abs(old_gram_drift)))
                    / max(1.0, float(np.linalg.norm(comparator["gram"], ord=2))),
                    "rho": float(old_solver["rho"]) - float(comparator["solver"]["rho"]),
                    "gram_rtol": R32_OLD_GRAM_RTOL,
                    "rho_tolerance": R32_OLD_RHO_TOLERANCE,
                },
            }
            gate = _r32_tangent_gate(
                comparator_gram=comparator["gram"], old_gram=coordinate["old_gram"],
                added_gram=coordinate["added_gram"], old_solver=old_solver,
                full_solver=full_solver, comparator_rho=float(comparator["solver"]["rho"]),
            )
            base.full_root._assert_full_root_sentinels(model, sentinels)
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            if terminal_surface != surface or base._frozen_surface(model, names) != frozen_surface:
                raise OtaSqpHold("HOLD: no-update r32 tangent screen mutated model state")
            runtime = opened.receipt.to_artifact_dict()
            linearization = {
                "action_owner": comparator["action_owner"],
                "action_tokens": action_tokens,
                "action_sha256": comparator["action_sha256"],
                "action_row_sha256": comparator["action_row_sha256"],
                "prefix_sha256": comparator["prefix_sha256"],
                "prefix_eos_sha256": comparator["prefix_eos_sha256"],
                "target_panel": target_panel, "cut_bindings": stripped_cuts,
                "cut_margins": live_cut_margins, "step_zero_margin_drift": margin_drift,
                "gradient_count": len(gradients),
                "gradient_payload": {
                    "shape": [ACTIVE_GRADIENTS, expected_elements], "dtype": "torch.float32",
                    "bytes": ACTIVE_GRADIENTS * expected_elements * 4,
                    "stored_in_receipt": False,
                },
                "coordinate_split": {
                    **coordinate["coordinate_counts"],
                    "new_a_nonzero_gradient_elements": coordinate["new_a_nonzero_gradient_elements"],
                    "new_b_live_gradient_rows": coordinate["new_b_live_gradient_rows"],
                    "live_new_b_parameter_count": coordinate["live_new_b_parameter_count"],
                },
                "gram": {
                    label: {
                        "shape": list(matrix.shape), "matrix": matrix.tolist(),
                        "sha256": base._hash(matrix.tolist()),
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
                        "maximum_eigenvalue": float(np.linalg.eigvalsh(matrix).max()),
                    }
                    for label, matrix in (
                        ("old_r16_coordinates", coordinate["old_gram"]),
                        ("added_coordinates", coordinate["added_gram"]),
                        ("full_r32", coordinate["full_gram"]),
                    )
                },
                "old_coordinate_solver": old_solver,
                "full_r32_solver": full_solver,
            }
            del gradients, target_gradients, cut_gradients, parameters, model
            torch.cuda.empty_cache()
        artifact_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        receipt = {
            "schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID,
            "status": gate["status"],
            "stop_reason": (
                "added_coordinate_gram_psd_nonzero_and_strictly_improved_feasible_final_min_rho"
                if gate["added_component_established"]
                else "added_coordinates_did_not_strictly_improve_feasible_final_min_rho"
            ),
            "run_id": run_id, "mode": "function_preserving_r32_no_update_tangent_screen",
            "runner_source_snapshot": source_snapshot,
            "bindings": {
                "r16_receipt": str(r16_receipt),
                "r16_receipt_sha256": str(r16_receipt_sha256).lower(),
                "r16_checkpoint": r16["checkpoint"], "r16_surface_sha256": V2_SURFACE_SHA256,
                "r16_natural_sha256": START_NATURAL_SHA256,
                "comparator_receipt": str(comparator_receipt),
                "comparator_receipt_sha256": str(comparator_receipt_sha256).lower(),
                "comparator_bundle_sha256": comparator["comparator_bundle_sha256"],
                "prompt_token_sha256": PROMPT_TOKEN_SHA256, "image_sha256": IMAGE_SHA256,
            },
            "protocol": {
                "world_size": 1, "rank": R32, "lora_alpha": ALPHA32,
                "use_dora": True, "use_rslora": False, "alpha_over_rank": ALPHA32 / R32,
                "optimizer": None, "ce": None, "fisher": None,
                "natural_candidate_update": False, "aligner_update": False,
                "tied_embedding_update": False, "parameter_update": False,
                "trained_state_saved": False, "trust_delta": TRUST_DELTA,
                "safety_eta": SAFETY_ETA,
            },
            "mechanical_checkpoint": expansion,
            "step_zero": {
                "identity": step_zero_identity, "surface": surface,
                "frozen_surface_sha256": frozen_surface,
            },
            "sealed_r16_comparator": {
                key: value for key, value in comparator.items() if key != "gram"
            } | {
                "gram": comparator["gram"].tolist(),
            },
            "linearization": linearization, "comparison": gate,
            "counts": counters,
            "resources": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(local_rank)),
                "device_total_memory_bytes": int(torch.cuda.get_device_properties(local_rank).total_memory),
                "artifact_bytes_before_receipt": artifact_bytes,
            },
            "wall_time_seconds": time.monotonic() - started, "runtime": runtime,
            "claim_boundary": (
                "Single-image, exact-v2-prefix, no-update local tangent comparison only. "
                "It is not a natural candidate, training result, reachability proof, capacity proof, "
                "owner gain, transfer result, or authorization to train r32."
            ),
        }
        base._atomic_json(output / "receipt.json", receipt)
        return output
    except BaseException as error:
        base._atomic_json(output / "receipt.json", {
            "schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID,
            "status": "HOLD", "hold_type": f"r32_{stage}_hold",
            "stop_reason": str(error), "run_id": run_id,
            "mode": "function_preserving_r32_no_update_tangent_screen",
            "runner_source_snapshot": source_snapshot,
            "bindings": {
                "r16_receipt": str(r16_receipt),
                "r16_receipt_sha256": str(r16_receipt_sha256),
                "comparator_receipt": str(comparator_receipt),
                "comparator_receipt_sha256": str(comparator_receipt_sha256),
            },
            "mechanical_checkpoint": expansion, "surface": surface,
            "frozen_surface_sha256": frozen_surface,
            "linearization_failure": linearization, "counts": counters,
            "wall_time_seconds": time.monotonic() - started,
            "error_type": type(error).__name__, "traceback": traceback.format_exc(),
            "protocol": {
                "parameter_update": False, "trained_state_saved": False,
                "optimizer": None, "ce": None, "fisher": None,
            },
        })
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(
    *, run_id: str, start_receipt: Path | None = None,
    start_receipt_sha256: str | None = None,
) -> Path:
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise OtaSqpHold("HOLD: OTA-SQP-lite requires torchrun --nproc_per_node=8")
    continuation = start_receipt is not None or start_receipt_sha256 is not None
    if continuation and (start_receipt is None or start_receipt_sha256 is None):
        raise OtaSqpHold("HOLD: continuation requires both predecessor receipt path and SHA-256")
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
    start: Mapping[str, Any] | None = None
    start_checkpoint: Path | None = None if continuation else START_CHECKPOINT
    supplied_receipt_sha = (
        str(start_receipt_sha256) if start_receipt_sha256 is not None else START_RECEIPT_SHA256
    )
    try:
        dist.init_process_group(backend="nccl")
        rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            raise OtaSqpHold("HOLD: LOCAL_RANK absent")
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        source_snapshot = base._rank0_call(lambda: _prepare_output(output))
        owns_output = True
        bindings, start = base._rank0_call(
            lambda: _resolve_start_context(start_receipt, start_receipt_sha256),
        )
        start_checkpoint = Path(str(start["checkpoint"]))
        supplied_receipt_sha = str(start["receipt_sha256"])
        _parent_admission, _parent_setup, target = base._bindings()
        setup = base._setup_for_checkpoint(start_checkpoint)
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
            if initial_surface["aggregate_sha256"] != start["surface_sha256"]:
                raise OtaSqpHold("HOLD: supplied start DoRA surface drifted")
            base._surface_agreement(initial_surface, dist.group.WORLD)
            frozen_before = base._frozen_surface(model, names)
            if frozen_before != start["frozen_surface_sha256"]:
                raise OtaSqpHold("HOLD: supplied start frozen surface drifted")

            def baseline_and_aliases(
                *, model: Any = model, tokenizer: Any = tokenizer,
                native_inputs: Mapping[str, Any] = native_inputs,
            ) -> dict[str, Any]:
                nonlocal counters
                parent_ledger = base._parse_and_match(
                    tokenizer=tokenizer, generated_token_ids=parent_tokens,
                    label="ota-current-real-prefix-parent33", raw_example=setup["raw_example"],
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
                    label="ota-current-loaded-baseline",
                )
                if (
                    packet["evaluation"]["generated_token_ids"] != current_tokens
                    or not packet["corridor_gate"]["passed"]
                    or packet["promotion_gate"]["passed"]
                ):
                    raise OtaSqpHold("HOLD: loaded start baseline did not reproduce its cold working state")
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
                pre_solver: Mapping[str, Any] | None = None
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
                    pre_solver = {
                        **linearization_manifest,
                        "target_margins": [float(item["margin"]) for item in target_panel["per_token"]],
                        "cut_margins": cut_margins,
                        "gram": gram.tolist(),
                        "gram_sha256": base._hash(gram.tolist()),
                        "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
                    }
                    solution = _solve_final_margin_span(
                        pre_solver["target_margins"],
                        cut_margins, gram, eta=SAFETY_ETA, delta=TRUST_DELTA,
                    )
                    direction = _materialize_span_direction(gradients, solution["coefficients"])
                    materialized_norm = math.sqrt(max(0.0, projected._fp64_dot(direction, direction)))
                    if not math.isclose(
                        materialized_norm, float(solution["ordinary_l2_norm"]),
                        rel_tol=1.0e-5, abs_tol=SOLVER_TOLERANCE,
                    ):
                        raise OtaSqpHold("HOLD: materialized SQP displacement norm drifted")
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
                        "materialized_direction_l2_norm": materialized_norm,
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
                        "diagnostic": pre_solver,
                    }
            dist.broadcast_object_list(gradient_packet, src=0)
            if not gradient_packet[0]["ok"]:
                linearization = gradient_packet[0].get("diagnostic")
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
                checkpoint = output / (
                    f"checkpoint-selected-update-{int(start['accepted_update_ordinal']) + 1}"
                    if continuation else "checkpoint-selected-smoke"
                )
                saved = base._rank0_call(lambda model=model: {
                    "checkpoint": str(checkpoint),
                    "readback": base.full_root._save_weights_only_checkpoint(
                        model=model, source_checkpoint=start_checkpoint, destination=checkpoint,
                    ),
                })
                if rank == 0:
                    counters["checkpoint_saves"] += 1
            terminal_surface = base.full_root._full_root_surface_snapshot(names, parameters)[1]
            frozen_after = base._frozen_surface(model, names)
            if frozen_after != frozen_before:
                raise OtaSqpHold("HOLD: OTA step mutated the frozen non-DoRA surface")
            runtime = opened.receipt.to_artifact_dict()
            _agree_hash(runtime, label="runtime identity")

        del model, tokenizer, native_inputs, prompts, opened, parameters, names, sentinels, clones, direction
        torch.cuda.empty_cache()
        dist.barrier()
        cold_checkpoint = start_checkpoint if saved is None else Path(str(saved["checkpoint"]))
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
                or cold["frozen_surface"] != start["frozen_surface_sha256"]
            ):
                raise OtaSqpHold("HOLD: no-selection terminal incumbent cold identity drifted")
            status = (
                "continuation_bounded_negative_no_admissible_radius"
                if continuation else "bounded_negative_no_admissible_radius"
            )
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
                or cold["frozen_surface"] != start["frozen_surface_sha256"]
                or cold_candidate.get("action_panel") != selected.get("action_panel")
                or cold_candidate.get("active_cut_margins") != selected.get("active_cut_margins")
            ):
                raise OtaSqpHold("HOLD: selected warm/cold save-reload identity drifted")
            if cold_gate["promotion"]["passed"]:
                status = (
                    "continuation_cold_match_level_promotion"
                    if continuation else "cold_match_level_promotion"
                )
                stop_reason = "clean_cold_34_owner_promotion"
            elif cold_gate["working"]["passed"]:
                status = (
                    "continuation_selected_working_cold_reproduced"
                    if continuation else "smoke_selected_working_cold_reproduced"
                )
                stop_reason = (
                    "one_step_continuation_complete"
                    if continuation else "one_relinearization_smoke_complete"
                )
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
            if continuation:
                accounting = _continuation_receipt_fields(
                    start=start, status=status,
                    terminal_candidate=None if selected is None else cold["candidate"],
                )
            else:
                terminal_signature = (
                    None if selected is None else _relinearization_signature(
                        cold["candidate"]["matched_owner_ids"],
                        int(cold["candidate"]["first_natural_divergence"]),
                    )
                )
                accounting = {
                    "attempted_update_ordinal": 1,
                    "accepted_update_ordinal": int(selected is not None),
                    "relinearization_signature": terminal_signature,
                }
            receipt = {
                "schema_version": SCHEMA_VERSION, "unit_id": UNIT_ID,
                "status": status, "stop_reason": stop_reason, "run_id": run_id,
                **accounting,
                "runner_source_snapshot": source_snapshot,
                "bindings": bindings | {
                    "start_natural_sha256": start["natural_token_ids_sha256"],
                    "start_surface_sha256": start["surface_sha256"],
                    "frozen_surface_sha256": start["frozen_surface_sha256"],
                    "optimization_prefix_sha256": start["optimization_prefix_sha256"],
                    "optimization_prefix_eos_sha256": token_ids_sha256([
                        *start["optimization_prefix_token_ids"], base.EOS,
                    ]),
                },
                "start_state": start,
                "protocol": {
                    "tier": (
                        "bounded_one_step_continuation" if continuation
                        else "production_shaped_smoke_only"
                    ),
                    "world_size": WORLD_SIZE,
                    "relinearizations": 1,
                    "foreground_updates_run": int(continuation),
                    "radii_by_rank": {str(index): RADII[index] for index in range(WORLD_SIZE)},
                    "base_step_scale": base.LEARNING_RATE,
                    "safety_eta": SAFETY_ETA, "trust_delta": TRUST_DELTA,
                    "surface": "588 FP32 language DoRA tensors / 18006016 elements",
                    "optimizer": None, "ce": None, "fisher": None,
                    "fixed_gt32_G6_control": {"run": False, "reason": "one-step main arm only"},
                    "r32": {"gate": "not_opened_by_one_step_r16", "run": False},
                    "aligner": {"gate": "not_opened_by_one_step_r16", "run": False},
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
                "start_receipt": None if start_receipt is None else str(start_receipt),
                "start_checkpoint": None if start_checkpoint is None else str(start_checkpoint),
                "start_receipt_sha256": supplied_receipt_sha,
                "saved_checkpoint": saved, "counts_rank0_partial": counters,
                "linearization_failure": linearization,
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
    parser.add_argument(
        "--check-start-receipt", action="store_true",
        help="CPU-verify one explicit predecessor OTA receipt/checkpoint",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="run exactly one world8 r16 relinearization")
    parser.add_argument(
        "--continue-one", action="store_true",
        help="run exactly one world8 relinearization from an explicit OTA predecessor",
    )
    parser.add_argument(
        "--screen-r32", action="store_true",
        help="run the one-GPU function-preserving r32 no-update tangent screen",
    )
    parser.add_argument(
        "--check-r32-inputs", action="store_true",
        help="CPU-verify the explicit v2 source and sealed v3b comparator",
    )
    parser.add_argument("--start-receipt", type=Path)
    parser.add_argument("--start-receipt-sha256")
    parser.add_argument("--r16-receipt", type=Path)
    parser.add_argument("--r16-receipt-sha256")
    parser.add_argument("--comparator-receipt", type=Path)
    parser.add_argument("--comparator-receipt-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    modes = sum(map(bool, (
        args.check_bindings, args.check_start_receipt, args.smoke, args.continue_one,
        args.screen_r32, args.check_r32_inputs,
    )))
    if modes > 1:
        raise SystemExit("incompatible OTA modes; select exactly one")
    r32_values = (
        args.r16_receipt, args.r16_receipt_sha256,
        args.comparator_receipt, args.comparator_receipt_sha256,
    )
    r32_complete = all(value is not None for value in r32_values)
    r32_supplied = any(value is not None for value in r32_values)
    if args.check_bindings:
        if args.run_id or args.start_receipt or args.start_receipt_sha256 or r32_supplied:
            raise SystemExit("--check-bindings is CPU-only and cannot be combined with a smoke run")
        print(json.dumps(_binding_receipt() | {"start_state": _start_state()}, indent=2, sort_keys=True))
        return
    if args.check_start_receipt:
        if (
            args.run_id or args.start_receipt is None or args.start_receipt_sha256 is None
            or r32_supplied
        ):
            raise SystemExit(
                "--check-start-receipt requires --start-receipt and --start-receipt-sha256 only"
            )
        print(json.dumps(
            _continuation_start_state(args.start_receipt, args.start_receipt_sha256),
            indent=2, sort_keys=True,
        ))
        return
    if args.smoke:
        if not args.run_id or args.start_receipt or args.start_receipt_sha256 or r32_supplied:
            raise SystemExit("--smoke requires --run-id and is incompatible with continuation inputs")
        print(run(run_id=args.run_id))
        return
    if args.continue_one:
        if (
            not args.run_id or args.start_receipt is None or args.start_receipt_sha256 is None
            or r32_supplied
        ):
            raise SystemExit(
                "--continue-one requires --start-receipt, --start-receipt-sha256, and --run-id"
            )
        print(run(
            run_id=args.run_id, start_receipt=args.start_receipt,
            start_receipt_sha256=args.start_receipt_sha256,
        ))
        return
    if args.check_r32_inputs:
        if args.run_id or args.start_receipt or args.start_receipt_sha256 or not r32_complete:
            raise SystemExit(
                "--check-r32-inputs requires the r16/comparator receipt paths and SHA-256 values only"
            )
        inputs = _r32_inputs(*r32_values)
        comparator = inputs["comparator"]
        print(json.dumps({
            "status": "verified", "r16_receipt": inputs["r16"]["receipt"],
            "r16_receipt_sha256": inputs["r16"]["receipt_sha256"],
            "r16_checkpoint": inputs["r16"]["checkpoint"],
            "r16_surface_sha256": inputs["r16"]["surface_sha256"],
            "comparator_receipt": comparator["receipt"],
            "comparator_receipt_sha256": comparator["receipt_sha256"],
            "comparator_bundle_sha256": comparator["comparator_bundle_sha256"],
            "action_sha256": comparator["action_sha256"],
            "active_cuts_sha256": comparator["active_cuts_sha256"],
            "gram_sha256": comparator["gram_sha256"],
            "solver_sha256": comparator["solver_sha256"],
        }, indent=2, sort_keys=True))
        return
    if args.screen_r32:
        if (
            not args.run_id or args.start_receipt or args.start_receipt_sha256
            or not r32_complete
        ):
            raise SystemExit(
                "--screen-r32 requires --run-id and the r16/comparator receipt paths and SHA-256 values"
            )
        print(screen_r32(
            run_id=args.run_id,
            r16_receipt=args.r16_receipt,
            r16_receipt_sha256=args.r16_receipt_sha256,
            comparator_receipt=args.comparator_receipt,
            comparator_receipt_sha256=args.comparator_receipt_sha256,
        ))
        return
    raise SystemExit(
        "select --check-bindings, --check-start-receipt, --smoke, --continue-one, "
        "--check-r32-inputs, or --screen-r32"
    )


if __name__ == "__main__":
    main()
