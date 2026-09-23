"""Fixed-prefix numerical-address assay for the accepted early and late fits.

The CLI binds the frozen plan, input rows, prior cells, and both step-984
compositions; it builds native inputs once and then runs the two checkpoints
sequentially. The core performs only the frozen 24 forwards per checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment import evaluation
from src.inference.bound_requests import build_bound_native_requests
from src.qwen import coordinate_codebook
from src.qwen.native import prepare_native_inputs, prepare_replay


CASE_IMAGE_IDS = (134886, 162581, 366711, 421834)
CONDITIONS = ("correct", "identity_override", "x_plus", "x_minus", "y_plus", "y_minus")
IDENTITY_TOLERANCE = 2e-4
MAX_FORWARDS_PER_CHECKPOINT = 24
MAX_FORWARDS_BOTH_CHECKPOINTS = 48
COORDINATE_COUNT = 1000
WARP_STRENGTH = 0.2
FROZEN_PLAN_SHA256 = "871e08f8d29811d38359118d355a91b5c903548e6390ed5bc5469ff859498202"
FROZEN_PLAN_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-23-codebook-discriminator-preparation/semantic-assay-plan.json"
)
DEFAULT_ADMISSION_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-coordinate-codebook-three-loss/evaluation-admission.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-23-codebook-discriminator/semantic-address-assay.json"
)
FROZEN_DATASET_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-coordinate-codebook-scale-preparation/train-1024.coord.jsonl"
)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _row_major_addresses(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _warp(values: torch.Tensor, *, axis: int | None, sign: int) -> torch.Tensor:
    if axis not in (None, 0, 1) or sign not in (-1, 0, 1):
        raise ValueError("address warp requires x/y axis and sign -1/0/+1")
    result = values.clone()
    if axis is not None and sign:
        u = result[..., axis]
        result[..., axis] = u + sign * WARP_STRENGTH * u * (1.0 - u)
    if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
        raise ValueError("warped addresses must remain finite within [0, 1]")
    return result


def _warp_patch_edges(edges: torch.Tensor, *, axis: int | None, sign: int) -> torch.Tensor:
    if not isinstance(edges, torch.Tensor) or edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("patch addresses must have shape [raw_patches, 4]")
    if not edges.numel() or not bool(torch.isfinite(edges).all()):
        raise ValueError("patch addresses must be nonempty and finite")
    if bool(((edges < 0) | (edges > 1)).any()):
        raise ValueError("patch addresses must lie in [0, 1]")
    if bool((edges[:, 0] > edges[:, 1]).any()) or bool((edges[:, 2] > edges[:, 3]).any()):
        raise ValueError("patch edge tuples must remain monotone")
    result = edges.clone()
    if axis is not None and sign:
        start = 0 if axis == 0 else 2
        end = start + 2
        u = result[:, start:end]
        result[:, start:end] = u + sign * WARP_STRENGTH * u * (1.0 - u)
    if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
        raise ValueError("warped patch edges must remain finite within [0, 1]")
    if bool((result[:, 0] > result[:, 1]).any()) or bool((result[:, 2] > result[:, 3]).any()):
        raise ValueError("warped patch edge tuples must remain monotone")
    unchanged = (2, 3) if axis == 0 and sign else (0, 1) if axis == 1 and sign else (0, 1, 2, 3)
    if not torch.equal(result[:, unchanged], edges[:, unchanged]):
        raise AssertionError("axis warp changed the other patch coordinates or tuple order")
    return result


def _condition_warp(condition: str) -> tuple[int | None, int]:
    if condition in ("correct", "identity_override"):
        return None, 0
    axis = 0 if condition.startswith("x_") else 1 if condition.startswith("y_") else None
    if axis is None or not condition.endswith(("plus", "minus")):
        raise ValueError(f"unknown frozen assay condition: {condition}")
    return axis, (1 if condition.endswith("plus") else -1)


@contextmanager
def _checked_early_forward(
    codebook: Any,
    expected_grid: torch.Tensor,
    edge_override: torch.Tensor | None,
) -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    original = codebook.inject_early
    if getattr(codebook, "_edge_coordinates_override", None) is not None:
        raise RuntimeError("early assay cannot start with an active patch-edge override")

    def checked(hidden_states: torch.Tensor, grid: torch.Tensor, *, edge_coordinates: torch.Tensor | None = None) -> torch.Tensor:
        if len(calls) >= 1:
            raise AssertionError("early address injection ran more than once for one image forward")
        observed_grid = torch.as_tensor(grid).detach().cpu()
        if not torch.equal(observed_grid, expected_grid.detach().cpu()):
            raise AssertionError("early address override reached a different image grid")
        if edge_override is None:
            if edge_coordinates is not None:
                raise AssertionError("correct early condition unexpectedly received an override")
        elif not isinstance(edge_coordinates, torch.Tensor) or not torch.equal(
            edge_coordinates.detach().float().cpu(), edge_override.detach().float().cpu()
        ):
            raise AssertionError("early address override did not reach the actual injection caller")
        calls.append({"raw_patch_count": int(hidden_states.shape[0]), "override": edge_coordinates is not None})
        return original(hidden_states, grid, edge_coordinates=edge_coordinates)

    try:
        with patch.object(codebook, "inject_early", checked):
            if edge_override is None:
                yield calls
            else:
                with codebook.override_patch_edges(edge_override):
                    yield calls
    finally:
        restored = codebook.inject_early
        if (
            getattr(restored, "__self__", None) is not getattr(original, "__self__", None)
            or getattr(restored, "__func__", restored) is not getattr(original, "__func__", original)
        ):
            raise AssertionError("early injection caller was not restored after the forward")
        if getattr(codebook, "_edge_coordinates_override", None) is not None:
            raise AssertionError("early patch-edge override was not restored after the forward")


@contextmanager
def _checked_late_forward(
    *,
    expected_merged_hw: tuple[int, int],
    axis: int | None,
    sign: int,
) -> Iterator[list[dict[str, Any]]]:
    original = coordinate_codebook._normalized_addresses
    calls: list[dict[str, Any]] = []

    def checked(height: int, width: int, *, device: torch.device) -> torch.Tensor:
        if calls:
            raise AssertionError("late address lookup ran more than once for one image forward")
        if (int(height), int(width)) != expected_merged_hw:
            raise AssertionError(
                f"late address grid {(height, width)} differs from frozen {expected_merged_hw}"
            )
        base = original(height, width, device=device)
        expected = _row_major_addresses(height, width, device=device)
        if base.shape != (height * width, 2) or not torch.equal(base, expected):
            raise AssertionError("late address lookup changed grid shape or row-major order")
        warped = _warp(base, axis=axis, sign=sign)
        calls.append({"merged_grid_hw": [int(height), int(width)], "address_count": int(base.shape[0])})
        return warped

    try:
        with patch.object(coordinate_codebook, "_normalized_addresses", checked):
            yield calls
    finally:
        if coordinate_codebook._normalized_addresses is not original:
            raise AssertionError("late normalized-address wrapper was not restored after the forward")


def _readout(logits: torch.Tensor, coordinate_ids: tuple[int, ...]) -> dict[str, Any]:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("causal slot logits must be a finite full-vocabulary vector")
    if len(coordinate_ids) != COORDINATE_COUNT or len(set(coordinate_ids)) != COORDINATE_COUNT:
        raise ValueError("readout requires 1,000 unique coordinate token IDs")
    if min(coordinate_ids) < 0 or max(coordinate_ids) >= logits.numel():
        raise ValueError("coordinate token IDs exceed the full vocabulary logits")
    values = logits.float()
    ids = torch.tensor(coordinate_ids, dtype=torch.long, device=values.device)
    coordinate_logits = values.index_select(0, ids)
    full_log_normalizer = torch.logsumexp(values, dim=0)
    coordinate_log_normalizer = torch.logsumexp(coordinate_logits, dim=0)
    family_mass = torch.exp(coordinate_log_normalizer - full_log_normalizer)
    probabilities = torch.softmax(coordinate_logits, dim=0)
    bins = torch.arange(COORDINATE_COUNT, device=values.device, dtype=probabilities.dtype) / COORDINATE_COUNT
    conditional_mean = (probabilities * bins).sum()
    cdf = probabilities.cumsum(0)
    return {
        "coordinate_logits": coordinate_logits.detach().cpu().tolist(),
        "full_vocabulary_logsumexp": float(full_log_normalizer.detach().cpu()),
        "coordinate_family_logsumexp": float(coordinate_log_normalizer.detach().cpu()),
        "coordinate_family_mass": float(family_mass.detach().cpu()),
        "conditional_mean_coordinate_b_over_1000": float(conditional_mean.detach().cpu()),
        "mass_weighted_first_moment": float((family_mass * conditional_mean).detach().cpu()),
        "coordinate_cdf": cdf.detach().cpu().tolist(),
    }


def _contrast(plus: Mapping[str, Any], minus: Mapping[str, Any]) -> dict[str, float]:
    plus_cdf = torch.tensor(plus["coordinate_cdf"], dtype=torch.float64)
    minus_cdf = torch.tensor(minus["coordinate_cdf"], dtype=torch.float64)
    if plus_cdf.numel() != COORDINATE_COUNT or minus_cdf.numel() != COORDINATE_COUNT:
        raise ValueError("coordinate CDF must retain all 1,000 legal bins")
    return {
        "plus_minus_conditional_mean_coordinate": float(
            plus["conditional_mean_coordinate_b_over_1000"]
            - minus["conditional_mean_coordinate_b_over_1000"]
        ),
        "plus_minus_mass_weighted_first_moment": float(
            plus["mass_weighted_first_moment"] - minus["mass_weighted_first_moment"]
        ),
        "plus_minus_coordinate_family_mass": float(
            plus["coordinate_family_mass"] - minus["coordinate_family_mass"]
        ),
        "coordinate_distribution_wasserstein1": float((plus_cdf - minus_cdf).abs().sum() / COORDINATE_COUNT),
    }


def _directional_responses(conditions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "x1": {
            "same_axis": "x",
            "plus_minus": _contrast(conditions["x_plus"]["x1"], conditions["x_minus"]["x1"]),
            "wrong_axis": "y",
            "wrong_axis_plus_minus": _contrast(conditions["y_plus"]["x1"], conditions["y_minus"]["x1"]),
        },
        "y1": {
            "same_axis": "y",
            "plus_minus": _contrast(conditions["y_plus"]["y1"], conditions["y_minus"]["y1"]),
            "wrong_axis": "x",
            "wrong_axis_plus_minus": _contrast(conditions["x_plus"]["y1"], conditions["x_minus"]["y1"]),
        },
    }


def _native_case(
    case: Mapping[str, Any], *, coordinate_ids: tuple[int, ...], spatial_merge_size: int
) -> tuple[dict[str, Any], tuple[int, ...], torch.Tensor, tuple[int, int]]:
    prepared = case.get("prepared")
    if not isinstance(prepared, Mapping):
        raise ValueError("each frozen case needs prepared native inputs and executed identities")
    native_inputs = prepared.get("native_inputs")
    if not isinstance(native_inputs, Mapping):
        raise ValueError("prepared case is missing maintained processor inputs")
    pixel_values = native_inputs.get("pixel_values")
    if not isinstance(pixel_values, torch.Tensor) or not pixel_values.numel():
        raise ValueError("prepared case is missing processor pixel_values")
    prompt_ids = tuple(int(value) for value in prepared.get("executed_prompt_token_ids", ()))
    if not prompt_ids:
        raise ValueError("prepared case is missing executed prompt IDs")
    if len(prompt_ids) != int(case["prompt_token_count"]):
        raise ValueError("executed prompt token count differs from the frozen case")
    if _json_digest(list(prompt_ids)) != str(case["prompt_token_ids_sha256"]):
        raise ValueError("executed prompt IDs differ from the frozen case hash")
    input_ids = native_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("one-image native inputs must contain one prompt row")
    attention_mask = native_inputs.get("attention_mask")
    if attention_mask is None:
        observed_prompt = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
    else:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("native prompt attention mask differs from input_ids")
        observed_prompt = tuple(
            int(value)
            for value in input_ids[0][attention_mask[0].bool()].detach().cpu().tolist()
        )
    if observed_prompt != prompt_ids:
        raise ValueError("native input prompt row differs from the executed prompt identity")
    executed_media = tuple(str(value) for value in prepared.get("executed_media_sha256", ()))
    expected_media = tuple(str(value) for value in case["media_sha256"])
    if len(expected_media) != 1 or executed_media != expected_media:
        raise ValueError("executed media identity differs from the frozen single-image case")
    grid = native_inputs.get("image_grid_thw")
    if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != (1, 3):
        raise ValueError("semantic assay requires one native still-image grid")
    grid_cpu = grid.detach().to(device="cpu", dtype=torch.long)
    expected_grid = tuple(int(value) for value in case["grid_thw"])
    if (
        tuple(grid_cpu[0].tolist()) != expected_grid
        or expected_grid[0] != 1
        or min(expected_grid[1:]) <= 0
    ):
        raise ValueError("executed native image grid differs from the frozen case")
    if spatial_merge_size <= 0 or any(value % spatial_merge_size for value in expected_grid[1:]):
        raise ValueError("native image grid is not divisible by the loaded merge size")
    merged_hw = tuple(int(value) for value in case["merged_grid_hw"])
    if merged_hw != (expected_grid[1] // spatial_merge_size, expected_grid[2] // spatial_merge_size):
        raise ValueError("frozen merged grid is not the native grid divided by merge size")
    raw_patch_count = int(case["raw_patch_count"])
    if raw_patch_count != expected_grid[1] * expected_grid[2]:
        raise ValueError("frozen raw patch count differs from the native grid")
    x_query = case["first_x_query"]
    y_query = case["first_y_query"]
    if x_query.get("coordinate_role") != "x1" or y_query.get("coordinate_role") != "y1":
        raise ValueError("frozen query roles must be first x1 and y1")
    x_prefix = tuple(int(value) for value in x_query["response_prefix_token_ids"])
    y_prefix = tuple(int(value) for value in y_query["response_prefix_token_ids"])
    x_target = int(x_query["target_token_id"])
    y_target = int(y_query["target_token_id"])
    if y_prefix != (*x_prefix, x_target) or y_query.get("earlier_gt_coordinates") != ["x1"]:
        raise ValueError("y1 query must contain only the fixed first GT x1 after its shared prefix")
    bins = tuple(int(value) for value in case["first_gt_coord_bins"])
    if len(bins) != 2 or any(value < 0 or value >= COORDINATE_COUNT for value in bins):
        raise ValueError("frozen first GT coordinate bins must be legal")
    id_to_bin = {token: index for index, token in enumerate(coordinate_ids)}
    if id_to_bin.get(x_target) != bins[0] or id_to_bin.get(y_target) != bins[1]:
        raise ValueError("frozen x1/y1 target token IDs disagree with their coordinate bins")
    if int(case["image_id"]) not in CASE_IMAGE_IDS:
        raise ValueError("case is outside the four frozen assay images")
    if case.get("annotation_unique_first_description") is not True:
        raise ValueError("case is not one of the frozen annotation-unique descriptions")
    return {"native_inputs": native_inputs, "x_prefix": (*prompt_ids, *x_prefix), "x_target": x_target,
            "y_target": y_target, "grid": grid, "media_sha256": list(executed_media),
            "prompt_token_ids_sha256": _json_digest(list(prompt_ids))}, prompt_ids, grid, merged_hw


def _forward_two_slots(
    model: Any,
    native_inputs: Mapping[str, Any],
    prompt_plus_x_prefix: tuple[int, ...],
    x_target: int,
) -> torch.Tensor:
    replay = prepare_replay(
        model,
        native_inputs,
        prompt_token_ids=prompt_plus_x_prefix,
        continuation_token_ids=(x_target,),
        compact_logits=True,
    )
    inputs = replay.inputs
    history = inputs.get("input_ids")
    mask = inputs.get("attention_mask")
    if not isinstance(history, torch.Tensor) or history.ndim != 2 or history.shape[0] != 1:
        raise ValueError("exact replay must contain one causal history")
    if not isinstance(mask, torch.Tensor) or mask.shape != history.shape:
        raise ValueError("exact replay attention mask differs from its history")
    observed = tuple(int(value) for value in history[0][mask[0].bool()].detach().cpu().tolist())
    if observed != (*prompt_plus_x_prefix, x_target):
        raise ValueError("exact replay inserted a token beyond the fixed x1 target")
    if inputs.get("logits_to_keep") != 2:
        raise ValueError("shared causal x1/y1 replay must retain exactly its final two logits")
    output = model(**inputs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError("model forward must return exactly the causal x1 and y1 logits")
    return logits[0].float()


def run_checkpoint_semantic_assay(
    qwen: Any,
    *,
    checkpoint: str,
    prepared_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one accepted step-984 checkpoint over the four frozen cases."""

    if checkpoint not in ("early16", "late16"):
        raise ValueError("checkpoint label must be the accepted early16 or late16 fit")
    if len(prepared_cases) != len(CASE_IMAGE_IDS) or tuple(int(case["image_id"]) for case in prepared_cases) != CASE_IMAGE_IDS:
        raise ValueError(f"prepared cases must be the frozen ordered image IDs {CASE_IMAGE_IDS}")
    model = qwen.model
    if bool(getattr(model, "training", False)):
        raise ValueError("semantic assay requires an evaluation-mode checkpoint")
    codebook = getattr(model, "coordinate_codebook", None)
    expected_mode = "early_patch_edges" if checkpoint == "early16" else "late_masked_center"
    if codebook is None or getattr(codebook, "mode", None) != expected_mode or not bool(getattr(codebook, "enabled", False)):
        raise ValueError("loaded checkpoint codebook mode or enabled state differs from the frozen fit")
    coordinate_ids = tuple(int(value) for value in qwen.token_identity.coordinate_token_ids)
    if len(coordinate_ids) != COORDINATE_COUNT or len(set(coordinate_ids)) != COORDINATE_COUNT:
        raise ValueError("loaded tokenizer must expose the frozen 1,000 coordinate token IDs")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("loaded checkpoint has no model parameters")
    device = parameter.device

    # Bind every native input before spending any teacher forwards.
    validated_cases = []
    for case in prepared_cases:
        spatial_merge_size = int(codebook.spatial_merge_size)
        runtime, _, grid, merged_hw = _native_case(
            case, coordinate_ids=coordinate_ids, spatial_merge_size=spatial_merge_size
        )
        grid_on_device = grid.to(device=device, dtype=torch.long)
        edges = None
        if checkpoint == "early16":
            edges = codebook.patch_edge_coordinates(grid_on_device, device=device)
            if tuple(edges.shape) != (int(case["raw_patch_count"]), 4):
                raise ValueError("early codebook edge count differs from the frozen raw patch count")
        validated_cases.append((case, runtime, grid, grid_on_device, merged_hw, edges))

    rows: list[dict[str, Any]] = []
    forwards = 0
    for case, runtime, grid, grid_on_device, merged_hw, edges in validated_cases:
        native_inputs = runtime["native_inputs"]
        condition_readouts: dict[str, Any] = {}
        condition_calls: dict[str, Any] = {}
        correct_full_logits: torch.Tensor | None = None
        full_vocab_identity_drift: float | None = None
        for condition in CONDITIONS:
            axis, sign = _condition_warp(condition)
            logits: torch.Tensor
            if checkpoint == "early16":
                assert edges is not None
                edge_override = None if condition == "correct" else _warp_patch_edges(edges, axis=axis, sign=sign)
                with _checked_early_forward(codebook, grid_on_device, edge_override) as calls:
                    with torch.inference_mode():
                        logits = _forward_two_slots(model, native_inputs, runtime["x_prefix"], runtime["x_target"])
                if len(calls) != 1:
                    raise AssertionError("early codebook injection did not run exactly once")
                if getattr(codebook, "_edge_coordinates_override", None) is not None:
                    raise AssertionError("early patch override was not restored after its forward")
                override_count = int(edge_override is not None)
                invocation_count = len(calls)
            elif condition == "correct":
                with torch.inference_mode():
                    logits = _forward_two_slots(model, native_inputs, runtime["x_prefix"], runtime["x_target"])
                override_count = 0
                invocation_count = 0
            else:
                with _checked_late_forward(expected_merged_hw=merged_hw, axis=axis, sign=sign) as calls:
                    with torch.inference_mode():
                        logits = _forward_two_slots(model, native_inputs, runtime["x_prefix"], runtime["x_target"])
                if len(calls) != 1:
                    raise AssertionError("late address override did not run exactly once")
                override_count = 1
                invocation_count = len(calls)
            forwards += 1
            if logits.ndim != 2 or logits.shape[0] != 2 or max(coordinate_ids) >= logits.shape[1]:
                raise ValueError("full vocabulary logits do not cover the coordinate family")
            if condition == "correct":
                correct_full_logits = logits.detach().float().cpu()
            elif condition == "identity_override":
                if correct_full_logits is None:
                    raise AssertionError("correct full-vocabulary logits must precede identity replay")
                full_vocab_identity_drift = float(
                    (correct_full_logits - logits.detach().float().cpu()).abs().max()
                )
            condition_readouts[condition] = {
                "x1": _readout(logits[0], coordinate_ids),
                "y1": _readout(logits[1], coordinate_ids),
            }
            condition_calls[condition] = {
                "model_forwards": 1,
                "address_override_invocations": override_count,
                "checked_caller_invocations": invocation_count,
            }
        identity_drift = {}
        for role in ("x1", "y1"):
            identity_drift[role] = float(
                (
                    torch.tensor(condition_readouts["correct"][role]["coordinate_logits"])
                    - torch.tensor(condition_readouts["identity_override"][role]["coordinate_logits"])
                ).abs().max()
            )
        if full_vocab_identity_drift is None:
            raise AssertionError("full-vocabulary identity drift was not captured")
        if float(full_vocab_identity_drift) > IDENTITY_TOLERANCE:
            raise AssertionError(
                f"identity override full-vocabulary drift {full_vocab_identity_drift} exceeds {IDENTITY_TOLERANCE}"
            )
        rows.append({
            "image_id": int(case["image_id"]),
            "row_id": str(case["row_id"]),
            "cohort": str(case["cohort"]),
            "first_description": str(case["first_description"]),
            "first_gt_coord_bins": list(case["first_gt_coord_bins"]),
            "executed_prompt_token_ids_sha256": runtime["prompt_token_ids_sha256"],
            "executed_media_sha256": runtime["media_sha256"],
            "grid_thw": [int(value) for value in grid.detach().cpu()[0].tolist()],
            "merged_grid_hw": list(merged_hw),
            "raw_patch_count": int(case["raw_patch_count"]),
            "conditions": condition_readouts,
            "calls": condition_calls,
            "identity_coordinate_logit_max_abs_by_role": identity_drift,
            "identity_full_vocabulary_logit_max_abs": float(full_vocab_identity_drift),
            "directional_responses": _directional_responses(condition_readouts),
        })
    if forwards > MAX_FORWARDS_PER_CHECKPOINT:
        raise AssertionError("per-checkpoint semantic assay exceeded 24 teacher forwards")
    return {
        "schema": "coordinate_codebook_discriminator.semantic_assay_result.v1",
        "status": "complete",
        "checkpoint": checkpoint,
        "checkpoint_step": 984,
        "teacher_forwards": forwards,
        "max_teacher_forwards_this_checkpoint": MAX_FORWARDS_PER_CHECKPOINT,
        "max_teacher_forwards_both_checkpoints": MAX_FORWARDS_BOTH_CHECKPOINTS,
        "identity_tolerance_full_vocabulary_max_abs": IDENTITY_TOLERANCE,
        "coordinate_token_ids": list(coordinate_ids),
        "mass_weighted_first_moment_definition": "coordinate_family_mass times conditional mean over the 1000 coordinate bins; not a coordinate expectation over non-coordinate vocabulary items",
        "interpretation_limit": "fixed-prefix numerical sensitivity only; the y1 x-warp holds GT x1 fixed, and the assay makes no whole-box transport, owner, or training-benefit claim",
        "cases": rows,
    }


def _check_binding(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = binding(Path(str(expected["path"])))
    if observed != dict(expected):
        raise ValueError(f"frozen file binding changed: {expected['path']}")
    return observed


def _checkpoint_bindings(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    checkpoints = plan.get("checkpoint_bindings")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"early16", "late16"}:
        raise ValueError("frozen plan must bind only early16 and late16")
    result: dict[str, list[dict[str, Any]]] = {}
    for label in ("early16", "late16"):
        entry = checkpoints[label]
        root = Path(str(entry["checkpoint_root"])).resolve(strict=True)
        if root.name != "step-984" or str(root) != str(entry["checkpoint_root"]):
            raise ValueError(f"{label} is not bound to its exact step-984 root")
        expected = entry.get("payload_bindings")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{label} has no frozen payload bindings")
        observed = [_check_binding(item) for item in expected]
        if sorted(expected, key=lambda item: item["path"]) != sorted(observed, key=lambda item: item["path"]):
            raise ValueError(f"{label} payload bindings changed")
        result[label] = observed
    return result


def _coordinate_token(value: Any) -> int:
    if not isinstance(value, str) or not value.startswith("<|coord_") or not value.endswith("|>"):
        raise ValueError(f"non-canonical coordinate token: {value!r}")
    result = int(value[len("<|coord_") : -2])
    if not 0 <= result < COORDINATE_COUNT:
        raise ValueError(f"coordinate token is outside norm1000: {value!r}")
    return result


def _ground_truth_signature(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for obj in row["objects"]:
        result.append({
            "owner_id": str(obj["coco_ann_id"]),
            "description": str(obj["desc"]),
            "coord_bins": [_coordinate_token(value) for value in obj["bbox_2d"]],
        })
    return result


def _pixel_tensor_sha256(native_inputs: Mapping[str, Any]) -> str:
    pixels = native_inputs.get("pixel_values")
    if not isinstance(pixels, torch.Tensor) or not pixels.numel():
        raise ValueError("native image projection is missing pixel_values")
    raw = pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _validate_prior_cells(case: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    expected_gt = _ground_truth_signature(row)
    bindings = case.get("cell_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"source", "early16", "late16"}:
        raise ValueError(f"{case['row_id']} lacks source/early16/late16 retained cells")
    for label in ("source", "early16", "late16"):
        binding_record = _check_binding(bindings[label])
        cell = json.loads(Path(binding_record["path"]).read_text())
        cell_case = cell.get("case", {})
        if (
            cell.get("status") != "complete"
            or int(cell_case.get("image_id", -1)) != int(case["image_id"])
            or str(cell_case.get("row_id")) != str(case["row_id"])
            or str(cell_case.get("cohort")) != str(case["cohort"])
        ):
            raise ValueError(f"{label} retained cell identity differs for {case['row_id']}")
        cell_gt = [
            {"owner_id": str(item["owner_id"]), "description": str(item["description"]),
             "coord_bins": [int(value) for value in item["coord_bins"]]}
            for item in cell.get("gt", ())
        ]
        if cell_gt != expected_gt:
            raise ValueError(f"{label} retained cell ground truth differs for {case['row_id']}")
        prompt_ids = [int(value) for value in cell.get("prompt_token_ids", ())]
        if (
            len(prompt_ids) != int(case["prompt_token_count"])
            or _json_digest(prompt_ids) != str(case["prompt_token_ids_sha256"])
        ):
            raise ValueError(f"{label} retained cell prompt differs for {case['row_id']}")
        if tuple(str(value) for value in cell.get("media_sha256", ())) != tuple(case["media_sha256"]):
            raise ValueError(f"{label} retained cell media differs for {case['row_id']}")
    return {"gt": expected_gt, "cell_bindings": dict(bindings)}


def _frozen_inputs(plan_path: Path, admission_path: Path) -> dict[str, Any]:
    if binding(plan_path)["sha256"] != FROZEN_PLAN_SHA256:
        raise ValueError("semantic-assay plan SHA256 differs from the frozen contract")
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema") != "coordinate_codebook_discriminator.semantic_assay_plan.v1"
        or plan.get("status") != "proposal_cpu_only_no_model_calls"
        or int(plan.get("model_calls", -1)) != 0
    ):
        raise ValueError("semantic-assay plan is not the frozen CPU-only proposal")
    if plan.get("coordinate_token_ids") is None or len(plan["coordinate_token_ids"]) != COORDINATE_COUNT:
        raise ValueError("semantic-assay plan must bind all 1,000 coordinate token IDs")
    if len(set(int(value) for value in plan["coordinate_token_ids"])) != COORDINATE_COUNT:
        raise ValueError("semantic-assay plan coordinate token IDs are not unique")
    denominator = plan.get("selection_denominator", {})
    if (
        denominator.get("eligible") != 4
        or denominator.get("checkpoints") != 2
        or denominator.get("conditions_per_case") != len(CONDITIONS)
        or denominator.get("teacher_forwards_max") != MAX_FORWARDS_BOTH_CHECKPOINTS
        or plan.get("intervention", {}).get("conditions") != list(CONDITIONS)
    ):
        raise ValueError("semantic-assay plan differs from the frozen 4 x 2 x 6 contract")
    cases = plan.get("cases")
    if (
        not isinstance(cases, list)
        or tuple(int(case["image_id"]) for case in cases) != CASE_IMAGE_IDS
        or any(case.get("annotation_unique_first_description") is not True for case in cases)
    ):
        raise ValueError("semantic-assay cases differ from the frozen ordered four-image cohort")

    input_bindings = plan.get("input_bindings")
    if not isinstance(input_bindings, list) or not input_bindings:
        raise ValueError("semantic-assay plan has no frozen input bindings")
    verified_inputs = [_check_binding(item) for item in input_bindings]
    datasets = [item for item in verified_inputs if Path(item["path"]).name == FROZEN_DATASET_PATH.name]
    if len(datasets) != 1 or Path(datasets[0]["path"]) != FROZEN_DATASET_PATH:
        raise ValueError("semantic-assay plan does not bind the frozen train-1024 input")
    dataset = FROZEN_DATASET_PATH
    rows = evaluation._load_rows(dataset)
    physical_lines = dataset.read_text().splitlines()
    if sum(bool(line.strip()) for line in physical_lines) != len(rows):
        raise ValueError("frozen dataset contains blank rows that invalidate physical row numbers")
    bound_rows: list[dict[str, Any]] = []
    prior_cells = []
    for case in cases:
        row_number = int(case["dataset_row_number"])
        if row_number < 1 or row_number > len(rows):
            raise ValueError(f"frozen row number is outside dataset: {row_number}")
        row = rows[row_number - 1]
        admission_row = row.get("_admission", {})
        if (
            int(row.get("image_id", -1)) != int(case["image_id"])
            or str(admission_row.get("row_id")) != str(case["row_id"])
            or str(admission_row.get("cohort")) != str(case["cohort"])
            or len(row.get("objects", ())) != int(case["source_gt_count"])
        ):
            raise ValueError(f"frozen dataset row identity differs for {case['row_id']}")
        gt = _ground_truth_signature(row)
        if (
            not gt
            or gt[0]["description"] != str(case["first_description"])
            or gt[0]["coord_bins"][:2] != [int(value) for value in case["first_gt_coord_bins"]]
        ):
            raise ValueError(f"frozen first description/GT differs for {case['row_id']}")
        prior_cells.append(_validate_prior_cells(case, row))
        bound_rows.append(row)

    admission_binding = binding(admission_path)
    admission = json.loads(admission_path.read_text())
    if not isinstance(admission.get("source_config"), Mapping):
        raise ValueError("admission file is missing its maintained source_config")
    return {
        "plan": plan,
        "admission": admission,
        "dataset": dataset,
        "rows": bound_rows,
        "prior_cells": prior_cells,
        "plan_binding": binding(plan_path),
        "admission_binding": admission_binding,
        "input_bindings": verified_inputs,
        "dataset_binding": datasets[0],
        "checkpoint_payload_bindings": _checkpoint_bindings(plan),
    }


def _prepare_native_cases(
    qwen: Any,
    *,
    plan_cases: Sequence[Mapping[str, Any]],
    admission: Mapping[str, Any],
    dataset: Path,
    input_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(plan_cases) != len(CASE_IMAGE_IDS) or len(input_rows) != len(CASE_IMAGE_IDS):
        raise ValueError("native preparation requires all four frozen input rows")
    if tuple(int(case["image_id"]) for case in plan_cases) != CASE_IMAGE_IDS:
        raise ValueError("native preparation case order differs from the frozen plan")
    if tuple(int(row["image_id"]) for row in input_rows) != CASE_IMAGE_IDS:
        raise ValueError("native preparation input row order differs from the frozen plan")

    config = evaluation._config(admission, dataset)
    config_sha256 = _json_digest(config)
    prepared_cases: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for case, input_row in zip(plan_cases, input_rows, strict=True):
        row = evaluation._normalize_case(qwen, input_row, config, dataset)
        if (
            str(row["row_id"]) != str(case["row_id"])
            or str(row.get("cohort")) != str(case["cohort"])
        ):
            raise ValueError(f"normalized native case identity differs for {case['row_id']}")
        gt = evaluation._gt(row)
        gt_signature = [
            {"owner_id": str(item["owner_id"]), "description": str(item["description"]),
             "coord_bins": [int(value) for value in item["coord_bins"]]}
            for item in gt
        ]
        if gt_signature != _ground_truth_signature(input_row):
            raise ValueError(f"normalized native ground truth differs for {case['row_id']}")
        if not gt or gt[0]["description"] != str(case["first_description"]):
            raise ValueError(f"native first description differs for {case['row_id']}")
        image_plan = row["image_plan"]
        if (
            int(image_plan["backend_prompt_token_count"]) != int(case["prompt_token_count"])
            or [int(value) for value in image_plan["observed_image_grid_thw"]] != [int(value) for value in case["grid_thw"]]
            or int(image_plan["merged_visual_tokens"]) != int(case["merged_token_count"])
        ):
            raise ValueError(f"normalized image/prompt plan differs for {case['row_id']}")

        requests, _ = build_bound_native_requests(qwen, config, [row])
        if len(requests) != 1 or requests[0].request_id != str(case["row_id"]):
            raise ValueError(f"maintained native request identity differs for {case['row_id']}")
        batch = prepare_native_inputs(
            qwen.processor, requests, device="cpu", record_media_identity=True
        )
        prompt_ids = tuple(int(value) for value in batch.prompt_token_ids[0])
        media_sha256 = tuple(str(value) for value in (batch.media_sha256 or ()))
        grid = batch.image_grids[0]
        if (
            len(prompt_ids) != int(case["prompt_token_count"])
            or _json_digest(list(prompt_ids)) != str(case["prompt_token_ids_sha256"])
            or media_sha256 != tuple(str(value) for value in case["media_sha256"])
            or grid is None
            or tuple(int(value) for value in grid) != tuple(int(value) for value in case["grid_thw"])
        ):
            raise ValueError(f"executed processor prompt/media/grid differs for {case['row_id']}")
        prepared = dict(case)
        prepared["prepared"] = {
            "native_inputs": batch.inputs,
            "executed_prompt_token_ids": prompt_ids,
            "executed_media_sha256": media_sha256,
        }
        prepared_cases.append(prepared)
        identities.append({
            "image_id": int(case["image_id"]),
            "row_id": str(case["row_id"]),
            "cohort": str(case["cohort"]),
            "dataset_row_number": int(case["dataset_row_number"]),
            "first_description": str(case["first_description"]),
            "first_gt_coord_bins": [int(value) for value in case["first_gt_coord_bins"]],
            "image_plan_content_sha256": str(image_plan["image_content_sha256"]),
            "ground_truth": gt,
            "native_request_config_sha256": config_sha256,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": _json_digest(list(prompt_ids)),
            "prompt_token_ids": list(prompt_ids),
            "media_sha256": list(media_sha256),
            "pixel_values_sha256": _pixel_tensor_sha256(batch.inputs),
            "grid_thw": [int(value) for value in grid],
            "merged_grid_hw": [int(value) for value in case["merged_grid_hw"]],
            "raw_patch_count": int(case["raw_patch_count"]),
        })
    return prepared_cases, identities


def _runtime_bindings(
    identity: Mapping[str, Any], expected: Mapping[str, Any], admission: Mapping[str, Any]
) -> None:
    root = Path(str(expected["checkpoint_root"])).resolve(strict=True)
    if Path(str(identity.get("checkpoint_root"))).resolve(strict=True) != root:
        raise ValueError("loaded checkpoint root differs from the frozen step-984 binding")
    payloads = identity.get("payload_bindings")
    if not isinstance(payloads, list) or sorted(payloads, key=lambda item: item["path"]) != sorted(
        expected["payload_bindings"], key=lambda item: item["path"]
    ):
        raise ValueError("loaded runtime payload bindings differ from the frozen step-984 files")
    source_config = admission.get("source_config", {})
    expected_model_path = Path(source_config["model"]["base_model"]).resolve(strict=True)
    expected_launch = {
        "backend": "hf",
        "model_path": str(expected_model_path),
        "model_dtype": "bf16",
        "backend_options": {
            "hf": {
                "attn_implementation": "flash_attention_2",
                "patch_embed_linearization": "enabled",
                "adapter_runtime": "live_promoted",
                "coordinate_codebook_path": str(root / "coordinate_codebook"),
            }
        },
        "adapter": {"type": "dora", "name": "default", "path": str(root / "adapter")},
        "embedding_delta": {
            "path": str(root / "special_token_embeddings"),
            "source_gate_root": None,
        },
    }
    if identity.get("launch") != expected_launch:
        raise ValueError("loaded runtime launch configuration differs from the frozen native HF path")


def _common_runtime_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    launch = identity.get("launch")
    if not isinstance(launch, Mapping):
        raise ValueError("loaded runtime identity lacks its launch configuration")
    options = launch.get("backend_options")
    adapter = launch.get("adapter")
    embedding = launch.get("embedding_delta")
    if not isinstance(options, Mapping) or not isinstance(adapter, Mapping) or not isinstance(embedding, Mapping):
        raise ValueError("loaded runtime identity has incomplete backend configuration")
    hf_options = options.get("hf")
    if not isinstance(hf_options, Mapping):
        raise ValueError("loaded runtime has no HF backend options")
    shared_options = {
        key: hf_options.get(key)
        for key in ("attn_implementation", "patch_embed_linearization", "adapter_runtime")
    }
    return {
        "backend": launch.get("backend"),
        "model_path": launch.get("model_path"),
        "model_dtype": launch.get("model_dtype"),
        "shared_backend_options": shared_options,
        "adapter_type": adapter.get("type"),
        "adapter_name": adapter.get("name"),
        "embedding_source_gate_root": embedding.get("source_gate_root"),
    }


def _verify_shared_native_inputs(
    prepared_cases: Sequence[Mapping[str, Any]], identities: Sequence[Mapping[str, Any]]
) -> None:
    if len(prepared_cases) != len(identities):
        raise ValueError("shared native input identity count changed")
    for case, identity in zip(prepared_cases, identities, strict=True):
        prepared = case["prepared"]
        if (
            _pixel_tensor_sha256(prepared["native_inputs"]) != identity["pixel_values_sha256"]
            or _json_digest(list(prepared["executed_prompt_token_ids"])) != identity["prompt_token_ids_sha256"]
            or list(prepared["executed_media_sha256"]) != identity["media_sha256"]
        ):
            raise ValueError(f"shared native pixels/prompt/media changed for {case['row_id']}")


def _implementation_bindings() -> dict[str, Any]:
    native_module = sys.modules[prepare_native_inputs.__module__]
    requests_module = sys.modules[build_bound_native_requests.__module__]
    return {
        "semantic_assay": binding(Path(__file__)),
        "evaluation": binding(Path(evaluation.__file__)),
        "bound_requests": binding(Path(requests_module.__file__)),
        "native": binding(Path(native_module.__file__)),
        "coordinate_codebook": binding(Path(coordinate_codebook.__file__)),
    }


def _write_result(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run(plan_path: Path, admission_path: Path, output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    """Run and persist the frozen two-checkpoint assay, or persist a HOLD."""

    if output.exists():
        raise FileExistsError(f"refusing to replace semantic-assay output: {output}")
    inputs = _frozen_inputs(plan_path, admission_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = inputs["plan"]
    checkpoints = plan["checkpoint_bindings"]
    result: dict[str, Any] = {
        "schema": "coordinate_codebook_discriminator.semantic_assay_run.v1",
        "status": "HOLD",
        "scope": "fixed-prefix numerical sensitivity only",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output.resolve()),
        "plan": inputs["plan_binding"],
        "admission": inputs["admission_binding"],
        "dataset": inputs["dataset_binding"],
        "verified_input_bindings": inputs["input_bindings"],
        "tokenizer_bindings": [
            item for item in inputs["input_bindings"]
            if Path(item["path"]).name in {
                "tokenizer.json", "added_tokens.json", "tokenizer_config.json", "special_tokens_map.json"
            }
        ],
        "implementation_bindings": _implementation_bindings(),
        "checkpoint_bindings": {
            name: {"checkpoint_root": checkpoints[name]["checkpoint_root"],
                   "payload_bindings": inputs["checkpoint_payload_bindings"][name]}
            for name in ("early16", "late16")
        },
        "native_case_identities": [],
        "native_input_policy": (
            "Prepare one CPU native batch with the early16 processor and reuse the exact input tensors "
            "for late16; verify tensor, prompt, media, loaded tokenizer, common launch, and payload identity."
        ),
        "image_identity_note": (
            "image_plan_content_sha256 and processor media_sha256 identify different stages; "
            "only executed processor media_sha256 is compared with the frozen retained-cell media binding."
        ),
        "native_request_config_sha256": None,
        "common_runtime_identity": None,
        "checkpoints": {},
        "checkpoint_teacher_forwards": {},
        "teacher_forwards": 0,
        "max_teacher_forwards": MAX_FORWARDS_BOTH_CHECKPOINTS,
        "error": None,
    }
    started = time.monotonic()
    prepared_cases: list[dict[str, Any]] | None = None
    total_forwards = 0
    shared_runtime_identity: dict[str, Any] | None = None
    native_case_identities: list[dict[str, Any]] | None = None
    failure: BaseException | None = None
    for label in ("early16", "late16"):
        qwen = None
        hook = None
        counter = {"forwards": 0}
        try:
            qwen, identity = evaluation._load_runtime(
                str(checkpoints[label]["checkpoint_root"]), device, inputs["admission"]
            )
            _runtime_bindings(identity, checkpoints[label], inputs["admission"])
            loaded_ids = tuple(int(value) for value in qwen.token_identity.coordinate_token_ids)
            if loaded_ids != tuple(int(value) for value in plan["coordinate_token_ids"]):
                raise ValueError(f"{label} loaded coordinate tokenizer differs from the frozen plan")
            common_identity = _common_runtime_identity(identity)
            if shared_runtime_identity is None:
                shared_runtime_identity = common_identity
                result["common_runtime_identity"] = common_identity
            elif common_identity != shared_runtime_identity:
                raise ValueError(f"{label} common tokenizer/model launch identity differs from early16")
            if prepared_cases is None:
                prepared_cases, identities = _prepare_native_cases(
                    qwen,
                    plan_cases=plan["cases"],
                    admission=inputs["admission"],
                    dataset=inputs["dataset"],
                    input_rows=inputs["rows"],
                )
                result["native_case_identities"] = identities
                native_case_identities = identities
                result["native_request_config_sha256"] = identities[0]["native_request_config_sha256"]
            elif label == "late16":
                assert native_case_identities is not None
                if _json_digest(evaluation._config(inputs["admission"], inputs["dataset"])) != result["native_request_config_sha256"]:
                    raise ValueError("late16 native request configuration differs from early16")
                _verify_shared_native_inputs(prepared_cases, native_case_identities)
            model = qwen.model
            hook = model.register_forward_pre_hook(
                lambda *_args, **_kwargs: counter.__setitem__("forwards", counter["forwards"] + 1)
            )
            del model
            checkpoint_result = run_checkpoint_semantic_assay(
                qwen, checkpoint=label, prepared_cases=prepared_cases
            )
            hook.remove()
            hook = None
            observed_forwards = int(counter["forwards"])
            result["checkpoint_teacher_forwards"][label] = observed_forwards
            total_forwards += observed_forwards
            result["teacher_forwards"] = total_forwards
            if observed_forwards != MAX_FORWARDS_PER_CHECKPOINT or checkpoint_result["teacher_forwards"] != observed_forwards:
                raise AssertionError(f"{label} did not execute exactly 24 checked teacher forwards")
            if total_forwards > MAX_FORWARDS_BOTH_CHECKPOINTS:
                raise AssertionError("semantic assay exceeded the frozen 48 teacher-forward ceiling")
            checkpoint_result["checkpoint_identity"] = identity
            result["checkpoints"][label] = checkpoint_result
        except BaseException as exc:
            observed_forwards = int(counter["forwards"])
            if label not in result["checkpoint_teacher_forwards"]:
                result["checkpoint_teacher_forwards"][label] = observed_forwards
                total_forwards += observed_forwards
                result["teacher_forwards"] = total_forwards
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            failure = exc
        finally:
            if hook is not None:
                hook.remove()
            if qwen is not None:
                del qwen
            gc.collect()
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if failure is not None:
            break

    result["elapsed_seconds"] = time.monotonic() - started
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if failure is None and tuple(result["checkpoints"]) == ("early16", "late16") and total_forwards == MAX_FORWARDS_BOTH_CHECKPOINTS:
        result["status"] = "complete"
        result["interpretation_limit"] = (
            "Positive response supports fixed-prefix numerical sensitivity only. "
            "The y1 x-warp holds fixed GT x1 and does not establish whole-box transport or native benefit."
        )
    _write_result(output, result)
    if failure is not None:
        raise RuntimeError(f"semantic address assay persisted HOLD to {output}") from failure
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=FROZEN_PLAN_PATH)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = run(args.plan, args.admission, args.output, device=args.device)
    print(json.dumps({
        "status": result["status"],
        "teacher_forwards": result["teacher_forwards"],
        "output": result["output_path"],
    }))


if __name__ == "__main__":
    main()
