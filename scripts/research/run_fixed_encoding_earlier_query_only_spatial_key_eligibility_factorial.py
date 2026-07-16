#!/usr/bin/env python3
"""Run the image-139 earlier-query-only spatial-eligibility factorial.

The runner reuses the accepted query-only scorer and adds one missing causal
cell: regional image-key restriction on queries before current-row scoring.
It does not train, generate, resize, re-encode per arm, or modify shared model
code.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402
from scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover import (  # noqa: E402
    build_causal_key_eligibility_mask,
    derive_explicit_position_ids,
    first_differing_description_index,
    tensor_sha256,
)


UNIT_ID = "2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial"
QUERY_UNIT_ID = query.UNIT_ID
DEFAULT_QUERY_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/"
    "cohort-four-float32-20260715a/receipt.json"
)
QUERY_RECEIPT_SHA256 = "4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4"
IMAGE_ID = "139"
TOLERANCE = 1e-4

_FROZEN_QUERY_CASE: Mapping[str, Any] | None = None
_FROZEN_HARD_CASE: Mapping[str, Any] | None = None
_ORIGINAL_RUN_CASE = query._run_case


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_query_receipt(path: Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    observed = _sha256_file(resolved)
    if observed != QUERY_RECEIPT_SHA256:
        raise ValueError("query-only receipt SHA-256 does not match the frozen contract")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != QUERY_UNIT_ID:
        raise ValueError("query-only receipt has unexpected unit_id")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("query-only receipt has invalid results")
    by_id = {str(result.get("image_id")): result for result in results if isinstance(result, Mapping)}
    if set(by_id) != query.EXPECTED_ANCHOR_IDS or len(by_id) != len(results):
        raise ValueError("query-only receipt does not contain the exact frozen panel")
    return payload, observed


def earlier_query_range(*, prefix_length: int) -> tuple[int, int]:
    prefix = int(prefix_length)
    if prefix < 2:
        raise ValueError("prefix_length must be at least two")
    return 0, prefix - 2


def build_earlier_query_only_key_eligibility_mask(
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    prefix_length: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    length = int(sequence_length)
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    if any(value < 0 or value >= length for value in image | eligible):
        raise ValueError("image key position is outside the sequence")
    if not eligible.issubset(image):
        raise ValueError("eligible image keys must be a subset of image keys")
    _, query_end = earlier_query_range(prefix_length=prefix_length)
    if query_end >= length:
        raise ValueError("earlier-query range is outside the sequence")
    mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    blocked = image - eligible
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        mask[: query_end + 1, blocked_tensor] = False
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def inspect_factorial_mask_structure(
    earlier_mask: torch.Tensor,
    *,
    sequence_length: int,
    image_key_positions: Sequence[int],
    eligible_image_positions: Sequence[int],
    prefix_length: int,
    row_length: int,
) -> dict[str, Any]:
    """Prove the exact earlier/row partition and its all-query union."""

    length = int(sequence_length)
    if tuple(earlier_mask.shape) != (1, 1, length, length) or earlier_mask.dtype is not torch.bool:
        raise ValueError("earlier-only mask must be boolean [1,1,S,S]")
    device = earlier_mask.device
    baseline = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    observed = earlier_mask[0, 0]
    image = {int(value) for value in image_key_positions}
    eligible = {int(value) for value in eligible_image_positions}
    blocked = image - eligible
    _, earlier_end = earlier_query_range(prefix_length=prefix_length)
    row_start, row_end = query.query_row_range(
        prefix_length=prefix_length, row_length=row_length
    )
    expected = baseline.clone()
    if blocked:
        blocked_tensor = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        expected[: earlier_end + 1, blocked_tensor] = False
    row_mask = query.build_query_scoped_key_eligibility_mask(
        sequence_length=length,
        image_key_positions=sorted(image),
        eligible_image_positions=sorted(eligible),
        prefix_length=prefix_length,
        row_length=row_length,
        device=device,
    )[0, 0]
    all_mask = build_causal_key_eligibility_mask(
        sequence_length=length,
        image_key_positions=sorted(image),
        eligible_image_positions=sorted(eligible),
        device=device,
    )[0, 0]
    earlier_changed = observed != baseline
    row_changed = row_mask != baseline
    all_changed = all_mask != baseline
    relevant_end = int(prefix_length) + int(row_length) - 2
    relevant = slice(0, relevant_end + 1)
    union = earlier_changed[relevant] | row_changed[relevant]
    intersection = earlier_changed[relevant] & row_changed[relevant]
    non_image_changed = earlier_changed.clone()
    if image:
        image_tensor = torch.tensor(sorted(image), dtype=torch.long, device=device)
        non_image_changed[:, image_tensor] = False
    row_query_changes = int(earlier_changed[row_start : row_end + 1].sum().item())
    future_equal = torch.equal(
        torch.triu(observed, diagonal=1), torch.triu(baseline, diagonal=1)
    )
    receipt = {
        "passed": False,
        "earlier_query_range": {"start_inclusive": 0, "end_inclusive": earlier_end},
        "row_query_range": {"start_inclusive": row_start, "end_inclusive": row_end},
        "union_query_range": {"start_inclusive": 0, "end_inclusive": relevant_end},
        "excluded_unscored_final_query": int(prefix_length) + int(row_length) - 1,
        "blocked_image_key_count": len(blocked),
        "earlier_changed_cell_count": int(earlier_changed.sum().item()),
        "row_changed_cell_count": int(row_changed.sum().item()),
        "all_query_changed_cell_count_on_scored_slice": int(all_changed[relevant].sum().item()),
        "exact_expected_earlier_mask": bool(torch.equal(observed, expected)),
        "changed_row_scoring_cell_count": row_query_changes,
        "changed_non_image_key_cell_count": int(non_image_changed.sum().item()),
        "causal_future_key_blocking_unchanged": bool(future_equal),
        "earlier_and_row_changed_sets_disjoint": bool(not torch.any(intersection).item()),
        "earlier_row_union_equals_all_query_on_scored_slice": bool(
            torch.equal(union, all_changed[relevant])
        ),
        "mask_shape": list(earlier_mask.shape),
        "mask_dtype": str(earlier_mask.dtype),
    }
    receipt["passed"] = bool(
        receipt["blocked_image_key_count"] > 0
        and receipt["earlier_changed_cell_count"] > 0
        and receipt["exact_expected_earlier_mask"]
        and receipt["changed_row_scoring_cell_count"] == 0
        and receipt["changed_non_image_key_cell_count"] == 0
        and receipt["causal_future_key_blocking_unchanged"]
        and receipt["earlier_and_row_changed_sets_disjoint"]
        and receipt["earlier_row_union_equals_all_query_on_scored_slice"]
    )
    return receipt


def assess_cross_receipt_baseline(
    *, live: Mapping[str, Any], frozen: Mapping[str, Any], tolerance: float = TOLERANCE
) -> dict[str, Any]:
    owners: dict[str, Any] = {}
    passed = True
    for owner in ("target", "competitor"):
        live_owner = live[owner]
        frozen_owner = frozen[owner]
        drift = query._max_vector_drift(
            live_owner["token_log_probabilities"], frozen_owner["token_log_probabilities"]
        )
        ranks_equal = [int(value) for value in live_owner["selected_token_ranks"]] == [
            int(value) for value in frozen_owner["selected_token_ranks"]
        ]
        owners[owner] = {
            "max_abs_logprob_drift": drift,
            "selected_token_ranks_equal": ranks_equal,
            "passed": bool(drift <= tolerance and ranks_equal),
        }
        passed = passed and owners[owner]["passed"]
    return {"passed": bool(passed), "tolerance": float(tolerance), "owners": owners}


def _phase_means(arm: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    phase_map = query._phase_score_map(arm)
    return {
        phase: {
            "target": float(owners["target"]["mean"]),
            "competitor": float(owners["competitor"]["mean"]),
        }
        for phase, owners in phase_map.items()
    }


def compute_factorial(
    *,
    baseline: Mapping[str, Any],
    row_only: Mapping[str, Mapping[str, Any]],
    earlier_only: Mapping[str, Mapping[str, Any]],
    all_query: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute output-scale two-factor contrasts for each regional arm."""

    baseline_means = _phase_means(baseline)
    output: dict[str, Any] = {}
    for region in ("target", "competitor"):
        cells = {
            "unrestricted": baseline_means,
            "row_only": _phase_means(row_only[region]),
            "earlier_only": _phase_means(earlier_only[region]),
            "all_query": _phase_means(all_query[region]),
        }
        phases = set.intersection(*(set(value) for value in cells.values()))
        region_output: dict[str, Any] = {}
        for phase in sorted(phases):
            target_values = {cell: values[phase]["target"] for cell, values in cells.items()}
            competitor_values = {
                cell: values[phase]["competitor"] for cell, values in cells.items()
            }
            gammas = {
                cell: target_values[cell] - competitor_values[cell] for cell in cells
            }
            target_interaction = (
                target_values["all_query"]
                - target_values["earlier_only"]
                - target_values["row_only"]
                + target_values["unrestricted"]
            )
            competitor_interaction = (
                competitor_values["all_query"]
                - competitor_values["earlier_only"]
                - competitor_values["row_only"]
                + competitor_values["unrestricted"]
            )
            gamma_interaction = (
                gammas["all_query"]
                - gammas["earlier_only"]
                - gammas["row_only"]
                + gammas["unrestricted"]
            )
            region_output[phase] = {
                "target_row_means": target_values,
                "competitor_row_means": competitor_values,
                "gammas": gammas,
                "target_row_interaction": target_interaction,
                "competitor_row_interaction": competitor_interaction,
                "gamma_interaction": gamma_interaction,
                "interaction_identity_error": abs(
                    gamma_interaction - (target_interaction - competitor_interaction)
                ),
            }
        output[region] = region_output
    return output


def _score_earlier_arm(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    features: Any,
    grid_thw: Sequence[int],
    merge_size: int,
    prompt_ids: Sequence[int],
    target_row: Sequence[int],
    competitor_row: Sequence[int],
    image_token_id: int,
    eligible_indices: Sequence[int],
    image_grid_thw: torch.Tensor,
    terminal_token_id: int | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    scores: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for owner, row in (("target", target_row), ("competitor", competitor_row)):
        ids = torch.tensor([list(prompt_ids) + list(row)], dtype=torch.long, device=device)
        image_positions = [
            int(value) for value in torch.where(ids[0] == int(image_token_id))[0].tolist()
        ]
        eligible_positions = [image_positions[int(index)] for index in eligible_indices]
        positions = derive_explicit_position_ids(
            model,
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.long),
            image_grid_thw=image_grid_thw.to(device=device),
        )
        mask = build_earlier_query_only_key_eligibility_mask(
            sequence_length=ids.shape[1],
            image_key_positions=image_positions,
            eligible_image_positions=eligible_positions,
            prefix_length=len(prompt_ids),
            device=device,
        )
        structures[owner] = inspect_factorial_mask_structure(
            mask,
            sequence_length=ids.shape[1],
            image_key_positions=image_positions,
            eligible_image_positions=eligible_positions,
            prefix_length=len(prompt_ids),
            row_length=len(row),
        )
        scores[owner] = query._score_row(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            prompt_ids=prompt_ids,
            row=row,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            position_ids=positions,
            custom_mask=mask,
            terminal_token_id=terminal_token_id,
        )
        identities[owner] = {
            "input_ids_sha256": tensor_sha256(ids),
            "position_ids_sha256": tensor_sha256(positions),
        }
    first_diff = first_differing_description_index(
        target_row, competitor_row, description_length=len(target_row) - 8
    )
    arm: dict[str, Any] = {
        "target": scores["target"],
        "competitor": scores["competitor"],
        "structural_mask_receipt": structures,
        "input_and_position_identities": identities,
    }
    if first_diff is not None:
        arm["first_differing_description"] = {
            "target": {
                "sum": scores["target"]["token_log_probabilities"][first_diff],
                "mean": scores["target"]["token_log_probabilities"][first_diff],
                "count": 1,
            },
            "competitor": {
                "sum": scores["competitor"]["token_log_probabilities"][first_diff],
                "mean": scores["competitor"]["token_log_probabilities"][first_diff],
                "count": 1,
            },
        }
    return arm


def _run_case_factorial(**kwargs: Any) -> dict[str, Any]:
    if _FROZEN_QUERY_CASE is None or _FROZEN_HARD_CASE is None:
        raise RuntimeError("frozen factorial cases were not initialized")
    base = _ORIGINAL_RUN_CASE(**kwargs)
    target_earlier = _score_earlier_arm(
        model=kwargs["qwen"].model,
        model_inputs=kwargs["model_inputs"],
        features=kwargs["features"],
        grid_thw=kwargs["grid_thw"],
        merge_size=kwargs["merge_size"],
        prompt_ids=kwargs["prompt_ids"],
        target_row=kwargs["target_row"],
        competitor_row=kwargs["competitor_row"],
        image_token_id=kwargs["image_token_id"],
        eligible_indices=kwargs["target_indices"],
        image_grid_thw=kwargs["image_grid_thw"],
        terminal_token_id=kwargs["qwen"].tokenizer.eos_token_id,
    )
    competitor_earlier = _score_earlier_arm(
        model=kwargs["qwen"].model,
        model_inputs=kwargs["model_inputs"],
        features=kwargs["features"],
        grid_thw=kwargs["grid_thw"],
        merge_size=kwargs["merge_size"],
        prompt_ids=kwargs["prompt_ids"],
        target_row=kwargs["target_row"],
        competitor_row=kwargs["competitor_row"],
        image_token_id=kwargs["image_token_id"],
        eligible_indices=kwargs["competitor_indices"],
        image_grid_thw=kwargs["image_grid_thw"],
        terminal_token_id=kwargs["qwen"].tokenizer.eos_token_id,
    )
    frozen_baseline = _FROZEN_QUERY_CASE["arms"]["all_allowed_4d"]
    cross_receipt = assess_cross_receipt_baseline(
        live=base["arms"]["all_allowed_4d"], frozen=frozen_baseline
    )
    structure_passed = all(
        bool(receipt.get("passed"))
        for arm in (target_earlier, competitor_earlier)
        for receipt in arm["structural_mask_receipt"].values()
    )
    target_position_match = (
        target_earlier["input_and_position_identities"]["target"]["position_ids_sha256"]
        == str(_FROZEN_HARD_CASE.get("position_ids_sha256"))
    )
    factorial = compute_factorial(
        baseline=frozen_baseline,
        row_only={
            "target": _FROZEN_QUERY_CASE["arms"]["target_row_query_only_hard"],
            "competitor": _FROZEN_QUERY_CASE["arms"]["competitor_row_query_only_hard"],
        },
        earlier_only={"target": target_earlier, "competitor": competitor_earlier},
        all_query={
            "target": _FROZEN_HARD_CASE["arms"]["target_eligibility"],
            "competitor": _FROZEN_HARD_CASE["arms"]["competitor_eligibility"],
        },
    )
    base["earlier_query_only_arms"] = {
        "target": target_earlier,
        "competitor": competitor_earlier,
    }
    base["factorial"] = factorial
    base["factorial_execution_gate"] = {
        "passed": bool(
            base.get("no_op_trust_gate", {}).get("passed")
            and base.get("parent_continuity", {}).get("passed")
            and base.get("structural_mask_gate_passed")
            and cross_receipt["passed"]
            and structure_passed
            and target_position_match
        ),
        "cross_receipt_baseline": cross_receipt,
        "earlier_structural_receipts_passed": structure_passed,
        "target_position_ids_match_hard_parent": target_position_match,
    }
    base["classification"] = (
        "valid_factorial_execution"
        if base["factorial_execution_gate"]["passed"]
        else "invalid_factorial_execution_gate"
    )
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = query.build_parser()
    parser.description = __doc__
    parser.add_argument("--query-receipt", type=Path, default=DEFAULT_QUERY_RECEIPT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _FROZEN_QUERY_CASE, _FROZEN_HARD_CASE

    requested = query.validate_requested_image_ids(args.image_ids)
    if requested != [IMAGE_ID]:
        raise SystemExit("this factorial executes exactly image 139")
    try:
        frozen_query, query_sha = validate_query_receipt(args.query_receipt)
        hard_parent, hard_sha = query.validate_parent_receipt(args.parent_receipt)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if frozen_query.get("parent_receipt_sha256") != hard_sha:
        raise SystemExit("query-only receipt does not bind the frozen hard parent")
    query_by_id = {str(result["image_id"]): result for result in frozen_query["results"]}
    hard_by_id = {str(result["image_id"]): result for result in hard_parent["results"]}
    _FROZEN_QUERY_CASE = query_by_id[IMAGE_ID]
    _FROZEN_HARD_CASE = hard_by_id[IMAGE_ID]
    original = query._run_case
    query._run_case = _run_case_factorial
    try:
        payload = query.run(args)
    finally:
        query._run_case = original
    payload["schema_version"] = "fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial.v1"
    payload["unit_id"] = UNIT_ID
    payload["query_receipt"] = str(Path(args.query_receipt).expanduser().resolve())
    payload["query_receipt_sha256"] = query_sha
    payload["hard_parent_receipt_sha256"] = hard_sha
    result = payload["results"][0]
    result["factorial_execution_gate"]["feature_continuity_passed"] = bool(
        result.get("feature_continuity", {}).get("passed")
    )
    result["factorial_execution_gate"]["row_contract_passed"] = bool(
        result.get("row_contract", {}).get("passed")
    )
    result["factorial_execution_gate"]["passed"] = bool(
        result["factorial_execution_gate"]["passed"]
        and result["factorial_execution_gate"]["feature_continuity_passed"]
        and result["factorial_execution_gate"]["row_contract_passed"]
    )
    result["classification"] = (
        "valid_factorial_execution"
        if result["factorial_execution_gate"]["passed"]
        else "invalid_factorial_execution_gate"
    )
    payload["panel_decision"] = {
        "classification": result["classification"],
        "image_id": IMAGE_ID,
        "factorial_execution_gate_passed": result["factorial_execution_gate"]["passed"],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    payload = run(args)
    payload["runner_sha256"] = _sha256_file(Path(__file__))
    payload["normalized_argv"] = normalized_argv
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
