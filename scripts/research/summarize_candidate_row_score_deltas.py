#!/usr/bin/env python3
"""Pair complete candidate-row scores across matched prefix boundaries.

This is an experiment-local reporting tool.  It validates that the sorted and
random checkpoint scoring receipts used the same manifest and exact candidate
rows, then writes a compact JSON receipt and a human-readable Markdown table.
It reports likelihood deltas and ranks only; it intentionally does not infer a
mechanism from those values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CHECKPOINTS = ("sorted", "random")
METRICS = ("full_row", "description", "x1", "y1", "x2", "y2", "geometry", "closure")
SCHEMA_VERSION = "candidate_row_score_delta_summary.v1"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _as_list(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _pair_id(image_id: str, boundary: Mapping[str, Any]) -> str:
    return "::".join(
        (
            image_id,
            _string(boundary.get("case_id"), "boundary case_id"),
            _string(boundary.get("checkpoint_arm"), "boundary checkpoint_arm"),
            _string(boundary.get("comparison_id"), "boundary comparison_id"),
        )
    )


def _boundary_side(boundary_id: str) -> str:
    for side in ("left", "right"):
        if boundary_id.endswith(f"__{side}"):
            return side
    raise ValueError(f"boundary_id has no explicit left/right suffix: {boundary_id}")


def _index_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw_image in _as_list(manifest.get("images"), "manifest images"):
        image = _as_mapping(raw_image, "manifest image")
        image_id = _string(image.get("image_id"), "manifest image_id")
        for raw_boundary in _as_list(image.get("boundaries"), f"manifest {image_id} boundaries"):
            boundary = _as_mapping(raw_boundary, "manifest boundary")
            boundary_id = _string(boundary.get("boundary_id"), "manifest boundary_id")
            pair_id = _pair_id(image_id, boundary)
            side = _boundary_side(boundary_id)
            pair = indexed.setdefault(pair_id, {"image_id": image_id, "sides": {}})
            if side in pair["sides"]:
                raise ValueError(f"duplicate {side} boundary for {pair_id}")
            pair["sides"][side] = dict(boundary)
    for pair_id, pair in indexed.items():
        if set(pair["sides"]) != {"left", "right"}:
            raise ValueError(f"{pair_id} must have exactly one left and one right boundary")
        left = pair["sides"]["left"]
        right = pair["sides"]["right"]
        for key in ("case_id", "checkpoint_arm", "comparison_id"):
            if left[key] != right[key]:
                raise ValueError(f"{pair_id} left/right differ for {key}")
        if left["prefix_arm"] == right["prefix_arm"]:
            raise ValueError(f"{pair_id} left/right prefix arms must differ")
    return indexed


def _index_receipt(receipt: Mapping[str, Any], *, checkpoint: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_image in _as_list(receipt.get("images"), f"{checkpoint} receipt images"):
        image = _as_mapping(raw_image, f"{checkpoint} receipt image")
        image_id = _string(image.get("image_id"), f"{checkpoint} receipt image_id")
        for raw_boundary in _as_list(image.get("boundaries"), f"{checkpoint} receipt boundaries"):
            boundary = _as_mapping(raw_boundary, f"{checkpoint} receipt boundary")
            boundary_id = _string(boundary.get("boundary_id"), f"{checkpoint} receipt boundary_id")
            key = (image_id, boundary_id)
            if key in indexed:
                raise ValueError(f"{checkpoint} receipt duplicate boundary {key}")
            indexed[key] = boundary
    return indexed


def _receipt_images(receipt: Mapping[str, Any], *, checkpoint: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_image in _as_list(receipt.get("images"), f"{checkpoint} receipt images"):
        image = _as_mapping(raw_image, f"{checkpoint} receipt image")
        image_id = _string(image.get("image_id"), f"{checkpoint} receipt image_id")
        if image_id in result:
            raise ValueError(f"{checkpoint} receipt repeats image_id {image_id}")
        result[image_id] = image
    return result


def _runtime_validation(receipt: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    runtime = _as_mapping(receipt.get("runtime"), f"{checkpoint} runtime")
    if runtime.get("runtime_dtype_mode") != "fp32":
        raise ValueError(f"{checkpoint} receipt was not run with runtime_dtype_mode=fp32")
    if runtime.get("score_accumulation_dtype") != "torch.float32":
        raise ValueError(f"{checkpoint} receipt did not accumulate scores in torch.float32")
    model_dtype = _as_mapping(runtime.get("model_dtype"), f"{checkpoint} runtime model_dtype")
    dtype_names = _as_list(model_dtype.get("parameter_dtype_names"), f"{checkpoint} parameter dtype names")
    if list(dtype_names) != ["torch.float32"]:
        raise ValueError(f"{checkpoint} receipt model parameters are not exclusively torch.float32")
    return {
        "runtime_dtype_mode": runtime["runtime_dtype_mode"],
        "score_accumulation_dtype": runtime["score_accumulation_dtype"],
        "parameter_dtype_names": list(dtype_names),
        "backend": _as_mapping(runtime.get("backend_session"), f"{checkpoint} backend session").get("backend"),
        "attention_implementation": runtime.get("attention_implementation"),
        "effective_config_sha256": runtime.get("effective_config_sha256"),
    }


def _candidate_index(boundary: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in _as_list(boundary.get("candidate_scores"), f"{label} candidate_scores"):
        candidate = _as_mapping(raw_candidate, f"{label} candidate")
        candidate_id = _string(candidate.get("candidate_id"), f"{label} candidate_id")
        if candidate_id in result:
            raise ValueError(f"{label} repeats candidate_id {candidate_id}")
        result[candidate_id] = candidate
    return result


def _metric_value(candidate: Mapping[str, Any], metric: str, field: str) -> float:
    return _number(_as_mapping(candidate.get(metric), f"candidate {metric}").get(field), f"candidate {metric}.{field}")


def _candidate_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        result[metric] = {
            field: _metric_value(right, metric, field) - _metric_value(left, metric, field)
            for field in ("sum", "mean")
        }
    return result


def _rank_candidates(candidates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Full-row mean removes any possible row-length advantage.  Sum remains in
    # the output because the requested analysis needs both representations.
    ordered = sorted(
        candidates.values(),
        key=lambda item: (-_metric_value(item, "full_row", "mean"), -_metric_value(item, "full_row", "sum"), str(item["candidate_id"])),
    )
    return [
        {
            "rank": index + 1,
            "candidate_id": candidate["candidate_id"],
            "owner": candidate.get("owner"),
            "role": candidate.get("role"),
            "covered": candidate.get("covered"),
            "full_row": {
                "sum": _metric_value(candidate, "full_row", "sum"),
                "mean": _metric_value(candidate, "full_row", "mean"),
            },
        }
        for index, candidate in enumerate(ordered)
    ]


def _owner_candidates(manifest_boundary: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return left greedy owner, right greedy owner, and an ambiguity reason."""

    candidates = _as_list(manifest_boundary.get("candidates"), "manifest candidates")
    left = [item for item in candidates if "greedy_left_owner" in str(_as_mapping(item, "manifest candidate").get("role", ""))]
    right = [item for item in candidates if _as_mapping(item, "manifest candidate").get("role") == "greedy_right_owner"]
    if len(left) != 1:
        return None, None, "missing_or_nonunique_greedy_left_owner"
    if len(right) != 1:
        fallback = [item for item in candidates if "fallback" in str(_as_mapping(item, "manifest candidate").get("role", ""))]
        return str(_as_mapping(left[0], "left owner").get("candidate_id")), None, (
            "no_greedy_right_owner; sampled-owner fallback candidates are not treated as an observed greedy alternate"
            if fallback
            else "missing_or_nonunique_greedy_right_owner"
        )
    return (
        str(_as_mapping(left[0], "left owner").get("candidate_id")),
        str(_as_mapping(right[0], "right owner").get("candidate_id")),
        None,
    )


def _rank_position(ranking: Sequence[Mapping[str, Any]], candidate_id: str) -> int:
    for item in ranking:
        if item["candidate_id"] == candidate_id:
            return int(item["rank"])
    raise ValueError(f"candidate {candidate_id} missing from ranking")


def _alternate_favor(
    left_candidates: Mapping[str, Mapping[str, Any]],
    right_candidates: Mapping[str, Mapping[str, Any]],
    left_ranking: Sequence[Mapping[str, Any]],
    right_ranking: Sequence[Mapping[str, Any]],
    left_owner_id: str | None,
    right_owner_id: str | None,
) -> dict[str, Any]:
    if left_owner_id is None or right_owner_id is None:
        return {"status": "ambiguous", "favors_observed_alternate_owner": None}
    if left_owner_id not in left_candidates or right_owner_id not in left_candidates:
        raise ValueError("observed owner candidate absent from left score boundary")
    if left_owner_id not in right_candidates or right_owner_id not in right_candidates:
        raise ValueError("observed owner candidate absent from right score boundary")
    result: dict[str, Any] = {"status": "available"}
    for field in ("sum", "mean"):
        left_margin = _metric_value(left_candidates[right_owner_id], "full_row", field) - _metric_value(left_candidates[left_owner_id], "full_row", field)
        right_margin = _metric_value(right_candidates[right_owner_id], "full_row", field) - _metric_value(right_candidates[left_owner_id], "full_row", field)
        result[f"full_row_{field}_right_owner_relative_to_left_owner"] = {
            "left_prefix": left_margin,
            "right_prefix": right_margin,
            "right_minus_left": right_margin - left_margin,
        }
    result["favors_observed_alternate_owner"] = result["full_row_sum_right_owner_relative_to_left_owner"]["right_minus_left"] > 0.0
    result["observed_owner_ranks"] = {
        "left_prefix": {
            "left_owner": _rank_position(left_ranking, left_owner_id),
            "right_owner": _rank_position(left_ranking, right_owner_id),
        },
        "right_prefix": {
            "left_owner": _rank_position(right_ranking, left_owner_id),
            "right_owner": _rank_position(right_ranking, right_owner_id),
        },
    }
    return result


def _validate_boundary(
    manifest_boundary: Mapping[str, Any], receipt_boundary: Mapping[str, Any], *, label: str
) -> None:
    for key in ("boundary_id", "prefix_mode"):
        if receipt_boundary.get(key) != manifest_boundary.get(key):
            raise ValueError(f"{label} differs for {key}")
    manifest_prefix = _as_mapping(manifest_boundary.get("prefix"), f"{label} manifest prefix")
    # The receipt's top-level prefix hash is the executed base prompt plus the
    # selected generated history.  The manifest records only the selected
    # history; compare that against the receipt's explicit selector reference.
    recorded_prefix = _as_mapping(receipt_boundary.get("recorded_prefix_reference"), f"{label} recorded prefix")
    if recorded_prefix.get("token_ids_sha256") != manifest_prefix.get("token_ids_sha256"):
        raise ValueError(f"{label} selected-history token hash differs from manifest")
    expected_candidates = _as_list(manifest_boundary.get("candidates"), f"{label} manifest candidates")
    actual_candidates = _candidate_index(receipt_boundary, label)
    expected_by_id = {
        str(_as_mapping(item, f"{label} manifest candidate").get("candidate_id")): _as_mapping(item, f"{label} manifest candidate")
        for item in expected_candidates
    }
    expected_ids = set(expected_by_id)
    if set(actual_candidates) != expected_ids:
        raise ValueError(f"{label} candidate IDs differ from manifest")
    for candidate_id, candidate in actual_candidates.items():
        expected = expected_by_id[candidate_id]
        expected_row = _as_mapping(expected.get("row"), f"{label} manifest candidate row")
        if candidate.get("token_ids_sha256") != expected_row.get("token_ids_sha256"):
            raise ValueError(f"{label} candidate row token hash differs from manifest for {candidate_id}")
        for key in ("owner", "role", "covered", "category"):
            if candidate.get(key) != expected.get(key):
                raise ValueError(f"{label} candidate {candidate_id} differs from manifest for {key}")
    for metric in METRICS:
        for candidate in actual_candidates.values():
            for field in ("sum", "mean"):
                _metric_value(candidate, metric, field)
    terminal = _as_mapping(receipt_boundary.get("terminal_boundary"), f"{label} terminal boundary")
    _number(terminal.get("row_entry_minus_terminal"), f"{label} terminal row_entry_minus_terminal")


def summarize(
    *, manifest_path: Path, sorted_receipt_path: Path, random_receipt_path: Path
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    manifest_sha256 = _sha256_path(manifest_path)
    manifest_pairs = _index_manifest(manifest)

    receipts = {"sorted": _load_json(sorted_receipt_path), "random": _load_json(random_receipt_path)}
    runtime_validation = {checkpoint: _runtime_validation(receipts[checkpoint], checkpoint) for checkpoint in CHECKPOINTS}
    for checkpoint in CHECKPOINTS:
        receipt = receipts[checkpoint]
        if receipt.get("unit_id") != manifest.get("unit_id"):
            raise ValueError(f"{checkpoint} receipt unit_id differs from manifest")
        receipt_manifest = _as_mapping(receipt.get("manifest"), f"{checkpoint} receipt manifest")
        if receipt_manifest.get("sha256") != manifest_sha256:
            raise ValueError(f"{checkpoint} receipt manifest hash differs from supplied manifest")
    if receipts["sorted"]["manifest"]["sha256"] != receipts["random"]["manifest"]["sha256"]:
        raise ValueError("sorted/random receipt manifest hashes differ")

    receipt_indexes = {checkpoint: _index_receipt(receipts[checkpoint], checkpoint=checkpoint) for checkpoint in CHECKPOINTS}
    receipt_images = {checkpoint: _receipt_images(receipts[checkpoint], checkpoint=checkpoint) for checkpoint in CHECKPOINTS}
    expected_image_ids = {pair["image_id"] for pair in manifest_pairs.values()}
    if set(receipt_images["sorted"]) != expected_image_ids or set(receipt_images["random"]) != expected_image_ids:
        raise ValueError("sorted/random receipt image identifiers differ from manifest")
    image_identity: dict[str, Any] = {}
    for image_id in sorted(expected_image_ids):
        sorted_image = receipt_images["sorted"][image_id]
        random_image = receipt_images["random"][image_id]
        for key in ("source_image", "base_prompt_token_ids_sha256"):
            if sorted_image.get(key) != random_image.get(key):
                raise ValueError(f"sorted/random receipt image {image_id} differs for {key}")
        source_image = _as_mapping(sorted_image.get("source_image"), f"source image {image_id}")
        image_identity[image_id] = {
            "path": source_image.get("path"),
            "sha256": source_image.get("sha256"),
            "executed_media_sha256": source_image.get("executed_media_sha256"),
            "base_prompt_token_ids_sha256": sorted_image.get("base_prompt_token_ids_sha256"),
        }
    expected_keys = {
        (pair["image_id"], _string(boundary.get("boundary_id"), "manifest boundary_id"))
        for pair in manifest_pairs.values()
        for boundary in pair["sides"].values()
    }
    for checkpoint in CHECKPOINTS:
        actual_keys = set(receipt_indexes[checkpoint])
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"{checkpoint} receipt image/boundary set differs from manifest; missing={missing}, extra={extra}")

    pair_results: list[dict[str, Any]] = []
    ambiguity_count = 0
    for pair_id in sorted(manifest_pairs):
        manifest_pair = manifest_pairs[pair_id]
        left_manifest = manifest_pair["sides"]["left"]
        right_manifest = manifest_pair["sides"]["right"]
        left_boundary_id = _string(left_manifest.get("boundary_id"), "left boundary_id")
        right_boundary_id = _string(right_manifest.get("boundary_id"), "right boundary_id")
        left_owner_id, right_owner_id, owner_ambiguity = _owner_candidates(left_manifest)
        if owner_ambiguity:
            ambiguity_count += 1
        score_results: dict[str, Any] = {}
        for checkpoint in CHECKPOINTS:
            left = receipt_indexes[checkpoint][(manifest_pair["image_id"], left_boundary_id)]
            right = receipt_indexes[checkpoint][(manifest_pair["image_id"], right_boundary_id)]
            _validate_boundary(left_manifest, left, label=f"{checkpoint}/{pair_id}/left")
            _validate_boundary(right_manifest, right, label=f"{checkpoint}/{pair_id}/right")
            left_candidates = _candidate_index(left, f"{checkpoint}/{pair_id}/left")
            right_candidates = _candidate_index(right, f"{checkpoint}/{pair_id}/right")
            left_ranking = _rank_candidates(left_candidates)
            right_ranking = _rank_candidates(right_candidates)
            deltas = {
                candidate_id: _candidate_delta(left_candidates[candidate_id], right_candidates[candidate_id])
                for candidate_id in sorted(left_candidates)
            }
            left_terminal = _as_mapping(left.get("terminal_boundary"), f"{checkpoint}/{pair_id}/left terminal")
            right_terminal = _as_mapping(right.get("terminal_boundary"), f"{checkpoint}/{pair_id}/right terminal")
            score_results[checkpoint] = {
                "candidate_delta_right_minus_left": deltas,
                "candidate_rankings": {"left": left_ranking, "right": right_ranking},
                "terminal": {
                    "left_row_entry_minus_terminal": _number(left_terminal.get("row_entry_minus_terminal"), "left terminal margin"),
                    "right_row_entry_minus_terminal": _number(right_terminal.get("row_entry_minus_terminal"), "right terminal margin"),
                    "right_minus_left": _number(right_terminal.get("row_entry_minus_terminal"), "right terminal margin") - _number(left_terminal.get("row_entry_minus_terminal"), "left terminal margin"),
                },
                "observed_alternate_owner_check": _alternate_favor(
                    left_candidates,
                    right_candidates,
                    left_ranking,
                    right_ranking,
                    left_owner_id,
                    right_owner_id,
                ),
            }
        pair_results.append(
            {
                "pair_id": pair_id,
                "image_id": manifest_pair["image_id"],
                "case_id": left_manifest["case_id"],
                "checkpoint_arm_that_promoted_the_pair": left_manifest["checkpoint_arm"],
                "comparison_id": left_manifest["comparison_id"],
                "left_prefix_arm": left_manifest["prefix_arm"],
                "right_prefix_arm": right_manifest["prefix_arm"],
                "left_boundary_id": left_boundary_id,
                "right_boundary_id": right_boundary_id,
                "observed_owners": {
                    "left": left_owner_id,
                    "right": right_owner_id,
                    "ambiguity": owner_ambiguity,
                },
                "scores_by_checkpoint": score_results,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "interpretation": "paired score reporting only; candidate rows are not a common normalized distribution and terminal margins are reported separately",
            "pair_count": len(pair_results),
            "pair_count_with_observed_greedy_alternate_owner": len(pair_results) - ambiguity_count,
            "pair_count_with_owner_ambiguity": ambiguity_count,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            "receipts": {
                checkpoint: {"path": str(path), "sha256": _sha256_path(path)}
                for checkpoint, path in (("sorted", sorted_receipt_path), ("random", random_receipt_path))
            },
        },
        "validation": {
            "same_manifest_hash": True,
            "same_manifest_unit_id": True,
            "same_manifest_image_boundary_candidate_sets": True,
            "same_source_image_and_base_prompt_identity": True,
            "image_identity": image_identity,
            "runtime": runtime_validation,
        },
        "pairs": pair_results,
    }


def _markdown(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate-row score delta summary",
        "",
        "This is paired score reporting only. Candidate rows are not normalized into one shared probability distribution. The terminal margin is reported separately from row scores.",
        "",
        "## Validation",
        "",
        "- Both scoring receipts reference the same manifest hash and unit identifier.",
        "- Image identifiers, boundary identifiers, candidate identifiers, prefix token hashes, and score metric fields matched the manifest.",
        "- Both runs used full precision (`fp32`) model parameters and score accumulation.",
        "",
        "## Paired results",
        "",
        "Primary candidate ranks use full-row mean log probability (higher is better), with full-row sum as the tie-breaker.",
        "",
        "| Pair | Score checkpoint | Observed alternate owner | Alternate-owner relative sum change | Favours alternate? | Left ranks | Right ranks | Terminal margin change |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for pair in receipt["pairs"]:
        for checkpoint in CHECKPOINTS:
            scores = pair["scores_by_checkpoint"][checkpoint]
            alternate = scores["observed_alternate_owner_check"]
            if alternate["status"] == "available":
                change = alternate["full_row_sum_right_owner_relative_to_left_owner"]["right_minus_left"]
                favors = str(alternate["favors_observed_alternate_owner"])
                ranks = alternate["observed_owner_ranks"]
                left_ranks = f"L{ranks['left_prefix']['left_owner']}/R{ranks['left_prefix']['right_owner']}"
                right_ranks = f"L{ranks['right_prefix']['left_owner']}/R{ranks['right_prefix']['right_owner']}"
                alternate_name = pair["observed_owners"]["right"]
            else:
                change = None
                favors = "ambiguous"
                left_ranks = "n/a"
                right_ranks = "n/a"
                alternate_name = "n/a"
            terminal_change = scores["terminal"]["right_minus_left"]
            lines.append(
                "| {pair} | {checkpoint} | {alternate} | {change} | {favors} | {left_ranks} | {right_ranks} | {terminal:.6f} |".format(
                    pair=pair["pair_id"].replace("|", "\\|"),
                    checkpoint=checkpoint,
                    alternate=alternate_name,
                    change="n/a" if change is None else f"{change:.6f}",
                    favors=favors,
                    left_ranks=left_ranks,
                    right_ranks=right_ranks,
                    terminal=terminal_change,
                )
            )
    lines.extend(
        [
            "",
            "## Ambiguities",
            "",
        ]
    )
    ambiguities = [pair for pair in receipt["pairs"] if pair["observed_owners"]["ambiguity"]]
    if not ambiguities:
        lines.append("None.")
    else:
        for pair in ambiguities:
            lines.append(f"- `{pair['pair_id']}`: {pair['observed_owners']['ambiguity']}.")
    lines.extend(
        [
            "",
            "The JSON companion contains every candidate's right-minus-left delta for full row, description, each coordinate, geometry, and closure, plus the complete rankings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sorted-receipt", type=Path, required=True)
    parser.add_argument("--random-receipt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    receipt = summarize(
        manifest_path=args.manifest,
        sorted_receipt_path=args.sorted_receipt,
        random_receipt_path=args.random_receipt,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(receipt), encoding="utf-8")


if __name__ == "__main__":
    main()
