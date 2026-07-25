#!/usr/bin/env python3
"""Summarize the frozen seven-case transition Phase Zero fixed-prefix panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCORE_SCHEMA = "complete_candidate_row_scoring.receipt.v1"
RELEASE_SCHEMA = "transition_phase0_fixed_prefix_release.receipt.v1"
UNCOVERED_ROLES = {
    "gained_uncovered_owner",
    "lost_uncovered_owner",
    "future_retained_uncovered_owner",
}
COVERED_ROLE = "covered_owner_control"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_sha(receipt: Mapping[str, Any], *, label: str) -> str:
    manifest = receipt.get("manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sha256"), str):
        raise ValueError(f"{label} lacks manifest.sha256")
    return str(manifest["sha256"])


def _score_cases(receipt: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    if receipt.get("schema_version") != SCORE_SCHEMA:
        raise ValueError(f"{label} score receipt schema differs")
    images = receipt.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{label} score receipt lacks images")
    cases: dict[str, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError(f"{label} image is malformed")
        image_id = str(image.get("image_id"))
        boundaries = image.get("boundaries")
        if not isinstance(boundaries, list) or len(boundaries) != 1:
            raise ValueError(f"{label} image {image_id} must have one boundary")
        boundary = boundaries[0]
        if not isinstance(boundary, Mapping):
            raise ValueError(f"{label} image {image_id} boundary is malformed")
        case_id = str(boundary.get("boundary_id"))
        terminal = boundary.get("terminal_boundary")
        candidates = boundary.get("candidate_scores")
        if not isinstance(terminal, Mapping) or not isinstance(candidates, list):
            raise ValueError(f"{label} case {case_id} lacks scores")
        if case_id in cases:
            raise ValueError(f"{label} repeats case {case_id}")
        candidate_map: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{label} case {case_id} candidate is malformed")
            candidate_id = str(candidate.get("candidate_id"))
            full_row = candidate.get("full_row")
            row_entry = candidate.get("row_entry")
            if not isinstance(full_row, Mapping) or not isinstance(row_entry, Mapping):
                raise ValueError(f"{label} candidate {candidate_id} lacks row scores")
            token_count = int(full_row.get("count", 0))
            if token_count <= 1 or int(row_entry.get("count", 0)) != 1:
                raise ValueError(f"{label} candidate {candidate_id} has invalid counts")
            suffix_sum = float(full_row["sum"]) - float(row_entry["sum"])
            candidate_map[candidate_id] = {
                "candidate_id": candidate_id,
                "role": str(candidate.get("role")),
                "covered": candidate.get("covered"),
                "owner": candidate.get("owner"),
                "category": candidate.get("category"),
                "suffix_sum": suffix_sum,
                "suffix_token_mean": suffix_sum / (token_count - 1),
                "description_mean": float(candidate["description"]["mean"]),
                "geometry_mean": float(candidate["geometry"]["mean"]),
            }
        cases[case_id] = {
            "case_id": case_id,
            "image_id": image_id,
            "continue_minus_stop": float(terminal["row_entry_minus_terminal"]),
            "candidates": candidate_map,
        }
    return cases


def _release_rows(release_root: Path, manifest_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(release_root.rglob("*.json")):
        receipt = _read_json(path)
        if receipt.get("schema_version") != RELEASE_SCHEMA:
            raise ValueError(f"release receipt schema differs: {path}")
        if _manifest_sha(receipt, label=str(path)) != manifest_sha:
            raise ValueError(f"release receipt manifest differs: {path}")
        checkpoint = receipt.get("checkpoint")
        plan = receipt.get("release_plan")
        released = receipt.get("released_row")
        if not all(isinstance(value, Mapping) for value in (checkpoint, plan, released)):
            raise ValueError(f"release receipt is malformed: {path}")
        strict = [str(value) for value in released.get("strict_matched_owner_ids", [])]
        uncovered = [str(value) for value in released.get("uncovered_ledger_owner_ids", [])]
        covered = [str(value) for value in released.get("covered_prefix_owner_ids", [])]
        rows.append(
            {
                "path": str(path.resolve()),
                "checkpoint_role": str(checkpoint.get("checkpoint_role")),
                "case_id": str(plan.get("case_id")),
                "force_mode": str(plan.get("force_mode")),
                "expected_owner_id": plan.get("expected_owner_id"),
                "status": str(released.get("status")),
                "raw_generated_text": released.get("raw_generated_text"),
                "strict_matched_owner_ids": strict,
                "uncovered_owner_ids": uncovered,
                "covered_owner_ids_repeated": covered,
                "intended_owner_realized": released.get("intended_owner_realized"),
            }
        )
    return rows


def summarize(source_path: Path, treatment_path: Path, release_root: Path) -> dict[str, Any]:
    source_receipt = _read_json(source_path)
    treatment_receipt = _read_json(treatment_path)
    source_sha = _manifest_sha(source_receipt, label="Source")
    treatment_sha = _manifest_sha(treatment_receipt, label="transition step 36")
    if source_sha != treatment_sha:
        raise ValueError("score receipts use different manifests")
    source = _score_cases(source_receipt, label="Source")
    treatment = _score_cases(treatment_receipt, label="transition step 36")
    if set(source) != set(treatment):
        raise ValueError("score receipts have different case sets")

    cases: list[dict[str, Any]] = []
    for case_id in sorted(source):
        source_case = source[case_id]
        treatment_case = treatment[case_id]
        if set(source_case["candidates"]) != set(treatment_case["candidates"]):
            raise ValueError(f"candidate sets differ for {case_id}")
        candidate_rows: list[dict[str, Any]] = []
        for candidate_id in sorted(source_case["candidates"]):
            src = source_case["candidates"][candidate_id]
            tx = treatment_case["candidates"][candidate_id]
            for field in ("role", "covered", "owner", "category"):
                if src[field] != tx[field]:
                    raise ValueError(f"candidate metadata differs for {candidate_id}")
            candidate_rows.append(
                {
                    **{key: src[key] for key in ("candidate_id", "role", "covered", "owner", "category")},
                    "source_suffix_sum": src["suffix_sum"],
                    "treatment_suffix_sum": tx["suffix_sum"],
                    "delta_suffix_sum": tx["suffix_sum"] - src["suffix_sum"],
                    "source_suffix_token_mean": src["suffix_token_mean"],
                    "treatment_suffix_token_mean": tx["suffix_token_mean"],
                    "delta_description_mean": tx["description_mean"] - src["description_mean"],
                    "delta_geometry_mean": tx["geometry_mean"] - src["geometry_mean"],
                }
            )
        routing: dict[str, Any] | None = None
        uncovered = [row for row in candidate_rows if row["role"] in UNCOVERED_ROLES]
        covered = [row for row in candidate_rows if row["role"] == COVERED_ROLE]
        if uncovered and covered:
            source_gap = max(row["source_suffix_sum"] for row in uncovered) - max(
                row["source_suffix_sum"] for row in covered
            )
            treatment_gap = max(row["treatment_suffix_sum"] for row in uncovered) - max(
                row["treatment_suffix_sum"] for row in covered
            )
            routing = {
                "source_best_uncovered_minus_best_covered": source_gap,
                "treatment_best_uncovered_minus_best_covered": treatment_gap,
                "delta_best_uncovered_minus_best_covered": treatment_gap - source_gap,
            }
        source_margin = source_case["continue_minus_stop"]
        treatment_margin = treatment_case["continue_minus_stop"]
        cases.append(
            {
                "case_id": case_id,
                "image_id": source_case["image_id"],
                "source_continue_minus_stop": source_margin,
                "treatment_continue_minus_stop": treatment_margin,
                "delta_continue_minus_stop": treatment_margin - source_margin,
                "source_prefers_continue": source_margin > 0,
                "treatment_prefers_continue": treatment_margin > 0,
                "routing": routing,
                "candidates": candidate_rows,
            }
        )

    releases = _release_rows(release_root, source_sha)
    complete = [row for row in releases if row["force_mode"] == "complete-description"]
    opener = [row for row in releases if row["force_mode"] == "opener"]
    roles = ("source", "transition-step36")
    release_aggregate = {
        role: {
            "opener_run_count": sum(row["checkpoint_role"] == role for row in opener),
            "opener_with_uncovered_owner_count": sum(
                row["checkpoint_role"] == role and bool(row["uncovered_owner_ids"])
                for row in opener
            ),
            "complete_description_run_count": sum(
                row["checkpoint_role"] == role for row in complete
            ),
            "intended_owner_realized_count": sum(
                row["checkpoint_role"] == role and row["intended_owner_realized"] is True
                for row in complete
            ),
            "complete_description_with_any_uncovered_owner_count": sum(
                row["checkpoint_role"] == role and bool(row["uncovered_owner_ids"])
                for row in complete
            ),
            "covered_owner_repeat_count": sum(
                len(row["covered_owner_ids_repeated"])
                for row in releases
                if row["checkpoint_role"] == role
            ),
        }
        for role in roles
    }
    routing_cases = [case for case in cases if case["routing"] is not None]
    return {
        "schema_version": "transition_phase0_fixed_prefix_panel.summary.v1",
        "manifest_sha256": source_sha,
        "case_count": len(cases),
        "candidate_count": sum(len(case["candidates"]) for case in cases),
        "aggregate": {
            "boundary_margin_increased_count": sum(
                case["delta_continue_minus_stop"] > 0 for case in cases
            ),
            "source_stop_to_treatment_continue_count": sum(
                not case["source_prefers_continue"] and case["treatment_prefers_continue"]
                for case in cases
            ),
            "routing_case_count": len(routing_cases),
            "best_uncovered_vs_covered_gap_improved_count": sum(
                case["routing"]["delta_best_uncovered_minus_best_covered"] > 0
                for case in routing_cases
            ),
            "release": release_aggregate,
        },
        "cases": cases,
        "releases": releases,
        "claim_boundary": (
            "Fixed-prefix independent candidate scores are not a normalized candidate distribution, "
            "and forced release is not a free-rollout final-set outcome."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-score-receipt", type=Path, required=True)
    parser.add_argument("--treatment-score-receipt", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.source_score_receipt.resolve(strict=True),
        args.treatment_score_receipt.resolve(strict=True),
        args.release_root.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": summary["case_count"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
