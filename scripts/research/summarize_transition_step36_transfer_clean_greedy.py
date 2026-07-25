#!/usr/bin/env python3
"""Validate and compare the matched Source/transition-step36 transfer runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    compare_artifacts,
)


class SummaryError(ValueError):
    """Raised when a conclusion-bearing transfer artifact is incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def _stop_count(summary: dict[str, Any], reason: str) -> int:
    reasons = summary.get("decode_stop_reasons", {})
    if not isinstance(reasons, dict):
        raise SummaryError("decode_stop_reasons must be an object")
    value = reasons.get(reason, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SummaryError(f"invalid {reason!r} stop count: {value!r}")
    return value


def _row_id_counts(refs: Any) -> Counter[str]:
    if not isinstance(refs, list):
        raise SummaryError("owner reference collection must be a list")
    counts: Counter[str] = Counter()
    for value in refs:
        if not isinstance(value, dict) or value.get("row_id") is None:
            raise SummaryError("owner reference is missing row_id")
        counts[str(value["row_id"])] += 1
    return counts


def _parallelism_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    parallelism = manifest.get("parallelism")
    if not isinstance(parallelism, dict):
        raise SummaryError("run manifest is missing parallelism metadata")
    keys = (
        "execution_mode",
        "active_ranks",
        "per_device_batch_size",
        "shard_plan_fingerprint",
        "visible_cuda_tokens",
        "rank_to_device",
        "merge_status",
    )
    identity = {key: parallelism.get(key) for key in keys}
    required = {
        "execution_mode": "controller_worker",
        "active_ranks": 8,
        "per_device_batch_size": 4,
        "merge_status": "completed",
    }
    if {key: identity[key] for key in required} != required:
        raise SummaryError(f"unexpected parallelism identity: {identity!r}")
    return identity


def _image_level_owner_delta(
    geometry: dict[str, Any], *, expected_row_count: int
) -> dict[str, Any]:
    source_only = _row_id_counts(geometry.get("arm_a_only_owner_refs"))
    treatment_only = _row_id_counts(geometry.get("arm_b_only_owner_refs"))
    row_ids = source_only.keys() | treatment_only.keys()
    deltas = {
        row_id: treatment_only[row_id] - source_only[row_id] for row_id in row_ids
    }
    positive = sum(delta > 0 for delta in deltas.values())
    negative = sum(delta < 0 for delta in deltas.values())
    zero = expected_row_count - positive - negative
    if zero < 0:
        raise SummaryError("image-level owner delta accounting exceeds row count")
    net = sum(deltas.values())
    expected_net = int(geometry["arm_b_only_owner_count"]) - int(
        geometry["arm_a_only_owner_count"]
    )
    if net != expected_net:
        raise SummaryError(
            f"image-level owner delta mismatch: observed={net}, expected={expected_net}"
        )
    return {
        "positive_image_count": positive,
        "negative_image_count": negative,
        "unchanged_image_count": zero,
        "net_owner_delta": net,
        "mean_owner_delta_per_image": net / expected_row_count,
        "nonzero_image_deltas": dict(sorted(deltas.items())),
    }


def _validate_run(
    *,
    run_dir: Path,
    expected_rows: int,
    expected_fingerprint: str,
    expected_input_jsonl: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    required = {
        "terminal_status": "completed",
        "decode_success_count": expected_rows,
        "row_count": expected_rows,
        "raw_row_count": expected_rows,
        "diagnostic_row_count": expected_rows,
        "scored_row_count": expected_rows,
        "parser_failure_count": 0,
        "score_failure_count": 0,
        "image_validation_failure_count": 0,
        "scored_artifact_materialized": True,
    }
    observed = {key: summary.get(key) for key in required}
    if observed != required:
        raise SummaryError(
            f"artifact gate failed for {run_dir.name}: "
            f"observed={observed!r}, expected={required!r}"
        )
    expected_generation = {
        "batch_size": 4,
        "max_new_tokens": 3084,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    generation = summary.get("generation_policy")
    if not isinstance(generation, dict):
        raise SummaryError(f"missing generation policy for {run_dir.name}")
    observed_generation = {key: generation.get(key) for key in expected_generation}
    if observed_generation != expected_generation:
        raise SummaryError(
            f"generation policy mismatch for {run_dir.name}: "
            f"observed={observed_generation!r}, expected={expected_generation!r}"
        )
    if generation.get("do_sample") is not False or generation.get("output_scores") is not True:
        raise SummaryError(f"non-greedy or unscored generation policy for {run_dir.name}")
    fingerprints = manifest.get("resolved_config_fingerprints")
    if not isinstance(fingerprints, dict) or fingerprints.get("infer_config") != expected_fingerprint:
        raise SummaryError(f"config fingerprint mismatch for {run_dir.name}")
    dataset_identity = manifest.get("dataset_identity")
    if not isinstance(dataset_identity, dict) or dataset_identity.get(
        "input_jsonl"
    ) != expected_input_jsonl:
        raise SummaryError(f"dataset identity mismatch for {run_dir.name}")
    if manifest.get("generation_policy") != generation:
        raise SummaryError(f"summary/manifest generation policy mismatch for {run_dir.name}")
    if not (run_dir / "gt_vs_pred.jsonl").is_file():
        raise SummaryError(f"missing gt_vs_pred.jsonl for {run_dir.name}")
    if not (run_dir / "gt_vs_pred_scored.jsonl").is_file():
        raise SummaryError(f"missing scored artifact for {run_dir.name}")
    return summary, manifest


def summarize(*, matrix_receipt: Path, output_root: Path) -> dict[str, Any]:
    receipt_path = matrix_receipt.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != "transition_step36_transfer_infer_matrix.v1":
        raise SummaryError("unexpected matrix receipt schema")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != 4:
        raise SummaryError("transfer receipt must contain exactly four entries")
    if receipt.get("entry_count") != len(entries):
        raise SummaryError("transfer receipt entry count mismatch")
    artifact_root = Path(str(receipt["inference_artifact_root"])).resolve(strict=True)
    out = output_root.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite comparison root: {out}")
    out.mkdir(parents=True)

    by_split_role: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise SummaryError("malformed transfer matrix entry")
        key = (str(entry["split"]), str(entry["role"]))
        if key in by_split_role:
            raise SummaryError(f"duplicate transfer matrix entry: {key}")
        by_split_role[key] = entry

    rows: list[dict[str, Any]] = []
    parallelism_by_run: dict[str, Any] = {}
    for split in ("development", "heldout"):
        split_receipt = receipt.get("splits", {}).get(split)
        if not isinstance(split_receipt, dict):
            raise SummaryError(f"missing receipt for split {split}")
        expected_rows = int(split_receipt["row_count"])
        input_jsonl = str(split_receipt["input_jsonl"])
        run_data: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
        for role in ("source", "transition-step36"):
            entry = by_split_role.get((split, role))
            if entry is None:
                raise SummaryError(f"missing matrix entry for {split}/{role}")
            if int(entry["expected_row_count"]) != expected_rows:
                raise SummaryError(f"receipt row count mismatch for {split}/{role}")
            run_dir = artifact_root / str(entry["run_name"])
            summary, manifest = _validate_run(
                run_dir=run_dir,
                expected_rows=expected_rows,
                expected_fingerprint=str(entry["config_fingerprint"]),
                expected_input_jsonl=input_jsonl,
            )
            run_data[role] = (run_dir, summary, manifest)
            parallelism_by_run[run_dir.name] = _parallelism_identity(manifest)

        source_dir, source_summary, source_manifest = run_data["source"]
        treatment_dir, treatment_summary, treatment_manifest = run_data[
            "transition-step36"
        ]
        if _parallelism_identity(source_manifest) != _parallelism_identity(
            treatment_manifest
        ):
            raise SummaryError(f"Source/treatment parallelism mismatch for {split}")
        comparison = compare_artifacts(
            source_dir / "gt_vs_pred.jsonl",
            treatment_dir / "gt_vs_pred.jsonl",
        )
        comparison_path = out / f"{split}-source-vs-transition-step36.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paired = comparison["paired_deltas"]
        geometry = comparison["common_owner_geometry"]
        arm_a = comparison["arm_a"]
        arm_b = comparison["arm_b"]
        image_deltas = _image_level_owner_delta(
            geometry, expected_row_count=expected_rows
        )
        rows.append(
            {
                "split": split,
                "row_count": expected_rows,
                "source_owner_count": int(arm_a["unique_matched_gt_owners"]),
                "treatment_owner_count": int(arm_b["unique_matched_gt_owners"]),
                "gained_owner_count": int(geometry["arm_b_only_owner_count"]),
                "lost_owner_count": int(geometry["arm_a_only_owner_count"]),
                "net_owner_delta": int(paired["unique_matched_gt_owners"]),
                "positive_image_count": int(image_deltas["positive_image_count"]),
                "negative_image_count": int(image_deltas["negative_image_count"]),
                "unchanged_image_count": int(image_deltas["unchanged_image_count"]),
                "prediction_count_delta": int(paired["prediction_count"]),
                "strict_duplicate_candidate_delta": int(
                    paired["strict_physical_owner_duplicate_candidate_count"]
                ),
                "invalid_prediction_delta": int(paired["invalid_prediction_count"]),
                "common_owner_iou_delta": float(geometry["iou_delta"]["mean"]),
                "source_natural_stop_count": _stop_count(source_summary, "im_end"),
                "source_length_stop_count": _stop_count(source_summary, "length"),
                "treatment_natural_stop_count": _stop_count(
                    treatment_summary, "im_end"
                ),
                "treatment_length_stop_count": _stop_count(
                    treatment_summary, "length"
                ),
                "source_truncated_decode_count": int(
                    source_summary.get("truncated_decode_count", 0)
                ),
                "treatment_truncated_decode_count": int(
                    treatment_summary.get("truncated_decode_count", 0)
                ),
                "comparison_path": str(comparison_path),
                "image_level_owner_delta": image_deltas,
            }
        )

    result = {
        "schema_version": "transition_step36_transfer_clean_greedy_summary.v1",
        "matrix_receipt": str(receipt_path),
        "artifact_root": str(artifact_root),
        "generation": receipt["generation"],
        "parallelism_by_run": parallelism_by_run,
        "split_count": len(rows),
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    table_rows = [
        {key: value for key, value in row.items() if key not in {"comparison_path", "image_level_owner_delta"}}
        for row in rows
    ]
    header = tuple(table_rows[0])
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(row[key]) for key in header) for row in table_rows)
    (out / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        matrix_receipt=args.matrix_receipt,
        output_root=args.output_root,
    )
    print(json.dumps({"split_count": result["split_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
