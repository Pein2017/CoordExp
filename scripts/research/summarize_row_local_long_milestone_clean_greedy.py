#!/usr/bin/env python3
"""Validate and compare a receipt-defined clean-greedy checkpoint matrix."""

from __future__ import annotations

import argparse
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
    """Raised when the milestone matrix is incomplete or invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def summarize(
    *, matrix_receipt: Path, source_run_dir: Path, output_root: Path
) -> dict[str, Any]:
    receipt = _read_json(matrix_receipt.expanduser().resolve(strict=True))
    entries = receipt.get("entries")
    expected_entry_count = receipt.get("entry_count")
    if not isinstance(entries, list) or not entries:
        raise SummaryError("matrix receipt has no entries")
    if expected_entry_count != len(entries):
        raise SummaryError(
            "matrix receipt entry count mismatch: "
            f"declared={expected_entry_count!r}, observed={len(entries)}"
        )
    source = source_run_dir.expanduser().resolve(strict=True)
    source_summary = _read_json(source / "summary.json")
    source_artifact = source / "gt_vs_pred.jsonl"
    if source_summary.get("terminal_status") != "completed" or not source_artifact.is_file():
        raise SummaryError(f"Source run is incomplete: {source}")

    artifact_root = Path(str(receipt["inference_artifact_root"])).resolve(strict=True)
    out = output_root.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite comparison root: {out}")
    out.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise SummaryError("malformed matrix entry")
        run_name = str(raw_entry["run_name"])
        run_dir = artifact_root / run_name
        summary_path = run_dir / "summary.json"
        artifact_path = run_dir / "gt_vs_pred.jsonl"
        summary = _read_json(summary_path)
        required = {
            "terminal_status": "completed",
            "decode_success_count": 64,
            "row_count": 64,
            "parser_failure_count": 0,
            "score_failure_count": 0,
            "image_validation_failure_count": 0,
            "scored_artifact_materialized": True,
        }
        observed = {key: summary.get(key) for key in required}
        if observed != required:
            raise SummaryError(
                f"run artifact gate failed for {run_name}: "
                f"observed={observed!r}, expected={required!r}"
            )
        comparison = compare_artifacts(source_artifact, artifact_path)
        comparison_path = out / f"source-vs-{run_name}.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paired = comparison["paired_deltas"]
        geometry = comparison["common_owner_geometry"]
        arm_a = comparison["arm_a"]
        arm_b = comparison["arm_b"]
        stop_reasons = summary.get("decode_stop_reasons", {})
        rows.append(
            {
                "arm": str(raw_entry["arm"]),
                "step": int(raw_entry["step"]),
                "run_name": run_name,
                "config_fingerprint": str(raw_entry["config_fingerprint"]),
                "source_owner_count": int(arm_a["unique_matched_gt_owners"]),
                "treatment_owner_count": int(arm_b["unique_matched_gt_owners"]),
                "gained_owner_count": int(geometry["arm_b_only_owner_count"]),
                "lost_owner_count": int(geometry["arm_a_only_owner_count"]),
                "net_owner_delta": int(paired["unique_matched_gt_owners"]),
                "prediction_count_delta": int(paired["prediction_count"]),
                "strict_duplicate_candidate_delta": int(
                    paired["strict_physical_owner_duplicate_candidate_count"]
                ),
                "invalid_prediction_delta": int(paired["invalid_prediction_count"]),
                "common_owner_iou_delta": float(geometry["iou_delta"]["mean"]),
                "natural_stop_count": int(stop_reasons.get("im_end", 0)),
                "length_stop_count": int(stop_reasons.get("length", 0)),
                "truncated_decode_count": int(summary.get("truncated_decode_count", 0)),
                "comparison_path": str(comparison_path),
            }
        )

    rows.sort(key=lambda row: (str(row["arm"]), int(row["step"])))
    result = {
        "schema_version": "row_local_long_milestone_clean_greedy_summary.v1",
        "matrix_receipt": str(matrix_receipt.resolve()),
        "source_run_dir": str(source),
        "source_decode_stop_reasons": source_summary.get("decode_stop_reasons", {}),
        "row_count": len(rows),
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    header = tuple(key for key in rows[0] if key != "comparison_path")
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(row[key]) for key in header) for row in rows)
    (out / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-receipt", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        matrix_receipt=args.matrix_receipt,
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
    )
    print(json.dumps({"row_count": result["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
