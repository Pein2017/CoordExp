"""Cold-validate the fixed native escape panel and write its final CPU receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet-v3.json"
RECORD_PATHS = [
    ROOT / "smoke-02" / "records.jsonl",
    ROOT / "full" / "rank-1" / "records.jsonl",
    ROOT / "full" / "rank-2" / "records.jsonl",
    ROOT / "full" / "rank-3" / "records.jsonl",
]
TERMINALS = [
    ROOT / "smoke-02" / "terminal.json",
    ROOT / "full" / "rank-1" / "terminal.json",
    ROOT / "full" / "rank-2" / "terminal.json",
    ROOT / "full" / "rank-3" / "terminal.json",
]
FAILED_SMOKE = ROOT / "smoke-01" / "terminal.json"
OUT = ROOT / "full" / "analysis.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    packet = json.loads(PACKET.read_text())
    cases = {item["case_id"]: item for item in packet["cases"]}
    records = [row for path in RECORD_PATHS for row in load_rows(path)]
    if len(records) != 15:
        raise ValueError(f"expected 15 fixed-panel records, got {len(records)}")
    keys = [(row["case_id"], row["job_id"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate fixed-panel cell")

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_case.setdefault(row["case_id"], []).append(row)
    case_receipts = []
    for case_id, case in cases.items():
        rows = by_case[case_id]
        by_job = {row["job_id"]: row for row in rows}
        natural = by_job["natural_anchor"]
        h_only = by_job["h_only"]
        h = case["h_ids"]
        baseline = case["baseline_action_ids"]
        natural_equal = natural["action_ids"] == baseline
        h_equal = h_only["action_ids"] == baseline
        h_prefix_equal = h_only["prefix_ids"] == h
        h_free_equal = h_only["free_ids"] == baseline[len(h):]
        if not all((natural_equal, h_equal, h_prefix_equal, h_free_equal)):
            raise ValueError(f"baseline parity failed for {case_id}")
        if natural["prefix_ids"] or natural["free_ids"] != natural["action_ids"]:
            raise ValueError(f"natural action partition failed for {case_id}")

        candidates = {item["candidate_id"]: item for item in case["candidates"]}
        candidate_receipts = []
        for row in rows:
            if row["kind"] != "h_plus_c":
                continue
            candidate_id = row["forced_candidate"]["candidate_id"]
            candidate = candidates[candidate_id]
            prefix_equal = row["prefix_ids"] == h + candidate["c_ids"]
            partition_equal = row["action_ids"] == row["prefix_ids"] + row["free_ids"]
            credit = row["credit_identity"]
            forced_excluded = (
                credit["forced_candidate_row_count"] == 1
                and credit["forced_candidate_in_free_counts"] is False
                and credit["natural_owner_recovery_eligible"] is False
                and credit["conditional_free_suffix_review_eligible"] is True
                and credit["candidate_id"] == candidate_id
            )
            if not all((prefix_equal, partition_equal, forced_excluded)):
                raise ValueError(f"candidate identity failed for {candidate_id}")
            candidate_receipts.append({
                "candidate_id": candidate_id,
                "prefix_equals_exact_h_plus_c": prefix_equal,
                "action_equals_prefix_plus_fresh_free": partition_equal,
                "forced_candidate_excluded_from_free_credit": forced_excluded,
                "fresh_free_token_count": len(row["free_ids"]),
                "native_eos_observed": row["native_eos_observed"],
                "stop_reason": row["stop_reason"],
            })
        expected_candidates = len(case["candidates"])
        if len(candidate_receipts) != expected_candidates:
            raise ValueError(f"candidate count failed for {case_id}")
        case_receipts.append({
            "case_id": case_id,
            "cell_count": len(rows),
            "natural_full_action_equals_frozen_baseline": natural_equal,
            "h_only_full_action_equals_frozen_baseline": h_equal,
            "h_only_prefix_equals_exact_native_h": h_prefix_equal,
            "h_only_free_equals_baseline_suffix_after_h": h_free_equal,
            "same_batch_identity_within_case": len({row["batch_identity_sha256"] for row in rows}) == 1,
            "batch_identity_sha256": rows[0]["batch_identity_sha256"],
            "candidates": candidate_receipts,
        })
    if not all(item["same_batch_identity_within_case"] for item in case_receipts):
        raise ValueError("batch identity drift within a case")

    terminals = [json.loads(path.read_text()) for path in TERMINALS]
    for terminal in terminals:
        if terminal["status"] != "completed" or terminal["exit_code"] != 0:
            raise ValueError(f"terminal not accepted: rank {terminal['rank']}")
        if terminal["bound_errors"] or terminal["readback"]["status"] != "accepted":
            raise ValueError(f"bound/readback failure: rank {terminal['rank']}")
        if terminal["elapsed_seconds"] > 1500:
            raise ValueError(f"wall bound exceeded: rank {terminal['rank']}")
        if terminal["peak_cuda_allocated_bytes"] > 12 * 1024**3:
            raise ValueError(f"CUDA allocated bound exceeded: rank {terminal['rank']}")
        if terminal["peak_cuda_reserved_bytes"] > 12 * 1024**3:
            raise ValueError(f"CUDA reserved bound exceeded: rank {terminal['rank']}")
        if terminal["peak_rss_bytes"] > 16 * 1024**3:
            raise ValueError(f"RSS bound exceeded: rank {terminal['rank']}")

    failed = json.loads(FAILED_SMOKE.read_text())
    payload = {
        "schema": "native_escape_witness.full_analysis.v1",
        "status": "technical_panel_accepted_scientific_visual_acceptance_pending_root",
        "packet": str(PACKET),
        "packet_sha256": sha(PACKET),
        "records": [{"path": str(path), "sha256": sha(path), "count": len(load_rows(path))}
                    for path in RECORD_PATHS],
        "combined_records": {
            "path": str(ROOT / "full" / "records.jsonl"),
            "sha256": sha(ROOT / "full" / "records.jsonl"),
            "count": len(records),
        },
        "case_checks": case_receipts,
        "accepted_runtime": {
            "generation_invocations": sum(item["continuations"] for item in terminals),
            "generated_tokens": sum(item["new_tokens"] for item in terminals),
            "image_forwards": sum(item["image_forwards"] for item in terminals),
            "model_forwards": sum(item["model_forwards"] for item in terminals),
            "model_loads": sum(item["model_loads"] for item in terminals),
            "selected_generated_token_bound": sum(item["selected_generated_token_bound"] for item in terminals),
            "max_elapsed_seconds": max(item["elapsed_seconds"] for item in terminals),
            "max_peak_cuda_allocated_bytes": max(item["peak_cuda_allocated_bytes"] for item in terminals),
            "max_peak_cuda_reserved_bytes": max(item["peak_cuda_reserved_bytes"] for item in terminals),
            "max_peak_rss_bytes": max(item["peak_rss_bytes"] for item in terminals),
            "terminals": [{"path": str(path), "sha256": sha(path)} for path in TERMINALS],
        },
        "preserved_technical_failure_cost": {
            "path": str(FAILED_SMOKE),
            "sha256": sha(FAILED_SMOKE),
            "status": failed["status"],
            "exit_code": failed["exit_code"],
            "continuations": failed["continuations"],
            "model_loads": failed["model_loads"],
            "elapsed_seconds": failed["elapsed_seconds"],
            "peak_cuda_allocated_bytes": failed["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": failed["peak_cuda_reserved_bytes"],
            "peak_rss_bytes": failed["peak_rss_bytes"],
        },
        "all_runtime_including_preserved_failure": {
            "generation_invocations": sum(item["continuations"] for item in terminals) + failed["continuations"],
            "model_loads": sum(item["model_loads"] for item in terminals) + failed["model_loads"],
            "generated_tokens": sum(item["new_tokens"] for item in terminals) + failed["new_tokens"],
        },
        "claim_boundary": (
            "This receipt proves exact native h and h+c consumption, fresh suffix partitioning, "
            "baseline parity, readback, and declared resource bounds. It does not grant visual "
            "owner identity, GT correctness, duplicate-free enumeration, or training efficacy."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
