"""Freeze the root-admitted native escape execution packet.

This CPU-only step binds the immutable candidate manifest, the separate root
visual receipt, exact native h and c tokens, prior deterministic h-only suffixes,
the actual source-record producer packet, and the task-local runner/reducer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-native-escape-witness"
)
SOURCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-small-owner-repeat-origin"
)
MANIFEST = ROOT / "candidate_manifest.json"
SOURCE_PACKET = SOURCE / "packet.json"
RUNNER = ROOT / "run_probe.py"
REDUCER = ROOT / "reduce.py"
BUILDER = ROOT / "build_execution_packet.py"
OUT = ROOT / "packet-v3.json"
COHORT = ("351017", "417044", "477415", "502725")
CAP = 3084


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_record(path: Path, case_id: str, job_id: str) -> dict[str, Any]:
    matches = [row for row in records(path)
               if row.get("case_id") == case_id and row.get("job_id") == job_id]
    require(len(matches) == 1, f"expected one {case_id}/{job_id} record in {path}")
    return matches[0]


def producer_packet(record: dict[str, Any]) -> dict[str, Any]:
    target = record["packet_sha256"]
    matches = [path for path in SOURCE.glob("packet*.json") if sha(path) == target]
    require(len(matches) == 1, f"cannot resolve unique source producer packet {target}")
    return {"path": str(matches[0]), "sha256": target}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    require(args.out.resolve() == OUT, "execution packet path is fixed")
    require(not OUT.exists(), "execution packet already exists")
    receipt_path = args.visual_receipt.resolve()
    require(receipt_path.is_file(), "visual receipt missing")
    manifest = json.loads(MANIFEST.read_text())
    source_packet = json.loads(SOURCE_PACKET.read_text())
    receipt = json.loads(receipt_path.read_text())
    require(manifest["schema"] == "native_escape_witness.candidate_manifest.v1",
            "candidate manifest schema differs")
    require(manifest["status"] == "candidate_only", "candidate manifest status differs")
    manifest_sha = sha(MANIFEST)
    require(receipt["schema"] == "repeat_recovery.root_visual_admission.v1",
            "visual receipt schema differs")
    require(receipt["status"] == "root_accepted_for_conditional_acquisition",
            "visual receipt is not accepted")
    require(receipt["candidate_manifest_sha256"] == manifest_sha,
            "visual receipt does not bind candidate manifest")
    accepted = {item["candidate_id"]: item for item in receipt["candidates"]}
    manifest_candidates = [c for case in manifest["cases"] for c in case["candidates"]]
    require(set(accepted) == {c["candidate_id"] for c in manifest_candidates},
            "visual receipt candidate set differs")

    packet_cases: list[dict[str, Any]] = []
    source_cases = {case["case_id"]: case for case in source_packet["cases"]}
    case_bounds: dict[str, int] = {}
    record_provenance: list[dict[str, Any]] = []
    for manifest_case in manifest["cases"]:
        case_id = manifest_case["case_id"]
        require(case_id == COHORT[len(packet_cases)], "cohort order differs")
        source_case = source_cases[case_id]
        record_path = Path(manifest_case["source_translated_record"]["path"])
        require(sha(record_path) == manifest_case["source_translated_record"]["sha256"],
                f"{case_id}: source record file changed")
        translated = one_record(record_path, case_id, "early_original_translated")
        h_control = one_record(record_path, case_id, "early_original_native")
        producer = producer_packet(translated)
        require(h_control["packet_sha256"] == translated["packet_sha256"],
                f"{case_id}: h/control producer packet differs")
        record_provenance.append({
            "case_id": case_id,
            "records_path": str(record_path),
            "records_sha256": sha(record_path),
            "producer_packet": producer,
            "manifest_source_packet_note": (
                "candidate_manifest.source_packet binds the original scientific packet; "
                "producer_packet above binds the actually executed repaired packet when different"
            ),
        })
        candidates: list[dict[str, Any]] = []
        for candidate in manifest_case["candidates"]:
            admitted = accepted[candidate["candidate_id"]]
            for field in ("case_id", "c_ids_sha256", "h_ids_sha256"):
                require(admitted[field] == (
                    candidate[field] if field != "h_ids_sha256"
                    else manifest_case["h_ids_sha256"]
                ), f"{case_id}/{candidate['candidate_id']}: receipt {field} differs")
            require(admitted["status"] == "root_accepted",
                    f"{candidate['candidate_id']}: candidate not root accepted")
            candidates.append({
                **candidate,
                "visual_admission": "root_accepted",
                "visual_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha(receipt_path),
                    "note": admitted["note"],
                    "figure": admitted["figure"],
                    "figure_sha256": admitted["figure_sha256"],
                },
            })
        h_ids = manifest_case["h_ids"]
        require(h_ids == source_case["baseline_action_ids"][:len(h_ids)],
                f"{case_id}: h is not literal natural baseline prefix")
        candidate_bound = sum(CAP - len(h_ids) - len(c["c_ids"]) for c in candidates)
        case_bound = CAP + (CAP - len(h_ids)) + candidate_bound
        case_bounds[case_id] = case_bound
        packet_cases.append({
            "case_id": case_id,
            "source_case": source_case["source_case"],
            "prompt_token_ids": manifest_case["prompt_token_ids"],
            "baseline_action_ids": manifest_case["baseline_action_ids"],
            "golden": source_case["golden"],
            "h_ids": h_ids,
            "h_ids_sha256": digest_json(h_ids),
            "h_source_job_id": "early_original_native",
            "h_complete_row_count": manifest_case["h_complete_row_count"],
            "expected_h_only_first512_ids": h_control["free_ids"],
            "h_control_source": {
                "records_path": str(record_path),
                "records_sha256": sha(record_path),
                "producer_packet": producer,
                "job_id": h_control["job_id"],
                "free_ids_sha256": digest_json(h_control["free_ids"]),
            },
            "candidates": candidates,
            "selected_generated_token_bound": case_bound,
        })

    cell_count = sum(2 + len(case["candidates"]) for case in packet_cases)
    token_bound = sum(case_bounds.values())
    require(cell_count == 15 and token_bound == 45619,
            f"unexpected frozen bounds: cells={cell_count}, tokens={token_bound}")
    source_files = source_packet["source_files"]
    for path, expected in source_files.items():
        require(sha(Path(path)) == expected, f"prior source byte changed: {path}")
    packet = {
        "schema": "native_escape_witness.v1",
        "status": "root_visually_admitted_not_launched",
        "cap": CAP,
        "source_packet": str(SOURCE_PACKET),
        "source_packet_sha256": sha(SOURCE_PACKET),
        "candidate_manifest": str(MANIFEST),
        "candidate_manifest_sha256": manifest_sha,
        "visual_admission_receipt": str(receipt_path),
        "visual_admission_receipt_sha256": sha(receipt_path),
        "runner_sha256": sha(RUNNER),
        "task_local_producers": {
            str(RUNNER): sha(RUNNER),
            str(REDUCER): sha(REDUCER),
            str(BUILDER): sha(BUILDER),
            str(ROOT / "build_candidate_manifest.py"): sha(
                ROOT / "build_candidate_manifest.py"),
        },
        "source_files": source_files,
        "source_record_provenance": record_provenance,
        "anchor_adapter": source_packet["anchor_adapter"],
        "config": source_packet["config"],
        "cases": packet_cases,
        "execution_limits": {
            "continuation_cells": cell_count,
            "selected_generated_token_bound": token_bound,
            "model_forward_bound": token_bound,
            "image_forward_bound": cell_count,
            "model_load_bound": 4,
            "score_replays": 0,
            "training_steps": 0,
            "per_process_wall_seconds": 1500,
            "per_process_peak_cuda_bytes": 12 * 1024**3,
            "per_process_peak_rss_bytes": 16 * 1024**3,
            "case_generated_token_bounds": case_bounds,
            "smoke": {
                "case_id": "351017",
                "physical_gpu": 0,
                "continuation_cells": 4,
                "generated_token_bound": case_bounds["351017"],
                "reused_as_final_rank": True,
            },
        },
        "claim_boundary": {
            "forced_candidate_is_not_native_recovery": True,
            "eos_is_not_positive": True,
            "suffix_must_be_generated_fresh": True,
            "translated_history_is_never_emitted_as_h": True,
            "visual_admission_is_not_gt_extent_certification": True,
            "training_authorized": False,
        },
    }
    OUT.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "packet": str(OUT),
        "packet_sha256": sha(OUT),
        "continuation_cells": cell_count,
        "generated_token_bound": token_bound,
        "case_generated_token_bounds": case_bounds,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
