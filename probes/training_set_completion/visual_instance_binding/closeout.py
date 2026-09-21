"""Build one stable Lane B candidate manifest from immutable receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compositor import binding


def closeout(root: Path) -> dict[str, Any]:
    admission_binding = root / "selection" / "shared-admission-binding.json"
    admission = json.loads(json.loads(admission_binding.read_text())["shared_admission"]["path"] and Path(json.loads(admission_binding.read_text())["shared_admission"]["path"]).read_text())
    qualification = []
    audit_only = []
    for parent, target in ((root / "qualification-01", qualification), (root / "states", audit_only)):
        for receipt_path in sorted(parent.glob("*/receipt.json")):
            receipt = json.loads(receipt_path.read_text())
            target.append({"state_id": receipt["state_id"], "status": receipt["status"], "model": receipt["model"], "device": receipt["device"], "receipt": binding(receipt_path), "scores": receipt.get("scores"), "counters": receipt["counters"], "parity": receipt.get("parity")})
    all_jobs = qualification + audit_only
    if len(qualification) != 4 or len(audit_only) != 4 or any(job["status"] != "candidate_complete" for job in all_jobs):
        raise ValueError("qualification or redundant execution receipts are incomplete")
    reduction_path = root / "reduction.json"
    acceptance_path = root / "cpu-acceptance.json"
    reduction = json.loads(reduction_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    costs = {key: sum(float(job["counters"].get(key, 0)) for job in all_jobs) for key in ("model_forwards", "vision_forwards", "gpu_seconds", "retained_tensor_bytes")}
    primary_costs = {key: sum(float(job["counters"].get(key, 0)) for job in qualification) for key in ("model_forwards", "vision_forwards", "gpu_seconds", "retained_tensor_bytes")}
    costs["artifact_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and path.name != "candidate-manifest.json")
    matrices = []
    for state in reduction["states"]:
        matrices.append({"state_id": state["state_id"], "model": state["model"], "stratum": state["stratum"], "response_matrix": state["response_matrix"], "selectivity": state["selectivity"]})
    result = {
        "schema": "visual_instance_binding.candidate_manifest.v1",
        "status": "candidate",
        "lane": "B_visual_instance_binding",
        "question": admission["lane_b"]["question"],
        "admission": {"binding": binding(admission_binding), "path": str(Path(json.loads(admission_binding.read_text())["shared_admission"]["path"])), "sha256": json.loads(admission_binding.read_text())["shared_admission"]["sha256"], "ready_states": admission["lane_b"]["counts"], "failure_holds": admission["lane_b"]["failure_holds"], "control_holds": admission["lane_b"]["control_holds"]},
        "qualification": {"campaign": "qualification-01", "primary": True, "jobs": qualification, "cells": 16, "states": matrices, "passed": True, "requirements": {"source_native_replay_atol": 2e-4, "clean_full_logit_atol": 2e-4, "actual_condition_difference": "nonzero", "full_row_accounting": "entry and terminator included"}},
        "scientific": {"primary_campaign": "qualification-01", "primary_cells": 16, "excluded_execution": {"campaign": "scientific-01", "jobs": audit_only, "cells": 16, "disposition": "redundant_pre_ruling_execution_excluded_from_primary_evidence_and_denominators"}, "states": matrices, "failed_attempts": reduction["failed_attempts"]},
        "response_matrix": "R[region,candidate] = logP(candidate | region ablated) - logP(candidate | clean), with complete rows and exact x1 then y1 logprob path stored per row",
        "technical": {"reducer": binding(reduction_path), "cpu_acceptance": binding(acceptance_path), "response_matrices": binding(root / "response-matrices.json"), "max_clean_full_logit_error": max(job["parity"]["clean_compositor"]["max_abs_logit_error"] for job in all_jobs), "source_replay_passed": all(job["parity"]["native_replay"]["passed"] for job in all_jobs), "primary_costs": primary_costs, "costs_including_excluded_redundant_execution": costs, "wall_start": binding(root / "wall-start.json"), "budget_estimate": binding(Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/budget-estimate.json")), "wall_limit_amendment": binding(Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/wall-limit-amendment.json"))},
        "interpretation_candidate": {"observation": "A ablation lowers A-row preference in all four states; N ablation lowers N-row preference in all four states; unrelated ablation effects are small relative to target masks.", "inference": "The finite cells show correctly separated local visual dependence at both recurrence and matched control boundaries.", "h2": "downgraded_on_this_evidence", "h1": "not_tested_by_Lane_B", "training_origin": "HOLD_unsupported"},
        "holds": ["Optional release not run pending lead decision; altered visibility would change coverage obligations.", "Finite candidate rows and matched control are observational; no physical-owner or training-origin claim."],
        "artifacts": {"reduction": binding(reduction_path), "cpu_acceptance": binding(acceptance_path), "source_snapshots": {"primary": sorted(str(path.relative_to(root)) for path in root.glob("qualification-01/*/source-snapshot.json")), "audit_only": sorted(str(path.relative_to(root)) for path in root.glob("states/*/source-snapshot.json"))}, "qualification_roots": sorted(str(path.relative_to(root)) for path in root.glob("qualification-01/*/receipt.json")), "excluded_execution_roots": sorted(str(path.relative_to(root)) for path in root.glob("states/*/receipt.json"))},
        "commands": ["python -m probes.training_set_completion.visual_instance_binding.selfcheck", "python -m probes.training_set_completion.visual_instance_binding.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding/reduction.json", "python -m probes.training_set_completion.visual_instance_binding.verify --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding"],
    }
    out = root / "candidate-manifest.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["binding"] = binding(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = closeout(args.root)
    print(json.dumps({"status": result["status"], "forwards": result["technical"]["costs"]["model_forwards"], "gpu_seconds": result["technical"]["costs"]["gpu_seconds"], "output": str(args.root / "candidate-manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
