"""Build the stable candidate manifest for the Lane A recovery package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = OUTPUT / "2026-09-21-spatial-progress-recovery"
PREDECESSOR = OUTPUT / "2026-09-21-spatial-progress-gate"
ADMISSION = PREDECESSOR / "selection/shared-admission.json"
BUDGET = PREDECESSOR / "selection/budget-estimate.json"
WALL_LIMIT = PREDECESSOR / "selection/wall-limit-amendment.json"
DISPATCH = ROOT / "dispatch.json"
REDUCTION = ROOT / "reduction-candidate-v2.json"
CPU_ACCEPTANCE = ROOT / "diagnostics/cpu-acceptance.json"
RED_DIAGNOSTIC = ROOT / "diagnostics/caller-boundary-red.json"
CODE_ROOT = Path("/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/spatial_progress_gate")
RUNTIME = CODE_ROOT / "runtime.py"
REDUCER = CODE_ROOT / "reduce.py"
SELFCHECK = CODE_ROOT / "selfcheck.py"
SOURCE_PROVENANCE = Path("/data/CoordExp/.worktrees/research-probes/src/artifacts/source_provenance.py")
ORCHESTRATION_NOTES = Path("/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-21-spatial-progress-recovery/orchestration-notes.md")
OUT = ROOT / "candidate-manifest-recovery-v3.json"


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _receipt_entries(reduction: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    qualification = reduction["qualification"]
    entries.append({"kind": "qualification", "campaign": qualification["campaign"], "result": qualification["result"], "receipt": qualification["receipt"]})
    for cell in sorted(reduction["cells"], key=lambda value: value["id"]):
        entries.append({"kind": "primary_cell", "cell_id": cell["id"], "condition": cell["condition"], "scores": cell["scores"], "receipt": cell["receipt"]})
    return entries


def build() -> dict[str, Any]:
    admission = load(ADMISSION)
    reduction = load(REDUCTION)
    cpu = load(CPU_ACCEPTANCE)
    red = load(RED_DIAGNOSTIC)
    dispatch = load(DISPATCH)
    orchestration = bind(ORCHESTRATION_NOTES)
    if bind(ADMISSION)["sha256"] != "8863a4eb6d00ed9cb27cf8dd41e32429e7081e3bf363907b23bf4a7a0ba1d7d8":
        raise ValueError("frozen admission binding changed")
    if reduction.get("status") != "candidate" or len(reduction.get("cells", [])) != 8 or reduction.get("missing_cells"):
        raise ValueError("recovery reduction is not complete")
    if any(effect.get("fixed_crossed_N_effect") is None for effect in reduction["effects"]):
        raise ValueError("fixed crossing effects are incomplete")
    if cpu.get("status") != "passed" or red.get("status") != "reproduced":
        raise ValueError("CPU acceptance diagnostics are incomplete")
    receipts = _receipt_entries(reduction)
    for entry in receipts:
        receipt = load(Path(entry["receipt"]["path"]))
        if receipt.get("status") != "candidate_complete":
            raise ValueError(f"nonterminal attempt: {entry}")
    continuation = reduction["costs"]
    historical = {"model_forwards": 167, "vision_forwards": 167, "gpu_seconds": 533.4304532557726}
    cumulative = {
        "model_forwards": historical["model_forwards"] + continuation["model_forwards"],
        "vision_forwards": historical["vision_forwards"] + continuation["vision_forwards"],
        "gpu_seconds": historical["gpu_seconds"] + continuation["gpu_seconds"],
    }
    return {
        "schema": "spatial_progress_gate.recovery_candidate.v3",
        "status": "candidate",
        "acceptance": "pending_lead_review_no_self_acceptance",
        "execution_owner": {"model": dispatch["required_child_model"], "effort": dispatch["required_child_effort"], "status": "terminal", "descendants": []},
        "orchestration": {"notes": orchestration, "child": {"model": dispatch["required_child_model"], "effort": dispatch["required_child_effort"], "status": "terminal", "descendants": []}},
        "shared_admission": {"binding": bind(ADMISSION), "counts": admission["lane_a"]["counts"], "boundaries": [item["id"] for item in admission["lane_a"]["boundaries"]]},
        "qualification": {"campaign": reduction["qualification"]["campaign"], "result": reduction["qualification"]["result"], "receipt": reduction["qualification"]["receipt"]},
        "primary_cells": {"count": len(reduction["cells"]), "entries": receipts[1:], "alias": {"condition_id": "x1_after_N786", "alias_of": "native", "executed_once": True}},
        "effects": reduction["effects"],
        "reduction": bind(REDUCTION),
        "diagnostics": {"caller_boundary_red": bind(RED_DIAGNOSTIC), "cpu_acceptance": bind(CPU_ACCEPTANCE)},
        "attempts": {"qualification_HOLDs": reduction["qualification_HOLDs"], "failed_attempts": reduction["failed_attempts"], "terminal_attempts": receipts},
        "costs": {"historical": historical, "continuation": continuation, "cumulative": cumulative, "limits": {"allocated_gpu_hours": 8, "model_forwards": 100000, "retained_bytes": 17179869184}},
        "wall_limit": {"dispatch": bind(DISPATCH), "first_model_launch": reduction["costs"]["wall_elapsed_basis"]["start"], "elapsed_seconds": continuation["wall_elapsed_seconds"], "deadline_utc": dispatch["wall_deadline_utc"], "limit_seconds": dispatch["wall_limit_seconds"]},
        "source_runtime_bindings": {"admission": bind(ADMISSION), "budget_estimate": bind(BUDGET), "wall_limit_amendment": bind(WALL_LIMIT), "dispatch": bind(DISPATCH), "producer": bind(Path(__file__)), "runtime": bind(RUNTIME), "reducer": bind(REDUCER), "selfcheck": bind(SELFCHECK), "source_provenance": bind(SOURCE_PROVENANCE)},
        "job_closure": {"active_owned_jobs": 0, "terminal_attempts": len(receipts), "cell_processes": "all eight terminal; launcher exit 0", "child_status": "terminal", "child_descendants": [], "optional_release": "not_run_not_authorized_in_recovery"},
        "changed_paths": [
            "probes/training_set_completion/spatial_progress_gate/runtime.py",
            "probes/training_set_completion/spatial_progress_gate/selfcheck.py",
            "probes/training_set_completion/spatial_progress_gate/reduce.py",
            "probes/training_set_completion/spatial_progress_gate/integrate_candidate.py",
            "research/experiments/2026-09-21-spatial-progress-recovery/orchestration-notes.md",
        ],
        "acceptance_commands": [
            "python -m probes.training_set_completion.spatial_progress_gate.selfcheck",
            "python -m probes.training_set_completion.spatial_progress_gate.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-recovery --out <fresh-output>",
            "python -m probes.training_set_completion.spatial_progress_gate.integrate_candidate --check",
        ],
        "claim_boundary": "candidate measurements only; lead owns H1 interpretation and acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = (json.dumps(build(), indent=2, allow_nan=False) + "\n").encode()
    if args.check:
        if OUT.read_bytes() != data:
            raise AssertionError("candidate manifest is stale")
        print(json.dumps({"status": "passed", "candidate": bind(OUT)}, indent=2))
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "candidate", "candidate": bind(OUT)}, indent=2))


if __name__ == "__main__":
    main()
