"""Apply the frozen serial-versus-distributed complete-update acceptance gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, "/data/CoordExp/.worktrees/row-feedback-pilot-20260913")
from src.adapters.dora import inspect_dora_adapter_payload


LIMITS = {
    "component_and_loss_atol": 1e-6,
    "component_and_loss_rtol": 1e-5,
    "gradient_norm_relative_error_max": 1e-4,
    "adapter_diff_l2_over_serial_movement_max": 1e-3,
    "adapter_max_abs_difference": 2e-6,
}


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def read_bound(reference):
    actual = binding(reference["path"])
    assert all(actual[k] == reference[k] for k in reference), "bound file drift"
    return json.loads(Path(reference["path"]).read_text())


def compare_tensors(reference, candidate):
    assert set(reference) == set(candidate), "adapter tensor identity"
    squared, maximum, scalars = 0.0, 0.0, 0
    for key in sorted(reference):
        a, b = reference[key], candidate[key]
        assert a.shape == b.shape and a.dtype == b.dtype, key
        assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), key
        difference = b.double() - a.double()
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        scalars += a.numel()
    return {"difference_l2": math.sqrt(squared), "maximum_absolute_difference": maximum,
            "tensor_count": len(reference), "scalar_count": scalars}


def evaluate(packet_path, packet_sha256, candidate_path):
    packet_binding = binding(packet_path)
    assert packet_binding["sha256"] == packet_sha256, "root packet hash"
    packet = read_bound(packet_binding)
    gate = packet["technical_equivalence_gate"]
    assert {k: gate[k] for k in LIMITS} == LIMITS, "frozen tolerances changed"
    assert gate["status"] == "root_frozen_before_distributed_results"
    candidate_binding = binding(candidate_path)
    candidate = read_bound(candidate_binding)
    assert candidate["mode"] == "cost" and candidate["dose"] == 1
    assert candidate["status"] == "cost_only_disposable_distributed_complete_update_saved"
    assert candidate["packet"] == {k: packet_binding[k] for k in ("path", "sha256")}
    arm = candidate["arm"]
    serial_binding = gate["serial_reference_receipts"][arm]
    serial = read_bound(serial_binding)
    assert serial["arm"] == arm and serial["mode"] == "cost" and serial["dose"] == 1
    for key in ("anchor_adapter", "bank", "protection", "teacher_cache", "schedule_source",
                "seed", "optimizer", "loss", "schedule", "trainable_surface"):
        assert candidate[key] == serial[key], key
    assert candidate["code_bindings"] == packet["code_bindings"]
    assert candidate["topology"] == packet["execution"]["distributed"]
    for key in ("bank", "normal"):
        assert candidate["materialization"][key] == serial["materialization"][key], key
    assert candidate["all_trainable_tensors_finite"] is True
    assert candidate["frozen_parameter_versions_unchanged"] is True
    ranks = candidate["rank_receipts"]
    world = candidate["topology"]["world_size"]
    assert sorted(row["rank"] for row in ranks) == list(range(world))
    for row in ranks:
        rank = read_bound({k: row[k] for k in ("path", "sha256")})
        assert rank["status"] == "complete" and rank["world_size"] == world
        assert rank["trainable_state_sha256"] == candidate["trainable_state_sha256"]
        assert rank["all_trainable_tensors_finite"] is True
        assert rank["frozen_parameter_versions_unchanged"] is True
    assert len(serial["updates"]) == len(candidate["updates"]) == 1
    a, b = serial["updates"][0], candidate["updates"][0]
    for key in ("counts", "replays", "model_forwards", "image_forwards", "internal_slots",
                "visible_target_tokens", "optimizer_steps"):
        assert a[key] == b[key], key
    assert b["global_invariants"]["global_record_counts"] == {
        "entry_c": 2, "post_completion_w": 2, "normal_kl": 54}
    assert b["counts"]["normal_protected_tokens"] == 5759
    values = {"loss": (a["loss"], b["loss"])}
    values.update({k: (a["component_record_means"][k], b["component_record_means"][k])
                   for k in a["component_record_means"]})
    checks = {f"numeric_{k}": abs(x-y) <= LIMITS["component_and_loss_atol"]
              + LIMITS["component_and_loss_rtol"] * abs(x) for k, (x, y) in values.items()}
    norm_error = abs(b["gradient_norm_before_clip"] - a["gradient_norm_before_clip"]) / abs(a["gradient_norm_before_clip"])
    checks["gradient_norm"] = norm_error <= LIMITS["gradient_norm_relative_error_max"]
    tensors = []
    for receipt in (serial, candidate):
        saved = receipt["saved_adapter"]
        assert inspect_dora_adapter_payload(saved["root"]) == saved, "saved adapter descriptor drift"
        tensors.append(load_file(str(Path(saved["root"]) / "adapter_model.safetensors"), device="cpu"))
    delta = compare_tensors(*tensors)
    assert delta["tensor_count"] == 588 and delta["scalar_count"] == 18006016
    delta["l2_over_serial_update_movement"] = delta["difference_l2"] / a["adapter_movement_l2"]
    checks["adapter_l2"] = delta["l2_over_serial_update_movement"] <= LIMITS["adapter_diff_l2_over_serial_movement_max"]
    checks["adapter_maximum"] = delta["maximum_absolute_difference"] <= LIMITS["adapter_max_abs_difference"]
    return {"schema": "row_feedback.distributed_equivalence_gate.v1", "arm": arm,
            "status": "gate_passed_pending_root_acceptance" if all(checks.values()) else "gate_failed",
            "packet": packet_binding, "serial_receipt": serial_binding,
            "candidate_receipt": candidate_binding, "world_size": world,
            "limits": LIMITS, "checks": checks, "loss_values_serial_candidate": values,
            "gradient_norm_relative_error": norm_error, "adapter_difference": delta,
            "boundary": "One disposable complete-update numerical gate only; no scientific fit result."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.packet, args.packet_sha256, args.candidate)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "adapter_difference": result["adapter_difference"], "output": args.output}))
    raise SystemExit(0 if all(result["checks"].values()) else 1)


if __name__ == "__main__":
    main()
