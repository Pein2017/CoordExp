#!/usr/bin/env python3
"""Replay the final, bounded batch-numerics diagnostic on GPU 0.

Run from the research-probes checkout:

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/throughput/final_diagnostic.py

It consumes the immutable N16 materialization packet and writes runtime data
only below the configured output root.  It does not train a scientific arm,
save an adapter, or select capacity.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from probes.dora_owner_learning import repeat_recovery_train as old
from probes.dora_owner_learning import margin_preserved_train as margin_engine
from probes.parallel_owner_research import training
from probes.owner_successor_scale.replay import batched_aligned_logits
from src.adapters.dora import select_dora_parameters
from src.qwen.native import prepare_replay

PACKET = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/training/preparation/inputs-v2.json")
ADAPTER = "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/training/full-fixedP-N16-v2/adapter"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/throughput")
CONTRACT = Path(__file__).with_name("final-technical-contract-v3.json")


def _metric(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    dot = left_sq = right_sq = diff_sq = maximum = 0.0
    for a, b in zip(left, right, strict=True):
        a64, b64 = a.detach().double(), b.detach().double()
        diff = a64 - b64
        dot += float((a64 * b64).sum())
        left_sq += float(a64.square().sum())
        right_sq += float(b64.square().sum())
        diff_sq += float(diff.square().sum())
        maximum = max(maximum, float(diff.abs().max()))
    return {
        "max_abs": maximum,
        "relative_l2": math.sqrt(diff_sq) / max(math.sqrt(left_sq), 1e-30),
        "cosine": dot / max(math.sqrt(left_sq * right_sq), 1e-30),
    }


def main() -> None:
    packet = old.load_json(PACKET)
    anchor, normals, margins = training._load_dependencies(packet)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    qwen, frontend, config, raw = training._load_model(
        anchor, adapter_path=ADAPTER, device=device,
        evidence_dir=OUTPUT / "final-diagnostic-model", packet=packet,
    )
    model = qwen.model
    named = select_dora_parameters(model, towers=("language",), adapter_name="default")
    assert len(named) == old.EXPECTED_TRAINABLE_TENSORS
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in named:
        parameter.requires_grad_(True)

    positive = training._materialize(packet["positive_records"][0], qwen=qwen, frontend=frontend, config=config, raw=raw)
    conditional = training._materialize(packet["conditional_records"][0], qwen=qwen, frontend=frontend, config=config, raw=raw)
    normal_entries = [training._materialize(training._normal_record(case), qwen=qwen, frontend=frontend, config=config, raw=raw) for case in normals]
    scales = {key: training.local_scale(packet, key, world_size=8) for key in training.COMPONENTS}
    source_theta = [parameter.detach().clone() for _, parameter in named]

    def restore(theta: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for (_, parameter), source in zip(named, theta, strict=True):
                parameter.copy_(source)
        model.zero_grad(set_to_none=True)

    def serial(entries: list[dict[str, object]]) -> list[torch.Tensor]:
        result = []
        for entry in entries:
            record = entry["record"]
            exact = prepare_replay(model, entry["inputs"], prompt_token_ids=entry["prompt_ids"], continuation_token_ids=record["target_token_ids"])
            result.append(exact.aligned_logits(model(**exact.inputs).logits).float())
        return result

    def source_reference(entry: dict[str, object]) -> torch.Tensor:
        record = entry["record"]
        return old._reference_logp(model, entry["inputs"], entry["prompt_ids"], record["target_token_ids"], record["kl_positions"]).to(device)

    normal = normal_entries[0]
    conditional_ref, normal_ref = source_reference(conditional), source_reference(normal)

    def loss_step(entries: list[tuple[str, dict[str, object]]], refs: dict[str, torch.Tensor], *, batched: bool, optimizer: torch.optim.Optimizer) -> dict[str, object]:
        logits = batched_aligned_logits(model, [entry for _, entry in entries]) if batched else serial([entry for _, entry in entries])
        components = {key: 0.0 for key in training.COMPONENTS}
        total = None
        for (kind, entry), value in zip(entries, logits, strict=True):
            record = entry["record"]
            target = torch.tensor(record["target_token_ids"], device=device)
            loss, stats = training.item_loss(value, target, kind=kind, scales=scales, positions=record.get("kl_positions", []), reference_logp=refs.get(record["record_id"]), margin=margins.get(record["record_id"]))
            for key in components:
                components[key] += float(stats.get(key, 0.0))
            total = loss if total is None else total + loss
        assert total is not None
        total.backward()
        gradients = [parameter.grad.detach().clone() for _, parameter in named]
        nonzero = [bool(torch.count_nonzero(gradient)) for gradient in gradients]
        raw_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], packet["clip_gradient_norm"], error_if_nonfinite=True, foreach=False))
        clip_factor = min(1.0, packet["clip_gradient_norm"] / raw_norm)
        before = [parameter.detach().clone() for _, parameter in named]
        optimizer.step()
        deltas = [parameter.detach() - prior for (_, parameter), prior in zip(named, before, strict=True)]
        return {"logits": logits, "components": components, "gradients": gradients, "deltas": deltas, "nonzero": nonzero, "raw_gradient_norm": raw_norm, "clip_factor": clip_factor}

    # Logical update 1: the common warm start.  Its state is then cloned for
    # the serial and batched realizations of logical update 2.
    warm_optimizer = torch.optim.AdamW([parameter for _, parameter in named], **packet["optimizer"])
    warm = loss_step([("positive", positive), ("conditional", conditional), ("normal", normal)], {conditional["record"]["record_id"]: conditional_ref, normal["record"]["record_id"]: normal_ref}, batched=False, optimizer=warm_optimizer)
    warm_theta = [parameter.detach().clone() for _, parameter in named]
    warm_optimizer_state = copy.deepcopy(warm_optimizer.state_dict())

    # Inspect the already registered normal bank only; no label, target, or
    # floor is edited.  This is after warm-start because KL is now nonzero.
    active = None
    with torch.no_grad():
        for candidate in normal_entries:
            logits = serial([candidate])[0]
            record = candidate["record"]
            penalty, detail = margin_engine.worst_margin_penalty(logits, torch.tensor(record["target_token_ids"], device=device), margins[record["record_id"]])
            if float(penalty) > 0:
                active = (candidate, float(penalty), detail)
                break
    if active is not None and active[0]["record"]["record_id"] != normal["record"]["record_id"]:
        restore(source_theta)
        normal = active[0]
        normal_ref = source_reference(normal)
    restore(warm_theta)

    entries = [("positive", positive), ("conditional", conditional), ("normal", normal)]
    refs = {conditional["record"]["record_id"]: conditional_ref, normal["record"]["record_id"]: normal_ref}
    def branch(name: str, *, batched: bool) -> dict[str, object]:
        restore(warm_theta)
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], **packet["optimizer"])
        optimizer.load_state_dict(copy.deepcopy(warm_optimizer_state))
        result = loss_step(entries, refs, batched=batched, optimizer=optimizer)
        return {"name": name, "batched": batched, **result}

    serial_branch, batch_branch = branch("serial", batched=False), branch("batched", batched=True)
    bounds = json.loads(CONTRACT.read_text())["bounds"]
    comparison = {
        "logits": _metric(serial_branch["logits"], batch_branch["logits"]),
        "component_abs": {key: abs(serial_branch["components"][key] - batch_branch["components"][key]) for key in training.COMPONENTS},
        "gradient": _metric(serial_branch["gradients"], batch_branch["gradients"]),
        "optimizer_delta": _metric(serial_branch["deltas"], batch_branch["deltas"]),
        "same_nonzero_selected_tensors": serial_branch["nonzero"] == batch_branch["nonzero"],
        "clip_factor_abs_difference": abs(serial_branch["clip_factor"] - batch_branch["clip_factor"]),
    }
    passed = comparison["logits"]["max_abs"] <= bounds["max_logit_abs"] and all(value <= bounds["each_component_loss_abs"] for value in comparison["component_abs"].values()) and comparison["gradient"]["relative_l2"] <= bounds["gradient_global_relative_l2"] and comparison["gradient"]["cosine"] >= bounds["gradient_cosine_minimum"] and comparison["optimizer_delta"]["relative_l2"] <= bounds["optimizer_delta_relative_l2"] and comparison["optimizer_delta"]["cosine"] >= bounds["optimizer_delta_cosine_minimum"] and comparison["same_nonzero_selected_tensors"] and comparison["clip_factor_abs_difference"] <= bounds["clip_factor_abs_difference"]
    result = {
        "schema": "owner_successor_scale.final_warm_start_diagnostic.v1",
        "status": "passed" if passed else "failed",
        "contract": str(CONTRACT.resolve()), "packet": str(PACKET), "adapter": ADAPTER,
        "logical_optimizer_updates": 2,
        "records": [{"kind": kind, "record_id": entry["record"]["record_id"], "target_tokens": len(entry["record"]["target_token_ids"]), "mask_count": len(entry["record"].get("kl_positions", []))} for kind, entry in entries],
        "warm_start_components": warm["components"],
        "active_margin": None if active is None else {"record_id": active[0]["record"]["record_id"], "penalty": active[1], "detail": active[2]},
        "comparison": comparison, "bounds": bounds,
        "branch_components": {branch["name"]: branch["components"] for branch in (serial_branch, batch_branch)},
        "branch_raw_gradient_norm": {branch["name"]: branch["raw_gradient_norm"] for branch in (serial_branch, batch_branch)},
        "resources": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
    }
    (OUTPUT / "final-warm-start-diagnostic.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
