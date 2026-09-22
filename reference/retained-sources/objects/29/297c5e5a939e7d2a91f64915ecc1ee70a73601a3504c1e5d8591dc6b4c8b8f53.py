#!/usr/bin/env python3
"""Measure bounded incremental-backward replay capacity.

Run from the checkout:
``PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python .../capacity_probe.py``.
This is a no-update measurement of the frozen common CE/KL/margin workload.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import torch

from probes.dora_owner_learning import repeat_recovery_train as old
from probes.dora_owner_learning.geometric_dedup_train import install_language_decoder_checkpointing
from probes.parallel_owner_research import training
from probes.owner_successor_scale.replay import batched_aligned_logits
from src.adapters.dora import select_dora_parameters

PACKET = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/training/preparation/inputs-v2.json")
ADAPTER = "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/training/full-fixedP-N16-v2/adapter"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/throughput")
CONTRACT = Path(__file__).with_name("capacity-contract-v1.json")


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    packet = old.load_json(PACKET)
    anchor, normals, margins = training._load_dependencies(packet)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    qwen, frontend, config, raw = training._load_model(anchor, adapter_path=ADAPTER, device=device, evidence_dir=OUTPUT / "capacity-probe-model", packet=packet)
    model = qwen.model
    named = select_dora_parameters(model, towers=("language",), adapter_name="default")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in named:
        parameter.requires_grad_(True)
    checkpoint = install_language_decoder_checkpointing(model)

    positives = [training._materialize(record, qwen=qwen, frontend=frontend, config=config, raw=raw) for record in packet["positive_records"][:4]]
    conditional = [training._materialize(record, qwen=qwen, frontend=frontend, config=config, raw=raw) for record in packet["conditional_records"][:4]]
    normal = [training._materialize(training._normal_record(case), qwen=qwen, frontend=frontend, config=config, raw=raw) for case in normals[:8]]
    items = [("positive", entry) for entry in positives] + [("conditional", entry) for entry in conditional] + [("normal", entry) for entry in normal]
    # Real detached source distributions are cached once on CPU; they enter a
    # microbatch only while that microbatch graph is live.
    refs: dict[str, torch.Tensor] = {}
    for kind, entry in items:
        if kind != "positive":
            record = entry["record"]
            refs[record["record_id"]] = old._reference_logp(model, entry["inputs"], entry["prompt_ids"], record["target_token_ids"], record["kl_positions"])
    scales = {key: training.local_scale(packet, key, world_size=8) for key in training.COMPONENTS}
    ordered = sorted(items, key=lambda item: len(item[1]["prompt_ids"]) + len(item[1]["record"]["target_token_ids"]))
    target_tokens = sum(len(entry["record"]["target_token_ids"]) for _, entry in ordered)
    total_tokens = sum(len(entry["prompt_ids"]) + len(entry["record"]["target_token_ids"]) for _, entry in ordered)
    results = []
    for enabled in contract["checkpointing"]:
        for microbatch_size in contract["candidates"]:
            checkpoint["enabled"] = enabled
            checkpoint["phase"] = f"capacity_b{microbatch_size}_{'on' if enabled else 'off'}"
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.monotonic()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            components = {key: 0.0 for key in training.COMPONENTS}
            forwards = 0
            error = None
            try:
                for offset in range(0, len(ordered), microbatch_size):
                    group = ordered[offset : offset + microbatch_size]
                    logits_rows = batched_aligned_logits(model, [entry for _, entry in group])
                    total = None
                    for (kind, entry), logits in zip(group, logits_rows, strict=True):
                        record = entry["record"]
                        target = torch.tensor(record["target_token_ids"], device=device)
                        reference = refs.get(record["record_id"])
                        loss, stats = training.item_loss(logits, target, kind=kind, scales=scales, positions=record.get("kl_positions", []), reference_logp=None if reference is None else reference.to(device), margin=margins.get(record["record_id"]))
                        for key in components:
                            components[key] += float(stats.get(key, 0.0))
                        total = loss if total is None else total + loss
                    assert total is not None and bool(torch.isfinite(total))
                    total.backward()
                    forwards += 1
                    del logits_rows, total, loss
                    model.zero_grad(set_to_none=True)
                end.record(); torch.cuda.synchronize(device)
                wall, device_ms = time.monotonic() - start, begin.elapsed_time(end)
                peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
                status = "passed" if peak_allocated <= contract["limits"]["allocated_bytes_per_card"] else "over_allocated"
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                status, error, wall, device_ms = "oom", str(exc), time.monotonic() - start, None
                peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
            results.append({"microbatch_size": microbatch_size, "activation_checkpointing": enabled, "status": status, "error": error, "wall_seconds": wall, "device_milliseconds": device_ms, "target_tokens": target_tokens, "total_tokens": total_tokens, "forwards": forwards, "target_tokens_per_second": target_tokens / wall, "total_tokens_per_second": total_tokens / wall, "components": components, "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved, "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    passing = [row for row in results if row["status"] == "passed"]
    selected = max(passing, key=lambda row: (row["target_tokens_per_second"], -row["microbatch_size"])) if passing else None
    result = {"schema": "owner_successor_scale.capacity_probe.v1", "status": "passed" if selected else "failed", "contract": str(CONTRACT.resolve()), "packet": str(PACKET), "adapter": ADAPTER, "no_optimizer_steps": True, "records": [{"kind": kind, "record_id": entry["record"]["record_id"], "target_tokens": len(entry["record"]["target_token_ids"])} for kind, entry in ordered], "results": results, "selected": selected, "receipt_bytes": 0}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result["receipt_bytes"] = len(encoded.encode())
    (OUTPUT / "capacity-probe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
