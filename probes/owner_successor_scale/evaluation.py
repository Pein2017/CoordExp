"""N16-anchor natural evaluation on the immutable owner-successor panel.

This thin entrypoint reuses the accepted native materializer, generator,
parser, scorer, overlap ledger, and cold consumer.  It owns only the N16
anchor identity, two-GPU slice gate, and remaining-254 no-rerun orchestration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from probes.native_owner_scale import evaluation as native


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-13-owner-successor-scale-throughput/evaluation"
CONFIRMATION = ROOT / "confirmation-selection.json"
OLD_EVALUATION_PACKET = BASE / "2026-09-12-native-owner-scale-and-state/evaluation/packet-v5.json"
N16_RECEIPT = BASE / "2026-09-12-native-owner-scale-and-state/scale/training/full-fixedP-N16-v2/provisional.json"
N16_ADAPTER_SHA256 = "092b47a56b2b50475e97e0a2f1fcbd058ba6f53f131cc7750af992ff4922db35"
N16_ADAPTER_FINGERPRINT = "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"
PANEL_SIZE = 256
SLICE_SIZE = 2
CAP = 3084
GPUS = (6, 7)
ARM = "N16-anchor"
_NATIVE_VALIDATE_BASELINE = native.validate_baseline_packet


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def binding(path: str | Path) -> dict[str, Any]:
    return native.binding(path)


def publish(path: str | Path, value: Any) -> None:
    native.publish(path, value)


def _script_binding() -> dict[str, Any]:
    return binding(Path(__file__).resolve())


def _configure_native() -> None:
    """Configure accepted helpers for this immutable N16 anchor route."""

    native.ROOT = ROOT
    native.BASELINE_GPUS = GPUS
    native.BASELINE_ARM = ARM
    native.BASELINE_SLICE_IDS = tuple(int(x) for x in read(CONFIRMATION)["image_ids"][:SLICE_SIZE])
    # v1 packet validation accepts a receipt-compatible producer hash.  The
    # packet is owned by this entrypoint, not by the earlier helper module.
    native.BASELINE_LAUNCH_PRODUCER_SHA256 = _script_binding()["sha256"]
    # The accepted helper's validator hard-codes its own module path.  Keep
    # its complete field/identity checks, substituting only the producer path
    # with this thin entrypoint's receipt.
    native.validate_baseline_packet = _native_validation_proxy


def _native_validation_proxy(packet: dict[str, Any], *, allow_legacy_producer: bool = False) -> None:
    value = copy.deepcopy(packet)
    value["producer"] = native.binding(Path(native.__file__).resolve())
    _NATIVE_VALIDATE_BASELINE(value, allow_legacy_producer=allow_legacy_producer)


def _n16_adapter() -> dict[str, Any]:
    receipt = read(N16_RECEIPT)
    adapter = copy.deepcopy(receipt["saved_adapter"])
    if adapter["root"].endswith("/adapter") is False:
        raise ValueError("N16 adapter root identity")
    if adapter["fingerprint"] != N16_ADAPTER_FINGERPRINT:
        raise ValueError("N16 adapter fingerprint changed")
    model_file = next(file for file in adapter["files"] if file["relative_path"] == "adapter_model.safetensors")
    if model_file["sha256"] != N16_ADAPTER_SHA256:
        raise ValueError("N16 adapter weights SHA256 changed")
    return adapter


def _build_evaluation_packet(output: Path) -> dict[str, Any]:
    if (output / "packet.json").exists():
        raise ValueError(f"evaluation packet already exists: {output / 'packet.json'}")
    selection = read(CONFIRMATION)
    if selection.get("status") != "frozen_cpu_no_model_calls":
        raise ValueError("confirmation selection is not immutable CPU-frozen")
    if len(selection["image_ids"]) != PANEL_SIZE or len(selection["blind_review_ids"]) != 32:
        raise ValueError("confirmation panel denominator changed")
    old = read(OLD_EVALUATION_PACKET)
    adapter = _n16_adapter()
    # Reuse the accepted native CPU materializer and its exact original prompt
    # config.  It loads processor/tokenizer metadata only (load_model=False).
    native.STABLE_PACKET = OLD_EVALUATION_PACKET
    records = native._native_records(selection)
    model = copy.deepcopy(old["model"])
    model["current_adapter"] = adapter
    config = copy.deepcopy(old["config"])
    config["adapter"] = {"name": "default", "path": adapter["root"], "type": "dora"}
    config["data"]["input_jsonl"] = str(native.NATIVE_SOURCE)
    packet = {
        # Keep the accepted native packet schema so its materializer/consumer
        # can be reused without introducing a second evaluator contract.
        "schema": "native_owner_scale_state.evaluation.packet.v1",
        "status": "cpu_prepared_n16_anchor_no_model_calls",
        "selection": binding(CONFIRMATION),
        "records": records,
        "model": model,
        "config": config,
        "native_source": binding(native.NATIVE_SOURCE),
        "generation": {
            "dtype": "fp32",
            "attention": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": CAP,
            "eos_token_id": native.EOS,
            "natural_prefix_ids": [],
            "forced_credit": False,
        },
        "n16_anchor": {
            "adapter": adapter,
            "adapter_weights_sha256": N16_ADAPTER_SHA256,
            "adapter_fingerprint": N16_ADAPTER_FINGERPRINT,
            "training_receipt": binding(N16_RECEIPT),
            "claim_boundary": "Frozen N16 anchor natural baseline only; no training/pretraining-disjointness claim.",
        },
        # The accepted baseline helper calls this endpoint slot ``stable50``;
        # here it is explicitly rebound to the frozen N16 anchor adapter.
        "stable50": {
            "adapter": adapter,
            "source_embedding": model["source_embedding"],
            "endpoint_role": "frozen N16 anchor natural baseline",
        },
        "consumer": {
            "module": "probes.owner_successor_scale.evaluation",
            "metric_path": "accepted native parser -> score -> overlap/repeat ledger",
            "claim_boundary": "Natural anchor rows only; GT-unmatched remains unknown and no visual review is performed here.",
        },
        "producer": _script_binding(),
        "provenance_boundary": "Consumes the immutable confirmation-selection.json; user labels do not alter raw benchmark identity.",
    }
    publish(output / "packet.json", packet)
    return packet


def prepare(*, output: str | Path = ROOT) -> dict[str, Any]:
    output = Path(output)
    packet = _build_evaluation_packet(output)
    _configure_native()
    # Accepted preparation writes the phase contract; patch its packet receipt
    # to this entrypoint and expose only the N16-anchor route.
    baseline_path = output / "baseline-packet.json"
    result = native.prepare_baseline(packet_path=output / "packet.json", output_path=baseline_path)
    baseline = read(baseline_path)
    # The accepted helper emits an intermediate packet under the same target;
    # retain it as provenance, then publish this entrypoint's bound contract.
    baseline_intermediate = output / "baseline-packet.accepted-helper.json"
    baseline_path.rename(baseline_intermediate)
    baseline["producer"] = _script_binding()
    baseline["arm"] = ARM
    baseline["model"] = packet["model"]
    baseline["config"] = packet["config"]
    baseline["generation"] = packet["generation"]
    baseline["adapter"] = packet["n16_anchor"]["adapter"]
    baseline["stored_adapter"] = packet["n16_anchor"]["adapter"]
    baseline["claim_boundary"] = "Frozen N16 anchor natural baseline only; no candidate/A/B evaluation or pretraining/SFT-disjointness claim."
    publish(baseline_path, baseline)
    request_path = output / "baseline-launch-request.json"
    request = read(request_path)
    request_intermediate = output / "baseline-launch-request.accepted-helper.json"
    request_path.rename(request_intermediate)
    request["packet"] = binding(baseline_path)
    request["arm"] = ARM
    request["physical_gpus"] = list(GPUS)
    request["slice"]["launch"] = f"CUDA_VISIBLE_DEVICES={GPUS[0]} python -m probes.owner_successor_scale.evaluation launch --packet {baseline_path} --phase slice --gpu {GPUS[0]}"
    request["remaining254"]["launch_after_slice_consumer"] = f"CUDA_VISIBLE_DEVICES={GPUS[0]},{GPUS[1]} python -m probes.owner_successor_scale.evaluation launch --packet {baseline_path} --phase full"
    publish(request_path, request)
    return {"packet": binding(output / "packet.json"), "baseline": binding(baseline_path), "slice_ids": baseline["phases"]["slice"]["image_ids"], "remaining": len(baseline["phases"]["full"]["image_ids"]), "model_calls": 0}


def _validate_packet(packet: dict[str, Any]) -> None:
    _configure_native()
    native.validate_baseline_packet(packet)
    evaluation = read(packet["input_packet"]["path"])
    if evaluation["status"] != "cpu_prepared_n16_anchor_no_model_calls":
        raise ValueError("wrong evaluation packet status")
    if evaluation["n16_anchor"]["adapter_fingerprint"] != N16_ADAPTER_FINGERPRINT:
        raise ValueError("wrong N16 adapter fingerprint")
    if packet["arm"] != ARM or packet["adapter"]["fingerprint"] != N16_ADAPTER_FINGERPRINT:
        raise ValueError("wrong N16 baseline identity")
    if packet["generation"]["max_new_tokens"] != CAP or packet["generation"]["dtype"] != "fp32" or packet["generation"]["attention"] != "sdpa":
        raise ValueError("generation settings changed")


def _worker(*, packet: Path, phase: str, shard: int, output: Path) -> None:
    _configure_native()
    value = read(packet)
    _validate_packet(value)
    native.baseline_worker(packet_path=packet, phase=phase, shard=shard, output=output)


def launch(*, packet_path: str | Path, phase: str) -> dict[str, Any]:
    _configure_native()
    packet_path = Path(packet_path)
    packet = read(packet_path)
    _validate_packet(packet)
    phase_packet = native._baseline_phase(packet, phase)
    if phase == "full":
        gate = Path(packet["phases"]["slice"]["consumer_root"]) / "result.json"
        if not gate.exists() or read(gate).get("status") != "cold_verified_baseline_only":
            raise ValueError("slice must cold-validate before full launch")
    output = Path(phase_packet["run_root"])
    if output.exists():
        raise ValueError(f"phase output occupied: {output}")
    output.mkdir(parents=True)
    processes = []
    logs = []
    for shard, gpu in enumerate(GPUS):
        cmd = [sys.executable, "-m", "probes.owner_successor_scale.evaluation", "worker", "--packet", str(packet_path), "--phase", phase, "--shard", str(shard), "--output", str(output / f"shard-{shard}")]
        log = (output / f"shard-{shard}.log").open("x")
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        processes.append(subprocess.Popen(cmd, cwd=native.WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT))
    publish(output / "launch.json", {"schema": "owner_successor_scale.evaluation.launch.v1", "status": "running", "packet": binding(packet_path), "phase": phase, "image_ids": phase_packet["image_ids"], "physical_gpus": list(GPUS), "commands": [p.args for p in processes], "no_retry": True})
    exits = []
    for shard, (process, log) in enumerate(zip(processes, logs, strict=True)):
        exits.append({"shard": shard, "exit_code": process.wait()})
        log.close()
    publish(output / "outer-exits.json", exits)
    if not all(x["exit_code"] == 0 for x in exits):
        raise RuntimeError(f"natural {phase} producer failed; partial artifacts preserved")
    return {"status": "completed", "phase": phase, "images": len(phase_packet["image_ids"]), "output": str(output)}


def merge(*, packet_path: str | Path, phase: str) -> dict[str, Any]:
    _configure_native()
    result = native.baseline_merge(packet_path=packet_path, phase=phase)
    return result


def consume(*, packet_path: str | Path, phase: str, rows_path: str | Path, output: str | Path) -> dict[str, Any]:
    _configure_native()
    result = native.consume_baseline(packet_path=packet_path, phase=phase, rows_path=rows_path, output=output)
    result_path = Path(output) / "result.json"
    value = read(result_path)
    value["arm"] = ARM
    value["claim_boundary"] = "Frozen N16 anchor natural baseline only; GT-unmatched remains unknown; no candidate/A/B or visual-review claim."
    value["next_step"] = "No remaining anchor phase; root may later bind a consumer, but this task stops after 256 cold-validated rows."
    result_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value


def finalize(*, packet_path: str | Path, output: str | Path = ROOT) -> dict[str, Any]:
    """Seal the combined 2+254 cold-validated anchor endpoint and costs."""

    _configure_native()
    packet_path = Path(packet_path)
    packet = read(packet_path)
    _validate_packet(packet)
    output = Path(output)
    result_path = output / "anchor-result.json"
    rows_path = output / "anchor-rows.jsonl"
    if result_path.exists() or rows_path.exists():
        raise ValueError("anchor final artifacts already exist")
    selection = read(CONFIRMATION)
    panel_ids = [int(x) for x in selection["image_ids"]]
    slice_ids = [int(x) for x in packet["phases"]["slice"]["image_ids"]]
    remaining_ids = [int(x) for x in packet["phases"]["full"]["image_ids"]]
    if slice_ids != panel_ids[:SLICE_SIZE] or set(slice_ids) & set(remaining_ids) or set(slice_ids + remaining_ids) != set(panel_ids):
        raise ValueError("anchor phase union changed")
    slice_result = read(Path(packet["phases"]["slice"]["consumer_root"]) / "result.json")
    full_result = read(Path(packet["phases"]["full"]["consumer_root"]) / "result.json")
    if slice_result.get("status") != "cold_verified_baseline_only" or full_result.get("status") != "cold_verified_baseline_only":
        raise ValueError("both anchor phases must be cold-verified")
    slice_rows = native.read_jsonl(Path(packet["phases"]["slice"]["run_root"]) / "rows.jsonl")
    full_rows = native.read_jsonl(Path(packet["phases"]["full"]["run_root"]) / "rows.jsonl")
    rows_by_id = {int(row["image_id"]): row for row in slice_rows + full_rows}
    if len(rows_by_id) != PANEL_SIZE or set(rows_by_id) != set(panel_ids):
        raise ValueError("anchor combined rows are not exact 256 identity")
    rows = [rows_by_id[image_id] for image_id in panel_ids]
    native._write_jsonl_exclusive(rows_path, rows)
    burden = native.accepted.burden(rows)
    quality = native.aggregate_scores([row["score"] for row in rows])
    from src.eval.detection_categories import COCO_80_CATEGORY_IDS, normalize_coco_category_name

    raw_descriptions = [prediction["description"] for row in rows for prediction in row["parsed"]["pred"]]
    out_of_coco80 = [
        description
        for description in raw_descriptions
        if normalize_coco_category_name(description) not in COCO_80_CATEGORY_IDS
    ]
    terminals = []
    for phase in ("slice", "full"):
        phase_root = Path(packet["phases"][phase]["run_root"])
        for shard in (0, 1):
            terminal = read(phase_root / f"shard-{shard}" / "terminal.json")
            terminals.append({
                "phase": phase,
                "shard": shard,
                "physical_gpu": terminal["physical_gpu"],
                "status": terminal["status"],
                "exit_code": terminal["exit_code"],
                "continuations": terminal["continuations"],
                "new_tokens": terminal["new_tokens"],
                "model_forwards": terminal["model_forwards"],
                "image_forwards": terminal["image_forwards"],
                "model_loads": terminal["model_loads"],
                "elapsed_seconds": terminal["elapsed_seconds"],
                "peak_cuda_allocated_bytes": terminal["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": terminal["peak_cuda_reserved_bytes"],
                "peak_rss_bytes": terminal["peak_rss_bytes"],
                "terminal": binding(phase_root / f"shard-{shard}" / "terminal.json"),
            })
    result = {
        "schema": "owner_successor_scale.evaluation.anchor_result.v1",
        "status": "cold_verified_n16_anchor_256",
        "arm": ARM,
        "packet": binding(packet_path),
        "selection": binding(CONFIRMATION),
        "rows": binding(rows_path),
        "panel": {"images": PANEL_SIZE, "image_ids_sha256": native.digest(panel_ids), "slice_ids": slice_ids, "remaining254_ids_sha256": native.digest(remaining_ids)},
        "denominators": {"fresh256": PANEL_SIZE, "slice": len(slice_ids), "remaining254": len(remaining_ids)},
        "quality": quality,
        "burden": burden,
        "raw_class_inventory": {
            "raw_prediction_count": len(raw_descriptions),
            "canonical_coco80_prediction_count": len(raw_descriptions) - len(out_of_coco80),
            "out_of_coco80_prediction_count": len(out_of_coco80),
            "out_of_coco80_descriptions": dict(sorted(Counter(out_of_coco80).items())),
            "policy": "raw descriptions retained in anchor-rows.jsonl; out-of-COCO80 values are separately reported and not silently relabeled or filtered",
        },
        "costs": {
            "model_loads": sum(x["model_loads"] for x in terminals),
            "model_forwards": sum(x["model_forwards"] for x in terminals),
            "image_forwards": sum(x["image_forwards"] for x in terminals),
            "new_tokens": sum(x["new_tokens"] for x in terminals),
            "elapsed_seconds_by_worker": [{"phase": x["phase"], "shard": x["shard"], "physical_gpu": x["physical_gpu"], "seconds": x["elapsed_seconds"]} for x in terminals],
            "elapsed_seconds_max_worker": max(x["elapsed_seconds"] for x in terminals),
            "peak_cuda_allocated_bytes_max": max(x["peak_cuda_allocated_bytes"] for x in terminals),
            "peak_cuda_reserved_bytes_max": max(x["peak_cuda_reserved_bytes"] for x in terminals),
            "peak_rss_bytes_max": max(x["peak_rss_bytes"] for x in terminals),
        },
        "terminals": terminals,
        "claim_boundary": "Frozen N16 anchor natural endpoint only; raw COCO benchmark unchanged; GT-unmatched and out-of-COCO80 descriptions remain separately reported/unknown; no candidate/A/B, visual review, training-disjointness or pretraining-disjointness claim.",
        "stop": "All 256 rows cold-validated; no slice rerun; root chooses any later consumer or paired candidate route.",
    }
    publish(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, default=ROOT)
    p = sub.add_parser("launch"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--phase", choices=("slice", "full"), required=True)
    p = sub.add_parser("worker"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--phase", choices=("slice", "full"), required=True); p.add_argument("--shard", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("merge"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--phase", choices=("slice", "full"), required=True)
    p = sub.add_parser("consume"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--phase", choices=("slice", "full"), required=True); p.add_argument("--rows", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("finalize"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare": print(json.dumps(prepare(output=args.output), sort_keys=True))
    elif args.command == "launch": print(json.dumps(launch(packet_path=args.packet, phase=args.phase), sort_keys=True))
    elif args.command == "worker": _worker(packet=args.packet, phase=args.phase, shard=args.shard, output=args.output)
    elif args.command == "merge": print(json.dumps(merge(packet_path=args.packet, phase=args.phase), sort_keys=True))
    elif args.command == "consume": print(json.dumps(consume(packet_path=args.packet, phase=args.phase, rows_path=args.rows, output=args.output), sort_keys=True))
    elif args.command == "finalize": print(json.dumps(finalize(packet_path=args.packet, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
