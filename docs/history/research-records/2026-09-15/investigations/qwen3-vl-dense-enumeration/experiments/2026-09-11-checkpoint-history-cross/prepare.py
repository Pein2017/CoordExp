"""Build the frozen CPU packet for the checkpoint/history cross."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
PRIOR = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-branch-vs-repeat-event")
DEFAULT_ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross")
ENDPOINT_PACKET = PRIOR / "endpoint-preparation" / "packet.json"
ENDPOINT_CONSUMER = PRIOR / "endpoint-A" / "consumer.json"
POSITIVE_RECEIPT = PRIOR / "full-A" / "receipt.json"
ENDPOINT_PRODUCER = PRIOR / "endpoint_eval.py"
WITNESS_RUNNER = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-native-escape-witness/run_probe.py")
WITNESS_REDUCER = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-native-escape-witness/reduce.py")

EXPECTED = {
    ENDPOINT_PACKET: "560006e73f3f0fc416e7d58751fe96c936aba9478fb6b320ab6118db2bcd5053",
    ENDPOINT_CONSUMER: "d61454f058793fd40338f19d1ebeac9023a3889df58912c3b960224dd8b91a37",
    POSITIVE_RECEIPT: "dd14419bf11a07aa36c783b26207addf0e4294242d13e4d6219d10674dfeaa40",
    ENDPOINT_PRODUCER: "abc4391a37d6edaa8e50d9a728bf86f602fbabad82f8cfe8d5310c86b365ec39",
}
OFFSETS = {
    39654: {"first_difference": 15, "h_end": 9, "stable50_end": 18, "positive32_end": 18},
    351017: {"first_difference": 10, "h_end": 9, "stable50_end": 19, "positive32_end": 20},
    417044: {"first_difference": 15, "h_end": 9, "stable50_end": 19, "positive32_end": 19},
    477415: {"first_difference": 1, "h_end": 0, "stable50_end": 9, "positive32_end": 9},
}
CHECKPOINTS = ("stable50", "positive32")
EOS, PAD, CAP = 151645, 151643, 3084
ROW_START, REF_END, BOX_START, ROW_END = 151646, 151647, 151648, 151649
STABLE_FINGERPRINT = "024e46a512491b15d8715218c9fe7970707e7449f8b354b40ae37122c2c7ac9b"
STABLE_RUNTIME_FINGERPRINT = "b3cf3ce468f9d40c7f6a629cf1a6265b8ba218e31d1078ef32af13ec15e63434"
POSITIVE_FINGERPRINT = "8c7d9e841be36fcdadb546d3024a32938124da2b1e12c3dd3ea80769a194dc68"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))
    require(load(path) == value, f"publication readback differs: {path}")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_difference(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def split_complete_rows(ids: list[int], *, allow_empty: bool = False) -> list[list[int]]:
    if not ids:
        require(allow_empty, "empty row history")
        return []
    rows: list[list[int]] = []
    cursor = 0
    while cursor < len(ids):
        require(ids[cursor] == ROW_START, f"non-row token at {cursor}")
        try:
            end = ids.index(ROW_END, cursor) + 1
        except ValueError as exc:
            raise ValueError(f"incomplete row at {cursor}") from exc
        row = ids[cursor:end]
        require(REF_END in row and BOX_START in row and row.index(REF_END) < row.index(BOX_START), "row wrappers differ")
        coords = row[row.index(BOX_START) + 1:-1]
        require(len(coords) == 4 and all(151670 <= value <= 152669 for value in coords), "coordinate token structure differs")
        require(EOS not in row and PAD not in row, "row contains terminal/pad")
        rows.append(row)
        cursor = end
    return rows


def validate_action(ids: list[int], stop: str) -> None:
    require(ids and len(ids) <= CAP and PAD not in ids and EOS not in ids[:-1], "cached action token corruption")
    require((stop == "im_end" and ids[-1] == EOS) or
            (stop == "length" and len(ids) == CAP and EOS not in ids), "cached stop/action mismatch")


def adapter_payload(payload: Mapping[str, Any], *, fingerprint: str, base: str,
                    runtime_fingerprint: str | None = None) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    require(payload.get("fingerprint") == fingerprint and payload.get("kind") == "dora_adapter", "adapter identity differs")
    root = Path(str(payload.get("root", "")))
    require(root.is_absolute(), "adapter root must be absolute")
    for item in payload.get("files", []):
        path = root / item["relative_path"]
        require(path.is_file() and path.stat().st_size == item["size_bytes"] and file_hash(path) == item["sha256"], "adapter file differs")
    observed = inspect_dora_adapter_payload(root, base)
    if runtime_fingerprint is None:
        require(observed == dict(payload), "adapter cold inspection differs")
    else:
        require(observed["fingerprint"] == runtime_fingerprint, "runtime adapter fingerprint differs")
        require(observed["root"] == payload["root"] and observed["files"] == payload["files"] and
                observed["semantic_identity"] == payload["semantic_identity"] and
                observed["tensor_manifest"] == payload["tensor_manifest"] and
                observed["kind"] == payload["kind"], "adapter payload/semantics differ across metadata versions")
        require(payload["version"] == "coordexp-swift-dora-adapter-v1" and
                observed["version"] == "coordexp-infras-dora-adapter-v1", "unexpected adapter metadata-version drift")
    return observed


def callable_frontend_preflight(config_payload: Mapping[str, Any], checkpoints: Mapping[str, Any]) -> dict[str, Any]:
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    config = InferConfig.model_validate(config_payload)
    require(config.backend.type == "hf" and config.model.dtype == "fp32", "requires native HF FP32")
    require(config.backend.hf.attn_implementation == "sdpa" and
            config.backend.hf.patch_embed_linearization == "enabled", "requires SDPA patch linearization")
    effective = {}
    for name, checkpoint in checkpoints.items():
        value = checkpoint_config(config, checkpoint["adapter"]["root"])
        require(str(value.adapter.path) == checkpoint["adapter"]["root"], f"{name}: explicit adapter override failed")
        effective[name] = {"adapter_path": str(value.adapter.path), "backend": value.backend.type,
                           "dtype": value.model.dtype, "attention": value.backend.hf.attn_implementation}
    policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0,
                                    repetition_penalty=1.0, use_model_defaults=False)
    required = {
        "load_policy": load_policy,
        "build_requests": build_requests,
        "native_record": native_record,
        "prepare_native_inputs": prepare_native_inputs,
        "generate_continuations": generate_continuations,
    }
    return {
        "effective_checkpoints": effective,
        "native_policy": {"temperature": policy.temperature, "top_p": policy.top_p,
                          "top_k": policy.top_k, "repetition_penalty": policy.repetition_penalty,
                          "use_model_defaults": policy.use_model_defaults},
        "callables": {name: {"module": value.__module__, "signature": str(inspect.signature(value))}
                      for name, value in required.items()},
    }


def build_packet() -> dict[str, Any]:
    for path, expected in EXPECTED.items():
        require(path.is_file() and file_hash(path) == expected, f"frozen source differs: {path}")
    endpoint = load_module("history_cross_endpoint", ENDPOINT_PRODUCER)
    prior_packet = load(ENDPOINT_PACKET)
    endpoint.validate_packet(prior_packet)
    endpoint.merge(ENDPOINT_PACKET, POSITIVE_RECEIPT, PRIOR / "endpoint-A", arm="A", verify=True)
    positive_rows = load(ENDPOINT_CONSUMER)
    require(len(positive_rows) == 384, "endpoint consumer population differs")
    positive_by_id = {int(row["image_id"]): row for row in positive_rows}
    stable_by_id = {int(row["image_id"]): row for row in prior_packet["eval_records"]}

    receipt = load(POSITIVE_RECEIPT)
    base = prior_packet["model"]["base_model_path"]
    stable_source_adapter = dict(prior_packet["model"]["current_adapter"])
    stable_adapter = adapter_payload(stable_source_adapter, fingerprint=STABLE_FINGERPRINT, base=base,
                                     runtime_fingerprint=STABLE_RUNTIME_FINGERPRINT)
    positive_adapter = endpoint.validate_adapter_receipt(receipt, arm="A", packet=prior_packet)
    require(positive_adapter["fingerprint"] == POSITIVE_FINGERPRINT, "positive32 fingerprint differs")
    checkpoints = {
        "stable50": {"adapter": stable_adapter, "source_adapter_identity": stable_source_adapter,
                     "metadata_identity_transition": {"payload_equal": True,
                         "source_version": stable_source_adapter["version"], "source_fingerprint": stable_source_adapter["fingerprint"],
                         "runtime_version": stable_adapter["version"], "runtime_fingerprint": stable_adapter["fingerprint"]},
                     "source": {"kind": "endpoint_packet_current_adapter",
                       "path": str(ENDPOINT_PACKET), "sha256": EXPECTED[ENDPOINT_PACKET]}},
        "positive32": {"adapter": positive_adapter, "source": {"kind": "completed_full_A_receipt",
                         "path": str(POSITIVE_RECEIPT), "sha256": EXPECTED[POSITIVE_RECEIPT]}},
    }
    frontend = callable_frontend_preflight(prior_packet["config"], checkpoints)

    images = []
    workers = []
    total_bound = 0
    for image_index, (image_id, offsets) in enumerate(OFFSETS.items()):
        stable = stable_by_id[image_id]
        positive = positive_by_id[image_id]
        require(stable["example_id"] == positive["example_id"], f"{image_id}: example association differs")
        stable_ids, positive_ids = list(stable["stable_ids"]), list(positive["action_ids"])
        stable_stop, positive_stop = stable["stable_score"]["stop_reason"], positive["stop_reason"]
        validate_action(stable_ids, stable_stop)
        validate_action(positive_ids, positive_stop)
        require(first_difference(stable_ids, positive_ids) == offsets["first_difference"], f"{image_id}: first difference differs")
        h_end = offsets["h_end"]
        h_ids = stable_ids[:h_end]
        require(h_ids == positive_ids[:h_end], f"{image_id}: h differs")
        h_rows = split_complete_rows(h_ids, allow_empty=True)
        histories = {}
        for name, ids, stop in (("stable50", stable_ids, stable_stop), ("positive32", positive_ids, positive_stop)):
            end = offsets[f"{name}_end"]
            history = ids[:end]
            branch = ids[h_end:end]
            require(split_complete_rows(branch) == [branch], f"{image_id}/{name}: branch is not one complete row")
            rows = split_complete_rows(history)
            require(len(rows) == len(h_rows) + 1, f"{image_id}/{name}: history row count differs")
            require(EOS not in history and end < len(ids), f"{image_id}/{name}: history terminal/budget differs")
            histories[name] = {
                "h_ids": h_ids,
                "h_end_exclusive": h_end,
                "h_complete_row_count": len(h_rows),
                "branch_row_ids": branch,
                "branch_row_ids_sha256": digest(branch),
                "history_ids": history,
                "history_ids_sha256": digest(history),
                "history_end_exclusive": end,
                "history_complete_row_count": len(rows),
                "full_action_ids": ids,
                "full_action_ids_sha256": digest(ids),
                "full_stop_reason": stop,
                "expected_on_diagonal_free_ids": ids[end:],
                "expected_on_diagonal_free_ids_sha256": digest(ids[end:]),
                "free_budget": CAP - end,
            }
        case = stable["case"]
        plan = case["image_plan"]
        require(positive["prompt_token_ids_sha256"] == endpoint.digest(stable["prompt_token_ids"]), f"{image_id}: prompt digest differs")
        require(positive["executed_media_sha256"] == plan["executed_media_sha256"] and
                positive["observed_image_grid_thw"] == plan["observed_image_grid_thw"], f"{image_id}: media/grid differs")
        images.append({
            "image_id": image_id,
            "example_id": stable["example_id"],
            "split": positive["split"],
            "offsets": offsets,
            "prompt_token_ids": stable["prompt_token_ids"],
            "prompt_token_ids_sha256": digest(stable["prompt_token_ids"]),
            "image": {"path": case["image_path"], "sha256": plan["image_content_sha256"],
                      "executed_media_sha256": plan["executed_media_sha256"],
                      "observed_image_grid_thw": plan["observed_image_grid_thw"],
                      "width": case["image_width"], "height": case["image_height"]},
            "source_case": case,
            "golden": stable["golden"],
            "histories": histories,
            "source_records": {
                "stable50": {"packet": str(ENDPOINT_PACKET), "packet_sha256": EXPECTED[ENDPOINT_PACKET],
                             "stable_action_field": "eval_records[].stable_ids"},
                "positive32": {"consumer": str(ENDPOINT_CONSUMER), "consumer_sha256": EXPECTED[ENDPOINT_CONSUMER],
                              "request_id": positive["request_id"], "adapter_fingerprint": positive["adapter_fingerprint"]},
            },
        })
        per_checkpoint_bound = histories["stable50"]["free_budget"] + histories["positive32"]["free_budget"]
        require(per_checkpoint_bound <= 6150, f"{image_id}: worker token bound exceeded")
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            worker = image_index * 2 + checkpoint_index
            workers.append({"worker": worker, "checkpoint": checkpoint, "image_id": image_id,
                            "on_diagonal_history": checkpoint,
                            "cross_history": "positive32" if checkpoint == "stable50" else "stable50",
                            "calls": 2, "model_loads": 1, "image_forwards": 2,
                            "max_generated_tokens": per_checkpoint_bound})
            total_bound += per_checkpoint_bound
    require(total_bound == 49082 and len(workers) == 8, "global token/worker bound differs")
    source_files = {}
    for path in (HERE / "unit.md", HERE / "prepare.py", HERE / "run_cross.py", HERE / "test_checkpoint_history_cross.py",
                 ENDPOINT_PACKET, ENDPOINT_CONSUMER, POSITIVE_RECEIPT, ENDPOINT_PRODUCER, WITNESS_RUNNER, WITNESS_REDUCER):
        require(path.is_file(), f"missing packet source: {path}")
        source_files[str(path)] = file_hash(path)
    return {
        "schema": "checkpoint_history_cross.packet.v1",
        "status": "candidate_cpu_prepared_no_gpu_grant",
        "question": "Does the first divergent complete row seed subsequent behavior, or do checkpoint parameters change continuation under the same literal history?",
        "claim_boundary": "Exposed four-image checkpoint/history cross; no GT-positive-row, physical-owner, hallucination, KV-circuit, generalization, training, or promotion claim.",
        "unit": {"path": str(HERE / "unit.md"), "sha256": file_hash(HERE / "unit.md")},
        "prior_endpoint": {"packet": str(ENDPOINT_PACKET), "packet_sha256": EXPECTED[ENDPOINT_PACKET],
                           "positive_consumer": str(ENDPOINT_CONSUMER), "positive_consumer_sha256": EXPECTED[ENDPOINT_CONSUMER]},
        "model": prior_packet["model"],
        "config": prior_packet["config"],
        "raw_config_adapter_path_is_source_trap": prior_packet["config"]["adapter"]["path"],
        "checkpoints": checkpoints,
        "frontend_cpu_preflight": frontend,
        "policy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                   "use_model_defaults": False, "trace": "none", "eos_token_id": EOS,
                   "pad_token_id": PAD, "total_action_cap": CAP},
        "images": images,
        "workers": workers,
        "limits": {"workers": 8, "cells": 16, "calls": 16, "model_loads": 8,
                   "image_forwards": 16, "global_generated_tokens_max": 49082,
                   "worker_generated_tokens_max": 6150, "seconds_per_worker": 1500,
                   "cuda_allocated_bytes_max": 12 * 1024**3, "cuda_reserved_bytes_max": 12 * 1024**3,
                   "rss_bytes_max": 16 * 1024**3, "training_steps": 0},
        "source_files": source_files,
    }


def validate_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    require(packet.get("schema") == "checkpoint_history_cross.packet.v1" and
            packet.get("status") == "candidate_cpu_prepared_no_gpu_grant", "packet status/schema differs")
    for raw, expected in packet["source_files"].items():
        path = Path(raw)
        require(path.is_absolute() and path.is_file() and file_hash(path) == expected, f"packet source changed: {path}")
    require(dict(packet) == build_packet(), "packet is not the current frozen projection")
    return {"images": len(packet["images"]), "workers": len(packet["workers"]),
            "cells": packet["limits"]["cells"], "global_generated_tokens_max": packet["limits"]["global_generated_tokens_max"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    require(args.out_root.is_absolute() and not args.out_root.exists(), "--out-root must be a fresh absolute path")
    packet = build_packet()
    packet_path = args.out_root / "preparation" / "packet.json"
    publish(packet_path, packet)
    invariants = validate_packet(packet)
    receipt = {"schema": "checkpoint_history_cross.preparation_receipt.v1", "status": "candidate_no_gpu_grant",
               "packet": str(packet_path), "packet_sha256": file_hash(packet_path), "invariants": invariants}
    publish(args.out_root / "preparation" / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
