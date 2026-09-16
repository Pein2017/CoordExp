"""Frozen post-fit A/B natural evaluation against the same N16 anchor.

The module owns only the transport between two completed owner-successor fits
and the already accepted native evaluator.  It deliberately reuses the old640
N16 rows and the new confirmation256 N16 rows; it never regenerates an anchor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.native_owner_scale import evaluation as native
from src.data.geometry import iou_xyxy


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
UNIT_ROOT = BASE / "2026-09-13-owner-successor-scale-throughput"
ROOT = UNIT_ROOT / "evaluation/paired-preparation"
OLD_ROOT = BASE / "2026-09-12-native-owner-scale-and-state/evaluation"
OLD_COMPLETION = OLD_ROOT / "completion-v11.json"
OLD_PACKET = OLD_ROOT / "candidate-panel-bound-v11.json"
NEW_RESULT = UNIT_ROOT / "evaluation/anchor-result.json"
NEW_PACKET = UNIT_ROOT / "evaluation/packet.json"
NEW_SELECTION = UNIT_ROOT / "evaluation/confirmation-selection.json"
SEALED_TRAINING_INPUT = UNIT_ROOT / "training/inputs-sealed-v1.json"
SEALED_TRAINING_SHA256 = "0c3659f2b8e3ac7c0172f72bce6abae74e2e7d0c9776eaa15f463deec4a2f548"
N16_ADAPTER_SHA256 = "092b47a56b2b50475e97e0a2f1fcbd058ba6f53f131cc7750af992ff4922db35"
N16_ADAPTER_FINGERPRINT = "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"
CAP = 3084
EOS = 151645
OLD_COUNT = 640
CONFIRMATION_COUNT = 256
TOTAL_COUNT = OLD_COUNT + CONFIRMATION_COUNT
BLIND_COUNT = 32
WORKERS = 8
ARMS = ("A", "B")
BLIND_SALT = "owner-successor-scale-throughput-paired-blind-v1:"
POSITION_BINS = ((0, 8, "0-7"), (8, 16, "8-15"), (16, 32, "16-31"),
                 (32, 64, "32-63"), (64, 10**9, "64+"))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return native.read_jsonl(path)


def binding(path: str | Path) -> dict[str, Any]:
    return native.binding(path)


def _verify_binding(value: Mapping[str, Any], label: str) -> Path:
    require(isinstance(value, Mapping) and value.get("path") and value.get("sha256"),
            f"missing {label} binding")
    path = Path(str(value["path"]))
    require(path.is_file(), f"missing {label}")
    require(binding(path) == dict(value), f"{label} binding changed")
    return path


def _training_binding(path: str | Path) -> dict[str, str]:
    """Return the exact two-field shape emitted by owner_successor_scale training."""
    path = Path(path)
    require(path.is_file(), f"missing training artifact: {path}")
    return {"path": str(path), "sha256": native.file_hash(path)}


def _verify_training_binding(value: Mapping[str, Any], label: str) -> Path:
    """Verify, without widening, the training producer's {path, sha256} contract."""
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            f"{label} training binding shape")
    path = Path(str(value["path"]))
    require(path.is_file(), f"missing {label}")
    require(dict(value) == _training_binding(path), f"{label} training binding changed")
    return path


def _training_projection(value: Mapping[str, Any], label: str) -> dict[str, str]:
    """Project an evaluation-owned three-field binding to the training contract."""
    require(isinstance(value, Mapping) and value.get("path") and value.get("sha256"),
            f"missing {label}")
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def publish(path: str | Path, value: Any) -> None:
    native.publish(path, value)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"output occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _script_binding() -> dict[str, Any]:
    return binding(Path(__file__).resolve())


def _generation_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "dtype": "fp32", "attention": "sdpa", "temperature": 0.0,
        "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
        "max_new_tokens": CAP, "eos_token_id": EOS,
        "natural_prefix_ids": [], "forced_credit": False,
    }
    require(all(value.get(key) == expected_value for key, expected_value in expected.items()),
            "prompt/budget generation contract changed")


def _same_prompt(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for key in ("data", "template", "generation", "scoring", "backend", "model"):
        require(left["config"][key] == right["config"][key],
                f"old/new native config differs at {key}")
    _generation_contract(left["generation"])
    _generation_contract(right["generation"])


def _exact_rows(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                *, arm: str, adapter_fingerprint: str, packet_sha256: str | None = None) -> None:
    expected = [str(record["example_id"]) for record in records]
    actual = [str(row.get("example_id")) for row in rows]
    require(len(rows) == len(records), f"{arm} row omission")
    require(len(set(actual)) == len(actual), f"{arm} row duplication")
    require(actual == expected, f"{arm} row order/population changed")
    for row, record in zip(rows, records, strict=True):
        require(row.get("arm") == arm, f"{arm} row arm mismatch")
        require(int(row.get("image_id")) == int(record["image_id"]), f"{arm} image mismatch")
        require(str(row.get("request_id")) == str(record["example_id"]), f"{arm} request mismatch")
        require(row.get("adapter_fingerprint") == adapter_fingerprint, f"{arm} wrong-anchor adapter")
        require(row.get("prefix_ids") == row.get("forced_ids") == [], f"{arm} is not natural")
        require(row.get("remaining_budget") == CAP, f"{arm} row budget mismatch")
        require(row.get("prompt_token_ids_sha256") == native.digest(record["prompt_token_ids"]),
                f"{arm} prompt mismatch")
        plan = record["case"]["image_plan"]
        require(row.get("executed_media_sha256") == plan["executed_media_sha256"],
                f"{arm} media mismatch")
        require(row.get("observed_image_grid_thw") == plan["observed_image_grid_thw"],
                f"{arm} image grid mismatch")
        observed = {"prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
                    "executed_media_sha256": row.get("executed_media_sha256"),
                    "observed_image_grid_thw": row.get("observed_image_grid_thw")}
        require(row.get("batch_identity_sha256") == native.digest(observed),
                f"{arm} batch identity mismatch")
        if packet_sha256 is not None:
            require(row.get("packet_sha256") == packet_sha256, f"{arm} packet mismatch")


def _tagged_records(old_packet: Mapping[str, Any], new_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    old_records = copy.deepcopy(old_packet["records"])
    new_records = copy.deepcopy(new_packet["records"])
    for row in new_records:
        row["source_record_sha256"] = native.digest({k: v for k, v in row.items() if k != "source_record_sha256"})
        row["source_panel"] = "confirmation256"
        row["evaluation_stratum"] = "confirmation256"
    records = [*old_records, *new_records]
    ids = [int(row["image_id"]) for row in records]
    require(len(ids) == len(set(ids)) == TOTAL_COUNT, "old640/confirmation256 overlap or denominator")
    return records


def _cold_reconsume_anchor(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]],
                           old_packet: Mapping[str, Any], new_packet: Mapping[str, Any]) -> None:
    """Replay tokenizer, accepted parser, scorer and overlap accounting on CPU."""
    from transformers import AutoTokenizer
    from src.eval.native_rows import native_detection_record as native_record

    tokenizer = AutoTokenizer.from_pretrained(new_packet["model"]["base_model_path"], local_files_only=True)
    native.validate_endpoint_rows(
        old_rows, old_packet, arm="scaled_terminal", tokenizer=tokenizer,
        expected_image_ids=[int(record["image_id"]) for record in old_packet["records"]],
    )
    for row, frozen in zip(new_rows, new_packet["records"], strict=True):
        ids = list(row["action_ids"])
        native.accepted._checked_action(ids, row["stop_reason"], CAP)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        require(text == row["text"], "new256 anchor token/text mismatch")
        parsed = native_record(text, frozen["case"], frozen["golden"], row["stop_reason"])
        require(parsed == row["parsed"], "new256 anchor cold parser mismatch")
        require(native.score(parsed, seed=None, length=len(ids), stop=row["stop_reason"]) == row["score"],
                "new256 anchor cold scorer mismatch")
        require(native.overlap_counts(parsed) == row["overlap_counts"],
                "new256 anchor cold overlap mismatch")


def _blind_freeze(records: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], output: Path) -> dict[str, Any]:
    blind_ids = [int(value) for value in selection["blind_review_ids"]]
    require(len(blind_ids) == len(set(blind_ids)) == BLIND_COUNT, "source-blind32 denominator")
    by_id = {int(record["image_id"]): record for record in records}
    require(set(blind_ids) <= set(by_id), "source-blind32 outside combined panel")
    value = {
        "schema": "owner_successor_scale.paired_evaluation.blind_freeze.v1",
        "status": "source_identities_frozen_before_trained_outputs",
        "selection": binding(NEW_SELECTION),
        "image_ids": blind_ids,
        "image_ids_sha256": native.digest(blind_ids),
        "items": [{
            "review_id": f"owner-successor-paired:{image_id}",
            "image_id": image_id,
            "example_id": by_id[image_id]["example_id"],
            "literal_source_canvas_path": by_id[image_id]["case"]["image_path"],
            "image_width": by_id[image_id]["case"]["image_width"],
            "image_height": by_id[image_id]["case"]["image_height"],
        } for image_id in blind_ids],
        "boundary": "One image per later review item; no prediction, arm, GT-negative, physical-owner, or exhaustive-recall label exists at freeze time.",
    }
    publish(output / "blind-review-freeze.json", value)
    return value


def _validate_prepared(packet: Mapping[str, Any], *, allow_bound: bool = True) -> None:
    statuses = {"trained_endpoints_pending"}
    if allow_bound:
        statuses.add("ready_for_root_accepted_endpoint")
    require(packet.get("schema") == "owner_successor_scale.paired_evaluation.packet.v1",
            "paired packet schema")
    require(packet.get("status") in statuses, "paired packet status")
    require(len(packet.get("records", [])) == TOTAL_COUNT, "paired record denominator")
    ids = [int(row["image_id"]) for row in packet["records"]]
    require(len(set(ids)) == TOTAL_COUNT, "paired duplicate record identity")
    require(packet.get("denominators") == {
        "old_exposed640": OLD_COUNT, "confirmation256": CONFIRMATION_COUNT,
        "combined896": TOTAL_COUNT, "source_blind32": BLIND_COUNT,
    }, "paired denominators changed")
    _generation_contract(packet.get("generation", {}))
    require(_verify_binding(packet["training_input"], "sealed training input").resolve()
            == SEALED_TRAINING_INPUT.resolve(), "trained receipt input path mismatch")
    require(packet["training_input"]["sha256"] == SEALED_TRAINING_SHA256,
            "trained receipt input SHA mismatch")
    require(_verify_binding(packet["anchor"]["rows"], "combined N16 anchor rows"),
            "combined anchor rows")
    require(packet["anchor"]["adapter_fingerprint"] == N16_ADAPTER_FINGERPRINT
            and packet["anchor"]["adapter_weights_sha256"] == N16_ADAPTER_SHA256,
            "wrong-anchor binding")
    _verify_binding(packet["blind_review"]["freeze"], "blind review freeze")


def prepare(*, output: str | Path = ROOT) -> dict[str, Any]:
    """Cold-bind and actually consume the two completed N16 row sources."""
    output = Path(output)
    require(not output.exists(), f"paired preparation output occupied: {output}")
    output.mkdir(parents=True)
    require(binding(SEALED_TRAINING_INPUT)["sha256"] == SEALED_TRAINING_SHA256,
            "sealed training packet changed")
    old_completion, old_packet = read(OLD_COMPLETION), read(OLD_PACKET)
    new_result, new_packet, selection = read(NEW_RESULT), read(NEW_PACKET), read(NEW_SELECTION)
    require(old_completion.get("status") == "mechanically_complete_physical_review_pending",
            "old640 completion status")
    require(old_completion["packet"] == binding(OLD_PACKET), "old640 completion packet binding")
    require(new_result.get("status") == "cold_verified_n16_anchor_256", "new N16 anchor status")
    require(new_result["selection"] == binding(NEW_SELECTION), "new selection/result binding")
    require(new_result["rows"] == binding(new_result["rows"]["path"]), "new anchor row binding")
    _same_prompt(old_packet, new_packet)
    require(old_packet["scaled_terminal"]["adapter"]["fingerprint"] == N16_ADAPTER_FINGERPRINT,
            "old640 is not the N16 candidate anchor")
    old_weights = next(item for item in old_packet["scaled_terminal"]["adapter"]["files"]
                       if item["relative_path"] == "adapter_model.safetensors")
    require(old_weights["sha256"] == N16_ADAPTER_SHA256,
            "old640 N16 weights changed")
    require(new_packet["n16_anchor"]["adapter_fingerprint"] == N16_ADAPTER_FINGERPRINT
            and new_packet["n16_anchor"]["adapter_weights_sha256"] == N16_ADAPTER_SHA256,
            "new256 wrong N16 anchor")
    records = _tagged_records(old_packet, new_packet)
    old_rows_path = _verify_binding(old_completion["rows"]["scaled_terminal"], "old640 N16 rows")
    old_rows, new_rows = read_jsonl(old_rows_path), read_jsonl(new_result["rows"]["path"])
    _exact_rows(old_rows, old_packet["records"], arm="scaled_terminal",
                adapter_fingerprint=N16_ADAPTER_FINGERPRINT,
                packet_sha256=old_completion["packet"]["sha256"])
    baseline_packet = _verify_binding(new_result["packet"], "new256 anchor packet")
    _exact_rows(new_rows, new_packet["records"], arm="N16-anchor",
                adapter_fingerprint=N16_ADAPTER_FINGERPRINT,
                packet_sha256=binding(NEW_PACKET)["sha256"])
    require(all(row.get("baseline_packet_sha256") == binding(baseline_packet)["sha256"]
                for row in new_rows), "new256 baseline packet mismatch")
    _cold_reconsume_anchor(old_rows, new_rows, old_packet, new_packet)
    anchor_rows = [*old_rows, *new_rows]
    anchor_path = output / "n16-anchor-rows-896.jsonl"
    _write_jsonl(anchor_path, anchor_rows)
    freeze = _blind_freeze(records, selection, output)
    old_strata = copy.deepcopy(old_packet["strata"])
    strata = {**old_strata, "old_exposed640": {
        "role": "old_exposed640", "count": OLD_COUNT,
        "image_ids": [int(row["image_id"]) for row in old_packet["records"]],
    }, "confirmation256": {
        "role": "confirmation256", "count": CONFIRMATION_COUNT,
        "image_ids": [int(row["image_id"]) for row in new_packet["records"]],
    }}
    packet = {
        "schema": "owner_successor_scale.paired_evaluation.packet.v1",
        "status": "trained_endpoints_pending",
        "records": records,
        "config": copy.deepcopy(new_packet["config"]),
        "generation": copy.deepcopy(new_packet["generation"]),
        "model": copy.deepcopy(new_packet["model"]),
        "strata": strata,
        "denominators": {"old_exposed640": OLD_COUNT, "confirmation256": CONFIRMATION_COUNT,
                         "combined896": TOTAL_COUNT, "source_blind32": BLIND_COUNT},
        "anchor": {
            "label": "N16-anchor", "adapter_fingerprint": N16_ADAPTER_FINGERPRINT,
            "adapter_weights_sha256": N16_ADAPTER_SHA256, "rows": binding(anchor_path),
            "old640_rows": binding(old_rows_path), "confirmation256_rows": new_result["rows"],
            "old_completion": binding(OLD_COMPLETION), "confirmation_result": binding(NEW_RESULT),
            "reuse_only_no_regeneration": True,
        },
        "training_input": binding(SEALED_TRAINING_INPUT),
        "endpoints": {arm: {"status": "pending_exact_full256_receipt_and_cold_check"} for arm in ARMS},
        "blind_review": {"freeze": binding(output / "blind-review-freeze.json"),
                         "source_blind": True, "single_image_items": True,
                         "visual_labels": "none_pending_root_Luna_review"},
        "source_packets": {"old640": binding(OLD_PACKET), "confirmation256": binding(NEW_PACKET),
                           "confirmation_selection": binding(NEW_SELECTION)},
        "baseline_cpu_consumer": {
            "status": "cold_consumed_existing_N16_rows",
            "images": len(anchor_rows), "cold_reparsed_rows": len(anchor_rows),
            "quality": native.aggregate_scores([row["score"] for row in anchor_rows]),
            "burden": native.accepted.burden(anchor_rows),
            "self_owner_changes": native.accepted.owner_change_counts(
                [row["score"] for row in anchor_rows], [row["score"] for row in anchor_rows]),
        },
        "producer": _script_binding(),
        "claim_boundary": "Preparation and anchor consumption only. No trained A/B endpoint, improvement, physical-owner label, promotion, or publication claim.",
    }
    publish(output / "packet.json", packet)
    _validate_prepared(packet, allow_bound=False)
    readiness = {
        "schema": "owner_successor_scale.paired_evaluation.readiness.v1",
        "status": "ready_waiting_for_two_full256_cold_receipts",
        "packet": binding(output / "packet.json"), "anchor_images_cold_consumed": TOTAL_COUNT,
        "missing_boundary": "Actual arm A and arm B 256-update receipts and their passing cold-checks; the one-update smoke is ineligible.",
        "bind_command": f"python -m probes.owner_successor_scale.paired_evaluation bind --packet {output/'packet.json'} --receipt-a <full-A/receipt.json> --cold-a <full-A/cold-check.json> --receipt-b <full-B/receipt.json> --cold-b <full-B/cold-check.json> --output {output/'packet-bound.json'}",
    }
    publish(output / "readiness.json", readiness)
    return {"packet": binding(output / "packet.json"), "readiness": binding(output / "readiness.json"),
            "anchor_rows": binding(anchor_path), "anchor_images": TOTAL_COUNT,
            "blind_images": len(freeze["image_ids"]), "model_calls": 0}


def _verify_endpoint(receipt_path: Path, cold_path: Path, *, arm: str,
                     training_input: Mapping[str, Any]) -> dict[str, Any]:
    receipt, cold = read(receipt_path), read(cold_path)
    require(receipt.get("schema") == "owner_successor_scale.training.receipt.v1"
            and receipt.get("status") == "technically_completed_cold_pending", f"arm {arm} receipt status")
    require(receipt.get("arm") == arm, f"arm {arm} trained-receipt mismatch")
    receipt_input = receipt.get("input", {})
    _verify_training_binding(receipt_input, f"arm {arm} training input")
    require(receipt_input == _training_projection(training_input, "paired sealed training input"),
            f"arm {arm} trained-receipt input mismatch")
    require(cold.get("schema") == "owner_successor_scale.training.cold_check.v1"
            and cold.get("status") == "passed", f"arm {arm} cold-check status")
    cold_receipt = cold.get("training_receipt", {})
    _verify_training_binding(cold_receipt, f"arm {arm} cold training receipt")
    require(cold_receipt == _training_binding(receipt_path), f"arm {arm} cold/receipt mismatch")
    require(receipt.get("updates") == 256 and receipt.get("world_size") == 8,
            f"arm {arm} is not a full256 endpoint")
    require(len(receipt.get("terminals", [])) == 8 and len(receipt.get("update_records", [])) == 8
            and len(receipt.get("global_updates", [])) == 256, f"arm {arm} receipt denominator")
    for name in ("terminals", "update_records", "refresh_records", "final_protection_records"):
        require(len(receipt.get(name, [])) == 8, f"arm {arm} {name} denominator")
        for index, item in enumerate(receipt[name]):
            _verify_training_binding(item, f"arm {arm} {name}[{index}]")
    adapter = copy.deepcopy(receipt.get("saved_adapter"))
    native._verify_adapter_files(adapter, f"arm {arm} adapter")
    source = receipt.get("source_adapter", {})
    require(source.get("fingerprint") == N16_ADAPTER_FINGERPRINT,
            f"arm {arm} source is not the frozen N16 anchor")
    require(Path(str(adapter.get("root", ""))).resolve() != Path(str(source.get("root", ""))).resolve(),
            f"arm {arm} saved root aliases the N16 source root")
    return {"schema": "owner_successor_scale.paired_evaluation.endpoint.v1", "status": "cold_verified",
            "arm": arm, "updates": 256, "world_size": 8, "training_input": dict(training_input),
            "receipt": binding(receipt_path), "cold_check": binding(cold_path), "adapter": adapter}


def bind_endpoints(*, packet_path: str | Path, receipt_a: str | Path, cold_a: str | Path,
                   receipt_b: str | Path, cold_b: str | Path, output: str | Path) -> dict[str, Any]:
    packet_path, output = Path(packet_path), Path(output)
    packet = read(packet_path)
    _validate_prepared(packet, allow_bound=False)
    require(not output.exists(), f"bound packet occupied: {output}")
    bound = copy.deepcopy(packet)
    bound["endpoints"] = {
        "A": _verify_endpoint(Path(receipt_a), Path(cold_a), arm="A", training_input=packet["training_input"]),
        "B": _verify_endpoint(Path(receipt_b), Path(cold_b), arm="B", training_input=packet["training_input"]),
    }
    require(bound["endpoints"]["A"]["adapter"]["root"] != bound["endpoints"]["B"]["adapter"]["root"],
            "A/B adapter roots must be distinct")
    bound["status"] = "ready_for_root_accepted_endpoint"
    bound["base_packet"] = binding(packet_path)
    bound["producer"] = _script_binding()
    bound["launch_contract"] = {
        "workers_per_arm": WORKERS, "images_per_arm": TOTAL_COUNT,
        "max_new_tokens_per_image": CAP, "max_new_tokens_per_arm": TOTAL_COUNT * CAP,
        "model_loads_per_arm": WORKERS, "retry_policy": "none; preserve partial shards",
        "root_gate": "binding is technical readiness only; root separately accepts and schedules each full endpoint",
    }
    publish(output, bound)
    _validate_prepared(bound)
    return bound


def _validate_bound(packet: Mapping[str, Any]) -> None:
    _validate_prepared(packet)
    require(packet["status"] == "ready_for_root_accepted_endpoint", "endpoints not bound")
    for arm in ARMS:
        endpoint = packet["endpoints"][arm]
        require(endpoint.get("status") == "cold_verified" and endpoint.get("arm") == arm,
                f"arm {arm} endpoint binding")
        _verify_binding(endpoint["receipt"], f"arm {arm} receipt")
        _verify_binding(endpoint["cold_check"], f"arm {arm} cold check")
        native._verify_adapter_files(endpoint["adapter"], f"arm {arm} adapter")


def worker(*, packet_path: str | Path, arm: str, shard: int, physical_gpu: int,
           output: str | Path) -> None:
    """One GPU producer using the accepted native materializer/parser/scorer."""
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.eval.native_rows import native_detection_record as native_record
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    packet_path, output = Path(packet_path), Path(output)
    packet = read(packet_path)
    _validate_bound(packet)
    require(arm in ARMS and shard in range(WORKERS), "paired worker arm/shard")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu)
            and torch.cuda.device_count() == 1, "paired worker explicit single GPU")
    require(not output.exists(), "paired worker output occupied")
    output.mkdir(parents=True)
    records = packet["records"][shard::WORKERS]
    adapter = packet["endpoints"][arm]["adapter"]
    terminal: dict[str, Any] = {"schema": "owner_successor_scale.paired_evaluation.terminal.v1",
        "status": "running", "arm": arm, "shard": shard, "physical_gpu": physical_gpu,
        "packet": binding(packet_path), "expected_continuations": len(records), "continuations": 0,
        "new_tokens": 0, "model_forwards": 0, "image_forwards": 0, "model_loads": 0}
    publish(output / "launch.json", terminal)
    handles: list[Any] = []
    started = time.monotonic()
    try:
        config = checkpoint_config(InferConfig.model_validate(packet["config"]), adapter["root"])
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live = identity["model_identity"]["adapter"]
        require(live["adapter_path"] == adapter["root"] and live.get("merged_adapters", []) == [],
                "loaded paired adapter identity changed")
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
                and identity["effective_settings"]["observed_attn_implementation"] == "sdpa",
                "live paired FP32/SDPA identity changed")
        publish(output / "model.json", {"status": "live_model_loaded_before_inference", "arm": arm,
                "packet": binding(packet_path), "identity": identity, "adapter": adapter,
                "generation": packet["generation"]})
        publish(output / "source.json", {"status": "frozen_input_bound_before_inference", "arm": arm,
                "packet": binding(packet_path), "image_ids": [int(r["image_id"]) for r in records],
                "records": [{"image_id": int(r["image_id"]), "example_id": r["example_id"],
                    "image_path": r["case"]["image_path"],
                    "prompt_token_ids_sha256": native.digest(r["prompt_token_ids"]),
                    "executed_media_sha256": r["case"]["image_plan"]["executed_media_sha256"],
                    "observed_image_grid_thw": r["case"]["image_plan"]["observed_image_grid_thw"],
                    "source_panel": r["source_panel"], "evaluation_stratum": r["evaluation_stratum"]}
                    for r in records]})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(
            lambda *_: terminal.__setitem__("model_forwards", terminal["model_forwards"] + 1)))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "paired visual hook ambiguity")
        handles.append(visual[0].register_forward_pre_hook(
            lambda *_: terminal.__setitem__("image_forwards", terminal["image_forwards"] + 1)))
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0,
                                        repetition_penalty=1.0, use_model_defaults=False)
        with (output / "rows.jsonl").open("x") as stream:
            for frozen in records:
                case = native._candidate_materialized_case(frozen["case"], packet["config"])
                requests, _ = build_requests(qwen, packet["config"], [case])
                batch = prepare_native_inputs(qwen.processor, requests, device="cuda:0",
                                              record_media_identity=True)
                plan = frozen["case"]["image_plan"]
                require(list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"]
                        and batch.media_sha256[0] == plan["executed_media_sha256"]
                        and list(batch.image_grids[0]) == plan["observed_image_grid_thw"],
                        "frozen paired natural input identity changed")
                with torch.inference_mode():
                    generated = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP],
                        eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                ids, stop = list(generated.token_ids), generated.stop_reason
                require(generated.request_id == frozen["example_id"], "paired request identity changed")
                native.accepted._checked_action(ids, stop, CAP)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = native_record(text, frozen["case"], frozen["golden"], stop)
                observed = {"prompt_token_ids_sha256": native.digest(frozen["prompt_token_ids"]),
                    "executed_media_sha256": batch.media_sha256[0],
                    "observed_image_grid_thw": list(batch.image_grids[0])}
                row = {"schema": "owner_successor_scale.paired_evaluation.natural.v1", "arm": arm,
                    "shard": shard, "example_id": frozen["example_id"], "image_id": frozen["image_id"],
                    "split": frozen["split"], "request_id": generated.request_id, "action_ids": ids,
                    "prefix_ids": [], "forced_ids": [], "remaining_budget": CAP, "text": text,
                    "stop_reason": stop, "parsed": parsed,
                    "score": native.score(parsed, seed=None, length=len(ids), stop=stop),
                    "overlap_counts": native.overlap_counts(parsed), "batch_identity_sha256": native.digest(observed),
                    "packet_sha256": binding(packet_path)["sha256"],
                    "adapter_fingerprint": adapter["fingerprint"], **observed}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                terminal["continuations"] += 1; terminal["new_tokens"] += len(ids)
        require(terminal["continuations"] == len(records), "paired shard denominator")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(output / "terminal.json", terminal)


def launch(*, packet_path: str | Path, arm: str, output: str | Path,
           physical_gpus: Sequence[int]) -> dict[str, Any]:
    packet_path, output = Path(packet_path), Path(output)
    _validate_bound(read(packet_path))
    gpus = [int(value) for value in physical_gpus]
    require(arm in ARMS and len(gpus) == len(set(gpus)) == WORKERS, "paired launch arm/GPU denominator")
    require(not output.exists(), "paired launch output occupied; no automatic retry")
    output.mkdir(parents=True)
    processes, logs, commands = [], [], []
    for shard, gpu in enumerate(gpus):
        command = [sys.executable, "-m", "probes.owner_successor_scale.paired_evaluation", "worker",
                   "--packet", str(packet_path), "--arm", arm, "--shard", str(shard),
                   "--physical-gpu", str(gpu), "--output", str(output / f"shard-{shard}")]
        log = (output / f"shard-{shard}.log").open("x")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   TOKENIZERS_PARALLELISM="false")
        processes.append(subprocess.Popen(command, cwd=WORKTREE, env=env, stdout=log,
                                          stderr=subprocess.STDOUT)); logs.append(log); commands.append(command)
    publish(output / "launch.json", {"status": "running", "packet": binding(packet_path), "arm": arm,
            "images": TOTAL_COUNT, "physical_gpus": gpus, "commands": commands, "no_retry": True})
    exits = []
    for shard, (process, log) in enumerate(zip(processes, logs, strict=True)):
        exits.append({"shard": shard, "exit_code": process.wait()}); log.close()
    publish(output / "outer-exits.json", exits)
    require(all(row["exit_code"] == 0 for row in exits), "paired producer failed; preserve partial outputs")
    return {"status": "completed", "arm": arm, "images": TOTAL_COUNT, "output": str(output)}


def merge(*, packet_path: str | Path, arm: str, output: str | Path) -> dict[str, Any]:
    packet_path, output = Path(packet_path), Path(output)
    packet = read(packet_path); _validate_bound(packet); require(arm in ARMS, "paired merge arm")
    exits = read(output / "outer-exits.json")
    require(len(exits) == WORKERS and all(row["exit_code"] == 0 for row in exits), "paired outer exits")
    rows, terminals = [], []
    for shard in range(WORKERS):
        terminal = read(output / f"shard-{shard}/terminal.json")
        require(terminal.get("status") == "completed" and terminal.get("arm") == arm,
                "paired shard terminal")
        shard_rows = read_jsonl(output / f"shard-{shard}/rows.jsonl")
        require(len(shard_rows) == len(packet["records"][shard::WORKERS]), "paired shard row count")
        rows.extend(shard_rows); terminals.append(binding(output / f"shard-{shard}/terminal.json"))
    by_id = {str(row["example_id"]): row for row in rows}
    expected = [str(record["example_id"]) for record in packet["records"]]
    require(len(by_id) == len(rows) == TOTAL_COUNT and set(by_id) == set(expected),
            "paired merged omission/duplication")
    ordered = [by_id[key] for key in expected]
    _write_jsonl(output / "rows.jsonl", ordered)
    result = {"schema": "owner_successor_scale.paired_evaluation.merge.v1",
              "status": "raw_merged_pending_cold_consumer", "packet": binding(packet_path),
              "arm": arm, "rows": binding(output / "rows.jsonl"), "images": TOTAL_COUNT,
              "terminals": terminals}
    publish(output / "merge.json", result)
    return result


def _validate_endpoint_rows(rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any],
                            *, arm: str, tokenizer: Any) -> list[dict[str, Any]]:
    _exact_rows(rows, packet["records"], arm=arm,
                adapter_fingerprint=packet["endpoints"][arm]["adapter"]["fingerprint"],
                packet_sha256=packet["_file_sha256"])
    from src.eval.native_rows import native_detection_record as native_record
    checked = []
    for row, frozen in zip(rows, packet["records"], strict=True):
        ids = list(row["action_ids"]); native.accepted._checked_action(ids, row["stop_reason"], CAP)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        require(text == row["text"], f"arm {arm} token/text mismatch")
        parsed = native_record(text, frozen["case"], frozen["golden"], row["stop_reason"])
        require(parsed == row["parsed"], f"arm {arm} cold parser mismatch")
        score = native.score(parsed, seed=None, length=len(ids), stop=row["stop_reason"])
        require(score == row["score"], f"arm {arm} cold scorer mismatch")
        require(native.overlap_counts(parsed) == row["overlap_counts"], f"arm {arm} overlap mismatch")
        checked.append(dict(row))
    return checked


def _position_bin(index: int) -> str:
    return next(label for start, end, label in POSITION_BINS if start <= index < end)


def _position_changes(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for threshold in ("50", "60", "80"):
        counts = {kind: Counter() for kind in ("gained_after", "lost_before", "retained_before", "retained_after")}
        for left, right in zip(before, after, strict=True):
            lm = {str(m["owner"]): int(m["pred_index"]) for m in left["score"][threshold]["matches"]}
            rm = {str(m["owner"]): int(m["pred_index"]) for m in right["score"][threshold]["matches"]}
            for owner in rm.keys() - lm.keys(): counts["gained_after"][_position_bin(rm[owner])] += 1
            for owner in lm.keys() - rm.keys(): counts["lost_before"][_position_bin(lm[owner])] += 1
            for owner in lm.keys() & rm.keys():
                counts["retained_before"][_position_bin(lm[owner])] += 1
                counts["retained_after"][_position_bin(rm[owner])] += 1
        result[threshold] = {key: dict(value) for key, value in counts.items()}
    return result


def _first_owner(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if not left["parsed"]["pred"]:
        return {"status": "anchor_has_no_valid_first_row"}
    first = left["parsed"]["pred"][0]
    gt_match = next((m for m in left["score"]["50"]["matches"] if int(m["pred_index"]) == 0), None)
    overlaps = [(iou_xyxy(first["bbox"], row["bbox"]), row["description"] == first["description"], index)
                for index, row in enumerate(right["parsed"]["pred"])]
    best_any = max(overlaps, default=(0.0, False, None))
    best_same = max((item for item in overlaps if item[1]), default=(0.0, True, None))
    if gt_match is None:
        status = "neutral_unmatched_anchor_first_owner"
    else:
        status = "retained_known_owner" if str(gt_match["owner"]) in set(right["score"]["50"]["owners"]) else "lost_known_owner"
    return {"status": status, "anchor_description": first["description"],
            "anchor_gt_owner": None if gt_match is None else str(gt_match["owner"]),
            "strict_geometry_any_class": best_any[0] > 0.95,
            "strict_geometry_same_class": best_same[0] > 0.95,
            "best_any_iou": best_any[0], "best_same_class_iou": best_same[0]}


def _pair_ledger(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"image_id": left["image_id"], "example_id": left["example_id"],
             "owner_changes": native.accepted.owner_changes(left["score"], right["score"]),
             "first_owner": _first_owner(left, right),
             "repeat_drift": {threshold: {"anchor": left["overlap_counts"][threshold],
                 "candidate": right["overlap_counts"][threshold],
                 "delta": right["overlap_counts"][threshold] - left["overlap_counts"][threshold]}
                 for threshold in ("80", "90", "95")}}
            for left, right in zip(before, after, strict=True)]


def _repeat_summary(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for threshold in ("80", "90", "95"):
        pairs = [(int(l["overlap_counts"][threshold]), int(r["overlap_counts"][threshold]))
                 for l, r in zip(before, after, strict=True)]
        result[threshold] = {"anchor_later_rows": sum(x for x, _ in pairs),
            "candidate_later_rows": sum(y for _, y in pairs),
            "images_cleared": sum(x > 0 and y == 0 for x, y in pairs),
            "images_retained": sum(x > 0 and y > 0 for x, y in pairs),
            "images_new_loop": sum(x == 0 and y > 0 for x, y in pairs)}
    result["strict_definition"] = "each later valid row counted once if native-pixel any-class IoU > 0.95 with any earlier valid row"
    return result


def _first_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values = Counter(row["first_owner"]["status"] for row in rows)
    values["neutral_strict_geometry_same_class"] = sum(
        row["first_owner"]["status"] == "neutral_unmatched_anchor_first_owner"
        and row["first_owner"]["strict_geometry_same_class"] for row in rows)
    return dict(values)


def _blind_queue(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], packet: Mapping[str, Any],
                 output: Path) -> dict[str, Any]:
    freeze = read(packet["blind_review"]["freeze"]["path"])
    by_arm = {arm: {int(row["image_id"]): row for row in rows} for arm, rows in rows_by_arm.items()}
    queue, source = [], []
    for item in freeze["items"]:
        image_id = int(item["image_id"]); candidates = []
        for arm in ("N16-anchor", "A", "B"):
            for index, prediction in enumerate(by_arm[arm][image_id]["parsed"]["pred"]):
                proposal = {"description": prediction["description"], "bbox": list(prediction["bbox"]),
                            "bbox_format": prediction.get("bbox_format", "xyxy")}
                candidates.append((native.digest({"salt": BLIND_SALT, "image_id": image_id,
                                                   "arm": arm, "index": index, "proposal": proposal}),
                                   arm, index, proposal))
        candidates.sort()
        proposals = []
        for ordinal, (_, arm, index, proposal) in enumerate(candidates):
            proposal_id = hashlib.sha256(f"{BLIND_SALT}{image_id}:{ordinal}:{native.digest(proposal)}".encode()).hexdigest()
            proposals.append({"proposal_id": proposal_id, **proposal})
            source.append({"image_id": image_id, "proposal_id": proposal_id,
                           "source_arm": arm, "source_prediction_index": index})
        queue.append({"schema": "owner_successor_scale.paired_evaluation.blind_item.v1",
                      **item, "proposals": proposals, "source_blind": True,
                      "physical_labels": [], "review_status": "pending_root_Luna_single_image_review",
                      "scope": "mixed proposals only; no exhaustive recall or automatic negative"})
    require(len(queue) == BLIND_COUNT, "blind queue denominator")
    _write_jsonl(output / "blind-review-queue.jsonl", queue)
    publish(output / "blind-review-source-map.json", {
        "schema": "owner_successor_scale.paired_evaluation.blind_source_map.v1",
        "status": "sealed_not_for_reviewer", "rows": source,
        "boundary": "Transport mapping only; no visual or physical-owner label."})
    return {"queue": binding(output / "blind-review-queue.jsonl"),
            "source_map": binding(output / "blind-review-source-map.json"),
            "images": len(queue), "proposals": len(source), "visual_labels": 0}


def consume(*, packet_path: str | Path, rows_a: str | Path, rows_b: str | Path,
            output: str | Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    packet_path, output = Path(packet_path), Path(output)
    packet = read(packet_path); _validate_bound(packet)
    packet["_file_sha256"] = binding(packet_path)["sha256"]
    require(not output.exists(), "paired consumer output occupied")
    tokenizer = AutoTokenizer.from_pretrained(packet["model"]["base_model_path"], local_files_only=True)
    checked = {"A": _validate_endpoint_rows(read_jsonl(rows_a), packet, arm="A", tokenizer=tokenizer),
               "B": _validate_endpoint_rows(read_jsonl(rows_b), packet, arm="B", tokenizer=tokenizer)}
    anchor = read_jsonl(packet["anchor"]["rows"]["path"])
    require(len(anchor) == TOTAL_COUNT, "anchor row omission")
    require([r["example_id"] for r in anchor] == [r["example_id"] for r in packet["records"]],
            "anchor/packet population mismatch")
    output.mkdir(parents=True)
    _write_jsonl(output / "A-consumer.jsonl", checked["A"])
    _write_jsonl(output / "B-consumer.jsonl", checked["B"])
    all_results = {}
    for arm in ARMS:
        ledger = _pair_ledger(anchor, checked[arm])
        _write_jsonl(output / f"{arm}-paired-ledger.jsonl", ledger)
        by_id_left = {int(row["image_id"]): row for row in anchor}
        by_id_right = {int(row["image_id"]): row for row in checked[arm]}
        strata = {}
        for name in ("old_exposed640", "train11", "reference54", "legacy_retention",
                     "fresh256", "remaining_retention", "confirmation256"):
            ids = [int(value) for value in packet["strata"][name]["image_ids"]]
            left, right = [by_id_left[i] for i in ids], [by_id_right[i] for i in ids]
            strata[name] = {"images": len(ids), "anchor": {
                "quality": native.aggregate_scores([r["score"] for r in left]),
                "burden": native.accepted.burden(left)}, "candidate": {
                "quality": native.aggregate_scores([r["score"] for r in right]),
                "burden": native.accepted.burden(right)},
                "paired": native.paired_summary(left, right),
                "repeat_drift": _repeat_summary(left, right),
                "position_coverage": _position_changes(left, right)}
        all_results[arm] = {"endpoint": packet["endpoints"][arm],
            "anchor_vs_candidate": native.paired_summary(anchor, checked[arm]),
            "anchor": {"quality": native.aggregate_scores([r["score"] for r in anchor]),
                       "burden": native.accepted.burden(anchor)},
            "candidate": {"quality": native.aggregate_scores([r["score"] for r in checked[arm]]),
                          "burden": native.accepted.burden(checked[arm])},
            "first_owner_preservation": _first_summary(ledger),
            "repeat_drift": _repeat_summary(anchor, checked[arm]),
            "position_coverage": _position_changes(anchor, checked[arm]),
            "paired_ledger": binding(output / f"{arm}-paired-ledger.jsonl"), "strata": strata}
    anchor_by_id = {int(row["image_id"]): row for row in anchor}
    blind = _blind_queue({"N16-anchor": list(anchor_by_id.values()), **checked}, packet, output)
    result = {"schema": "owner_successor_scale.paired_evaluation.result.v1", "status": "cold_verified",
        "packet": binding(packet_path), "rows": {"N16-anchor": packet["anchor"]["rows"],
            "A": binding(output / "A-consumer.jsonl"), "B": binding(output / "B-consumer.jsonl")},
        "denominators": packet["denominators"], "arms": all_results, "blind_review": blind,
        "uncertain_owner_policy": "GT-unmatched predictions and source-blind unreviewed proposals remain neutral; geometry overlap is diagnostic, not a physical-owner label.",
        "decision_boundary": "No automatic promotion. Root interprets A and B against the same N16 anchor jointly across owner gain/retention/loss, first-owner preservation, repeats/drift, raw/invalid burden, stop behavior, and output position."}
    publish(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, default=ROOT)
    p = sub.add_parser("bind"); p.add_argument("--packet", type=Path, required=True)
    for flag in ("receipt-a", "cold-a", "receipt-b", "cold-b", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p = sub.add_parser("worker"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--shard", type=int, required=True); p.add_argument("--physical-gpu", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("launch"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--gpus", required=True)
    p = sub.add_parser("merge"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("consume"); p.add_argument("--packet", type=Path, required=True); p.add_argument("--rows-a", type=Path, required=True); p.add_argument("--rows-b", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(output=args.output)
    elif args.command == "bind": result = bind_endpoints(packet_path=args.packet, receipt_a=args.receipt_a,
        cold_a=args.cold_a, receipt_b=args.receipt_b, cold_b=args.cold_b, output=args.output)
    elif args.command == "worker": worker(packet_path=args.packet, arm=args.arm, shard=args.shard,
        physical_gpu=args.physical_gpu, output=args.output); return
    elif args.command == "launch": result = launch(packet_path=args.packet, arm=args.arm, output=args.output,
        physical_gpus=[int(x) for x in args.gpus.split(",")])
    elif args.command == "merge": result = merge(packet_path=args.packet, arm=args.arm, output=args.output)
    else: result = consume(packet_path=args.packet, rows_a=args.rows_a, rows_b=args.rows_b, output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
