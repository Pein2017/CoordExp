"""Minimal native producer and cold merger for the checkpoint/history cross."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

EOS, PAD, CAP = 151645, 151643, 3084
WALL_SECONDS = 1500


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))
    require(load(path) == value, f"publication readback differs: {path}")


def append(stream: Any, value: Any) -> None:
    stream.write(canonical_bytes(value).decode())
    stream.flush()
    os.fsync(stream.fileno())


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def prepare_module() -> Any:
    return module("checkpoint_history_prepare", HERE / "prepare.py")


def witness_modules(packet: Mapping[str, Any]) -> tuple[Any, Any]:
    paths = {Path(path).name: Path(path) for path in packet["source_files"]}
    return (module("checkpoint_history_witness_runner", paths["run_probe.py"]),
            module("checkpoint_history_witness_reducer", paths["reduce.py"]))


def checked_suffix(ids: list[int], *, budget: int, stop: str) -> None:
    require(ids and len(ids) <= budget and PAD not in ids and EOS not in ids[:-1], "free suffix token corruption")
    require((stop == "im_end" and ids[-1] == EOS) or
            (stop == "length" and len(ids) == budget and EOS not in ids), "free suffix stop/budget mismatch")


def owners(card: Mapping[str, Any]) -> dict[str, list[str]]:
    return {threshold: list(card[threshold]["owners"]) for threshold in ("50", "60", "80")}


def analyze_cell(packet: Mapping[str, Any], image: Mapping[str, Any], history_name: str,
                 free_ids: list[int], stop_reason: str, tokenizer: Any) -> dict[str, Any]:
    from probes.dora_owner_learning.candidate_opportunity import score
    from probes.source_rweak_row_cross.run import native_record

    history = image["histories"][history_name]
    prefix_ids = list(history["history_ids"])
    checked_suffix(free_ids, budget=history["free_budget"], stop=stop_reason)
    action_ids = prefix_ids + free_ids
    full_text = tokenizer.decode(action_ids, skip_special_tokens=False)
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    free_text = tokenizer.decode(free_ids, skip_special_tokens=False)
    parsed_full = native_record(full_text, image["source_case"], image["golden"], stop_reason)
    parsed_prefix = native_record(prefix_text, image["source_case"], image["golden"], "conditional")
    parsed_free = native_record(free_text, image["source_case"], image["golden"], stop_reason)
    runner, reducer = witness_modules(packet)
    partition = runner.partition_parser_rows(parsed_full, prefix_row_count=history["history_complete_row_count"])
    require(len(partition["observed_prefix_rows"]) == history["history_complete_row_count"], "prefix parser row accounting differs")
    full_score = score(parsed_full, seed=None, length=len(action_ids), stop=stop_reason)
    prefix_score = score(parsed_prefix, seed=None, length=len(prefix_ids), stop="conditional")
    free_score = score(parsed_free, seed=None, length=len(free_ids), stop=stop_reason)
    record_for_repeat = {"parsed": parsed_full, "parser_partition": partition}
    repeat_orders = reducer.strict_repeat_orders(record_for_repeat)
    burden = reducer.burden(record_for_repeat)
    first_ledger = partition["free_rows"][0] if partition["free_rows"] else None
    return {
        "history": history_name,
        "prefix_ids": prefix_ids,
        "prefix_token_count": len(prefix_ids),
        "prefix_complete_row_count": history["history_complete_row_count"],
        "free_ids": free_ids,
        "free_token_count": len(free_ids),
        "free_budget": history["free_budget"],
        "action_ids": action_ids,
        "action_token_count": len(action_ids),
        "stop_reason": stop_reason,
        "eos": stop_reason == "im_end",
        "cap": stop_reason == "length",
        "first_free_action": {"token_id": free_ids[0], "decoded": tokenizer.decode([free_ids[0]], skip_special_tokens=False)},
        "first_free_parser_row_or_fragment": first_ledger,
        "parsed_full": parsed_full,
        "parsed_prefix": parsed_prefix,
        "parsed_free": parsed_free,
        "parser_partition": partition,
        "forced_free_accounting": {
            "forced_history_token_count": len(prefix_ids),
            "forced_history_row_count": history["history_complete_row_count"],
            "forced_history_in_free_counts": False,
            "free_generated_orders": [row.get("generated_order") for row in partition["free_rows"]],
        },
        "scores": {"full": full_score, "prefix": prefix_score, "free": free_score},
        "owner_sets": {"full": owners(full_score), "prefix": owners(prefix_score), "free": owners(free_score)},
        "strict_class_blind_native_pixel_iou_gt_0_95_later_free_row_orders": repeat_orders,
        "strict_class_blind_native_pixel_iou_gt_0_95_later_free_row_count": len(repeat_orders),
        "free_drop_burden": burden,
        "claim_boundary": "Forced prefix excluded from free-row counts; annotation-relative owners only; no GT-positive-row claim.",
    }


def validate_packet(packet_path: Path) -> dict[str, Any]:
    require(packet_path.is_absolute() and packet_path.is_file(), "--packet must be an existing absolute path")
    packet = load(packet_path)
    prepare = prepare_module()
    prepare.validate_packet(packet)
    require(file_hash(packet_path) == digest(packet), "packet bytes are not canonical")
    return packet


def execute(packet_path: Path, out_root: Path, worker_id: int) -> None:
    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    require(out_root.is_absolute(), "--out-root must be absolute")
    packet = validate_packet(packet_path)
    require(0 <= worker_id < 8 and os.environ.get("CUDA_VISIBLE_DEVICES") == str(worker_id), "worker/GPU identity differs")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one assigned CUDA device required")
    worker = packet["workers"][worker_id]
    require(worker["worker"] == worker_id, "worker packet order differs")
    image = next(row for row in packet["images"] if row["image_id"] == worker["image_id"])
    checkpoint = packet["checkpoints"][worker["checkpoint"]]
    adapter = checkpoint["adapter"]
    run = out_root / f"worker-{worker_id}"
    require(not run.exists(), "occupied worker output")
    run.mkdir(parents=True)
    packet_sha = file_hash(packet_path)
    terminal = {"schema": "checkpoint_history_cross.terminal.v1", "status": "running", "exit_code": None,
                "worker": worker_id, "checkpoint": worker["checkpoint"], "image_id": worker["image_id"],
                "pid": os.getpid(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "packet_sha256": packet_sha, "adapter_fingerprint": adapter["fingerprint"],
                "model_loads": 0, "continuations": 0, "image_forwards": 0, "model_forwards": 0,
                "generated_free_tokens": 0}
    publish(run / "launch.json", {**terminal, "command": "execute"})
    start = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    handles = []
    counters = {"model": 0, "image": 0}
    error = None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("1500 second worker bound")))
        signal.alarm(WALL_SECONDS)
        torch.cuda.reset_peak_memory_stats()
        raw_config = InferConfig.model_validate(packet["config"])
        config = checkpoint_config(raw_config, adapter["root"])
        require(str(config.adapter.path) == adapter["root"] and str(raw_config.adapter.path) == packet["raw_config_adapter_path_is_source_trap"],
                "explicit checkpoint adapter override differs")
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live = identity["model_identity"]["adapter"]
        require(live["adapter_path"] == adapter["root"] and live["merged_adapters"] == [], "live adapter differs/merged")
        require(identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"] and
                identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live numeric policy differs")
        require(inspect_dora_adapter_payload(adapter["root"], packet["model"]["base_model_path"]) == adapter,
                "live adapter payload differs")
        publish(run / "model.json", {"schema": "checkpoint_history_cross.model.v1", "checkpoint": worker["checkpoint"],
                                     "adapter": adapter, "identity": identity})
        publish(run / "config.json", {"schema": "checkpoint_history_cross.config.v1", "raw_config": packet["config"],
                                      "effective_config": config.model_dump(mode="json"),
                                      "explicit_adapter_override": adapter["root"], "policy": packet["policy"]})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: counters.__setitem__("model", counters["model"] + 1)))
        visuals = [value for name, value in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")
        handles.append(visuals[0].register_forward_pre_hook(lambda *_: counters.__setitem__("image", counters["image"] + 1)))
        requests, _ = build_requests(qwen, packet["config"], [image["source_case"]])
        require(list(requests[0].expected_token_ids) == image["prompt_token_ids"], "frontend prompt differs")
        batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True)
        require(list(batch.prompt_token_ids[0]) == image["prompt_token_ids"] and
                batch.media_sha256[0] == image["image"]["executed_media_sha256"] and
                list(batch.image_grids[0]) == image["image"]["observed_image_grid_thw"], "live prompt/media/grid differs")
        batch_receipt = {"request_id": batch.request_ids[0], "prompt_token_ids_sha256": image["prompt_token_ids_sha256"],
                         "executed_media_sha256": batch.media_sha256[0],
                         "observed_image_grid_thw": list(batch.image_grids[0])}
        batch_receipt["batch_identity_sha256"] = digest(batch_receipt)
        publish(run / "batch.json", {"schema": "checkpoint_history_cross.batch.v1", **batch_receipt})
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0,
                                        repetition_penalty=1.0, use_model_defaults=False)
        records = []
        with (run / "records.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
            for order, history_name in enumerate((worker["on_diagonal_history"], worker["cross_history"])):
                history = image["histories"][history_name]
                result = generate_continuations(qwen.model, batch, extensions=[history["history_ids"]],
                    budgets=[history["free_budget"]], eos_token_id=EOS,
                    pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                require(result.request_id == image["example_id"], "request association differs")
                free_ids = list(result.token_ids)
                analysis = analyze_cell(packet, image, history_name, free_ids, result.stop_reason, qwen.tokenizer)
                on_diagonal = order == 0
                exact = (free_ids == history["expected_on_diagonal_free_ids"] and
                         analysis["action_ids"] == history["full_action_ids"] and
                         result.stop_reason == history["full_stop_reason"])
                record = {"schema": "checkpoint_history_cross.cell.v1", "worker": worker_id,
                          "checkpoint": worker["checkpoint"], "image_id": image["image_id"],
                          "example_id": image["example_id"], "history": history_name,
                          "cell_order": order, "on_diagonal": on_diagonal,
                          "exact_cached_natural_reproduction": exact if on_diagonal else None,
                          "packet_sha256": packet_sha, "adapter_fingerprint": adapter["fingerprint"],
                          "analysis": analysis, **batch_receipt}
                append(stream, record)
                records.append(record)
                terminal["continuations"] += 1
                terminal["generated_free_tokens"] += len(free_ids)
                if on_diagonal:
                    require(exact, "on-diagonal cached natural reproduction failed; cross-history cell withheld")
        require(len(records) == terminal["continuations"] == 2 and counters["image"] == 2 and
                terminal["generated_free_tokens"] == counters["model"] <= worker["max_generated_tokens"],
                "worker counter/bound differs")
        require(load_jsonl(run / "records.jsonl") == records, "records cold readback differs")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        error = exc
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - start,
                        model_forwards=counters["model"], image_forwards=counters["image"],
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(torch.device("cuda:0")),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(torch.device("cuda:0")),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        publish(run / "terminal.json", terminal)
    if error is not None:
        raise error


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def validate_cell(packet: Mapping[str, Any], record: Mapping[str, Any], worker: Mapping[str, Any],
                  image: Mapping[str, Any], tokenizer: Any, packet_sha: str) -> dict[str, Any]:
    require(record["schema"] == "checkpoint_history_cross.cell.v1" and record["worker"] == worker["worker"] and
            record["checkpoint"] == worker["checkpoint"] and record["image_id"] == worker["image_id"] and
            record["packet_sha256"] == packet_sha, "cell identity differs")
    require(record["adapter_fingerprint"] == packet["checkpoints"][worker["checkpoint"]]["adapter"]["fingerprint"], "cell adapter differs")
    analysis = record["analysis"]
    expected = analyze_cell(packet, image, record["history"], analysis["free_ids"], analysis["stop_reason"], tokenizer)
    require(analysis == expected, "cold cell analysis differs")
    history = image["histories"][record["history"]]
    if record["on_diagonal"]:
        exact = (analysis["free_ids"] == history["expected_on_diagonal_free_ids"] and
                 analysis["action_ids"] == history["full_action_ids"] and
                 analysis["stop_reason"] == history["full_stop_reason"])
        require(record["exact_cached_natural_reproduction"] is True and exact, "cold on-diagonal parity differs")
    else:
        require(record["exact_cached_natural_reproduction"] is None, "cross cell has parity label")
    return dict(record)


def merge(packet_path: Path, out_root: Path, *, verify: bool) -> dict[str, Any]:
    from tokenizers import Tokenizer

    packet = validate_packet(packet_path)
    packet_sha = file_hash(packet_path)
    tokenizer = Tokenizer.from_file(packet["model"]["base_model_path"] + "/tokenizer.json")
    all_records = []
    terminals = []
    for worker in packet["workers"]:
        run = out_root / f"worker-{worker['worker']}"
        terminal = load(run / "terminal.json")
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0 and
                terminal["packet_sha256"] == packet_sha and terminal["worker"] == worker["worker"], "worker terminal differs")
        require(terminal["model_loads"] == 1 and terminal["continuations"] == 2 and
                terminal["image_forwards"] == 2 and terminal["model_forwards"] == terminal["generated_free_tokens"] and
                terminal["generated_free_tokens"] <= worker["max_generated_tokens"], "worker counters differ")
        limits = packet["limits"]
        require(terminal["elapsed_seconds"] <= limits["seconds_per_worker"] and
                terminal["peak_cuda_allocated_bytes"] <= limits["cuda_allocated_bytes_max"] and
                terminal["peak_cuda_reserved_bytes"] <= limits["cuda_reserved_bytes_max"] and
                terminal["peak_rss_bytes"] <= limits["rss_bytes_max"], "worker resource bound exceeded")
        model = load(run / "model.json")
        adapter = packet["checkpoints"][worker["checkpoint"]]["adapter"]
        require(model["adapter"] == adapter and model["identity"]["model_identity"]["adapter"]["adapter_path"] == adapter["root"] and
                model["identity"]["model_identity"]["adapter"]["merged_adapters"] == [], "cold model identity differs")
        rows = load_jsonl(run / "records.jsonl")
        require(len(rows) == 2 and rows[0]["history"] == worker["on_diagonal_history"] and rows[0]["on_diagonal"] is True and
                rows[1]["history"] == worker["cross_history"] and rows[1]["on_diagonal"] is False, "cell order/gate differs")
        image = next(row for row in packet["images"] if row["image_id"] == worker["image_id"])
        all_records.extend(validate_cell(packet, row, worker, image, tokenizer, packet_sha) for row in rows)
        terminals.append(terminal)
    require(len(all_records) == 16 and sum(row["on_diagonal"] for row in all_records) == 8, "16-cell coverage differs")
    comparisons = []
    for image in packet["images"]:
        for history in ("stable50", "positive32"):
            rows = [row for row in all_records if row["image_id"] == image["image_id"] and row["history"] == history]
            require({row["checkpoint"] for row in rows} == {"stable50", "positive32"}, "same-history checkpoint pair differs")
            rows.sort(key=lambda row: row["checkpoint"])
            left, right = rows
            left_ids, right_ids = left["analysis"]["free_ids"], right["analysis"]["free_ids"]
            comparisons.append({"image_id": image["image_id"], "history": history,
                                "checkpoints": [left["checkpoint"], right["checkpoint"]],
                                "same_first_free_action": left_ids[0] == right_ids[0],
                                "same_free_action_ids": left_ids == right_ids,
                                "common_free_action_prefix_tokens": common_prefix_length(left_ids, right_ids),
                                "stop_reasons": [left["analysis"]["stop_reason"], right["analysis"]["stop_reason"]],
                                "first_free_actions": [left["analysis"]["first_free_action"], right["analysis"]["first_free_action"]]})
    reduction = {"schema": "checkpoint_history_cross.reduction.v1", "cells": 16,
                 "on_diagonal_exact": sum(row["exact_cached_natural_reproduction"] is True for row in all_records),
                 "same_history_checkpoint_comparisons": comparisons,
                 "claim_boundary": packet["claim_boundary"]}
    resources = {"schema": "checkpoint_history_cross.resources.v1", "model_loads": sum(row["model_loads"] for row in terminals),
                 "continuations": sum(row["continuations"] for row in terminals),
                 "image_forwards": sum(row["image_forwards"] for row in terminals),
                 "model_forwards": sum(row["model_forwards"] for row in terminals),
                 "generated_free_tokens": sum(row["generated_free_tokens"] for row in terminals),
                 "allocated_gpu_seconds": sum(row["elapsed_seconds"] for row in terminals), "workers": terminals}
    require(resources["model_loads"] == 8 and resources["continuations"] == resources["image_forwards"] == 16 and
            resources["generated_free_tokens"] == resources["model_forwards"] <= packet["limits"]["global_generated_tokens_max"],
            "global resource accounting differs")
    outputs = {"consumer.json": all_records, "reduction.json": reduction, "resources.json": resources}
    if verify:
        for name, value in outputs.items():
            require(load(out_root / name) == value, f"cold merged output differs: {name}")
    else:
        for name, value in outputs.items():
            publish(out_root / name, value)
    return {"status": "verified" if verify else "merged", "cells": 16, "on_diagonal_exact": 8,
            "consumer_sha256": file_hash(out_root / "consumer.json") if (out_root / "consumer.json").exists() else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--packet", type=Path, required=True)
    execute_parser.add_argument("--out-root", type=Path, required=True)
    execute_parser.add_argument("--worker", type=int, required=True)
    for command in ("merge", "verify"):
        value = sub.add_parser(command)
        value.add_argument("--packet", type=Path, required=True)
        value.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "execute":
        execute(args.packet, args.out_root, args.worker)
    else:
        print(json.dumps(merge(args.packet, args.out_root, verify=args.command == "verify"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
