"""Native fixed-prefix scoring and continuation runner for one admitted case."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback
from typing import Any


WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in os.sys.path:
    os.sys.path.insert(0, str(WORKTREE))

from probes.dora_owner_learning.route_access import checkpoint_config, score_logits  # noqa: E402
from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy  # noqa: E402


EOS = 151645
ROW_START = 151646
REF_END = 151647
BOX_START = 151648
ROW_END = 151649
CAP = 3084
WALL_SECONDS = 1500


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def publish(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def append_jsonl(stream: Any, value: Any) -> None:
    stream.write(canonical_bytes(value).decode())
    stream.flush()
    os.fsync(stream.fileno())


def validate_sources(source_files: dict[str, str]) -> None:
    require(isinstance(source_files, dict) and source_files, "missing source hash map")
    for name, expected in source_files.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file(), f"missing source: {name}")
        require(file_hash(path) == expected, f"source hash changed: {name}")


def split_rows(ids: list[int]) -> tuple[list[list[int]], list[int]]:
    rows: list[list[int]] = []
    index = 0
    while index < len(ids) and ids[index] != EOS:
        if ids[index] != ROW_START:
            return rows, ids[index:]
        try:
            end = ids.index(ROW_END, index) + 1
        except ValueError:
            return rows, ids[index:]
        rows.append(ids[index:end])
        index = end
    return rows, ids[index:]


def row_fields(ids: list[int]) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    require(ids[0] == ROW_START and ids[-1] == ROW_END and REF_END in ids and BOX_START in ids, "bad complete row")
    desc = tuple(ids[1:ids.index(REF_END)])
    coords = tuple(value - 151670 for value in ids[ids.index(BOX_START) + 1:-1])
    require(len(coords) == 4, "row coordinate arity")
    valid = 0 <= min(coords) and max(coords) <= 999 and coords[0] < coords[2] and coords[1] < coords[3]
    return desc, coords, valid


def iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def free_behavior(prefix_ids: list[int], free_ids: list[int], candidates: dict[str, Any], stop: str, *, image_width: int, image_height: int) -> dict[str, Any]:
    prefix_rows, prefix_tail = split_rows(prefix_ids)
    free_rows, free_tail = split_rows(free_ids)
    require(not prefix_tail, "prefix contains non-row tail")
    if stop == "im_end":
        require(free_ids and free_ids[-1] == EOS and free_tail == [EOS], "EOS stop/token mismatch")
        free_tail = []
    else:
        require(stop == "length" and EOS not in free_ids, "length stop/token mismatch")
    candidate_rows = {label: value["token_ids"] for label, value in candidates.items()}
    exact = []
    close_pixel_same_description = []
    close_bin_same_description = []
    invalid = 0
    strict_pixel_class_blind = 0
    strict_bin_class_blind = 0
    earlier = list(prefix_rows)
    for row in free_rows:
        exact_labels = [label for label, candidate in candidate_rows.items() if row == candidate]
        desc, coords, valid = row_fields(row)
        pixel_labels = []
        bin_labels = []
        if valid:
            pixels = coord_bins_to_pixel_xyxy(coords, image_width=image_width, image_height=image_height, field="free_row")
            for label, candidate in candidate_rows.items():
                cdesc, ccoords, cvalid = row_fields(candidate)
                if cvalid and desc == cdesc:
                    cpixels = coord_bins_to_pixel_xyxy(ccoords, image_width=image_width, image_height=image_height, field=f"candidate_{label}")
                    if iou_xyxy(pixels, cpixels) > 0.95:
                        pixel_labels.append(label)
                    if iou(coords, ccoords) > 0.95:
                        bin_labels.append(label)
            prior_fields = [row_fields(old) for old in earlier]
            if any(old_valid and iou_xyxy(pixels, coord_bins_to_pixel_xyxy(old_coords, image_width=image_width, image_height=image_height, field="prior_row")) > 0.95
                   for _, old_coords, old_valid in prior_fields):
                strict_pixel_class_blind += 1
            if any(old_valid and iou(coords, old_coords) > 0.95 for _, old_coords, old_valid in prior_fields):
                strict_bin_class_blind += 1
        else:
            invalid += 1
        exact.append(exact_labels)
        close_pixel_same_description.append(pixel_labels)
        close_bin_same_description.append(bin_labels)
        earlier.append(row)
    first = None
    if free_rows:
        desc, coords, valid = row_fields(free_rows[0])
        pixels = list(coord_bins_to_pixel_xyxy(coords, image_width=image_width, image_height=image_height, field="first_free_row")) if valid else None
        first = {"token_ids": free_rows[0], "description_token_ids": list(desc), "coord_bins": list(coords), "pixel_box_xyxy": pixels, "geometry_valid": valid, "exact_candidate_labels": exact[0], "native_pixel_iou_gt_0_95_candidate_labels_same_description": close_pixel_same_description[0], "auxiliary_coord_bin_iou_gt_0_95_candidate_labels_same_description": close_bin_same_description[0]}
    return {
        "complete_free_rows": len(free_rows),
        "first_complete_free_row": first,
        "exact_candidate_recurrences": {label: sum(label in labels for labels in exact) for label in candidates},
        "native_pixel_iou_gt_0_95_candidate_recurrences_same_description": {label: sum(label in labels for labels in close_pixel_same_description) for label in candidates},
        "native_pixel_class_blind_strict_repeat_rows_against_prefix_and_prior_free": strict_pixel_class_blind,
        "auxiliary_coord_bin_iou_gt_0_95_candidate_recurrences_same_description": {label: sum(label in labels for labels in close_bin_same_description) for label in candidates},
        "auxiliary_coord_bin_class_blind_strict_repeat_rows_against_prefix_and_prior_free": strict_bin_class_blind,
        "metric_definitions": {"candidate_incidence": "same-description IoU>0.95 against fixed A/B/C candidate rows in native projected pixel coordinates", "strict_repeat": "class-blind IoU>0.95 against any earlier valid prefix/free row in native projected pixel coordinates", "auxiliary_bin_iou": "coord-bin-space sensitivity diagnostic only; never the primary repeat or candidate incidence"},
        "geometry_invalid_complete_free_rows": invalid,
        "raw_unparsed_tail_ids": free_tail,
        "raw_unparsed_tail_token_count": len(free_tail),
        "stop_reason": stop,
        "eos": stop == "im_end",
        "horizon": stop == "length",
    }


def validate_packet(packet: dict[str, Any], case_id: str) -> dict[str, Any]:
    require(packet.get("schema") == "repeat_multiplicity.v1" and packet.get("status") == "frozen_root_admitted", "packet is not frozen/final")
    validate_sources(packet["source_files"])
    require(case_id in packet["admitted_case_ids"], f"case is not admitted: {case_id}")
    matches = [case for case in packet["cases"] if case["case_id"] == case_id]
    require(len(matches) == 1 and matches[0]["admission_status"] == "admitted_by_root", "case admission mismatch")
    case = matches[0]
    require(len(case["cells"]) == 6 and len({cell["cell_id"] for cell in case["cells"]}) == 6, "cell family mismatch")
    require(len(case["candidates"]["A"]["token_ids"]) == len(case["candidates"]["B"]["token_ids"]), "A/B candidate length differs")
    lengths = {len(cell["prefix_ids"]) for cell in case["cells"]}
    suffixes = {tuple(cell["prefix_row_labels"][-3:]) for cell in case["cells"]}
    require(len(lengths) == 1 and suffixes == {("A", "B", "C")}, "prefix length/suffix mismatch")
    for cell in case["cells"]:
        rebuilt = [token for label in cell["prefix_row_labels"] for token in case["candidates"][label]["token_ids"]]
        require(rebuilt == cell["prefix_ids"] and len(rebuilt) + cell["budget"] <= CAP and cell["budget"] <= 512, f"{cell['cell_id']}: prefix/budget mismatch")
    return case


def effective_config(packet: dict[str, Any]) -> Any:
    from src.config.inference import InferConfig
    config = InferConfig.model_validate(packet["config"])
    require(config.adapter is not None and config.embedding_delta is not None, "Source policy components missing")
    require(config.backend.type == "hf" and config.model.dtype == "fp32", "requires HF FP32")
    require(config.backend.hf.attn_implementation == "sdpa" and config.backend.hf.patch_embed_linearization == "enabled", "requires SDPA patch linearization")
    return checkpoint_config(config, Path(packet["anchor_adapter"]))


def run(packet_path: Path, out_dir: Path, case_id: str) -> dict[str, Any]:
    require(packet_path.is_absolute() and packet_path.is_file(), "--packet must be an existing absolute path")
    require(out_dir.is_absolute() and not out_dir.exists(), "--out-dir must be a new absolute path")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "CUDA_VISIBLE_DEVICES must be explicit")
    visible = [part for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if part.strip()]
    require(len(visible) == 1, "exactly one physical GPU must be visible")
    packet = json.loads(packet_path.read_text())
    case = validate_packet(packet, case_id)
    config = effective_config(packet)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()
    packet_sha = file_hash(packet_path)
    terminal = {"schema": "repeat_multiplicity.terminal.v1", "status": "running", "case_id": case_id, "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "packet_sha256": packet_sha, "model_loads": 0, "generation_calls": 0, "score_replays": 0, "model_forwards": 0, "image_forwards": 0, "generated_free_tokens": 0, "started_unix": time.time()}
    publish(out_dir / "launch.json", {"schema": "repeat_multiplicity.launch.v1", "pid": os.getpid(), "packet": str(packet_path), "packet_sha256": packet_sha, "case_id": case_id, "cells": [cell["cell_id"] for cell in case["cells"]]})
    publish(out_dir / "config.json", {"schema": "repeat_multiplicity.config.v1", "source_config": packet["config"], "effective_config": config.model_dump(mode="json"), "policy": packet["policy"]})
    start = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    handles = []
    torch = None
    records = []
    error = None
    counters = {"model": 0, "image": 0}
    process_cuda_peaks = {"allocated": 0, "reserved": 0}
    try:
        def expire(_sig: int, _frame: Any) -> None:
            raise TimeoutError(f"hard wall exceeded {WALL_SECONDS} seconds")
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(WALL_SECONDS)
        import torch as torch_module
        torch = torch_module
        from probes.dora_owner_learning.runtime import load_policy
        from probes.source_rweak_row_cross.run import build_requests, native_record
        from src.qwen.generation import NativeGenerationPolicy, generate_continuations
        from src.qwen.native import prepare_native_inputs, prepare_replay
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "runner requires one visible CUDA device")
        load_tick = time.monotonic()
        qwen, model_identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["load_seconds"] = time.monotonic() - load_tick
        terminal["model_loads"] = 1
        require(model_identity["effective_settings"]["observed_attn_implementation"] == "sdpa", "live attention is not SDPA")
        require(model_identity["effective_settings"]["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"], "live parameters are not FP32")
        require(model_identity["model_identity"]["adapter"].get("adapter_path") == packet["anchor_adapter"], "live adapter mismatch")
        publish(out_dir / "model.json", {"schema": "repeat_multiplicity.model.v1", "identity": model_identity, "eos_token_id": EOS, "pad_token_id": qwen.tokenizer.pad_token_id})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "ambiguous visual module")
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: counters.__setitem__("model", counters["model"] + 1)))
        handles.append(visuals[0].register_forward_pre_hook(lambda *_: counters.__setitem__("image", counters["image"] + 1)))
        requests, _ = build_requests(qwen, packet["config"], [case["source_case"]])
        batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True)
        require(len(batch.request_ids) == 1 and list(batch.prompt_token_ids[0]) == case["prompt_token_ids"], "live prompt association mismatch")
        require(tuple(batch.image_grids[0]) == tuple(case["source_case"]["image_plan"]["observed_image_grid_thw"]), "live image grid mismatch")
        publish(out_dir / "batch.json", {"schema": "repeat_multiplicity.batch.v1", "request_id": batch.request_ids[0], "prompt_token_ids": list(batch.prompt_token_ids[0]), "image_grid_thw": list(batch.image_grids[0]), "image_path": case["source_case"]["image_path"]})
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, repetition_penalty=1.0, top_k=0, use_model_defaults=False)
        def accumulate_cuda_peaks() -> None:
            process_cuda_peaks["allocated"] = max(process_cuda_peaks["allocated"], torch.cuda.max_memory_allocated(torch.device("cuda:0")))
            process_cuda_peaks["reserved"] = max(process_cuda_peaks["reserved"], torch.cuda.max_memory_reserved(torch.device("cuda:0")))
        accumulate_cuda_peaks()
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        with (out_dir / "records.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
            before_model, before_image = counters["model"], counters["image"]
            natural = generate_continuations(qwen.model, batch, extensions=[[]], budgets=[CAP], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
            natural_ids = list(natural.token_ids)
            require(natural_ids == case["baseline_action_ids"], "fresh natural anchor differs from saved Stable50")
            natural_text = qwen.tokenizer.decode(natural_ids, skip_special_tokens=False)
            natural_record = native_record(natural_text, {"row_id": case["source_row_id"]}, case["golden"], natural.stop_reason)
            row = {"schema": "repeat_multiplicity.record.v1", "kind": "natural_anchor", "case_id": case_id, "packet_sha256": packet_sha, "action_ids": natural_ids, "text": natural_text, "stop_reason": natural.stop_reason, "exact_saved_anchor_match": True, "parsed": natural_record, "model_forwards": counters["model"] - before_model, "image_forwards": counters["image"] - before_image}
            append_jsonl(stream, row); records.append(row)
            terminal["generation_calls"] += 1; terminal["generated_free_tokens"] += len(natural_ids)
            for cell in case["cells"]:
                prefix = cell["prefix_ids"]
                scores = {}
                score_before_allocated = torch.cuda.memory_allocated(torch.device("cuda:0"))
                accumulate_cuda_peaks()
                torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
                for label in cell["score_candidates"]:
                    candidate = [EOS] if label == "EOS" else case["candidates"][label]["token_ids"]
                    continuation = prefix + candidate
                    before_model, before_image = counters["model"], counters["image"]
                    replay = prepare_replay(qwen.model, batch.inputs, prompt_token_ids=case["prompt_token_ids"], continuation_token_ids=continuation)
                    logits = replay.aligned_logits(qwen.model(**replay.inputs).logits)[-len(candidate):]
                    targets = replay.target_ids[-len(candidate):]
                    scored = score_logits(logits, targets, prompt_length=len(case["prompt_token_ids"]) + len(prefix))
                    require(scored["length"] == len(candidate) and [p["target_id"] for p in scored["positions"]] == candidate, "candidate score alignment")
                    scores[label] = scored
                    terminal["score_replays"] += 1
                    require(counters["model"] - before_model == 1 and counters["image"] - before_image == 1, "score replay forward accounting")
                    del logits, replay
                scoring_peak = torch.cuda.max_memory_allocated(torch.device("cuda:0"))
                scoring_peak_reserved = torch.cuda.max_memory_reserved(torch.device("cuda:0"))
                score_summary = {label: {"token_count": value["length"], "sum_logprob": value["sum_logprob"], "mean_logprob": value["mean_logprob"], "first_non_argmax": value["first_non_argmax"], "positions": value["positions"]} for label, value in scores.items()}
                score_summary["A_minus_B"] = {"sum_log_odds": scores["A"]["sum_logprob"] - scores["B"]["sum_logprob"], "mean_logprob_difference": scores["A"]["mean_logprob"] - scores["B"]["mean_logprob"], "equal_serialized_length": scores["A"]["length"] == scores["B"]["length"]}
                before_model, before_image = counters["model"], counters["image"]
                result = generate_continuations(qwen.model, batch, extensions=[prefix], budgets=[cell["budget"]], eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                free_ids = list(result.token_ids)
                require(len(free_ids) <= cell["budget"], "free budget accounting")
                require((result.stop_reason == "length" and len(free_ids) == cell["budget"] and EOS not in free_ids) or
                        (result.stop_reason == "im_end" and bool(free_ids) and free_ids[-1] == EOS and EOS not in free_ids[:-1]),
                        "native stop/token accounting")
                action = prefix + free_ids
                text = qwen.tokenizer.decode(action, skip_special_tokens=False)
                free_text = qwen.tokenizer.decode(free_ids, skip_special_tokens=False)
                parsed_full = native_record(text, {"row_id": case["source_row_id"]}, case["golden"], result.stop_reason)
                parsed_free = native_record(free_text, {"row_id": case["source_row_id"]}, case["golden"], result.stop_reason)
                behavior = free_behavior(prefix, free_ids, case["candidates"], result.stop_reason, image_width=case["source_case"]["image_width"], image_height=case["source_case"]["image_height"])
                row = {"schema": "repeat_multiplicity.record.v2", "kind": "conditional", "case_id": case_id, "cell_id": cell["cell_id"], "packet_sha256": packet_sha, "prefix_row_labels": cell["prefix_row_labels"], "prefix_ids": prefix, "free_ids": free_ids, "action_ids": action, "budget": cell["budget"], "text": text, "free_text": free_text, "stop_reason": result.stop_reason, "scores": score_summary, "free_behavior": behavior, "parsed_full": parsed_full, "parsed_free": parsed_free, "model_forwards": counters["model"] - before_model, "image_forwards": counters["image"] - before_image, "scoring_peak_cuda_allocated_bytes": scoring_peak, "scoring_peak_cuda_reserved_bytes": scoring_peak_reserved, "scoring_pre_cuda_allocated_bytes": score_before_allocated}
                append_jsonl(stream, row); records.append(row)
                terminal["generation_calls"] += 1; terminal["generated_free_tokens"] += len(free_ids)
        require(len(records) == 7 and terminal["generation_calls"] == 7 and terminal["score_replays"] == 24, "finite case panel accounting")
        durable = [json.loads(line) for line in (out_dir / "records.jsonl").read_text().splitlines() if line.strip()]
        require(durable == records, "cold JSONL readback differs")
        require(durable[0]["kind"] == "natural_anchor" and [row["cell_id"] for row in durable[1:]] == [cell["cell_id"] for cell in case["cells"]], "cold cell order mismatch")
        readback = {"schema": "repeat_multiplicity.readback.v1", "case_id": case_id, "records": len(durable), "natural_anchors": 1, "conditional_cells": 6, "score_replays": 24, "all_A_B_lengths_equal": all(row["scores"]["A_minus_B"]["equal_serialized_length"] for row in durable[1:]), "all_prefixes_exact": all(row["prefix_ids"] == cell["prefix_ids"] for row, cell in zip(durable[1:], case["cells"]))}
        publish(out_dir / "readback.json", readback)
        terminal["status"] = "complete"
        terminal["exit_code"] = 0
        terminal["readback"] = readback
    except BaseException as exc:
        error = exc
        terminal["status"] = "failed"
        terminal["exit_code"] = 1
        terminal["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            handle.remove()
        terminal["elapsed_seconds"] = time.monotonic() - start
        terminal["model_forwards"] = counters["model"]
        terminal["image_forwards"] = counters["image"]
        terminal["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and torch.cuda.is_available():
            process_cuda_peaks["allocated"] = max(process_cuda_peaks["allocated"], torch.cuda.max_memory_allocated(torch.device("cuda:0")))
            process_cuda_peaks["reserved"] = max(process_cuda_peaks["reserved"], torch.cuda.max_memory_reserved(torch.device("cuda:0")))
            terminal["peak_cuda_allocated_bytes"] = process_cuda_peaks["allocated"]
            terminal["peak_cuda_reserved_bytes"] = process_cuda_peaks["reserved"]
        publish(out_dir / "terminal.json", terminal)
    if error is not None:
        raise error
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = run(args.packet.resolve(), args.out_dir.resolve(), args.case_id)
    print(json.dumps({key: result[key] for key in ("status", "case_id", "generation_calls", "score_replays", "model_forwards", "image_forwards", "generated_free_tokens", "elapsed_seconds", "peak_cuda_allocated_bytes", "peak_rss_bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
