"""Eight-cell free-history continuation diagnostic for frozen N16/A endpoints.

CPU validation is always available. GPU execution is fail-closed behind an
exact root grant and writes raw generated IDs before parsing or interpretation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.candidate_opportunity import file_hash, require
from probes.dora_owner_learning.route_access import checkpoint_config
from probes.owner_successor_scale.history import _check_model_identity
from probes.parallel_owner_research.history import complete_rows, continuation_ledger
from probes.source_rweak_row_cross.run import build_requests, native_record
from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy


REPO = Path("/data/CoordExp/.worktrees/research-probes")
TRAIN_RAW = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"
)
CAP = 3084
EOS = 151645
PACKET_SCHEMA = "label_vs_compilation.free_h_execution_packet.v1"
GRANT_SCHEMA = "label_vs_compilation.free_h_root_grant.v1"


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _binding(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    require(path.is_file(), f"missing bound file: {path}")
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def _publish(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(path, value)


def _verify_binding(value: Mapping[str, Any], label: str) -> None:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{label} binding fields")
    require(_binding(value["path"]) == dict(value), f"{label} binding changed")


def _without_eos(values: Sequence[int]) -> list[int]:
    result = [int(value) for value in values]
    require(result, "empty token sequence")
    if result[-1] == EOS:
        result.pop()
    require(EOS not in result, "EOS inside token sequence")
    return result


def _complete_literal(values: Sequence[int], label: str) -> list[list[int]]:
    try:
        return complete_rows(_without_eos(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def validate_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = copy.deepcopy(dict(value))
    require(packet.get("schema") == PACKET_SCHEMA, "execution packet schema")
    require(packet.get("status") == "sealed_root_launch_grant_pending", "execution packet status")
    for label, source in packet["sources"].items():
        _verify_binding(source, f"source {label}")
    require(packet["sources"]["runner"] == _binding(Path(__file__).resolve()), "live runner identity")
    require(packet["sources"]["materialization_train_raw"]["path"] == str(TRAIN_RAW), "materialization source path")
    runtime = packet["runtime"]
    require(
        runtime == {
            "backend": "hf_native",
            "dtype": "fp32",
            "attention_implementation": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "eos_token_id": EOS,
            "total_assistant_token_cap_including_supplied_h": CAP,
            "strict_repeat": "any_class_iou_strictly_greater_than_0.95",
            "unknown_policy": "neutral",
            "forced_h_credit": False,
        },
        "runtime semantics changed",
    )
    require(packet["tmux"]["session"] == "label-vs-compilation-diag-v1", "tmux identity")
    cells = packet["cells"]
    require(len(cells) == 8, "exact eight diagnostic cells")
    require([cell["cell_id"] for cell in cells] == list(range(8)), "cell ordering")
    require([cell["physical_gpu"] for cell in cells] == list(range(8)), "one cell per physical GPU")
    require(len({(cell["arm"], cell["image_id"]) for cell in cells}) == 8, "endpoint/case identity")
    require({cell["arm"] for cell in cells} == {"N16", "A"}, "endpoint identities")
    require({cell["image_id"] for cell in cells} == {477415, 351017, 417044, 388795}, "case identities")
    for cell in cells:
        require(cell["adapter"] == packet["models"][cell["arm"]], f"cell {cell['cell_id']} adapter")
        h_ids = cell["h"]["token_ids"]
        c_ids = cell["c"]["token_ids"]
        w_ids = cell["w"]["token_ids"]
        require(_digest(h_ids) == cell["h"]["token_ids_sha256"], "h digest")
        require(_digest(c_ids) == cell["c"]["token_ids_sha256"], "c digest")
        require(_digest(w_ids) == cell["w"]["token_ids_sha256"], "w digest")
        require(len(_complete_literal(h_ids, "h")) == cell["h"]["complete_rows"], "h complete rows")
        require(len(_complete_literal(c_ids, "c")) == len(_complete_literal(w_ids, "w")) == 1, "c/w row grammar")
        require(cell["prefix_token_count"] == len(h_ids), "prefix count")
        require(cell["remaining_budget"] == CAP - len(h_ids) > 0, "remaining budget")
        require(cell["scored_c"]["argmax_target_tokens"] == len(c_ids), "c not all-token argmax")
        require(cell["scientific_call"] == "one_complete_free_h_continuation", "cell call semantics")
        require(cell["supplied_credit"] == {"h": False, "c": True, "free_suffix": True}, "credit semantics")
    bounds = packet["bounds"]
    require(bounds["complete_continuations"] == 8, "continuation bound")
    require(bounds["model_loads"] == 8 and bounds["image_forwards"] == 8, "load/image bounds")
    require(bounds["max_new_tokens"] == sum(cell["remaining_budget"] for cell in cells), "token bound")
    require(bounds["fallback_h_plus_c_calls"] == 0, "fallback is not authorized")
    return packet


def validate_grant(packet_path: str | Path, grant_path: str | Path, output: str | Path) -> dict[str, Any]:
    grant = _read(grant_path)
    require(grant.get("schema") == GRANT_SCHEMA and grant.get("status") == "granted", "root grant")
    require(grant.get("packet") == _binding(packet_path), "grant packet binding")
    require(grant.get("runner") == _binding(Path(__file__).resolve()), "grant runner binding")
    require(grant.get("output_root") == str(Path(output)), "grant output identity")
    require(grant.get("tmux_session") == "label-vs-compilation-diag-v1", "grant tmux identity")
    require(grant.get("authorized_cell_ids") == list(range(8)), "grant cell scope")
    require(grant.get("authorized_complete_continuations") == 8, "grant continuation scope")
    require(grant.get("fallback_h_plus_c_calls") == 0, "grant cannot authorize implicit fallback")
    return grant


def _runtime_config(config_source: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config_source))
    value["data"] = copy.deepcopy(value["data"])
    value["data"]["input_jsonl"] = str(TRAIN_RAW)
    left, right = copy.deepcopy(value), copy.deepcopy(dict(config_source))
    left.pop("data", None)
    right.pop("data", None)
    require(left == right, "runtime config changed non-data inference semantics")
    return value


def _cell(packet: Mapping[str, Any], cell_id: int) -> dict[str, Any]:
    values = [cell for cell in packet["cells"] if int(cell["cell_id"]) == cell_id]
    require(len(values) == 1, "cell lookup")
    return values[0]


def _subsequence_positions(values: Sequence[int], target: Sequence[int]) -> list[int]:
    target = list(target)
    if not target:
        return []
    return [index for index in range(len(values) - len(target) + 1) if list(values[index:index + len(target)]) == target]


def _geometry(rows: Sequence[Mapping[str, Any]], target: Mapping[str, Any]) -> dict[str, Any]:
    matches = []
    for index, row in enumerate(rows):
        overlap = float(iou_xyxy(row["bbox"], target["bbox_xyxy_pixels"]))
        matches.append({
            "pred_index": index,
            "description": row["description"],
            "same_class": row["description"] == target["description"],
            "iou": overlap,
        })
    matches.sort(key=lambda item: (-item["iou"], item["pred_index"]))
    same = [item for item in matches if item["same_class"]]
    return {
        "best_any_class": matches[0] if matches else None,
        "best_same_class": same[0] if same else None,
        "any_class_iou_gt_0_5": bool(matches and matches[0]["iou"] > 0.5),
        "same_class_iou_gt_0_5": bool(same and same[0]["iou"] > 0.5),
        "strict_repeat_iou_gt_0_95": bool(matches and matches[0]["iou"] > 0.95),
    }


def _assessment(cell: Mapping[str, Any], free_ids: Sequence[int], ledger: Mapping[str, Any]) -> dict[str, Any]:
    predictions = ledger["free_parsed"]["pred"]
    c_ids = cell["c"]["token_ids"]
    w_ids = cell["w"]["token_ids"]
    literal_c = list(free_ids[:len(c_ids)]) == list(c_ids)
    observed_owners = [str(owner) for owner in ledger["free_score"]["50"]["owners"]]
    required_owners = [str(item["owner_id"]) for item in cell["annotation_owner_obligations"]]
    return {
        "literal_c_prefix_parity": literal_c,
        "scientific_interpretation": (
            "eligible_after_free_c_parity" if literal_c
            else "STOP_cell_free_c_disagrees_with_exact_sequential_argmax"
        ),
        "literal_c_occurrences": _subsequence_positions(free_ids, c_ids),
        "literal_w_occurrences": _subsequence_positions(free_ids, w_ids),
        "c_geometry": _geometry(predictions, cell["c"]["literal_target"]),
        "w_geometry": _geometry(predictions, cell["w"]["literal_target"]),
        "annotation_owner_obligations": {
            "required": required_owners,
            "observed_free_tp50": observed_owners,
            "retained": sorted(set(required_owners) & set(observed_owners)),
            "missing": sorted(set(required_owners) - set(observed_owners)),
            "gained_beyond_obligations": sorted(set(observed_owners) - set(required_owners)),
            "qualification": "GT-relative annotation owners only; unknown remains neutral and reviewed physical c/w are separate.",
        },
    }


def worker(packet_path: Path, grant_path: Path, output: Path, cell_id: int, physical_gpu: int) -> None:
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    packet = validate_packet(_read(packet_path))
    validate_grant(packet_path, grant_path, packet["tmux"]["output_root"])
    cell = _cell(packet, cell_id)
    require(physical_gpu == cell["physical_gpu"], "physical GPU assignment")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "visible physical GPU")
    require(torch.cuda.device_count() == 1, "worker requires one visible GPU")
    require(not output.exists(), "cell output collision")
    output.mkdir(parents=True)
    started = time.monotonic()
    terminal: dict[str, Any] = {
        "schema": "label_vs_compilation.free_h_cell_terminal.v1",
        "status": "running",
        "cell_id": cell_id,
        "arm": cell["arm"],
        "image_id": cell["image_id"],
        "physical_gpu": physical_gpu,
        "model_loads": 0,
        "continuations": 0,
        "model_forwards": 0,
        "image_forwards": 0,
    }
    _publish(output / "launch.json", {**terminal, "packet": _binding(packet_path), "grant": _binding(grant_path)})
    handles: list[Any] = []
    try:
        paired = _read(packet["sources"]["paired_packet"]["path"])
        frozen = next(row for row in paired["records"] if int(row["image_id"]) == int(cell["image_id"]))
        config_source = _runtime_config(paired["config"])
        config = checkpoint_config(InferConfig.model_validate(config_source), cell["adapter"]["root"])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        _check_model_identity(identity, cell["arm"], cell["adapter"])
        terminal["model_loads"] = 1
        _publish(output / "model.json", {"identity": identity, "adapter": cell["adapter"], "cell_id": cell_id})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)

        def model_forward(*_: Any) -> None:
            terminal["model_forwards"] += 1

        def image_forward(*_: Any) -> None:
            terminal["image_forwards"] += 1

        handles.append(qwen.model.register_forward_pre_hook(model_forward))
        visual = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visual) == 1, "single native visual module")
        handles.append(visual[0].register_forward_pre_hook(image_forward))
        requests, _ = build_requests(qwen, config_source, [frozen["case"]])
        batch = prepare_native_inputs(
            qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True
        )
        require(list(batch.prompt_token_ids[0]) == cell["prompt_token_ids"], "prompt identity")
        require(batch.media_sha256[0] == cell["executed_media_sha256"], "media identity")
        require(list(batch.image_grids[0]) == cell["observed_image_grid_thw"], "image grid identity")
        prefix = list(cell["h"]["token_ids"])
        prefix_text = qwen.tokenizer.decode(prefix, skip_special_tokens=False)
        require(qwen.tokenizer.encode(prefix_text, add_special_tokens=False) == prefix, "h tokenizer roundtrip")
        policy = NativeGenerationPolicy(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_model_defaults=False,
        )
        with torch.inference_mode():
            generated = generate_continuations(
                qwen.model,
                batch,
                extensions=[prefix],
                budgets=[cell["remaining_budget"]],
                eos_token_id=EOS,
                pad_token_id=qwen.tokenizer.pad_token_id,
                policy=policy,
                trace="none",
            )[0]
        free_ids = [int(value) for value in generated.token_ids]
        require(free_ids and len(prefix) + len(free_ids) <= CAP, "continuation budget")
        require(
            (generated.stop_reason == "im_end" and free_ids[-1] == EOS and EOS not in free_ids[:-1])
            or (
                generated.stop_reason == "length"
                and len(free_ids) == cell["remaining_budget"]
                and EOS not in free_ids
            ),
            "native terminal identity",
        )
        raw = {
            "schema": "label_vs_compilation.free_h_raw_generation.v1",
            "packet": _binding(packet_path),
            "grant": _binding(grant_path),
            "cell_id": cell_id,
            "arm": cell["arm"],
            "image_id": cell["image_id"],
            "physical_gpu": physical_gpu,
            "prompt_token_ids_sha256": _digest(cell["prompt_token_ids"]),
            "executed_media_sha256": batch.media_sha256[0],
            "h_token_ids": prefix,
            "h_token_ids_sha256": _digest(prefix),
            "free_token_ids": free_ids,
            "free_token_ids_sha256": _digest(free_ids),
            "remaining_budget": cell["remaining_budget"],
            "stop_reason": generated.stop_reason,
        }
        # Durability boundary: literal generated IDs are committed before decode/parser work.
        _publish(output / "raw-generation.json", raw)
        free_text = qwen.tokenizer.decode(free_ids, skip_special_tokens=False)
        ledger = continuation_ledger(
            prefix_text,
            free_text,
            {"case": frozen["case"], "golden": frozen["golden"]},
            len(prefix),
            len(free_ids),
            generated.stop_reason,
        )
        row = {
            "schema": "label_vs_compilation.free_h_cell_row.v1",
            "packet": _binding(packet_path),
            "grant": _binding(grant_path),
            "raw_generation": _binding(output / "raw-generation.json"),
            "cell_id": cell_id,
            "arm": cell["arm"],
            "image_id": cell["image_id"],
            "prefix_token_count": len(prefix),
            "remaining_budget": cell["remaining_budget"],
            "free_token_ids_sha256": _digest(free_ids),
            "free_text": free_text,
            "stop_reason": generated.stop_reason,
            "free_score": ledger["free_score"],
            "full_score": ledger["full_score"],
            "full_overlap_counts": ledger["full_overlap_counts"],
            "burden": ledger["burden"],
            "assessment": _assessment(cell, free_ids, ledger),
            "supplied_h_earns_no_credit": True,
        }
        _publish(output / "row.json", row)
        require(terminal["image_forwards"] == 1, "exact one image forward")
        require(terminal["model_forwards"] <= cell["remaining_budget"] + 1, "model forward bound")
        terminal.update(status="completed", exit_code=0, continuations=1)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        bounds = packet["bounds"]
        if terminal["status"] == "completed" and not (
            terminal["elapsed_seconds"] <= bounds["max_rank_seconds"]
            and terminal["peak_cuda_allocated_bytes"] <= bounds["max_cuda_allocated_bytes_per_cell"]
            and terminal["peak_cuda_reserved_bytes"] <= bounds["max_cuda_reserved_bytes_per_cell"]
            and terminal["peak_rss_bytes"] <= bounds["max_rss_bytes_per_cell"]
        ):
            terminal.update(status="failed", exit_code=1, error="cell resource bound exceeded")
        _publish(output / "terminal.json", terminal)
    require(terminal["status"] == "completed", "cell terminal completion")


def consume(packet_path: Path, grant_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    packet = validate_packet(_read(packet_path))
    validate_grant(packet_path, grant_path, output)
    paired = _read(packet["sources"]["paired_packet"]["path"])
    by_image = {int(row["image_id"]): row for row in paired["records"]}
    tokenizer = AutoTokenizer.from_pretrained(packet["models"]["base_model_path"], local_files_only=True)
    rows = []
    parity_stops = []
    totals = {"model_forwards": 0, "image_forwards": 0, "elapsed_seconds": 0.0}
    for cell in packet["cells"]:
        root = output / f"cell-{cell['cell_id']:02d}-{cell['arm'].lower()}-{cell['image_id']}"
        terminal = _read(root / "terminal.json")
        require(
            terminal["status"] == "completed"
            and terminal["cell_id"] == cell["cell_id"]
            and terminal["arm"] == cell["arm"]
            and terminal["image_id"] == cell["image_id"]
            and terminal["physical_gpu"] == cell["physical_gpu"]
            and terminal["model_loads"] == 1
            and terminal["continuations"] == 1,
            f"cell {cell['cell_id']} terminal",
        )
        raw = _read(root / "raw-generation.json")
        row = _read(root / "row.json")
        require(raw["packet"] == row["packet"] == _binding(packet_path), "row packet identity")
        require(raw["grant"] == row["grant"] == _binding(grant_path), "row grant identity")
        require(row["raw_generation"] == _binding(root / "raw-generation.json"), "raw binding")
        require(raw["cell_id"] == row["cell_id"] == cell["cell_id"], "row cell identity")
        require(raw["h_token_ids"] == cell["h"]["token_ids"], "row h identity")
        require(raw["h_token_ids_sha256"] == _digest(raw["h_token_ids"]), "row h digest")
        require(raw["free_token_ids_sha256"] == row["free_token_ids_sha256"] == _digest(raw["free_token_ids"]), "free digest")
        require(raw["remaining_budget"] == row["remaining_budget"] == cell["remaining_budget"], "row budget")
        free_ids = raw["free_token_ids"]
        require(len(free_ids) <= cell["remaining_budget"], "free token bound")
        free_text = tokenizer.decode(free_ids, skip_special_tokens=False)
        require(free_text == row["free_text"], "token/text identity")
        prefix_text = tokenizer.decode(cell["h"]["token_ids"], skip_special_tokens=False)
        frozen = by_image[cell["image_id"]]
        ledger = continuation_ledger(
            prefix_text,
            free_text,
            {"case": frozen["case"], "golden": frozen["golden"]},
            cell["prefix_token_count"],
            len(free_ids),
            raw["stop_reason"],
        )
        require(row["free_score"] == ledger["free_score"], "free score recompute")
        require(row["full_score"] == ledger["full_score"], "full score recompute")
        require(row["full_overlap_counts"] == ledger["full_overlap_counts"], "overlap recompute")
        require(row["burden"] == ledger["burden"], "burden recompute")
        assessment = _assessment(cell, free_ids, ledger)
        require(row["assessment"] == assessment, "assessment recompute")
        if not assessment["literal_c_prefix_parity"]:
            parity_stops.append(cell["cell_id"])
        rows.append({
            "cell_id": cell["cell_id"],
            "arm": cell["arm"],
            "image_id": cell["image_id"],
            "row": _binding(root / "row.json"),
            "terminal": _binding(root / "terminal.json"),
            "natural_root_context": cell["natural_root_context"],
            "assessment": assessment,
            "free_score": ledger["free_score"],
            "burden": ledger["burden"],
        })
        for key in totals:
            totals[key] += terminal[key]
    require(len(rows) == 8, "exact consumer denominator")
    result = {
        "schema": "label_vs_compilation.free_h_result.v1",
        "status": (
            "completed_all_cells_free_c_parity_eligible"
            if not parity_stops else "completed_with_cell_free_c_parity_stops"
        ),
        "packet": _binding(packet_path),
        "grant": _binding(grant_path),
        "consumer": _binding(Path(__file__).resolve()),
        "counts": {
            "cells": 8,
            "models": 2,
            "images": 4,
            "complete_continuations": 8,
            "free_c_parity_pass": 8 - len(parity_stops),
            "free_c_parity_stop": len(parity_stops),
        },
        "parity_stop_cell_ids": parity_stops,
        "rows": rows,
        "resources": {
            **totals,
            "gpu_hours_sum": totals["elapsed_seconds"] / 3600.0,
            "qualification": "sum of rank wall time, not measured utilization",
        },
        "interpretation_boundary": packet["interpretation_boundary"],
    }
    _publish(output / "result.json", result)
    return result


def launch(packet_path: Path, grant_path: Path, output: Path) -> None:
    packet = validate_packet(_read(packet_path))
    require(str(output) == packet["tmux"]["output_root"], "packet output root")
    grant = validate_grant(packet_path, grant_path, output)
    require(os.environ.get("TMUX"), "launch must run inside tmux")
    session = subprocess.run(
        ["tmux", "display-message", "-p", "#S"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    require(session == packet["tmux"]["session"], "wrong tmux session")
    require(not output.exists(), "execution output collision")
    output.mkdir(parents=True)
    started = time.monotonic()
    launches = []
    processes: list[tuple[subprocess.Popen[Any], Any]] = []
    terminal: dict[str, Any] = {
        "schema": "label_vs_compilation.free_h_launch_terminal.v1",
        "status": "running",
        "packet": _binding(packet_path),
        "grant": _binding(grant_path),
        "tmux_session": packet["tmux"]["session"],
        "complete_continuations": 0,
    }
    _publish(output / "launch.json", terminal)
    try:
        for cell in packet["cells"]:
            cell_output = output / f"cell-{cell['cell_id']:02d}-{cell['arm'].lower()}-{cell['image_id']}"
            log_path = output / f"cell-{cell['cell_id']:02d}.log"
            log = log_path.open("x")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--packet", str(packet_path),
                "--grant", str(grant_path),
                "--output", str(cell_output),
                "--cell-id", str(cell["cell_id"]),
                "--physical-gpu", str(cell["physical_gpu"]),
            ]
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env={
                    **os.environ,
                    "PYTHONPATH": str(REPO),
                    "CUDA_VISIBLE_DEVICES": str(cell["physical_gpu"]),
                    "TOKENIZERS_PARALLELISM": "false",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            launches.append({
                "cell_id": cell["cell_id"],
                "arm": cell["arm"],
                "image_id": cell["image_id"],
                "physical_gpu": cell["physical_gpu"],
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            })
            processes.append((process, log))
        _publish(output / "process-launch.json", {"launches": launches})
        exits = []
        for item, (process, log) in zip(launches, processes, strict=True):
            exits.append({**item, "exit_code": process.wait()})
            log.close()
        _publish(output / "process-exits.json", {"results": exits})
        require(all(item["exit_code"] == 0 for item in exits), "cell failure; preserve outputs, no retry")
        result = consume(packet_path, grant_path, output)
        terminal.update(
            status="completed_consumed",
            exit_code=0,
            complete_continuations=result["counts"]["complete_continuations"],
            result=_binding(output / "result.json"),
        )
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for process, log in processes:
            if not log.closed and process.poll() is not None:
                log.close()
        terminal["elapsed_seconds"] = time.monotonic() - started
        _publish(output / "terminal.json", terminal)
    require(terminal["status"] == "completed_consumed", "launch terminal completion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="CPU-only exact packet validation")
    verify.add_argument("--packet", type=Path, required=True)
    worker_parser = sub.add_parser("worker", help="Run one root-granted single-GPU cell")
    worker_parser.add_argument("--packet", type=Path, required=True)
    worker_parser.add_argument("--grant", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--cell-id", type=int, required=True)
    worker_parser.add_argument("--physical-gpu", type=int, required=True)
    consume_parser = sub.add_parser("consume", help="CPU exact recomputation of eight completed cells")
    consume_parser.add_argument("--packet", type=Path, required=True)
    consume_parser.add_argument("--grant", type=Path, required=True)
    consume_parser.add_argument("--output", type=Path, required=True)
    launch_parser = sub.add_parser("launch", help="Root-granted eight-GPU launcher; must run in tmux")
    launch_parser.add_argument("--packet", type=Path, required=True)
    launch_parser.add_argument("--grant", type=Path, required=True)
    launch_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        packet = validate_packet(_read(args.packet))
        print(json.dumps({
            "status": "cpu_valid_root_launch_grant_pending",
            "packet": str(args.packet),
            "cells": len(packet["cells"]),
            "max_new_tokens": packet["bounds"]["max_new_tokens"],
        }, sort_keys=True))
    elif args.command == "worker":
        worker(args.packet, args.grant, args.output, args.cell_id, args.physical_gpu)
    elif args.command == "consume":
        result = consume(args.packet, args.grant, args.output)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
    else:
        launch(args.packet, args.grant, args.output)


if __name__ == "__main__":
    main()
