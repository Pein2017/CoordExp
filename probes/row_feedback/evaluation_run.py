"""Finite natural-endpoint execution and source-blind review preparation.

The runner owns no fitting or scheduling decisions.  It consumes a root-bound
post-fit adapter, the immutable endpoint selection, and one explicitly assigned
GPU.  CPU merge/review commands preserve the fixed endpoint identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.row_feedback.evaluation import (
    EOS_TOKEN_ID,
    MAX_VISIBLE_TOKENS,
    REVIEW_SALT,
    _score_output,
    _valid_envelope,
    binding,
    digest,
    file_hash,
    make_output_envelope,
    publish,
    read,
    read_jsonl,
    require,
    validate_selection,
)


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_adapter_binding(path: str | Path, *, arm: str) -> dict[str, Any]:
    from src.adapters.dora import inspect_dora_adapter_payload

    value = read(path)
    require(value.get("schema") == "row_feedback.endpoint_adapter_binding.v1", "adapter binding schema")
    require(value.get("status") == "root_bound_after_paired_fit", "adapter binding status")
    require(value.get("arm") == arm and arm in ("S", "F"), "adapter binding arm")
    require(value.get("dose") in (16, 32, 64), "adapter binding dose")
    adapter = value.get("adapter")
    require(isinstance(adapter, Mapping), "adapter descriptor")
    root = Path(str(adapter.get("root", ""))).resolve()
    require(root.is_dir(), "adapter root")
    actual_adapter = inspect_dora_adapter_payload(root)
    require(actual_adapter == dict(adapter), "adapter payload differs from root binding")
    for label in ("fit_receipt", "execution_packet"):
        ref = value.get(label)
        require(isinstance(ref, Mapping), f"{label} binding")
        require(binding(ref["path"]) == dict(ref), f"{label} changed")
    fit_receipt = read(value["fit_receipt"]["path"])
    saved_adapter = fit_receipt.get("saved_adapter")
    if saved_adapter is None and isinstance(fit_receipt.get("arms"), Mapping):
        saved_adapter = fit_receipt["arms"].get(arm, {}).get("saved_adapter")
    require(saved_adapter == dict(adapter), "fit receipt saved adapter differs")
    if "arm" in fit_receipt:
        require(fit_receipt["arm"] == arm, "fit receipt arm differs")
    packet = read(value["execution_packet"]["path"])
    packet_dose = packet.get("dose")
    if packet_dose is None and isinstance(packet.get("schedule"), Mapping):
        packet_dose = packet["schedule"].get("dose")
    require(packet_dose == value["dose"], "execution packet dose differs")
    return value


def shard_ids(image_ids: Sequence[int], *, shard_index: int, shard_count: int) -> list[int]:
    require(0 < shard_count <= len(image_ids), "shard count")
    require(0 <= shard_index < shard_count, "shard index")
    return [int(item) for item in image_ids[shard_index::shard_count]]


def add_materialization_receipt(
    receipts: dict[str, dict[str, Any]], *, image_id: int, materialized: Mapping[str, Any]
) -> None:
    """Record per-image prompt/input identity without imposing equal image-token lengths."""

    prompt_ids = materialized.get("prompt_ids")
    require(isinstance(prompt_ids, list) and prompt_ids, "materialized prompt IDs")
    materialization = materialized.get("materialization")
    require(isinstance(materialization, Mapping), "materialization receipt")
    prompt_sha = digest(prompt_ids)
    require(materialization.get("prompt_token_ids_sha256") == prompt_sha, "materialized prompt digest")
    prepared_sha = materialized.get("prepared_inputs_sha256")
    require(isinstance(prepared_sha, str) and len(prepared_sha) == 64, "prepared input digest")
    key = str(image_id)
    require(key not in receipts, "duplicate materialization receipt")
    receipts[key] = {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": prompt_sha,
        "prepared_inputs_sha256": prepared_sha,
        "executed_media_sha256": materialization["executed_media_sha256"],
        "observed_image_grid_thw": materialization["observed_image_grid_thw"],
    }


def _shared_model_identity(config: Any) -> dict[str, Any]:
    backend_hf = getattr(config.backend, "hf", None)
    return {
        "base_model": str(config.model.base_model),
        "dtype": str(config.model.dtype),
        "attention": None if backend_hf is None else str(backend_hf.attn_implementation),
        "embedding_delta": config.embedding_delta.model_dump(mode="json"),
        "template": config.template.model_dump(mode="json"),
        "generation": config.generation.model_dump(mode="json"),
    }


def _natural_result(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "arm",
        "visible_token_ids",
        "text",
        "finish_reason",
        "eos",
        "cap",
        "visible_generated_tokens",
        "internal_slot_count",
        "physical_token_count",
        "model_forwards",
        "image_forwards",
        "slot_work",
        "timing",
        "decode_contract",
    )
    return {key: value[key] for key in keys}


def run_shard(
    *,
    selection_path: str | Path,
    adapter_binding_path: str | Path,
    arm: str,
    shard_index: int,
    shard_count: int,
    physical_gpu: int,
    output: str | Path,
) -> dict[str, Any]:
    import torch
    from probes.row_feedback import runtime

    output = Path(output)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "physical GPU binding")
    require(torch.cuda.device_count() == 1, "endpoint worker requires exactly one visible GPU")
    require(not output.exists(), "endpoint shard output already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    phase = "preflight"
    try:
        selection = read(selection_path)
        validate_selection(selection, verify_files=True)
        selection_sha = file_hash(selection_path)
        adapter_binding = validate_adapter_binding(adapter_binding_path, arm=arm)
        ids = shard_ids(selection["image_ids"], shard_index=shard_index, shard_count=shard_count)
        records = {int(item["image_id"]): item for item in selection["records"]}
        launch = {
            "schema": "row_feedback.endpoint_shard_launch.v1",
            "status": "running",
            "pid": os.getpid(),
            "arm": arm,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "physical_gpu": physical_gpu,
            "image_ids": ids,
            "selection": binding(selection_path),
            "adapter_binding": binding(adapter_binding_path),
            "max_visible_tokens": MAX_VISIBLE_TOKENS,
            "code": {
                "runner": binding(Path(__file__)),
                "consumer": binding(Path(__file__).with_name("evaluation.py")),
                "runtime": binding(Path(runtime.__file__)),
            },
        }
        publish(output / "launch.json", launch)
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        phase = "load"
        load_started = time.monotonic()
        qwen, frontend, config, loaded_identity = runtime.load_feedback_policy(
            adapter_path=adapter_binding["adapter"]["root"], device=device
        )
        load_seconds = time.monotonic() - load_started
        model_identity = _shared_model_identity(config)
        materialization_by_image: dict[str, dict[str, Any]] = {}
        envelopes = []
        for ordinal, image_id in enumerate(ids):
            phase = f"materialize:{image_id}"
            materialized = runtime.materialize_endpoint_record(
                qwen,
                frontend,
                config,
                records[image_id],
                native_source=selection["native_source"],
            )
            add_materialization_receipt(
                materialization_by_image, image_id=image_id, materialized=materialized
            )
            phase = f"generate:{image_id}"
            generated = runtime.generate_visible(
                qwen,
                materialized["inputs"],
                prompt_ids=materialized["prompt_ids"],
                history_ids=(),
                arm=arm,
                max_visible_tokens=MAX_VISIBLE_TOKENS,
                eos_token_id=EOS_TOKEN_ID,
            )
            envelope = make_output_envelope(
                selection_sha256=selection_sha,
                record=records[image_id],
                arm=arm,
                prompt_token_ids=materialized["prompt_ids"],
                prepared_inputs_sha256=materialized["prepared_inputs_sha256"],
                model_identity=model_identity,
                adapter_identity=adapter_binding["adapter"],
                runtime=_natural_result(generated),
            )
            envelope["materialization"] = materialized["materialization"]
            envelope["shard"] = {"index": shard_index, "count": shard_count, "ordinal": ordinal}
            publish(output / "rows" / f"{ordinal:03d}-{image_id}.json", envelope)
            envelopes.append(envelope)
        phase = "finalize"
        _write_text_exclusive(
            output / "rows.jsonl",
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in envelopes),
        )
        terminal = {
            "schema": "row_feedback.endpoint_shard_terminal.v1",
            "status": "completed",
            "exit_code": 0,
            "arm": arm,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "physical_gpu": physical_gpu,
            "image_ids": ids,
            "rows": binding(output / "rows.jsonl"),
            "launch": binding(output / "launch.json"),
            "selection": binding(selection_path),
            "adapter_binding": binding(adapter_binding_path),
            "loaded_identity": loaded_identity,
            "materialization_by_image": materialization_by_image,
            "resources": {
                "load_seconds": load_seconds,
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            },
            "counters": {
                "images": len(envelopes),
                "visible_generated_tokens": sum(row["runtime"]["visible_generated_tokens"] for row in envelopes),
                "internal_slot_count": sum(row["runtime"]["internal_slot_count"] for row in envelopes),
                "model_forwards": sum(row["runtime"]["model_forwards"] for row in envelopes),
                "image_forwards": sum(row["runtime"]["image_forwards"] for row in envelopes),
            },
        }
        publish(output / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.endpoint_shard_failure.v1",
            "status": "failed",
            "exit_code": 1,
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        publish(output / "failure.json", failure)
        raise


def merge_shards(
    *, selection_path: str | Path, arm: str, shard_dirs: Sequence[str | Path], output: str | Path
) -> dict[str, Any]:
    output = Path(output)
    require(not output.exists(), "merged endpoint output already exists")
    output.mkdir(parents=True)
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    expected = [int(item) for item in selection["image_ids"]]
    selection_sha = file_hash(selection_path)
    records = {int(item["image_id"]): item for item in selection["records"]}
    rows: dict[int, dict[str, Any]] = {}
    terminals = []
    shard_indices: set[int] = set()
    common_shard_count: int | None = None
    common_adapter_binding: Mapping[str, Any] | None = None
    for directory in map(Path, shard_dirs):
        terminal_path = directory / "terminal.json"
        terminal = read(terminal_path)
        require(terminal.get("status") == "completed" and terminal.get("arm") == arm, "shard terminal")
        require(terminal["launch"] == binding(directory / "launch.json"), "shard launch binding")
        require(terminal["selection"] == binding(selection_path), "shard selection binding")
        require(terminal["rows"] == binding(directory / "rows.jsonl"), "shard rows binding")
        shard_index = int(terminal["shard_index"])
        shard_count = int(terminal["shard_count"])
        require(shard_index not in shard_indices, "duplicate shard index")
        shard_indices.add(shard_index)
        if common_shard_count is None:
            common_shard_count = shard_count
            common_adapter_binding = terminal["adapter_binding"]
        require(shard_count == common_shard_count, "mixed shard counts")
        require(terminal["adapter_binding"] == common_adapter_binding, "mixed adapter bindings")
        shard_rows = read_jsonl(directory / "rows.jsonl")
        require(
            [int(row["image_id"]) for row in shard_rows] == [int(item) for item in terminal["image_ids"]],
            "shard terminal/row identities",
        )
        for row in shard_rows:
            image_id = int(row["image_id"])
            require(image_id not in rows, "duplicate image across shards")
            require(image_id in records, "unknown merged image")
            _valid_envelope(
                row,
                arm=arm,
                selection_sha256=selection_sha,
                record=records[image_id],
            )
            rows[image_id] = row
        terminals.append(binding(terminal_path))
    require(common_shard_count == len(shard_dirs), "missing shard terminal")
    require(shard_indices == set(range(common_shard_count)), "shard index coverage")
    require(set(rows) == set(expected), "merged endpoint32 coverage")
    ordered = [rows[image_id] for image_id in expected]
    merged_path = output / "rows.jsonl"
    _write_text_exclusive(
        merged_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
    )
    receipt = {
        "schema": "row_feedback.endpoint_merge_receipt.v1",
        "status": "completed",
        "arm": arm,
        "images": len(ordered),
        "selection": binding(selection_path),
        "shard_terminals": terminals,
        "rows": binding(merged_path),
    }
    publish(output / "merge-receipt.json", receipt)
    return receipt


def prepare_blind_review(
    *,
    selection_path: str | Path,
    s_rows_path: str | Path,
    f_rows_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    require(not output.exists(), "blind review output already exists")
    output.mkdir(parents=True)
    selection = read(selection_path)
    validate_selection(selection, verify_files=True)
    selection_sha = file_hash(selection_path)
    records = {int(row["image_id"]): row for row in selection["records"]}
    arms = {
        "S": {int(row["image_id"]): row for row in read_jsonl(s_rows_path)},
        "F": {int(row["image_id"]): row for row in read_jsonl(f_rows_path)},
    }
    expected = set(int(item) for item in selection["image_ids"])
    require(set(arms["S"]) == set(arms["F"]) == expected, "blind-review endpoint coverage")
    queue_rows, source_rows = [], []
    for image_id in selection["dense_review_ids"]:
        record = records[int(image_id)]
        proposals = []
        for arm in ("S", "F"):
            envelope = arms[arm][int(image_id)]
            _valid_envelope(envelope, arm=arm, selection_sha256=selection_sha, record=record)
            parsed = _score_output(envelope, record)["parser"]["predictions"]
            for source_index, prediction in enumerate(parsed):
                proposal = {
                    "description": prediction["description"],
                    "bbox": list(prediction["bbox"]),
                    "bbox_format": "xyxy_native_pixels",
                }
                proposals.append((arm, source_index, proposal))
        proposals.sort(
            key=lambda item: digest(
                {"salt": REVIEW_SALT, "image_id": image_id, "source_slot": item[0], "proposal": item[2]}
            )
        )
        public = []
        for ordinal, (arm, source_index, proposal) in enumerate(proposals):
            proposal_id = hashlib.sha256(
                f"{REVIEW_SALT}{image_id}:{ordinal}:{digest(proposal)}".encode()
            ).hexdigest()
            public.append({"proposal_id": proposal_id, **proposal})
            source_rows.append(
                {
                    "image_id": int(image_id),
                    "proposal_id": proposal_id,
                    "source_arm": arm,
                    "source_prediction_index": source_index,
                }
            )
        queue_rows.append(
            {
                "review_id": f"row-feedback-dense:{image_id}",
                "image_id": int(image_id),
                "image_path": record["image_path"],
                "width": record["width"],
                "height": record["height"],
                "proposals": public,
                "source_blind": True,
            }
        )
    queue = {
        "schema": "row_feedback.dense8_blind_review_queue.v1",
        "status": "ready_for_physical_review",
        "selection": binding(selection_path),
        "images": len(queue_rows),
        "rows": queue_rows,
        "review_questions": [
            "Is each proposal a visible instance with the stated class and plausible geometry?",
            "Do overlapping same-class proposals represent distinct instances, duplicate re-entry, or geometry aliases?",
        ],
        "boundary": "Mixed S/F valid proposals only; no GT or arm labels. Proposal review is not exhaustive recall and unmatched is not automatically hallucination.",
    }
    source_map = {
        "schema": "row_feedback.dense8_blind_review_source_map.v1",
        "status": "sealed_not_for_reviewer",
        "selection": binding(selection_path),
        "rows": source_rows,
    }
    publish(output / "queue.json", queue)
    publish(output / "source-map.json", source_map)
    manifest = {
        "schema": "row_feedback.dense8_blind_review_manifest.v1",
        "status": "ready_for_render",
        "queue": binding(output / "queue.json"),
        "source_map": binding(output / "source-map.json"),
        "renderer_contract": "Render original image_path and literal proposal xyxy boxes; show queue only to reviewer.",
    }
    publish(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--selection", type=Path, required=True)
    worker.add_argument("--adapter-binding", type=Path, required=True)
    worker.add_argument("--arm", choices=("S", "F"), required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    worker.add_argument("--physical-gpu", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--selection", type=Path, required=True)
    merge.add_argument("--arm", choices=("S", "F"), required=True)
    merge.add_argument("--shard", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("prepare-review")
    review.add_argument("--selection", type=Path, required=True)
    review.add_argument("--s-rows", type=Path, required=True)
    review.add_argument("--f-rows", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        result = run_shard(
            selection_path=args.selection,
            adapter_binding_path=args.adapter_binding,
            arm=args.arm,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            physical_gpu=args.physical_gpu,
            output=args.output,
        )
    elif args.command == "merge":
        result = merge_shards(
            selection_path=args.selection,
            arm=args.arm,
            shard_dirs=args.shard,
            output=args.output,
        )
    else:
        result = prepare_blind_review(
            selection_path=args.selection,
            s_rows_path=args.s_rows,
            f_rows_path=args.f_rows,
            output=args.output,
        )
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
