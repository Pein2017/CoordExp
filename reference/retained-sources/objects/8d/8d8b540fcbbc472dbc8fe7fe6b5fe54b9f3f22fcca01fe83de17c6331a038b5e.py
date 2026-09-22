from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch

from probes.row_feedback import runtime


RUN_DIR = Path(__file__).resolve().parent
BANK_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-row-feedback-pilot/data-v2/supervision-bank.json"
)
BANK_SHA256 = "1f0a2c2b82e3bf696a7c9721db0d5b3c5ee34eee01f8f29ef669446c4428cd1c"
RUNTIME_PATH = Path(
    "/data/CoordExp/.worktrees/row-feedback-pilot-20260913/"
    "probes/row_feedback/runtime.py"
)
RUNTIME_SHA256 = "94e9a2f472af2a1e60f06f215f72a2a362c9b2773918f98a7f85f983f1ef57f6"
RECORD_ID = "coco2017_train_000000219546:702889:h+c->w"
MAX_VISIBLE_TOKENS = 3084


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def compact_result(result: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    wall = float(result["timing"]["wall_seconds"])
    visible = int(result["visible_generated_tokens"])
    forwards = int(result["model_forwards"])
    seconds_per_visible = wall / visible if visible else None
    seconds_per_forward = wall / forwards if forwards else None
    return {
        "raw_output": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "arm": result["arm"],
        "finish_reason": result["finish_reason"],
        "eos": result["eos"],
        "cap": result["cap"],
        "visible_generated_tokens": visible,
        "visible_token_ids_sha256": hashlib.sha256(
            json.dumps(result["visible_token_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
        "internal_slot_count": result["internal_slot_count"],
        "slot_work": result["slot_work"],
        "model_forwards": forwards,
        "image_forwards": result["image_forwards"],
        "physical_token_count": result["physical_token_count"],
        "decode_contract": result["decode_contract"],
        "runtime_reported_wall_seconds": wall,
        "seconds_per_generated_visible_token": seconds_per_visible,
        "seconds_per_model_forward": seconds_per_forward,
        "linear_3084_visible_seconds": (
            None if seconds_per_visible is None else seconds_per_visible * MAX_VISIBLE_TOKENS
        ),
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("cost probe owns physical GPU0 only")
    if sha256(RUNTIME_PATH) != RUNTIME_SHA256:
        raise RuntimeError("runtime code changed after consumer freeze")
    if sha256(BANK_PATH) != BANK_SHA256:
        raise RuntimeError("supervision bank binding changed")
    if (RUN_DIR / "cost-receipt.json").exists():
        raise RuntimeError("cost receipt already exists")

    process_started = time.monotonic()
    phase = "load_inputs"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        bank = json.loads(BANK_PATH.read_text())
        records = [record for record in bank["records"] if record["record_id"] == RECORD_ID]
        if len(records) != 1:
            raise RuntimeError("cost record is missing or duplicated")
        record = records[0]
        source_adapter = record["teacher"]["source_adapter"]
        adapter = Path(source_adapter["root"]).resolve()
        if adapter != runtime.ANCHOR_ADAPTER.resolve():
            raise RuntimeError("cost probe is not bound to the original N16 adapter")

        phase = "model_load"
        load_started = time.monotonic()
        gate = runtime.embedding_source_gate_receipt()
        qwen, frontend, config, identity = runtime.load_feedback_policy(
            adapter_path=adapter,
            device=device,
        )
        qwen.model.eval()
        torch.cuda.synchronize(device)
        load_wall = time.monotonic() - load_started
        load_peak_allocated = torch.cuda.max_memory_allocated(device)
        load_peak_reserved = torch.cuda.max_memory_reserved(device)

        phase = "materialize"
        materialize_started = time.monotonic()
        materialized = runtime.materialize_bank_records(qwen, frontend, config, bank)[RECORD_ID]
        torch.cuda.synchronize(device)
        materialize_wall = time.monotonic() - materialize_started

        arms: dict[str, Any] = {}
        arm_peak_allocated: list[int] = []
        arm_peak_reserved: list[int] = []
        for arm in ("S", "F"):
            phase = f"generate_{arm}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            arm_started = time.monotonic()
            result = runtime.generate_visible(
                qwen,
                materialized["inputs"],
                prompt_ids=materialized["prompt_ids"],
                history_ids=(),
                arm=arm,
                max_visible_tokens=MAX_VISIBLE_TOKENS,
            )
            torch.cuda.synchronize(device)
            synchronized_wall = time.monotonic() - arm_started
            if result["visible_generated_tokens"] != len(result["visible_token_ids"]):
                raise RuntimeError(f"{arm} visible accounting mismatch")
            if result["visible_generated_tokens"] > MAX_VISIBLE_TOKENS:
                raise RuntimeError(f"{arm} exceeded visible budget")
            if result["decode_contract"]["max_visible_tokens"] != MAX_VISIBLE_TOKENS:
                raise RuntimeError(f"{arm} decode contract changed")
            raw = runtime._jsonable_result(result)
            raw["synchronized_wall_seconds"] = synchronized_wall
            raw_path = RUN_DIR / f"{arm}-raw-output.json"
            write_json(raw_path, raw)
            compact = compact_result(result, raw_path)
            compact["synchronized_wall_seconds"] = synchronized_wall
            compact["synchronized_seconds_per_generated_visible_token"] = (
                synchronized_wall / result["visible_generated_tokens"]
                if result["visible_generated_tokens"]
                else None
            )
            compact["synchronized_seconds_per_model_forward"] = (
                synchronized_wall / result["model_forwards"]
                if result["model_forwards"]
                else None
            )
            compact["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            compact["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            arm_peak_allocated.append(compact["peak_cuda_allocated_bytes"])
            arm_peak_reserved.append(compact["peak_cuda_reserved_bytes"])
            arms[arm] = compact

        phase = "receipt"
        receipt = {
            "schema": "row_feedback.natural_cost_receipt.v1",
            "status": "technical_cost_pair_complete",
            "claim_boundary": (
                "Technical runtime cost evidence only; outputs are not quality evidence and do not "
                "select a fit dose."
            ),
            "runtime": {"path": str(RUNTIME_PATH), "sha256": RUNTIME_SHA256},
            "bank": {"path": str(BANK_PATH), "sha256": BANK_SHA256},
            "record_id": RECORD_ID,
            "history_token_count": 0,
            "same_materialized_input_for_both_arms": True,
            "materialization": materialized["materialization"],
            "prepared_inputs_sha256": materialized["prepared_inputs_sha256"],
            "source_adapter": {
                "root": str(adapter),
                "fingerprint": source_adapter["fingerprint"],
                "files": source_adapter["files"],
            },
            "loaded_identity": identity,
            "source_gate": gate,
            "timing": {
                "model_load_wall_seconds": load_wall,
                "materialization_wall_seconds": materialize_wall,
                "process_wall_seconds": time.monotonic() - process_started,
            },
            "arms": arms,
            "resources": {
                "peak_cuda_allocated_bytes": max([load_peak_allocated, *arm_peak_allocated]),
                "peak_cuda_reserved_bytes": max([load_peak_reserved, *arm_peak_reserved]),
                "peak_rss_bytes": max_rss_bytes(),
            },
        }
        write_json(RUN_DIR / "cost-receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "output": str(RUN_DIR)}, sort_keys=True))
    except BaseException as exc:
        write_json(
            RUN_DIR / "failure.json",
            {
                "schema": "row_feedback.natural_cost_failure.v1",
                "status": "failed",
                "phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.monotonic() - process_started,
            },
        )
        raise


if __name__ == "__main__":
    main()
