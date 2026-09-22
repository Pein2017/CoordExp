"""CPU-only acceptance replay for the completed third-fit native readbacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.acquisition import binding, digest, publish, read, require
import probes.training_set_completion.recover_readback as recovery


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum").resolve()
ROOT = BASE / "third-fit-readback-acceptance-v1"
RUN_ROOT = BASE / "third-fit-v1"
TRAINING_ROOT = RUN_ROOT / "training"
READBACK_ROOT = RUN_ROOT / "readback-recovery"
TRAINING_MANIFEST = BASE / "third-fit-preparation-v1/manifest.json"
VERIFY = Path(__file__).resolve()
WRAPPER = RUN_ROOT / "run.py"
RECOVERY_SOURCE = WORKTREE / "probes/training_set_completion/recover_readback.py"
STEPS = (16, 32, 64)
IMAGES = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
CAP = 3084
EOS = 151645


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _readback_config(manifest_path: Path) -> None:
    recovery.MANIFEST = manifest_path.resolve()
    recovery.TRAIN = TRAINING_ROOT.resolve()
    recovery.ROOT = READBACK_ROOT.resolve()


def _row_key(step: int, image_id: int) -> str:
    return f"step-{step:05d}-image-{image_id:012d}"


def _check_stop(ids: list[int], stop: str) -> str:
    require(ids and len(ids) <= CAP, "token count/cap")
    if stop == "im_end":
        require(ids[-1] == EOS and EOS not in ids[:-1], "invalid natural EOS")
        return "eos"
    require(stop == "length" and len(ids) == CAP and EOS not in ids, "invalid length-cap termination")
    return "cap"


def _compare_envelope_row(envelope: Mapping[str, Any], durable: Mapping[str, Any]) -> None:
    fields = ("route_id", "image_id", "empty_assistant_prefix", "prompt_token_ids",
              "generated_token_ids", "generated_token_ids_sha256", "decode_stop_reason",
              "raw_decode_text", "executed_media_sha256", "observed_image_grid_thw")
    require(all(envelope.get(key) == durable.get(key) for key in fields), "envelope/durable row mismatch")


def _replay_existing(candidate: Mapping[str, Any], existing_path: Path) -> dict[str, Any]:
    """Compare a freshly computed candidate without mutating the immutable receipt.

    Adding this replay switch necessarily changes this verifier's own source
    hash.  The original receipt therefore remains authoritative for its
    historical verifier binding; only that self-binding is normalized while
    comparing the evidence payload.
    """
    existing = read(existing_path)
    require(existing.get("schema") == candidate.get("schema"), "existing receipt schema")
    existing_verifier = existing.get("sources", {}).get("verifier", {})
    current_verifier = candidate.get("sources", {}).get("verifier", {})
    require(existing_verifier.get("path") == current_verifier.get("path") == str(VERIFY), "existing verifier path")
    left = json.loads(json.dumps(candidate))
    right = json.loads(json.dumps(existing))
    left["sources"]["verifier"] = {"path": str(VERIFY)}
    right["sources"]["verifier"] = {"path": str(VERIFY)}
    require(left == right, "existing receipt differs from fresh candidate")
    return {"status": "verified_existing", "candidate_equivalent": True,
            "existing_receipt": binding(existing_path),
            "historical_verifier_binding": existing_verifier,
            "current_verifier_binding": current_verifier}


def verify(*, verify_existing: bool = False) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from probes.source_rweak_row_cross.run import native_record
    from probes.training_set_completion.training import validate_manifest

    training = read(TRAINING_MANIFEST)
    validate_manifest(training)
    require(training["runtime"] == {"checkpoint_steps": [16, 32, 64], "eos_token_id": EOS,
                                     "max_model_forwards": 704, "updates": 64, "wall_seconds": 3600}, "training runtime")
    readback_manifest_path = READBACK_ROOT / "manifest.json"
    _readback_config(readback_manifest_path)
    readback_manifest = read(readback_manifest_path)
    recovery.validate(readback_manifest)
    require(readback_manifest["training_manifest"] == binding(TRAINING_MANIFEST), "readback/training manifest binding")
    require(readback_manifest["bound_roots"] == {"manifest": str(TRAINING_MANIFEST), "training": str(TRAINING_ROOT), "readback": str(READBACK_ROOT)}, "bound roots")
    require(readback_manifest["producer"] == binding(WRAPPER), "wrapper source binding")
    require(readback_manifest["reused_producer"] == binding(RECOVERY_SOURCE), "recovery source binding")
    require(readback_manifest["runtime"] == {"cap": CAP, "eos": EOS, "worker_seconds": 3600,
                                               "policy": {"empty_assistant_prefix": True, "temperature": 0.0,
                                                          "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}}, "readback policy")
    require([int(route["image_id"]) for route in readback_manifest["routes"]] == list(IMAGES), "route cohort")

    outer_terminal = read(RUN_ROOT / "terminal.json")
    require(outer_terminal["status"] == "completed_unscored" and outer_terminal["phase"] == "completed", "outer terminal")
    require(outer_terminal["manifest"] == binding(TRAINING_MANIFEST), "outer manifest binding")
    require(outer_terminal["wrapper"] == binding(WRAPPER) and outer_terminal["reused_producer"] == binding(RECOVERY_SOURCE), "outer source bindings")
    training_terminal = read(TRAINING_ROOT / "terminal.json")
    require(training_terminal["status"] == "completed" and training_terminal["updates"] == 64 and training_terminal["model_forwards"] == 704, "training terminal")
    require([int(item["step"]) for item in training_terminal["checkpoints"]] == list(STEPS), "training checkpoints")

    checkpoint_fingerprints: dict[str, str] = {}
    for step in STEPS:
        actual = recovery.adapter_identity(step, training["model_config"])
        expected = readback_manifest["checkpoint_adapters"][str(step)]
        require(actual == expected, f"checkpoint {step} fingerprint")
        checkpoint_fingerprints[str(step)] = actual["fingerprint"]
        checkpoint_root = TRAINING_ROOT / "checkpoints" / f"step-{step:05d}"
        require((checkpoint_root / "state.pt").is_file() and (checkpoint_root / "adapter").is_dir(), f"checkpoint {step} files")

    exits = read(READBACK_ROOT / "exits.json")["exits"]
    require(len(exits) == 8 and all(item["exit_code"] == 0 for item in exits), "worker exits")
    for shard, item in enumerate(exits):
        command = item["command"]
        require(command[1] == str(WRAPPER), "child wrapper entry")
        require(command[command.index("--manifest") + 1] == str(readback_manifest_path), "child manifest root")
        require(command[command.index("--training-root") + 1] == str(TRAINING_ROOT), "child training root")
        require(command[command.index("--readback-root") + 1] == str(READBACK_ROOT), "child readback root")
        require(command[command.index("--shard") + 1] == str(shard) and command[command.index("--gpu") + 1] == str(shard), "child shard/GPU")
        require(command[0].startswith("/root/miniconda3/envs/ms/bin/"), "child interpreter")

    partitions = readback_manifest["partitions"]
    require(len(partitions) == 8 and sum(len(group) for group in partitions) == 33 and max(map(len, partitions)) <= 5, "partition shape")
    expected_loads = sum(len({int(job["step"]) for job in group}) for group in partitions)
    require(expected_loads == 9, "expected model loads")
    terminal_summaries = []
    for shard, group in enumerate(partitions):
        terminal = read(READBACK_ROOT / "terminals" / f"shard-{shard}.json")
        require(terminal["status"] == "completed" and terminal["shard"] == shard and terminal["gpu"] == shard, "worker terminal identity")
        require(terminal["expected"] == terminal["completed"] == len(group), "worker terminal count")
        require(terminal["model_loads"] == len({int(job["step"]) for job in group}), "worker model-load count")
        require(terminal["manifest"] == binding(readback_manifest_path), "worker manifest binding")
        terminal_summaries.append({"shard": shard, "gpu": shard, "jobs": len(group), "model_loads": terminal["model_loads"], "generated_tokens": terminal["generated_tokens"]})
    require(sum(item["model_loads"] for item in terminal_summaries) == 9, "actual nine model loads")
    require(terminal_summaries[2]["model_loads"] == 2, "GPU2 step16-to64 reload")
    require((READBACK_ROOT / "model/shard-2-step-00016.json").is_file() and (READBACK_ROOT / "model/shard-2-step-00064.json").is_file(), "GPU2 reload receipts")

    acquisition = read(training["acquisition_manifest"]["path"])
    acquisition_by_image = {int(record["image_id"]): record for record in acquisition["records"]}
    routes = {int(route["image_id"]): route for route in readback_manifest["routes"]}
    tokenizer = AutoTokenizer.from_pretrained(training["model_config"]["model"]["base_model"], local_files_only=True)
    rows_by_step: dict[str, dict[str, Any]] = {}
    parser_counts = {str(step): {"rows": 0, "valid_predictions": 0, "dropped_predictions": 0} for step in STEPS}
    stop_counts: dict[str, dict[str, int]] = {}
    for step in STEPS:
        envelope = read(READBACK_ROOT / f"readback-step-{step}.json")
        require(envelope["status"] == "completed_unscored" and envelope["manifest"] == binding(TRAINING_MANIFEST), f"step {step} envelope")
        require(envelope["adapter"] == readback_manifest["checkpoint_adapters"][str(step)], f"step {step} adapter envelope")
        require(len(envelope["rows"]) == len(IMAGES) and [int(row["image_id"]) for row in envelope["rows"]] == sorted(IMAGES), f"step {step} envelope cohort")
        durable_bindings = envelope["recovery"]["per_image_rows"]
        require(len(durable_bindings) == len(IMAGES), f"step {step} durable binding count")
        seen: set[int] = set()
        stop_counts[str(step)] = {"eos": 0, "cap": 0}
        for envelope_row, durable_binding in zip(envelope["rows"], durable_bindings, strict=True):
            image_id = int(envelope_row["image_id"])
            require(image_id in routes and image_id not in seen, f"step {step} row image identity")
            seen.add(image_id)
            row_path = READBACK_ROOT / "rows" / f"{_row_key(step, image_id)}.json"
            require(durable_binding == binding(row_path), f"step {step} row binding")
            durable = read(row_path)
            recovery.validate_row_checkpoint(durable, {"step": step, "image_id": image_id}, readback_manifest)
            _compare_envelope_row(envelope_row, durable)
            route = routes[image_id]
            require(durable["route_id"] == route["route_id"] and durable["image_id"] == image_id, "route identity")
            require(durable["empty_assistant_prefix"] is True and durable["prompt_token_ids"] == route["prompt_token_ids"], "prompt identity")
            require(digest(durable["prompt_token_ids"]) == durable["prompt_token_ids_sha256"], "prompt hash")
            image_path = Path(route["case"]["image_path"])
            require(_hash(image_path) == route["image_identity"]["image_content_sha256"], "original image bytes")
            require(durable["executed_media_sha256"] == route["image_identity"]["executed_media_sha256"], "executed media identity")
            require(durable["observed_image_grid_thw"] == route["image_identity"]["observed_image_grid_thw"], "image grid identity")
            ids = durable["generated_token_ids"]
            require(isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids), "raw token IDs")
            require(digest(ids) == durable["generated_token_ids_sha256"], "raw token hash")
            text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            require(text == durable["raw_decode_text"], "raw token decode")
            stop_kind = _check_stop(ids, durable["decode_stop_reason"])
            stop_counts[str(step)][stop_kind] += 1
            parsed = native_record(text, route["case"], acquisition_by_image[image_id]["golden"], durable["decode_stop_reason"])
            require(parsed["raw_decode_text"] == text and parsed["decode_stop_reason"] == durable["decode_stop_reason"], "native parser replay")
            parser_counts[str(step)]["rows"] += 1
            parser_counts[str(step)]["valid_predictions"] += int(parsed["valid_prediction_count"])
            parser_counts[str(step)]["dropped_predictions"] += int(parsed["dropped_prediction_count"])
            rows_by_step[f"{step}:{image_id}"] = {"path": str(row_path), "sha256": _hash(row_path), "token_count": len(ids), "stop": stop_kind}
        require(seen == set(IMAGES), f"step {step} complete row set")

    require(stop_counts == {"16": {"eos": 9, "cap": 2}, "32": {"eos": 11, "cap": 0}, "64": {"eos": 10, "cap": 1}}, "stop counts")
    result = read(READBACK_ROOT / "result.json")
    require(result["status"] == "candidate_ready" and result["request_count"] == 33, "recovery result")
    require(result["manifest"] == binding(readback_manifest_path), "result manifest binding")
    require(result["stop_counts"] == {"im_end": 30, "length": 3}, "result stop counts")
    require(result["envelopes"] == [binding(READBACK_ROOT / f"readback-step-{step}.json") for step in STEPS], "result envelopes")
    require(result["generated_tokens"] == sum(item["token_count"] for item in rows_by_step.values()), "result token total")

    receipt: dict[str, Any] = {
        "schema": "training_set_completion.third_fit_readback_acceptance.v1",
        "status": "candidate_ready",
        "scope": "CPU-only replay of completed third-fit readbacks; no physical-owner or model-quality decision.",
        "sources": {
            "training_manifest": binding(TRAINING_MANIFEST),
            "readback_manifest": binding(readback_manifest_path),
            "outer_terminal": binding(RUN_ROOT / "terminal.json"),
            "training_terminal": binding(TRAINING_ROOT / "terminal.json"),
            "recovery_result": binding(READBACK_ROOT / "result.json"),
            "wrapper": binding(WRAPPER),
            "reused_recovery_source": binding(RECOVERY_SOURCE),
            "verifier": binding(VERIFY)
        },
        "coverage": {"steps": list(STEPS), "rows": 33, "images_per_step": 11, "worker_count": 8, "model_loads": 9, "gpu2_reload_steps": [16, 64]},
        "stop_counts": stop_counts,
        "parser_replay": parser_counts,
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "checks": {
            "training_status_and_counters": True,
            "checkpoint_adapter_fingerprints": True,
            "prompt_media_grid_identity": True,
            "raw_token_hash_and_decode": True,
            "native_parser_replay": True,
            "cap_and_stop_contract": True,
            "durable_row_envelope_correspondence": True,
            "worker_partition_and_gpu_isolation_receipts": True,
            "wrapper_and_unchanged_recovery_bindings": True
        },
        "claim_boundary": "All rows remain raw unscored evidence; unmatched, parser-dropped, repeat, false, unknown, and geometry/class semantics are outside this receipt."
    }
    if verify_existing:
        return _replay_existing(receipt, ROOT / "receipt.json")
    publish(ROOT / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "receipt.json")
    parser.add_argument("--verify-existing", action="store_true", help="compare against immutable receipt.json without writing")
    args = parser.parse_args()
    require(args.output.resolve() == (ROOT / "receipt.json").resolve(), "fixed acceptance receipt path")
    value = verify(verify_existing=args.verify_existing)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
