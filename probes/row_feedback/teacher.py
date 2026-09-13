"""Fresh native-N16 full-vocabulary teacher cache for row-feedback protection.

The cache owns no slot implementation and performs no optimization.  It
replays the exact protection-record prompts/actions through the native no-slot
model and persists only selected visible-target ordinal log probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from probes.row_feedback import data


SCHEMA = "row_feedback.native_n16_teacher_cache.v1"
TENSOR_KEY = "log_probs"
NORMALIZATION_ATOL = 3e-5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_digest(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _positions_sha256(positions: Sequence[int]) -> str:
    return _json_digest(list(positions))


def _normalization_max_abs(log_probs: Any) -> float:
    import torch

    values = torch.logsumexp(log_probs, dim=-1).abs()
    return float(values.max().item()) if values.numel() else 0.0


def _check_log_probs(tensor: Any, *, positions: Sequence[int], vocabulary_size: int | None = None) -> None:
    import torch

    _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.float32,
             "teacher tensor must be CPU contiguous float32")
    _require(tensor.ndim == 2 and tensor.shape[0] == len(positions) and tensor.shape[1] > 1,
             "teacher tensor shape/visible ordinals")
    _require(tensor.is_contiguous() and bool(torch.isfinite(tensor).all()), "teacher tensor finite contiguous")
    if vocabulary_size is not None:
        _require(tensor.shape[1] == vocabulary_size, "teacher vocabulary dimension")
    _require(_normalization_max_abs(tensor) <= NORMALIZATION_ATOL, "teacher log-prob normalization")


def _source_packet(protection_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = data.binding(protection_path)
    protection = data._read_bound(reference)
    data.validate_protection_records(protection)
    return reference, protection


def _verify_teacher_adapter(identity: Mapping[str, Any]) -> None:
    root = Path(identity["root"])
    _require(root.is_dir() and identity.get("fingerprint"), "N16 teacher adapter identity")
    for item in identity["files"]:
        candidate = root / item["relative_path"]
        _require(candidate.is_file() and data.file_hash(candidate) == item["sha256"],
                 f"N16 teacher adapter drift: {candidate}")


def _load_teacher_policy(*, anchor: Mapping[str, Any], adapter_path: str, device: Any,
                         output: Path) -> tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    """Load the native N16 through the runtime's explicit accepted gate route."""
    from src.config.inference import load_research_infer_config
    from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
    from probes.parallel_owner_research import training
    from probes.row_feedback import runtime

    base = load_research_infer_config(training.CONFIG).config
    _require(base.model_dump(mode="json") == anchor["config"], "live base config identity")
    source_gate = runtime.embedding_source_gate_receipt()
    qwen, frontend, config, identity = runtime.load_feedback_policy(
        adapter_path=adapter_path, device=device, source_gate_root=source_gate["root"],
    )
    qwen.model.eval()
    _require(str(config.embedding_delta.source_gate_root) == source_gate["root"],
             "teacher did not use the explicit source gate")
    inspected = inspect_special_token_embedding_delta_payload(
        identity["model_identity"]["embedding_delta"]["identity"]["delta_path"],
        anchor["model_identity"]["base_model"],
    )
    composition = training.old.loaded_composition_evidence(
        loaded_identity=identity, expected_base=anchor["model_identity"]["base_model"],
        expected_adapter=adapter_path, expected_embedding=anchor["source_embedding"],
        inspected_embedding=inspected,
    )
    _require(composition["passed"], "loaded N16 composition differs from frozen source")
    _require(qwen.token_identity.im_end_token_ids == (training.old.EOS,)
             and qwen.tokenizer.pad_token_id == training.old.PAD,
             "native terminal token identity")
    _write_exclusive(output / "loaded-model.json", identity)
    _write_exclusive(output / "loaded-composition-check.json", composition)
    _write_exclusive(output / "embedding-source-gate.json", source_gate)
    return qwen, frontend, config, identity, source_gate


def _save_tensor(path: Path, tensor: Any) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    _require(not path.exists(), f"refuse to overwrite teacher tensor: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({TENSOR_KEY: tensor}, str(path), metadata={"format": "pt"})
    loaded = load_file(str(path), device="cpu")
    _require(set(loaded) == {TENSOR_KEY} and loaded[TENSOR_KEY].dtype == tensor.dtype
             and loaded[TENSOR_KEY].shape == tensor.shape and loaded[TENSOR_KEY].is_contiguous(),
             "teacher safetensors save/load identity")
    _require(_tensor_digest(loaded[TENSOR_KEY]) == _tensor_digest(tensor), "teacher tensor roundtrip")
    return {
        "path": str(path.resolve()), "sha256": data.file_hash(path), "tensor_key": TENSOR_KEY,
        "dtype": "float32", "shape": list(tensor.shape), "tensor_sha256": _tensor_digest(tensor),
    }


def _cache_record(*, model: Any, entry: Mapping[str, Any], record: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import torch
    from src.qwen.native import prepare_replay

    target_ids = record["action_ids"]
    positions = record["kl_positions"]
    with torch.no_grad():
        replay = prepare_replay(model, entry["inputs"], prompt_token_ids=entry["prompt_ids"],
                                continuation_token_ids=target_ids)
        logits = replay.aligned_logits(model(**replay.inputs).logits)
        _require(replay.target_ids.tolist() == target_ids, f"literal target alignment: {record['key']}")
        _require(logits.dtype == torch.float32 and logits.shape[0] == len(target_ids),
                 f"FP32 native logits: {record['key']}")
        selected = torch.log_softmax(logits[list(positions)].float(), dim=-1).detach().cpu().contiguous()
    _check_log_probs(selected, positions=positions, vocabulary_size=logits.shape[1])
    tensor = _save_tensor(output / "tensors" / f"{record['key']}.safetensors", selected)
    return {
        "key": record["key"], "example_id": record["example_id"],
        "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
        "action_ids_sha256": record["action_ids_sha256"],
        "kl_positions": positions, "kl_positions_sha256": _positions_sha256(positions),
        "visible_target_ordinals": positions, "log_probs": tensor,
        "normalization_max_abs_logsumexp": _normalization_max_abs(selected),
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest["schema"] == SCHEMA and manifest["status"] == "complete", "teacher manifest status")
    protection = data._read_bound(manifest["source_protection_records"])
    data.validate_protection_records(protection)
    _require(manifest["native_n16_teacher"] == protection["fresh_n16_teacher"], "teacher identity binding")
    gate = manifest["embedding_source_gate"]
    _require(gate["source_study_passed"] and gate["roundtrip_probe_passed"]
             and gate["probe_receipt_ok"], "teacher embedding source gate status")
    for field in ("source_study", "roundtrip_probe"):
        reference = gate[field]
        _require(data.file_hash(reference["path"]) == reference["sha256"],
                 f"teacher embedding source gate drift: {field}")
    records = manifest["records"]
    source_by_key = {record["key"]: record for record in protection["records"]}
    _require([record["key"] for record in records] == manifest["normal_keys"], "manifest record order")
    _require(set(manifest["normal_keys"]).issubset(source_by_key), "teacher key outside protection corpus")
    if manifest["mode"] == "full":
        _require(manifest["normal_keys"] == protection["normal_keys"], "full cache key coverage")
    positions_total = 0
    for record in records:
        source = source_by_key[record["key"]]
        _require(record["example_id"] == source["example_id"], "teacher example identity")
        _require(record["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"], "teacher prompt identity")
        _require(record["action_ids_sha256"] == source["action_ids_sha256"], "teacher action identity")
        _require(record["kl_positions"] == source["kl_positions"]
                 and record["visible_target_ordinals"] == record["kl_positions"], "teacher ordinal identity")
        _require(record["kl_positions_sha256"] == _positions_sha256(record["kl_positions"]), "teacher position hash")
        tensor = record["log_probs"]
        _require(tensor["tensor_key"] == TENSOR_KEY and tensor["dtype"] == "float32"
                 and tensor["shape"][0] == len(record["kl_positions"]), "teacher tensor metadata")
        _require(data.file_hash(tensor["path"]) == tensor["sha256"], "teacher tensor file hash")
        positions_total += len(record["kl_positions"])
    _require(manifest["denominators"] == {"records": len(records), "protected_positions": positions_total},
             "teacher cache denominators")
    without_hash = dict(manifest)
    manifest_hash = without_hash.pop("manifest_sha256")
    _require(manifest_hash == _json_digest(without_hash), "teacher manifest hash")


def load_teacher_record(manifest_path: str | Path, key: str, *, expected_prompt_sha256: str,
                        expected_action_sha256: str, expected_positions: Sequence[int]):
    """Runtime-facing safe loader returning a CPU FP32 full-vocabulary tensor."""
    from safetensors.torch import load_file

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    record = next((row for row in manifest["records"] if row["key"] == key), None)
    _require(record is not None, f"teacher key not found: {key}")
    _require(record["prompt_token_ids_sha256"] == expected_prompt_sha256, "runtime prompt binding")
    _require(record["action_ids_sha256"] == expected_action_sha256, "runtime action binding")
    _require(record["visible_target_ordinals"] == list(expected_positions), "runtime ordinal binding")
    tensor_ref = record["log_probs"]
    tensor = load_file(tensor_ref["path"], device="cpu")[TENSOR_KEY].contiguous()
    _check_log_probs(tensor, positions=expected_positions, vocabulary_size=tensor_ref["shape"][1])
    _require(_tensor_digest(tensor) == tensor_ref["tensor_sha256"], "runtime tensor binding")
    return tensor


def generate_cache(*, protection_path: str | Path, output_dir: str | Path, device: str, limit: int | None = None) -> dict[str, Any]:
    """Run the real no-slot native N16 cache path on one assigned GPU."""
    import torch
    from probes.row_feedback import runtime

    protection_ref, protection = _source_packet(protection_path)
    all_records = protection["records"]
    selected = all_records if limit is None else all_records[:limit]
    _require(selected and (limit is None or limit > 0), "positive teacher cache limit")
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"refuse to overwrite teacher output: {output}")
    _require(device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") == "2",
             "teacher cache is assigned only to physical GPU2")
    output.mkdir(parents=True)
    started = time.monotonic()
    packet = data._read_bound(protection["sources"]["native_n16_training_input"])
    anchor = data._read_bound(packet["anchor_input"])
    identity = protection["fresh_n16_teacher"]
    _verify_teacher_adapter(identity)
    status, phase, error = "failed", "preload", None
    cached: list[dict[str, Any]] = []
    try:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        qwen, frontend, _config, _loaded_identity, source_gate = _load_teacher_policy(
            anchor=anchor, adapter_path=identity["root"], device=torch.device(device), output=output,
        )
        loaded_model = data.binding(output / "loaded-model.json")
        loaded_composition = data.binding(output / "loaded-composition-check.json")
        loaded_identity = json.loads((output / "loaded-model.json").read_text())
        _require(loaded_identity["model_identity"]["adapter"]["adapter_path"] == identity["root"],
                 "loaded adapter root differs from frozen N16 teacher")
        _require(json.loads((output / "loaded-composition-check.json").read_text())["passed"], "N16 composition")
        model = qwen.model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        phase = "native_replay"
        for frozen in selected:
            entry = runtime.materialize_record(qwen, frontend, _config, frozen)
            cached.append(_cache_record(model=model, entry=entry, record=frozen, output=output))
        elapsed = time.monotonic() - started
        manifest = {
            "schema": SCHEMA, "status": "complete", "mode": "full" if limit is None else "one_record_smoke",
            "claim_boundary": "Native no-slot, no-grad FP32/SDPA N16 teacher cache only; no optimizer, slot route, or scientific endpoint claim.",
            "source_protection_records": protection_ref,
            "native_n16_teacher": identity,
            "loaded_model": loaded_model, "loaded_composition_check": loaded_composition,
            "embedding_source_gate": source_gate,
            "effective_settings": {"dtype": "float32", "attention": "sdpa", "slots": "none",
                                   "temperature": 1.0, "gradients": "disabled"},
            "normal_keys": [record["key"] for record in cached], "records": cached,
            "denominators": {"records": len(cached),
                             "protected_positions": sum(len(record["kl_positions"]) for record in cached)},
            "resources": {"elapsed_seconds": elapsed,
                          "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                          "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                          "model_forwards": len(cached), "image_forwards": len(cached),
                          "physical_gpu": 2},
            "command": [sys.executable, "-m", "probes.row_feedback.teacher", "--protection-records",
                        str(protection_path), "--output-dir", str(output), "--device", device,
                        *([] if limit is None else ["--limit", str(limit)])],
        }
        manifest["manifest_sha256"] = _json_digest(manifest)
        validate_manifest(manifest)
        _write_exclusive(output / "manifest.json", manifest)
        status, phase = "complete", "saved"
        return manifest
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if torch.cuda.is_available():
            resources = {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                         "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
        else:
            resources = {}
        terminal = {"schema": "row_feedback.native_n16_teacher_terminal.v1", "status": status,
                    "phase": phase, "error": error, "elapsed_seconds": time.monotonic() - started,
                    "resources": resources, "physical_gpu": 2}
        _write_exclusive(output / "terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protection-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = generate_cache(protection_path=args.protection_records, output_dir=args.output_dir,
                            device=args.device, limit=args.limit)
    print(json.dumps({"manifest": str((args.output_dir / "manifest.json").resolve()),
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
