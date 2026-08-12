#!/usr/bin/env python3
"""Attest one fresh HF inference from the terminal Wave 7 resume child."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.dora import inspect_dora_adapter_payload  # noqa: E402
from src.artifacts.checkpoint_payload import (  # noqa: E402
    admit_inference_checkpoint_payload_identity,
    load_inference_checkpoint_payload_manifest,
)
from src.config.inference import load_infer_config  # noqa: E402
from src.data.jsonl import load_raw_examples  # noqa: E402
from src.inference.artifacts import (  # noqa: E402
    IMAGE_PLAN_NAME,
    MANIFEST_NAME,
    PARSE_DIAGNOSTICS_NAME,
    PROVENANCE_NAME,
    RAW_NAME,
    SCORED_NAME,
    SUMMARY_NAME,
    TOKEN_TRACE_NAME,
    validate_scored_artifact_set,
)
from src.qwen.parity import (  # noqa: E402
    assert_absent_artifact_target,
    canonical_json_bytes,
    write_strict_json_atomic,
)
from src.qwen.special_token_embeddings import (  # noqa: E402
    inspect_special_token_embedding_delta_payload,
)
from scripts.probes.coordexp_swift.wave7_exact_resume_sequence import (  # noqa: E402
    RECEIPT_SCHEMA as SEQUENCE_SCHEMA,
)


FINAL_COMPARISON_SCHEMA = "coordexp-swift-wave7-exact-resume-comparison-v2"
MARKER_SCHEMA = "coordexp-swift-wave7-exact-resume-postrun-attempt-marker-v1"
RECEIPT_SCHEMA = "coordexp-swift-wave7-exact-resume-postrun-v1"
PUBLICATION_FAILURE_SCHEMA = (
    "coordexp-swift-wave7-exact-resume-postrun-publication-failure-v1"
)
FAILURE_SCHEMA = "coordexp-swift-wave7-exact-resume-postrun-failure-v1"
CHECKPOINT_PUBLICATION_SCHEMA = (
    "coordexp-swift-inference-checkpoint-payload-publication"
)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSONL_ROWS = 10_000
MAX_CHILD_OUTPUT_TAIL_BYTES = 64 * 1024
MAX_GPU_CSV_STDOUT_BYTES = 1024 * 1024
MAX_GPU_CSV_STDERR_BYTES = 64 * 1024
OUTPUT_DRAIN_JOIN_SECONDS = 5.0
GPU_MEMORY_TOTAL_MIB = 81_920
GPU_MEMORY_USED_CEILING_MIB = 49_152
GPU_MEMORY_HEADROOM_FLOOR_MIB = 32_768
GPU_STABILITY_SECONDS = 2.0
GPU_OBSERVATION_TIMEOUT_SECONDS = 120.0
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 30 * 60.0
POSTRUN_LEAVES = (
    "final-child-infer.yaml",
    "inference-attempt-marker.json",
    "final-child-infer",
    "postrun-receipt.json",
    "postrun-receipt.publication-failure.json",
)
OUTPUT_FILES = frozenset(
    {
        RAW_NAME,
        SCORED_NAME,
        TOKEN_TRACE_NAME,
        PROVENANCE_NAME,
        PARSE_DIAGNOSTICS_NAME,
        IMAGE_PLAN_NAME,
        SUMMARY_NAME,
        MANIFEST_NAME,
        "configs/resolved.json",
        "configs/resolved.yaml",
    }
)
FIXTURE = (
    REPO_ROOT / "tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.single.jsonl"
).resolve()
BASE_MODEL = (
    REPO_ROOT / "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
).resolve()
EXPECTED_ROW_ID = "coco2017_train_000000000030__smoke2obj"
FIXTURE_IDENTITY_FIELDS = (
    "row_id",
    "row_index",
    "example_id",
    "image_path",
    "image_width",
    "image_height",
    "gt",
)
SYSTEM_PROMPT = (
    "You are a general-purpose object detection and grounding assistant. Output "
    "one compact detection row per object by concatenating rows directly with no "
    "separator, no newline, and no extra text. Use this row pattern exactly: "
    "<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|coord_100|>"
    "<|coord_200|><|coord_300|><|coord_400|><|box_end|>. Descriptions are raw "
    "class text; bbox coords are four coord tokens in x1 y1 x2 y2 order."
)
USER_PROMPT = (
    "Locate each clearly visible object instance in the image. Return compact rows "
    "using this exact pattern: <|object_ref_start|>{desc}<|object_ref_end|>"
    "<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
    "<|box_end|>. Use one row per object; concatenate rows directly with no "
    "separator and do not insert newline characters. Restrict `desc` to this "
    "COCO-80 class list: person, bicycle, car, motorcycle, airplane, bus, train, "
    "truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, "
    "bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, "
    "umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, "
    "baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, "
    "wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, "
    "broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, "
    "bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, "
    "microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, "
    "teddy bear, hair drier, toothbrush."
)


class Wave7PostrunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.context = dict(context or {})
        super().__init__(message)


class _BoundedByteTail:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.total_bytes = 0
        self._tail = bytearray()
        self._lock = threading.Lock()

    def append(self, value: bytes) -> None:
        with self._lock:
            self.total_bytes += len(value)
            self._tail.extend(value)
            overflow = len(self._tail) - self.maximum_bytes
            if overflow > 0:
                del self._tail[:overflow]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            encoded = bytes(self._tail)
            total = self.total_bytes
        text = encoded.decode("utf-8", errors="replace")
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > self.maximum_bytes:
            text = text_bytes[-self.maximum_bytes :].decode("utf-8", errors="ignore")
        return {
            "tail": text,
            "total_bytes": total,
            "truncated": total > self.maximum_bytes,
            "tail_cap_bytes": self.maximum_bytes,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Wave7PostrunError(
            "identity file is unreadable",
            code="wave7_postrun.identity_drift",
            context={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_loads(encoded: bytes, *, owner: str) -> Any:
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7PostrunError(
            f"{owner} exceeds the JSON size bound",
            code="wave7_postrun.json_oversize",
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant: {value}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise Wave7PostrunError(
            f"{owner} is not strict JSON",
            code="wave7_postrun.json_malformed",
            context={"owner": owner, "error_type": type(exc).__name__},
        ) from exc


def _strict_json_file(path: Path, *, owner: str) -> dict[str, Any]:
    _regular_file(path, owner=owner)
    value = _strict_json_loads(path.read_bytes(), owner=owner)
    if not isinstance(value, dict):
        raise Wave7PostrunError(
            f"{owner} must be one JSON object", code="wave7_postrun.json_schema"
        )
    return value


def _strict_jsonl(path: Path, *, owner: str) -> list[dict[str, Any]]:
    _regular_file(path, owner=owner)
    encoded = path.read_bytes()
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7PostrunError(
            f"{owner} exceeds the JSONL size bound",
            code="wave7_postrun.json_oversize",
        )
    lines = encoded.splitlines()
    if len(lines) > MAX_JSONL_ROWS or any(not line.strip() for line in lines):
        raise Wave7PostrunError(
            f"{owner} has invalid JSONL topology",
            code="wave7_postrun.output_schema",
        )
    rows = [_strict_json_loads(line, owner=f"{owner} row") for line in lines]
    if not all(isinstance(row, dict) for row in rows):
        raise Wave7PostrunError(
            f"{owner} rows must be objects", code="wave7_postrun.output_schema"
        )
    return rows


def _canonical_path(value: str | Path, *, owner: str, must_exist: bool) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise Wave7PostrunError(
            f"{owner} must be absolute", code="wave7_postrun.path_topology"
        )
    resolved = requested.resolve(strict=must_exist)
    if requested != resolved:
        raise Wave7PostrunError(
            f"{owner} must already be canonical",
            code="wave7_postrun.path_topology",
            context={"path": str(requested), "resolved": str(resolved)},
        )
    cursor = resolved if resolved.exists() else resolved.parent
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise Wave7PostrunError(
                f"{owner} cannot traverse symlinks",
                code="wave7_postrun.path_topology",
            )
        cursor = cursor.parent
    return resolved


def _regular_file(path: Path, *, owner: str) -> None:
    try:
        info = path.stat()
    except OSError as exc:
        raise Wave7PostrunError(
            f"{owner} is missing", code="wave7_postrun.input_missing"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise Wave7PostrunError(
            f"{owner} must be one ordinary file",
            code="wave7_postrun.path_topology",
        )


def _contained_real_descendant(
    root: Path,
    relative: Path,
    *,
    owner: str,
    expect_directory: bool,
) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise Wave7PostrunError(
            f"{owner} relative path is unsafe",
            code="wave7_postrun.path_topology",
        )
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise Wave7PostrunError(
                f"{owner} descendant component is missing",
                code="wave7_postrun.path_topology",
                context={"path": str(current)},
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise Wave7PostrunError(
                f"{owner} descendant component cannot be a symlink",
                code="wave7_postrun.path_topology",
                context={"path": str(current)},
            )
    resolved = current.resolve(strict=True)
    if resolved != current or not resolved.is_relative_to(root):
        raise Wave7PostrunError(
            f"{owner} escaped its r5 root",
            code="wave7_postrun.path_topology",
            context={"path": str(current), "resolved": str(resolved)},
        )
    if expect_directory and not stat.S_ISDIR(current.stat().st_mode):
        raise Wave7PostrunError(
            f"{owner} must be a directory",
            code="wave7_postrun.path_topology",
        )
    return current


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    return {
        **unsigned,
        "receipt_payload_sha256": _sha256_bytes(canonical_json_bytes(unsigned)),
    }


def _authenticate_receipt(
    path: Path, *, schema: str, status: str, owner: str
) -> dict[str, Any]:
    payload, _ = _authenticate_receipt_snapshot(
        path, schema=schema, status=status, owner=owner
    )
    return payload


def _authenticate_receipt_snapshot(
    path: Path, *, schema: str, status: str, owner: str
) -> tuple[dict[str, Any], str]:
    _regular_file(path, owner=owner)
    encoded = path.read_bytes()
    payload = _strict_json_loads(encoded, owner=owner)
    if not isinstance(payload, dict):
        raise Wave7PostrunError(
            f"{owner} must be one JSON object", code="wave7_postrun.json_schema"
        )
    observed = payload.get("receipt_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    recomputed = _sha256_bytes(canonical_json_bytes(unsigned))
    if (
        not _is_digest(observed)
        or observed != recomputed
        or payload.get("schema") != schema
        or payload.get("status") != status
    ):
        raise Wave7PostrunError(
            f"{owner} signature, schema, or status is invalid",
            code="wave7_postrun.receipt_authentication",
        )
    return payload, _sha256_bytes(encoded)


def _binding_matches(
    binding: Any,
    *,
    path: Path,
    payload: Mapping[str, Any],
    file_sha256: str | None = None,
) -> bool:
    if not isinstance(binding, Mapping):
        return False
    required = {
        "path",
        "file_sha256",
        "receipt_payload_sha256",
        "schema",
        "status",
        "validated",
        "mismatches",
    }
    return (
        set(binding) == required
        and binding.get("path") == str(path)
        and binding.get("file_sha256") == (file_sha256 or _sha256_file(path))
        and binding.get("receipt_payload_sha256")
        == payload.get("receipt_payload_sha256")
        and binding.get("schema") == FINAL_COMPARISON_SCHEMA
        and binding.get("status") == "passed"
        and binding.get("validated") is True
        and binding.get("mismatches") == []
    )


def authenticate_inputs(
    *, r5_root: Path, sequence_receipt: Path, final_comparison: Path
) -> dict[str, Any]:
    root = _canonical_path(r5_root, owner="r5 root", must_exist=True)
    if not root.is_dir() or root.is_symlink():
        raise Wave7PostrunError(
            "r5 root must be one real directory", code="wave7_postrun.path_topology"
        )
    sequence_path = _canonical_path(
        sequence_receipt, owner="sequence receipt", must_exist=True
    )
    final_path = _canonical_path(
        final_comparison, owner="final comparison", must_exist=True
    )
    if (
        sequence_path == final_path
        or not sequence_path.is_relative_to(root)
        or not final_path.is_relative_to(root)
    ):
        raise Wave7PostrunError(
            "input receipts must be distinct leaves under r5",
            code="wave7_postrun.path_topology",
        )
    sequence, sequence_file_sha256 = _authenticate_receipt_snapshot(
        sequence_path, schema=SEQUENCE_SCHEMA, status="passed", owner="sequence receipt"
    )
    final, final_file_sha256 = _authenticate_receipt_snapshot(
        final_path,
        schema=FINAL_COMPARISON_SCHEMA,
        status="passed",
        owner="final comparison",
    )
    if final.get("mismatches") != [] or not _binding_matches(
        sequence.get("final_receipt"),
        path=final_path,
        payload=final,
        file_sha256=final_file_sha256,
    ):
        raise Wave7PostrunError(
            "sequence does not bind the passed comparator-v2 receipt",
            code="wave7_postrun.receipt_binding",
        )
    expected_child = _contained_real_descendant(
        root,
        Path("runs/resume_child"),
        owner="resume child run",
        expect_directory=True,
    )
    run = final.get("runs", {}).get("resume_child")
    validation = final.get("event_validation", {}).get("resume_child")
    if (
        not isinstance(run, Mapping)
        or run.get("path") != str(expected_child)
        or not isinstance(validation, Mapping)
        or validation.get("event_versions") != [2]
        or not isinstance(validation.get("events"), list)
        or len(validation["events"]) != 1
    ):
        raise Wave7PostrunError(
            "final comparison does not identify the exact resume child event",
            code="wave7_postrun.checkpoint_selection",
        )
    event = validation["events"][0]
    progress = event.get("committed_progress") if isinstance(event, Mapping) else None
    identity = (
        event.get("inference_payload_identity") if isinstance(event, Mapping) else None
    )
    if (
        not isinstance(progress, Mapping)
        or progress.get("completed_steps") != 5
        or progress.get("optimizer_update_status") != "applied"
        or progress.get("finite_status") != "finite"
        or not isinstance(identity, Mapping)
        or identity.get("schema") != CHECKPOINT_PUBLICATION_SCHEMA
        or identity.get("schema_version") != 2
    ):
        raise Wave7PostrunError(
            "resume child latest committed inference identity is not step 5",
            code="wave7_postrun.checkpoint_selection",
        )
    checkpoint = _contained_real_descendant(
        root,
        Path("runs/resume_child/checkpoints/step-5"),
        owner="resume child step-5 checkpoint",
        expect_directory=True,
    )
    try:
        admitted = admit_inference_checkpoint_payload_identity(checkpoint, identity)
    except BaseException as exc:
        raise Wave7PostrunError(
            "committed step-5 inference payload failed admission",
            code="wave7_postrun.payload_identity",
            context={"error_code": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    if admitted != dict(identity):
        raise Wave7PostrunError(
            "committed step-5 identity admission was not exact",
            code="wave7_postrun.payload_identity",
        )
    return {
        "r5_root": root,
        "sequence_path": sequence_path,
        "sequence": sequence,
        "sequence_file_sha256": sequence_file_sha256,
        "final_path": final_path,
        "final": final,
        "final_file_sha256": final_file_sha256,
        "checkpoint_dir": checkpoint,
        "inference_payload_identity": dict(identity),
    }


def build_derived_config(*, r5_root: Path, checkpoint_dir: Path) -> dict[str, Any]:
    postrun_root = (r5_root / "postrun").resolve()
    return {
        "schema_version": 1,
        "run": {
            "name": "final-child-infer",
            "artifact_root": str(postrun_root),
            "output_dir": "final-child-infer",
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(BASE_MODEL),
            "dtype": "bf16",
            "processor": {"do_resize": False},
        },
        "data": {"input_jsonl": str(FIXTURE)},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "geo_sorted",
            "object_order_seed": None,
            "assistant_format": "object_box_closed",
            "prompt": {"system": SYSTEM_PROMPT, "user": USER_PROMPT},
        },
        "backend": {
            "type": "hf",
            "hf": {
                "attn_implementation": "flash_attention_2",
                "patch_embed_linearization": "enabled",
            },
        },
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {
            "write_token_trace": True,
            "write_parse_diagnostics": True,
            "include_raw_model_logprob": False,
        },
        "debug": {"smoke": True, "dry_run": False},
        "adapter": {
            "type": "dora",
            "path": str(checkpoint_dir / "adapter"),
            "name": "default",
        },
        "embedding_delta": {"path": str(checkpoint_dir / "special_token_embeddings")},
    }


def _write_config(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    target = assert_absent_artifact_target(path)
    try:
        encoded = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=False)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if target.exists() and target.is_file():
            target.unlink()
        raise
    resolved = load_infer_config(target)
    if resolved.config_dict != dict(value):
        raise Wave7PostrunError(
            "derived inference config did not resolve exactly",
            code="wave7_postrun.config_semantics",
        )
    return {
        "path": str(target),
        "file_sha256": _sha256_file(target),
        "resolved_fingerprint": resolved.fingerprint,
        "config": resolved.config_dict,
    }


def _bounded_nvidia_csv(command: list[str], *, owner: str) -> list[str]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )
    except OSError as exc:
        raise Wave7PostrunError(
            f"{owner} could not be sampled",
            code="wave7_postrun.gpu_inventory",
        ) from exc
    try:
        stdout, stderr, drain_threads = _start_output_drainers(
            process,
            stdout_cap_bytes=MAX_GPU_CSV_STDOUT_BYTES,
            stderr_cap_bytes=MAX_GPU_CSV_STDERR_BYTES,
            thread_name_prefix="wave7-postrun-gpu-inventory",
        )
    except BaseException:
        _terminate_process_group(process)
        raise
    timed_out = False
    return_code: int | None = None
    cleanup_errors: list[dict[str, Any]] = []
    try:
        return_code = process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            _terminate_process_group(process)
        except BaseException as exc:
            cleanup_errors.append(_error_evidence(exc))
        for thread in drain_threads:
            thread.join(timeout=OUTPUT_DRAIN_JOIN_SECONDS)
    stdout_snapshot = stdout.snapshot()
    stderr_snapshot = stderr.snapshot()
    alive_drainers = [thread.name for thread in drain_threads if thread.is_alive()]
    if alive_drainers:
        raise Wave7PostrunError(
            f"{owner} output drainers did not terminate",
            code="wave7_postrun.gpu_inventory",
            context={"threads": alive_drainers},
        )
    for stream_name, snapshot, maximum_bytes in (
        ("stdout", stdout_snapshot, MAX_GPU_CSV_STDOUT_BYTES),
        ("stderr", stderr_snapshot, MAX_GPU_CSV_STDERR_BYTES),
    ):
        if snapshot["total_bytes"] > maximum_bytes:
            raise Wave7PostrunError(
                f"{owner} {stream_name} is oversized",
                code="wave7_postrun.gpu_output_oversize",
                context={
                    "stream": stream_name,
                    "total_bytes": snapshot["total_bytes"],
                    "maximum_bytes": maximum_bytes,
                },
            )
    diagnostics = {
        "return_code": return_code,
        "timed_out": timed_out,
        "stdout": stdout_snapshot,
        "stderr": stderr_snapshot,
        "cleanup_errors": cleanup_errors,
    }
    if timed_out or return_code != 0 or cleanup_errors:
        raise Wave7PostrunError(
            f"{owner} could not be sampled",
            code="wave7_postrun.gpu_inventory",
            context=diagnostics,
        )
    encoded = stdout_snapshot["tail"]
    return [line.strip() for line in encoded.splitlines() if line.strip()]


def _gpu_sample() -> dict[str, Any]:
    gpu_lines = _bounded_nvidia_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        owner="GPU device inventory",
    )
    compute_lines = _bounded_nvidia_csv(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        owner="GPU compute inventory",
    )
    devices: list[dict[str, Any]] = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or not all(fields):
            raise Wave7PostrunError(
                "GPU device row is malformed", code="wave7_postrun.gpu_inventory"
            )
        index, gpu_uuid, total, used, utilization = fields
        if not all(value.isdigit() for value in (index, total, used, utilization)):
            raise Wave7PostrunError(
                "GPU device row values are malformed",
                code="wave7_postrun.gpu_inventory",
            )
        total_value = int(total)
        used_value = int(used)
        devices.append(
            {
                "index": int(index),
                "gpu_uuid": gpu_uuid,
                "memory_total_mib": total_value,
                "memory_used_mib": used_value,
                "memory_headroom_mib": total_value - used_value,
                "utilization_gpu_percent": int(utilization),
            }
        )
    if (
        not devices
        or [row["index"] for row in devices] != list(range(len(devices)))
        or len({row["gpu_uuid"] for row in devices}) != len(devices)
    ):
        raise Wave7PostrunError(
            "GPU physical inventory is not exact", code="wave7_postrun.gpu_inventory"
        )
    known_uuids = {row["gpu_uuid"] for row in devices}
    compute: list[dict[str, Any]] = []
    for line in compute_lines:
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) != 2
            or fields[0] not in known_uuids
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
        ):
            raise Wave7PostrunError(
                "GPU compute row is malformed", code="wave7_postrun.gpu_inventory"
            )
        compute.append({"gpu_uuid": fields[0], "driver_pid": int(fields[1])})
    compute.sort(key=lambda row: (row["gpu_uuid"], row["driver_pid"]))
    if len({(row["gpu_uuid"], row["driver_pid"]) for row in compute}) != len(compute):
        raise Wave7PostrunError(
            "GPU compute inventory contains duplicates",
            code="wave7_postrun.gpu_inventory",
        )
    return {
        "checked_at": _utc_now(),
        "gpu_inventory": devices,
        "compute_inventory": compute,
    }


def select_gpu(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_devices = first.get("gpu_inventory")
    second_devices = second.get("gpu_inventory")
    first_compute = first.get("compute_inventory")
    second_compute = second.get("compute_inventory")
    if not all(
        isinstance(value, list)
        for value in (first_devices, second_devices, first_compute, second_compute)
    ):
        raise Wave7PostrunError(
            "stable GPU samples are malformed", code="wave7_postrun.gpu_selection"
        )
    first_topology = [(row.get("index"), row.get("gpu_uuid")) for row in first_devices]
    second_topology = [
        (row.get("index"), row.get("gpu_uuid")) for row in second_devices
    ]
    if first_topology != second_topology or first_compute != second_compute:
        raise Wave7PostrunError(
            "GPU UUID or process inventory changed between stable samples",
            code="wave7_postrun.gpu_stability",
        )
    candidates: list[dict[str, Any]] = []
    for left, right in zip(first_devices, second_devices, strict=True):
        if all(
            row.get("memory_total_mib") == GPU_MEMORY_TOTAL_MIB
            and isinstance(row.get("memory_used_mib"), int)
            and not isinstance(row.get("memory_used_mib"), bool)
            and row["memory_used_mib"] <= GPU_MEMORY_USED_CEILING_MIB
            and row.get("memory_headroom_mib", -1) >= GPU_MEMORY_HEADROOM_FLOOR_MIB
            for row in (left, right)
        ):
            candidates.append(left)
    if not candidates:
        raise Wave7PostrunError(
            "no physical GPU satisfies the shared-GPU capacity contract",
            code="wave7_postrun.gpu_capacity",
        )
    selected = min(candidates, key=lambda row: int(row["index"]))
    uuid = selected["gpu_uuid"]
    baseline = [dict(row) for row in first_compute if row.get("gpu_uuid") == uuid]
    return {
        "physical_index": int(selected["index"]),
        "gpu_uuid": uuid,
        "preexisting_compute_inventory": baseline,
    }


def _validate_selection(selection: Mapping[str, Any]) -> None:
    if set(selection) != {
        "physical_index",
        "gpu_uuid",
        "preexisting_compute_inventory",
    }:
        raise Wave7PostrunError(
            "selected GPU binding fields are not exact",
            code="wave7_postrun.gpu_selection",
        )
    index = selection["physical_index"]
    gpu_uuid = selection["gpu_uuid"]
    inventory = selection["preexisting_compute_inventory"]
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid
        or not isinstance(inventory, list)
    ):
        raise Wave7PostrunError(
            "selected GPU binding is malformed",
            code="wave7_postrun.gpu_selection",
        )
    keys: list[tuple[str, int]] = []
    for row in inventory:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"gpu_uuid", "driver_pid"}
            or row.get("gpu_uuid") != gpu_uuid
            or isinstance(row.get("driver_pid"), bool)
            or not isinstance(row.get("driver_pid"), int)
            or row["driver_pid"] <= 0
        ):
            raise Wave7PostrunError(
                "preexisting selected-GPU process binding is malformed",
                code="wave7_postrun.gpu_selection",
            )
        keys.append((gpu_uuid, row["driver_pid"]))
    if len(keys) != len(set(keys)):
        raise Wave7PostrunError(
            "preexisting selected-GPU process binding has duplicates",
            code="wave7_postrun.gpu_selection",
        )


def _stable_gpu_selection() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = _gpu_sample()
    first_ns = time.monotonic_ns()
    time.sleep(GPU_STABILITY_SECONDS)
    second = _gpu_sample()
    second_ns = time.monotonic_ns()
    if second_ns - first_ns < int(GPU_STABILITY_SECONDS * 1_000_000_000):
        raise Wave7PostrunError(
            "GPU stability interval was shorter than declared",
            code="wave7_postrun.gpu_stability",
        )
    return select_gpu(first, second), [first, second]


def _proc_stat_parent_pid(encoded: str) -> int:
    """Parse PPID without splitting the parenthesized comm field."""

    closing_parenthesis = encoded.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("proc stat comm field is unterminated")
    trailing_fields = encoded[closing_parenthesis + 1 :].split()
    if len(trailing_fields) < 2:
        raise ValueError("proc stat lacks state and parent pid")
    parent_pid = int(trailing_fields[1])
    if parent_pid < 0:
        raise ValueError("proc stat parent pid is negative")
    return parent_pid


def _descendant_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            encoded = (entry / "stat").read_text(encoding="utf-8")
            parents[int(entry.name)] = _proc_stat_parent_pid(encoded)
        except (OSError, ValueError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def _wait_for_gpu_observation(
    *, process: subprocess.Popen[Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    baseline = {
        int(row["driver_pid"]) for row in selection["preexisting_compute_inventory"]
    }
    deadline = time.monotonic() + GPU_OBSERVATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sample = _gpu_sample()
        descendants = _descendant_pids(process.pid)
        added = [
            dict(row)
            for row in sample["compute_inventory"]
            if row["gpu_uuid"] == selection["gpu_uuid"]
            and int(row["driver_pid"]) not in baseline
            and int(row["driver_pid"]) in descendants
        ]
        if added:
            return added
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise Wave7PostrunError(
        "inference child was not observed on the selected GPU UUID",
        code="wave7_postrun.gpu_observation",
    )


def _assert_gpu_cleanup(
    *,
    observed: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    task_keys = {(str(row["gpu_uuid"]), int(row["driver_pid"])) for row in observed}
    baseline_keys: set[tuple[str, int]] = set()
    selected_uuid: str | None = None
    if selection is not None:
        _validate_selection(selection)
        selected_uuid = str(selection["gpu_uuid"])
        baseline_keys = {
            (selected_uuid, int(row["driver_pid"]))
            for row in selection["preexisting_compute_inventory"]
        }
    samples: list[dict[str, Any]] = []
    for index in range(2):
        sample = _gpu_sample()
        survivors = [
            dict(row)
            for row in sample["compute_inventory"]
            if (
                (str(row["gpu_uuid"]), int(row["driver_pid"])) in task_keys
                or (
                    selected_uuid is not None
                    and str(row["gpu_uuid"]) == selected_uuid
                    and (selected_uuid, int(row["driver_pid"])) not in baseline_keys
                )
            )
        ]
        if survivors:
            raise Wave7PostrunError(
                "a task-added GPU PID survived inference cleanup",
                code="wave7_postrun.gpu_cleanup",
                context={"survivors": survivors},
            )
        samples.append(sample)
        if index == 0:
            time.sleep(GPU_STABILITY_SECONDS)
    return samples


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise Wave7PostrunError(
            "inference process group ownership cannot be verified",
            code="wave7_postrun.process_cleanup",
        ) from exc
    return True


def _wait_process_group_gone(process: subprocess.Popen[Any], *, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        process.poll()
        if not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    term_grace: float = 10.0,
    kill_grace: float = 10.0,
) -> None:
    if not _process_group_exists(process.pid):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_process_group_gone(process, seconds=term_grace):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_process_group_gone(process, seconds=kill_grace):
        raise Wave7PostrunError(
            "inference process group survived bounded TERM/KILL cleanup",
            code="wave7_postrun.process_cleanup",
            context={"process_group_id": process.pid},
        )


def _drain_stream(stream: Any, sink: _BoundedByteTail) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            if not isinstance(chunk, bytes):
                chunk = str(chunk).encode("utf-8", errors="replace")
            sink.append(chunk)
    finally:
        try:
            stream.close()
        except BaseException:
            pass


def _start_output_drainers(
    process: subprocess.Popen[Any],
    *,
    stdout_cap_bytes: int = MAX_CHILD_OUTPUT_TAIL_BYTES,
    stderr_cap_bytes: int = MAX_CHILD_OUTPUT_TAIL_BYTES,
    thread_name_prefix: str = "wave7-postrun",
) -> tuple[_BoundedByteTail, _BoundedByteTail, tuple[threading.Thread, ...]]:
    if process.stdout is None or process.stderr is None:
        raise Wave7PostrunError(
            "inference output pipes were not created",
            code="wave7_postrun.output_drain",
        )
    stdout = _BoundedByteTail(stdout_cap_bytes)
    stderr = _BoundedByteTail(stderr_cap_bytes)
    threads = (
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout),
            name=f"{thread_name_prefix}-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr),
            name=f"{thread_name_prefix}-stderr",
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    return stdout, stderr, threads


def _error_evidence(exc: BaseException) -> dict[str, Any]:
    return {
        "code": getattr(exc, "code", type(exc).__name__),
        "message": str(exc),
        "context": dict(getattr(exc, "context", {}) or {}),
    }


def launch_inference(
    *,
    config_path: Path,
    selection: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    _validate_selection(selection)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise Wave7PostrunError(
            "inference timeout must be positive and finite",
            code="wave7_postrun.timeout",
        )
    argv = [sys.executable, "-m", "src.infer", "--config", str(config_path)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selection["physical_index"]),
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )
    except OSError as exc:
        raise Wave7PostrunError(
            "fresh inference child could not be launched",
            code="wave7_postrun.launch",
        ) from exc
    try:
        stdout_sink, stderr_sink, drain_threads = _start_output_drainers(process)
    except BaseException:
        _terminate_process_group(process)
        raise
    observed: list[dict[str, Any]] = []
    primary_failure: Wave7PostrunError | None = None
    cleanup_errors: list[dict[str, Any]] = []
    return_code: int | None = None
    timed_out = False
    try:
        observed = _wait_for_gpu_observation(process=process, selection=selection)
        return_code = process.wait(timeout=float(timeout_seconds))
        if return_code != 0:
            raise Wave7PostrunError(
                "fresh inference child exited nonzero",
                code="wave7_postrun.child_nonzero",
                context={"return_code": return_code},
            )
    except subprocess.TimeoutExpired:
        timed_out = True
        primary_failure = Wave7PostrunError(
            "fresh inference child exceeded its timeout",
            code="wave7_postrun.child_timeout",
        )
    except Wave7PostrunError as exc:
        primary_failure = exc
    except BaseException as exc:
        primary_failure = Wave7PostrunError(
            "fresh inference child failed unexpectedly",
            code="wave7_postrun.child_runtime",
            context={"error_type": type(exc).__name__},
        )
    finally:
        try:
            _terminate_process_group(process)
        except BaseException as exc:
            cleanup_errors.append(_error_evidence(exc))
        try:
            cleanup_samples = _assert_gpu_cleanup(
                observed=observed,
                selection=selection,
            )
        except BaseException as exc:
            cleanup_samples = []
            cleanup_errors.append(_error_evidence(exc))
        for thread in drain_threads:
            thread.join(timeout=OUTPUT_DRAIN_JOIN_SECONDS)
        alive_drainers = [thread.name for thread in drain_threads if thread.is_alive()]
        if alive_drainers:
            cleanup_errors.append(
                {
                    "code": "wave7_postrun.output_drain",
                    "message": "inference output drainers did not terminate",
                    "context": {"threads": alive_drainers},
                }
            )
    diagnostics = {
        "return_code": return_code,
        "post_cleanup_return_code": process.poll(),
        "timed_out": timed_out,
        "timeout_seconds": float(timeout_seconds),
        "stdout": stdout_sink.snapshot(),
        "stderr": stderr_sink.snapshot(),
        "cleanup_errors": cleanup_errors,
    }
    if cleanup_errors:
        first = cleanup_errors[0]
        if primary_failure is not None:
            diagnostics["primary_failure"] = _error_evidence(primary_failure)
        raise Wave7PostrunError(
            str(first["message"]),
            code=str(first["code"]),
            context=diagnostics,
        ) from primary_failure
    if primary_failure is not None:
        raise Wave7PostrunError(
            str(primary_failure),
            code=primary_failure.code,
            context={**primary_failure.context, **diagnostics},
        ) from primary_failure
    return {
        "argv": argv,
        "shell": False,
        "attempt_count": 1,
        "return_code": return_code,
        "selected_gpu": dict(selection),
        "observed_task_compute_inventory": observed,
        "cleanup_samples": cleanup_samples,
        "duration_seconds": time.monotonic() - started,
        "timing_semantics": "descriptive_only_not_compared",
        "stdout": diagnostics["stdout"],
        "stderr": diagnostics["stderr"],
        "failure": None,
    }


def _inventory(root: Path) -> frozenset[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise Wave7PostrunError(
                "output contains a non-ordinary entry",
                code="wave7_postrun.output_inventory",
            )
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return frozenset(result)


def _contains_mapping(value: Any, expected: Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        if dict(value) == dict(expected):
            return True
        return any(_contains_mapping(child, expected) for child in value.values())
    if isinstance(value, list):
        return any(_contains_mapping(child, expected) for child in value)
    return False


def _runtime_delta_matches_inspector(value: Any, delta: Mapping[str, Any]) -> bool:
    if _contains_mapping(value, delta):
        return True
    if not isinstance(value, Mapping) or value.get("status") != "loaded":
        return False
    identity = value.get("identity")
    load = value.get("load")
    semantic = delta.get("semantic_identity")
    root = delta.get("root")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(load, Mapping)
        or not isinstance(semantic, Mapping)
        or not isinstance(root, str)
    ):
        return False
    root_path = Path(root)
    return (
        identity.get("status") == "validated"
        and identity.get("delta_path") == root
        and identity.get("metadata_path")
        == str(root_path / "special_token_embeddings.json")
        and identity.get("metadata") == dict(semantic)
        and identity.get("base_model_path") == semantic.get("base_model_path")
        and load.get("loaded") is True
        and load.get("tensor_path")
        == str(root_path / "special_token_embeddings.safetensors")
        and load.get("metadata_path")
        == str(root_path / "special_token_embeddings.json")
        and load.get("tensor_shape") == semantic.get("tensor_shape")
        and load.get("source_tensor_dtype") == semantic.get("tensor_dtype")
    )


def _expected_fixture_identity() -> dict[str, Any]:
    examples = load_raw_examples(FIXTURE)
    if len(examples) != 1:
        raise Wave7PostrunError(
            "postrun fixture must contain exactly one raw example",
            code="wave7_postrun.fixture_identity",
        )
    example = examples[0]
    if example.example_id != EXPECTED_ROW_ID:
        raise Wave7PostrunError(
            "postrun fixture example id drifted",
            code="wave7_postrun.fixture_identity",
        )
    return {
        "row_id": EXPECTED_ROW_ID,
        "row_index": 0,
        "example_id": EXPECTED_ROW_ID,
        "image_path": str(example.image.path),
        "image_width": example.image.width,
        "image_height": example.image.height,
        "gt": [obj.to_artifact_dict() for obj in example.objects],
    }


def _fixture_identity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in FIXTURE_IDENTITY_FIELDS}


def validate_outputs(
    *, output_dir: Path, config_value: Mapping[str, Any], checkpoint_dir: Path
) -> dict[str, Any]:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise Wave7PostrunError(
            "canonical inference output directory is missing",
            code="wave7_postrun.output_inventory",
        )
    try:
        validate_scored_artifact_set(output_dir)
    except BaseException as exc:
        raise Wave7PostrunError(
            "production scored artifact validation failed",
            code="wave7_postrun.scored_artifacts",
            context={"error_code": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    observed_inventory = _inventory(output_dir)
    if observed_inventory != OUTPUT_FILES:
        raise Wave7PostrunError(
            "fresh inference output inventory is not exact",
            code="wave7_postrun.output_inventory",
            context={
                "missing": sorted(OUTPUT_FILES - observed_inventory),
                "unexpected": sorted(observed_inventory - OUTPUT_FILES),
            },
        )
    summary = _strict_json_file(output_dir / SUMMARY_NAME, owner="inference summary")
    manifest = _strict_json_file(output_dir / MANIFEST_NAME, owner="run manifest")
    resolved = _strict_json_file(
        output_dir / "configs/resolved.json", owner="resolved inference config"
    )
    raw_rows = _strict_jsonl(output_dir / RAW_NAME, owner="raw inference artifact")
    scored_rows = _strict_jsonl(
        output_dir / SCORED_NAME, owner="scored inference artifact"
    )
    for name in (TOKEN_TRACE_NAME, PARSE_DIAGNOSTICS_NAME, IMAGE_PLAN_NAME):
        _strict_jsonl(output_dir / name, owner=name)
    expected_fixture_identity = _expected_fixture_identity()
    raw_identity = (
        {} if len(raw_rows) != 1 else _fixture_identity_projection(raw_rows[0])
    )
    scored_identity = (
        {} if len(scored_rows) != 1 else _fixture_identity_projection(scored_rows[0])
    )
    if (
        summary.get("row_count") != 1
        or summary.get("raw_row_count") != 1
        or summary.get("scored_row_count") != 1
        or summary.get("scored_artifact_materialized") is not True
        or len(raw_rows) != 1
        or len(scored_rows) != 1
        or raw_identity != expected_fixture_identity
        or scored_identity != expected_fixture_identity
        or raw_identity != scored_identity
    ):
        raise Wave7PostrunError(
            "one-row artifact counts or fixture identity drifted",
            code="wave7_postrun.output_rows",
        )
    model_identity = manifest.get("model_identity")
    if (
        manifest.get("backend") != "hf"
        or manifest.get("backend_mode") != "generate"
        or not isinstance(model_identity, Mapping)
        or not model_identity
    ):
        raise Wave7PostrunError(
            "run manifest lacks the required HF runtime-model identity",
            code="wave7_postrun.backend_identity",
        )
    if not isinstance(resolved.get("config"), Mapping) or dict(
        resolved["config"]
    ) != dict(config_value):
        raise Wave7PostrunError(
            "resolved inference config differs from the derived semantics",
            code="wave7_postrun.config_semantics",
        )
    try:
        adapter = inspect_dora_adapter_payload(
            checkpoint_dir / "adapter",
            expected_base_model_path=BASE_MODEL,
        )
        delta = inspect_special_token_embedding_delta_payload(
            checkpoint_dir / "special_token_embeddings",
            expected_base_model_path=BASE_MODEL,
        )
        checkpoint_manifest = load_inference_checkpoint_payload_manifest(checkpoint_dir)
    except BaseException as exc:
        raise Wave7PostrunError(
            "checkpoint payload inspectors failed after inference",
            code="wave7_postrun.payload_identity",
            context={"error_code": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    without_root_adapter = {
        key: value for key, value in adapter.items() if key != "root"
    }
    without_root_delta = {key: value for key, value in delta.items() if key != "root"}
    if (
        checkpoint_manifest.get("adapter", {}).get("inspector_identity")
        != without_root_adapter
        or checkpoint_manifest.get("special_token_embedding_delta", {}).get(
            "inspector_identity"
        )
        != without_root_delta
        or not _contains_mapping(manifest.get("adapter_identity"), adapter)
        or not _runtime_delta_matches_inspector(
            manifest.get("embedding_delta_identity"), delta
        )
    ):
        raise Wave7PostrunError(
            "artifact adapter/delta identity differs from current inspectors",
            code="wave7_postrun.runtime_payload_identity",
        )
    return {
        "inventory": sorted(observed_inventory),
        "summary": summary,
        "row_id": EXPECTED_ROW_ID,
        "backend": {"type": "hf", "mode": "generate"},
        "model_identity": dict(model_identity),
        "adapter_identity": adapter,
        "embedding_delta_identity": delta,
        "resolved_config": resolved,
    }


def semantic_projection(value: Any) -> Any:
    """Drop explicitly descriptive observations from a receipt projection."""

    descriptive = {
        "checked_at",
        "completed_at",
        "duration_seconds",
        "published_at",
        "sample_monotonic_ns",
        "started_at",
        "timing_semantics",
    }
    if isinstance(value, Mapping):
        return {
            key: semantic_projection(child)
            for key, child in value.items()
            if key not in descriptive
        }
    if isinstance(value, list):
        return [semantic_projection(child) for child in value]
    return value


def _identity(
    path: Path,
    payload: Mapping[str, Any] | None = None,
    *,
    frozen_file_sha256: str | None = None,
) -> dict[str, Any]:
    observed_file_sha256 = _sha256_file(path)
    if frozen_file_sha256 is not None and observed_file_sha256 != frozen_file_sha256:
        raise Wave7PostrunError(
            "authenticated input receipt bytes drifted",
            code="wave7_postrun.identity_drift",
            context={
                "path": str(path),
                "expected_file_sha256": frozen_file_sha256,
                "observed_file_sha256": observed_file_sha256,
            },
        )
    result = {"path": str(path), "file_sha256": observed_file_sha256}
    if payload is not None:
        result["payload_sha256"] = payload["receipt_payload_sha256"]
        result["schema"] = payload["schema"]
        result["status"] = payload["status"]
    return result


def _authenticated_input_identities(admitted: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence_receipt": _identity(
            admitted["sequence_path"],
            admitted["sequence"],
            frozen_file_sha256=admitted["sequence_file_sha256"],
        ),
        "final_comparison": _identity(
            admitted["final_path"],
            admitted["final"],
            frozen_file_sha256=admitted["final_file_sha256"],
        ),
    }


def _publish_signed(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = assert_absent_artifact_target(path)
    signed = _signed_payload(payload)
    write_strict_json_atomic(target, signed)
    reloaded = _authenticate_receipt(
        target,
        schema=str(payload["schema"]),
        status=str(payload["status"]),
        owner="published postrun receipt",
    )
    if reloaded != signed:
        raise Wave7PostrunError(
            "published receipt reload differs from its signed payload",
            code="wave7_postrun.publication_reload",
        )
    return signed


def _publish_final_or_sidecar(
    *,
    receipt_path: Path,
    sidecar_path: Path,
    payload: Mapping[str, Any],
    marker_path: Path,
) -> dict[str, Any]:
    try:
        receipt = _publish_signed(receipt_path, payload)
    except BaseException as exc:
        if receipt_path.exists() or receipt_path.is_symlink():
            raise Wave7PostrunError(
                "postrun receipt publication is ambiguous",
                code="wave7_postrun.publication_collision",
            ) from exc
        sidecar = {
            "schema": PUBLICATION_FAILURE_SCHEMA,
            "status": "failed",
            "receipt_path": str(receipt_path),
            "attempt_marker": _identity(marker_path),
            "error_code": getattr(exc, "code", type(exc).__name__),
            "error_type": type(exc).__name__,
            "published_at": _utc_now(),
        }
        try:
            _publish_signed(sidecar_path, sidecar)
        except BaseException as sidecar_exc:
            raise Wave7PostrunError(
                "postrun receipt and publication-failure sidecar both failed",
                code="wave7_postrun.publication_failure",
                context={"sidecar_error_type": type(sidecar_exc).__name__},
            ) from sidecar_exc
        raise Wave7PostrunError(
            "postrun receipt publication failed; a signed sidecar was published",
            code="wave7_postrun.publication_failure",
        ) from exc
    if sidecar_path.exists() or sidecar_path.is_symlink():
        raise Wave7PostrunError(
            "receipt and publication-failure sidecar are mutually exclusive",
            code="wave7_postrun.publication_collision",
        )
    return receipt


def _publish_failure_sidecar(
    *,
    sidecar_path: Path,
    receipt_path: Path,
    marker_path: Path,
    exc: BaseException,
) -> dict[str, Any]:
    return _publish_signed(
        sidecar_path,
        {
            "schema": FAILURE_SCHEMA,
            "status": "failed",
            "receipt_path": str(receipt_path),
            "attempt_marker": _identity(marker_path),
            "failure": _error_evidence(exc),
            "failed_at": _utc_now(),
        },
    )


def run(
    *,
    r5_root: Path,
    sequence_receipt: Path,
    final_comparison: Path,
    output: Path,
    timeout_seconds: float = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
) -> int:
    targets: dict[str, Path] | None = None
    try:
        admitted = authenticate_inputs(
            r5_root=r5_root,
            sequence_receipt=sequence_receipt,
            final_comparison=final_comparison,
        )
        root = admitted["r5_root"]
        postrun_root = root / "postrun"
        if postrun_root.exists():
            if not postrun_root.is_dir() or postrun_root.is_symlink():
                raise Wave7PostrunError(
                    "postrun root has unsafe topology",
                    code="wave7_postrun.path_topology",
                )
        else:
            postrun_root.mkdir(mode=0o755)
        targets = {name: postrun_root / name for name in POSTRUN_LEAVES}
        requested_output = _canonical_path(
            output, owner="postrun receipt output", must_exist=False
        )
        if requested_output != targets["postrun-receipt.json"]:
            raise Wave7PostrunError(
                "output must be the canonical postrun receipt leaf",
                code="wave7_postrun.path_topology",
            )
        for name, target in targets.items():
            if target.exists() or target.is_symlink():
                raise Wave7PostrunError(
                    "postrun target collision",
                    code="wave7_postrun.target_collision",
                    context={"target": name, "path": str(target)},
                )
        config_value = build_derived_config(
            r5_root=root, checkpoint_dir=admitted["checkpoint_dir"]
        )
        config_identity = _write_config(targets["final-child-infer.yaml"], config_value)
        selection, gpu_samples = _stable_gpu_selection()
        input_identities = _authenticated_input_identities(admitted)
        marker = _publish_signed(
            targets["inference-attempt-marker.json"],
            {
                "schema": MARKER_SCHEMA,
                "status": "started",
                "sequence_receipt": input_identities["sequence_receipt"],
                "final_comparison": input_identities["final_comparison"],
                "checkpoint_dir": str(admitted["checkpoint_dir"]),
                "inference_payload_identity": admitted["inference_payload_identity"],
                "config": config_identity,
                "selected_gpu": selection,
                "gpu_admission_samples": gpu_samples,
                "argv": [
                    sys.executable,
                    "-m",
                    "src.infer",
                    "--config",
                    str(targets["final-child-infer.yaml"]),
                ],
                "shell": False,
                "attempt_count": 1,
                "started_at": _utc_now(),
                "timing_semantics": "descriptive_only_not_compared",
            },
        )
        _authenticated_input_identities(admitted)
        admit_inference_checkpoint_payload_identity(
            admitted["checkpoint_dir"], admitted["inference_payload_identity"]
        )
        launch = launch_inference(
            config_path=targets["final-child-infer.yaml"],
            selection=selection,
            timeout_seconds=timeout_seconds,
        )
        input_identities = _authenticated_input_identities(admitted)
        admit_inference_checkpoint_payload_identity(
            admitted["checkpoint_dir"], admitted["inference_payload_identity"]
        )
        outputs = validate_outputs(
            output_dir=targets["final-child-infer"],
            config_value=config_value,
            checkpoint_dir=admitted["checkpoint_dir"],
        )
        receipt = _publish_final_or_sidecar(
            receipt_path=requested_output,
            sidecar_path=targets["postrun-receipt.publication-failure.json"],
            marker_path=targets["inference-attempt-marker.json"],
            payload={
                "schema": RECEIPT_SCHEMA,
                "status": "passed",
                "sequence_receipt": input_identities["sequence_receipt"],
                "final_comparison": input_identities["final_comparison"],
                "attempt_marker": _identity(
                    targets["inference-attempt-marker.json"], marker
                ),
                "checkpoint_dir": str(admitted["checkpoint_dir"]),
                "inference_payload_identity": admitted["inference_payload_identity"],
                "config": config_identity,
                "launch": launch,
                "outputs": outputs,
                "claim_scope": {
                    "establishes": [
                        "fresh_hf_inference_completed_from_resume_child_step_5",
                        "canonical_one_row_scored_artifacts_validated",
                        "checkpoint_payload_identity_stable_before_and_after_inference",
                    ],
                    "does_not_establish": [
                        "prediction_validity_or_quality",
                        "throughput_or_resource_promotion",
                    ],
                },
                "completed_at": _utc_now(),
                "timing_semantics": "descriptive_only_not_compared",
            },
        )
        return 0 if receipt.get("status") == "passed" else 2
    except BaseException as exc:
        if targets is not None:
            marker_path = targets["inference-attempt-marker.json"]
            receipt_path = targets["postrun-receipt.json"]
            sidecar_path = targets["postrun-receipt.publication-failure.json"]
            if (
                marker_path.is_file()
                and not marker_path.is_symlink()
                and not receipt_path.exists()
                and not receipt_path.is_symlink()
                and not sidecar_path.exists()
                and not sidecar_path.is_symlink()
            ):
                try:
                    _publish_failure_sidecar(
                        sidecar_path=sidecar_path,
                        receipt_path=receipt_path,
                        marker_path=marker_path,
                        exc=exc,
                    )
                except BaseException as sidecar_exc:
                    print(
                        "wave7_postrun.failure_sidecar: "
                        f"{getattr(sidecar_exc, 'code', type(sidecar_exc).__name__)}: "
                        f"{sidecar_exc}",
                        file=sys.stderr,
                    )
        print(
            f"{getattr(exc, 'code', 'wave7_postrun.failure')}: {exc}",
            file=sys.stderr,
        )
        return 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--r5-root", type=Path, required=True)
    run_parser.add_argument("--sequence-receipt", type=Path, required=True)
    run_parser.add_argument("--final-comparison", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SECONDS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        r5_root=args.r5_root,
        sequence_receipt=args.sequence_receipt,
        final_comparison=args.final_comparison,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
