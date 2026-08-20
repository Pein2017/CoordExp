from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from scripts.probes.coordexp_swift import wave7_exact_resume_interrupt as interrupt
from scripts.probes.coordexp_swift import wave7_exact_resume_compare as compare
from src.artifacts.training_state import (
    TrainingStatePublication,
    build_resume_compatibility_projection,
    publish_training_state,
    serialize_rank_training_state,
)
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
)
from src.training.exact_resume import build_exact_resume_identities


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/probes/coordexp_swift/wave7_exact_resume_compare.py"
WORLD_SIZE = 8
PROVENANCE_SHA256 = "a" * 64
COMMIT = "b" * 40
INTERRUPT_SOURCE_SHA256 = (
    "7018e7e177d74940e9397caa6152f5335d55e68bb1e98e55705725a749ccdbdd"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _resolved_config(
    run_dir: Path,
    *,
    name: str,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    config_source = run_dir.parent / "configs" / f"{name}.json"
    _json(config_source, {"name": name, "schema_version": 1})
    resume = {
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "mode": "exact_same_world_size",
    }
    return {
        "config": {
            "checkpoint": {"save_final": True, "steps": [3, 5]},
            "data": {"train": {"path": "/data/train.jsonl"}},
            "eval": {"forward": {"steps": [3]}},
            "model": {"precision": "bf16"},
            "optimizer": {"lr": 0.001},
            "packing": {"policy": "source_order_next_fit"},
            "resume": resume,
            "run": {
                "artifact_root": str(run_dir.parent),
                "name": name,
                "output_dir": str(run_dir),
            },
            "runtime": {"world_size": WORLD_SIZE},
            "training": {
                "forward_input_provider_mode": "synchronous",
                "precision": "bf16",
                "seed": 17,
            },
        },
        "resolution": {
            "entry_config_path": f"/configs/{name}.yaml",
            "fingerprint": hashlib.sha256(name.encode()).hexdigest(),
            "loader_version": "coordexp-swift-config-v1",
            "path_origins": (
                {}
                if checkpoint_dir is None
                else {
                    "resume.checkpoint_dir": {
                        "declared_path": str(checkpoint_dir),
                        "declaring_config_path": f"/configs/{name}.yaml",
                        "resolved_path": str(checkpoint_dir),
                    }
                }
            ),
            "schema_version": 1,
            "sources": [
                {
                    "path": str(config_source),
                    "sha256": _sha256(config_source),
                }
            ],
        },
    }


def _identities(config: dict[str, Any]) -> dict[str, str]:
    return dict(
        build_exact_resume_identities(
            base_model="1" * 64,
            cache="2" * 64,
            dependencies="3" * 64,
            policy="4" * 64,
            resolved_config=config,
            topology="5" * 64,
            trainable_surface="6" * 64,
        )
    )


def _rank_payload(rank: int, step: int, *, tensor_drift: bool = False):
    torch.manual_seed(1000 + step * 10 + rank)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for _ in range(step):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01 * (rank + 1))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if tensor_drift and rank == 0:
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
    python_state = random.Random(3000 + step * 10 + rank).getstate()
    numpy_state = np.random.RandomState(4000 + step * 10 + rank).get_state()
    cpu_generator = torch.Generator().manual_seed(5000 + step * 10 + rank)
    cuda_generator = torch.Generator().manual_seed(6000 + step * 10 + rank)
    next_index = step * 2
    common = {
        "checkpoint_step": step,
        "next_rank_local_micro_step": next_index,
        "rank": rank,
        "resolved_grad_accum_steps": 2,
        "resolved_max_steps": 5,
        "total_rank_local_micro_steps": 10,
        "world_size": WORLD_SIZE,
    }
    cursor = {
        "data": {
            "schema": "coordexp-swift-rank-data-cursor-v1",
            **common,
            "runtime_counters": {
                "optimizer_step_count": step,
                "scheduler_step_count": step,
                "zero_grad_count": step,
            },
        },
        "pack": {
            "schema": "coordexp-swift-rank-pack-cursor-v1",
            **common,
            "cache": {
                "fingerprint": "2" * 64,
                "format_version": "coordexp-swift-pack-cache-v3",
            },
            "next_pack": (
                None
                if step == 5
                else {
                    "example_ids": [f"rank-{rank}-example-{next_index}"],
                    "pack_index": next_index,
                    "sequence_length": 4,
                }
            ),
        },
    }
    return serialize_rank_training_state(
        rank=rank,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        python_rng_state=python_state,
        numpy_rng_state=numpy_state,
        torch_cpu_rng_state=cpu_generator.get_state(),
        torch_cuda_rng_states=(cuda_generator.get_state(),),
        cursor=cursor,
        next_rank_local_micro_step=next_index,
    )


def _write_inference_payload(checkpoint_dir: Path) -> None:
    adapter = checkpoint_dir / "adapter"
    adapter.mkdir(parents=True)
    _json(adapter / "adapter_config.json", {"peft_type": "LORA", "use_dora": True})
    save_file(
        {
            "base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 3),
            "base_model.model.q_proj.lora_B.default.weight": torch.ones(3, 2),
            "base_model.model.q_proj.lora_magnitude_vector.default.weight": (
                torch.ones(3)
            ),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    delta = checkpoint_dir / "special_token_embeddings"
    delta.mkdir()
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.ones(1, 2)},
        str(delta / "special_token_embeddings.safetensors"),
    )
    _json(
        delta / "special_token_embeddings.json",
        {
            "base_config_sha256": "7" * 64,
            "base_model_path": "/models/base",
            "semantics": SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
            "tensor_dtype": "float32",
            "tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
            "tensor_shape": [1, 2],
            "tie_word_embeddings": True,
            "token_ids": [100],
            "token_strings": ["<coord_100>"],
            "tokenizer_sha256": "8" * 64,
        },
    )


def _publish_checkpoint(
    run_dir: Path,
    *,
    step: int,
    config: dict[str, Any],
    run_id: str,
    segment_id: str,
    continuation_index: int,
    tensor_drift: bool = False,
) -> Any:
    checkpoint_dir = run_dir / "checkpoints" / f"step-{step}"
    checkpoint_dir.mkdir(parents=True)
    _write_inference_payload(checkpoint_dir)
    return publish_training_state(
        checkpoint_dir,
        TrainingStatePublication(
            parent_run_id=run_id,
            parent_segment_id=segment_id,
            checkpoint_step=step,
            continuation_index=continuation_index,
            world_size=WORLD_SIZE,
            identities=_identities(config),
            scheduler_applicable=True,
            scaler_applicable=False,
            rank_payloads=tuple(
                _rank_payload(rank, step, tensor_drift=tensor_drift)
                for rank in range(WORLD_SIZE)
            ),
            accumulation_microstep=0,
            resolved_config=config,
            resume_compatibility=build_resume_compatibility_projection(config),
        ),
    )


def _train_row(
    step: int,
    *,
    aggregate_metric_drift: bool = False,
    loss_drift: bool = False,
) -> dict[str, Any]:
    return {
        "acc_top1": 0.5 + step / 100 + (0.004 if aggregate_metric_drift else 0.0),
        "finite_status": "finite",
        "input_build_seconds": 0.01 * step,
        "input_wait_seconds": 0.001 * step,
        "loss/base_ce/weighted": 1.0 / step,
        "loss/total": 1.0 / step + (0.1 if loss_drift else 0.0),
        "lr/group_0": 0.001 * (0.9**step),
        "micro_step_count": 2,
        "non_finite_fields": [],
        "optimizer_update_status": "applied",
        "per_rank_measurement": {
            str(rank): {"step_duration_seconds": 0.1 * step + rank / 1000}
            for rank in range(WORLD_SIZE)
        },
        "resource/cpu_max_rss_bytes": float(1000 + step),
        "split": "train",
        "step": step,
        "step_duration_seconds": 0.1 * step,
    }


def _eval_row() -> dict[str, Any]:
    return {
        "acc_top1": 0.75,
        "loss/total": 0.5,
        "non_finite_fields": [],
        "pack_count": 8,
        "split": "eval",
        "step": 3,
        "trigger_reasons": ["scheduled"],
    }


def _run_receipt(
    run_dir: Path,
    *,
    name: str,
    run_id: str,
    segment_id: str,
    config: dict[str, Any],
    completed_steps: int,
    continuation: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    phase = {
        "completed_at": "2026-08-10T00:00:00+00:00",
        "duration_seconds": 1.25,
        "started_at": "2026-08-10T00:00:00+00:00",
        "status": "completed",
    }
    publication_steps = (
        [3, 5]
        if completed_steps == 5 and continuation["continuation_index"] == 0
        else [completed_steps]
    )
    publication_events = []
    for step in publication_steps:
        checkpoint_dir = run_dir / f"checkpoints/step-{step}"
        manifest_path = checkpoint_dir / "training_state/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        publication_events.append(
            {
                "checkpoint_identity": {
                    "checkpoint_step": step,
                    "resolved_path": str(checkpoint_dir),
                    "training_state_aggregate_digest": manifest["aggregate_digest"],
                    "training_state_manifest_file_sha256": _sha256(manifest_path),
                },
                "checkpoint_path": f"checkpoints/step-{step}",
                "completed_at": f"2026-08-10T00:00:0{step}+00:00",
                "duration_clock": "monotonic",
                "duration_seconds": 0.25 + step / 100,
                "exact_training_state_enabled": True,
                "failure_code": None,
                "is_final": step == 5,
                "started_at": f"2026-08-10T00:00:0{step}+00:00",
                "status": "completed",
                "step": step,
            }
        )
    return {
        "artifact_root": str(run_dir.parent),
        "checkpoint_event_count": (
            2 if completed_steps == 5 and continuation["continuation_index"] == 0 else 1
        ),
        "collision_outcome": "created",
        "completed_at": (
            "2026-08-10T00:00:01+00:00" if status == "completed" else None
        ),
        "completed_steps": completed_steps,
        "config_fingerprint": config["resolution"]["fingerprint"],
        "consumed_packs": completed_steps * 2,
        "continuation": continuation,
        "created_at": "2026-08-10T00:00:00+00:00",
        "final_finite_status": "finite" if status == "completed" else None,
        "final_optimizer_update_status": ("applied" if status == "completed" else None),
        "forward_input_provider_mode": "synchronous",
        "forward_input_provider_resolution": {
            "requested_mode": "synchronous",
            "resolved_mode": "synchronous",
        },
        "materializations": {},
        "measurement": {
            "accepted_measured_steps": max(0, completed_steps - 2),
            "active_phase": None,
            "context": {
                "comparison_arm": name,
                "wall_clock_scope": "training_entry_to_terminal_artifact",
                "warmup_exclusion_steps": 2,
            },
            "entry_to_terminal": {
                "boundary": "training_entry_to_terminal_state_durable_before_measurement_annotation",
                "clock": "monotonic",
                "completed_at": (
                    "2026-08-10T00:00:01+00:00" if status == "completed" else None
                ),
                "duration_seconds": 5.0 if status == "completed" else None,
                "started_at": "2026-08-10T00:00:00+00:00",
                "status": "completed" if status == "completed" else "running",
            },
            "expected_measured_steps": 3,
            "failure_phase": None,
            "last_completed_phase": (
                "checkpoint_publication" if status == "completed" else None
            ),
            "phase_order": ["checkpoint_publication"] if status == "completed" else [],
            "phases": {"checkpoint_publication": phase}
            if status == "completed"
            else {},
            "checkpoint_publication_events": publication_events,
            "resource_high_water": None,
            "schema_version": 1,
            "steady_state_eligible": status == "completed",
            "terminal_phase": (
                "checkpoint_publication" if status == "completed" else None
            ),
            "terminal_phase_status": "completed" if status == "completed" else None,
        },
        "policy_identities": {
            "attention_proof": {"proof_policy": "first_micro_step"},
            "cache": {"train": {"fingerprint": "2" * 64}},
            "eval_reduction": {"effective_mode": "disjoint_shard"},
            "input_provider": {"mode": "synchronous"},
            "packing": {"policy": "source_order_next_fit"},
            "profile_sync_timings": {"enabled": False},
            "resume": {
                "mode": "exact_same_world_size",
                "supported_boundary": "optimizer_step_accumulation_zero",
                "world_size_policy": "exact_same_world_size",
            },
            "upstream_runtime_baseline": {"baseline_sha256": "9" * 64},
        },
        "provenance": {
            "dependencies": {"baseline_sha256": "9" * 64},
            "repository": {
                "commit": {"status": "available", "value": COMMIT},
                "execution_relevant_changes": {
                    "count": 1,
                    "path_classes": {"source": {"tracked": 1, "untracked": 0}},
                    "truncated": False,
                },
                "execution_relevant_digest": {
                    "status": "available",
                    "value": PROVENANCE_SHA256,
                },
                "state": "dirty",
                "tracked_changes_present": True,
                "untracked_changes_present": False,
            },
            "runtime": {"world_size": WORLD_SIZE},
            "schema_version": 3,
        },
        "resolved_config_path": "resolved_config.json",
        "resolved_max_steps": 5,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "run_name": name,
        "runtime": {"world_size": WORLD_SIZE},
        "status": status,
        "terminal_error": None,
        "updated_at": "2026-08-10T00:00:01+00:00",
        "warning_counts": {},
    }


def _write_logs(
    run_dir: Path,
    train_steps: range,
    *,
    aggregate_metric_drift: bool = False,
    include_eval: bool,
    loss_drift: bool = False,
) -> None:
    rows = [
        _train_row(
            step,
            aggregate_metric_drift=aggregate_metric_drift and step == 4,
            loss_drift=loss_drift and step == 4,
        )
        for step in train_steps
    ]
    if include_eval:
        rows.append(_eval_row())
    (run_dir / "logging.jsonl").write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _base_continuation(segment_id: str) -> dict[str, Any]:
    return {
        "continuation_index": 0,
        "parent": None,
        "schema_version": 1,
        "segment_id": segment_id,
    }


def _write_interrupt_evidence(
    root: Path,
    parent_run: Path,
    parent_manifest: Any,
) -> tuple[Path, Path]:
    marker = root / "interruption-marker.json"
    receipt = root / "termination-receipt.json"
    manifest_path = parent_run / "checkpoints/step-3/training_state/manifest.json"
    checkpoint = {
        "aggregate_digest": parent_manifest.aggregate_digest,
        "checkpoint_step": 3,
        "file_sha256": _sha256(manifest_path),
        "path": str(manifest_path),
        "rank_count": WORLD_SIZE,
        "size": manifest_path.stat().st_size,
        "world_size": WORLD_SIZE,
    }
    run_path = parent_run / "run.json"
    run_bytes = run_path.read_bytes()
    parent_payload = json.loads(run_bytes)
    checkpoint_publication_event = next(
        event
        for event in parent_payload["measurement"]["checkpoint_publication_events"]
        if event["step"] == 3
    )
    checkpoint_publication = {
        "event": checkpoint_publication_event,
        "run_file_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "run_path": str(run_path),
        "run_size": len(run_bytes),
    }
    source_owner = root / "source-owner.py"
    source_owner.write_text("SOURCE_IDENTITY = 1\n", encoding="utf-8")
    source_paths = (SCRIPT, Path(interrupt.__file__).resolve(), source_owner)
    source_hashes = [
        {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}
        for path in source_paths
    ]
    parent_config = json.loads(
        (parent_run / "resolved_config.json").read_text(encoding="utf-8")
    )
    config_paths = [Path(row["path"]) for row in parent_config["resolution"]["sources"]]
    config_hashes = [
        {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}
        for path in config_paths
    ]
    target_binding = {
        "artifact_root": str(parent_run.parent),
        "launcher_config": str(config_paths[0]),
        "resolved_config_fingerprint": parent_config["resolution"]["fingerprint"],
        "resolved_config_sources": parent_config["resolution"]["sources"],
        "run_dir": str(parent_run),
        "run_name": "interrupted-parent",
    }
    launcher_argv = ["python", "train.py", "--config", str(config_paths[0])]
    launcher_sha256 = hashlib.sha256(
        json.dumps(launcher_argv, separators=(",", ":")).encode()
    ).hexdigest()
    script_identity = {
        "path": str(Path(interrupt.__file__).resolve()),
        "sha256": _sha256(Path(interrupt.__file__).resolve()),
    }
    marker_payload = {
        "captured_pgid_set": [123],
        "captured_pid_set": [123],
        "captured_process_graph": [
            {
                "depth": 0,
                "pgid": 123,
                "pid": 123,
                "ppid": 1,
                "session_id": 123,
                "start_time_ticks": 1,
                "state": "S",
            }
        ],
        "checkpoint": checkpoint,
        "checkpoint_publication_event": checkpoint_publication_event,
        "checkpoint_publication_run": {
            "file_sha256": checkpoint_publication["run_file_sha256"],
            "path": checkpoint_publication["run_path"],
            "size": checkpoint_publication["run_size"],
        },
        "config_hashes": config_hashes,
        "gpu_compute_apps_baseline": [],
        "launch": {"pgid": 123, "pid": 123},
        "launcher": {"argv": launcher_argv, "argv_sha256": launcher_sha256},
        "marker_target": str(marker),
        "receipt_target": str(receipt),
        "schema": interrupt.MARKER_SCHEMA,
        "script": script_identity,
        "source_hashes": source_hashes,
        "target_binding": target_binding,
        "timestamp": "2026-08-10T00:00:00+00:00",
    }
    _json(marker, marker_payload)
    tree = interrupt._snapshot_run_tree(parent_run)
    receipt_payload = {
        "checkpoint": checkpoint,
        "checkpoint_publication": checkpoint_publication,
        "config_hashes": config_hashes,
        "duration_seconds": 1.0,
        "errors": [],
        "events": [],
        "finished_at": "2026-08-10T00:00:01+00:00",
        "launch": {
            "attempted": True,
            "argv": launcher_argv,
            "argv_sha256": launcher_sha256,
            "pgid": 123,
            "pid": 123,
            "returncode": -15,
        },
        "marker": {
            "expected_file_sha256": _sha256(marker),
            "file_sha256": _sha256(marker),
            "final_file_sha256": _sha256(marker),
            "path": str(marker),
            "published": True,
            "strict_payload_equal": True,
            "unchanged": True,
        },
        "postconditions": {
            "checkpoint_publication_event_unchanged": True,
            "checkpoint_publication_run_final_file_sha256": checkpoint_publication[
                "run_file_sha256"
            ],
            "final_json_absent": True,
            "late_write_detected": False,
            "manifest_final_aggregate_digest": parent_manifest.aggregate_digest,
            "manifest_final_file_sha256": _sha256(manifest_path),
            "manifest_unchanged": True,
            "max_logged_train_step": 3,
            "nvidia_compute_apps": [],
            "nvidia_compute_apps_added": [],
            "nvidia_compute_apps_baseline": [],
            "run_not_completed": True,
            "run_status": "initialized",
            "stability_after": {
                "entry_count": tree["entry_count"],
                "fingerprint": tree["fingerprint"],
            },
            "stability_before": {
                "entry_count": tree["entry_count"],
                "fingerprint": tree["fingerprint"],
            },
            "step_5_absent": True,
        },
        "request": {
            "expected_checkpoint_step": 3,
            "parent_run_dir": str(parent_run),
            "timeout_seconds": 10.0,
        },
        "schema": interrupt.RECEIPT_SCHEMA,
        "script": script_identity,
        "source_hashes": source_hashes,
        "started_at": "2026-08-10T00:00:00+00:00",
        "status": "passed",
        "termination": {
            "capture_completed_monotonic": 1.0,
            "captured_process_graph": marker_payload["captured_process_graph"],
            "cleanup_errors": [],
            "captured_pgids": [123],
            "captured_pids": [123],
            "duration_seconds": 0.1,
            "launcher_exited": True,
            "launcher_returncode": -15,
            "post_marker_discovered_pids": [],
            "reaped": [{"pid": 123, "wait_status": 15}],
            "remaining_pgids": [],
            "remaining_pids": [],
        },
        "target_binding": target_binding,
    }
    _json(receipt, receipt_payload)
    return marker, receipt


def _fixture(root: Path, *, mutation: str | None = None) -> dict[str, Path]:
    reference = (root / "uninterrupted").resolve()
    parent = (root / "interrupted-parent").resolve()
    child = (root / "resume-child").resolve()
    for path in (reference, parent, child):
        path.mkdir()
    reference_config = _resolved_config(
        reference, name="uninterrupted", checkpoint_dir=None
    )
    parent_config = _resolved_config(
        parent, name="interrupted-parent", checkpoint_dir=None
    )
    parent_checkpoint = parent / "checkpoints/step-3"
    child_config = _resolved_config(
        child, name="resume-child", checkpoint_dir=parent_checkpoint
    )
    reference_step3 = _publish_checkpoint(
        reference,
        step=3,
        config=reference_config,
        run_id="run-reference",
        segment_id="segment-reference",
        continuation_index=0,
    )
    _publish_checkpoint(
        reference,
        step=5,
        config=reference_config,
        run_id="run-reference",
        segment_id="segment-reference",
        continuation_index=0,
    )
    parent_step3 = _publish_checkpoint(
        parent,
        step=3,
        config=parent_config,
        run_id="run-parent",
        segment_id="segment-parent",
        continuation_index=0,
    )
    child_step5 = _publish_checkpoint(
        child,
        step=5,
        config=child_config,
        run_id="run-child",
        segment_id="segment-child",
        continuation_index=1,
        tensor_drift=mutation == "tensor_drift",
    )
    del reference_step3, child_step5
    _json(
        reference / "checkpoints/final.json",
        {"checkpoint_path": "checkpoints/step-5", "step": 5},
    )
    _json(
        child / "checkpoints/final.json",
        {"checkpoint_path": "checkpoints/step-5", "step": 5},
    )
    _json(
        reference / "checkpoints/best.json",
        {
            "checkpoint_path": "checkpoints/step-3",
            "selector": "acc_top1",
            "step": 3,
            "value": 0.75,
        },
    )
    _json(
        parent / "checkpoints/best.json",
        {
            "checkpoint_path": "checkpoints/step-3",
            "selector": "acc_top1",
            "step": 3,
            "value": 0.75,
        },
    )
    for run_dir, config in (
        (reference, reference_config),
        (parent, parent_config),
        (child, child_config),
    ):
        _json(run_dir / "resolved_config.json", config)
    _write_logs(reference, range(1, 6), include_eval=True)
    _write_logs(parent, range(1, 4), include_eval=True)
    _write_logs(
        child,
        range(4, 6),
        aggregate_metric_drift=mutation == "aggregate_metric_drift",
        include_eval=False,
        loss_drift=mutation == "loss_drift",
    )
    parent_manifest_path = parent / "checkpoints/step-3/training_state/manifest.json"
    parent_manifest_sha256 = _sha256(parent_manifest_path)
    child_continuation = {
        "continuation_index": 1,
        "parent": {
            "checkpoint_identity": {
                "checkpoint_step": 3,
                "resolved_path": str(parent_checkpoint),
                "training_state_aggregate_digest": parent_step3.manifest.aggregate_digest,
                "training_state_manifest_file_sha256": parent_manifest_sha256,
            },
            "continuation_index": 0,
            "run_id": "run-parent",
            "segment_id": "segment-parent",
        },
        "schema_version": 1,
        "segment_id": "segment-child",
    }
    if mutation == "lineage_mismatch":
        child_continuation["parent"]["checkpoint_identity"][
            "training_state_aggregate_digest"
        ] = "f" * 64
    receipts = (
        (
            reference,
            _run_receipt(
                reference,
                name="uninterrupted",
                run_id="run-reference",
                segment_id="segment-reference",
                config=reference_config,
                completed_steps=5,
                continuation=_base_continuation("segment-reference"),
                status="completed",
            ),
        ),
        (
            parent,
            _run_receipt(
                parent,
                name="interrupted-parent",
                run_id="run-parent",
                segment_id="segment-parent",
                config=parent_config,
                completed_steps=3,
                continuation=_base_continuation("segment-parent"),
                status="initialized",
            ),
        ),
        (
            child,
            _run_receipt(
                child,
                name="resume-child",
                run_id="run-child",
                segment_id="segment-child",
                config=child_config,
                completed_steps=5,
                continuation=child_continuation,
                status="completed",
            ),
        ),
    )
    for run_dir, receipt_payload in receipts:
        _json(run_dir / "run.json", receipt_payload)
    if mutation == "best_missing":
        (parent / "checkpoints/best.json").unlink()
    elif mutation == "best_drift":
        _json(
            parent / "checkpoints/best.json",
            {
                "checkpoint_path": "checkpoints/step-3",
                "selector": "acc_top1",
                "step": 3,
                "value": 0.70,
            },
        )
    elif mutation in {
        "publication_extra_field",
        "publication_identity_mismatch",
        "publication_invalid",
        "publication_missing",
    }:
        run_path = reference / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if mutation == "publication_missing":
            run["measurement"]["checkpoint_publication_events"] = []
        elif mutation == "publication_invalid":
            run["measurement"]["checkpoint_publication_events"][0][
                "duration_seconds"
            ] = -0.1
        elif mutation == "publication_identity_mismatch":
            run["measurement"]["checkpoint_publication_events"][0][
                "checkpoint_identity"
            ]["training_state_aggregate_digest"] = "f" * 64
        else:
            run["measurement"]["checkpoint_publication_events"][0]["unexpected"] = True
        _json(run_path, run)
    marker, receipt = _write_interrupt_evidence(root, parent, parent_step3.manifest)
    if mutation == "missing_rank":
        shutil.rmtree(child / "checkpoints/step-5/training_state/rank-00007")
    elif mutation == "cursor_drift":
        path = child / "checkpoints/step-5/training_state/rank-00000/cursor.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "rng_drift":
        path = child / "checkpoints/step-5/training_state/rank-00000/rng-python.bin"
        path.write_bytes(path.read_bytes() + b"x")
    elif mutation == "pruning":
        shutil.rmtree(parent_checkpoint)
    elif mutation == "extra_fields":
        value = json.loads((reference / "run.json").read_text(encoding="utf-8"))
        value["unexpected"] = True
        _json(reference / "run.json", value)
    elif mutation == "inventory_mismatch":
        (root / "source-owner.py").write_text("SOURCE_IDENTITY = 2\n", encoding="utf-8")
    elif mutation == "config_inventory_mismatch":
        Path(parent_config["resolution"]["sources"][0]["path"]).write_text(
            '{"name":"changed"}\n', encoding="utf-8"
        )
    elif mutation == "interrupt_script_mismatch":
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        marker_value["script"]["sha256"] = "f" * 64
        _json(marker, marker_value)
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_value["script"]["sha256"] = "f" * 64
        marker_hash = _sha256(marker)
        for key in (
            "expected_file_sha256",
            "file_sha256",
            "final_file_sha256",
        ):
            receipt_value["marker"][key] = marker_hash
        _json(receipt, receipt_value)
    elif mutation == "interrupt_publication_mismatch":
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_value["checkpoint_publication"]["event"]["duration_seconds"] += 1.0
        _json(receipt, receipt_value)
    elif mutation == "launcher_hash_mismatch":
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        marker_value["launcher"]["argv_sha256"] = "f" * 64
        _json(marker, marker_value)
        receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_value["launch"]["argv_sha256"] = "f" * 64
        marker_hash = _sha256(marker)
        for key in (
            "expected_file_sha256",
            "file_sha256",
            "final_file_sha256",
        ):
            receipt_value["marker"][key] = marker_hash
        _json(receipt, receipt_value)
    elif mutation == "target_binding_mismatch":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["target_binding"]["resolved_config_fingerprint"] = "f" * 64
        _json(receipt, value)
    return {
        "child": child,
        "marker": marker,
        "output": root / "comparison.json",
        "parent": parent,
        "receipt": receipt,
        "reference": reference,
    }


def _compare_args(
    paths: dict[str, Path], *, source_mismatch: bool = False
) -> list[str]:
    source_hash = (
        "f" * 64
        if source_mismatch
        else (_sha256(SCRIPT) if SCRIPT.is_file() else "0" * 64)
    )
    return [
        "--uninterrupted-run-dir",
        str(paths["reference"]),
        "--interrupted-parent-run-dir",
        str(paths["parent"]),
        "--resume-child-run-dir",
        str(paths["child"]),
        "--interruption-marker",
        str(paths["marker"]),
        "--termination-receipt",
        str(paths["receipt"]),
        "--output",
        str(paths["output"]),
        "--expected-source-sha256",
        source_hash,
        "--expected-interrupt-source-sha256",
        INTERRUPT_SOURCE_SHA256,
        "--expected-provenance-sha256",
        PROVENANCE_SHA256,
    ]


def _run_compare(paths: dict[str, Path], *, source_mismatch: bool = False):
    command = [
        sys.executable,
        str(SCRIPT),
        *_compare_args(paths, source_mismatch=source_mismatch),
    ]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_accepted_fixture_publishes_bounded_passed_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run_compare(paths)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["mismatches"] == []
    assert receipt["runs"]["uninterrupted"]["role"] == "uninterrupted"
    assert receipt["runs"]["interrupted_parent"]["role"] == "interrupted_parent"
    assert receipt["runs"]["resume_child"]["role"] == "resume_child"
    assert receipt["lineage"]["continuation_index"] == 1
    assert receipt["state_comparisons"]["step_3"]["status"] == "passed"
    assert receipt["state_comparisons"]["step_5"]["status"] == "passed"
    assert receipt["run_contract"]["run_counts"]["interrupted_parent"] == {
        "checkpoint_event_count": 1,
        "completed_steps": 3,
        "consumed_packs": 6,
    }
    assert receipt["log_comparison"]["excluded_observations"]["resume_child"][0][
        "values"
    ]["step_duration_seconds"] == pytest.approx(0.4)
    assert receipt["checkpoint_selection"]["status"] == "passed"
    assert receipt["checkpoint_selection"]["combined_parent_child"]["step"] == 3
    assert receipt["publication_measurements"]["status"] == "passed"
    assert (
        receipt["publication_measurements"]["cadence_cost"]["total_checkpoint_count"]
        == 4
    )
    assert receipt["claim_boundaries"]["inference_validation"] == (
        "bounded_structural_production_validators_not_runtime_model_load"
    )
    assert receipt["claim_boundaries"]["task_8_2_completion"] == (
        "not_claimed_without_independent_runtime_inference_load"
    )
    snapshots = receipt["parent_tree_toctou"]
    assert snapshots["initial"] == snapshots["prepublication"]
    assert snapshots["prepublication"] == snapshots["final_precommit"]
    assert receipt["claim_boundaries"]["cadence_benefit"] == (
        "not_claimed_beyond_five_step_smoke"
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_rank", None),
        ("cursor_drift", None),
        ("rng_drift", None),
        ("tensor_drift", None),
        ("loss_drift", "wave7_compare.logging_value"),
        ("aggregate_metric_drift", "wave7_compare.logging_value"),
        ("lineage_mismatch", "wave7_compare.lineage_mismatch"),
        ("pruning", "wave7_compare.parent_tree_changed"),
        ("extra_fields", "wave7_compare.schema"),
        ("best_missing", "wave7_compare.checkpoint_selection"),
        ("best_drift", "wave7_compare.checkpoint_selection"),
        ("publication_missing", "wave7_compare.checkpoint_publication"),
        ("publication_invalid", "wave7_compare.checkpoint_publication"),
        ("publication_identity_mismatch", "wave7_compare.checkpoint_publication"),
        ("publication_extra_field", "wave7_compare.checkpoint_publication"),
        ("inventory_mismatch", "wave7_compare.interrupt_inventory"),
        ("config_inventory_mismatch", "wave7_compare.interrupt_inventory"),
        ("interrupt_script_mismatch", "wave7_compare.interrupt_inventory"),
        (
            "interrupt_publication_mismatch",
            "wave7_compare.interruption_binding",
        ),
        ("launcher_hash_mismatch", "wave7_compare.interruption_binding"),
        ("target_binding_mismatch", "wave7_compare.interruption_binding"),
    ],
)
def test_contract_mutations_publish_failed_receipt(
    tmp_path: Path,
    mutation: str,
    expected_code: str | None,
) -> None:
    paths = _fixture(tmp_path, mutation=mutation)

    result = _run_compare(paths)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["mismatches"]
    if expected_code is not None:
        assert expected_code in {row["code"] for row in receipt["mismatches"]}


def test_expected_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run_compare(paths, source_mismatch=True)

    assert result.returncode == 1
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert {row["code"] for row in receipt["mismatches"]} == {
        "wave7_compare.source_mismatch"
    }


def test_output_inside_parent_is_rejected_without_parent_mutation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths["output"] = paths["parent"] / "comparison.json"
    before = interrupt._snapshot_run_tree(paths["parent"])

    result = _run_compare(paths)

    assert result.returncode == 2
    assert not paths["output"].exists()
    after = interrupt._snapshot_run_tree(paths["parent"])
    assert after["fingerprint"] == before["fingerprint"]
    assert after["entry_count"] == before["entry_count"]


def test_marker_and_receipt_must_be_disjoint_without_output_publication(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths["marker"] = paths["receipt"]

    result = _run_compare(paths)

    assert result.returncode == 2
    assert not paths["output"].exists()


def test_joint_alternate_interrupt_source_and_rebound_evidence_fails_preoutput(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    original_path = Path(interrupt.__file__).resolve()
    alternate_path = tmp_path / "alternate_interrupt.py"
    alternate_path.write_bytes(
        original_path.read_bytes() + b"\n# alternate live source\n"
    )
    alternate_identity = {
        "path": str(alternate_path),
        "sha256": _sha256(alternate_path),
        "size": alternate_path.stat().st_size,
    }
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    for payload in (marker, receipt):
        payload["script"] = {
            "path": alternate_identity["path"],
            "sha256": alternate_identity["sha256"],
        }
        payload["source_hashes"] = [
            alternate_identity if row["path"] == str(original_path) else row
            for row in payload["source_hashes"]
        ]
    _json(paths["marker"], marker)
    marker_hash = _sha256(paths["marker"])
    for key in (
        "expected_file_sha256",
        "file_sha256",
        "final_file_sha256",
    ):
        receipt["marker"][key] = marker_hash
    _json(paths["receipt"], receipt)
    monkeypatch.setattr(compare.interrupt, "__file__", str(alternate_path))

    result = compare.main(_compare_args(paths))

    assert result == 2
    assert not paths["output"].exists()


def test_publish_boundary_parent_race_never_leaves_durable_passed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    request = compare.parse_request(_compare_args(paths))
    original_publish = compare._publish
    injected = False

    def publish_then_race(path: Path, payload: dict[str, Any]) -> None:
        nonlocal injected
        original_publish(path, payload)
        if not injected:
            injected = True
            (paths["parent"] / "late-write.txt").write_text(
                "injected at publication boundary\n", encoding="utf-8"
            )

    monkeypatch.setattr(compare, "_publish", publish_then_race)

    result = compare.execute(request)

    assert result != 0
    if paths["output"].exists():
        receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert receipt["status"] == "failed"


def test_hardlink_commit_gap_parent_race_never_leaves_durable_passed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    request = compare.parse_request(_compare_args(paths))
    original_commit = compare._commit_staged_receipt
    injected = False

    def commit_after_parent_write(
        stage: Path,
        output: Path,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal injected
        if not injected:
            injected = True
            (paths["parent"] / "hardlink-gap-write.txt").write_text(
                "injected on entry to hardlink commit\n", encoding="utf-8"
            )
        return original_commit(stage, output, payload, **kwargs)

    monkeypatch.setattr(compare, "_commit_staged_receipt", commit_after_parent_write)

    result = compare.execute(request)

    assert result != 0
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "wave7_compare.parent_tree_changed" in {
        row["code"] for row in receipt["mismatches"]
    }
