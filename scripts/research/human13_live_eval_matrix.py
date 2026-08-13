#!/usr/bin/env python3
"""Plan or execute the Human-13 full-panel checkpoint evaluation matrix.

This module is intentionally only an orchestration seam.  Model loading and
decoding remain in :mod:`human13_live_eval`; dry-run never imports a backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_human13_k_union_manifest import load_manifest
from scripts.research.human13_live_eval import (
    checkpoint_payload_sha256,
    evaluate_hf_checkpoint,
    source_outputs_from_manifest,
    write_outputs_jsonl,
)

MILESTONES = (1, 2, 4, 8, 16)


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve(strict=True).read_bytes()).hexdigest()


def _hash_file_or_contract(path: Path | None, contract: Mapping[str, object]) -> str:
    if path is not None and path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_milestones(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError("milestones must be a non-empty subset of 1,2,4,8,16")
    if any(isinstance(item, bool) or item not in MILESTONES for item in result):
        raise ValueError("milestones must be a subset of 1,2,4,8,16")
    return tuple(item for item in MILESTONES if item in result)


def _structural_readback(path: Path, milestone: int) -> dict[str, object]:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir() or root.name != f"step-{milestone}" or root.parent.name != "checkpoints":
        raise ValueError(f"checkpoint {root} does not match checkpoints/step-{milestone}")
    children = {item.name for item in root.iterdir()}
    if children != {"adapter", "special_token_embeddings"}:
        raise ValueError("checkpoint must contain only adapter and special_token_embeddings")
    for child in (root / "adapter", root / "special_token_embeddings"):
        if not child.is_dir() or not any(item.is_file() for item in child.rglob("*")):
            raise ValueError(f"checkpoint payload is empty: {child}")
    return {"status": "structural", "path": str(root), "children": sorted(children)}


@dataclass(frozen=True)
class Human13EvalJob:
    milestone: int
    checkpoint_path: Path
    output_path: Path
    checkpoint_payload_sha256: str
    readback: Mapping[str, object]


@dataclass(frozen=True)
class Human13EvalPlan:
    manifest_path: Path
    manifest: Any
    manifest_sha256: str
    arm_id: str
    run_id: str
    run_root: Path
    output_dir: Path
    milestones: tuple[int, ...]
    resolved_arm_plan_sha256: str
    resolved_config_sha256: str
    jobs: tuple[Human13EvalJob, ...]
    source_outputs: tuple[Mapping[str, object], ...]
    execution_ready: bool = False
    actions: Mapping[str, int] = field(default_factory=lambda: {"model_load": 0, "decode": 0, "gpu_allocation": 0})


def _load_source_records(path: Path) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for line in path.expanduser().resolve(strict=True).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("source discovery JSONL rows must be objects")
        image_id = value.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id in records:
            raise ValueError("source discovery rows require unique integer image_id")
        records[image_id] = value
    return records


def _checkpoint_map(
    *,
    run_root: Path | None,
    explicit_checkpoints: Mapping[int, str | Path] | None,
    milestones: tuple[int, ...],
) -> tuple[Path, dict[int, Path]]:
    if (run_root is None) == (explicit_checkpoints is None):
        raise ValueError("provide exactly one of run_root or explicit_checkpoints")
    if run_root is not None:
        root = run_root.expanduser().resolve(strict=True)
        paths = {step: root / "checkpoints" / f"step-{step}" for step in milestones}
        return root, paths
    assert explicit_checkpoints is not None
    if set(explicit_checkpoints) != set(milestones):
        raise ValueError("explicit checkpoints must cover exactly the requested milestones")
    paths = {step: Path(explicit_checkpoints[step]).expanduser().resolve(strict=True) for step in milestones}
    common = Path(paths[milestones[0]])
    root = common.parent.parent if common.parent.name == "checkpoints" else common.parent
    return root, paths


def build_eval_plan(
    *,
    manifest_path: str | Path,
    arm_id: str,
    output_dir: str | Path,
    milestones: Sequence[int] = MILESTONES,
    run_root: str | Path | None = None,
    explicit_checkpoints: Mapping[int, str | Path] | None = None,
    source_discovery_records: Mapping[int, Mapping[str, Any]] | None = None,
    resolved_arm_plan: str | Path | None = None,
    resolved_config: str | Path | None = None,
) -> Human13EvalPlan:
    selected = _validate_milestones(milestones)
    if not arm_id:
        raise ValueError("arm_id must be non-empty")
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_file, require_full_panel=True)
    manifest_digest = manifest_sha256(manifest_file)
    root, paths = _checkpoint_map(
        run_root=None if run_root is None else Path(run_root),
        explicit_checkpoints=explicit_checkpoints,
        milestones=selected,
    )
    output_root = Path(output_dir).expanduser().resolve()
    run_id = root.name
    plan_digest = _hash_file_or_contract(Path(resolved_arm_plan) if resolved_arm_plan else None, {"arm_id": arm_id, "run_root": str(root)})
    config_digest = _hash_file_or_contract(Path(resolved_config) if resolved_config else None, {"surface": "hf-fp32-sdpa-batch1-rp1-full13"})
    jobs: list[Human13EvalJob] = []
    for step in selected:
        checkpoint = paths[step]
        payload_digest = checkpoint_payload_sha256(checkpoint)
        readback = _structural_readback(checkpoint, step)
        jobs.append(
            Human13EvalJob(
                milestone=step,
                checkpoint_path=checkpoint,
                output_path=output_root / f"{arm_id}.milestone-{step}.jsonl",
                checkpoint_payload_sha256=payload_digest,
                readback=readback,
            )
        )
    source_outputs: tuple[Mapping[str, object], ...] = ()
    if source_discovery_records is not None:
        source_outputs = tuple(
            source_outputs_from_manifest(
                manifest=manifest,
                manifest_sha256=manifest_digest,
                source_discovery_records=source_discovery_records,
                resolved_arm_plan_sha256=plan_digest,
                resolved_config_sha256=config_digest,
                run_root=str(root),
                checkpoint_path=str(getattr(manifest.binding.source, "checkpoint_path")),
            )
        )
    return Human13EvalPlan(
        manifest_path=manifest_file,
        manifest=manifest,
        manifest_sha256=manifest_digest,
        arm_id=arm_id,
        run_id=run_id,
        run_root=root,
        output_dir=output_root,
        milestones=selected,
        resolved_arm_plan_sha256=plan_digest,
        resolved_config_sha256=config_digest,
        jobs=tuple(jobs),
        source_outputs=source_outputs,
        actions={"model_load": 0, "decode": 0, "gpu_allocation": 0},
    )


def execute_eval_plan(
    plan: Human13EvalPlan,
    *,
    execute: bool = False,
    user_model_gpu_authority: bool = False,
    source_config_path: str | Path = "configs/coordexp_swift/infer/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml",
) -> dict[str, object]:
    if not execute:
        raise PermissionError("execution requires --execute")
    if not user_model_gpu_authority:
        raise PermissionError("execution requires --user-model-gpu-authority")
    receipt_path = plan.output_dir / f"{plan.arm_id}.receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite Human-13 receipt: {receipt_path}")
    started = time.perf_counter()
    outputs: list[dict[str, object]] = []
    for job in plan.jobs:
        rows = evaluate_hf_checkpoint(
            manifest=plan.manifest,
            manifest_sha256=plan.manifest_sha256,
            checkpoint_path=job.checkpoint_path,
            arm_id=plan.arm_id,
            milestone=job.milestone,
            run_id=plan.run_id,
            run_root=str(plan.run_root),
            resolved_arm_plan_sha256=plan.resolved_arm_plan_sha256,
            resolved_config_sha256=plan.resolved_config_sha256,
            source_config_path=source_config_path,
        )
        expected_images = tuple(int(image.image_id) for image in plan.manifest.images)
        observed_images = tuple(int(row["image_id"]) for row in rows)
        if len(rows) != len(expected_images) or set(observed_images) != set(expected_images):
            raise ValueError(
                f"milestone {job.milestone} did not return the complete Human-13 panel"
            )
        write_outputs_jsonl(job.output_path, rows)
        payload = job.output_path.read_bytes()
        outputs.append({"milestone": job.milestone, "path": str(job.output_path), "sha256": hashlib.sha256(payload).hexdigest(), "row_count": len(rows)})
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "arm_id": plan.arm_id,
        "manifest_sha256": plan.manifest_sha256,
        "resolved_arm_plan_sha256": plan.resolved_arm_plan_sha256,
        "resolved_config_sha256": plan.resolved_config_sha256,
        "backend_contract": "hf-fp32-sdpa-batch1-rp1-full13",
        "milestones": list(plan.milestones),
        "checkpoints": [{"milestone": job.milestone, "path": str(job.checkpoint_path), "sha256": job.checkpoint_payload_sha256, "readback": dict(job.readback)} for job in plan.jobs],
        "outputs": outputs,
        "runtime_seconds": time.perf_counter() - started,
    }
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt


def _parse_checkpoints(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        try:
            raw_step, raw_path = value.split("=", 1)
            step = int(raw_step)
        except (ValueError, TypeError):
            raise ValueError("--checkpoint must be MILESTONE=PATH") from None
        if step in result:
            raise ValueError("duplicate --checkpoint milestone")
        result[step] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--milestones", default=",".join(str(item) for item in MILESTONES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-discovery", type=Path)
    parser.add_argument("--source-config", type=Path, default=Path("configs/coordexp_swift/infer/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    args = parser.parse_args(argv)
    selected = tuple(int(item) for item in args.milestones.split(",") if item)
    explicit = _parse_checkpoints(args.checkpoint) if args.checkpoint else None
    records = _load_source_records(args.source_discovery) if args.source_discovery else None
    plan = build_eval_plan(
        manifest_path=args.manifest,
        arm_id=args.arm_id,
        output_dir=args.output_dir,
        milestones=selected,
        run_root=args.run_root,
        explicit_checkpoints=explicit,
        source_discovery_records=records,
    )
    if not args.execute:
        print(json.dumps({"status": "dry_run", "execution_ready": False, "arm_id": plan.arm_id, "milestones": list(plan.milestones), "jobs": [{"milestone": job.milestone, "checkpoint_path": str(job.checkpoint_path), "output_path": str(job.output_path), "checkpoint_payload_sha256": job.checkpoint_payload_sha256, "readback": dict(job.readback)} for job in plan.jobs], "actions": dict(plan.actions)}, sort_keys=True))
        return 0
    receipt = execute_eval_plan(plan, execute=True, user_model_gpu_authority=args.user_model_gpu_authority, source_config_path=args.source_config)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
