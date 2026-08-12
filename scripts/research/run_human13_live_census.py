#!/usr/bin/env python3
"""Run the exact Human-13 no-update census from the frozen Source state.

Dry-run is the default and performs identity validation only.  The processor,
models, forwards, and artifact writes are reachable solely through explicit
``--execute --user-model-gpu-authority``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


RECEIPT_SCHEMA_VERSION = "human13_live_census_execution_receipt.v1"
ZERO_ACTIONS = {
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "artifact_writes": 0,
    "gpu_allocations": 0,
}


class LiveCensusError(RuntimeError):
    """Raised when the live census cannot preserve its frozen contract."""


def execute_cli(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    k_root: str | Path,
    a1_config_path: str | Path,
    repo_root: str | Path,
    census_output: str | Path,
    receipt_output: str | Path,
    execute: bool,
    authority: bool,
    panel_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate or execute one immutable Source-state census."""

    if execute and authority is not True:
        raise LiveCensusError("execute mode requires explicit model/GPU authority")
    census_path = Path(census_output).expanduser().resolve()
    receipt_path = Path(receipt_output).expanduser().resolve()
    if execute:
        _preflight_outputs(census_path, receipt_path)

    plan_kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "source_root": source_root,
        "k_root": k_root,
    }
    if panel_path is not None:
        plan_kwargs["panel_path"] = panel_path
    census_plan = _load_census_plan(**plan_kwargs)
    model_plan = _build_model_plan(a1_config_path)
    if getattr(model_plan, "arm_id", None) != "A1":
        raise LiveCensusError("live no-update census requires the frozen A1 config")
    validation = _validate_model_plan(model_plan)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    if _sha256_file(manifest_file) != census_plan.manifest_sha256:
        raise LiveCensusError("census plan and manifest bytes differ")

    base_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "mode": "dry_run" if not execute else "execute",
        "execution_ready": False,
        "manifest": {
            "path": str(manifest_file),
            "sha256": census_plan.manifest_sha256,
        },
        "census_plan": {
            "schema_version": census_plan.schema_version,
            "sha256": census_plan.plan_sha256,
            "discovery_binding_sha256": census_plan.discovery_binding_sha256,
        },
        "source_discovery": _discovery_identity(source_root),
        "k_discovery": _discovery_identity(k_root),
        "model_plan": _artifact(model_plan),
        "source_validation": _artifact(validation),
    }
    if not execute:
        return {**base_receipt, "actions": dict(ZERO_ACTIONS)}

    root = Path(repo_root).expanduser().resolve(strict=True)
    started = time.perf_counter()
    manifest = _load_manifest(manifest_file)
    processor_components = _load_processor_components(model_plan)
    skeletons = _build_skeletons(manifest, processor_components, root)
    prepared = _prepare_capture(plan=census_plan, prompt_skeletons=skeletons)
    pack_count = len(prepared.packed_plan.packs)
    if pack_count <= 0:
        raise LiveCensusError("prepared census has no physical packs")
    assembly = _assemble_model(model_plan, pack_count=pack_count, repo_root=root)
    if int(getattr(assembly.runtime, "optimizer_step_count", -1)) != 0:
        raise LiveCensusError("Source census assembly must have zero optimizer steps")
    with _open_hf_scorer(root) as hf_scorer:
        captured = _capture(
            plan=census_plan,
            prompt_skeletons=skeletons,
            packed_model=assembly.model,
            packed_runtime=assembly.runtime,
            tokenizer=assembly.components.tokenizer,
            hf_scorer=hf_scorer,
        )
    census = _run_census(plan=census_plan, evidence=captured.evidence)
    census_bytes = _canonical_bytes(census)
    census_sha256 = hashlib.sha256(census_bytes).hexdigest()
    counters = dict(captured.receipt)
    packed_forwards = int(counters.get("packed_forward_count", -1))
    hf_forwards = int(counters.get("hf_forward_count", -1))
    if packed_forwards != pack_count or hf_forwards <= 0:
        raise LiveCensusError("captured census forward counts differ from its plan")
    receipt = {
        **base_receipt,
        "execution_ready": True,
        "census": {
            "path": str(census_path),
            "schema_version": census.get("schema_version"),
            "sha256": census_sha256,
        },
        "runtime": {
            **counters,
            "total_forward_count": packed_forwards + hf_forwards,
            "optimizer_steps": 0,
            "backwards": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": _peak_memory_bytes(assembly),
        },
        "actions": {
            "model_loads": 2,
            "forwards": packed_forwards + hf_forwards,
            "backwards": 0,
            "optimizer_constructions": 1,
            "optimizer_steps": 0,
            "artifact_writes": 2,
            "gpu_allocations": 2,
        },
    }
    receipt_bytes = _canonical_bytes(receipt)
    _write_exclusive(census_path, census_bytes)
    _write_exclusive(receipt_path, receipt_bytes)
    return receipt


def _preflight_outputs(census_path: Path, receipt_path: Path) -> None:
    if census_path == receipt_path:
        raise LiveCensusError("census and execution receipt outputs must be distinct")
    for path in (census_path, receipt_path):
        if path.exists():
            raise LiveCensusError(f"live census output already exists: {path}")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveCensusError(f"live census artifact is not canonical JSON: {exc}") from exc


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LiveCensusError(f"live census output already exists: {path}") from exc
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise LiveCensusError(f"short write for live census artifact: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discovery_identity(root: str | Path) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve(strict=True)
    receipt = (resolved / "receipt.json").resolve(strict=True)
    records = (resolved / "trajectories.jsonl").resolve(strict=True)
    return {
        "root": str(resolved),
        "receipt_path": str(receipt),
        "receipt_sha256": _sha256_file(receipt),
        "records_path": str(records),
        "records_sha256": _sha256_file(records),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_artifact_dict", None)
    if callable(method):
        result = method()
        if isinstance(result, Mapping):
            return dict(result)
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return {"repr": repr(value)}


def _load_census_plan(**kwargs: Any) -> Any:
    from scripts.research.materialize_human13_no_update_census import (
        load_canonical_census_plan,
    )

    return load_canonical_census_plan(**kwargs)


def _build_model_plan(path: str | Path) -> Any:
    from scripts.research.human13_live_model import build_human13_live_model_plan

    return build_human13_live_model_plan(path)


def _validate_model_plan(plan: Any) -> Any:
    from scripts.research.human13_live_model import validate_human13_live_model_plan

    return validate_human13_live_model_plan(plan)


def _load_manifest(path: str | Path) -> Any:
    from scripts.research.build_human13_k_union_manifest import load_manifest

    return load_manifest(path, require_full_panel=True)


def _load_processor_components(model_plan: Any) -> Any:
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options

    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=model_plan.source.base_model_path,
            dtype=model_plan.mixed_precision,
            attn_implementation=model_plan.attn_implementation,
            patch_embed_linearization=model_plan.patch_embed_linearization,
            load_model=False,
        )
    )


def _build_skeletons(manifest: Any, components: Any, repo_root: Path) -> Any:
    from scripts.research.human13_live_model import build_human13_processor_skeletons

    return build_human13_processor_skeletons(
        manifest, components, repo_root=repo_root
    )


def _prepare_capture(**kwargs: Any) -> Any:
    from scripts.research.human13_live_census import prepare_census_capture

    return prepare_census_capture(**kwargs)


def _assemble_model(plan: Any, *, pack_count: int, repo_root: Path) -> Any:
    from scripts.research.human13_live_model import assemble_human13_live_model

    return assemble_human13_live_model(
        plan, pack_count=pack_count, repo_root=repo_root
    )


def _open_hf_scorer(repo_root: Path) -> Any:
    from scripts.research.human13_hf_census import open_source_hf_census_scorer

    return open_source_hf_census_scorer(repo_root=repo_root)


def _capture(**kwargs: Any) -> Any:
    from scripts.research.human13_live_census import capture_human13_live_census

    return capture_human13_live_census(**kwargs)


def _run_census(**kwargs: Any) -> Any:
    from scripts.research.materialize_human13_no_update_census import (
        run_census_with_exact_logits,
    )

    return run_census_with_exact_logits(**kwargs)


def _peak_memory_bytes(assembly: Any) -> int | None:
    device = getattr(getattr(assembly, "accelerator", None), "device", None)
    if device is None:
        return None
    try:
        import torch

        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(resolved))
    except (RuntimeError, TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--k-root", type=Path, required=True)
    parser.add_argument("--a1-config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--panel-path", type=Path)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = execute_cli(
        manifest_path=args.manifest,
        source_root=args.source_root,
        k_root=args.k_root,
        a1_config_path=args.a1_config,
        repo_root=args.repo_root,
        census_output=args.census_output,
        receipt_output=args.receipt_output,
        execute=args.execute,
        authority=args.user_model_gpu_authority,
        panel_path=args.panel_path,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveCensusError",
    "RECEIPT_SCHEMA_VERSION",
    "ZERO_ACTIONS",
    "build_parser",
    "execute_cli",
    "main",
]
