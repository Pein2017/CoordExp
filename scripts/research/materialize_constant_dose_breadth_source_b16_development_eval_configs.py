#!/usr/bin/env python3
"""Materialize immutable Source@B16 development inference configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config.inference import (  # noqa: E402
    ResolvedInferConfig,
    load_infer_config,
    validate_infer_input_paths,
)


EXPERIMENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen"
)
SOURCE_TEMPLATE = (
    REPOSITORY_ROOT / "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_step4887_candidate_pool_2432_source_b16_vllm.yaml"
)
SOURCE_INPUT = EXPERIMENT_ROOT / "candidate-pool-v1/candidate-pool-2432.coord.jsonl"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "train-992-events-v1"
INFERENCE_ARTIFACT_ROOT = EXPERIMENT_ROOT / "development-source-b16-vllm-eval-v1"

ARMS = ("broad", "concentrated")
SEEDS = (19, 23)
STEPS = (10, 20, 30, 31)
EXPECTED_CONFIG_COUNT = len(ARMS) * len(SEEDS) * len(STEPS)

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
REQUIRED_EMBEDDING_FILES = (
    "special_token_embeddings.json",
    "special_token_embeddings.safetensors",
)


class MaterializationError(ValueError):
    """Raised when immutable inputs cannot identify one runnable config."""


@dataclass(frozen=True)
class TreatmentCheckpoint:
    arm: str
    seed: int
    step: int
    run_dir: Path
    adapter_dir: Path
    embedding_delta_dir: Path

    @property
    def slug(self) -> str:
        return f"{self.arm}-seed-{self.seed}-step-{self.step}"

    @property
    def config_name(self) -> str:
        return f"source-b16-development-{self.slug}.yaml"

    @property
    def run_name(self) -> str:
        return f"source-b16-development-{self.slug}-vllm"


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MaterializationError(f"missing {label}: {resolved}")
    return resolved


def _require_payload_files(
    directory: Path,
    *,
    label: str,
    required_files: tuple[str, ...],
) -> Path:
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise MaterializationError(f"missing {label} directory: {resolved}")
    missing = [name for name in required_files if not (resolved / name).is_file()]
    if missing:
        raise MaterializationError(
            f"incomplete {label} payload at {resolved}; missing files: {missing}"
        )
    return resolved


def resolve_treatment_checkpoint(
    checkpoint_root: Path,
    *,
    arm: str,
    seed: int,
    step: int,
) -> TreatmentCheckpoint:
    seed_root = checkpoint_root.resolve() / arm / f"seed-{seed}"
    runs_root = seed_root / "runs"
    run_dirs = sorted(path.resolve() for path in runs_root.glob("*") if path.is_dir())
    if len(run_dirs) != 1:
        raise MaterializationError(
            "expected exactly one treatment run directory for "
            f"arm={arm} seed={seed}, found {len(run_dirs)} under {runs_root}: "
            f"{[path.name for path in run_dirs]}"
        )
    run_dir = run_dirs[0]

    checkpoints_root = run_dir / "checkpoints"
    checkpoint_dirs = [
        path.resolve()
        for path in checkpoints_root.glob(f"step-{step}")
        if path.is_dir()
    ]
    if len(checkpoint_dirs) != 1:
        raise MaterializationError(
            "expected exactly one checkpoint directory for "
            f"arm={arm} seed={seed} step={step}, found {len(checkpoint_dirs)} "
            f"under {checkpoints_root}"
        )
    checkpoint_dir = checkpoint_dirs[0]
    adapter_dir = _require_payload_files(
        checkpoint_dir / "adapter",
        label="adapter",
        required_files=REQUIRED_ADAPTER_FILES,
    )
    embedding_delta_dir = _require_payload_files(
        checkpoint_dir / "special_token_embeddings",
        label="special-token embedding delta",
        required_files=REQUIRED_EMBEDDING_FILES,
    )
    return TreatmentCheckpoint(
        arm=arm,
        seed=seed,
        step=step,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        embedding_delta_dir=embedding_delta_dir,
    )


def _derived_config_payload(
    checkpoint: TreatmentCheckpoint,
    *,
    template_reference: str,
    source_input: Path,
    inference_artifact_root: Path,
) -> dict[str, object]:
    artifact_root = (
        inference_artifact_root.resolve()
        / checkpoint.arm
        / f"seed-{checkpoint.seed}"
        / f"step-{checkpoint.step}"
    )
    return {
        "extends": template_reference,
        "run": {
            "name": checkpoint.run_name,
            "artifact_root": str(artifact_root),
            "collision_policy": "fail",
        },
        "model": {"dtype": "bf16"},
        "backend": {
            "type": "vllm",
            "vllm": {
                "gpu_memory_utilization": 0.70,
                "max_model_len": 4096,
            },
        },
        "data": {"input_jsonl": str(source_input)},
        "generation": {
            "batch_size": 32,
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "adapter": {
            "type": "dora",
            "path": str(checkpoint.adapter_dir),
            "name": "default",
        },
        "embedding_delta": {"path": str(checkpoint.embedding_delta_dir)},
    }


def _validate_generated_config(
    config_path: Path,
    *,
    checkpoint: TreatmentCheckpoint,
    source_input: Path,
    inference_artifact_root: Path,
) -> ResolvedInferConfig:
    resolved = load_infer_config(config_path)
    validate_infer_input_paths(
        resolved,
        fields=("data.input_jsonl", "adapter.path", "embedding_delta.path"),
    )
    config = resolved.config
    expected_artifact_root = (
        inference_artifact_root.resolve()
        / checkpoint.arm
        / f"seed-{checkpoint.seed}"
        / f"step-{checkpoint.step}"
    )
    observed = {
        "run.name": config.run.name,
        "run.artifact_root": config.run.artifact_root,
        "run.collision_policy": config.run.collision_policy,
        "model.dtype": config.model.dtype,
        "backend.type": config.backend.type,
        "backend.vllm.max_model_len": getattr(config.backend, "vllm").max_model_len,
        "data.input_jsonl": config.data.input_jsonl,
        "generation.batch_size": config.generation.batch_size,
        "generation.max_new_tokens": config.generation.max_new_tokens,
        "generation.temperature": config.generation.temperature,
        "generation.top_p": config.generation.top_p,
        "generation.repetition_penalty": config.generation.repetition_penalty,
        "adapter.path": config.adapter.path if config.adapter else None,
        "embedding_delta.path": (
            config.embedding_delta.path if config.embedding_delta else None
        ),
    }
    expected = {
        "run.name": checkpoint.run_name,
        "run.artifact_root": str(expected_artifact_root),
        "run.collision_policy": "fail",
        "model.dtype": "bf16",
        "backend.type": "vllm",
        "backend.vllm.max_model_len": 4096,
        "data.input_jsonl": str(source_input),
        "generation.batch_size": 32,
        "generation.max_new_tokens": 2048,
        "generation.temperature": 0.0,
        "generation.top_p": 1.0,
        "generation.repetition_penalty": 1.0,
        "adapter.path": str(checkpoint.adapter_dir),
        "embedding_delta.path": str(checkpoint.embedding_delta_dir),
    }
    if observed != expected:
        raise MaterializationError(
            f"generated config contract mismatch for {config_path}: "
            f"observed={observed!r}, expected={expected!r}"
        )
    return resolved


def materialize_configs(
    *,
    source_template: Path,
    checkpoint_root: Path,
    output_config_dir: Path,
    inference_artifact_root: Path,
) -> tuple[Path, ...]:
    """Write and validate one immutable directory containing all 16 configs."""

    template = _require_file(source_template, label="Source@B16 template")
    frozen_input = _require_file(SOURCE_INPUT, label="frozen Source@B16 input")
    checkpoint_root = checkpoint_root.expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise MaterializationError(f"missing checkpoint root: {checkpoint_root}")

    output_dir = output_config_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable generated-config directory: {output_dir}"
        )

    checkpoints = tuple(
        resolve_treatment_checkpoint(
            checkpoint_root,
            arm=arm,
            seed=seed,
            step=step,
        )
        for arm in ARMS
        for seed in SEEDS
        for step in STEPS
    )
    if len(checkpoints) != EXPECTED_CONFIG_COUNT:
        raise AssertionError(
            f"internal treatment grid mismatch: {len(checkpoints)} configs"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging_dir.mkdir()
    template_reference = os.path.relpath(template, start=output_dir)
    try:
        staged_paths: list[Path] = []
        for checkpoint in checkpoints:
            config_path = staging_dir / checkpoint.config_name
            payload = _derived_config_payload(
                checkpoint,
                template_reference=template_reference,
                source_input=frozen_input,
                inference_artifact_root=inference_artifact_root,
            )
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            _validate_generated_config(
                config_path,
                checkpoint=checkpoint,
                source_input=frozen_input,
                inference_artifact_root=inference_artifact_root,
            )
            staged_paths.append(config_path)
        staging_dir.rename(output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return tuple(output_dir / path.name for path in staged_paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-template", type=Path, default=SOURCE_TEMPLATE)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument(
        "--inference-artifact-root", type=Path, default=INFERENCE_ARTIFACT_ROOT
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    outputs = materialize_configs(
        source_template=arguments.source_template,
        checkpoint_root=arguments.checkpoint_root,
        output_config_dir=arguments.output_config_dir,
        inference_artifact_root=arguments.inference_artifact_root,
    )
    print(
        json.dumps(
            {
                "config_count": len(outputs),
                "output_config_dir": str(arguments.output_config_dir.resolve()),
                "config_names": [path.name for path in outputs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
