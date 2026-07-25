#!/usr/bin/env python3
"""Materialize the 40-config clean-greedy long-training milestone matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
    load_infer_config,
    validate_infer_input_paths,
)


EXPERIMENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-24-prefix-local-and-on-policy-owner-set-training"
)
SOURCE_TEMPLATE = (
    REPOSITORY_ROOT / "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_row_local_64_source_hf.yaml"
)
INPUT_JSONL = EXPERIMENT_ROOT / "inference-panel-pairwise-64-v2/panel.coord.jsonl"
STEPS = (9, 18, 27, 36, 45, 54, 63, 72, 81, 90)

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
REQUIRED_EMBEDDING_FILES = (
    "special_token_embeddings.json",
    "special_token_embeddings.safetensors",
)


class MaterializationError(ValueError):
    """Raised when one immutable checkpoint cannot form a runnable config."""


@dataclass(frozen=True)
class Arm:
    slug: str
    checkpoint_run_dir: Path


ARMS = (
    Arm(
        slug="pairwise-lr1e5",
        checkpoint_run_dir=EXPERIMENT_ROOT
        / "train-pairwise-1440-long-learning-rate-1e-5-v1/runs/"
        "qwen3_vl_2b_row_local_complete_action_pairwise_1440_event_long_learning_rate_1e-5",
    ),
    Arm(
        slug="pairwise-lr3e6",
        checkpoint_run_dir=EXPERIMENT_ROOT
        / "train-pairwise-1440-long-learning-rate-3e-6-v1/runs/"
        "qwen3_vl_2b_row_local_complete_action_pairwise_1440_event_long_learning_rate_3e-6",
    ),
    Arm(
        slug="transition-lr1e5",
        checkpoint_run_dir=EXPERIMENT_ROOT
        / "train-first-divergence-1440-long-learning-rate-1e-5-v1/runs/"
        "qwen3_vl_2b_first_divergence_transition_1440_event_long_learning_rate_1e-5",
    ),
    Arm(
        slug="owner-conditioned-lr1e5",
        checkpoint_run_dir=EXPERIMENT_ROOT
        / "train-owner-conditioned-1440-long-learning-rate-1e-5-v1/runs/"
        "qwen3_vl_2b_owner_conditioned_candidate_1440_event_long_learning_rate_1e-5",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MaterializationError(f"missing {label}: {resolved}")
    return resolved


def _require_payload(
    directory: Path, *, label: str, required_files: tuple[str, ...]
) -> Path:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise MaterializationError(f"missing {label} directory: {resolved}")
    missing = [name for name in required_files if not (resolved / name).is_file()]
    if missing:
        raise MaterializationError(
            f"incomplete {label} at {resolved}; missing files: {missing}"
        )
    return resolved


def _config_name(arm: Arm, step: int) -> str:
    return f"row-local-long-{arm.slug}-step-{step:02d}-64-hf.yaml"


def _run_name(arm: Arm, step: int) -> str:
    return f"qwen3-vl-2b-row-local-long-{arm.slug}-step-{step:02d}-64-hf"


def _checkpoint_payload(arm: Arm, step: int) -> tuple[Path, Path]:
    checkpoint = arm.checkpoint_run_dir.resolve() / "checkpoints" / f"step-{step}"
    adapter = _require_payload(
        checkpoint / "adapter",
        label=f"{arm.slug} step-{step} adapter",
        required_files=REQUIRED_ADAPTER_FILES,
    )
    embedding_delta = _require_payload(
        checkpoint / "special_token_embeddings",
        label=f"{arm.slug} step-{step} special-token embedding delta",
        required_files=REQUIRED_EMBEDDING_FILES,
    )
    return adapter, embedding_delta


def materialize(
    *,
    output_config_dir: Path,
    inference_artifact_root: Path,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    selected_entries: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, object]:
    template = _require_file(SOURCE_TEMPLATE, label="64-image Source HF template")
    input_jsonl = _require_file(INPUT_JSONL, label="frozen 64-image input panel")
    output_dir = output_config_dir.expanduser().resolve()
    artifact_root = inference_artifact_root.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite config directory: {output_dir}")
    if artifact_root.exists():
        raise FileExistsError(
            f"refusing to reuse inference artifact root: {artifact_root}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    template_reference = os.path.relpath(template, start=output_dir)
    if batch_size <= 0:
        raise MaterializationError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise MaterializationError("max_new_tokens must be positive")
    arm_by_slug = {arm.slug: arm for arm in ARMS}
    requested = selected_entries or tuple(
        (arm.slug, step) for arm in ARMS for step in STEPS
    )
    if len(set(requested)) != len(requested):
        raise MaterializationError(f"duplicate selected entries: {requested!r}")
    for arm_slug, step in requested:
        if arm_slug not in arm_by_slug:
            raise MaterializationError(f"unknown arm: {arm_slug}")
        if step not in STEPS:
            raise MaterializationError(f"unsupported milestone step: {step}")

    entries: list[dict[str, object]] = []
    try:
        for arm_slug, step in requested:
            arm = arm_by_slug[arm_slug]
            adapter, embedding_delta = _checkpoint_payload(arm, step)
            config_path = staging / _config_name(arm, step)
            payload = {
                "extends": template_reference,
                "run": {
                    "name": _run_name(arm, step),
                    "artifact_root": str(artifact_root),
                    "collision_policy": "fail",
                },
                "data": {"input_jsonl": str(input_jsonl)},
                "generation": {
                    "batch_size": batch_size,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                },
                "scoring": {"enabled": True},
                "adapter": {
                    "type": "dora",
                    "path": str(adapter),
                    "name": "default",
                },
                "embedding_delta": {"path": str(embedding_delta)},
            }
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            resolved = load_infer_config(config_path)
            validate_infer_input_paths(
                resolved,
                fields=("data.input_jsonl", "adapter.path", "embedding_delta.path"),
            )
            config = resolved.config
            observed = (
                config.run.name,
                Path(config.run.artifact_root),
                config.run.collision_policy,
                Path(config.data.input_jsonl),
                config.generation.batch_size,
                config.generation.max_new_tokens,
                config.generation.temperature,
                config.generation.top_p,
                config.generation.repetition_penalty,
                Path(config.adapter.path) if config.adapter else None,
                Path(config.embedding_delta.path) if config.embedding_delta else None,
            )
            expected = (
                _run_name(arm, step),
                artifact_root,
                "fail",
                input_jsonl,
                batch_size,
                max_new_tokens,
                0.0,
                1.0,
                1.0,
                adapter,
                embedding_delta,
            )
            if observed != expected:
                raise MaterializationError(
                    f"resolved config mismatch for {config_path}: "
                    f"observed={observed!r}, expected={expected!r}"
                )
            entries.append(
                {
                    "arm": arm.slug,
                    "step": step,
                    "config_name": config_path.name,
                    "run_name": config.run.name,
                    "config_fingerprint": resolved.fingerprint,
                    "adapter_path": str(adapter),
                    "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
                    "embedding_delta_path": str(embedding_delta),
                    "embedding_delta_sha256": _sha256(
                        embedding_delta / "special_token_embeddings.safetensors"
                    ),
                }
            )
        if len(entries) != len(requested):
            raise AssertionError(f"unexpected matrix size: {len(entries)}")
        receipt = {
            "schema_version": "row_local_long_milestone_infer_matrix.v1",
            "source_template": str(template),
            "input_jsonl": str(input_jsonl),
            "input_jsonl_sha256": _sha256(input_jsonl),
            "inference_artifact_root": str(artifact_root),
            "generation": {
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "entry_count": len(entries),
            "entries": entries,
        }
        (staging / "matrix-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {**receipt, "output_config_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument("--inference-artifact-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="ARM:STEP",
        help="materialize only the named arm and milestone; may be repeated",
    )
    args = parser.parse_args()
    selected_entries: tuple[tuple[str, int], ...] | None = None
    if args.entry:
        parsed: list[tuple[str, int]] = []
        for value in args.entry:
            try:
                arm_slug, raw_step = value.rsplit(":", maxsplit=1)
                parsed.append((arm_slug, int(raw_step)))
            except ValueError as exc:
                raise SystemExit(
                    f"invalid --entry {value!r}; expected ARM:STEP"
                ) from exc
        selected_entries = tuple(parsed)
    result = materialize(
        output_config_dir=args.output_config_dir,
        inference_artifact_root=args.inference_artifact_root,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        selected_entries=selected_entries,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
