#!/usr/bin/env python3
"""Materialize the matched Source/transition-step36 transfer evaluation configs."""

from __future__ import annotations

import argparse
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
from src.data.jsonl import load_raw_examples  # noqa: E402


SOURCE_TEMPLATE = (
    REPOSITORY_ROOT
    / "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_row_local_64_source_hf.yaml"
)
CANDIDATE_POOL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/candidate-pool-v1"
)
SPLITS = {
    "development": (CANDIDATE_POOL_ROOT / "development.jsonl", 256),
    "heldout": (CANDIDATE_POOL_ROOT / "heldout.jsonl", 128),
}
TRANSITION_CHECKPOINT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-24-prefix-local-and-on-policy-owner-set-training/"
    "train-first-divergence-1440-long-learning-rate-1e-5-v1/runs/"
    "qwen3_vl_2b_first_divergence_transition_1440_event_long_learning_rate_1e-5/"
    "checkpoints/step-36"
)
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
REQUIRED_EMBEDDING_FILES = (
    "special_token_embeddings.json",
    "special_token_embeddings.safetensors",
)


class MaterializationError(ValueError):
    """Raised when the fixed transfer comparison cannot be materialized safely."""


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


def _config_name(split: str, role: str) -> str:
    return f"transition-step36-transfer-{split}-{role}-hf.yaml"


def _run_name(split: str, role: str) -> str:
    return f"qwen3-vl-2b-transition-step36-transfer-{split}-{role}-max3084-b4-hf"


def materialize(
    *, output_config_dir: Path, inference_artifact_root: Path
) -> dict[str, object]:
    template = _require_file(SOURCE_TEMPLATE, label="Source HF template")
    resolved_template = load_infer_config(template)
    validate_infer_input_paths(
        resolved_template, fields=("adapter.path", "embedding_delta.path")
    )
    template_config = resolved_template.config
    if template_config.adapter is None or template_config.embedding_delta is None:
        raise MaterializationError(
            "Source template must resolve its production adapter and embedding delta"
        )
    source_adapter = _require_payload(
        Path(template_config.adapter.path),
        label="Source production adapter",
        required_files=REQUIRED_ADAPTER_FILES,
    )
    source_embedding_delta = _require_payload(
        Path(template_config.embedding_delta.path),
        label="Source production special-token embedding delta",
        required_files=REQUIRED_EMBEDDING_FILES,
    )
    adapter = _require_payload(
        TRANSITION_CHECKPOINT / "adapter",
        label="transition step-36 adapter",
        required_files=REQUIRED_ADAPTER_FILES,
    )
    embedding_delta = _require_payload(
        TRANSITION_CHECKPOINT / "special_token_embeddings",
        label="transition step-36 special-token embedding delta",
        required_files=REQUIRED_EMBEDDING_FILES,
    )
    output_dir = output_config_dir.expanduser().resolve()
    artifact_root = inference_artifact_root.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite config directory: {output_dir}")
    if artifact_root.exists():
        raise FileExistsError(
            f"refusing to reuse inference artifact root: {artifact_root}"
        )

    split_receipts: dict[str, dict[str, object]] = {}
    example_ids_by_split: dict[str, set[str]] = {}
    for split, (raw_input, expected_rows) in SPLITS.items():
        input_jsonl = _require_file(raw_input, label=f"{split} input JSONL")
        examples = load_raw_examples(input_jsonl)
        if len(examples) != expected_rows:
            raise MaterializationError(
                f"{split} row count mismatch: observed={len(examples)}, "
                f"expected={expected_rows}"
            )
        example_ids = {example.example_id for example in examples}
        if len(example_ids) != expected_rows:
            raise MaterializationError(f"{split} contains duplicate example IDs")
        missing_images = [
            str(example.image.path)
            for example in examples
            if not example.image.path.is_file()
        ]
        if missing_images:
            raise MaterializationError(
                f"{split} contains missing images: {missing_images[:3]}"
            )
        example_ids_by_split[split] = example_ids
        split_receipts[split] = {
            "input_jsonl": str(input_jsonl),
            "input_jsonl_sha256": _sha256(input_jsonl),
            "row_count": expected_rows,
        }
    overlap = example_ids_by_split["development"] & example_ids_by_split["heldout"]
    if overlap:
        raise MaterializationError(
            f"development and heldout example IDs overlap: {sorted(overlap)[:3]}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    template_reference = os.path.relpath(template, start=output_dir)
    entries: list[dict[str, object]] = []
    try:
        for split, (input_jsonl, expected_rows) in SPLITS.items():
            input_jsonl = input_jsonl.resolve()
            for role in ("source", "transition-step36"):
                run_name = _run_name(split, role)
                payload: dict[str, object] = {
                    "extends": template_reference,
                    "run": {
                        "name": run_name,
                        "artifact_root": str(artifact_root),
                        "collision_policy": "fail",
                    },
                    "data": {"input_jsonl": str(input_jsonl)},
                    "generation": {
                        "batch_size": 4,
                        "max_new_tokens": 3084,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "repetition_penalty": 1.0,
                    },
                    "scoring": {"enabled": True},
                }
                if role == "transition-step36":
                    payload["adapter"] = {
                        "type": "dora",
                        "path": str(adapter),
                        "name": "default",
                    }
                    payload["embedding_delta"] = {"path": str(embedding_delta)}

                config_path = staging / _config_name(split, role)
                config_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                )
                resolved = load_infer_config(config_path)
                fields = ["data.input_jsonl"]
                if role == "transition-step36":
                    fields.extend(("adapter.path", "embedding_delta.path"))
                validate_infer_input_paths(resolved, fields=tuple(fields))
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
                    config.scoring.enabled,
                    Path(config.adapter.path) if config.adapter else None,
                    Path(config.embedding_delta.path)
                    if config.embedding_delta
                    else None,
                )
                expected = (
                    run_name,
                    artifact_root,
                    "fail",
                    input_jsonl,
                    4,
                    3084,
                    0.0,
                    1.0,
                    1.0,
                    True,
                    adapter if role == "transition-step36" else source_adapter,
                    embedding_delta
                    if role == "transition-step36"
                    else source_embedding_delta,
                )
                if observed != expected:
                    raise MaterializationError(
                        f"resolved config mismatch for {config_path}: "
                        f"observed={observed!r}, expected={expected!r}"
                    )
                entries.append(
                    {
                        "split": split,
                        "role": role,
                        "expected_row_count": expected_rows,
                        "config_name": config_path.name,
                        "run_name": run_name,
                        "config_fingerprint": resolved.fingerprint,
                    }
                )

        receipt = {
            "schema_version": "transition_step36_transfer_infer_matrix.v1",
            "source_template": str(template),
            "source_template_sha256": _sha256(template),
            "source_adapter_path": str(source_adapter),
            "source_adapter_sha256": _sha256(
                source_adapter / "adapter_model.safetensors"
            ),
            "source_embedding_delta_path": str(source_embedding_delta),
            "source_embedding_delta_sha256": _sha256(
                source_embedding_delta / "special_token_embeddings.safetensors"
            ),
            "inference_artifact_root": str(artifact_root),
            "generation": {
                "batch_size": 4,
                "max_new_tokens": 3084,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            },
            "scoring_enabled": True,
            "splits": split_receipts,
            "split_example_id_overlap_count": 0,
            "transition_checkpoint": str(TRANSITION_CHECKPOINT.resolve()),
            "adapter_path": str(adapter),
            "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
            "embedding_delta_path": str(embedding_delta),
            "embedding_delta_sha256": _sha256(
                embedding_delta / "special_token_embeddings.safetensors"
            ),
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
    args = parser.parse_args()
    result = materialize(
        output_config_dir=args.output_config_dir,
        inference_artifact_root=args.inference_artifact_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
