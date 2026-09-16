"""Prepare the bounded Source256 A/B qualification and paired64 manifests."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from probes.training_set_completion import source256_training as runtime
from probes.training_set_completion import training


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-16-source256-fixed-prefix-completion/runtime"
)
SOURCE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "dora_owner_learning/configs/source256.yaml"
)
SCHEMA = "training_set_completion.source256_trial.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _model_config(*, arm: str, output: Path) -> dict[str, Any]:
    from src.config.inference import load_research_infer_config

    resolved = load_research_infer_config(SOURCE_CONFIG)
    config = copy.deepcopy(resolved.config_dict)
    config["run"].update(
        name=f"source256-fixed-prefix-{arm}",
        artifact_root=str((output / arm).resolve()),
        output_dir=None,
        collision_policy="fail",
    )
    return config


def build_training_manifest(
    *,
    preparation_path: Path,
    arm: str,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    require(arm in runtime.ARMS and mode in ("qualification", "main"), "arm/mode")
    checked = runtime.validate_preparation(read(preparation_path))
    contract = checked["runtime_contract"]
    source_adapter = training.inspect_dora_adapter_payload(
        contract["adapter_root"], contract["base_model_root"]
    )
    mode_runtime = {
        "qualification": {
            "updates": 2,
            "checkpoint_steps": [2],
            "max_model_forwards": 128,
            "max_model_calls": 64,
            "wall_seconds": 3600,
        },
        "main": {
            "updates": 64,
            "checkpoint_steps": [16, 32, 64],
            "max_model_forwards": 4096,
            "max_model_calls": 2048,
            "wall_seconds": 14_400,
        },
    }[mode]
    manifest: dict[str, Any] = {
        "schema": runtime.MANIFEST_SCHEMA,
        "status": "candidate_ready",
        "arm": arm,
        "mode": mode,
        "sources": {
            "producer": training.binding(Path(runtime.__file__)),
            "source_config": training.binding(SOURCE_CONFIG),
        },
        "preparation": training.binding(preparation_path),
        "source_adapter": source_adapter,
        "model_config": _model_config(arm=arm, output=output),
        "optimizer": copy.deepcopy(training.DEFAULT_OPTIMIZER),
        "scheduler": {
            "type": "cosine",
            "total_updates": 64,
            "warmup_updates": 0,
            "min_lr_ratio": 0.0,
        },
        "objective": {
            "ce_reduction": "sample_equal",
            "geometry_reduction": "sample_equal",
            "branch_weights": {"common": 0.5, "variable": 0.5},
        },
        "validity_hinge": {
            "weight": 0.01,
            "margin": 1 / 999,
            "coordinate_token_ids": contract["coordinate_token_ids"],
            "coordinate_bin_values": list(range(1000)),
        },
        "runtime": {
            **mode_runtime,
            "seed": 19,
            "world_size": 4,
            "effective_image_batch": 64,
            "branch_image_count": 32,
            "microbatch_size": 2,
            "activation_checkpointing": True,
            "fresh_optimizer": True,
            "gradient_clip_norm": 1.0,
            "eos_token_id": contract["eos_token_id"],
        },
        "content_sha256": None,
    }
    manifest["content_sha256"] = training.digest(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    runtime.validate_training_manifest(manifest)
    return manifest


def prepare(
    *, preparation_path: Path, output: Path, mode: str
) -> dict[str, Any]:
    require(not output.exists(), f"trial output collision: {output}")
    manifests = {
        arm: build_training_manifest(
            preparation_path=preparation_path,
            arm=arm,
            mode=mode,
            output=output,
        )
        for arm in runtime.ARMS
    }
    output.mkdir(parents=True)
    paths: dict[str, dict[str, Any]] = {}
    for arm, manifest in manifests.items():
        path = output / arm / "training-manifest.json"
        training.publish(path, manifest)
        paths[arm] = training.binding(path)
    value = {
        "schema": SCHEMA,
        "status": "candidate_ready_for_GPU_qualification"
        if mode == "qualification"
        else "candidate_ready_for_paired64_after_qualification",
        "mode": mode,
        "preparation": training.binding(preparation_path),
        "gate": runtime.validate_preparation(read(preparation_path))["gate"],
        "arms": paths,
        "topology": {
            "implementation": "four ranks per arm",
            "qualification": "A and B concurrently on disjoint four-GPU groups",
            "main_default": "A and B concurrently on disjoint four-GPU groups",
        },
        "launch": "held for lead release; preparation does not launch GPU work",
        "producer": training.binding(Path(__file__)),
    }
    training.publish(output / "trial.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "main"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                preparation_path=args.preparation,
                output=args.output,
                mode=args.mode,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
