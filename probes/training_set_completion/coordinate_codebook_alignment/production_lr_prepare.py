"""Prepare the two authorized conditional-LR production packets.

This is a CPU-only preparation entry.  It derives configs from the frozen
nominal config, verifies their semantic delta, and writes isolated evaluation
queues for the already frozen fit panel.  It never loads model weights or
launches training/evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.artifacts import binding
from src.config.loader import load_train_config


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-coordinate-codebook-alignment"
)
REPO = Path(__file__).resolve().parents[3]
NOMINAL = REPO / "configs/research/coordinate_codebook_alignment/nominal.yaml"
PLAN_V4 = ROOT / "production-plan-v4"
SELECTION = ROOT / "selection-v4"
DEFAULT_OUTPUT = ROOT / "production-lr-plan-v1"

SCALES = (("0p3x", 0.3), ("3x", 3.0))
STEPS = (32, 128, 512)
FIT_PATH = SELECTION / "fit.coord.jsonl"
LR_PATHS = (
    ("optimizer", "groups", "adapters", "language", "lr"),
    ("optimizer", "groups", "adapters", "vision", "lr"),
    ("optimizer", "groups", "adapters", "aligner", "lr"),
    ("optimizer", "groups", "token_embeddings", "lr"),
    ("optimizer", "groups", "coordinate_codebook", "lr"),
)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object row in {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _lookup(mapping: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        mapping = mapping[key]
    return mapping


def _set(mapping: Any, path: tuple[str, ...], value: Any) -> None:
    parent = _lookup(mapping, path[:-1])
    parent[path[-1]] = value


def _without_lr_and_name(value: Any, path: tuple[str, ...] = ()) -> Any:
    if path == ("run", "name") or path in LR_PATHS:
        return None
    if isinstance(value, dict):
        return {
            key: _without_lr_and_name(item, (*path, str(key)))
            for key, item in value.items()
            if (*path, str(key)) != ("run", "name") and (*path, str(key)) not in LR_PATHS
        }
    if isinstance(value, list):
        return [_without_lr_and_name(item, path) for item in value]
    return value


def _diff_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if type(left) is not type(right):
        return [".".join(path)]
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        result: list[str] = []
        for key in keys:
            if key not in left or key not in right:
                result.append(".".join((*path, str(key))))
            else:
                result.extend(_diff_paths(left[key], right[key], (*path, str(key))))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [".".join(path)]
        result: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_diff_paths(a, b, (*path, str(index))))
        return result
    return [] if left == right else [".".join(path)]


def _config_payload(scale: float, tag: str, nominal_payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(nominal_payload)
    payload["run"]["name"] = f"conditional-{tag}-seed1729"
    for path in LR_PATHS:
        # Decimal multiplication avoids binary-float artifacts such as
        # ``6.000000000000001e-05`` in the authored YAML.
        product = Decimal(str(_lookup(payload, path))) * Decimal(str(scale))
        _set(payload, path, float(product))
    return payload


def _verify_config(
    config_path: Path,
    nominal_resolved: Any,
    nominal_payload: dict[str, Any],
    expected_scale: float,
    tag: str,
) -> dict[str, Any]:
    resolved = load_train_config(config_path)
    payload = resolved.config_dict
    nominal = nominal_resolved.config_dict
    reduced_delta = _diff_paths(
        _without_lr_and_name(nominal), _without_lr_and_name(payload)
    )
    if reduced_delta:
        raise ValueError(f"unexpected semantic config delta for {tag}: {reduced_delta}")
    if payload["run"]["name"] != f"conditional-{tag}-seed1729":
        raise ValueError(f"run name mismatch for {tag}")
    expected = {}
    for path in LR_PATHS:
        base = _lookup(nominal_payload, path)
        actual = _lookup(payload, path)
        expected_value = float(Decimal(str(base)) * Decimal(str(expected_scale)))
        if actual != expected_value:
            raise ValueError(f"LR mismatch at {'.'.join(path)}: {actual} != {expected_value}")
        expected[".".join(path)] = {"nominal": base, "conditional": actual}
    return {
        "path": binding(config_path),
        "fingerprint": resolved.fingerprint,
        "scale": expected_scale,
        "run_name": payload["run"]["name"],
        "semantic_delta_paths": ["run.name", *[".".join(path) for path in LR_PATHS]],
        "lr_values": expected,
        "loader_version": resolved.loader_version,
    }


def _queue(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    checkpoint_ref: str,
    config_path: Path,
) -> dict[str, Any]:
    specs = []
    for index, row in enumerate(sorted(rows, key=lambda item: int(item["image_id"]))):
        admission = row.get("_admission")
        if not isinstance(admission, dict) or not admission.get("row_id"):
            raise ValueError(f"fit row lacks admission row_id: {row.get('image_id')}")
        image_id = int(row["image_id"])
        specs.append(
            {
                "queue_index": index,
                "image_id": image_id,
                "row_id": str(admission["row_id"]),
                "condition": condition,
                "checkpoint_ref": checkpoint_ref,
                "config": str(config_path.resolve(strict=True)),
                "dataset": str(FIT_PATH.resolve(strict=True)),
                "cell_key": f"{condition}-{image_id}",
                "cohort": str(admission["cohort"]),
            }
        )
    return {
        "schema": "coordinate_codebook_alignment.evaluation_queue.v1",
        "status": "planned",
        "model_calls": 0,
        "monitor": False,
        "teacher_metrics": True,
        "empty_prefix": True,
        "max_new_tokens": 3084,
        "repetition_penalty": 1.0,
        "sampling": False,
        "specs": specs,
    }


def prepare(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output}")
    for path in (NOMINAL, PLAN_V4 / "manifest.json", PLAN_V4 / "pack-schedule.json", SELECTION / "admission.json", FIT_PATH):
        path.resolve(strict=True)

    nominal_payload = yaml.safe_load(NOMINAL.read_text(encoding="utf-8"))
    if not isinstance(nominal_payload, dict):
        raise ValueError("nominal config is not a YAML mapping")
    nominal_resolved = load_train_config(NOMINAL)
    rows = _rows(FIT_PATH)
    if len(rows) != 32:
        raise ValueError(f"expected 32 frozen fit rows, got {len(rows)}")
    row_ids = [str(row.get("_admission", {}).get("row_id", "")) for row in rows]
    image_ids = [int(row["image_id"]) for row in rows]
    if len(set(row_ids)) != 32 or len(set(image_ids)) != 32:
        raise ValueError("fit row or image identities are not unique")

    configs: dict[str, dict[str, Any]] = {}
    config_paths: dict[str, Path] = {}
    for tag, scale in SCALES:
        config_path = REPO / f"configs/research/coordinate_codebook_alignment/conditional-{tag}.yaml"
        if config_path.exists():
            raise FileExistsError(f"refusing to overwrite config: {config_path}")
        payload = _config_payload(scale, tag, nominal_payload)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths[tag] = config_path
        configs[tag] = _verify_config(config_path, nominal_resolved, nominal_payload, scale, tag)

    output.mkdir(parents=True)
    queues: dict[str, dict[str, Any]] = {}
    condition_names = []
    for tag, _scale in SCALES:
        for step in STEPS:
            condition = f"conditional-{tag}-step{step}"
            condition_names.append(condition)
            queue_path = output / "queues" / condition / "queue.json"
            queue = _queue(
                rows,
                condition=condition,
                checkpoint_ref=condition,
                config_path=config_paths[tag],
            )
            _write_json(queue_path, queue)
            queues[str(queue_path.relative_to(output))] = binding(queue_path)

    plan_manifest_path = PLAN_V4 / "manifest.json"
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    expected_plan_fit = plan_manifest["source"]["fit"]["sha256"]
    if expected_plan_fit != binding(FIT_PATH)["sha256"]:
        raise ValueError("production-plan-v4 fit binding does not match selection-v4 fit")
    schedule_path = PLAN_V4 / "pack-schedule.json"
    manifest = {
        "schema": "coordinate_codebook_alignment.conditional_lr_plan.v1",
        "status": "candidate_ready_for_parent_review",
        "model_calls": 0,
        "source": {
            "nominal_config": binding(NOMINAL),
            "production_plan_v4": binding(plan_manifest_path),
            "admission": binding(SELECTION / "admission.json"),
            "fit": binding(FIT_PATH),
            "pack_schedule": binding(schedule_path),
        },
        "training_contract": {
            "seed": 1729,
            "updates": 512,
            "effective_batch_size": 8,
            "train_order": "seeded_shuffle",
            "checkpoint_steps": list(STEPS),
            "source_config_fingerprint": nominal_resolved.fingerprint,
            "pack_schedule": binding(schedule_path),
            "fit_row_ids": row_ids,
        },
        "configs": configs,
        "evaluation": {
            "fit_count": len(rows),
            "conditions": condition_names,
            "decode": {
                "empty_prefix": True,
                "max_new_tokens": 3084,
                "repetition_penalty": 1.0,
                "sampling": False,
                "teacher_metrics": True,
            },
            "queues": queues,
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"status": "candidate_ready_for_parent_review", "model_calls": 0, "manifest": str(result)}))
