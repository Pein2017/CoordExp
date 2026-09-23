"""Prepare the authorized nominal seed-2718 repeat packet, CPU-only."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.artifacts import binding
from src.config.loader import load_train_config
from src.training.pack_cache import _rank_local_pack_indices, load_all_micro_steps_from_cache
from src.training.schedule import resolve_planned_step_schedule


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-22-coordinate-codebook-alignment"
).resolve()
REPO = Path(__file__).resolve().parents[3]
NOMINAL = REPO / "configs/research/coordinate_codebook_alignment/nominal.yaml"
REPEAT_CONFIG = REPO / "configs/research/coordinate_codebook_alignment/nominal-repeat-2718.yaml"
SELECTION = ROOT / "selection-v4"
CACHE = ROOT / "packing-cache/5ee09639e76f478c60debc4996c69db3924bf76b952b0673d7812726928a8ede"
PLAN_V4 = ROOT / "production-plan-v4"
REDUCTION = ROOT / "production/reductions/nominal-complete-v1.json"
TRAINING_RECEIPT = ROOT / "production/nominal-training-complete-v1.json"
OUTPUT = ROOT / "repeat-plan-v1"
SEED = 2718
STEPS = (32, 128, 512)
EFFECTIVE_BATCH = 8


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _diff_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if type(left) is not type(right):
        return [".".join(path)]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
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


def _without_repeat_fields(value: Any, path: tuple[str, ...] = ()) -> Any:
    if path in {("run", "name"), ("runtime", "seed")}:
        return None
    if isinstance(value, dict):
        return {
            key: _without_repeat_fields(item, (*path, str(key)))
            for key, item in value.items()
            if (*path, str(key)) not in {("run", "name"), ("runtime", "seed")}
        }
    if isinstance(value, list):
        return [_without_repeat_fields(item, path) for item in value]
    return value


def _make_repeat_config() -> dict[str, Any]:
    if REPEAT_CONFIG.exists():
        raise FileExistsError(f"refusing to overwrite repeat config: {REPEAT_CONFIG}")
    nominal_payload = yaml.safe_load(NOMINAL.read_text(encoding="utf-8"))
    repeat_payload = copy.deepcopy(nominal_payload)
    repeat_payload["run"]["name"] = "nominal-seed2718"
    repeat_payload["runtime"]["seed"] = SEED
    REPEAT_CONFIG.write_text(yaml.safe_dump(repeat_payload, sort_keys=False), encoding="utf-8")
    nominal = load_train_config(NOMINAL)
    repeat = load_train_config(REPEAT_CONFIG)
    if _diff_paths(_without_repeat_fields(nominal.config_dict), _without_repeat_fields(repeat.config_dict)):
        raise ValueError("repeat config has an unexpected semantic delta")
    if repeat.config.run.name != "nominal-seed2718" or repeat.config.runtime.seed != SEED:
        raise ValueError("repeat run name or seed mismatch")
    return {
        "binding": binding(REPEAT_CONFIG),
        "fingerprint": repeat.fingerprint,
        "semantic_delta_paths": ["run.name", "runtime.seed"],
        "nominal_fingerprint": nominal.fingerprint,
        "run_name": repeat.config.run.name,
        "seed": repeat.config.runtime.seed,
    }


def _pack_schedule(fit_rows: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    cache_manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    micro_steps = load_all_micro_steps_from_cache(CACHE, expected_fingerprint=str(cache_manifest["fingerprint"]))
    schedule = resolve_planned_step_schedule(config, packs_per_epoch=len(micro_steps), world_size=1)
    distributed = resolve_planned_step_schedule(config, packs_per_epoch=len(micro_steps), world_size=8)
    if schedule.resolved_max_steps != 512 or schedule.runtime_batch.effective_batch_size != EFFECTIVE_BATCH:
        raise ValueError("repeat schedule did not resolve to 512 updates and ebs 8")
    indices = _rank_local_pack_indices(schedule, rank=0, world_size=1, micro_step_count=len(micro_steps), shuffle_seed=SEED)
    rank_indices = [_rank_local_pack_indices(distributed, rank=rank, world_size=8, micro_step_count=len(micro_steps), shuffle_seed=SEED) for rank in range(8)]
    interleaved = tuple(rank_indices[rank][step] for step in range(512) for rank in range(8))
    if tuple(indices) != interleaved:
        raise ValueError("repeat seeded schedule is not rank-major equivalent")
    if len(indices) != 512 * EFFECTIVE_BATCH:
        raise ValueError("repeat schedule has the wrong number of presentations")
    row_ids = {str(row["_admission"]["row_id"]): int(row["image_id"]) for row in fit_rows}
    cache_ids = {str(item) for micro in micro_steps for item in micro.metadata["example_ids"]}
    if cache_ids != set(row_ids):
        raise ValueError("repeat schedule cache identities differ from frozen fit panel")
    packs = []
    for cache_index, micro in enumerate(micro_steps):
        ids = [str(item) for item in micro.metadata["example_ids"]]
        packs.append({"cache_index": cache_index, "pack_id": int(micro.metadata["pack_id"]), "row_ids": ids, "image_ids": [row_ids[item] for item in ids]})
    steps = []
    for step in range(512):
        selected = indices[step * EFFECTIVE_BATCH : (step + 1) * EFFECTIVE_BATCH]
        steps.append({"step": step + 1, "cache_indices": [int(index) for index in selected], "packs": [packs[int(index)] for index in selected]})
    return {
        "schema": "coordinate_codebook_alignment.repeat_schedule.v1",
        "seed": SEED,
        "shuffle_owner": "src.training.pack_cache._rank_local_pack_indices",
        "world_size": 1,
        "distributed_world_size": 8,
        "rank_major_equivalence_verified": True,
        "updates": 512,
        "effective_batch_size": EFFECTIVE_BATCH,
        "packs_per_epoch": len(micro_steps),
        "schedule": schedule.to_artifact_dict(),
        "distributed_schedule": distributed.to_artifact_dict(),
        "cache": {"manifest": binding(CACHE / "manifest.json"), "chunks": [binding(CACHE / item["path"]) for item in cache_manifest["chunks"]]},
        "packs": packs,
        "steps": steps,
    }


def _queue(rows: list[dict[str, Any]], condition: str, config_path: Path) -> dict[str, Any]:
    specs = []
    for index, row in enumerate(sorted(rows, key=lambda item: int(item["image_id"]))):
        admission = row["_admission"]
        image_id = int(row["image_id"])
        specs.append({"queue_index": index, "image_id": image_id, "row_id": str(admission["row_id"]), "condition": condition, "checkpoint_ref": condition, "config": str(config_path.resolve(strict=True)), "dataset": str((SELECTION / "fit.coord.jsonl").resolve(strict=True)), "cell_key": f"{condition}-{image_id}", "cohort": str(admission["cohort"])})
    return {"schema": "coordinate_codebook_alignment.evaluation_queue.v1", "status": "planned", "model_calls": 0, "monitor": False, "teacher_metrics": True, "empty_prefix": True, "max_new_tokens": 3084, "repetition_penalty": 1.0, "sampling": False, "specs": specs}


def prepare(output: Path = OUTPUT) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite repeat plan: {output}")
    for path in (NOMINAL, PLAN_V4 / "manifest.json", SELECTION / "admission.json", SELECTION / "fit.coord.jsonl", CACHE / "manifest.json", REDUCTION, TRAINING_RECEIPT):
        path.resolve(strict=True)
    config_meta = _make_repeat_config()
    resolved = load_train_config(REPEAT_CONFIG)
    fit_rows = _rows(SELECTION / "fit.coord.jsonl")
    if len(fit_rows) != 32 or len({str(row["_admission"]["row_id"]) for row in fit_rows}) != 32:
        raise ValueError("frozen fit panel is not exactly 32 unique admitted rows")
    schedule = _pack_schedule(fit_rows, resolved.config)
    output.mkdir(parents=True)
    schedule_path = output / "pack-schedule-seed2718.json"
    _write(schedule_path, schedule)
    queues = {}
    conditions = []
    for step in STEPS:
        condition = f"nominal-repeat-step{step}"
        conditions.append(condition)
        queue_path = output / "queues" / condition / "queue.json"
        _write(queue_path, _queue(fit_rows, condition, REPEAT_CONFIG))
        queues[str(queue_path.relative_to(output))] = binding(queue_path)
    chosen_checkpoint_root = ROOT / "training/nominal-seed1729/checkpoints/step-512"
    manifest = {
        "schema": "coordinate_codebook_alignment.repeat_plan.v1",
        "status": "candidate_ready_for_parent_review",
        "model_calls": 0,
        "source": {
            "nominal_config": binding(NOMINAL),
            "repeat_config": config_meta["binding"],
            "production_plan": binding(PLAN_V4 / "manifest.json"),
            "admission": binding(SELECTION / "admission.json"),
            "fit": binding(SELECTION / "fit.coord.jsonl"),
            "chosen_nominal_reduction": binding(REDUCTION),
            "nominal_training_receipt": binding(TRAINING_RECEIPT),
            "chosen_checkpoint_root": str(chosen_checkpoint_root.resolve(strict=True)),
        },
        "training": {"seed": SEED, "updates": 512, "effective_batch_size": EFFECTIVE_BATCH, "checkpoint_steps": list(STEPS), "train_order": "seeded_shuffle", "config_fingerprint": resolved.fingerprint, "pack_schedule": binding(schedule_path), "fit_row_ids": [str(row["_admission"]["row_id"]) for row in fit_rows]},
        "config_delta": {"paths": config_meta["semantic_delta_paths"], "nominal_fingerprint": config_meta["nominal_fingerprint"], "repeat_fingerprint": config_meta["fingerprint"]},
        "evaluation": {"fit_count": 32, "conditions": conditions, "decode": {"empty_prefix": True, "max_new_tokens": 3084, "repetition_penalty": 1.0, "sampling": False, "teacher_metrics": True}, "queues": queues},
    }
    manifest_path = output / "manifest.json"
    _write(manifest_path, manifest)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"status": "candidate_ready_for_parent_review", "model_calls": 0, "manifest": str(result)}))
