"""Freeze the CPU production packet for the nominal coordinate-codebook fit.

This preparation entry only reads the maintained config/cache and frozen input
bindings. It writes JSON manifests and queue specifications; it never loads
model weights or launches a model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from probes.training_set_completion.artifacts import binding
from src.config.loader import load_train_config
from src.training.pack_cache import _rank_local_pack_indices, load_all_micro_steps_from_cache
from src.training.schedule import resolve_planned_step_schedule


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment")
REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/research/coordinate_codebook_alignment/nominal.yaml"
SELECTION = ROOT / "selection-v4"
RUNTIME_INPUTS = ROOT / "runtime-inputs-v2"
CACHE = ROOT / "packing-cache/5ee09639e76f478c60debc4996c69db3924bf76b952b0673d7812726928a8ede"
PLAN = ROOT / "production-plan-v1"
SEED = 1729
UPDATES = 512
EFFECTIVE_BATCH = 8
FIT_CONDITIONS = ("source", "nominal-step32", "nominal-step128", "nominal-step512")
MONITOR_CONDITIONS = ("source", "chosen-final")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verify_projection() -> None:
    projection = json.loads((RUNTIME_INPUTS / "projection.json").read_text())
    for record in projection:
        source = Path(record["source"]["path"]).resolve(strict=True)
        runtime = Path(record["runtime"]["path"]).resolve(strict=True)
        if binding(source)["sha256"] != record["source"]["sha256"] or binding(runtime)["sha256"] != record["runtime"]["sha256"]:
            raise ValueError(f"projection binding drift: {source} / {runtime}")
        source_rows = _rows(source)
        runtime_rows = _rows(runtime)
        if len(source_rows) != len(runtime_rows) or len(source_rows) != int(record["count"]):
            raise ValueError(f"projection count mismatch: {source}")
        for source_row, runtime_row in zip(source_rows, runtime_rows, strict=True):
            source_images = [Path(str(item)).resolve(strict=True) for item in source_row["images"]]
            runtime_images = [(runtime.parent / str(item)).resolve(strict=True) for item in runtime_row["images"]]
            if source_images != runtime_images or any(hashlib.sha256(left.read_bytes()).digest() != hashlib.sha256(right.read_bytes()).digest() for left, right in zip(source_images, runtime_images, strict=True)):
                raise ValueError(f"projection image identity mismatch: {source}")
            source_fields = {key: value for key, value in source_row.items() if key not in {"_admission", "images"}}
            runtime_fields = {key: value for key, value in runtime_row.items() if key != "images"}
            if source_fields != runtime_fields:
                raise ValueError(f"projection non-image field mismatch: {source}")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _pack_schedule(config: Any, fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache_manifest = json.loads((CACHE / "manifest.json").read_text())
    micro_steps = load_all_micro_steps_from_cache(CACHE, expected_fingerprint=str(cache_manifest["fingerprint"]))
    schedule = resolve_planned_step_schedule(config, packs_per_epoch=len(micro_steps), world_size=1)
    distributed_schedule = resolve_planned_step_schedule(config, packs_per_epoch=len(micro_steps), world_size=8)
    if schedule.resolved_max_steps != UPDATES or schedule.runtime_batch.effective_batch_size != EFFECTIVE_BATCH:
        raise ValueError("nominal schedule does not resolve to 512 updates and effective batch 8")
    indices = _rank_local_pack_indices(schedule, rank=0, world_size=1, micro_step_count=len(micro_steps), shuffle_seed=SEED)
    rank_indices = [_rank_local_pack_indices(distributed_schedule, rank=rank, world_size=8, micro_step_count=len(micro_steps), shuffle_seed=SEED) for rank in range(8)]
    distributed_indices = tuple(rank_indices[rank][step] for step in range(UPDATES) for rank in range(8))
    if tuple(indices) != distributed_indices:
        raise ValueError("logical seeded schedule differs from rank-major world-size-8 schedule")
    if len(indices) != UPDATES * EFFECTIVE_BATCH:
        raise ValueError("maintained seeded schedule length mismatch")
    row_ids = {str(row["_admission"]["row_id"]): int(row["image_id"]) for row in fit_rows}
    cache_ids = {str(example_id) for micro in micro_steps for example_id in micro.metadata["example_ids"]}
    if cache_ids != set(row_ids):
        raise ValueError(f"packing cache row identities differ from fit admission: missing={sorted(set(row_ids)-cache_ids)} extra={sorted(cache_ids-set(row_ids))}")
    packs = []
    for cache_index, micro in enumerate(micro_steps):
        ids = [str(item) for item in micro.metadata["example_ids"]]
        packs.append({"cache_index": cache_index, "pack_id": int(micro.metadata["pack_id"]), "row_ids": ids, "image_ids": [row_ids[item] for item in ids]})
    steps = []
    for step in range(UPDATES):
        selected = indices[step * EFFECTIVE_BATCH : (step + 1) * EFFECTIVE_BATCH]
        steps.append({"step": step + 1, "cache_indices": [int(index) for index in selected], "packs": [packs[int(index)] for index in selected]})
    return {"seed": SEED, "shuffle_owner": "src.training.pack_cache._rank_local_pack_indices", "world_size": 1, "distributed_world_size_equivalent": 8, "rank_major_equivalence_verified": True, "updates": UPDATES, "effective_batch_size": EFFECTIVE_BATCH, "packs_per_epoch": len(micro_steps), "schedule": schedule.to_artifact_dict(), "distributed_schedule": distributed_schedule.to_artifact_dict(), "cache": {"manifest": binding(CACHE / "manifest.json"), "chunks": [binding(CACHE / item["path"]) for item in cache_manifest["chunks"]]}, "packs": packs, "steps": steps}


def _queue(name: str, rows: list[dict[str, Any]], conditions: tuple[str, ...], *, monitor: bool) -> dict[str, Any]:
    specs = []
    for condition in conditions:
        for row in sorted(rows, key=lambda item: int(item["image_id"])):
            image_id = int(row["image_id"])
            specs.append({"queue_index": len(specs), "image_id": image_id, "row_id": str(row["_admission"]["row_id"]), "condition": condition, "checkpoint_ref": condition, "dataset": str((SELECTION / ("monitor.coord.jsonl" if monitor else "fit.coord.jsonl")).resolve(strict=True)), "cell_key": f"{condition}-{image_id}", "cohort": str(row["_admission"]["cohort"])})
    return {"schema": "coordinate_codebook_alignment.evaluation_queue.v1", "status": "planned", "model_calls": 0, "monitor": monitor, "teacher_metrics": True, "empty_prefix": True, "max_new_tokens": 3084, "repetition_penalty": 1.0, "specs": specs}


def prepare(output: Path = PLAN) -> Path:
    if output.exists():
        raise FileExistsError(f"production preparation already exists: {output}")
    resolved = load_train_config(CONFIG)
    _verify_projection()
    fit = _rows(SELECTION / "fit.coord.jsonl")
    monitor = _rows(SELECTION / "monitor.coord.jsonl")
    if len(fit) != 32 or len(monitor) != 64:
        raise ValueError("frozen admission counts changed")
    if {str(row["_admission"]["row_id"]) for row in fit} & {str(row["_admission"]["row_id"]) for row in monitor}:
        raise ValueError("fit and monitor identities overlap")
    schedule = _pack_schedule(resolved.config, fit)
    output.mkdir(parents=True)
    queues = {}
    for condition in FIT_CONDITIONS:
        queue = _queue(condition, fit, (condition,), monitor=False)
        path = output / "queues" / f"fit-{condition}" / "queue.json"
        _write(path, queue)
        queues[str(path.relative_to(output))] = binding(path)
    for condition in MONITOR_CONDITIONS:
        monitor_queue = _queue(condition, monitor, (condition,), monitor=True)
        monitor_path = output / "queues" / f"monitor-{condition}" / "queue.json"
        _write(monitor_path, monitor_queue)
        queues[str(monitor_path.relative_to(output))] = binding(monitor_path)
    _write(output / "pack-schedule.json", schedule)
    manifest = {
        "schema": "coordinate_codebook_alignment.production_plan.v1",
        "status": "candidate_ready_for_parent_review",
        "model_calls": 0,
        "source": {"config": binding(CONFIG), "admission": binding(SELECTION / "admission.json"), "fit": binding(SELECTION / "fit.coord.jsonl"), "monitor": binding(SELECTION / "monitor.coord.jsonl"), "runtime_projection": binding(RUNTIME_INPUTS / "projection.json"), "runtime_fit": binding(RUNTIME_INPUTS / "fit.coord.jsonl"), "runtime_monitor": binding(RUNTIME_INPUTS / "monitor.coord.jsonl")},
        "training": {"config_fingerprint": resolved.fingerprint, "seed": SEED, "updates": UPDATES, "effective_batch_size": EFFECTIVE_BATCH, "precision": "bf16", "attention": "flash_attention_2", "train_order": "seeded_shuffle", "checkpoint_steps": [32, 128, 512], "pack_schedule": binding(output / "pack-schedule.json")},
        "packing_cache": {"root": str(CACHE.resolve(strict=True)), "manifest": binding(CACHE / "manifest.json"), "micro_step_count": schedule["packs_per_epoch"], "row_identity_contract": "each cache pack binds metadata.example_ids to selection-v4 _admission.row_id"},
        "evaluation": {"fit_image_count": len(fit), "fit_conditions": list(FIT_CONDITIONS), "monitor_image_count": len(monitor), "monitor_conditions": list(MONITOR_CONDITIONS), "input_dataset_source": "selection-v4", "monitor_selection_policy": "freeze monitor image identities now; evaluate source and the placeholder chosen-final only after fit-only checkpoint selection; no checkpoint is selected by this preparation and monitor is not launched", "decode": {"empty_prefix": True, "max_new_tokens": 3084, "repetition_penalty": 1.0, "sampling": False, "teacher_metrics": True}, "queues": queues},
        "future_lr_variants": {"status": "prepared_not_launched", "scales": {"0.3x": 0.3, "3x": 3.0}, "base_group_lrs": resolved.config.optimizer.model_dump(mode="json").get("groups", {})},
    }
    _write(output / "manifest.json", manifest)
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLAN)
    args = parser.parse_args()
    path = prepare(args.output)
    print(json.dumps({"status": "candidate_ready_for_parent_review", "model_calls": 0, "manifest": str(path)}))
