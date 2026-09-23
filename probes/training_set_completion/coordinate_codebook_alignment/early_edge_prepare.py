"""Freeze the early-edge run from the accepted three-loss packet and cells."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
from pathlib import Path

from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment.scale_prepare import PARENT, REPO, V3
from src.config.loader import load_train_config
from src.training.schedule import resolve_planned_step_schedule


ROOT = PARENT / "2026-09-23-early-edge-codebook"
THREE = PARENT / "2026-09-22-coordinate-codebook-three-loss"
RUN = ROOT / "early-edge1024-seed1729-16epoch"
RUN_EIGHT = ROOT / "early-edge1024-seed1729-16epoch-eight-v1"
CONDITIONS = (("early_edge_epoch8", "three_loss_epoch8", 492),
              ("early_edge_epoch16", "three_loss_epoch16", 984))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def prepare() -> None:
    old = json.loads((THREE / "training-config-v1.json").read_text())
    config = copy.deepcopy(old)
    config["model"]["coordinate_codebook"].update(
        mode="early_patch_edges", projection_seed=1729,
    )
    config["optimizer"]["groups"]["coordinate_codebook_projection"] = {
        "lr": 2e-5, "weight_decay": 0.0,
    }
    config["run"].update(
        artifact_root=str(ROOT), name=RUN.name, output_dir=RUN.name,
    )
    _write(ROOT / "training-config-v1.json", config)
    changes = []
    def diff(a: object, b: object, key: str = "") -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for child in sorted(set(a) | set(b)):
                diff(a.get(child), b.get(child), f"{key}.{child}".strip("."))
        elif a != b:
            changes.append({"field": key, "before": a, "after": b})
    diff(old, config)
    assert {item["field"] for item in changes} == {
        "model.coordinate_codebook.mode",
        "model.coordinate_codebook.projection_seed",
        "optimizer.groups.coordinate_codebook_projection",
        "run.artifact_root", "run.name", "run.output_dir",
    }
    _write(ROOT / "config-delta-v1.json", {
        "source": binding(THREE / "training-config-v1.json"), "changes": changes,
    })
    schedule = resolve_planned_step_schedule(
        load_train_config(ROOT / "training-config-v1.json").config,
        packs_per_epoch=492, world_size=4,
    )
    assert schedule.resolved_max_steps == 984 and schedule.tail_fill_pack_count == 0
    prior = json.loads((THREE / "analytical-cells.json").read_text())
    by_condition = {
        name: [s for s in prior["specs"] if s["condition"] == name]
        for name in ("source", "three_loss_epoch8", "three_loss_epoch16")
    }
    assert [len(by_condition[k]) for k in by_condition] == [1280, 96, 1280]
    reused = []
    for condition, rows in by_condition.items():
        for spec in rows:
            item = copy.deepcopy(spec)
            path = (Path(item["reuse"]["path"]) if condition == "source"
                    else THREE / "production" / condition / "cells" / f"{item['cell_key']}.json")
            receipt = binding(path)
            if condition == "source":
                assert receipt["sha256"] == item["reuse"]["sha256"]
            item["reuse"] = receipt
            reused.append(item)
    assert len({(s["condition"], s["row_id"]) for s in reused}) == 2656
    evaluation = {}
    new = []
    for condition, old_condition, step in CONDITIONS:
        checkpoint = RUN / "checkpoints" / f"step-{step}"
        queue = copy.deepcopy(json.loads((THREE / "queues" / old_condition / "queue.json").read_text()))
        specs = []
        for index, source in enumerate(by_condition[old_condition]):
            item = copy.deepcopy(source)
            item.update(condition=condition, checkpoint_ref=str(checkpoint),
                        cell_key=f"{condition}-{item['image_id']}", queue_index=index)
            item.pop("reuse", None)
            specs.append(item)
        queue.update(specs=specs, status="frozen_early_edge_authorized")
        path = ROOT / "queues" / condition / "queue.json"
        _write(path, queue)
        assert not path.with_name("queue-state.json").exists()
        evaluation[condition] = {
            "queue": str(path), "checkpoint": str(checkpoint),
            "output": str(ROOT / "production" / condition), "cells": len(specs),
        }
        new.extend(specs)
    assert len(new) == 1376 and len(reused) + len(new) == 4032
    _write(ROOT / "analytical-cells.json", {
        "specs": reused + new,
        "retained32_row_ids": prior["retained32_row_ids"],
        "count": 4032, "new": 1376, "reused": 2656,
        "trajectory_cells": 2656,
    })
    _write(ROOT / "evaluation-plan-v1.json", evaluation)
    _write(ROOT / "packet-v1.json", {
        "status": "CPU_packet", "root": str(ROOT), "model_calls": 0,
        "training_schedule": binding(THREE / "packing-exposure-v1.json"),
        "admission": binding(THREE / "evaluation-admission.json"),
        "bindings": [binding(ROOT / name) for name in (
            "training-config-v1.json", "config-delta-v1.json",
            "analytical-cells.json", "evaluation-plan-v1.json",
        )],
    })


def prepare_eight() -> None:
    """Rebind the identical global packs to the lead-authorized eight-rank route."""
    old = json.loads((ROOT / "training-config-v1.json").read_text())
    config = copy.deepcopy(old)
    config["run"].update(name=RUN_EIGHT.name, output_dir=RUN_EIGHT.name)
    _write(ROOT / "training-config-v2.json", config)
    schedule = resolve_planned_step_schedule(
        load_train_config(ROOT / "training-config-v2.json").config,
        packs_per_epoch=492, world_size=8,
    )
    assert (schedule.resolved_max_steps, schedule.runtime_batch.resolved_grad_accum_steps,
            schedule.runtime_batch.effective_batch_size, schedule.tail_fill_pack_count) == (984, 1, 8, 0)
    prior_path = THREE / "packing-exposure-v1.json"
    prior = json.loads(prior_path.read_text())
    exposure = copy.deepcopy(prior)
    exposure["schedule"] = schedule.to_artifact_dict()
    exposure["rank_checks"]["8"] = {
        "gradient_accumulation": 1, "microsteps_per_rank": 984, "rank_major_exact": True,
    }
    exposure["topology_amendment"] = {
        "source": binding(prior_path), "world_size": 8, "gradient_accumulation": 1,
        "global_pack_indices_unchanged": True,
    }
    assert exposure["global_pack_indices"] == prior["global_pack_indices"]
    assert exposure["packs"] == prior["packs"]
    assert len(exposure["global_pack_indices"]) == 7872
    _write(ROOT / "packing-exposure-v2.json", exposure)

    analytical = json.loads((ROOT / "analytical-cells.json").read_text())
    evaluation = {}
    for condition, _, step in CONDITIONS:
        checkpoint = RUN_EIGHT / "checkpoints" / f"step-{step}"
        old_queue = ROOT / "queues" / condition / "queue.json"
        queue = json.loads(old_queue.read_text())
        for item in queue["specs"]:
            assert item["condition"] == condition
            item["checkpoint_ref"] = str(checkpoint)
        path = ROOT / "queues-eight" / condition / "queue.json"
        _write(path, queue)
        evaluation[condition] = {
            "queue": str(path), "checkpoint": str(checkpoint),
            "output": str(ROOT / "production" / condition), "cells": len(queue["specs"]),
        }
    for item in analytical["specs"]:
        if item["condition"] in evaluation:
            item["checkpoint_ref"] = evaluation[item["condition"]]["checkpoint"]
    assert len(analytical["specs"]) == 4032
    _write(ROOT / "analytical-cells-v2.json", analytical)
    _write(ROOT / "evaluation-plan-v2.json", evaluation)
    diagnostic = json.loads((ROOT / "diagnostic-plan-v1.json").read_text())
    diagnostic["checkpoint"] = str(RUN_EIGHT / "checkpoints/step-984")
    _write(ROOT / "diagnostic-plan-v2.json", diagnostic)
    ruling = REPO / "research/experiments/2026-09-23-early-edge-codebook/lead-ruling-02-eight-gpu-sequential.md"
    _write(ROOT / "packet-v2.json", {
        "status": "CPU_eight_rank_packet", "root": str(ROOT), "model_calls": 0,
        "source_packet": binding(ROOT / "packet-v1.json"), "authority": binding(ruling),
        "bindings": [binding(ROOT / name) for name in (
            "training-config-v2.json", "packing-exposure-v2.json",
            "analytical-cells-v2.json", "evaluation-plan-v2.json",
            "diagnostic-plan-v2.json", "qualification/actual-failed-grid-v1.json",
        )] + [binding(Path(item["queue"])) for item in evaluation.values()],
    })


def freeze_launch(*, version: str = "v1") -> None:
    """Capture the final implementation and bind every model-entry input."""
    from src.artifacts.source_provenance import preserve_source
    from src.config.paths import resolve_run_directory
    from src.qwen import load_qwen_components
    from src.training.pack_cache import build_packing_cache_fingerprint, load_cache_manifest

    if version not in {"v1", "v2"}:
        raise ValueError("only bound early-edge launches v1/v2 are supported")
    run_dir = RUN if version == "v1" else RUN_EIGHT
    cfg = load_train_config(ROOT / f"training-config-{version}.json").config
    assert resolve_run_directory(cfg, cwd=REPO).run_dir == run_dir and not run_dir.exists()
    assert cfg.run.collision_policy == "fail"
    components = load_qwen_components(cfg, load_model=False)
    fingerprint = build_packing_cache_fingerprint(cfg, components, dataset=cfg.data.train, split="train")
    cache = V3 / "packing-cache" / fingerprint
    assert fingerprint == "78406920e488c24dfbe4fd37711dcbc45ff54322e8a6bc558de470a1d19c0e18"
    assert load_cache_manifest(cache, expected_fingerprint=fingerprint)["micro_step_count"] == 492
    old_launch = json.loads((THREE / "launch-v1.json").read_text())
    inherited_inputs = []
    for item in old_launch["bindings"]:
        path = Path(item["path"])
        if str(path).startswith(str(REPO)) or "/queues/" in str(path):
            continue
        assert binding(path)["sha256"] == item["sha256"], path
        inherited_inputs.append(item)
    evaluation = json.loads((ROOT / f"evaluation-plan-{version}.json").read_text())
    assert len({Path(item["queue"]).parent for item in evaluation.values()}) == 2
    assert all(not Path(item["queue"]).with_name("queue-state.json").exists()
               for item in evaluation.values())
    authority = REPO / "research/experiments/2026-09-23-early-edge-codebook/unit.md"
    sources = list((REPO / "src").rglob("*.py"))
    sources += list(Path(__file__).parent.glob("*.py"))
    sources += [authority, REPO / "probes/training_set_completion/artifacts.py"]
    if version == "v2":
        sources.append(REPO / "research/experiments/2026-09-23-early-edge-codebook/lead-ruling-02-eight-gpu-sequential.md")
    captures = []
    for source in sorted(set(sources)):
        capture = preserve_source(source, run_root=ROOT / f"launch-{version}.json",
                                  relative_name=source.relative_to(REPO))
        captures.append({"current": binding(source), "capture": binding(capture)})
    for name in ("transformers.models.qwen3_vl.modeling_qwen3_vl",
                 "transformers.models.qwen3_vl.processing_qwen3_vl",
                 "peft.tuners.lora.layer", "peft.tuners.lora.dora"):
        source = Path(importlib.import_module(name).__file__)
        capture = preserve_source(source, run_root=ROOT / f"launch-{version}.json",
                                  relative_name=Path("runtime") / f"{name}.py")
        captures.append({"current": binding(source), "capture": binding(capture)})
    local_inputs = (
        "training-config-v1.json", "config-delta-v1.json", "analytical-cells.json",
        "evaluation-plan-v1.json", "packet-v1.json", "qualification/tiny-config-v1.json",
        "preparation/pre-entry-reduction-v1.json", "diagnostic-plan-v1.json",
    ) if version == "v1" else (
        "training-config-v2.json", "packing-exposure-v2.json", "analytical-cells-v2.json",
        "evaluation-plan-v2.json", "packet-v2.json", "diagnostic-plan-v2.json",
        "qualification/actual-failed-grid-v1.json", "qualification/first-four-rank-failure-v1.json",
    )
    records = inherited_inputs + [binding(ROOT / name) for name in local_inputs]
    records += [binding(Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment/runtime-inputs-v2/qualification.coord.jsonl"))]
    records += [binding(THREE / "evaluation-admission.json")]
    records += [binding(THREE / "packing-exposure-v1.json") if version == "v1"
                else binding(ROOT / "packing-exposure-v2.json")]
    records += [binding(Path(item["queue"])) for item in evaluation.values()]
    records += [binding(authority), binding(ROOT / "preparation/implementation-allowlist-v3.json")]
    records += [item["current"] for item in captures]
    argv = ["python", "-B", "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=4" if version == "v1" else "--nproc_per_node=8", "-m",
            "probes.training_set_completion.coordinate_codebook_alignment.three_loss_train"
            if version == "v1" else "probes.training_set_completion.coordinate_codebook_alignment.early_edge_train",
            "--config", str(ROOT / f"training-config-{version}.json"),
            "--output", str(ROOT / ("first-production.json" if version == "v1" else "first-production-eight-v1.json")),
            "--packing-plan", str(THREE / "packing-exposure-v1.json" if version == "v1"
                                  else ROOT / "packing-exposure-v2.json")]
    _write(ROOT / f"launch-{version}.json", {
        "status": "mechanical_launch_checks_passed", "root": str(ROOT), "run_dir": str(run_dir),
        "cache_root": str(V3 / "packing-cache"), "cache_fingerprint": fingerprint,
        "training_argv": argv, "training_environment": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3" if version == "v1" else "0,1,2,3,4,5,6,7", "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1", "coordexp_infras_PACK_CACHE_ROOT": str(V3 / "packing-cache"),
        },
        "evaluation": evaluation,
        "evaluation_admission": str(THREE / "evaluation-admission.json"),
        "condition_order_during": ["early_edge_epoch8", "early_edge_epoch16"] if version == "v1" else [],
        "condition_order_after": ["early_edge_epoch16", "early_edge_epoch8"],
        "bindings": records, "source_captures": captures,
        "limits": old_launch["limits"], "model_calls": 0,
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "peft", "accelerate", "flash-attn")},
        "reuse_qualification": old_launch["reuse_qualification"],
    })


if __name__ == "__main__":
    prepare()
