"""CPU contract tests for the S K10/H20 crossover runner."""

from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest
import torch

from scripts.research import materialize_s_k10_h20_crossover_plan as planner
from scripts.research import natural_boundary_attention_actuators as attention
from scripts.research import run_s_k10_h20_crossover_shard as runner
from scripts.research import run_s_primary_natural_boundary_gate as gate


REAL_PRE_GPU_V3 = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-07-s-k10-h20-natural-crossover/pre-gpu-receipt-v3/"
    "pre-gpu-receipt.json"
)
real_crossover_artifacts = pytest.mark.skipif(
    not REAL_PRE_GPU_V3.is_file(),
    reason="frozen crossover source artifacts are unavailable",
)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner._canonical(value) + b"\n")
    return path


def _full_event(index: int) -> dict[str, Any]:
    event_id = planner.EVENT_IDS[index]
    target_owner_id = event_id
    covered_owner_id = f"covered:{index}"
    prefix_token_ids = [151646, 1000 + index, 151649]
    prefix_sha256 = runner.sha256_json(prefix_token_ids)
    geometry: dict[str, Any] = {
        "target_owner_id": target_owner_id,
        "covered_A_owner_id": covered_owner_id,
        "regions": {"a_exclusive": [index + 1], "b_exclusive": [index + 2]},
    }
    geometry["geometry_sha256"] = runner.sha256_json(geometry)
    event: dict[str, Any] = {
        "admission": {"status": "admitted"},
        "checkpoint": "S",
        "eligibility": {"launch_eligible": True},
        "event_id": event_id,
        "event_index": index,
        "image_id": 100 + index,
        "geometry": geometry,
        "geometry_sha256": geometry["geometry_sha256"],
        "geometry_source": {"status": "bound"},
        "geometry_supersession": {"status": "current"},
        "image_cell_region_receipts": {},
        "image_cell_regions": {"a_exclusive": [index + 1], "b_exclusive": [index + 2]},
        "natural_boundary": {
            "history_sha256": prefix_sha256,
            "history_token_ids": prefix_token_ids,
            "opener_injected": False,
            "opener_seeded": False,
            "pre_opener_natural": True,
            "prefix_sha256": prefix_sha256,
            "prefix_token_ids": prefix_token_ids,
        },
        "owner_refs": {
            "covered_A_owner_id": covered_owner_id,
            "covered_owner_ids": [covered_owner_id],
            "derived_panel_object_index": index,
            "gt_owner_id": target_owner_id,
            "source_panel_object_index": index + 10,
        },
        "same_class_competitor_owner_id": None,
        "step": runner.STEP,
        "substrate": "four-coordinate geo_sorted_xy",
    }
    event["event_sha256"] = runner.sha256_json(event)
    return event


def _plan_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return planner._source_event_binding(
        {
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "target_owner_id": event["owner_refs"]["gt_owner_id"],
        },
        event,
    )


def _rehash_event(event: dict[str, Any]) -> None:
    event.pop("event_sha256", None)
    event["event_sha256"] = runner.sha256_json(event)


def _path_ref(path: Path, *, directory: bool = False) -> dict[str, Any]:
    digest = runner._sha256_directory(path) if directory else runner._sha256_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "kind": "directory" if directory else "file",
    }


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested
            for item in value.values()
            for nested in _nested_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _nested_keys(item)}
    return set()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, dict[str, Any]]:
    full_events = [_full_event(index) for index in range(3)]
    events = [_plan_event(event) for event in full_events]
    manifest = _write_json(tmp_path / "manifest.json", {"events": full_events})
    census = _write_json(tmp_path / "census.json", {"rows": []})
    config = (tmp_path / "config.yaml").resolve()
    panel = (tmp_path / "panel.jsonl").resolve()
    cohort = (tmp_path / "cohort.json").resolve()
    cohort_manifest = _write_json(tmp_path / "cohort-manifest.json", {"status": "sealed"})
    config.write_text("model: S\n", encoding="utf-8")
    panel.write_text("{}\n", encoding="utf-8")
    cohort.write_text("{}\n", encoding="utf-8")
    h0_root = tmp_path / "h0-root"
    h0_dir = h0_root / "h0"
    base_model = tmp_path / "base-model"
    h0_dir.mkdir(parents=True, exist_ok=True)
    base_model.mkdir(exist_ok=True)
    (h0_root / "root.txt").write_text("root\n", encoding="utf-8")
    (h0_dir / "h0.txt").write_text("h0\n", encoding="utf-8")
    (base_model / "weights.bin").write_bytes(b"weights")
    execution_root = (tmp_path / "execution").resolve()
    final_root = (tmp_path / "final").resolve()
    device_plan = dict(planner.DEVICE_PLAN)
    plan: dict[str, Any] = {
        "schema_version": planner.SCHEMA_VERSION,
        "status": "planned",
        "unit_id": runner.UNIT_ID,
        "primary": dict(planner.PRIMARY),
        "source_bindings": {
            "evidence": {"path": str(tmp_path / "evidence.json")},
            "receipt": {"path": str(tmp_path / "source.receipt.json")},
            "manifest": {"path": str(manifest), "raw_sha256": runner._sha256_file(manifest)},
            "census": {"path": str(census), "raw_sha256": runner._sha256_file(census)},
            "original_plan": {"path": str(tmp_path / "original-plan.json")},
            "gate_result": {"path": str(tmp_path / "gate.json")},
        },
        "events": events,
        "event_count": 3,
        "image_count": 3,
        "cell_order": list(runner.CELLS),
        "technical_control": "C00",
        "cells": {cell: {"cell_id": cell} for cell in runner.CELLS},
        "operator_contract": {
            "admission_mode": planner.OPENER_MODE,
            "opener_injected": False,
            "opener_generated_by_model": True,
            "use_cache": False,
            "max_rows": planner.MAX_ROWS,
            "max_row_tokens": planner.MAX_ROW_TOKENS,
            "full_endpoint_vectors_required": True,
            "no_training": True,
        },
        "device_plan": device_plan,
        "shards": [
            {
                "shard_index": index,
                "shard_id": f"shard-{index:03d}",
                "physical_device": device_plan[f"shard-{index:03d}"],
                "logical_device": "cuda:0",
                "event": events[index],
                "events": [events[index]],
            }
            for index in range(3)
        ],
        **planner.FROZEN_FLAGS,
        "no_legacy_no_2x2_reuse": True,
        "no_legacy_plan_reselection": True,
    }
    plan["self_sha256"] = runner.sha256_json(plan)
    plan_path = _write_json(tmp_path / "plan.json", plan)

    input_paths = {
        "manifest": manifest,
        "census": census,
        "execution_plan": plan_path,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model,
    }
    input_bindings = {
        key: _path_ref(path, directory=key in {"h0_root", "h0_dir", "base_model_dir"})
        for key, path in input_paths.items()
    }
    input_hashes = {
        "manifest_raw_sha256": input_bindings["manifest"]["sha256"],
        "manifest_self_sha256": "a" * 64,
        "census_v3_raw_sha256": input_bindings["census"]["sha256"],
        "census_v3_self_sha256": "b" * 64,
        "execution_plan_raw_sha256": input_bindings["execution_plan"]["sha256"],
        "execution_plan_sha256": plan["self_sha256"],
        "config_sha256": input_bindings["config"]["sha256"],
        "panel_sha256": input_bindings["panel"]["sha256"],
        "cohort_sha256": input_bindings["cohort"]["sha256"],
        "cohort_manifest_raw_sha256": input_bindings["cohort_manifest"]["sha256"],
        "h0_root_sha256": input_bindings["h0_root"]["sha256"],
        "h0_dir_sha256": input_bindings["h0_dir"]["sha256"],
        "base_model_inventory_sha256": input_bindings["base_model_dir"]["sha256"],
    }
    receipt: dict[str, Any] = {
        "schema_version": "s_k10_h20_crossover_pre_gpu_receipt.v1",
        "status": "sealed_pre_gpu",
        "unit_id": runner.UNIT_ID,
        "plan": {"path": str(plan_path), "sha256": runner._sha256_file(plan_path)},
        "plan_self_sha256": plan["self_sha256"],
        "roots": {
            "execution_root": {"path": str(execution_root), "status": "reserved_absent_pre_gpu"},
            "final_root": {"path": str(final_root), "status": "reserved_absent_pre_gpu"},
        },
        "input_bindings": input_bindings,
        "input_paths": {key: str(path.resolve()) for key, path in input_paths.items()},
        "input_hashes": input_hashes,
        "runtime": {"python_version": "test", "torch_version": "test", "transformers_version": "test"},
        "device_plan": device_plan,
        "source_files": {"runner": {"path": str(Path(runner.__file__).resolve()), "sha256": "c" * 64}},
        "event_bindings": [
            {key: event[key] for key in ("event_index", "event_id", "image_id", "event_sha256", "prefix_sha256", "geometry_sha256")}
            for event in events
        ],
        "no_training": True,
        "launch_scope": "one_fixed_three_shard_no_training_launch",
    }
    receipt["self_sha256"] = runner.sha256_json(receipt)
    receipt_path = _write_json(tmp_path / "pre-gpu.json", receipt)

    def fake_plan_validate(path: str | Path) -> dict[str, Any]:
        assert Path(path).resolve() == plan_path.resolve()
        return {"plan": copy.deepcopy(plan), "plan_sha256": plan["self_sha256"], "self_sha256": plan["self_sha256"], "path": str(plan_path)}

    def fake_receipt_validate(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        assert Path(path).resolve() == receipt_path.resolve()
        assert kwargs["phase"] == "runtime"
        selected_shard = kwargs["shard_id"]
        assert selected_shard in device_plan
        assert kwargs["output_root"] == str(execution_root / selected_shard)
        return {"receipt": copy.deepcopy(receipt), "self_sha256": receipt["self_sha256"], "receipt_sha256": runner.sha256_json(receipt), "path": str(receipt_path)}

    monkeypatch.setattr(planner, "validate_plan", fake_plan_validate)
    from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer

    monkeypatch.setattr(sealer, "validate_pre_gpu_receipt", fake_receipt_validate)
    runtime_identity = {
        "schema_version": "s_k10_h20_crossover_runtime_identity_binding.v1",
        "unit_id": runner.UNIT_ID,
        "pre_gpu_receipt_path": str(receipt_path),
        "pre_gpu_receipt_sha256": runner._sha256_file(receipt_path),
        "pre_gpu_receipt_self_sha256": receipt["self_sha256"],
        "plan_self_sha256": plan["self_sha256"],
        "input_paths": {key: str(path.resolve()) for key, path in input_paths.items()},
        "input_hashes": input_hashes,
        "code_hashes": {"runner": "c" * 64},
        "runtime": receipt["runtime"],
        "device_assignment": {
            "shard_id": "shard-000",
            "shard_index": 0,
            "physical_device": device_plan["shard-000"],
            "observed_cuda_visible_devices": device_plan["shard-000"],
            "logical_device": "cuda:0",
        },
    }

    def fake_runtime_binding(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        selected_shard = kwargs["shard_id"]
        selected_index = int(selected_shard.removeprefix("shard-"))
        physical_device = device_plan[selected_shard]
        assert kwargs["observed_cuda_visible_devices"] == physical_device
        selected_identity = copy.deepcopy(runtime_identity)
        selected_identity["device_assignment"] = {
            "shard_id": selected_shard,
            "shard_index": selected_index,
            "physical_device": physical_device,
            "observed_cuda_visible_devices": physical_device,
            "logical_device": "cuda:0",
        }
        return selected_identity

    def fake_runtime_validate(value: Mapping[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assignment = value["device_assignment"]
        selected_shard = assignment["shard_id"]
        assert assignment["physical_device"] == device_plan[selected_shard]
        assert assignment["observed_cuda_visible_devices"] == device_plan[selected_shard]
        return {"identity": dict(value)}

    monkeypatch.setattr(sealer, "runtime_identity_binding", fake_runtime_binding)
    monkeypatch.setattr(sealer, "validate_runtime_identity", fake_runtime_validate)
    return plan_path, receipt_path, manifest, {
        "census": census,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "cohort_manifest": cohort_manifest,
        "h0_root": h0_root,
        "h0_dir": h0_dir,
        "base_model_dir": base_model,
        "output": execution_root / "shard-000",
        "execution_root": execution_root,
        "final_root": final_root,
        "plan": plan,
        "receipt": receipt,
        "full_events": full_events,
    }


def _natural_result(mask_hash: str, mask_shape: list[int], *, parity: bool = False) -> dict[str, Any]:
    scalar = {
        "use_cache": False,
        "input_ids_sha256": "b" * 64,
        "mrope_hash": "mrope",
        "attention_actuation_receipt": {
            "mask_sha256": mask_hash,
            "mask_shape": mask_shape,
            "layer_consumption_attestation": {
                "passed": True,
                "layer_count": 28,
                "layer_indices": list(range(28)),
            },
        },
    }
    result: dict[str, Any] = {
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "runtime_scalar_forward_count": 1,
        "runtime_scalar_receipts": [scalar],
    }
    if parity:
        result["full_logit_parity"] = {
            "passed": True,
            "per_forward_max_abs_delta": 0.0,
            "reference_step_count": 1,
            "candidate_step_count": 1,
        }
    return result


def _event_runner(_event: object, **_kwargs: object) -> dict[str, Any]:
    k01 = attention.build_k01(sequence_length=8)
    k10 = attention.build_k10(
        sequence_length=8,
        image_key_positions=(1, 2, 3),
        b_exclusive_positions=(2,),
        query_position=7,
    )
    h20 = attention.build_h20(
        sequence_length=8,
        query_positions=(7,),
        latest_row_key_positions=(5, 6),
    )
    c11 = attention.compose_k10_h20(k10, h20)
    return {
        "technical_control": {
            "K00": {
                "transport_arm_id": "K00",
                "result": {"admission_mode": "pre_opener_natural", "opener_injected": False},
            }
        },
        "cells": {
            "C00": {"result": _natural_result(attention.sha256_tensor(k01.mask), list(k01.mask.shape), parity=True)},
            "C10": {"result": _natural_result(attention.sha256_tensor(k10.mask), list(k10.mask.shape))},
            "C01": {"result": _natural_result(attention.sha256_tensor(h20.mask), list(h20.mask.shape))},
            "C11": {"result": _natural_result(attention.sha256_tensor(c11.mask), list(c11.mask.shape))},
        },
    }


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, output_name: str = "out", **overrides: object) -> dict[str, Any]:
    plan_path, receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    kwargs: dict[str, object] = {
        "shard_id": "shard-000",
        "manifest": manifest,
        "census": sources["census"],
        "config": sources["config"],
        "panel": sources["panel"],
        "cohort": sources["cohort"],
        "cohort_manifest": sources["cohort_manifest"],
        "h0_root": sources["h0_root"],
        "h0_dir": sources["h0_dir"],
        "base_model_dir": sources["base_model_dir"],
        "event_runner": _event_runner,
    }
    kwargs.update(overrides)
    output = sources["output"] if output_name == "out" else tmp_path / output_name
    return runner.run_crossover_shard(plan_path, receipt_path, output, **kwargs)  # type: ignore[arg-type]


def test_event_from_manifest_resolves_planner_projection_and_retains_full_event() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)

    assert set(planned_event) == {
        "event_index",
        "event_id",
        "image_id",
        "event_sha256",
        "prefix_sha256",
        "geometry_sha256",
        "target_owner_id",
        "covered_owner_ids",
        "source_qualification",
    }
    assert "prefix_sha256" not in full_event

    resolved = runner._event_from_manifest({"events": [full_event]}, planned_event)

    assert resolved == full_event
    assert resolved["natural_boundary"]["history_token_ids"] == [151646, 1000, 151649]
    assert resolved["geometry"]["target_owner_id"] == planner.EVENT_IDS[0]


@pytest.mark.parametrize("shard_index", range(3))
def test_validate_preflight_accepts_production_shaped_event_for_every_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_index: int,
) -> None:
    plan_path, receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    shard_id = f"shard-{shard_index:03d}"
    output_root = sources["execution_root"] / shard_id

    preflight = runner.validate_preflight(
        plan_path,
        receipt_path,
        output_root,
        shard_id=shard_id,
        manifest=manifest,
        census=sources["census"],
        config=sources["config"],
        panel=sources["panel"],
        cohort=sources["cohort"],
        cohort_manifest=sources["cohort_manifest"],
        h0_root=sources["h0_root"],
        h0_dir=sources["h0_dir"],
        base_model_dir=sources["base_model_dir"],
    )

    assert preflight.event == sources["full_events"][shard_index]
    assert not output_root.exists()


def test_manifest_projection_rejects_prefix_history_mismatch() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)
    full_event["natural_boundary"]["history_sha256"] = "f" * 64
    _rehash_event(full_event)

    with pytest.raises(runner.CrossoverRunnerError, match="natural_boundary.history_sha256"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_manifest_projection_rejects_nested_geometry_self_hash_mismatch() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)
    full_event["geometry"]["regions"]["a_exclusive"] = [999]
    _rehash_event(full_event)

    with pytest.raises(runner.CrossoverRunnerError, match="geometry.geometry_sha256"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_manifest_projection_rejects_nested_top_level_geometry_mismatch() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)
    full_event["geometry_sha256"] = "f" * 64
    _rehash_event(full_event)

    with pytest.raises(runner.CrossoverRunnerError, match="geometry_sha256"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_manifest_projection_rejects_target_owner_mismatch() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)
    full_event["owner_refs"]["gt_owner_id"] = "gt:wrong:owner"
    _rehash_event(full_event)

    with pytest.raises(runner.CrossoverRunnerError, match="owner_refs.gt_owner_id"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_manifest_projection_rejects_covered_owner_order_mismatch() -> None:
    full_event = _full_event(0)
    full_event["owner_refs"]["covered_owner_ids"].append("covered:second")
    _rehash_event(full_event)
    planned_event = _plan_event(full_event)
    full_event["owner_refs"]["covered_owner_ids"].reverse()
    _rehash_event(full_event)
    planned_event["event_sha256"] = full_event["event_sha256"]

    with pytest.raises(runner.CrossoverRunnerError, match="covered_owner_ids"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_manifest_projection_rejects_full_event_self_hash_drift() -> None:
    full_event = _full_event(0)
    planned_event = _plan_event(full_event)
    full_event["eligibility"]["launch_eligible"] = False

    with pytest.raises(runner.CrossoverRunnerError, match="event_sha256"):
        runner._event_from_manifest({"events": [full_event]}, planned_event)


def test_plan_projection_requires_exact_source_qualification() -> None:
    planned_event = _plan_event(_full_event(0))
    planned_event["source_qualification"]["h20_nondegenerate"] = False

    with pytest.raises(runner.CrossoverRunnerError, match="source_qualification"):
        runner._require_event_ref(planned_event, label="plan event")


def test_admitted_history_is_required_and_never_self_defaults() -> None:
    full_event = _full_event(0)
    expected = full_event["natural_boundary"]["history_token_ids"]
    runner._require_admitted_history(expected, full_event)

    del full_event["natural_boundary"]["history_token_ids"]
    with pytest.raises(runner.CrossoverTechnicalInvalid, match="required.*history_token_ids"):
        runner._require_admitted_history(expected, full_event)


def test_source_preflight_calls_validate_inputs_never_loads_or_creates_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    calls = {"configure": 0, "validate": 0, "production": 0, "load": 0}

    class SentinelExecutor:
        _adapter = None
        _orchestrator = None

        def configure_pre_gpu_identity(
            self,
            identity: Mapping[str, Any],
            *,
            runtime_versions: Mapping[str, str],
        ) -> None:
            calls["configure"] += 1
            assert dict(runtime_versions) == identity["runtime"]

        def _validate_inputs(
            self,
            event: Mapping[str, Any],
            arm_order: tuple[str, ...],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
            from scripts.research import run_s_natural_boundary_k_n_h_cohort as cohort

            calls["validate"] += 1
            assert arm_order == cohort.ARM_ORDER
            owner_refs = event["owner_refs"]
            row = {
                "checkpoint": "S",
                "gt_owner_id": owner_refs["gt_owner_id"],
                "image_id": event["image_id"],
                "source_panel_object_index": owner_refs["source_panel_object_index"],
                "derived_panel_object_index": owner_refs["derived_panel_object_index"],
            }
            environment = {
                "manifest": "S_NATURAL_BOUNDARY_MANIFEST",
                "census": "S_NATURAL_BOUNDARY_CENSUS",
                "config": "S_NATURAL_BOUNDARY_CONFIG",
                "panel": "S_NATURAL_BOUNDARY_PANEL",
                "cohort": "S_NATURAL_BOUNDARY_COHORT",
                "cohort_manifest": "S_NATURAL_BOUNDARY_COHORT_MANIFEST",
                "h0_root": "S_NATURAL_BOUNDARY_H0_ROOT",
                "h0_dir": "S_NATURAL_BOUNDARY_H0_DIR",
                "pre_gpu_receipt": "S_NATURAL_BOUNDARY_PRE_GPU_RECEIPT",
            }
            return dict(event), row, {
                label: Path(os.environ[name]) for label, name in environment.items()
            }

        def _load_once(self, *_args: Any, **_kwargs: Any) -> None:
            calls["load"] += 1
            raise AssertionError("source preflight must never load a model")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def fake_production_path(
        preflight: runner.CrossoverPreflight,
        *,
        paths: Mapping[str, Path],
    ) -> dict[str, Any]:
        calls["production"] += 1
        assert preflight.event["event_id"] == planner.EVENT_IDS[0]
        assert set(paths) == {
            "manifest",
            "census",
            "config",
            "panel",
            "cohort",
            "cohort_manifest",
            "h0_root",
            "h0_dir",
            "pre_gpu_receipt",
        }
        return {
            "status": "passed",
            "gpu_used": False,
            "model_loaded": False,
        }

    monkeypatch.setattr(
        runner, "_preflight_model_free_production_path", fake_production_path
    )
    output_root = sources["execution_root"] / "shard-000"
    result = runner.preflight_crossover(
        plan_path,
        receipt_path,
        output_root,
        shard_id="shard-000",
        manifest=manifest,
        census=sources["census"],
        config=sources["config"],
        panel=sources["panel"],
        cohort=sources["cohort"],
        cohort_manifest=sources["cohort_manifest"],
        h0_root=sources["h0_root"],
        h0_dir=sources["h0_dir"],
        base_model_dir=sources["base_model_dir"],
        executor=SentinelExecutor(),
    )

    assert calls == {"configure": 1, "validate": 1, "production": 1, "load": 0}
    assert result["status"] == "passed"
    assert result["gpu_used"] is False
    assert result["model_loaded"] is False
    assert result["resolved_identity"]["event_id"] == planner.EVENT_IDS[0]
    assert result["resolved_census_row"]["gt_owner_id"] == planner.EVENT_IDS[0]
    assert result["resolved_paths"]["manifest"]["sha256"] == runner._sha256_file(manifest)
    assert result["self_sha256"] == runner.sha256_json(
        {key: value for key, value in result.items() if key != "self_sha256"}
    )
    assert not output_root.exists()


def _preseal_source_kwargs(sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest": sources["manifest"]
        if "manifest" in sources
        else sources["input_paths"]["manifest"],
        "census": sources["census"],
        "config": sources["config"],
        "panel": sources["panel"],
        "cohort": sources["cohort"],
        "cohort_manifest": sources["cohort_manifest"],
        "h0_root": sources["h0_root"],
        "h0_dir": sources["h0_dir"],
        "base_model_dir": sources["base_model_dir"],
        "execution_root": sources["execution_root"],
        "final_root": sources["final_root"],
    }


def test_preseal_source_preflight_binds_plan_sources_code_tests_and_absent_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    sources["manifest"] = manifest
    production_calls = 0

    def fake_production_path(
        preflight: runner.CrossoverPreflight,
        *,
        paths: Mapping[str, Path],
    ) -> dict[str, Any]:
        nonlocal production_calls
        production_calls += 1
        assert preflight.pre_gpu_receipt == {}
        assert preflight.pre_gpu_receipt_path == ""
        assert preflight.pre_gpu_receipt_raw_sha256 == "0" * 64
        assert preflight.pre_gpu_receipt_self_sha256 == "0" * 64
        assert set(paths) == {
            "manifest",
            "census",
            "config",
            "panel",
            "cohort",
            "cohort_manifest",
            "h0_root",
            "h0_dir",
        }
        body = {
            "status": "passed",
            "full_runtime_cohort": {"status": "passed", "event_count": 11},
            "selected_factory_contexts": [
                {"event_id": event_id} for event_id in planner.EVENT_IDS
            ],
            "gpu_used": False,
            "model_loaded": False,
            "model_loader_called": False,
            "output_root_created": False,
        }
        return body | {"receipt_sha256": runner.sha256_json(body)}

    monkeypatch.setattr(
        runner, "_preflight_model_free_production_path", fake_production_path
    )

    result = runner.preflight_crossover_sources(
        plan_path, **_preseal_source_kwargs(sources)
    )

    assert production_calls == 1
    assert result["schema_version"] == runner.SOURCE_PREFLIGHT_SCHEMA_VERSION
    assert result["status"] == "passed"
    assert result["phase"] == "preseal_model_free_production"
    assert result["receipt_independent"] is True
    assert result["plan_binding"] == {
        "path": str(plan_path.resolve()),
        "raw_sha256": runner._sha256_file(plan_path),
        "self_sha256": sources["plan"]["self_sha256"],
    }
    assert set(result["source_bindings"]) == {
        "manifest",
        "census",
        "execution_plan",
        "config",
        "panel",
        "cohort",
        "cohort_manifest",
        "h0_root",
        "h0_dir",
        "base_model_dir",
    }
    assert result["source_bindings"]["manifest"] == _path_ref(manifest)
    assert result["reserved_roots"] == {
        "execution_root": {
            "path": str(sources["execution_root"]),
            "status": "reserved_absent_pre_gpu",
        },
        "final_root": {
            "path": str(sources["final_root"]),
            "status": "reserved_absent_pre_gpu",
        },
    }
    assert set(result["code_bindings"]) == set(
        __import__(
            "scripts.research.seal_s_k10_h20_crossover_pre_gpu_receipt",
            fromlist=["CODE_ROLE_PATHS"],
        ).CODE_ROLE_PATHS
    )
    assert set(result["test_bindings"]) == set(
        __import__(
            "scripts.research.seal_s_k10_h20_crossover_pre_gpu_receipt",
            fromlist=["TEST_PATHS"],
        ).TEST_PATHS
    )
    assert not {
        key for key in _nested_keys(result) if key.startswith("pre_gpu_receipt")
    }
    assert "0" * 64 not in json.dumps(result, sort_keys=True)
    assert result["self_sha256"] == runner.sha256_json(
        {key: value for key, value in result.items() if key != "self_sha256"}
    )
    assert not sources["execution_root"].exists()
    assert not sources["final_root"].exists()


def test_preseal_source_preflight_rejects_operator_plan_tamper_before_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    sources["manifest"] = manifest
    tampered = copy.deepcopy(sources["plan"])
    tampered["operator_contract"]["max_rows"] += 1
    tampered.pop("self_sha256")
    tampered["self_sha256"] = runner.sha256_json(tampered)
    tampered_path = _write_json(tmp_path / "tampered-plan.json", tampered)
    monkeypatch.setattr(
        planner,
        "validate_plan",
        lambda path: {
            "path": str(Path(path).resolve()),
            "plan": copy.deepcopy(tampered),
            "plan_sha256": tampered["self_sha256"],
            "self_sha256": tampered["self_sha256"],
        },
    )
    monkeypatch.setattr(
        runner,
        "_preflight_model_free_production_path",
        lambda *_args, **_kwargs: pytest.fail("production path must not run"),
    )

    with pytest.raises(runner.CrossoverRunnerError, match="max_rows"):
        runner.preflight_crossover_sources(
            tampered_path, **_preseal_source_kwargs(sources)
        )


def test_preseal_source_preflight_rejects_manifest_source_tamper_before_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    sources["manifest"] = manifest
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_document["tampered"] = True
    _write_json(manifest, manifest_document)
    monkeypatch.setattr(
        runner,
        "_preflight_model_free_production_path",
        lambda *_args, **_kwargs: pytest.fail("production path must not run"),
    )

    with pytest.raises(runner.CrossoverRunnerError, match="manifest differs"):
        runner.preflight_crossover_sources(
            plan_path, **_preseal_source_kwargs(sources)
        )


@pytest.mark.parametrize("root_key", ("execution_root", "final_root"))
def test_preseal_source_preflight_rejects_existing_reserved_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_key: str,
) -> None:
    plan_path, _receipt_path, manifest, sources = _fixture(tmp_path, monkeypatch)
    sources["manifest"] = manifest
    sources[root_key].mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "_preflight_model_free_production_path",
        lambda *_args, **_kwargs: pytest.fail("production path must not run"),
    )

    with pytest.raises(runner.CrossoverRunnerError, match="must be absent"):
        runner.preflight_crossover_sources(
            plan_path, **_preseal_source_kwargs(sources)
        )


@real_crossover_artifacts
def test_real_artifact_preseal_source_preflight_covers_full_cohort_and_selected_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _receipt_info = runner._read_json(REAL_PRE_GPU_V3, "real pre-GPU receipt")
    plan_path = Path(receipt["input_paths"]["execution_plan"])
    manifest_path = Path(receipt["input_paths"]["manifest"])
    source_paths = dict(receipt["input_paths"])
    execution_root = tmp_path / "must-remain-absent-execution"
    final_root = tmp_path / "must-remain-absent-final"
    dispatched: list[dict[str, Any]] = []
    original_arm = gate._arm_attention_callback

    def spy_arm(callback: Any, arm_id: str) -> Any:
        wrapped = original_arm(callback, arm_id)

        def observe(*args: Any, **kwargs: Any) -> Any:
            result = wrapped(*args, **kwargs)
            dispatched.append(
                {"callback": callback, "arm_id": arm_id, "result": result}
            )
            return result

        return observe

    monkeypatch.setattr(gate, "_arm_attention_callback", spy_arm)

    result = runner.preflight_crossover_sources(
        plan_path,
        manifest=manifest_path,
        census=source_paths["census"],
        config=source_paths["config"],
        panel=source_paths["panel"],
        cohort=source_paths["cohort"],
        cohort_manifest=source_paths["cohort_manifest"],
        h0_root=source_paths["h0_root"],
        h0_dir=source_paths["h0_dir"],
        base_model_dir=source_paths["base_model_dir"],
        execution_root=execution_root,
        final_root=final_root,
    )
    production = result["model_free_production_path"]

    assert result["status"] == "passed"
    assert result["phase"] == "preseal_model_free_production"
    assert result["receipt_independent"] is True
    assert result["source_bindings"] == {
        label: _path_ref(
            Path(source_paths[label]),
            directory=label in {"h0_root", "h0_dir", "base_model_dir"},
        )
        for label in result["source_bindings"]
    }
    assert production["full_runtime_cohort"]["event_count"] == 11
    assert len(production["full_runtime_cohort"]["processor_context_bindings"]) == 11
    assert [item["event_id"] for item in production["selected_factory_contexts"]] == list(
        planner.EVENT_IDS
    )
    selected_contexts = production["selected_factory_contexts"]
    assert len(dispatched) == len(selected_contexts) == runner.SHARD_COUNT
    assert [item["arm_id"] for item in dispatched] == ["K10"] * runner.SHARD_COUNT
    assert [item["result"]["receipt"] for item in dispatched] == [
        item["c11_receipt"] for item in selected_contexts
    ]
    for dispatch in dispatched:
        callback = dispatch["callback"]
        output = dispatch["result"]
        sequence_length = int(output["attention_mask"].shape[-1])
        direct_k10 = attention.build_scalar_step_factory(
            "K10", **callback.k10_kwargs
        ).build(sequence_length, query_position=sequence_length - 1, device="cpu")
        direct_h20 = attention.build_scalar_step_factory(
            "H20", **callback.h20_kwargs
        ).build(sequence_length, query_position=sequence_length - 1, device="cpu")
        direct = attention.compose_k10_h20(direct_k10, direct_h20)
        expected_receipt = direct.receipt()
        expected_receipt.update(
            {
                "transport_arm_id": "K10",
                "inner_actuator_id": "C11",
                "composition_arm_id": "C11",
            }
        )
        assert torch.equal(output["attention_mask"], direct.attention_mask)
        assert output["receipt"] == expected_receipt
    assert all(
        set(item["attention_factory_contracts"])
        == set(item["attention_factory_arms"])
        and set(item["built_factory_receipts"]) == {"K01", "K10", "H20"}
        and all(
            receipt["layer_consumption_attestation"]["required"] is True
            and receipt["layer_consumption_attestation"]["status"] == "unattested"
            and receipt["layer_consumption_attestation"][
                "exact_same_tensor_all_layers_required"
            ]
            is True
            and receipt["all_layer_consumption_attestation"]["required"] is True
            and receipt["all_layer_consumption_attestation"]["status"]
            == "unattested"
            and receipt["all_layer_consumption_attestation"][
                "exact_same_tensor_all_layers_required"
            ]
            is True
            for receipt in item["built_factory_receipts"].values()
        )
        and item["built_factory_receipts"]["K01"]["no_op_parity"]["status"]
        == "unassessed"
        and item["k14_reference_positions"]["K14T"]
        and item["k14_reference_positions"]["K14B"]
        and item["c11_receipt"]["cell_id"] == "C11"
        and item["c11_receipt"]["layer_consumption_attestation"]["required"]
        is True
        and item["c11_receipt"]["layer_consumption_attestation"]["status"]
        == "unattested"
        and item["c11_receipt"]["all_layer_consumption_attestation"][
            "exact_same_tensor_all_layers_required"
        ]
        is True
        for item in production["selected_factory_contexts"]
    )
    assert production["processor_only"]["model_present"] is False
    assert production["processor_only"]["backend_session_opened"] is False
    assert result["gpu_used"] is False
    assert result["model_loaded"] is False
    assert result["backend_session_opened"] is False
    assert result["output_root_created"] is False
    assert not {
        key for key in _nested_keys(result) if key.startswith("pre_gpu_receipt")
    }
    assert "0" * 64 not in json.dumps(result, sort_keys=True)
    assert result["self_sha256"] == runner.sha256_json(
        {key: value for key, value in result.items() if key != "self_sha256"}
    )
    assert not execution_root.exists()
    assert not final_root.exists()


def test_preflight_rejects_existing_output_before_event_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def fail_runner(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("event runner must not be called")

    with pytest.raises(runner.CrossoverRunnerError, match="fresh output root"):
        _run(tmp_path, monkeypatch, output_name="existing", event_runner=fail_runner)
    assert not called


def test_runner_writes_exact_result_siblings_and_identity_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(tmp_path, monkeypatch)
    document = result["result"]
    assert document["schema_version"] == runner.SCHEMA_VERSION
    assert document["unit_id"] == runner.UNIT_ID
    assert list(document["cells"]) == list(runner.CELLS)
    assert document["technical_control"]["K00"]["transport_arm_id"] == "K00"
    assert document["cells"]["C11"]["transport_arm_id"] == "K10"
    assert document["cells"]["C11"]["composition_arm_id"] == "C11"
    root = Path(result["output_root"])
    assert {item.name for item in root.iterdir()} == {
        "result.json",
        "runtime_identity.json",
        "terminal_summary.json",
        "aggregate.receipt.json",
    }
    runtime = runner._read_json(root / "runtime_identity.json", "runtime")[0]
    terminal = runner._read_json(root / "terminal_summary.json", "terminal")[0]
    aggregate = runner._read_json(root / "aggregate.receipt.json", "aggregate")[0]
    assert runtime["schema_version"] == runner.RUNTIME_IDENTITY_SCHEMA_VERSION
    assert terminal["schema_version"] == runner.TERMINAL_SCHEMA_VERSION
    assert aggregate["schema_version"] == runner.AGGREGATE_RECEIPT_SCHEMA_VERSION
    assert aggregate["result_sha256"] == document["result_sha256"]


def test_runner_strips_nonpersistent_callback_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDoseScoreBias:
        pass

    def callback_runner(*_args: object, **_kwargs: object) -> dict[str, Any]:
        outputs = _event_runner(*_args, **_kwargs)
        outputs["cells"]["C10"]["result"]["score_bias"] = FixedDoseScoreBias()
        outputs["cells"]["C10"]["result"]["score_bias_callback"] = lambda *_a: None
        outputs["cells"]["C10"]["result"]["attention_mask"] = attention.build_k10(
            sequence_length=8,
            image_key_positions=(1, 2),
            b_exclusive_positions=(2,),
            query_position=7,
        ).mask
        return outputs

    result = _run(tmp_path, monkeypatch, event_runner=callback_runner)
    serialized = json.dumps(result["result"])
    assert "FixedDoseScoreBias" not in serialized
    assert "score_bias_callback" not in serialized
    assert '"attention_mask"' not in serialized


def test_runner_failure_is_immutable_and_preserves_scientific_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def malformed(*_args: object, **_kwargs: object) -> object:
        return {"technical_control": {"K00": {"result": {}}}, "cells": {}}

    with pytest.raises(runner.CrossoverTechnicalInvalid):
        _run(tmp_path, monkeypatch, event_runner=malformed)
    failure_root = tmp_path / "execution" / "shard-000"
    assert (failure_root / "failure.json").is_file()
    assert (failure_root / "failure.stderr").is_file()
    before = (failure_root / "failure.stderr").read_bytes()
    with pytest.raises(runner.CrossoverRunnerError):
        _run(tmp_path, monkeypatch, event_runner=malformed)
    assert (failure_root / "failure.stderr").read_bytes() == before


def _cpu_c11_callback() -> tuple[runner._C11CompositionCallback, Any, Any]:
    common = {
        "image_key_positions": (1, 2, 3),
        "b_exclusive_positions": (2,),
        "latest_row_key_positions": (5, 6),
        "layer_count": 28,
        "head_count": 16,
        "device": "cpu",
        "dtype": torch.bool,
    }
    k10_factory = attention.build_scalar_step_factory("K10", **common)
    h20_factory = attention.build_scalar_step_factory("H20", **common)
    return runner._C11CompositionCallback(k10_factory, h20_factory), k10_factory, h20_factory


def test_c11_callback_uses_gate_keyword_dispatch_and_matches_direct_composition() -> None:
    callback, k10_factory, h20_factory = _cpu_c11_callback()
    positional = [
        parameter
        for parameter in inspect.signature(callback).parameters.values()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    assert positional == []

    wrapped = gate._arm_attention_callback(callback, "K10")
    input_ids = torch.zeros((1, 8), dtype=torch.long)
    result = wrapped(
        {"context": "ignored"},
        input_ids=input_ids,
        step=0,
        row_index=0,
    )

    direct_k10_factory = attention.build_scalar_step_factory(
        "K10", **runner._factory_kwargs(k10_factory)
    )
    direct_h20_factory = attention.build_scalar_step_factory(
        "H20", **runner._factory_kwargs(h20_factory)
    )
    direct = attention.compose_k10_h20(
        direct_k10_factory.build(8, query_position=7, device="cpu"),
        direct_h20_factory.build(8, query_position=7, device="cpu"),
    )
    expected_receipt = direct.receipt()
    expected_receipt.update(
        {
            "transport_arm_id": "K10",
            "inner_actuator_id": "C11",
            "composition_arm_id": "C11",
        }
    )

    assert torch.equal(result["attention_mask"], direct.attention_mask)
    assert result["receipt"] == expected_receipt
    assert result["actuator_id"] == "C11"
    assert result["protocol"] == runner._C11CompositionCallback.protocol
    assert result["score_bias"] is None
    assert result["score_bias_callback"] is None
    assert callback.last_receipt == expected_receipt
    assert callback.receipts == [expected_receipt]
    assert not any(
        name in callback.__dict__ for name in ("k_factory", "h_factory", "composed", "actuator")
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "integer sequence_length"),
        ({"sequence_length": 8}, "integer query_position"),
        ({"sequence_length": 8, "query_position": 7}, "requires device"),
        (
            {"sequence_length": 0, "query_position": 0, "device": "cpu"},
            "positive sequence_length",
        ),
        (
            {"sequence_length": 8.0, "query_position": 7, "device": "cpu"},
            "integer sequence_length",
        ),
        (
            {"sequence_length": 8, "query_position": 8, "device": "cpu"},
            "within sequence_length",
        ),
        (
            {"sequence_length": 8, "query_position": 7, "device": "not-a-device"},
            "device is invalid",
        ),
        (
            {"sequence_length": 8, "query_position": 7, "device": 1},
            "device must be a torch.device or string",
        ),
    ],
)
def test_c11_callback_rejects_missing_or_invalid_scalar_protocol(
    kwargs: dict[str, Any], message: str
) -> None:
    callback, _k10_factory, _h20_factory = _cpu_c11_callback()
    with pytest.raises(runner.CrossoverTechnicalInvalid, match=message):
        callback(**kwargs)


class _FakeDecoderLayer(torch.nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        return hidden_states


class _FakeDecoderModel(torch.nn.Module):
    """A 28-layer CPU stand-in that consumes one exact mask per layer."""

    def __init__(self, layer_count: int = 28) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _FakeDecoderLayer() for _ in range(layer_count)
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def _real_c11_live_receipt() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Build the real C11 receipt and bind a real all-28 attestor forward.

    This mirrors ``run_s_primary_natural_boundary_gate``: the observed
    attestor receipt overwrites only the top-level attestation of the actuator
    receipt and of the scalar call receipt, so the composed receipt's K10/H20
    ``children`` keep their construction placeholders.
    """

    callback, _k10_factory, _h20_factory = _cpu_c11_callback()
    output = callback(sequence_length=8, query_position=7, device="cpu")
    mask = output["attention_mask"]
    model = _FakeDecoderModel()
    hidden = torch.zeros((1, mask.shape[-1], 4))
    attestor = attention.LayerMaskConsumptionAttestor(model, mask)
    with attestor:
        model(hidden, attention_mask=mask)
    observed = attestor.receipt()
    assert observed["passed"] is True
    assert observed["layer_count"] == 28
    assert observed["layer_indices"] == list(range(28))

    actuation = {
        **output["receipt"],
        "layer_consumption_attestation": dict(observed),
        "all_layer_consumption_attestation": dict(observed),
    }
    scalar = {
        "use_cache": False,
        "input_ids_sha256": "a" * 64,
        "mrope_hash": "b" * 64,
        "attention_actuation_receipt": actuation,
        "layer_consumption_attestation": dict(observed),
    }
    result = {
        "admission_mode": "pre_opener_natural",
        "opener_injected": False,
        "synthetic_opener_injections": 0,
        "runtime_scalar_forward_count": 1,
        "runtime_scalar_receipts": [scalar],
    }
    return result, dict(output["receipt"]), str(output["receipt"]["mask_sha256"])


def test_live_validation_accepts_real_c11_children_with_top_level_runtime_pass() -> None:
    result, composed_receipt, mask_sha256 = _real_c11_live_receipt()

    children = composed_receipt["children"]
    assert [child["arm_id"] for child in children] == ["K10", "H20"]
    for child in children:
        for key in runner.ATTESTATION_KEYS:
            assert (
                attention.construction_consumption_placeholder_kind(
                    child["receipt"][key]
                )
                == "base"
            )
    for key in runner.ATTESTATION_KEYS:
        assert (
            attention.construction_consumption_placeholder_kind(composed_receipt[key])
            == "composed"
        )

    validated = runner._validate_natural_result(
        result,
        label="C11",
        expected_mask_sha256=mask_sha256,
        require_mask=True,
    )
    assert validated["runtime_scalar_forward_count"] == 1


@pytest.mark.parametrize("mutation", ["missing", "failed", "construction_only"])
def test_live_validation_requires_a_passed_top_level_runtime_attestation(
    mutation: str,
) -> None:
    result, composed_receipt, mask_sha256 = _real_c11_live_receipt()
    scalar = result["runtime_scalar_receipts"][0]
    if mutation == "missing":
        scalar.pop("layer_consumption_attestation")
        for key in runner.ATTESTATION_KEYS:
            scalar["attention_actuation_receipt"].pop(key)
        message = r"lacks a runtime all-28-layer consumption attestation"
    elif mutation == "construction_only":
        # The pre-forward receipt the callback returns, never overwritten.
        scalar["attention_actuation_receipt"] = dict(composed_receipt)
        scalar.pop("layer_consumption_attestation")
        message = r"lacks a runtime all-28-layer consumption attestation"
    else:
        scalar["layer_consumption_attestation"]["passed"] = False
        message = r"lacks passed all-layer consumption attestation"

    with pytest.raises(runner.CrossoverTechnicalInvalid, match=message):
        runner._validate_natural_result(
            result,
            label="C11",
            expected_mask_sha256=mask_sha256,
            require_mask=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ["fake_pass", "attested_status", "extra_key", "dropped_key", "emptied"],
)
def test_live_validation_rejects_a_malformed_c11_child_attestation(
    mutation: str,
) -> None:
    result, _composed_receipt, mask_sha256 = _real_c11_live_receipt()
    child = result["runtime_scalar_receipts"][0]["attention_actuation_receipt"][
        "children"
    ][0]["receipt"]
    placeholder = dict(child["layer_consumption_attestation"])
    if mutation == "fake_pass":
        placeholder["passed"] = True
    elif mutation == "attested_status":
        placeholder["status"] = "attested"
    elif mutation == "extra_key":
        placeholder["observed_layers"] = list(range(28))
    elif mutation == "dropped_key":
        placeholder.pop("exact_same_tensor_all_layers_required")
    else:
        placeholder = {}
    child["layer_consumption_attestation"] = placeholder

    assert attention.construction_consumption_placeholder_kind(placeholder) is None
    with pytest.raises(runner.CrossoverTechnicalInvalid):
        runner._validate_natural_result(
            result,
            label="C11",
            expected_mask_sha256=mask_sha256,
            require_mask=True,
        )


def test_preseal_accepts_real_construction_placeholders() -> None:
    callback, k10_factory, h20_factory = _cpu_c11_callback()
    k01_factory = attention.build_scalar_step_factory(
        "K01", **runner._factory_kwargs(k10_factory)
    )
    for arm, factory in (("K01", k01_factory), ("K10", k10_factory), ("H20", h20_factory)):
        receipt = factory.build(8, query_position=7, device="cpu").receipt()
        consumption, all_layer = runner.require_unattested_consumption(
            receipt, label=f"factory {arm}"
        )
        assert consumption == receipt["layer_consumption_attestation"]
        assert all_layer == receipt["all_layer_consumption_attestation"]

    c11_receipt = callback(sequence_length=8, query_position=7, device="cpu")["receipt"]
    consumption, all_layer = runner.require_unattested_consumption(
        c11_receipt, label="C11 factory"
    )
    assert consumption["declared_sequence_length"] == 8
    assert consumption["declared_layer_count"] == 28
    assert consumption["passed"] is False
    assert all_layer == consumption


@pytest.mark.parametrize(
    "mutation",
    [
        "fake_pass",
        "attested_status",
        "extra_key",
        "dropped_key",
        "wrong_schema_version",
        "wrong_layer_count",
        "not_a_mapping",
    ],
)
def test_preseal_rejects_construction_placeholder_drift(mutation: str) -> None:
    callback, _k10_factory, _h20_factory = _cpu_c11_callback()
    receipt = callback(sequence_length=8, query_position=7, device="cpu")["receipt"]
    placeholder: Any = dict(receipt["layer_consumption_attestation"])
    if mutation == "fake_pass":
        placeholder["passed"] = True
    elif mutation == "attested_status":
        placeholder["status"] = "attested"
    elif mutation == "extra_key":
        placeholder["observed_layers"] = list(range(28))
    elif mutation == "dropped_key":
        placeholder.pop("declared_sequence_length")
    elif mutation == "wrong_schema_version":
        placeholder["schema_version"] = "natural_boundary_attention_actuators.v1"
    elif mutation == "wrong_layer_count":
        placeholder["declared_layer_count"] = 27
    else:
        placeholder = "unattested"
    receipt["layer_consumption_attestation"] = placeholder

    with pytest.raises(
        runner.CrossoverTechnicalInvalid,
        match=r"does not retain the pre-forward required/unattested consumption contract",
    ):
        runner.require_unattested_consumption(receipt, label="C11 factory")


def test_cpu_preflight_probe_is_canonical_and_does_not_launch_gpu() -> None:
    document = runner.run_cpu_preflight_probe()
    assert document["schema_version"] == runner.PRE_GPU_PROBE_SCHEMA_VERSION
    assert document["no_gpu_launch"] is True
    assert document["source_preflight"] == {"status": "not_exercised"}
    assert document["gpu_used"] is False
    assert document["model_loaded"] is False
    assert document["composition"]["cell_id"] == "C11"
    assert document["self_sha256"] == runner.sha256_json({key: value for key, value in document.items() if key != "self_sha256"})


def test_cpu_preflight_cli_runs_from_repo_checkout() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--cpu-preflight"],
        cwd=runner.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    assert document["schema_version"] == runner.PRE_GPU_PROBE_SCHEMA_VERSION
    assert document["status"] == "passed"
    assert document["no_gpu_launch"] is True
    assert document["source_preflight"] == {"status": "not_exercised"}


def test_cpu_preflight_cli_rejects_partial_source_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = runner.main(["--cpu-preflight", "--plan", "/tmp/partial-plan.json"])

    assert code == 2
    document = json.loads(capsys.readouterr().err)
    assert document["status"] == "blocked"
    assert document["source_preflight"]["status"] == "blocked"
    assert "partial source preflight arguments" in document["source_preflight"]["error"]
    assert document["gpu_used"] is False
    assert document["model_loaded"] is False


def test_cpu_preflight_cli_full_arguments_emit_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: dict[str, Any] = {}

    def fake_source_preflight(
        plan: Path,
        receipt: Path,
        output_root: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        requested.update(
            {
                "plan": plan,
                "receipt": receipt,
                "output_root": output_root,
                **kwargs,
            }
        )
        return {
            "schema_version": runner.SOURCE_PREFLIGHT_SCHEMA_VERSION,
            "status": "passed",
            "gpu_used": False,
            "model_loaded": False,
        }

    monkeypatch.setattr(runner, "preflight_crossover", fake_source_preflight)
    monkeypatch.setattr(
        attention,
        "run_installed_qwen_cpu_probe",
        lambda: {"status": "passed", "model_loaded": False},
    )
    arguments = {
        "--plan": tmp_path / "plan.json",
        "--pre-gpu-receipt": tmp_path / "receipt.json",
        "--manifest": tmp_path / "manifest.json",
        "--output-root": tmp_path / "execution" / "shard-001",
    }
    argv = ["--cpu-preflight", "--shard-id", "shard-001"]
    for name, value in arguments.items():
        argv.extend((name, str(value)))

    code = runner.main(argv)

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "passed"
    assert document["source_preflight"]["status"] == "passed"
    assert document["source_preflight"]["gpu_used"] is False
    assert document["source_preflight"]["model_loaded"] is False
    assert requested["plan"] == arguments["--plan"]
    assert requested["receipt"] == arguments["--pre-gpu-receipt"]
    assert requested["manifest"] == arguments["--manifest"]
    assert requested["shard_id"] == "shard-001"
    assert requested["output_root"] == arguments["--output-root"]


def test_preseal_source_preflight_cli_emits_receipt_independent_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: dict[str, Any] = {}

    def fake_preseal(plan: Path, **kwargs: Any) -> dict[str, Any]:
        requested.update({"plan": plan, **kwargs})
        body = {
            "schema_version": runner.SOURCE_PREFLIGHT_SCHEMA_VERSION,
            "status": "passed",
            "phase": "preseal_model_free_production",
            "unit_id": runner.UNIT_ID,
            "receipt_independent": True,
            "gpu_used": False,
            "model_loaded": False,
            "backend_session_opened": False,
            "output_root_created": False,
        }
        return body | {"self_sha256": runner.sha256_json(body)}

    monkeypatch.setattr(runner, "preflight_crossover_sources", fake_preseal)
    arguments = {
        "--plan": tmp_path / "plan.json",
        "--manifest": tmp_path / "manifest.json",
        "--census": tmp_path / "census.json",
        "--config": tmp_path / "config.yaml",
        "--panel": tmp_path / "panel.jsonl",
        "--cohort": tmp_path / "cohort.json",
        "--cohort-manifest": tmp_path / "cohort.manifest.json",
        "--h0-root": tmp_path / "h0-root",
        "--h0-dir": tmp_path / "h0-root" / "h0",
        "--base-model-dir": tmp_path / "base-model",
        "--execution-root": tmp_path / "execution-v3",
        "--final-root": tmp_path / "evidence-v3",
    }
    argv = ["--preseal-source-preflight"]
    for name, value in arguments.items():
        argv.extend((name, str(value)))

    code = runner.main(argv)

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "passed"
    assert document["receipt_independent"] is True
    assert not {
        key for key in _nested_keys(document) if key.startswith("pre_gpu_receipt")
    }
    assert requested["plan"] == arguments["--plan"]
    for name, value in arguments.items():
        if name == "--plan":
            continue
        assert requested[name.removeprefix("--").replace("-", "_")] == value


def test_directory_inventory_hash_matches_pre_gpu_sealer_schema(tmp_path: Path) -> None:
    from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer

    root = tmp_path / "inventory"
    (root / "prefix" / "nested").mkdir(parents=True)
    (root / "prefix.json").write_text("outer sibling\n", encoding="utf-8")
    (root / "prefix" / "nested.json").write_text(
        "nested sibling\n", encoding="utf-8"
    )
    (root / "prefix" / "nested" / "weights.bin").write_bytes(b"weights")
    path_order = [
        child.relative_to(root).as_posix()
        for child in sorted(root.rglob("*"))
        if child.is_file()
    ]
    assert path_order != sorted(path_order)
    expected, _ = sealer._directory_inventory(root, "inventory")
    assert runner._sha256_directory(root) == expected
