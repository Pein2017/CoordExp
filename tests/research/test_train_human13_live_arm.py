from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.research import train_human13_live_arm as live_train


@dataclass(frozen=True)
class _Panel:
    panel_sha256: str
    owner_count: int
    images: tuple[object, ...]


@dataclass(frozen=True)
class _Binding:
    panel: _Panel


@dataclass(frozen=True)
class _Denominators:
    panel_image_count: int
    target_image_count: int
    target_owner_count: int
    replay_image_count: int
    replay_owner_count: int
    duplicate_image_count: int
    duplicate_event_count: int


@dataclass(frozen=True)
class _Manifest:
    binding: _Binding
    images: tuple[object, ...]
    denominators: _Denominators
    full_panel: bool = True


def _image(image_id: int, *, owners: int, selected: int, replay: int, dup: int):
    return SimpleNamespace(
        image_id=image_id,
        owners=tuple(
            SimpleNamespace(owner_id=f"gt:{image_id}:{i}") for i in range(owners)
        ),
        selected_rows=tuple(
            SimpleNamespace(row_id=f"h:{image_id}:{i}") for i in range(selected)
        ),
        replay_row_ids=tuple(f"g:{image_id}:{i}" for i in range(replay)),
        duplicate_events=tuple(
            SimpleNamespace(event_id=f"d:{image_id}:{i}") for i in range(dup)
        ),
    )


def test_schedule_repeats_one_complete_panel_per_update_and_checkpoints_milestones() -> (
    None
):
    micro_steps = ("pack-0", "pack-1", "pack-2")

    schedule = live_train.build_training_schedule(pack_count=3, max_updates=16)
    stream = live_train.repeat_panel_micro_steps(micro_steps, max_updates=16)

    assert schedule.resolved_max_steps == 16
    assert schedule.runtime_batch.resolved_grad_accum_steps == 3
    assert tuple(event.planned_step_id for event in schedule.events["checkpoint"]) == (
        1,
        2,
        4,
        8,
        16,
    )
    assert schedule.events["final"][0].planned_step_id == 16
    assert len(stream) == 48
    assert stream[:6] == micro_steps + micro_steps


def test_vertical_schedule_is_exactly_one_update_and_one_checkpoint() -> None:
    schedule = live_train.build_training_schedule(pack_count=2, max_updates=1)

    assert schedule.resolved_max_steps == 1
    assert tuple(event.planned_step_id for event in schedule.events["checkpoint"]) == (
        1,
    )
    assert schedule.events["final"][0].planned_step_id == 1


def test_vertical_projection_uses_only_selected_image_and_local_denominators() -> None:
    first = _image(10, owners=4, selected=2, replay=1, dup=3)
    second = _image(20, owners=7, selected=5, replay=4, dup=0)
    manifest = _Manifest(
        binding=_Binding(_Panel("a" * 64, 11, ("image-10", "image-20"))),
        images=(first, second),
        denominators=_Denominators(2, 2, 7, 2, 5, 1, 3),
    )

    projected = live_train.project_vertical_manifest(manifest, image_id=10)

    assert tuple(image.image_id for image in projected.images) == (10,)
    assert projected.binding.panel.owner_count == 4
    assert projected.denominators == _Denominators(1, 1, 2, 1, 1, 1, 3)
    assert manifest.denominators == _Denominators(2, 2, 7, 2, 5, 1, 3)


def test_repeat_panel_micro_steps_rejects_empty_or_unsupported_update_count() -> None:
    with pytest.raises(live_train.LiveTrainingError, match="micro-step"):
        live_train.repeat_panel_micro_steps((), max_updates=16)
    with pytest.raises(live_train.LiveTrainingError, match="1 or 16"):
        live_train.repeat_panel_micro_steps(("pack",), max_updates=2)


def test_runtime_receipt_is_canonical_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "runtime-receipt.json"
    payload = {
        "schema_version": "human13_live_training_receipt.v1",
        "arm_id": "A1",
        "execution_ready": True,
    }

    live_train.write_immutable_receipt(path, payload)

    encoded = path.read_bytes()
    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == payload
    with pytest.raises(live_train.LiveTrainingError, match="already exists"):
        live_train.write_immutable_receipt(path, payload)


class _Writer:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[dict[str, Any]] = []
        self.finalizations: list[dict[str, Any]] = []

    def append_logging_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def finalize(self, **kwargs: Any) -> None:
        self.finalizations.append(kwargs)


class _CheckpointWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls: list[dict[str, Any]] = []

    def write_checkpoint(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            checkpoint_dir=self.run_dir / "checkpoints" / f"step-{kwargs['step']}"
        )


class _Trainer:
    last: "_Trainer | None" = None

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.pack_stream = tuple(kwargs["pack_stream"])
        type(self).last = self

    def run(self) -> Any:
        pack_count = self.schedule.runtime_batch.resolved_grad_accum_steps
        for step in range(1, self.schedule.resolved_max_steps + 1):
            self.runtime.optimizer_step_count += 1
            observation = SimpleNamespace(
                planned_step_id=step,
                micro_step_count=pack_count,
                optimizer_update_status="applied",
                finite_status="finite",
                loss_bundle_artifact={"total_loss": float(step)},
                scheduler_artifact={"step": step},
                post_backward_artifact={"status": "pass"},
            )
            self.on_completed_step(observation)
            for event in self.schedule.events["checkpoint"]:
                if event.planned_step_id == step:
                    self.on_checkpoint(event, observation)
            for event in self.schedule.events["final"]:
                if event.planned_step_id == step:
                    self.on_final(event, observation)
        return SimpleNamespace(
            completed_steps=self.schedule.resolved_max_steps,
            consumed_micro_steps=len(self.pack_stream),
            latest_observation=observation,
            to_artifact_dict=lambda: {
                "completed_steps": self.schedule.resolved_max_steps,
                "consumed_micro_steps": len(self.pack_stream),
            },
        )


def _prepared(tmp_path: Path, *, vertical_image_id: int | None = None) -> Any:
    runtime = SimpleNamespace(optimizer_step_count=0, accelerator="accelerator")
    assembly = SimpleNamespace(
        runtime=runtime,
        model="model",
        plan=SimpleNamespace(arm_id="A1", to_artifact_dict=lambda: {"arm_id": "A1"}),
        validation=SimpleNamespace(to_artifact_dict=lambda: {"source": "valid"}),
        trainable_surface_receipt={"surface": "language-dora"},
        memory_saver_receipt={"enabled": True},
    )
    payload = SimpleNamespace(
        arm_id="A1",
        micro_steps=(SimpleNamespace(pack=SimpleNamespace(length=7)),) * 2,
        execution_plan="execution-plan",
        packed_plan=SimpleNamespace(
            performance_counters=lambda **kwargs: SimpleNamespace(
                to_artifact_dict=lambda: {
                    "pack_count": 2,
                    "packed_tokens": 14,
                    "logical_tokens": 14,
                    **kwargs,
                }
            )
        ),
    )
    return live_train.PreparedHuman13LiveArm(
        arm_id="A1",
        resolved_plan=SimpleNamespace(raw={"arm_id": "A1"}),
        resolved_plan_path=tmp_path / "resolved-plan.json",
        resolved_plan_sha256="a" * 64,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="b" * 64,
        sealed_manifest=SimpleNamespace(manifest="full"),
        execution_manifest="projected" if vertical_image_id else "full",
        model_plan=assembly.plan,
        payload=payload,
        assembly=assembly,
        run_dir=tmp_path / "run",
        vertical_image_id=vertical_image_id,
    )


def test_training_session_uses_existing_trainer_and_writes_all_checkpoint_readbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    writer = _Writer(prepared.run_dir)
    checkpoint_writer = _CheckpointWriter(prepared.run_dir)
    receipts: list[dict[str, Any]] = []
    monkeypatch.setattr(live_train, "_initialize_run_writer", lambda *a, **k: writer)
    monkeypatch.setattr(
        live_train,
        "_build_checkpoint_writer",
        lambda _run_dir: checkpoint_writer,
    )
    monkeypatch.setattr(
        live_train,
        "_readback_checkpoint",
        lambda checkpoint_dir, *, expected_step, assembly: SimpleNamespace(
            to_artifact_dict=lambda: {
                "step": expected_step,
                "checkpoint_dir": str(checkpoint_dir),
            }
        ),
    )
    monkeypatch.setattr(
        live_train,
        "write_immutable_receipt",
        lambda _path, payload: receipts.append(dict(payload)) or _path,
    )
    monkeypatch.setattr(live_train, "_validate_prepared_payload", lambda prepared: None)
    monkeypatch.setattr(live_train, "_loss_runner", lambda plan: "loss-runner")
    monkeypatch.setattr(live_train, "_loss_context_factory", lambda: "context-factory")
    monkeypatch.setattr(
        live_train,
        "_checkpoint_kwargs",
        lambda assembly: {"accelerator": "accelerator", "adapter_name": "default"},
    )

    result = live_train.train_prepared_arm(
        prepared,
        max_updates=16,
        trainer_type=_Trainer,
        updated_at="2026-08-12T00:00:00Z",
    )

    assert result.completed_steps == 16
    assert len(_Trainer.last.pack_stream) == 32
    assert [call["step"] for call in checkpoint_writer.calls] == [1, 2, 4, 8, 16]
    assert [call["is_final"] for call in checkpoint_writer.calls] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [row["step"] for row in writer.rows] == list(range(1, 17))
    assert writer.finalizations[-1]["status"] == "completed"
    assert receipts[0]["runtime_entry_contract"] == "human13_runtime.v1"
    assert receipts[0]["applied_update_count"] == 16
    assert [item["step"] for item in receipts[0]["checkpoint_readbacks"]] == [
        1,
        2,
        4,
        8,
        16,
    ]


def test_vertical_training_session_writes_only_step_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, vertical_image_id=14038)
    writer = _Writer(prepared.run_dir)
    checkpoint_writer = _CheckpointWriter(prepared.run_dir)
    monkeypatch.setattr(live_train, "_initialize_run_writer", lambda *a, **k: writer)
    monkeypatch.setattr(
        live_train, "_build_checkpoint_writer", lambda _: checkpoint_writer
    )
    monkeypatch.setattr(
        live_train,
        "_readback_checkpoint",
        lambda checkpoint_dir, *, expected_step, assembly: SimpleNamespace(
            to_artifact_dict=lambda: {"step": expected_step}
        ),
    )
    monkeypatch.setattr(
        live_train, "write_immutable_receipt", lambda path, payload: path
    )
    monkeypatch.setattr(live_train, "_validate_prepared_payload", lambda prepared: None)
    monkeypatch.setattr(live_train, "_loss_runner", lambda plan: "loss-runner")
    monkeypatch.setattr(live_train, "_loss_context_factory", lambda: "context-factory")
    monkeypatch.setattr(
        live_train,
        "_checkpoint_kwargs",
        lambda assembly: {"accelerator": "accelerator", "adapter_name": "default"},
    )

    result = live_train.train_prepared_arm(
        prepared,
        max_updates=1,
        trainer_type=_Trainer,
        updated_at="2026-08-12T00:00:00Z",
    )

    assert result.completed_steps == 1
    assert len(_Trainer.last.pack_stream) == 2
    assert [call["step"] for call in checkpoint_writer.calls] == [1]


def test_prepare_vertical_a1_composes_parent_manifest_projection_payload_and_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "resolved-plan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    calls: list[object] = []
    resolved = SimpleNamespace(
        arm_id="A1",
        updates=True,
        manifest_sha256="b" * 64,
        output_root=tmp_path / "a1",
        raw={"arm_id": "A1"},
    )
    full_manifest = SimpleNamespace(images=(SimpleNamespace(image_id=14038),))
    sealed = SimpleNamespace(manifest=full_manifest, manifest_sha256="b" * 64)
    projected = SimpleNamespace(
        images=(SimpleNamespace(image_id=14038),), full_panel=False
    )
    model_plan = SimpleNamespace(arm_id="A1")
    processor_components = SimpleNamespace(name="processor")
    skeletons = {14038: "skeleton"}
    materialized = SimpleNamespace(name="segments")
    payload = SimpleNamespace(
        arm_id="A1",
        micro_steps=("pack-0", "pack-1"),
        packed_plan=SimpleNamespace(packs=("pack-0", "pack-1")),
    )
    assembly = SimpleNamespace(components=SimpleNamespace(processor="live-processor"))

    monkeypatch.setattr(
        live_train,
        "_validate_resolved_plan",
        lambda path, manifest: calls.append("validate-plan") or resolved,
    )
    monkeypatch.setattr(
        live_train,
        "_resolve_arm_config",
        lambda plan, arm_config, repo_root: calls.append("resolve-config")
        or tmp_path / "a1.yaml",
    )
    monkeypatch.setattr(
        live_train,
        "_build_model_plan",
        lambda path: calls.append("model-plan") or model_plan,
    )
    monkeypatch.setattr(
        live_train,
        "_validate_model_plan",
        lambda plan: calls.append("validate-model-plan") or "validated",
    )
    monkeypatch.setattr(
        live_train,
        "_load_sealed_manifest",
        lambda path: calls.append("sealed-parent") or sealed,
    )
    monkeypatch.setattr(
        live_train,
        "_load_processor_components",
        lambda plan: calls.append("processor-components") or processor_components,
    )
    monkeypatch.setattr(
        live_train,
        "_build_skeletons",
        lambda manifest, components, repo_root: calls.append(
            ("skeletons", manifest, components)
        )
        or skeletons,
    )
    monkeypatch.setattr(
        live_train,
        "project_vertical_manifest",
        lambda manifest, image_id: calls.append(("projection", manifest, image_id))
        or projected,
    )
    monkeypatch.setattr(
        live_train,
        "_materialize_segments",
        lambda manifest, selected_skeletons: calls.append(
            ("segments", manifest, selected_skeletons)
        )
        or materialized,
    )
    monkeypatch.setattr(
        live_train,
        "_build_vocab_groups",
        lambda components: calls.append(("vocab", components)) or "vocab",
    )
    monkeypatch.setattr(
        live_train,
        "_build_live_payload",
        lambda **kwargs: calls.append(("payload", kwargs)) or payload,
    )
    monkeypatch.setattr(
        live_train,
        "_assemble_live_model",
        lambda plan, pack_count, repo_root: calls.append(("assembly", plan, pack_count))
        or assembly,
    )
    monkeypatch.setattr(
        live_train,
        "_attach_live_image_processor",
        lambda value, components: calls.append(("attach", value, components)) or value,
    )

    prepared = live_train.prepare_live_arm(
        resolved_plan_path=tmp_path / "resolved-plan.json",
        manifest_path=tmp_path / "manifest.json",
        repo_root=tmp_path,
        vertical_image_id=14038,
    )

    assert prepared.arm_id == "A1"
    assert prepared.sealed_manifest is sealed
    assert prepared.execution_manifest is projected
    assert prepared.vertical_image_id == 14038
    assert prepared.run_dir == resolved.output_root / "vertical-slice-image-14038"
    assert ("assembly", model_plan, 2) in calls
    segment_call = next(
        item for item in calls if isinstance(item, tuple) and item[0] == "segments"
    )
    assert segment_call[1:] == (projected, {14038: "skeleton"})


def test_execute_requires_explicit_authority_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        live_train,
        "prepare_live_arm",
        lambda **kwargs: pytest.fail("authority failure must precede live preparation"),
    )

    with pytest.raises(live_train.LiveTrainingError, match="authority"):
        live_train.execute_cli(
            resolved_plan_path=tmp_path / "plan.json",
            manifest_path=tmp_path / "manifest.json",
            repo_root=tmp_path,
            execute=True,
            authority=False,
            vertical_slice=True,
            image_id=14038,
            max_updates=1,
        )


def test_materialize_resolved_plan_from_receipt_strips_helper_path_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a1" / "resolved_plan.json"
    plan = {
        "schema_version": "human13_resolved_arm_plan.v1",
        "arm_id": "A1",
        "resolved_plan_path": str(destination),
        "output_root": str(tmp_path / "a1"),
    }
    receipt = {
        "schema_version": "human13_materialized_plans.v1",
        "plans": [plan],
    }

    path = live_train.materialize_resolved_plan(receipt, arm_id="A1")

    assert path == destination
    assert json.loads(path.read_text(encoding="utf-8")) == {
        key: value for key, value in plan.items() if key != "resolved_plan_path"
    }
    with pytest.raises(live_train.LiveTrainingError, match="already exists"):
        live_train.materialize_resolved_plan(receipt, arm_id="A1")


def test_materialize_resolved_plan_rejects_stale_noncanonical_destination(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema_version": "human13_materialized_plans.v1",
        "plans": [
            {
                "arm_id": "A1",
                "output_root": str(tmp_path / "expected"),
                "resolved_plan_path": str(tmp_path / "stale" / "resolved_plan.json"),
            }
        ],
    }

    with pytest.raises(live_train.LiveTrainingError, match="output_root"):
        live_train.materialize_resolved_plan(receipt, arm_id="A1")


def test_direct_script_entry_bootstraps_repo_imports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/research/train_human13_live_arm.py",
            "--resolved-plan",
            str(tmp_path / "missing-plan.json"),
            "--manifest",
            str(tmp_path / "missing-manifest.json"),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "No module named 'scripts.research'" not in completed.stderr
    assert "resolved plan does not exist" in completed.stderr
