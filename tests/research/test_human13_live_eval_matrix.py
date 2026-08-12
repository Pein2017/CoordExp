from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research import human13_live_eval_matrix as matrix


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        binding=SimpleNamespace(
            artifact_root="/artifacts",
            source=SimpleNamespace(checkpoint_path="/source"),
        ),
        images=tuple(SimpleNamespace(image_id=i) for i in (1, 2, 3)),
    )


def _checkpoint(root: Path, step: int) -> Path:
    path = root / "checkpoints" / f"step-{step}"
    (path / "adapter").mkdir(parents=True)
    (path / "special_token_embeddings").mkdir()
    (path / "adapter" / "payload").write_text(str(step), encoding="ascii")
    (path / "special_token_embeddings" / "payload").write_text(
        str(step), encoding="ascii"
    )
    return path


def test_dry_run_plans_exact_subset_and_performs_cpu_readback(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    for step in (1, 2, 4, 8, 16):
        _checkpoint(run_root, step)
    seen: list[tuple[int, Path]] = []
    monkeypatch.setattr(matrix, "load_manifest", lambda *_args, **_kwargs: _manifest())
    monkeypatch.setattr(matrix, "manifest_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        matrix,
        "source_outputs_from_manifest",
        lambda **kwargs: seen.append((0, Path(kwargs["checkpoint_path"]))) or ({},),
    )

    plan = matrix.build_eval_plan(
        manifest_path=tmp_path / "manifest.json",
        arm_id="A1",
        run_root=run_root,
        milestones=(1, 4, 16),
        output_dir=tmp_path / "out",
        source_discovery_records={1: {}, 2: {}, 3: {}},
    )

    assert [job.milestone for job in plan.jobs] == [1, 4, 16]
    assert [job.checkpoint_path.name for job in plan.jobs] == [
        "step-1",
        "step-4",
        "step-16",
    ]
    assert plan.execution_ready is False
    assert plan.actions == {"model_load": 0, "decode": 0, "gpu_allocation": 0}
    assert seen == [(0, Path("/source"))]


def test_execute_requires_user_model_gpu_authority(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "run", 1)
    monkeypatch.setattr(matrix, "load_manifest", lambda *_args, **_kwargs: _manifest())
    monkeypatch.setattr(matrix, "manifest_sha256", lambda _path: "a" * 64)
    plan = matrix.build_eval_plan(
        manifest_path=tmp_path / "manifest.json",
        arm_id="A1",
        explicit_checkpoints={1: checkpoint},
        milestones=(1,),
        output_dir=tmp_path / "out",
    )
    with pytest.raises(PermissionError, match="user-model-gpu-authority"):
        matrix.execute_eval_plan(plan, execute=True)


def test_execute_writes_immutable_milestone_jsonl_and_receipt(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "run", 1)
    monkeypatch.setattr(matrix, "load_manifest", lambda *_args, **_kwargs: _manifest())
    monkeypatch.setattr(matrix, "manifest_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(matrix, "source_outputs_from_manifest", lambda **_kwargs: ())
    monkeypatch.setattr(
        matrix,
        "evaluate_hf_checkpoint",
        lambda **kwargs: tuple(
            {"milestone": kwargs["milestone"], "image_id": image_id}
            for image_id in (1, 2, 3)
        ),
    )
    plan = matrix.build_eval_plan(
        manifest_path=tmp_path / "manifest.json",
        arm_id="A1",
        explicit_checkpoints={1: checkpoint},
        milestones=(1,),
        output_dir=tmp_path / "out",
    )

    receipt = matrix.execute_eval_plan(
        plan,
        execute=True,
        user_model_gpu_authority=True,
    )
    output = tmp_path / "out" / "A1.milestone-1.jsonl"
    assert output.read_text(encoding="utf-8") == (
        '{"image_id":1,"milestone":1}\n'
        '{"image_id":2,"milestone":1}\n'
        '{"image_id":3,"milestone":1}\n'
    )
    assert receipt["status"] == "completed"
    assert receipt["outputs"][0]["sha256"]
    assert (tmp_path / "out" / "A1.receipt.json").is_file()
    with pytest.raises(FileExistsError):
        matrix.execute_eval_plan(
            plan,
            execute=True,
            user_model_gpu_authority=True,
        )


def test_cli_dry_run_does_not_load_model(tmp_path, monkeypatch, capsys):
    run_root = tmp_path / "run"
    _checkpoint(run_root, 1)
    monkeypatch.setattr(matrix, "load_manifest", lambda *_args, **_kwargs: _manifest())
    monkeypatch.setattr(matrix, "manifest_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        matrix,
        "evaluate_hf_checkpoint",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    assert (
        matrix.main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--arm-id",
                "A1",
                "--run-root",
                str(run_root),
                "--milestones",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
