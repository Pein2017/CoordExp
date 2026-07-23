from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.research.materialize_constant_dose_breadth_source_b16_development_eval_configs as materializer
from scripts.research.materialize_constant_dose_breadth_source_b16_development_eval_configs import (
    ARMS,
    EXPECTED_CONFIG_COUNT,
    SEEDS,
    SOURCE_INPUT,
    STEPS,
    MaterializationError,
    materialize_configs,
)
from src.config.inference import load_infer_config


def _write_source_template(config_root: Path, fixture_root: Path) -> Path:
    base_model = fixture_root / "base-model"
    base_model.mkdir(parents=True)
    source_input = fixture_root / "source.jsonl"
    source_input.write_text("{}\n", encoding="utf-8")
    source_adapter = fixture_root / "source-adapter"
    source_adapter.mkdir()
    source_embedding = fixture_root / "source-embedding"
    source_embedding.mkdir()
    template = config_root / "source-b16-template.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "run": {
                    "name": "source-b16-template",
                    "artifact_root": str(fixture_root / "source-artifacts"),
                    "collision_policy": "fail",
                },
                "model": {
                    "base_model": str(base_model),
                    "dtype": "bf16",
                    "processor": {"do_resize": False},
                },
                "data": {"input_jsonl": str(source_input)},
                "template": {
                    "object_field_order": "desc_first",
                    "object_ordering": "geo_sorted",
                    "assistant_format": "object_box_closed",
                    "prompt": {"user": "List every object."},
                },
                "backend": {
                    "type": "vllm",
                    "vllm": {
                        "gpu_memory_utilization": 0.70,
                        "max_model_len": 4096,
                    },
                },
                "generation": {
                    "batch_size": 32,
                    "max_new_tokens": 2048,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                },
                "scoring": {"enabled": True},
                "artifacts": {
                    "write_token_trace": True,
                    "write_parse_diagnostics": True,
                },
                "adapter": {
                    "type": "dora",
                    "path": str(source_adapter),
                    "name": "default",
                },
                "embedding_delta": {"path": str(source_embedding)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return template


def _write_checkpoint_payload(checkpoint_dir: Path) -> None:
    adapter = checkpoint_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    embedding = checkpoint_dir / "special_token_embeddings"
    embedding.mkdir()
    (embedding / "special_token_embeddings.json").write_text("{}\n", encoding="utf-8")
    (embedding / "special_token_embeddings.safetensors").write_bytes(b"embedding")


def _fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    config_root = tmp_path / "configs/coordexp_swift/infer/research"
    template = _write_source_template(config_root, tmp_path / "fixture")
    source_input = tmp_path / "candidate-pool-2432.coord.jsonl"
    source_input.write_text("{}\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = checkpoint_root / arm / f"seed-{seed}" / "runs" / "single-run"
            for step in STEPS:
                _write_checkpoint_payload(run_dir / "checkpoints" / f"step-{step}")
    output_dir = config_root / "generated" / "development-source-b16-v1"
    artifact_root = tmp_path / "artifacts" / "development-source-b16-v1"
    return template, source_input, checkpoint_root, output_dir, artifact_root


def test_materializes_and_loads_the_complete_immutable_treatment_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert SOURCE_INPUT == Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-22-constant-dose-image-breadth-treatment-screen/"
        "candidate-pool-v1/candidate-pool-2432.coord.jsonl"
    )
    template, source_input, checkpoint_root, output_dir, artifact_root = _fixture_tree(
        tmp_path
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(materializer, "SOURCE_INPUT", source_input)

    outputs = materialize_configs(
        source_template=template,
        checkpoint_root=checkpoint_root,
        output_config_dir=output_dir,
        inference_artifact_root=artifact_root,
    )

    expected_names = {
        f"source-b16-development-{arm}-seed-{seed}-step-{step}.yaml"
        for arm in ARMS
        for seed in SEEDS
        for step in STEPS
    }
    assert len(outputs) == EXPECTED_CONFIG_COUNT == 16
    assert {path.name for path in outputs} == expected_names
    assert {path.name for path in output_dir.glob("*.yaml")} == expected_names

    for path in outputs:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["extends"] == "../../source-b16-template.yaml"
        assert raw["model"] == {"dtype": "bf16"}
        assert raw["backend"]["type"] == "vllm"
        assert raw["backend"]["vllm"]["gpu_memory_utilization"] == 0.70
        assert raw["backend"]["vllm"]["max_model_len"] == 4096
        assert raw["generation"] == {
            "batch_size": 32,
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        }
        resolved = load_infer_config(path).config
        assert raw["data"] == {"input_jsonl": str(source_input.resolve())}
        assert resolved.data.input_jsonl == str(source_input.resolve())
        assert resolved.run.collision_policy == "fail"
        assert resolved.generation.n == 1
        assert resolved.scoring.enabled is True
        assert resolved.adapter is not None
        assert resolved.embedding_delta is not None
        assert Path(resolved.adapter.path).parent.name.startswith("step-")
        assert Path(resolved.embedding_delta.path).parent.name.startswith("step-")

    with pytest.raises(FileExistsError, match="refusing to overwrite immutable"):
        materialize_configs(
            source_template=template,
            checkpoint_root=checkpoint_root,
            output_config_dir=output_dir,
            inference_artifact_root=artifact_root,
        )


def test_ambiguous_run_fails_without_publishing_a_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, source_input, checkpoint_root, output_dir, artifact_root = _fixture_tree(
        tmp_path
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(materializer, "SOURCE_INPUT", source_input)
    ambiguous_run = checkpoint_root / "broad/seed-19/runs/second-run"
    ambiguous_run.mkdir()

    with pytest.raises(MaterializationError, match="exactly one treatment run"):
        materialize_configs(
            source_template=template,
            checkpoint_root=checkpoint_root,
            output_config_dir=output_dir,
            inference_artifact_root=artifact_root,
        )
    assert not output_dir.exists()


def test_missing_checkpoint_payload_fails_without_publishing_a_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, source_input, checkpoint_root, output_dir, artifact_root = _fixture_tree(
        tmp_path
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(materializer, "SOURCE_INPUT", source_input)
    missing_payload = (
        checkpoint_root
        / "concentrated/seed-23/runs/single-run/checkpoints/step-31/adapter/"
        "adapter_model.safetensors"
    )
    missing_payload.unlink()

    with pytest.raises(MaterializationError, match="incomplete adapter payload"):
        materialize_configs(
            source_template=template,
            checkpoint_root=checkpoint_root,
            output_config_dir=output_dir,
            inference_artifact_root=artifact_root,
        )
    assert not output_dir.exists()
