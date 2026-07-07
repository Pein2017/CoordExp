from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.errors import ConfigContractError
from src.config.fingerprint import sha256_json
from src.config.loader import load_train_config
from src.config.models import CoordGaussianRPSLossConfig, OptimizerGroupConfig
from src.config.paths import resolve_run_directory
from src.config.resolve import (
    estimate_full_logits_bytes,
    resolve_effective_batch_runtime,
    resolve_qwen_runtime_controls,
)
from src.config.writer import write_resolved_config_artifacts
from src.training.schedule import resolve_planned_step_schedule


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_smoke_config_loads_and_writes_resolved_artifacts(tmp_path: Path) -> None:
    resolved = load_train_config(FIXTURE_CONFIG)

    fixture_root = FIXTURE_CONFIG.resolve().parent
    assert resolved.config.data.train.path == str(fixture_root / "examples.jsonl")
    assert resolved.config.data.eval is not None
    assert resolved.config.data.eval.path == str(fixture_root / "examples.jsonl")
    assert resolved.config.training.effective_batch_size == 2
    authored_model_path = (
        "/data/CoordExp/model_cache/models/Qwen/"
        "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
    )
    assert resolved.config.model.base_model == str(Path(authored_model_path).resolve())
    assert len(resolved.fingerprint) == 64
    assert resolved.path_origins["data.train.path"].declared_path == "examples.jsonl"
    assert resolved.path_origins["model.base_model"].declared_path == authored_model_path

    artifacts = write_resolved_config_artifacts(resolved, tmp_path / "run-a")
    payload = json.loads(artifacts.json_path.read_text())

    assert artifacts.yaml_path.exists()
    assert payload["resolution"]["fingerprint"] == resolved.fingerprint
    assert payload["config"]["data"]["train"]["path"] == str(fixture_root / "examples.jsonl")


@pytest.mark.parametrize("field", ("lr", "weight_decay"))
def test_optimizer_group_config_rejects_non_finite_scalars(field: str) -> None:
    payload = {"lr": 1.0e-4, "weight_decay": 0.0}
    payload[field] = float("inf")

    with pytest.raises(ValueError):
        OptimizerGroupConfig(**payload)


def test_coord_gaussian_rps_loss_config_rejects_tiny_temperature() -> None:
    with pytest.raises(ValueError):
        CoordGaussianRPSLossConfig(weight=1.0, temperature=1e-45)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        ("checkpoint.every_fraction", float("inf")),
        ("adapter.dropout", float("inf")),
        ("losses.protected.base_ce.weight", float("inf")),
        ("losses.protected.token_type_gate.weight", float("inf")),
        ("losses.protected.coord_gaussian_rps.weight", float("inf")),
        ("losses.protected.coord_gaussian_rps.gaussian_weight", float("inf")),
        ("losses.protected.coord_gaussian_rps.rps_weight", float("inf")),
        ("losses.protected.coord_gaussian_rps.temperature", float("inf")),
        (
            "losses.protected.coord_gaussian_rps.gaussian_r95_axis_fraction",
            float("inf"),
        ),
        ("optimizer.epsilon", float("inf")),
        ("optimizer.betas", [0.9, float("inf")]),
        ("optimizer.scheduler.warmup_ratio", float("inf")),
        ("training.max_grad_norm", float("inf")),
        ("training.logging.every_fraction", float("inf")),
        ("eval.forward.every_fraction", float("inf")),
    ],
)
def test_config_rejects_non_finite_float_fields(
    tmp_path: Path,
    field_path: str,
    value: Any,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    _set_nested(payload, field_path, value)
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError):
        load_train_config(config_path)


def test_fingerprint_and_resolved_writer_reject_non_finite_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigContractError) as fingerprint_exc:
        sha256_json({"bad": float("inf")})

    assert fingerprint_exc.value.code == "config.fingerprint_non_finite"

    class FakeResolvedConfig:
        fingerprint = "bad"

        def to_artifact_dict(self) -> dict[str, Any]:
            return {"config": {"bad": float("inf")}, "resolution": {}}

    with pytest.raises(ConfigContractError) as writer_exc:
        write_resolved_config_artifacts(FakeResolvedConfig(), tmp_path)

    assert writer_exc.value.code == "config.resolved_artifact_non_finite"


def test_three_level_inheritance_required_and_path_origins(tmp_path: Path) -> None:
    base = tmp_path / "configs" / "base.yaml"
    direction = tmp_path / "configs" / "directions" / "demo" / "base.yaml"
    run = tmp_path / "configs" / "directions" / "demo" / "run.yaml"
    direction.parent.mkdir(parents=True)

    _write_yaml(base, _minimal_config())
    base_payload = yaml.safe_load(base.read_text())
    base_payload["run"]["name"] = "REQUIRED"
    base_payload["model"]["logits_memory_budget_bytes"] = "REQUIRED"
    base_payload["checkpoint"]["steps"] = [99]
    _write_yaml(base, base_payload)

    _write_yaml(
        direction,
        {
            "extends": "../../base.yaml",
            "model": {"logits_memory_budget_bytes": 4_000_000_000},
            "adapter": {"path": "adapters/demo"},
            "data": {
                "train": {"path": "data/train.jsonl"},
                "eval": {"path": "data/eval.jsonl"},
            },
            "checkpoint": {"steps": [7, 8]},
        },
    )
    _write_yaml(
        run,
        {
            "schema_version": 1,
            "extends": "base.yaml",
            "run": {"name": "demo-run"},
            "checkpoint": {"steps": [2, 4]},
        },
    )

    resolved = load_train_config(run)

    assert resolved.config.run.name == "demo-run"
    assert resolved.config.checkpoint.steps == (2, 4)
    assert resolved.config.data.train.path == str((direction.parent / "data/train.jsonl").resolve())
    assert resolved.config.model.base_model == str(
        (base.parent / "model_cache/models/Qwen/example").resolve()
    )
    assert resolved.config.adapter.path == str((direction.parent / "adapters/demo").resolve())
    assert resolved.path_origins["model.base_model"].declaring_config_path == base.resolve()
    assert resolved.path_origins["data.train.path"].declaring_config_path == direction.resolve()
    assert resolved.path_origins["adapter.path"].declaring_config_path == direction.resolve()
    assert [source.path for source in resolved.sources] == [
        base.resolve(),
        direction.resolve(),
        run.resolve(),
    ]


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        ("training.grad_accum_steps", 2),
        ("template.language", "en"),
        ("checkpoint.save_steps", 5),
        ("model.adapter.type", "dora"),
        ("runtime.accelerate.unknown_key", True),
    ],
)
def test_unknown_keys_and_legacy_aliases_fail(
    tmp_path: Path,
    field_path: str,
    value: Any,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    _set_nested(payload, field_path, value)
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError, match="schema validation"):
        load_train_config(config_path)


def test_train_order_defaults_to_source_order_and_rejects_shuffle(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    payload = _minimal_config()
    payload["data"].pop("train_order", None)
    _write_yaml(default_path, payload)

    resolved = load_train_config(default_path)

    assert resolved.config.data.train_order == "source_order"

    shuffle_path = tmp_path / "shuffle.yaml"
    payload = _minimal_config()
    payload["data"]["train_order"] = "shuffle"
    _write_yaml(shuffle_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(shuffle_path)

    assert "source_order" in str(exc_info.value)
    assert "shuffle" in str(exc_info.value)


def test_geometry_flip_augmentation_defaults_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["data"].pop("augmentation", None)
    _write_yaml(config_path, payload)

    resolved = load_train_config(config_path)

    geometry_flips = resolved.config.data.augmentation.train.geometry_flips
    assert geometry_flips.enabled is False
    assert geometry_flips.horizontal_prob == 0.0
    assert geometry_flips.vertical_prob == 0.0


def test_geometry_flip_augmentation_accepts_probabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["data"]["augmentation"] = {
        "train": {
            "geometry_flips": {
                "enabled": True,
                "horizontal_prob": 0.5,
                "vertical_prob": 0.25,
            }
        }
    }
    _write_yaml(config_path, payload)

    resolved = load_train_config(config_path)

    geometry_flips = resolved.config.data.augmentation.train.geometry_flips
    assert geometry_flips.enabled is True
    assert geometry_flips.horizontal_prob == pytest.approx(0.5)
    assert geometry_flips.vertical_prob == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_prob", -0.01),
        ("horizontal_prob", 1.01),
        ("vertical_prob", float("inf")),
        ("vertical_prob", "0.5"),
    ],
)
def test_geometry_flip_augmentation_rejects_invalid_probabilities(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["data"]["augmentation"] = {
        "train": {
            "geometry_flips": {
                "enabled": True,
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            }
        }
    }
    payload["data"]["augmentation"]["train"]["geometry_flips"][field] = value
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert "data.augmentation.train.geometry_flips" in exc_info.value.context["field"]
    assert field in exc_info.value.context["field"]


def test_geometry_flip_augmentation_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["data"]["augmentation"] = {
        "train": {
            "geometry_flips": {
                "enabled": True,
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
                "include_composed": True,
            }
        }
    }
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert "include_composed" in exc_info.value.context["field"]


def test_coord_gaussian_rps_loss_config_defaults_disabled_and_loads_plugin_params(
    tmp_path: Path,
) -> None:
    default_path = tmp_path / "default.yaml"
    payload = _minimal_config()
    _write_yaml(default_path, payload)

    default_config = load_train_config(default_path).config
    default_coord_loss = default_config.losses.protected.coord_gaussian_rps
    assert default_coord_loss.weight == 0.0
    assert default_coord_loss.gaussian_weight == pytest.approx(0.5)
    assert default_coord_loss.rps_weight == pytest.approx(0.2)
    assert default_coord_loss.gaussian_r95_axis_fraction == pytest.approx(0.04)
    assert default_coord_loss.gaussian_r95_cap_bins == 8
    assert default_coord_loss.gaussian_r95_min_bins == 1
    assert default_coord_loss.gaussian_r95_fallback_bins == 8

    enabled_path = tmp_path / "enabled.yaml"
    payload = _minimal_config()
    payload["losses"]["protected"]["coord_gaussian_rps"] = {
        "weight": 1.0,
        "gaussian_weight": 0.5,
        "rps_weight": 0.2,
        "temperature": 1.0,
        "gaussian_r95_axis_fraction": 0.04,
        "gaussian_r95_cap_bins": 8,
        "gaussian_r95_min_bins": 1,
        "gaussian_r95_fallback_bins": 8,
    }
    _write_yaml(enabled_path, payload)

    enabled_coord_loss = (
        load_train_config(enabled_path).config.losses.protected.coord_gaussian_rps
    )
    assert enabled_coord_loss.weight == pytest.approx(1.0)
    assert enabled_coord_loss.gaussian_weight == pytest.approx(0.5)
    assert enabled_coord_loss.rps_weight == pytest.approx(0.2)


def test_legacy_dlora_adapter_spelling_explains_v1_dora_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["adapter"]["type"] = "dlora"
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert "V1 uses adapter.type: dora" in exc_info.value.message


def test_legacy_dlora_message_survives_multiple_schema_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["model"]["logits_memory_budget_bytes"] = -1
    payload["adapter"]["type"] = "dlora"
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert "V1 uses adapter.type: dora" in exc_info.value.message
    assert exc_info.value.context["error_count"] >= 2


def test_nested_extends_and_cycles_fail(tmp_path: Path) -> None:
    nested = tmp_path / "nested.yaml"
    payload = _minimal_config()
    payload["data"]["extends"] = "other.yaml"
    _write_yaml(nested, payload)
    with pytest.raises(ConfigContractError, match="top level"):
        load_train_config(nested)

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _write_yaml(a, {"schema_version": 1, "extends": "b.yaml"})
    _write_yaml(b, {"extends": "a.yaml"})
    with pytest.raises(ConfigContractError, match="cycle"):
        load_train_config(a)


def test_required_placeholder_and_root_schema_version_fail(tmp_path: Path) -> None:
    required_path = tmp_path / "required.yaml"
    payload = _minimal_config()
    payload["run"]["name"] = "REQUIRED"
    _write_yaml(required_path, payload)
    with pytest.raises(ConfigContractError, match="REQUIRED"):
        load_train_config(required_path)

    no_schema = tmp_path / "no_schema.yaml"
    payload = _minimal_config()
    payload.pop("schema_version")
    _write_yaml(no_schema, payload)
    with pytest.raises(ConfigContractError, match="schema_version"):
        load_train_config(no_schema)


def test_child_null_does_not_delete_inherited_optional_value(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent_payload = _minimal_config()
    parent_payload["template"]["prompt"]["system"] = "parent-system"
    _write_yaml(parent, parent_payload)
    _write_yaml(
        child,
        {
            "schema_version": 1,
            "extends": "parent.yaml",
            "template": {"prompt": {"system": None}},
        },
    )

    with pytest.raises(ConfigContractError, match="null cannot delete"):
        load_train_config(child)

    root_null = tmp_path / "root_null.yaml"
    root_payload = _minimal_config()
    root_payload["template"]["prompt"]["system"] = None
    _write_yaml(root_null, root_payload)
    assert load_train_config(root_null).config.template.prompt.system is None


def test_runtime_batch_and_qwen_control_checks(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["training"]["effective_batch_size"] = 4
    _write_yaml(config_path, payload)
    resolved = load_train_config(config_path)

    batch = resolve_effective_batch_runtime(resolved.config, world_size=2)
    assert batch.resolved_grad_accum_steps == 2

    with pytest.raises(ConfigContractError, match="divide evenly"):
        resolve_effective_batch_runtime(resolved.config, world_size=3)

    assert estimate_full_logits_bytes(
        global_max_length=12_000,
        vocab_size=100_000,
        dtype="bf16",
    ) == 2_400_000_000
    with pytest.raises(ConfigContractError, match="budget"):
        resolve_qwen_runtime_controls(
            resolved.config,
            tokenizer_vocab_size=3_000_000,
            model_logits_dtype="bf16",
        )
    qwen_controls = resolve_qwen_runtime_controls(
        resolved.config,
        tokenizer_vocab_size=100_000,
        model_logits_dtype="bf16",
    )
    assert qwen_controls.estimated_logits_bytes == 2_400_000_000

    tiny_budget_path = tmp_path / "tiny_budget.yaml"
    payload["model"]["logits_memory_budget_bytes"] = 1
    _write_yaml(tiny_budget_path, payload)
    tiny_budget = load_train_config(tiny_budget_path)
    with pytest.raises(ConfigContractError) as exc_info:
        resolve_qwen_runtime_controls(
            tiny_budget.config,
            tokenizer_vocab_size=100_000,
            model_logits_dtype="bf16",
        )
    assert exc_info.value.code == "config.logits_memory_budget"
    assert exc_info.value.context["estimated_bytes"] == 2_400_000_000
    assert exc_info.value.context["budget_bytes"] == 1

    with pytest.raises(ConfigContractError) as fp32_exc_info:
        resolve_qwen_runtime_controls(
            resolved.config,
            tokenizer_vocab_size=100_000,
            model_logits_dtype="fp32",
        )
    assert fp32_exc_info.value.context["estimated_bytes"] == 4_800_000_000
    assert fp32_exc_info.value.context["budget_bytes"] == 4_000_000_000
    assert fp32_exc_info.value.context["model_logits_dtype"] == "fp32"
    assert fp32_exc_info.value.context["compute_precision"] == "bf16"


def test_run_directory_collision_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _minimal_config()
    payload["run"]["artifact_root"] = "artifacts"
    _write_yaml(config_path, payload)
    resolved = load_train_config(config_path)

    existing = tmp_path / "artifacts" / "run"
    existing.mkdir(parents=True)
    with pytest.raises(ConfigContractError, match="already exists"):
        resolve_run_directory(resolved.config, cwd=tmp_path)

    payload["run"]["collision_policy"] = "timestamp"
    _write_yaml(config_path, payload)
    resolved = load_train_config(config_path)
    run_dir = resolve_run_directory(
        resolved.config,
        cwd=tmp_path,
        timestamp="20260630T000000Z",
    )
    assert run_dir.run_dir == tmp_path / "artifacts" / "run-20260630T000000Z"

    run_dir.run_dir.mkdir()
    with pytest.raises(ConfigContractError, match="timestamped"):
        resolve_run_directory(
            resolved.config,
            cwd=tmp_path,
            timestamp="20260630T000000Z",
        )


def test_trace_config_writes_resolved_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "trace-run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.trace_config",
            "--config",
            str(FIXTURE_CONFIG),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(result.stdout)

    assert summary["artifacts"]["json_path"] == str(
        run_dir / "configs" / "resolved.json"
    )
    assert (run_dir / "configs" / "resolved.yaml").exists()

    sentinel = run_dir / "configs" / "resolved.json"
    before = sentinel.read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.trace_config",
            "--config",
            str(FIXTURE_CONFIG),
            "--run-dir",
            str(run_dir),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert sentinel.read_text() == before


def test_production_relaunch_configs_load_strictly() -> None:
    prod_path = Path(
        "configs/coordexp_swift/prod/"
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_"
        "accelerate8_ebs64_4epoch_warmup0p1.yaml"
    )
    smoke_path = Path(
        "configs/coordexp_swift/smoke/"
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_"
        "accelerate8_ebs64_2step_warmup0p1_eval_patchproof.yaml"
    )

    prod = load_train_config(prod_path).config
    smoke = load_train_config(smoke_path).config

    assert prod.run.name == (
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_"
        "accelerate8_ebs64_4epoch_warmup0p1"
    )
    assert prod.adapter.rank == 16
    assert prod.adapter.alpha == 32
    assert prod.optimizer.scheduler.warmup_ratio == 0.1
    assert prod.optimizer.scheduler.warmup_steps is None
    assert prod.training.effective_batch_size == 64
    assert prod.training.max_steps is None
    assert prod.training.epochs == 4
    assert prod.training.max_grad_norm == 1.0
    assert prod.runtime.seed == 17
    assert prod.runtime.backend == "accelerate"
    assert prod.runtime.accelerate is not None
    assert prod.runtime.accelerate.gradient_accumulation_steps is None
    assert prod.adapter.target_towers == ("language",)
    assert prod.adapter.target_modules == "all_linear"
    assert prod.template.object_field_order == "desc_first"
    assert prod.template.object_ordering == "geo_sorted"
    assert prod.packing.global_max_length == 12_000

    assert smoke.run.name == (
        "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_"
        "accelerate8_ebs64_2step_warmup0p1_eval_patchproof"
    )
    assert smoke.adapter.rank == 16
    assert smoke.adapter.alpha == 32
    assert smoke.optimizer.scheduler.warmup_ratio == 0.1
    assert smoke.training.effective_batch_size == 64
    assert smoke.training.max_steps == 2
    assert smoke.eval.forward.steps == (1,)
    assert smoke.checkpoint.steps == (2,)
    assert smoke.runtime.accelerate is not None
    assert smoke.runtime.accelerate.gradient_accumulation_steps is None


def test_coord_gaussian_rps_length12000_smoke_config_loads_strictly() -> None:
    smoke_path = Path(
        "configs/coordexp_swift/smoke/"
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
        "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
    )

    smoke = load_train_config(smoke_path).config
    batch = resolve_effective_batch_runtime(smoke, world_size=8)

    assert smoke.run.name == (
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
        "accelerate8_ebs24_2step_warmup0p1_eval_patchproof"
    )
    assert smoke.adapter.rank == 16
    assert smoke.adapter.alpha == 32
    assert smoke.template.object_field_order == "desc_first"
    assert smoke.template.object_ordering == "geo_sorted"
    assert smoke.data.train.sample_limit == 256
    assert smoke.data.eval is not None
    assert smoke.data.eval.sample_limit == 64
    assert smoke.data.augmentation.train.geometry_flips.enabled is True
    assert smoke.data.augmentation.train.geometry_flips.horizontal_prob == pytest.approx(
        0.5
    )
    assert smoke.data.augmentation.train.geometry_flips.vertical_prob == pytest.approx(
        0.2
    )
    assert smoke.packing.global_max_length == 12_000
    assert smoke.training.effective_batch_size == 24
    assert smoke.training.max_steps == 2
    assert smoke.optimizer.scheduler.warmup_ratio == 0.1
    assert smoke.runtime.seed == 17
    assert smoke.runtime.backend == "accelerate"
    assert smoke.runtime.accelerate is not None
    assert smoke.runtime.accelerate.gradient_accumulation_steps is None
    assert batch.resolved_grad_accum_steps == 3

    protected = smoke.losses.protected
    assert protected.base_ce.weight == pytest.approx(1.0)
    assert protected.token_type_gate.weight == pytest.approx(0.25)
    assert protected.token_type_gate.groups == (
        "desc_text",
        "schema",
        "coordinate",
        "eos",
    )
    assert protected.coord_gaussian_rps.weight == pytest.approx(1.0)
    assert protected.coord_gaussian_rps.gaussian_weight == pytest.approx(0.5)
    assert protected.coord_gaussian_rps.rps_weight == pytest.approx(0.2)
    assert protected.coord_gaussian_rps.gaussian_r95_axis_fraction == pytest.approx(
        0.04
    )
    assert protected.coord_gaussian_rps.gaussian_r95_cap_bins == 8
    assert protected.coord_gaussian_rps.gaussian_r95_min_bins == 1
    assert protected.coord_gaussian_rps.gaussian_r95_fallback_bins == 8


def test_coord_gaussian_rps_prod_config_loads_strictly() -> None:
    prod_path = Path(
        "configs/coordexp_swift/prod/"
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
        "accelerate8_ebs24_8epoch_warmup0p1.yaml"
    )

    prod = load_train_config(prod_path).config
    batch = resolve_effective_batch_runtime(prod, world_size=8)
    schedule = resolve_planned_step_schedule(
        prod,
        packs_per_epoch=14_660,
        world_size=8,
        source_config_path=str(prod_path),
    )

    assert prod.run.name == (
        "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
        "accelerate8_ebs24_8epoch_warmup0p1"
    )
    assert prod.adapter.rank == 16
    assert prod.adapter.alpha == 32
    assert prod.adapter.target_towers == ("language",)
    assert prod.adapter.target_modules == "all_linear"
    assert prod.template.object_field_order == "desc_first"
    assert prod.template.object_ordering == "geo_sorted"
    assert prod.data.train.sample_limit is None
    assert prod.data.eval is not None
    assert prod.data.eval.sample_limit is None
    assert prod.data.augmentation.train.geometry_flips.enabled is True
    assert prod.data.augmentation.train.geometry_flips.horizontal_prob == pytest.approx(
        0.5
    )
    assert prod.data.augmentation.train.geometry_flips.vertical_prob == pytest.approx(
        0.2
    )
    assert prod.packing.global_max_length == 12_000
    assert prod.training.effective_batch_size == 24
    assert prod.training.max_steps is None
    assert prod.training.epochs == 8
    assert prod.training.max_grad_norm == 1.0
    assert prod.training.logging.every_fraction == pytest.approx(0.002)
    assert prod.training.logging.steps == (1,)
    assert prod.optimizer.scheduler.warmup_ratio == 0.1
    assert prod.optimizer.scheduler.warmup_steps is None
    assert prod.runtime.seed == 17
    assert prod.runtime.backend == "accelerate"
    assert prod.runtime.accelerate is not None
    assert prod.runtime.accelerate.gradient_accumulation_steps is None
    assert batch.resolved_grad_accum_steps == 3
    assert schedule.resolved_max_steps == 4887
    assert [
        event.planned_step_id for event in schedule.events["training.logging"][:4]
    ] == [1, 10, 20, 30]
    assert schedule.events["training.logging"][-1].planned_step_id == 4887
    assert prod.eval.forward.every_fraction == pytest.approx(0.1)
    assert prod.eval.forward.steps == ()
    assert prod.checkpoint.every_fraction == pytest.approx(0.3)
    assert prod.checkpoint.steps == ()

    protected = prod.losses.protected
    assert protected.base_ce.weight == pytest.approx(1.0)
    assert protected.token_type_gate.weight == pytest.approx(0.2)
    assert protected.token_type_gate.groups == (
        "desc_text",
        "schema",
        "coordinate",
        "eos",
    )
    assert protected.coord_gaussian_rps.weight == pytest.approx(1.0)
    assert protected.coord_gaussian_rps.gaussian_weight == pytest.approx(0.5)
    assert protected.coord_gaussian_rps.rps_weight == pytest.approx(0.2)
    assert protected.coord_gaussian_rps.temperature == pytest.approx(1.0)
    assert protected.coord_gaussian_rps.gaussian_r95_axis_fraction == pytest.approx(
        0.04
    )
    assert protected.coord_gaussian_rps.gaussian_r95_cap_bins == 8
    assert protected.coord_gaussian_rps.gaussian_r95_min_bins == 1
    assert protected.coord_gaussian_rps.gaussian_r95_fallback_bins == 8


def _minimal_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {"name": "run", "artifact_root": "artifacts"},
        "model": {
            "base_model": "model_cache/models/Qwen/example",
            "attn_implementation": "flash_attention_2",
            "logits_memory_budget_bytes": 4_000_000_000,
            "processor": {
                "do_resize": False,
                "max_raw_pixels": 2_000_000,
                "max_merged_visual_tokens": 4096,
            },
            "special_token_embeddings": {
                "groups": {
                    "coordinate_tokens": "default_coord_0_999",
                    "wrapper_tokens": "default_object_box_wrappers",
                }
            },
        },
        "adapter": {
            "type": "dora",
            "target_towers": ["language"],
            "target_modules": "all_linear",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "bias": "none",
        },
        "data": {
            "train": {"path": "train.jsonl", "sample_limit": 2},
            "eval": {"path": "eval.jsonl", "sample_limit": 2},
        },
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {
                "system": None,
                "user": "Describe each requested object with its bounding box.",
            },
        },
        "packing": {"global_max_length": 12_000},
        "losses": {
            "normalizer": "segment_balanced",
            "protected": {
                "base_ce": {"weight": 1.0},
                "token_type_gate": {
                    "weight": 1.0,
                    "groups": ["desc_text", "schema", "coordinate", "eos"],
                },
            },
        },
        "optimizer": {
            "name": "adamw_torch",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "kwargs": {},
            "groups": {
                "adapters": {"language": {"lr": 2.0e-4, "weight_decay": 0.0}},
                "token_embeddings": {"lr": 1.0e-4, "weight_decay": 0.0},
            },
            "scheduler": {
                "name": "cosine_with_warmup",
                "warmup_ratio": 0.0,
                "warmup_steps": None,
                "kwargs": {},
            },
        },
        "training": {
            "mode": "supervised",
            "epochs": 1,
            "max_steps": 5,
            "effective_batch_size": 2,
            "precision": "bf16",
            "max_grad_norm": 1.0,
            "logging": {"every_fraction": None, "steps": [1, 2, 3, 4, 5]},
        },
        "runtime": {"backend": "single", "seed": 17},
        "eval": {
            "forward": {"every_fraction": None, "steps": [2, 4]},
            "inference": {"enabled": False},
        },
        "checkpoint": {"every_fraction": 0.4, "steps": [], "save_final": True},
        "debug": {"dry_run_writes_artifacts": True},
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
