from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig, TrainingConfig
from src.training.pipelines.stage1_compact_trie_ce import Stage1CompactTrieCEPipeline
from src.training.pipelines.stage1_json_ce import Stage1JsonCEPipeline
from src.training.pipelines.stage2_two_channel import Stage2TwoChannelPipeline
from src.training.surfaces import TrainingSurfaceResolver


def _removed_key(*parts: str) -> str:
    return "_".join(parts)


def _shadow_config(surface_id: str = "stage1_json_ce") -> dict[str, object]:
    supervision: dict[str, object]
    if surface_id == "stage2_two_channel":
        supervision = {
            "mode": "two_channel",
            "channels": {
                "a": {"source": "ground_truth"},
                "b": {"source": "rollout"},
            },
        }
    elif surface_id == "stage1_compact_trie_ce":
        supervision = {"mode": "compact_trie"}
    else:
        supervision = {"mode": "json_ce"}

    return {
        "run": {"id": "shadow-smoke", "scope": "smoke"},
        "surface": {"id": surface_id},
        "data": {"train_jsonl": "train.jsonl", "validation_jsonl": "val.jsonl"},
        "template": {"id": "json_chat"},
        "supervision": supervision,
        "objectives": _surface_objectives(surface_id),
        "observability": {"level": "minimal"},
        "artifacts": {"output_root": "output/shadow"},
        "runtime": {"trainer": "shadow"},
    }


def _surface_objectives(surface_id: str) -> dict[str, dict[str, object]]:
    if surface_id == "stage1_compact_trie_ce":
        return {
            "trie_ce": {"enabled": True, "weight": 1.0},
            "coord_soft_ce": {"enabled": False, "weight": 0.25},
        }
    if surface_id == "stage2_two_channel":
        return {
            "token_ce": {"enabled": True, "weight": 1.0},
            "box_regression": {"enabled": False, "weight": 0.1},
        }

    return {"token_ce": {"enabled": True, "weight": 1.0}}


def test_shadow_config_accepts_only_canonical_top_level_domains() -> None:
    cfg = _shadow_config()

    resolved = TrainingSurfaceResolver().resolve(cfg)

    assert resolved.top_level_domains == (
        "run",
        "surface",
        "data",
        "template",
        "supervision",
        "objectives",
        "observability",
        "artifacts",
        "runtime",
    )


def test_shadow_resolver_outputs_are_detached_from_mutable_payload() -> None:
    cfg = _shadow_config()

    resolved = TrainingSurfaceResolver().resolve(cfg)
    cfg["run"]["id"] = "mutated"  # type: ignore[index]

    assert resolved.run["id"] == "shadow-smoke"
    with pytest.raises(TypeError):
        resolved.run["id"] = "blocked"  # type: ignore[index]


def test_shadow_config_rejects_unknown_top_level_domain() -> None:
    cfg = _shadow_config()
    cfg["custom"] = {"legacy_extra": True}

    with pytest.raises(ValueError, match="Unknown top-level"):
        TrainingSurfaceResolver().resolve(cfg)


def test_shadow_config_rejects_missing_required_domain() -> None:
    cfg = _shadow_config()
    del cfg["runtime"]

    with pytest.raises(ValueError, match="Missing required top-level"):
        TrainingSurfaceResolver().resolve(cfg)


@pytest.mark.parametrize(
    ("surface_id", "pipeline_type", "pipeline_id"),
    [
        ("stage1_json_ce", Stage1JsonCEPipeline, "stage1_json_ce"),
        (
            "stage1_compact_trie_ce",
            Stage1CompactTrieCEPipeline,
            "stage1_compact_trie_ce",
        ),
        ("stage2_two_channel", Stage2TwoChannelPipeline, "stage2_two_channel"),
    ],
)
def test_surface_id_selects_shadow_pipeline(
    surface_id: str,
    pipeline_type: type[object],
    pipeline_id: str,
) -> None:
    resolved = TrainingSurfaceResolver().resolve(_shadow_config(surface_id))

    assert isinstance(resolved.pipeline, pipeline_type)
    assert resolved.pipeline.identity.surface_id == surface_id
    assert resolved.pipeline.identity.pipeline_id == pipeline_id


def test_stage1_json_rejects_stage2_supervision_sections() -> None:
    cfg = _shadow_config("stage1_json_ce")
    cfg["supervision"] = {
        "mode": "json_ce",
        "stage2_assignment": {"strategy": "greedy_iou"},
    }

    with pytest.raises(ValueError, match="stage2_assignment"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage2_two_channel_requires_channel_supervision() -> None:
    cfg = _shadow_config("stage2_two_channel")
    cfg["supervision"] = {"mode": "two_channel"}

    with pytest.raises(ValueError, match="supervision.channels"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage2_two_channel_rejects_non_mapping_channel_descriptor() -> None:
    cfg = _shadow_config("stage2_two_channel")
    cfg["supervision"] = {
        "mode": "two_channel",
        "channels": {
            "a": True,
            "b": {"source": "rollout"},
        },
    }

    with pytest.raises(TypeError, match=r"supervision\.channels\.a"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage2_two_channel_channel_descriptor_requires_source() -> None:
    cfg = _shadow_config("stage2_two_channel")
    cfg["supervision"] = {
        "mode": "two_channel",
        "channels": {
            "a": {"source": "ground_truth"},
            "b": {"objective_scope": "rollout"},
        },
    }

    with pytest.raises(ValueError, match=r"supervision\.channels\.b\.source"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage1_compact_trie_surface_requires_enabled_trie_objective() -> None:
    cfg = _shadow_config("stage1_compact_trie_ce")
    cfg["objectives"] = {"token_ce": {"enabled": True, "weight": 1.0}}

    with pytest.raises(ValueError, match="requires enabled objectives.*trie_ce"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage1_compact_trie_surface_accepts_trie_coord_profile() -> None:
    cfg = _shadow_config("stage1_compact_trie_ce")
    cfg["objectives"] = {
        "trie_ce": {"enabled": True, "weight": 1.0},
        "coord_soft_ce": {"enabled": True, "weight": 0.25},
    }

    resolved = TrainingSurfaceResolver().resolve(cfg)

    assert [entry.objective_id for entry in resolved.objectives.enabled_objectives] == [
        "trie_ce",
        "coord_soft_ce",
    ]


def test_stage1_json_surface_rejects_trie_objectives() -> None:
    cfg = _shadow_config("stage1_json_ce")
    cfg["objectives"] = {
        "token_ce": {"enabled": True, "weight": 1.0},
        "trie_ce": {"enabled": False, "weight": 1.0},
    }

    with pytest.raises(ValueError, match="does not support objective keys.*trie_ce"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage2_two_channel_surface_requires_enabled_token_objective() -> None:
    cfg = _shadow_config("stage2_two_channel")
    cfg["objectives"] = {"box_regression": {"enabled": True, "weight": 0.1}}

    with pytest.raises(ValueError, match="requires enabled objectives.*token_ce"):
        TrainingSurfaceResolver().resolve(cfg)


def test_stage2_two_channel_surface_rejects_trie_objective_profile() -> None:
    cfg = _shadow_config("stage2_two_channel")
    cfg["objectives"] = {
        "token_ce": {"enabled": True, "weight": 1.0},
        "trie_ce": {"enabled": True, "weight": 0.5},
    }

    with pytest.raises(ValueError, match="does not support objective keys.*trie_ce"):
        TrainingSurfaceResolver().resolve(cfg)


def test_shadow_resolver_rejects_non_finite_shared_domain_metadata() -> None:
    cfg = _shadow_config()
    cfg["run"] = {"id": "shadow-smoke", "scope": "smoke", "seed": float("inf")}

    with pytest.raises(ValueError, match=r"run\.seed"):
        TrainingSurfaceResolver().resolve(cfg)


@pytest.mark.parametrize(
    "field",
    ["owner", "expiry", "notes", "surface_or_pipeline_opt_in"],
)
def test_experimental_block_requires_strict_fields(field: str) -> None:
    cfg = _shadow_config()
    cfg["experimental"] = {
        "owner": "coordexp",
        "expiry": "2026-06-30",
        "notes": "Temporary shadow resolver smoke.",
        "surface_or_pipeline_opt_in": True,
    }
    del cfg["experimental"][field]  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        TrainingSurfaceResolver().resolve(cfg)


def test_experimental_block_requires_explicit_opt_in() -> None:
    cfg = _shadow_config()
    cfg["run"] = {"id": "prod-shadow", "scope": "production"}
    cfg["experimental"] = {
        "owner": "coordexp",
        "expiry": "2026-06-30",
        "notes": "Temporary production-like smoke.",
        "surface_or_pipeline_opt_in": False,
    }

    with pytest.raises(ValueError, match="surface_or_pipeline_opt_in"):
        TrainingSurfaceResolver().resolve(cfg)


def test_experimental_block_accepts_explicit_opt_in() -> None:
    cfg = _shadow_config()
    cfg["experimental"] = {
        "owner": "coordexp",
        "expiry": "2026-06-30",
        "notes": "Temporary shadow resolver smoke.",
        "surface_or_pipeline_opt_in": True,
    }

    resolved = TrainingSurfaceResolver().resolve(cfg)

    assert resolved.experimental is not None
    assert resolved.experimental.owner == "coordexp"


def test_objective_profile_requires_at_least_one_enabled_objective() -> None:
    with pytest.raises(ValueError, match="enable at least one objective"):
        TrainingSurfaceResolver().resolve_objectives(
            {
                "token_ce": {"enabled": False, "weight": 1.0},
                "trie_ce": {"enabled": False, "weight": 1.0},
            }
        )


def test_removed_mechanism_keys_fail_fast_anywhere() -> None:
    cfg = _shadow_config()
    removed_objective = _removed_key("loss", "duplicate", "burst", "unlikelihood")
    cfg["objectives"] = {
        "token_ce": {"enabled": True},
        removed_objective: {"enabled": True},
    }

    with pytest.raises(ValueError, match=removed_objective):
        TrainingSurfaceResolver().resolve(cfg)


@pytest.mark.parametrize(
    "removed_key",
    [
        _removed_key("adjacent", "repulsion", "weight"),
        _removed_key("continue", "over", "eos", "margin"),
        _removed_key("eos", "loosen"),
        _removed_key(
            "forced",
            "continuation",
        ),
        _removed_key("separator", "continue", "weight"),
        _removed_key("stop", "gate"),
        _removed_key("stop", "signal", "damping"),
    ],
)
def test_removed_nested_objective_config_keys_fail_fast(removed_key: str) -> None:
    cfg = _shadow_config()
    cfg["objectives"] = {
        "token_ce": {
            "enabled": True,
            "config": {
                "safe": 1,
                "nested": {removed_key: 0.1},
            },
        },
    }

    with pytest.raises(ValueError, match=removed_key):
        TrainingSurfaceResolver().resolve(cfg)


@pytest.mark.parametrize(
    ("config_relpath", "expected_type"),
    [
        (
            "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_tiny.yaml",
            LatestDetectionTrainingConfig,
        ),
        (
            "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml",
            LatestDetectionTrainingConfig,
        ),
        ("configs/stage2_two_channel/smoke/a_only.yaml", TrainingConfig),
        ("configs/stage2_two_channel/prod/a_only.yaml", TrainingConfig),
    ],
)
def test_current_config_loader_configs_remain_passthrough(
    config_relpath: str,
    expected_type: type[object],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    cfg = ConfigLoader.load_materialized_training_config(str(repo_root / config_relpath))

    assert isinstance(cfg, expected_type)
