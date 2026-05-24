import pytest
import torch

from src.trainers.teacher_forcing.contracts import PipelineResult
from src.trainers.teacher_forcing.objective_atoms import project_stage2_objective_atoms


def _t(x: float) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float32)


def _token_ce_spec(weight: float = 1.0) -> dict:
    return {
        "name": "token_ce",
        "enabled": True,
        "weight": float(weight),
        "channels": ["A", "B"],
        "config": {},
    }


def _schema_format_ce_spec(weight: float = 1.0) -> dict:
    return {
        "name": "schema_format_ce",
        "enabled": True,
        "weight": float(weight),
        "channels": ["B"],
        "config": {"schema_ce_weight": 1.0},
    }


def test_project_stage2_objective_atoms_is_strictly_additive_for_token_ce() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(1.5),
        module_losses={"token_ce": _t(1.5)},
        metrics={},
        state={
            "token_ce_struct_contrib": _t(0.4),
            "token_ce_desc_contrib": _t(0.1),
        },
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=[_token_ce_spec(weight=3.0)],
        text_provenance="B_rollout_text",
        coord_provenance=None,
        emit_text=True,
        emit_coord=False,
        require_additive=True,
    )

    assert atoms["loss/B_rollout_text/struct_ce"] == pytest.approx(1.2)
    assert atoms["loss/B_rollout_text/desc_ce"] == pytest.approx(0.3)
    assert sum(atoms.values()) == pytest.approx(1.5)


def test_project_stage2_objective_atoms_projects_schema_format_ce() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(0.75),
        module_losses={"schema_format_ce": _t(0.75)},
        metrics={},
        state={"schema_format_ce_contrib": _t(0.5)},
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=[_schema_format_ce_spec(weight=1.5)],
        text_provenance="B_rollout_text",
        coord_provenance=None,
        emit_text=True,
        emit_coord=False,
        require_additive=True,
    )

    assert atoms == {"loss/B_rollout_text/schema_format_ce": pytest.approx(0.75)}


def test_project_stage2_objective_atoms_allows_disabling_text_emission() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(0.0),
        module_losses={"token_ce": _t(0.0)},
        metrics={},
        state={
            "token_ce_struct_contrib": _t(0.0),
            "token_ce_desc_contrib": _t(0.0),
        },
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=[_token_ce_spec()],
        text_provenance=None,
        coord_provenance=None,
        emit_text=False,
        emit_coord=False,
        require_additive=True,
    )

    assert atoms == {}


def test_project_stage2_objective_atoms_raises_on_mismatch() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(1.0),
        module_losses={"token_ce": _t(1.0)},
        metrics={},
        state={
            "token_ce_struct_contrib": _t(0.1),
            "token_ce_desc_contrib": _t(0.1),
        },
    )

    with pytest.raises(ValueError, match=r"Stage2 atom projection mismatch"):
        _ = project_stage2_objective_atoms(
            pipeline_result=pipeline_result,
            objective_specs=[_token_ce_spec()],
            text_provenance="B_rollout_text",
            coord_provenance=None,
            emit_text=True,
            emit_coord=False,
            require_additive=True,
        )


@pytest.mark.parametrize("module_name", ["bbox_geo", "bbox_size_aux", "coord_reg"])
def test_project_stage2_objective_atoms_rejects_removed_modules(
    module_name: str,
) -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(0.0),
        module_losses={module_name: _t(0.0)},
        metrics={},
        state={},
    )

    with pytest.raises(
        ValueError,
        match=rf"stage2 atom projection does not know how to project module '{module_name}'",
    ):
        _ = project_stage2_objective_atoms(
            pipeline_result=pipeline_result,
            objective_specs=[
                {
                    "name": module_name,
                    "enabled": True,
                    "weight": 1.0,
                    "channels": ["A", "B"],
                    "config": {},
                }
            ],
            text_provenance=None,
            coord_provenance=None,
            emit_text=False,
            emit_coord=False,
            require_additive=False,
        )
