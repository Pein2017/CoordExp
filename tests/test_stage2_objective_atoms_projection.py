import pytest
import torch

from src.trainers.teacher_forcing.contracts import PipelineResult
from src.trainers.teacher_forcing.objective_atoms import project_stage2_objective_atoms


def _t(x: float) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float32)


def test_project_stage2_objective_atoms_is_strictly_additive() -> None:
    objective_specs = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "loss_dead_anchor_suppression",
            "enabled": True,
            "weight": 0.25,
            "channels": ["B"],
            "config": {},
        },
        {
            "name": "bbox_geo",
            "enabled": True,
            "weight": 2.0,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "bbox_size_aux",
            "enabled": True,
            "weight": 0.25,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "coord_reg",
            "enabled": True,
            "weight": 0.5,
            "channels": ["A", "B"],
            "config": {},
        },
    ]

    state = {
        "token_ce_struct_contrib": _t(0.3),
        "token_ce_desc_contrib": _t(0.2),
        "loss_dead_anchor_suppression_contrib": _t(0.4),
        "bbox_smoothl1_contrib": _t(0.4),
        "bbox_ciou_contrib": _t(0.1),
        "bbox_log_wh_contrib": _t(0.3),
        "bbox_log_area_contrib": _t(0.05),
        "bbox_oversize_contrib": _t(0.0),
        "coord_token_ce_contrib": _t(0.05),
        "coord_soft_ce_contrib": _t(0.0),
        "coord_w1_contrib": _t(0.01),
        "coord_el1_contrib": _t(0.0),
        "coord_ehuber_contrib": _t(0.0),
        "coord_entropy_contrib": _t(-0.02),
        "coord_gate_contrib": _t(0.0),
        "text_gate_contrib": _t(0.0),
    }

    token_loss = _t(0.3 + 0.2)
    dead_anchor_suppression_loss = _t(0.4)
    bbox_loss = _t(0.4 + 0.1)
    bbox_size_aux_loss = _t(0.3 + 0.05 + 0.0)
    coord_loss = _t(0.05 + 0.01 - 0.02)

    module_losses = {
        "token_ce": 1.0 * token_loss,
        "loss_dead_anchor_suppression": 0.25 * dead_anchor_suppression_loss,
        "bbox_geo": 2.0 * bbox_loss,
        "bbox_size_aux": 0.25 * bbox_size_aux_loss,
        "coord_reg": 0.5 * coord_loss,
    }
    total_loss = (
        module_losses["token_ce"]
        + module_losses["loss_dead_anchor_suppression"]
        + module_losses["bbox_geo"]
        + module_losses["bbox_size_aux"]
        + module_losses["coord_reg"]
    )

    pipeline_result = PipelineResult(
        total_loss=total_loss,
        module_losses=module_losses,
        metrics={},
        state=state,
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        text_provenance="B_rollout_text",
        coord_provenance="B_coord",
        emit_text=True,
        emit_coord=True,
        require_additive=True,
    )

    assert atoms["loss/B_rollout_text/struct_ce"] == pytest.approx(0.3)
    assert atoms["loss/B_rollout_text/desc_ce"] == pytest.approx(0.2)
    assert atoms["loss/B_rollout_text/loss_dead_anchor_suppression"] == pytest.approx(0.25 * 0.4)
    assert atoms["loss/B_coord/bbox_smoothl1"] == pytest.approx(2.0 * 0.4)
    assert atoms["loss/B_coord/bbox_ciou"] == pytest.approx(2.0 * 0.1)
    assert atoms["loss/B_coord/bbox_log_wh"] == pytest.approx(0.25 * 0.3)
    assert atoms["loss/B_coord/bbox_log_area"] == pytest.approx(0.25 * 0.05)
    assert atoms["loss/B_coord/coord_token_ce"] == pytest.approx(0.5 * 0.05)
    assert atoms["loss/B_coord/coord_w1"] == pytest.approx(0.5 * 0.01)
    assert atoms["loss/B_coord/coord_entropy"] == pytest.approx(0.5 * -0.02)

    assert sum(atoms.values()) == pytest.approx(float(total_loss.detach().cpu().item()))


def test_project_stage2_objective_atoms_emits_dead_anchor_suppression_text_atom() -> None:
    objective_specs = [
        {
            "name": "loss_dead_anchor_suppression",
            "enabled": True,
            "weight": 1.5,
            "channels": ["B"],
            "config": {},
        },
    ]

    pipeline_result = PipelineResult(
        total_loss=_t(0.3),
        module_losses={"loss_dead_anchor_suppression": _t(0.3)},
        metrics={},
        state={"loss_dead_anchor_suppression_contrib": _t(0.2)},
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        text_provenance="B_rollout_text",
        coord_provenance=None,
        emit_text=True,
        emit_coord=False,
        require_additive=False,
    )

    assert set(atoms.keys()) == {"loss/B_rollout_text/loss_dead_anchor_suppression"}
    assert atoms["loss/B_rollout_text/loss_dead_anchor_suppression"] == pytest.approx(0.3)


def test_project_stage2_objective_atoms_allows_disabling_coord_emission() -> None:
    objective_specs = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "bbox_geo",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "bbox_size_aux",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
        {
            "name": "coord_reg",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
    ]

    module_losses = {
        "token_ce": _t(0.25),
        "bbox_geo": _t(0.0),
        "coord_reg": _t(0.0),
    }
    pipeline_result = PipelineResult(
        total_loss=module_losses["token_ce"],
        module_losses=module_losses,
        metrics={},
        state={
            "token_ce_struct_contrib": _t(0.25),
            "token_ce_desc_contrib": _t(0.0),
        },
    )

    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        text_provenance="A1_text",
        coord_provenance=None,
        emit_text=True,
        emit_coord=False,
        require_additive=True,
    )

    assert set(atoms.keys()) == {"loss/A1_text/struct_ce"}
    assert atoms["loss/A1_text/struct_ce"] == pytest.approx(0.25)


def test_project_stage2_objective_atoms_raises_on_mismatch() -> None:
    objective_specs = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "config": {},
        },
    ]

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
            objective_specs=objective_specs,
            text_provenance="B_rollout_text",
            coord_provenance="B_coord",
            emit_text=True,
            emit_coord=True,
            require_additive=True,
        )
