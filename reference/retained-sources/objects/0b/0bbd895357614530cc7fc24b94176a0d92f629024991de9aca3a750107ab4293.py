from __future__ import annotations


import pytest
import torch


from probes.logit_lens import base as probe


def _row(offset: int) -> list[int]:
    return [
        probe.OBJECT_REF_START,
        100 + offset,
        probe.OBJECT_REF_END,
        probe.BOX_START,
        probe.COORD_START + offset,
        probe.COORD_START + offset + 1,
        probe.COORD_START + offset + 2,
        probe.COORD_START + offset + 3,
        probe.BOX_END,
    ]


def test_select_sites_aligns_state_t_to_actual_token_t_plus_one() -> None:
    generated = [*_row(0), *_row(10), *_row(20), 151645]
    sites = probe.select_sites(prompt_token_count=7, generated_token_ids=generated)
    assert len(sites) <= probe.MAX_SITES
    assert sites[0]["position"] == 6
    assert sites[0]["actual_next_token_id"] == probe.OBJECT_REF_START
    by_label = {label: site for site in sites for label in site["labels"]}
    assert by_label["first_row_coord_1_decision"]["actual_next_token_id"] == probe.COORD_START
    assert by_label["middle_row_coord_4_decision"]["actual_next_token_id"] == probe.COORD_START + 13
    assert by_label["last_row_boundary"]["actual_next_token_id"] == 151645
    assert by_label["terminal_eos_decision"]["actual_next_token_id"] == 151645


def test_select_sites_last_boundary_without_observed_next_is_retained() -> None:
    sites = probe.select_sites(prompt_token_count=3, generated_token_ids=_row(0))
    boundary = next(site for site in sites if "last_row_boundary" in site["labels"])
    assert boundary["position"] == 3 + len(_row(0)) - 1
    assert boundary["actual_next_token_id"] is None


class _MutatingDeepStack(torch.nn.Module):
    def __init__(self, layers: int, visual_mask: torch.Tensor) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(layers)])
        self.norm = torch.nn.Identity()
        self.visual_mask = visual_mask

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < probe.DEEPSTACK_LAYER_COUNT:
                value[self.visual_mask] += float(index + 1)
        return self.norm(value)


def test_capture_clones_before_inplace_injection_and_detects_text_sensitivity() -> None:
    visual = torch.tensor([[False, True, False, True]])
    model = _MutatingDeepStack(4, visual)
    value = torch.zeros(1, 4, 2)
    with probe.DeepStackLensCapture(
        layers=model.layers,
        norm=model.norm,
        selected_positions=[0, 2],
        visual_mask=visual,
    ) as capture:
        model(value)
    capture.validate()
    assert capture.boundaries[0]["visual_delta_l2"] > 0
    assert capture.boundaries[0]["text_exact_equal"] is True
    assert torch.equal(capture.residuals[0], torch.zeros(2, 2))

    capture.boundaries[0]["text_exact_equal"] = False
    with pytest.raises(RuntimeError, match="changed text"):
        capture.validate()


def test_site_selection_rejects_empty_generation() -> None:
    with pytest.raises(RuntimeError, match="requires prompt and generated"):
        probe.select_sites(prompt_token_count=4, generated_token_ids=[])
