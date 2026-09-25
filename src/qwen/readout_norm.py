"""Frozen output-only norm coefficients for the tied selected-delta profile.

This verifier neither chooses a cohort nor applies an intervention. The untied
profile retains its separate effective-row construction and identity checks.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from src.artifacts.utf8_json import literal_binding as _binding
from src.qwen.input_identity import tensor_hash as _tensor_hash
from src.qwen.special_token_embeddings import SelectedDeltaInputEmbedding, SelectedDeltaOutputHead


def validate_panel_sources(panel: Mapping[str, Any]) -> None:
    sources = panel.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("panel sources must be a list")
    for expected in sources:
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ValueError("panel source binding is malformed")
        path = Path(expected["path"])
        if _binding(path) != dict(expected):
            raise AssertionError(f"frozen source changed: {path}")

def load_fixed_tied_coefficients(
    *,
    panel: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    expected = panel.get("coefficient_binding", panel.get("coefficients"))
    if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
        raise ValueError("panel needs a coefficient binding")
    path = Path(expected["path"])
    if _binding(path) != dict(expected):
        raise AssertionError("fixed coefficient binding changed")
    coordinate_list = panel.get("coordinate_ids")
    if not isinstance(coordinate_list, list) or len(coordinate_list) != 1000:
        raise ValueError("panel must bind all 1000 coordinate IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate_list):
        raise ValueError("coordinate IDs must be integers")
    coordinate_ids = torch.tensor(coordinate_list, device=device, dtype=torch.long)
    expected_ids = [tokenizer.convert_tokens_to_ids(f"<|coord_{index}|>") for index in range(1000)]
    if coordinate_ids.tolist() != expected_ids:
        raise AssertionError("loaded tokenizer coordinate IDs differ from panel")
    head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    if not isinstance(head, SelectedDeltaOutputHead) or not isinstance(
        embedding, SelectedDeltaInputEmbedding
    ):
        raise AssertionError("loaded model lacks the selected delta input/output seam")
    if head.bias is not None:
        raise AssertionError("readout bias must be absent")
    lookup = {int(token): index for index, token in enumerate(head.selected_token_ids.tolist())}
    if any(int(token) not in lookup for token in coordinate_ids.tolist()):
        raise AssertionError("coordinate IDs are not covered by shared output delta")
    selected = torch.tensor(
        [lookup[int(token)] for token in coordinate_ids.tolist()], device=device
    )
    effective = head.base.weight[coordinate_ids].detach() + head.shared_embed_delta[selected].detach()
    if not torch.equal(effective, embedding(coordinate_ids).detach()):
        raise AssertionError("effective output and input coordinate rows differ")
    coefficients = torch.load(path, map_location="cpu", weights_only=True)
    if _tensor_hash(effective) != coefficients["effective_rows_sha256"]:
        raise AssertionError("effective coordinate row hash differs from frozen coefficients")
    norms = effective.cpu().double().norm(dim=1)
    if not torch.equal(norms, coefficients["norms"]):
        raise AssertionError("effective coordinate row norms differ from frozen coefficients")
    if not torch.equal(norms.median() / norms, coefficients["factors"]):
        raise AssertionError("frozen norm factors differ from actual effective rows")
    factors = coefficients["factors"].to(device=device, dtype=torch.float64)
    receipt = {
        "coefficient_binding": _binding(path),
        "effective_rows_sha256": _tensor_hash(effective),
        "effective_row_norms_sha256": _tensor_hash(norms),
        "factor_sha256": _tensor_hash(factors),
        "effective_dtype": str(effective.dtype),
        "bias_exists": False,
        "base_weight_tied_to_input": head.base.weight.data_ptr() == embedding.base.weight.data_ptr(),
        "shared_delta_tied_to_input": head.shared_embed_delta.data_ptr()
        == embedding.shared_embed_delta.data_ptr(),
        "input_coordinate_sha256": _tensor_hash(embedding(coordinate_ids)),
        "base_coordinate_sha256": _tensor_hash(head.base.weight[coordinate_ids]),
        "shared_delta_sha256": _tensor_hash(head.shared_embed_delta),
    }
    return coordinate_ids, factors, receipt
