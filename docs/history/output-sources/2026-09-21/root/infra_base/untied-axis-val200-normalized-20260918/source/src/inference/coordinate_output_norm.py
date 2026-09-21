"""Output-only coordinate row-norm policy, matching the natural-readout probe."""

import torch
from transformers import LogitsProcessor

from src.common.errors import RuntimeContractError
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead


class CoordinateOutputNorm(LogitsProcessor):
    def __init__(self, model, tokenizer):
        head = model.get_output_embeddings()
        if not isinstance(head, SelectedDeltaOutputHead) or head.bias is not None:
            raise RuntimeContractError(
                "coordinate norm policy requires a bias-free selected-delta output head",
                code="hf_backend.coordinate_output_norm_head",
            )
        ids = [tokenizer.convert_tokens_to_ids(f"<|coord_{i}|>") for i in range(1000)]
        selected = head.selected_token_ids.tolist()
        if len(set(ids)) != 1000 or any(i not in selected for i in ids):
            raise RuntimeContractError(
                "coordinate norm policy requires all 1000 coordinate tokens in the output delta",
                code="hf_backend.coordinate_output_norm_tokens",
            )
        self.ids = torch.tensor(ids, device=head.weight.device)
        positions = torch.tensor([selected.index(i) for i in ids], device=head.weight.device)
        with torch.no_grad():
            # Match the reference: compose base + FP32 delta, then measure in FP64.
            rows = head.base.weight[self.ids].detach() + head.shared_embed_delta[positions].detach()
            norms = rows.double().norm(dim=1)
            if not torch.isfinite(norms).all() or not (norms > 0).all():
                raise RuntimeContractError(
                    "coordinate output norms must be finite and positive",
                    code="hf_backend.coordinate_output_norm_values",
                )
            self.factors = norms.median() / norms

    def __call__(self, input_ids, scores):
        transformed = scores.clone()
        transformed[:, self.ids] = (scores[:, self.ids].double() * self.factors).to(scores.dtype)
        return transformed
