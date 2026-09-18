"""Fresh-process check of both checkpoint deltas through the native HF loader.

Run from the checkout root with PYTHONPATH=. and an authored inference YAML.
This is a mechanism check, not model-quality evidence.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.adapters.dora import load_inference_dora_adapter
from src.artifacts.checkpoint_payload import build_inference_checkpoint_payload_identity
from src.config.inference import load_infer_config
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import load_inference_embedding_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    config = load_infer_config(args.config).config
    root = Path(config.embedding_delta.path)
    identity = build_inference_checkpoint_payload_identity(root.parent)
    tensors = load_file(str(root / "special_token_embeddings.safetensors"))
    assert set(tensors) == {"input_embed_delta", "output_embed_delta"}
    qwen = load_qwen_components_from_options(QwenLoadOptions(
        base_model=config.model.base_model, dtype=config.model.dtype,
        attn_implementation=config.backend.hf.attn_implementation,
        patch_embed_linearization=config.backend.hf.patch_embed_linearization,
        load_model=True,
    ))
    load_inference_dora_adapter(config=config, qwen=qwen)
    load_inference_embedding_delta(config=config, qwen=qwen)
    model = qwen.model.to("cuda").eval()
    left = model.get_input_embeddings().shared_embed_delta
    right = model.get_output_embeddings().shared_embed_delta
    assert left is not right and left.data_ptr() != right.data_ptr()
    for key, parameter in (("input_embed_delta", left), ("output_embed_delta", right)):
        assert torch.equal(parameter.cpu(), tensors[key]), key
        assert parameter.dtype == torch.float32 and parameter.abs().sum() > 0
    assert not torch.equal(left, right)
    selected_id = model.get_input_embeddings().selection.token_ids[104]
    ids = torch.tensor([[100, selected_id, 42]], device="cuda")
    def forward():
        return model(input_ids=ids, use_cache=False).logits.detach().clone()
    with torch.inference_mode():
        reference = forward()
        assert torch.isfinite(reference).all()
        sensitivity = {}
        for name, parameter in (("input", left), ("output", right)):
            original = parameter.clone()
            parameter[104].add_(torch.linspace(-0.5, 0.5, parameter.shape[1], device="cuda"))
            changed = forward()
            sensitivity[name] = float((changed - reference).abs().max())
            assert sensitivity[name] > 0, name
            parameter.copy_(original)
            assert torch.equal(forward(), reference), name
    receipt = {
        "status": "passed", "checkpoint": str(root.parent),
        "checkpoint_identity": identity,
        "both_tensors_exact_after_native_hf_reload": True,
        "independent_storage": True, "both_nonzero_and_distinct": True,
        "delta_shape": list(left.shape), "delta_dtype": str(left.dtype),
        "forward_sensitivity_max_abs": sensitivity,
        "forward_restored_bitwise": True,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
