"""Model/input assembly for the fixed Source256 learning profiles."""

from pathlib import Path

from src.adapters.dora import attach_dora_adapter, select_dora_parameters
from src.inference.inputs import plan_examples
from src.qwen.native import prepare_native_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import attach_embedding_delta

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/source256.yaml"
PACKAGED_SOURCE_GATE_ROOT = Path(__file__).resolve().parent / "configs/source-gate-root"


def bind_source256_language_dora(
    model, *, expected_tensor_count, expected_scalar_count, adapter_name="default",
):
    """Bind the fixed Source256 surface and return its frozen complement."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = select_dora_parameters(model, towers=("language",), adapter_name=adapter_name)
    if not (
        len(named) == expected_tensor_count == 588
        and sum(parameter.numel() for _, parameter in named) == expected_scalar_count == 18_006_016
        and all(
            "language_model" in name
            and not any(part in name for part in ("visual", "merger", "embed_tokens", "lm_head"))
            for name, _ in named
        )
    ):
        raise ValueError("Source256 language DoRA training surface changed")
    for _, parameter in named:
        parameter.requires_grad_(True)
    selected = {id(parameter) for _, parameter in named}
    frozen = tuple((name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in selected)
    return named, frozen


def build_request(raw, *, config, qwen, row_index=0):
    planned, = plan_examples(
        [raw], config=config, components=qwen, row_indices=[row_index],
    )
    return planned.request, planned.image, planned.prompt


def materialize(qwen, request):
    return prepare_native_inputs(
        qwen.processor, (request,), device=next(qwen.model.parameters()).device,
        record_media_identity=True,
    )


def load_policy(config, *, device):
    """Load the fixed native policy once; mode and placement are recipe choices."""
    if (config.backend.type != "hf" or config.model.dtype != "fp32"
            or config.backend.hf.attn_implementation != "sdpa"
            or config.adapter is None or config.embedding_delta is None):
        raise ValueError("Source256 requires FP32/SDPA with explicit DoRA and embedding payloads")
    qwen = load_qwen_components_from_options(QwenLoadOptions(
        base_model=config.model.base_model, dtype=config.model.dtype,
        attn_implementation=config.backend.hf.attn_implementation,
        patch_embed_linearization=config.backend.hf.patch_embed_linearization,
        load_model=True,
    ))
    adapter_receipt = attach_dora_adapter(
        qwen.model, adapter_path=config.adapter.path,
        base_model_path=qwen.base_model_path, adapter_name=config.adapter.name,
    )
    source_gate_root = (
        Path(config.embedding_delta.source_gate_root).expanduser().resolve(strict=True)
        if config.embedding_delta.source_gate_root
        else PACKAGED_SOURCE_GATE_ROOT.resolve(strict=True)
    )
    embedding_receipt = attach_embedding_delta(
        delta_path=config.embedding_delta.path,
        qwen=qwen,
        source_gate_root=source_gate_root,
    )
    qwen.model.to(device)
    qwen.model.eval()
    descriptor = {
        "schema_version": "source256_loaded_policy.v1",
        "scope": "loaded-model-identity-only",
        "model_identity": {
            "base": {"path": str(qwen.base_model_path)},
            "adapter": adapter_receipt,
            "embedding_delta": embedding_receipt,
            "embedding_source_gate_root": str(source_gate_root),
        },
        "effective_settings": {
            "observed_attn_implementation": getattr(qwen.model.config, "_attn_implementation", None),
            "observed_model_dtype": {
                "parameter_dtype_names": sorted({str(parameter.dtype) for parameter in qwen.model.parameters()}),
            },
        },
    }
    return qwen, descriptor
