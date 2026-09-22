"""Model/input assembly for the fixed Source256 learning profiles."""

from pathlib import Path

from src.adapters.dora import attach_dora_adapter
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.inference.image_plan import plan_image_batch
from src.inference.prompt import build_prompt_record
from src.qwen.native import NativeRequest, prepare_native_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import attach_embedding_delta

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/source256.yaml"


def processor_config(config):
    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def template_config(config):
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system, user=config.template.prompt.user
        ),
    )


def build_request(raw, *, config, qwen, row_index=0):
    image = plan_image_batch(
        [raw], components=qwen, processor_config=processor_config(config),
        row_indices=[row_index],
    ).rows[0]
    prompt = build_prompt_record(
        raw, template_config(config), processor=qwen.processor, row_index=row_index,
        merged_visual_tokens=image.merged_visual_tokens,
        object_order_seed=config.template.object_order_seed,
    )
    request = NativeRequest(
        request_id=str(raw.example_id), chat_text=prompt.chat_text,
        image=image.image_path,
        expected_token_ids=tuple(prompt.expected_executed_prompt_token_ids),
        expected_image_grid=tuple(image.expected_image_grid_thw),
        expected_image_size=(image.decoded_width, image.decoded_height),
        image_sha256=image.image_content_sha256,
        logical_transform=image.logical_transform_id,
    )
    return request, image, prompt


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
    embedding_receipt = attach_embedding_delta(
        delta_path=config.embedding_delta.path, qwen=qwen,
        source_gate_root=config.embedding_delta.source_gate_root or Path(__file__).resolve().parents[2],
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
        },
        "effective_settings": {
            "observed_attn_implementation": getattr(qwen.model.config, "_attn_implementation", None),
            "observed_model_dtype": {
                "parameter_dtype_names": sorted({str(parameter.dtype) for parameter in qwen.model.parameters()}),
            },
        },
    }
    return qwen, descriptor
