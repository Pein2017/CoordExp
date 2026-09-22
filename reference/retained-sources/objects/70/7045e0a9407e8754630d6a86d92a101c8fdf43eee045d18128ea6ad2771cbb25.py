"""Concrete Source/overfit model attachment and native request preparation."""
from __future__ import annotations
from types import SimpleNamespace
from src.config.inference import InferConfig
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig

def _processor_config(config: InferConfig) -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )

def _template_config(config: InferConfig) -> TemplateConfig:
    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def load_source_components(launch):
    """Retain one public model reference and share it with strict HF decoding."""
    from src.adapters.dora import attach_dora_adapter
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    from src.qwen.special_token_embeddings import attach_embedding_delta

    options = launch.backend_options['hf']
    qwen = load_qwen_components_from_options(QwenLoadOptions(
        base_model=launch.model_path, dtype=launch.model_dtype,
        attn_implementation=options['attn_implementation'],
        patch_embed_linearization=options['patch_embed_linearization'], load_model=True,
    ))
    adapter = launch.adapter
    embedding = launch.embedding_delta
    return SimpleNamespace(
        qwen=qwen,
        adapter_receipt=attach_dora_adapter(
            qwen.model, adapter_path=adapter['path'], base_model_path=qwen.base_model_path,
            adapter_name=adapter['name'],
        ),
        embedding_delta_receipt=attach_embedding_delta(
            delta_path=embedding['path'], qwen=qwen, source_gate_root=embedding['source_gate_root'],
        ),
    )



def materialize_request(components, request):
    from src.qwen.native import NativeRequest, prepare_native_inputs, model_device

    native = prepare_native_inputs(components.processor, [NativeRequest(
        request_id=request.request_id, chat_text=request.chat_text, image=request.image_path,
        expected_token_ids=request.expected_executed_prompt_token_ids,
        expected_image_grid=request.expected_image_grid_thw,
        expected_image_size=(request.decoded_image_width, request.decoded_image_height),
        image_sha256=request.image_sha256, logical_transform=request.logical_transform_id,
    )], device="cpu" if components.model is None else model_device(components.model), record_media_identity=True)
    return native.inputs, native.prompt_token_ids, native.image_grids, native.media_sha256
