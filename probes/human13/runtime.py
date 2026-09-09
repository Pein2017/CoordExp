"""Source FP32/SDPA batch-one runtime choices for finite-panel interventions."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

def derive_hf_fp32_sdpa_batch_one_launch(launch: Any) -> Any:
    """Derive batch-one census execution without changing content identities."""

    nested = getattr(launch, "backend_options", {}).get("hf", {})
    if (
        getattr(launch, "backend", None) != "hf"
        or getattr(launch, "model_dtype", None) != "fp32"
        or nested.get("attn_implementation") != "sdpa"
    ):
        raise ValueError("HF runtime requires the Source fp32/SDPA launch")
    if is_dataclass(launch) and not isinstance(launch, type):
        return replace(cast(Any, launch), batch_size=1)
    try:
        values = vars(launch)
    except TypeError as exc:
        raise ValueError("HF runtime requires a content-bound launch record") from exc
    return SimpleNamespace(**{**values, "batch_size": 1})


def validate_hf_fp32_sdpa_batch_one(launch: Any, receipt: Any) -> dict[str, object]:
    """Validate and return the runtime identity observed by the HF receipt."""

    derive_hf_fp32_sdpa_batch_one_launch(launch)
    if getattr(launch, "batch_size", None) != 1:
        raise ValueError("HF runtime requires the Source fp32/SDPA batch-one launch")
    if getattr(receipt, "backend", None) != "hf":
        raise ValueError("HF observed backend must be exactly hf")
    settings = getattr(receipt, "effective_settings", None)
    if not isinstance(settings, Mapping):
        raise ValueError("HF observed runtime settings are missing")
    if settings.get("batch_size") != 1:
        raise ValueError("HF observed batch_size must be exactly one")
    observed_dtype = settings.get("observed_model_dtype")
    observed_names = (
        observed_dtype.get("parameter_dtype_names")
        if isinstance(observed_dtype, Mapping)
        else None
    )
    if not isinstance(observed_names, list) or observed_names != ["torch.float32"]:
        raise ValueError("HF observed runtime must be exclusively fp32")
    observed_attention = settings.get("observed_attn_implementation")
    if observed_attention != "sdpa":
        raise ValueError("HF observed runtime must use SDPA")
    validate = getattr(receipt, "validate_for_launch", None)
    if not callable(validate):
        raise ValueError("HF observed runtime receipt is untyped")
    validate(launch)
    identity = {
        "backend": receipt.backend,
        "backend_mode": receipt.backend_mode,
        "backend_version": receipt.backend_version,
        "batch_size": settings["batch_size"],
        "observed_model_dtype_names": list(observed_names),
        "observed_attn_implementation": observed_attention,
        "generation_config_fingerprint": receipt.generation_config_fingerprint,
        "model_identity": dict(receipt.model_identity),
        "tokenizer_identity": dict(receipt.tokenizer_identity),
        "processor_identity": dict(receipt.processor_identity),
    }
    return identity



def physical_image_id(example: Any) -> int | str:
    """Return the dataset image id, falling back to the internal example id."""

    metadata = getattr(example, "metadata", {})
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if isinstance(source, Mapping) and source.get("image_id") is not None:
        value = source["image_id"]
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    return str(example.example_id)


def _processor_config(config: Any) -> Any:
    from src.config.models import ProcessorConfig

    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def _template_config(config: Any) -> Any:
    from src.config.models import TemplateConfig, TemplatePromptConfig

    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def _build_requests(config: Any, frontend: Any, raw_examples: Sequence[Any]) -> list[Any]:
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record

    plans = plan_image_batch(
        list(raw_examples),
        components=frontend.qwen,
        processor_config=_processor_config(config),
        row_indices=list(range(len(raw_examples))),
    )
    by_id = {row.row_id: row for row in plans.rows}
    requests: list[Any] = []
    for index, example in enumerate(raw_examples):
        row = by_id[example.example_id]
        record = build_prompt_record(
            example,
            _template_config(config),
            processor=frontend.qwen.processor,
            row_index=index,
            merged_visual_tokens=row.merged_visual_tokens,
            object_order_seed=config.template.object_order_seed,
        )
        from src.inference.backend import DecodeRequest, GenerationPolicy

        requests.append(
            DecodeRequest(
                request_id=str(example.example_id),
                chat_text=record.chat_text,
                input_prompt_token_ids=tuple(record.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
                image_path=row.image_path,
                declared_image_width=row.declared_width,
                declared_image_height=row.declared_height,
                decoded_image_width=row.decoded_width,
                decoded_image_height=row.decoded_height,
                image_sha256=row.image_content_sha256,
                expected_image_grid_thw=tuple(row.expected_image_grid_thw),
                logical_transform_id=row.logical_transform_id,
                generation_policy=GenerationPolicy(max_new_tokens=1),
            )
        )
    return requests



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


def prepare_decision_history(components, request, route_ids):
    """Literal frozen route with one selected causal decision per target token."""
    import torch
    from src.qwen.native import NativeRequest, prepare_native_inputs, exact_history_inputs, model_device

    native = prepare_native_inputs(components.processor, [NativeRequest(
        request_id=request.request_id, chat_text=request.chat_text, image=request.image_path,
        expected_token_ids=request.expected_executed_prompt_token_ids,
        expected_image_grid=request.expected_image_grid_thw,
        expected_image_size=(request.decoded_image_width, request.decoded_image_height),
        image_sha256=request.image_sha256, logical_transform=request.logical_transform_id,
    )], device=model_device(components.model))
    prompt = native.prompt_token_ids[0]
    positions = torch.arange(len(prompt) - 1, len(prompt) - 1 + len(route_ids),
                             dtype=torch.long, device=model_device(components.model))
    inputs = exact_history_inputs(components.model, native.inputs, [(*prompt, *route_ids)],
                                  pad_token_id=0)
    inputs['logits_to_keep'] = positions
    return inputs, prompt, positions
