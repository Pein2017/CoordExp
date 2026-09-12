"""Concrete Source/overfit model attachment and native request preparation."""
from __future__ import annotations
from types import SimpleNamespace


def input_source_hashes():
    """Fresh execution dependencies; historical receipts remain unchanged."""
    from pathlib import Path
    from src.config.fingerprint import sha256_file

    root = Path(__file__).resolve().parents[2]
    return {path: sha256_file(root / path) for path in (
        "probes/logit_lens/runtime.py", "probes/logit_lens/base.py",
        "probes/logit_lens/causal.py", "src/inference/inputs.py",
        "src/inference/prompt.py", "src/inference/image_plan.py",
        "src/qwen/encoding.py", "src/qwen/images.py", "src/qwen/native.py",
        "src/templates/renderer.py",
    )}


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
