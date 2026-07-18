#!/usr/bin/env python3
"""Real dynamic-HF versus materialized-HF parity qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config.fingerprint import sha256_json
from src.config.inference import load_infer_config
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.data import load_raw_examples
from src.inference.execution_model import (
    bind_execution_model_parity,
    resolve_execution_model,
)
from src.inference.execution_model_parity import (
    EXECUTION_MODEL_PARITY_NAME,
    build_execution_model_parity_receipt,
    compare_execution_models,
    write_execution_model_parity_receipt,
)
from src.inference.hf_backend import _load_hf_components
from src.inference.image_plan import plan_image_batch
from src.inference.prompt import build_prompt_record
from src.inference.runtime import prepare_backend_launch
from src.qwen.images import apply_logical_image_transform, rgb_image_sha256
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.tokens import DEFAULT_WRAPPER_TOKENS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()

    resolved = load_infer_config(args.config)
    config = resolved.config
    if config.backend.type != "hf" or config.adapter is None or config.embedding_delta is None:
        raise ValueError("parity probe requires an HF config with adapter and embedding delta")
    generation_fingerprint = sha256_json(config.generation.model_dump(mode="json"))
    execution_model = resolve_execution_model(
        base_model_path=config.model.base_model,
        target_dtype=config.model.dtype,
        adapter_path=config.adapter.path,
        adapter_name=config.adapter.name,
        embedding_delta_path=config.embedding_delta.path,
    )
    dynamic_launch = prepare_backend_launch(
        config,
        generation_config_fingerprint=generation_fingerprint,
    )
    dynamic_loaded = _load_hf_components(dynamic_launch)
    dynamic_qwen = dynamic_loaded.qwen
    materialized_qwen = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=str(execution_model["model_path"]),
            dtype=config.model.dtype,
            attn_implementation=config.backend.hf.attn_implementation,
            patch_embed_linearization=config.backend.hf.patch_embed_linearization,
            load_model=True,
        )
    )
    device = torch.device(args.device)
    dynamic_model = dynamic_qwen.model.to(device).eval()
    materialized_model = materialized_qwen.model.to(device).eval()

    examples = list(load_raw_examples(config.data.input_jsonl))
    raw_example = examples[args.row_index]
    image_batch = plan_image_batch(
        [raw_example],
        components=dynamic_qwen,
        processor_config=ProcessorConfig(
            do_resize=False,
            max_raw_pixels=1_000_000_000,
            max_merged_visual_tokens=1_000_000,
        ),
        row_indices=[args.row_index],
    )
    image_row = image_batch.rows[0]
    prompt_record = build_prompt_record(
        raw_example,
        TemplateConfig(
            object_field_order=config.template.object_field_order,
            object_ordering=config.template.object_ordering,
            assistant_format=config.template.assistant_format,
            prompt=TemplatePromptConfig(
                system=config.template.prompt.system,
                user=config.template.prompt.user,
            ),
        ),
        processor=dynamic_qwen.processor,
        row_index=args.row_index,
        merged_visual_tokens=image_row.merged_visual_tokens,
    )
    with Image.open(raw_example.image.path) as source_image:
        image = apply_logical_image_transform(
            source_image.convert("RGB"),
            image_row.logical_transform_id,
            example_id=raw_example.example_id,
            image_path=raw_example.image.path,
        )
    try:
        encoded = dynamic_qwen.processor(
            text=[prompt_record.chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        native_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(encoded).items()
        }
        selected_token_ids = [
            *(
                dynamic_qwen.token_identity.wrapper_token_ids[token]
                for token in DEFAULT_WRAPPER_TOKENS
            ),
            *dynamic_qwen.token_identity.coordinate_token_ids,
        ]
        comparison = compare_execution_models(
            dynamic_model=dynamic_model,
            materialized_model=materialized_model,
            native_inputs=native_inputs,
            selected_token_ids=selected_token_ids,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "repetition_penalty": config.generation.repetition_penalty,
                "eos_token_id": dynamic_qwen.token_identity.im_end_token_ids[0],
                "pad_token_id": dynamic_qwen.tokenizer.pad_token_id,
                "use_cache": True,
            },
        )
        if comparison["dynamic_generated_ids"] != comparison["materialized_generated_ids"]:
            mismatch = next(
                (
                    index
                    for index, (dynamic_id, materialized_id) in enumerate(
                        zip(
                            comparison["dynamic_generated_ids"],
                            comparison["materialized_generated_ids"],
                            strict=False,
                        )
                    )
                    if dynamic_id != materialized_id
                ),
                min(
                    len(comparison["dynamic_generated_ids"]),
                    len(comparison["materialized_generated_ids"]),
                ),
            )
            print(
                json.dumps(
                    {
                        "parity_failure": "greedy_generated_ids",
                        "first_mismatch_index": mismatch,
                        "comparison": comparison,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        fixture_identity = {
            "row_id": raw_example.example_id,
            "row_index": args.row_index,
            "input_jsonl_sha256": _sha256_file(Path(config.data.input_jsonl)),
            "prompt_ids_sha256": sha256_json(prompt_record.prompt_token_ids),
            "processor_fingerprint": sha256_json(
                dynamic_qwen.processor_identity.to_artifact_dict()
            ),
            "image_file_sha256": _sha256_file(raw_example.image.path),
            "executed_media_sha256": rgb_image_sha256(image),
            "generation_fingerprint": generation_fingerprint,
            "max_new_tokens": args.max_new_tokens,
        }
        parity = build_execution_model_parity_receipt(
            execution_model=execution_model,
            fixture_identity=fixture_identity,
            comparison=comparison,
        )
        output = (
            Path(args.output).resolve()
            if args.output
            else Path(str(execution_model["receipt_path"])).with_name(
                EXECUTION_MODEL_PARITY_NAME
            )
        )
        write_execution_model_parity_receipt(output, parity)
        qualified = bind_execution_model_parity(
            execution_model,
            parity,
            parity_path=output,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(output),
                    "composition_key": qualified["composition_key"],
                    "snapshot_fingerprint": qualified["snapshot_fingerprint"],
                    "parity_digest": parity["digest"],
                    "comparison": comparison,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        image.close()
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
