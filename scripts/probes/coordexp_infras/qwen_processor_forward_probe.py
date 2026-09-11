#!/usr/bin/env python3
"""Probe Qwen3-VL no-resize processor and optional forward contracts.

This is intentionally a small, local-only Wave 1B probe.  The default path
loads the processor, validates no-resize image/token expansion, and writes a
receipt.  Model forward is opt-in because the local 2B model can be expensive.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


DEFAULT_MODEL_PATH = Path(
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
DEFAULT_OUTPUT_DIR = Path("outputs/probes/coordexp_swift/qwen_processor_forward")


def _as_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _tensor_to_int_list(value: torch.Tensor | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value.detach().cpu().tolist()]


def _require_int(value: Any, name: str) -> int:
    if value is None:
        raise RuntimeError(f"processor image_processor missing {name}")
    return int(value)


def _processor_image_facts(processor: Any) -> tuple[int, int, int]:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = _require_int(getattr(image_processor, "patch_size", None), "patch_size")
    merge_size = _require_int(getattr(image_processor, "merge_size", None), "merge_size")
    temporal_patch_size = _require_int(
        getattr(image_processor, "temporal_patch_size", None),
        "temporal_patch_size",
    )
    return patch_size, merge_size, temporal_patch_size


def _make_synthetic_image(height: int, width: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256)
    return image


def _make_messages(image: Image.Image, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]


def _render_prompt(processor: Any, image: Image.Image, text: str) -> str:
    rendered = processor.apply_chat_template(
        _make_messages(image, text),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("processor.apply_chat_template(..., tokenize=False) must return str")
    return rendered


def _call_processor(processor: Any, image: Image.Image, text: str) -> dict[str, Any]:
    rendered = _render_prompt(processor, image, text)
    inputs = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
        do_resize=False,
    )
    return {"rendered_text": rendered, "inputs": dict(inputs)}


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _verify_processor_contract(
    *,
    processor: Any,
    height: int,
    width: int,
    processor_result: dict[str, Any],
) -> dict[str, Any]:
    patch_size, merge_size, temporal_patch_size = _processor_image_facts(processor)
    factor = patch_size * merge_size
    if height % factor != 0 or width % factor != 0:
        raise RuntimeError(
            f"synthetic image dimensions must be divisible by patch_size*merge_size={factor}: "
            f"height={height} width={width}"
        )

    inputs = processor_result["inputs"]
    image_grid_thw = inputs.get("image_grid_thw")
    pixel_values = inputs.get("pixel_values")
    input_ids = inputs.get("input_ids")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise RuntimeError("processor output missing tensor image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor):
        raise RuntimeError("processor output missing tensor pixel_values")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("processor output missing tensor input_ids")

    expected_grid = [1, height // patch_size, width // patch_size]
    observed_grid = [int(v) for v in image_grid_thw[0].detach().cpu().tolist()]
    if observed_grid != expected_grid:
        raise RuntimeError(f"image_grid_thw mismatch: observed={observed_grid} expected={expected_grid}")

    expected_raw_patch_rows = expected_grid[0] * expected_grid[1] * expected_grid[2]
    if int(pixel_values.shape[0]) != expected_raw_patch_rows:
        raise RuntimeError(
            "pixel_values raw patch rows mismatch: "
            f"observed={tuple(pixel_values.shape)} expected_rows={expected_raw_patch_rows}"
        )
    expected_pixel_width = 3 * temporal_patch_size * patch_size * patch_size
    if int(pixel_values.shape[1]) != expected_pixel_width:
        raise RuntimeError(
            "pixel_values feature width mismatch: "
            f"observed={tuple(pixel_values.shape)} expected_width={expected_pixel_width}"
        )

    expected_visual_tokens = expected_raw_patch_rows // (merge_size * merge_size)
    tokenizer = processor.tokenizer
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    observed_placeholder_count = int((input_ids == image_pad_id).sum().item())
    if observed_placeholder_count != expected_visual_tokens:
        raise RuntimeError(
            "expanded <|image_pad|> count mismatch: "
            f"observed={observed_placeholder_count} expected={expected_visual_tokens}"
        )

    im_end_ids = _token_ids(tokenizer, "<|im_end|>")
    im_end_newline_ids = _token_ids(tokenizer, "<|im_end|>\n")
    newline_ids = _token_ids(tokenizer, "\n")
    if len(im_end_ids) != 1:
        raise RuntimeError(f"<|im_end|> must be one token, got {im_end_ids}")
    if im_end_newline_ids != im_end_ids + newline_ids:
        raise RuntimeError(
            "<|im_end|> newline split mismatch: "
            f"im_end={im_end_ids} newline={newline_ids} combined={im_end_newline_ids}"
        )

    return {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(getattr(processor, "image_processor", None)).__name__,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "required_spatial_factor": factor,
        "height": height,
        "width": width,
        "do_resize": False,
        "expected_image_grid_thw": [expected_grid],
        "observed_image_grid_thw": _as_list(image_grid_thw),
        "expected_raw_patch_rows": expected_raw_patch_rows,
        "observed_pixel_values_shape": list(pixel_values.shape),
        "expected_pixel_values_shape": [expected_raw_patch_rows, expected_pixel_width],
        "expected_merged_visual_tokens": expected_visual_tokens,
        "image_pad_token_id": image_pad_id,
        "observed_image_pad_placeholder_count": observed_placeholder_count,
        "rendered_text_image_pad_literal_count": processor_result["rendered_text"].count("<|image_pad|>"),
        "im_end_ids": im_end_ids,
        "newline_ids": newline_ids,
        "im_end_newline_ids": im_end_newline_ids,
        "im_end_newline_split_verified": True,
        "input_ids_shape": list(input_ids.shape),
    }


def _load_qwen_model(model_path: Path, dtype: str, device: str) -> Any:
    from transformers import Qwen3VLForConditionalGeneration

    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    return Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=None,
        local_files_only=True,
    ).to(device)


def _segment_position_ids(model: Any, segment: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = segment["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    grid = segment["image_grid_thw"]
    rope_owner = getattr(model, "model", model)
    get_rope_index = getattr(rope_owner, "get_rope_index", None)
    if get_rope_index is None:
        raise RuntimeError(
            "loaded Qwen3-VL model does not expose get_rope_index on the wrapper or inner model"
        )
    mrope_position_ids, _rope_delta = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=grid,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )
    if mrope_position_ids.shape[0] != 3:
        raise RuntimeError(f"expected 3-row Qwen MRoPE positions, got {tuple(mrope_position_ids.shape)}")
    text_position_ids = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    ).view(1, 1, -1)
    return torch.cat([text_position_ids, mrope_position_ids], dim=0)


def _segment_boundaries(lengths: list[int]) -> list[int]:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + int(length))
    return boundaries


def _row_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu()
    return {
        "first": int(values[0].item()),
        "last": int(values[-1].item()),
        "min": int(values.min().item()),
        "max": int(values.max().item()),
        "unique_count": int(torch.unique(values).numel()),
    }


def _position_summaries(
    position_parts: list[torch.Tensor],
    segment_lengths: list[int],
    segment_boundaries: list[int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    row_names = ["text", "temporal", "height", "width"]
    for index, (part, length) in enumerate(zip(position_parts, segment_lengths, strict=True)):
        summaries.append(
            {
                "segment_index": index,
                "start_position": segment_boundaries[index],
                "end_position": segment_boundaries[index + 1],
                "length": int(length),
                "shape": list(part.shape),
                "rows": {
                    row_name: _row_summary(part[row_idx, 0])
                    for row_idx, row_name in enumerate(row_names)
                },
            }
        )
    return summaries


def _whole_pack_rope_contrast(
    model: Any,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    segment_boundaries: list[int],
) -> dict[str, Any]:
    rope_owner = getattr(model, "model", model)
    whole_position_ids, _rope_delta = rope_owner.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=torch.ones_like(input_ids),
    )
    start_values = [int(whole_position_ids[0, 0, start].item()) for start in segment_boundaries[:-1]]
    return {
        "executed": True,
        "whole_pack_mrope_shape": list(whole_position_ids.shape),
        "segment_start_values_from_whole_pack_mrope_text_row": start_values,
        "second_segment_text_start_continuous": bool(len(start_values) > 1 and start_values[1] != 0),
        "reason_disallowed": "whole-pack get_rope_index does not reset segment-local text positions",
    }


def _install_integrated_fa2_capture() -> tuple[dict[str, Any], Any]:
    import transformers.modeling_flash_attention_utils as flash_utils

    observed: dict[str, Any] = {
        "flash_fn_called": False,
        "flash_varlen_fn_called": False,
        "pad_fn_called": False,
        "unpad_fn_called": False,
        "lazy_import_implementations": [],
        "varlen_calls": [],
        "ordinary_flash_calls": [],
    }

    def fake_flash_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["flash_fn_called"] = True
        q = args[0]
        observed["ordinary_flash_calls"].append(
            {"q_shape": list(q.shape), "q_device": str(q.device), "q_dtype": str(q.dtype)}
        )
        return torch.zeros_like(q)

    def fake_flash_varlen_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        observed["flash_varlen_fn_called"] = True
        observed["varlen_calls"].append(
            {
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "q_device": str(q.device),
                "q_dtype": str(q.dtype),
                "cu_seqlens_q": _tensor_to_int_list(cu_seqlens_q),
                "cu_seqlens_k": _tensor_to_int_list(cu_seqlens_k),
                "max_seqlen_q": None if max_seqlen_q is None else int(max_seqlen_q),
                "max_seqlen_k": None if max_seqlen_k is None else int(max_seqlen_k),
                "flash_kwargs": {
                    key: str(value) if isinstance(value, torch.dtype) else value
                    for key, value in kwargs.items()
                },
            }
        )
        return torch.zeros_like(q)

    def fake_pad_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["pad_fn_called"] = True
        raise AssertionError("Unexpected integrated Qwen padding pad path reached")

    def fake_unpad_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        observed["unpad_fn_called"] = True
        raise AssertionError("Unexpected integrated Qwen padding unpad path reached")

    def fake_process_flash_kwargs_fn(**kwargs: Any) -> dict[str, Any]:
        return {
            "causal": bool(kwargs["is_causal"]),
            "dropout_p": float(kwargs["dropout"]),
            "softmax_scale": kwargs.get("softmax_scale"),
        }

    def fake_lazy_import_flash_attention(implementation: str | None = None) -> tuple[Any, Any]:
        observed["lazy_import_implementations"].append(implementation)
        return (
            (fake_flash_fn, fake_flash_varlen_fn, fake_pad_fn, fake_unpad_fn),
            fake_process_flash_kwargs_fn,
        )

    original_lazy_import = flash_utils.lazy_import_flash_attention
    flash_utils.lazy_import_flash_attention = fake_lazy_import_flash_attention
    return observed, original_lazy_import


def _restore_integrated_fa2_capture(original_lazy_import: Any) -> None:
    import transformers.modeling_flash_attention_utils as flash_utils

    flash_utils.lazy_import_flash_attention = original_lazy_import


def _run_model_forward(
    *,
    processor: Any,
    model_path: Path,
    dtype: str,
    device: str,
    first_segment: dict[str, Any],
    second_segment: dict[str, Any],
) -> dict[str, Any]:
    model = _load_qwen_model(model_path, dtype, device)
    model.eval()

    segments = [first_segment["inputs"], second_segment["inputs"]]
    segment_lengths = [int(segment["input_ids"].shape[1]) for segment in segments]
    boundaries = _segment_boundaries(segment_lengths)
    input_ids = torch.cat([segment["input_ids"] for segment in segments], dim=1).to(device)
    pixel_values = torch.cat([segment["pixel_values"] for segment in segments], dim=0).to(device)
    image_grid_thw = torch.cat([segment["image_grid_thw"] for segment in segments], dim=0).to(device)
    cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    max_segment_length = max(segment_lengths)

    position_parts = []
    cursor = 0
    for segment in segments:
        segment_on_device = {
            "input_ids": segment["input_ids"].to(device),
            "image_grid_thw": segment["image_grid_thw"].to(device),
        }
        position_part = _segment_position_ids(model, segment_on_device)
        if int(position_part[0, 0, 0].item()) != 0:
            raise RuntimeError("segment-local text position_ids did not reset to zero")
        cursor += int(position_part.shape[-1])
        position_parts.append(position_part)
    if cursor != int(input_ids.shape[1]):
        raise RuntimeError("packed position length mismatch")
    position_ids = torch.cat(position_parts, dim=-1).to(device)
    if list(position_ids.shape) != [4, 1, int(input_ids.shape[1])]:
        raise RuntimeError(f"position_ids must be [4, 1, seq], got {tuple(position_ids.shape)}")
    reset_values = [int(position_ids[0, 0, start].item()) for start in boundaries[:-1]]
    if reset_values != [0 for _ in boundaries[:-1]]:
        raise RuntimeError(f"text position ids did not reset at segment starts: {reset_values}")
    whole_pack_contrast = _whole_pack_rope_contrast(model, input_ids, image_grid_thw, boundaries)

    fa2_observed, original_lazy_import = _install_integrated_fa2_capture()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                labels=None,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max_segment_length,
                max_length_k=max_segment_length,
            )
    finally:
        _restore_integrated_fa2_capture(original_lazy_import)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("model forward did not return tensor logits")
    expected_logits_shape = [1, int(input_ids.shape[1]), int(model.config.text_config.vocab_size)]
    if list(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"logits shape mismatch: observed={tuple(logits.shape)} expected={expected_logits_shape}"
        )

    matching_text_calls = [
        call
        for call in fa2_observed["varlen_calls"]
        if call["cu_seqlens_q"] == boundaries
        and call["cu_seqlens_k"] == boundaries
        and call["max_seqlen_q"] == max_segment_length
        and call["max_seqlen_k"] == max_segment_length
    ]
    if not matching_text_calls:
        raise RuntimeError(
            "integrated Qwen FA2 capture did not observe a text varlen call matching "
            f"packed boundaries {boundaries}"
        )

    return {
        "executed": True,
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "labels": None,
        "use_cache": False,
        "inputs_embeds_used": False,
        "packed_segment_count": 2,
        "packed_segment_lengths": segment_lengths,
        "packed_segment_boundaries": boundaries,
        "packed_segment_start_positions": boundaries[:-1],
        "position_ids_shape": list(position_ids.shape),
        "position_ids_row_meaning": ["text", "temporal", "height", "width"],
        "position_ids_per_segment_summary": _position_summaries(
            position_parts, segment_lengths, boundaries
        ),
        "text_position_reset_values": reset_values,
        "text_reset_matches_segment_starts": True,
        "text_reset_matches_fa2_splits": reset_values == [0 for _ in boundaries[:-1]],
        "whole_pack_get_rope_index_contrast": whole_pack_contrast,
        "integrated_fa2": {
            "executed": True,
            "observed_branch": "padding_free_varlen",
            "attention_mask": None,
            "cu_seq_lens_q": boundaries,
            "cu_seq_lens_k": boundaries,
            "max_length_q": max_segment_length,
            "max_length_k": max_segment_length,
            "matching_text_varlen_call_count": len(matching_text_calls),
            "first_matching_text_varlen_call": matching_text_calls[0],
            "flash_fn_called": bool(fa2_observed["flash_fn_called"]),
            "flash_varlen_fn_called": bool(fa2_observed["flash_varlen_fn_called"]),
            "pad_fn_called": bool(fa2_observed["pad_fn_called"]),
            "unpad_fn_called": bool(fa2_observed["unpad_fn_called"]),
            "lazy_import_implementations": fa2_observed["lazy_import_implementations"],
            "total_varlen_call_count": len(fa2_observed["varlen_calls"]),
        },
        "input_ids_shape": list(input_ids.shape),
        "pixel_values_shape": list(pixel_values.shape),
        "image_grid_thw": _as_list(image_grid_thw),
        "logits_shape": list(logits.shape),
        "expected_logits_shape": expected_logits_shape,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe local Qwen3-VL no-resize processor and optional forward contract."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height", type=_positive_int, default=None)
    parser.add_argument("--width", type=_positive_int, default=None)
    parser.add_argument("--skip-model-forward", action="store_true", default=True)
    parser.add_argument(
        "--run-model-forward",
        action="store_false",
        dest="skip_model_forward",
        help="Opt in to loading the model and running a tiny two-segment packed forward.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    patch_size, merge_size, _temporal_patch_size = _processor_image_facts(processor)
    factor = patch_size * merge_size
    height = args.height or factor * 2
    width = args.width or factor * 3

    image = _make_synthetic_image(height, width)
    first_segment = _call_processor(processor, image, "Describe the synthetic image.")
    processor_contract = _verify_processor_contract(
        processor=processor,
        height=height,
        width=width,
        processor_result=first_segment,
    )

    receipt: dict[str, Any] = {
        "probe": "qwen_processor_forward_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "flash_attn": _package_version("flash_attn"),
        },
        "model_path": str(args.model_path),
        "processor_contract": processor_contract,
        "model_forward": {
            "executed": False,
            "reason": "skipped by default; pass --run-model-forward to load the model",
        },
    }

    if not args.skip_model_forward:
        second_segment = _call_processor(processor, image, "Count the visible color bands.")
        receipt["model_forward"] = _run_model_forward(
            processor=processor,
            model_path=args.model_path,
            dtype=args.dtype,
            device=args.device,
            first_segment=first_segment,
            second_segment=second_segment,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "qwen_forward_contract.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(receipt_path))


if __name__ == "__main__":
    main()
