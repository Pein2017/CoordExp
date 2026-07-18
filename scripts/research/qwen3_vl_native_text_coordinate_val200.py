#!/usr/bin/env python
"""Inference-only native Qwen3-VL text-coordinate val200 baseline.

This is intentionally experiment-local. It reuses Swift's Qwen loader, image
materialization, HF generation backend, and detection evaluator artifact shape,
but does not broaden the stable coord-token parser or inference schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from src.config.fingerprint import sha256_file, sha256_json
from src.config.models import ProcessorConfig
from src.data import RawExample, load_raw_examples
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.detection_categories import COCO_80_CLASS_NAMES
from src.inference.backend import (
    BackendLaunch,
    DecodeRequest,
    GenerationPolicy,
    open_backend_session,
    validate_decode_results,
)
from src.inference.hf_backend import open_hf_backend_session
from src.inference.image_plan import plan_image_batch, verify_processor_model_vision_parity
from src.qwen.runtime_loading import QwenProcessorIdentity


RAW_NAME = "gt_vs_pred.jsonl"
SCORED_NAME = "gt_vs_pred_scored.jsonl"
PROVENANCE_NAME = "gt_vs_pred_scored.jsonl.provenance.json"
TRACE_NAME = "pred_token_trace.jsonl"
DIAGNOSTICS_NAME = "parse_diagnostics.jsonl"
IMAGE_PLAN_NAME = "image_plan.jsonl"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "run_manifest.json"
SCORE_POLICY = {
    "id": "native-json-object-span-logprob-mean-v1",
    "formula": "exp(mean(logprob over native JSON object span tokens))",
    "coordinate_surface": "native_text_norm1000",
}
SCORE_POLICY_FINGERPRINT = sha256_json(SCORE_POLICY)
COCO_CLASSES = frozenset(COCO_80_CLASS_NAMES)
JSON_DECODER = json.JSONDecoder()
LEGACY_RE = re.compile(
    r"<\|object_ref_start\|>(?P<label>.*?)<\|object_ref_end\|>"
    r"<\|box_start\|>\((?P<x1>\d+),(?P<y1>\d+)\),\((?P<x2>\d+),(?P<y2>\d+)\)"
    r"<\|box_end\|>",
    re.DOTALL,
)
NATIVE_JSON_FENCE_RE = re.compile(
    r"^```json\s*(?P<payload>\[.*\])\s*```(?:<\|im_end\|>)?$",
    re.DOTALL,
)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("config must be a YAML mapping")
    _apply_generation_overrides(config, args)
    rows = list(load_raw_examples(config["data"]["input_jsonl"]))
    indices = _select_indices(rows, args.indices, args.limit)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        _merge_and_write_root(
            output_dir=output_dir,
            config=config,
            config_path=config_path,
            rows=rows,
            indices=indices,
            components=_load_native_metadata_components(config),
            output_format=args.format,
        )
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed native baseline requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)

    rank_seed = int(config["generation"]["seed"]) + rank
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    assigned = [(index, rows[index]) for index in indices if index % world_size == rank]
    print(f"rank={rank}/{world_size} assigned_rows={len(assigned)}", flush=True)
    components = _load_native_metadata_components(config)
    verify_processor_model_vision_parity(
        processor_identity=components.processor_identity,
        model_config=components.config,
    )
    generation_fingerprint = sha256_json(config["generation"])
    generation_policy = GenerationPolicy(
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        repetition_penalty=float(config["generation"]["repetition_penalty"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
    )
    launch = BackendLaunch(
        backend="hf",
        model_path=str(components.base_model_path),
        model_dtype=str(config["model"]["dtype"]),
        batch_size=int(config["generation"]["batch_size"]),
        generation_config_fingerprint=generation_fingerprint,
        backend_options={
            "hf": {
                "attn_implementation": str(config["model"]["attn_implementation"]),
                "patch_embed_linearization": "disabled",
            }
        },
    )

    rank_dir = output_dir / "shards" / f"rank-{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_rows: list[dict[str, Any]] = []
    rank_traces: list[dict[str, Any]] = []
    rank_diagnostics: list[dict[str, Any]] = []
    rank_image_plan: list[dict[str, Any]] = []
    started = time.time()
    batch_size = int(config["generation"]["batch_size"])
    session_opener = lambda backend_launch: open_hf_backend_session(
        backend_launch,
        components_loader=lambda active_launch: _load_native_backend_components(
            config,
            active_launch,
        ),
    )
    with open_backend_session(launch, opener=session_opener) as session:
        for batch_start in range(0, len(assigned), batch_size):
            batch = assigned[batch_start : batch_start + batch_size]
            examples = [example for _, example in batch]
            image_plan = plan_image_batch(
                examples,
                components=components,
                processor_config=ProcessorConfig(
                    do_resize=False,
                    max_raw_pixels=1_000_000_000,
                    max_merged_visual_tokens=1_000_000,
                ),
                row_indices=[index for index, _ in batch],
            )
            rank_image_plan.extend(row.to_artifact_dict() for row in image_plan.rows)
            image_plan_by_row_id = {row.row_id: row for row in image_plan.rows}
            requests = []
            for row_index, example in batch:
                messages = _messages(config, example, output_format=args.format)
                chat_text = _chat_text(components.processor, messages)
                input_prompt_ids = _prompt_ids(components.tokenizer, chat_text)
                plan_row = image_plan_by_row_id[example.example_id]
                executed_prompt_ids = _expand_image_placeholder(
                    components.tokenizer,
                    input_prompt_ids,
                    merged_visual_tokens=plan_row.merged_visual_tokens,
                    row_id=example.example_id,
                )
                requests.append(
                    DecodeRequest(
                        request_id=example.example_id,
                        chat_text=chat_text,
                        input_prompt_token_ids=tuple(input_prompt_ids),
                        expected_executed_prompt_token_ids=tuple(executed_prompt_ids),
                        image_path=plan_row.image_path,
                        declared_image_width=plan_row.declared_width,
                        declared_image_height=plan_row.declared_height,
                        decoded_image_width=plan_row.decoded_width,
                        decoded_image_height=plan_row.decoded_height,
                        image_sha256=plan_row.image_content_sha256,
                        expected_image_grid_thw=tuple(plan_row.expected_image_grid_thw),
                        generation_policy=generation_policy,
                    )
                )
            results = validate_decode_results(
                requests=requests,
                results=session.decode(requests),
                receipt=session.receipt,
            )
            by_id = {result.request_id: result for result in results}
            for row_index, example in batch:
                result = by_id[example.example_id]
                parsed = _parse_output(
                    result.parser_text,
                    output_format=args.format,
                    row_id=example.example_id,
                    row_index=row_index,
                    image_width=example.image.width,
                    image_height=example.image.height,
                )
                rank_rows.append(
                    _row_payload(
                        example=example,
                        row_index=row_index,
                        result=result,
                        parsed=parsed,
                        tokenizer=components.tokenizer,
                    )
                )
                rank_diagnostics.extend(parsed["diagnostics"])
                rank_traces.extend(_trace_rows(result))
                rank_traces.extend(
                    _score_trace_rows(
                        result,
                        parsed["predictions"],
                        row_id=example.example_id,
                    )
                )
            print(
                f"rank={rank} completed={min(batch_start + batch_size, len(assigned))}/{len(assigned)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    _write_jsonl(rank_dir / RAW_NAME, rank_rows)
    _write_jsonl(rank_dir / TRACE_NAME, rank_traces)
    _write_jsonl(rank_dir / DIAGNOSTICS_NAME, rank_diagnostics)
    _write_jsonl(rank_dir / IMAGE_PLAN_NAME, rank_image_plan)
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        _merge_and_write_root(
            output_dir=output_dir,
            config=config,
            config_path=config_path,
            rows=rows,
            indices=indices,
            components=components,
            output_format=args.format,
        )
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default=None, help="comma-separated source row indices")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--format", choices=("native_json", "compact_legacy"), default="native_json")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def _apply_generation_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    generation = config["generation"]
    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    generation.update(applied)
    generation.setdefault("seed", 42)
    config["invocation_overrides"] = applied


class _NativeTokenIdentity:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "required_token_count": 0,
            "coord_token_count": 0,
            "special_coordinate_surface": "absent_native_text_coordinates",
            "im_end_token_ids": [int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))],
            "tokenizer_vocab_size": int(len(self.tokenizer)),
        }


def _load_native_components(config: dict[str, Any]) -> Any:
    base_model = Path(config["model"]["base_model"]).expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    hf_config = AutoConfig.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=True
    )
    dtype = torch.bfloat16 if config["model"]["dtype"] == "bf16" else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model),
        dtype=dtype,
        attn_implementation=config["model"]["attn_implementation"],
        device_map=None,
        local_files_only=True,
    )
    vision = getattr(hf_config, "vision_config")
    processor_identity = QwenProcessorIdentity(
        processor_class=type(processor).__name__,
        tokenizer_class=type(tokenizer).__name__,
        image_processor_class=type(processor.image_processor).__name__,
        patch_size=int(processor.image_processor.patch_size),
        merge_size=int(processor.image_processor.merge_size),
        temporal_patch_size=int(processor.image_processor.temporal_patch_size),
    )
    model_identity = {
        "config_class": type(hf_config).__name__,
        "model_type": str(getattr(hf_config, "model_type", "")),
        "architectures": list(getattr(hf_config, "architectures", None) or ()),
        "text_vocab_size": int(getattr(hf_config.text_config, "vocab_size")),
        "vision_config": {
            "patch_size": int(vision.patch_size),
            "spatial_merge_size": int(vision.spatial_merge_size),
            "temporal_patch_size": int(vision.temporal_patch_size),
        },
    }
    from types import SimpleNamespace

    return SimpleNamespace(
        base_model_path=base_model,
        processor=processor,
        tokenizer=tokenizer,
        config=hf_config,
        model=model,
        model_identity=model_identity,
        processor_identity=processor_identity,
        token_identity=_NativeTokenIdentity(tokenizer),
    )


def _load_native_backend_components(config: dict[str, Any], launch: BackendLaunch) -> Any:
    components = _load_native_components(config)
    if components.base_model_path != Path(launch.model_path).expanduser().resolve():
        raise RuntimeError("native backend loader model path differs from the backend launch")
    from types import SimpleNamespace

    return SimpleNamespace(
        qwen=components,
        adapter_receipt=None,
        embedding_delta_receipt=None,
    )


def _load_native_metadata_components(config: dict[str, Any]) -> Any:
    base_model = Path(config["model"]["base_model"]).expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    hf_config = AutoConfig.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=True
    )
    vision = getattr(hf_config, "vision_config")
    from types import SimpleNamespace

    return SimpleNamespace(
        base_model_path=base_model,
        processor=processor,
        tokenizer=tokenizer,
        config=hf_config,
        model_identity={
            "config_class": type(hf_config).__name__,
            "model_type": str(getattr(hf_config, "model_type", "")),
            "architectures": list(getattr(hf_config, "architectures", None) or ()),
            "text_vocab_size": int(getattr(hf_config.text_config, "vocab_size")),
            "vision_config": {
                "patch_size": int(vision.patch_size),
                "spatial_merge_size": int(vision.spatial_merge_size),
                "temporal_patch_size": int(vision.temporal_patch_size),
            },
        },
        processor_identity=QwenProcessorIdentity(
            processor_class=type(processor).__name__,
            tokenizer_class=type(tokenizer).__name__,
            image_processor_class=type(processor.image_processor).__name__,
            patch_size=int(processor.image_processor.patch_size),
            merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=int(processor.image_processor.temporal_patch_size),
        ),
        token_identity=_NativeTokenIdentity(tokenizer),
    )


def _select_indices(rows: list[RawExample], raw_indices: str | None, limit: int | None) -> list[int]:
    if raw_indices:
        indices = [int(value) for value in raw_indices.split(",") if value.strip()]
    else:
        indices = list(range(len(rows)))
    if limit is not None:
        indices = indices[:limit]
    if not indices or min(indices) < 0 or max(indices) >= len(rows) or len(set(indices)) != len(indices):
        raise ValueError("selected row indices must be unique and within the input range")
    return indices


def _messages(config: dict[str, Any], example: RawExample, *, output_format: str) -> list[dict[str, Any]]:
    user_text = _render_user_prompt(config, output_format=output_format)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(example.image.path)},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def _render_user_prompt(config: dict[str, Any], *, output_format: str) -> str:
    prompt_config = config["prompt"]
    if prompt_config.get("template") != "pretrained_processor_chat":
        raise ValueError("native baseline requires prompt.template=pretrained_processor_chat")
    if str(prompt_config.get("system") or "").strip():
        raise ValueError("Qwen3-VL's pretrained template has no default system message")
    ontology = list(config["ontology"]["classes"])
    if ontology != list(COCO_80_CLASS_NAMES):
        raise ValueError("ontology.classes must exactly match the canonical ordered COCO-80 class list")
    user_text = str(prompt_config["user"]).strip().replace(
        "{coco80_classes}", ", ".join(ontology)
    )
    if "{coco80_classes}" in user_text:
        raise ValueError("failed to render the COCO-80 class list into the user prompt")
    if output_format == "compact_legacy":
        user_text = (
            "Locate every clearly visible COCO-80 object instance in the image. Return only concatenated rows "
            "with exactly this pattern: <|object_ref_start|>label<|object_ref_end|>"
            "<|box_start|>(x1,y1),(x2,y2)<|box_end|>. Use integer normalized coordinates from 0 to 1000. "
            "Use only these COCO-80 labels: " + ", ".join(COCO_80_CLASS_NAMES) + "."
        )
    return user_text


def _parse_output(text: str, *, output_format: str, row_id: str, row_index: int, image_width: int, image_height: int) -> dict[str, Any]:
    if output_format == "native_json":
        return _parse_native_json(text, row_id=row_id, row_index=row_index, image_width=image_width, image_height=image_height)
    return _parse_compact_legacy(text, row_id=row_id, row_index=row_index, image_width=image_width, image_height=image_height)


def _parse_compact_legacy(text: str, *, row_id: str, row_index: int, image_width: int, image_height: int) -> dict[str, Any]:
    stripped = text.strip()
    matches = list(LEGACY_RE.finditer(stripped))
    diagnostics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    cursor = 0
    for order, match in enumerate(matches):
        if stripped[cursor:match.start()].strip():
            diagnostics.append(_diag(row_id, row_index, "format", "unmatched_legacy_text", order=order))
        cursor = match.end()
        label = match.group("label").strip()
        bbox = [int(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        span_id = f"{row_id}:legacy-span-{order}"
        if label not in COCO_CLASSES:
            dropped.append({"object_span_id": span_id, "reason": "invalid_coco_label", "label": label})
            diagnostics.append(_diag(row_id, row_index, "object", "invalid_coco_label", order=order, label=label))
            continue
        try:
            pixel_bbox = coord_bins_to_pixel_xyxy(bbox, image_width=image_width, image_height=image_height, field=f"pred[{order}].bbox")
        except Exception as exc:
            dropped.append({"object_span_id": span_id, "reason": "invalid_geometry", "bbox": bbox})
            diagnostics.append(_diag(row_id, row_index, "object", "invalid_geometry", order=order, detail=str(exc)))
            continue
        predictions.append({
            "object_span_id": span_id,
            "description": label,
            "label": label,
            "bbox": list(pixel_bbox),
            "bbox_2d": bbox,
            "bbox_format": "xyxy",
            "coord_bins": bbox,
            "generated_order": order,
            "raw_span_text": match.group(0),
        })
    if stripped[cursor:].strip():
        diagnostics.append(_diag(row_id, row_index, "format", "unmatched_legacy_text"))
    if not matches:
        diagnostics.append(_diag(row_id, row_index, "format", "no_legacy_rows"))
    if predictions and dropped:
        status = "accepted_with_drops"
    elif predictions:
        status = "accepted" if not diagnostics else "accepted_with_unmatched_text"
    elif dropped:
        status = "all_spans_dropped"
    else:
        status = "empty"
    return {"predictions": predictions, "dropped": dropped, "status": status, "metric_bearing": bool(predictions) and status in {"accepted", "accepted_with_drops"}, "diagnostics": diagnostics}


def _chat_text(processor: Any, messages: list[dict[str, Any]]) -> str:
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(chat_text, str) or not chat_text:
        raise ValueError("native Qwen chat template must return non-empty text")
    return chat_text


def _prompt_ids(tokenizer: Any, chat_text: str) -> list[int]:
    encoded = tokenizer(chat_text, add_special_tokens=False)
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, list) and len(input_ids) == 1 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("native Qwen tokenizer must return non-empty input_ids")
    return [int(value) for value in input_ids]


def _expand_image_placeholder(
    tokenizer: Any,
    input_prompt_ids: list[int],
    *,
    merged_visual_tokens: int,
    row_id: str,
) -> list[int]:
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        raise ValueError("native Qwen tokenizer is missing <|image_pad|>")
    image_token_id = int(image_token_id)
    image_indices = [
        index for index, token_id in enumerate(input_prompt_ids) if token_id == image_token_id
    ]
    if len(image_indices) != 1:
        raise ValueError(
            f"native Qwen prompt for {row_id} must contain exactly one image placeholder"
        )
    image_index = image_indices[0]
    return [
        *input_prompt_ids[:image_index],
        *([image_token_id] * int(merged_visual_tokens)),
        *input_prompt_ids[image_index + 1 :],
    ]


def _parse_native_json(text: str, *, row_id: str, row_index: int, image_width: int, image_height: int) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    source_text = text.strip()
    stripped = source_text
    payload_offset = 0
    envelope = "bare_json"
    fenced = NATIVE_JSON_FENCE_RE.fullmatch(stripped)
    if fenced is not None:
        stripped = fenced.group("payload").strip()
        payload_offset = source_text.find(stripped)
        envelope = "pretrained_json_fence"
    elif stripped.endswith("<|im_end|>"):
        stripped = stripped[: -len("<|im_end|>")].rstrip()
    status = "accepted"
    if not stripped.startswith("["):
        diagnostics.append(_diag(row_id, row_index, "json", "response_does_not_start_with_array"))
        return {"predictions": [], "dropped": [], "status": "invalid_json", "metric_bearing": False, "diagnostics": diagnostics}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        diagnostics.append(_diag(row_id, row_index, "json", "invalid_json", detail=str(exc)))
        return {"predictions": [], "dropped": [], "status": "invalid_json", "metric_bearing": False, "diagnostics": diagnostics}
    if not isinstance(parsed, list):
        diagnostics.append(_diag(row_id, row_index, "json", "top_level_not_array"))
        return {"predictions": [], "dropped": [], "status": "invalid_json", "metric_bearing": False, "diagnostics": diagnostics}
    cursor = stripped.find("[") + 1
    dropped: list[dict[str, Any]] = []
    for order, item in enumerate(parsed):
        while cursor < len(stripped) and (stripped[cursor].isspace() or stripped[cursor] == ","):
            cursor += 1
        start = cursor
        try:
            raw_item, end = JSON_DECODER.raw_decode(stripped, cursor)
        except json.JSONDecodeError:
            raw_item, end = item, cursor
        cursor = end
        object_span_id = f"{row_id}:native-span-{order}"
        if not isinstance(raw_item, dict) or set(raw_item) != {"bbox_2d", "label"}:
            dropped.append({"object_span_id": object_span_id, "reason": "object_schema"})
            diagnostics.append(_diag(row_id, row_index, "object", "object_schema", order=order))
            continue
        label = raw_item["label"]
        bbox = raw_item["bbox_2d"]
        if not isinstance(label, str) or label not in COCO_CLASSES:
            dropped.append({"object_span_id": object_span_id, "reason": "invalid_coco_label", "label": label})
            diagnostics.append(_diag(row_id, row_index, "object", "invalid_coco_label", order=order, label=label))
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
            dropped.append({"object_span_id": object_span_id, "reason": "invalid_norm1000_bbox", "bbox_2d": bbox})
            diagnostics.append(_diag(row_id, row_index, "object", "invalid_norm1000_bbox", order=order))
            continue
        try:
            pixel_bbox = coord_bins_to_pixel_xyxy(bbox, image_width=image_width, image_height=image_height, field=f"pred[{order}].bbox_2d")
        except Exception as exc:
            dropped.append({"object_span_id": object_span_id, "reason": "invalid_geometry", "bbox_2d": bbox})
            diagnostics.append(_diag(row_id, row_index, "object", "invalid_geometry", order=order, detail=str(exc)))
            continue
        predictions.append(
            {
                "object_span_id": object_span_id,
                "description": label,
                "label": label,
                "bbox": list(pixel_bbox),
                "bbox_2d": list(bbox),
                "bbox_format": "xyxy",
                "coord_bins": list(bbox),
                "generated_order": order,
                "raw_item": raw_item,
                "raw_span_text": stripped[start:end],
                "raw_char_span": [payload_offset + start, payload_offset + end],
            }
        )
    if predictions and dropped:
        status = "accepted_with_drops"
    elif not predictions and dropped:
        status = "all_spans_dropped"
    elif not predictions:
        status = "empty"
    return {
        "predictions": predictions,
        "dropped": dropped,
        "status": status,
        "metric_bearing": status.startswith("accepted"),
        "diagnostics": diagnostics,
        "response_envelope": envelope,
    }


def _row_payload(*, example: RawExample, row_index: int, result: Any, parsed: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    predictions = parsed["predictions"]
    for prediction in predictions:
        _attach_score(prediction, result, tokenizer)
    return {
        "row_id": example.example_id,
        "row_index": row_index,
        "example_id": example.example_id,
        "image_path": str(example.image.path),
        "image_width": example.image.width,
        "image_height": example.image.height,
        "image_id": int(example.metadata.get("image_id", example.example_id.split("_")[-1])),
        "gt": [obj.to_artifact_dict() for obj in example.objects],
        "pred": predictions,
        "raw_decode_text": result.raw_generated_text,
        "decode_stop_reason": result.stop_reason,
        "parser_id": "native-qwen3-vl-json-v2",
        "parser_policy": "qwen3vl_pretrained_fenced_or_bare_json_array_bbox_2d_norm1000_strict_keys",
        "response_envelope": parsed.get("response_envelope"),
        "metric_bearing": parsed["metric_bearing"],
        "parse_status": parsed["status"],
        "valid_prediction_count": len(predictions),
        "dropped_prediction_count": len(parsed["dropped"]),
        "dropped_predictions": parsed["dropped"],
        "token_trace": [_trace_dict(item) for item in result.token_trace],
    }


def _attach_score(prediction: dict[str, Any], result: Any, tokenizer: Any) -> None:
    span = _locate_token_span(result, prediction, tokenizer)
    if span is None:
        prediction["score_error"] = "native_json_span_token_alignment_missing"
        return
    start, end = span
    selected = [item for item in result.token_trace[start:end] if not item.is_pad and not item.is_stop and item.logprob is not None]
    if not selected:
        prediction["score_error"] = "native_json_span_has_no_scored_tokens"
        return
    logprobs = [float(item.logprob) for item in selected]
    score = math.exp(sum(logprobs) / len(logprobs))
    prediction["score"] = score
    prediction["pred_score_version"] = 1
    prediction["pred_score_source"] = {
        "kind": "token_trace_selected_logprob_mean",
        "row_id": result.request_id,
        "object_span_id": prediction["object_span_id"],
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "generated_step_indices": [item.step_index for item in selected],
        "token_ids": [item.token_id for item in selected],
        "token_text": [item.token_text for item in selected],
        "selected_logprobs": logprobs,
        "selected_count": len(selected),
    }


def _locate_token_span(result: Any, prediction: dict[str, Any], tokenizer: Any) -> tuple[int, int] | None:
    raw_char_span = prediction.get("raw_char_span")
    if (
        isinstance(raw_char_span, list)
        and len(raw_char_span) == 2
        and all(isinstance(value, int) for value in raw_char_span)
    ):
        char_start, char_end = raw_char_span
        offset = 0
        token_start = None
        token_end = None
        for index, item in enumerate(result.token_trace):
            token_text = str(item.token_text)
            next_offset = offset + len(token_text)
            if token_start is None and next_offset > char_start:
                token_start = index
            if offset < char_end:
                token_end = index + 1
            offset = next_offset
        if token_start is not None and token_end is not None and token_start < token_end:
            return token_start, token_end

    encoded = tokenizer(prediction["raw_span_text"], add_special_tokens=False)
    needle = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if isinstance(needle, list) and needle and isinstance(needle[0], list):
        needle = needle[0]
    if not isinstance(needle, list) or not needle:
        return None
    needle = [int(value) for value in needle]
    token_ids = [int(value) for value in result.generated_token_ids]
    for start in range(0, len(token_ids) - len(needle) + 1):
        if token_ids[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def _trace_dict(item: Any) -> dict[str, Any]:
    return {
        "generated_step_index": item.step_index,
        "token_id": item.token_id,
        "token_text": item.token_text,
        "logprob": item.logprob,
        "is_stop": item.is_stop,
        "is_pad": item.is_pad,
        "backend": item.backend,
        "backend_mode": item.backend_mode,
        "response_family": item.response_family,
    }


def _trace_rows(result: Any) -> list[dict[str, Any]]:
    return [{"trace_type": "generated_token", "row_id": result.request_id, **_trace_dict(item)} for item in result.token_trace]


def _score_trace_rows(result: Any, predictions: list[dict[str, Any]], *, row_id: str) -> list[dict[str, Any]]:
    return [
        {"trace_type": "native_score_status", "row_id": row_id, "object_span_id": pred["object_span_id"], "score": pred.get("score"), "score_error": pred.get("score_error")}
        for pred in predictions
    ]


def _merge_and_write_root(*, output_dir: Path, config: dict[str, Any], config_path: Path, rows: list[RawExample], indices: list[int], components: Any, output_format: str) -> None:
    shard_rows = []
    traces = []
    diagnostics = []
    image_plan = []
    for shard in sorted((output_dir / "shards").glob("rank-*/")):
        shard_rows.extend(_read_jsonl(shard / RAW_NAME))
        traces.extend(_read_jsonl(shard / TRACE_NAME))
        diagnostics.extend(_read_jsonl(shard / DIAGNOSTICS_NAME))
        image_plan.extend(_read_jsonl(shard / IMAGE_PLAN_NAME))
    shard_rows.sort(key=lambda row: int(row["row_index"]))
    traces.sort(key=lambda row: (str(row.get("row_id")), int(row.get("generated_step_index", -1))))
    diagnostics.sort(key=lambda row: (int(row.get("row_index", -1)), str(row.get("diagnostic_type"))))
    image_plan.sort(key=lambda row: int(row["row_index"]))
    scored_rows = []
    for row in shard_rows:
        pred = []
        for item in row["pred"]:
            if "score" in item and "pred_score_source" in item:
                pred.append(item)
        scored_rows.append({key: row[key] for key in ("row_id", "row_index", "example_id", "image_path", "image_width", "image_height", "image_id", "gt") } | {"pred": pred})
    _write_jsonl(output_dir / RAW_NAME, shard_rows)
    _write_jsonl(output_dir / SCORED_NAME, scored_rows)
    _write_jsonl(output_dir / TRACE_NAME, traces)
    _write_jsonl(output_dir / DIAGNOSTICS_NAME, diagnostics)
    _write_jsonl(output_dir / IMAGE_PLAN_NAME, image_plan)
    resolved = dict(config)
    resolved["resolved"] = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_row_count": len(indices),
        "selected_row_indices": indices,
        "model_identity": _model_identity(components),
        "processor_identity": components.processor_identity.to_artifact_dict(),
        "tokenizer_identity": components.token_identity.to_artifact_dict(),
        "coordinate_surface": "native_text_norm1000" if output_format == "native_json" else "native_legacy_norm1000",
        "output_format": output_format,
        "adapter": None,
        "embedding_delta": None,
        "backend": "hf",
        "decode": "sampling" if float(config["generation"]["temperature"]) > 0.0 else "greedy",
        "chat_template": "pretrained_processor_chat",
        "rendered_user_prompt": _render_user_prompt(config, output_format=output_format),
    }
    _write_json(output_dir / "resolved_config.json", resolved)
    (output_dir / "resolved_config.path").write_text(str(output_dir / "resolved_config.json") + "\n", encoding="utf-8")
    raw_sha = sha256_file(output_dir / RAW_NAME)
    scored_sha = sha256_file(output_dir / SCORED_NAME)
    row_ids = [str(row["row_id"]) for row in scored_rows]
    row_ids_sha256 = hashlib.sha256(json.dumps(row_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    provenance = {
        "artifact_schema_version": 1,
        "raw_artifact": {"path": RAW_NAME, "sha256": raw_sha},
        "scored_artifact": {"path": SCORED_NAME, "sha256": scored_sha},
        "detection_template_id": f"qwen3-vl-{output_format}-v1",
        "prompt_policy_fingerprint": sha256_json(
            {
                "chat_template": "pretrained_processor_chat",
                "rendered_user_prompt": _render_user_prompt(config, output_format=output_format),
            }
        ),
        "decode_policy_fingerprint": sha256_json(config["generation"]),
        "generation_config_fingerprint": sha256_json(config["generation"]),
        "generation_policy": config["generation"],
        "parallelism": {"world_size": int(os.environ.get("WORLD_SIZE", "1")), "rows": len(indices)},
        "model_identity": _model_identity(components),
        "model_identity_fingerprint": sha256_json(_model_identity(components)),
        "processor_identity": components.processor_identity.to_artifact_dict(),
        "processor_identity_fingerprint": sha256_json(components.processor_identity.to_artifact_dict()),
        "tokenizer_identity": components.token_identity.to_artifact_dict(),
        "adapter_identity": None,
        "embedding_delta_identity": None,
        "template_identity": {
            "chat_template": "pretrained_processor_chat",
            "format": output_format,
            "coordinate_scale": 1000,
        },
        "parser_policy": "qwen3vl_pretrained_fenced_or_bare_json_array_bbox_2d_norm1000_strict_keys" if output_format == "native_json" else "native_legacy_wrapped_xyxy_norm1000",
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
        "row_binding": {"row_count": len(row_ids), "row_ids_sha256": row_ids_sha256},
    }
    _write_json(output_dir / PROVENANCE_NAME, provenance)
    parser_counts = Counter(str(row["parse_status"]) for row in shard_rows)
    stop_counts = Counter(str(row["decode_stop_reason"]) for row in shard_rows)
    summary = {
        "terminal_status": "completed",
        "row_count": len(shard_rows),
        "raw_row_count": len(shard_rows),
        "scored_row_count": len(scored_rows),
        "gt_object_count": sum(len(row["gt"]) for row in shard_rows),
        "raw_prediction_count": sum(len(row["pred"]) for row in shard_rows),
        "scored_prediction_count": sum(len(row["pred"]) for row in scored_rows),
        "parser_status_counts": dict(sorted(parser_counts.items())),
        "decode_stop_reason_counts": dict(sorted(stop_counts.items())),
        "invalid_output_count": sum(1 for row in shard_rows if row["parse_status"] == "invalid_json"),
        "length_truncation_count": sum(1 for row in shard_rows if row["decode_stop_reason"] == "length"),
        "natural_termination_count": sum(1 for row in shard_rows if row["decode_stop_reason"] == "im_end"),
        "strict_json_row_count": sum(1 for row in shard_rows if row["metric_bearing"]),
        "score_alignment_missing_count": sum(1 for row in shard_rows for pred in row["pred"] if "score" not in pred),
        "benchmark_eligible": bool(len(shard_rows) == len(indices) and len(indices) == len(rows)),
        "coordinate_surface": "native_text_norm1000" if output_format == "native_json" else "native_legacy_norm1000",
        "model_family": "original_qwen3_vl_2b_instruct_base_only",
    }
    _write_json(output_dir / SUMMARY_NAME, summary)
    manifest = {
        "artifact_schema_version": 1,
        "artifacts": {"gt_vs_pred": RAW_NAME, "gt_vs_pred_scored": SCORED_NAME, "provenance": PROVENANCE_NAME, "trace": TRACE_NAME, "diagnostics": DIAGNOSTICS_NAME, "image_plan": IMAGE_PLAN_NAME, "summary": SUMMARY_NAME},
        "terminal_status": "completed",
        "benchmark_eligible": summary["benchmark_eligible"],
        "model_identity": provenance["model_identity"],
        "coordinate_surface": "native_text_norm1000" if output_format == "native_json" else "native_legacy_norm1000",
        "parser_policy": provenance["parser_policy"],
        "score_policy_fingerprint": SCORE_POLICY_FINGERPRINT,
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)


def _model_identity(components: Any) -> dict[str, Any]:
    model_identity = components.model_identity
    if hasattr(model_identity, "to_artifact_dict"):
        model_identity = model_identity.to_artifact_dict()
    return {"family": "base-only", "base": {"path": str(components.base_model_path), "model": model_identity}, "adapter": None, "embedding_delta": None}


def _diag(row_id: str, row_index: int, kind: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"row_id": row_id, "row_index": row_index, "diagnostic_type": kind, "reason": reason, **extra}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
