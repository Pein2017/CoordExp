from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch

from src.common.paths import resolve_image_path_strict
from src.common.semantic_desc import normalize_desc
from src.common.prediction_parsing import extract_special_tokens
from src.common.prediction_parsing import load_prediction_dict
from src.config.prompts import get_template_prompts
from src.config.schema import CoordTokensConfig
from src.coord_tokens.template_adapter import apply_coord_template_adapter
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine
from src.trainers.stage2_rollout_aligned import (
    _ensure_system_prompt_message,
    _strip_trailing_assistant_turns_for_rollout,
)

_IM_END = "<|im_end|>"


@dataclass(frozen=True)
class Stage2ParitySample:
    line_idx: int
    image: Optional[str]
    messages: List[Dict[str, Any]]
    assistant_payload: Optional[str]
    objects: Any
    width: int
    height: int


@dataclass(frozen=True)
class RolloutTextResult:
    text: str
    raw_output_json: Optional[Dict[str, Any]]
    pred_count: int
    errors: Tuple[str, ...]


def build_stage2_vllm_sample(
    record: Mapping[str, Any],
    *,
    line_idx: int,
    prompt_variant: str,
    object_field_order: Literal["desc_first", "geometry_first"],
) -> Stage2ParitySample:
    system_prompt, user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    builder = JSONLinesBuilder(
        user_prompt=user_prompt,
        emit_norm="norm1000",
        coord_tokens_enabled=True,
        object_field_order=object_field_order,
    )
    merged = builder.build(record)
    messages_raw = merged.get("messages")
    if not isinstance(messages_raw, list):
        raise ValueError("JSONLinesBuilder output must contain messages as a list")
    messages = _strip_trailing_assistant_turns_for_rollout(messages_raw)
    messages = _ensure_system_prompt_message(messages, system_prompt)
    image = None
    images_raw = record.get("images")
    if isinstance(images_raw, list) and images_raw:
        image = str(images_raw[0])
    elif isinstance(record.get("image"), str):
        image = str(record.get("image"))
    return Stage2ParitySample(
        line_idx=int(line_idx),
        image=image,
        messages=[dict(msg) for msg in messages],
        assistant_payload=(
            str(merged["assistant_payload"])
            if merged.get("assistant_payload") is not None
            else None
        ),
        objects=merged.get("objects"),
        width=int(record.get("width") or 0),
        height=int(record.get("height") or 0),
    )


def init_stage2_style_vllm_engine(
    *,
    model_dir: str,
    system_prompt: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    enforce_eager: bool,
) -> Any:
    from swift.llm import VllmEngine, get_model_tokenizer
    from swift.llm.template import get_template

    _, processor = get_model_tokenizer(
        model_dir,
        torch_dtype=torch.bfloat16,
        load_model=False,
        download_model=False,
    )
    template_type = (
        getattr(getattr(processor, "model_meta", None), "template", None)
        or "qwen3_vl"
    )
    template = get_template(
        template_type=template_type,
        processor=processor,
        default_system=system_prompt,
        max_length=max_model_len,
        truncation_strategy="delete",
    )
    coord_cfg = CoordTokensConfig(enabled=True, skip_bbox_norm=True)
    apply_coord_template_adapter(template, coord_cfg)
    template.set_mode("vllm")
    engine = VllmEngine(
        model_dir,
        torch_dtype=torch.bfloat16,
        template=template,
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        enforce_eager=bool(enforce_eager),
        disable_custom_all_reduce=True,
        enable_lora=False,
        enable_prefix_caching=True,
        engine_kwargs={"mm_processor_kwargs": {"do_resize": False}},
    )
    return engine


def run_stage2_style_vllm(
    *,
    engine: Any,
    samples: Sequence[Stage2ParitySample],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    batch_size: int,
) -> List[RolloutTextResult]:
    from swift.llm import InferRequest, RequestConfig

    request_config = RequestConfig(
        n=1,
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=-1,
        repetition_penalty=float(repetition_penalty),
        stop=[_IM_END],
        return_details=True,
        seed=int(seed),
    )
    infer_requests = [InferRequest(messages=sample.messages) for sample in samples]
    results: List[RolloutTextResult] = []
    step = max(1, int(batch_size))
    for start in range(0, len(infer_requests), step):
        outputs = engine.infer(
            infer_requests[start : start + step],
            request_config=request_config,
            use_tqdm=False,
        )
        if len(outputs) != len(infer_requests[start : start + step]):
            raise RuntimeError(
                "VllmEngine returned {} outputs for {} requests".format(
                    len(outputs), len(infer_requests[start : start + step])
                )
            )
        for output in outputs:
            text = ""
            if isinstance(output, Exception):
                results.append(
                    RolloutTextResult(
                        text=f"<exception: {type(output).__name__}>",
                        raw_output_json=None,
                        pred_count=0,
                        errors=(type(output).__name__,),
                    )
                )
                continue
            try:
                text = str(output.choices[0].message.content or "")
            except Exception:
                text = ""
            parsed = load_prediction_dict(text)
            objects = (
                list(parsed.get("objects") or [])
                if isinstance(parsed, Mapping)
                else []
            )
            errors: List[str] = []
            if not objects:
                errors.append("empty_pred")
            results.append(
                RolloutTextResult(
                    text=text,
                    raw_output_json=(dict(parsed) if isinstance(parsed, Mapping) else None),
                    pred_count=int(len(objects)),
                    errors=tuple(errors),
                )
            )
    return results


def load_records_by_indices(
    jsonl_path: Path, indices: Sequence[int]
) -> List[Tuple[int, Dict[str, Any]]]:
    wanted = {int(idx) for idx in indices}
    if not wanted:
        return []
    max_idx = max(wanted)
    found: List[Tuple[int, Dict[str, Any]]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx > max_idx:
                break
            if line_idx not in wanted:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                found.append((int(line_idx), obj))
    found.sort(key=lambda item: item[0])
    return found


def collect_stage2_parity_gt_vs_pred(
    *,
    jsonl_path: Path,
    records: Sequence[Mapping[str, Any]],
    root_image_dir: Path,
    checkpoint_path: Path,
    prompt_variant: str,
    object_field_order: Literal["desc_first", "geometry_first"],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    enforce_eager: bool,
    seed: int,
    out_path: Path,
    pred_token_trace_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_token_trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["ROOT_IMAGE_DIR"] = str(root_image_dir)

    helper_cfg = InferenceConfig(
        gt_jsonl=str(jsonl_path),
        model_checkpoint=str(checkpoint_path),
        mode="coord",
        prompt_variant=str(prompt_variant),
        object_field_order=object_field_order,
        device="cpu",
        root_image_dir=str(root_image_dir),
        backend_type="hf",
    )
    helper_gen = GenerationConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=float(repetition_penalty),
        batch_size=int(batch_size),
        seed=int(seed),
    )
    helper = InferenceEngine(helper_cfg, helper_gen)

    indexed_records = list(enumerate(records))
    stage2_samples: List[Stage2ParitySample] = []
    loaded_images: List[Tuple[str, Any]] = []
    gt_rows: List[List[Dict[str, Any]]] = []
    widths: List[int] = []
    heights: List[int] = []
    image_fields: List[str] = []

    for line_idx, record in indexed_records:
        sample = build_stage2_vllm_sample(
            record,
            line_idx=int(line_idx),
            prompt_variant=prompt_variant,
            object_field_order=object_field_order,
        )
        stage2_samples.append(sample)
        img_path, image = helper._prepare_image(jsonl_path, dict(record))
        if image is None:
            raise FileNotFoundError(f"failed to load image: {img_path}")
        loaded_images.append((str(img_path), image))
        width = int(record.get("width") or image.width)
        height = int(record.get("height") or image.height)
        widths.append(width)
        heights.append(height)
        image_fields.append(str((record.get("images") or [""])[0]))
        gt_errors: List[str] = []
        gt_objs = helper._compact_objects(
            helper._process_gt(dict(record), width=width, height=height, errors=gt_errors)
        )
        gt_rows.append(gt_objs)

    system_prompt, _user_prompt = get_template_prompts(
        ordering="sorted",
        coord_mode="coord_tokens",
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    engine = init_stage2_style_vllm_engine(
        model_dir=str(checkpoint_path),
        system_prompt=system_prompt,
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        enforce_eager=bool(enforce_eager),
    )
    rollout_results = run_stage2_style_vllm(
        engine=engine,
        samples=stage2_samples,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        seed=int(seed),
        batch_size=int(batch_size),
    )
    del engine

    emitted: List[Dict[str, Any]] = []
    error_counts: Dict[str, int] = {}
    pred_token_trace_path.write_text("", encoding="utf-8")
    for idx, result in enumerate(rollout_results):
        pred_errors: List[str] = list(result.errors)
        preds = helper._compact_objects(
            helper._process_pred(
                str(result.text),
                width=int(widths[idx]),
                height=int(heights[idx]),
                errors=pred_errors,
            )
        )
        for code in pred_errors:
            error_counts[str(code)] = int(error_counts.get(str(code), 0) + 1)
        emitted.append(
            {
                "image": image_fields[idx],
                "width": int(widths[idx]),
                "height": int(heights[idx]),
                "mode": "coord",
                "coord_mode": "pixel",
                "gt": gt_rows[idx],
                "pred": preds,
                "raw_output_json": result.raw_output_json,
                "raw_special_tokens": extract_special_tokens(str(result.text)),
                "raw_ends_with_im_end": str(result.text).endswith(_IM_END),
                "errors": list(pred_errors),
                "error_entries": [],
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in emitted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": "coord",
        "determinism": "best_effort",
        "errors_total": int(sum(error_counts.values())),
        "errors_by_code": error_counts,
        "counters": dict(error_counts),
        "error_codes": sorted(error_counts.keys()),
        "total_read": int(len(records)),
        "total_emitted": int(len(emitted)),
        "backend": {
            "type": "stage2_parity_vllm",
            "model_checkpoint": str(checkpoint_path),
        },
        "generation": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
        },
        "infer": {
            "gt_jsonl": str(jsonl_path),
            "prompt_variant": str(prompt_variant),
            "object_field_order": str(object_field_order),
            "root_image_dir": str(root_image_dir),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
