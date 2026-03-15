#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.rollout_parity import (
    build_stage2_vllm_sample,
    init_stage2_style_vllm_engine,
    load_records_by_indices,
    run_stage2_style_vllm,
)
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--indices", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-variant", type=str, default="coco_80")
    parser.add_argument(
        "--object-field-order",
        type=str,
        default="desc_first",
        choices=["desc_first", "geometry_first"],
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=3084)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=14000)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-image-dir", type=str, default=None)
    return parser.parse_args()


def _parse_indices(raw: str) -> List[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def main() -> None:
    args = parse_args()
    indices = _parse_indices(args.indices)
    records = load_records_by_indices(args.jsonl, indices)
    if not records:
        raise SystemExit("No records resolved for the provided indices.")

    if args.root_image_dir:
        root_image_dir = str(Path(args.root_image_dir).resolve())
    else:
        root_image_dir = str(args.jsonl.resolve().parent)
    os.environ["ROOT_IMAGE_DIR"] = root_image_dir

    stage2_samples = [
        build_stage2_vllm_sample(
            record,
            line_idx=line_idx,
            prompt_variant=args.prompt_variant,
            object_field_order=args.object_field_order,
        )
        for line_idx, record in records
    ]

    system_prompt = str(stage2_samples[0].messages[0]["content"])
    stage2_engine = init_stage2_style_vllm_engine(
        model_dir=args.ckpt,
        system_prompt=system_prompt,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        enforce_eager=bool(args.enforce_eager),
    )
    stage2_results = run_stage2_style_vllm(
        engine=stage2_engine,
        samples=stage2_samples,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )
    del stage2_engine

    infer_cfg = InferenceConfig(
        gt_jsonl=str(args.jsonl),
        model_checkpoint=str(args.ckpt),
        mode="coord",
        device=str(args.device),
        limit=0,
        backend_type="vllm",
        prompt_variant=str(args.prompt_variant),
        object_field_order=str(args.object_field_order),
        root_image_dir=root_image_dir,
        backend={
            "type": "vllm",
            "mode": "local",
            "server_options": {
                "vllm_tensor_parallel_size": int(args.tensor_parallel_size),
                "vllm_gpu_memory_utilization": float(args.gpu_memory_utilization),
                "vllm_max_model_len": int(args.max_model_len),
            },
        },
    )
    gen_cfg = GenerationConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=float(args.repetition_penalty),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    infer_engine = InferenceEngine(infer_cfg, gen_cfg)
    infer_engine.load_model()

    infer_images = []
    for _line_idx, record in records:
        _img_path, image = infer_engine._prepare_image(args.jsonl, record)
        if image is None:
            raise RuntimeError("Failed to load image for infer-engine parity path.")
        infer_images.append(image)
    infer_outputs = infer_engine._generate_vllm_local_batch(infer_images)

    rows: List[Dict[str, Any]] = []
    for (line_idx, record), sample, stage2_res, infer_res in zip(
        records, stage2_samples, stage2_results, infer_outputs
    ):
        infer_text = str(infer_res.text or "")
        infer_raw = infer_text and infer_text or ""
        parsed_infer = None
        infer_pred_count = 0
        if infer_raw:
            from src.common.prediction_parsing import load_prediction_dict

            parsed_infer = load_prediction_dict(infer_raw)
            if isinstance(parsed_infer, dict):
                infer_pred_count = int(len(parsed_infer.get("objects") or []))
        rows.append(
            {
                "line_idx": int(line_idx),
                "image": sample.image,
                "prompt_variant": args.prompt_variant,
                "object_field_order": args.object_field_order,
                "stage2_message_head": sample.messages[:2],
                "stage2_text": stage2_res.text,
                "stage2_raw_output_json": stage2_res.raw_output_json,
                "stage2_pred_count": int(stage2_res.pred_count),
                "stage2_errors": list(stage2_res.errors),
                "infer_text": infer_text,
                "infer_raw_output_json": parsed_infer,
                "infer_pred_count": int(infer_pred_count),
                "infer_error": (
                    type(infer_res.error).__name__ if infer_res.error is not None else None
                ),
                "dataset_record_head": {
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "images": record.get("images"),
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
