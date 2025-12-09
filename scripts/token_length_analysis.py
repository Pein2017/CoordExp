"""
Analyze token lengths for LVIS JSONL and simulate packing efficiency.

This script avoids loading model weights and images. It builds chat
messages via the JSONLinesBuilder, tokenizes with the tokenizer's chat
template, and replaces the single <|image_pad|> placeholder with the
expected number of vision tokens computed from image dimensions and the
vision patch/merge sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from transformers import AutoProcessor, AutoTokenizer

from src.config.loader import ConfigLoader
from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import sort_objects_by_topleft


@dataclass(frozen=True)
class LengthStats:
    count: int
    mean: float
    median: float
    p95: float
    p99: float
    max_len: int
    min_len: int


@dataclass(frozen=True)
class PackingResult:
    packing_length: int
    packs: int
    avg_samples_per_pack: float
    avg_fill: float
    max_pack_len: int
    steps_per_epoch: int
    optimizer_steps_per_epoch: int
    suggested_effective_batch: int
    suggested_grad_accum: int


def _load_config(config_path: Path) -> Tuple[Dict, Dict]:
    """Load YAML with inheritance resolved and return (raw_cfg, custom_cfg)."""
    raw_cfg = ConfigLoader.load_yaml_with_extends(str(config_path))
    custom = raw_cfg.get("custom") or {}
    if not isinstance(custom, dict):
        raise ValueError("custom section must be a mapping")
    return raw_cfg, custom


def _build_prompts(raw_cfg: Dict) -> Tuple[str, str]:
    prompts = ConfigLoader.resolve_prompts(raw_cfg)
    return prompts.system or "", prompts.user or ""


def _prepare_builder(user_prompt: str, custom_cfg: Dict) -> JSONLinesBuilder:
    coord_tokens_cfg = custom_cfg.get("coord_tokens") or {}
    coord_enabled = bool(coord_tokens_cfg.get("enabled", False))
    emit_norm = custom_cfg.get("emit_norm", "none")
    json_format = custom_cfg.get("json_format", "standard")
    mode = "summary" if custom_cfg.get("use_summary") else "dense"
    return JSONLinesBuilder(
        user_prompt=user_prompt,
        emit_norm=emit_norm,
        mode=mode,
        json_format=json_format,
        coord_tokens_enabled=coord_enabled,
    )


def _compute_image_tokens(width: int, height: int, patch_size: int) -> int:
    """Return raw vision patch tokens before any spatial merging."""
    grid_h = math.ceil(height / patch_size)
    grid_w = math.ceil(width / patch_size)
    return int(grid_h * grid_w)


def _tokenize_record(
    record: Dict,
    *,
    tokenizer,
    builder: JSONLinesBuilder,
    system_prompt: str,
    image_token_id: int,
    patch_size: int,
    merge_size: int | None,
    max_length: int | None,
    object_ordering: str,
) -> tuple[int, int, int]:
    """Return (text_len_with_placeholders, total_len_raw_patches, total_len_after_merge)."""
    rec = dict(record)
    objs = rec.get("objects") or []
    if object_ordering == "sorted":
        rec["objects"] = sort_objects_by_topleft(objs)
    elif object_ordering == "random":
        rec["objects"] = list(objs)

    merged = builder.build_many([rec])
    messages = merged["messages"]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    input_ids: Sequence[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors=None
    )
    placeholder_count = input_ids.count(image_token_id)

    width = int(rec.get("width") or 1)
    height = int(rec.get("height") or 1)
    num_images = len(rec.get("images") or [])
    if num_images == 0 and placeholder_count > 0:
        num_images = placeholder_count
    image_tokens_raw = sum(
        _compute_image_tokens(width, height, patch_size) for _ in range(num_images)
    )
    if merge_size and merge_size > 1:
        image_tokens_merged = math.ceil(image_tokens_raw / (merge_size**2))
    else:
        image_tokens_merged = image_tokens_raw

    text_len = len(input_ids)
    if max_length is not None:
        text_len = min(text_len, max_length)
    base_text = max(text_len - placeholder_count, 0)
    total_raw = base_text + image_tokens_raw
    total_merged = base_text + image_tokens_merged

    return text_len, total_raw, total_merged


def compute_lengths(
    jsonl_path: Path,
    *,
    tokenizer,
    builder: JSONLinesBuilder,
    system_prompt: str,
    patch_size: int,
    merge_size: int,
    max_length: int | None,
    object_ordering: str,
    sample_limit: int | None = None,
) -> tuple[List[int], List[int], List[int]]:
    text_lengths: List[int] = []
    total_lengths_raw: List[int] = []
    total_lengths_merged: List[int] = []
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if sample_limit is not None and idx >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text_len, total_raw, total_merged = _tokenize_record(
                record,
                tokenizer=tokenizer,
                builder=builder,
                system_prompt=system_prompt,
                image_token_id=image_token_id,
                patch_size=patch_size,
                merge_size=merge_size,
                max_length=max_length,
                object_ordering=object_ordering,
            )
            text_lengths.append(text_len)
            total_lengths_raw.append(total_raw)
            total_lengths_merged.append(total_merged)
    return text_lengths, total_lengths_raw, total_lengths_merged


def summarize_lengths(lengths: Sequence[int]) -> LengthStats:
    if not lengths:
        raise ValueError("No lengths computed")
    sorted_lens = sorted(lengths)
    count = len(sorted_lens)
    return LengthStats(
        count=count,
        mean=float(statistics.mean(sorted_lens)),
        median=float(statistics.median(sorted_lens)),
        p95=float(sorted_lens[int(count * 0.95)]),
        p99=float(sorted_lens[int(count * 0.99)]),
        max_len=int(sorted_lens[-1]),
        min_len=int(sorted_lens[0]),
    )


def make_histogram(lengths: Sequence[int], bin_size: int = 512) -> Dict[str, int]:
    hist: Dict[str, int] = defaultdict(int)
    for length in lengths:
        bucket = int(length // bin_size * bin_size)
        key = f"{bucket}-{bucket + bin_size - 1}"
        hist[key] += 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0])))


def _stats_dict(lengths: Sequence[int], bin_size: int) -> Dict[str, object]:
    stats = summarize_lengths(lengths)
    hist = make_histogram(lengths, bin_size=bin_size)
    return {
        "count": stats.count,
        "mean": stats.mean,
        "median": stats.median,
        "p95": stats.p95,
        "p99": stats.p99,
        "max": stats.max_len,
        "min": stats.min_len,
        "histogram": hist,
    }


def simulate_packing(
    lengths: Sequence[int],
    packing_length: int,
    *,
    world_size: int,
    per_device_batch: int,
    target_effective_base: int,
) -> PackingResult:
    """First-fit decreasing bin pack to estimate packs and optimizer steps."""
    lengths_sorted = sorted(lengths, reverse=True)
    bins: List[List[int]] = []
    bin_tokens: List[int] = []
    for length in lengths_sorted:
        placed = False
        for i, total in enumerate(bin_tokens):
            if total + length <= packing_length:
                bins[i].append(length)
                bin_tokens[i] += length
                placed = True
                break
        if not placed:
            bins.append([length])
            bin_tokens.append(length)

    packs = len(bins)
    total_tokens = sum(bin_tokens)
    avg_fill = total_tokens / (packs * packing_length)
    avg_samples_per_pack = len(lengths) / packs
    max_pack_len = max(bin_tokens)

    steps_per_epoch = math.ceil(packs / (world_size * per_device_batch))

    # Choose effective_batch_size so that base-sample effective ~ target_effective_base
    suggested_effective = max(
        1, math.ceil(target_effective_base / avg_samples_per_pack)
    )
    suggested_grad_accum = max(
        1, math.ceil(suggested_effective / (per_device_batch * world_size))
    )
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / suggested_grad_accum)

    return PackingResult(
        packing_length=packing_length,
        packs=packs,
        avg_samples_per_pack=avg_samples_per_pack,
        avg_fill=avg_fill,
        max_pack_len=max_pack_len,
        steps_per_epoch=steps_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        suggested_effective_batch=suggested_effective,
        suggested_grad_accum=suggested_grad_accum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Token length analysis + packing sim")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config path (YAML with extends)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="JSONL path; defaults to custom.train_jsonl in config",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional sample cap for quick tests",
    )
    parser.add_argument(
        "--binsize",
        type=int,
        default=512,
        help="Histogram bin size",
    )
    parser.add_argument(
        "--pack-lengths",
        type=int,
        nargs="+",
        default=[12000, 16000, 20000],
        help="Packing lengths to simulate",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of GPUs for effective batch computation",
    )
    parser.add_argument(
        "--per-device-train-batch",
        type=int,
        default=None,
        help="Override per_device_train_batch_size; defaults to training.per_device_train_batch_size or 1",
    )
    parser.add_argument(
        "--target-effective",
        type=int,
        default=128,
        help="Baseline effective batch size in base samples",
    )
    args = parser.parse_args()

    raw_cfg, custom_cfg = _load_config(args.config)
    system_prompt, user_prompt = _build_prompts(raw_cfg)
    jsonl_path = args.jsonl or Path(custom_cfg["train_jsonl"])
    object_ordering = str(custom_cfg.get("object_ordering", "sorted"))

    model_path = raw_cfg.get("model", {}).get("model")
    if not model_path:
        raise ValueError("model.model must be set in the config")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    patch_size = getattr(processor.image_processor, "patch_size", 16)
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    max_length = raw_cfg.get("global_max_length") or raw_cfg.get("template", {}).get(
        "max_length"
    )
    max_length = int(max_length) if max_length is not None else None

    builder = _prepare_builder(user_prompt, custom_cfg)

    text_lengths, total_lengths_raw, total_lengths_merged = compute_lengths(
        jsonl_path,
        tokenizer=tokenizer,
        builder=builder,
        system_prompt=system_prompt,
        patch_size=patch_size,
        merge_size=merge_size,
        max_length=max_length,
        object_ordering=object_ordering,
        sample_limit=args.sample_limit,
    )
    text_stats = _stats_dict(text_lengths, bin_size=args.binsize)
    total_stats_raw = _stats_dict(total_lengths_raw, bin_size=args.binsize)
    total_stats_merged = _stats_dict(total_lengths_merged, bin_size=args.binsize)

    per_device_batch = args.per_device_train_batch or int(
        raw_cfg.get("training", {}).get("per_device_train_batch_size", 1)
    )

    packing_results = [
        simulate_packing(
            text_lengths,
            pack_len,
            world_size=args.world_size,
            per_device_batch=per_device_batch,
            target_effective_base=args.target_effective,
        )
        for pack_len in args.pack_lengths
    ]

    output = {
        "config": str(args.config),
        "jsonl": str(jsonl_path),
        "patch_size": patch_size,
        "merge_size": merge_size,
        "max_length": max_length,
        "lengths": {
            "text_tokens": text_stats,
            "text_plus_vision_raw": total_stats_raw,
            "text_plus_vision_merged": total_stats_merged,
        },
        "packing": [
            {
                "packing_length": r.packing_length,
                "packs": r.packs,
                "avg_samples_per_pack": r.avg_samples_per_pack,
                "avg_fill": r.avg_fill,
                "max_pack_len": r.max_pack_len,
                "steps_per_epoch": r.steps_per_epoch,
                "optimizer_steps_per_epoch": r.optimizer_steps_per_epoch,
                "suggested_effective_batch": r.suggested_effective_batch,
                "suggested_grad_accum": r.suggested_grad_accum,
            }
            for r in packing_results
        ],
        "params": {
            "world_size": args.world_size,
            "per_device_train_batch_size": per_device_batch,
            "target_effective_base": args.target_effective,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
